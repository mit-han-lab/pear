from __future__ import annotations

import collections
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import ClassVar

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReplicaSnapshot:
    """The replicas over the last minute: token rates per second, requests now, and the share of queried prompt tokens
    served from a cache (GPU prefix hits plus LMCache hits among the GPU misses); None for a number whose metric a
    replica does not report."""

    live: int
    known: int
    gen_tps: float | None
    prompt_tps: float | None
    running: int | None
    waiting: int | None
    cache_hit: float | None
    preempt_per_min: float | None

    MISSING: ClassVar[str] = "n/a"  # shown in place of a number the snapshot does not have

    def line(self) -> str:
        """The snapshot in the log's form."""
        return (
            f"engines {self.live}/{self.known} gen {self.gen_text} tok/s prompt {self.prompt_text} tok/s "
            f"run {self.running_text} wait {self.waiting_text} cachehit {self.cache_hit_text} "
            f"preempt {self.preempt_text}/min"
        )

    # each number as the log's line and the panel's rows show it
    @property
    def gen_text(self) -> str:
        return self.shown(self.gen_tps, "{:.1f}k", 1e-3)

    @property
    def prompt_text(self) -> str:
        return self.shown(self.prompt_tps, "{:.0f}k", 1e-3)

    @property
    def running_text(self) -> str:
        return self.shown(self.running, "{}")

    @property
    def waiting_text(self) -> str:
        return self.shown(self.waiting, "{}")

    @property
    def cache_hit_text(self) -> str:
        return self.shown(self.cache_hit, "{:.1f}%", 100)

    @property
    def preempt_text(self) -> str:
        return self.shown(self.preempt_per_min, "{:.0f}")

    @classmethod
    def shown(cls, number: float | None, pattern: str, scale: float = 1) -> str:
        """A number times scale in its pattern, MISSING without the number."""
        return cls.MISSING if number is None else pattern.format(number * scale)


@dataclass
class ReplicaSample:
    """One replica's answer at one time: the value of each vLLM metric of ReplicaMetrics.METRICS, None for one the
    answer lacks."""

    time: float
    values: dict[str, float | None]


class ReplicaMetrics:
    """The vLLM ``/metrics`` endpoints of every replica in the cluster, aggregated over a sliding window."""

    # the vLLM metrics a sample holds: six counters, then two gauges
    METRICS = (
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "external_prefix_cache_hits_total",
        "prompt_tokens_total",
        "generation_tokens_total",
        "num_preemptions_total",
        "num_requests_running",
        "num_requests_waiting",
    )
    WINDOW_S = 60.0  # rates over the samples of the last minute
    HISTORY = 64  # samples kept per replica
    PROBE_TIMEOUT_S = 2.0  # per /metrics request: a replica that misses it is down for this refresh
    PROBE_BYTES = 4096  # of a listener's answer while discovering: enough to see a vllm: counter
    DISCOVERY_TIMEOUT_S = 15.0  # per round; a loaded node that misses it is retried on the next one
    # the vLLM server actors' process name as `ss -p` prints it: the kernel's comm field, the first 15 characters of the
    # title Ray gives an actor's process, ray::<its class name> (pear/rollout/server.py)
    SERVER_ACTOR = "ray::PearVLLMSe"

    def __init__(self) -> None:
        self.history: dict[str, collections.deque] = {}
        self.absent_metrics: set[str] = set()  # the vLLM metrics a replica's answer lacked, each said once

    def discover(self) -> list[str]:
        """One discovery round over every alive node; returns the addresses of the replicas it added."""
        task = ray.remote(num_cpus=0)(self._node_addresses)
        # as verl places its agent-loop workers: a node without CPUs (a simulator node) runs none of the training's
        # processes
        nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
        node_strategies = [NodeAffinitySchedulingStrategy(node_id=n, soft=False) for n in nodes]
        refs = [task.options(scheduling_strategy=strategy).remote() for strategy in node_strategies]
        ready_refs, _ = ray.wait(refs, num_returns=len(refs), timeout=self.DISCOVERY_TIMEOUT_S)

        added_addresses = sorted({address for ref in ready_refs for address in ray.get(ref)} - set(self.history))
        for added_address in added_addresses:
            self.history[added_address] = collections.deque(maxlen=self.HISTORY)

        return added_addresses

    def snapshot(self) -> ReplicaSnapshot:
        """Every known replica sampled once, with its rates over the window; a number is None when a replica does not
        report a metric it is made of."""
        windows = []  # of each live replica: its sample of now and its oldest one within the window
        for address, history in self.history.items():
            latest_sample = self._sample(address)
            if latest_sample is None:
                continue
            history.append(latest_sample)
            windows.append(
                (latest_sample, next(sample for sample in history if latest_sample.time - sample.time <= self.WINDOW_S))
            )

        queried_token_n, hits, external_hits = (
            self._increase(windows, metric)
            for metric in ("prefix_cache_queries_total", "prefix_cache_hits_total", "external_prefix_cache_hits_total")
        )
        cache_hit = None
        if queried_token_n and hits is not None and external_hits is not None:
            cache_hit = min(hits + external_hits, queried_token_n) / queried_token_n
        preempt_per_s = self._rate(windows, "num_preemptions_total")

        return ReplicaSnapshot(
            live=len(windows),
            known=len(self.history),
            gen_tps=self._rate(windows, "generation_tokens_total"),
            prompt_tps=self._rate(windows, "prompt_tokens_total"),
            running=self._level(windows, "num_requests_running"),
            waiting=self._level(windows, "num_requests_waiting"),
            cache_hit=cache_hit,
            preempt_per_min=None if preempt_per_s is None else preempt_per_s * 60,
        )

    @staticmethod
    def _node_addresses() -> list[str]:
        """Runs on a node: ``ip:port`` of every vLLM server actor listener that answers /metrics with vLLM counters."""
        listener_addresses = set()

        for line in subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout.splitlines():
            match = re.search(r"\s(\d+\.\d+\.\d+\.\d+):(\d+)\s", line)
            if ReplicaMetrics.SERVER_ACTOR in line and match and not match.group(1).startswith(("0.0.0.0", "127.")):
                address = f"{match.group(1)}:{match.group(2)}"
                if b"vllm:" in (ReplicaMetrics._get(address, ReplicaMetrics.PROBE_BYTES) or b""):
                    listener_addresses.add(address)

        return sorted(listener_addresses)

    def _sample(self, address: str) -> ReplicaSample | None:
        """The replica's metrics now, None for one its answer lacks; None when it does not answer."""
        text = self._get(address)
        if text is None:
            return None

        sample = ReplicaSample(time.time(), {})
        for metric in self.METRICS:
            match = re.search(rf"^vllm:{metric}\{{[^}}]*\}} ([0-9.e+]+)", text.decode(), re.M)
            sample.values[metric] = float(match.group(1)) if match else None

        # said once per metric: the numbers made of it show as missing from here on
        absent_names = {metric for metric in self.METRICS if sample.values[metric] is None} - self.absent_metrics
        if absent_names:
            self.absent_metrics |= absent_names
            names = ", ".join(f"vllm:{name}" for name in sorted(absent_names))
            logger.warning("engine %s does not report %s: the numbers made of it show as missing", address, names)

        return sample

    @staticmethod
    def _level(windows: list[tuple[ReplicaSample, ReplicaSample]], metric: str) -> int | None:
        """A gauge summed over the replicas, None when one of them does not report it."""
        levels = [latest_sample.values[metric] for latest_sample, _ in windows]

        return None if None in levels else int(sum(levels))

    @staticmethod
    def _increase(windows: list[tuple[ReplicaSample, ReplicaSample]], metric: str) -> float | None:
        """A counter's increase over the replicas' windows, None when one of them does not report it."""
        value_pairs = [
            (latest_sample.values[metric], base_sample.values[metric]) for latest_sample, base_sample in windows
        ]
        if any(None in pair for pair in value_pairs):
            return None

        return sum(latest_value - base_value for latest_value, base_value in value_pairs)

    @staticmethod
    def _rate(windows: list[tuple[ReplicaSample, ReplicaSample]], metric: str) -> float | None:
        """A counter's increase per second summed over the replicas, None when one of them does not report it."""
        if any(
            None in (latest_sample.values[metric], base_sample.values[metric]) for latest_sample, base_sample in windows
        ):
            return None

        return sum(
            (latest_sample.values[metric] - base_sample.values[metric]) / (latest_sample.time - base_sample.time)
            for latest_sample, base_sample in windows
            if latest_sample.time > base_sample.time
        )

    @staticmethod
    def _get(address: str, limit: int | None = None) -> bytes | None:
        try:
            url = f"http://{address}/metrics"
            with urllib.request.urlopen(url, timeout=ReplicaMetrics.PROBE_TIMEOUT_S) as response:
                return response.read(limit)
        except Exception:  # not every listener speaks HTTP, and a restarting replica answers later
            return None
