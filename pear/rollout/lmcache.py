from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

from pear.utils.logging_utils import get_logger
from pear.utils.net_utils import free_port
from pear.utils.system_utils import die_with_parent
from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = get_logger(__name__)


class LMCacheSidecar:
    """One ``lmcache server`` process bound to its GPUs' NUMA node: its start, vLLM's connector to it and its clear."""

    HOST = "127.0.0.1"  # the sidecar serves its own replica, on the replica's machine
    NUMA_HEADROOM_GB = 100  # free memory the sidecar's NUMA node must keep after the tier is pinned
    BASE_PORT = 5560  # + local replica index; the vLLM connector's port
    HTTP_BASE_PORT = 18080  # + local replica index; the sidecar's HTTP API (cache clear)
    MAX_WORKERS = 8  # sidecar worker threads
    PIN_TIMEOUT_S = 1800.0  # how long the sidecar may take to pin its tier
    # eviction: the tier is pinned up front and a store that cannot reserve space is skipped, so a high watermark
    # only trades headroom for capacity; small rounds evict the superseded recurrent pages without taking live
    # prefixes
    EVICTION_POLICY = "LRU"
    EVICTION_RATIO = 0.05  # share of the tier freed per eviction round
    EVICTION_WATERMARK = 0.8  # tier fill level that triggers eviction
    POLL_S = 5.0  # between checks of the sidecar's log while it pins
    CLEAR_PATH = "/cache/clear"  # the sidecar's HTTP API path that clears the whole tier
    CLEAR_TIMEOUT_S = 120.0  # a force-clear is bookkeeping; beyond this the sidecar is stuck
    ERROR_TEXT_CHARS = 200  # of the sidecar's answer, in the error of a failed clear
    NUMA_MIN_LOCAL_FRACTION = 0.95  # share of the sidecar's resident memory that must be on its NUMA node
    PIN_PATTERN = re.compile(
        rb"now total is (\d+) MB / (\d+) MB"
    )  # the pin-progress line: MB pinned so far / MB in the tier

    def __init__(self, cuda_visible_devices: str, l1_size_gb: int, chunk_size: int, log_path: Path) -> None:
        self.cuda_visible_devices = cuda_visible_devices
        self.l1_size_gb = l1_size_gb
        self.chunk_size = chunk_size
        self.log_path = log_path

        gpus = [int(g) for g in cuda_visible_devices.split(",") if g.strip()]
        index = min(gpus) // len(gpus)  # 0 for the node's first TP group, 1 for the second: distinct ports
        # shared nodes: fixed ports collide with other users' servers
        self.port = free_port(self.BASE_PORT + index)
        self.http_port = free_port(self.HTTP_BASE_PORT + index)

    def start(self) -> None:
        """Start ``lmcache server`` bound to its GPUs' NUMA node and wait until its CPU tier is fully pinned."""
        # LMCache's allocator drops the NUMA node id, so placement is enforced with numactl --membind: a tier on the
        # other node exhausts it once two replicas share a node (swap, stalls, heartbeat loss)
        numa_node = self._numa_node(self.cuda_visible_devices)
        available_mb = self._available_mb(numa_node)
        if available_mb < (self.l1_size_gb + self.NUMA_HEADROOM_GB) * 1024:
            raise RuntimeError(
                f"NUMA node {numa_node} has {available_mb} MB available, "
                f"need {self.l1_size_gb} GB tier + {self.NUMA_HEADROOM_GB} GB"
            )

        command = [
            "numactl",
            f"--cpunodebind={numa_node}",
            f"--membind={numa_node}",
            # the CLI of this actor's own interpreter (``lmcache`` console script = ``lmcache.cli.main:main``)
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            self.HOST,
            "--port",
            str(self.port),
            "--http-port",
            str(self.http_port),
            "--chunk-size",
            str(self.chunk_size),
            "--separate-object-groups",
            "--l1-size-gb",
            str(self.l1_size_gb),
            "--eviction-policy",
            self.EVICTION_POLICY,
            "--eviction-ratio",
            str(self.EVICTION_RATIO),
            "--eviction-trigger-watermark",
            str(self.EVICTION_WATERMARK),
            "--max-workers",
            str(self.MAX_WORKERS),
            "--supported-transfer-mode",
            "lmcache_driven",
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            "LMCACHE_DISABLE_BANNER": "1",
            "LMCACHE_LOG_LEVEL": "INFO",
        }  # the pin-progress line that signals readiness is logged at INFO

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # the log is appended to (a resumed run reuses its RUN dir), so only the bytes written by this server count:
        # an earlier server's final "now total is N MB / N MB" line must not pass as this one's readiness
        log_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        with self.log_path.open("ab") as log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, preexec_fn=die_with_parent
            )

        self._wait_pinned(process, numa_node, log_offset)

    def connector_config(self) -> dict:
        """vLLM's kv_transfer_config of the replica whose KV cache this sidecar keeps."""
        return {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"lmcache.mp.host": f"tcp://{self.HOST}", "lmcache.mp.port": self.port},
        }

    async def clear(self) -> int:
        """Clear the whole tier and return the status of the sidecar's answer, which is 200, or raise on another."""
        url = f"http://{self.HOST}:{self.http_port}{self.CLEAR_PATH}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.CLEAR_TIMEOUT_S)) as session:
            async with session.post(url) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"LMCache invalidate {self.CLEAR_PATH} -> {response.status}: "
                        f"{(await response.text())[: self.ERROR_TEXT_CHARS]}"
                    )

                return response.status

    def _wait_pinned(self, process: subprocess.Popen, numa_node: int, log_offset: int) -> None:
        """Wait until the server's log shows its whole tier pinned, with at least NUMA_MIN_LOCAL_FRACTION on
        ``numa_node``."""
        deadline = time.monotonic() + self.PIN_TIMEOUT_S

        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"lmcache server exited with {process.returncode}, see {self.log_path}")
            matches = self.PIN_PATTERN.findall(self.log_path.read_bytes()[log_offset:])
            if matches and matches[-1][0] == matches[-1][1]:
                numastat_output = subprocess.run(
                    ["numastat", "-p", str(process.pid)], capture_output=True, text=True, check=True
                ).stdout
                total = [line for line in numastat_output.splitlines() if line.startswith("Total")][-1].split()
                placement = [int(float(x)) for x in total[1:-1]]  # resident MB per NUMA node
                local_fraction = placement[numa_node] / max(1, sum(placement))
                if local_fraction < self.NUMA_MIN_LOCAL_FRACTION:
                    process.kill()
                    raise RuntimeError(
                        f"lmcache server on port {self.port} is not on NUMA node {numa_node}: {placement} MB per node"
                    )
                logger.info(
                    "lmcache server on port %d pinned %s MB, %.1f%% on NUMA node %d",
                    self.port,
                    matches[-1][1].decode(),
                    100 * local_fraction,
                    numa_node,
                )
                return
            time.sleep(self.POLL_S)

        process.kill()
        raise RuntimeError(
            f"lmcache server on port {self.port} did not finish pinning within {self.PIN_TIMEOUT_S}s, "
            f"see {self.log_path}"
        )

    @staticmethod
    def _numa_node(cuda_visible_devices: str) -> int:
        """The NUMA node of the first visible GPU, from its PCI bus id in sysfs."""
        gpu = int(cuda_visible_devices.split(",")[0])
        bus = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # sysfs names the PCI domain with 4 digits, nvidia-smi with 8
        numa_node = int(Path(f"/sys/bus/pci/devices/{bus[4:].lower()}/numa_node").read_text())
        if numa_node < 0:
            raise RuntimeError(f"GPU {gpu} ({bus}) reports no NUMA node")

        return numa_node

    @staticmethod
    def _available_mb(numa_node: int) -> int:
        """Memory the kernel can hand to a new allocation on ``numa_node``: free pages plus reclaimable page cache and
        slab."""
        # numactl -H's "free" excludes the cache
        fields = {}

        for line in Path(f"/sys/devices/system/node/node{numa_node}/meminfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                fields[parts[2].rstrip(":")] = int(parts[3])

        if "MemFree" not in fields:
            raise RuntimeError(f"cannot read meminfo of NUMA node {numa_node}")

        return sum(fields.get(k, 0) for k in ("MemFree", "Active(file)", "Inactive(file)", "SReclaimable")) // 1024


class LMCacheVLLMServer(vLLMHttpServer):
    """vLLM's HTTP server with an LMCache sidecar: started with the engine, its tier cleared whenever the prefix
    cache is."""

    lmcache_sidecar: LMCacheSidecar | None = None  # None: the replica runs without LMCache

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)

        lmcache_config = self.config.custom["lmcache"]  # the recipe's rollout.custom.lmcache block
        if not lmcache_config["enable"]:
            return
        if self.config.enable_sleep_mode:
            # a sleep-mode replica's KV cache (vLLM's VMM allocator) cannot be shared with the sidecar over CUDA IPC,
            # so such a replica runs without LMCache; base.yaml turns sleep mode off
            logger.info(
                "replica %d node %d: sleep mode is on, LMCache sidecar not attached", self.replica_rank, self.node_rank
            )
            return

        log_dir = Path(lmcache_config["log_dir"])
        sidecar = LMCacheSidecar(
            os.environ[get_visible_devices_keyword()],
            int(lmcache_config["l1_size_gb"]),
            int(lmcache_config["chunk_size"]),
            log_dir / f"lmcache_replica{self.replica_rank}_node{self.node_rank}.log",
        )
        sidecar.start()

        # the replica's only once it runs: a start that raised leaves the replica without one
        self.lmcache_sidecar = sidecar
        engine_kwargs["kv_transfer_config"] = sidecar.connector_config()
        logger.info(
            "replica %d node %d: LMCache sidecar on port %d (http %d)",
            self.replica_rank,
            self.node_rank,
            sidecar.port,
            sidecar.http_port,
        )

    async def clear_kv_cache(self) -> None:
        # Called after the new weights are loaded, while generation is still paused: reset_prefix_cache does not
        # reach the sidecar's tier, which would serve KV of the old weights to the resumed episodes. Fails closed on
        # anything but a 200.
        await super().clear_kv_cache()

        if self.lmcache_sidecar is None:
            return

        status = await self.lmcache_sidecar.clear()
        # live visibility: one line per clear in the driver log (the sidecar's own log has the object counts)
        logger.warning("LMCache invalidate %s -> %s (replica %s)", LMCacheSidecar.CLEAR_PATH, status, self.replica_rank)
