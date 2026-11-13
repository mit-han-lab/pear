from __future__ import annotations

import collections
import time
from typing import Any

import ray

from pear.utils.logging_utils import get_logger
from pear.utils.ray_utils import ON_RAY_HEAD

logger = get_logger(__name__)


class ProgressTracker:
    """Detached Ray actor (see ``handle``) holding one record per finished trajectory; the progress panel reads its
    ``stats``."""

    NAME = "pear_progress"  # the actor's name in the Ray cluster
    NAMESPACE = "pear"  # its Ray namespace

    def __init__(self) -> None:
        self.start_time = time.time()
        self.records: list[dict[str, Any]] = []

    @classmethod
    def report_trajectory(cls, record: dict[str, Any]) -> None:
        """Fire-and-forget report from an agent loop at the end of a trajectory; never raises."""
        try:
            cls.handle().report.remote(record)
        except Exception as exc:  # progress must never fail a trajectory
            logger.warning("progress report failed: %s", exc)

    def report(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def stats(self, rollout_n: int) -> dict[str, Any]:
        """The latest step's trajectories: counts, end reasons, per-task successes and, by prompt uid, the judged groups
        (every trajectory rewarded) and the kept ones (mixed rewards: what the DAPO filter keeps)."""
        step = max((r["step"] for r in self.records), default=None)
        step_records = [r for r in self.records if r["step"] == step]

        groups: dict[Any, list[bool]] = collections.defaultdict(list)
        tasks: dict[str, list[int]] = {}
        for r in step_records:
            groups[r["uid"]].append(bool(r["success"]))
            counts = tasks.setdefault(r["task"], [0, 0])
            counts[0] += 1
            counts[1] += int(bool(r["success"]))

        judged_groups = [g for g in groups.values() if len(g) == rollout_n]
        kept_groups = [g for g in judged_groups if rollout_n == 1 or 0 < sum(g) < rollout_n]

        return {
            "started": self.start_time,
            "step": step,
            "finished": len(step_records),
            "success": sum(int(bool(r["success"])) for r in step_records),
            "accept": rollout_n * len(kept_groups),
            "judged": len(judged_groups),
            "accept_groups": len(kept_groups),
            "zero": sum(1 for g in judged_groups if not any(g)),
            "one": sum(1 for g in judged_groups if all(g)),
            # avg@n: the mean pass rate over every judged group, the rejected all-zero and all-one ones included
            "avg_at_n": sum(sum(g) / len(g) for g in judged_groups) / len(judged_groups) if judged_groups else None,
            "turns": sum(int(r["num_chunks"]) for r in self.records),
            "end_reason": dict(collections.Counter(r["end_reason"] for r in step_records)),
            "tasks": tasks,
        }

    @classmethod
    def handle(cls) -> ray.actor.ActorHandle:
        """The cluster's tracker actor, created on first use (from any process of the Ray cluster)."""
        actor_cls = ray.remote(num_cpus=0)(cls)
        options = dict(
            name=cls.NAME, namespace=cls.NAMESPACE, get_if_exists=True, lifetime="detached", resources=ON_RAY_HEAD
        )

        return actor_cls.options(**options).remote()
