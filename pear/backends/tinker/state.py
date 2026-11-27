from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from pear.utils.file_utils import write_text_atomically

from .rollout import Trajectory


@dataclass
class Group:
    uid: str
    task_index: int
    dispatch_step: int
    trajectories: list[Trajectory | None]

    @property
    def complete(self) -> bool:
        return all(t is not None for t in self.trajectories)


@dataclass
class RunState:
    step: int = 0
    epoch: int = 0
    cursor: int = 0
    next_group: int = 0
    groups: list[Group] = field(default_factory=list)
    counters: dict[str, int] = field(
        default_factory=lambda: {"completed": 0, "filtered": 0, "stale": 0, "episodes": 0, "successes": 0}
    )

    def next_task(self, task_count: int, seed: int, shuffle: bool) -> int:
        order = list(range(task_count))
        if shuffle:
            random.Random(seed + self.epoch).shuffle(order)
        index = order[self.cursor]
        self.cursor += 1
        if self.cursor == task_count:
            self.cursor, self.epoch = 0, self.epoch + 1
        return index


class Checkpoints:
    """Pair Tinker's optimizer checkpoint with an atomic snapshot of the coordinator's own state."""

    VERSION = 1

    def __init__(self, config) -> None:
        self.config = config
        self.root = Path(config.run_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[RunState, str] | None:
        meta = self.reference()
        if meta is None:
            return None
        if meta["signature"] != self.config.signature():
            raise ValueError(
                "checkpoint recipe differs from this run; use a new run_dir and init_checkpoint to fork it"
            )
        # Only checkpoints written in this run directory are loaded; never accept a pickle from the model service.
        path = self.root / f"step_{int(meta['step']):08d}.pkl"
        with path.open("rb") as file:
            state = pickle.load(file)
        if not isinstance(state, RunState) or state.step != meta["step"]:
            raise ValueError("checkpoint state and manifest disagree")
        return state, meta["model_path"]

    def reference(self) -> dict | None:
        """Evaluation needs only the model reference, independent of training data and coordinator state."""
        manifest = self.root / "latest.json"
        if not manifest.exists():
            return None
        meta = json.loads(manifest.read_text())
        if meta["version"] != self.VERSION:
            raise ValueError("unsupported coordinator checkpoint version")
        return meta

    async def save(self, client, state: RunState) -> str:
        # Sampling exports are ephemeral; save_state persists model weights and optimizer state.
        saved = await client.save_state(f"step_{state.step:08d}")
        self._write_state(state)
        write_text_atomically(
            self.root / "latest.json",
            json.dumps(
                {
                    "version": self.VERSION,
                    "step": state.step,
                    "model_path": saved.path,
                    "signature": self.config.signature(),
                },
                indent=2,
            ),
        )
        old = sorted(self.root.glob("step_*.pkl"))[: -self.config.keep_checkpoints]
        for checkpoint in old:
            checkpoint.unlink()
        return saved.path

    def save_pending(self, state: RunState, groups: list[Group]) -> None:
        """Keep an expensive batch before submitting its update, paired with the preceding saved optimizer."""
        meta = self.reference()
        if meta is not None and meta["step"] == state.step:
            self._write_state(replace(state, groups=[*groups, *state.groups]))

    def _write_state(self, state: RunState) -> None:
        path = self.root / f"step_{state.step:08d}.pkl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
