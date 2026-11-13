from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class DriverLog:
    """What the driver log says: the sampler's last poll (prompt groups the next batch can draw on, groups in
    flight) and the trainer's phase (trainer_base.marked_timer; ``gen`` = it waits for its batch) with its start;
    while it trains, since when and how long the last finished step trained (the run's metrics files)."""

    available_group_n: int | None = None
    inflight_group_n: int | None = None
    phase: str | None = None
    phase_start: float | None = None
    training_start: float | None = None  # the first phase after the last gen: the trainer took its batch
    expected_training_s: float | None = None

    POLL_PATTERN: ClassVar[re.Pattern] = re.compile(r"pending: (\d+), running: (\d+), finished: (\d+)")
    PHASE_PATTERN: ClassVar[re.Pattern] = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+:phase ([a-z_]+)")
    # the driver's lines
    STAMP_PATTERN: ClassVar[re.Pattern] = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+:")
    STAMP_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"  # their local time: as text, it sorts as time does
    WAIT_PHASES: ClassVar[tuple[str, ...]] = ("gen", "step")  # the trainer waits for its batch
    # bytes read back from a log's end: the metrics' last record; the driver log's first window, doubled until it holds
    # the step's start
    TAIL_BYTES: ClassVar[int] = 400_000
    TRAINING_TIMERS: ClassVar[tuple[str, ...]] = (
        "old_log_prob",
        "adv",
        "update_actor",
        "update_weights",
        "save_checkpoint",
    )

    @classmethod
    def read(cls, path: str, batch_groups: int, launch_time: float) -> DriverLog:
        """What the driver log at ``path`` says of the launch started at ``launch_time``; nothing while it does not
        exist yet."""
        try:
            text = cls._launch_tail(path, launch_time)
        except OSError:  # the panel starts with the run, before the driver writes its log
            return cls()

        state = cls()
        phases = list(cls.PHASE_PATTERN.finditer(text))
        stamps = [(time.mktime(time.strptime(match.group(1), cls.STAMP_FORMAT)), match.group(2)) for match in phases]
        if stamps:
            state.phase_start, state.phase = stamps[-1]
            waits = [i for i, (_, name) in enumerate(stamps) if name in cls.WAIT_PHASES]
            if state.is_training:
                state.training_start = stamps[waits[-1] + 1][0] if waits else stamps[0][0]
                state.expected_training_s = cls._expected_training_s(os.path.dirname(path))

        polls = list(cls.POLL_PATTERN.finditer(text))
        if polls:
            # the sampler polls once a minute while the trainer waits, after its evictions: `finished` is what the
            # next batch can draw on (groups carried over from earlier steps included); a later phase other than
            # gen/step means the trainer has since taken its batch from them
            pending_n, running_n, finished_n = (int(g) for g in polls[-1].groups())
            is_batch_taken = any(
                match.group(2) not in cls.WAIT_PHASES for match in phases if match.start() > polls[-1].end()
            )
            state.available_group_n = max(finished_n - batch_groups, 0) if is_batch_taken else finished_n
            state.inflight_group_n = pending_n + running_n

        return state

    @property
    def is_training(self) -> bool:
        """Whether the trainer has taken its batch: its phase is past the wait for it."""
        return self.phase not in (None, *self.WAIT_PHASES)

    @classmethod
    def _launch_tail(cls, path: str, launch_time: float) -> str:
        """The log's end back to the start of the trainer's current step (its gen/step phase line), cut after the last
        line stamped before ``launch_time`` (an earlier launch's)."""
        launch_stamp = time.strftime(cls.STAMP_FORMAT, time.localtime(launch_time))
        size, window = os.path.getsize(path), cls.TAIL_BYTES

        with open(path, "rb") as f:
            while True:
                f.seek(max(size - window, 0))
                text = f.read().decode("utf-8", errors="replace")
                earlier_stamps = [match for match in cls.STAMP_PATTERN.finditer(text) if match.group(1) < launch_stamp]
                if earlier_stamps:
                    end = text.find("\n", earlier_stamps[-1].end())
                    return text[end + 1 :] if end >= 0 else ""
                if window >= size or any(
                    match.group(2) in cls.WAIT_PHASES for match in cls.PHASE_PATTERN.finditer(text)
                ):
                    return text
                window *= 2

    @classmethod
    def _expected_training_s(cls, run_dir: str) -> float | None:
        """The trainer-side seconds of the last finished step (old log-probs to checkpoint), from its metrics record."""
        for path in sorted(glob.glob(os.path.join(run_dir, "metrics_*.jsonl")), key=os.path.getmtime, reverse=True):
            with open(path, "rb") as f:
                f.seek(max(os.path.getsize(path) - cls.TAIL_BYTES, 0))
                lines = f.read().decode("utf-8", errors="replace").splitlines()

            for line in reversed(lines):
                try:
                    data = json.loads(line).get("data", {})
                except ValueError:  # the first line of the tail is cut
                    continue
                if "timing_s/update_actor" in data:
                    return sum(data.get(f"timing_s/{timer}", 0.0) for timer in cls.TRAINING_TIMERS)

        return None
