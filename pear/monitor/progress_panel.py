from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import ray
from ray import cloudpickle

from pear.monitor.driver_log import DriverLog
from pear.monitor.metrics import ReplicaMetrics, ReplicaSnapshot
from pear.monitor.progress_tracker import ProgressTracker
from pear.utils.file_utils import write_text_atomically


# The live panel, on the run's Ray head, for example:
#   RAY_ADDRESS=<head ip>:6379 PYTHONPATH=<repo> python -m pear.monitor.progress_panel \
#     --target 128 --group 8 --run <run>
class ProgressPanel:
    """The progress panel: the tracker's stats, the driver log and the replicas' metrics, every INTERVAL_S."""

    INTERVAL_S = 10.0  # between refreshes
    REDISCOVER_EVERY = 6  # refreshes between discovery rounds: replicas appear after launch
    DRIVER_LOG_NAME = "train.log"  # the driver log in a run's directory: sampler groups and trainer phase
    NUMBERS_NAME = "progress.json"  # the file in a run's directory a panel rewrites with its numbers

    def __init__(self, target_trajectory_n: int, rollout_n: int, run_dir: str, writes_numbers: bool) -> None:
        self.target_trajectory_n = target_trajectory_n
        self.rollout_n = rollout_n
        self.run_dir = os.path.abspath(run_dir)
        self.driver_log_path = os.path.join(self.run_dir, self.DRIVER_LOG_NAME)
        self.numbers_path = os.path.join(self.run_dir, self.NUMBERS_NAME) if writes_numbers else None

    @classmethod
    def from_command_line(cls) -> ProgressPanel:
        """The panel of the command line (the recipes' --target, --group, --run and --json)."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--target", type=int, required=True, help="trajectories a step trains on (batch x group)")
        parser.add_argument("--group", type=int, required=True, help="trajectories per episode (rollout.n)")
        parser.add_argument("--run", required=True, help="the run's directory: its train.log is the driver log")
        parser.add_argument(
            "--json", action="store_true", help="rewrite the run's progress.json every refresh with the numbers"
        )

        args = parser.parse_args()

        return cls(args.target, args.group, args.run, args.json)

    def run(self) -> None:
        # the panel only watches the run: the cluster's worker and raylet logs belong to train.log, not to it
        ray.init(address="auto", namespace=ProgressTracker.NAMESPACE, logging_level="ERROR", log_to_driver=False)
        # a run script ends the panel with SIGTERM: it leaves as on Ctrl-C, not through the handler Ray installed, which
        # writes the signal's stack into the panel's log
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        # what the panel sends to the cluster's workers, the tracker's actor class and the replicas' discovery task,
        # goes by value: a worker of the panel's job need not find pear on its path, so the panel starts in any
        # directory
        for shipped_class in (ProgressTracker, ReplicaMetrics):
            cloudpickle.register_pickle_by_value(sys.modules[shipped_class.__module__])
        tracker = ProgressTracker.handle()
        replica_metrics = ReplicaMetrics()
        view = PanelView(
            self.target_trajectory_n,
            self.rollout_n,
            os.path.basename(self.run_dir),
            is_tty=sys.stdout.isatty(),
        )

        try:
            for refresh in itertools.count():
                if (
                    refresh % self.REDISCOVER_EVERY == 0
                    and (added_addresses := replica_metrics.discover())
                    and not view.is_tty
                ):
                    print(
                        f"{time.strftime(PanelView.CLOCK_FORMAT)} engines added: {', '.join(added_addresses)}",
                        flush=True,
                    )
                stats = ray.get(tracker.stats.remote(self.rollout_n))
                # the tracker starts with the launch (this panel or its first trajectory); older lines: earlier launches
                driver = DriverLog.read(self.driver_log_path, view.batch_groups, stats["started"])
                snapshot = replica_metrics.snapshot()
                view.draw(stats, driver, snapshot)

                if self.numbers_path:  # replaced atomically: a reader may open it at any time
                    stats.update(available_groups=driver.available_group_n, inflight_groups=driver.inflight_group_n)
                    numbers = {
                        "t": time.time(),
                        "target": self.target_trajectory_n,
                        "group": self.rollout_n,
                        "stats": stats,
                        "engines": snapshot.line(),
                        "engine_metrics": asdict(snapshot),
                        "trainer": {"phase": driver.phase, "since": driver.phase_start},
                    }
                    write_text_atomically(self.numbers_path, json.dumps(numbers))

                time.sleep(self.INTERVAL_S)
        except KeyboardInterrupt:  # Ctrl-C or SIGTERM ends the panel
            pass
        finally:
            view.close()


class PanelView:
    """The progress panel's view: a block of lines redrawn in place on a terminal, one line per refresh in a log."""

    DIM, BOLD, GREEN, YELLOW, CYAN = "2", "1", "32", "33", "36"  # ANSI SGR codes
    # ANSI control sequences
    HIDE_CURSOR_AND_CLEAR = "\x1b[?25l\x1b[2J"  # the cursor hidden, the screen cleared
    ROW_END = "\x1b[K\n"  # the rest of the row cleared, then the next row
    HOME = "\x1b[H"  # the cursor at the top: the block is redrawn over the previous one
    CLEAR_BELOW = "\x1b[J"  # what the previous block left below
    SHOW_CURSOR = "\x1b[?25h\n"  # the cursor shown again, on a line of its own
    CLOCK_FORMAT = "%H:%M:%S"  # a refresh's local time
    TREND_EVERY_S = 120.0  # between the cache-hit trend's points
    TREND_POINTS = 16  # of that trend shown
    BAR_WIDTH = 40  # a bar's columns
    LABEL_WIDTH = 10  # a row label's columns
    FALLBACK_SIZE = (120, 24)  # columns and rows when the terminal does not report its size

    def __init__(self, target_trajectory_n: int, rollout_n: int, run_name: str, is_tty: bool) -> None:
        self.batch_groups = target_trajectory_n // rollout_n
        self.run_name = run_name
        self.is_tty = is_tty
        self.trend: collections.deque = collections.deque(maxlen=self.TREND_POINTS)
        self.trend_at = 0.0

        if is_tty:
            sys.stdout.write(self.HIDE_CURSOR_AND_CLEAR)

    def draw(self, stats: dict[str, Any], driver: DriverLog, snapshot: ReplicaSnapshot) -> None:
        now = time.time()
        if snapshot.cache_hit is not None and now - self.trend_at >= self.TREND_EVERY_S:
            self.trend.append(snapshot.cache_hit)
            self.trend_at = now

        if self.is_tty:
            width = shutil.get_terminal_size(self.FALLBACK_SIZE).columns
            rows = self._rows(stats, driver, snapshot, now, width)
            block = "".join(self._fit(segments, width) + self.ROW_END for segments in rows)
            sys.stdout.write(self.HOME + block + self.CLEAR_BELOW)
        else:
            sys.stdout.write(self._line(stats, driver, snapshot, now) + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        if self.is_tty:
            sys.stdout.write(self.SHOW_CURSOR)
            sys.stdout.flush()

    def _rows(
        self, stats: dict[str, Any], driver: DriverLog, snapshot: ReplicaSnapshot, now: float, width: int
    ) -> Iterator[list[tuple[str, str]]]:
        step = f"step {stats['step']}" if stats["step"] is not None else None  # known once a trajectory of it finished
        phase = f"{driver.phase} {self._duration(now - driver.phase_start)}" if driver.phase else "no phase yet"
        head = " · ".join(part for part in ("pear", self.run_name, step, phase) if part)
        clock = time.strftime(self.CLOCK_FORMAT, time.localtime(now))
        yield [(head, self.BOLD), (" " * max(width - len(head) - len(clock), 1), ""), (clock, self.DIM)]

        if driver.is_training:  # the batch is taken: the bar is the trainer's time against the last step's
            elapsed_s, expected_s = now - driver.training_start, driver.expected_training_s
            expected_text = f" of ~{self._duration(expected_s)}" if expected_s else ""
            bar = self._bar(elapsed_s / expected_s if expected_s else 0.0, self.YELLOW)
            yield [
                self._label("training"),
                *bar,
                (f"  {driver.phase} · {self._duration(elapsed_s)}{expected_text}", ""),
            ]
        else:
            ready_group_n, words = self._batch(stats, driver)
            yield [self._label("batch"), *self._bar(ready_group_n / self.batch_groups, self.GREEN), (f"  {words}", "")]

        finished_n = stats["finished"]
        if not finished_n:
            yield [self._label("episodes"), ("none finished yet", self.DIM)]
        else:
            rate = stats["turns"] / max(now - stats["started"], 1.0)
            avg = f" · avg@n {stats['avg_at_n']:.3f}" if stats["avg_at_n"] is not None else ""
            yield [
                self._label("episodes"),
                (f"{finished_n} done · ", ""),
                (f"{stats['success']} success ({100 * stats['success'] / finished_n:.1f}%)", self.GREEN),
                (f"{avg} · {rate:.1f} turns/s", ""),
            ]
            groups = f"{stats['judged']} judged · {stats['accept_groups']} kept · {stats['zero']} all-fail · "
            yield [self._label("groups"), (groups + f"{stats['one']} all-pass", "")]
            reasons = collections.Counter(stats["end_reason"]).most_common()
            yield [self._label("ended by"), (" · ".join(f"{k} {v}" for k, v in reasons), "")]
            # <task> <successes>/<done> per task, in natural order (task_2 before task_10)
            tasks = stats["tasks"]
            task_texts = [f"{name} {tasks[name][1]}/{tasks[name][0]}" for name in sorted(tasks, key=self._natural_key)]
            yield [self._label("tasks"), ("  ".join(task_texts), "")]

        if not snapshot.known:
            yield [self._label("engines"), ("none discovered yet", self.DIM)]
            return

        load = (
            f"{snapshot.live}/{snapshot.known} up · gen {snapshot.gen_text} tok/s "
            f"· prompt {snapshot.prompt_text} tok/s · {snapshot.running_text} running "
            f"· {snapshot.waiting_text} waiting · {snapshot.preempt_text} preempt/min"
        )
        yield [self._label("engines"), (load, "")]
        trend = (
            [
                (f"   every {self.TREND_EVERY_S / 60:.0f} min, newest last: ", self.DIM),
                (" ".join(f"{100 * v:.1f}" for v in self.trend), self.CYAN),
            ]
            if self.trend
            else []
        )
        yield [self._label("cache hit"), (snapshot.cache_hit_text, ""), *trend]

    def _line(self, stats: dict[str, Any], driver: DriverLog, snapshot: ReplicaSnapshot, now: float) -> str:
        """One refresh as a log line: the same numbers, `` | ``-separated."""
        parts = [time.strftime(self.CLOCK_FORMAT, time.localtime(now))]
        if stats["step"] is not None:
            parts.append(f"step {stats['step']}")
        if driver.phase:
            parts.append(f"{driver.phase} {self._duration(now - driver.phase_start)}")

        if driver.is_training:
            expected_text = f" of ~{self._duration(driver.expected_training_s)}" if driver.expected_training_s else ""
            parts.append(f"training {self._duration(now - driver.training_start)}{expected_text}")
        else:
            parts.append(self._batch(stats, driver)[1])

        finished_n = stats["finished"]
        if finished_n:
            avg = f" avg@n {stats['avg_at_n']:.3f}" if stats["avg_at_n"] is not None else ""
            parts.append(
                f"done {finished_n} succ {stats['success']} ({100 * stats['success'] / finished_n:.1f}%){avg} "
                f"judged {stats['judged']} kept {stats['accept_groups']} "
                f"zero {stats['zero']} one {stats['one']}"
            )
            parts.append(" ".join(f"{k}={v}" for k, v in sorted(stats["end_reason"].items())))
        parts.append(snapshot.line() if snapshot.known else "engines none yet")

        return " | ".join(parts)

    def _batch(self, stats: dict[str, Any], driver: DriverLog) -> tuple[int, str]:
        """Groups ready for the next batch and the words after the bar."""
        if driver.available_group_n is None:  # no sampler poll in the driver log yet: this step's kept groups
            return stats["accept_groups"], f"{stats['accept_groups']}/{self.batch_groups} groups kept this step"

        inflight_text = f" · {driver.inflight_group_n} in flight" if driver.inflight_group_n else ""

        return driver.available_group_n, f"{driver.available_group_n}/{self.batch_groups} groups ready{inflight_text}"

    def _label(self, name: str) -> tuple[str, str]:
        return f"{name:<{self.LABEL_WIDTH}}", self.DIM

    def _bar(self, fraction: float, style: str) -> list[tuple[str, str]]:
        filled_column_n = round(self.BAR_WIDTH * min(max(fraction, 0.0), 1.0))

        return [("█" * filled_column_n, style), ("░" * (self.BAR_WIDTH - filled_column_n), self.DIM)]

    @staticmethod
    def _fit(segments: list[tuple[str, str]], width: int) -> str:
        """The segments cut to ``width`` columns, each in its ANSI style."""
        styled_texts, columns_left = [], width

        for text, style in segments:
            text = text[:columns_left]
            columns_left -= len(text)
            # ANSI SGR: the style on, the text, every style off
            styled_texts.append(f"\x1b[{style}m{text}\x1b[0m" if style and text else text)

        return "".join(styled_texts)

    @staticmethod
    def _duration(duration_s: float) -> str:
        minutes, seconds = divmod(int(max(duration_s, 0)), 60)
        if minutes >= 60:
            return f"{minutes // 60}h{minutes % 60:02d}m"

        return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"

    @staticmethod
    def _natural_key(name: str) -> list[int | str]:
        """``name`` as its runs of digits (numbers) and the text between them."""
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


if __name__ == "__main__":
    ProgressPanel.from_command_line().run()
