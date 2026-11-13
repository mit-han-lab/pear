from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pear.envs.robolab.scores import RoboLabBenchmarkScores

# RoboLab's task table, the columns the scores read: a task's class name is the first word of its task_name
TASK_TABLE = """task_name,difficulty_label,episode_s,attributes
BananaInBowlTask (a banana and a bowl),simple,30,"color, spatial"
BlocksInBinTask,moderate,60,stacking
CupsTask,complex,90,"counting, sorting, an attribute of no axis"
"""


class Run:
    """A run directory as the scores read it: the tool's calls (sim/actions_<pid>.jsonl), the agent loop's ends of its
    episodes (train.log), the evaluation's records as it writes them (val_dumps/eval_<step>.jsonl) and, once every
    episode is recorded, its summary (val_dumps/eval_<step>_summary.json)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        (path / "sim").mkdir(parents=True)
        (path / "train.log").write_text("a line of the driver's\n")

    def episode(
        self, uid: str, task: str, action_n: int, env_end_reason: str | None, subtask_score: float | None
    ) -> None:
        """The tool's calls of an episode: its open and ``action_n`` actions, the last one's result with RoboLab's end
        and subtask progress."""
        calls = [
            {
                "uid": uid,
                "env": {"task": task},
                "time": float(number),
                "action": {"arm": {"z": 1.0}} if number else None,
                "env_end_reason": None,
                "subtask_score": None,
            }
            for number in range(action_n + 1)
        ]
        calls[-1].update(env_end_reason=env_end_reason, subtask_score=subtask_score)

        with (self.path / "sim" / "actions_1.jsonl").open("a") as log:
            log.writelines(json.dumps(call) + "\n" for call in calls)

    def end(self, task: str, action_n: int, end_reason: str, success: bool, line_n: int = 1) -> None:
        """The agent loop's line of an episode's end, into which Ray's log dedup folded ``line_n`` lines."""
        record = {
            "env": {"task": task},
            "task": task,
            "success": success,
            "end_reason": end_reason,
            "executed_steps": action_n,
        }
        dedup_mark = f" [repeated {line_n}x across cluster]" if line_n > 1 else ""

        with (self.path / "train.log").open("a") as log:
            log.write(f"(AgentLoopWorker pid=7) EPISODE_END {record}{dedup_mark}\n")

    def record(self, uid: str, task: str, score: float) -> None:
        """An episode's record of the run's evaluation."""
        (self.path / "val_dumps").mkdir(exist_ok=True)
        record = {"score": score, "extra_fields": {"episode": {"uid": uid, "env": {"task": task}}}}

        with (self.path / "val_dumps" / "eval_7.jsonl").open("a") as dump:
            dump.write(json.dumps(record) + "\n")

    def summary(self) -> None:
        """The summary the run's evaluation writes at its end (the scores read only that it is there)."""
        (self.path / "val_dumps" / "eval_7_summary.json").write_text("{}")


@pytest.fixture
def task_table(tmp_path) -> Path:
    table = tmp_path / "task_table.csv"
    table.write_text(TASK_TABLE)

    return table


@pytest.fixture
def ended_run(tmp_path) -> Run:
    """A run whose evaluation ended: a success, RoboLab's time limit, the protocol's budget and a divergence."""
    run = Run(tmp_path / "ended")
    episodes = (
        ("a", "BananaInBowlTask", 1, "success", 1.0, "env", True),
        ("b", "BananaInBowlTask", 2, "time_limit", 0.5, "env", False),
        ("c", "BlocksInBinTask", 1, None, None, "budget", False),
        ("d", "CupsTask", 1, "physics_diverged", None, "env", False),
    )
    for uid, task, action_n, env_end_reason, subtask_score, end_reason, success in episodes:
        run.episode(uid, task, action_n, env_end_reason, subtask_score)
        run.end(task, action_n, end_reason, success)
        run.record(uid, task, float(success))
    run.summary()

    return run


@pytest.fixture
def stopped_run(tmp_path) -> Run:
    """A run that stopped before its records: two successes whose lines Ray folded into one, an episode the protocol
    ended at three quarters of its subtasks, and an episode still running."""
    run = Run(tmp_path / "stopped")
    run.episode("e", "BananaInBowlTask", 1, "success", 1.0)
    run.episode("f", "BananaInBowlTask", 1, "success", 1.0)
    run.end("BananaInBowlTask", 1, "env", True, line_n=2)
    run.episode("g", "BlocksInBinTask", 2, None, 0.75)
    run.end("BlocksInBinTask", 2, "parse", False)
    run.episode("h", "CupsTask", 0, None, None)

    return run


def test_an_ended_evaluation_is_scored_from_its_records(task_table, ended_run, tmp_path, capsys):
    summary_path = tmp_path / "summary.json"
    args = argparse.Namespace(runs=[str(ended_run.path)], task_table=str(task_table), json=str(summary_path))

    RoboLabBenchmarkScores(args).run()

    summary = json.loads(summary_path.read_text())
    assert summary["overall"] == {"won": 1, "episodes": 4, "sr": 25.0, "score": 37.5}  # (1 + 0.5 + 0 + 0) / 4
    assert summary["by_difficulty"]["simple"] == {"won": 1, "episodes": 2, "sr": 50.0, "score": 75.0}
    assert summary["by_axis"]["relational"] == {"won": 1, "episodes": 3, "sr": 33.3, "score": 50.0}
    assert summary["by_axis"]["procedural"]["episodes"] == 2 and summary["by_axis"]["visual"]["episodes"] == 2
    assert summary["ends"] == {"success": 1, "env_end": 2, "budget": 1}
    assert summary["tasks"][0] == {
        "task": "BananaInBowlTask",
        "difficulty": "simple",
        "limit_s": 30,
        "axes": ["relational", "visual"],
        "won": 1,
        "episodes": 2,
        "sr": 50.0,
        "score": 75.0,
        "ends": {"success": 1, "env_end": 1},
        "env_ends": {"success": 1, "time_limit": 1},
    }
    assert [row["task"] for row in summary["tasks"]] == ["BananaInBowlTask", "BlocksInBinTask", "CupsTask"]
    assert capsys.readouterr().out.splitlines()[0] == "RoboLab-120: 1/4 SR 25.0% Score 37.5"


def test_a_run_that_stopped_before_its_records_is_scored_from_its_logs(task_table, stopped_run, capsys):
    args = argparse.Namespace(runs=[str(stopped_run.path)], task_table=str(task_table), json=None)

    RoboLabBenchmarkScores(args).run()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{stopped_run.path}: no records; 3 ended episodes from its logs, unfinished {{'CupsTask': 1}}"
    assert lines[1] == "RoboLab-120: 2/3 SR 66.7% Score 91.7"  # (1 + 1 + 0.75) / 3
    assert lines[4] == "ends: {'success': 2, 'parse': 1}"
    assert len(lines) == 7  # the two tasks with an ended episode, each on its line


def test_an_evaluation_without_its_summary_is_called_unfinished_and_scored_from_its_logs(
    task_table, stopped_run, capsys
):
    stopped_run.record("e", "BananaInBowlTask", 1.0)  # two of its three ended episodes reached the records
    stopped_run.record("g", "BlocksInBinTask", 0.0)
    args = argparse.Namespace(runs=[str(stopped_run.path)], task_table=str(task_table), json=None)

    RoboLabBenchmarkScores(args).run()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        f"{stopped_run.path}: unfinished evaluation, 2 episodes recorded and no summary; "
        "3 ended episodes from its logs, unfinished {'CupsTask': 1}"
    )
    assert lines[1] == "RoboLab-120: 2/3 SR 66.7% Score 91.7"  # its three ended episodes, not the two recorded


def test_the_runs_an_evaluation_was_dealt_into_are_scored_together(task_table, ended_run, stopped_run, tmp_path):
    summary_path = tmp_path / "summary.json"
    runs = [str(ended_run.path), str(stopped_run.path)]

    RoboLabBenchmarkScores(argparse.Namespace(runs=runs, task_table=str(task_table), json=str(summary_path))).run()

    summary = json.loads(summary_path.read_text())
    assert summary["overall"] == {"won": 3, "episodes": 7, "sr": 42.9, "score": 60.7}  # (3 + 0.5 + 0.75) / 7
    assert summary["ends"] == {"success": 3, "env_end": 2, "budget": 1, "parse": 1}
    assert summary["tasks"][1]["ends"] == {"budget": 1, "parse": 1} and summary["tasks"][1]["score"] == 37.5
