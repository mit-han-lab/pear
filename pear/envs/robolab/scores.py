from __future__ import annotations

import argparse
import ast
import collections
import csv
import glob
import json
import os
import re
from collections.abc import Iterator

from pear.agent_loop.episode import EndReason


class RoboLabBenchmarkScores:
    """A RoboLab-120 evaluation with N episodes per task, over the runs it was dealt into, scored as RoboLab's
    leaderboard scores one: the success rate (SR) and the mean episode score (a success 1, else its last subtask
    progress), overall, by difficulty and by competency axis, and per task with its episodes' ends."""

    # RoboLab's mapping of task attributes to its three competency axes
    AXIS_BY_ATTRIBUTE = {
        "size": "visual",
        "color": "visual",
        "semantics": "visual",
        "spatial": "relational",
        "conjunction": "relational",
        "counting": "relational",
        "stacking": "procedural",
        "sorting": "procedural",
        "reorientation": "procedural",
        "affordance": "procedural",
    }
    DIFFICULTIES = ("simple", "moderate", "complex")
    EPISODE_END_PATTERN = re.compile(r"EPISODE_END (\{.*\})")  # the agent loop's record of an episode's end
    # Ray's log dedup: the line stands for N lines
    REPEAT_COUNT_PATTERN = re.compile(r"\[repeated (\d+)x across cluster\]")
    # in place of a records file's extension: the evaluation's summary, which it writes once every episode is recorded
    SUMMARY_SUFFIX = "_summary.json"

    def __init__(self, args: argparse.Namespace) -> None:
        self.run_dirs = args.runs
        self.task_table_path = args.task_table
        self.summary_path = args.json  # where the summary is written too, if anywhere

    @classmethod
    def from_command_line(cls) -> RoboLabBenchmarkScores:
        """The scores of the runs and the task table the command line names."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument(
            "--runs",
            nargs="+",
            required=True,
            help="the run dirs (val_dumps/eval_<step>.jsonl, sim/actions_*.jsonl, train.log)",
        )
        parser.add_argument("--task-table", required=True, help="RoboLab's robolab/tasks/_metadata/task_table.csv")
        parser.add_argument("--json", help="also write the summary here")

        return cls(parser.parse_args())

    def run(self) -> None:
        """Prints the evaluation's scores, and writes them as JSON where the command line says."""
        tasks = self.task_table()
        episodes, ends = [], collections.defaultdict(collections.Counter)
        for run_dir in self.run_dirs:
            episodes += self.episodes(run_dir)
            for key, counts in self.loop_ends(run_dir).items():
                ends[key].update(counts)

        rows = []
        for task in sorted({episode["task"] for episode in episodes}):
            task_episodes = [episode for episode in episodes if episode["task"] == task]
            rows.append(
                {
                    "task": task,
                    **tasks[task],
                    **self.scores(task_episodes),
                    "ends": dict(ends[task]),
                    "env_ends": dict(collections.Counter(episode["env_end"] for episode in task_episodes)),
                }
            )

        summary = {
            "overall": self.scores(episodes),
            "by_difficulty": {
                difficulty: self.scores(
                    [episode for episode in episodes if tasks[episode["task"]]["difficulty"] == difficulty]
                )
                for difficulty in self.DIFFICULTIES
            },
            "by_axis": {
                axis: self.scores([episode for episode in episodes if axis in tasks[episode["task"]]["axes"]])
                for axis in sorted(set(self.AXIS_BY_ATTRIBUTE.values()))
            },
            "ends": dict(sum(ends.values(), collections.Counter())),
            "tasks": rows,
        }
        self.print_summary(summary)
        if self.summary_path:
            with open(self.summary_path, "w") as summary_file:
                json.dump(summary, summary_file, indent=1)

    def task_table(self) -> dict[str, dict]:
        """RoboLab's task table: per task class its difficulty, competency axes and time limit."""
        tasks = {}
        with open(self.task_table_path) as table:
            for row in csv.DictReader(table):
                attributes = [attribute.strip() for attribute in row["attributes"].split(",") if attribute.strip()]
                tasks[row["task_name"].split(" ")[0]] = {
                    "difficulty": row["difficulty_label"],
                    "limit_s": int(row["episode_s"]),
                    "axes": sorted(
                        {
                            self.AXIS_BY_ATTRIBUTE[attribute]
                            for attribute in attributes
                            if attribute in self.AXIS_BY_ATTRIBUTE
                        }
                    ),
                }

        return tasks

    def episodes(self, run_dir: str) -> list[dict]:
        """A run's episodes from its records, each with its success and, from the tool's last call for its uid,
        RoboLab's end (success | time_limit | physics_diverged; None: the protocol ended it) and last subtask progress;
        from its logs (``logged_episodes``) when its evaluation did not finish: no records, or records without their
        summary."""
        last_calls, action_counts = {}, collections.Counter()
        for path in glob.glob(os.path.join(run_dir, "sim", "actions_*.jsonl")):
            with open(path) as log:
                for line in log:
                    call = json.loads(line)
                    action_counts[call["uid"]] += call["action"] is not None
                    if call["uid"] not in last_calls or call["time"] >= last_calls[call["uid"]]["time"]:
                        last_calls[call["uid"]] = call

        record_paths = glob.glob(os.path.join(run_dir, "val_dumps", "eval_*.jsonl"))
        if not record_paths:
            return self.logged_episodes(run_dir, last_calls, action_counts, "no records")

        [record_path] = record_paths
        if not os.path.exists(os.path.splitext(record_path)[0] + self.SUMMARY_SUFFIX):
            with open(record_path) as dump:
                record_n = sum(1 for _ in dump)
            records_state = f"unfinished evaluation, {record_n} episodes recorded and no summary"

            return self.logged_episodes(run_dir, last_calls, action_counts, records_state)

        episodes = []
        with open(record_path) as dump:
            for line in dump:
                record = json.loads(line)
                episode = record["extra_fields"]["episode"]
                call = last_calls[episode["uid"]]
                episodes.append(
                    {
                        "task": episode["env"]["task"],
                        "success": record["score"] > 0,
                        "env_end": call["env_end_reason"],
                        "subtask_score": call["subtask_score"] or 0.0,
                    }
                )

        return episodes

    def logged_episodes(
        self, run_dir: str, last_calls: dict[str, dict], action_counts: collections.Counter, records_state: str
    ) -> list[dict]:
        """A run's ended episodes from its logs alone, printed with ``records_state`` (why not from its records): each
        EPISODE_END line (its task, end and actions) matched to a uid of the tool's calls with that task, as many
        actions and, for RoboLab's ends, an end of RoboLab's; the uids left were still running when the run stopped,
        and are counted apart (``unfinished_counts``)."""
        free_uids = collections.defaultdict(list)  # (task, actions, ended by RoboLab) -> uids
        for uid, call in last_calls.items():
            free_uids[(call["env"]["task"], action_counts[uid], call["env_end_reason"] is not None)].append(uid)

        episodes = []
        for record, line_n in self.episode_ends(run_dir):
            for _ in range(line_n):
                uid = free_uids[(record["task"], record["executed_steps"], record["end_reason"] == EndReason.ENV)].pop()
                episodes.append(
                    {
                        "task": record["task"],
                        "success": bool(record["success"]),
                        "env_end": last_calls[uid]["env_end_reason"],
                        "subtask_score": last_calls[uid]["subtask_score"] or 0.0,
                    }
                )

        unfinished_counts = collections.Counter(key[0] for key, uids in free_uids.items() for _ in uids)
        print(
            f"{run_dir}: {records_state}; {len(episodes)} ended episodes from its logs, "
            f"unfinished {dict(unfinished_counts)}"
        )

        return episodes

    def loop_ends(self, run_dir: str) -> dict[str, collections.Counter]:
        """Per task the agent loop's ends of a run's episodes: success, env_end (RoboLab's, without success), parse or
        budget, each line weighted by the count Ray's log dedup folded into it."""
        ends = collections.defaultdict(collections.Counter)
        for record, line_n in self.episode_ends(run_dir):
            end = (
                record["end_reason"]
                if record["end_reason"] != EndReason.ENV
                else ("success" if record["success"] else "env_end")
            )
            ends[record["env"]["task"]][end] += line_n

        return ends

    def episode_ends(self, run_dir: str) -> Iterator[tuple[dict, int]]:
        """The agent loop's records of a run's episode ends (the EPISODE_END lines of its train.log), each with the
        number of lines Ray's log dedup folded into its line."""
        with open(os.path.join(run_dir, "train.log"), errors="replace") as log:
            for line in log:
                match = self.EPISODE_END_PATTERN.search(line)
                if match is None:
                    continue
                record, repeat_match = ast.literal_eval(match.group(1)), self.REPEAT_COUNT_PATTERN.search(line)
                yield record, int(repeat_match.group(1)) if repeat_match else 1

    @staticmethod
    def scores(episodes: list[dict]) -> dict:
        """The episodes' successes, count, success rate and mean score (a success scores 1), the rates in percent."""
        success_n = sum(episode["success"] for episode in episodes)
        score = sum(1.0 if episode["success"] else episode["subtask_score"] for episode in episodes)
        episode_n = len(episodes)

        return {
            "won": success_n,
            "episodes": episode_n,
            "sr": round(100 * success_n / episode_n, 1) if episode_n else None,
            "score": round(100 * score / episode_n, 1) if episode_n else None,
        }

    @staticmethod
    def print_summary(summary: dict) -> None:
        overall = summary["overall"]
        print(f"RoboLab-120: {overall['won']}/{overall['episodes']} SR {overall['sr']}% Score {overall['score']}")
        for group in ("by_difficulty", "by_axis"):
            print(
                f"{group}: "
                + ", ".join(
                    f"{label} {scores['won']}/{scores['episodes']} SR {scores['sr']}% Score {scores['score']}"
                    for label, scores in summary[group].items()
                )
            )
        print(f"ends: {summary['ends']}")

        for row in summary["tasks"]:
            print(
                f"  {row['task']:40s} {row['won']:2d}/{row['episodes']} Score {row['score']:5.1f}  "
                f"{row['difficulty']:9s} {','.join(row['axes']):22s} limit {row['limit_s']:3d}s ends {row['ends']} "
                f"env {row['env_ends']}"
            )


if __name__ == "__main__":
    RoboLabBenchmarkScores.from_command_line().run()
