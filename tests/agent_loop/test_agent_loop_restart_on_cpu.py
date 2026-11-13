import ast
import asyncio
import json
import types

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from pear.monitor.progress_tracker import ProgressTracker
from pear.sim.env import EpisodeLost
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

ENV_CLASS = "tests.StackingEnv"  # the env class of the trajectory's tool, which its row selects
RESPONSE_LENGTH = 256_000  # the run's response tokens, the trajectory's budget (its tool sets none)
# what verl passes with every row: its prompt group's uid and the training step
ROW = {"uid": "the row's prompt group", "global_steps": 3}


class RecordingTool:
    """The environment tool as PearAgentLoop.run sees it: records the episodes it closes, next to the runs that played
    them."""

    env_class = types.SimpleNamespace(task_name=lambda env_args: env_args["task"])
    config = {}  # the tool config's entries: no response_budget

    def __init__(self) -> None:
        self.events = []  # ("run", episode) for each run of the base class, ("close", episode) for each close

    async def close(self, episode: Episode) -> None:
        self.events.append(("close", episode))


class RestartHarness:
    """PearAgentLoop.run without tokenizer or server: each run of the base class raises the next of ``failures`` (None: the
    run ends with an output)."""

    def __init__(self, monkeypatch, *failures: BaseException | None) -> None:
        self.failures = list(failures)

        async def base_run(loop, sampling_params, **kwargs):
            loop.tool.events.append(("run", loop.episode))
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
            loop.end_reason = EndReason.ENV  # a run that ends has ended its episode
            return types.SimpleNamespace(reward_score=None, extra_fields={})

        monkeypatch.setattr(ToolAgentLoop, "run", base_run)
        # the progress tracker is a Ray actor: a report from here would start a Ray cluster in the test's process
        self.records = []  # what the loop reported of its ended episodes
        monkeypatch.setattr(ProgressTracker, "report_trajectory", staticmethod(self.records.append))
        self.loop = object.__new__(PearAgentLoop)
        self.loop.response_length = RESPONSE_LENGTH
        self.loop.tools = {ENV_CLASS: RecordingTool()}

    def run(self):
        extra_info = {"tool_selection": [ENV_CLASS], "env": json.dumps({"task": "stack_bowls"})}

        return asyncio.run(self.loop.run({}, extra_info=extra_info, **ROW))

    def played_new_episodes(self, runs: int) -> bool:
        """Whether its ``runs`` runs each played a new episode of the row's env args, closed right after it."""
        events = self.loop.tool.events
        played = [episode for event, episode in events if event == "run"]
        return (
            len({id(episode) for episode in played}) == runs == len(played)
            and events == [(event, episode) for episode in played for event in ("run", "close")]
            and all(episode.env_args == {"task": "stack_bowls"} for episode in played)
        )


def test_a_trajectory_whose_environment_was_lost_starts_over_in_a_new_episode(monkeypatch):
    harness = RestartHarness(monkeypatch, EpisodeLost("the process crashed"), None)

    output = harness.run()

    assert harness.played_new_episodes(runs=2) and output.extra_fields["episode"]["env"] == {"task": "stack_bowls"}
    # the reward is the episode's success, and these runs carried out no action
    assert output.reward_score == 0.0 and output.extra_fields["reward_extra_info"] == {"success": 0.0, "done": 0.0}


def test_the_end_record_names_the_end_reason_by_its_plain_text(monkeypatch):
    # the EPISODE_END line is the record's repr, which a family's scores read back with ast.literal_eval
    harness = RestartHarness(monkeypatch, None)

    harness.run()

    [record] = harness.records
    assert type(record["end_reason"]) is str
    assert ast.literal_eval(repr(record))["end_reason"] == "env"


@pytest.mark.parametrize(
    "failures",
    [[EpisodeLost("the process crashed")] * (PearAgentLoop.MAX_RESTARTS + 1), [RuntimeError("the trajectory failed")]],
)
def test_a_trajectory_fails_on_its_last_lost_run_or_any_other_error_its_episodes_closed(monkeypatch, failures):
    harness = RestartHarness(monkeypatch, *failures)

    with pytest.raises(type(failures[0])):
        harness.run()

    assert harness.played_new_episodes(runs=len(failures))
