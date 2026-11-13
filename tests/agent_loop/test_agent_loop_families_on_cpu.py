import asyncio
import json
import types

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.dataset import EpisodeDataset
from pear.agent_loop.episode import EndReason, Episode
from pear.monitor.progress_tracker import ProgressTracker
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

ARM_ENV, TABLE_ENV = "tests.ArmEnv", "tests.TableEnv"  # two families' env classes, their tools' names
RUN_RESPONSE_LENGTH = 256_000  # the run's response tokens (MAX_RESPONSE)
ARM_RESPONSE_BUDGET = 160_000  # one family's, below the run's
SAMPLE_LIMIT = 16  # verl's data.*_max_samples, which would cut an episode file's episodes
# what verl passes with every row: its prompt group's uid and the training step
ROW = {"uid": "the row's prompt group", "global_steps": 3}


class FamilyTool:
    """An environment tool as PearAgentLoop.run sees it: records the episodes it closes."""

    env_class = types.SimpleNamespace(task_name=lambda env_args: env_args["task"])

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}  # the tool config's entries, as EnvironmentTool keeps them
        self.closed: list[Episode] = []

    async def close(self, episode: Episode) -> None:
        self.closed.append(episode)


def test_a_trajectory_plays_its_episode_in_the_tool_its_row_selects(monkeypatch):
    played_tools = []

    async def base_run(loop, sampling_params, **kwargs):
        played_tools.append(loop.tool)
        loop.end_reason = EndReason.ENV
        return types.SimpleNamespace(reward_score=None, extra_fields={})

    monkeypatch.setattr(ToolAgentLoop, "run", base_run)
    # the progress tracker is a Ray actor: a report from here would start a Ray cluster in the test's process
    monkeypatch.setattr(ProgressTracker, "report_trajectory", staticmethod(lambda record: None))
    loop = object.__new__(PearAgentLoop)
    loop.response_length = RUN_RESPONSE_LENGTH
    arm_tool, table_tool = FamilyTool(), FamilyTool()
    loop.tools = {ARM_ENV: arm_tool, TABLE_ENV: table_tool}

    extra_info = {"tool_selection": [TABLE_ENV], "env": json.dumps({"task": "wipe_table"})}
    asyncio.run(loop.run({}, extra_info=extra_info, **ROW))

    assert played_tools == [table_tool] and arm_tool.closed == []
    assert [episode.env_args for episode in table_tool.closed] == [{"task": "wipe_table"}]


def test_the_episodes_of_several_families_make_one_dataset_each_row_selecting_its_env_class(tmp_path):
    arm_spec, table_spec = tmp_path / "arm.json", tmp_path / "table.json"
    arm_spec.write_text(
        json.dumps({"env_class": ARM_ENV, "data_source": "task", "env": {"task": ["reach"], "trial": "0-2"}})
    )
    table_spec.write_text(json.dumps({"env_class": TABLE_ENV, "data_source": "task", "env": {"task": ["wipe", "set"]}}))
    dataset = object.__new__(EpisodeDataset)
    dataset.data_files, dataset.max_samples = [str(arm_spec), str(table_spec), str(table_spec)], -1  # the table's twice

    dataset._read_files_and_tokenize()

    rows = list(dataset.dataframe)
    assert [(row["data_source"], row["extra_info"]["tool_selection"]) for row in rows] == (
        [("reach", [ARM_ENV])] * 3 + [("wipe", [TABLE_ENV]), ("set", [TABLE_ENV])] * 2
    )
    # each row exactly its episode file's env args, none of the other family's keys
    assert [json.loads(row["extra_info"]["env"]) for row in rows[:4]] == (
        [{"task": "reach", "trial": t} for t in range(3)] + [{"task": "wipe"}]
    )


def test_a_trajectory_takes_its_tool_s_response_budget_and_the_run_s_otherwise(monkeypatch):
    budgets = []

    async def base_run(loop, sampling_params, **kwargs):
        budgets.append(loop.response_budget)
        loop.end_reason = EndReason.ENV
        return types.SimpleNamespace(reward_score=None, extra_fields={})

    monkeypatch.setattr(ToolAgentLoop, "run", base_run)
    monkeypatch.setattr(ProgressTracker, "report_trajectory", staticmethod(lambda record: None))
    loop = object.__new__(PearAgentLoop)
    loop.response_length = RUN_RESPONSE_LENGTH
    # the tool config's value is the env var's string (oc.env), and null where the family sets none
    loop.tools = {
        ARM_ENV: FamilyTool({"response_budget": str(ARM_RESPONSE_BUDGET)}),
        TABLE_ENV: FamilyTool({"response_budget": None}),
    }

    for env_class in (ARM_ENV, TABLE_ENV):
        extra_info = {"tool_selection": [env_class], "env": json.dumps({"task": "reach"})}
        asyncio.run(loop.run({}, extra_info=extra_info, **ROW))

    assert budgets == [ARM_RESPONSE_BUDGET, RUN_RESPONSE_LENGTH]


def test_a_dataset_refuses_a_sample_limit():
    # an episode file says itself how many episodes it has
    dataset = object.__new__(EpisodeDataset)
    dataset.data_files, dataset.max_samples = [], SAMPLE_LIMIT

    with pytest.raises(ValueError, match="max_samples is not supported"):
        dataset._read_files_and_tokenize()
