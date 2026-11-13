from __future__ import annotations

import asyncio
import json
import os

import pytest
from PIL import Image

from pear.agent_loop.episode import Episode
from pear.agent_loop.protocol import InvalidAction
from pear.envs.libero.env import LiberoEnv
from pear.envs.libero_plus.env import LiberoPlusEnv
from pear.envs.libero_pro.env import LiberoProEnv
from pear.envs.robodojo.env import RoboDojoEnv
from pear.envs.robolab.env import RoboLabEnv
from pear.envs.tool import EnvironmentTool
from pear.sim.env import BaseEnv, EpisodeLost

ENVS = [LiberoEnv, LiberoPlusEnv, LiberoProEnv, RoboDojoEnv, RoboLabEnv]
# env args with what every family's task name reads
ENV_ARGS = {
    "suite": "libero_goal",
    "task_id": 3,
    "category": "camera",
    "perturbation": "lan",
    "task_name": "stack_bowls",
    "task": "BananaInBowlTask",
}


class FakeClient:
    """A client as the tool sees it: records its calls; each step executes two native actions and the episode ends at
    the second; an episode is lost (as with a crashed simulator process) where a test sets ``lost_at``."""

    lost_at = None  # "open" or "step"

    def __init__(self, env_class: type[BaseEnv]) -> None:
        self.cameras, self.calls = env_class.CAMERAS, []

    async def open(self, uid: str, env_args: dict):
        self.calls.append("open")
        return self._result(None)

    async def step(self, uid: str, action):
        self.calls.append("step")
        return self._result(action)

    def _result(self, action):
        if self.lost_at == self.calls[-1]:
            raise EpisodeLost("the server process crashed")
        done, actions = self.calls.count("step") == 2, [] if action is None else [action] * 2
        rewards = [0.0, float(done)] if actions else []  # the task done at the episode's last native action
        return {
            "images": None if done else [Image.new("RGB", (6, 4))] * len(self.cameras),
            "state": {"arm": {"x": 1.0}},
            "actions": actions,
            "rewards": rewards,
            "done": done,
            "executed": len(actions),
            "outcome": None if action is None else "reached",
            "instruction": "stack the bowls",
            "env_end_reason": "success" if done else None,
            "standard_camera": True,
            "subtask_score": None,
            "subtask_info": None,
        }


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    """Gives every family's env class FakeClient as its kind's client."""
    for env_class in ENVS:
        monkeypatch.setattr(env_class, "CLIENT_CLASS_PATH", f"{FakeClient.__module__}.{FakeClient.__qualname__}")


class TrajectoryHarness:
    """The environment tool of a family's env class on a fake client, called as PearAgentLoop calls it for one
    trajectory."""

    def __init__(self, env_class: type[BaseEnv], tmp_path) -> None:
        self.tool = EnvironmentTool(
            {"env_class": f"{env_class.__module__}.{env_class.__qualname__}", "log_dir": str(tmp_path)}
        )
        self.zero = {part: dict.fromkeys(quantities, 0.0) for part, quantities in env_class.SCHEMA.items()}
        self.log = tmp_path / f"actions_{os.getpid()}.jsonl"

    def play(self, *actions) -> list[tuple]:
        """The answers to the open of a new episode of ENV_ARGS and to ``actions``."""
        return asyncio.run(self._play(actions))

    async def _play(self, actions):
        self.episode = Episode(ENV_ARGS)
        answers = [await self.tool.open(self.episode)]
        for action in actions:
            answers.append(await self.tool.step(self.episode, action))
            self.episode.update(answers[-1][1])  # as PearAgentLoop records a model's action
        return answers


@pytest.mark.parametrize("env_class", ENVS)
def test_a_trajectory_s_answers_open_and_step_its_episode(env_class, tmp_path):
    harness = TrajectoryHarness(env_class, tmp_path)

    [(content, opened), (_, stepped), (ended, _)] = harness.play(harness.zero, harness.zero)

    assert harness.tool.client.calls == ["open", "step", "step"]
    assert content[0]["text"] == json.dumps({"arm": {"x": 1.0}}) + "\n\n" + EnvironmentTool.REMINDER
    assert [block["text"] for block in content[1::2]] == [f"[{name}]" for name in env_class.CAMERAS]
    assert all(isinstance(block["image"], Image.Image) for block in content[2::2])
    assert stepped["actions"] == [harness.zero] * 2 and ended is None  # the episode ended: no next user turn
    # named by its config's env class, which the episodes' rows select it by (EpisodeDataset)
    assert harness.tool.name == f"{env_class.__module__}.{env_class.__qualname__}"
    assert "stack the bowls" in harness.tool.system_prompt(opened)
    records = [json.loads(line) for line in harness.log.read_text().splitlines()]
    assert [(record["steps_before"], record["executed"], record["success"]) for record in records] == [
        (0, 0, False),
        (0, 2, False),
        (2, 2, True),
    ]
    assert all(record["env"] == ENV_ARGS and set(env_class.LOG_KEYS) <= set(record) for record in records)
    # every call carries its episode's uid, which tells a task's concurrent episodes apart (the val dump's episode uid)
    assert {record["uid"] for record in records} == {harness.episode.uid}


@pytest.mark.parametrize("env_class", ENVS)
def test_an_unusable_answer_raises_its_reason_and_never_reaches_the_client(env_class, tmp_path):
    zero = TrajectoryHarness(env_class, tmp_path).zero
    part, quantity = next(iter(env_class.SCHEMA.items()))[0], env_class.SCHEMA[next(iter(env_class.SCHEMA))][0]
    unusable = {
        "must map exactly the parts": [[zero], {p: v for p, v in zero.items() if p != part}],
        "must give exactly its quantities": [{**zero, part: {}}, {**zero, part: 1.0}],
        "must be a finite number": [
            {**zero, part: {**zero[part], quantity: value}}
            for value in (float("nan"), float("inf"), True, 10**400, "1")
        ],
    }

    for reason, actions in unusable.items():
        for action in actions:
            harness = TrajectoryHarness(env_class, tmp_path)
            with pytest.raises(InvalidAction, match=reason):
                harness.play(action)
            assert harness.tool.client.calls == ["open"]


@pytest.mark.parametrize("lost_at, logged_call_n", [("open", 0), ("step", 1)])
def test_a_call_that_loses_its_episode_leaves_no_record_in_the_call_log(monkeypatch, tmp_path, lost_at, logged_call_n):
    harness = TrajectoryHarness(RoboDojoEnv, tmp_path)
    monkeypatch.setattr(FakeClient, "lost_at", lost_at)

    with pytest.raises(EpisodeLost):  # the agent loop starts the trajectory over in a new episode
        harness.play(harness.zero)

    # the log holds the calls the environment carried out: none of a lost open, the open before a lost step
    records = harness.log.read_text().splitlines() if harness.log.exists() else []
    assert len(records) == logged_call_n


def test_a_perturbed_libero_plus_viewpoint_keeps_the_agentview_line_s_first_clause(tmp_path):
    tool = TrajectoryHarness(LiberoPlusEnv, tmp_path).tool

    standard = tool.system_prompt({"instruction": "x", "standard_camera": True})
    perturbed = tool.system_prompt({"instruction": "x", "standard_camera": False})

    assert "+x points toward the bottom of the image" in standard
    assert "- agentview: a fixed camera.\n" in perturbed and "+x points toward the bottom" not in perturbed


def test_each_family_names_the_task_of_an_episode():
    names = {env_class: env_class.task_name(ENV_ARGS) for env_class in ENVS}

    assert names == {
        LiberoEnv: "libero_goal/task_3",
        LiberoPlusEnv: "libero_goal/task_3/camera",
        LiberoProEnv: "libero_goal/task_3/lan",
        RoboDojoEnv: "stack_bowls",
        RoboLabEnv: "BananaInBowlTask",
    }
