import asyncio
import json
import logging
import types

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from pear.envs.libero.env import LiberoEnv
from pear.envs.tool import EnvironmentTool
from verl.experimental.agent_loop.tool_agent_loop import AgentState

ENV_ARGS = {"suite": "libero_goal", "task_id": 3, "trial_id": 0}
ZERO = {"arm": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, "gripper": {"width": 0.0}}
# json blocks that are no action of LIBERO's schema, each with the tool's reason
INVALID_ACTIONS = [
    ([ZERO], "the json block must map exactly the parts arm, gripper to their quantities"),
    ({"arm": ZERO["arm"]}, "the json block must map exactly the parts arm, gripper to their quantities"),
    ({**ZERO, "gripper": {}}, "'gripper' must give exactly its quantities width"),
    ({**ZERO, "arm": 1.0}, "'arm' must give exactly its quantities x, y, z, roll, pitch, yaw"),
    ({**ZERO, "gripper": {"width": float("nan")}}, "gripper.width must be a finite number"),
    ({**ZERO, "arm": {**ZERO["arm"], "yaw": "1"}}, "arm.yaw must be a finite number"),
]


class UnreachedClient:
    """A client of an episode whose simulator no call may reach."""

    def __init__(self, env_class: type) -> None:
        self.env_class = env_class

    async def step(self, uid: str, action) -> dict:
        raise AssertionError(f"the action {action} reached the simulator")


@pytest.fixture
def loop(tmp_path, monkeypatch) -> PearAgentLoop:
    """A PearAgentLoop in an open episode of LIBERO's tool, without tokenizer or server."""
    monkeypatch.setattr(LiberoEnv, "CLIENT_CLASS_PATH", f"{__name__}.{UnreachedClient.__qualname__}")
    loop = object.__new__(PearAgentLoop)
    loop.tool = EnvironmentTool({"env_class": "pear.envs.libero.env.LiberoEnv", "log_dir": str(tmp_path)})
    loop.episode = Episode(ENV_ARGS)
    loop.end_reason = None

    return loop


@pytest.mark.parametrize(("action", "reason"), INVALID_ACTIONS)
def test_an_invalid_action_ends_the_episode_as_a_parse_failure_with_the_tool_s_reason(
    loop, tmp_path, monkeypatch, caplog, action, reason
):
    # the loop's logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(logging.getLogger("pear.agent_loop.agent_loop"), "handlers", [caplog.handler])
    agent_data = types.SimpleNamespace(metrics={}, tool_calls=[types.SimpleNamespace(arguments=json.dumps(action))])

    state = asyncio.run(loop._handle_processing_tools_state(agent_data))

    assert (state, loop.end_reason) == (AgentState.TERMINATED, EndReason.PARSE)
    assert caplog.messages == [f"PARSE_FAILURE env={ENV_ARGS} error={reason!r}"]
    # the answer is no action of the episode: nothing of it is accounted, and the call log has no line of it
    episode = loop.episode
    assert (episode.num_chunks, episode.actions, episode.chunk_rewards) == (0, [], [])
    assert (episode.done, episode.success) == (False, False)
    assert list(agent_data.metrics) == ["tool_calls"]
    assert list(tmp_path.iterdir()) == []
