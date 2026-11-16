import asyncio
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from pear.agent_loop.protocol import InvalidAction
from pear.envs.libero.env import LiberoEnv
from pear.envs.tool import EnvironmentTool
from verl.experimental.agent_loop.tool_agent_loop import AgentState


class Client:
    def __init__(self, env_class):
        pass

    async def open(self, uid, env_args):
        assert env_args["policy_finish"] is True
        return {
            "images": [Image.new("RGB", (1, 1)) for _ in LiberoEnv.CAMERAS],
            "state": {},
            "actions": [],
            "rewards": [],
            "done": False,
            "executed": 0,
            "outcome": None,
            "instruction": "Place the bowl on the plate.",
        }

    async def step(self, uid, action):
        assert action == {"finish": True}
        return {
            "images": None,
            "state": {},
            "actions": [],
            "rewards": [],
            "done": True,
            "executed": 0,
            "outcome": "finish",
            "instruction": "Place the bowl on the plate.",
            "task_success": True,
        }


def test_a_policy_finish_reaches_libero_and_terminates_the_agent_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(LiberoEnv, "CLIENT_CLASS_PATH", f"{__name__}.{Client.__qualname__}")
    loop = object.__new__(PearAgentLoop)
    loop.tool = EnvironmentTool(
        {"env_class": "pear.envs.libero.env.LiberoEnv", "log_dir": str(tmp_path), "policy_finish": True}
    )
    loop.episode = Episode({"suite": "libero_goal", "task_id": 3, "trial_id": 0})
    loop.end_reason = None
    agent_data = SimpleNamespace(metrics={}, tool_calls=[SimpleNamespace(arguments=json.dumps({"finish": True}))])
    observation, initial = asyncio.run(loop.tool.open(loop.episode))
    prompt = loop.tool.system_prompt(initial)
    assert '{"finish": true}' in prompt
    assert '{"finish": true}' in observation[0]["text"]
    assert "the episode ends by itself" not in prompt

    state = asyncio.run(loop._handle_processing_tools_state(agent_data))

    assert (state, loop.end_reason) == (AgentState.TERMINATED, EndReason.FINISH)
    assert (loop.episode.done, loop.episode.success, loop.episode.actions) == (True, True, [])


@pytest.mark.parametrize("action", [{"finish": 1}, {"finish": True, "arm": {}}])
def test_malformed_finish_stays_an_invalid_action(tmp_path, monkeypatch, action):
    monkeypatch.setattr(LiberoEnv, "CLIENT_CLASS_PATH", f"{__name__}.{Client.__qualname__}")
    tool = EnvironmentTool(
        {"env_class": "pear.envs.libero.env.LiberoEnv", "log_dir": str(tmp_path), "policy_finish": True}
    )

    with pytest.raises(InvalidAction):
        asyncio.run(tool.step(Episode({}), action))
