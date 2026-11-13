import asyncio
import json
import types

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from verl.experimental.agent_loop.tool_agent_loop import AgentState

EPISODE_ACTIONS = 100  # the actions of a long episode, which the environment alone ends


class FakeTool:
    """The environment: each action executes one native step; the episode ends with the ``done_at``-th action."""

    def __init__(self, done_at: int) -> None:
        self.done_at = done_at
        self.calls = 0

    async def step(self, episode, action: dict) -> tuple[str, dict]:
        self.calls += 1
        done = self.calls >= self.done_at
        return "the observation", {"actions": [[0.0] * 7], "rewards": [float(done)], "done": done}


class EpisodeHarness:
    """A PearAgentLoop's tool turns without tokenizer or server: every action goes to FakeTool, generating resumes at once."""

    def __init__(self, done_at: int) -> None:
        self.loop = object.__new__(PearAgentLoop)
        self.loop.tool = FakeTool(done_at)
        self.loop.episode = Episode({"suite": "libero_10", "task_id": 0, "trial_id": 0})
        self.loop.end_reason = None
        self.loop._resume_generating = self.resume
        self.agent_data = types.SimpleNamespace(
            metrics={}, tool_calls=[types.SimpleNamespace(arguments=json.dumps({}))]
        )

    async def resume(self, agent_data, message: dict) -> AgentState:
        return AgentState.GENERATING

    def run(self, actions: int) -> list[AgentState]:
        return [asyncio.run(self.loop._handle_processing_tools_state(self.agent_data)) for _ in range(actions)]


def test_only_the_environment_ends_an_episode_however_many_actions_it_takes():
    harness = EpisodeHarness(done_at=EPISODE_ACTIONS)

    states = harness.run(EPISODE_ACTIONS)

    assert states[:-1] == [AgentState.GENERATING] * (EPISODE_ACTIONS - 1)
    assert states[-1] == AgentState.TERMINATED
    assert harness.loop.end_reason is EndReason.ENV
    assert harness.loop.episode.num_chunks == EPISODE_ACTIONS


@pytest.mark.parametrize(
    ("done", "rewards", "is_success"),
    [(True, [0.0, 1.0], True), (True, [0.0, 0.0], False), (False, [1.0], False), (False, [], False)],
)
def test_an_episode_succeeds_when_the_action_that_ends_it_earned_a_reward(done, rewards, is_success):
    episode = Episode({"suite": "libero_10", "task_id": 0, "trial_id": 0})

    episode.update({"actions": [[0.0] * 7] * len(rewards), "rewards": rewards, "done": done})

    assert (episode.done, episode.success) == (done, is_success)
