import asyncio
import types

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

RESPONSE_LENGTH = 100_000  # the episode's token budget
USED = 40_000  # response tokens of the earlier turns
TURN_TOKENS = 32_769  # the tokens of one long turn, within the budget that remains


class FakeGeneration:
    """The base class's generation step: records the allowance it got and generates a set number of tokens."""

    def __init__(self, tokens: int | None) -> None:
        self.tokens = tokens  # None: all of the allowance
        self.max_tokens = None

    async def __call__(self, agent_data, sampling_params, ignore_termination=False) -> AgentState:
        self.max_tokens = sampling_params["max_tokens"]
        generated = self.max_tokens if self.tokens is None else self.tokens
        agent_data.response_ids = [1] * generated
        agent_data.response_mask = agent_data.response_mask + [1] * generated
        return AgentState.TERMINATED


class TurnHarness:
    """One generating turn of a PearAgentLoop without tools, tokenizer or server: only the turn's budgeting runs."""

    def __init__(self, monkeypatch, tokens: int | None) -> None:
        monkeypatch.setenv("PEAR_LOG_TURNS", "0")  # the turn log needs the tokenizer
        self.generation = FakeGeneration(tokens)
        monkeypatch.setattr(ToolAgentLoop, "_handle_generating_state", self.generation)

        self.loop = object.__new__(PearAgentLoop)
        self.loop.response_budget = RESPONSE_LENGTH
        self.loop.end_reason = None
        self.agent_data = types.SimpleNamespace(response_mask=[1] * USED, response_ids=[], tool_calls=["the action"])

    def run(self) -> AgentState:
        return asyncio.run(self.loop._handle_generating_state(self.agent_data, {"temperature": 1.0}))


def test_a_turn_may_use_the_whole_remaining_budget(monkeypatch):
    harness = TurnHarness(monkeypatch, tokens=TURN_TOKENS)

    state = harness.run()

    assert harness.generation.max_tokens == RESPONSE_LENGTH - USED
    assert state == AgentState.PROCESSING_TOOLS
    assert harness.loop.end_reason is None


def test_a_turn_that_spends_the_budget_ends_the_episode_by_budget(monkeypatch):
    harness = TurnHarness(monkeypatch, tokens=None)

    state = harness.run()

    assert state == AgentState.TERMINATED
    assert harness.loop.end_reason is EndReason.BUDGET
