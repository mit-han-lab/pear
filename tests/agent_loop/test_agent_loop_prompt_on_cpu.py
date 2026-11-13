import asyncio
import json
import types

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from pear.monitor.progress_tracker import ProgressTracker
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema

ENV_CLASS = "tests.StackingEnv"  # the env class of the row's tool: the name of its schema (EnvironmentTool)
RESPONSE_LENGTH = 256_000  # the run's response tokens
INSTRUCTION = "stack the bowls"  # the episode's task, in the system prompt
OBSERVATION = "the first frames"  # the initial observation, the first user turn
# what verl passes with every row: its prompt group's uid and the training step
ROW = {"uid": "the row's prompt group", "global_steps": 3}


class OpeningTool:
    """The environment tool as an episode's first prompt sees it, its schema named by its env class as EnvironmentTool's."""

    env_class = types.SimpleNamespace(task_name=lambda env_args: env_args["task"])
    config = {}  # the tool config's entries: no response_budget
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=ENV_CLASS,
            description="Execute the next action and return the observation.",
            parameters=OpenAIFunctionParametersSchema(type="object", properties={}),
        ),
    )

    def __init__(self) -> None:
        self.closed: list[Episode] = []

    async def open(self, episode: Episode) -> tuple[list[dict], dict]:
        return [{"type": "text", "text": OBSERVATION}], {"instruction": INSTRUCTION}

    def system_prompt(self, result: dict) -> str:
        return f"Your task: {result['instruction']}"

    async def close(self, episode: Episode) -> None:
        self.closed.append(episode)


class ChatTemplate:
    """verl's prompt builder with a chat template that renders the tools it is given as a "# Tools" section of the system
    turn, as the models' own does; one token per character."""

    def __init__(self, rendered_tools: list[dict] | None = None) -> None:
        self.rendered_tools = rendered_tools  # the tools it renders whatever it is given (None: those it is given)
        self.given_tools: list[list[dict] | None] = []  # the tools each build was given

    async def build_initial_tokens(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> list[int]:
        self.given_tools.append(tools)

        turns = []
        for message in messages:
            content = message["content"]
            text = content if isinstance(content, str) else "".join(block["text"] for block in content)
            turns.append(f"<|im_start|>{message['role']}\n{text}")
        rendered = tools if self.rendered_tools is None else self.rendered_tools
        if rendered:
            turns[0] += "\n\n# Tools\n\n<tools>\n" + "\n".join(json.dumps(tool) for tool in rendered) + "\n</tools>"

        return [ord(char) for char in "<|im_end|>\n".join(turns)]

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(map(chr, token_ids))


class PromptHarness:
    """PearAgentLoop.run over verl's ToolAgentLoop.run with ``template`` for the model's chat template and no server: the
    episode ends at its first generation, which records the prompt."""

    def __init__(self, monkeypatch, template: ChatTemplate) -> None:
        # the progress tracker is a Ray actor: a report from here would start a Ray cluster in the test's process
        monkeypatch.setattr(ProgressTracker, "report_trajectory", staticmethod(lambda record: None))
        self.template = template
        self.prompts: list[str] = []  # the prompt of each generation

        self.loop = object.__new__(PearAgentLoop)
        self.loop.tools = {ENV_CLASS: OpeningTool()}
        # as verl's constructor leaves them, the schemas of the run's tools: PEAR's loop renders none of them
        self.loop.tool_schemas = [OpeningTool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
        self.loop.response_length = RESPONSE_LENGTH
        self.loop.processor, self.loop.mm_processor_kwargs = None, {}  # a text model's
        self.loop.rollout_config = types.SimpleNamespace(full_determinism=False)
        self.loop.tokenizer = types.SimpleNamespace(decode=template.decode)
        self.loop.ct_build_initial_tokens = template.build_initial_tokens

        async def generate(agent_data, sampling_params):
            self.prompts.append(template.decode(agent_data.prompt_ids))
            self.loop.end_reason = EndReason.BUDGET  # the loop ends no episode without the reason its record names
            return AgentState.TERMINATED

        self.loop._handle_generating_state = generate

    async def run(self) -> AgentLoopOutput:
        self.loop.loop = asyncio.get_running_loop()  # verl's constructor takes the loop it runs on
        extra_info = {"tool_selection": [ENV_CLASS], "env": json.dumps({"task": "stack_bowls"})}

        return await self.loop.run({}, raw_prompt=[], extra_info=extra_info, **ROW)


def test_the_prompt_of_a_row_that_selects_its_tool_renders_no_tool_schema(monkeypatch):
    harness = PromptHarness(monkeypatch, ChatTemplate())

    asyncio.run(harness.run())

    # verl's run selects the row's tool and its schema (extra_info.tool_selection); the template is given none of it
    [prompt] = harness.prompts
    assert harness.template.given_tools == [[]]
    assert INSTRUCTION in prompt and OBSERVATION in prompt and "# Tools" not in prompt and ENV_CLASS not in prompt


def test_a_prompt_that_renders_a_tool_schema_all_the_same_fails_the_episode(monkeypatch):
    # a template that gets the schema another way, as a verl rendering its tools from elsewhere would
    schema = OpeningTool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
    harness = PromptHarness(monkeypatch, ChatTemplate(rendered_tools=[schema]))

    with pytest.raises(RuntimeError, match="renders a tool schema"):
        asyncio.run(harness.run())

    # never generated from, and its episode closed
    assert harness.prompts == [] and len(harness.loop.tools[ENV_CLASS].closed) == 1
