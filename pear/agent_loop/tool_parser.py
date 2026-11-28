from __future__ import annotations

from pear.agent_loop.protocol import JsonBlockProtocol
from pear.agent_loop.protocol import NoToolCall as NoToolCall
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.ray_utils import get_event_loop


class JsonBlockToolParser(ToolParser):
    """verl's tool parser for an answer that ends with one fenced ```json block, the arguments of the episode's single
    tool (no <tool_call> tags: the system prompt carries the protocol)."""

    # the marker that ends a model's thinking, set by the model family's parser: the answer is what follows the last
    # one; None: the whole text is the answer
    THINK_END: str | None = None

    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        """Returns ``(text, [the call])`` to verl's ToolAgentLoop after every generation, and ``(text, [])`` for an
        answer without a usable ```json block, whose reason ``parse`` gives."""
        try:
            return await self.parse(responses_ids, tools)
        except NoToolCall as exc:
            return exc.text, []

    async def parse(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[FunctionCall]]:
        """Returns ``(text, [the call])`` from the last ```json block of the answer and raises NoToolCall when the
        answer has no usable one."""
        text = await get_event_loop().run_in_executor(None, self.tokenizer.decode, responses_ids)
        arguments = JsonBlockProtocol.parse(text, self.THINK_END)
        return text, [FunctionCall(name=tools[0].function.name, arguments=arguments)]
