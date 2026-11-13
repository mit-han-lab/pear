from __future__ import annotations

import json

from pear.agent_loop.protocol import JsonBlockProtocol
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
        answer = text.rpartition(self.THINK_END)[2] if self.THINK_END else text
        blocks = JsonBlockProtocol.JSON_BLOCK_PATTERN.findall(answer)
        if not blocks:
            raise NoToolCall("no ```json block found", text, answer)

        try:
            json.loads(blocks[-1])  # the call carries the block's text: here it only has to parse
        except json.JSONDecodeError as exc:
            raise NoToolCall(
                f"the ```json block is not valid JSON ({exc.msg} at line {exc.lineno})", text, answer
            ) from None

        return text, [FunctionCall(name=tools[0].function.name, arguments=blocks[-1])]


class NoToolCall(ValueError):
    """Raised for an answer without a usable ```json block; its message is the reason, ``text`` what the model
    generated and ``answer`` what follows its thinking."""

    def __init__(self, reason: str, text: str, answer: str) -> None:
        super().__init__(reason)
        self.text = text
        self.answer = answer
