import asyncio
from types import SimpleNamespace

import pytest

from pear.agent_loop.tool_parser import JsonBlockToolParser, NoToolCall
from pear.models.qwen3_5.tool_parser import Qwen3_8ToolParser
from verl.experimental.agent_loop.tool_parser import ToolParser

TOOLS = [SimpleNamespace(function=SimpleNamespace(name="act"))]
ACTION = '{"x": 1.5}'


class CharacterTokenizer:
    """A tokenizer whose ids are the code points of the text."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(code_point) for code_point in ids)


class Answer:
    """A generated text as a parser receives it, as token ids."""

    @staticmethod
    def parsed(parser_class: type[JsonBlockToolParser], text: str) -> tuple[str, list]:
        return asyncio.run(parser_class(CharacterTokenizer()).parse([ord(character) for character in text], TOOLS))


def test_the_format_qwen3_8_builds_the_familys_parser():
    # the registration runs on import of the family's module, which the recipes have verl import in every process
    assert type(ToolParser.get_tool_parser("qwen3_8", CharacterTokenizer())) is Qwen3_8ToolParser


def test_the_call_is_the_last_json_block_after_the_thinking():
    text = f'<think>```json\n{{"x": 0}}\n```</think>first ```json\n{{"x": 0.5}}\n``` then ```json\n{ACTION}\n```'

    decoded, calls = Answer.parsed(Qwen3_8ToolParser, text)

    assert decoded == text
    assert [(call.name, call.arguments) for call in calls] == [("act", ACTION)]


def test_a_block_inside_the_thinking_is_no_answer_of_the_family():
    text = f"<think>```json\n{ACTION}\n```</think>no block here"

    with pytest.raises(NoToolCall, match="no ```json block found") as raised:
        Answer.parsed(Qwen3_8ToolParser, text)

    assert raised.value.answer == "no block here"
    # without a family's marker the whole text is the answer
    assert Answer.parsed(JsonBlockToolParser, text)[1][0].arguments == ACTION


def test_an_invalid_json_block_gives_the_reason_and_no_call():
    text = '</think>```json\n{"x": }\n```'

    with pytest.raises(NoToolCall, match="not valid JSON"):
        Answer.parsed(Qwen3_8ToolParser, text)

    parser = Qwen3_8ToolParser(CharacterTokenizer())
    assert asyncio.run(parser.extract_tool_calls([ord(character) for character in text], TOOLS)) == (text, [])
