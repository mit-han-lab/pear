from __future__ import annotations

import json
import re
import sys
from typing import Any


class JsonBlockProtocol:
    """The conversation a family holds with the model, the same in a simulated episode and on a robot: its system
    prompt, a turn's text and the action an answer's json block gives."""

    JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)  # a fenced json block and its object

    @staticmethod
    def system_prompt(template: str, environment: str, instruction: str) -> str:
        """Returns ``template`` with the family's facts as its ENVIRONMENT section (no section without facts) and the
        episode's task."""
        section = f"## ENVIRONMENT\n\n{environment}\n\n" if environment else ""

        return template.replace("{environment}", section).replace("{instruction}", instruction)

    @staticmethod
    def turn_text(state: dict[str, dict[str, float]] | None, reminder: str) -> str:
        """Returns the text of a user turn: the parts' quantities as JSON, then ``reminder``."""
        return json.dumps(state) + "\n\n" + reminder

    @classmethod
    def parse(cls, text: str, think_end: str | None = None) -> str:
        """The last JSON object after the model's thinking, kept as text for the tool adapter."""
        answer = text.rpartition(think_end)[2] if think_end else text
        blocks = cls.JSON_BLOCK_PATTERN.findall(answer)
        if not blocks:
            raise NoToolCall("no ```json block found", text, answer)
        try:
            json.loads(blocks[-1])
        except json.JSONDecodeError as exc:
            raise NoToolCall(
                f"the ```json block is not valid JSON ({exc.msg} at line {exc.lineno})", text, answer
            ) from None
        return blocks[-1]

    @staticmethod
    def check_action(action: Any, schema: dict[str, tuple[str, ...]]) -> None:
        """Raises InvalidAction unless ``action`` gives every quantity of every part of ``schema`` as a finite
        number."""
        if not isinstance(action, dict) or set(action) != set(schema):
            raise InvalidAction(f"the json block must map exactly the parts {', '.join(schema)} to their quantities")

        for part, quantities in action.items():
            if not isinstance(quantities, dict) or set(quantities) != set(schema[part]):
                raise InvalidAction(f"{part!r} must give exactly its quantities {', '.join(schema[part])}")
            for name, value in quantities.items():
                # NaN and inf fail the bound, as an int past every float does (a bool is no number here)
                if type(value) not in (int, float) or not abs(value) <= sys.float_info.max:
                    raise InvalidAction(f"{part}.{name} must be a finite number")


class InvalidAction(ValueError):
    """Raised for an answer whose json block is no action of the family's schema; its message is the reason."""


class NoToolCall(ValueError):
    """An answer without a usable JSON block, with its original text and the answer after thinking."""

    def __init__(self, reason: str, text: str, answer: str) -> None:
        super().__init__(reason)
        self.text = text
        self.answer = answer
