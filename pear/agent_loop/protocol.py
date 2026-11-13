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
