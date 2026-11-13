import json
import math
from typing import Any

from pear.agent_loop.protocol import JsonBlockProtocol
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.schemas import ActionFormat


class ActionParser:
    """Reads the action from a complete final answer."""

    THINK_START = "<think>"  # what opens a model's thinking: an answer that holds it is refused

    @classmethod
    def parse(cls, answer: str, action_format: ActionFormat) -> StrictModel:
        """Returns the action of the json block that ends ``answer``, as the format's action model; raises ValueError
        for an answer that holds a thinking block, ends otherwise or gives no action of the format."""
        if cls.THINK_START in answer:
            raise ValueError("Final answer unexpectedly contains a thinking block")
        blocks = list(JsonBlockProtocol.JSON_BLOCK_PATTERN.finditer(answer))
        if not blocks or answer[blocks[-1].end() :].strip():
            raise ValueError("Final answer must end with a fenced json action")

        action_fields = json.loads(blocks[-1].group(1), object_pairs_hook=cls._unique_object)
        schema = action_format.schema
        if not isinstance(action_fields, dict) or set(action_fields) != set(schema):
            raise ValueError("Expected exactly the parts " + ", ".join(schema))
        for part, names in schema.items():
            quantities = action_fields[part]
            if not isinstance(quantities, dict) or set(quantities) != set(names):
                raise ValueError(f"Invalid quantities for {part}")
            if any(
                type(quantity) not in (int, float) or not math.isfinite(quantity) for quantity in quantities.values()
            ):
                raise ValueError("Action quantities must be finite numbers, not booleans")

        return action_format.action_model.model_validate(action_fields)

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Returns the JSON object of ``pairs``; raises ValueError for a key it holds twice."""
        unique_object = {}
        for key, item in pairs:
            if key in unique_object:
                raise ValueError(f"Duplicate action field: {key}")
            unique_object[key] = item

        return unique_object
