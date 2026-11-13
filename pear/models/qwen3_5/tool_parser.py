from pear.agent_loop.tool_parser import JsonBlockToolParser
from verl.experimental.agent_loop.tool_parser import ToolParser


@ToolParser.register("qwen3_8")
class Qwen3_8ToolParser(JsonBlockToolParser):
    """PEAR's ```json answer as Qwen3.8 writes it (rollout.multi_turn.format=qwen3_8): after its last </think>."""

    THINK_END = "</think>"
