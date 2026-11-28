from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from pear.envs.session import EnvironmentSession
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
from verl.utils.import_utils import resolve_config_path


class EnvironmentTool(EnvironmentSession, BaseTool):
    """An EnvironmentSession exposed as a verl tool, named by the env class its tool config selects."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None) -> None:
        # one instance per agent-loop process (PEAR's trainer config keeps verl's dataset from building the tools on
        # the driver); the loop's episodes run on its client
        EnvironmentSession.__init__(self, config)
        BaseTool.__init__(self, config, tool_schema)

    @staticmethod
    def load_configs(tool_config_path: str) -> list[dict[str, Any]]:
        """Returns the config of every tool of a tool config file, the variables it names resolved in this process."""
        tools = OmegaConf.load(resolve_config_path(tool_config_path)).tools

        return [OmegaConf.to_container(tool.config, resolve=True) for tool in tools]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # verl's BaseTool requires a named schema and keys its tools by that name: the tool config's env class, by
        # which a row selects its tool (EpisodeDataset); the model never sees the schema (the system prompt carries the
        # protocol)
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name=self.config["env_class"],
                description="Execute the next action and return the observation.",
                parameters=OpenAIFunctionParametersSchema(type="object", properties={}),
            ),
        )
