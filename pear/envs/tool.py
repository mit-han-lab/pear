from __future__ import annotations

import json
import os
import pkgutil
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from pear.agent_loop.episode import Episode
from pear.agent_loop.protocol import JsonBlockProtocol
from pear.sim.env import BaseEnv
from pear.sim.schemas import EnvResult
from pear.utils.file_utils import append_line
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
from verl.utils.import_utils import resolve_config_path


class EnvironmentTool(BaseTool):
    """An environment family as the agent loop sees it: the env class its tool config names (``env_class``) gives the
    protocol's system prompt its facts, and its kind's client carries out each answer's action in the trajectory's
    episode, whose result becomes the next user turn (the robot's quantities and labelled frames)."""

    # the protocol every family shares: {environment} takes the family's facts as an ENVIRONMENT section, {instruction}
    # the episode's task
    SYSTEM_PROMPT_TEMPLATE = Path(__file__).with_name("system_prompt.md").read_text()
    # repeated every turn: without it the model drops the json block
    REMINDER = "The current camera frames follow. End your response with one json block containing the next action."

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None) -> None:
        # one instance per agent-loop process (PEAR's trainer config keeps verl's dataset from building the tools on
        # the driver); the loop's episodes run on its client
        self.env_class: type[BaseEnv] = pkgutil.resolve_name(config["env_class"])  # the family's facts and simulator
        self.client = pkgutil.resolve_name(self.env_class.CLIENT_CLASS_PATH)(self.env_class)  # its kind's client
        self.log_path = Path(str(config["log_dir"])) / f"actions_{os.getpid()}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(config, tool_schema)

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

    async def open(self, episode: Episode) -> tuple[list[dict[str, Any]] | None, EnvResult]:
        """Opens ``episode`` and returns the first user turn's content with the result of its initial observation."""
        steps_before, start_time = len(episode.actions), time.monotonic()
        result = await self.client.open(episode.uid, episode.env_args)
        self._log_call(episode, steps_before, None, result, time.monotonic() - start_time)

        return self._observation(result), result

    def system_prompt(self, result: EnvResult) -> str:
        """Returns the episode's system prompt, from the result of its initial observation."""
        return JsonBlockProtocol.system_prompt(
            self.SYSTEM_PROMPT_TEMPLATE, self.env_class.environment_prompt(result), result["instruction"]
        )

    async def step(self, episode: Episode, action: Any) -> tuple[list[dict[str, Any]] | None, EnvResult]:
        """Carries out the answer's ``action`` in ``episode`` and returns the next user turn's content (None once the
        episode ended) with the result; raises InvalidAction for a json block that is no action."""
        steps_before, start_time = len(episode.actions), time.monotonic()
        JsonBlockProtocol.check_action(action, self.env_class.SCHEMA)
        result = await self.client.step(episode.uid, action)
        self._log_call(episode, steps_before, action, result, time.monotonic() - start_time)

        return self._observation(result), result

    async def close(self, episode: Episode) -> None:
        """Ends the episode on its client."""
        await self.client.close(episode.uid)

    def _log_call(self, episode: Episode, steps_before: int, action: Any, result: EnvResult, duration_s: float) -> None:
        """Writes one JSON line per call in log_dir/actions_<pid>.jsonl: the episode's uid and env args, the native
        actions before the call, the action, what it did and how long the call took (the wait for its simulator
        included)."""
        # the episodes can be read while they run; the val dump comes only at the end of a validation. The uid tells
        # apart the calls of a task's concurrent episodes (the val dump's extra_fields.episode.uid)
        record = {
            "time": round(time.time(), 1),
            "seconds": round(duration_s, 2),
            "uid": episode.uid,
            "env": episode.env_args,
            "steps_before": steps_before,
            "action": action,
            "outcome": result["outcome"],
            "executed": result["executed"],
            "done": result["done"],
            "success": Episode.is_success(result),
            "state": result["state"],
            **{key: result[key] for key in self.env_class.LOG_KEYS},
        }

        append_line(self.log_path, json.dumps(record))

    def _observation(self, result: EnvResult) -> list[dict[str, Any]] | None:
        """Returns the content of the user turn for one result (None once the episode ended): the parts' quantities
        and the reminder, then "[camera]" and the frame per camera."""
        if result["images"] is None:
            return None

        text = JsonBlockProtocol.turn_text(result["state"], self.REMINDER)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        for name, image in zip(self.env_class.CAMERAS, result["images"]):
            blocks.append({"type": "text", "text": f"[{name}]"})
            blocks.append({"type": "image", "image": image})

        return blocks
