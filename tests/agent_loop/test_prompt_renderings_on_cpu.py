import asyncio
import pkgutil
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import Episode
from pear.envs.tool import EnvironmentTool
from repository import REPO
from stored_text import StoredText
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

ENVS = REPO / "pear" / "envs"
FAMILIES = sorted(path.parents[1].name for path in ENVS.glob("*/configs/tool_config.yaml"))
RENDERINGS = Path(__file__).with_name("prompt_renderings")  # the stored first prompts, <name>.txt
# set: a test writes its stored rendering instead of comparing with it, for a prompt change that is meant
UPDATE_VARIABLE = "PEAR_UPDATE_PROMPT_RENDERINGS"
INSTRUCTION = "put the black bowl on the plate"  # the task of the simulator's first answer
FRAME_SIZE = (8, 8)  # of the first answer's frames: a rendering shows a frame as its block's header alone
# the one family whose facts depend on the first answer: an episode under a perturbed camera gets other words
PERTURBED_CAMERA_FAMILY = "libero_plus"
# a rendering's name, its family and whether its episode's camera is the one the family's facts describe
CASES = [
    *((family, family, True) for family in FAMILIES),
    (f"{PERTURBED_CAMERA_FAMILY}_perturbed_camera", PERTURBED_CAMERA_FAMILY, False),
]


class FirstObservationClient:
    """A client that answers an episode's open with a fixed first observation: the instruction, the robot's quantities
    at zero and a frame per camera."""

    standard_camera = True  # of the episodes it opens

    def __init__(self, env_class: type) -> None:
        self.env_class = env_class

    async def open(self, uid: str, env_args: dict) -> dict:
        return {
            "instruction": INSTRUCTION,
            "state": {part: dict.fromkeys(quantities, 0.0) for part, quantities in self.env_class.SCHEMA.items()},
            "images": [Image.new("RGB", FRAME_SIZE)] * len(self.env_class.CAMERAS),
            "actions": [],
            "rewards": [],
            "done": False,
            "executed": 0,
            "outcome": None,
            "standard_camera": self.standard_camera,
            # what the family's call log records besides the protocol's keys
            **dict.fromkeys(self.env_class.LOG_KEYS),
        }


class FirstPrompt:
    """A family's first prompt as PearAgentLoop builds it: its tool on a client that gives the fixed first observation,
    and what the loop then hands verl's prompt builder."""

    def __init__(self, family: str, standard_camera: bool, tmp_path: Path, monkeypatch) -> None:
        # the family's env class, by its shipped tool config, whose environment variables stay unresolved
        tool_config = OmegaConf.load(ENVS / family / "configs" / "tool_config.yaml")
        [tool] = OmegaConf.to_container(tool_config, resolve=False)["tools"]
        env_class_path = tool["config"]["env_class"]
        client_class_path = f"{FirstObservationClient.__module__}.{FirstObservationClient.__qualname__}"
        monkeypatch.setattr(pkgutil.resolve_name(env_class_path), "CLIENT_CLASS_PATH", client_class_path)
        monkeypatch.setattr(FirstObservationClient, "standard_camera", standard_camera)
        # verl's prompt building is where the first prompt leaves PEAR's code
        monkeypatch.setattr(ToolAgentLoop, "_handle_pending_state", self.record)

        self.messages: list[dict] = []  # the conversation verl would render
        self.tool_schemas: list[dict] | None = None  # the tools the loop has verl render with it
        self.loop = object.__new__(PearAgentLoop)
        self.loop.tool = EnvironmentTool({"env_class": env_class_path, "log_dir": str(tmp_path)})
        self.loop.tools = {self.loop.tool.name: self.loop.tool}
        self.loop.episode = Episode({})
        self.loop.tokenizer = types.SimpleNamespace(decode=lambda token_ids: "")

        asyncio.run(self.open())

    async def open(self) -> None:
        """The loop's first state: the episode's open and the first prompt."""
        self.loop.loop = asyncio.get_running_loop()  # verl's constructor takes the loop it runs on

        await self.loop._handle_pending_state(types.SimpleNamespace(messages=[]), {})

    async def record(self, agent_data, sampling_params: dict) -> AgentState:
        """In place of verl's prompt building: keeps what it would render."""
        self.messages = agent_data.messages
        self.tool_schemas = self.loop._prompt_tool_schemas(agent_data)
        agent_data.prompt_ids = []

        return AgentState.GENERATING

    def text(self) -> str:
        """The conversation as text: a line per turn, then its text or a line per block with the block's text."""
        lines = []

        for message in self.messages:
            lines.append(f"==== {message['role']} ====")
            if isinstance(message["content"], str):
                lines.append(message["content"])
                continue
            for block in message["content"]:
                lines.append(f"---- {block['type']} ----")
                if block["type"] == "text":
                    lines.append(block["text"])

        return "\n".join(lines) + "\n"


@pytest.mark.parametrize(("name", "family", "standard_camera"), CASES, ids=[name for name, _, _ in CASES])
def test_a_family_s_first_prompt_is_its_stored_rendering(name, family, standard_camera, tmp_path, monkeypatch):
    stored_rendering = RENDERINGS / f"{name}.txt"

    prompt = FirstPrompt(family, standard_camera, tmp_path, monkeypatch)

    # the system prompt carries the whole protocol: no tool schema goes to the model with it
    assert prompt.tool_schemas == []
    StoredText.assert_matches(stored_rendering, prompt.text(), UPDATE_VARIABLE)
