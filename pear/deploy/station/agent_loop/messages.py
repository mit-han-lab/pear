from pathlib import Path

import numpy as np
from PIL import Image

from pear.agent_loop.protocol import JsonBlockProtocol
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput, ReferenceImage
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.schemas import ActionFormat, Observation

JPEG_QUALITY = 90  # of every image a model is sent: a turn's camera frames and a task's reference images


class SystemPrompt:
    """The station's system prompt: its template with the environment of an action format and the task."""

    # read at each render: {environment} takes the action format's environment as an ENVIRONMENT section,
    # {instruction} the task
    TEMPLATE_PATH = Path(__file__).with_name("system_prompt.md")

    @classmethod
    def render(cls, instruction: str, action_format: ActionFormat) -> str:
        """Returns the system prompt of the task ``instruction`` in ``action_format``."""
        template = cls.TEMPLATE_PATH.read_text()

        return JsonBlockProtocol.system_prompt(template, action_format.environment_prompt, instruction)


class ObservationMessage:
    """What the model is shown each turn: the measured state, the last action's result and the camera frames."""

    @staticmethod
    def build(
        observation: Observation,
        state: StrictModel,
        action_format: ActionFormat,
        directory: Path,
        attempt_id: str,
        feedback: str | None,
        reference_images: tuple[ReferenceImage, ...],
    ) -> ModelInput:
        """Returns the turn's text and images for the model, its camera frames saved as JPEG files in
        ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        decimals = action_format.decimals
        # + 0.0 makes a quantity rounded to -0.0 read 0.0
        shown_state = {
            part: {name: round(float(value), decimals[name]) + 0.0 for name, value in quantities.items()}
            for part, quantities in state.model_dump().items()
        }
        frames_phrase = "frame follows" if len(action_format.cameras) == 1 else "frames follow"
        reminder = (
            f"The current camera {frames_phrase}. End your response with one json block containing the next action."
        )
        text = JsonBlockProtocol.turn_text(shown_state, reminder)
        if feedback:
            text += "\n\nPrevious execution: " + feedback

        camera_images = []
        for camera in action_format.cameras:
            rgb = np.asarray(observation["images"][camera.role])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Expected a uint8 RGB image from the {camera.name} camera")
            timestamps = observation["camera_timestamps"][camera.role]
            if timestamps["received_monotonic_s"] > observation["captured_monotonic_s"]:
                raise ValueError("Image cannot postdate observation capture")
            path = directory / f"{attempt_id}-{camera.name}.jpg"
            Image.fromarray(rgb).save(path, "JPEG", quality=JPEG_QUALITY)
            camera_images.append(CameraImage(camera.name, path))

        return ModelInput(text, tuple(camera_images), reference_images)
