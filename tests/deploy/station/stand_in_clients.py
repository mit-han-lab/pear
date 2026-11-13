import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient
from pear.deploy.station.agent_loop.clients.schemas import (
    EventHandler,
    HealthStatus,
    ModelEvent,
    ModelHealth,
    ModelInput,
    ModelOutput,
    ReasoningKind,
)


class ModelReply:
    """A model's answer that carries an action."""

    @staticmethod
    def output(action: Any) -> ModelOutput:
        """Returns the answer a model gives for ``action``, a dict or an action's model: a fenced json block."""
        if hasattr(action, "model_dump"):
            action = action.model_dump(exclude={"type"})

        return ModelOutput("```json\n" + json.dumps(action) + "\n```")


class DemonstrationActions:
    """The demonstration action of each action format of the robots: the arms move a little up, then down, and the
    grippers hold."""

    BOB_CM = 0.3  # how far an arm moves up or down in one demonstration action

    @classmethod
    def action(cls, format_name: str, call_count: int, is_moving: bool) -> dict[str, dict[str, float]]:
        """Returns the action of the format ``format_name`` after ``call_count`` earlier ones: up after an even
        count, down after an odd one, and no move at all unless ``is_moving``."""
        lift_cm = (cls.BOB_CM if call_count % 2 == 0 else -cls.BOB_CM) if is_moving else 0.0
        actions = {"bimanual_pose": cls.bimanual_pose, "single_arm_pose": cls.single_arm_pose}

        return actions[format_name](lift_cm)

    @staticmethod
    def bimanual_pose(lift_cm: float) -> dict[str, dict[str, float]]:
        """Returns the action of the two arms' pose format that lifts both arms by ``lift_cm``."""
        still_arm = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)

        return {
            "left_arm": {**still_arm, "z": lift_cm},
            "right_arm": {**still_arm, "z": lift_cm},
            "left_gripper": {"width": 0.0},
            "right_gripper": {"width": 0.0},
        }

    @staticmethod
    def single_arm_pose(lift_cm: float) -> dict[str, dict[str, float]]:
        """Returns the action of the one arm's pose format that lifts the arm by ``lift_cm``."""
        return {"arm": dict(x=0.0, y=0.0, z=lift_cm, pitch=0.0, roll=0.0), "gripper": {"width": 0.0}}


class StandInClient(ModelClient):
    """The base of a stand-in model, which reaches no service."""

    LABEL = "Stand-in"  # names the client in its messages

    def check(self) -> ModelHealth:
        """Returns the health of a model that has no connection to check."""
        return {"status": HealthStatus.OK.value, "message": "Stand-in model: no connection to check"}


class StandInScriptedClient(StandInClient):
    """A stand-in model that answers what a test's function returns and keeps every input it was given."""

    def __init__(
        self,
        system_prompt: str,
        answer: Callable[[ModelInput, EventHandler, threading.Event], ModelOutput],
        *,
        timeout_s: float,
    ) -> None:
        super().__init__(system_prompt, timeout_s=timeout_s)
        self.answer = answer  # gives the answer to an input, with the request's event handler and cancellation
        self.model_inputs: list[ModelInput] = []  # the input of every request, in order

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        """Returns what the test's function answers to ``model_input``."""
        self.model_inputs.append(model_input)

        # in a thread of its own: the function may wait for the test, and may ask another stand-in, whose request
        # runs an event loop of its own
        return await asyncio.to_thread(self.answer, model_input, on_event, cancellation)


class StandInDemonstrationClient(StandInClient):
    """A stand-in model that narrates three sentences and answers the demonstration action of its action format."""

    DELAY_S = 0.3  # how long one answer takes
    # the reasoning that streams before an answer, a piece at a time
    NARRATION = ("Simulation only. ", "Observe the robot. ", "Propose a finite action.")

    def __init__(
        self,
        system_prompt: str,
        *,
        timeout_s: float,
        action_format: str,
        delay_s: float = DELAY_S,
        is_moving: bool = True,
        context_limit: int | None = None,
    ) -> None:
        super().__init__(system_prompt, timeout_s=timeout_s)
        self.action_format = action_format  # the name of the action format the answers are in
        self.delay_s = delay_s
        self.is_moving = is_moving  # whether the answers move the arms
        self.context_limit = context_limit  # the answers after which the context is full; None: it never is
        self.call_count = 0  # the answers given so far

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        """Returns the next demonstration action as the model's answer, after the narration."""
        if self.context_limit is not None and self.call_count >= self.context_limit:
            raise ContextLimit("Demonstration context capacity reached")

        for text in self.NARRATION:
            # the simulated key is the stand-in's own, which the stored texts of a task's events list
            on_event(
                ModelEvent.REASONING_DELTA,
                {"text": text, "reasoning_kind": ReasoningKind.NONE.value, "simulated": True},
            )
            # a stop ends the narration; the stand-in keeps no time of its own
            await self._race(asyncio.sleep(self.delay_s / len(self.NARRATION)), cancellation, timeout_s=None)

        action = DemonstrationActions.action(self.action_format, self.call_count, self.is_moving)
        self.call_count += 1
        output = ModelReply.output(action)
        on_event(ModelEvent.ANSWER_DELTA, {"text": output.answer})

        return output
