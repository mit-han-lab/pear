import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, SessionLost
from pear.deploy.station.agent_loop.clients.schemas import (
    ModelEvent,
    ModelInput,
    ModelOutput,
    ModelPhase,
    ReferenceImage,
)
from pear.deploy.station.agent_loop.messages import ObservationMessage
from pear.deploy.station.agent_loop.schemas import TurnEvent, TurnPhase
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.schemas import ActionFormat, BriefStatus, ControlSnapshot, DetailedStatus, Observation
from pear.deploy.station.exceptions import OperatorPaused
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


# takes one event of a turn, the model's or the agent loop's: its kind and its data
TurnEventHandler = Callable[[ModelEvent | TurnEvent, dict[str, Any]], None]


@dataclass
class Turn:
    """One turn: the robot's observation, what the model was shown, its answer, the robot's result of its action and
    whether the action went to the robot."""

    attempt_id: str  # what the turn's events and files are known by
    observation: Observation  # the robot's observation the turn's input was made from
    model_input: ModelInput
    model_output: ModelOutput | None = None  # None until the model has answered
    station_result: MotionResult | None = None  # the robot's result of the turn's action; None until it has one
    is_dispatched: bool = False  # whether the action went to the robot


class ModelCall:
    """A model request in a worker thread, watched until its answer arrives while the arms hold the observed pose,
    settled; the operator's pause, its deadline and arms that leave the pose stop it."""

    THREAD_NAME = "agent-loop-model-call"  # of the thread that waits for the model's answer

    def __init__(
        self,
        robot: BaseRobot,
        reference_observation: Observation,
        call: Callable[[], ModelOutput],
        *,
        timeout_s: float,
        cancellation: threading.Event,
        abort: Callable[[], None],
    ) -> None:
        self.robot = robot
        self.reference_observation = reference_observation  # the arms hold its pose, settled, while the model works
        self.call = call  # makes the request and returns the model's answer
        self.timeout_s = timeout_s  # how long the model may take
        self.cancellation = cancellation  # the operator's pause
        self.abort = abort  # stops the request
        # what the worker ends with: (True, the answer) or (False, the error that ended the request)
        self._result_queue: queue.Queue[tuple[bool, ModelOutput | BaseException]] = queue.Queue(maxsize=1)
        self._received_result: tuple[bool, ModelOutput | BaseException] | None = None  # taken from the queue

    def run(self) -> ModelOutput:
        """Returns the model's answer once it has arrived while the arms still hold the reference pose, settled."""
        OperatorPaused.check_cancelled(self.cancellation)
        self.robot.wait_stationary(self.reference_observation, cancellation=self.cancellation)
        thread = threading.Thread(target=self._work, daemon=True, name=self.THREAD_NAME)
        thread.start()

        try:
            return self._monitor()
        except BaseException as error:
            self._stop(thread, error)
            raise

    def _work(self) -> None:
        """Puts the model's answer, or the error that ended the request, into the result queue."""
        try:
            self._result_queue.put((True, self.call()))
        except BaseException as error:
            self._result_queue.put((False, error))

    def _monitor(self) -> ModelOutput:
        """Returns the answer once it has arrived and the arms hold the reference pose, settled; raises the request's
        error, and raises at the operator's pause, at the deadline and once the arms no longer hold the pose."""
        control_period_s = self.robot.control_period_s
        deadline = time.monotonic() + self.timeout_s

        while True:
            OperatorPaused.check_cancelled(self.cancellation)
            is_holding = self.robot.is_holding(self.reference_observation)
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError("Model request timed out; no action will execute")

            if self._received_result is not None:
                has_answer, answer_or_error = self._received_result
                if not has_answer:
                    raise answer_or_error
                if is_holding:
                    return answer_or_error
                time.sleep(min(control_period_s, remaining_s))
                continue

            try:
                self._received_result = self._result_queue.get(timeout=min(control_period_s, remaining_s))
            except queue.Empty:
                pass

    def _stop(self, thread: threading.Thread, error: BaseException) -> None:
        """Stops the arms and the request and waits for the worker; raises SessionLost when the worker does not end
        or the request lost the model's conversation."""
        self.robot.stop()
        self.abort()
        thread.join(ModelClient.CLEANUP_TIMEOUT_S)
        if thread.is_alive():
            raise SessionLost("Model worker did not stop; clear the task before continuing") from error

        # a pause must not hide that the request lost the model's conversation
        if self._received_result is None:
            try:
                self._received_result = self._result_queue.get_nowait()
            except queue.Empty:
                pass
        if (
            self._received_result is not None
            and not self._received_result[0]
            and isinstance(self._received_result[1], SessionLost)
        ):
            raise self._received_result[1] from error


class StationAgentLoop:
    """A task's model session on a robot: each step observes, asks the model, reads the action of its answer and has
    the robot execute it; the conversation itself is the client's."""

    def __init__(
        self,
        robot: BaseRobot,
        client: ModelClient,
        *,
        action_format: ActionFormat,
        timeout_s: float,
        directory: Path,
        reference_images: tuple[ReferenceImage, ...],
    ) -> None:
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Model timeout must be finite and positive")
        if robot.action_mode != ActionMode.CARTESIAN:
            raise ValueError(f"A model session needs a Cartesian Station, not one in {robot.action_mode} mode")
        if robot.session_id is None:
            raise ValueError("Open a Cartesian Station before creating a model session")

        self.robot = robot
        self.client = client
        self.action_format = action_format
        self.timeout_s = timeout_s  # how long the model may take to answer a turn
        self.directory = Path(directory)  # the task's folder, where each turn's camera frames are saved
        # the task's images, in every turn's input; a client sends them once
        self.reference_images = tuple(reference_images)
        self.episode_id = str(uuid4())  # the session's identity, which its records carry
        self.turn_count = 0  # the turns observed so far
        self.last_turn: Turn | None = None
        self.previous_result: MotionResult | None = None  # the result of the last turn, told to the model in the next
        self.blocked_reason: str | None = None  # why the session takes no further turn; None while it does
        self._feedback: str | None = None  # what the next turn tells the model of the last result; None: nothing
        self._request_cancellation = threading.Event()  # stops the running model request: a new one each step
        self._step_lock = threading.Lock()  # held by the one step that runs

    def step(self, *, cancellation: threading.Event, on_event: TurnEventHandler, attempt_id: str) -> MotionResult:
        """Observes, asks the model, reads the action of its answer and has the robot execute it; returns the robot's
        result, which an action that did not complete has too."""
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("A model step is already running")
        turn = None
        self._request_cancellation = threading.Event()

        try:
            OperatorPaused.check_cancelled(cancellation)
            if self.blocked_reason:
                raise ContextLimit(self.blocked_reason)
            observation, state, model_input = self._observe(attempt_id, cancellation, on_event)
            turn = self.last_turn = Turn(attempt_id, observation, model_input)
            self.turn_count += 1

            action = self._ask_model(turn, observation, state, cancellation, on_event)

            return self._dispatch(turn, action, observation, cancellation, on_event)
        except (ContextLimit, SessionLost) as error:
            self.blocked_reason = str(error)
            self.robot.stop()
            raise
        except OperatorPaused as error:
            self.robot.stop()
            if turn is not None:
                self.record(self.failure_result(error))
            raise
        except Exception as error:
            self.robot.stop()
            if getattr(error, "details", None):
                on_event(TurnEvent.DIAGNOSTIC, error.details)
            if turn is None:
                raise
            logger.exception("Turn %s ended without a completed action", turn.attempt_id)
            status = MotionStatus.REJECTED if isinstance(error, ValueError) else MotionStatus.FAILED
            return self.record(self.failure_result(error, status=status))
        except BaseException:
            self.robot.stop()
            raise
        finally:
            self._step_lock.release()

    def failure_result(
        self,
        error: BaseException,
        *,
        status: MotionStatus = MotionStatus.FAILED,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus | None = None,
    ) -> MotionResult:
        """Returns the result of the turn's action that did not complete, measured at ``snapshot`` (by default
        now)."""
        return self.robot.failure_result(error, format_name=self.action_format.name, status=status, snapshot=snapshot)

    def record(self, result: MotionResult) -> MotionResult:
        """Keeps ``result`` as the last turn's, which the next turn tells the model of, and returns it."""
        self.last_turn.station_result = result
        self.previous_result = result
        self._feedback = self.action_format.result_feedback(result, is_dispatched=self.last_turn.is_dispatched)

        return result

    def close(self) -> None:
        """Stops the running model request, waits for the step to end and closes the client."""
        self.client.cancel()
        if not self._step_lock.acquire(timeout=ModelClient.CLEANUP_TIMEOUT_S):
            raise RuntimeError("Model step still running; cleanup incomplete")

        try:
            self.client.close()
        finally:
            self._step_lock.release()

    def _observe(
        self, attempt_id: str, cancellation: threading.Event, on_event: TurnEventHandler
    ) -> tuple[Observation, StrictModel, ModelInput]:
        """Returns the observation of the arms at rest, the state the model reads of it and the turn's input for the
        model."""
        on_event(ModelEvent.PHASE, {"phase": TurnPhase.OBSERVING.value})
        self.robot.wait_stationary(cancellation=cancellation)
        observation = self.robot.observe()
        state = self.robot.model_state(observation["joint_positions"], self.action_format.name)
        model_input = ObservationMessage.build(
            observation, state, self.action_format, self.directory, attempt_id, self._feedback, self.reference_images
        )

        return observation, state, model_input

    def _ask_model(
        self,
        turn: Turn,
        observation: Observation,
        state: StrictModel,
        cancellation: threading.Event,
        on_event: TurnEventHandler,
    ) -> StrictModel:
        """Reports what the model is shown, asks the model and returns the action of its answer."""
        OperatorPaused.check_cancelled(cancellation)
        on_event(
            TurnEvent.OBSERVATION,
            {
                "turn_id": turn.attempt_id,
                "observation_id": observation["observation_id"],
                "captured_monotonic_s": observation["captured_monotonic_s"],
                "state": state.model_dump(mode="json"),
                "text": turn.model_input.text,
                "images": {image.camera_name: str(image.path) for image in turn.model_input.images},
                "image_uuids": {image.camera_name: image.uuid for image in turn.model_input.images},
                "previous_result": self.previous_result.to_record() if self.previous_result else None,
            },
        )
        on_event(ModelEvent.PHASE, {"phase": ModelPhase.THINKING.value})

        model_output = ModelCall(
            self.robot,
            observation,
            lambda: self._infer(turn, on_event),
            timeout_s=self.timeout_s,
            cancellation=cancellation,
            abort=self._abort,
        ).run()
        on_event(TurnEvent.MODEL_OUTPUT, asdict(model_output))
        OperatorPaused.check_cancelled(cancellation)

        return ActionParser.parse(model_output.answer, self.action_format)

    def _infer(self, turn: Turn, on_event: TurnEventHandler) -> ModelOutput:
        """Returns the model's answer to the turn's input, which the turn keeps even when the operator stops it before
        its action goes to the robot."""
        turn.model_output = self.client.infer(
            turn.model_input, on_event=on_event, cancellation=self._request_cancellation
        )

        return turn.model_output

    def _abort(self) -> None:
        """Stops the running model request: its own cancellation and the client's."""
        self._request_cancellation.set()
        self.client.cancel()

    def _dispatch(
        self,
        turn: Turn,
        action: StrictModel,
        observation: Observation,
        cancellation: threading.Event,
        on_event: TurnEventHandler,
    ) -> MotionResult:
        """Has the robot execute ``action`` from the observed pose and returns its result, which the session keeps."""
        OperatorPaused.check_cancelled(cancellation)
        on_event(TurnEvent.ACTION, action.model_dump(mode="json"))
        OperatorPaused.check_cancelled(cancellation)
        turn.is_dispatched = True
        result = self.robot.execute(
            action,
            reference_observation=observation,
            cancellation=cancellation,
            report_progress=lambda phase: on_event(ModelEvent.PHASE, {"phase": phase.value}),
        )
        OperatorPaused.check_cancelled(cancellation)
        if self.robot.last_error_diagnostics:
            on_event(TurnEvent.DIAGNOSTIC, self.robot.last_error_diagnostics)

        return self.record(result)
