import threading
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop, Turn
from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient
from pear.deploy.station.agent_loop.clients.schemas import ModelEvent, ReferenceImage
from pear.deploy.station.agent_loop.schemas import TurnEvent
from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.schemas import RobotStatus
from pear.deploy.station.exceptions import OperatorPaused
from pear.deploy.station.gui.console.phase import Phase
from pear.deploy.station.records.task_events import TaskEventKind
from pear.deploy.station.records.task_files import TaskFiles

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.gui.console.console import Console

RECORD_URL = "/api/records/{task_id}/{name}"  # where the page's service serves a file of a task's folder


class TaskRun:
    """The console's task: its record, its model session and its turns, which run one after another on the
    console's worker, with each turn's records and events."""

    UNREACHABLE_RETRIES = 5  # consecutive pre-dispatch IK rejections retried with a fresh observation before pausing
    # shown when an unreachable action is asked for again
    RETRY_NOTICE = "IK rejected before motion. Requesting a fresh action ({retry}/{retries})."
    ATTEMPT_ID_CHARS = 12  # hex characters of an attempt id, unique within a task
    STATISTICS_EVENTS = (ModelEvent.USAGE, ModelEvent.PERFORMANCE)  # events whose final value outlives a cancelled turn
    # a result's status -> the phase its turn ends in
    OUTCOME_PHASES = {
        MotionStatus.COMPLETED: Phase.COMPLETED,
        MotionStatus.REJECTED: Phase.REJECTED,
        MotionStatus.FAILED: Phase.FAILED,
    }
    RESUME_CONTEXT_EVENT = "resume_context"  # the event of a run that continues after an action did not complete
    INTERRUPTED_EVENT = "interrupted"  # the event of a turn that the operator paused after its model request

    def __init__(self, console: "Console") -> None:
        self.console = console
        self.task: dict[str, Any] | None = (
            None  # the task's record (id, instruction, model, images); None without a task
        )
        self.model_client: ModelClient | None = None  # the model session's client; None before it is made
        self.session: StationAgentLoop | None = None  # the model session; None before it starts and once it ended
        self.total_steps = 0  # the actions that completed and settled
        self.turn_number = 0  # of the latest turn
        self.active_attempt_id: str | None = None  # of the running or the last turn
        self.active_turn: dict[str, Any] | None = None  # what the page shows of that turn
        self.resume_pending = False  # whether an action did not complete, so the next run reports what happened
        self.inference_started_at: float | None = None  # wall time at which the model began on the latest turn

    @property
    def files(self) -> TaskFiles:
        """The task's folder."""
        return TaskFiles(self.console.root / self.task["id"])

    def run(self, cancellation: threading.Event) -> None:
        """Runs turns until the operator pauses, an action does not complete, or the model session ends."""
        console = self.console
        retries = 0  # of consecutive actions offline IK rejected
        is_unfinished = False
        turn_before_step = None

        try:
            if self.session and self.resume_pending:
                self._resume_failed_turn()
            if not self.session:
                self.new_session()

            while True:
                OperatorPaused.check_cancelled(cancellation)
                attempt_id = self._start_turn()
                is_unfinished = True
                turn_before_step = self.session.last_turn

                result = self.session.step(
                    cancellation=cancellation,
                    on_event=partial(self.on_event, cancellation, attempt_id),
                    attempt_id=attempt_id,
                )
                OperatorPaused.check_cancelled(cancellation)

                self._save_execution_result(result)
                is_unfinished = False

                if self._is_unreachable(result) and retries < self.UNREACHABLE_RETRIES:
                    retries += 1
                    message = self.RETRY_NOTICE.format(retry=retries, retries=self.UNREACHABLE_RETRIES)
                    console.emit(ModelEvent.NOTICE, {"attempt": attempt_id, "retry": retries, "message": message})
                    continue

                if result.status != MotionStatus.COMPLETED or not result.is_settled:
                    # a local motion or planning failure keeps the model's history: the actual result is reported
                    # after a manual resume
                    self.resume_pending = True
                    raise RuntimeError(
                        result.error.get("message") if result.error else "Action did not complete and settle"
                    )

                retries = 0
                self.total_steps += 1
        except Exception as error:
            failure = self._reconcile_fault(error, turn_before_step) if is_unfinished else error
            self._end_turn(failure, is_unfinished)
            raise failure from None

    def new_session(self) -> None:
        """Starts the task's model session with a new client, and records it; refused while a session exists."""
        console = self.console
        if self.session:
            raise RuntimeError("Clear the existing episode before creating another")

        self.model_client = console.new_client(console.model, self.task["instruction"])
        directory = self.files.directory
        self.session = StationAgentLoop(
            console.robot,
            self.model_client,
            action_format=console.action_format,
            timeout_s=console.models_config.models[console.model].timeout_s,
            directory=directory,
            reference_images=tuple(
                ReferenceImage(directory / image["file"], image["id"]) for image in self.task.get("images", [])
            ),
        )
        self.resume_pending = False

        session_record = {
            "episode_id": self.session.episode_id,
            "model": console.model,
            "action_format": console.action_format.name,
            "system_prompt": self.model_client.system_prompt,
        }
        TaskFiles.write_json(self.files.session_path, session_record)
        session_started = {"episode_id": self.session.episode_id, "message": "Model session started"}
        console.emit(TaskEventKind.SESSION, session_started)

    def on_event(
        self, cancellation: threading.Event, attempt_id: str, kind: ModelEvent | TurnEvent, data: dict[str, Any]
    ) -> None:
        """Publishes one event of the turn ``attempt_id`` while it is the active turn; a cancelled turn publishes
        only its final statistics."""
        console = self.console
        if attempt_id != self.active_attempt_id:
            return
        if cancellation.is_set() and not (kind in self.STATISTICS_EVENTS and data.get("final") is True):
            return

        if kind == ModelEvent.PERFORMANCE:
            console.performance = {**data, "attempt": attempt_id, "received_at": time.time()}
            console.emit(kind, {**data, "attempt": attempt_id}, is_saved=bool(data.get("final")))
            return
        if kind == ModelEvent.USAGE:
            console.emit(kind, {**data, "attempt": attempt_id}, is_saved=data.get("final", True))
            return

        if kind == ModelEvent.PHASE:
            phase = Phase(data["phase"])
            console.set_phase(phase)
            if phase == Phase.THINKING and self.inference_started_at is None:
                self.inference_started_at = time.time()
            if self.active_turn:
                self.active_turn["phase"] = phase.value
        elif kind == TurnEvent.OBSERVATION:
            task_id = self.task["id"]
            observation_path = self.files.turn_record_path(attempt_id, TaskFiles.OBSERVATION_RECORD)
            TaskFiles.write_json(observation_path, data)
            details_url = RECORD_URL.format(task_id=task_id, name=observation_path.name)
            image_urls = {
                camera_name: RECORD_URL.format(task_id=task_id, name=Path(path).name)
                for camera_name, path in data["images"].items()
            }
            data = {**data, "details_url": details_url, "images": image_urls}
        elif kind == TurnEvent.MODEL_OUTPUT:
            TaskFiles.write_json(self.files.turn_record_path(attempt_id, TaskFiles.OUTPUT_RECORD), data)
        console.emit(kind, {**data, "attempt": attempt_id})

    def close_session(self) -> None:
        """Ends the model session, or closes a client that never started one; a cleanup that fails blocks the
        session."""
        if self.session:
            try:
                self.session.close()
            except Exception as error:
                self.session.blocked_reason = "Model cleanup incomplete: " + str(error)
                raise
        elif self.model_client:
            self.model_client.close()

        self.session = self.model_client = None

    def _resume_failed_turn(self) -> None:
        """Continues a session whose last action did not complete, with a fresh observation of what happened."""
        console = self.console
        console.robot.wait_settled(console.SETTLE_TIMEOUT_S)
        if self.session.blocked_reason:
            raise ContextLimit(self.session.blocked_reason)

        self.resume_pending = False
        resume_context = {
            "episode_id": self.session.episode_id,
            "message": "Continuing with a fresh observation and actual execution feedback",
        }
        console.emit(self.RESUME_CONTEXT_EVENT, resume_context)

    def _start_turn(self) -> str:
        """Opens the next turn under a new attempt id, which the turn's events carry, and returns the id."""
        console = self.console
        attempt_id = uuid4().hex[: self.ATTEMPT_ID_CHARS]
        self.active_attempt_id = attempt_id
        console.performance = None
        self.inference_started_at = None
        self.turn_number += 1

        self.active_turn = {
            "attempt": attempt_id,
            "turn_number": self.turn_number,
            "started_at": time.time(),
            "episode_id": self.session.episode_id,
            "phase": Phase.OBSERVING.value,
        }
        console.emit(TaskEventKind.TURN_STARTED, self.active_turn)

        return attempt_id

    def _save_execution_result(self, result: MotionResult) -> None:
        """Records what the active turn's action did, beside the model's output that asked for it."""
        attempt_id = self.active_attempt_id
        last_turn = self.session.last_turn
        if last_turn.model_output:
            output_path = self.files.turn_record_path(attempt_id, TaskFiles.OUTPUT_RECORD)
            TaskFiles.write_json(output_path, asdict(last_turn.model_output))

        result_record = result.to_record()
        result_path = self.files.turn_record_path(attempt_id, TaskFiles.RESULT_RECORD)
        TaskFiles.write_json(
            result_path, {"turn_id": attempt_id, "dispatched": last_turn.is_dispatched, **result_record}
        )
        self.active_turn["phase"] = self.OUTCOME_PHASES[result.status].value
        self.console.emit(TaskEventKind.EXECUTION, {"attempt": attempt_id, **result_record})

    def _reconcile_fault(self, error: Exception, turn_before_step: Turn | None) -> Exception:
        """Records a robot fault that ended an unfinished turn; the fault replaces ``error`` as the turn's failure,
        which this returns."""
        console = self.console
        robot_state = console.robot.status(with_snapshot=True)
        if not self.active_turn or robot_state["status"] != RobotStatus.FAULT:
            return error

        # the watchdog's cancellation still blocks inference and dispatch; the latched fault is reconciled here,
        # without a cancellable event callback or a wait for HOLD, which a faulted control loop never reaches
        failure = error
        try:
            console.robot.check_fault()
        except Exception as fault:
            failure = fault
        last_turn = self.session.last_turn
        if last_turn is not None and last_turn is not turn_before_step:
            result = self.session.record(self.session.failure_result(failure, snapshot=robot_state))
            self._save_execution_result(result)
            self.resume_pending = True

        details = getattr(failure, "details", None)
        # a diagnostic the turn already published, found among the events kept for browsers, is not sent again
        with console.lock:
            if details and not any(
                event["type"] == TurnEvent.DIAGNOSTIC
                and event["data"].get("attempt") == self.active_attempt_id
                and event["data"].get("code") == details["code"]
                for event in console.events
            ):
                console.emit(TurnEvent.DIAGNOSTIC, {**details, "attempt": self.active_attempt_id})

        return failure

    def _end_turn(self, failure: Exception, is_unfinished: bool) -> None:
        """Closes the active turn after ``failure`` ended the run."""
        console = self.console
        if isinstance(failure, OperatorPaused):
            if is_unfinished and self.active_turn:
                self.active_turn["phase"] = Phase.INTERRUPTED.value
                turn_end_data = {"attempt": self.active_attempt_id, "phase": Phase.INTERRUPTED.value}
                console.emit(TaskEventKind.TURN_END, turn_end_data)
                self._record_interruption()
        elif self.active_turn:
            phase = Phase.CONTEXT_LIMIT if isinstance(failure, ContextLimit) else Phase.FAILED
            self.active_turn["phase"] = phase.value
            turn_end_data = {"attempt": self.active_attempt_id, "phase": phase.value, "message": str(failure)}
            console.emit(TaskEventKind.TURN_END, turn_end_data, is_saved=False)

    def _record_interruption(self) -> None:
        """Records an operator's pause of a turn whose model request was made, as the turn's failed result."""
        console = self.console
        session = self.session
        if not session or not session.last_turn or session.last_turn.attempt_id != self.active_attempt_id:
            return

        if console.robot_status() != RobotStatus.FAULT:
            console.robot.wait_settled(console.SETTLE_TIMEOUT_S)
        interrupted_result = session.failure_result(
            OperatorPaused("The operator interrupted this turn; remaining motion was cancelled.")
        )
        self._save_execution_result(session.record(interrupted_result))

        self.active_turn["phase"] = Phase.INTERRUPTED.value
        interrupted = {
            "message": "Step interrupted; remaining action will not execute.",
            "attempt": self.active_attempt_id,
            "session_retained": not session.blocked_reason,
        }
        console.emit(self.INTERRUPTED_EVENT, interrupted)

    @staticmethod
    def _is_unreachable(result: MotionResult) -> bool:
        """Whether offline IK rejected the action before any motion."""
        return bool(
            result.status == MotionStatus.REJECTED
            and result.is_settled
            and result.error
            and result.error.get("code") == UnreachablePoseError.__name__
        )
