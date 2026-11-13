import pkgutil
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.agent_loop.clients.health_checks import HealthChecks
from pear.deploy.station.agent_loop.clients.schemas import ModelHealth
from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.cameras import CameraFeed
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.schemas import ActionFormat, RobotStatus
from pear.deploy.station.exceptions import OperatorPaused
from pear.deploy.station.gui.console.commands import Commands
from pear.deploy.station.gui.console.home_review import HomeReview, HomeStatus
from pear.deploy.station.gui.console.lease import ControlLease
from pear.deploy.station.gui.console.phase import Phase
from pear.deploy.station.gui.console.preview import Preview
from pear.deploy.station.gui.console.task_run import TaskRun
from pear.deploy.station.gui.console.watchdog import Watchdog
from pear.deploy.station.records.annotations import Annotations
from pear.deploy.station.records.image_uploads import ImageUploads
from pear.deploy.station.records.task_events import TaskEvents
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.render_job import RenderedRecordings
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Console:
    """The page's console of one robot: the robot, its cameras, the recordings and the tasks' records, one operation
    at a time on its worker, and a STOP that waits for nothing."""

    WORKER_JOIN_S = 15.0  # for a stopped operation to end
    SETTLE_TIMEOUT_S = 10.0  # for a stopped or interrupted motion to settle
    CHECK_INSTRUCTION = "Connection check"  # the task of a client made only to check its connection
    EVENT_CAPACITY = 500  # recent events a reconnecting browser can catch up on
    WORKER_THREAD_NAME = "gui-operation"  # of the thread that runs the operations, one at a time
    PAUSE_REASON = "Paused by operator"  # why the robot stopped when the operator paused it
    STOPPING_EVENT = "stopping"  # the event of a stop request
    ERROR_EVENT = "error"  # the event of an operation's failure

    def __init__(self, config: DictConfig, *, root: Path, uploads_dir: Path) -> None:
        self.root = root  # the folder of the tasks' folders
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_config: ModelsConfig = ModelsConfig.from_station_config(config)

        # what the robot, the previews and the recordings read the cameras through
        self.camera_feed = CameraFeed(None)
        self.robot: BaseRobot = pkgutil.resolve_name(config.robot_class)(config, cameras=self.camera_feed)
        self.uploads = ImageUploads(uploads_dir)
        self.recordings = RenderedRecordings(root, self.robot.CAMERAS)
        self.annotations = Annotations()  # the one writer of the tasks' tags, which the page's service shares

        # held by every write of the state several threads write: the phase, the STOP latch, the Home review and the
        # saved fault, so that a check of that state under it stays true until the matching write
        self.lock = threading.RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=self.EVENT_CAPACITY)  # the recent events, read under lock
        self.last_event_id = 0  # of the latest event: the events are numbered from 1
        self.lease = ControlLease(lambda: self.is_busy)
        self.model = self.models_config.default_model  # the model of the next task
        self.health_checks = HealthChecks(lambda model: self.new_client(model, self.CHECK_INSTRUCTION))

        self.worker: threading.Thread | None = None  # the thread of the running or the last operation
        self.cancellation = threading.Event()  # set to cancel the running operation
        self._phase = Phase.DISABLED
        self.error: str | None = None  # the failure the page shows; None without one
        self.is_stop_latched = False  # whether STOP MOTION holds every motion until it is reset
        self.performance: dict[str, Any] | None = None  # the inference report of the operation's latest turn
        self.home_status: dict[str, Any] | None = None  # the Home's stage and status, as the page receives them
        # (the robot's session, the fault's generation) of the last fault saved
        self.saved_fault_identity: tuple[str | None, int | None] | None = None
        self.task_run = TaskRun(self)  # the task, its model session and its turns
        self.home_review = HomeReview(self.lock)
        self.commands = Commands(self)

        roles = tuple(camera.role for camera in self.robot.CAMERAS)
        self.preview = Preview(self.camera_feed, roles, self.robot.open_cameras)
        self.watchdog = Watchdog(self)

    @property
    def phase(self) -> Phase:
        """What the console is doing, as the page shows it."""
        return self._phase

    @property
    def is_busy(self) -> bool:
        """Whether an operation runs on the console's worker."""
        return self.worker is not None and self.worker.is_alive()

    @property
    def action_format(self) -> ActionFormat:
        """The action format the selected model answers in."""
        return self.models_config.action_format(self.model)

    def set_phase(self, phase: Phase) -> None:
        """Shows ``phase``; the one writer of the phase, under the lock."""
        with self.lock:
            self._phase = phase

    def stop(self, *, is_latching: bool = False, reason: str = PAUSE_REASON) -> dict[str, Any]:
        """Cancels the running operation, stops the robot and returns how long the stop request took."""
        # no inference, event file, network, lifecycle or console lock before the stop itself
        started_at = time.monotonic()
        self.cancellation.set()
        try:
            self.robot.stop()
            stop_request_seconds = time.monotonic() - started_at
        finally:
            self.home_review.answered.set()  # a Home review that waits for the operator sees the cancellation
            with self.lock:
                # a failed stop request still latches, so no further motion starts until the reset
                if is_latching:
                    self.is_stop_latched = True
                self.set_phase(Phase.STOPPING)
                self.performance = None
            self.recordings.stop(reason, stopped_at=started_at)

        stopping = {"message": reason, "stop_request_seconds": stop_request_seconds}
        self.emit(self.STOPPING_EVENT, stopping, is_saved=False)

        return {"stop_requested": True, "stop_request_seconds": stop_request_seconds}

    def snapshot(self) -> dict[str, Any]:
        """Returns the state the page shows; the console's own part is read under the lock, as one moment."""
        robot_state = self.robot.status(with_snapshot=True)
        camera_status = self.preview.status()
        recording = self.recordings.snapshot()

        with self.lock:
            task_run = self.task_run
            phase, session, active_turn = self._phase, task_run.session, task_run.active_turn
            if phase == Phase.EXECUTING and robot_state.get("motion", {}).get("settling"):
                phase = Phase.SETTLING

            return {
                "phase": phase.value,
                "profile": self.robot.profile(),
                "model": self.model,
                "models": [
                    {"id": name, "label": offered_model.label, "action_format": offered_model.action_format}
                    for name, offered_model in self.models_config.models.items()
                ],
                "action_format": self.action_format.name,
                "active_turn": dict(active_turn) if active_turn else None,
                "robot": robot_state,
                "model_health": self.health_checks.get(self.model),
                "supports_reference_images": self.supports_reference_images(),
                "busy": self.is_busy,
                "stop_latched": self.is_stop_latched,
                "error": self.error,
                "task": task_run.task,
                "total_steps": task_run.total_steps,
                "sequence": session.turn_count if session else 0,
                "owner": self.lease.owner,
                "home_plan": self.home_review.plan,
                "home_status": self.home_status,
                "performance": self.performance,
                "inference_started_at": task_run.inference_started_at,
                "recording": recording,
                "cameras": camera_status,
            }

    def check_model(self, *, is_forced: bool = False) -> ModelHealth:
        """Returns the selected model's connection: from a recent check unless ``is_forced``, else from a client made
        for it."""
        return self.health_checks.check(self.model, is_forced=is_forced)

    def supports_reference_images(self) -> bool:
        """Whether the selected model accepts a task's images, as its last connection check tells."""
        client_type = self.models_config.models[self.model].client_type

        return client_type.supports_reference_images(self.health_checks.get(self.model))

    def emit(self, kind: str, data: dict[str, Any], *, is_saved: bool = True) -> dict[str, Any]:
        """Publishes one event to the recording and the browsers and, when ``is_saved`` and there is a task, to the
        task's event file; returns the event."""
        self.recordings.on_event(kind, data)

        with self.lock:
            active_turn, task_id = self.task_run.active_turn, (self.task_run.task or {}).get("id")
            # an event of the running turn carries the turn's fields
            if active_turn and data.get("attempt") == active_turn["attempt"]:
                data = {**active_turn, **data}
            else:
                data = dict(data)
            self.last_event_id += 1
            event = {
                "id": self.last_event_id,
                "type": str(kind),
                "data": data,
                "time": time.time(),
                "monotonic_s": time.monotonic(),  # the clock of the recordings' timelines and the cameras' frames
                "task_id": task_id,
            }

            if is_saved and task_id:
                TaskEvents(TaskFiles(self.root / task_id).events_path).append(event)
            self.events.append(event)

        return event

    def launch(
        self,
        operation: Callable[[threading.Event], None],
        *,
        cancellation: threading.Event | None = None,
        is_recorded: bool = False,
    ) -> None:
        """Runs ``operation(cancellation)`` on the console's worker; a failure stops the robot and is shown to the
        operator."""
        with self.lock:
            if self.is_busy:
                raise ValueError("Wait for the current operation to finish stopping.")
            cancellation = self.cancellation = cancellation if cancellation is not None else threading.Event()
            OperatorPaused.check_cancelled(cancellation)

            self.worker = threading.Thread(
                target=self._operate,
                args=(operation, cancellation, is_recorded),
                name=self.WORKER_THREAD_NAME,
                daemon=True,
            )
            self.worker.start()

    def _operate(
        self, operation: Callable[[threading.Event], None], cancellation: threading.Event, is_recorded: bool
    ) -> None:
        """The worker's body: one operation, then its failure handling, the end of its recording and the withdrawal
        of a Home plan under review."""
        try:
            operation(cancellation)
        except OperatorPaused:
            self.set_phase(Phase.STOPPING)
        except Exception as error:
            logger.exception("An operation of the console failed")
            self.robot.stop()
            self.record_error(error)
            self.performance = None

            if isinstance(error, HomeFailed):
                self.home_status = {**error.details, "status": HomeStatus.FAILED.value}
                TaskFiles.write_json(self.root / f"home-failed-{time.time_ns()}.json", self.home_status)
            if self.robot_status() == RobotStatus.FAULT:
                self.set_phase(Phase.FAULT)
            else:
                self.set_phase(Phase.HOME_FAILED if isinstance(error, HomeFailed) else Phase.STOPPING)
        finally:
            if is_recorded:
                self.recordings.stop(self.error or "Task paused")
            with self.lock:
                self.home_review.plan = None

    def close(self) -> None:
        """Stops the robot and releases it, the cameras and the model session; the motors must be disabled first."""
        self.stop(reason="GUI shutdown")
        if self.worker is not None:
            self.worker.join(self.WORKER_JOIN_S)
        # the robot is shut down only once its motors are explicitly disabled
        if self.robot_status() != RobotStatus.OFF:
            raise RuntimeError("Disable motors before shutting down the GUI service")

        self.watchdog.close()
        self.recordings.close()
        try:
            self.task_run.close_session()
        finally:
            self.robot.close()
            self.preview.close()

    def new_client(self, model: str, instruction: str) -> ModelClient:
        """Returns a new client of ``model``, its system prompt the task ``instruction`` in the model's format."""
        offered_model = self.models_config.models[model]

        return offered_model.new_client(SystemPrompt.render(instruction, self.models_config.action_format(model)))

    def record_error(self, error: BaseException) -> None:
        """Shows an operation's failure to the operator and saves the robot's fault, if it has one."""
        self.error = f"{type(error).__name__}: {error}"
        self.emit(self.ERROR_EVENT, {"message": self.error}, is_saved=False)
        self.save_control_diagnostic()

    def save_control_diagnostic(self) -> None:
        """Saves the robot's current fault, once for each generation of a fault."""
        robot_state = self.robot.status(with_snapshot=True)
        fault_identity = (self.robot.session_id, robot_state.get("generation"))
        # claimed under the lock: the watchdog and the worker may both find the same fault unsaved
        with self.lock:
            if not robot_state.get("fault") or fault_identity == self.saved_fault_identity:
                return
            self.saved_fault_identity = fault_identity

        try:
            fault_record = {"fault": robot_state["fault"], "details": robot_state["fault_details"]}
            TaskFiles.write_json(self.root / f"control-fault-{time.time_ns()}.json", fault_record)
        except (OSError, ValueError):
            logger.exception("Could not persist control fault record")

    def robot_status(self) -> str:
        """Returns the robot's status, the word of a RobotStatus such as OFF, READY or FAULT."""
        return self.robot.status()["status"]
