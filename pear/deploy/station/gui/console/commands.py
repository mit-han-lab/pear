import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pear.deploy.station.agent_loop.clients.schemas import ModelEvent
from pear.deploy.station.control.motion.homing import HomeStage
from pear.deploy.station.control.schemas import HomePlan, RobotStatus
from pear.deploy.station.exceptions import OperatorPaused
from pear.deploy.station.gui.console.home_review import HomeStatus
from pear.deploy.station.gui.console.phase import STOPPED_STATUSES, Phase
from pear.deploy.station.gui.console.task_run import RECORD_URL
from pear.deploy.station.records.annotations import Annotations
from pear.deploy.station.records.task_events import TaskEventKind
from pear.deploy.station.records.task_files import TaskFiles
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.gui.console.console import Console

logger = get_logger(__name__)


@dataclass(frozen=True)
class Command:
    """One of the operator's commands: its handler, and how the console and the page's service admit it."""

    handler: Callable[[dict[str, Any]], dict[str, Any] | None]  # takes the command's data; returns its answer
    is_immediate: bool = False  # runs at once, never behind another command or the running operation
    needs_control: bool = True  # only the tab that holds control may give it
    is_latch_blocked: bool = False  # refused while STOP MOTION is latched
    stops_first: bool = False  # stops the running operation instead of waiting for it
    clears_error: bool = True  # clears the failure the page shows once it is admitted
    clears_performance: bool = False  # clears the inference report the page shows once it is admitted


class Commands:
    """The operator's commands: each is admitted on the caller's thread, and a long one then runs on the console's
    worker."""

    MAX_INSTRUCTION_CHARS = 20000  # longest task instruction accepted, in characters
    # the stage a robot's Home reports -> the phase shown
    HOME_STAGE_PHASES = {
        HomeStage.OPENING: Phase.OPENING_GRIPPERS,
        HomeStage.CHECKING: Phase.CHECKING_DIRECT_HOME,
        HomeStage.HOMING: Phase.HOMING,
        HomeStage.COMPLETED: Phase.HOME_COMPLETE,
    }

    def __init__(self, console: "Console") -> None:
        self.console = console

        # the commands, by the name the page sends
        self.table = {
            "pause": Command(self._pause, is_immediate=True),
            "stop": Command(self._stop, is_immediate=True, needs_control=False),
            "home_confirm": Command(self._confirm_home, is_immediate=True),
            "reset_stop": Command(self._reset_stop, clears_error=False),
            "reconnect_cameras": Command(self._reconnect_cameras),
            "enable": Command(self._enable, is_latch_blocked=True),
            "recover_control": Command(self._recover_control),
            "start": Command(self._start, is_latch_blocked=True, clears_performance=True),
            "resume": Command(self._resume, is_latch_blocked=True, clears_performance=True),
            "clear": Command(self._clear, stops_first=True, clears_performance=True),
            "disable": Command(self._disable, stops_first=True, clears_performance=True),
            "home": Command(self._home, is_latch_blocked=True, stops_first=True, clears_performance=True),
            "model": Command(self._select_model, clears_performance=True),
            "recording": Command(self._set_recording),
        }

    def run(self, name: str, data: dict[str, Any] | None) -> dict[str, Any]:
        """Admits the command ``name`` with ``data`` and returns its answer; STOP and pause never wait for anything."""
        console = self.console
        data = data or {}
        command = self.table.get(name)
        if command is None:
            raise ValueError("Unknown command")
        if command.is_immediate:
            return command.handler(data)

        if command.stops_first and console.is_busy:
            self._stop_before(name)
        if console.is_busy:
            raise ValueError("Wait for the current operation or pause it first")
        if command.is_latch_blocked and console.is_stop_latched:
            raise ValueError("Reset STOP MOTION before continuing")

        if command.clears_error:
            console.error = None
        if command.clears_performance:
            console.performance = None
        answer = command.handler(data)

        return {"accepted": True} if answer is None else answer

    def _stop_before(self, name: str) -> None:
        """Stops the running operation and waits for it to end and for the robot to settle."""
        console = self.console
        previous_worker = console.worker
        console.stop(reason="Stopping before " + name)

        previous_worker.join(console.WORKER_JOIN_S)
        if previous_worker.is_alive():
            raise ValueError("Old operation is still stopping; no new operation was started")
        if console.robot_status() not in (RobotStatus.OFF, RobotStatus.FAULT):
            console.robot.wait_settled(console.SETTLE_TIMEOUT_S)

    def _pause(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pauses the running operation and stops the robot."""
        return self.console.stop()

    def _stop(self, data: dict[str, Any]) -> dict[str, Any]:
        """Stops the robot and latches STOP MOTION until it is reset."""
        return self.console.stop(is_latching=True, reason="STOP MOTION activated")

    def _confirm_home(self, data: dict[str, Any]) -> dict[str, Any]:
        """Answers the Home plan under review."""
        self.console.home_review.confirm(data.get("plan_id"), data.get("confirm") is True)

        return {"accepted": True}

    def _reset_stop(self, data: dict[str, Any]) -> dict[str, Any]:
        """Releases the STOP MOTION latch of a stopped, stable robot without moving it."""
        console = self.console

        # under the lock a STOP that arrives meanwhile latches after this reset, never before it
        with console.lock:
            robot_status = console.robot_status()
            if robot_status not in STOPPED_STATUSES:
                raise ValueError("Robot must be stopped and stable before reset")
            console.is_stop_latched = False
            console.set_phase(
                Phase.idle_for(robot_status, is_latched=False, has_task=console.task_run.task is not None)
            )

        return {"reset": True}

    def _reconnect_cameras(self, data: dict[str, Any]) -> None:
        """Replaces the cameras' reader between operations; the motors, the task and its model session stay."""
        console = self.console
        with console.lock:
            previous_phase = console.phase  # shown again once the cameras are back
            console.set_phase(Phase.RECONNECTING_CAMERAS)

        console.launch(lambda cancellation: self._replace_cameras(cancellation, previous_phase))

    def _replace_cameras(self, cancellation: threading.Event, previous_phase: Phase) -> None:
        """Reconnects the cameras, then shows ``previous_phase`` again; it never moves the arms."""
        console = self.console
        console.preview.reconnect(cancellation)

        console.emit(ModelEvent.NOTICE, {"message": "Camera reconnect started. Motors and the task are unchanged."})
        console.set_phase(previous_phase)
        # a STOP or a fault during the reconnect still ends it as a stopped operation
        OperatorPaused.check_cancelled(cancellation)

    def _enable(self, data: dict[str, Any]) -> None:
        """Enables the motors, holding where the arms are, once every camera delivers frames."""
        console = self.console
        if console.robot_status() != RobotStatus.OFF:
            raise ValueError("Motors are already enabled")
        if console.camera_feed.reader is None or (console.preview.is_encoding and console.preview.error_text):
            raise ValueError("Cameras are unavailable. Reconnect cameras first.")
        console.camera_feed.read()  # reconnecting, partial or stale cameras never enable the motors
        console.set_phase(Phase.ENABLING)

        console.launch(self._open_robot)

    def _open_robot(self, cancellation: threading.Event) -> None:
        """Opens the robot's session, which enables the motors, and waits until the arms hold still."""
        console = self.console
        OperatorPaused.check_cancelled(cancellation)
        console.robot.open()
        OperatorPaused.check_cancelled(cancellation)
        console.robot.wait_stationary()
        console.set_phase(Phase.HOLDING)

    def _recover_control(self, data: dict[str, Any]) -> None:
        """Recovers a faulted robot's control at its current pose, keeping the task paused."""
        console = self.console

        with console.lock:
            if console.robot_status() != RobotStatus.FAULT:
                raise ValueError("Control recovery requires a faulted robot")
            console.save_control_diagnostic()
            # the old request's cancellation stays set, for its late model events too
            console.cancellation.set()
            cancellation = console.cancellation = threading.Event()
            console.set_phase(Phase.RECOVERING_CONTROL)

            console.launch(self._recover, cancellation=cancellation)

    def _recover(self, cancellation: threading.Event) -> None:
        """Reacquires the robot's hold at its current pose, then shows the console idle."""
        console = self.console
        try:
            console.robot.recover(cancellation=cancellation)
        except OperatorPaused:
            raise
        except Exception as error:
            raise RuntimeError(f"Control recovery failed: {error}") from error

        with console.lock:
            OperatorPaused.check_cancelled(cancellation)
            robot_status = console.robot_status()
            if robot_status != RobotStatus.READY:
                raise RuntimeError("Robot is no longer stable after recovery")
            console.error = None
            idle_phase = Phase.idle_for(
                robot_status, is_latched=console.is_stop_latched, has_task=console.task_run.task is not None
            )
            console.set_phase(idle_phase)
            message = "Control recovered at the current pose. Motors remain enabled. Choose Resume or Home."
            console.emit(ModelEvent.NOTICE, {"message": message})

    def _start(self, data: dict[str, Any]) -> None:
        """Creates the record of a task from its instruction, tags and images, and runs its turns."""
        console = self.console
        task_run = console.task_run
        if task_run.task:
            raise ValueError("Clear the current task before starting a new one")
        instruction = str(data.get("instruction", "")).strip()
        if not 1 <= len(instruction) <= self.MAX_INSTRUCTION_CHARS:
            raise ValueError(f"Enter a task of 1–{self.MAX_INSTRUCTION_CHARS} characters")
        tags = Annotations.clean_tags(data.get("tags", []))
        image_ids = data.get("image_ids", [])
        if image_ids and not console.supports_reference_images():
            raise ValueError("This model service does not yet support task images")
        if not console.preview.is_encoding or console.preview.error_text:
            raise ValueError("Restore live camera feeds before starting a task")
        if console.robot_status() != RobotStatus.READY:
            raise ValueError("Enable motors and wait for stable holding first")

        offered_model = console.models_config.models[console.model]
        task = {
            "id": uuid4().hex,
            "instruction": instruction,
            "created": time.time(),
            "model": console.model,
            "model_label": offered_model.label,
        }
        if len(console.robot.ACTION_FORMATS) > 1:
            task["action_format"] = console.action_format.name
        reasoning_effort = getattr(offered_model.client_type, "REASONING_EFFORT", None)  # a client that names one
        if reasoning_effort:
            task["reasoning_effort"] = reasoning_effort  # shown beside the model's name in the styled task video

        files = TaskFiles(console.root / task["id"])
        files.directory.mkdir()
        try:
            attached_images = console.uploads.attach(image_ids, console.lease.owner, files)
            # the page fetches each image where the page's service serves the task's files
            task["images"] = [
                {**image, "url": RECORD_URL.format(task_id=task["id"], name=image["file"])} for image in attached_images
            ]
            if tags:
                console.annotations.update(files, tags=tags)
        except Exception:
            shutil.rmtree(files.directory)
            raise

        task_run.task = task
        TaskFiles.write_json(files.task_path, task)
        for upload_id in image_ids:
            try:
                console.uploads.delete(upload_id, console.lease.owner)
            except ValueError as error:
                # the task's copies are made: a draft that stays cannot hold the task back
                logger.warning("Draft image %s was not removed: %s", upload_id, error)

        task_run.total_steps = task_run.turn_number = 0
        task_run.active_attempt_id = task_run.active_turn = None
        console.emit(TaskEventKind.TASK, task)
        self._start_recorded_loop()

    def _resume(self, data: dict[str, Any]) -> None:
        """Runs the paused task's turns again, in its model session unless that session ended."""
        console = self.console
        task_run = console.task_run
        if not task_run.task or console.robot_status() != RobotStatus.READY:
            raise ValueError("A paused task and stable enabled robot are required")
        if not console.preview.is_encoding or console.preview.error_text:
            raise ValueError("Restore live camera feeds before resuming")
        if task_run.session and task_run.session.blocked_reason:
            raise ValueError(task_run.session.blocked_reason + "; clear the task to restart")

        self._start_recorded_loop()

    def _clear(self, data: dict[str, Any]) -> None:
        """Ends the task and its model session; its records stay in the history."""
        self.console.launch(self._clear_task)

    def _clear_task(self, cancellation: threading.Event) -> None:
        """Closes the task's model session, then ends the task."""
        console = self.console
        task_run = console.task_run
        task_run.close_session()

        console.emit(TaskEventKind.TASK_END, {"message": "Task cleared; history saved."})
        task_run.task, task_run.total_steps = None, 0
        task_run.active_attempt_id = task_run.active_turn = None
        console.set_phase(Phase.idle_for(console.robot_status(), is_latched=False, has_task=False))

    def _disable(self, data: dict[str, Any]) -> None:
        """Disables the motors once the operator confirms the arms are supported, even when the model's cleanup
        fails."""
        if data.get("confirm") is not True:
            raise ValueError("Confirm arms and objects are supported before disabling")

        self.console.launch(self._close_robot)

    def _close_robot(self, cancellation: threading.Event) -> None:
        """Closes the task's model session and the robot's session, which disables the motors."""
        console = self.console
        task_run = console.task_run
        cleanup_error = None
        try:
            task_run.close_session()
        except Exception as error:
            logger.exception("The model session did not close")
            cleanup_error = error

        # an explicit disable of the motors must not depend on a reachable model server
        console.robot.close()
        if cleanup_error is None:
            task_run.session = task_run.model_client = task_run.task = None
            task_run.active_turn = task_run.active_attempt_id = None
        console.set_phase(Phase.DISABLED)
        if cleanup_error is not None:
            message = "Motors disabled; model session cleanup failed: " + str(cleanup_error)
            console.record_error(RuntimeError(message))

    def _home(self, data: dict[str, Any]) -> None:
        """Ends the task and returns the arms Home once the operator approves the plan."""
        console = self.console
        robot = console.robot
        if console.robot_status() != RobotStatus.READY:
            raise ValueError("Home requires stable enabled motors")

        home_options: dict[str, Any] = {}
        if robot.HOME_ARM_CHOICES:
            home_options["side"] = data.get("side", robot.HOME_ARM_CHOICES[0])
            if home_options["side"] not in robot.HOME_ARM_CHOICES:
                raise ValueError("Invalid home side")
        # False: return only the arm joints and keep the current gripper targets
        opens_grippers = data.get("open_grippers", True)
        if not isinstance(opens_grippers, bool) or (not opens_grippers and not robot.HOME_CAN_KEEP_GRIPPERS):
            raise ValueError("Invalid home gripper choice")
        if robot.HOME_CAN_KEEP_GRIPPERS:
            home_options["opens_grippers"] = opens_grippers

        console.launch(lambda cancellation: self._return_home(cancellation, home_options, opens_grippers))

    def _return_home(self, cancellation: threading.Event, home_options: dict[str, Any], opens_grippers: bool) -> None:
        """Ends the task and its model session, then returns the arms Home with ``home_options`` once the operator
        approves the plan."""
        console = self.console
        task_run = console.task_run
        console.home_status = {"stage": HomeStage.CHECKING.value, "status": HomeStatus.RUNNING.value}
        task_run.close_session()
        if task_run.task:
            console.emit(TaskEventKind.TASK_END, {"message": "Task ended for home."})
        task_run.task = None
        console.set_phase(Phase.CHECKING_HOME)

        home_report = console.robot.home(
            cancellation=cancellation,
            report_progress=lambda progress: self._report_home_progress(progress, cancellation),
            approve_plan=lambda plan: self._review_home(plan, cancellation),
            **home_options,
        )
        TaskFiles.write_json(console.root / f"home-{time.time_ns()}.json", home_report)

        console.home_status = {
            "stage": HomeStage.COMPLETED.value,
            "status": HomeStatus.COMPLETED.value,
            "results": home_report["results"],
        }
        console.set_phase(Phase.HOME_COMPLETE)
        reached_message = (
            "Grippers opened and Home reached" if opens_grippers else "Arms reached Home; grippers unchanged"
        )
        console.emit(ModelEvent.NOTICE, {"message": f"{reached_message}. Start a new task when ready."})

    def _report_home_progress(self, progress: dict[str, Any], cancellation: threading.Event) -> None:
        """Shows one stage of the Home."""
        console = self.console
        OperatorPaused.check_cancelled(cancellation)
        phase = self.HOME_STAGE_PHASES[progress["stage"]]

        console.home_status = {**progress, "status": HomeStatus.RUNNING.value}
        console.set_phase(phase)
        console.emit(ModelEvent.NOTICE, {"message": phase.value, "home": console.home_status})

    def _review_home(self, plan: HomePlan, cancellation: threading.Event) -> bool:
        """Records a Home plan, publishes it and returns the operator's answer; a STOP ends the wait."""
        console = self.console
        review_plan = console.home_review.prepare(plan)
        TaskFiles.write_json(console.root / f"home-plan-{time.time_ns()}.json", review_plan)
        console.set_phase(Phase.CONFIRM_HOME)

        return console.home_review.wait_for_decision(review_plan, cancellation)

    def _select_model(self, data: dict[str, Any]) -> None:
        """Chooses the model of the next task."""
        console = self.console
        model = data.get("model")
        console.models_config.require(model)
        if console.task_run.task or console.task_run.session:
            raise ValueError("Clear the current task before changing models")

        console.model = model

    def _set_recording(self, data: dict[str, Any]) -> None:
        """Turns the recording of tasks on or off."""
        if type(data.get("enabled")) is not bool:
            raise ValueError("Recording enabled must be a boolean")

        self.console.recordings.is_enabled = data["enabled"]

    def _start_recorded_loop(self) -> None:
        """Starts recording the task and runs its turns on the worker."""
        console = self.console
        task_run = console.task_run
        # replaced before the recording starts: a STOP while it starts sets the cancellation the worker is given
        cancellation = console.cancellation = threading.Event()

        try:
            console.recordings.start(
                dict(task_run.task), console.camera_feed, lambda: console.robot.status(with_snapshot=True)
            )
            console.launch(task_run.run, cancellation=cancellation, is_recorded=True)
        except BaseException:
            console.recordings.stop("Task did not start")
            raise
