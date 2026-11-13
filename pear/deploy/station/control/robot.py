from __future__ import annotations

import copy
import pkgutil
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.joint.operations import GripperOperation
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.stability import StabilityError
from pear.deploy.station.control.motion.actions import ActionAdapter, JointPositions, StrictModel
from pear.deploy.station.control.motion.execution import Execution
from pear.deploy.station.control.motion.kinematics import BaseKinematics
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.schemas import (
    ActionFormat,
    ActionPhase,
    BriefStatus,
    Camera,
    ControlSnapshot,
    ControlState,
    DetailedStatus,
    HomePlan,
    HomeReport,
    Observation,
    RobotStatus,
)
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class ActionMode(StrEnum):
    """Which actions a robot takes."""

    CARTESIAN = "cartesian"  # every action of the robot, as a model's session gives them: the planner is opened
    JOINT = "joint"  # chunks of joint targets alone, as maintenance gives them: no planner is opened


class BaseRobot(Execution):
    """The base of a robot's class: one robot's session over its control loop, with its lifetime, its observations,
    its blocking actions and Home; a robot's class sets its facts and its classes and implements its own steps."""

    # the robot's facts
    NAME: str  # what the command line and the page's service call the robot: the name of its package
    TITLE: str  # what the page calls the robot
    SUBTITLE: str  # the line under the title that the page receives with the robot's profile
    ARMS_TEXT: str  # what the robot moves, in words, as the page receives it with the profile: "Both arms"
    CAMERAS: tuple[Camera, ...]  # its cameras, in the order the page shows them
    ACTION_FORMATS: dict[str, ActionFormat]  # its action formats by name, the default first
    HOME_ARM_CHOICES: tuple[str, ...] = ()  # the arms Home may return, the default first; empty: no choice
    HOME_CAN_KEEP_GRIPPERS = False  # whether Home may leave the grippers as they are
    JOINT_NAMES: tuple[str, ...] = ()  # one arm's joints as the page names them; empty: the page shows no joints

    # the classes the robot is built from
    CONFIG_PATH: Path  # its defaults: the config.yaml of its package
    ROBOT_CONFIG_CLASS: type[RobotConfig]  # its settings: the robot node of a station's configuration
    CONTROL_LOOP_CLASS: type[JointControlLoop]
    DRIVER_CLASS: type[JointDriver]  # made with the two configurations
    CAMERA_SOURCE_CLASS: type[CameraSource]  # made with the robot's settings
    ACTION_ADAPTER: ActionAdapter  # reads an action as one of its action models
    JOINT_ACTION_CLASS: type[JointPositions]  # its chunk of joint targets, the one action joint mode takes

    STATION_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "station.yaml"  # what every station starts from
    PLANNING_TIMEOUT_S = 30.0  # for planning one action or Home
    HOME_TIMEOUT_S = 420.0  # for a whole Home, the operator's review included
    FOREIGN_OBSERVATION = "Observation belongs to a different station session"  # the refusal of another session's

    def __init__(
        self,
        config: DictConfig,
        *,
        cameras: CameraSource | None = None,
        action_mode: ActionMode = ActionMode.CARTESIAN,
    ) -> None:
        if action_mode not in ActionMode:
            raise ValueError("Expected joint/cartesian mode")

        self._station_config = config  # what the two configurations are made from, again at every open
        self._control_config, self._robot_config = self.make_configs()  # of the open or the next session

        self.action_mode = action_mode
        self._cameras = cameras  # the caller's: every observation reads them, and the robot never closes them

        self._control_loop: JointControlLoop | None = None  # None while closed: making a robot opens no device
        self.session_id: str | None = None  # the identity of the open session; None while closed
        self.kinematics: BaseKinematics | None = None  # of the open session; None while closed and in joint mode
        self.last_error_diagnostics: dict[str, Any] | None = None  # what the last failed action's error recorded
        self._execution_lock = threading.Lock()  # held by the one action or recovery that runs
        self._lifecycle_lock = threading.RLock()  # held while the session opens, closes or is read as a whole

    @classmethod
    def load_config(cls, config_path: Path, overrides: Sequence[str]) -> DictConfig:
        """Returns the configuration of the station whose own file is ``config_path``: the config every station
        starts from, the defaults of the robot the file names, the file and the ``key=value`` ``overrides``, each read
        over the ones before it."""
        station_config = OmegaConf.load(config_path)
        if malformed_overrides := [override for override in overrides if "=" not in override]:
            raise ValueError(f"An override is key=value, which {', '.join(malformed_overrides)} is not")
        override_config = OmegaConf.from_dotlist(list(overrides))
        own_config = OmegaConf.merge(station_config, override_config)
        if "robot_class" not in own_config:
            raise ValueError(f"{config_path} names no robot: its robot_class is the class of the station's robot")
        try:
            robot_class = pkgutil.resolve_name(own_config.robot_class)
        except AttributeError as error:
            # a path that names nothing in the installed packages: no installed robot, refused in one sentence
            raise ValueError(f"robot_class names {own_config.robot_class}, which is no robot's class") from error
        if not (isinstance(robot_class, type) and issubclass(robot_class, BaseRobot)):
            raise ValueError(f"robot_class names {own_config.robot_class}, which is no robot's class")

        shared_config = OmegaConf.load(cls.STATION_CONFIG_PATH)
        config = OmegaConf.merge(
            shared_config, OmegaConf.load(robot_class.CONFIG_PATH), station_config, override_config
        )
        if unknown_keys := sorted(set(config) - set(shared_config)):
            raise ValueError(f"{config_path} holds keys that no station's configuration has: {', '.join(unknown_keys)}")
        # the directory of the station's own file, the base of the relative paths of the robot's settings: the
        # directory the path names, not the one of a file it links to
        config.config_dir = str(Path(config_path).absolute().parent)

        return config

    def make_configs(self) -> tuple[ControlConfig, RobotConfig]:
        """Returns the control loop's configuration and the robot's settings, made from the station's configuration;
        a value the station leaves ??? reads as a template's placeholder, which the robot's motion blockers name."""
        config = self._station_config
        control_config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ControlConfig), config.control))

        robot_node = OmegaConf.merge(OmegaConf.structured(self.ROBOT_CONFIG_CLASS), config.robot)
        for key in self.ROBOT_CONFIG_CLASS.RELATIVE_PATH_KEYS:
            # a value left ??? becomes a placeholder below; OmegaConf's select fails inside a section left ??? unless
            # it raises for it
            try:
                path = OmegaConf.select(robot_node, key, throw_on_missing=True)
            except MissingMandatoryValue:
                continue
            if path is not None and not Path(path).is_absolute():
                OmegaConf.update(robot_node, key, str((Path(config.config_dir) / path).resolve()))
        for key in OmegaConf.missing_keys(robot_node):
            OmegaConf.update(robot_node, key, self.ROBOT_CONFIG_CLASS.PLACEHOLDER.format(key=key))

        return control_config, OmegaConf.to_object(robot_node)

    @property
    def control_config(self) -> ControlConfig:
        """A copy of the control loop's configuration of the open or the next session."""
        return copy.deepcopy(self._control_config)

    @property
    def robot_config(self) -> RobotConfig:
        """A copy of the robot's settings of the open or the next session."""
        return copy.deepcopy(self._robot_config)

    @property
    def control_period_s(self) -> float:
        """The control period of the open or the next session, s."""
        return self._control_config.control_period_s

    @classmethod
    def profile(cls) -> dict[str, Any]:
        """Returns what the page lays the robot out with."""
        return {
            "name": cls.NAME,
            "title": cls.TITLE,
            "subtitle": cls.SUBTITLE,
            "arms": cls.ARMS_TEXT,
            "cameras": [{"role": camera.role, "label": camera.label} for camera in cls.CAMERAS],
            "home_sides": list(cls.HOME_ARM_CHOICES),
            "home_keep_grippers": cls.HOME_CAN_KEEP_GRIPPERS,
            "joints": list(cls.JOINT_NAMES),
            "formats": list(cls.ACTION_FORMATS),
        }

    @property
    def control_loop(self) -> JointControlLoop:
        """The control loop of the open session, for the robot's own motion code and commands; raises while the
        robot is closed."""
        control_loop = self._control_loop
        if control_loop is None:
            raise RuntimeError("Open the station first")

        return control_loop

    def open(self) -> Self:
        """Makes the two configurations again, opens the planner and the control loop, which enables the arms, and
        returns the robot; a failure closes what was opened."""
        with self._lifecycle_lock:
            if self._control_loop is not None:
                raise RuntimeError("Station already open")
            self._control_config, self._robot_config = self.make_configs()

            try:
                if self.action_mode == ActionMode.CARTESIAN:
                    self.open_planner()
                control_config, robot_config = self.control_config, self.robot_config
                driver = self.DRIVER_CLASS(control_config, robot_config)
                self._control_loop = self.CONTROL_LOOP_CLASS.open(driver, control_config, robot_config)
                self.session_id = str(uuid4())

                return self
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Stops, and releases the arms and the planner; a later ``open`` starts a new session."""
        with self._lifecycle_lock:
            try:
                if self._control_loop is not None:
                    try:
                        self._control_loop.shutdown()
                    finally:
                        if self._control_loop.status == ControlState.CLOSED:
                            self._control_loop = None
                            self.session_id = None
            finally:
                self.close_planner()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    @abstractmethod
    def open_planner(self) -> None:
        """Makes planning ready for the robot's action mode, or replaces a planner that failed; the caller holds the
        lifecycle lock."""

    @abstractmethod
    def close_planner(self) -> None:
        """Releases what ``open_planner`` made ready."""

    def open_cameras(self) -> CameraSource:
        """Returns a new source of the robot's cameras, which the caller closes."""
        return self.CAMERA_SOURCE_CLASS(self.robot_config)

    def status(self, with_snapshot: bool = False) -> BriefStatus | DetailedStatus:
        """Returns whether the robot is OFF, READY (a settled HOLD), BUSY or in FAULT; with ``with_snapshot``, the
        whole control snapshot too."""
        control_loop = self._control_loop
        if control_loop is None:
            return {"status": RobotStatus.OFF.value}

        snapshot = control_loop.snapshot()
        control_state = snapshot["status"]
        if control_state == ControlState.CLOSED:
            robot_status = RobotStatus.OFF
        elif control_state == ControlState.FAULT:
            robot_status = RobotStatus.FAULT
        elif control_state == ControlState.HOLD and snapshot["arrival"]["settled"]:
            robot_status = RobotStatus.READY
        else:
            robot_status = RobotStatus.BUSY

        if with_snapshot:
            return {**snapshot, "status": robot_status.value, "control_state": control_state}
        return {
            "status": robot_status.value,
            "settled": snapshot["arrival"]["settled"],
            "reached": snapshot["arrival"]["reached"],
            "fault": snapshot["fault"],
        }

    def observe(self) -> Observation:
        """Returns the measured joints, the cameras' frames and the control snapshot as one observation of the
        session."""
        with self._lifecycle_lock:
            control_loop, cameras, session_id = self._control_loop, self._cameras, self.session_id
            snapshot = self.control_loop.snapshot()
        SnapshotCheck.validate_snapshot(snapshot, self._control_config)

        try:
            frames = cameras.read() if cameras is not None else {}
        except BaseException:
            control_loop.stop()
            raise
        if session_id != self.session_id:
            raise RuntimeError("Station session changed during observation")

        return {
            "observation_id": str(uuid4()),
            "session_id": session_id,
            "captured_monotonic_s": time.monotonic(),
            "joint_positions": SnapshotCheck.state_vector(snapshot),
            "images": {role: frame["rgb"] for role, frame in frames.items()},
            "camera_timestamps": {
                role: {key: value for key, value in frame.items() if key != "rgb"} for role, frame in frames.items()
            },
            "control_snapshot": snapshot,
        }

    def execute(
        self,
        action: Any,
        *,
        reference_observation: Observation | None = None,
        timeout_s: float | None = None,
        cancellation: threading.Event | None = None,
        report_progress: Callable[[ActionPhase], None] | None = None,
    ) -> MotionResult:
        """Runs one action from a settled HOLD at ``reference_observation`` (by default a new observation), within
        ``timeout_s`` if one is given, and returns its result, which an action that did not complete has too."""
        if timeout_s is not None:
            SnapshotCheck.check_timeout(timeout_s, "Timeout")

        with self.action_guard(cancellation) as guard:
            control_loop = guard.control_loop
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            self.last_error_diagnostics = None
            parsed_action = None

            try:
                parsed_action = self.ACTION_ADAPTER.validate(action)
                is_joint_action = isinstance(parsed_action, self.JOINT_ACTION_CLASS)
                if self.action_mode == ActionMode.JOINT and not is_joint_action:
                    raise ValueError("Joint mode only accepts joint_positions")
                self.wait_for_settled_hold(deadline=deadline, guard=guard)
                observation = self.observe() if reference_observation is None else reference_observation
                self.verify_observation(observation)

                gripper_report = None
                if is_joint_action:
                    snapshot = self.run_sequence(parsed_action, observation, guard=guard, deadline=deadline)
                else:
                    guard.check()
                    if report_progress is not None:
                        report_progress(ActionPhase.PLANNING)
                    prepared_action = self.plan(observation, parsed_action, deadline=deadline)
                    guard.check()
                    if report_progress is not None:
                        report_progress(ActionPhase.EXECUTING)
                    gripper_report = self.run_prepared(observation, prepared_action, guard=guard, deadline=deadline)
                    snapshot = control_loop.snapshot()
                guard.check()

                return self.result(snapshot, action=parsed_action, gripper_report=gripper_report)
            except Exception as error:
                logger.exception("An action did not complete")
                self.last_error_diagnostics = getattr(error, "details", None)
                control_loop.stop()
                is_rejected = isinstance(error, ValueError) and not guard.has_dispatched

                return self.result(
                    control_loop.snapshot(),
                    status=MotionStatus.REJECTED if is_rejected else MotionStatus.FAILED,
                    error=error,
                    action=parsed_action,
                )

    @abstractmethod
    def plan(self, observation: Observation, action: StrictModel, *, deadline: float | None) -> PreparedAction:
        """Returns the arm move and the gripper request of the validated ``action`` from ``observation``, planned
        by ``deadline`` (None: the action has no deadline)."""

    @abstractmethod
    def result(
        self,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus = MotionStatus.COMPLETED,
        error: BaseException | None = None,
        action: StrictModel | None = None,
        gripper_report: dict[str, Any] | None = None,
    ) -> MotionResult:
        """Returns the robot's result of ``action`` from a control snapshot or the robot's status;
        ``gripper_report`` is the report of the grippers' operation, if the action ran one."""

    def home(
        self,
        *,
        cancellation: threading.Event | None = None,
        timeout_s: float = HOME_TIMEOUT_S,
        report_progress: Callable[[dict[str, Any]], None] | None = None,
        approve_plan: Callable[[HomePlan], bool] | None = None,
        **home_options: Any,
    ) -> HomeReport:
        """Returns the arms Home with the robot's ``home_options`` once ``approve_plan`` approves the plan, all within
        ``timeout_s``, and returns the Home report; ``report_progress`` receives each stage."""
        SnapshotCheck.check_timeout(timeout_s, "Timeout")
        deadline = time.monotonic() + timeout_s
        self.control_loop.snapshot()  # a closed robot is refused before a missing Home is
        if self._robot_config.home is None:
            raise ValueError("No home pose configured")

        with self.action_guard(cancellation) as guard:
            return self.run_home(
                guard, deadline=deadline, report_progress=report_progress, approve_plan=approve_plan, **home_options
            )

    @abstractmethod
    def run_home(
        self,
        guard: ActionGuard,
        *,
        deadline: float,
        report_progress: Callable[[dict[str, Any]], None] | None,
        approve_plan: Callable[[HomePlan], bool] | None,
        **home_options: Any,
    ) -> HomeReport:
        """Plans Home, has ``approve_plan`` approve the plan and runs it under ``guard`` by ``deadline``, and returns
        the Home report."""

    def recover(
        self, *, cancellation: threading.Event | None = None, timeout_s: float | None = None
    ) -> ControlSnapshot:
        """Reacquires a hold after a fault, within ``timeout_s`` (by default the control loop's own), without
        replacing the session or enabling motors, and returns the snapshot then."""
        control_loop = self._claim_execution()

        try:
            snapshot = control_loop.recover(cancellation=cancellation, timeout_s=timeout_s)
            self.last_error_diagnostics = None

            return snapshot
        finally:
            self._execution_lock.release()

    def stop(self) -> None:
        """Brakes any operation into a HOLD."""
        control_loop = self._control_loop
        if control_loop is not None:
            control_loop.stop()

    def check_fault(self) -> None:
        """Raises the fault the control loop has latched, with its diagnostics, if it has one."""
        control_loop = self._control_loop
        if control_loop is not None:
            SnapshotCheck.raise_for_fault(control_loop.snapshot())

    def wait_stationary(
        self, observation: Observation | None = None, *, cancellation: threading.Event | None = None
    ) -> ControlSnapshot:
        """Waits, outside an action, for a settled HOLD, of the operation of ``observation`` if one is given, and
        returns its snapshot; a stop, a fault or another operation ends the wait with an error."""
        return self.wait_for_settled_hold(observation, guard=ActionGuard(self.control_loop, cancellation))

    def is_holding(self, observation: Observation) -> bool:
        """Whether the arms still hold the operation of ``observation`` and have settled; raises once they no longer
        hold it."""
        return self.hold_snapshot(observation)["arrival"]["settled"]

    def wait_settled(self, timeout_s: float) -> ControlSnapshot:
        """Waits for a stop or an action to end in a settled HOLD and returns its snapshot."""
        return self.control_loop.wait_for_hold(timeout_s)

    def model_state(self, joint_positions: Sequence[float], format_name: str) -> StrictModel:
        """Returns what the model reads of the measured ``joint_positions`` in the action format ``format_name``."""
        self._check_format(format_name)
        if self.kinematics is None:
            raise RuntimeError("Open a Cartesian Station first")

        return self.kinematics.quantities(joint_positions)

    def failure_result(
        self,
        error: BaseException,
        *,
        format_name: str,
        status: MotionStatus = MotionStatus.FAILED,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus | None = None,
    ) -> MotionResult:
        """Returns the result of an action of the format ``format_name`` that did not complete, measured at
        ``snapshot`` (by default now)."""
        self._check_format(format_name)

        return self.result(
            self.status(with_snapshot=True) if snapshot is None else snapshot, status=status, error=error
        )

    @contextmanager
    def action_guard(self, cancellation: threading.Event | None = None) -> Iterator[ActionGuard]:
        """Runs one action alone on the open session, under the guard this yields; any failure stops the arms."""
        with self._lifecycle_lock:
            self.control_loop.snapshot()  # a read of the open session: a closed robot is refused before the claim
            control_loop = self._claim_execution()
            guard = ActionGuard(control_loop, cancellation)

        try:
            yield guard
        except BaseException:
            control_loop.stop()
            raise
        finally:
            self._execution_lock.release()

    def wait_for_settled_hold(
        self, observation: Observation | None = None, *, deadline: float | None = None, guard: ActionGuard
    ) -> ControlSnapshot:
        """Waits under ``guard``, until ``deadline``, for a settled HOLD, of the operation of ``observation`` if one
        is given, and returns its snapshot."""
        return SnapshotCheck.wait_ready(
            lambda: self.hold_snapshot(observation),
            timeout_s=self.remaining_s(deadline, self._control_config.settle_timeout_s),
            control_period_s=self._control_config.control_period_s,
            guard=guard,
            make_timeout_error=lambda snapshot: StabilityError(
                "Station did not pass the continuous stability window", snapshot["settling"]
            ),
        )

    def verify_observation(self, observation: Observation) -> ControlSnapshot:
        """Returns the fresh snapshot of the HOLD when the open session still holds the operation of
        ``observation``, and raises otherwise."""
        control_loop = self._control_loop
        if control_loop is None:
            raise ValueError(self.FOREIGN_OBSERVATION)

        with control_loop.lock:
            snapshot = self.hold_snapshot(observation)
            if control_loop is not self._control_loop:
                raise ValueError("Station session changed during validation")

            return snapshot

    def submit_grippers(self, guard: ActionGuard, observation: Observation, gripper_request: Any) -> GripperOperation:
        """Starts the gripper operation for ``gripper_request`` under ``guard`` while the arms hold ``observation``,
        settled, and returns it."""
        control_loop = guard.control_loop

        with control_loop.lock:
            if not self.verify_observation(observation)["arrival"]["settled"]:
                raise RuntimeError("Gripper reference is no longer stable")

            return guard.submit(control_loop.begin_grippers, gripper_request)

    def hold_snapshot(self, observation: Observation | None = None) -> ControlSnapshot:
        """Returns the fresh feedback of a HOLD, of the session and the operation of ``observation`` if one is
        given, and raises when the arms do not hold it."""
        snapshot = SnapshotCheck.validate_snapshot(
            self.control_loop.snapshot(), self._control_config, requires_hold=True
        )
        if observation is not None:
            if observation.get("session_id") != self.session_id:
                raise ValueError(self.FOREIGN_OBSERVATION)
            SnapshotCheck.validate_reference(snapshot, observation)

        return snapshot

    def _claim_execution(self) -> JointControlLoop:
        """Returns the control loop of the open session with the one place of a running action or recovery taken,
        and raises while the robot is closed or another one runs; the caller releases the execution lock."""
        with self._lifecycle_lock:
            control_loop = self.control_loop
            if not self._execution_lock.acquire(blocking=False):
                raise RuntimeError("A station action is already running")

            return control_loop

    def _check_format(self, format_name: str) -> None:
        """Raises unless ``format_name`` names an action format of the robot."""
        if format_name not in self.ACTION_FORMATS:
            raise ValueError(f"{self.TITLE} has no action format {format_name!r}")
