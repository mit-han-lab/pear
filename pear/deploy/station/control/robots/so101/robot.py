import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.motion.actions import ActionAdapter, StrictModel
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.motion.results import MotionStatus
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.formats import SINGLE_ARM_POSE_FORMAT
from pear.deploy.station.control.robots.so101.hardware.cameras import WristCamera
from pear.deploy.station.control.robots.so101.hardware.driver import SO101Driver
from pear.deploy.station.control.robots.so101.joint.control_loop import SO101ControlLoop
from pear.deploy.station.control.robots.so101.layout import JOINTS
from pear.deploy.station.control.robots.so101.motion.actions import GripperAction, JointPositions, SingleArmDelta
from pear.deploy.station.control.robots.so101.motion.homing import SO101Homing
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.robots.so101.motion.planning import SO101Planning
from pear.deploy.station.control.robots.so101.motion.results import SO101MotionResult
from pear.deploy.station.control.schemas import (
    BriefStatus,
    ControlSnapshot,
    DetailedStatus,
    HomePlan,
    HomeReport,
    Observation,
)


class SO101Robot(BaseRobot):
    """The SO-101: one follower arm on its Feetech servos with a wrist camera, whose kinematics and planning run in
    the caller's thread."""

    NAME = "so101"
    TITLE = "SO-101"
    SUBTITLE = "SINGLE-ARM ROBOTICS"
    ARMS_TEXT = "One arm"
    CAMERAS = (WristCamera.CAMERA,)
    ACTION_FORMATS = {SINGLE_ARM_POSE_FORMAT.name: SINGLE_ARM_POSE_FORMAT}
    JOINT_NAMES = tuple(joint.replace("_", " ").capitalize() for joint in JOINTS)  # Shoulder pan, Shoulder lift, ...

    CONFIG_PATH = Path(__file__).with_name("config.yaml")
    ROBOT_CONFIG_CLASS = SO101Config
    CONTROL_LOOP_CLASS = SO101ControlLoop
    DRIVER_CLASS = SO101Driver
    CAMERA_SOURCE_CLASS = WristCamera
    KINEMATICS_CLASS = SO101Kinematics  # the model of the arm that Cartesian actions are solved on
    ACTION_ADAPTER = ActionAdapter(JointPositions, SingleArmDelta, GripperAction)
    JOINT_ACTION_CLASS = JointPositions

    def __init__(
        self,
        config: DictConfig,
        *,
        cameras: CameraSource | None = None,
        action_mode: ActionMode = ActionMode.CARTESIAN,
    ) -> None:
        super().__init__(config, cameras=cameras, action_mode=action_mode)

        # a Cartesian robot's kinematics load now, before the page connects: the first load holds the GIL for seconds,
        # which during an enable would starve the page's heartbeats; without geometry the open refuses instead
        if action_mode == ActionMode.CARTESIAN and self.robot_config.geometry is not None:
            self.KINEMATICS_CLASS.shared_instance(self.robot_config)

    def home(
        self,
        *,
        cancellation: threading.Event | None = None,
        timeout_s: float = BaseRobot.HOME_TIMEOUT_S,
        report_progress: Callable[[dict[str, Any]], None] | None = None,
        approve_plan: Callable[[HomePlan], bool] | None = None,
    ) -> HomeReport:
        """Opens the gripper and returns the arm directly Home; an SO-101 Home takes no options."""
        return super().home(
            cancellation=cancellation, timeout_s=timeout_s, report_progress=report_progress, approve_plan=approve_plan
        )

    def open_planner(self) -> None:
        """Makes the shared kinematics the session's, loaded before the motors are enabled."""
        if self._robot_config.geometry is None:
            raise ValueError("Cartesian motion requires geometry")

        self.kinematics = self.KINEMATICS_CLASS.shared_instance(self.robot_config)

    def close_planner(self) -> None:
        """Forgets the session's kinematics; the shared ones stay loaded."""
        self.kinematics = None

    def plan(self, observation: Observation, action: StrictModel, *, deadline: float | None) -> PreparedAction:
        """Returns the arm move and the gripper request of ``action`` from ``observation``, which the arm must still
        hold before and after; refused once ``deadline`` has passed."""
        self.remaining_s(deadline, self.PLANNING_TIMEOUT_S)  # planning runs in this thread: refused once late

        return self.run_while_observation_holds(
            observation, lambda: SO101Planning.checked_action(self._robot_config, self.kinematics, observation, action)
        )

    def run_while_observation_holds(self, observation: Observation, planning_step: Callable[[], Any]) -> Any:
        """Runs ``planning_step`` bound to ``observation``, whose operation the arm must hold before and after, and
        returns what it returns."""
        self.verify_observation(observation)
        planned = planning_step()
        self.verify_observation(observation)

        return planned

    def result(
        self,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus = MotionStatus.COMPLETED,
        error: BaseException | None = None,
        action: StrictModel | None = None,
        gripper_report: dict[str, Any] | None = None,
    ) -> SO101MotionResult:
        """Returns the result of ``action`` from a control snapshot or the robot's status; a pose delta's carries the
        model's state of the measured joints."""
        return SO101MotionResult.from_snapshot(
            snapshot,
            status=status,
            error=error,
            kinematics=self.kinematics if isinstance(action, SingleArmDelta) else None,
            gripper_outcome=None if gripper_report is None else gripper_report["outcome"],
        )

    def failure_result(
        self,
        error: BaseException,
        *,
        format_name: str,
        status: MotionStatus = MotionStatus.FAILED,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus | None = None,
    ) -> SO101MotionResult:
        """Returns the result of an action of the format ``format_name`` that did not complete, measured at
        ``snapshot`` (by default now), with the model's state of the measured joints."""
        self._check_format(format_name)

        return SO101MotionResult.from_snapshot(
            self.status(with_snapshot=True) if snapshot is None else snapshot,
            status=status,
            error=error,
            kinematics=self.kinematics,
        )

    def run_home(
        self,
        guard: ActionGuard,
        *,
        deadline: float,
        report_progress: Callable[[dict[str, Any]], None] | None,
        approve_plan: Callable[[HomePlan], bool] | None,
    ) -> HomeReport:
        """Plans Home, has ``approve_plan`` approve the plan and runs it under ``guard`` by ``deadline``, and returns
        the Home report."""
        return SO101Homing(self, guard, report_progress).run(approve_plan=approve_plan, deadline=deadline)
