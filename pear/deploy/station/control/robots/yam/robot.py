from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.motion.actions import ActionAdapter, StrictModel
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.motion.results import MotionStatus
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE, CAMERAS
from pear.deploy.station.control.robots.yam.hardware.cameras import LiveCameras
from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.layout import BOTH_ARMS, LAYOUT
from pear.deploy.station.control.robots.yam.motion.actions import (
    BimanualDelta,
    CartesianDelta,
    GripperAction,
    JointPositions,
)
from pear.deploy.station.control.robots.yam.motion.homing import YAMHoming
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.motion.planner import PlannerProcess
from pear.deploy.station.control.robots.yam.motion.results import YAMMotionResult
from pear.deploy.station.control.schemas import (
    BriefStatus,
    ControlSnapshot,
    DetailedStatus,
    HomePlan,
    HomeReport,
    Observation,
)


class YAMRobot(BaseRobot):
    """Both YAM arms with their wrist and overhead cameras: actions are planned in a planner process, and the arms
    move through the SDK's process."""

    NAME = "yam"
    TITLE = "YAM"
    SUBTITLE = "BIMANUAL ROBOTICS"
    ARMS_TEXT = "Both arms"
    CAMERAS = CAMERAS
    ACTION_FORMATS = {BIMANUAL_POSE.name: BIMANUAL_POSE}
    HOME_ARM_CHOICES = (BOTH_ARMS, *LAYOUT.arms)
    HOME_CAN_KEEP_GRIPPERS = True

    CONFIG_PATH = Path(__file__).with_name("config.yaml")
    ROBOT_CONFIG_CLASS = YAMConfig
    CONTROL_LOOP_CLASS = YAMControlLoop
    DRIVER_CLASS = YAMDriver
    CAMERA_SOURCE_CLASS = LiveCameras
    ACTION_ADAPTER = ActionAdapter(JointPositions, CartesianDelta, BimanualDelta, GripperAction)
    JOINT_ACTION_CLASS = JointPositions

    def __init__(
        self,
        config: DictConfig,
        *,
        cameras: CameraSource | None = None,
        action_mode: ActionMode = ActionMode.CARTESIAN,
    ) -> None:
        super().__init__(config, cameras=cameras, action_mode=action_mode)
        self.planner: PlannerProcess | None = None  # None while closed, and in joint mode before a Home

    def make_configs(self) -> tuple[ControlConfig, YAMConfig]:
        """Returns the control loop's configuration and the robot's settings, refused when a gripper's contact must
        hold as long as a settle may take."""
        control_config, robot_config = super().make_configs()
        if robot_config.gripper_contact_confirm_s >= control_config.settle_timeout_s:
            raise ValueError("settle_timeout_s must exceed gripper_contact_confirm_s")

        return control_config, robot_config

    def home(
        self,
        side: str = BOTH_ARMS,
        *,
        opens_grippers: bool = True,
        cancellation: threading.Event | None = None,
        timeout_s: float = BaseRobot.HOME_TIMEOUT_S,
        report_progress: Callable[[dict[str, Any]], None] | None = None,
        approve_plan: Callable[[HomePlan], bool] | None = None,
    ) -> HomeReport:
        """Opens the grippers of ``side`` unless ``opens_grippers`` is false, returns ``side`` directly Home within
        ``timeout_s`` and returns the Home report."""
        if side not in self.HOME_ARM_CHOICES:
            raise ValueError("Home side must be left, right or both")
        if not isinstance(opens_grippers, bool):
            raise ValueError("open_grippers must be true or false")

        return super().home(
            cancellation=cancellation,
            timeout_s=timeout_s,
            report_progress=report_progress,
            approve_plan=approve_plan,
            side=side,
            opens_grippers=opens_grippers,
        )

    def open_planner(self) -> None:
        """Starts the planner process, or replaces one that failed; in Cartesian mode the model's forward kinematics
        are made here too."""
        if self.planner is not None:
            if self.planner.is_closed:
                self.planner = PlannerProcess(self.control_config, self.robot_config)
            return

        kinematics = None
        if self.action_mode == ActionMode.CARTESIAN:
            if self._robot_config.geometry is None:
                raise ValueError("Cartesian motion requires station geometry")
            # the model state's forward kinematics run here; the inverse kinematics and the moves in the planner
            kinematics = YAMKinematics(self.robot_config)
        self.planner = PlannerProcess(self.control_config, self.robot_config)
        self.kinematics = kinematics

    def close_planner(self) -> None:
        """Stops the planner process."""
        if self.planner is not None:
            self.planner.close()
            self.planner = self.kinematics = None

    def plan(self, observation: Observation, action: StrictModel, *, deadline: float | None) -> PreparedAction:
        """Returns the inverse kinematics and the arm move of ``action`` from ``observation``, which the planner
        process plans by ``deadline``; a planner that failed is replaced first."""
        with self._lifecycle_lock:
            self.open_planner()

        return self.planner.plan(
            self, observation, action, timeout_s=self.remaining_s(deadline, self.PLANNING_TIMEOUT_S)
        )

    def result(
        self,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus = MotionStatus.COMPLETED,
        error: BaseException | None = None,
        action: StrictModel | None = None,
        gripper_report: dict[str, Any] | None = None,
    ) -> YAMMotionResult:
        """Returns the result of an action from a control snapshot, with each gripper's outcome; every action gets
        the model's state."""
        return YAMMotionResult.from_snapshot(
            snapshot,
            status=status,
            error=error,
            kinematics=self.kinematics,
            gripper_outcomes=gripper_report["outcomes"] if gripper_report else None,
        )

    def run_home(
        self,
        guard: ActionGuard,
        *,
        deadline: float,
        report_progress: Callable[[dict[str, Any]], None] | None,
        approve_plan: Callable[[HomePlan], bool] | None,
        side: str,
        opens_grippers: bool,
    ) -> HomeReport:
        """Plans Home of ``side``, has ``approve_plan`` approve the plan and runs it under ``guard``, and returns the
        Home report."""
        with self._lifecycle_lock:
            guard.check()
            self.open_planner()

        homing = YAMHoming(self, guard, report_progress, side, opens_grippers)

        return homing.run(approve_plan=approve_plan, deadline=deadline)
