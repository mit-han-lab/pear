from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.trajectory import JointMove
from pear.deploy.station.control.motion.homing import HomeStage, Homing
from pear.deploy.station.control.robots.yam.joint.grippers import GripperOutcome
from pear.deploy.station.control.schemas import ControlSnapshot, HomePlan, HomeReport, Observation

if TYPE_CHECKING:  # an annotation only: the robot's module imports this one
    from pear.deploy.station.control.robots.yam.robot import YAMRobot


class YAMHoming(Homing):
    """Home of both YAM arms or one: the planner process plans and rechecks the direct return, and the grippers of
    the returning arms open first unless they are kept."""

    START_TARGETS_KEY = "start_targets"
    GOAL_TARGETS_KEY = "goal_targets"
    OPENING_KEY = "opening_commands"

    def __init__(
        self,
        robot: YAMRobot,
        guard: ActionGuard,
        report_progress: Callable[[dict[str, Any]], None] | None,
        side: str,
        opens_grippers: bool,
    ) -> None:
        super().__init__(robot, guard, report_progress)
        self.side = side  # the arm that returns, or both
        self.opens_grippers = opens_grippers  # whether the returning arms' grippers open first

    def run(self, *, approve_plan: Callable[[HomePlan], bool] | None, deadline: float) -> HomeReport:
        """Runs Home and returns its report, which starts with the side and whether the grippers opened."""
        report = super().run(approve_plan=approve_plan, deadline=deadline)

        return {"side": self.side, "open_grippers": self.opens_grippers, **report}

    def plan(self, observation: Observation, deadline: float) -> tuple[HomePlan, JointMove | None]:
        """Returns the plan of Home from ``observation``, which the planner process checks, and its arm move."""
        robot = self.robot
        plan, move = robot.planner.plan_home(
            robot,
            observation,
            self.side,
            opens_grippers=self.opens_grippers,
            timeout_s=robot.remaining_s(deadline, robot.PLANNING_TIMEOUT_S),
        )
        self.arms = plan["active_sides"]

        return plan, move

    def is_opened(self, gripper_report: dict[str, Any]) -> bool:
        """Whether every gripper of the report is open."""
        return all(outcome == GripperOutcome.OPEN for outcome in gripper_report["outcomes"].values())

    def recheck(self, observation: Observation, move: JointMove, deadline: float) -> None:
        """Raises unless the planner process finds the approved ``move`` still valid with ``observation``."""
        robot = self.robot
        robot.planner.recheck_home(
            robot, observation, move, timeout_s=robot.remaining_s(deadline, robot.PLANNING_TIMEOUT_S)
        )

    def tolerance(self) -> np.ndarray:
        """Returns Home's tolerance of each coordinate of an arm; a kept gripper, which may hold an object, is not
        judged."""
        robot_config = self.robot_config
        gripper_tolerance = robot_config.gripper_tolerance if self.opens_grippers else np.inf

        return np.array(robot_config.home.joint_tolerance + [gripper_tolerance])

    def accuracy(self, snapshot: ControlSnapshot) -> dict[str, Any]:
        """Returns each returning arm's error against its targets, judged by Home's own tolerance."""
        accuracy = {}

        for arm in self.arms:
            arm_accuracy = self.arm_accuracy(snapshot, arm)
            accuracy[arm] = {
                "reached": arm_accuracy.pop("reached"),
                "gripper_open": arm_accuracy.pop("gripper_open"),
                "gripper_kept": not self.opens_grippers,
                **arm_accuracy,
            }

        return accuracy

    def report_stage(self, stage: HomeStage, **data: Any) -> None:
        """Enters ``stage`` and reports it with ``data``, and with the returning arms while the grippers open and the
        arms return."""
        if stage in (HomeStage.OPENING, HomeStage.HOMING):
            data = {"sides": self.arms, **data}

        super().report_stage(stage, **data)
