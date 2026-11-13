from typing import Any

import numpy as np

from pear.deploy.station.control.joint.trajectory import JointMove
from pear.deploy.station.control.motion.homing import Homing
from pear.deploy.station.control.robots.so101.joint.grippers import GripperOutcome
from pear.deploy.station.control.robots.so101.layout import ARM
from pear.deploy.station.control.robots.so101.motion.planning import SO101Planning
from pear.deploy.station.control.schemas import ControlSnapshot, HomePlan, Observation


class SO101Homing(Homing):
    """The SO-101's return to Home: the gripper opens, then the arm returns directly; the plan and its recheck run in
    the caller's thread, bound to the observation they start from."""

    # the keys of the plan that hold the joint vector once the gripper has opened, at Home, and whether it opens
    START_TARGETS_KEY = "start_target"
    GOAL_TARGETS_KEY = "goal_target"
    OPENING_KEY = "open_gripper"

    def plan(self, observation: Observation, deadline: float) -> tuple[HomePlan, JointMove | None]:
        """Returns the checked plan of Home from ``observation`` and its move, while the arm still holds the operation
        of ``observation``."""
        return self.robot.run_while_observation_holds(
            observation, lambda: SO101Planning.checked_home(self.control_config, self.robot_config, observation)
        )

    def is_opened(self, gripper_report: dict[str, Any]) -> bool:
        """Whether the report of the jaw's operation says that it is open."""
        return gripper_report["outcome"] == GripperOutcome.OPEN

    def recheck(self, observation: Observation, move: JointMove, deadline: float) -> None:
        """Raises unless the approved ``move`` still starts at the held targets and stays within the bounds with the
        feedback of ``observation``."""
        self.robot.run_while_observation_holds(
            observation, lambda: SO101Planning.recheck_home_move(self.robot_config, observation, move)
        )

    def tolerance(self) -> np.ndarray:
        """Returns Home's tolerance of each coordinate: of each arm joint and of the gripper."""
        return np.array(self.robot_config.home.joint_tolerance + [self.robot_config.gripper_tolerance])

    def accuracy(self, snapshot: ControlSnapshot) -> dict[str, Any]:
        """Returns the arm's error against the targets it is judged against, by Home's own tolerance."""
        return self.arm_accuracy(snapshot, ARM)
