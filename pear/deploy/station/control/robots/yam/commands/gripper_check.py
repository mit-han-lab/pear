from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # an annotation only: the command line imports this module without the station's packages
    from pear.deploy.station.control.robots.yam.robot import YAMRobot


class GripperCheckMode(StrEnum):
    """How long a gripper check lets the gripper move."""

    SHORT = "short"  # a short time of its endpoint trajectory, then a stop
    FULL = "full"  # the whole gripper operation, to its endpoint


class GripperCheck:
    """One supervised check of a gripper through the robot's gripper operation, under an action's guard: in full, or
    for a short time only."""

    SHORT_CHECK_S = 0.4  # how long a short check lets the gripper's endpoint trajectory run
    HOLD_TIMEOUT_S = 10.0  # for the hold after a short check
    STOP_REASON = "commissioning_short_check_complete"  # how the stop of a short check ends the motion

    def __init__(self, robot: YAMRobot, arm: str, opens: bool, mode: GripperCheckMode) -> None:
        self.robot = robot  # open, in joint mode
        self.arm, self.opens, self.mode = arm, opens, mode

    def run(self) -> dict[str, Any]:
        """Moves the gripper as the mode says and returns the snapshots of the hold before and after, with what the
        gripper's operation did."""
        robot = self.robot
        report = {"side": self.arm, "opening": self.opens, "mode": self.mode.value}

        with robot.action_guard() as guard:
            control_loop = guard.control_loop
            robot.wait_for_settled_hold(guard=guard)
            observation = robot.observe()
            guard.check()
            report["before"] = robot.hold_snapshot()

            if self.mode == GripperCheckMode.FULL:
                gripper_request = {self.arm: self.opens}
                report.update(
                    robot.operate_grippers(gripper_request, observation=observation, guard=guard, deadline=None)
                )
            else:
                # the gripper's endpoint trajectory runs for a short time only, with no partial opening of its own and
                # the control loop's limits as they are
                operation = robot.submit_grippers(guard, observation, {self.arm: self.opens})
                start_time = time.monotonic()
                while not operation.is_complete:
                    guard.check()
                    if time.monotonic() - start_time >= self.SHORT_CHECK_S:
                        guard.submit(control_loop.stop, reason=self.STOP_REASON)
                        break
                    time.sleep(control_loop.control_config.control_period_s)
                guard.check()
                control_loop.wait_for_hold(self.HOLD_TIMEOUT_S, guard=guard)
                report.update(
                    completed=operation.is_complete,
                    short_check_stopped=not operation.is_complete,
                    elapsed_seconds=time.monotonic() - start_time,
                )

            report["after"] = robot.hold_snapshot()

            return report
