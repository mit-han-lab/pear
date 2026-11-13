from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import RobotCommand

if TYPE_CHECKING:  # an annotation only: the station's packages are imported where the command runs
    from pear.deploy.station.control.robots.so101.robot import SO101Robot

# the station's packages and the robot's own modules, the commands' module among them, are imported inside the
# methods: the command line lists the commands without them, and the commands' module imports this one


class SafetyCheck(RobotCommand):
    """MOVES the arm: enables it, goes Home, tries a step it cannot reach, moves down and STOPs part way, returns to
    the start and releases it."""

    NAME = "safety"
    HELP = "MOVES the arm: Home, unreachable step, STOP, back"
    MOVES_ROBOT = True
    UNREACHABLE_X_CM = 30.0  # the step forward the arm cannot reach from the Home of the package's config.yaml
    DOWN_Z_CM = -6.0  # the step down that the STOP interrupts
    STOP_AFTER_S = 0.4  # into the downward move
    MOVE_START_TIMEOUT_S = 10.0  # for the downward move to begin
    MOVE_START_POLL_S = 0.005  # between two looks at whether it began
    JOIN_TIMEOUT_S = 30.0  # for the downward action's thread to end after the STOP
    TARGET_CHANGE_TOLERANCE = 1e-9  # the change of the targets, rad, below which a refused step moved nothing
    TARGET_CHANGE_DECIMALS = 6  # of the change of the targets in the report, rad
    PROGRESS_DECIMALS = 3  # of the share of the move made before the STOP
    SECONDS_DECIMALS = 2  # of the time from the STOP to the hold

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the confirmation that the arm may move."""
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        MaintenanceSteps.add_confirmation(parser)

    def run(self) -> dict[str, dict[str, Any]]:
        """Returns what the step out of reach and the STOP each did and whether each passed."""
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        # the settings alone first: an uncommissioned arm is refused before the kinematics load
        self.new_robot(ActionMode.JOINT).robot_config.require_commissioned()
        print(
            "Backend real: enable the arm, go Home, try an unreachable step, move down and STOP part way, return to the "
            "start, then disable torque.",
            file=sys.stderr,
            flush=True,
        )

        report = {}
        robot = self.new_robot(ActionMode.CARTESIAN)
        with robot:
            start = MaintenanceSteps.held_targets(robot)
            try:
                robot.home(report_progress=MaintenanceSteps.print_home_stage)
                report["unreachable"] = self._unreachable_step(robot)
                report["stop"] = self._stopped_step(robot)
            finally:
                MaintenanceSteps.park(robot, start)

        return report

    def _unreachable_step(self, robot: SO101Robot) -> dict[str, Any]:
        """Tries the step out of reach and returns whether it was refused without a change of the targets."""
        import numpy as np

        from pear.deploy.station.control.motion.results import MotionStatus
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        print(f"unreachable step: x +{self.UNREACHABLE_X_CM:g} cm", file=sys.stderr, flush=True)
        targets_before = np.array(MaintenanceSteps.held_targets(robot))
        result = robot.execute(MaintenanceSteps.arm_delta(x=self.UNREACHABLE_X_CM))
        targets_after = np.array(MaintenanceSteps.held_targets(robot))

        target_change_rad = float(np.abs(targets_after - targets_before).max())
        return {
            "status": result.status.value,
            "error": result.error,
            "target_change_rad": round(target_change_rad, self.TARGET_CHANGE_DECIMALS),
            "passed": result.status == MotionStatus.REJECTED
            and np.allclose(targets_after, targets_before, atol=self.TARGET_CHANGE_TOLERANCE),
        }

    def _stopped_step(self, robot: SO101Robot) -> dict[str, Any]:
        """Moves the arm down, STOPs it part way and returns how far it went, how it ended and whether it held."""
        import numpy as np

        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps
        from pear.deploy.station.control.robots.so101.layout import ARM, GRIPPER_INDEX
        from pear.deploy.station.control.schemas import ControlState

        print(f"STOP {self.STOP_AFTER_S} s into a z {self.DOWN_Z_CM:g} cm move", file=sys.stderr, flush=True)
        control_loop = robot.control_loop
        origin = np.array(MaintenanceSteps.held_targets(robot))
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.update(result=robot.execute(MaintenanceSteps.arm_delta(z=self.DOWN_Z_CM)))
        )

        worker.start()
        try:
            deadline = time.monotonic() + self.MOVE_START_TIMEOUT_S
            while control_loop.status != ControlState.RUNNING and worker.is_alive():
                if time.monotonic() > deadline:
                    raise TimeoutError("The move did not start")
                time.sleep(self.MOVE_START_POLL_S)
            with control_loop.lock:
                goal = np.array(control_loop.requested_position())
            time.sleep(self.STOP_AFTER_S)
            stopped_at = time.monotonic()
        finally:
            # never return to the start while the move's action may still dispatch
            robot.stop()
            worker.join(self.JOIN_TIMEOUT_S)

        hold_snapshot = control_loop.wait_for_hold(MaintenanceSteps.HOLD_TIMEOUT_S)
        final_targets = np.array(hold_snapshot["targets"][ARM])
        span = goal[:GRIPPER_INDEX] - origin[:GRIPPER_INDEX]
        progress = float(np.dot(final_targets[:GRIPPER_INDEX] - origin[:GRIPPER_INDEX], span) / np.dot(span, span))
        stopped_result = outcome.get("result")

        return {
            "status": stopped_result.status.value if stopped_result else None,
            "error": stopped_result.error if stopped_result else None,
            "progress": round(progress, self.PROGRESS_DECIMALS),
            "stop_to_hold_s": round(time.monotonic() - stopped_at, self.SECONDS_DECIMALS),
            "end_reason": hold_snapshot["motion"]["end_reason"],
            "runtime_status": hold_snapshot["status"],
            "settled": hold_snapshot["arrival"]["settled"],
            "passed": 0 < progress < 1 and hold_snapshot["status"] == ControlState.HOLD,
        }
