import argparse
import sys
from pathlib import Path
from typing import Any

from pear.deploy.station.control.commands import RobotCommand

# the station's packages and the robot's own modules, the commands' module among them, are imported inside the
# methods: the command line lists the commands without them, and the commands' module imports this one


class HomeCheck(RobotCommand):
    """MOVES the arm: enables it, opens the gripper, goes Home, photographs the wrist view, returns to the start and
    releases it."""

    NAME = "home"
    HELP = "MOVES the arm: Home, wrist photo, back to the start"
    MOVES_ROBOT = True
    SECONDS_DECIMALS = 2  # of the move's duration in the report

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the confirmation that the arm may move."""
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        MaintenanceSteps.add_confirmation(parser)

    def run(self) -> dict[str, Any]:
        """Returns how long the move to Home took, the arm at Home with Home's accuracy, the photos and the arm back
        at its start."""
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        photo_dir = Path(self.load_config().run_dir) / MaintenanceSteps.PHOTO_FOLDER
        robot = self.new_robot(ActionMode.JOINT)
        robot.robot_config.require_commissioned()  # before the wrist camera opens
        print(
            "Backend real: enable the arm, open the gripper, go Home, take a wrist photo, return to the start, then "
            "disable torque.",
            file=sys.stderr,
            flush=True,
        )
        # the camera opens before the robot is made with it; a station without a camera's section takes no photo
        cameras = None if robot.robot_config.camera is None else robot.open_cameras()

        try:
            robot = self.new_robot(ActionMode.JOINT, cameras)
            with robot:
                start = MaintenanceSteps.held_targets(robot)
                try:
                    home_report = robot.home(report_progress=MaintenanceSteps.print_home_stage)
                    photos = MaintenanceSteps.photograph(robot, photo_dir)
                finally:
                    park_snapshot = MaintenanceSteps.park(robot, start)
        finally:
            if cameras is not None:
                cameras.close()

        return {
            "trajectory_seconds": round(home_report["plan"]["trajectory_seconds"], self.SECONDS_DECIMALS),
            "home": MaintenanceSteps.summary(home_report["after"]) | {"accuracy": home_report["results"]},
            "photos": photos,
            "returned": MaintenanceSteps.summary(park_snapshot),
        }
