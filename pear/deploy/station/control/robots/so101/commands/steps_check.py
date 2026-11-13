import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pear.deploy.station.control.commands import RobotCommand

# the station's packages and the robot's own modules, the commands' module among them, are imported inside the
# methods: the command line lists the commands without them, and the commands' module imports this one


class StepsCheck(RobotCommand):
    """MOVES the arm: enables it, goes Home, makes small Cartesian steps, returns to the start and releases it."""

    NAME = "steps"
    HELP = "MOVES the arm: Home, small Cartesian deltas, back"
    MOVES_ROBOT = True
    # the steps without --deltas, cm and rad: each quantity there and back, from the Home of the package's config.yaml
    DEFAULT_DELTAS = [
        {"x": 2},
        {"x": -2},
        {"y": 2},
        {"y": -2},
        {"z": -2},
        {"z": 2},
        {"pitch": 0.2},
        {"pitch": -0.2},
        {"roll": 0.3},
        {"roll": -0.3},
    ]
    MEASURED_DECIMALS = 3  # of a step's measured change
    ERROR_DECIMALS = 4  # of a step's largest joint error, rad

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the confirmation, the steps and whether they are photographed."""
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        MaintenanceSteps.add_confirmation(parser)
        parser.add_argument("--deltas", help='JSON list, for example \'[{"x": 2}, {"x": -2}]\'')
        parser.add_argument("--photos", action="store_true", help="Save the wrist view at Home and after each step")

    def run(self) -> list[dict[str, Any]]:
        """Returns each step's commanded and measured change, its status, whether it reached its target, its largest
        joint error and its error."""
        from pear.deploy.station.control.motion.results import MotionStatus
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps
        from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX
        from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics

        # the settings alone first: an uncommissioned arm is refused before the camera opens and the kinematics load
        settings_robot = self.new_robot(ActionMode.JOINT)
        settings_robot.robot_config.require_commissioned()
        deltas = json.loads(self.args.deltas) if self.args.deltas else self.DEFAULT_DELTAS
        for delta in deltas:
            if unknown_quantities := set(delta) - set(SO101Kinematics.ARM_QUANTITIES):
                raise ValueError(f"The arm has no quantities {sorted(unknown_quantities)}")
        print(
            f"Backend real: enable the arm, go Home, run {len(deltas)} Cartesian steps, return to the start, then "
            "disable torque.",
            file=sys.stderr,
            flush=True,
        )

        photo_dir = None
        if self.args.photos:
            photo_folder = f"steps-{time.strftime(MaintenanceSteps.STAMP_FORMAT)}"
            photo_dir = Path(self.load_config().run_dir) / MaintenanceSteps.PHOTO_FOLDER / photo_folder
        cameras = None if photo_dir is None else settings_robot.open_cameras()

        report = []
        try:
            robot = self.new_robot(ActionMode.CARTESIAN, cameras)
            with robot:
                start = MaintenanceSteps.held_targets(robot)
                try:
                    robot.home(report_progress=MaintenanceSteps.print_home_stage)
                    MaintenanceSteps.photograph(robot, photo_dir, "00-home")

                    for delta in deltas:
                        observation = robot.observe()
                        quantities_before = robot.kinematics.quantities(observation["joint_positions"]).arm
                        arm_changes = {
                            quantity: float(delta.get(quantity, 0)) for quantity in SO101Kinematics.ARM_QUANTITIES
                        }
                        result = robot.execute(
                            MaintenanceSteps.arm_delta(**arm_changes), reference_observation=observation
                        )
                        quantities_after = result.measured_state.arm if result.measured_state else None
                        report.append(
                            {
                                "commanded": {quantity: value for quantity, value in arm_changes.items() if value},
                                "measured": {
                                    quantity: round(
                                        getattr(quantities_after, quantity) - getattr(quantities_before, quantity),
                                        self.MEASURED_DECIMALS,
                                    )
                                    for quantity in SO101Kinematics.ARM_QUANTITIES
                                }
                                if quantities_after
                                else None,
                                "status": result.status.value,
                                "reached": result.target_reached,
                                "max_joint_error": round(
                                    max(map(abs, result.tracking_error[:GRIPPER_INDEX])), self.ERROR_DECIMALS
                                )
                                if result.tracking_error
                                else None,
                                "error": result.error,
                            }
                        )
                        print(f"step {len(report)}: {report[-1]}", file=sys.stderr, flush=True)

                        label = "_".join(f"{quantity}{value:+g}" for quantity, value in delta.items())
                        MaintenanceSteps.photograph(robot, photo_dir, f"{len(report):02d}-{label}")
                        if result.status == MotionStatus.FAILED:
                            break
                finally:
                    MaintenanceSteps.park(robot, start)
        finally:
            if cameras is not None:
                cameras.close()

        return report
