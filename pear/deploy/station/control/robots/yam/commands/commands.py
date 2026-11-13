from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import RobotCommand, RobotCommands, Smoke
from pear.deploy.station.control.robots.yam.commands.accuracy_check import YAMAccuracyCheck

# its module imports nothing of the station's packages: the command line declares its choices without them
from pear.deploy.station.control.robots.yam.commands.gripper_check import GripperCheck, GripperCheckMode

if TYPE_CHECKING:  # annotations only: the station's packages are imported where a command runs
    from pear.deploy.station.control.joint.layout import JointLayout
    from pear.deploy.station.control.robot import BaseRobot
    from pear.deploy.station.control.schemas import ControlSnapshot, DetailedStatus


class ArmCommand(RobotCommand):
    """The base of a command about one of the robot's arms, which the command line names."""

    # the robot's arms, as its layout names them: written here, as the command line declares them without the
    # station's packages
    ARM_CHOICES = ("left", "right")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the arm, by the word the operator types."""
        parser.add_argument("side", choices=cls.ARM_CHOICES)


class Check(RobotCommand):
    """Lists the CAN interfaces and the USB cameras the operating system sees, and what keeps the arms from moving;
    sends no CAN frame."""

    NAME = "check"
    HELP = "OS device inventory; no CAN frames or SDK robot creation"

    def run(self) -> dict[str, Any]:
        """Returns the devices, and the reasons the arms must not move yet: the first value the station's file has
        not given, then what the arms lack."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages: where a command runs
        from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks

        robot_config = self.new_robot(ActionMode.JOINT).robot_config

        return StartupChecks.inventory() | {"motion_blockers": robot_config.motion_blockers()}


class Preflight(RobotCommand):
    """Runs the checks of enabling the arms with register reads alone, powering no motor."""

    NAME = "preflight"
    HELP = "Validate startup using register reads"

    def run(self) -> dict[str, dict[str, str]]:
        """Returns what was checked of each arm."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages
        from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks

        return StartupChecks.preflight(self.new_robot(ActionMode.JOINT).robot_config)


class CalibrateGripper(ArmCommand):
    """MOVES one empty gripper to measure its endpoints, the arms disabled, and keeps a record of the run in the run
    directory."""

    NAME = "calibrate-gripper"
    HELP = "MOVE one empty gripper to measure endpoints; arms stay disabled"
    MOVES_ROBOT = True

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the arm and the confirmation that the gripper may move."""
        super().add_arguments(parser)
        parser.add_argument("--yes", action="store_true", help="Confirm that the real gripper moves")

    def run(self) -> dict[str, Any]:
        """Returns the gripper's endpoints and where the record of the run is; a placeholder left in the settings is
        refused before the gripper's CAN channel opens."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages
        from pear.deploy.station.control.robots.yam.commands.gripper_calibration import GripperCalibration

        run_dir = Path(self.load_config().run_dir)
        robot_config = self.new_robot(ActionMode.JOINT).robot_config
        if placeholder_key := robot_config.first_placeholder():
            raise ValueError(
                f"{placeholder_key} is a placeholder in {self.args.config}; see the yam setup in README.md"
            )

        return GripperCalibration.run(robot_config, self.args.side, run_dir)


class CheckGripper(ArmCommand):
    """MOVES one gripper through the robot's gripper operation, to its endpoint or for a short time, with both arms
    enabled, and reports the arms' hold before and after."""

    NAME = "gripper-check"
    HELP = "MOVES one gripper through the station's gripper operation; supervised on real arms"
    MOVES_ROBOT = True
    OPEN_ENDPOINT = "open"  # the endpoint that opens the gripper
    ENDPOINTS = (OPEN_ENDPOINT, "close")  # where a check moves the gripper

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the arm, the endpoint, how long the gripper moves and the confirmation that it may move."""
        super().add_arguments(parser)
        parser.add_argument("endpoint", choices=cls.ENDPOINTS)
        parser.add_argument(
            "--mode", choices=[mode.value for mode in GripperCheckMode], default=GripperCheckMode.SHORT.value
        )
        parser.add_argument("--yes", action="store_true", help="Confirm that the real gripper is empty and may move")

    def run(self) -> dict[str, Any]:
        """Returns the check's report."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages

        args = self.args
        robot = self.new_robot(ActionMode.JOINT)

        with robot:
            return GripperCheck(
                robot, args.side, args.endpoint == self.OPEN_ENDPOINT, GripperCheckMode(args.mode)
            ).run()


class Cameras(RobotCommand):
    """Lists the serial number, the port, the model and the firmware of every connected RealSense camera."""

    NAME = "cameras"
    HELP = "RealSense identity and serials"
    READS_CONFIG = False

    def run(self) -> list[dict[str, Any]]:
        """Returns each camera's identity."""
        from pear.deploy.station.control.robots.yam.hardware.cameras import RealSenseCameras  # with its SDK

        return RealSenseCameras.discover()


class Snapshot(RobotCommand):
    """Captures every configured camera at once into the run directory, with an index page of their images."""

    NAME = "snapshot"
    HELP = "Capture all configured cameras simultaneously"
    DEFAULT_FRAME_COUNT = 120  # of the frames each camera streams for a snapshot
    MAX_FRAME_COUNT = 18000  # the most frames a camera may stream for a snapshot
    OUTPUT_FOLDER = "cameras"  # of the run directory, where the snapshots go

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares how many frames each camera streams."""
        parser.add_argument("--frames", type=int, default=cls.DEFAULT_FRAME_COUNT)

    def run(self) -> list[dict[str, Any]]:
        """Returns each camera's record of the snapshot."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages
        from pear.deploy.station.control.robots.yam.hardware.cameras import RealSenseCameras

        run_dir = Path(self.load_config().run_dir)
        cameras = self.new_robot(ActionMode.JOINT).robot_config.cameras
        frame_count = self.args.frames
        if not 1 <= frame_count <= self.MAX_FRAME_COUNT:
            raise ValueError(f"Use 1..{self.MAX_FRAME_COUNT} frames")

        return RealSenseCameras.snapshot(
            [camera.serial for camera in cameras],
            run_dir / self.OUTPUT_FOLDER,
            frame_count,
            {camera.serial: camera.role for camera in cameras},
        )


class YAMSmoke(Smoke):
    """MOVES the robot: enables both arms, moves the first joint of one arm a small distance and back, and releases
    the arms."""

    HELP = "Small joint-1 move and return; supervised on real hardware"
    MAX_DELTA_RAD = 0.05
    DEFAULT_DELTA_RAD = 0.03
    NOTICE = (
        "Backend real: enable both followers, jog {side} joint 1 by {delta} rad, return, then disable. "
        "Arms must be supported at the starting pose."
    )
    REPORTS_WHILE_ENABLED = True

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the arm whose first joint moves, the distance and the confirmation that the arms may move."""
        parser.add_argument("--side", choices=ArmCommand.ARM_CHOICES, default=ArmCommand.ARM_CHOICES[0])
        super().add_arguments(parser)
        parser.add_argument("--yes", action="store_true", help="Confirm motion on the real arms")

    def moved_joint(self, layout: JointLayout) -> tuple[str, int]:
        """Returns the arm the command line names and its first joint."""
        return self.args.side, 0

    def initial_snapshot(self, robot: BaseRobot) -> ControlSnapshot | DetailedStatus:
        """Returns the robot's status with its snapshot at once: the move starts at the targets the arms hold."""
        return robot.status(with_snapshot=True)


class SweepFriction(RobotCommand):
    """MOVES the robot: homes both arms and opens their grippers, sweeps each joint of one arm both ways at constant
    speeds, at a lifted and a low pose, with the arm's friction feedforward off, to measure its friction and the
    gravity model's error, homes the arms again, keeps a record in the run directory and prints the arm's lines of the
    station's file; without --yes it prints its plan and moves nothing."""

    NAME = "friction-sweep"
    HELP = "MOVES an arm: identify gravity and friction (see its --help)"
    MOVES_ROBOT = True
    HAS_DRY_RUN = True
    OUTPUT_FOLDER = "friction"  # of the run directory, where the records go
    # the sweep's poses, as FrictionSweep.POSES names them: written here, as the command line declares them without
    # the station's packages
    POSE_CHOICES = ("lifted", "low")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the arm, its poses, and its joints and the speeds, each comma separated."""
        parser.add_argument("--side", choices=ArmCommand.ARM_CHOICES, required=True)
        parser.add_argument("--poses", nargs="+", choices=cls.POSE_CHOICES, default=list(cls.POSE_CHOICES))
        parser.add_argument("--joints", help="1-based, comma separated")
        parser.add_argument("--speeds")
        parser.add_argument("--yes", action="store_true", help="Required to move real hardware")
        # the sweep refuses joints or speeds as the command line refuses a word: with the usage, status 2
        parser.set_defaults(refuse_arguments=parser.error)

    @staticmethod
    def feedforward_off_overrides(side: str) -> list[str]:
        """Returns the overrides that turn the friction feedforward of the arm ``side`` off, read over the station's
        file: a sweep with it on would measure what the feedforward leaves."""
        return [
            f"robot.arms.{side}.friction_feedforward_nm=[0, 0, 0, 0, 0, 0]",
            f"robot.arms.{side}.friction_load_ratio=[0, 0, 0, 0, 0, 0]",
        ]

    def run(self) -> None:
        """Prints the sweep's plan and, with --yes, runs it, printing its results as it goes."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages
        from pear.deploy.station.control.robots.yam.commands.friction_sweep import FrictionSweep

        args = self.args
        joints, speeds = FrictionSweep.checked_arguments(args.joints, args.speeds, args.refuse_arguments)

        run_dir = Path(self.load_config().run_dir)
        robot = self.new_robot(ActionMode.JOINT, extra_overrides=self.feedforward_off_overrides(args.side))
        sweep = FrictionSweep(robot, args.side, run_dir / self.OUTPUT_FOLDER, joints, speeds, list(args.poses))
        sweep.run(is_confirmed=args.yes)


class YAMCommands(RobotCommands):
    """Maintenance and commissioning commands of the YAM robot."""

    ROBOT_CLASS_PATH = "pear.deploy.station.control.robots.yam.robot.YAMRobot"
    HELP = "Maintenance commands of the YAM station"
    CONFIRMATION_REFUSAL = "{command} moves the real hardware; add --yes to confirm"
    COMMAND_CLASSES = (
        Check,
        Preflight,
        CalibrateGripper,
        CheckGripper,
        Cameras,
        Snapshot,
        YAMSmoke,
        SweepFriction,
        YAMAccuracyCheck,
    )
