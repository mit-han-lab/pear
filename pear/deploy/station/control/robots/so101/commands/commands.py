from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import RobotCommand, RobotCommands, Smoke
from pear.deploy.station.control.robots.so101.commands.accuracy_check import SO101AccuracyCheck
from pear.deploy.station.control.robots.so101.commands.friction_sweep import FrictionSweep
from pear.deploy.station.control.robots.so101.commands.home_check import HomeCheck
from pear.deploy.station.control.robots.so101.commands.safety_check import SafetyCheck
from pear.deploy.station.control.robots.so101.commands.steps_check import StepsCheck

if TYPE_CHECKING:  # annotations only: the station's packages are imported where a command runs
    import numpy as np

    from pear.deploy.station.control.joint.layout import JointLayout
    from pear.deploy.station.control.robots.so101.robot import SO101Robot
    from pear.deploy.station.control.schemas import ControlSnapshot, DetailedStatus

# the station's packages and the robot's own modules are imported inside the methods: the command line lists the
# commands without them


class MaintenanceSteps:
    """The steps the SO-101's maintenance commands share: the hold the arm starts from and the return to it, Home's
    progress, a snapshot's summary, a frame's photo and a pose delta."""

    HOLD_TIMEOUT_S = 10.0  # for the settled hold the arm starts from and each of a check's steps ends in
    PARK_TIMEOUT_S = 60.0  # for the arm to arrive back at its start
    SUMMARY_DECIMALS = 4  # of the vectors of a snapshot's summary
    JPEG_QUALITY = 90  # of a photo
    PHOTO_FOLDER = "cameras"  # of the run directory, where the photos go
    STAMP_FORMAT = "%Y%m%d-%H%M%S"  # of the local time that names a photo, or one run's folder of photos

    @classmethod
    def held_targets(cls, robot: SO101Robot) -> tuple[float, ...]:
        """Returns the arm's targets once it holds them, settled."""
        from pear.deploy.station.control.robots.so101.layout import ARM

        return robot.control_loop.wait_for_hold(cls.HOLD_TIMEOUT_S)["targets"][ARM]

    @classmethod
    def park(cls, robot: SO101Robot, start: Sequence[float]) -> ControlSnapshot:
        """Moves the arm back to the targets ``start``, clipped into its bounds, at Home's speed, after a recovery
        from a fault, and returns the snapshot of the hold it ends in."""
        import numpy as np

        from pear.deploy.station.control.robots.so101.layout import ARM
        from pear.deploy.station.control.schemas import ControlState

        control_loop = robot.control_loop
        print("returning to the start", file=sys.stderr, flush=True)
        if control_loop.status == ControlState.FAULT:
            robot.recover()
        control_loop.wait_for_hold(control_loop.HOLD_WAIT_S)

        limits, home = robot.robot_config.limits, robot.robot_config.home
        goal = np.clip(start, limits.lower_bounds, limits.upper_bounds)
        control_loop.move({ARM: tuple(goal)}, speed_scale=home.speed_scale, caps=home.limits)

        return control_loop.wait_for_hold(cls.PARK_TIMEOUT_S)

    @staticmethod
    def print_home_stage(progress: dict[str, Any]) -> None:
        """Prints the stage Home has entered."""
        print(f"home: {progress['stage']}", file=sys.stderr, flush=True)

    @staticmethod
    def add_confirmation(parser: argparse.ArgumentParser) -> None:
        """Declares --yes, the confirmation every command that moves the arm takes."""
        parser.add_argument("--yes", action="store_true", help="Confirm motion on the real arm")

    @classmethod
    def summary(cls, snapshot: ControlSnapshot | DetailedStatus) -> dict[str, Any]:
        """Returns the arm's target, measured position, signed error and arrival in ``snapshot``, rounded."""
        from pear.deploy.station.control.robots.so101.layout import ARM

        return {
            "target": [round(value, cls.SUMMARY_DECIMALS) for value in snapshot["targets"][ARM]],
            "measured": [round(value, cls.SUMMARY_DECIMALS) for value in snapshot["arms"][ARM]["position"]],
            "signed_error": [round(value, cls.SUMMARY_DECIMALS) for value in snapshot["arrival"]["signed_error"]],
            "reached": snapshot["arrival"]["reached"],
        }

    @classmethod
    def photograph(cls, robot: SO101Robot, photo_dir: Path | None, name: str | None = None) -> dict[str, Any]:
        """Saves each camera's frame of a new observation in ``photo_dir`` and returns their descriptions by camera
        role; without a directory, takes none."""
        if photo_dir is None:
            return {}

        return {role: cls.save_frame(rgb, photo_dir, role, name) for role, rgb in robot.observe()["images"].items()}

    @classmethod
    def save_frame(cls, rgb: np.ndarray, output_dir: Path, role: str, name: str | None = None) -> dict[str, Any]:
        """Writes one camera frame as a JPEG, named ``name`` (by default the time) and the camera's role, and returns
        its path and shape."""
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{name or time.strftime(cls.STAMP_FORMAT)}-{role}.jpg"
        Image.fromarray(rgb).save(path, "JPEG", quality=cls.JPEG_QUALITY)

        return {"path": str(path), "shape": list(rgb.shape)}

    @staticmethod
    def arm_delta(**arm_changes: float) -> dict[str, Any]:
        """Returns the pose delta action that changes the arm's quantities by ``arm_changes`` and leaves the
        gripper."""
        from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics

        zero_delta = {quantity: 0.0 for quantity in SO101Kinematics.ARM_QUANTITIES}
        return {"type": "single_arm_delta", "arm": zero_delta | arm_changes, "gripper": {"width": 0.0}}


class Check(RobotCommand):
    """Lists the serial ports and the video devices and what keeps the arm from moving; opens no device."""

    NAME = "check"
    HELP = "Serial and camera inventory; opens no device"

    def run(self) -> dict[str, Any]:
        """Returns the devices by their stable paths and the reasons the arm must not move, a value the station's file
        leaves ??? first."""
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.hardware.startup import StartupChecks

        robot_config = self.new_robot(ActionMode.JOINT).robot_config

        return StartupChecks.inventory() | {"motion_blockers": robot_config.motion_blockers()}


class Preflight(RobotCommand):
    """Pings the servos and reads their registers, with torque left as found."""

    NAME = "preflight"
    HELP = "Ping servos and read registers; torque untouched"

    def run(self) -> dict[str, Any]:
        """Returns the servos' registers, their faults and whether the checks passed."""
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.hardware.startup import StartupChecks

        return StartupChecks.preflight(self.new_robot(ActionMode.JOINT).robot_config)


class Pose(RobotCommand):
    """Reads the measured joints and the model's pose of them, with torque left as found."""

    NAME = "pose"
    HELP = "Measured joints and FK; torque untouched"
    WATCH_INTERVAL_S = 0.5  # between two printed readings of --watch
    RADIAN_DECIMALS = 4  # of a joint in rad
    DEGREE_DECIMALS = 1  # of a joint in degrees and of the gripper's percent

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares whether the command keeps reading."""
        parser.add_argument("--watch", action="store_true", help="Print continuously until Ctrl-C")

    def run(self) -> dict[str, Any]:
        """Returns one reading of the joints in rad, in degrees (the gripper in percent) and as the model's pose; with
        --watch, prints one each interval until it is interrupted."""
        import numpy as np

        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.hardware.arms import SO101Hardware
        from pear.deploy.station.control.robots.so101.hardware.startup import StartupChecks
        from pear.deploy.station.control.robots.so101.layout import MOTORS

        robot = self.new_robot(ActionMode.JOINT)
        robot_config = robot.robot_config
        kinematics = robot.KINEMATICS_CLASS.shared_instance(robot_config)
        joint_offsets = np.array(robot_config.hardware.joint_offsets)

        with StartupChecks.connected_bus(robot_config.hardware) as bus:
            while True:
                values = bus.sync_read("Present_Position", num_retry=StartupChecks.READ_RETRIES)
                position = SO101Hardware.bus_position(values, joint_offsets)
                reading = {
                    "joints_rad": [round(float(value), self.RADIAN_DECIMALS) for value in position],
                    # the gripper's opening in percent
                    "joints_deg": {
                        motor: round(math.degrees(value) if motor != "gripper" else 100 * value, self.DEGREE_DECIMALS)
                        for motor, value in zip(MOTORS, position)
                    },
                    "single_arm_pose": kinematics.quantities(position).model_dump(),
                }
                if not self.args.watch:
                    return reading

                print(json.dumps(reading), flush=True)
                time.sleep(self.WATCH_INTERVAL_S)


class Snapshot(RobotCommand):
    """Saves one frame of the wrist camera in the run directory."""

    NAME = "snapshot"
    HELP = "Save one wrist camera frame"

    def run(self) -> dict[str, Any]:
        """Returns the path and the shape of the saved frame."""
        from pear.deploy.station.control.robot import ActionMode

        output_dir = Path(self.load_config().run_dir) / MaintenanceSteps.PHOTO_FOLDER
        robot = self.new_robot(ActionMode.JOINT)

        cameras = robot.open_cameras()
        try:
            frame = cameras.read()[robot.robot_config.camera.role]
        finally:
            cameras.close()

        return MaintenanceSteps.save_frame(frame["rgb"], output_dir, robot.robot_config.camera.role)


class Calibrate(RobotCommand):
    """Calibrates the arm with lerobot's own procedure, which waits for ENTER, into the station's calibration_dir;
    torque stays off."""

    NAME = "calibrate"
    HELP = "Calibrate with lerobot's procedure; torque stays off, waits for ENTER"

    def run(self) -> dict[str, str]:
        """Returns the path of the calibration file; a placeholder left in the settings is refused before the serial
        port opens."""
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        from pear.deploy.station.control.robot import ActionMode

        robot_config = self.new_robot(ActionMode.JOINT).robot_config
        if placeholder_key := robot_config.first_placeholder():
            raise ValueError(
                f"{placeholder_key} is a placeholder in {self.args.config}; see the so101 setup in README.md"
            )
        hardware_config = robot_config.hardware
        follower = SO101Follower(
            SO101FollowerConfig(
                port=hardware_config.port,
                id=hardware_config.calibration_id,
                calibration_dir=hardware_config.calibration_dir,
            )
        )

        follower.bus.connect()  # the handshake only: torque is left as found, off
        try:
            follower.calibrate()
        finally:
            follower.disconnect()  # writes torque-disable, which changes nothing: torque was never enabled

        return {"calibration": str(follower.calibration_fpath)}


class SO101Smoke(Smoke):
    """MOVES the arm: enables it, moves one joint a small distance and back, and releases it."""

    HELP = "MOVES the arm: one joint out and back"
    MAX_DELTA_RAD = 0.1  # the farthest the SO-101's smoke moves a joint
    DEFAULT_DELTA_RAD = 0.05  # how far it moves one by default
    NOTICE = (
        "Backend real: enable the arm, move {joint} by {delta:+.3f} rad, return, then disable torque. "
        "Keep the arm supported if it is not resting on its stops."
    )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the confirmation, the joint that moves and the distance."""
        from pear.deploy.station.control.robots.so101.layout import JOINTS

        MaintenanceSteps.add_confirmation(parser)
        parser.add_argument("--joint", choices=list(JOINTS), default=JOINTS[0])
        super().add_arguments(parser)

    def run(self) -> dict[str, Any] | None:
        """Refuses an uncommissioned arm, as every command that moves it does first, then moves the joint and back."""
        from pear.deploy.station.control.robot import ActionMode

        self.new_robot(ActionMode.JOINT).robot_config.require_commissioned()

        return super().run()

    def moved_joint(self, layout: JointLayout) -> tuple[str, int]:
        """Returns the one arm and the joint the command line names."""
        return layout.arms[0], layout.coordinates.index(self.args.joint)

    def summary(self, snapshot: ControlSnapshot | DetailedStatus) -> dict[str, Any]:
        """Returns the arm's target, measured position, signed error and arrival in ``snapshot``, rounded."""
        return MaintenanceSteps.summary(snapshot)


class SO101Commands(RobotCommands):
    """Maintenance commands of the SO-101."""

    ROBOT_CLASS_PATH = "pear.deploy.station.control.robots.so101.robot.SO101Robot"
    HELP = "Maintenance commands of the SO-101 station"
    CONFIRMATION_REFUSAL = "{command} moves the arm; add --yes to confirm"
    COMMAND_CLASSES = (
        Check,
        Preflight,
        Pose,
        Snapshot,
        SO101Smoke,
        HomeCheck,
        StepsCheck,
        SafetyCheck,
        Calibrate,
        FrictionSweep,
        SO101AccuracyCheck,
    )
