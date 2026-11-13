import math
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

from pear.deploy.station.control.joint.config import ConfigSection, HomeConfig, Limits, RobotConfig
from pear.deploy.station.control.robots.so101.layout import ARM, LAYOUT, VECTOR_LENGTH

ARM_JOINT_LENGTH = LAYOUT.joint_count  # of a vector of settings with one value for each arm joint
WRIST_ROLE = "wrist"  # the one camera's role, by which the format reads its frames


@dataclass
class SO101Limits(Limits):
    """The arm's limits, with the acceleration and jerk caps a controlled STOP brakes with."""

    # a controlled STOP brakes with these; None: with the move caps
    stop_acceleration: list[float] | None = None
    stop_jerk: list[float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        stop_and_move_caps = ((self.stop_acceleration, self.acceleration), (self.stop_jerk, self.jerk))
        if any(
            stop_caps is not None and any(stop_cap < move_cap for stop_cap, move_cap in zip(stop_caps, move_caps))
            for stop_caps, move_caps in stop_and_move_caps
        ):
            raise ValueError("A STOP must not brake more gently than a move")

    def braking_caps(self) -> tuple[list[float], list[float]]:
        """Returns the acceleration and jerk caps a controlled STOP brakes with: the STOP caps, a move's where they are
        unset."""
        return self.stop_acceleration or self.acceleration, self.stop_jerk or self.jerk


@dataclass
class HardwareConfig(ConfigSection):
    """The arm's Feetech bus: its device, its lerobot calibration and the servos' settings."""

    port: str = MISSING  # the bus's serial device
    calibration_id: str = MISSING  # the name of lerobot's calibration file in calibration_dir
    # where the calibration file is; a relative directory is relative to the station's own file
    calibration_dir: Path = MISSING
    # a software zero correction on top of lerobot's calibration, rad: the joint's angle is lerobot's reading plus it;
    # the servos' registers stay untouched
    joint_offsets: list[float] = MISSING
    # the control periods between two polls of the servos' Status, and of their Torque_Enable half a cycle apart
    status_poll_ticks: int = MISSING
    position_p_coefficient: int = MISSING  # the servos' position P gain

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_length("joint_offsets", ARM_JOINT_LENGTH)
        self.check_range("joint_offsets", at_least=-0.2, at_most=0.2)
        self.check_range("status_poll_ticks", at_least=2, at_most=100)
        self.check_range("position_p_coefficient", at_least=8, at_most=64)


@dataclass
class CameraConfig(ConfigSection):
    """The wrist camera's device and the frames it is asked for."""

    path: str = MISSING  # the camera's video device
    role: str = MISSING  # the role its frames are keyed by: wrist, the format's one camera
    width: int = MISSING  # pixels
    height: int = MISSING  # pixels
    fps: int = MISSING  # frames a second
    fourcc: str | None = MISSING  # the pixel format the device is asked for; None: the device's own

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.role != WRIST_ROLE:
            raise ValueError(f"CameraConfig.role must be {WRIST_ROLE!r}")
        self.check_range("width", above=0)
        self.check_range("height", above=0)
        self.check_range("fps", above=0)


@dataclass
class KinematicsConfig(ConfigSection):
    """What the inverse kinematics solve on and how far they go: the arm's description, the bounds of an increment
    and of the residuals, and the gripper's fingertip widths."""

    # the arm's description, by default the package's own, whose joints are zero mid-range as lerobot calibrates them; a
    # relative path is relative to the station's own file
    urdf: Path = Path(__file__).with_name("description") / "so101_new_calib.urdf"
    max_rotation_rad: float = MISSING  # the largest pitch or roll increment an action may ask
    position_residual_m: float = MISSING  # the largest position error of a solution
    rotation_residual_rad: float = MISSING  # the largest orientation error of a solution
    ik_iterations: int = MISSING  # the most steps one solve takes
    gripper_width_cm: list[float] = MISSING  # the fingertips' distance when the gripper is closed and when it is open

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("max_rotation_rad", above=0, at_most=math.pi)
        self.check_range("position_residual_m", above=0, at_most=0.01)
        self.check_range("rotation_residual_rad", above=0, at_most=0.1)
        self.check_range("ik_iterations", at_least=10, at_most=2000)
        self.check_length("gripper_width_cm", 2)
        closed_width, open_width = self.gripper_width_cm
        if not 0 <= closed_width < open_width:
            raise ValueError("gripper_width_cm must be [closed, open] with closed < open")


@dataclass
class SO101HomeConfig(HomeConfig):
    """The arm's Home: its pose and how close each joint must come to it."""

    position: list[float] = MISSING  # the joint vector at Home, rad and the gripper's opening
    joint_tolerance: list[float] = MISSING  # how close each arm joint must come to Home, rad

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_length("position", VECTOR_LENGTH)
        self.check_length("joint_tolerance", ARM_JOINT_LENGTH)
        self.check_range("joint_tolerance", above=0, at_most=0.1)


@dataclass
class FeedforwardConfig(ConfigSection):
    """The goal offsets of the five arm joints that the feedforward adds to a target, rad; zeros disable a term."""

    gravity_compliance: list[float] = MISSING  # the offset for each N*m of gravity torque the joint carries
    friction_offset_rad: list[float] = MISSING  # the offset against friction while the joint moves
    friction_load_ratio: list[float] = MISSING  # what each N*m of the joint's gravity load adds to it

    def __post_init__(self) -> None:
        super().__post_init__()

        for name in ("gravity_compliance", "friction_offset_rad", "friction_load_ratio"):
            self.check_length(name, ARM_JOINT_LENGTH)
        self.check_range("gravity_compliance", at_least=0, at_most=0.3)
        self.check_range("friction_offset_rad", at_least=0, at_most=0.05)
        self.check_range("friction_load_ratio", at_least=0, at_most=0.1)


@dataclass
class SO101Config(RobotConfig):
    """The SO-101's settings: its arm's bus, its wrist camera, its kinematics, its limits, its Home and the
    feedforward of its servos."""

    LAYOUT = LAYOUT  # the arm's joint vector
    RELATIVE_PATH_KEYS = ("hardware.calibration_dir", "geometry.urdf")  # the two paths, which may be relative

    hardware: HardwareConfig | None = None  # None: the arm must not move (a motion blocker)
    camera: CameraConfig | None = None  # None: the station has no camera
    geometry: KinematicsConfig | None = None  # None: no Cartesian motion
    limits: SO101Limits = MISSING
    home: SO101HomeConfig | None = None  # None: the robot has no Home
    feedforward: FeedforwardConfig | None = None  # None: the servos get no goal offsets

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("feedback_position_tolerance", at_most=0.2)
        limits = self.limits
        if self.home is not None and any(
            not lower_bound <= coordinate <= upper_bound
            for coordinate, lower_bound, upper_bound in zip(
                self.home.position, limits.lower_bounds, limits.upper_bounds
            )
        ):
            raise ValueError("Home must lie inside the position limits")
        if self.feedforward is not None and self.geometry is None:
            raise ValueError("feedforward needs geometry.urdf for the gravity model")

    def arm_limits(self) -> dict[str, SO101Limits]:
        """Returns the limits of the one arm."""
        return {ARM: self.limits}

    def motion_blockers(self) -> list[str]:
        """Returns the reasons the real arm must not move under these settings: a template's placeholder first, then
        no bus, or no calibration file."""
        hardware = self.hardware
        blockers = []
        if placeholder_key := self.first_placeholder():
            blockers.append(f"{placeholder_key} is a placeholder; see the so101 setup in README.md")
        if hardware is None:
            blockers.append("missing hardware section")
        elif not (hardware.calibration_dir / f"{hardware.calibration_id}.json").exists():
            blockers.append(f"missing calibration file for id {hardware.calibration_id!r}")

        return blockers
