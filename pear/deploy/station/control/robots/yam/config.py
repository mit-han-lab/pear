import math
from dataclasses import dataclass

from omegaconf import MISSING

from pear.deploy.station.control.joint.config import ConfigSection, HomeConfig, Limits, RobotConfig
from pear.deploy.station.control.robots.yam.layout import LAYOUT


@dataclass
class ArmConfig(ConfigSection):
    """One follower arm: its CAN bus, its SDK models and gains, its gripper's calibration, its feedforward and its
    limits."""

    ARM_TYPES = ("yam", "yam_pro", "yam_ultra", "yam_ultra_2", "big_yam")  # the SDK's arm models
    GRIPPER_TYPES = ("crank_4310", "linear_3507", "linear_4310", "flexible_4310")  # the SDK's gripper models

    channel: str = MISSING  # the arm's SocketCAN interface
    adapter_serial: str = MISSING  # the serial number of the arm's USB-CAN adapter
    limits: Limits = MISSING
    arm_type: str | None = None  # one of ARM_TYPES; None: not commissioned
    joint_position_gains: list[float] | None = None  # J1-J6 MIT Kp in the SDK's motor order; None: the SDK's own
    # J1-J6 software zero correction, true joint angle = SDK reading + offset: the motors' zeros stay untouched, and
    # the commands are shifted back by the same amount
    joint_offsets: list[float] = MISSING
    gripper_type: str | None = None  # one of GRIPPER_TYPES; None: not commissioned
    gripper_limits: list[float] | None = None  # the closed and the open endpoint, motor rad; None: not calibrated
    gripper_stroke_range: list[float] | None = None  # the measured hardware's least and greatest stroke, rad
    gripper_force_n: float = MISSING  # the SDK's nominal force setting, not a calibrated fingertip force
    # whether a moving command carries the planned joint velocity as the MIT velocity target; false is the SDK's
    # position-only command, whose Kd brakes every move, so that a joint lags by about Kd * v / Kp
    uses_velocity_feedforward: bool = MISSING
    # J1-J6 factors of the SDK's gravity model, which replace its own instead of scaling them; None: the SDK's own
    gravity_compensation_factors: list[float] | None = None
    friction_feedforward_nm: list[float] = MISSING  # J1-J6 Coulomb friction fed forward along the plan; zero: none
    friction_load_ratio: list[float] = MISSING  # J1-J6 friction per Nm of the gravity load a joint carries

    def __post_init__(self) -> None:
        super().__post_init__()

        for name in ("joint_position_gains", "joint_offsets", "gravity_compensation_factors"):
            self.check_length(name, LAYOUT.joint_count)
        for name in ("friction_feedforward_nm", "friction_load_ratio"):
            self.check_length(name, LAYOUT.joint_count)
        for name in ("gripper_limits", "gripper_stroke_range"):
            self.check_length(name, 2)  # the closed and the open endpoint; the least and the greatest stroke
        self.check_range("joint_position_gains", at_least=0, at_most=500)
        self.check_range("joint_offsets", at_least=-0.2, at_most=0.2)
        self.check_range("gripper_force_n", above=0)
        self.check_range("gravity_compensation_factors", at_least=0.5, at_most=1.5)
        self.check_range("friction_feedforward_nm", at_least=0, at_most=3)
        self.check_range("friction_load_ratio", at_least=0, at_most=0.5)
        if self.arm_type is not None and self.arm_type not in self.ARM_TYPES:
            raise ValueError(f"{type(self).__name__}.arm_type must be one of {', '.join(self.ARM_TYPES)}")
        if self.gripper_type is not None and self.gripper_type not in self.GRIPPER_TYPES:
            raise ValueError(f"{type(self).__name__}.gripper_type must be one of {', '.join(self.GRIPPER_TYPES)}")

        has_friction_feedforward = any(self.friction_feedforward_nm) or any(self.friction_load_ratio)
        if has_friction_feedforward and not self.uses_velocity_feedforward:
            raise ValueError("Friction feedforward requires uses_velocity_feedforward")
        if self.gripper_stroke_range is not None:
            min_stroke, max_stroke = self.gripper_stroke_range
            if not 0 < min_stroke < max_stroke:
                raise ValueError("Gripper stroke range must be positive and ordered")
            if self.gripper_limits is not None:
                stroke = abs(self.gripper_limits[1] - self.gripper_limits[0])
                if not min_stroke <= stroke <= max_stroke:
                    raise ValueError("Gripper endpoints disagree with the measured stroke range")


@dataclass
class CameraCrop(ConfigSection):
    """A window of a camera's RGB frame, in pixels, which the preview and the observations both show."""

    x_offset: int = MISSING  # of the window's left edge from the frame's
    y_offset: int = MISSING  # of the window's top edge from the frame's
    width: int = MISSING
    height: int = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("x_offset", at_least=0)
        self.check_range("y_offset", at_least=0)
        self.check_range("width", above=0)
        self.check_range("height", above=0)
        frame_width, frame_height = CameraConfig.FRAME_WIDTH, CameraConfig.FRAME_HEIGHT
        if self.x_offset + self.width > frame_width or self.y_offset + self.height > frame_height:
            raise ValueError(f"Camera crop must fit within the {frame_width}x{frame_height} RGB frame")


@dataclass
class CameraConfig(ConfigSection):
    """One RealSense camera of the robot: its serial number, its role among the robot's cameras and its crop."""

    FRAME_WIDTH = 640  # pixels of every camera's RGB frame
    FRAME_HEIGHT = 480
    FRAME_RATE = 30  # frames a second of every camera's RGB stream
    ROLES = ("left", "right", "top")  # the roles of the robot's cameras

    serial: str = MISSING  # the camera's serial number in the RealSense SDK, not its USB serial
    role: str = MISSING  # one of ROLES
    crop: CameraCrop | None = None  # None: the whole frame

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.role not in self.ROLES:
            raise ValueError(f"{type(self).__name__}.role must be one of {', '.join(self.ROLES)}")


@dataclass
class CartesianGeometry(ConfigSection):
    """Where the two arms' bases are in the robot's frame, and how far a solved pose may be from the one asked for."""

    base_spacing_m: float = MISSING  # between the two arms' bases
    max_rotation_rad: float = MISSING  # the largest rotation an action may ask of an arm
    position_residual_m: float = MISSING  # how far a solved tool position may be from the one asked for
    rotation_residual_rad: float = MISSING  # how far a solved tool rotation may be from the one asked for

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("base_spacing_m", above=0, below=2)
        self.check_range("max_rotation_rad", above=0, at_most=math.pi)
        self.check_range("position_residual_m", above=0, at_most=0.003)
        self.check_range("rotation_residual_rad", above=0, at_most=0.02)


@dataclass
class YAMHomeConfig(HomeConfig):
    """Home of the YAM arms: each arm's Home, and how close each joint must end to it."""

    MAX_JOINT_TOLERANCE = 0.035  # the loosest a joint's Home tolerance may be, rad

    left_pose: list[float] = MISSING  # the left arm's Home: J1-J6 in rad, then the gripper's opening
    right_pose: list[float] = MISSING  # the right arm's Home: J1-J6 in rad, then the gripper's opening
    joint_tolerance: list[float] = MISSING  # how close each joint J1-J6 must end to its Home, rad

    def __post_init__(self) -> None:
        super().__post_init__()

        for arm in LAYOUT.arms:
            self.check_length(f"{arm}_pose", LAYOUT.coordinate_count)
        self.check_length("joint_tolerance", LAYOUT.joint_count)
        if any(not 0 < tolerance <= self.MAX_JOINT_TOLERANCE for tolerance in self.joint_tolerance):
            raise ValueError(f"Home joint tolerances must be positive and at most {self.MAX_JOINT_TOLERANCE} rad")

    def arm_pose(self, arm: str) -> list[float]:
        """Returns the Home of ``arm``: its joints, then its gripper's opening."""
        return getattr(self, f"{arm}_pose")


@dataclass
class YAMConfig(RobotConfig):
    """The settings of a YAM robot: its two follower arms, their cameras, the Cartesian geometry, Home, and the
    tolerances of their control."""

    LAYOUT = LAYOUT

    arms: dict[str, ArmConfig] = MISSING  # by the arm's name
    cameras: list[CameraConfig] = MISSING
    geometry: CartesianGeometry | None = None  # None: the robot takes no Cartesian action
    home: YAMHomeConfig | None = None
    gripper_contact_confirm_s: float = MISSING  # how long a gripper's contact must hold, steady and loaded
    # the share of the SDK's holding torque that a gripper's measured effort must reach for a contact
    gripper_contact_effort_ratio: float = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("gripper_contact_confirm_s", at_least=0.1, at_most=1)
        self.check_range("gripper_contact_effort_ratio", above=0)
        if set(self.arms) != set(LAYOUT.arms):
            raise ValueError(f"Exactly {' and '.join(LAYOUT.arms)} follower arms are required")
        for name in ("channel", "adapter_serial"):
            if len({getattr(arm_config, name) for arm_config in self.arms.values()}) != len(LAYOUT.arms):
                raise ValueError(f"Follower {name} must be distinct")
        if len({camera.serial for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Camera serials must be unique")
        if len({camera.role for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Camera roles must be unique")

        if self.home is not None:
            for arm, arm_config in self.arms.items():
                limits = arm_config.limits
                home_pose = self.home.arm_pose(arm)
                pose_bounds = zip(home_pose, limits.lower_bounds, limits.upper_bounds)
                if any(not lower_bound <= value <= upper_bound for value, lower_bound, upper_bound in pose_bounds):
                    raise ValueError(f"{arm}: home must lie inside configured position bounds")

    def arm_limits(self) -> dict[str, Limits]:
        """Returns each arm's limits, by the arm's name."""
        return {arm: arm_config.limits for arm, arm_config in self.arms.items()}

    def motion_blockers(self) -> list[str]:
        """Returns the reasons the real arms must not move under these settings: a template's placeholder first, then
        an arm's models or its gripper's calibration missing."""
        blockers = []
        if placeholder_key := self.first_placeholder():
            blockers.append(f"{placeholder_key} is a placeholder; see the yam setup in README.md")

        for arm, arm_config in self.arms.items():
            for name in ("arm_type", "gripper_type", "gripper_limits"):
                if getattr(arm_config, name) is None:
                    blockers.append(f"{arm}: missing {name}")
            if arm_config.gripper_limits and arm_config.gripper_limits[0] == arm_config.gripper_limits[1]:
                blockers.append(f"{arm}: gripper calibration endpoints are equal")

        return blockers

    def feeds_velocity(self, arm: str) -> bool:
        """Whether ``arm`` receives the planned joint velocity as its MIT velocity target while it moves."""
        return self.arms[arm].uses_velocity_feedforward
