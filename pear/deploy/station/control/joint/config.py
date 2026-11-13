import math
import operator
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import MISSING

from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.limits import JointLimits


@dataclass
class ConfigSection:
    """A section of a station's configuration, whose numbers are all finite."""

    def __post_init__(self) -> None:
        for section_field in fields(self):
            self.check_finite(section_field.name, getattr(self, section_field.name))

    def check_range(
        self,
        name: str,
        *,
        above: float | None = None,
        at_least: float | None = None,
        below: float | None = None,
        at_most: float | None = None,
    ) -> None:
        """Raises unless every number of the field ``name`` lies within the bounds given (a field left None holds
        none)."""
        value = getattr(self, name)
        numbers = value if isinstance(value, list) else [] if value is None else [value]
        bounds = {
            "greater than": (operator.gt, above),
            "at least": (operator.ge, at_least),
            "less than": (operator.lt, below),
            "at most": (operator.le, at_most),
        }

        for words, (is_within, bound) in bounds.items():
            if bound is not None and not all(is_within(number, bound) for number in numbers):
                raise ValueError(f"{type(self).__name__}.{name} must be {words} {bound}")

    def check_length(self, name: str, length: int) -> None:
        """Raises unless the field ``name`` holds ``length`` values (a field left None holds none)."""
        value = getattr(self, name)
        if value is not None and len(value) != length:
            raise ValueError(f"{type(self).__name__}.{name} must hold {length} values, not {len(value)}")

    @classmethod
    def check_finite(cls, name: str, value: Any) -> None:
        """Raises unless every number of ``value``, the value of the field ``name``, is finite."""
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{cls.__name__}.{name} must be finite")

        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        for child in children:
            cls.check_finite(name, child)


@dataclass
class ControlConfig(ConfigSection):
    """The settings of the joints' control loop: the control node of a station's configuration."""

    rate_hz: float = MISSING  # control periods a second
    state_timeout_s: float = MISSING  # the age past which a reading of the joints is stale
    # how late a sequence of targets may run; no limit of the gap between two control periods
    deadline_tolerance_s: float = MISSING
    settle_timeout_s: float = MISSING  # the longest wait for the arms to come to rest
    settle_window_s: float = MISSING  # the readings a judgement of rest looks back over
    action_timeout_s: float = MISSING  # the longest gap between two targets of a sequence

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("rate_hz", above=0, at_most=100)
        self.check_range("state_timeout_s", above=0, at_most=1)
        self.check_range("deadline_tolerance_s", above=0, at_most=1)
        self.check_range("settle_timeout_s", above=0)
        self.check_range("settle_window_s", above=0)
        self.check_range("action_timeout_s", above=0, at_most=5)
        if self.settle_window_s >= self.settle_timeout_s:
            raise ValueError("settle_timeout_s must exceed settle_window_s")

    @property
    def control_period_s(self) -> float:
        """The length of one control period."""
        return 1 / self.rate_hz


@dataclass
class MotionLimits(ConfigSection):
    """The velocity, acceleration and jerk caps of one arm's coordinates."""

    velocity: list[float] = MISSING
    acceleration: list[float] = MISSING
    jerk: list[float] = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()

        if any(cap <= 0 for caps in (self.velocity, self.acceleration, self.jerk) for cap in caps):
            raise ValueError("Motion limits must be positive")

    def check_lengths(self, coordinate_count: int) -> None:
        """Raises unless every vector holds one value for each of an arm's ``coordinate_count`` coordinates."""
        for limit_field in fields(self):
            self.check_length(limit_field.name, coordinate_count)


@dataclass
class Limits(MotionLimits):
    """One arm's limits: its motion caps, its position bounds and how far a joint may be from its target."""

    lower_bounds: list[float] = MISSING
    upper_bounds: list[float] = MISSING
    tracking_error: list[float] = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()

        if any(lower_bound >= upper_bound for lower_bound, upper_bound in zip(self.lower_bounds, self.upper_bounds)):
            raise ValueError("Every lower limit must be below its upper limit")
        if any(tracking_bound <= 0 for tracking_bound in self.tracking_error):
            raise ValueError("Tracking bounds must be positive")

    def braking_caps(self) -> tuple[list[float], list[float]]:
        """Returns the acceleration and jerk caps a controlled STOP brakes with (by default a move's)."""
        return self.acceleration, self.jerk


@dataclass
class HomeConfig(ConfigSection):
    """What every robot's Home holds besides its pose: how fast the arms return and when they have arrived."""

    speed_scale: float = MISSING  # the share of the move caps a return to Home uses
    limits: MotionLimits | None = None  # the caps of a return, one arm's for every arm; None: the move caps
    settle_window_s: float = MISSING  # the readings Home's own judgement of arrival looks back over
    arrival_timeout_s: float = MISSING  # the longest wait after the move for the arms to arrive at Home

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("speed_scale", above=0, at_most=1)
        self.check_range("settle_window_s", above=0)
        self.check_range("arrival_timeout_s", above=0)


@dataclass
class RobotConfig(ConfigSection, ABC):
    """What the settings of every robot hold for its joints' control: its tolerances, its Home and its arms'
    limits."""

    LAYOUT: ClassVar[JointLayout]  # the robot's joint vector, which the vectors of its settings follow
    # the dotted keys under robot: whose value is a path, a relative one being relative to the station's own file
    RELATIVE_PATH_KEYS: ClassVar[tuple[str, ...]] = ()
    # what a value the station leaves ??? reads as: a template's placeholder, which the motion blockers name
    PLACEHOLDER: ClassVar[str] = "<{key}>"

    position_tolerance: float = MISSING  # how close an arm joint is to its target at an arrival, rad
    gripper_tolerance: float = MISSING  # how close a gripper's opening is to its target at an arrival
    feedback_position_tolerance: float = MISSING  # how far measured feedback may rest outside the position bounds
    velocity_tolerance: float = MISSING  # the mean speed below which a coordinate is at rest
    home: HomeConfig | None = None  # the Home pose and how the arms return to it; None: the robot has none

    def __post_init__(self) -> None:
        super().__post_init__()

        self.check_range("position_tolerance", above=0)
        self.check_range("gripper_tolerance", above=0, at_most=0.2)
        self.check_range("feedback_position_tolerance", above=0)
        self.check_range("velocity_tolerance", above=0)

        layout = self.LAYOUT
        for limits in self.arm_limits().values():
            limits.check_lengths(layout.coordinate_count)
            if layout.gripper_value(limits.lower_bounds) < 0 or layout.gripper_value(limits.upper_bounds) > 1:
                raise ValueError("Gripper limits must lie within [0, 1]")
        if self.home is not None and self.home.limits is not None:
            self.home.limits.check_lengths(layout.coordinate_count)

    @abstractmethod
    def arm_limits(self) -> dict[str, Limits]:
        """Returns the limits of each arm the settings name, by the arm's name."""

    @abstractmethod
    def motion_blockers(self) -> list[str]:
        """Returns the reasons the real arms must not move under these settings."""

    def require_commissioned(self) -> None:
        """Refuses real motion while the settings still list a reason against it."""
        if blockers := self.motion_blockers():
            raise RuntimeError("Real motion is not commissioned:\n" + "\n".join(blockers))

    def first_placeholder(self) -> str | None:
        """Returns the dotted key of the first value that is still a template's placeholder, in the order of the
        settings; None when there is none."""
        return self.placeholder_key(asdict(self), "")

    def joint_limits(self) -> JointLimits:
        """Returns the arms' limits as the limits of the joint vector, arm after arm."""
        arm_limits = [self.arm_limits()[arm] for arm in self.LAYOUT.arms]
        vectors = {
            name: tuple(value for limits in arm_limits for value in getattr(limits, name))
            for name in ("lower_bounds", "upper_bounds", "velocity", "acceleration", "jerk", "tracking_error")
        }
        braking_caps = [limits.braking_caps() for limits in arm_limits]

        return JointLimits(
            **vectors,
            stop_acceleration=tuple(cap for acceleration, _ in braking_caps for cap in acceleration),
            stop_jerk=tuple(cap for _, jerk in braking_caps for cap in jerk),
            feedback_tolerance=self.feedback_position_tolerance,
        )

    def feeds_velocity(self, arm: str) -> bool:
        """Whether ``arm`` receives the planned joint velocity with each moving command (by default every arm)."""
        return True

    @classmethod
    def placeholder_key(cls, value: Any, key: str) -> str | None:
        """Returns the dotted key, under ``key``, of the first text in ``value`` that is a placeholder, <...>; None when
        there is none."""
        if isinstance(value, (str, Path)):
            text = str(value)
            return key if text.startswith("<") and text.endswith(">") else None
        if isinstance(value, dict):
            children = [(f"{key}.{child_key}" if key else str(child_key), child) for child_key, child in value.items()]
        elif isinstance(value, (list, tuple)):
            children = [(f"{key}.{index}" if key else str(index), child) for index, child in enumerate(value)]
        else:
            return None

        for child_key, child in children:
            if child_placeholder_key := cls.placeholder_key(child, child_key):
                return child_placeholder_key

        return None
