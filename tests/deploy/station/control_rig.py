from __future__ import annotations

import time
from typing import Any

import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.schemas import MOVING_CONTROL_STATES


class StandInClock:
    """The monotonic clock and the sleep of the station's code during a test: time stands still until the test or a
    sleep advances it, and each sleep steps the control loop the test attached, as the loop's thread would."""

    START_S = 100.0  # the clock's first reading

    def __init__(self) -> None:
        self.now = self.START_S  # what the monotonic clock reads
        self.control_loop: JointControlLoop | None = None  # what each sleep steps once; None: a sleep only advances

    def install(self, monkeypatch: pytest.MonkeyPatch) -> StandInClock:
        """Has every module read this clock and sleep on it until the test ends, and returns the clock."""
        monkeypatch.setattr(time, "monotonic", self.monotonic)
        monkeypatch.setattr(time, "sleep", self.sleep)

        return self

    def monotonic(self) -> float:
        """Returns the clock's reading."""
        return self.now

    def sleep(self, duration_s: float) -> None:
        """Advances the clock by ``duration_s`` and steps the attached control loop once."""
        self.now += duration_s
        if self.control_loop is not None:
            self.control_loop.step()


class Ticker:
    """Steps a control loop by hand on the stand-in clock, one control period at a time."""

    STEPS = 2000  # the most control periods an operation may take

    def __init__(self, control_loop: JointControlLoop, clock: StandInClock) -> None:
        self.control_loop, self.clock = control_loop, clock

    def tick(self, count: int = 1, *, period_s: float | None = None) -> None:
        """Advances the clock by ``period_s`` (one control period by default) and steps, ``count`` times."""
        for _ in range(count):
            self.clock.now += self.control_loop.control_config.control_period_s if period_s is None else period_s
            self.control_loop.step()

    def until_hold(self) -> int:
        """Steps until the control loop neither runs nor stops an operation, and returns the steps taken."""
        for step in range(self.STEPS):
            if self.control_loop.status not in MOVING_CONTROL_STATES:
                return step
            self.tick()

        raise AssertionError(f"Still {self.control_loop.status} after {self.STEPS} steps")


class ControlRig:
    """A robot's control loop on a stand-in driver and the stand-in clock, made with the loop's constructor and stepped
    by the test; as a context, the loop, its driver and its ticker, the loop closed at the end."""

    def __init__(
        self,
        control_loop_class: type[JointControlLoop],
        driver_class: type[JointDriver],
        control_config: ControlConfig,
        robot_config: RobotConfig,
        clock: StandInClock,
    ) -> None:
        self.driver = driver_class(control_config, robot_config)
        self.control_loop = control_loop_class(self.driver, control_config, robot_config)
        self.ticker = Ticker(self.control_loop, clock)

    def __enter__(self) -> tuple[JointControlLoop, JointDriver, Ticker]:
        return self.control_loop, self.driver, self.ticker

    def __exit__(self, *_: Any) -> None:
        self.control_loop.close()


class SteppedControlLoop(JointControlLoop):
    """A control loop that runs no thread of its own: the test steps it, by hand or through the stand-in clock's
    sleeps; a robot's stepped loop is built on this and on the robot's own loop."""

    def start(self) -> None:
        """Starts no thread."""


class Robots:
    """Each robot's configuration in the tests of the control loop, and a joint vector inside its bounds."""

    @staticmethod
    def load(robot_name: str) -> tuple[ControlConfig, RobotConfig]:
        """Returns the two configurations of the tests' station of the robot ``robot_name``, its arms' limits wide
        enough for a move to go anywhere a test of the control loop sends it."""
        return ConfigFiles.load_configs(robot_name, *ConfigFiles.WIDE_LIMITS[robot_name])

    @staticmethod
    def rest(robot_config: RobotConfig) -> tuple[float, ...]:
        """Returns a joint vector inside every bound of ``robot_config``: the middle of each coordinate's range."""
        limits = robot_config.joint_limits()

        return tuple((lower + upper) / 2 for lower, upper in zip(limits.lower_bounds, limits.upper_bounds))
