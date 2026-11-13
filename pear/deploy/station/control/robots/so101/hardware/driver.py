import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.hardware.arm_state import ServoFault, SO101ArmState
from pear.deploy.station.control.robots.so101.hardware.arms import SO101Hardware
from pear.deploy.station.control.robots.so101.joint.feedforward import Feedforward
from pear.deploy.station.control.robots.so101.layout import ARM, GRIPPER_INDEX, VECTOR_LENGTH
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class SO101Driver(JointDriver):
    """The SO-101's servos as the control loop's driver: a target carries the feedforward's goal offset, and a bus
    glitch keeps the last reading."""

    HARDWARE_CLASS = SO101Hardware  # the arm the driver enables, reads and commands
    OFFSET_EPSILON = 1e-6  # a held target's offset this close to the one sent needs no new command

    def __init__(self, control_config: ControlConfig, robot_config: SO101Config) -> None:
        self.limits = robot_config.limits  # whose position bounds an offset never pushes a target past
        self.hardware = self.HARDWARE_CLASS(control_config, robot_config)

        # the feedforward is built with the arm enabled; a failure releases the arm
        feedforward_config = robot_config.feedforward
        try:
            self.feedforward = (
                None if feedforward_config is None else Feedforward(feedforward_config, robot_config.geometry.urdf)
            )
        except BaseException:
            self.hardware.close()
            raise

        self.offset = np.zeros(VECTOR_LENGTH)  # the goal offset last sent
        self.state: SO101ArmState | None = None  # the last reading returned
        self.has_commanded = False  # whether a target was sent, after which a bus glitch is tolerated

    def read(self) -> dict[str, SO101ArmState]:
        """Returns the arm's reading; after a bus glitch, the previous one."""
        try:
            self.state = self.hardware.read()
        except ServoFault:  # a servo's fault is never a glitch
            raise
        except Exception as error:
            # nor is a failed read before the first command, while enabling settles; a glitch's reading ages until the
            # control loop finds it stale
            if not self.has_commanded:
                raise
            logger.warning("feedback read failed: %s", error, exc_info=error)

        return {ARM: self.state}

    def read_for_recovery(self) -> dict[str, SO101ArmState]:
        """Returns the arm's reading after checking that every servo is healthy and still holds torque."""
        self.state = self.hardware.read_for_recovery()

        return {ARM: self.state}

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Commands the arm's target with the feedforward of its planned velocity; without velocities it is held."""
        target = targets[ARM]
        if velocities is None:
            self._hold(target)
        else:
            self._move(target, velocities[ARM])

        self.has_commanded = True

    def refresh_hold(self, targets: dict[str, tuple[float, ...]]) -> None:
        """Sends the held target again while the friction offset of the last move fades."""
        self._move(targets[ARM], None, skips_unchanged=True)

    def telemetry(self) -> dict[str, Any]:
        """Returns the arm joints' current feedforward offsets, rad."""
        return {"feedforward": self.offset[:GRIPPER_INDEX].tolist()}

    def close(self) -> None:
        """Releases the arm, which disables its torque."""
        self.hardware.close()

    def _hold(self, target: Sequence[float]) -> None:
        """Holds ``target`` with the gravity offset alone."""
        offset = np.zeros(VECTOR_LENGTH)
        if self.feedforward is not None:
            # a fault's hold never depends on the gravity model
            try:
                offset = self.feedforward.hold_offset(target)
            except Exception:
                logger.exception("Gravity feedforward failed; holding without it")

        self.hardware.send(self._goal(target, offset))
        self.offset = offset

    def _move(
        self, target: Sequence[float], planned_velocity: Sequence[float] | None, *, skips_unchanged: bool = False
    ) -> None:
        """Commands ``target`` with the feedforward of ``planned_velocity``, unless ``skips_unchanged`` and its offset
        is the one sent."""
        offset = np.zeros(VECTOR_LENGTH)
        if self.feedforward is not None:
            offset = self.feedforward.offset(target, planned_velocity, time.monotonic())
        if skips_unchanged and np.allclose(offset, self.offset, rtol=0, atol=self.OFFSET_EPSILON):
            return

        self.hardware.send(self._goal(target, offset))
        self.offset = offset

    def _goal(self, target: Sequence[float], offset: np.ndarray) -> tuple[float, ...]:
        """Returns ``target`` plus ``offset``, never further past a bound than ``target`` itself."""
        target_vector = np.asarray(target, dtype=float)
        lower_bound = np.minimum(self.limits.lower_bounds, target_vector)
        upper_bound = np.maximum(self.limits.upper_bounds, target_vector)

        return tuple(float(coordinate) for coordinate in np.clip(target_vector + offset, lower_bound, upper_bound))
