import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.motion.actions import StrictModel


class UnreachablePoseError(ValueError):
    """Raised when a requested pose fails the offline inverse kinematics; nothing was commanded."""


class BaseKinematics(ABC):
    """The base of a robot's kinematics: what the model reads of the measured joints, and the parts of a solve that
    need no model of an arm."""

    REACH_BISECTIONS = 7  # halvings of a requested increment when reporting how far an arm can go
    # the errors of a solve that count as out of reach while the reachable fraction of an increment is sought
    UNREACHABLE_ERRORS: tuple[type[Exception], ...] = (UnreachablePoseError,)

    @abstractmethod
    def quantities(self, joint_positions: Sequence[float]) -> StrictModel:
        """Returns what the model reads of the measured ``joint_positions``, rounded the way it is shown."""

    @classmethod
    def reachable_fraction(cls, solve: Callable[[float], Any]) -> float:
        """Returns the largest fraction of a requested increment that ``solve`` solves, found by bisection."""
        low_fraction, high_fraction = 0.0, 1.0

        for _ in range(cls.REACH_BISECTIONS):
            middle_fraction = (low_fraction + high_fraction) / 2
            try:
                solve(middle_fraction)
                low_fraction = middle_fraction
            except cls.UNREACHABLE_ERRORS:
                high_fraction = middle_fraction

        return low_fraction

    @staticmethod
    def rotation_distance(first_rotation: np.ndarray, second_rotation: np.ndarray) -> float:
        """Returns the angle between two rotations, each a 3x3 matrix."""
        return math.acos(float(np.clip((np.trace(first_rotation.T @ second_rotation) - 1) / 2, -1, 1)))

    @staticmethod
    def truncated_text(value: float, decimals: int) -> str:
        """Returns ``value`` truncated toward zero to ``decimals`` places, as text with its sign."""
        value = math.trunc(value * 10**decimals) / 10**decimals

        # an unsigned zero: "+0.0 of -2" would read as the opposite direction
        return f"{value:+.{decimals}f}" if value else f"{0:.{decimals}f}"
