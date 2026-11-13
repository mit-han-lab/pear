from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pear.deploy.station.control.joint.layout import JointLayout

if TYPE_CHECKING:  # annotations only: the configuration's module imports this one
    from ruckig import Trajectory

    from pear.deploy.station.control.joint.config import MotionLimits


@dataclass(frozen=True)
class JointLimits:
    """A robot's joint bounds and motion caps, one value for each coordinate of its joint vector."""

    EPSILON = 1e-9  # rounding slack of every bound comparison
    CAP_NAMES = ("velocity", "acceleration", "jerk")  # the motion caps, in the order of their derivatives

    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]
    jerk: tuple[float, ...]
    tracking_error: tuple[float, ...]
    stop_acceleration: tuple[float, ...]  # a controlled STOP brakes with these
    stop_jerk: tuple[float, ...]
    feedback_tolerance: float  # how far measured feedback may rest outside the bounds

    def motion_caps(self, speed_scale: float = 1.0, caps: MotionLimits | None = None) -> list[list[float]]:
        """Returns the velocity, acceleration and jerk caps, within ``caps`` (one arm's values, for every arm),
        time-scaled by ``speed_scale``."""
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("Speed scale must be finite and within (0, 1]")

        motion_caps = []
        for order, name in enumerate(self.CAP_NAMES, 1):
            arm_caps = getattr(caps, name) if caps is not None else None
            motion_caps.append(
                [
                    (min(cap, arm_caps[index % len(arm_caps)]) if arm_caps is not None else cap) * speed_scale**order
                    for index, cap in enumerate(getattr(self, name))
                ]
            )

        return motion_caps

    def braking_caps(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Returns the acceleration and jerk caps of a controlled STOP."""
        return self.stop_acceleration, self.stop_jerk

    def check_positions(self, vector: Sequence[float], layout: JointLayout, tolerance: float = 0.0) -> None:
        """Raises unless ``vector`` lies within the bounds, widened by ``tolerance`` for measured feedback."""
        lower_bounds, upper_bounds = np.array(self.lower_bounds), np.array(self.upper_bounds)
        texts = layout.texts

        self._check_bounds(
            layout.vector(vector),
            lower_bounds,
            upper_bounds,
            tolerance,
            layout,
            texts.outside_bounds,
            texts.outside_bounds_coordinate,
        )

    def check_command(self, target: Sequence[float], previous_target: Sequence[float], layout: JointLayout) -> None:
        """Raises if ``target`` lies outside a bound and further outside it than ``previous_target``."""
        lower_bounds, upper_bounds = self.bounds(layout.vector(previous_target))
        texts = layout.texts

        self._check_bounds(
            layout.vector(target),
            lower_bounds,
            upper_bounds,
            0.0,
            layout,
            texts.command_outside_bounds,
            texts.command_outside_bounds_coordinate,
        )

    def check_trajectory(self, trajectory: Trajectory, start: Sequence[float]) -> None:
        """Raises if Ruckig's position extrema of ``trajectory`` leave the bounds for a motion from ``start``."""
        lower_bounds, upper_bounds = self.bounds(start)

        for extremum, lower_bound, upper_bound in zip(trajectory.position_extrema, lower_bounds, upper_bounds):
            if extremum.min < lower_bound - self.EPSILON or extremum.max > upper_bound + self.EPSILON:
                raise ValueError("Ruckig trajectory crosses a position limit")

    def bounds(self, start: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns the position bounds for a motion from ``start``, which may rest outside a limit and move only
        inward."""
        return np.minimum(self.lower_bounds, start), np.maximum(self.upper_bounds, start)

    def _check_bounds(
        self,
        values: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        allowance: float,
        layout: JointLayout,
        message_template: str,
        coordinate_template: str,
    ) -> None:
        """Raises the robot's message for the first arm of which a value lies outside its bounds by more than
        ``allowance``."""
        outside_mask = (values < lower_bounds - allowance - self.EPSILON) | (
            values > upper_bounds + allowance + self.EPSILON
        )
        coordinate_count = layout.coordinate_count

        for arm_index, arm in enumerate(layout.arms):
            indices = [index for index in np.flatnonzero(outside_mask) if index // coordinate_count == arm_index]
            if indices:
                details = "; ".join(
                    coordinate_template.format(
                        axis=layout.texts.bound_axes[index % coordinate_count],
                        value=values[index],
                        unit=layout.axis_unit(index),
                        lower=lower_bounds[index],
                        upper=upper_bounds[index],
                        allowance=allowance,
                    )
                    for index in indices
                )
                raise ValueError(message_template.format(arm=arm, details=details))
