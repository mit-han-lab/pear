from collections.abc import Sequence
from typing import Any

import numpy as np


class FrictionFeedforward:
    """The signed friction feedforward of each joint, which follows the planned velocity: it ramps in, stays latched
    after the plan stops and then fades."""

    MIN_MOVING_SPEED = 0.005  # the planned speed above which a joint counts as moving, rad/s
    LATCH_S = 0.2  # how long the feedforward stays after the plan stops
    FADE_S = 0.3  # how long it then takes to fade out
    ENGAGE_S = 0.05  # how long engaging and reversing take, so that no step reaches the motors
    MAX_STEP_S = 0.1  # the most one update advances the ramps: a stalled caller must not advance them by a whole stall

    def __init__(self, magnitude: Sequence[float], load_ratio: Sequence[float]) -> None:
        # each joint's feedforward at no load, in its user's unit (a torque or a goal offset), and what each unit of
        # the gravity load the joint carries adds to it (gearbox friction)
        self.magnitude, self.load_ratio = self.checked_values(magnitude, load_ratio)
        self.reset()

    def reset(self) -> None:
        """Forgets every joint's motion: no feedforward until the plan moves again."""
        joint_count = len(self.magnitude)
        self.level = np.zeros(joint_count)  # the signed fraction of the magnitude, in [-1, 1]
        self.direction = np.zeros(joint_count)
        self.moved_at = np.full(joint_count, -np.inf)
        self.updated_at: float | None = None

    def update(self, planned_velocity: Sequence[float] | None, now: float, load: Sequence[float] | None) -> np.ndarray:
        """Advances to ``now`` and returns each joint's feedforward for the gravity load it carries."""
        # the sign comes from the plan, never from the measured velocity, which chatters around zero at rest
        velocity = self.joint_velocity(planned_velocity)
        load = self.joint_load(load)
        elapsed_s = 0.0 if self.updated_at is None else float(np.clip(now - self.updated_at, 0.0, self.MAX_STEP_S))
        self.updated_at = now

        is_moving = np.abs(velocity) > self.MIN_MOVING_SPEED
        self.direction[is_moving] = np.sign(velocity[is_moving])
        self.moved_at[is_moving] = now
        is_latched = now - self.moved_at <= self.LATCH_S
        target_level = np.where(is_latched, self.direction, 0.0)
        rate = np.where(target_level != 0, 1 / self.ENGAGE_S, 1 / self.FADE_S)
        self.level += np.clip(target_level - self.level, -rate * elapsed_s, rate * elapsed_s)

        return (self.magnitude + self.load_ratio * np.abs(load)) * self.level

    def joint_velocity(self, planned_velocity: Sequence[float] | None) -> np.ndarray:
        """Returns the planned velocity of each joint: zeros when it is missing or malformed, so that the feedforward
        fades."""
        return self._joint_values(planned_velocity)

    def joint_load(self, load: Sequence[float] | None) -> Any:
        """Returns the gravity load each joint carries: zeros when it is missing or malformed."""
        return self._joint_values(load)

    def _joint_values(self, values: Sequence[float] | None) -> np.ndarray:
        """Returns ``values`` as one finite number for each joint: zeros when they are missing or malformed."""
        if values is not None:
            array = np.asarray(values, dtype=float)
            if array.shape == self.magnitude.shape and np.isfinite(array).all():
                return array

        return np.zeros_like(self.magnitude)

    @staticmethod
    def checked_values(magnitude: Sequence[float], load_ratio: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns each joint's magnitude and load ratio, refused unless both are finite and non-negative, one for
        each joint."""
        magnitude_values = np.asarray(magnitude, dtype=float)
        if magnitude_values.ndim != 1 or (magnitude_values < 0).any() or not np.isfinite(magnitude_values).all():
            raise ValueError("Friction torques must be finite and non-negative")
        load_ratio_values = np.asarray(load_ratio, dtype=float)
        if load_ratio_values.shape != magnitude_values.shape or not (
            np.isfinite(load_ratio_values).all() and (load_ratio_values >= 0).all()
        ):
            raise ValueError("Friction load ratios must be finite, non-negative, one per joint")

        return magnitude_values, load_ratio_values
