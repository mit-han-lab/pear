from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.feedforward import FrictionFeedforward
from pear.deploy.station.control.robots.so101.config import FeedforwardConfig
from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX, JOINTS, VECTOR_LENGTH

# pinocchio is imported where a gravity model loads: it is heavy, and only a driver with feedforward needs it


class GravityModel:
    """The gravity torque on the five arm joints, N*m, from the URDF's masses at the station's joint angles."""

    MODEL_CACHE_SIZE = 4  # the loaded descriptions the gravity models of a process share

    def __init__(self, urdf: Path) -> None:
        import pinocchio

        self.model, self._position_indices, self._velocity_indices = self.load(str(urdf))
        self.data = self.model.createData()  # one per instance: pinocchio's data is not shared
        self._compute_gravity = pinocchio.computeGeneralizedGravity

    def torque(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Returns the gravity torque of each arm joint at the joint vector ``joint_positions``."""
        full_configuration = np.zeros(self.model.nq)
        full_configuration[self._position_indices] = np.asarray(joint_positions, dtype=float)[: len(JOINTS)]

        # positive: what the joint must supply, so that its servo settles that far below its goal
        return self._compute_gravity(self.model, self.data, full_configuration)[self._velocity_indices].copy()

    @staticmethod
    @lru_cache(maxsize=MODEL_CACHE_SIZE)
    def load(urdf: str) -> tuple[Any, list[int], list[int]]:
        """Returns pinocchio's model of the description ``urdf`` and the arm joints' configuration and velocity
        indices, loaded once per process."""
        import pinocchio

        model = pinocchio.buildModelFromUrdf(urdf)
        joint_ids = [model.getJointId(name) for name in JOINTS]
        if any(joint_id >= model.njoints for joint_id in joint_ids):
            raise ValueError(f"{urdf} lacks one of the joints {JOINTS}")
        position_indices = [model.joints[joint_id].idx_q for joint_id in joint_ids]
        velocity_indices = [model.joints[joint_id].idx_v for joint_id in joint_ids]

        return model, position_indices, velocity_indices


class SO101FrictionFeedforward(FrictionFeedforward):
    """The arm's friction feedforward, a goal offset in rad: the robot's own checks of its values, its planned velocity
    cut to the arm joints, and its gravity load used as it comes."""

    def joint_velocity(self, planned_velocity: Sequence[float] | None) -> np.ndarray:
        """Returns the arm joints' part of the planned velocity: zeros when it is missing or malformed."""
        velocity = np.zeros_like(self.magnitude)
        if planned_velocity is not None:
            plan = np.asarray(planned_velocity, dtype=float)[: len(self.magnitude)]
            if plan.shape == velocity.shape and np.isfinite(plan).all():
                velocity = plan

        return velocity

    def joint_load(self, load: Sequence[float] | None) -> Any:
        """Returns the gravity load as it is given: the gravity model's torque of the commanded target."""
        return load

    @staticmethod
    def checked_values(magnitude: Sequence[float], load_ratio: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns each joint's offset and load ratio, refused unless one of each for every joint, non-negative."""
        offset_values = np.asarray(magnitude, dtype=float)
        load_ratio_values = np.asarray(load_ratio, dtype=float)
        if offset_values.shape != load_ratio_values.shape or offset_values.ndim != 1:
            raise ValueError("Expected one friction offset and load ratio per joint")
        if (offset_values < 0).any() or (load_ratio_values < 0).any():
            raise ValueError("Friction offsets and load ratios must be non-negative")

        return offset_values, load_ratio_values


class Feedforward:
    """The goal offset of a commanded target: gravity always, friction while it lasts."""

    def __init__(self, feedforward_config: FeedforwardConfig, urdf: Path) -> None:
        self.compliance = np.asarray(feedforward_config.gravity_compliance, dtype=float)
        self.gravity_model = GravityModel(urdf)
        self.friction_feedforward = SO101FrictionFeedforward(
            feedforward_config.friction_offset_rad, feedforward_config.friction_load_ratio
        )

    def offset(self, target: Sequence[float], planned_velocity: Sequence[float] | None, now: float) -> np.ndarray:
        """Returns the goal offset of ``target``, which moves at the arm's ``planned_velocity``; the gripper never gets
        one."""
        torque = self.gravity_model.torque(target)

        # the servos run a P-only position loop and take no torque command, so a joint stops short by the torque it
        # carries over P: the offset is added to its goal instead
        offset = np.zeros(VECTOR_LENGTH)
        offset[:GRIPPER_INDEX] = self.compliance * torque + self.friction_feedforward.update(
            planned_velocity, now, torque
        )
        return offset

    def hold_offset(self, target: Sequence[float]) -> np.ndarray:
        """Returns the goal offset of holding ``target``: gravity alone, the friction's state cleared."""
        self.friction_feedforward.reset()

        offset = np.zeros(VECTOR_LENGTH)
        offset[:GRIPPER_INDEX] = self.compliance * self.gravity_model.torque(target)
        return offset
