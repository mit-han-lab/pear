from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.trajectory import GripperProfile, JointMove
from pear.deploy.station.control.schemas import Observation


@dataclass(frozen=True)
class PreparedAction:
    """A validated action: what its grippers are asked to do and its arm move, each if the action has one."""

    gripper_request: Any  # what the robot's gripper operation is asked for; None: no gripper moves
    move: JointMove | None  # None: the arms hold


class HomePlanningError(ValueError):
    """Raised when the direct return to Home cannot be validated."""


class HomePlanning:
    """What the planning of every robot's direct return to Home shares: the move with its check against the position
    bounds, the recheck once the grippers have opened, and how long a plan says Home takes."""

    TARGET_TOLERANCE = 1e-8  # between the approved and the current targets
    MOVE_TOLERANCE = 1e-10  # below this a Home move is not needed
    # the refusal of an approved move whose start the arms no longer hold
    TARGETS_CHANGED = "Home targets changed; check and confirm again"

    @classmethod
    def home_move(
        cls, robot_config: RobotConfig, start: np.ndarray, goal: np.ndarray, offset: np.ndarray
    ) -> JointMove | None:
        """Returns the checked direct move from the targets ``start`` to the targets ``goal`` at Home's speed, None
        when the arms hold the goal already; ``offset`` is how far the measured joints are from the start."""
        limits, home = robot_config.joint_limits(), robot_config.home

        move = None
        if np.max(abs(start - goal)) > cls.MOVE_TOLERANCE:
            move = JointMove.build(
                start, goal, limits, robot_config.LAYOUT, speed_scale=home.speed_scale, caps=home.limits
            )

        try:
            cls.check_home_bounds(robot_config, move, start, offset)
        except ValueError as error:
            raise HomePlanningError(f"Direct Home move rejected: {error}") from error

        return move

    @staticmethod
    def time_estimates(
        control_config: ControlConfig,
        robot_config: RobotConfig,
        move: JointMove | None,
        opening_s: float,
        has_opening: bool,
    ) -> dict[str, float]:
        """Returns the entries of a Home plan that say how long Home takes: its move, its grippers' opening of
        ``opening_s``, and the whole operation with the settling after each of the two."""
        move_s = move.duration_s if move is not None else 0.0
        settle_s = max(control_config.settle_window_s, robot_config.home.settle_window_s)

        return {
            "trajectory_seconds": move_s,
            "opening_seconds_estimate": opening_s,
            "operation_seconds_estimate": opening_s + move_s + (has_opening + (move is not None)) * settle_s,
        }

    @classmethod
    def recheck_home_move(cls, robot_config: RobotConfig, observation: Observation, move: JointMove | None) -> None:
        """Raises unless the approved ``move`` still starts at the targets the arms hold in ``observation``, taken
        once the grippers have opened, and stays within the position bounds with its feedback."""
        measured_positions = cls.measured_positions(observation, robot_config.LAYOUT)
        targets = SnapshotCheck.target_vector(observation["control_snapshot"])
        if move is not None and not np.allclose(targets, move.start, atol=cls.TARGET_TOLERANCE, rtol=0):
            raise HomePlanningError(cls.TARGETS_CHANGED)

        cls.check_home_bounds(robot_config, move, targets, measured_positions - targets)

    @staticmethod
    def check_home_bounds(
        robot_config: RobotConfig, move: JointMove | None, start: np.ndarray, offset: np.ndarray
    ) -> None:
        """Raises unless the planned motion from the targets ``start`` stays within the position bounds, also when
        the offset of the measured joints shifts it."""
        # a start held past a limit since enabling may only move inward; the measured pose may rest past a limit
        # within the feedback tolerance, and the shifted motion may not go beyond it
        limits, layout = robot_config.joint_limits(), robot_config.LAYOUT
        measured_positions = start + offset
        limits.check_positions(measured_positions, layout, limits.feedback_tolerance)
        lower_bounds, upper_bounds = limits.bounds(start)
        shifted_limits = dataclasses.replace(
            limits,
            lower_bounds=tuple(np.minimum(lower_bounds, measured_positions)),
            upper_bounds=tuple(np.maximum(upper_bounds, measured_positions)),
        )

        if move is None:
            points = [start]
        else:
            # the least and the greatest position of each coordinate along the move, as Ruckig names them
            extrema = move.trajectory.position_extrema
            points = [np.array([getattr(axis, bound) for axis in extrema]) for bound in ("min", "max")]
        for point in points:
            limits.check_command(point, start, layout)
            shifted_limits.check_positions(point + offset, layout)

    @staticmethod
    def measured_positions(observation: Observation, layout: JointLayout) -> np.ndarray:
        """Returns the measured joint vector of ``observation`` with every gripper's opening within [0, 1]."""
        # the control loop accepts a gripper's feedback within its allowance: a reading just past an endpoint of
        # the opening is no commanded target
        measured_positions = np.asarray(observation["joint_positions"], dtype=float).copy()
        gripper_indices = list(layout.gripper_indices)
        measured_positions[gripper_indices] = np.clip(measured_positions[gripper_indices], 0, 1)

        return measured_positions

    @classmethod
    def needs_opening(cls, target_opening: float, measured_opening: float, gripper_tolerance: float) -> bool:
        """Whether a gripper opens before the arms return: its target or its measured opening is not the open
        endpoint."""
        return bool(
            abs(target_opening - GripperProfile.OPEN_ENDPOINT) > cls.TARGET_TOLERANCE
            or abs(measured_opening - GripperProfile.OPEN_ENDPOINT) > gripper_tolerance
        )
