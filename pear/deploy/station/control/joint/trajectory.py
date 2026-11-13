from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig, Synchronization, Trajectory

from pear.deploy.station.control.joint.config import ControlConfig, MotionLimits
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.limits import JointLimits


@dataclass
class CommandSample:
    """One control period's commanded joint vector and its derivatives."""

    position: Sequence[float] | np.ndarray
    velocity: Sequence[float] | np.ndarray
    acceleration: Sequence[float] | np.ndarray
    is_finished: bool
    requested_target: Sequence[float] | np.ndarray | None = None  # a new held target the sample commits, if any


class JointTrajectory:
    """The online Ruckig state that finite sequences, holds and controlled stops share."""

    CAP_INPUTS = ("max_velocity", "max_acceleration", "max_jerk")  # Ruckig's inputs of the three motion caps

    def __init__(self, limits: JointLimits, layout: JointLayout, control_config: ControlConfig) -> None:
        self.limits, self.layout, self.control_config = limits, layout, control_config
        self.generator = Ruckig(layout.vector_length, control_config.control_period_s)
        self.input = InputParameter(layout.vector_length)
        self.output = OutputParameter(layout.vector_length)
        self.duration_s = 0.0  # of the trajectory last computed

    def hold(self, position: Sequence[float]) -> None:
        """Rests at ``position`` with zero derivatives, within the station's limits."""
        position = [float(value) for value in self.layout.vector(position)]

        self.head_for(position)
        self.input.current_position = position
        self.input.current_velocity = self.input.current_acceleration = [0.0] * self.layout.vector_length

    def head_for(self, target: Sequence[float], motion_caps: Sequence[Sequence[float]] | None = None) -> None:
        """Heads for rest at ``target`` in position mode, within ``motion_caps`` (by default the station's limits)."""
        motion_caps = self.limits.motion_caps() if motion_caps is None else motion_caps
        for name, caps in zip(self.CAP_INPUTS, motion_caps):
            setattr(self.input, name, list(caps))

        self.input.control_interface = ControlInterface.Position
        self.input.target_position = [float(value) for value in target]
        self.input.target_velocity = self.input.target_acceleration = [0.0] * self.layout.vector_length

    def set_target(self, target: Sequence[float]) -> None:
        """Heads for a new checked target at the station's limits, keeping the old input if the target is rejected."""
        self.limits.check_positions(target, self.layout)
        names = ("target_position", "control_interface", *self.CAP_INPUTS)
        previous_inputs = {name: getattr(self.input, name) for name in names}

        try:
            self.head_for(self.layout.vector(target))
            self.compute_trajectory()
        except BaseException:
            for name, value in previous_inputs.items():
                setattr(self.input, name, value)
            raise

    def advance(self) -> CommandSample:
        """Returns the sample of the next control period."""
        result = self.generator.update(self.input, self.output)
        if result < 0:
            raise RuntimeError(f"Trajectory update failed: {result}")

        return CommandSample(
            self.output.new_position,
            self.output.new_velocity,
            self.output.new_acceleration,
            result == Result.Finished,
        )

    def accept(self, sample: CommandSample) -> None:
        """Continues from ``sample``, which was sent."""
        self.input.current_position = list(sample.position)
        self.input.current_velocity = list(sample.velocity)
        self.input.current_acceleration = list(sample.acceleration)
        if sample.requested_target is not None:
            self.input.target_position = list(sample.requested_target)

    def stop(self, *, from_standstill: bool, braking_caps: tuple[Sequence[float], Sequence[float]] | None) -> None:
        """Brakes to rest, from standstill if ``from_standstill``, with the acceleration and jerk caps
        ``braking_caps`` if given."""
        zeros = [0.0] * self.layout.vector_length
        if from_standstill:
            self.input.current_velocity = self.input.current_acceleration = zeros
        if braking_caps is not None:
            self.input.max_acceleration, self.input.max_jerk = (list(caps) for caps in braking_caps)
        self.input.control_interface = ControlInterface.Velocity
        self.input.target_velocity = self.input.target_acceleration = zeros

        self.compute_trajectory()

    def compute_trajectory(self) -> None:
        """Computes the trajectory to the input's target, and raises if Ruckig rejects it or it leaves the bounds."""
        trajectory = Trajectory(self.layout.vector_length)
        if self.generator.calculate(self.input, trajectory) < 0:
            raise ValueError("Trajectory rejected")
        self.limits.check_trajectory(trajectory, self.input.current_position)

        self.duration_s = trajectory.duration


class GripperProfile:
    """One gripper's trajectory to an endpoint of its opening, advanced one control period at a time."""

    OPEN_ENDPOINT = 1.0  # the opening of an open gripper
    CLOSED_ENDPOINT = 0.0  # the opening of a closed gripper

    def __init__(
        self, start: float, opens: bool, limits: MotionLimits, control_period_s: float, start_velocity: float = 0.0
    ) -> None:
        self.generator = Ruckig(1, control_period_s)
        self.input = self.ruckig_input(start, opens, limits)
        self.input.current_velocity = [start_velocity]
        self.output = OutputParameter(1)
        self.endpoint = self.input.target_position[0]  # the opening the gripper heads for

    def advance(self) -> tuple[float, float, float, bool]:
        """Returns the gripper's next position, velocity and acceleration, and whether the trajectory has finished."""
        result = self.generator.update(self.input, self.output)
        if result < 0:
            raise RuntimeError(f"Gripper trajectory update failed: {result}")

        return (
            self.output.new_position[0],
            self.output.new_velocity[0],
            self.output.new_acceleration[0],
            result == Result.Finished,
        )

    def accept(self, position: float) -> None:
        """Continues from the sample last returned, at the ``position`` that was commanded for it."""
        self.output.pass_to_input(self.input)
        self.input.current_position = [position]

    @classmethod
    def duration_s(cls, start: float, opens: bool, limits: MotionLimits) -> float:
        """Returns how long a gripper takes from rest at the opening ``start`` to its endpoint."""
        trajectory = Trajectory(1)
        if Ruckig(1).calculate(cls.ruckig_input(start, opens, limits), trajectory) < 0:
            raise ValueError("Ruckig rejected the gripper endpoint")

        return trajectory.duration

    @classmethod
    def ruckig_input(cls, start: float, opens: bool, limits: MotionLimits) -> InputParameter:
        """Returns Ruckig's input of the trajectory from rest at the opening ``start`` to the open or the closed
        endpoint, within the gripper's caps of its arm's ``limits``."""
        ruckig_input = InputParameter(1)
        ruckig_input.current_position = [float(start)]
        ruckig_input.current_velocity = ruckig_input.current_acceleration = [0.0]
        ruckig_input.target_position = [cls.OPEN_ENDPOINT if opens else cls.CLOSED_ENDPOINT]
        ruckig_input.target_velocity = ruckig_input.target_acceleration = [0.0]
        ruckig_input.max_velocity = [JointLayout.gripper_value(limits.velocity)]
        ruckig_input.max_acceleration = [JointLayout.gripper_value(limits.acceleration)]
        ruckig_input.max_jerk = [JointLayout.gripper_value(limits.jerk)]

        return ruckig_input


@dataclass(frozen=True)
class JointMove:
    """An immutable rest-to-rest move, checked against the station's limits when built."""

    MAX_DURATION_S = 120.0  # longer moves are rejected rather than run

    start: tuple[float, ...]
    goal: tuple[float, ...]
    motion_caps: tuple[tuple[float, ...], ...]  # the velocity, acceleration and jerk caps of the move

    @classmethod
    def build(
        cls,
        start: Sequence[float],
        goal: Sequence[float],
        limits: JointLimits,
        layout: JointLayout,
        *,
        speed_scale: float = 1.0,
        caps: MotionLimits | None = None,
    ) -> JointMove:
        """Returns a checked move from the held command ``start``, which may rest just outside a limit, to
        ``goal``."""
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        vector_shape = (layout.vector_length,)
        if start.shape != vector_shape or goal.shape != vector_shape or not np.isfinite([start, goal]).all():
            raise ValueError(f"Expected finite {layout.vector_length}D start and goal")
        limits.check_positions(start, layout, limits.feedback_tolerance)
        limits.check_command(goal, start, layout)

        motion_caps = tuple(tuple(cap_values) for cap_values in limits.motion_caps(speed_scale, caps))
        move = cls(tuple(start), tuple(goal), motion_caps)
        limits.check_trajectory(move.trajectory, start)

        return move

    @cached_property
    def trajectory(self) -> Trajectory:
        """The Ruckig trajectory of this move."""
        vector_length = len(self.start)
        ruckig_input = InputParameter(vector_length)
        ruckig_input.current_position, ruckig_input.target_position = self.start, self.goal
        ruckig_input.max_velocity, ruckig_input.max_acceleration, ruckig_input.max_jerk = self.motion_caps
        # rest-to-rest phase synchronization keeps the joint-space path straight, for several arms at once too
        ruckig_input.synchronization = Synchronization.Phase

        trajectory = Trajectory(vector_length)
        if Ruckig(vector_length).calculate(ruckig_input, trajectory) < 0:
            raise ValueError("Ruckig rejected the joint target")
        if not np.isfinite(trajectory.duration) or trajectory.duration > self.MAX_DURATION_S:
            raise ValueError(f"Joint trajectory exceeds {self.MAX_DURATION_S:g} seconds")

        return trajectory

    @property
    def duration_s(self) -> float:
        """The move's duration."""
        return self.trajectory.duration

    def at_time(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the position, velocity and acceleration ``seconds`` into the move."""
        if not np.isfinite(seconds):
            raise ValueError("Trajectory time must be finite")

        return tuple(np.asarray(vector) for vector in self.trajectory.at_time(np.clip(seconds, 0, self.duration_s)))

    def __getstate__(self) -> dict[str, Any]:
        # native Ruckig objects cannot cross a process boundary: each process builds the trajectory once
        return {key: value for key, value in self.__dict__.items() if key != "trajectory"}
