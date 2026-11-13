from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.trajectory import CommandSample, GripperProfile, JointMove, JointTrajectory
from pear.deploy.station.control.schemas import MotionEndReason


class OperationInterrupted(RuntimeError):
    """Raised by a bounded operation to request a controlled stop, which is no hardware fault."""

    def __init__(self, reason: MotionEndReason) -> None:
        super().__init__(reason.value)
        self.reason = reason  # how the motion ends


class GripperEndpointTimeout(TimeoutError):
    """Raised when a gripper did not reach the endpoint it was sent to before the deadline."""


class Operation(ABC):
    """What a control loop asks once every control period for the next command."""

    stops_from_standstill = False  # whether a stop brakes from standstill instead of the commanded motion

    @abstractmethod
    def advance(self, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]) -> CommandSample:
        """Returns the next command sample."""

    def fault_targets(
        self, error: Exception, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        """Returns the targets to hold after ``error`` faulted this operation (by default the targets held)."""
        return targets


class TargetOperation(Operation):
    """Advances the shared online trajectory; also the control loop's controlled stop."""

    def __init__(self, trajectory: JointTrajectory) -> None:
        self.trajectory = trajectory

    def advance(self, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]) -> CommandSample:
        """Returns the next command sample of the trajectory."""
        return self.trajectory.advance()


class JointMoveOperation(TargetOperation):
    """Plays a checked JointMove at the control rate."""

    def __init__(self, trajectory: JointTrajectory, move: JointMove) -> None:
        super().__init__(trajectory)
        self.move, self.elapsed_s = move, 0.0

    def advance(self, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]) -> CommandSample:
        """Returns the move's command sample one control period on."""
        control_period_s = self.trajectory.control_config.control_period_s
        self.elapsed_s = min(self.elapsed_s + control_period_s, self.move.duration_s)
        position, velocity, acceleration = self.move.at_time(self.elapsed_s)

        return CommandSample(position, velocity, acceleration, self.elapsed_s >= self.move.duration_s)


class SequenceOperation(TargetOperation):
    """Streams a finite chunk of absolute targets at its own rate, interrupting when it falls behind."""

    MAX_ROWS = 1000  # targets in one chunk
    PHASE_EPSILON = 1e-10  # rounding slack of the phase of a row

    def __init__(self, trajectory: JointTrajectory, positions: Sequence[Sequence[float]], rate_hz: float) -> None:
        super().__init__(trajectory)
        layout, control_config = trajectory.layout, trajectory.control_config

        self.positions = np.array(positions, dtype=float, copy=True)
        if (
            self.positions.ndim != 2
            or self.positions.shape[1:] != (layout.vector_length,)
            or not 1 <= len(self.positions) <= self.MAX_ROWS
        ):
            raise ValueError(f"Expected 1..{self.MAX_ROWS} finite {layout.vector_length}D targets")
        if not np.isfinite(rate_hz) or not 1 / control_config.action_timeout_s < rate_hz <= control_config.rate_hz:
            raise ValueError("Action rate must exceed timeout rate and not exceed controller rate")
        for row in self.positions:
            trajectory.limits.check_positions(row, layout)

        self.positions.setflags(write=False)
        self.rate_hz = float(rate_hz)
        self.index = 0  # of the next row to hand to the trajectory
        self.row_phase = 0.0  # the share of a row's time that has passed since the last row, 1 when the next is due
        self.start_time = None  # when the first row was handed over

    @property
    def is_complete(self) -> bool:
        """Whether every target was handed to the trajectory."""
        return self.index == len(self.positions)

    def advance(self, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]) -> CommandSample:
        """Returns the next command sample, heading for the next row once its time has come."""
        control_config = self.trajectory.control_config
        if not self.is_complete:
            if self.start_time is None:
                self.start_time, self.row_phase = now, 1.0
            else:
                self.row_phase += self.rate_hz * control_config.control_period_s
            if self.row_phase >= 1 - self.PHASE_EPSILON:
                lateness_s = now - (self.start_time + self.index / self.rate_hz)
                if lateness_s > control_config.control_period_s + control_config.deadline_tolerance_s:
                    raise OperationInterrupted(MotionEndReason.SEQUENCE_DEADLINE)
                self.trajectory.set_target(self.positions[self.index])
                self.row_phase -= 1
                self.index += 1

        sample = self.trajectory.advance()
        sample.is_finished &= self.is_complete

        return sample


class GripperOperation(Operation):
    """The base of a robot's gripper operation, which moves grippers to an endpoint of their opening while every
    arm target stays held."""

    stops_from_standstill = True  # a stop brakes from standstill, not from the grippers' own profiles

    def __init__(self, control_config: ControlConfig, robot_config: RobotConfig, opens_by_arm: dict[str, bool]) -> None:
        self.control_config, self.robot_config = control_config, robot_config
        self.opens_by_arm = opens_by_arm  # whether the gripper of each arm the operation moves opens or closes
        # per arm, the measured opening a gripper stopped at on an object, at which its arrival is then judged
        self.blocked_gripper_openings: dict[str, float] | None = None

    def new_profile(self, arm: str, start: float, start_velocity: float = 0.0) -> GripperProfile:
        """Returns the trajectory of ``arm``'s gripper from the opening ``start`` to its endpoint, within the arm's
        caps."""
        limits = self.robot_config.arm_limits()[arm]

        return GripperProfile(
            start, self.opens_by_arm[arm], limits, self.control_config.control_period_s, start_velocity
        )

    def fault_targets(
        self, error: Exception, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        """Returns the targets to hold after ``error``: a gripper whose opening was obstructed is released to its
        measured opening, if that is fresh."""
        targets = dict(targets)
        if isinstance(error, GripperEndpointTimeout):
            for arm, opens in self.opens_by_arm.items():
                state = states[arm]
                age_s = now - state.sampled_monotonic_s
                if opens and ArmState.is_fresh(age_s, self.control_config.state_timeout_s):
                    measured_opening = float(np.clip(JointLayout.gripper_value(state.position), 0, 1))
                    targets[arm] = JointLayout.with_gripper(targets[arm], measured_opening)

        return targets

    @abstractmethod
    def report(self) -> dict[str, Any]:
        """Returns what the robot's reports keep of the operation once it has ended, under the robot's own keys."""
