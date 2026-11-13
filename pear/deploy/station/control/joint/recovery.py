from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.stability import StabilityWindow
from pear.deploy.station.control.joint.trajectory import JointTrajectory
from pear.deploy.station.control.schemas import RecoveryPhase
from pear.deploy.station.exceptions import OperatorPaused

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.control.joint.control_loop import JointControlLoop


@dataclass
class RecoveryRequest:
    """One caller's request to reacquire a hold after a fault; the control thread advances it."""

    window: StabilityWindow
    deadline: float
    cancellation: threading.Event | None = None
    is_done: bool = False
    error: Exception | None = None
    phase: RecoveryPhase = RecoveryPhase.CHECKING_STABILITY
    trajectory: JointTrajectory | None = None  # the release from the old targets to the new hold
    anchor_pose: dict[str, tuple[float, ...]] | None = None  # the measured pose the new hold was taken from
    has_automatic_timeout: bool = False  # whether the policy chose the timeout

    def check_deadline(self) -> None:
        """Raises the window's stability error once the request's deadline has passed."""
        if time.monotonic() >= self.deadline:
            raise self.window.stability_error("Recovery requires fresh stable feedback")


class RecoveryPolicy(ABC):
    """How a robot's control loop holds its arms at a fault and reacquires a settled hold afterwards."""

    @abstractmethod
    def default_timeout_s(self, control_config: ControlConfig) -> float:
        """Returns how long a recovery may take when its caller gives no timeout."""

    @abstractmethod
    def fault_targets(self, control_loop: JointControlLoop) -> dict[str, tuple[float, ...]]:
        """Returns the targets to hold after a fault."""

    @abstractmethod
    def step(
        self, control_loop: JointControlLoop, request: RecoveryRequest, feedback: dict[str, Any], is_stable: bool
    ) -> None:
        """Advances ``request`` by one control period with the window's verdict on ``feedback``."""


class MeasuredHoldRecovery(RecoveryPolicy):
    """A fault holds the arms where they are measured, and a recovery takes the still arms' pose as the hold: for
    stiff position servos, which keep pushing toward an old target."""

    TIMEOUT_SETTLE_COUNT = 2  # settle timeouts a recovery may take by default

    def default_timeout_s(self, control_config: ControlConfig) -> float:
        """Returns how long a recovery may take when its caller gives no timeout."""
        return self.TIMEOUT_SETTLE_COUNT * control_config.settle_timeout_s

    def fault_targets(self, control_loop: JointControlLoop) -> dict[str, tuple[float, ...]]:
        """Returns the targets to hold after a fault: the measured arm joints if fresh, with each gripper's
        command."""
        try:
            control_loop.check_fresh(control_loop.states)
        except RuntimeError:
            return control_loop.targets

        return self._measured_targets(control_loop)

    def step(
        self, control_loop: JointControlLoop, request: RecoveryRequest, feedback: dict[str, Any], is_stable: bool
    ) -> None:
        """Advances ``request`` by one control period with the window's verdict on ``feedback``."""
        if request.phase == RecoveryPhase.CHECKING_STABILITY:
            if not is_stable:
                return

            # the arms are still: their measured pose becomes the hold, and a held object is not released; the pose
            # may rest past a limit, as an arm folded onto its stop, only within the feedback tolerance
            targets = self._measured_targets(control_loop)
            limits, layout = control_loop.limits, control_loop.layout
            limits.check_positions(layout.join(targets), layout, limits.feedback_tolerance)
            OperatorPaused.check_cancelled(request.cancellation)
            control_loop.transfer_hold(targets)
            request.phase = RecoveryPhase.VERIFYING_HOLD
            request.window.reset()
            return

        control_loop.check_tracking(feedback)
        if is_stable:
            control_loop.complete_recovery(request)

    def _measured_targets(self, control_loop: JointControlLoop) -> dict[str, tuple[float, ...]]:
        """Returns each arm's measured joints with its gripper's command as the targets."""
        layout = control_loop.layout

        return {
            arm: layout.with_gripper(control_loop.states[arm].position, layout.gripper_value(control_loop.targets[arm]))
            for arm in layout.arms
        }


class ReleaseRecovery(RecoveryPolicy):
    """A fault keeps the old targets, and a recovery releases them slowly toward the measured pose: for arms that
    comply with an old target."""

    MIN_TIMEOUT_S = 1.0  # the shortest default recovery
    RELEASE_CAPS = (0.1, 0.2, 1.0)  # velocity, acceleration and jerk caps of the release, per coordinate
    RELEASE_MARGIN_S = 1.0  # past the release and its settling before an automatic timeout
    START_EPSILON = 1e-9  # targets this close to the measured pose need no release

    def default_timeout_s(self, control_config: ControlConfig) -> float:
        """Returns how long a recovery may take when its caller gives no timeout; a release extends it."""
        return max(self.MIN_TIMEOUT_S, control_config.settle_timeout_s)

    def fault_targets(self, control_loop: JointControlLoop) -> dict[str, tuple[float, ...]]:
        """Returns the targets to hold after a fault: the old ones."""
        return control_loop.targets

    def step(
        self, control_loop: JointControlLoop, request: RecoveryRequest, feedback: dict[str, Any], is_stable: bool
    ) -> None:
        """Advances ``request`` by one control period with the window's verdict on ``feedback``."""
        if request.phase == RecoveryPhase.CHECKING_STABILITY:
            if not is_stable:
                return
            self._plan_release(control_loop, request)

        self._check_drift(control_loop, request)
        if request.phase == RecoveryPhase.RELEASING_TARGET:
            self._release(control_loop, request, is_stable)
        else:
            self._verify_hold(control_loop, request, feedback, is_stable)

    def _plan_release(self, control_loop: JointControlLoop, request: RecoveryRequest) -> None:
        """Plans the release from the failed targets to the measured arm pose, clipped into the bounds."""
        layout, limits = control_loop.layout, control_loop.limits
        lower_bounds, upper_bounds = layout.split(limits.lower_bounds), layout.split(limits.upper_bounds)
        # each gripper keeps its command, so a force-limited grasp holds
        targets = {
            arm: layout.with_gripper(
                np.clip(control_loop.states[arm].position, lower_bounds[arm], upper_bounds[arm]),
                layout.gripper_value(control_loop.targets[arm]),
            )
            for arm in layout.arms
        }
        start, goal = layout.join(control_loop.targets), layout.join(targets)
        # targets held since enabling may rest past a limit; the release may only move them inward
        limits.check_positions(start, layout, limits.feedback_tolerance)
        limits.check_command(goal, start, layout)

        trajectory = control_loop.new_trajectory()
        trajectory.hold(start)
        # these caps bound how fast the target changes, not the contact forces
        release_caps = [np.minimum(caps, cap) for caps, cap in zip(limits.motion_caps(), self.RELEASE_CAPS)]
        trajectory.head_for(goal, release_caps)
        trajectory.compute_trajectory()
        request.trajectory = trajectory
        request.anchor_pose = {arm: control_loop.states[arm].position for arm in layout.arms}

        if np.allclose(start, goal, atol=self.START_EPSILON, rtol=0):
            request.phase = RecoveryPhase.VERIFYING_HOLD
            return
        request.phase = RecoveryPhase.RELEASING_TARGET
        if request.has_automatic_timeout:
            settling_s = control_loop._control_config.settle_timeout_s + self.RELEASE_MARGIN_S
            request.deadline = time.monotonic() + trajectory.duration_s + settling_s

    def _check_drift(self, control_loop: JointControlLoop, request: RecoveryRequest) -> None:
        """Raises if an arm joint moved away from the pose the hold was taken from: a release never chases a drifting
        arm."""
        layout = control_loop.layout
        # the loop's own settings, not a copy: this runs every control period of a recovery
        tolerance = control_loop._robot_config.position_tolerance

        for arm in layout.arms:
            measured_joints = np.array(layout.joint_values(control_loop.states[arm].position))
            drift = abs(measured_joints - np.array(layout.joint_values(request.anchor_pose[arm])))
            if np.any(drift > tolerance):
                index = int(np.argmax(drift))
                name = layout.axis_name(layout.arm_axis_labels(arm, index))
                raise RuntimeError(f"Recovery blocked: {name} moved {drift[index]:.6f} rad during hold transfer")

    def _release(self, control_loop: JointControlLoop, request: RecoveryRequest, is_stable: bool) -> None:
        """Moves the hold one control period along the release, and verifies the new hold once the release ends."""
        if not is_stable:
            raise request.window.stability_error("Recovery blocked: motion during hold transfer")

        layout = control_loop.layout
        sample = request.trajectory.advance()
        control_loop.limits.check_command(sample.position, layout.join(control_loop.targets), layout)
        OperatorPaused.check_cancelled(request.cancellation)
        control_loop.transfer_hold(layout.split(sample.position), control_loop.planned_velocities(sample))
        request.trajectory.accept(sample)

        if sample.is_finished:
            request.phase = RecoveryPhase.VERIFYING_HOLD
            request.window.reset()

    def _verify_hold(
        self, control_loop: JointControlLoop, request: RecoveryRequest, feedback: dict[str, Any], is_stable: bool
    ) -> None:
        """Completes the recovery once the arms track the new hold and are stable."""
        control_loop.check_tracking(feedback)
        if is_stable:
            control_loop.complete_recovery(request)
