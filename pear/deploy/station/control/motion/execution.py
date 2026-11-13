from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.operations import GripperOperation
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.motion.actions import JointPositions
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.schemas import ControlSnapshot, MotionEndReason, Observation


class Execution(ABC):
    """The action level a robot's class is built on: how a chunk of joint targets and a prepared action are carried
    out under the action's guard, from the three steps of its session that the robot gives."""

    def run_sequence(
        self, action: JointPositions, observation: Observation, *, guard: ActionGuard, deadline: float | None
    ) -> ControlSnapshot:
        """Streams the chunk of joint targets ``action`` from the hold of ``observation``, waits for its HOLD until
        ``deadline`` and returns the snapshot then."""
        control_loop = guard.control_loop
        self.submit_when_settled(
            guard,
            control_loop.start_sequence,
            action.positions,
            rate_hz=action.rate_hz,
            observation=observation,
            deadline=deadline,
        )

        duration_s = len(action.positions) / action.rate_hz
        snapshot = control_loop.wait_for_hold(
            self.remaining_s(deadline, duration_s + control_loop.SEQUENCE_MARGIN_S), guard=guard
        )
        if snapshot["motion"]["end_reason"] != MotionEndReason.FINISHED:
            raise RuntimeError(f"Sequence interrupted: {snapshot['motion']['end_reason']}")

        return snapshot

    def run_prepared(
        self, observation: Observation, prepared_action: PreparedAction, *, guard: ActionGuard, deadline: float | None
    ) -> dict[str, Any] | None:
        """Runs the arm move of ``prepared_action`` from the hold of ``observation``, then what it asks of the
        grippers, all until ``deadline``, and returns the grippers' report (None when it asks nothing of them)."""
        control_loop, move = guard.control_loop, prepared_action.move
        if move is not None:
            self.submit_when_settled(guard, control_loop.dispatch_joint_move, move, observation, deadline=deadline)
            margin_s = control_loop.control_config.settle_timeout_s + control_loop.MOVE_MARGIN_S
            control_loop.wait_for_hold(self.remaining_s(deadline, move.duration_s + margin_s), guard=guard)
            if control_loop.snapshot()["motion"]["end_reason"] != MotionEndReason.FINISHED:
                raise RuntimeError("Cartesian action was interrupted")

        if prepared_action.gripper_request is None:
            return None

        guard.check()
        observation_after_move = self.observe()
        guard.check()

        return self.operate_grippers(
            prepared_action.gripper_request, observation=observation_after_move, guard=guard, deadline=deadline
        )

    def operate_grippers(
        self, gripper_request: Any, *, observation: Observation, guard: ActionGuard, deadline: float | None
    ) -> dict[str, Any]:
        """Moves the grippers as ``gripper_request`` asks while the arms hold ``observation``, until ``deadline``,
        and returns the report of the operation."""
        control_loop = guard.control_loop
        guard.check()
        start_time = time.monotonic()

        try:
            timeout_s = self.remaining_s(deadline, control_loop.GRIPPER_TIMEOUT_S)
            operation = self.submit_grippers(guard, observation, gripper_request)
            control_loop.wait_for_hold(timeout_s, guard=guard)
        except BaseException:
            control_loop.stop()
            raise

        return {
            "kind": "gripper",
            "completed": True,
            **operation.report(),
            "elapsed_seconds": time.monotonic() - start_time,
        }

    def submit_when_settled(
        self,
        guard: ActionGuard,
        control_loop_method: Callable[..., bool],
        *args: Any,
        deadline: float | None,
        **kwargs: Any,
    ) -> None:
        """Submits a method of the control loop through ``guard`` until it starts its operation, which it does not
        while the hold has not settled; gives up at ``deadline`` or once the settle timeout has passed."""
        control_config = guard.control_loop.control_config

        SnapshotCheck.poll(
            lambda: (guard.submit(control_loop_method, *args, **kwargs), None),
            control_config.control_period_s,
            self.remaining_s(deadline, control_config.settle_timeout_s),
            lambda _: TimeoutError("No stable window before dispatch; no new command sent"),
        )

    @abstractmethod
    def observe(self) -> Observation:
        """Returns the measured joints, the cameras' frames and the control snapshot as one observation of the
        session."""

    @abstractmethod
    def wait_for_settled_hold(
        self, observation: Observation | None = None, *, deadline: float | None = None, guard: ActionGuard
    ) -> ControlSnapshot:
        """Waits under ``guard``, until ``deadline``, for a settled HOLD, of the operation of ``observation`` if one
        is given, and returns its snapshot."""

    @abstractmethod
    def submit_grippers(self, guard: ActionGuard, observation: Observation, gripper_request: Any) -> GripperOperation:
        """Starts the gripper operation for ``gripper_request`` under ``guard`` while the arms hold ``observation``,
        settled, and returns it."""

    @staticmethod
    def remaining_s(deadline: float | None, limit_s: float) -> float:
        """Returns how long is left until ``deadline``, at most ``limit_s``, and raises once it has passed."""
        if deadline is None:
            return limit_s

        remaining_s = min(limit_s, deadline - time.monotonic())
        if remaining_s <= 0:
            raise TimeoutError("Action total timeout exceeded")

        return remaining_s
