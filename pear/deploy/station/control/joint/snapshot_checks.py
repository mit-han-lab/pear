from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.stability import StabilityError
from pear.deploy.station.control.schemas import ControlSnapshot, ControlState, Observation

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.control.joint.action_guard import ActionGuard


class SnapshotCheck:
    """The readings and the checks of a control snapshot that every robot applies, whatever its arms, and the wait
    they repeat."""

    @classmethod
    def wait_ready(
        cls,
        read_snapshot: Callable[[], ControlSnapshot],
        *,
        timeout_s: float,
        control_period_s: float,
        guard: ActionGuard,
        make_timeout_error: Callable[[ControlSnapshot], Exception],
        accepts: Callable[[ControlSnapshot], bool] | None = None,
    ) -> ControlSnapshot:
        """Returns the first checked snapshot of ``read_snapshot`` that shows a settled HOLD which ``accepts``
        accepts, never resetting the history; past ``timeout_s`` raises the error made from the last snapshot."""
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("Wait timeout must be finite and nonnegative")

        def read_settled() -> tuple[bool, ControlSnapshot]:
            guard.check()
            snapshot = read_snapshot()
            is_accepted = accepts(snapshot) if accepts is not None else True
            guard.check()
            is_settled_hold = snapshot["status"] == ControlState.HOLD and snapshot["arrival"]["settled"]

            return is_settled_hold and is_accepted, snapshot

        return cls.poll(read_settled, control_period_s, timeout_s, make_timeout_error)

    @staticmethod
    def poll(
        attempt: Callable[[], tuple[bool, Any]],
        control_period_s: float,
        timeout_s: float = math.inf,
        make_timeout_error: Callable[[Any], Exception] | None = None,
    ) -> Any:
        """Returns the result of the first ``attempt`` that is done, trying once every control period; when
        ``timeout_s`` pass first, raises the error made from the last result (by default the wait has no timeout of
        its own: an attempt ends it)."""
        deadline = time.monotonic() + timeout_s

        while True:
            is_done, result = attempt()
            if is_done:
                return result
            if time.monotonic() >= deadline:
                raise make_timeout_error(result)

            time.sleep(control_period_s)

    @classmethod
    def validate_snapshot(
        cls, snapshot: ControlSnapshot, control_config: ControlConfig, *, requires_hold: bool = False
    ) -> ControlSnapshot:
        """Returns ``snapshot`` unless it shows a fault, an unexpected status or feedback too old to trust."""
        cls.raise_for_fault(snapshot)

        allowed_states = (
            (ControlState.HOLD,) if requires_hold else (ControlState.HOLD, ControlState.RUNNING, ControlState.STOPPING)
        )
        if snapshot["status"] not in allowed_states:
            raise RuntimeError(
                f"Station requires valid {'HOLD' if requires_hold else 'feedback'}: "
                f"{snapshot['status']}: {snapshot['fault']}"
            )
        if any(
            not ArmState.is_fresh(feedback["age_seconds"], control_config.state_timeout_s)
            for feedback in snapshot["arms"].values()
        ):
            raise RuntimeError("Stale station feedback")

        return snapshot

    @classmethod
    def validate_reference(cls, snapshot: ControlSnapshot, observation: Observation | None) -> None:
        """Raises unless ``observation`` is valid data of the snapshot's operation, whose targets it then still
        holds."""
        if observation is None:
            return

        if observation["control_snapshot"]["generation"] != snapshot["generation"]:
            raise ValueError("Observation invalidated by another operation")
        reference = np.asarray(observation["joint_positions"])
        if reference.shape != cls.target_vector(snapshot).shape or not np.isfinite(reference).all():
            raise ValueError("Invalid observation reference")

    @staticmethod
    def raise_for_fault(snapshot: ControlSnapshot) -> None:
        """Raises the latched fault of ``snapshot`` with its diagnostics, if it has one."""
        if snapshot["status"] != ControlState.FAULT and not snapshot["fault"]:
            return

        details = snapshot.get("fault_details") or {}
        if details.get("code") == StabilityError.__name__:
            raise StabilityError(details["reason"], details["metrics"])
        raise RuntimeError(f"Station fault: {snapshot['status']}: {snapshot['fault']}")

    @staticmethod
    def check_timeout(timeout_s: float, subject: str) -> None:
        """Raises unless ``timeout_s``, the timeout that ``subject`` names in the error, is finite and positive."""
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"{subject} must be finite and positive")

    @staticmethod
    def state_vector(snapshot: ControlSnapshot) -> np.ndarray:
        """Returns the measured position of every arm in ``snapshot`` as one vector, in the snapshot's arm order,
        which is the robot's."""
        return np.array([value for feedback in snapshot["arms"].values() for value in feedback["position"]])

    @staticmethod
    def target_vector(snapshot: ControlSnapshot) -> np.ndarray:
        """Returns the held command targets in ``snapshot`` as one vector, in the order of its measured arms."""
        return np.array([value for arm in snapshot["arms"] for value in snapshot["targets"][arm]])
