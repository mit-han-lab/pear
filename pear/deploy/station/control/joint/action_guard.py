from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.schemas import ControlState
from pear.deploy.station.exceptions import OperatorPaused

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.control.joint.control_loop import JointControlLoop


class ActionGuard:
    """Binds every phase of one caller's action to the operation generation its own submissions started."""

    def __init__(self, control_loop: JointControlLoop, cancellation: threading.Event | None = None) -> None:
        self.control_loop = control_loop
        self.cancellation = cancellation  # set by the operator to pause the action

        # a stop, a fault, a close or a competing operation advances the generation, which ends every later phase
        with control_loop.lock:
            self.generation = control_loop.generation
        self.has_dispatched = False  # whether a submission of this action started an operation

    def check(self) -> None:
        """Raises if the action was cancelled, the control loop faulted or closed, or another operation began."""
        control_loop = self.control_loop
        OperatorPaused.check_cancelled(self.cancellation)

        with control_loop.lock:
            if control_loop.status == ControlState.FAULT:
                SnapshotCheck.raise_for_fault(control_loop.snapshot())
            if control_loop.status == ControlState.CLOSED:
                raise RuntimeError(f"Action interrupted: {control_loop.status}: {control_loop.fault}")
            if self.generation != control_loop.generation:
                raise RuntimeError("Action cancelled by stop or another operation")

    def submit(self, control_loop_method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Calls a method of the control loop under its lock if the action is still current, adopts the operation it
        starts and returns what it returns."""
        control_loop = self.control_loop

        with control_loop.lock:
            self.check()
            result = control_loop_method(*args, **kwargs)
            has_advanced = self.generation != control_loop.generation  # the method started an operation or stopped
            if has_advanced:
                control_loop.attach_cancellation(self.cancellation)
            self.has_dispatched |= has_advanced
            self.generation = control_loop.generation

            return result
