from __future__ import annotations

from pear.deploy.station.control.motion.kinematics import BaseKinematics
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.robots.so101.joint.grippers import GripperOutcome
from pear.deploy.station.control.schemas import BriefStatus, ControlSnapshot, DetailedStatus


class SO101MotionResult(MotionResult):
    """An SO-101 action's result, in which a jaw stopped before it closed has not reached its target."""

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus,
        error: BaseException | None,
        kinematics: BaseKinematics | None,
        gripper_outcome: str | None = None,
    ) -> SO101MotionResult:
        """Returns the result of an action from a control snapshot or the robot's status, with the model's state of
        the measured joints when ``kinematics`` are given."""
        return cls.build(
            snapshot,
            status=status,
            error=error,
            to_model_state=None if kinematics is None else kinematics.quantities,
            is_gripper_reached=gripper_outcome != GripperOutcome.STOPPED_BEFORE_CLOSED,
            gripper_outcome=gripper_outcome,
        )
