from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pear.deploy.station.control.motion.kinematics import BaseKinematics
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.robots.yam.joint.grippers import GripperOutcome
from pear.deploy.station.control.schemas import BriefStatus, ControlSnapshot, DetailedStatus


@dataclass(frozen=True)
class YAMMotionResult(MotionResult):
    """A YAM action's result, with each gripper's outcome."""

    gripper_outcomes: dict[str, str] = field(default_factory=dict)  # the outcome of each gripper that moved, by arm

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus,
        error: BaseException | None,
        kinematics: BaseKinematics | None,
        gripper_outcomes: dict[str, str] | None,
    ) -> YAMMotionResult:
        """Returns the result of an action from a control snapshot; a gripper that resistance stopped is not
        reached."""
        outcomes = dict(gripper_outcomes or {})

        return cls.build(
            snapshot,
            status=status,
            error=error,
            to_model_state=None if kinematics is None else kinematics.quantities,
            is_gripper_reached=GripperOutcome.RESISTANCE_DETECTED not in outcomes.values(),
            gripper_outcomes=outcomes,
            gripper_outcome=next(iter(outcomes.values())) if len(outcomes) == 1 else None,
        )

    def to_record(self) -> dict[str, Any]:
        """Returns the result as a task's records keep it, with each gripper's outcome before the error."""
        record = super().to_record()
        error = record.pop("error")

        return {**record, "gripper_outcomes": self.gripper_outcomes, "error": error}
