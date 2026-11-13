from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.schemas import BriefStatus, ControlSnapshot, DetailedStatus
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class MotionStatus(StrEnum):
    """How an action ended, by the word its result carries."""

    COMPLETED = "completed"
    REJECTED = "rejected"  # refused before anything was sent to the arms
    FAILED = "failed"


@dataclass(frozen=True)
class MotionResult:
    """The outcome of an action: its status, settling and arrival, the measured pose, the gripper's outcome and the
    error that ended it."""

    status: MotionStatus
    is_settled: bool
    target_reached: bool
    measured_position: list[float] | None  # the measured joint vector; None of a robot that is closed
    tracking_error: list[float] | None  # the measured joints' signed errors against what the action requested
    measured_state: StrictModel | None = None  # what the model reads of the measured joints
    gripper_outcome: str | None = None  # how the one gripper that moved ended, in the robot's word
    error: dict[str, str] | None = None  # the code and the message of what ended the action

    @classmethod
    def build(
        cls,
        snapshot: ControlSnapshot | DetailedStatus | BriefStatus,
        *,
        status: MotionStatus,
        error: BaseException | None,
        to_model_state: Callable[[list[float]], StrictModel] | None,
        is_gripper_reached: bool,
        **result_fields: Any,
    ) -> MotionResult:
        """Returns the result of an action from a control snapshot or a robot's status; ``to_model_state`` turns the
        measured joints into what the model reads, and ``result_fields`` are the fields a robot's result adds."""
        arrival = snapshot.get("arrival", {})
        measured_position = SnapshotCheck.state_vector(snapshot).tolist() if "arms" in snapshot else None

        model_state = None
        if measured_position is not None and to_model_state is not None:
            try:
                model_state = to_model_state(measured_position)
            except (ValueError, RuntimeError):
                logger.exception("The measured joints have no model state")

        return cls(
            status=status,
            is_settled=arrival.get("settled", False),
            target_reached=status == MotionStatus.COMPLETED and arrival.get("reached", False) and is_gripper_reached,
            measured_position=measured_position,
            tracking_error=arrival.get("signed_error"),
            measured_state=model_state,
            error={"code": type(error).__name__, "message": str(error)} if error else None,
            **result_fields,
        )

    def to_record(self) -> dict[str, Any]:
        """Returns the result as a task's records and the page keep it, under their keys and in JSON values."""
        return {
            "status": self.status.value,
            "settled": self.is_settled,
            "reached": self.target_reached,
            "measured_position": self.measured_position,
            "tracking_error": self.tracking_error,
            "measured_state": None if self.measured_state is None else self.measured_state.model_dump(mode="json"),
            "gripper_outcome": self.gripper_outcome,
            "error": self.error,
        }
