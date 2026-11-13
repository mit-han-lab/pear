import copy
from dataclasses import dataclass, field
from typing import Any

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.schemas import ArmFeedback


@dataclass(frozen=True)
class SO101ArmState(ArmState):
    """The SO-101 arm's reading, whose velocity is a finite difference of consecutive reads, with its servos' status."""

    servo_status: dict[str, Any] = field(default_factory=dict)  # the latest low-rate poll of the servos' status

    def feedback(self, now: float) -> ArmFeedback:
        """Returns this reading as a snapshot's feedback at the monotonic time ``now``, with the servos' status."""
        return {**super().feedback(now), "servo": copy.deepcopy(self.servo_status)}


class ServoFault(RuntimeError):
    """Raised when a servo reported an error or lost torque."""
