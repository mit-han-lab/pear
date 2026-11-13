import math
from dataclasses import dataclass
from typing import Any

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.layout import JointLayout


@dataclass(frozen=True, kw_only=True)
class YAMArmState(ArmState):
    """One YAM arm's reading, with the SDK's view of its force-limited gripper."""

    effort: tuple[float, ...] | None = None
    sdk_gripper_target: float | None = None  # the SDK's force-limited command, as a normalized opening
    sdk_gripper_force_limited: bool | None = None
    sdk_gripper_force_limit_n: float | None = None
    sdk_gripper_target_effort_nm: float | None = None  # the SDK's holding torque, its friction included

    def check(self, layout: JointLayout) -> None:
        """Raises if this reading is malformed, a force-limited gripper without the SDK's telemetry included."""
        super().check(layout)

        if self.effort is not None:
            layout.arm_vector(self.effort)
        if self.sdk_gripper_force_limited:
            target, torque = self.sdk_gripper_target, self.sdk_gripper_target_effort_nm
            if (
                self.effort is None
                or target is None
                or not math.isfinite(target)
                or not 0 <= target <= 1
                or torque is None
                or not math.isfinite(torque)
                or torque == 0
            ):
                raise ValueError("Invalid SDK gripper force-limiting feedback")

    def feedback(self, now: float) -> dict[str, Any]:
        """Returns this reading as a snapshot's feedback at the monotonic time ``now``, with the SDK's gripper
        telemetry."""
        return {
            **super().feedback(now),
            "sdk_gripper_force_limited": self.sdk_gripper_force_limited,
            "sdk_gripper_target": self.sdk_gripper_target,
            "sdk_gripper_force_limit_n": self.sdk_gripper_force_limit_n,
            "sdk_gripper_target_effort_nm": self.sdk_gripper_target_effort_nm,
            **({"effort": self.effort} if self.effort is not None else {}),
        }
