from dataclasses import dataclass

from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.schemas import ArmFeedback


@dataclass(frozen=True)
class ArmState:
    """One arm's measured coordinates; a robot may subclass it to carry its own telemetry."""

    CLOCK_SLACK_S = 0.001  # how far a reading's stamp may run ahead of the local clock

    position: tuple[float, ...]
    velocity: tuple[float, ...]
    sampled_monotonic_s: float  # local monotonic time of the reading, not of the caller's read
    sample_id: int | None = None  # tells a repeated reading from a new one at the same time

    def check(self, layout: JointLayout) -> None:
        """Raises if this reading is not one finite arm vector of ``layout``."""
        layout.arm_vector(self.position)
        layout.arm_vector(self.velocity)

    def feedback(self, now: float) -> ArmFeedback:
        """Returns this reading as a snapshot's feedback at the monotonic time ``now``."""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "sampled_monotonic_s": self.sampled_monotonic_s,
            "sample_id": self.sample_id,
            "age_seconds": now - self.sampled_monotonic_s,
        }

    @staticmethod
    def is_fresh(age_s: float, timeout_s: float, slack_s: float = 0.0) -> bool:
        """Whether a reading ``age_s`` old is recent: at most ``timeout_s`` old and ``slack_s`` ahead of the clock."""
        return -slack_s <= age_s <= timeout_s
