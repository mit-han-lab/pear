from abc import ABC, abstractmethod
from typing import Any

from pear.deploy.station.control.joint.arm_state import ArmState


class JointDriver(ABC):
    """The joint hardware a control loop reads and commands: one state and one target for each arm."""

    @abstractmethod
    def read(self) -> dict[str, ArmState]:
        """Returns the latest reading of every arm."""

    @abstractmethod
    def read_for_recovery(self) -> dict[str, ArmState]:
        """Returns a reading of every arm, after verifying that the hardware is healthy enough to hold again."""

    @abstractmethod
    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Commands every arm's target with its planned velocity; without velocities the targets are held."""

    def refresh_hold(self, targets: dict[str, tuple[float, ...]]) -> None:  # noqa: B027  (a default, not abstract)
        """Renews what the driver adds to the held targets, once every control period of a HOLD (by default
        nothing)."""

    def telemetry(self) -> dict[str, Any]:
        """Returns what the driver adds to the motion of a snapshot (by default nothing)."""
        return {}

    @abstractmethod
    def close(self) -> None:
        """Releases the hardware."""
