import threading
import time
from collections.abc import Callable


class ControlLease:
    """The one browser tab that may command the robot, held for as long as its heartbeats arrive."""

    LEASE_S = 3.0  # a holder silent for longer loses control, and its running operation is paused

    def __init__(self, is_busy: Callable[[], bool]) -> None:
        self.is_busy = is_busy  # whether an operation runs; then only the watchdog ends an expired lease
        self.lock = threading.RLock()
        self.owner: str | None = None  # the id of the tab that holds control; None while no tab does
        self.heartbeat_at = 0.0  # when the holder's last heartbeat arrived, on the monotonic clock

    def claim(self, client_id: str) -> bool:
        """Gives control to ``client_id`` when nobody holds it or an idle holder went silent; returns whether
        ``client_id`` holds it."""
        with self.lock:
            if self.owner is None or (self.is_expired() and not self.is_busy()):
                self.owner, self.heartbeat_at = client_id, time.monotonic()
            return self.heartbeat(client_id)

    def heartbeat(self, client_id: str) -> bool:
        """Renews the lease of ``client_id``; returns whether it holds control and its operation has not lost it."""
        with self.lock:
            if self.owner != client_id or (self.is_expired() and self.is_busy()):
                return False
            self.heartbeat_at = time.monotonic()
            return True

    def require(self, client_id: str) -> None:
        """Refuses a command from a tab that does not hold control."""
        if self.owner != client_id or self.is_expired():
            raise ValueError("This tab does not hold control. Reconnect to claim the idle workspace.")

    def lost_owner(self) -> str | None:
        """Returns the holder that went silent while an operation runs, whose operation is paused before control
        moves; None while there is none."""
        with self.lock:
            return self.owner if self.owner is not None and self.is_expired() and self.is_busy() else None

    def release(self, client_id: str) -> None:
        """Frees the control ``client_id`` holds for the next tab that claims it."""
        with self.lock:
            if self.owner == client_id:
                self.owner = None

    def is_expired(self) -> bool:
        """Whether the holder has been silent for longer than the lease."""
        return time.monotonic() - self.heartbeat_at > self.LEASE_S
