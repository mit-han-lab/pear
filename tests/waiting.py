import time
from collections.abc import Callable


class Waiting:
    """A test's wait for what another thread or process does."""

    @staticmethod
    def until(condition: Callable[[], bool], timeout_s: float, poll_s: float) -> None:
        """Returns once ``condition()`` holds, looked at every ``poll_s`` seconds, and fails the test after
        ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        while not condition():
            assert time.monotonic() < deadline, "the condition never held"
            time.sleep(poll_s)
