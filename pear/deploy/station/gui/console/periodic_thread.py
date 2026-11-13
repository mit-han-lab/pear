import threading
import time
from collections.abc import Callable

from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PeriodicThread:
    """A background thread that does a step every period until it is closed: the first step at once or after one
    period, each next one a period after a step's start or a whole period after its end; a step that fails is
    logged with its traceback, and the thread then goes on or ends."""

    MIN_WAIT_S = 0.001  # between two steps paced from their starts, even when one took longer than the period
    JOIN_S = 2.0  # for the step in progress at close

    def __init__(
        self,
        step: Callable[[], object],
        period_s: float,
        thread_name: str,
        *,
        is_waiting_first: bool = False,
        is_paced_from_start: bool = False,
        is_ended_by_failure: bool = False,
    ) -> None:
        self.step = step
        self.period_s = period_s
        self.is_waiting_first = is_waiting_first  # whether the first step waits one period
        self.is_paced_from_start = is_paced_from_start  # whether a period counts from a step's start, not its end
        self.is_ended_by_failure = is_ended_by_failure  # whether a step that fails ends the thread
        self.exit_event = threading.Event()  # set at close
        self.thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Ends the steps and waits up to JOIN_S for the one in progress; the thread may outlive the wait."""
        self.exit_event.set()
        self.thread.join(self.JOIN_S)

    def _run(self) -> None:
        """Does a step every period until the thread is closed, or until a step fails when that ends it."""
        if self.is_waiting_first and self.exit_event.wait(self.period_s):
            return

        while not self.exit_event.is_set():
            start_time = time.monotonic()
            try:
                self.step()
            except Exception:
                logger.exception("A step of %s failed", self.thread.name)
                if self.is_ended_by_failure:
                    return
            if self.is_paced_from_start:
                self.exit_event.wait(max(self.MIN_WAIT_S, self.period_s - (time.monotonic() - start_time)))
            else:
                self.exit_event.wait(self.period_s)
