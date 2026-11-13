import threading
import time
from collections.abc import Callable

from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.agent_loop.clients.schemas import HealthStatus, ModelHealth
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HealthChecks:
    """The last connection check of each model, which answers later polls until it is too old."""

    MAX_AGE_S = 30.0  # a check answers the automatic polls for this long before a new client checks again
    # the result shown for a model no check has reached yet
    UNCHECKED_HEALTH: ModelHealth = {"status": HealthStatus.UNCHECKED.value, "message": "Connection not checked"}

    def __init__(self, new_client: Callable[[str], ModelClient]) -> None:
        self.new_client = new_client  # model id -> a client built only to check its connection
        self.lock = threading.Lock()  # held by the one check that runs at a time
        self.results: dict[str, tuple[float, ModelHealth]] = {}  # model id -> (when it was checked, result)

    def get(self, model: str) -> ModelHealth:
        """Returns the last check of ``model``."""
        last_check = self.results.get(model)

        return last_check[1] if last_check else self.UNCHECKED_HEALTH

    def check(self, model: str, *, is_forced: bool = False) -> ModelHealth:
        """Checks ``model`` with a new client unless a recent check answers; a forced one reuses only a later check."""
        asked_at = time.monotonic()

        with self.lock:
            last_check = self.results.get(model)
            if last_check and (
                last_check[0] >= asked_at or not is_forced and asked_at - last_check[0] < self.MAX_AGE_S
            ):
                return last_check[1]

            client = None
            try:
                client = self.new_client(model)
                result = client.check()
            except Exception as exc:
                logger.exception("Model connection failed: %s", exc)
                result = {"status": HealthStatus.OFFLINE.value, "message": str(exc)}
            finally:
                if client:
                    client.close()
            self.results[model] = (time.monotonic(), result)

        return result
