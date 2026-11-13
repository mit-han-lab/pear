import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from pear.deploy.station.agent_loop.clients.metrics import TokenRate
from pear.deploy.station.agent_loop.clients.schemas import EventHandler, ModelHealth, ModelInput, ModelOutput, Usage
from pear.deploy.station.exceptions import OperatorPaused


class ContextLimit(ValueError):
    """Raised when a conversation's context cannot continue safely."""


class SessionLost(RuntimeError):
    """Raised when a model's conversation cannot be continued safely."""


@dataclass(kw_only=True)
class TurnProgress:
    """One request of a client while it runs: who hears of it, what stops it and what has streamed of it so far."""

    on_event: EventHandler  # takes what the request reports
    cancellation: threading.Event  # set to stop the request
    start_time: float  # when the request began, on the monotonic clock
    first_output_s: float | None = None  # from the start to the first streamed text
    usage: Usage = field(default_factory=dict)  # the latest report of the token counts
    token_rate: TokenRate = field(default_factory=TokenRate)  # the generated tokens a second, on the client's clock

    def note_output(self) -> bool:
        """Notes that text streamed and returns whether it is the request's first."""
        if self.first_output_s is not None:
            return False

        self.first_output_s = time.monotonic() - self.start_time

        return True


class ModelClient(ABC):
    """The base of a model service's client, which takes a turn's text and images and returns the model's answer: one
    request at a time, which a stop ends early, until the client is closed."""

    LABEL = "Model"  # names the client in its messages
    CANCEL_POLL_S = 0.02  # how often a running request looks for a stop
    CLEANUP_TIMEOUT_S = 10.0  # how long close waits for a running request to end
    RELATIVE_PATH_ARGUMENTS: tuple[str, ...] = ()  # arguments that are paths, relative ones to the station's own file

    def __init__(self, system_prompt: str, *, timeout_s: float) -> None:
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s  # how long the model may take to answer a turn

        self._abort_event = threading.Event()  # set by cancel: stops the running request
        self._request_lock = threading.Lock()  # held by the one request that runs
        self._is_closed = False
        self._lost_reason: str | None = None  # why the model's conversation cannot go on; None while it can

    @abstractmethod
    def check(self) -> ModelHealth:
        """Returns what a check of the model's connection found; raises when the model cannot be reached."""

    def infer(self, model_input: ModelInput, *, on_event: EventHandler, cancellation: threading.Event) -> ModelOutput:
        """Returns the model's answer to ``model_input`` and reports the request's progress to ``on_event``; setting
        ``cancellation`` or calling ``cancel`` stops the request."""
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError(f"A {self.LABEL} request is already active")

        try:
            if self._is_closed:
                raise SessionLost(f"{self.LABEL} client is closed")
            if self._lost_reason:
                raise SessionLost(self._lost_reason)
            self._check_usable()
            OperatorPaused.check_cancelled(cancellation)
            self._abort_event.clear()

            return asyncio.run(self._infer_async(model_input, on_event, cancellation))
        finally:
            self._request_lock.release()

    def cancel(self) -> None:
        """Stops the request that runs, if one does."""
        self._abort_event.set()

    def close(self) -> None:
        """Stops the request that runs, waits for it to end and frees what the client holds; raises SessionLost when
        the request does not end in time."""
        self.cancel()
        if not self._request_lock.acquire(timeout=self.CLEANUP_TIMEOUT_S):
            raise SessionLost(f"{self.LABEL} request did not finish cancellation")

        try:
            if not self._is_closed:
                self._release()
                self._is_closed = True
        finally:
            self._request_lock.release()

    @classmethod
    def supports_reference_images(cls, health: ModelHealth) -> bool:
        """Returns whether a model of this client takes a task's reference images, given its last connection check."""
        return True

    def _check_usable(self) -> None:  # noqa: B027  (a client refuses no request by default)
        """Raises when the client can take no request (by default it takes every one)."""

    @abstractmethod
    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        """Returns the model's answer to ``model_input``, as ``infer`` does, inside the request's event loop."""

    async def _race(self, request: Awaitable[Any], cancellation: threading.Event, *, timeout_s: float | None) -> Any:
        """Returns what ``request`` returns; a stop ends it first and is refused, and past ``timeout_s`` it ends with
        TimeoutError."""
        request_task = asyncio.ensure_future(request)
        stop_task = asyncio.create_task(self._wait_stopped(cancellation))

        try:
            async with asyncio.timeout(timeout_s):
                finished_tasks, _ = await asyncio.wait((request_task, stop_task), return_when=asyncio.FIRST_COMPLETED)
                # a complete answer is kept: the agent loop still looks for a stop
                if request_task in finished_tasks and not request_task.cancelled():
                    return request_task.result()
                raise OperatorPaused(f"{self.LABEL} inference interrupted; no partial action is valid")
        finally:
            for task in (request_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(request_task, stop_task, return_exceptions=True)

    async def _wait_stopped(self, cancellation: threading.Event) -> None:
        """Returns once the request is stopped."""
        while not self._is_stopped(cancellation):
            await asyncio.sleep(self.CANCEL_POLL_S)

    def _is_stopped(self, cancellation: threading.Event) -> bool:
        """Returns whether the request is stopped: its cancellation is set or the client was cancelled."""
        return cancellation.is_set() or self._abort_event.is_set()

    def _release(self) -> None:  # noqa: B027  (a client may hold nothing)
        """Frees what the client holds, once no request runs (by default nothing)."""

    def _lost_session(self, reason: str) -> SessionLost:
        """Remembers that the model's conversation is lost for ``reason`` and returns the error that says so."""
        self._lost_reason = reason

        return SessionLost(reason)

    @staticmethod
    def _service_url(base_url: str) -> str:
        """Returns a model service's ``base_url`` without a slash at its end; raises for one that is no plain HTTP(S)
        URL."""
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.hostname
            or parsed_url.username
            or parsed_url.password
        ):
            raise ValueError("Model base_url must be HTTP(S) without embedded credentials")

        return base_url.rstrip("/")
