import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

import aiohttp

from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, SessionLost, TurnProgress
from pear.deploy.station.agent_loop.clients.metrics import Performance, RequestStatus
from pear.deploy.station.agent_loop.clients.schemas import (
    EventHandler,
    HealthStatus,
    ModelEvent,
    ModelHealth,
    ModelInput,
    ModelOutput,
    ModelPhase,
    ReasoningKind,
    Usage,
)
from pear.deploy.station.exceptions import OperatorPaused


class _TurnError(ValueError):
    """Raised for a turn the model failed while its session is confirmed alive."""


class ResultStatus(StrEnum):
    """How the session service says a turn ended, each by the word of the turn's result."""

    COMPLETED = "completed"  # it ended with the model's answer
    INTERRUPTED = "interrupted"  # an interrupt ended it
    LOST = "lost"  # its session is gone


@dataclass(kw_only=True)
class ClaudeCodeProgress(TurnProgress):
    """One turn of the session while its request runs: its identity and the compactions the service reports of it."""

    previous_compaction_count: int  # the episode's compactions before this turn
    id: str = field(default_factory=lambda: str(uuid4()))  # what the service knows the turn by
    session_created: asyncio.Event = field(default_factory=asyncio.Event)  # set once the service holds the session
    compaction_count: int = 0  # the most compactions the service has reported for this turn
    is_terminal_emitted: bool = False  # whether the turn's last performance report went out


class ClaudeCodeClient(ModelClient):
    """A client of a Claude Code session service, which keeps the conversation."""

    LABEL = "Claude Code"  # names the client in its messages
    INTERRUPTED_MESSAGE = "Claude Code turn interrupted"  # what a stopped turn is refused with
    TIMED_OUT_MESSAGE = "Claude Code timed out; session stopped safely"  # what a turn past its time is refused with
    PROVIDER = "claude_code"  # names the client in its performance reports
    CHECK_TIMEOUT_S = 3  # of the health request and of connecting to the service
    EVENT_LIMIT_BYTES = 32 * 1024 * 1024  # of one streamed event, which holds a whole reasoning and result
    INTERRUPT_MARGIN_S = 1  # of the cleanup timeout kept back for what follows the interrupt
    SERVICE_NAME = "pear-station-claude-code"  # what the service calls itself in its health answer
    PROTOCOL = 1  # the version of the service's protocol the client speaks
    RESULT_EVENT = "result"  # the service's last event of a turn, which holds the answer
    # the service's events that are passed on as they arrive
    FORWARDED_EVENTS = (ModelEvent.PHASE, ModelEvent.REASONING_DELTA, ModelEvent.ANSWER_DELTA, ModelEvent.NOTICE)
    REFERENCES_READY = "ready"  # the service's word for task images its context holds
    NO_COMPACTION = "none"  # the service's word for a turn without a compaction
    CONTEXT_ERROR_WORD = "context"  # what a failed turn's error holds when the conversation outgrew its context
    # what a turn is refused with once a compaction dropped the task's images
    COMPACTED_IMAGES_MESSAGE = "Image context was compacted; clear the task to restart"

    def __init__(self, system_prompt: str, *, timeout_s: float, base_url: str, model: str) -> None:
        if type(model) is not str or not model:
            raise ValueError("model must be the name the service has the model under, a non-empty string")

        super().__init__(system_prompt, timeout_s=timeout_s)
        self.base_url = self._service_url(base_url)
        self.model = model

        self.session_id = str(uuid4())
        self.session_url = self.base_url + "/sessions/" + self.session_id

        self._is_session_created = self._was_creation_attempted = False
        self._compaction_count = 0  # the episode's compactions so far
        self._references_sent = False  # whether a turn has carried the task's images
        self._supports_references: bool | None = None  # whether the service takes task images; None before a check
        self._supports_reference_recovery = False  # whether the service restores task images after a compaction
        self._reference_count = 0  # the task images the service was sent
        self._has_references = False  # whether the task has images
        self._image_context_blocked = False  # whether the task's images may be gone from the context

    def check(self) -> ModelHealth:
        """Returns the health of a service that has the model; raises when the service or the model is unavailable."""
        return asyncio.run(self._health())

    @classmethod
    def supports_reference_images(cls, health: ModelHealth) -> bool:
        """Returns whether the service's last connection check said it takes a task's reference images."""
        return health.get("reference_images") is True

    async def _health(self) -> ModelHealth:
        """Returns the health the service reports of itself and the model, and remembers what it says it does with a
        task's images; raises when the service or the model is unavailable."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.CHECK_TIMEOUT_S)) as http_session:
            async with http_session.get(self.base_url + "/health") as response:
                response.raise_for_status()
                health_answer = await response.json()

        if (
            health_answer.get("service") != self.SERVICE_NAME
            or health_answer.get("protocol") != self.PROTOCOL
            or health_answer.get("model") != self.model
            or health_answer.get("status") != HealthStatus.OK
        ):
            raise ValueError("Claude Code service or upstream model is unavailable")

        capabilities = health_answer.get("capabilities", {})
        self._supports_references = capabilities.get("reference_images") is True
        self._supports_reference_recovery = capabilities.get("reference_image_recovery") is True

        return {
            "status": HealthStatus.OK.value,
            "message": "Claude Code session service connected",
            "reference_images": self._supports_references,
        }

    def _check_usable(self) -> None:
        """Raises once the task's images may be gone from the context."""
        if self._image_context_blocked:
            raise ContextLimit(self.COMPACTED_IMAGES_MESSAGE)

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        """Returns the model's answer to ``model_input`` as one turn of the service's session, which is lost when a
        turn leaves its state in doubt."""
        references = model_input.reference_images if not self._references_sent else ()
        if references:
            if self._supports_references is None:
                await asyncio.to_thread(self.check)
            if not self._supports_references:
                raise ValueError("Claude Code service does not support task images; update it first")
            self._has_references = True
            self._reference_count = len(references)

        turn = ClaudeCodeProgress(
            on_event=on_event,
            cancellation=cancellation,
            start_time=time.monotonic(),
            previous_compaction_count=self._compaction_count,
        )
        if self._is_session_created:
            turn.session_created.set()
        payload = {
            "turn_id": turn.id,
            "text": model_input.text,
            "images": [{"role": image.camera_name, "data": image.base64()} for image in model_input.images],
        }
        if references:
            payload["reference_images"] = [{"data": image.base64()} for image in references]

        connect_timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.CHECK_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=connect_timeout) as http_session:
            request_task = asyncio.create_task(self._request(http_session, turn, payload))
            stop_task = asyncio.create_task(self._wait_stopped(cancellation))
            is_stop_confirmed = False

            try:
                finished_tasks, _ = await asyncio.wait(
                    (request_task, stop_task), timeout=self.timeout_s, return_when=asyncio.FIRST_COMPLETED
                )
                if request_task in finished_tasks:
                    return request_task.result()

                # a timeout too needs the service's confirmed interrupt: a dropped connection does not stop the turn
                has_timed_out = not finished_tasks
                await self._interrupt(http_session, turn, request_task)
                is_stop_confirmed = True
                if has_timed_out:
                    raise TimeoutError(self.TIMED_OUT_MESSAGE)
                raise OperatorPaused(self.INTERRUPTED_MESSAGE)
            except (OperatorPaused, _TurnError, TimeoutError, ContextLimit) as exc:
                # only an explicit result of the model, or a timeout whose stop the service confirmed, keeps the session
                if isinstance(exc, TimeoutError) and not is_stop_confirmed:
                    raise self._lost_session("Claude Code cancellation could not be confirmed") from exc
                raise
            except Exception as exc:
                raise self._lost_session(f"Claude Code session state is uncertain: {exc}") from exc
            finally:
                for unfinished_task in (request_task, stop_task):
                    if not unfinished_task.done():
                        unfinished_task.cancel()
                await asyncio.gather(request_task, stop_task, return_exceptions=True)
                if turn.usage and not turn.is_terminal_emitted:
                    # a stream that broke must not leave the saved usage in progress
                    if not turn.usage.get("final"):
                        turn.usage = {**turn.usage, "partial": True, "final": True}
                        on_event(ModelEvent.USAGE, turn.usage)
                    self._emit_performance(
                        turn, RequestStatus.FAILED if self._lost_reason else RequestStatus.INTERRUPTED
                    )

    async def _request(
        self, http_session: aiohttp.ClientSession, turn: ClaudeCodeProgress, payload: dict[str, Any]
    ) -> ModelOutput:
        """Creates the session if the service has none yet, sends the turn and reads its events up to the result."""
        if not self._is_session_created:
            await self._create_session(http_session, turn)
        if self._is_stopped(turn.cancellation):
            raise OperatorPaused("Paused before submitting the observation")

        async with http_session.post(self.session_url + "/turns", json=payload) as response:
            response.raise_for_status()
            self._references_sent = True

            # the events are split here: aiohttp's own line buffer is too small for a whole reasoning or result
            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                if len(buffer) > self.EVENT_LIMIT_BYTES:
                    raise SessionLost("Claude Code event exceeded the size limit")

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    output = self._receive(turn, json.loads(line))
                    if output is not None:
                        return output

        raise SessionLost("Connection ended without a terminal turn result")

    async def _create_session(self, http_session: aiohttp.ClientSession, turn: ClaudeCodeProgress) -> None:
        """Opens this client's session on the service with its system prompt and model."""
        self._was_creation_attempted = True

        session_request = {"session_id": self.session_id, "system": self.system_prompt, "model": self.model}
        async with http_session.post(self.base_url + "/sessions", json=session_request) as response:
            response.raise_for_status()
            session_answer = await response.json()
        if session_answer.get("session_id") != self.session_id:
            raise SessionLost("Session identity changed during creation")

        self._is_session_created = True
        turn.session_created.set()

    def _receive(self, turn: ClaudeCodeProgress, event: dict[str, Any]) -> ModelOutput | None:
        """Takes one event of the turn's stream and returns the turn's answer once its result arrives, None before."""
        if event.get("session_id") != self.session_id or event.get("turn_id") != turn.id:
            raise SessionLost("Unexpected session or turn identity")

        kind, data = event["type"], event["data"]
        if kind == self.RESULT_EVENT:
            return self._result(turn, data)

        if kind == ModelEvent.USAGE:
            self._update_usage(turn, data)
        elif (
            not self._is_stopped(turn.cancellation)
            and kind in self.FORWARDED_EVENTS
            # the service's other phases are its own
            and (kind != ModelEvent.PHASE or data.get("phase") in (ModelPhase.THINKING, ModelPhase.COMPACTING))
        ):
            if kind in (ModelEvent.REASONING_DELTA, ModelEvent.ANSWER_DELTA) and turn.note_output():
                self._emit_performance(turn, RequestStatus.STREAMING)
            turn.on_event(ModelEvent(kind), data)

        return None

    def _result(self, turn: ClaudeCodeProgress, data: dict[str, Any]) -> ModelOutput:
        """Returns the answer of a completed turn; raises for a turn that was interrupted, lost or failed, and for one
        whose task images are gone from the context."""
        status = data.get("status")
        reference_context = data.get("reference_context", {})
        if (
            self._has_references
            and self._supports_reference_recovery
            and status not in (ResultStatus.INTERRUPTED, ResultStatus.LOST)
            and not self._is_stopped(turn.cancellation)
            and (
                reference_context.get("status") != self.REFERENCES_READY
                or reference_context.get("image_count") != self._reference_count
                or not reference_context.get("verified_request_id")
            )
        ):
            self._image_context_blocked = True
        if self._image_context_blocked:
            self._emit_performance(turn, RequestStatus.FAILED)
            raise ContextLimit(
                "Reference image recovery was not confirmed; clear the task to restart"
                if self._supports_reference_recovery
                else self.COMPACTED_IMAGES_MESSAGE
            )

        if status == ResultStatus.INTERRUPTED or self._is_stopped(turn.cancellation):
            self._emit_performance(turn, RequestStatus.INTERRUPTED)
        elif status == ResultStatus.COMPLETED:
            self._emit_performance(turn, RequestStatus.COMPLETED)
        else:
            self._emit_performance(turn, RequestStatus.FAILED)

        if status == ResultStatus.LOST:
            raise SessionLost(data.get("error") or "Native session lost")
        if status == ResultStatus.INTERRUPTED or self._is_stopped(turn.cancellation):
            raise OperatorPaused(self.INTERRUPTED_MESSAGE)
        if status != ResultStatus.COMPLETED:
            if self._has_references and self.CONTEXT_ERROR_WORD in str(data.get("error", "")).lower():
                self._image_context_blocked = True
                raise ContextLimit(data["error"])
            raise _TurnError(data.get("error") or "Claude Code turn failed")

        # the service's own word for how much of the reasoning it gives stays as it is
        reasoning_kind = data.get("reasoning_kind", ReasoningKind.NONE.value)

        return ModelOutput(data["answer"], data.get("reasoning", ""), reasoning_kind, turn.usage)

    def _update_usage(self, turn: ClaudeCodeProgress, data: Usage) -> None:
        """Keeps the service's latest usage report, with the episode's compactions in it, and passes it on while the
        turn runs."""
        compaction = data.get("compaction")
        if isinstance(compaction, dict):
            if (
                self._has_references
                and not self._supports_reference_recovery
                and (compaction.get("status") not in (None, self.NO_COMPACTION) or compaction.get("count", 0) > 0)
            ):
                self._image_context_blocked = True

            count = compaction.get("count")
            if type(count) is int and count >= 0:
                # a count that streams is the turn's total so far: a later, lower one takes no compaction back
                turn.compaction_count = max(turn.compaction_count, count)
            self._compaction_count = turn.previous_compaction_count + turn.compaction_count
            data = {**data, "compaction": {**compaction, "episode_count": self._compaction_count}}

        turn.usage = data
        if not self._is_stopped(turn.cancellation) or data.get("final"):
            turn.on_event(ModelEvent.USAGE, data)
        if not self._is_stopped(turn.cancellation):
            self._emit_performance(turn, RequestStatus.STREAMING)

    async def _interrupt(
        self, http_session: aiohttp.ClientSession, turn: ClaudeCodeProgress, request_task: asyncio.Task
    ) -> None:
        """Asks the service to interrupt the turn and takes the turn's last events; raises TimeoutError when the
        service does not confirm in time."""
        async with asyncio.timeout(self.CLEANUP_TIMEOUT_S - self.INTERRUPT_MARGIN_S):
            # the session's creation may still be completing: neither it nor its observation is sent again
            await turn.session_created.wait()

            async with http_session.post(self.session_url + "/turns/" + turn.id + "/interrupt") as response:
                response.raise_for_status()
                await response.json()

            try:
                await request_task  # the last usage and the result arrive before a resume is allowed
            except OperatorPaused:
                pass

    def _emit_performance(self, turn: ClaudeCodeProgress, status: RequestStatus) -> None:
        """Reports the turn's timings; a status other than streaming makes the report the turn's last."""
        turn.is_terminal_emitted = status != RequestStatus.STREAMING
        now = time.monotonic()
        request_usages = turn.usage.get("requests", [])
        for request_usage in request_usages:
            turn.token_rate.add(request_usage["id"], request_usage.get("output"), now)

        samples = [
            (request_usage.get("output"), request_usage.get("generation_time_ms")) for request_usage in request_usages
        ]
        # only requests that have all ended have a decode interval on the server's clock
        is_decode_known = status != RequestStatus.STREAMING and all(
            request_usage.get("complete") for request_usage in request_usages
        )
        decode_metrics = Performance.decode_rate(samples if is_decode_known else [])

        turn.on_event(
            ModelEvent.PERFORMANCE,
            {
                **Performance.report(self.PROVIDER, turn.start_time, turn.first_output_s, turn.usage, status=status),
                "partial": bool(turn.usage.get("partial")),
                "client_decode_throughput_tps": turn.token_rate.tokens_per_second(),
                **decode_metrics,
            },
        )

    def _release(self) -> None:
        """Has the service end the session, if the client asked it for one."""
        if self._was_creation_attempted:
            asyncio.run(self._delete_session())

    async def _delete_session(self) -> None:
        """Has the service end this client's session."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.CLEANUP_TIMEOUT_S)) as http_session:
            async with http_session.delete(self.session_url) as response:
                response.raise_for_status()
