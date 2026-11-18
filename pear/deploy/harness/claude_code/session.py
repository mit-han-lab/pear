import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, StreamEvent, SystemMessage
from omegaconf import DictConfig

from pear.deploy.station.agent_loop.clients.claude_code import ClaudeCodeClient, ResultStatus
from pear.deploy.station.agent_loop.clients.schemas import ModelEvent, ModelPhase, ReasoningKind, Usage
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only: the proxy imports the session
    from pear.deploy.harness.claude_code.proxy import UpstreamProxy

logger = get_logger(__name__)


class SessionLost(RuntimeError):
    """Raised when the state of a session's conversation can no longer be confirmed."""


class TurnUsage:
    """The token counts of one turn's model requests, as the turn's usage events report them to the client."""

    NO_COMPACTION = "none"  # the compaction status of a turn without one
    MEASURED = "measured"  # the status of a context whose tokens the model server counted
    PENDING = "pending"  # the status of a context no request has measured since the last compaction

    def __init__(
        self,
        context_limit: int,
        context_input: int | None,
        compacted_input: int = 0,
        summarized_input: int | None = None,
    ) -> None:
        self.context_limit = context_limit  # the tokens the model's context holds
        # the tokens of the latest main request's prompt, the previous turn's at first; None until a request measures
        self.context_input = context_input
        # the conversation's tokens the compactions so far took out of the context, which the conversation still counts
        self.compacted_input = compacted_input
        # the prompt of a compaction's summary request, until the main request after it measures what it left
        self.summarized_input = summarized_input
        self.summary_input: int | None = None  # the prompt of the turn's latest summary request
        self.compaction = {"status": self.NO_COMPACTION, "count": 0}  # the turn's compactions
        self.requests: dict[str, dict[str, Any]] = {}  # by request id: its input, output, decode time and completion

    def carried(self) -> tuple[int | None, int, int | None]:
        """Returns what the next turn's usage starts from: the context's tokens and the compactions' account."""
        return self.context_input, self.compacted_input, self.summarized_input

    def measure_context(self, prompt_tokens: int) -> None:
        """Takes a main request's prompt as the context's tokens: after a compaction, what it left of the summarized
        conversation, the rest of which the compaction took out."""
        if self.summarized_input is not None:
            self.compacted_input += self.summarized_input - prompt_tokens
            self.summarized_input = None
        self.context_input = prompt_tokens

    def compact(self) -> None:
        """Takes a completed compaction: the next main request measures what it left of the conversation its
        summary request held."""
        self.summarized_input = self.summary_input
        self.context_input = None

    def open_request(self, request_id: str) -> None:
        """Begins the counts of a request the model server accepted."""
        self.requests[request_id] = {"input": None, "output": None, "generation_time_ms": None, "complete": False}

    def update(self, request_id: str, **counts: Any) -> None:
        """Takes what the stream or the model server's metrics report of a request."""
        self.requests[request_id].update(counts)

    def drop_request(self, request_id: str) -> None:
        """Forgets a request the model server ended before any output."""
        del self.requests[request_id]

    def snapshot(self, *, is_final: bool) -> Usage:
        """Returns the turn's counts: the sums over its requests, the context and the compactions."""
        rows = list(self.requests.values())
        inputs = [row["input"] for row in rows if row["input"] is not None]
        outputs = [row["output"] for row in rows if row["output"] is not None]

        return {
            "input": sum(inputs) if inputs else None,
            "output": sum(outputs) if outputs else None,
            "final": is_final,
            # lower bounds: a request that did not end, or one whose counts the server did not report
            "partial": not all(row["complete"] for row in rows) or len(inputs) < len(rows) or len(outputs) < len(rows),
            "context": {
                "input": self.context_input,
                # the conversation's tokens: the context's, with those the compactions took out of it
                "conversation_input": None if self.context_input is None else self.context_input + self.compacted_input,
                "context_limit": self.context_limit,
                "status": self.PENDING if self.context_input is None else self.MEASURED,
            },
            "compaction": dict(self.compaction),
            "requests": [
                {"id": request_id, **{key: row[key] for key in ("output", "generation_time_ms", "complete")}}
                for request_id, row in self.requests.items()
            ],
        }


@dataclass
class Turn:
    """One turn of a session: its observation, the events the client reads of it, and what keeps its answer from the
    robot."""

    id: str
    content: list[dict[str, Any]]  # the blocks of the turn's user message, as Claude Code is given it
    images: dict[str, dict[str, Any]]  # the turn's camera images in their order, by the placeholders in its message
    usage: TurnUsage
    events: asyncio.Queue = field(default_factory=asyncio.Queue)  # what the turn's stream sends the client
    cancellation: asyncio.Event = field(default_factory=asyncio.Event)  # set to interrupt the turn
    done: asyncio.Event = field(default_factory=asyncio.Event)  # set once the turn's result is out
    refusal: str | None = None  # why the turn's answer must not reach the robot; None while it may
    reasoning: str = ""  # the model's reasoning streamed so far
    last_usage_time: float = 0.0  # when the turn's usage was last reported, on the monotonic clock


class ClaudeCodeSession:
    """One task's conversation: a Claude Code process without tools, which one asyncio task owns and gives one turn
    at a time, its model requests going through the upstream proxy."""

    # of one turn's request with its images in base64, and of one message between the CLI and the SDK
    MESSAGE_LIMIT_BYTES = 1 << 28
    INTERRUPT_TIMEOUT_S = 8  # for an interrupted turn's result: within the client's cleanup timeout
    CLOSE_TIMEOUT_S = 9  # for a closed session's process to end: within the client's cleanup timeout
    USAGE_INTERVAL_S = 0.25  # between two usage reports of a turn while it streams
    # the CLI's own limit of one model request, past any turn's: the station's interrupt ends a turn
    REQUEST_TIMEOUT_MS = 3_600_000
    PLACEHOLDER_KEY = "unused"  # the API key Claude Code sends: the proxy asks for none
    # what Claude Code holds of a camera image, which the proxy sends the model in its place: the CLI keeps no image of
    # its own, whose number per request it limits by dropping the oldest
    IMAGE_PLACEHOLDER = "[pear image {key}]"
    CLI_FLAGS = {"bare": None, "no-session-persistence": None}  # no hooks, plugins or memory; no transcript on disk
    # no traffic but the model requests: no update, telemetry or error report
    CLI_ENVIRONMENT = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
    }
    SUCCESS = "success"  # the subtype of a result the CLI completed
    FAILED = "failed"  # the status of a turn without an answer the robot may execute
    STATUS_MESSAGE = "status"  # the subtype of the CLI's messages of what it does
    COMPACTING = "compacting"  # the status of the CLI while it compacts
    COMPACT_BOUNDARY = "compact_boundary"  # the subtype of the CLI's message that a compaction replaced the history
    # what the operator reads in the turn of a compaction, with the context's tokens before it
    COMPACTED_NOTICE = "Context compacted: the conversation of {tokens:,} tokens was replaced by the model's summary"
    ABORTED = "aborted"  # the error of a compaction that an interrupt ended
    # the compaction statuses the client reads
    RUNNING, COMPLETED, INTERRUPTED = "running", "completed", "interrupted"

    def __init__(
        self, session_id: str, system: str, config: DictConfig, proxy: "UpstreamProxy", context_limit: int
    ) -> None:
        self.id = session_id
        self.system = system  # the station's system prompt, the only one the model server is sent
        self.config = config
        self.proxy = proxy
        self.context_limit = context_limit  # the tokens the model server's context holds
        # the CLI's working directory and configuration, and the records of the model requests
        self.directory = Path(config.run_dir) / session_id
        self.references: list[dict[str, Any]] = []  # the task's images, which the proxy puts at the head of a request
        self.reference_request_id: str | None = None  # the latest main request the model server was sent with them
        self.images: dict[str, dict[str, Any]] = {}  # the camera images of Claude Code's history, by placeholder
        self.turn: Turn | None = None  # the turn that runs
        self.turn_ids: set[str] = set()  # of every turn submitted: none is run twice
        self.cancelled_turn_ids: set[str] = set()  # of the turns interrupted, before they started too
        # what the next turn's usage starts from: the context's tokens and the compactions' account, the last turn's
        self.context: tuple[int | None, int, int | None] = (None, 0, None)
        self.lost_reason: str | None = None  # why the conversation cannot go on; None while it can
        self.is_closing = False
        self.commands: asyncio.Queue[Turn | None] = asyncio.Queue()  # the turns to run; None ends the session
        self.started = asyncio.get_running_loop().create_future()  # done once the CLI runs with the model's context
        self.owner = asyncio.create_task(self._own(), name=f"claude-code-session-{session_id}")

    def submit(
        self,
        turn_id: str,
        content: list[dict[str, Any]],
        images: dict[str, dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> Turn:
        """Queues a turn and returns it; refuses a reused id, a second running turn and task images after the first
        turn."""
        if self.lost_reason or self.is_closing:
            raise SessionLost(self.lost_reason or "The session is closed")
        if turn_id in self.turn_ids:
            raise ValueError("Turn ID already used: a turn is never run twice")
        if self.turn:
            raise ValueError("A turn is already running")
        if references and self.turn_ids:
            raise ValueError("The task's images come with the first turn only")

        if references:
            self.references = references
        self.images |= images
        turn = Turn(turn_id, content, images, TurnUsage(self.context_limit, *self.context))
        if turn_id in self.cancelled_turn_ids:
            turn.cancellation.set()
        self.turn_ids.add(turn_id)
        self.turn = turn
        self.commands.put_nowait(turn)

        return turn

    async def interrupt(self, turn_id: str) -> None:
        """Interrupts the turn ``turn_id``, before it starts too, and returns once its result is out; raises
        SessionLost when the conversation's state is lost."""
        self.cancelled_turn_ids.add(turn_id)
        turn = self.turn
        if turn and turn.id == turn_id:
            turn.cancellation.set()
            await turn.done.wait()

        if self.lost_reason:
            raise SessionLost(self.lost_reason)

    async def close(self) -> None:
        """Ends the session: its running turn, then its process."""
        self.is_closing = True
        if self.turn:
            self.turn.cancellation.set()
        self.commands.put_nowait(None)

        try:
            await asyncio.wait_for(asyncio.shield(self.owner), self.CLOSE_TIMEOUT_S)
        except TimeoutError:
            self.owner.cancel()
            await asyncio.gather(self.owner, return_exceptions=True)

    def report_usage(self, turn: Turn, *, is_final: bool = False, is_forced: bool = False) -> None:
        """Sends the turn's usage: at most every USAGE_INTERVAL_S while it streams, at once when forced or final."""
        now = time.monotonic()
        if is_final or is_forced or now - turn.last_usage_time >= self.USAGE_INTERVAL_S:
            turn.last_usage_time = now
            self._emit(turn, ModelEvent.USAGE, turn.usage.snapshot(is_final=is_final))

    async def _own(self) -> None:
        """Runs the CLI for the session's life and its turns one at a time; whatever ends the CLI loses the
        conversation."""
        base_url = self.proxy.open(self)

        try:
            async with ClaudeSDKClient(options=self._options(base_url)) as client:
                # the CLI plans its compactions around the context it is told: it must be the model server's
                context = await client.get_context_usage()
                if context["rawMaxTokens"] != self.context_limit or not context["isAutoCompactEnabled"]:
                    raise SessionLost(
                        f"Claude Code plans a context of {context['rawMaxTokens']} tokens with autocompact "
                        f"{context['isAutoCompactEnabled']}; the model server's holds {self.context_limit}"
                    )
                self.started.set_result(None)

                while (turn := await self.commands.get()) is not None:
                    if turn.cancellation.is_set():
                        self._finish(turn, None)
                    else:
                        await self._run(client, turn)
        except BaseException as error:  # noqa: BLE001  (whatever ends the CLI ends the conversation)
            self.lost_reason = self.lost_reason or f"Claude Code session ended: {type(error).__name__}: {error}"
            if not self.started.done():
                self.started.set_exception(SessionLost(self.lost_reason))
            if self.turn:
                self._finish(self.turn, None)
        finally:
            await self.proxy.close(self.id)

    def _options(self, base_url: str) -> ClaudeAgentOptions:
        """Returns the CLI's options: the station's system prompt, no tool, server or setting of its own, prompts
        delivered as written, and the model reached through the proxy with the model server's context."""
        return ClaudeAgentOptions(
            system_prompt=self.system,
            tools=[],
            mcp_servers={},
            strict_mcp_config=True,
            setting_sources=[],
            verbatim_prompts=True,
            include_partial_messages=True,
            model=self.config.model,
            cwd=self.directory,
            max_buffer_size=self.MESSAGE_LIMIT_BYTES,
            extra_args=dict(self.CLI_FLAGS),
            env={
                **self.CLI_ENVIRONMENT,
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": self.PLACEHOLDER_KEY,
                "CLAUDE_CONFIG_DIR": str(self.directory / "config"),
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(self.context_limit),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.config.cli_max_output_tokens),
                "API_TIMEOUT_MS": str(self.REQUEST_TIMEOUT_MS),
            },
            stderr=lambda line: logger.warning("Claude Code session %s: %s", self.id, line),
        )

    async def _run(self, client: ClaudeSDKClient, turn: Turn) -> None:
        """Runs one turn to its result; an interrupt whose end the CLI does not confirm in time loses the session."""
        await client.query(self._prompt(turn))
        reader = asyncio.create_task(self._receive(client, turn))
        stop = asyncio.create_task(turn.cancellation.wait())

        try:
            await asyncio.wait((reader, stop), return_when=asyncio.FIRST_COMPLETED)
            if reader.done():
                # the result waits for the last counts of the turn's requests, which follow their streams
                await self.proxy.settle(self.id)
            else:
                async with asyncio.timeout(self.INTERRUPT_TIMEOUT_S):
                    await client.interrupt()
                    await self.proxy.cancel(self.id)
                    await reader
            result = reader.result()
        except BaseException as error:
            self.lost_reason = f"Claude Code did not confirm the end of a turn: {type(error).__name__}: {error}"
            raise
        finally:
            for task in (reader, stop):
                task.cancel()
            await asyncio.gather(reader, stop, return_exceptions=True)

        self._finish(turn, result)

    @staticmethod
    async def _prompt(turn: Turn) -> AsyncIterator[dict[str, Any]]:
        """Yields the turn's user message as the SDK streams it to the CLI."""
        yield {"type": "user", "message": {"role": "user", "content": turn.content}, "parent_tool_use_id": None}

    async def _receive(self, client: ClaudeSDKClient, turn: Turn) -> ResultMessage:
        """Passes the CLI's messages of a turn on as the turn's events and returns its result."""
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                return message
            if isinstance(message, StreamEvent):
                self._on_stream(turn, message.event)
            elif isinstance(message, SystemMessage):
                self._on_system(turn, message)

        raise SessionLost("Claude Code ended a turn without its result")

    def _on_stream(self, turn: Turn, event: dict[str, Any]) -> None:
        """Passes a streamed event of the model's answer on: its phase, its reasoning and its text; an interrupted
        turn passes nothing on."""
        if turn.cancellation.is_set():
            return

        delta = event.get("delta") or {}
        if event.get("type") == "message_start":
            self._emit(turn, ModelEvent.PHASE, {"phase": ModelPhase.THINKING.value})
        if delta.get("thinking"):
            turn.reasoning += delta["thinking"]
            self._emit(
                turn, ModelEvent.REASONING_DELTA, {"text": delta["thinking"], "reasoning_kind": ReasoningKind.FULL}
            )
        if delta.get("text"):
            self._emit(turn, ModelEvent.ANSWER_DELTA, {"text": delta["text"]})

    def _on_system(self, turn: Turn, message: SystemMessage) -> None:
        """Counts the compactions the CLI reports, which the turn's usage carries, and tells the operator of each one
        that completes."""
        compaction = turn.usage.compaction
        if message.subtype == self.STATUS_MESSAGE and message.data.get("status") == self.COMPACTING:
            # the CLI repeats the status while it compacts, every 30 s: the phase begins at the first alone, whose time
            # the page counts from
            is_beginning = compaction["status"] != self.RUNNING
            compaction["status"] = self.RUNNING
            if is_beginning and not turn.cancellation.is_set():
                self._emit(turn, ModelEvent.PHASE, {"phase": ModelPhase.COMPACTING.value})
        elif message.subtype == self.STATUS_MESSAGE and message.data.get("compact_result") == self.FAILED:
            compaction["status"] = (
                self.INTERRUPTED if message.data.get("compact_error") == self.ABORTED else self.FAILED
            )
        elif message.subtype == self.COMPACT_BOUNDARY:
            compaction.update(status=self.COMPLETED, count=compaction["count"] + 1)
            turn.usage.compact()
            if turn.usage.summarized_input is not None and not turn.cancellation.is_set():
                notice = self.COMPACTED_NOTICE.format(tokens=turn.usage.summarized_input)
                self._emit(turn, ModelEvent.NOTICE, {"message": notice})
        else:
            return

        self.report_usage(turn, is_forced=True)

    def _finish(self, turn: Turn, result: ResultMessage | None) -> None:
        """Sends the turn's last usage and its result, whose answer only a completed turn has."""
        status, answer, error = self.FAILED, "", None
        if self.lost_reason:
            status, error = ResultStatus.LOST, self.lost_reason
        elif turn.cancellation.is_set():
            status = ResultStatus.INTERRUPTED
        elif turn.refusal:
            error = turn.refusal
        elif result.subtype == self.SUCCESS and not result.is_error:
            status, answer = ResultStatus.COMPLETED, result.result or ""
        else:
            error = result.result or "; ".join(result.errors or ()) or result.subtype
        if turn.usage.compaction["status"] == self.RUNNING:
            is_stopped = status in (ResultStatus.INTERRUPTED, ResultStatus.LOST)
            turn.usage.compaction["status"] = self.INTERRUPTED if is_stopped else self.FAILED

        self.context = turn.usage.carried()
        self.report_usage(turn, is_final=True)
        self._emit(
            turn,
            ClaudeCodeClient.RESULT_EVENT,
            {
                "status": status,
                "answer": answer,
                "reasoning": turn.reasoning,
                "reasoning_kind": ReasoningKind.FULL if turn.reasoning else ReasoningKind.NONE,
                "error": error,
                "reference_context": {
                    "status": ClaudeCodeClient.REFERENCES_READY,
                    "image_count": len(self.references),
                    "verified_request_id": self.reference_request_id,
                },
            },
        )
        turn.done.set()
        self.turn = None

    def _emit(self, turn: Turn, kind: str, data: dict[str, Any]) -> None:
        """Queues one event of the turn's stream."""
        turn.events.put_nowait({"session_id": self.id, "turn_id": turn.id, "type": kind, "data": data})
