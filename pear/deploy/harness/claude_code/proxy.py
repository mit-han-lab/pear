import asyncio
import base64
import hashlib
import io
import json
import socket
import time
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import aiohttp
from aiohttp import web
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from pear.deploy.harness.claude_code.engine_metrics import EngineMetrics
from pear.deploy.harness.claude_code.session import ClaudeCodeSession, Turn
from pear.utils.file_utils import write_text_atomically


class RequestAudit:
    """The records of a session's model requests, one file each in the order they arrived: what Claude Code asked,
    what the model server was sent and how it answered, with a digest in place of each image's data."""

    RECORD_NAME = "{number:05d}-{kind}.json"  # numbered in the order the requests arrived
    JSON_INDENT = 2

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.record_count = 0

        self.directory.mkdir(parents=True)

    def write(
        self,
        kind: str,
        turn_id: str | None,
        claude_code_request: dict[str, Any],
        upstream_request: dict[str, Any] | None,
        outcome: dict[str, Any],
    ) -> None:
        """Records one request: its kind, its turn, both forms of its body and its outcome."""
        self.record_count += 1
        record = {
            "kind": kind,
            "turn_id": turn_id,
            "time": time.time(),
            "claude_code_request": self.digested(claude_code_request),
            "upstream_request": self.digested(upstream_request),
            "outcome": outcome,
        }
        path = self.directory / self.RECORD_NAME.format(number=self.record_count, kind=kind)
        write_text_atomically(path, json.dumps(record, indent=self.JSON_INDENT) + "\n")

    @classmethod
    def digested(cls, value: Any) -> Any:
        """Returns ``value`` with the data of every base64 image source replaced by its SHA-256 and size."""
        if isinstance(value, list):
            return [cls.digested(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") == "base64" and "data" in value:
            data = base64.b64decode(value["data"])
            return {
                **{key: item for key, item in value.items() if key != "data"},
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }

        return {key: cls.digested(item) for key, item in value.items()}


class UpstreamProxy:
    """The model server as every session's Claude Code reaches it, on this machine's loopback address: each request
    rewritten for the served model, given all the output its context leaves, counted and recorded."""

    HOST = "127.0.0.1"  # only the sessions' own processes reach it
    MAIN = "main"  # a request of the conversation, whose answer the turn returns
    SUMMARY = "summary"  # a request of a compaction, whose answer becomes the conversation's summary
    REFUSED = "refused"  # a request that reached no model server
    SUMMARY_MARKER = "CRITICAL: Respond with TEXT ONLY."  # how the pinned CLI opens the instruction of a compaction
    # what the model server is sent of a request beside its rewritten parts: its sampling and its reasoning effort are
    # the server's and the config's
    FORWARDED_FIELDS = ("model", "messages", "max_tokens", "stream")
    COUNTED_FIELDS = ("model", "messages", "system")  # what the model server is sent of Claude Code's own counts
    # a refusal that Claude Code reports at once instead of retrying it, and a failure that it retries
    REFUSAL_STATUS = 409
    RETRY_STATUS = 500
    TOO_LONG_STATUS = 400
    SHOULD_RETRY_HEADER = "x-should-retry"
    # the error whose message Claude Code answers with a compaction
    TOO_LONG_MESSAGE = "prompt is too long: {tokens} tokens > {limit} maximum"
    # the reasons a turn ends without an action: the client's own word for a lost image context does not occur in them
    CUT_MESSAGE = "The answer reached the model's token limit and was cut; no action executes"
    CAMERAS_MESSAGE = "The camera images did not reach the model unchanged; no action executes"
    PLACEHOLDER_HEAD = ClaudeCodeSession.IMAGE_PLACEHOLDER.partition("{")[0]  # how every image's placeholder begins
    MAX_TOKENS_STOP = "max_tokens"  # the stop reason of an answer the server cut
    END_TURN_STOP = "end_turn"  # the stop reason of an answer the model ended itself
    SUMMARY_END = "</summary>"  # how the robot's instruction has a summary end
    # the failures Claude Code asks again after: a stream without output, an answer to a compaction that is no summary
    NO_OUTPUT_MESSAGE = "The model server sent no output"
    NO_SUMMARY_MESSAGE = "The model wrote no summary within its output limit"
    FRAME_SEPARATOR = b"\n\n"  # between two events of a stream
    DATA_PREFIX = b"data:"  # of the line of an event that holds its JSON
    FIRST_BLOCK_EVENT = "content_block_start"  # the first event of a stream that carries output
    PROMPT_EVENTS = ("message_start", "message")  # what reports a prompt's tokens: a stream's first event, an answer
    # a prompt's tokens, which the model server splits into what its cache held, what it added and the rest
    PROMPT_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    BLANK_FORMAT, BLANK_MEDIA_TYPE = "JPEG", "image/jpeg"  # of the pictures a count carries

    def __init__(
        self, config: DictConfig, http_session: aiohttp.ClientSession, context_limit: int, summary_instruction: str
    ) -> None:
        self.upstream = config.upstream.rstrip("/")
        self.chat_template_kwargs = OmegaConf.to_container(config.chat_template_kwargs)
        self.http_session = http_session
        self.context_limit = context_limit
        self.summary_instruction = summary_instruction  # in place of the CLI's own in a compaction's request
        self.metrics = EngineMetrics(http_session, self.upstream + "/metrics", config.metrics_interval_s)
        self.sessions: dict[str, ClaudeCodeSession] = {}  # by session id
        self.running_requests: dict[str, set[asyncio.Task]] = {}  # by session id: the handlers that run
        self.audits: dict[str, RequestAudit] = {}  # by session id: the records of its requests
        self.blank_pictures: dict[tuple[int, int], str] = {}  # base64 JPEGs, by width and height
        self.url = ""  # the proxy's address, once it listens

    async def __aenter__(self) -> Self:
        app = web.Application(client_max_size=ClaudeCodeSession.MESSAGE_LIMIT_BYTES)
        app.router.add_post("/{session_id}/v1/messages", self.messages)
        app.router.add_post("/{session_id}/v1/messages/count_tokens", self.count_tokens)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        listener = socket.create_server((self.HOST, 0))
        await web.SockSite(self.runner, listener).start()
        self.url = f"http://{self.HOST}:{listener.getsockname()[1]}"

        return self

    async def __aexit__(self, *exception: object) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)
        await self.runner.cleanup()

    def open(self, session: ClaudeCodeSession) -> str:
        """Routes the requests of ``session`` and returns the base URL its Claude Code is given."""
        self.sessions[session.id] = session
        self.running_requests[session.id] = set()
        self.audits[session.id] = RequestAudit(session.directory / "requests")

        return f"{self.url}/{session.id}"

    async def close(self, session_id: str) -> None:
        """Ends the requests of a session and its route."""
        await self.cancel(session_id)
        self.sessions.pop(session_id, None)
        self.running_requests.pop(session_id, None)
        self.audits.pop(session_id, None)

    async def cancel(self, session_id: str) -> None:
        """Ends the requests a session has running; the model server stops each with its connection."""
        tasks = list(self.running_requests.get(session_id, ()))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def settle(self, session_id: str) -> None:
        """Waits for the requests a session has running to end, past their streams: their last counts and their
        records."""
        tasks = list(self.running_requests.get(session_id, ()))
        if tasks:
            await asyncio.wait(tasks)

    async def messages(self, request: web.Request) -> web.StreamResponse:
        """Answers one model request of a session's Claude Code: refused outside a turn the session accepts,
        otherwise rewritten, sized to its context and relayed to the model server."""
        session = self.sessions.get(request.match_info["session_id"])
        audit = self.audits.get(request.match_info["session_id"])
        claude_code_request = await request.json()
        turn = session.turn if session else None
        if turn is None or turn.refusal or turn.cancellation.is_set():
            if audit:
                audit.write(self.REFUSED, None, claude_code_request, None, {"status": self.REFUSAL_STATUS})
            return self.refusal("The session takes no model request now")

        kind = self.kind_of(claude_code_request)
        upstream_request = self.rewrite(claude_code_request, session, kind)
        if kind == self.MAIN:
            # Claude Code's history only ever loses messages: an image its request no longer holds is not asked again,
            # but for the turn's own, which join the history after a compaction the turn begins with
            held = {
                self.placeholder_of(block)
                for message in claude_code_request["messages"]
                for block in self.blocks(message["content"])
            } | turn.images.keys()
            session.images = {
                placeholder: image for placeholder, image in session.images.items() if placeholder in held
            }
        outcome: dict[str, Any] = {}
        self.running_requests[session.id].add(asyncio.current_task())
        try:
            return await self.answer(request, session, turn, kind, upstream_request, outcome)
        finally:
            self.running_requests[session.id].discard(asyncio.current_task())
            audit.write(kind, turn.id, claude_code_request, upstream_request, outcome)

    async def count_tokens(self, request: web.Request) -> web.Response:
        """Answers Claude Code's own count of a prompt with the model server's, of the images its placeholders stand
        for."""
        body = await request.json()
        counted = {key: body[key] for key in self.COUNTED_FIELDS if key in body}
        if session := self.sessions.get(request.match_info["session_id"]):
            counted["messages"] = [self.drawn(message, session.images) for message in counted["messages"]]
        counted = self.counted(counted)
        async with self.http_session.post(self.upstream + "/v1/messages/count_tokens", json=counted) as response:
            return web.Response(status=response.status, body=await response.read(), content_type="application/json")

    def counted(self, request: dict[str, Any]) -> dict[str, Any]:
        """Returns ``request`` as the model server counts it: each picture a blank one of its size, whose tokens are
        the same. vLLM keeps the pictures it counts in its API server's cache as if its engine had them, and the
        request that then sends them fails on the engine's cache miss; blank ones are never sent."""
        messages = [
            {**message, "content": [self.blank(block) if block["type"] == "image" else block for block in content]}
            if isinstance(content := message["content"], list)
            else message
            for message in request["messages"]
        ]

        return {**request, "messages": messages}

    def blank(self, block: dict[str, Any]) -> dict[str, Any]:
        """Returns an image block of a blank picture of the size of ``block``'s."""
        size = Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))).size
        if size not in self.blank_pictures:
            picture = io.BytesIO()
            Image.new("RGB", size).save(picture, self.BLANK_FORMAT)
            self.blank_pictures[size] = base64.b64encode(picture.getvalue()).decode("ascii")
        source = {"type": "base64", "media_type": self.BLANK_MEDIA_TYPE, "data": self.blank_pictures[size]}

        return {**block, "source": source}

    def kind_of(self, claude_code_request: dict[str, Any]) -> str:
        """Returns whether a request is a compaction's, which holds the CLI's instruction, or the conversation's."""
        # the instruction may come before the CLI's note to go on with a cut summary; the summary replaces it
        is_summary = any(self.instructs_summary(message) for message in claude_code_request["messages"])

        return self.SUMMARY if is_summary else self.MAIN

    def rewrite(self, claude_code_request: dict[str, Any], session: ClaudeCodeSession, kind: str) -> dict[str, Any]:
        """Returns the request the model server is sent: the station's system prompt alone, each camera image in
        place of its placeholder, the task's images at the head of the first user message, the chat template's
        settings and, for a compaction, the robot's summary instruction in place of the CLI's."""
        messages = claude_code_request["messages"]
        if kind == self.SUMMARY:
            messages = [
                self.instructed(message) if self.instructs_summary(message) else message for message in messages
            ]
        messages = [self.drawn(message, session.images) for message in messages]
        messages[0] = {**messages[0], "content": [*session.references, *self.blocks(messages[0]["content"])]}
        upstream_request = {
            key: claude_code_request[key] for key in self.FORWARDED_FIELDS if key in claude_code_request
        }

        return {
            **upstream_request,
            "messages": messages,
            "system": session.system,
            "chat_template_kwargs": self.chat_template_kwargs,
        }

    def instructed(self, message: dict[str, Any]) -> dict[str, Any]:
        """Returns the message of a compaction's instruction with the robot's instruction in place of the CLI's and
        without its camera images and their labels."""
        instruction = {"type": "text", "text": self.summary_instruction}
        content: list[dict[str, Any]] = []
        # the CLI adds its instruction to the last turn's message, whose images draw the model to act on them instead
        for block in self.blocks(message["content"]):
            if self.is_placeholder(block):
                if content and content[-1]["type"] == "text":
                    content.pop()  # the camera's label, before its image (ClaudeCodeService.observation)
            else:
                content.append(instruction if self.is_instruction(block) else block)

        return {**message, "content": content}

    async def answer(
        self,
        request: web.Request,
        session: ClaudeCodeSession,
        turn: Turn,
        kind: str,
        upstream_request: dict[str, Any],
        outcome: dict[str, Any],
    ) -> web.StreamResponse:
        """Sizes the request's output to what its context leaves, checks the turn's camera images and relays it;
        ``outcome`` gets what the record of the request keeps."""
        counted = self.counted({key: upstream_request[key] for key in (*self.COUNTED_FIELDS, "chat_template_kwargs")})
        async with self.http_session.post(self.upstream + "/v1/messages/count_tokens", json=counted) as response:
            response.raise_for_status()
            prompt_tokens = (await response.json())["input_tokens"]
        room = self.context_limit - prompt_tokens
        outcome.update(prompt_tokens=prompt_tokens)
        if room <= 0:
            outcome.update(status=self.TOO_LONG_STATUS)
            message = self.TOO_LONG_MESSAGE.format(tokens=prompt_tokens, limit=self.context_limit)
            error = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
            return web.json_response(error, status=self.TOO_LONG_STATUS)
        # the main request may use all the output its context leaves, as the model was trained; a summary keeps the
        # CLI's budget
        upstream_request["max_tokens"] = room if kind == self.MAIN else min(upstream_request["max_tokens"], room)
        messages = upstream_request["messages"]
        if self.holds_placeholder(messages) or (kind == self.MAIN and not self.cameras_arrived(turn, messages[-1])):
            turn.refusal = self.CAMERAS_MESSAGE
            outcome.update(status=self.REFUSAL_STATUS)
            return self.refusal(turn.refusal)

        return await self.forward(request, session, turn, kind, upstream_request, outcome)

    async def forward(
        self,
        request: web.Request,
        session: ClaudeCodeSession,
        turn: Turn,
        kind: str,
        upstream_request: dict[str, Any],
        outcome: dict[str, Any],
    ) -> web.StreamResponse:
        """Sends the request to the model server and returns its answer to Claude Code, while the server's metrics
        count its output; an answer cut at the end of the context leaves the turn with no action."""
        request_id = uuid4().hex
        finished = asyncio.Event()
        # the stream's own output count, which replaces the metrics' once it arrives, and a compaction's answer text
        counts: dict[str, Any] = {"output": None, "text": ""}

        def on_output(tokens: int) -> None:
            # the metrics are read from before the server answers, which it does once it has rendered the prompt
            if counts["output"] is None and request_id in turn.usage.requests:
                turn.usage.update(request_id, output=tokens)
                session.report_usage(turn)

        try:
            baseline = await self.metrics.read()
        except (aiohttp.ClientError, KeyError, ValueError):
            baseline = None
        follower = asyncio.create_task(self.metrics.follow(baseline, on_output, finished)) if baseline else None
        try:
            async with self.http_session.post(self.upstream + "/v1/messages", json=upstream_request) as response:
                outcome.update(status=response.status, request_id=request_id)
                if response.status != web.HTTPOk.status_code:
                    return web.Response(
                        status=response.status, body=await response.read(), content_type="application/json"
                    )

                turn.usage.open_request(request_id)
                if kind == self.MAIN:
                    session.reference_request_id = request_id
                session.report_usage(turn, is_forced=True)
                if upstream_request.get("stream"):
                    reply = await self.relay(request, response, session, turn, request_id, kind, counts)
                else:
                    message = await response.json()
                    self.observe(session, turn, request_id, kind, counts, message)
                    is_failure = kind == self.SUMMARY and not self.summarizes(counts)
                    reply = self.failure(self.NO_SUMMARY_MESSAGE) if is_failure else web.json_response(message)
        finally:
            finished.set()
            generation_time_ms = await follower if follower else None
            if request_id in turn.usage.requests:
                turn.usage.update(request_id, generation_time_ms=generation_time_ms)
                session.report_usage(turn, is_forced=True)

        # what Claude Code was answered: a failure in place of the server's answer when it is no summary
        outcome.update(
            stop_reason=counts.get("stop_reason"), usage=turn.usage.requests.get(request_id), reply_status=reply.status
        )

        return reply

    async def relay(
        self,
        request: web.Request,
        response: aiohttp.ClientResponse,
        session: ClaudeCodeSession,
        turn: Turn,
        request_id: str,
        kind: str,
        counts: dict[str, Any],
    ) -> web.StreamResponse:
        """Passes the server's event stream on as it arrives once it carries output, and a compaction's whole once it is
        a summary; any other is a failure Claude Code sends the request again after."""
        # Claude Code takes an empty stream as an empty answer, and any answer of its compaction as the summary
        reply: web.StreamResponse | None = None
        held_frames: list[bytes] = []
        buffer = b""
        has_output = False

        async for chunk in response.content.iter_any():
            buffer += chunk
            *frames, buffer = buffer.split(self.FRAME_SEPARATOR)
            for frame in frames:
                event = self.event_of(frame)
                self.observe(session, turn, request_id, kind, counts, event)
                if reply is not None:
                    await reply.write(frame + self.FRAME_SEPARATOR)
                    continue
                held_frames.append(frame)
                has_output = has_output or event.get("type") == self.FIRST_BLOCK_EVENT
                if has_output and kind == self.MAIN:
                    reply = await self.opened(request, response, held_frames)

        if reply is None and has_output and self.summarizes(counts):
            reply = await self.opened(request, response, held_frames)
        if reply is None:
            if not has_output:
                # no answer came of it: its retry is counted instead
                turn.usage.drop_request(request_id)
            return self.failure(self.NO_SUMMARY_MESSAGE if has_output else self.NO_OUTPUT_MESSAGE)
        await reply.write(buffer)
        await reply.write_eof()

        return reply

    def observe(
        self,
        session: ClaudeCodeSession,
        turn: Turn,
        request_id: str,
        kind: str,
        counts: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        """Takes the counts and the stop reason an event of the stream, or a whole answer, reports, before Claude Code
        has it: an answer cut at its token limit leaves the turn without an action before Claude Code asks to go on."""
        message = event.get("message", event)
        usage = event.get("usage") or message.get("usage")
        if kind == self.SUMMARY:
            texts = [block.get("text") or "" for block in message.get("content") or [] if block.get("type") == "text"]
            counts["text"] += ((event.get("delta") or {}).get("text") or "") + "".join(texts)
        if event.get("type") in self.PROMPT_EVENTS:
            prompt_tokens = sum(usage.get(key) or 0 for key in self.PROMPT_KEYS)
            turn.usage.update(request_id, input=prompt_tokens)
            if kind == self.MAIN:
                turn.usage.measure_context(prompt_tokens)
            else:
                turn.usage.summary_input = prompt_tokens
        stop_reason = (event.get("delta") or message).get("stop_reason")
        if stop_reason:
            counts.update(output=usage["output_tokens"], stop_reason=stop_reason)
            turn.usage.update(request_id, output=usage["output_tokens"], complete=True)
        if kind == self.MAIN and stop_reason == self.MAX_TOKENS_STOP:
            turn.refusal = self.CUT_MESSAGE
        session.report_usage(turn)

    @classmethod
    async def opened(
        cls, request: web.Request, response: aiohttp.ClientResponse, held_frames: list[bytes]
    ) -> web.StreamResponse:
        """Returns the stream of the answer to Claude Code, begun with the frames held of it."""
        reply = web.StreamResponse(headers={"Content-Type": response.headers["Content-Type"]})
        await reply.prepare(request)
        await reply.write(b"".join(frame + cls.FRAME_SEPARATOR for frame in held_frames))

        return reply

    @classmethod
    def summarizes(cls, counts: dict[str, Any]) -> bool:
        """Returns whether an answer the model ended itself holds a summary."""
        return counts.get("stop_reason") == cls.END_TURN_STOP and cls.SUMMARY_END in counts["text"]

    @classmethod
    def failure(cls, message: str) -> web.Response:
        """Returns the answer of a failure that Claude Code sends its request again after."""
        error = {"type": "error", "error": {"type": "api_error", "message": message}}

        return web.json_response(error, status=cls.RETRY_STATUS, headers={cls.SHOULD_RETRY_HEADER: "true"})

    @classmethod
    def drawn(cls, message: dict[str, Any], images: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Returns ``message`` with each placeholder of a camera image that ``images`` keeps replaced by the
        image."""
        if not isinstance(message["content"], list):
            return message

        content = [images.get(cls.placeholder_of(block), block) for block in message["content"]]

        return {**message, "content": content}

    @classmethod
    def instructs_summary(cls, message: dict[str, Any]) -> bool:
        """Returns whether ``message`` holds the CLI's instruction of a compaction."""
        return any(cls.is_instruction(block) for block in cls.blocks(message["content"]))

    @classmethod
    def is_instruction(cls, block: dict[str, Any]) -> bool:
        """Returns whether ``block`` is the CLI's instruction of a compaction."""
        return block["type"] == "text" and block["text"].startswith(cls.SUMMARY_MARKER)

    @classmethod
    def is_placeholder(cls, block: dict[str, Any]) -> bool:
        """Returns whether ``block`` is the placeholder of a camera image."""
        return (cls.placeholder_of(block) or "").startswith(cls.PLACEHOLDER_HEAD)

    @staticmethod
    def placeholder_of(block: dict[str, Any]) -> str | None:
        """Returns what a block holds as a placeholder of an image is written: a text block's text without the newline
        Claude Code ends an interrupted turn's message with, which it gives no image; None for any other block."""
        return block["text"].strip() if block["type"] == "text" else None

    @classmethod
    def holds_placeholder(cls, messages: list[dict[str, Any]]) -> bool:
        """Returns whether ``messages`` still hold a placeholder of a camera image: of one the session does not keep,
        or one Claude Code changed."""
        return any(
            block["type"] == "text" and cls.PLACEHOLDER_HEAD in block["text"]
            for message in messages
            for block in cls.blocks(message["content"])
        )

    @classmethod
    def cameras_arrived(cls, turn: Turn, message: dict[str, Any]) -> bool:
        """Returns whether ``message`` ends with the turn's camera images, unchanged and in their order."""
        sent = [block["source"]["data"] for block in turn.images.values()]
        received = [block["source"].get("data") for block in cls.blocks(message["content"]) if block["type"] == "image"]

        return not sent or received[-len(sent) :] == sent

    @classmethod
    def refusal(cls, message: str) -> web.Response:
        """Returns the answer that ends Claude Code's request without a retry."""
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}

        return web.json_response(error, status=cls.REFUSAL_STATUS, headers={cls.SHOULD_RETRY_HEADER: "false"})

    @classmethod
    def event_of(cls, frame: bytes) -> dict[str, Any]:
        """Returns the JSON of a stream's event, empty for a frame that holds none."""
        data = b"".join(line[len(cls.DATA_PREFIX) :] for line in frame.splitlines() if line.startswith(cls.DATA_PREFIX))

        return json.loads(data) if data.strip() else {}

    @staticmethod
    def blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Returns a message's content as its list of blocks."""
        return content if isinstance(content, list) else [{"type": "text", "text": content}]
