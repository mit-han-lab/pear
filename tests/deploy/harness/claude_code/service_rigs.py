import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from PIL import Image

from pear.deploy.harness.claude_code.main import ClaudeCodeService
from pear.deploy.harness.claude_code.proxy import UpstreamProxy
from pear.deploy.harness.claude_code.session import ClaudeCodeSession, Turn, TurnUsage
from pear.deploy.station.agent_loop.clients.claude_code import ClaudeCodeClient
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput, ReferenceImage
from repository import REPO


class Pictures:
    """Small JPEG images of one color each, as a turn's camera frames and a task's images carry them."""

    # the bytes the images share: a 16x12 JPEG's header and tables, then each color's own scan
    HEADER = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSop"
        "GR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
        "KCj/wAARCAAMABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQID"
        "AAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlq"
        "c3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3"
        "+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEI"
        "FEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImK"
        "kpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwA="
    )
    SCANS = {
        "red": "e2a8a28af00fd6cfffd9",
        "green": "b1451457c11f991fffd9",
        "blue": "f22a28a2bf5b380fffd9",
        "yellow": "f6ca28a2bf183d13ffd9",
    }

    @classmethod
    def jpeg(cls, color: str) -> bytes:
        return cls.HEADER + bytes.fromhex(cls.SCANS[color])

    @classmethod
    def base64(cls, color: str) -> str:
        return base64.b64encode(cls.jpeg(color)).decode("ascii")

    @classmethod
    def block(cls, color: str) -> dict[str, Any]:
        """The image as a block of a message."""
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": cls.base64(color)}}

    @staticmethod
    def of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The image blocks of ``messages``, in order."""
        return [
            block
            for message in messages
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "image"
        ]

    @staticmethod
    def size(block: dict[str, Any]) -> tuple[int, int]:
        """The width and height of an image block's picture."""
        return Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))).size

    @staticmethod
    def undrawn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``messages`` with their images' data left out."""
        return [
            {
                **message,
                "content": [
                    {**block, "source": {**block["source"], "data": None}} if block["type"] == "image" else block
                    for block in message["content"]
                ],
            }
            if isinstance(message["content"], list)
            else message
            for message in messages
        ]


class StandInUpstream:
    """A stand-in of the model server: vLLM's Anthropic messages and token counts, its model list, its health and its
    Prometheus metrics; it answers with the streams a test scripts and records what it is sent."""

    MODEL = "served-model"
    CONTEXT_LIMIT = 260096
    ANSWER = 'Holding still.\n```json\n{"arm": {"x": 0}}\n```'
    SUMMARY = "<summary>The arm holds still.</summary>"
    SUMMARY_TAG = "<summary>"  # what the robot's compaction instruction names
    OUTPUT_TOKENS = 120  # of a default answer
    THINKING_CHUNKS = 3  # of a default answer's reasoning
    METRICS = {
        "vllm:generation_tokens_total": "counter",
        "vllm:request_decode_time_seconds": "histogram",
        "vllm:num_requests_running": "gauge",
    }

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []  # the bodies of the message requests, in order
        self.counted: list[dict[str, Any]] = []  # the bodies of the token counts, in order
        self.answers: list[dict[str, Any]] = []  # the scripted answers, used in order before the default one
        self.summary_answers: list[dict[str, Any]] = []  # those of compactions alone, used before the answers
        self.prompt_tokens = 4000  # what a count and a stream's first event report
        self.is_healthy = True
        self.generation_tokens = 0
        self.decode_seconds = 0.0
        self.finished_requests = 0
        self.running_requests = 0

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 28)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/v1/messages", self.messages)
        app.router.add_post("/v1/messages/count_tokens", self.count_tokens)

        return app

    async def health(self, request: web.Request) -> web.Response:
        return web.Response(status=200 if self.is_healthy else 503)

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": self.MODEL, "max_model_len": self.CONTEXT_LIMIT}]})

    async def metrics(self, request: web.Request) -> web.Response:
        lines = [f"# TYPE {name} {kind}" for name, kind in self.METRICS.items()]
        lines += [
            f'vllm:generation_tokens_total{{model_name="{self.MODEL}"}} {float(self.generation_tokens)}',
            f'vllm:request_decode_time_seconds_sum{{model_name="{self.MODEL}"}} {self.decode_seconds}',
            f'vllm:request_decode_time_seconds_count{{model_name="{self.MODEL}"}} {float(self.finished_requests)}',
            f'vllm:num_requests_running{{model_name="{self.MODEL}"}} {float(self.running_requests)}',
        ]

        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def count_tokens(self, request: web.Request) -> web.Response:
        self.counted.append(await request.json())

        return web.json_response({"input_tokens": self.prompt_tokens})

    async def messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        is_summary = self.SUMMARY_TAG in json.dumps(body["messages"])  # also past a cut summary the CLI asks again of
        scripted = self.summary_answers if is_summary and self.summary_answers else self.answers
        answer = scripted.pop(0) if scripted else {}
        if "status" in answer:
            return web.json_response(answer["error"], status=answer["status"])
        # the server renders a long prompt before it answers at all
        await asyncio.sleep(answer.get("header_delay_s", 0))

        text = answer.get("text", self.SUMMARY if is_summary else self.ANSWER)  # None: the model only reasons
        cached_tokens = answer.get("cached_tokens", 0)
        prompt_tokens = answer.get("prompt_tokens", self.prompt_tokens)
        prompt_usage = {"input_tokens": prompt_tokens - cached_tokens, "cache_read_input_tokens": cached_tokens}
        output_tokens = answer.get("output_tokens", self.OUTPUT_TOKENS)
        stop_reason = answer.get("stop_reason", "end_turn")
        if not body.get("stream"):
            self.finish(output_tokens, answer)
            content = [{"type": "thinking", "thinking": "think", "signature": "0"}, {"type": "text", "text": text}]
            message = {"id": "msg", "type": "message", "role": "assistant", "content": content, "model": self.MODEL}
            usage = {**prompt_usage, "output_tokens": output_tokens}
            return web.json_response({**message, "stop_reason": stop_reason, "usage": usage})

        return await self.stream(request, answer, text, prompt_usage, output_tokens, stop_reason)

    async def stream(
        self,
        request: web.Request,
        answer: dict[str, Any],
        text: str,
        prompt_usage: dict[str, int],
        output_tokens: int,
        stop_reason: str,
    ) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"})
        await response.prepare(request)
        delay_s = answer.get("delay_s", 0)
        self.running_requests += 1

        async def event(name: str, data: dict[str, Any]) -> None:
            await response.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            await asyncio.sleep(delay_s)

        start = {"id": "msg", "type": "message", "role": "assistant", "content": [], "model": self.MODEL}
        start |= {"stop_reason": None, "usage": {**prompt_usage, "output_tokens": 0}}
        try:
            await event("message_start", {"type": "message_start", "message": start})
            if answer.get("is_empty"):
                await response.write_eof()
                return response
            thinking_block = {"type": "thinking", "thinking": ""}
            await event(
                "content_block_start", {"type": "content_block_start", "index": 0, "content_block": thinking_block}
            )
            for _ in range(answer.get("thinking_chunks", self.THINKING_CHUNKS)):
                self.generation_tokens += 1
                delta = {"type": "thinking_delta", "thinking": "think "}
                await event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta})
            signature = {"type": "signature_delta", "signature": uuid4().hex}
            await event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": signature})
            await event("content_block_stop", {"type": "content_block_stop", "index": 0})
            if text is not None:
                text_block = {"type": "text", "text": ""}
                start = {"type": "content_block_start", "index": 1, "content_block": text_block}
                await event("content_block_start", start)
                delta = {"type": "text_delta", "text": text}
                await event("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": delta})
                await event("content_block_stop", {"type": "content_block_stop", "index": 1})
            usage = {**prompt_usage, "output_tokens": output_tokens}
            final = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            }
            await event("message_delta", final)
            await event("message_stop", {"type": "message_stop"})
            await response.write_eof()
        finally:
            self.running_requests -= 1
            self.finish(output_tokens - answer.get("thinking_chunks", self.THINKING_CHUNKS), answer)

        return response

    def finish(self, output_tokens: int, answer: dict[str, Any]) -> None:
        """Counts a request's output and decode interval in the metrics."""
        self.generation_tokens += output_tokens
        self.decode_seconds += answer.get("decode_s", 0.5)
        self.finished_requests += 1


class ServiceConfig:
    """The service's config composed with overrides, as its entry point composes it."""

    @staticmethod
    def of(overrides: list[str]) -> DictConfig:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "pear" / "configs"), version_base=None):
            return compose(config_name="claude_code", overrides=["metrics_interval_s=0.05", *overrides])


class ClaudeCodeRequests:
    """Request bodies shaped as the pinned Claude Code CLI sends them: its own system blocks before the station's,
    and the fields the model server is not sent."""

    MARKER = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."  # the head of the CLI's compaction instruction
    CLI_SYSTEM = (
        "x-anthropic-billing-header: cc_version=0; cc_entrypoint=sdk-py;",
        "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
    )
    CLI_MAX_TOKENS = 32768

    @classmethod
    def main(cls, system: str, messages: list[dict[str, Any]], *, is_stream: bool = True) -> dict[str, Any]:
        return {
            "model": StandInUpstream.MODEL,
            "messages": messages,
            "system": [{"type": "text", "text": text} for text in (*cls.CLI_SYSTEM, system)],
            "tools": [],
            "metadata": {"user_id": "{}"},
            "max_tokens": cls.CLI_MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "context_management": {"edits": []},
            "output_config": {"effort": "xhigh"},
            "temperature": 1,
            "stream": is_stream,
        }

    @classmethod
    def summary(cls, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        *earlier, last = messages
        instruction = {"type": "text", "text": cls.MARKER + "\n\nYour task is to create a detailed summary."}
        return cls.main(system, [*earlier, {**last, "content": [*last["content"], instruction]}])

    @staticmethod
    def placeholder(turn_id: str, index: int) -> str:
        """What Claude Code holds of a turn's image of a camera."""
        return ClaudeCodeSession.IMAGE_PLACEHOLDER.format(key=f"{turn_id}/camera_{index}")

    @classmethod
    def observation(cls, text: str, colors: tuple[str, ...], turn_id: str) -> dict[str, Any]:
        """A turn's user message as the service gives it to Claude Code: the text, then each camera's label and the
        placeholder of its image."""
        content = [{"type": "text", "text": text}]
        for index, _ in enumerate(colors):
            content += [
                {"type": "text", "text": f"[camera_{index}]"},
                {"type": "text", "text": cls.placeholder(turn_id, index)},
            ]

        return {"role": "user", "content": content}

    @classmethod
    def images(cls, colors: tuple[str, ...], turn_id: str) -> dict[str, dict[str, Any]]:
        """A turn's camera images as the service keeps them, by their placeholders."""
        return {cls.placeholder(turn_id, index): Pictures.block(color) for index, color in enumerate(colors)}

    @staticmethod
    def observation_as_sent(text: str, colors: tuple[str, ...]) -> dict[str, Any]:
        """A turn's user message as the model server is sent it: the text, then each camera's label and image."""
        content = [{"type": "text", "text": text}]
        for index, color in enumerate(colors):
            content += [{"type": "text", "text": f"[camera_{index}]"}, Pictures.block(color)]

        return {"role": "user", "content": content}

    @staticmethod
    def answer(text: str = StandInUpstream.ANSWER) -> dict[str, Any]:
        thinking = {"type": "thinking", "thinking": "think", "signature": "0"}
        return {"role": "assistant", "content": [thinking, {"type": "text", "text": text}]}


class StandInSession:
    """What the proxy reads and calls of a session: its id, its system prompt, the task's images, the camera images
    of its history, its running turn, its directory and its usage reports."""

    SYSTEM = "You are controlling a robot."

    def __init__(self, directory: Path, references: list[dict[str, Any]], turn_colors: tuple[str, ...]) -> None:
        self.id = str(uuid4())
        self.system = self.SYSTEM
        self.references = references
        self.reference_request_id: str | None = None
        self.directory = directory / self.id
        self.usage_reports: list[dict[str, Any]] = []
        turn_id = str(uuid4())
        content = ClaudeCodeRequests.observation("observe", turn_colors, turn_id)["content"]
        self.images = ClaudeCodeRequests.images(turn_colors, turn_id)
        usage = TurnUsage(StandInUpstream.CONTEXT_LIMIT, None)
        self.turn: Turn | None = Turn(turn_id, content, dict(self.images), usage)

    def report_usage(self, turn: Turn, *, is_final: bool = False, is_forced: bool = False) -> None:
        self.usage_reports.append(turn.usage.snapshot(is_final=is_final))


class ProxyRig:
    """The proxy before a stand-in model server, with one stand-in session routed, in a test's event loop."""

    SUMMARY_INSTRUCTION = "Write the robot's summary inside <summary> tags."

    def __init__(self, tmp_path: Path, references: list[dict[str, Any]], turn_colors: tuple[str, ...]) -> None:
        self.tmp_path = tmp_path
        self.upstream = StandInUpstream()
        self.session = StandInSession(tmp_path, references, turn_colors)

    async def __aenter__(self) -> "ProxyRig":
        self.upstream_server = TestServer(self.upstream.app())
        await self.upstream_server.start_server()
        config = ServiceConfig.of(
            [
                "port=0",
                f"upstream={self.upstream_server.make_url('')}",
                f"model={StandInUpstream.MODEL}",
                f"run_dir={self.tmp_path}",
            ]
        )
        self.http_session = ClientSession()
        self.proxy = UpstreamProxy(config, self.http_session, StandInUpstream.CONTEXT_LIMIT, self.SUMMARY_INSTRUCTION)
        await self.proxy.__aenter__()
        self.url = self.proxy.open(self.session)

        return self

    async def __aexit__(self, *exception: object) -> None:
        await self.proxy.__aexit__(*exception)
        await self.http_session.close()
        await self.upstream_server.close()

    async def send(self, body: dict[str, Any], path: str = "/v1/messages") -> tuple[int, dict[str, str], bytes]:
        """Posts a request as Claude Code does and returns the proxy's status, headers and whole body, once the
        request's counts and record are done."""
        async with self.http_session.post(self.url + path, json=body) as response:
            answer = response.status, dict(response.headers), await response.read()
        await self.proxy.settle(self.session.id)

        return answer


class ServiceRig:
    """The service before a stand-in model server, both on this machine, in a test's event loop, with the station's
    own client to reach it."""

    def __init__(self, tmp_path: Path, overrides: tuple[str, ...] = ()) -> None:
        self.tmp_path = tmp_path
        self.overrides = overrides
        self.upstream = StandInUpstream()

    async def __aenter__(self) -> "ServiceRig":
        self.upstream_server = TestServer(self.upstream.app())
        await self.upstream_server.start_server()
        config = ServiceConfig.of(
            [
                "port=0",
                f"upstream={self.upstream_server.make_url('')}",
                f"model={StandInUpstream.MODEL}",
                f"run_dir={self.tmp_path / 'runs'}",
                *self.overrides,
            ]
        )
        self.service = ClaudeCodeService(config)
        self.service_server = TestServer(self.service.app())
        await self.service_server.start_server()

        return self

    async def __aexit__(self, *exception: object) -> None:
        await self.service_server.close()
        await self.upstream_server.close()

    def client(self, system: str = StandInSession.SYSTEM, **arguments: Any) -> ClaudeCodeClient:
        base_url = str(self.service_server.make_url("")).rstrip("/")
        return ClaudeCodeClient(system, timeout_s=60, base_url=base_url, model=StandInUpstream.MODEL, **arguments)

    def records(self, client: ClaudeCodeClient) -> Path:
        """The directory of the records of a client's session's model requests."""
        return self.tmp_path / "runs" / client.session_id / "requests"

    def observation(self, text: str, colors: tuple[str, ...], references: tuple[str, ...] = ()) -> ModelInput:
        """A turn's input with its camera frames and task images saved as files."""
        cameras = tuple(CameraImage(f"camera_{index}", self.saved(color)) for index, color in enumerate(colors))
        return ModelInput(text, cameras, tuple(ReferenceImage(self.saved(color)) for color in references))

    def saved(self, color: str) -> Path:
        path = self.tmp_path / f"{color}.jpg"
        path.write_bytes(Pictures.jpeg(color))
        return path
