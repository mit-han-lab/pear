import asyncio
import base64
import binascii
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import aiohttp
import hydra
from aiohttp import web
from omegaconf import DictConfig

from pear.deploy.harness.claude_code.proxy import UpstreamProxy
from pear.deploy.harness.claude_code.session import ClaudeCodeSession, SessionLost
from pear.deploy.station.agent_loop.clients.claude_code import ClaudeCodeClient
from pear.deploy.station.agent_loop.clients.schemas import HealthStatus


class ClaudeCodeService:
    """The Claude Code session service of the robot station's ClaudeCodeClient, from its composed config: one session
    at a time, each a Claude Code process whose model requests reach the model server through the upstream proxy."""

    # what a session's Claude Code inherits of this process's environment: none of a login shell's keys or proxies
    KEPT_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")
    UPSTREAM_CHECK_TIMEOUT_S = 1.5  # of a health check of the model server: within the client's own check
    MAX_IMAGE_BYTES = 10 * 1024 * 1024  # of one image of a turn, decoded
    JPEG_MAGIC = b"\xff\xd8"  # what a JPEG file begins with
    LABEL = "[{role}]"  # the text before a camera's image, as the station's VLLMClient sends it
    SUMMARY_PROMPT = Path(__file__).with_name("summary_prompt.md")  # a compaction's instruction to the model

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.session: ClaudeCodeSession | None = None  # the one session, which a newer one replaces
        self.closed_ids: set[str] = set()  # of the sessions that ended: none is opened again

    def run(self) -> None:
        """Serves until a stop signal."""
        for name in list(os.environ):
            if name not in self.KEPT_ENVIRONMENT:
                del os.environ[name]

        web.run_app(
            self.app(), host=self.config.host, port=self.config.port, access_log=None, handler_cancellation=True
        )

    def app(self) -> web.Application:
        """Returns the service's web application, whose startup checks the model server and starts the proxy."""
        app = web.Application(client_max_size=ClaudeCodeSession.MESSAGE_LIMIT_BYTES)
        app.cleanup_ctx.append(self.lifetime)
        app.router.add_get("/health", self.health)
        app.router.add_post("/sessions", self.create)
        app.router.add_post("/sessions/{session_id}/turns", self.turn)
        app.router.add_post("/sessions/{session_id}/turns/{turn_id}/interrupt", self.interrupt)
        app.router.add_delete("/sessions/{session_id}", self.delete)

        return app

    async def lifetime(self, app: web.Application) -> AsyncIterator[None]:
        """Holds the connections to the model server and the proxy while the service runs: the model's context is
        what the model server reports of the served model."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http_session:
            async with http_session.get(self.config.upstream.rstrip("/") + "/v1/models") as response:
                response.raise_for_status()
                served_models = {model["id"]: model for model in (await response.json())["data"]}
            if self.config.model not in served_models:
                raise ValueError(f"The model server does not serve {self.config.model}: {', '.join(served_models)}")

            context_limit = served_models[self.config.model]["max_model_len"]
            summary_instruction = self.SUMMARY_PROMPT.read_text()
            async with UpstreamProxy(self.config, http_session, context_limit, summary_instruction) as proxy:
                self.proxy = proxy
                self.context_limit = context_limit
                yield
                if self.session:
                    await self.session.close()

    async def health(self, request: web.Request) -> web.Response:
        """Answers with the service's identity and protocol, the model and whether the model server answers."""
        answer = {
            "service": ClaudeCodeClient.SERVICE_NAME,
            "protocol": ClaudeCodeClient.PROTOCOL,
            "model": self.config.model,
            "capabilities": {"reference_images": True, "reference_image_recovery": True},
        }
        timeout = aiohttp.ClientTimeout(total=self.UPSTREAM_CHECK_TIMEOUT_S)
        try:
            async with self.proxy.http_session.get(self.proxy.upstream + "/health", timeout=timeout) as response:
                response.raise_for_status()
        except (aiohttp.ClientError, TimeoutError) as error:
            offline = {**answer, "status": HealthStatus.OFFLINE, "message": f"The model server is unavailable: {error}"}
            return web.json_response(offline, status=web.HTTPServiceUnavailable.status_code)

        return web.json_response({**answer, "status": HealthStatus.OK})

    async def create(self, request: web.Request) -> web.Response:
        """Opens a session with the station's system prompt, in place of the one that runs, once its Claude Code
        runs with the model server's context."""
        body = await request.json()
        session_id = self.uuid_of(body.get("session_id"))
        if session_id in self.closed_ids or (self.session and self.session.id == session_id):
            raise web.HTTPConflict(text="Session ID already used")
        if body.get("model") != self.config.model:
            raise web.HTTPBadRequest(text=f"The service serves {self.config.model}")
        if not isinstance(body.get("system"), str) or not body["system"]:
            raise web.HTTPBadRequest(text="Expected a system prompt")

        if self.session:
            self.closed_ids.add(self.session.id)
            await self.session.close()
        self.session = ClaudeCodeSession(session_id, body["system"], self.config, self.proxy, self.context_limit)
        try:
            await self.session.started
        except SessionLost as error:
            raise web.HTTPServiceUnavailable(text=str(error)) from error

        return web.json_response({"session_id": session_id})

    async def turn(self, request: web.Request) -> web.StreamResponse:
        """Runs one turn of the session and streams its events, one JSON object a line, up to its result; a client
        that leaves before the result interrupts the turn."""
        session = self.session_of(request)
        body = await request.json()
        turn_id = self.uuid_of(body.get("turn_id"))
        references = body.get("reference_images", [])
        if not isinstance(references, list) or not all(isinstance(image, dict) for image in references):
            raise web.HTTPBadRequest(text="Expected a list of task images")
        content, images = self.observation(body, turn_id)
        try:
            turn = session.submit(
                turn_id, content, images, [self.image_block(image.get("data")) for image in references]
            )
        except SessionLost as error:
            raise web.HTTPGone(text=str(error)) from error
        except ValueError as error:
            raise web.HTTPConflict(text=str(error)) from error

        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        try:
            await response.prepare(request)
            while True:
                event = await turn.events.get()
                await response.write((json.dumps(event, allow_nan=False) + "\n").encode())
                if event["type"] == ClaudeCodeClient.RESULT_EVENT:
                    break
            await response.write_eof()

            return response
        finally:
            if not turn.done.is_set():
                try:
                    await asyncio.shield(session.interrupt(turn_id))
                except SessionLost:
                    pass

    async def interrupt(self, request: web.Request) -> web.Response:
        """Interrupts a turn, before it starts too, and answers once its result is out."""
        session = self.session_of(request)
        try:
            await session.interrupt(self.uuid_of(request.match_info["turn_id"]))
        except SessionLost as error:
            raise web.HTTPGone(text=str(error)) from error

        return web.json_response({"status": "stopped"})

    async def delete(self, request: web.Request) -> web.Response:
        """Ends a session for good."""
        session_id = self.uuid_of(request.match_info["session_id"])
        self.closed_ids.add(session_id)
        if self.session and self.session.id == session_id:
            session, self.session = self.session, None
            await session.close()

        return web.json_response({"closed": True})

    def session_of(self, request: web.Request) -> ClaudeCodeSession:
        """Returns the session a request names; refuses one that is not the service's or whose conversation is
        lost."""
        session = self.session
        if not session or session.id != self.uuid_of(request.match_info["session_id"]):
            raise web.HTTPGone(text="Session unavailable; clear the task")
        if session.lost_reason:
            raise web.HTTPGone(text=session.lost_reason)

        return session

    def observation(self, body: dict[str, Any], turn_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Returns a turn's user message as Claude Code is given it, the text, then each camera's label and the
        placeholder of its image, and the images by their placeholders; refuses a malformed turn."""
        text, cameras = body.get("text"), body.get("images")
        if not isinstance(text, str) or not isinstance(cameras, list) or not all(isinstance(c, dict) for c in cameras):
            raise web.HTTPBadRequest(text="Expected a text and a list of camera images")
        roles = [camera.get("role") for camera in cameras]
        if not all(isinstance(role, str) and role for role in roles) or len(set(roles)) != len(roles):
            raise web.HTTPBadRequest(text="Expected one image of each camera, each under its role")

        blocks, images = [{"type": "text", "text": text}], {}
        for camera in cameras:
            placeholder = ClaudeCodeSession.IMAGE_PLACEHOLDER.format(key=f"{turn_id}/{camera['role']}")
            images[placeholder] = self.image_block(camera.get("data"))
            blocks += [
                {"type": "text", "text": self.LABEL.format(role=camera["role"])},
                {"type": "text", "text": placeholder},
            ]

        return blocks, images

    def image_block(self, data: object) -> dict[str, Any]:
        """Returns a JPEG's base64 as an image block of a message; refuses what is not a JPEG within the size."""
        try:
            image = base64.b64decode(data, validate=True) if isinstance(data, str) else b""
        except binascii.Error:
            image = b""
        if not image.startswith(self.JPEG_MAGIC) or len(image) > self.MAX_IMAGE_BYTES:
            raise web.HTTPBadRequest(text="Expected a JPEG image in base64")

        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

    @staticmethod
    def uuid_of(value: object) -> str:
        """Returns ``value`` when it is a UUID's text; refuses it otherwise."""
        try:
            UUID(value)
        except (ValueError, TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text="Expected a UUID") from error

        return value


@hydra.main(config_path="../../../configs", config_name="claude_code", version_base=None)
def main(config: DictConfig) -> None:
    ClaudeCodeService(config).run()


if __name__ == "__main__":
    main()
