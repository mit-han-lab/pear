import asyncio
import json
import time
from collections.abc import Awaitable
from typing import Any

from aiohttp import WSMsgType, web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.preview import Preview
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class LiveRoutes:
    """The one WebSocket of each browser, which carries the events, the state and the camera previews."""

    MAX_MESSAGE_BYTES = 16384  # of a message a browser sends
    HEARTBEAT_S = 10  # between two pings of a socket, which end a connection that no longer answers
    FRAME_PERIOD_S = 1 / Preview.FRAME_RATE  # between two rounds of sending, and two reads of the feed

    def __init__(self, console: Console) -> None:
        self.console = console
        self.live_feed = LiveFeed(console)  # read by a background thread of the application, sent by every socket

    def add_routes(self, router: web.UrlDispatcher) -> None:
        """Adds the route to ``router``."""
        router.add_get("/api/live", self.live)

    async def live(self, request: web.Request) -> web.WebSocketResponse:
        """Sends a browser the events after the one it names, each new state and fresh previews, and takes its
        heartbeats, until it goes."""
        # one socket carries the previews and the events: a browser's few HTTP/1 connections to the service stay free
        # for the commands
        socket = web.WebSocketResponse(max_msg_size=self.MAX_MESSAGE_BYTES, heartbeat=self.HEARTBEAT_S, compress=False)
        await socket.prepare(request)
        live_stream = LiveStream(self.console, self.live_feed, socket, int(request.query.get("after", "0")))
        sender = asyncio.create_task(live_stream.run())

        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    data = json.loads(message.data)
                    if data.get("type") == "heartbeat":
                        self.console.lease.heartbeat(data.get("client_id"))
                if sender.done():
                    break
        except (ConnectionError, asyncio.CancelledError):
            pass  # the browser went, or the service stops
        except ValueError:
            logger.warning("A browser's live message could not be read; its socket is closed", exc_info=True)
        finally:
            sender.cancel()
            [send_outcome] = await asyncio.gather(sender, return_exceptions=True)
            if isinstance(send_outcome, Exception):
                logger.error("A live socket stopped sending", exc_info=send_outcome)
            await socket.close()

        return socket


class LiveFeed:
    """The recent events and the state that every live socket sends, read once a period off the event loop."""

    STATE_PERIOD_S = 0.2  # between two states encoded for the sockets

    def __init__(self, console: Console) -> None:
        self.console = console
        self.events: tuple[dict[str, Any], ...] = ()  # the console's recent events, the oldest first
        # the latest state message, encoded once for every socket; None before the first
        self.state_message: str | None = None
        self.state_at = 0.0  # when the latest state was read, on the monotonic clock

    def read(self) -> None:
        """Copies the recent events and, once a state period, encodes the state message; runs on a background
        thread."""
        with self.console.lock:
            self.events = tuple(self.console.events)

        if time.monotonic() - self.state_at >= self.STATE_PERIOD_S:
            self.state_message = json.dumps({"type": "state", "data": self.console.snapshot()})
            self.state_at = time.monotonic()


class LiveStream:
    """What one browser's socket is sent: the new events, each new state and fresh previews."""

    SEND_TIMEOUT_S = 2  # for one message to be sent; a socket that takes longer is closed
    PREVIEW_MAX_AGE_S = 0.5  # older preview frames are not sent

    def __init__(self, console: Console, live_feed: LiveFeed, socket: web.WebSocketResponse, after_id: int) -> None:
        self.console = console
        self.live_feed = live_feed
        self.socket = socket
        self.after_id = after_id  # of the last event sent
        self.sent_frame_numbers: dict[str, int] = {}  # of the last preview sent of each camera role
        self.sent_state_message: str | None = None  # the state message sent last

    async def run(self) -> None:
        """Sends until the socket closes; a socket that cannot take a message is closed."""
        try:
            while not self.socket.closed:
                await self.send_events()
                await self.send_state()
                await self.send_previews()
                await asyncio.sleep(LiveRoutes.FRAME_PERIOD_S)
        except (ConnectionError, TimeoutError):
            await self.socket.close()

    async def send_events(self) -> None:
        """Sends the events after the last one sent."""
        for event in [event for event in self.live_feed.events if event["id"] > self.after_id]:
            await self.send(self.socket.send_json(event))
            self.after_id = event["id"]

    async def send_state(self) -> None:
        """Sends the state message unless it was sent."""
        state_message = self.live_feed.state_message
        if state_message is not None and state_message is not self.sent_state_message:
            await self.send(self.socket.send_str(state_message))
            self.sent_state_message = state_message

    async def send_previews(self) -> None:
        """Sends each camera's preview that is fresh and was not sent, as its camera's index and its JPEG."""
        for index, camera in enumerate(self.console.robot.CAMERAS):
            frame = self.console.preview.get(camera.role)
            if frame is None or frame.frame_number == self.sent_frame_numbers.get(camera.role):
                continue
            if time.monotonic() - frame.received_monotonic_s < self.PREVIEW_MAX_AGE_S:
                await self.send(self.socket.send_bytes(bytes([index]) + frame.jpeg))
                self.sent_frame_numbers[camera.role] = frame.frame_number

    async def send(self, send_awaitable: Awaitable[None]) -> None:
        """Waits for one message to be sent, at most SEND_TIMEOUT_S."""
        await asyncio.wait_for(send_awaitable, self.SEND_TIMEOUT_S)
