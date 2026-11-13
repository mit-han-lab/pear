import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from aiohttp import web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.periodic_thread import PeriodicThread
from pear.deploy.station.gui.server.command_cache import CommandCache
from pear.deploy.station.gui.server.console_routes import ConsoleRoutes
from pear.deploy.station.gui.server.history_routes import HistoryRoutes
from pear.deploy.station.gui.server.live_routes import LiveRoutes
from pear.deploy.station.gui.server.media_routes import MediaRoutes
from pear.deploy.station.gui.server.security import Security
from pear.deploy.station.gui.server.service_identity import ServiceIdentity


class ServerApp:
    """The aiohttp application of one console: the checks before every route, the routes, the page and the background
    threads."""

    MAX_BODY_BYTES = 16384  # of a request's body; an upload's route reads more
    MODEL_POLL_S = 5  # between two checks of the selected model's connection
    UPLOAD_CLEANUP_S = 60  # between two removals of the expired draft images
    STATIC_DIR = Path(__file__).parents[1] / "static"  # the page: its index and the assets the index loads
    ASSETS_FOLDER = "assets"  # in the page's folder
    INDEX_NAME = "index.html"  # in the page's folder

    def __init__(self, console: Console, token: str, identity: ServiceIdentity) -> None:
        self.console = console
        self.token = token  # the browser token a login gives
        self.identity = identity  # who the service is, which /health answers
        self.command_cache = CommandCache(console)
        self.live_routes = LiveRoutes(console)  # its feed is read by a background thread of the application

    def build(self) -> web.Application:
        """Returns the application: the security checks before every route, the routes, the page and the background
        threads."""
        security = Security(self.token, self.console.robot.NAME)
        app = web.Application(middlewares=[security.middleware], client_max_size=self.MAX_BODY_BYTES)
        app.on_response_prepare.append(security.prepare)

        app.router.add_post(Security.OPEN_PATH, security.login)
        ConsoleRoutes(self.console, self.command_cache, self.identity).add_routes(app.router)
        self.live_routes.add_routes(app.router)
        HistoryRoutes(self.console).add_routes(app.router)
        MediaRoutes(self.console).add_routes(app.router)
        app.router.add_get("/", self.index)
        if (self.STATIC_DIR / self.ASSETS_FOLDER).exists():
            app.router.add_static(f"/{self.ASSETS_FOLDER}", self.STATIC_DIR / self.ASSETS_FOLDER)

        app.on_cleanup.append(self.close)
        app.cleanup_ctx.append(self.poll)

        return app

    async def index(self, request: web.Request) -> web.FileResponse:
        """Answers the page."""
        return web.FileResponse(self.STATIC_DIR / self.INDEX_NAME)

    async def poll(self, app: web.Application) -> AsyncIterator[None]:
        """Reads the live feed, checks the selected model and removes the expired drafts in the background while the
        application runs."""
        periodic_threads = [
            PeriodicThread(self.live_routes.live_feed.read, LiveRoutes.FRAME_PERIOD_S, "gui-live-feed"),
            PeriodicThread(self.console.check_model, self.MODEL_POLL_S, "gui-model-check"),
            PeriodicThread(self.console.uploads.cleanup, self.UPLOAD_CLEANUP_S, "gui-upload-cleanup"),
        ]

        yield

        # no step starts any more; one in progress is not waited for, and may still finish
        for periodic_thread in periodic_threads:
            periodic_thread.exit_event.set()

    async def close(self, app: web.Application) -> None:
        """Waits for the commands that run, then closes the console."""
        await self.command_cache.drain()
        await asyncio.to_thread(self.console.close)
