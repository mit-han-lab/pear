import hmac
import secrets
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from aiohttp import web

from pear.deploy.station.gui.server.error_responses import ErrorResponses
from pear.deploy.station.gui.server.launcher import ServiceConfig


class Security:
    """The checks every request passes (this machine's address, the page's origin, a session for the API), the login
    that opens a session, and the headers of every response."""

    LOOPBACK_HOSTS = (ServiceConfig.LOOPBACK, "localhost", "::1")  # as a URL's host, without the IPv6 brackets
    OPEN_PATH = "/api/auth"  # the login: the one path of the API that a browser without a session may use
    API_PREFIX = "/api/"  # of the paths that need a session
    MAX_SESSIONS = 32  # sessions kept open; a login beyond them ends the oldest
    SESSION_BYTES = 32  # of randomness in a new session's id
    LOGIN_HINT = "Open the workspace with the URL that `pear-station gui` prints"  # the answer without a session
    RESPONSE_STARTED_KEY = web.RequestKey("response_started", bool)  # set on a request once its response has begun
    # added to every response
    HEADERS = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": (
            "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        ),
    }

    def __init__(self, token: str, robot_name: str) -> None:
        self.token = token  # the browser token a login gives; no login succeeds without one
        # services for different robots may share a browser host: their sessions are kept apart
        self.cookie_name = f"{robot_name}_session"
        self.sessions: OrderedDict[str, None] = OrderedDict()  # the ids of the open sessions, the oldest first

    @web.middleware
    async def middleware(
        self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        """Checks the request, then returns its handler's response or the answer to the handler's failure."""
        self.check(request)

        try:
            response = await handler(request)
        except web.HTTPException:
            raise
        except Exception as error:
            if request.get(self.RESPONSE_STARTED_KEY):
                raise  # a streamed response has begun, so no second response can follow it
            answer, status = ErrorResponses.answer(error, f"{request.method} {request.path}")
            response = web.json_response(answer, status=status)

        return response

    def check(self, request: web.Request) -> None:
        """Refuses a request for another host, a request from another origin, and one to the API without a session."""
        if request.url.host not in self.LOOPBACK_HOSTS:
            raise web.HTTPForbidden(text="Loopback access only")
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text="Cross-origin request rejected")
        if request.path.startswith(self.API_PREFIX) and request.path != self.OPEN_PATH:
            if request.cookies.get(self.cookie_name) not in self.sessions:
                raise web.HTTPUnauthorized(text=self.LOGIN_HINT)

    async def prepare(self, request: web.Request, response: web.StreamResponse) -> None:
        """Adds the headers to a response about to begin, and marks the request so a later failure is not answered
        again."""
        response.headers.update(self.HEADERS)
        request[self.RESPONSE_STARTED_KEY] = True

    async def login(self, request: web.Request) -> web.Response:
        """Opens a session for a browser that gives the token, unless the browser has one open."""
        body = await request.json()
        if not self.token or not hmac.compare_digest(str(body.get("token", "")), self.token):
            raise web.HTTPUnauthorized()
        if request.cookies.get(self.cookie_name) in self.sessions:
            return web.json_response({"authenticated": True})

        session_id = secrets.token_urlsafe(self.SESSION_BYTES)
        if len(self.sessions) >= self.MAX_SESSIONS:
            self.sessions.popitem(last=False)
        self.sessions[session_id] = None
        response = web.json_response({"authenticated": True})
        response.set_cookie(self.cookie_name, session_id, httponly=True, samesite="Strict", path="/")

        return response
