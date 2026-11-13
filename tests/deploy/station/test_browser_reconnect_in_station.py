import json
import time

import pytest
from playwright.sync_api import BrowserContext, Page, Request, Route, WebSocketRoute, expect

EXPECT_TIMEOUT_S = 5.0  # how long a poll looks for its value: as long as Playwright's assertions wait by default
POLL_MS = 100  # between two looks of a poll, while the browser's messages reach the test's handlers
SOCKET_LOSS_CODE = 1011  # the close code of the stand-in's lost connection: the server met an unexpected condition
OFFLINE_TIMEOUT_MS = 10000  # for the page to show the connection lost once the browser is offline
ONLINE_TIMEOUT_MS = 15000  # for the page to show it connected again once the browser is back online

pytestmark = pytest.mark.usefixtures("station_login")


def test_browser_reconnects_automatically_after_transport_loss_without_enabling_motors_or_resuming(
    page: Page, context: BrowserContext
) -> None:
    initial = context.request.get("/api/state").json()
    state = {**initial, "owner": "", "robot": {"status": "OFF"}, "busy": False, "task": None, "phase": "Disabled"}
    socket: WebSocketRoute | None = None
    connections = 0
    commands: list[str] = []

    def answer_claim(route: Route) -> None:
        state["owner"] = route.request.post_data_json["client_id"]
        route.fulfill(json={"owner": True})

    def record_command(route: Route) -> None:
        commands.append(route.request.post_data_json["command"])
        route.fulfill(json={"accepted": True})

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket, connections
        connections += 1
        socket = ws
        ws.send(json.dumps({"type": "state", "data": state}))
        ws.on_message(lambda message: ws.send(json.dumps({"type": "state", "data": state})))

    page.route("**/api/control/claim", answer_claim)
    page.route("**/api/command", record_command)
    page.route_web_socket("**/api/live*", serve_live)

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    socket.close(code=SOCKET_LOSS_CODE, reason="Stand-in network loss")

    # the page's claim and its new connection reach the handlers only while a call of the test waits on the browser
    deadline_s = time.monotonic() + EXPECT_TIMEOUT_S
    while connections != 2 and time.monotonic() < deadline_s:
        page.wait_for_timeout(POLL_MS)
    assert connections == 2

    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    assert commands == []
    expect(page.get_by_role("button", name="Reconnect cameras", exact=True)).to_be_enabled()
    page.get_by_role("button", name="Reconnect cameras", exact=True).click()
    assert commands == ["reconnect_cameras"]
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    expect(page.locator(".status-card strong")).to_have_text("Disabled")


def test_real_browser_socket_reconnects_after_offline_and_online_and_does_not_replay_commands(
    page: Page, context: BrowserContext
) -> None:
    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    commands: list[str] = []

    def record_command(r: Request) -> None:
        if r.url.endswith("/api/command"):
            commands.append(r.post_data_json["command"])

    page.on("request", record_command)

    context.set_offline(True)
    expect(page.get_by_text("Disconnected", exact=True)).to_be_visible(timeout=OFFLINE_TIMEOUT_MS)
    context.set_offline(False)
    expect(page.get_by_text("Connected", exact=True)).to_be_visible(timeout=ONLINE_TIMEOUT_MS)
    assert commands == []

    state = context.request.get("/api/state").json()
    assert state["robot"]["status"] == "OFF"
    assert state["task"] is None
