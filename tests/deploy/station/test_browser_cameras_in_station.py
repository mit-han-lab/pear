import json
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Page, Route, WebSocketRoute, expect

from waiting import Waiting

LEASE_EXPIRY_WAIT_MS = 3300  # longer than the console's control lease: the tab of the test before no longer holds it
EXPECT_TIMEOUT_S = 5.0  # how long a poll looks for its value: as long as Playwright's assertions wait by default
POLL_S = 0.1  # between two looks of a poll

pytestmark = pytest.mark.usefixtures("station_login")


def test_one_failed_camera_leaves_healthy_previews_visible_and_gates_motor_enable(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    snapshot = context.request.get("/api/state").json()
    socket: WebSocketRoute | None = None
    state = {
        **snapshot,
        "owner": "",
        "robot": {"status": "OFF"},
        "task": None,
        "busy": False,
        "phase": "Disabled",
        "cameras": {
            "error": "top: Device disconnected",
            "errors": {"top": "Device disconnected"},
            "roles": {"left": {"age": 0.03}, "top": {"age": 99}, "right": {"age": 0.03}},
        },
    }

    def answer_claim(route: Route) -> None:
        state["owner"] = route.request.post_data_json["client_id"]
        route.fulfill(json={"claimed": True})

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws
        ws.send(json.dumps({"type": "state", "data": state}))
        ws.on_message(lambda message: ws.send(json.dumps({"type": "state", "data": state})))

    page.route("**/api/control/claim", answer_claim)
    page.route_web_socket("**/api/live*", serve_live)
    commands: list[str] = []

    def record_command(route: Route) -> None:
        commands.append(route.request.post_data_json["command"])
        route.fulfill(json={"accepted": True})

    page.route("**/api/command", record_command)

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    cards = page.locator(".camera-grid .camera")
    expect(cards.nth(0).locator(".signal-overlay")).to_have_count(0)
    expect(cards.nth(1).locator(".signal-overlay")).to_have_text("Device disconnected")
    expect(cards.nth(2).locator(".signal-overlay")).to_have_count(0)
    expect(page.get_by_role("button", name="Enable Motors", exact=True)).to_be_disabled()

    reconnect = page.get_by_role("button", name="Reconnect cameras", exact=True)
    reconnect.click()
    assert commands == ["reconnect_cameras"]
    page.screenshot(path=tmp_path / "camera-partial-failure.png", full_page=True)

    state = {**state, "robot": {"status": "READY"}, "phase": "Holding"}
    socket.send(json.dumps({"type": "state", "data": state}))
    expect(reconnect).to_be_enabled()
    state = {**state, "phase": "Thinking", "busy": True}
    socket.send(json.dumps({"type": "state", "data": state}))
    expect(reconnect).to_be_disabled()


def test_explicit_camera_reconnect_completes_on_stand_in_hardware_and_leaves_motors_off(
    page: Page, context: BrowserContext
) -> None:
    page.wait_for_timeout(LEASE_EXPIRY_WAIT_MS)
    page.goto("/")
    reconnect = page.get_by_role("button", name="Reconnect cameras", exact=True)
    expect(reconnect).to_be_enabled()

    with page.expect_response(
        lambda r: r.url.endswith("/api/command") and r.request.post_data_json["command"] == "reconnect_cameras"
    ) as response:
        reconnect.click()
    assert response.value.ok

    def is_reconnected() -> bool:
        s = context.request.get("/api/state").json()
        return not s["busy"] and s["cameras"]["error"] is None

    Waiting.until(is_reconnected, EXPECT_TIMEOUT_S, POLL_S)
    expect(reconnect).to_be_enabled()
    # fresh cameras, which the page calls LIVE: the station tells it of no other mode
    expect(page.locator(".camera-grid .camera-live")).to_have_text(["LIVE", "LIVE", "LIVE"])

    state = context.request.get("/api/state").json()
    assert state["robot"]["status"] == "OFF"
    assert state["task"] is None
    assert state["cameras"]["error"] is None
