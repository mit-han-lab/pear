import json
import time
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Page, Route, WebSocketRoute, expect

pytestmark = pytest.mark.usefixtures("station_login")


@pytest.mark.parametrize("outcome", ["ready", "blocked", "stop"])
def test_manual_control_recovery_keeps_the_task_and_handles_its_outcome(
    page: Page, context: BrowserContext, tmp_path: Path, outcome: str
) -> None:
    initial = context.request.get("/api/state").json()
    state = {
        **initial,
        "owner": "",
        "robot": {"status": "FAULT"},
        "busy": False,
        "task": {"id": "a" * 32, "instruction": "Keep the original task", "model": "demo", "created": time.time()},
        "phase": "Fault",
        "stop_latched": outcome == "stop",
        "error": "Transient control fault",
        "cameras": {"error": None, "roles": {"left": {"age": 0}, "right": {"age": 0}, "top": {"age": 0}}},
    }
    socket: WebSocketRoute | None = None
    commands: list[str] = []

    def publish() -> None:
        socket.send(json.dumps({"type": "state", "data": state}))

    page.route("**/api/state", lambda route: route.fulfill(json=state))
    # The made-up task has no recordings.
    page.route("**/api/history/*/recordings", lambda route: route.fulfill(json=[]))
    page.route(
        "**/api/history/*/recordings/merged",
        lambda route: route.fulfill(
            json={"segments": {"full": [], "motion": []}, "pending": None, "current": False, "merged": None}
        ),
    )

    def answer_claim(route: Route) -> None:
        state["owner"] = route.request.post_data_json["client_id"]
        route.fulfill(json={"owner": True})

    page.route("**/api/control/claim", answer_claim)

    def run_command(route: Route) -> None:
        nonlocal state
        command = route.request.post_data_json["command"]
        commands.append(command)
        if command == "recover_control":
            state = {**state, "busy": True, "phase": "Recovering control", "error": None}
            publish()
        route.fulfill(json={"accepted": True})

    page.route("**/api/command", run_command)

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws
        publish()
        ws.on_message(lambda message: publish())

    page.route_web_socket("**/api/live*", serve_live)

    page.goto("/")
    expect(page.get_by_role("button", name="Recover control", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Home", exact=True)).to_be_disabled()
    if outcome == "ready":
        page.screenshot(path=tmp_path / "recovery-fault.png")
    page.get_by_role("button", name="Recover control", exact=True).click()
    expect(page.get_by_role("button", name="Recovering…", exact=True)).to_be_disabled()
    expect(page.locator(".user-bubble")).to_have_text("Keep the original task")
    assert commands == ["recover_control"]

    state = {
        **state,
        "busy": False,
        "robot": {"status": "FAULT" if outcome == "blocked" else "READY"},
        "phase": "Fault" if outcome == "blocked" else "Stopped" if outcome == "stop" else "Paused",
        "error": "RuntimeError: Control recovery failed: right motor 4: over temperature"
        if outcome == "blocked"
        else None,
    }
    publish()
    expect(page.locator(".user-bubble")).to_have_text("Keep the original task")
    assert commands == ["recover_control"]

    if outcome == "blocked":
        expect(page.get_by_role("alert")).to_contain_text("over temperature")
        expect(page.locator(".status-card")).to_contain_text("Control recovery failed:")
        expect(page.get_by_role("button", name="Recover control", exact=True)).to_be_enabled()
        expect(page.get_by_role("button", name="Resume task", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="Home", exact=True)).to_be_disabled()
    elif outcome == "stop":
        expect(page.get_by_role("button", name="Reset stop", exact=True)).to_be_enabled()
        expect(page.get_by_role("button", name="Home", exact=True)).to_be_disabled()
    else:
        expect(page.get_by_role("button", name="Recover control", exact=True)).to_have_count(0)
        expect(page.get_by_role("button", name="Resume task", exact=True)).to_be_enabled()
        expect(page.get_by_role("button", name="Home", exact=True)).to_be_enabled()
        page.screenshot(path=tmp_path / "recovery-paused.png")
        page.get_by_role("button", name="Resume task", exact=True).click()
        assert commands == ["recover_control", "resume"]
