import json
import re
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import BrowserContext, Page, WebSocketRoute, expect

from waiting import Waiting

EXPECT_TIMEOUT_S = 5.0  # how long a poll looks for its value: as long as Playwright's assertions wait by default
POLL_S = 0.1  # between two looks of a poll
SCROLL_TOP_SCRIPT = "(el) => el.scrollTop"  # where a scrolled element shows its content from

pytestmark = pytest.mark.usefixtures("station_login")


def test_streamed_cards_preserve_expansion_scroll_context_usage_and_stable_numbering(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    snapshot = context.request.get("/api/state").json()
    socket: WebSocketRoute | None = None
    event_id = 0
    task = {
        "id": "a" * 32,
        "instruction": "Observe the workspace and keep both arms still",
        "created": 1700000000,
        "model": "demo",
    }
    state = {
        **snapshot,
        "task": task,
        "busy": True,
        "phase": "Thinking",
        "active_turn": {"attempt": "a260"},
        "turn_number": 260,
    }

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws
        ws.send(json.dumps({"type": "state", "data": state}))

    page.route_web_socket("**/api/live*", serve_live)

    def send(type: str, data: dict[str, Any]) -> None:
        nonlocal event_id
        event_id += 1
        socket.send(
            json.dumps(
                {
                    "id": event_id,
                    "task_id": task["id"],
                    "type": type,
                    "data": {"attempt": "a260", "turn_number": 260, "started_at": 1700000300, **data},
                }
            )
        )

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    send("turn_started", {"phase": "Thinking"})
    card = page.locator('[data-turn-number="260"]')
    details = card.locator(".details-toggle")
    expect(card.locator(".turn-head")).to_contain_text("Turn 260")
    expect(details).to_have_count(0)

    send("observation", {"images": {}, "state": {"left_arm": {"gripper": 0.25}}})
    expect(details).to_have_text("Observation & State")
    expect(card.locator(".turn-details")).to_have_count(0)
    details.click()
    expect(card.locator(".turn-details .turn-json")).to_contain_text('"gripper": 0.25')
    details.click()
    expect(card.locator(".turn-details")).to_have_count(0)

    send("reasoning_delta", {"text": "Reviewing the measured state. "})
    expect(card.locator(".thought-toggle")).to_have_attribute("aria-expanded", "false")
    card.locator(".thought-toggle").click()
    send("reasoning_delta", {"text": "Checking both wrists."})
    expect(card.locator(".thought-toggle")).to_have_attribute("aria-expanded", "true")
    expect(card.locator(".thought")).to_contain_text("Checking both wrists.")
    expect(card.locator(".thought-toggle")).to_have_text("Reasoning")

    context_usage = page.locator(".control-panel .performance > div", has_text="Context usage").locator("strong")
    expect(context_usage).to_have_text("Not provided")
    send("capacity", {"input": 1898, "context_limit": 260096})
    send("usage", {"input": 1898, "output": 202})
    expect(context_usage).to_have_text("1,898 / 260,096")
    expect(card.locator(".thought-toggle")).to_have_text("Reasoning · 202 tokens")

    send("answer_delta", {"text": "Keep both arms steady.\n" + "Long output " * 300})
    expect(card.locator(".turn-text")).to_be_visible()
    page.locator(".conversation-scroll").evaluate(
        "(el) => { el.scrollTop = 0; el.dispatchEvent(new Event('scroll')); }"
    )
    expect(page.get_by_role("button", name="Latest", exact=True)).to_be_visible()
    before = page.locator(".conversation-scroll").evaluate(SCROLL_TOP_SCRIPT)
    send("answer_delta", {"text": "\nMore text arrives."})
    Waiting.until(
        lambda: page.locator(".conversation-scroll").evaluate(SCROLL_TOP_SCRIPT) == before, EXPECT_TIMEOUT_S, POLL_S
    )
    assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")

    page.get_by_role("button", name="Latest", exact=True).click()
    send("turn_started", {"attempt": "a261", "turn_number": 261, "phase": "Thinking"})
    send("capacity", {"attempt": "a261", "turn_number": 261, "input": 2400, "context_limit": 260096})
    next = page.locator('[data-turn-number="261"]')
    # The panel shows the latest request's context, never a sum over turns.
    expect(context_usage).to_have_text("2,400 / 260,096")
    expect(next).to_have_class(re.compile("turn-active"))
    socket.send(
        json.dumps({"type": "state", "data": {**state, "active_turn": {"attempt": "a261"}, "turn_number": 261}})
    )
    page.screenshot(path=tmp_path / "pear-station-turn-cards.png", full_page=True)
