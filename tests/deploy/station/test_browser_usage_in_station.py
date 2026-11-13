import json
import time
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import BrowserContext, Page, WebSocketRoute, expect

CONTEXT_USAGE = '.control-panel .performance > div:has-text("Context usage") strong'  # the panel's context usage

pytestmark = pytest.mark.usefixtures("station_login")


def test_claude_code_generated_tokens_context_and_incomplete_usage_survive_live_and_history_views(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    snapshot = context.request.get("/api/state").json()
    task = {"id": "d" * 32, "instruction": "Offline Claude Code usage", "model": "claude_code"}
    socket: WebSocketRoute | None = None
    id = 0

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws

        ws.send(
            json.dumps(
                {
                    "type": "state",
                    "data": {
                        **snapshot,
                        "task": task,
                        "model": task["model"],
                        "busy": True,
                        "phase": "Thinking",
                        "active_turn": {"attempt": "cc1"},
                    },
                }
            )
        )

    page.route_web_socket("**/api/live*", serve_live)

    def send(type: str, data: dict[str, Any]) -> None:
        nonlocal id
        id += 1
        socket.send(
            json.dumps(
                {
                    "id": id,
                    "task_id": task["id"],
                    "type": type,
                    "time": time.time(),
                    "data": {"attempt": "cc1", "turn_number": 1, **data},
                }
            )
        )

        if type == "performance":
            socket.send(
                json.dumps(
                    {
                        "type": "state",
                        "data": {
                            **snapshot,
                            "task": task,
                            "model": task["model"],
                            "busy": not data.get("final"),
                            "phase": "Paused" if data.get("final") else "Thinking",
                            "active_turn": {"attempt": "cc1"},
                            "performance": data,
                        },
                    }
                )
            )

    panel = page.locator(".control-panel .performance")
    generated = panel.locator(":scope > div", has_text="Generated tokens").locator("strong")
    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    send("turn_started", {"phase": "Thinking"})
    send("reasoning_delta", {"text": "Compare the wrist views."})
    thought = page.locator('[data-turn-number="1"] .thought-toggle')

    usage = {
        "input": 29137,
        "output": 2498,
        "context": {"input": 3825, "context_limit": 260096},
        "partial": False,
        "final": True,
    }
    live = {**usage, "input": 1025, "output": 55, "partial": True, "final": False}
    send("usage", live)
    send("performance", {"output_tokens": 55, "elapsed_s": 2.2, "partial": True, "final": False, "status": "streaming"})
    expect(thought).to_have_text("Reasoning · 55 tokens")
    expect(generated).to_have_text("55")
    # The separately measured context fills the panel, not the call's input.
    expect(page.locator(CONTEXT_USAGE)).to_have_text("3,825 / 260,096")
    expect(panel).not_to_contain_text("Incomplete usage")

    send("usage", {**live, "output": 110})
    send(
        "performance", {"output_tokens": 110, "elapsed_s": 3.7, "partial": True, "final": False, "status": "streaming"}
    )
    expect(thought).to_have_text("Reasoning · 110 tokens")
    expect(generated).to_have_text("110")
    page.screenshot(path=tmp_path / "claude-code-live.png", full_page=True)

    send("usage", usage)
    expect(thought).to_have_text("Reasoning · 2,498 tokens")
    expect(page.locator(CONTEXT_USAGE)).to_have_text("3,825 / 260,096")
    # A context still being measured is neither the stale measurement nor the call's input.
    send("usage", {**usage, "context": {"input": None, "context_limit": 260096}})
    expect(page.locator(CONTEXT_USAGE)).to_have_text("Not provided")

    incomplete = {**usage, "output": None, "partial": True}
    send("usage", incomplete)
    incomplete_performance = {
        "output_tokens": 110,
        "elapsed_s": 5.5,
        "partial": True,
        "final": True,
        "status": "interrupted",
    }
    send("performance", incomplete_performance)
    expect(generated).to_have_text("≥ 110")
    expect(panel).to_contain_text("Incomplete usage · confirmed counts are lower bounds")
    expect(panel).to_contain_text("Interrupted · partial statistics")
    # Without the call's output count the thought shows the streamed one.
    expect(thought).to_have_text("Reasoning · ≥ 110 tokens")
    expect(page.locator(CONTEXT_USAGE)).to_have_text("3,825 / 260,096")

    saved = [
        {**e, "id": i + 1, "task_id": task["id"], "data": {"attempt": "cc1", "turn_number": 1, **e["data"]}}
        for i, e in enumerate(
            [
                {"type": "phase", "time": 1700000000, "data": {"phase": "Thinking"}},
                {
                    "type": "model_output",
                    "time": 1700000007.1,
                    "data": {"reasoning": "Compare the wrist views.", "answer": ""},
                },
                {"type": "usage", "data": incomplete},
                {"type": "performance", "data": incomplete_performance},
                {"type": "phase", "data": {"phase": "Interrupted"}},
            ]
        )
    ]
    page.route("**/api/history", lambda route: route.fulfill(json=[{**task, "created": time.time()}]))
    page.route(f"**/api/history/{task['id']}/events*", lambda route: route.fulfill(json=saved))
    page.get_by_role("button", name="History", exact=True).click()
    page.locator(".history-list button").first.click()
    expect(page.locator(".turn-card .thought-toggle")).to_have_text("Thought for 7.1 s · ≥ 110 tokens")
    expect(page.locator(CONTEXT_USAGE)).to_have_text("3,825 / 260,096")
    page.screenshot(path=tmp_path / "claude-code-usage.png", full_page=True)


def test_context_usage_follows_the_latest_measured_turn_through_a_compaction_live_and_in_history(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    snapshot = context.request.get("/api/state").json()
    task = {"id": "e" * 32, "instruction": "Offline context compaction", "model": "claude_code"}
    socket: WebSocketRoute | None = None
    id = 0
    events: list[dict[str, Any]] = []

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws
        ws.send(
            json.dumps(
                {
                    "type": "state",
                    "data": {**snapshot, "task": task, "model": task["model"], "busy": False, "phase": "Paused"},
                }
            )
        )

    page.route_web_socket("**/api/live*", serve_live)

    def send(turn: int, type: str, data: dict[str, Any]) -> None:
        nonlocal id
        id += 1
        event = {
            "id": id,
            "task_id": task["id"],
            "type": type,
            "time": time.time(),
            "data": {"attempt": f"cc{turn}", "turn_number": turn, **data},
        }
        events.append(event)
        socket.send(json.dumps(event))

    def usage(measured: int | None) -> dict[str, Any]:
        return {
            "input": 29137,
            "output": 2498,
            "context": {"input": measured, "context_limit": 260096},
            "partial": False,
            "final": True,
        }

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    send(1, "turn_started", {"phase": "Completed"})
    send(1, "usage", usage(3825))
    send(2, "turn_started", {"phase": "Completed"})
    send(2, "usage", usage(5120))
    expect(page.locator(CONTEXT_USAGE)).to_have_text("5,120 / 260,096")

    send(3, "turn_started", {"phase": "Compacting"})
    send(3, "reasoning", {"text": "Compacting the conversation."})
    send(3, "usage", {**usage(None), "final": False})
    # Until the compacted context is measured, the panel keeps the latest measurement.
    expect(page.locator('[data-turn-number="3"] .thought-toggle')).to_have_text("Reasoning · 2,498 tokens")
    expect(page.locator(CONTEXT_USAGE)).to_have_text("5,120 / 260,096")
    send(3, "usage", usage(1210))
    expect(page.locator(CONTEXT_USAGE)).to_have_text("1,210 / 260,096")

    page.route("**/api/history", lambda route: route.fulfill(json=[{**task, "created": time.time()}]))
    page.route(f"**/api/history/{task['id']}/events*", lambda route: route.fulfill(json=events))
    page.get_by_role("button", name="History", exact=True).click()
    page.locator(".history-list button").first.click()
    expect(page.locator(CONTEXT_USAGE)).to_have_text("1,210 / 260,096")
    page.screenshot(path=tmp_path / "context-after-compaction.png", full_page=True)
