import json
import time
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import BrowserContext, Locator, Page, WebSocketRoute, expect

pytestmark = pytest.mark.usefixtures("station_login")


def test_live_vllm_claude_code_cancellation_task_switch_and_history_retain_honest_metrics(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    snapshot = context.request.get("/api/state").json()
    socket: WebSocketRoute | None = None
    id = 0
    task = {"id": "b" * 32, "instruction": "Offline telemetry demonstration", "model": "vllm"}
    state = {
        **snapshot,
        "task": task,
        "model": "vllm",
        "busy": True,
        "phase": "Thinking",
        "performance": None,
        "active_turn": {"attempt": "t1"},
    }

    def serve_live(ws: WebSocketRoute) -> None:
        nonlocal socket
        socket = ws
        ws.send(json.dumps({"type": "state", "data": state}))

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
                    "data": {"attempt": "t1", "turn_number": 1, **data},
                }
            )
        )

    def update(patch: dict[str, Any]) -> None:
        nonlocal state
        state = {**state, **patch}
        socket.send(json.dumps({"type": "state", "data": state}))

    panel = page.locator(".control-panel .performance")

    def row(label: str) -> Locator:
        return panel.locator(":scope > div", has_text=label).locator("strong")

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    send("turn_started", {"phase": "Thinking"})
    send("reasoning_delta", {"text": "Reading the wrist views."})
    thought = page.locator('[data-turn-number="1"] .thought-toggle')
    expect(row("Generated tokens")).to_have_text("Pending")
    expect(row("Client decode throughput")).to_have_text("Not provided")
    expect(row("Context usage")).to_have_text("Not provided")

    vllm = {
        "request_id": "r1",
        "output_tokens": 55,
        "client_decode_throughput_tps": None,
        "elapsed_s": 3.5,
        "final": False,
    }
    send("performance", vllm)
    update({"performance": vllm})
    expect(row("Generated tokens")).to_have_text("55")
    expect(row("Client decode throughput")).to_have_text("Not provided")
    expect(thought).to_have_text("Reasoning · 55 tokens")
    send("usage", {"input": 1000, "output": 55, "final": False})
    expect(row("Context usage")).to_have_text("1,000")
    send("usage", {"input": 1000, "output": 55, "context_limit": 32768, "final": False})
    expect(row("Context usage")).to_have_text("1,000 / 32,768")

    growing = {**vllm, "output_tokens": 110, "client_decode_throughput_tps": 110, "elapsed_s": 4}
    send("usage", {"input": 1000, "output": 110, "context_limit": 32768, "final": False})
    send("performance", growing)
    update({"performance": growing})
    expect(thought).to_have_text("Reasoning · 110 tokens")
    expect(row("Generated tokens")).to_have_text("110")
    expect(row("Client decode throughput")).to_have_text("110.00 tok/s")
    growing = {**growing, "output_tokens": 175, "client_decode_throughput_tps": 120, "elapsed_s": 4.5}
    send("usage", {"input": 1000, "output": 175, "context_limit": 32768, "final": False})
    send("performance", growing)
    update({"performance": growing})
    expect(row("Client decode throughput")).to_have_text("120.00 tok/s")
    expect(row("Generated tokens")).to_have_text("175")
    page.screenshot(path=tmp_path / "performance-vllm-live.png", full_page=True)

    final = {**growing, "decode_throughput_tps": 1800, "final": True, "status": "completed"}
    send("performance", final)
    send("execution", {"status": "completed", "settled": True})
    update({"busy": False, "phase": "Paused", "performance": final})
    expect(row("Client decode throughput")).to_have_text("120.00 tok/s")
    expect(row("Inference elapsed")).to_have_text("4.50 s")

    saved = [
        {**e, "id": i + 1, "task_id": task["id"], "data": {"attempt": "t1", "turn_number": 1, **e["data"]}}
        for i, e in enumerate(
            [
                {"type": "turn_started", "data": {"phase": "Completed"}},
                {"type": "reasoning", "data": {"text": "Reading the wrist views."}},
                {"type": "usage", "data": {"input": 1000, "output": 175, "context_limit": 32768, "final": True}},
                {"type": "performance", "data": final},
            ]
        )
    ]
    page.route("**/api/history", lambda route: route.fulfill(json=[{**task, "created": time.time()}]))
    page.route(f"**/api/history/{task['id']}/events*", lambda route: route.fulfill(json=saved))
    page.get_by_role("button", name="History", exact=True).click()
    page.locator(".history-list button").first.click()
    expect(page.locator(".turn-card .thought-toggle")).to_have_text("Reasoning · 175 tokens")
    expect(row("Client decode throughput")).to_have_text("120.00 tok/s")
    expect(row("Context usage")).to_have_text("1,000 / 32,768")

    page.get_by_role("button", name="Workspace", exact=True).click()
    update(
        {
            "task": {**task, "id": "c" * 32, "model": "claude_code"},
            "model": "claude_code",
            "busy": True,
            "performance": None,
            "phase": "Thinking",
        }
    )
    # A new task starts without the previous task's figures.
    expect(row("Generated tokens")).to_have_text("Pending")
    expect(row("Client decode throughput")).to_have_text("Not provided")
    expect(row("Context usage")).to_have_text("Not provided")
    claude_code = {
        "output_tokens": 100,
        "client_decode_throughput_tps": 25,
        "elapsed_s": 10,
        "final": False,
        "status": "streaming",
    }
    update({"performance": claude_code})
    expect(row("Client decode throughput")).to_have_text("25.00 tok/s")
    claude_code = {**claude_code, "final": True, "status": "interrupted", "partial": True}
    update({"performance": claude_code, "busy": False, "phase": "Paused"})
    expect(row("Generated tokens")).to_have_text("≥ 100")
    expect(row("Inference elapsed")).to_have_text("10.00 s")
    expect(row("Client decode throughput")).to_have_text("25.00 tok/s")
    expect(panel).to_contain_text("Interrupted · partial statistics")
    page.screenshot(path=tmp_path / "performance-claude-code.png", full_page=True)
    # Server-only statistics must not stand in for a missing client measurement.
    update({"performance": {"decode_throughput_tps": 1800, "final": True}})
    expect(row("Client decode throughput")).to_have_text("Not provided")

    update({"performance": claude_code})
    expect(row("Client decode throughput")).to_have_text("25.00 tok/s")
    socket.close()
    expect(panel).to_contain_text("Connection lost")
    expect(row("Client decode throughput")).to_have_text("Not provided")
    expect(row("Generated tokens")).to_have_text("Not provided")
