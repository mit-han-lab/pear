import re
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Page, expect

from stand_in_cameras import StandInCameras
from stand_in_robots import StandInYAMRobot
from waiting import Waiting

# before a test opens the page: longer than the console's control lease, which the tab of the test before held
OPENING_WAIT_MS = 3200
TAKE_CONTROL_WAIT_MS = 3300  # before the reloaded page takes control: longer than the lease its first load held
CAMERA_COUNT = len(StandInYAMRobot.CAMERAS)  # the station's cameras: at both wrists and overhead
CAMERA_WIDTH = StandInCameras.FRAME_SIZE[0]  # of every camera's frame, which the previews keep
HOLD_TIMEOUT_MS = 15000  # for the motors to hold once enabled
COMPLETED_TURN_TIMEOUT_MS = 25000  # for the stand-in model's first turn to complete
PAUSE_TIMEOUT_MS = 20000  # for the task to pause
STEPS_TIMEOUT_S = 20.0  # for the resumed task to take a step
HOME_CHECK_TIMEOUT_MS = 20000  # for the check of the direct path Home to ask for confirmation
HOME_TIMEOUT_MS = 30000  # for the arms to come Home
DISABLE_TIMEOUT_MS = 15000  # for the motors to be disabled
EXPECT_TIMEOUT_S = 5.0  # how long a poll looks for its value: as long as Playwright's assertions wait by default
POLL_S = 0.1  # between two looks of a poll
LAPTOP_VIEWPORT = {"width": 1280, "height": 800}  # the window of a small laptop
CONNECTION_TIMEOUT_MS = 12000  # for the page to show its connection lost once offline, and back once online
TURN_NUMBERS_SCRIPT = "(nodes) => nodes.map((n) => n.getAttribute('data-turn-number'))"  # the numbers of the turns


@pytest.fixture(autouse=True)
def workspace(page: Page, station_login: None) -> None:
    """Opens the workspace in the test's page, connected, once the tab of the test before has lost control."""
    page.wait_for_timeout(OPENING_WAIT_MS)
    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()


def test_full_browser_task_pause_clear_home_stop_and_history_flow(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    page_errors: list[str] = []
    page.on("pageerror", lambda e: page_errors.append(e.message))
    expect(page.get_by_role("combobox", name="Arm configuration")).to_have_count(0)
    expect(page.get_by_role("combobox", name="Environment")).to_have_count(0)
    expect(page.get_by_role("button", name="Step Once", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Recover holding", exact=True)).to_have_count(0)
    # the station tells the page of no mode but the live one
    expect(page.get_by_text("LIVE ROBOT", exact=True)).to_be_visible()
    expect(page.get_by_role("combobox", name="Model", exact=True)).to_have_value("stand_in")
    expect(page.locator(".camera-grid img")).to_have_count(CAMERA_COUNT)
    Waiting.until(
        lambda: page.locator(".camera-grid img").evaluate_all(
            f"(imgs) => imgs.every((i) => i.naturalWidth === {CAMERA_WIDTH})"
        ),
        EXPECT_TIMEOUT_S,
        POLL_S,
    )
    page.evaluate("() => window.scrollTo(0, 0)")
    page.screenshot(path=tmp_path / "pear-station-workspace.png", full_page=True)

    page.get_by_role("button", name="Enable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Holding", timeout=HOLD_TIMEOUT_MS)
    instruction = page.get_by_role("textbox", name="Task instruction")
    instruction.fill("Lift both arms slightly, then return. Repeat until I pause.")
    page.get_by_role("button", name="Tags · 0", exact=True).click()
    page.get_by_role("combobox", name="Tags for the new task").fill("workspace-e2e")
    page.get_by_role("combobox", name="Tags for the new task").press("Enter")
    instruction.click()
    expect(page.get_by_role("dialog", name="Task tags")).to_have_count(0)
    expect(page.get_by_role("button", name="Tags · 1", exact=True)).to_be_visible()
    instruction.press("Enter")
    expect(page.get_by_role("button", name="Pause task", exact=True)).to_be_visible()
    started = context.request.get("/api/state").json()["task"]
    annotations = context.request.get(f"/api/history/{started['id']}/annotations").json()
    assert annotations["tags"] == ["workspace-e2e"]

    expect(page.locator(".result-chip.completed").first).to_be_visible(timeout=COMPLETED_TURN_TIMEOUT_MS)
    expect(page.locator(".action-card").first).to_be_visible()
    turn = page.locator(".turn-card").filter(has=page.locator(".result-chip.completed")).first
    turn.get_by_role("button", name="Observation & State", exact=True).click()
    expect(turn.locator(".turn-details img")).to_have_count(CAMERA_COUNT)
    expect(turn.locator(".turn-details .turn-json")).to_contain_text('"left_arm"')
    # The stand-in model reports no usage.
    expect(page.locator(".control-panel .performance > div", has_text="Context usage")).to_contain_text("Not provided")
    for label in ["Exact model input", "Action", "Token usage & capacity"]:
        expect(page.locator("summary").filter(has_text=re.compile("^" + label + "$"))).to_have_count(0)
    page.evaluate("() => window.scrollTo(0, 0)")
    page.screenshot(path=tmp_path / "pear-station-running.png", full_page=True)

    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    before_step = context.request.get("/api/state").json()["total_steps"]
    page.get_by_role("button", name="Resume task", exact=True).click()
    Waiting.until(
        lambda: context.request.get("/api/state").json()["total_steps"] > before_step, STEPS_TIMEOUT_S, POLL_S
    )
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)

    numbers = page.locator(".turn-card").evaluate_all(TURN_NUMBERS_SCRIPT)
    page.reload()
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    Waiting.until(
        lambda: page.locator(".turn-card").evaluate_all(TURN_NUMBERS_SCRIPT) == numbers, EXPECT_TIMEOUT_S, POLL_S
    )

    page.wait_for_timeout(TAKE_CONTROL_WAIT_MS)
    page.get_by_role("button", name="Take control", exact=True).click()
    second = context.new_page()
    second.goto("/")
    expect(second.get_by_text("Read-only tab")).to_be_visible()
    second.get_by_role("button", name="STOP MOTION", exact=True).click()
    expect(page.get_by_role("button", name="Reset stop")).to_be_visible()
    second.close()
    page.get_by_role("button", name="Reset stop").click()
    page.get_by_role("button", name="New task", exact=True).click()
    expect(page.get_by_text("What should we work on?")).to_be_visible()

    page.get_by_role("button", name="Home", exact=True).click()
    page.get_by_role("button", name="Check Home", exact=True).click()
    expect(page.get_by_text("Confirm direct Home")).to_be_visible(timeout=HOME_CHECK_TIMEOUT_MS)
    page.get_by_role("button", name="Workspace clear — Begin Home").click()
    expect(page.locator(".status-card strong")).to_have_text("Home complete", timeout=HOME_TIMEOUT_MS)
    page.get_by_role("button", name="Disable Motors", exact=True).click()
    page.locator(".modal").get_by_role("button", name="Disable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Disabled", timeout=DISABLE_TIMEOUT_MS)

    page.get_by_role("button", name="History", exact=True).click()
    expect(page.locator(".history-list button").first).to_be_visible()
    page.locator(".history-list button").first.click()
    expect(page.locator(".user-bubble")).to_have_text("Lift both arms slightly, then return. Repeat until I pause.")
    expect(page.get_by_role("link", name="Export task")).to_be_visible()
    assert page_errors == []


def test_small_laptop_camera_enlargement_and_disconnected_controls(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    page.set_viewport_size(LAPTOP_VIEWPORT)
    expect(page.get_by_role("button", name="STOP MOTION", exact=True)).to_be_in_viewport()
    expect(page.get_by_role("button", name="Send and start", exact=True)).to_be_in_viewport()
    expect(page.get_by_role("button", name="Enable Motors", exact=True)).to_be_in_viewport()

    instruction = page.get_by_role("textbox", name="Task instruction")
    instruction.fill("first line")
    instruction.press("Shift+Enter")
    instruction.press("Enter")
    expect(instruction).to_have_value("first line\n")

    page.get_by_role("button", name="Expand Overhead", exact=True).click()
    expect(page.locator(".camera.large")).to_be_visible()
    page.get_by_role("button", name="Close camera", exact=True).click()
    page.evaluate("() => window.scrollTo(0, 0)")
    page.screenshot(path=tmp_path / "pear-station-laptop.png", full_page=True)
    assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")

    context.set_offline(True)
    expect(page.get_by_text("Disconnected", exact=True)).to_be_visible(timeout=CONNECTION_TIMEOUT_MS)
    # Back online the page reconnects by itself and the alert goes, so its button is tried while still offline.
    page.get_by_role("button", name="Reconnect", exact=True).click()
    expect(page.get_by_text("Disconnected", exact=True)).to_be_visible()
    context.set_offline(False)
    expect(page.get_by_text("Connected", exact=True)).to_be_visible(timeout=CONNECTION_TIMEOUT_MS)
    expect(page.locator(".status-card strong")).to_have_text("Disabled")
