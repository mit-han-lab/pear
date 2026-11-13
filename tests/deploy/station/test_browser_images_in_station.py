import base64
import json
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, FilePayload, Page, Route, expect

from waiting import Waiting

# a tiny valid PNG, with names kept only in the page and the task's records: its base64 text and its bytes
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
PNG = base64.b64decode(PNG_BASE64)
LEASE_EXPIRY_WAIT_MS = 3300  # longer than the console's control lease: the tab of the test before no longer holds it
HOLD_TIMEOUT_MS = 60000  # for the motors to hold once enabled
STEPS_TIMEOUT_S = 20.0  # for the running task to take a step
PAUSE_TIMEOUT_MS = 20000  # for the task to pause
DISABLE_TIMEOUT_MS = 15000  # for the motors to be disabled
POLL_S = 0.1  # between two looks of a poll
MIN_EXPORT_BYTES = 100  # an export holds more than an empty archive


class ImageFiles:
    """The tiny PNG as the files the chooser of the task images takes."""

    @staticmethod
    def file(name: str) -> FilePayload:
        """Returns the PNG as the file ``name``."""
        return {"name": name, "mimeType": "image/png", "buffer": PNG}


@pytest.fixture(autouse=True)
def workspace(page: Page, station_login: None) -> None:
    """Opens the workspace in the test's page, connected and ready to take task images, once the tab of the test
    before has lost control."""
    page.wait_for_timeout(LEASE_EXPIRY_WAIT_MS)
    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Add images")).to_be_enabled()


def test_select_retry_drop_paste_remove_run_resume_history_and_export_images(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(e.message))
    page.get_by_label("Task instruction").fill("Match these goal images.")
    fail = True

    def upload_image(route: Route) -> None:
        nonlocal fail
        if fail:
            fail = False
            route.fulfill(
                status=409, content_type="application/json", body=json.dumps({"error": "Stand-in upload failure"})
            )
        else:
            route.continue_()

    page.route("**/api/images?*", upload_image)
    page.get_by_label("Choose task images").set_input_files(ImageFiles.file("first.png"))
    expect(page.get_by_text("Stand-in upload failure")).to_be_visible()
    expect(page.get_by_role("button", name="Send and start", exact=True)).to_be_disabled()
    page.get_by_role("button", name="Retry upload").click()
    expect(page.locator(".attachment-status.ready")).to_have_count(1)
    page.get_by_role("button", name="Preview task image 1").click()
    expect(page.get_by_role("dialog", name="Task image preview")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog", name="Task image preview")).to_have_count(0)

    page.locator(".composer").evaluate(
        """(el, data) => {
            const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'dropped.png', {type: 'image/png'}));
            el.dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer}));
        }""",
        PNG_BASE64,
    )
    page.get_by_label("Task instruction").evaluate(
        """(el, data) => {
            const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'pasted.png', {type: 'image/png'}));
            el.dispatchEvent(new ClipboardEvent('paste', {bubbles: true, cancelable: true, clipboardData: transfer}));
        }""",
        PNG_BASE64,
    )
    expect(page.locator(".attachment-status.ready")).to_have_count(3)
    page.get_by_role("button", name="Remove task image 2").click()
    expect(page.locator(".attachment-status.ready")).to_have_count(2)

    page.get_by_role("button", name="Enable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Holding", timeout=HOLD_TIMEOUT_MS)
    page.screenshot(path=tmp_path / "images-composer.png", full_page=True)
    page.get_by_role("button", name="Send and start", exact=True).click()
    expect(page.locator(".saved-images img")).to_have_count(2)
    Waiting.until(lambda: context.request.get("/api/state").json()["total_steps"] > 0, STEPS_TIMEOUT_S, POLL_S)
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    state = context.request.get("/api/state").json()
    assert [i["name"] for i in state["task"]["images"]] == ["first.png", "pasted.png"]

    page.get_by_role("button", name="Resume task", exact=True).click()
    Waiting.until(
        lambda: context.request.get("/api/state").json()["total_steps"] > state["total_steps"], STEPS_TIMEOUT_S, POLL_S
    )
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    page.screenshot(path=tmp_path / "images-task.png", full_page=True)

    page.get_by_role("button", name="New task", exact=True).click()
    expect(page.get_by_text("What should we work on?")).to_be_visible()
    expect(page.locator(".attachment")).to_have_count(0)
    page.get_by_role("button", name="Disable Motors", exact=True).click()
    page.locator(".modal").get_by_role("button", name="Disable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Disabled", timeout=DISABLE_TIMEOUT_MS)

    page.get_by_role("button", name="History", exact=True).click()
    page.locator(".history-list button").first.click()
    expect(page.locator(".saved-images img")).to_have_count(2)
    page.get_by_role("button", name="View task image 1").click()
    expect(page.get_by_role("dialog").locator("img")).to_be_visible()
    page.get_by_role("button", name="Close image preview").click()
    export_response = context.request.get(f"/api/history/{state['task']['id']}/export")
    assert export_response.ok
    assert len(export_response.body()) > MIN_EXPORT_BYTES
    assert errors == []


def test_invalid_files_and_four_image_limit_retain_the_draft(page: Page) -> None:
    page.get_by_label("Task instruction").fill("Keep this task text.")
    page.get_by_label("Choose task images").set_input_files(
        {"name": "invalid.png", "mimeType": "image/png", "buffer": b"invalid"}
    )
    expect(page.get_by_role("button", name="Retry upload")).to_be_visible()
    page.get_by_role("button", name="Remove task image 1").click()

    page.get_by_label("Choose task images").set_input_files(
        [ImageFiles.file("1.png"), ImageFiles.file("2.png"), ImageFiles.file("3.png"), ImageFiles.file("4.png")]
    )
    expect(page.locator(".attachment-status.ready")).to_have_count(4)
    expect(page.get_by_role("button", name="Add images")).to_be_disabled()
    page.get_by_label("Choose task images").set_input_files(ImageFiles.file("5.png"))
    expect(page.get_by_role("alert").first).to_contain_text("Choose up to four images")
    expect(page.get_by_label("Task instruction")).to_have_value("Keep this task text.")
