import re
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Page, expect

from pear.deploy.station.records.recording.ffmpeg import VIDEO_SIZE
from waiting import Waiting

LEASE_EXPIRY_WAIT_MS = 3300  # longer than the console's control lease: the tab of the test before no longer holds it
HOLD_TIMEOUT_MS = 15000  # for the motors to hold once enabled
MIN_STEPS = 2  # the steps the task takes before its first pause
STEPS_TIMEOUT_S = 25.0  # for the running task to take them
PAUSE_TIMEOUT_MS = 20000  # for the task to pause
RENDER_TIMEOUT_MS = 30000  # for a segment's styled videos to render in the background, and the merge to be offered
MERGE_TIMEOUT_MS = 15000  # for the merged video to render
DISABLE_TIMEOUT_MS = 15000  # for the motors to be disabled
EXPECT_TIMEOUT_S = 5.0  # how long a poll looks for its value: as long as Playwright's assertions wait by default
POLL_S = 0.1  # between two looks of a poll
# whether the video shown plays: it holds its current frame, at the width of a styled video
PLAYING_SCRIPT = f"(v) => v.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && v.videoWidth === {VIDEO_SIZE[0]}"

pytestmark = pytest.mark.usefixtures("station_login")


def test_record_a_task_play_styled_videos_resume_into_a_new_segment_and_disable_recording(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    page.wait_for_timeout(LEASE_EXPIRY_WAIT_MS)
    page.goto("/")
    toggle = page.get_by_role("switch", name="Recording", exact=True)
    expect(toggle).to_be_checked()
    page.get_by_role("button", name="Enable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Holding", timeout=HOLD_TIMEOUT_MS)
    page.get_by_role("textbox", name="Task instruction").fill("Lift both arms and return.")
    page.get_by_role("button", name="Send and start", exact=True).click()
    expect(toggle).to_be_disabled()
    Waiting.until(lambda: context.request.get("/api/state").json()["total_steps"] >= MIN_STEPS, STEPS_TIMEOUT_S, POLL_S)
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)

    # Each segment's styled videos render in the background as soon as it stops.
    recordings = page.get_by_role("region", name="Task recordings")
    expect(recordings.get_by_role("button", name=re.compile("Full video"))).to_be_visible(timeout=RENDER_TIMEOUT_MS)
    expect(recordings.get_by_role("button", name=re.compile("Motion only"))).to_be_visible(timeout=RENDER_TIMEOUT_MS)
    for name in (re.compile("Full video"), re.compile("Motion only")):
        recordings.get_by_role("button", name=name).click()
        Waiting.until(lambda: page.locator("video").evaluate(PLAYING_SCRIPT), EXPECT_TIMEOUT_S, POLL_S)
        page.get_by_role("button", name="Close video").click()
    expect(recordings.get_by_role("link", name="Download full video segment 1")).to_have_attribute(
        "href", re.compile(r"/001/styled/full\.mp4\?download=1$")
    )
    expect(recordings.get_by_role("link", name="Download motion only segment 1")).to_have_attribute(
        "href", re.compile(r"/001/styled/motion\.mp4\?download=1$")
    )
    expect(recordings.get_by_role("link", name="Download source files segment 1")).to_have_attribute(
        "href", re.compile(r"/001/source\.zip$")
    )
    page.screenshot(path=tmp_path / "recordings-saved.png", full_page=True)

    page.get_by_role("button", name="Resume task", exact=True).click()
    expect(recordings.get_by_text("Segment 2", exact=True)).to_be_visible()
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    expect(recordings.get_by_role("link", name="Download full video segment 2")).to_be_visible(
        timeout=RENDER_TIMEOUT_MS
    )

    # Both segments join into one full video; motion only keeps segment 1's motion.
    merged = recordings.get_by_role("group", name="Merged video")
    expect(merged.get_by_text("Not merged")).to_be_visible()
    merged.get_by_role("button", name="Merge: 2 segments").click(timeout=RENDER_TIMEOUT_MS)
    expect(merged.get_by_text("All 2 segments")).to_be_visible(timeout=MERGE_TIMEOUT_MS)
    for name in (re.compile("Full video"), re.compile("Motion only")):
        merged.get_by_role("button", name=name).click()
        Waiting.until(lambda: page.locator("video").evaluate(PLAYING_SCRIPT), EXPECT_TIMEOUT_S, POLL_S)
        page.get_by_role("button", name="Close video").click()
    expect(merged.get_by_role("link", name="Download merged full video")).to_have_attribute(
        "href", re.compile(r"/merged/full\.mp4\?download=1$")
    )
    expect(merged.get_by_role("button", name=re.compile("^Merge"))).to_have_count(0)
    page.screenshot(path=tmp_path / "recordings-merged.png", full_page=True)

    toggle.uncheck()
    page.get_by_role("button", name="Resume task", exact=True).click()
    expect(page.get_by_role("button", name="Pause task", exact=True)).to_be_visible()
    state = context.request.get("/api/state").json()
    assert state["recording"]["active"] is None
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    expect(recordings.get_by_text("Segment 3", exact=True)).to_have_count(0)

    toggle.check()
    page.get_by_role("button", name="New task", exact=True).click()
    page.get_by_role("button", name="Disable Motors", exact=True).click()
    page.locator(".modal").get_by_role("button", name="Disable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Disabled", timeout=DISABLE_TIMEOUT_MS)
