import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from playwright.sync_api import BrowserContext, Page, expect

# an environment variable: a YAML file of the real models to test, its default_model and models as a station's
# configuration gives them
MODELS_VARIABLE = "PEAR_STATION_MODELS"
HOLD_TIMEOUT_MS = 15000  # for the motors to hold once enabled
ANSWER_TIMEOUT_MS = 60000  # for the real model's first thought, and its first turn to complete
PAUSE_TIMEOUT_MS = 20000  # for the task to pause, and the new task to be offered

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get(MODELS_VARIABLE), reason=f"needs real models: {MODELS_VARIABLE} names a file of them"
    ),
    pytest.mark.usefixtures("station_login"),
]


@pytest.fixture(scope="module")
def station_models() -> tuple[str, ...]:
    """The overrides that give the module's station the real models of the file the environment names."""
    models_file = OmegaConf.load(os.environ[MODELS_VARIABLE])
    models = json.dumps(OmegaConf.to_container(models_file.models))

    return f"default_model={models_file.default_model}", f"models={models}"


def test_real_vllm_stream_in_browser_with_stand_in_robot_only(
    page: Page, context: BrowserContext, tmp_path: Path
) -> None:
    initial = context.request.get("/api/state").json()
    assert initial["robot"]["status"] == "OFF"
    assert initial["model"] == "vllm"

    page.goto("/")
    expect(page.get_by_text("Connected", exact=True)).to_be_visible()
    page.get_by_role("button", name="Check connection", exact=True).click()
    expect(page.get_by_text("Service ready", exact=True)).to_be_visible()
    page.get_by_role("button", name="Enable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Holding", timeout=HOLD_TIMEOUT_MS)

    page.get_by_role("textbox", name="Task instruction").fill(
        "Hold the current pose. Keep both arms and both grippers unchanged."
    )
    page.get_by_role("button", name="Send and start", exact=True).click()
    expect(page.locator(".thought-toggle").first).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(page.locator(".result-chip.completed").first).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    page.get_by_role("button", name="Pause task", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Paused", timeout=PAUSE_TIMEOUT_MS)
    completed = page.locator(".turn-card").filter(has=page.locator(".result-chip.completed")).first
    completed.locator(".thought-toggle").click()
    completed.scroll_into_view_if_needed()
    page.screenshot(path=tmp_path / "pear-station-vllm-live.png", full_page=True)

    page.get_by_role("button", name="New task", exact=True).click()
    expect(page.get_by_text("What should we work on?")).to_be_visible(timeout=PAUSE_TIMEOUT_MS)
    page.get_by_role("button", name="Disable Motors", exact=True).click()
    page.locator(".modal").get_by_role("button", name="Disable Motors", exact=True).click()
    expect(page.locator(".status-card strong")).to_have_text("Disabled")
    health = context.request.get("/api/model/health").json()
    assert health["status"] == "ok"
