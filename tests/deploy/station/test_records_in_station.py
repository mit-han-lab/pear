import asyncio
import io
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from config_files import ConfigFiles
from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server.app import ServerApp
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from pear.deploy.station.records.annotations import Annotations
from pear.deploy.station.records.context_usage import ContextUsage
from pear.deploy.station.records.image_uploads import ImageUploads
from pear.deploy.station.records.recording.captions import Captions
from pear.deploy.station.records.task_events import TaskEvents
from pear.deploy.station.records.task_files import TaskFiles
from recorded_task import RecordedTask
from stored_text import StoredText
from waiting import Waiting

LARGE_SIZE = (3000, 1500)  # wider than the longest side a normalized picture keeps
LONGEST_SIDE = 2048  # of a normalized picture
SMALL_PIXEL_LIMIT = 100  # a MAX_PIXELS that the default test picture exceeds
EXPIRED_AGE_S = ImageUploads.DRAFT_LIFETIME_S + 1  # just past the lifetime of an unattached upload
TASK_ID = "0123456789abcdef" * 2  # an id of a task's form
TURNS = 2  # the stand-in model reports a full context after this many turns, which pauses the task
TASK_TIMEOUT_S = 60  # for each step of a recorded task
TASK_COUNT = 130  # enough tasks that the history is not one page
FIRST_CREATED = 1_700_000_000  # creation time of the oldest task, in seconds
LIMIT_BYTES = 16384  # of a command body
OVERSIZED_COMMAND_CHARS = 20000  # more than a command body may hold
NOISE_SIZE = (512, 512)  # a noise picture whose PNG exceeds the command body limit
UPLOAD_WAIT_S = 5  # for a held upload to be released
STARTED_WAIT_S = 2  # for the held upload to begin
STOP_REPLY_S = 1  # within which a STOP must answer while an upload is held
THINKING_S = 10  # inference delay of a model that thinks for the whole test
INSTRUCTION = "Hold both arms"  # of the task whose record a test reads
# a model of the station's configuration that a vLLM server serves, whose client names a reasoning effort
VLLM_MODEL = (
    "models.vllm.label=vLLM",
    "models.vllm.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
    "models.vllm.client.base_url=http://127.0.0.1:8000/v1",
    "models.vllm.client.model=example-model",
    "models.vllm.timeout_s=300",
)
# a model of the station's configuration that a Claude Code service serves, whose last check decides its task images
CLAUDE_CODE_MODEL = (
    "models.claude_code.label=Claude Code",
    "models.claude_code.client_class=pear.deploy.station.agent_loop.clients.claude_code.ClaudeCodeClient",
    "models.claude_code.client.base_url=http://127.0.0.1:8001",
    "models.claude_code.client.model=example-model",
    "models.claude_code.timeout_s=300",
)


class Pictures:
    """Encoded test pictures."""

    @staticmethod
    def make(fmt="PNG", size=(80, 40), color="red", **kwargs):
        output = io.BytesIO()
        Image.new("RGBA" if fmt == "PNG" else "RGB", size, color).save(output, fmt, **kwargs)
        return output.getvalue()


def test_tags_are_trimmed_deduplicated_and_keep_first_spelling():
    assert Annotations.clean_tags(["  Kp400 ", "kp400", "摩擦  补偿", "", "KP400", "baseline"]) == [
        "Kp400",
        "摩擦 补偿",
        "baseline",
    ]
    assert Annotations.clean_tags([]) == []


def test_reordered_tags_are_saved_in_the_given_order(tmp_path):
    annotations, files = Annotations(), TaskFiles(tmp_path)

    annotations.update(files, tags=["b", "a", "c"])
    assert Annotations.read(files)["tags"] == ["b", "a", "c"]
    annotations.update(files, tags=["c", "b", "a"])
    assert Annotations.read(files)["tags"] == ["c", "b", "a"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("Kp400", "list"),
        ([3], "text"),
        (["x" * (Annotations.MAX_TAG_LENGTH + 1)], "40 characters"),
        (["bad\x07tag"], "control"),
        ([f"t{n}" for n in range(Annotations.MAX_TAGS + 1)], "at most 20"),
        (["t"] * (Annotations.MAX_TAG_INPUTS + 1), "list"),
    ],
)
def test_invalid_tags_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        Annotations.clean_tags(value)


def test_an_update_replaces_the_tags_and_leaves_task_json_untouched(tmp_path):
    annotations, files = Annotations(), TaskFiles(tmp_path)
    (tmp_path / "task.json").write_text("{}")
    before = os.stat(tmp_path / "task.json").st_mtime_ns

    assert Annotations.read(files) == {"tags": []}
    assert annotations.update(files, tags=["摩擦补偿"]) == {"tags": ["摩擦补偿"]}
    # the tags are kept as their text, not escaped
    assert "摩擦补偿" in files.annotations_path.read_text(encoding="utf-8")
    assert annotations.update(files, tags=["Kp400"]) == {"tags": ["Kp400"]}
    saved = files.annotations_path.read_text(encoding="utf-8")
    assert "Kp400" in saved and type(json.loads(saved)["updated"]) is float
    assert not list(tmp_path.glob("*.tmp"))
    assert os.stat(tmp_path / "task.json").st_mtime_ns == before

    with pytest.raises(ValueError):
        annotations.update(files, tags=["bad\x00"])
    assert Annotations.read(files)["tags"] == ["Kp400"]
    files.annotations_path.write_text("{damaged")
    assert Annotations.read(files) == {"tags": []}


def test_a_file_with_a_note_still_loads_its_tags_and_ignores_the_note(tmp_path):
    files = TaskFiles(tmp_path)
    files.annotations_path.write_text(json.dumps({"tags": ["Kp400"], "note": "old text", "updated": 1.0}))

    assert Annotations.read(files) == {"tags": ["Kp400"]}


def test_tag_counts_merge_case_variants_and_sort_by_use():
    rows = [
        {"tags": ["kp400"], "created": 3},
        {"tags": ["Kp400", "baseline"], "created": 1},
        {"tags": ["Alpha"], "created": 2},
        {},
    ]

    assert Annotations.tag_counts(rows) == [
        {"tag": "Kp400", "count": 2},
        {"tag": "Alpha", "count": 1},
        {"tag": "baseline", "count": 1},
    ]


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_a_large_picture_becomes_a_bounded_rgb_jpeg(fmt):
    data, size = ImageUploads.normalize(Pictures.make(fmt, LARGE_SIZE))

    assert size == (LONGEST_SIDE, LONGEST_SIDE // 2)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG" and image.mode == "RGB"
        assert image.size == size


def test_a_picture_is_turned_upright_and_its_transparency_becomes_white():
    exif = Image.Exif()
    exif[274] = 6

    assert ImageUploads.normalize(Pictures.make("JPEG", exif=exif))[1] == (40, 80)
    data, _ = ImageUploads.normalize(Pictures.make(color=(255, 0, 0, 0)))
    with Image.open(io.BytesIO(data)) as image:
        assert min(image.getpixel((10, 10))) > 250


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not an image", id="text"),
        pytest.param(b"x" * (ImageUploads.MAX_BYTES + 1), id="oversized"),
        pytest.param(Pictures.make("GIF"), id="gif"),
    ],
)
def test_invalid_input_is_rejected(data):
    with pytest.raises(ValueError):
        ImageUploads.normalize(data)


def test_an_animation_and_a_picture_over_the_pixel_limit_are_rejected(monkeypatch):
    out = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(
        out, "PNG", save_all=True, append_images=[Image.new("RGB", (10, 10), "blue")]
    )

    with pytest.raises(ValueError, match="Animated"):
        ImageUploads.normalize(out.getvalue())
    monkeypatch.setattr(ImageUploads, "MAX_PIXELS", SMALL_PIXEL_LIMIT)
    with pytest.raises(ValueError, match="pixels"):
        ImageUploads.normalize(Pictures.make())


def test_drafts_belong_to_their_browser_expire_and_attach_in_the_given_order(tmp_path):
    uploads = ImageUploads(tmp_path / "uploads")
    ids = [uploads.add("browser-one", f"{i}.png", Pictures.make())["id"] for i in range(2)]
    task = tmp_path / "task"
    task.mkdir()
    files = TaskFiles(task)

    with pytest.raises(ValueError, match="another browser"):
        uploads.attach(ids, "browser-two", files)
    for invalid in [[ids[0]] * 2, [ids[0]] * 5, "bad", ["../bad"]]:
        with pytest.raises(ValueError):
            uploads.attach(invalid, "browser-one", files)
    assert [r["id"] for r in uploads.attach(ids[::-1], "browser-one", files)] == ids[::-1]

    metadata = uploads.root / f"{ids[0]}.json"
    row = json.loads(metadata.read_text())
    row["created"] = time.time() - EXPIRED_AGE_S
    metadata.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="expired"):
        uploads.attach([ids[0]], "browser-one", files)
    uploads.cleanup()
    assert not metadata.exists() and not metadata.with_suffix(".jpg").exists()
    assert (task / f"reference-{ids[0]}.jpg").exists()


def test_events_without_turn_numbers_are_numbered_in_order_and_the_file_is_not_rewritten(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = []
    for number in range(260):
        for kind in ("observation", "reasoning_delta", "execution"):
            rows.append(
                {
                    "id": len(rows) + 1,
                    "time": 1000 + number,
                    "type": kind,
                    "data": {"attempt": f"attempt-{number}"},
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = path.read_bytes()

    first = list(TaskEvents(path).read_numbered())[:100]
    later = [e for e in TaskEvents(path).read_numbered() if e["id"] > first[-1]["id"]][:100]
    complete = list(TaskEvents(path).read_numbered())

    assert first + later == complete[:200]
    assert [e["data"]["turn_number"] for e in complete if e["type"] == "observation"] == list(range(1, 261))
    assert path.read_bytes() == original


def test_a_task_is_found_only_by_an_id_of_its_form_with_a_folder(tmp_path):
    (tmp_path / TASK_ID).mkdir()
    (tmp_path / "not-a-task").mkdir()

    assert TaskFiles.find(tmp_path, TASK_ID).directory == tmp_path / TASK_ID
    for wrong_id in ("f" * 32, "not-a-task", "..", TASK_ID.upper(), TASK_ID + "0", TASK_ID[:-1], ""):
        assert TaskFiles.find(tmp_path, wrong_id) is None, wrong_id


@pytest.mark.parametrize("robot", RecordedTask.stored_robots())
def test_every_file_of_a_stored_task_folder_has_the_name_task_files_gives_it(robot):
    [directory] = [path for path in (RecordedTask.STORED_TASKS / robot).iterdir() if path.is_dir()]
    files = TaskFiles.find(directory.parent, directory.name)
    attempt_ids = {
        event["data"]["attempt"] for event in TaskEvents(files.events_path).read() if "attempt" in event["data"]
    }
    turn_records = (TaskFiles.OBSERVATION_RECORD, TaskFiles.OUTPUT_RECORD, TaskFiles.RESULT_RECORD)

    names = {
        files.task_path.name,
        files.events_path.name,
        files.session_path.name,
        *(files.turn_record_path(attempt_id, record).name for attempt_id in attempt_ids for record in turn_records),
    }

    assert {path.name for path in directory.iterdir()} <= names


def test_context_is_latest_full_length_not_sum_of_requests_and_survives_new_turn():
    usage = ContextUsage()
    usage.on_event("task", {"id": "task-one"})
    usage.on_event("turn_started", {"turn_number": 1, "attempt": "one", "episode_id": "e"})
    report = {"attempt": "one", "input": 51696, "output": 187}
    first, _ = usage.on_event("usage", report)
    assert first["tokens"] == 51883
    report["input"] = 999
    assert first["input"] == 51696

    usage.on_event("turn_started", {"turn_number": 2, "attempt": "two", "episode_id": "e"})
    assert usage.latest_context is first
    assert usage.turn_usage is None
    usage.on_event("observation", {"attempt": "two", "request_id": "request-two"})
    usage.on_event("capacity", {"attempt": "two", "input": 53000, "context_limit": 100000})
    assert usage.latest_context["tokens"] == 53000 and usage.latest_context["output"] is None
    usage.on_event("usage", {"attempt": "two", "input": 53000, "output": 200})
    assert usage.latest_context["tokens"] == 53200
    assert usage.turn_usage["tokens"] == 53200

    # A repeated report replaces the snapshot, and compaction can reduce its size.
    usage.on_event("usage", {"attempt": "two", "input": 40000, "output": 200})
    assert usage.latest_context["tokens"] == 40200
    assert first["tokens"] == 51883
    # the next task starts without it
    assert usage.on_event("task", {"id": "task-two"}) == (None, None)


def test_stale_reports_and_invalid_numbers_are_ignored():
    usage = ContextUsage()
    usage.on_event("turn_started", {"turn_number": 1, "attempt": "one"})
    usage.on_event("observation", {"attempt": "one", "request_id": "request-one"})
    usage.on_event("usage", {"attempt": "one", "input": 51696, "output": 187})

    for invalid in (
        {"attempt": "old"},
        {},
        {"attempt": "one", "request_id": "old-request"},
    ):
        usage.on_event("usage", {**invalid, "input": 123})
        usage.on_event("capacity", {**invalid, "input": 123, "context_limit": 1000})
        assert usage.latest_context["tokens"] == 51883
        assert usage.latest_context["context_limit"] is None
    usage.on_event("capacity", {"attempt": "one", "context_limit": float("nan")})
    usage.on_event("usage", {"attempt": "one", "input": 20, "output": -1})
    assert usage.latest_context["tokens"] == 20 and usage.latest_context["output"] is None
    usage.on_event("usage", {"attempt": "one", "input": True, "output": "300"})
    assert usage.latest_context["tokens"] == 20
    json.dumps(usage.latest_context, allow_nan=False)  # raises when the snapshot holds a NaN


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("session", {"episode_id": "new"}),
        ("turn_started", {"turn_number": 2, "attempt": "two", "episode_id": "new"}),
        ("task_end", {}),
        ("task", {"id": "new-task"}),
    ],
)
def test_context_resets_for_new_history(kind, data):
    usage = ContextUsage()
    usage.on_event("turn_started", {"turn_number": 1, "attempt": "one", "episode_id": "old"})
    usage.on_event("usage", {"attempt": "one", "input": 100, "output": 20})

    assert usage.on_event(kind, data) == (None, None)
    assert Captions.context({"phase": "Thinking", "context_usage": usage.latest_context}) == "Context Usage: Pending"
    assert Captions.context({"phase": "Executing", "context_usage": None}) == "Context Usage: Not provided"


def test_pause_resume_retains_latest_context_usage():
    usage = ContextUsage()
    usage.on_event("turn_started", {"turn_number": 1, "attempt": "one", "episode_id": "old"})
    usage.on_event("usage", {"attempt": "one", "input": 100, "output": 20})
    before = usage.latest_context

    usage.on_event("resume_context", {"episode_id": "old"})

    assert usage.latest_context == before


def test_a_report_with_its_own_context_keeps_the_call_cost_apart():
    state = ContextUsage()
    state.on_event("turn_started", {"attempt": "a", "turn_number": 1})
    data = {
        "attempt": "a",
        "input": 29137,
        "output": 2498,
        "partial": False,
        "context": {"input": 3825, "context_limit": 260096, "status": "measured"},
        "compaction": {"status": "completed", "count": 1},
    }

    state.on_event("usage", data)

    assert state.turn_usage["tokens"] == 31635
    assert state.latest_context["tokens"] == 3825
    packet = {
        "turn_number": 1,
        "inference_usage": state.turn_usage,
        "context_usage": state.latest_context,
        "phase": "Paused",
    }
    assert "29,137" in Captions.inference(packet)
    assert "3,825 / 260,096" in Captions.context(packet)

    data["context"] = {"input": None, "context_limit": 260096, "status": "pending"}
    data["output"], data["partial"] = None, True
    state.on_event("usage", data)
    packet.update(inference_usage=state.turn_usage, context_usage=state.latest_context)
    assert "Pending" in Captions.context(packet)
    assert "at least 29,137" in Captions.inference(packet)
    assert "Not provided" in Captions.inference(packet)


class PageService:
    """The page's service of a test's console, which the test serves with aiohttp's test server; it disables the
    console's motors before it closes the console, as an operator does before stopping the service."""

    TOKEN = "t" * 40  # the browser token a login gives
    CONFIGURATION_ID = "tests"  # the digest the service's identity carries

    def __init__(self, console: Console) -> None:
        self.console = console

    def app(self) -> web.Application:
        """Returns the service's application."""
        identity = ServiceIdentity(self.console.robot.NAME, self.CONFIGURATION_ID)
        app = ServerApp(self.console, self.TOKEN, identity).build()
        # before the service's own cleanup, which closes the console: a console with its motors on refuses to close
        app.on_cleanup.insert(0, self.disable_motors)

        return app

    async def disable_motors(self, app: web.Application) -> None:
        """Disables the console's motors, if they are enabled: the service's first cleanup."""
        await asyncio.to_thread(ConsoleRig.disable, self.console)


class RecordLayout:
    """What the stored texts of the tasks' records hold: each file of a task's folders with its fields, and what the
    page's history reads of a stored task."""

    # identifiers and clock readings that differ on every run, replaced before the record files' names are compared
    VOLATILE_PATTERNS = (
        (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<uuid>"),
        (re.compile(r"[0-9a-f]{32}"), "<hex>"),
        (re.compile(r"\b[0-9a-f]{12}\b"), "<turn>"),
        (re.compile(r"\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?"), "<time>"),
        (re.compile(r"\d{8}T\d{6}[\dZ]*"), "<time>"),
        (re.compile(r"\d{10,}"), "<count>"),
    )

    @classmethod
    def layout(cls, root: Path) -> dict[str, Any]:
        """Each masked file name with its top-level fields; nested values depend on run timing, names do not."""
        layouts = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".json":
                fields = sorted(json.loads(path.read_text()))
            elif path.suffix == ".jsonl":
                fields = cls.events_by_type([json.loads(line) for line in path.read_text().splitlines()])
            else:
                fields = path.suffix
            layouts.setdefault(cls.mask(str(path.relative_to(root))), fields)
        return dict(sorted(layouts.items()))

    @classmethod
    async def read_history(cls, app: web.Application) -> dict[str, Any]:
        """Returns the fields of the tasks the page's history lists, and the events of its first task by type with
        their turn numbers."""
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.post("/api/auth", json={"token": PageService.TOKEN})).status == 200
            tasks = await (await client.get("/api/history")).json()
            assert tasks, "the recorded fixture holds at least one task"
            events = await (await client.get(f"/api/history/{tasks[0]['id']}/events")).json()
        finally:
            await client.close()

        return {
            "tasks": sorted({field for task in tasks for field in task}),
            "events": cls.events_by_type(events),
            "turn_numbers": sorted(
                {event["data"]["turn_number"] for event in events if "turn_number" in event["data"]}
            ),
        }

    @classmethod
    def mask(cls, text: str) -> str:
        """Returns ``text`` with run-specific identifiers and times replaced by placeholders."""
        for pattern, placeholder in cls.VOLATILE_PATTERNS:
            text = pattern.sub(placeholder, text)
        return text

    @staticmethod
    def events_by_type(events: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
        """For each event type, the event fields and the union of its data fields."""
        by_type = {}
        for event in events:
            fields = by_type.setdefault(event.get("type", "?"), {"event": set(), "data": set()})
            fields["event"].update(event)
            fields["data"].update(event.get("data") or {})
        return {
            kind: {part: sorted(names) for part, names in fields.items()} for kind, fields in sorted(by_type.items())
        }


class AnnotationHistory:
    """The tasks that a test of the history's annotations writes."""

    @staticmethod
    def make_task(root: Path, task_id: str, created: int) -> Path:
        """Writes the record of a task created at ``created`` and returns its folder."""
        directory = root / task_id
        directory.mkdir()
        task = {"id": task_id, "instruction": f"task {created}", "created": created, "model": "demo"}
        (directory / "task.json").write_text(json.dumps(task))
        os.utime(directory / "task.json", (created, created))
        return directory


class TaskRecord:
    """A task a test starts to read its record."""

    @staticmethod
    def started_task(controller: Console) -> dict[str, Any]:
        """Starts a task, reads its record, pauses it and returns the record."""
        controller.commands.run("start", {"instruction": INSTRUCTION})
        Waiting.until(lambda: controller.task_run.task is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        record = json.loads((controller.root / controller.task_run.task["id"] / "task.json").read_text())
        controller.commands.run("pause", None)
        Waiting.until(lambda: not controller.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        return record


@pytest.fixture
def page_app(console: Console) -> web.Application:
    """The application of the page's service of ``console``, for the test to serve."""
    return PageService(console).app()


def test_a_task_s_record_files_and_their_fields_are_the_stored_layout(robot_name, console, monkeypatch):
    controller = console
    monkeypatch.setattr(
        controller, "new_client", StandInModel(controller, delay_s=0, is_moving=True, context_limit=TURNS)
    )
    controller.commands.run("enable", None)
    Waiting.until(
        lambda: not controller.is_busy and controller.robot.status()["status"] == "READY",
        TASK_TIMEOUT_S,
        ConsoleRig.POLL_S,
    )
    controller.commands.run("start", {"instruction": "Pick up the red cube."})
    Waiting.until(lambda: not controller.is_busy, TASK_TIMEOUT_S, ConsoleRig.POLL_S)
    controller.commands.run("home", {})
    Waiting.until(lambda: controller.home_review.plan is not None, TASK_TIMEOUT_S, ConsoleRig.POLL_S)
    controller.commands.run("home_confirm", {"plan_id": controller.home_review.plan["plan_id"], "confirm": True})
    Waiting.until(
        lambda: not controller.is_busy and controller.phase == "Home complete", TASK_TIMEOUT_S, ConsoleRig.POLL_S
    )
    controller.commands.run("disable", {"confirm": True})
    Waiting.until(
        lambda: not controller.is_busy and controller.robot.status()["status"] == "OFF",
        TASK_TIMEOUT_S,
        ConsoleRig.POLL_S,
    )

    assert controller.task_run.total_steps == TURNS
    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"records-{robot_name}.txt",
        json.dumps(RecordLayout.layout(controller.root), indent=1, sort_keys=True) + "\n",
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )
    if os.environ.get(ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE):
        shutil.rmtree(RecordedTask.STORED_TASKS / robot_name, ignore_errors=True)
        shutil.copytree(controller.root, RecordedTask.STORED_TASKS / robot_name, ignore=shutil.ignore_patterns("*.jpg"))
        for record in (RecordedTask.STORED_TASKS / robot_name).rglob("*.json"):
            record.write_text(record.read_text().replace(str(controller.root), "records"))


def test_history_reads_recorded_tasks(robot_name, console, page_app):
    shutil.copytree(RecordedTask.STORED_TASKS / robot_name, console.root, dirs_exist_ok=True)

    history = asyncio.run(RecordLayout.read_history(page_app))

    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"history-{robot_name}.txt",
        json.dumps(history, indent=1) + "\n",
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )


@pytest.mark.parametrize("robot_name", ["yam"])
def test_history_lists_every_task_with_annotations_and_edits_keep_order(console, page_app):
    controller = console
    ids = [f"{n:032x}" for n in range(1, TASK_COUNT + 1)]
    for n, task_id in enumerate(ids):
        AnnotationHistory.make_task(controller.root, task_id, FIRST_CREATED + n)

    async def scenario():
        async with TestClient(TestServer(page_app)) as client:
            url = f"/api/history/{ids[0]}/annotations"
            assert (await client.get(url)).status == 401
            assert (await client.post(url, json={"tags": ["x"]})).status == 401
            await client.post("/api/auth", json={"token": PageService.TOKEN})
            assert await (await client.get(url)).json() == {"tags": []}
            response = await client.post(url, json={"tags": ["Kp400", " kp400", "摩擦 补偿"]})
            assert response.status == 200
            assert await response.json() == {"tags": ["Kp400", "摩擦 补偿"]}
            await client.post(f"/api/history/{ids[1]}/annotations", json={"tags": ["KP400"]})
            for bad in (
                {"tags": ["x" * (Annotations.MAX_TAG_LENGTH + 1)]},
                {"color": "red"},
                {"note": "x"},
                {"tags": [], "note": "x"},
                {},
                ["tags"],
            ):
                response = await client.post(url, json=bad)
                assert response.status == 409
                assert (await response.json())["error"]
            missing = f"/api/history/{'f' * 32}/annotations"
            assert (await client.post(missing, json={"tags": []})).status == 404
            rows = await (await client.get("/api/history")).json()
            # All tasks, newest first; tagging the oldest task did not move it.
            assert [row["id"] for row in rows] == ids[::-1]
            assert rows[-1]["tags"] == ["Kp400", "摩擦 补偿"]
            assert rows[0]["tags"] == [] and "note" not in rows[0]
            assert await (await client.get("/api/tags")).json() == [
                {"tag": "Kp400", "count": 2},
                {"tag": "摩擦 补偿", "count": 1},
            ]
            archive = await (await client.get(f"/api/history/{ids[0]}/export")).read()
            with zipfile.ZipFile(io.BytesIO(archive)) as files:
                assert json.loads(files.read(TaskFiles.ANNOTATIONS_NAME))["tags"] == ["Kp400", "摩擦 补偿"]

    asyncio.run(scenario())


def test_start_saves_tags_and_invalid_tags_create_no_task(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=THINKING_S))
    before = {path.name for path in c.root.iterdir()}
    with pytest.raises(ValueError, match="40 characters"):
        c.commands.run("start", {"instruction": "Hold", "tags": ["x" * (Annotations.MAX_TAG_LENGTH + 1)]})
    assert c.task_run.task is None and {path.name for path in c.root.iterdir()} == before
    try:
        c.commands.run("start", {"instruction": "Hold", "tags": ["Kp400", "kp400 ", "摩擦 补偿"]})
        Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        directory = c.root / c.task_run.task["id"]
        assert Annotations.read(TaskFiles(directory)) == {"tags": ["Kp400", "摩擦 补偿"]}
        assert "tags" not in json.loads((directory / "task.json").read_text())
    finally:
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("clear", None)
    Waiting.until(lambda: c.task_run.task is None and not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("start", {"instruction": "Hold without tags"})
    try:
        Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert not (c.root / c.task_run.task["id"] / TaskFiles.ANNOTATIONS_NAME).exists()
    finally:
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_upload_start_history_and_export(enabled_console, page_app):
    enabled = enabled_console

    async def run():
        async with TestClient(TestServer(page_app)) as client:

            def form(data):
                body = FormData()
                body.add_field("image", data, filename="goal.png", content_type="image/png")
                return body

            url = "/api/images?client_id=test-client"
            assert (await client.post(url, data=form(Pictures.make()))).status == 401
            await client.post("/api/auth", json={"token": PageService.TOKEN})
            assert (await client.post(url, json={"image": "fake"})).status == 409
            assert (await client.post(url, data=form(b"fake png"))).status == 409
            assert (await client.post(url, data=form(b"x" * (ImageUploads.MAX_BYTES + 1)))).status == 413
            assert (await client.post("/api/command", json={"large": "x" * OVERSIZED_COMMAND_CHARS})).status == 413
            # A compressed upload larger than the command body limit.
            out = io.BytesIO()
            Image.effect_noise(NOISE_SIZE, 100).convert("RGB").save(out, "PNG")
            assert len(out.getvalue()) > LIMIT_BYTES
            response = await client.post(url, data=form(out.getvalue()))
            assert response.status == 200
            image = await response.json()
            body = {
                "client_id": "test-client",
                "command_id": uuid4().hex,
                "command": "start",
                "data": {"instruction": "Match the image", "image_ids": [image["id"]]},
            }
            first = await client.post("/api/command", json=body)
            assert first.status == 200, await first.text()
            task = dict(enabled.task_run.task)
            assert (await client.post("/api/command", json=body)).status == 200
            assert enabled.task_run.task["id"] == task["id"]
            assert (await client.get(task["images"][0]["url"])).status == 200
            # the task runs: its model session starts with the task's images
            await asyncio.to_thread(
                Waiting.until, lambda: enabled.task_run.session is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S
            )
            assert enabled.task_run.session.reference_images[0].uuid == image["id"]
            assert len(enabled.task_run.session.reference_images) == 1
            rows = await (await client.get("/api/history")).json()
            assert rows[0]["images"] == task["images"]
            archive = await (await client.get(f"/api/history/{task['id']}/export")).read()
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                assert task["images"][0]["file"] in z.namelist()
            assert (await client.post(url, data=form(Pictures.make()))).status == 409

    asyncio.run(run())


def test_stop_does_not_wait_for_image_processing(console, page_app, monkeypatch):
    controller = console
    started, release = threading.Event(), threading.Event()
    original = controller.uploads.add

    def slow(*args):
        started.set()
        assert release.wait(UPLOAD_WAIT_S)
        return original(*args)

    monkeypatch.setattr(controller.uploads, "add", slow)

    async def run():
        async with TestClient(TestServer(page_app)) as client:
            await client.post("/api/auth", json={"token": PageService.TOKEN})
            body = FormData()
            body.add_field("image", Pictures.make(), filename="goal.png")
            upload = asyncio.create_task(client.post("/api/images?client_id=test-client", data=body))
            try:
                assert await asyncio.to_thread(started.wait, STARTED_WAIT_S)
                response = await asyncio.wait_for(
                    client.post(
                        "/api/command",
                        json={
                            "client_id": "other-browser",
                            "command_id": uuid4().hex,
                            "command": "stop",
                        },
                    ),
                    STOP_REPLY_S,
                )
                assert response.status == 200 and controller.is_stop_latched
            finally:
                release.set()
                assert (await upload).status == 200

    asyncio.run(run())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_old_claude_service_blocks_attachments_before_task_creation(robot_name, tmp_path, monkeypatch):
    with ConsoleRig(robot_name, tmp_path, *CLAUDE_CODE_MODEL) as enabled:
        ConsoleRig.enable(enabled)
        monkeypatch.setattr(
            enabled, "new_client", StandInModel(enabled, delay_s=ConsoleRig.MODEL_DELAY_S, is_moving=False)
        )
        enabled.model = "claude_code"
        assert enabled.check_model()["status"] == "ok"  # a service whose check lists no task images
        assert not enabled.snapshot()["supports_reference_images"]
        uid = enabled.uploads.add(enabled.lease.owner, "goal.png", Pictures.make())["id"]
        with pytest.raises(ValueError, match="does not yet support"):
            enabled.commands.run("start", {"instruction": "Match it", "image_ids": [uid]})
        assert enabled.task_run.task is None


@pytest.mark.parametrize("robot_name", ["yam"])
def test_effort_of_the_model_is_recorded(robot_name, tmp_path, monkeypatch):
    with ConsoleRig(robot_name, tmp_path, *VLLM_MODEL) as enabled:
        ConsoleRig.enable(enabled)
        monkeypatch.setattr(
            enabled, "new_client", StandInModel(enabled, delay_s=ConsoleRig.MODEL_DELAY_S, is_moving=False)
        )
        enabled.commands.run("model", {"model": "vllm"})

        record = TaskRecord.started_task(enabled)

        assert record["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_model_without_effort_records_no_effort(enabled_console):
    record = TaskRecord.started_task(enabled_console)

    assert "reasoning_effort" not in record


@pytest.mark.parametrize("robot_name", ["yam"])
def test_legacy_pagination_keeps_numbers_and_export_keeps_original_records(console, page_app):
    controller = console
    task_id = uuid.uuid4().hex
    root = controller.root / task_id
    root.mkdir()
    rows = []
    for number in range(260):
        for kind in ("observation", "action", "execution"):
            rows.append(
                {
                    "id": len(rows) + 1,
                    "time": 1000 + number,
                    "type": kind,
                    "task_id": task_id,
                    "data": {"attempt": f"attempt-{number}"},
                }
            )
    journal = "".join(json.dumps(e) + "\n" for e in rows).encode()
    (root / "events.jsonl").write_bytes(journal)
    originals = {
        "one-request.json": b'{"raw_input":"preserved"}',
        "one-response.json": b'{"action":"preserved"}',
        "one-head.jpg": b"snapshot",
    }
    for name, value in originals.items():
        (root / name).write_bytes(value)

    async def scenario():
        client = TestClient(TestServer(page_app))
        await client.start_server()
        try:
            await client.post("/api/auth", json={"token": PageService.TOKEN})
            collected = []
            after = 0
            while True:
                page = await (await client.get(f"/api/history/{task_id}/events?after={after}")).json()
                if not page:
                    break
                collected.extend(page)
                after = page[-1]["id"]
            numbers = [e["data"]["turn_number"] for e in collected if e["type"] == "observation"]
            assert numbers == list(range(1, 261))
            everything = await (await client.get(f"/api/history/{task_id}/events?all=1")).json()
            assert everything == collected and len(everything) == len(rows)
            tail = await (await client.get(f"/api/history/{task_id}/events?after=100&all=1")).json()
            assert [e["id"] for e in tail] == list(range(101, len(rows) + 1))
            archive = await (await client.get(f"/api/history/{task_id}/export")).read()
            with zipfile.ZipFile(io.BytesIO(archive)) as zip:
                assert zip.read("events.jsonl") == journal
                for name, value in originals.items():
                    assert zip.read(name) == value
            assert (root / "events.jsonl").read_bytes() == journal
        finally:
            await client.close()

    asyncio.run(scenario())
