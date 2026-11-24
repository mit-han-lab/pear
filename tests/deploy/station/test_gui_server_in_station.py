import asyncio
import base64
import io
import json
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from aiohttp import ClientPayloadError, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.agent_loop.clients.health_checks import HealthChecks
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server import error_responses
from pear.deploy.station.gui.server.app import ServerApp
from pear.deploy.station.gui.server.live_routes import LiveFeed, LiveRoutes
from pear.deploy.station.gui.server.security import Security
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from recorded_task import RecordedTask, RecordingWaits, StandInSegment
from stand_in_clients import StandInDemonstrationClient

CONFIGURATION_ID = "configuration"  # the digest of the configuration the services of these tests answer with
# the imported station's test models without Codex, offered beside the console's stand-in model
MODELS = (
    "models.vllm.label=vLLM",
    "models.vllm.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
    "models.vllm.client.base_url=http://127.0.0.1:8000/v1",
    "models.vllm.client.model=example-model",
    "models.claude_code.label=Claude Code",
    "models.claude_code.client_class=pear.deploy.station.agent_loop.clients.claude_code.ClaudeCodeClient",
    "models.claude_code.client.base_url=http://127.0.0.1:8001",
    "models.claude_code.client.model=example-model",
)
LOCK_HOLD_S = 2  # a worker holds the console's lock this long
STALL_LIMIT_S = 1  # the event loop answers within this while the lock is held
ERRORS_TOKEN = "e" * 40  # the browser token of the services that answer the failures of a command or a route
STREAMED_BODY = b"partial archive"  # what a stream sends before it fails
HEADERS_TOKEN = "h" * 40  # the browser token of the service whose answers' headers are read
SOURCES = RecordedTask.SOURCE_FILES  # each camera's own video in a segment of the YAM robot's cameras, by role
POLL_S = 0.2  # between two looks at a render over HTTP


class ServedConsole:
    """The web application of a test's console as the page's service builds it, answering /health as a service of
    the console's robot."""

    @staticmethod
    def app(console: Console, token: str) -> web.Application:
        """Returns the application of ``console`` whose login takes ``token``."""
        return ServerApp(console, token, ServiceIdentity(console.robot.NAME, CONFIGURATION_ID)).build()


class CommandRun:
    """A command a browser that holds control posts, which the console runs as the test gives it."""

    @staticmethod
    def post(
        console: Console,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        command: Callable[[str, dict[str, Any] | None], dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        """Returns the status and the body of the answer to the command, the logged records of the answers' failures
        captured at ERROR."""

        async def scenario():
            client = TestClient(TestServer(ServedConsole.app(console, ERRORS_TOKEN)))
            await client.start_server()
            try:
                await client.post("/api/auth", json={"token": ERRORS_TOKEN})
                console.commands.run = command
                body = {"client_id": "test-client", "command_id": str(uuid.uuid4()), "command": "model"}
                response = await client.post("/api/command", json=body)
                return response.status, await response.json()
            finally:
                await client.close()

        # the answers' logger keeps its records to itself: the test's capture in the place of its handler
        monkeypatch.setattr(error_responses.logger, "handlers", [caplog.handler])
        with caplog.at_level("ERROR"):
            return asyncio.run(scenario())


@pytest.fixture
def models_console(robot_name: str, tmp_path: Path) -> Iterator[Console]:
    """A console of the robot ``robot_name`` that offers the imported station's test models beside its stand-in."""
    with ConsoleRig(robot_name, tmp_path, *MODELS) as console:
        yield console


@pytest.mark.parametrize("robot_name", ["yam"])
def test_access_needs_the_token_and_the_control_claim_except_for_the_stop_command(models_console):
    console = models_console

    async def scenario():
        token = "a" * 40
        client = TestClient(TestServer(ServedConsole.app(console, token)))
        await client.start_server()
        try:
            assert (await client.get("/api/state")).status == 401
            assert (await client.post("/api/auth", json={"token": "wrong"})).status == 401
            assert (await client.post("/api/auth", json={"token": token})).status == 200
            headers = {"Origin": "https://untrusted.example"}
            assert (
                await client.post("/api/control/claim", json={"client_id": "browser-one"}, headers=headers)
            ).status == 403
            console.lease.release(ConsoleRig.CLIENT_ID)
            assert (await (await client.post("/api/control/claim", json={"client_id": "browser-one"})).json())["owner"]
            assert not (await (await client.post("/api/control/claim", json={"client_id": "browser-two"})).json())[
                "owner"
            ]
            cid = str(uuid.uuid4())
            body = {
                "client_id": "browser-one",
                "command_id": cid,
                "command": "model",
                "data": {"model": "claude_code"},
            }
            responses = await asyncio.gather(*[client.post("/api/command", json=body) for _ in range(3)])
            assert all(r.status == 200 for r in responses)
            assert console.model == "claude_code"
            assert (await client.post("/api/command", json={**body, "data": {"model": "vllm"}})).status == 409
            outsider = {
                "client_id": "browser-two",
                "command_id": str(uuid.uuid4()),
                "command": "enable",
            }
            assert (await client.post("/api/command", json=outsider)).status == 409
            outsider.update(command="stop", command_id=str(uuid.uuid4()))
            assert (await client.post("/api/command", json=outsider)).status == 200
            assert console.is_stop_latched
            assert (await client.get("/api/records/not-a-uuid/browser-token")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_an_empty_token_never_logs_in(console):
    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "")))
        await client.start_server()
        try:
            return [(await client.post("/api/auth", json=body)).status for body in ({"token": ""}, {})]
        finally:
            await client.close()

    assert asyncio.run(scenario()) == [401, 401]


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_scripted_client_keeps_control_with_http_heartbeats(console):
    # Supervised gripper checks run from scripts that do not open the /api/live socket.
    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "c" * 40)))
        await client.start_server()
        try:
            await client.post("/api/auth", json={"token": "c" * 40})
            console.lease.release(ConsoleRig.CLIENT_ID)
            claim = await client.post("/api/control/claim", json={"client_id": "script-one"})
            assert (await claim.json())["owner"]
            console.lease.heartbeat_at -= 2.5
            beat = await client.post("/api/control/heartbeat", json={"client_id": "script-one"})
            assert (await beat.json())["owner"]
            console.lease.require("script-one")
            other = await client.post("/api/control/heartbeat", json={"client_id": "script-two"})
            assert not (await other.json())["owner"]
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000", "[::1]", "localhost", "example.com:8000", ""]
)
def test_loopback_hosts_pass_and_others_are_refused(host):
    request = make_mocked_request("GET", "/health", headers={"Host": host})
    is_loopback = host.startswith(("127.0.0.1", "localhost", "[::1]"))

    if is_loopback:
        Security("token", "robot").check(request)
    else:
        with pytest.raises(web.HTTPForbidden):
            Security("token", "robot").check(request)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_one_snapshot_per_period_serves_every_socket(console, monkeypatch):
    snapshots = []
    monkeypatch.setattr(console, "snapshot", lambda: snapshots.append(1) or {"phase": "Disabled"})
    feed = LiveFeed(console)

    feed.read()
    feed.read()

    assert len(snapshots) == 1
    assert json.loads(feed.state_message) == {"type": "state", "data": {"phase": "Disabled"}}


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_held_console_lock_never_stalls_the_event_loop(console):
    held, release = threading.Event(), threading.Event()

    def hold():
        with console.lock:
            held.set()
            release.wait(LOCK_HOLD_S)

    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "k" * 40)))
        await client.start_server()
        holder = threading.Thread(target=hold)
        try:
            await client.post("/api/auth", json={"token": "k" * 40})
            stream = await client.ws_connect("/api/live")
            holder.start()
            await asyncio.to_thread(held.wait)
            started = time.monotonic()
            await asyncio.sleep(LiveRoutes.FRAME_PERIOD_S * 3)  # the stream ticks while the lock is held
            assert (await client.get("/health")).status == 200
            elapsed = time.monotonic() - started
            await stream.close()
            return elapsed
        finally:
            release.set()
            holder.join()
            await client.close()

    assert asyncio.run(scenario()) < STALL_LIMIT_S


@pytest.mark.parametrize("robot_name", ["yam"])
def test_one_socket_carries_state_and_every_preview_while_a_stop_answers_at_once(console, record_property):
    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "c" * 40)))
        await client.start_server()
        try:
            await client.post("/api/auth", json={"token": "c" * 40})
            console.lease.release(ConsoleRig.CLIENT_ID)
            await client.post("/api/control/claim", json={"client_id": "browser-one"})
            streams = [await client.ws_connect("/api/live") for _ in range(3)]
            roles = set()
            state = False
            for _ in range(30):
                message = await streams[0].receive(timeout=2)
                if message.type == WSMsgType.BINARY:
                    roles.add(message.data[0])
                elif message.type == WSMsgType.TEXT:
                    state |= message.json()["type"] == "state"
                if state and len(roles) == 3:
                    break
            assert state and roles == {0, 1, 2}
            await streams[0].send_json({"type": "heartbeat", "client_id": "browser-one"})
            started = time.monotonic()
            response = await client.post(
                "/api/command",
                json={"client_id": "read-only", "command_id": str(uuid.uuid4()), "command": "stop"},
            )
            assert response.status == 200
            elapsed = time.monotonic() - started
            record_property("stop_http_seconds", elapsed)
            assert elapsed < 0.2
            for stream in streams:
                await stream.close()
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_model_health_refreshes_after_its_connection_returns_without_commands(console, monkeypatch):
    checks = []

    class Service(StandInDemonstrationClient):
        def check(self):
            checks.append(1)
            return {"status": "ok" if len(checks) > 1 else "offline", "message": "Checked"}

    monkeypatch.setattr(console, "new_client", StandInModel(console, Service))
    monkeypatch.setattr(HealthChecks, "MAX_AGE_S", 0)  # every automatic poll checks again

    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "m" * 40)))
        await client.start_server()
        try:
            await client.post("/api/auth", json={"token": "m" * 40})
            for _ in range(70):
                state = await (await client.get("/api/state")).json()
                if state["model_health"]["status"] == "ok":
                    break
                await asyncio.sleep(0.1)
            assert state["model_health"]["status"] == "ok"
            assert state["robot"]["status"] == "OFF" and not state["busy"]
            assert state["task"] is None
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_an_unexpected_error_is_500_and_logged(console, caplog, monkeypatch):
    def command(name, data):
        raise KeyError("missing")

    status, body = CommandRun.post(console, caplog, monkeypatch, command)

    assert status == 500
    assert "missing" not in body["error"]
    assert any(record.exc_info and record.exc_info[0] is KeyError for record in caplog.records)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_an_operator_error_is_409(console, caplog, monkeypatch):
    def command(name, data):
        raise ValueError("Unknown model")

    status, body = CommandRun.post(console, caplog, monkeypatch, command)

    assert status == 409
    assert body == {"error": "Unknown model"}
    assert not caplog.records


@pytest.mark.parametrize("robot_name", ["yam"])
def test_an_unexpected_route_error_is_500(console, caplog, monkeypatch):
    monkeypatch.setattr(console, "snapshot", lambda: {}["missing"])

    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, ERRORS_TOKEN)))
        await client.start_server()
        try:
            await client.post("/api/auth", json={"token": ERRORS_TOKEN})
            return (await client.get("/api/state")).status
        finally:
            await client.close()

    # the answers' logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(error_responses.logger, "handlers", [caplog.handler])
    with caplog.at_level("ERROR"):
        assert asyncio.run(scenario()) == 500
    assert any(record.exc_info and record.exc_info[0] is KeyError for record in caplog.records)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_unauthenticated_text_points_at_the_login_url(console):
    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, ERRORS_TOKEN)))
        await client.start_server()
        try:
            response = await client.get("/api/state")
            return response.status, await response.text()
        finally:
            await client.close()

    status, text = asyncio.run(scenario())

    assert status == 401
    assert "pear-station gui" in text


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_failure_mid_stream_is_not_answered_again(console, caplog, monkeypatch):
    async def stream(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(STREAMED_BODY)
        raise ConnectionResetError("Client went away")

    async def scenario():
        app = ServedConsole.app(console, "f" * 40)
        app.router.add_get("/stream", stream)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/stream")
            with pytest.raises(ClientPayloadError):
                await response.read()
            return response.status
        finally:
            await client.close()

    # the answers' logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(error_responses.logger, "handlers", [caplog.handler])
    with caplog.at_level("ERROR", logger=error_responses.logger.name):
        assert asyncio.run(scenario()) == 200
    assert not [record for record in caplog.records if record.name == error_responses.logger.name]


@pytest.mark.parametrize("robot_name", ["yam"])
def test_streamed_refused_and_page_answers_carry_the_headers(console):
    async def stream(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"streamed")
        return response

    async def scenario():
        app = ServedConsole.app(console, HEADERS_TOKEN)
        app.router.add_get("/stream", stream)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            return {path: (await client.get(path)).headers for path in ("/stream", "/api/state", "/")}
        finally:
            await client.close()

    answers = asyncio.run(scenario())

    for path, headers in answers.items():
        assert {name: headers.get(name) for name in Security.HEADERS} == Security.HEADERS, path


@pytest.mark.parametrize("robot_name", ["so101"])
def test_the_service_names_follow_the_robot(console):
    async def scenario():
        client = TestClient(TestServer(ServedConsole.app(console, "s" * 40)))
        await client.start_server()
        try:
            health = await (await client.get("/health")).json()
            assert (health["service"], health["robot"]) == ("pear-station-gui", "so101")
            response = await client.post("/api/auth", json={"token": "s" * 40})
            assert "so101_session" in response.cookies
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_video_endpoints_need_a_login_and_serve_ranges(console):
    task_id = "b" * 32
    directory = console.root / task_id / "recordings" / "merged"
    directory.mkdir(parents=True)
    (directory / "merged.json").write_text(json.dumps({"styled": True}))
    (directory / "full.mp4").write_bytes(b"0123456789")

    async def scenario():
        async with TestClient(TestServer(ServedConsole.app(console, "video-token"))) as client:
            url = f"/api/recordings/{task_id}/merged/full.mp4"
            assert (await client.get(url)).status == 401
            await client.post("/api/auth", json={"token": "video-token"})
            response = await client.get(url, headers={"Range": "bytes=2-5"})
            assert response.status == 206
            assert await response.read() == b"2345"
            response = await client.get(url + "?download=1")
            assert "attachment" in response.headers["Content-Disposition"]
            assert (await client.get(url.replace("full.mp4", "merged.json"))).status == 404
            # a segment's camera videos are in the task's export, not served one by one
            assert (await client.get(f"/api/recordings/{task_id}/001/head.mp4")).status == 404

    asyncio.run(scenario())


class TaskDownloads:
    """A task folder as a station of today or of before leaves it, and what its downloads are named."""

    CREATED = 1795484448.8  # the task's start, on the station's clock
    TASK_ID = "c" * 32

    @classmethod
    def write(cls, root: Path) -> Path:
        """Writes the task's folder under ``root`` and returns it: its records, a turn's frame by today's name and
        one by the name of before, recordings with an unfinished and an empty file, and a temporary file."""
        directory = root / cls.TASK_ID
        segment = directory / "recordings" / "001"
        segment.mkdir(parents=True)
        (directory / "turns").mkdir()
        task = {"id": cls.TASK_ID, "instruction": "Lift the cup.", "created": cls.CREATED, "model": "vllm"}
        (directory / "task.json").write_text(json.dumps(task))
        (directory / "session.json").write_text(json.dumps({"system_prompt": "You drive a robot </script> safely."}))
        frames = {
            "head": f"/api/records/{cls.TASK_ID}/turns/001-head.jpg",
            "wrist": f"/api/records/{cls.TASK_ID}/a1b2-head.jpg",
        }
        events = [
            {"id": 1, "type": "turn_started", "data": {"attempt": "001", "turn_number": 1}, "time": cls.CREATED + 1},
            {"id": 2, "type": "observation", "data": {"attempt": "001", "images": frames}, "time": cls.CREATED + 1},
            {
                "id": 3,
                "type": "usage",
                "data": {"attempt": "001", "input": 900, "output": 100},
                "time": cls.CREATED + 2,
            },
            {"id": 4, "type": "model_output", "data": {"attempt": "001", "answer": "done"}, "time": cls.CREATED + 3},
        ]
        # every event the console saves carries the monotonic clock too
        rows = [json.dumps({**event, "monotonic_s": event["time"] - cls.CREATED}) + "\n" for event in events]
        (directory / "events.jsonl").write_text("".join(rows))
        (directory / "turns" / "001-head.jpg").write_bytes(b"today's jpeg")
        (directory / "a1b2-head.jpg").write_bytes(b"jpeg of before")
        (directory / "notes.tmp").write_text("partial write")
        (segment / "recording.json").write_text(json.dumps({"id": "001", "status": "saved"}))
        (segment / "timeline.jsonl").write_text('{"start_frame": 0}\n' * 200)
        for name in SOURCES.values():
            (segment / name).write_bytes(name.encode() * 50)
        (segment / "head.partial.mp4").write_bytes(b"incomplete")
        (segment / "encoder.log").write_text("")

        return directory

    @classmethod
    def name(cls) -> str:
        """Returns the name of the task's downloads."""
        return f"yam-{time.strftime('%Y-%m-%d-%H%M', time.localtime(cls.CREATED))}-{cls.TASK_ID[:8]}"

    @staticmethod
    def trace_data(page: str) -> dict[str, Any]:
        """Returns the records a trace page holds."""
        mark = '<script id="trace-data" type="application/json">'
        return json.loads(page[page.index(mark) + len(mark) : page.index("</script>", page.index(mark))])


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_export_is_one_folder_of_the_task_with_its_trace(console):
    directory = TaskDownloads.write(console.root)
    task_id, name = TaskDownloads.TASK_ID, TaskDownloads.name()

    async def removed(pattern):
        # The handler deletes its temporary archive just after the client has the body.
        for _ in range(100):
            if not list(console.root.glob(pattern)):
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{pattern} was left in the history directory")

    async def scenario():
        async with TestClient(TestServer(ServedConsole.app(console, "export-token"))) as client:
            url = f"/api/history/{task_id}/export"
            assert (await client.get(url)).status == 401
            await client.post("/api/auth", json={"token": "export-token"})
            response = await client.get(url)
            assert response.status == 200
            assert f'filename="{name}.zip"' in response.headers["Content-Disposition"]
            assert int(response.headers["Content-Length"]) == len(body := await response.read())
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                assert set(archive.namelist()) == {
                    f"{name}/{path}"
                    for path in (
                        "task.json",
                        "session.json",
                        "events.jsonl",
                        "turns/001-head.jpg",
                        "a1b2-head.jpg",
                        "trace.html",
                        "recordings/001/recording.json",
                        "recordings/001/timeline.jsonl",
                        *(f"recordings/001/{source}" for source in SOURCES.values()),
                    )
                }
                assert archive.read(f"{name}/recordings/001/head.mp4") == b"head.mp4" * 50
                kinds = {info.filename: info.compress_type for info in archive.infolist()}
                assert kinds[f"{name}/recordings/001/head.mp4"] == zipfile.ZIP_STORED
                assert kinds[f"{name}/turns/001-head.jpg"] == zipfile.ZIP_STORED
                assert kinds[f"{name}/events.jsonl"] == zipfile.ZIP_DEFLATED
                assert TaskDownloads.trace_data(archive.read(f"{name}/trace.html").decode())["task"]["id"] == task_id
            assert (directory / "events.jsonl").is_file()
            # Export leaves no copy of the task beside the history.
            await removed(".download-*")
            await removed("*.zip*")
            assert (await client.get(f"/api/history/{'0' * 32}/export")).status == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_trace_page_holds_the_records_with_every_picture_inlined(console):
    TaskDownloads.write(console.root)
    task_id, name = TaskDownloads.TASK_ID, TaskDownloads.name()

    async def scenario():
        async with TestClient(TestServer(ServedConsole.app(console, "trace-token"))) as client:
            url = f"/api/history/{task_id}/trace"
            assert (await client.get(url)).status == 401
            await client.post("/api/auth", json={"token": "trace-token"})
            response = await client.get(url)
            assert response.status == 200 and response.content_type == "text/html"
            assert f'filename="{name}-trace.html"' in response.headers["Content-Disposition"]
            page = await response.text()
            # the page loads nothing: its script and styles are in it
            assert 'src="/' not in page and 'href="/' not in page
            trace = TaskDownloads.trace_data(page)
            assert trace["robot"] == "YAM" and trace["task"]["instruction"] == "Lift the cup."
            # a "</script>" of the records cannot close the page's script element
            assert trace["system_prompt"] == "You drive a robot </script> safely."
            [observation] = [event for event in trace["events"] if event["type"] == "observation"]
            prefix = "data:image/jpeg;base64,"
            assert observation["data"]["images"] == {
                "head": prefix + base64.b64encode(b"today's jpeg").decode(),
                "wrist": prefix + base64.b64encode(b"jpeg of before").decode(),
            }
            assert trace["token_usage"] == 1000
            # a turn's frame is served from the turns' folder
            frame = await client.get(f"/api/records/{task_id}/turns/001-head.jpg")
            assert frame.status == 200 and await frame.read() == b"today's jpeg"
            for bad in ("..", ".hidden", "002-head.jpg"):
                assert (await client.get(f"/api/records/{task_id}/turns/{bad}")).status == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_segment_s_styled_videos_are_rendered_and_served_over_http(console):
    task_id = "9" * 32
    directory = console.root / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    task.add_segment(1, 30, 2, started_monotonic_s=started)
    base = f"/api/history/{task_id}/recordings"
    files = f"/api/recordings/{task_id}/001/styled"

    async def rendered(client):
        deadline = time.monotonic() + RecordingWaits.RENDER_TIMEOUT_S
        while (row := (await (await client.get(base)).json())[0])["render"] is not None:
            assert time.monotonic() < deadline, row
            await asyncio.sleep(POLL_S)
        return row

    async def scenario():
        async with TestClient(TestServer(ServedConsole.app(console, "styled-token"))) as client:
            assert (await client.post(f"{base}/001/styled")).status == 401
            await client.post("/api/auth", json={"token": "styled-token"})
            [row] = await (await client.get(base)).json()
            assert row["styled"] is None and row["render"] is None
            assert (await client.get(f"{files}/full.mp4")).status == 404
            response = await client.post(f"{base}/001/styled")
            assert response.status == 200
            assert (await response.json())["render"]["status"] in ("waiting", "queued", "rendering")
            row = await rendered(client)
            assert row["styled"]["frames"] == 30
            response = await client.get(f"{files}/full.mp4?download=1")
            assert response.status == 200
            assert f"yam-{task_id}-001-styled-full.mp4" in response.headers["Content-Disposition"]
            assert await response.read() == (directory / "recordings" / "001" / "styled" / "full.mp4").read_bytes()
            for bad in ("merged.json", "encoder.log", "full.partial.mp4", "head.mp4"):
                assert (await client.get(f"{files}/{bad}")).status == 404
            assert (await client.post(f"{base}/abc/styled")).status == 404
            assert (await client.post(f"{base}/007/styled")).status == 409

    asyncio.run(scenario())


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_task_s_merged_videos_are_rendered_and_served_over_http(console):
    task_id = "f" * 32
    directory = console.root / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    task.add_segment(1, 30, 2, started_monotonic_s=started)
    task.add_segment(2, 24, 3, started_monotonic_s=started + 1)
    base = f"/api/history/{task_id}/recordings/merged"
    files = f"/api/recordings/{task_id}/merged"

    async def merged(client):
        deadline = time.monotonic() + RecordingWaits.RENDER_TIMEOUT_S
        while (status := await (await client.get(base)).json())["render"] is not None:
            assert time.monotonic() < deadline, status
            await asyncio.sleep(POLL_S)
        return status

    async def scenario():
        async with TestClient(TestServer(ServedConsole.app(console, "merge-token"))) as client:
            assert (await client.get(base)).status == 401
            assert (await client.post(base)).status == 401
            await client.post("/api/auth", json={"token": "merge-token"})
            status = await (await client.get(base)).json()
            assert status["merged"] is None and status["segments"]["full"] == ["001", "002"]
            assert (await client.get(f"{files}/full.mp4")).status == 404
            response = await client.post(base)
            assert response.status == 200
            assert (await response.json())["render"]["status"] in ("waiting", "queued", "rendering")
            status = await merged(client)
            assert status["current"] and status["merged"]["frames"] == 54
            videos = [name for name in (status["merged"]["full"], status["merged"]["motion"]) if name]
            assert videos[0] == "full.mp4"
            for name in videos:
                response = await client.get(f"{files}/{name}?download=1")
                assert response.status == 200
                disposition = response.headers["Content-Disposition"]
                assert f"yam-{task_id}-merged-{name}" in disposition
                body = await response.read()
                assert body == (directory / "recordings" / "merged" / name).read_bytes()
            for bad in ("merged.json", "head.mp4", "source.zip", "full.partial.mp4", "encoder.log"):
                assert (await client.get(f"{files}/{bad}")).status == 404
            console.recordings.live_segments.append(StandInSegment(task_id, "003"))
            try:
                response = await client.post(base)
                assert response.status == 409
                assert "Segment 3" in (await response.json())["error"]
            finally:
                console.recordings.live_segments.pop()
            assert (await client.post(f"/api/history/{'0' * 32}/recordings/merged")).status == 404

    asyncio.run(scenario())
