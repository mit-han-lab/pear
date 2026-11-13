import asyncio
import json
import os
import pkgutil
import secrets
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config_files import ConfigFiles
from first_line import FirstLine
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.cli import GUICommand, StationCommand
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server import server as server_module
from pear.deploy.station.gui.server.app import ServerApp
from pear.deploy.station.gui.server.browser_token import BrowserToken
from pear.deploy.station.gui.server.launcher import GUILauncher
from pear.deploy.station.gui.server.server import GUIServer
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from pear.utils.net_utils import free_port
from repository import REPO

YAM_STATION = ConfigFiles.station_file("yam")  # the tests' YAM station, on stand-in hardware
# the models of the stations of these tests, the station's own models as overrides: the imported station's test
# models without Codex, the first of them the default
MODELS = (
    "default_model=vllm",
    "models.vllm.label=vLLM",
    "models.vllm.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
    "models.vllm.client.base_url=http://127.0.0.1:8000/v1",
    "models.vllm.client.model=example-model",
    "models.vllm.timeout_s=300",
    "models.claude_code.label=Claude Code",
    "models.claude_code.client_class=pear.deploy.station.agent_loop.clients.claude_code.ClaudeCodeClient",
    "models.claude_code.client.base_url=http://127.0.0.1:8001",
    "models.claude_code.client.model=example-model",
    "models.claude_code.timeout_s=300",
)
# the model a real service of these tests runs, in the place of the imported station's demonstration model
STAND_IN_MODEL = (
    "default_model=stand_in",
    "models.stand_in.label=Stand-in",
    "models.stand_in.client_class=stand_in_clients.StandInDemonstrationClient",
    "models.stand_in.client.action_format=bimanual_pose",
)
STARTUP_TIMEOUT_S = 60  # for the service in a new process to print its URL
EXIT_TIMEOUT_S = 15  # for a service to exit on SIGTERM
SHARED_MODE = 0o644  # readable by the group and by other users
SERVICE_TOKEN = "g" * 40  # the browser token of a service these tests serve in their own process
# the page's service in the background as these tests' launchers start it, on the standard library alone: it writes
# the token and its process id into the run directory, then answers every GET with the /health answer it is given
STAND_IN_SERVICE_PROGRAM = """
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class StandInService(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.server.health)

    def log_message(self, *args):
        pass

    @classmethod
    def main(cls):
        port, run_dir, health, token_name, token, pid_name = sys.argv[1:]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / token_name).write_text(token)
        (Path(run_dir) / pid_name).write_text(str(os.getpid()))
        server = HTTPServer(("127.0.0.1", int(port)), cls)
        server.health = health.encode()
        server.serve_forever()


StandInService.main()
"""


class StandInService:
    """The stand-in a test's launcher starts in the place of the page's service: what it answers, and the command
    that starts it."""

    TOKEN = "fixed-test-token"  # the browser token it writes
    INSTANCE_ID = "stand-in"  # of its run, in its /health answer
    PID_FILE_NAME = "stand-in.pid"  # in the run directory: its process id, by which the test stops it

    def __init__(self) -> None:
        self.configuration_id: str | None = None  # the digest it answers; None: its launcher's own
        self.version = ServiceIdentity.HEALTH_VERSION  # of the /health answer it gives

    def command(self, launcher: GUILauncher) -> list[str]:
        """Returns the command that starts the stand-in on the port and in the run directory of ``launcher``,
        answering as a service of the launcher's identity unless the test changed what it answers."""
        health = {
            "service": ServiceIdentity.SERVICE_NAME,
            "robot": launcher.identity.robot_name,
            "version": self.version,
            "instance": self.INSTANCE_ID,
            "configuration_id": self.configuration_id or launcher.identity.configuration_id,
        }
        service_config = launcher.service_config
        arguments = [str(service_config.port), str(service_config.run_dir), json.dumps(health)]
        arguments += [BrowserToken.FILE_NAME, self.TOKEN, self.PID_FILE_NAME]

        return [sys.executable, "-c", STAND_IN_SERVICE_PROGRAM, *arguments]

    def command_of(self, gui_command: GUICommand) -> list[str]:
        """Returns the command that starts the stand-in for the station `pear-station gui` names."""
        return self.command(GUILauncher.from_station_config(gui_command.config_path, gui_command.overrides))


class ForeignHandler(BaseHTTPRequestHandler):
    """Answers every request with the status and body set on its server."""

    def do_GET(self):
        self.send_response(self.server.status)
        self.send_header("Content-Type", self.server.content_type)
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, *args):
        pass


class GUIStation:
    """The tests' YAM station with its models as these tests launch its service."""

    @staticmethod
    def launcher(port: int, run_dir: Path, *overrides: str) -> GUILauncher:
        """Returns the launcher of the station's service on ``port``, run in ``run_dir``."""
        return GUILauncher(ConfigFiles.load("yam", *MODELS, f"port={port}", f"run_dir={run_dir}", *overrides))

    @staticmethod
    def detached_arguments(port: int, run_dir: Path) -> list[str]:
        """Returns the arguments of `pear-station gui --detach` for the station's service on ``port``."""
        return ["gui", "--config", str(YAM_STATION), "--detach", *MODELS, f"port={port}", f"run_dir={run_dir}"]


@pytest.fixture
def port() -> int:
    """A free port of this machine, for a test's service."""
    return free_port(0)


@pytest.fixture
def run_dir(tmp_path: Path) -> Iterator[Path]:
    """The run directory of a test's service; a stand-in still running there is stopped at the end."""
    yield tmp_path / "gui"
    pid_file = tmp_path / "gui" / StandInService.PID_FILE_NAME
    if pid_file.exists():
        os.kill(int(pid_file.read_text()), signal.SIGTERM)


@pytest.fixture
def stand_in_service(monkeypatch: pytest.MonkeyPatch) -> StandInService:
    """The stand-in that takes the page's service's place; `pear-station gui --detach` starts it too."""
    stand_in = StandInService()
    monkeypatch.setattr(GUICommand, "foreground_command", lambda gui_command: stand_in.command_of(gui_command))
    return stand_in


@pytest.fixture
def stand_in_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gives the processes a test starts the tests' stand-ins, which the service of a station of the tests imports."""
    paths = [os.environ.get("PYTHONPATH", ""), str(REPO), str(ConfigFiles.DIRECTORY)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in paths if path))


@pytest.fixture
def pear_station(monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    """Runs `pear-station` in this process with the given arguments, as its entry point reads them: a command line
    argparse refuses exits with its status, and a command returns its exit status."""

    def run(*arguments: str) -> int:
        monkeypatch.setattr(sys, "argv", ["pear-station", *arguments])
        station_command = StationCommand.from_command_line()
        try:
            station_command.run()
        except SystemExit as raised_exit:
            return raised_exit.code

        return 0

    return run


@pytest.fixture
def listener() -> Iterator[ThreadingHTTPServer]:
    """A web server on a free loopback port, answering what the test sets."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ForeignHandler)
    server.status, server.content_type, server.body = 200, "text/html", b""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_a_started_service_returns_the_login_url_with_its_token(port, run_dir, stand_in_service):
    launcher = GUIStation.launcher(port, run_dir)

    url = launcher.start_or_reuse(stand_in_service.command(launcher))

    assert url == f"http://127.0.0.1:{port}/#token=fixed-test-token"


def test_the_service_log_is_private(port, run_dir, stand_in_service):
    launcher = GUIStation.launcher(port, run_dir)

    launcher.start_or_reuse(stand_in_service.command(launcher))

    assert (run_dir / "service.log").stat().st_mode & 0o777 == 0o600


def test_the_service_log_keeps_errors_but_not_the_printed_url(port, run_dir):
    printing = "import sys; print('login url'); print('startup error', file=sys.stderr); raise SystemExit(3)"

    with pytest.raises(RuntimeError, match="GUI startup failed"):
        GUIStation.launcher(port, run_dir).start_or_reuse([sys.executable, "-c", printing])

    log = (run_dir / "service.log").read_text()
    assert "startup error" in log and "login url" not in log


def test_a_matching_service_is_reused_without_a_new_process(port, run_dir, stand_in_service):
    launcher = GUIStation.launcher(port, run_dir)
    started_url = launcher.start_or_reuse(stand_in_service.command(launcher))
    second = GUIStation.launcher(port, run_dir)

    assert second.start_or_reuse(["false"]) == started_url
    assert second.process is None


def test_a_service_of_another_configuration_is_refused(port, run_dir, stand_in_service):
    stand_in_service.configuration_id = "another-configuration"
    launcher = GUIStation.launcher(port, run_dir)

    with pytest.raises(RuntimeError, match="different or outdated"):
        launcher.start_or_reuse(stand_in_service.command(launcher))


def test_a_service_of_another_robot_is_refused(port, run_dir, stand_in_service):
    service = GUIStation.launcher(port, run_dir)
    service.start_or_reuse(stand_in_service.command(service))
    health = service.read_health()

    with pytest.raises(RuntimeError, match="different or outdated"):
        service.validate({**health, "robot": "another-robot"})


def test_an_outdated_service_is_refused(port, run_dir, stand_in_service):
    stand_in_service.version = ServiceIdentity.HEALTH_VERSION - 1
    launcher = GUIStation.launcher(port, run_dir)

    with pytest.raises(RuntimeError, match="different or outdated"):
        launcher.start_or_reuse(stand_in_service.command(launcher))


def test_a_startup_failure_points_to_the_log(port, run_dir):
    with pytest.raises(RuntimeError, match="GUI startup failed; inspect .*service.log"):
        GUIStation.launcher(port, run_dir).start_or_reuse([sys.executable, "-c", "raise SystemExit(3)"])


def test_a_listener_that_never_answers_cannot_be_verified(port, run_dir):
    service = GUIStation.launcher(port, run_dir)
    service.HEALTH_TIMEOUT_S = 0.2
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()

        with pytest.raises(RuntimeError, match="cannot be verified"):
            service.start_or_reuse(["false"])


def test_detach_then_reuse_print_the_same_url(port, run_dir, stand_in_service, pear_station, capsys):
    arguments = GUIStation.detached_arguments(port, run_dir)

    assert pear_station(*arguments) == 0
    first = capsys.readouterr().out
    assert pear_station(*arguments) == 0
    second = capsys.readouterr().out

    assert first == second == f"http://127.0.0.1:{port}/#token=fixed-test-token\n"


def test_the_foreground_command_reuses_a_matching_service(
    port, run_dir, stand_in_service, pear_station, monkeypatch, capsys
):
    served = []
    monkeypatch.setattr(GUIServer, "run", lambda server: served.append(server))
    foreground = [argument for argument in GUIStation.detached_arguments(port, run_dir) if argument != "--detach"]
    assert pear_station(*GUIStation.detached_arguments(port, run_dir)) == 0
    started = capsys.readouterr().out

    assert pear_station(*foreground) == 0

    assert capsys.readouterr().out == started and not served


def test_a_startup_failure_exits_with_an_error(port, run_dir, pear_station, monkeypatch, capsys):
    monkeypatch.setattr(
        GUICommand, "foreground_command", lambda gui_command: [sys.executable, "-c", "raise SystemExit(3)"]
    )

    assert pear_station(*GUIStation.detached_arguments(port, run_dir)) == 1
    assert "ERROR: GUI startup failed" in capsys.readouterr().err


def test_the_real_server_logs_in_only_with_the_printed_token(port, run_dir, pear_station, stand_in_path, capsys):
    arguments = [*GUIStation.detached_arguments(port, run_dir), *STAND_IN_MODEL]
    try:
        assert pear_station(*arguments) == 0
        url = capsys.readouterr().out.strip()
        token = url.split("#token=")[1]

        def login(value):
            request = Request(
                f"http://127.0.0.1:{port}/api/auth",
                data=json.dumps({"token": value}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                return urlopen(request, timeout=5).status
            except HTTPError as error:
                return error.code

        assert login("wrong") == 401
        assert login(token) == 200
        assert token not in (run_dir / "service.log").read_text()
    finally:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            os.kill(json.load(response)["pid"], signal.SIGTERM)


@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (200, "text/html", b"<!doctype html><title>Another app</title>"),
        (200, "application/json", b'["not", "an", "object"]'),
        (200, "application/json", b"null"),
        (404, "text/html", b"<h1>Not Found</h1>"),
    ],
    ids=["html", "json-list", "json-null", "not-found"],
)
def test_an_answer_that_is_not_a_health_object_names_a_different_service(
    listener, tmp_path, status, content_type, body
):
    listener.status, listener.content_type, listener.body = status, content_type, body

    with pytest.raises(RuntimeError, match="different or outdated service"):
        GUIStation.launcher(listener.server_address[1], tmp_path / "gui").start_or_reuse(["false"])

    assert not (tmp_path / "gui").exists()


def test_the_url_is_printed_once_the_service_listens(port, tmp_path, stand_in_path):
    log_dir = tmp_path / "gui"
    command = [sys.executable, "-m", "pear.deploy.station.cli", "gui", "--config", str(YAM_STATION)]
    command += [*STAND_IN_MODEL, f"port={port}", f"run_dir={log_dir}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        url = FirstLine.of(process, STARTUP_TIMEOUT_S).strip()
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass  # the service accepts connections as soon as its URL is out
    finally:
        os.kill(process.pid, signal.SIGTERM)
        _, log = process.communicate(timeout=EXIT_TIMEOUT_S)

    token = (log_dir / "browser-token").read_text()
    assert url == f"http://127.0.0.1:{port}/#token={token}"
    assert (log_dir / "browser-token").stat().st_mode & 0o777 == 0o600
    assert "listening" in log and token not in log


def test_the_token_is_created_private_and_reused(tmp_path):
    token = BrowserToken(tmp_path / "gui").create()

    assert len(token) >= BrowserToken.RANDOM_BYTE_COUNT
    assert (tmp_path / "gui" / "browser-token").stat().st_mode & 0o777 == 0o600
    assert BrowserToken(tmp_path / "gui").create() == token
    assert BrowserToken(tmp_path / "gui").read() == token


def test_creation_leaves_only_the_token_file(tmp_path):
    BrowserToken(tmp_path / "gui").create()

    assert [path.name for path in (tmp_path / "gui").iterdir()] == ["browser-token"]


def test_concurrent_creators_agree_on_one_complete_token(tmp_path, monkeypatch):
    log_dir = tmp_path / "gui"
    values = iter(["first-token", "second-token"])
    concurrent = []

    def token_urlsafe(size):
        value = next(values)
        if value == "first-token":
            concurrent.append(BrowserToken(log_dir).create())  # another service creates meanwhile
        return value

    monkeypatch.setattr(secrets, "token_urlsafe", token_urlsafe)
    token = BrowserToken(log_dir).create()

    assert concurrent == [token] and token == (log_dir / "browser-token").read_text()


def test_an_empty_token_is_refused(tmp_path):
    path = tmp_path / "browser-token"
    path.touch(mode=0o600)

    with pytest.raises(RuntimeError, match="empty.*remove it"):
        BrowserToken(tmp_path).create()
    with pytest.raises(RuntimeError, match="empty.*remove it"):
        BrowserToken(tmp_path).read()


def test_a_token_open_to_other_users_is_refused(tmp_path):
    path = tmp_path / "browser-token"
    path.write_text("shared-token")
    path.chmod(SHARED_MODE)

    with pytest.raises(RuntimeError, match="other users.*remove it"):
        BrowserToken(tmp_path).create()

    assert path.read_text() == "shared-token"


def test_the_identity_is_stable_and_follows_the_robot_class_and_the_run_directory(robot_name, tmp_path):
    config = ConfigFiles.load(robot_name, *MODELS)
    # the robot the stand-in robot of the tests stands in for, on its devices
    real_robot_class = pkgutil.resolve_name(config.robot_class).__bases__[0]
    real_config = ConfigFiles.load(
        robot_name, *MODELS, f"robot_class={real_robot_class.__module__}.{real_robot_class.__qualname__}"
    )
    identity = ServiceIdentity.of_config(config, tmp_path).configuration_id
    assert identity == ServiceIdentity.of_config(config, tmp_path).configuration_id
    assert identity != ServiceIdentity.of_config(real_config, tmp_path).configuration_id
    assert identity != ServiceIdentity.of_config(config, tmp_path / "other").configuration_id


def test_each_action_format_prompt_is_part_of_the_identity(robot_name, tmp_path, monkeypatch):
    config = ConfigFiles.load(robot_name, *MODELS)
    identity = ServiceIdentity.of_config(config, tmp_path).configuration_id
    render = SystemPrompt.render
    monkeypatch.setattr(
        SystemPrompt,
        "render",
        staticmethod(lambda instruction, action_format: render(instruction, action_format) + "\nchanged"),
    )
    assert identity != ServiceIdentity.of_config(config, tmp_path).configuration_id


def test_the_login_url_is_handed_over_once_listening_and_never_logged(tmp_path, caplog, monkeypatch):
    port = free_port(0)
    config = ConfigFiles.load("yam", *STAND_IN_MODEL, f"port={port}", f"run_dir={tmp_path}")
    token = BrowserToken(tmp_path).create()
    urls = []

    def on_listening(url):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()  # refused unless the service listens
        urls.append(url)

    server = GUIServer(config, on_listening)
    console = Console(config, root=tmp_path / "tasks", uploads_dir=tmp_path / "tasks-uploads")

    async def scenario():
        serving = asyncio.create_task(server.serve(console, token, ServiceIdentity("yam", "identity")))
        while not urls and not serving.done():
            await asyncio.sleep(0.01)
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)

    # the server's logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(server_module.logger, "handlers", [caplog.handler])
    with caplog.at_level("INFO", logger=server_module.logger.name):
        asyncio.run(asyncio.wait_for(scenario(), timeout=15))

    assert urls == [f"http://127.0.0.1:{port}/#token={token}"]
    messages = [record.getMessage() for record in caplog.records]
    assert any(f"http://127.0.0.1:{port}/" in message for message in messages)
    assert not any(token in message for message in messages)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_health_names_the_service_and_its_robot(console, tmp_path):
    config = ConfigFiles.load("yam", *MODELS, "port=0", f"run_dir={tmp_path}")
    identity = ServiceIdentity.of_config(config, tmp_path)

    async def scenario():
        client = TestClient(TestServer(ServerApp(console, SERVICE_TOKEN, identity).build()))
        await client.start_server()
        try:
            return await (await client.get("/health")).json()
        finally:
            await client.close()

    health = asyncio.run(scenario())

    assert health["service"] == ServiceIdentity.SERVICE_NAME and health["robot"] == console.robot.NAME
    assert (
        health["version"] == ServiceIdentity.HEALTH_VERSION and health["configuration_id"] == identity.configuration_id
    )
    GUILauncher(config).validate(health)  # `pear-station gui` would reuse it
