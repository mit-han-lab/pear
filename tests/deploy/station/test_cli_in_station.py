import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from config_files import ConfigFiles
from pear.deploy.station.cli import GUICommand, StationCommand
from pear.deploy.station.gui.server.launcher import GUILauncher
from pear.deploy.station.gui.server.server import GUIServer

# the robots the command line offers, each by the name of its package
ROBOT_NAMES = tuple(robot_name for robot_name, _ in StationCommand.robot_commands())
HELP_TIMEOUT_S = 120  # to import every robot package and print a usage
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


class GUICommandLine:
    """The command lines of `pear-station gui` that these tests run: the tests' YAM station with its models."""

    @staticmethod
    def arguments(run_dir: Path, *extra: str, config: Path = YAM_STATION, options: Sequence[str] = ()) -> list[str]:
        """Returns the arguments that serve the station ``config`` from ``run_dir``: its ``options``, then the models,
        the run directory and the ``extra`` overrides."""
        return ["gui", "--config", str(config), *options, *MODELS, f"run_dir={run_dir}", *extra]


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
def served(monkeypatch: pytest.MonkeyPatch) -> list[GUIServer]:
    """The servers foreground `pear-station gui` runs in its process, recorded instead of served."""
    servers = []
    monkeypatch.setattr(GUILauncher, "read_health", lambda launcher: None)
    monkeypatch.setattr(GUIServer, "run", lambda server: servers.append(server))
    return servers


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[GUILauncher | GUIServer]:
    """The services started in the background or in this process, recorded instead of started."""
    services = []
    monkeypatch.setattr(GUILauncher, "read_health", lambda launcher: None)
    monkeypatch.setattr(GUILauncher, "start", lambda launcher, command: services.append(launcher))
    monkeypatch.setattr(GUIServer, "run", lambda server: services.append(server))
    return services


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[GUILauncher | GUIServer]:
    """The launchers that got as far as looking at their port."""
    services = []

    def read_health(launcher):
        services.append(launcher)
        raise RuntimeError("The port was looked at")

    monkeypatch.setattr(GUILauncher, "read_health", read_health)
    monkeypatch.setattr(GUIServer, "run", lambda server: services.append(server))
    return services


def test_the_help_lists_gui_and_every_robot(pear_station, capsys):
    with pytest.raises(SystemExit) as result:
        pear_station("--help")

    assert result.value.code == 0
    output = capsys.readouterr().out
    for name in ("gui", *ROBOT_NAMES):
        assert name in output


def test_an_unknown_robot_is_an_invalid_choice(pear_station, capsys):
    with pytest.raises(SystemExit) as result:
        pear_station("nonexistent", "check")

    assert result.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_a_robot_class_of_no_installed_robot_is_refused(pear_station, capsys, tmp_path):
    robot_class = "robot_class=pear.deploy.station.control.robots.nonexistent.robot.NonexistentRobot"

    code = pear_station(*GUICommandLine.arguments(tmp_path / "gui", robot_class))

    assert code == 1
    assert "nonexistent" in capsys.readouterr().err


def test_an_unrecognized_argument_is_refused(pear_station, capsys):
    with pytest.raises(SystemExit) as result:
        pear_station("yam", "check", "--config", str(YAM_STATION), "--bogus")

    assert result.value.code == 2
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


def test_a_command_error_is_reported_with_exit_code_one(pear_station, capsys):
    code = pear_station("yam", "smoke", "--config", str(YAM_STATION), "--delta", "5", "--yes")

    assert code == 1
    assert "ERROR: Smoke-test delta must be nonzero and at most 0.05 rad" in capsys.readouterr().err


@pytest.mark.parametrize(
    "target", [[], *([name] for name in ROBOT_NAMES)], ids=lambda target: " ".join(["pear-station", *target])
)
def test_the_help_prints_in_a_fresh_process(target):
    command = [sys.executable, "-m", "pear.deploy.station.cli", *target, "--help"]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=HELP_TIMEOUT_S)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(f"usage: {' '.join(['pear-station', *target])}")


def test_the_foreground_command_serves_the_station_with_its_overrides(served, pear_station, tmp_path):
    code = pear_station(*GUICommandLine.arguments(tmp_path / "gui", "port=18999", "default_model=claude_code"))

    assert code == 0
    server = served[0]
    assert (server.config.robot_class, server.service_config.port, server.config.default_model) == (
        "stand_in_robots.StandInYAMRobot",
        18999,
        "claude_code",
    )
    assert (server.config.config_dir, server.service_config.run_dir) == (str(YAM_STATION.parent), tmp_path / "gui")


def test_the_default_port_is_the_robot_s_own(served, pear_station, tmp_path):
    pear_station(*GUICommandLine.arguments(tmp_path / "gui"))

    assert served[0].service_config.port == 18800


def test_a_station_without_models_is_reported(pear_station, capsys, tmp_path):
    code = pear_station("gui", "--config", str(YAM_STATION), f"run_dir={tmp_path / 'gui'}")

    assert code == 1
    assert "ERROR" in capsys.readouterr().err


def test_gui_needs_the_station_s_own_file(served, pear_station, capsys, tmp_path):
    with pytest.raises(SystemExit) as result:
        pear_station("gui", *MODELS, f"run_dir={tmp_path / 'gui'}")

    assert result.value.code == 2 and not served
    assert "--config" in capsys.readouterr().err


def test_a_station_without_a_run_directory_is_refused(served, pear_station, capsys):
    code = pear_station("gui", "--config", str(YAM_STATION), *MODELS)

    assert code == 1 and not served
    assert "run_dir" in capsys.readouterr().err


def test_a_station_file_that_names_no_robot_is_refused(started, pear_station, tmp_path, capsys):
    config = tmp_path / "station.yaml"
    config.write_text("default_model: vllm\nmodels: {}\n")

    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", config=config, options=["--detach"])) == 1
    assert "names no robot" in capsys.readouterr().err and not started


def test_an_unreadable_station_file_is_reported(started, pear_station, tmp_path, capsys):
    config = tmp_path / "station.yaml"
    config.write_text("robot_class: [unclosed\n")

    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", config=config, options=["--detach"])) == 1
    assert capsys.readouterr().err.startswith("ERROR:") and not started


def test_a_broken_station_configuration_is_reported_before_detaching(started, pear_station, tmp_path, capsys):
    config = tmp_path / "station.yaml"
    config.write_text("robot_class: stand_in_robots.StandInYAMRobot\nnot_a_station_field: 1\n")

    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", config=config, options=["--detach"])) == 1
    assert capsys.readouterr().err.startswith("ERROR:") and not started


def test_a_missing_station_configuration_is_reported_in_the_foreground(started, pear_station, tmp_path, capsys):
    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", config=tmp_path / "absent.yaml")) == 1
    assert "absent.yaml" in capsys.readouterr().err and not started


@pytest.mark.parametrize("launch", [[], ["--detach"]], ids=["foreground", "detach"])
def test_an_unknown_model_is_named(reached, pear_station, tmp_path, capsys, launch):
    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", "default_model=absent", options=launch)) == 1

    error = capsys.readouterr().err
    assert error.startswith("ERROR:") and "'absent'" in error and "claude_code" in error
    assert not reached


def test_a_format_the_robot_lacks_is_named(reached, pear_station, tmp_path, capsys):
    action_format = "models.claude_code.action_format=robodojo"

    assert pear_station(*GUICommandLine.arguments(tmp_path / "gui", action_format, options=["--detach"])) == 1

    error = capsys.readouterr().err
    assert "claude_code" in error and "'robodojo'" in error and "bimanual_pose" in error
    assert not reached


def test_an_offered_model_reaches_the_port(reached, pear_station, tmp_path):
    pear_station(*GUICommandLine.arguments(tmp_path / "gui", "default_model=claude_code"))

    assert [launcher.config.default_model for launcher in reached] == ["claude_code"]


def test_the_service_in_the_background_runs_the_command_line_s_configuration_and_overrides(tmp_path):
    overrides = [*MODELS, "port=18999", f"run_dir={tmp_path / 'gui'}", "default_model=claude_code"]

    assert GUICommand(YAM_STATION, overrides, is_detached=True).foreground_command() == [
        sys.executable,
        "-u",
        "-m",
        "pear.deploy.station.cli",
        "gui",
        "--config",
        str(YAM_STATION),
        *overrides,
    ]
