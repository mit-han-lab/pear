import argparse
import pkgutil
import sys
from pathlib import Path

import pytest

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.joint.config import RobotConfig
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from stand_in_robots import StandInRobots

ROBOT_COMMANDS = dict(StationCommand.robot_commands())  # each robot package's maintenance commands, by its name
ROBOTS_MODULE = StationCommand.ROBOTS_PACKAGE  # whose packages are the robots, each by its name
# each robot's first value that its shipped configuration leaves for a station to give
FIRST_PLACEHOLDER = {"yam": "arms.left.channel", "so101": "hardware.port"}
SO101_PORT = "/dev/ttyACM0"  # a serial port a station's file gives
# the model the station's recipe offers, as its command line gives it: a served model through its OpenAI API
RECIPE_MODEL = (
    "default_model=served",
    "models.served.label=Served",
    "models.served.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
    "models.served.client.base_url=http://localhost:8000/v1",
    "models.served.client.model=served",
)


class UnfilledStation:
    """A station's own file that names its robot and gives nothing else: the robot's shipped configuration as it is."""

    @staticmethod
    def write(robot_name: str, directory: Path) -> Path:
        """Writes the station's own file of the robot ``robot_name`` into ``directory`` and returns its path."""
        path = directory / "config.yaml"
        path.write_text(f"robot_class: {ROBOT_COMMANDS[robot_name].ROBOT_CLASS_PATH}\n")

        return path


def test_nested_value_is_named_by_its_dotted_field():
    assert RobotConfig.placeholder_key({"arms": {"left": {"channel": "<CAN interface>"}}}, "") == "arms.left.channel"


def test_list_item_is_named_by_its_index():
    assert RobotConfig.placeholder_key({"cameras": [{"serial": "1"}, {"serial": "<serial>"}]}, "") == "cameras.1.serial"


def test_concrete_values_are_not_placeholders():
    assert RobotConfig.placeholder_key({"channel": "can0", "note": "a < b", "rate": 50, "ids": ["<", "x>"]}, "") is None


def test_file_with_a_placeholder_is_rejected_with_field_path_and_docs(tmp_path):
    commands_class = ROBOT_COMMANDS["so101"]
    path = tmp_path / ConfigFiles.STATION_FILE_NAME
    path.write_text(
        f"robot_class: {commands_class.ROBOT_CLASS_PATH}\nrobot:\n  hardware:\n    port: <serial port, see README.md>\n"
    )
    parser = argparse.ArgumentParser()
    commands_class.add_arguments(parser)
    command = commands_class.from_arguments(parser.parse_args(["calibrate", "--config", str(path)])).command

    with pytest.raises(
        ValueError, match=rf"hardware.port is a placeholder in {path}; see the so101 setup in README.md"
    ):
        command.run()


def test_file_without_a_placeholder_is_accepted():
    config = ConfigFiles.load("so101", f"robot.hardware.port={SO101_PORT}")

    _, robot_config = ConfigFiles.make_configs(config)

    assert robot_config.first_placeholder() is None


@pytest.mark.parametrize("robot", ["yam", "so101"])
def test_unfilled_template_loads_for_device_discovery(robot, tmp_path, monkeypatch):
    commands_class = ROBOT_COMMANDS[robot]
    # the devices the operating system lists are not this test's subject: none
    startup_checks = pkgutil.resolve_name(f"{ROBOTS_MODULE}.{robot}.hardware.startup.StartupChecks")
    monkeypatch.setattr(startup_checks, "inventory", classmethod(lambda cls: {}))
    parser = argparse.ArgumentParser()
    commands_class.add_arguments(parser)
    args = parser.parse_args(["check", "--config", str(UnfilledStation.write(robot, tmp_path))])

    result = commands_class.from_arguments(args).command.run()

    assert (
        result["motion_blockers"][0]
        == f"{FIRST_PLACEHOLDER[robot]} is a placeholder; see the {robot} setup in README.md"
    )


@pytest.mark.parametrize("robot", ["yam", "so101"])
def test_preflight_names_the_placeholder_before_opening_a_device(robot, tmp_path, capsys, monkeypatch):
    arguments = ["--config", str(UnfilledStation.write(robot, tmp_path))]
    monkeypatch.setattr(sys, "argv", ["pear-station", robot, "preflight", *arguments])

    with pytest.raises(SystemExit) as exit_info:
        StationCommand.from_command_line().run()
    code = exit_info.value.code

    assert code == 1
    assert f"{FIRST_PLACEHOLDER[robot]} is a placeholder; see the {robot} setup in README.md" in capsys.readouterr().err


@pytest.mark.parametrize("robot", ["yam", "so101"])
def test_real_station_names_the_placeholder_before_opening_a_device(robot, tmp_path):
    station = pkgutil.resolve_name(ROBOT_COMMANDS[robot].ROBOT_CLASS_PATH)
    config = BaseRobot.load_config(UnfilledStation.write(robot, tmp_path), [])

    with pytest.raises(
        RuntimeError, match=f"{FIRST_PLACEHOLDER[robot]} is a placeholder; see the {robot} setup in README.md"
    ):
        station(config, action_mode=ActionMode.JOINT).open()


@pytest.mark.parametrize("robot", ["yam", "so101"])
def test_stand_in_config_opens(robot):
    config = ConfigFiles.load(robot)

    with StandInRobots.robot_class(robot)(config, cameras=None) as station:
        assert station.status()["status"]


@pytest.mark.parametrize("robot", ["yam", "so101"])
def test_models_config_loads_and_names_its_robot(robot):
    models = ModelsConfig.from_station_config(ConfigFiles.load(robot, *RECIPE_MODEL))

    assert models.robot_type.NAME == robot
    assert models.default_model in models.models
