import argparse
import pkgutil
import re
import sys

import pytest
from omegaconf import OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.robot import BaseRobot

ROBOT_COMMANDS = dict(StationCommand.robot_commands())  # each robot package's maintenance commands, by its name
YAM_PORT = 18900  # a port of the page's service that a station's own file gives, beside the robot's default


@pytest.mark.parametrize("name", ROBOT_COMMANDS)
def test_robot_declares_its_subcommands(name):
    commands_class = ROBOT_COMMANDS[name]
    parser = argparse.ArgumentParser(prog=f"pear-station {name}")

    commands_class.add_arguments(parser)

    assert callable(commands_class.run)


@pytest.mark.parametrize("name", ROBOT_COMMANDS)
def test_camera_hook_is_a_class_member(name):
    robot_class = pkgutil.resolve_name(ROBOT_COMMANDS[name].ROBOT_CLASS_PATH)
    assert "." in robot_class.open_cameras.__qualname__


def test_a_station_s_own_file_is_read_over_its_robot_s_defaults(tmp_path):
    robot_class = pkgutil.resolve_name(ROBOT_COMMANDS["yam"].ROBOT_CLASS_PATH)
    station_file = tmp_path / "config.yaml"
    station_file.write_text(f"robot_class: {ROBOT_COMMANDS['yam'].ROBOT_CLASS_PATH}\nport: {YAM_PORT}\n")

    config = BaseRobot.load_config(station_file, [])

    assert config.port == YAM_PORT
    defaults = OmegaConf.load(robot_class.CONFIG_PATH)
    assert config.robot.velocity_tolerance == defaults.robot.velocity_tolerance


def test_loading_an_unknown_robot_lists_the_installed_ones(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pear-station", "nonexistent", "check"])

    with pytest.raises(SystemExit):
        StationCommand.from_command_line()

    assert re.search(r"choose from .*yam", capsys.readouterr().err)


def test_available_robots_load_with_their_own_name():
    for name, commands_class in ROBOT_COMMANDS.items():
        assert pkgutil.resolve_name(commands_class.ROBOT_CLASS_PATH).NAME == name


def test_config_rejects_duplicate_devices():
    config = ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"])
    config.robot.arms.right.channel = config.robot.arms.left.channel
    with pytest.raises(ValueError):
        ConfigFiles.make_configs(config)
