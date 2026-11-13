import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.commands import AccuracyCheck
from pear.deploy.station.control.robots.so101.commands.accuracy_check import SO101AccuracyCheck
from pear.deploy.station.control.robots.so101.commands.commands import SO101Commands
from pear.deploy.station.control.robots.so101.joint.feedforward import GravityModel
from pear.deploy.station.control.robots.so101.layout import ARM

STATION_FILE = ConfigFiles.station_file("so101")  # the own file of the SO-101's station of the tests


@pytest.fixture
def run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[int, str, str]]:
    """Runs `pear-station so101 accuracy-check <options>` on the station of the tests with its quick settings and
    no wait for the fade, its run directory the test's; returns exit code, stdout and stderr."""
    monkeypatch.setattr(AccuracyCheck, "FADE_WAIT_S", 0.0)

    def execute(*options: str, overrides: Sequence[str] = ()) -> tuple[int, str, str]:
        command_line = [
            "accuracy-check",
            "--config",
            str(STATION_FILE),
            *options,
            *ConfigFiles.SO101_QUICK_OVERRIDES,
            *overrides,
            f"run_dir={tmp_path}",
        ]
        monkeypatch.setattr(sys, "argv", ["pear-station", "so101", *command_line])
        try:
            StationCommand.from_command_line().run()
            code = 0
        except SystemExit as stop:
            code = stop.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return execute


def test_the_poses_are_safe_and_load_lift_or_elbow_both_ways():
    _, robot_config = ConfigFiles.load_configs("so101")
    command = SO101AccuracyCheck(argparse.Namespace(), SO101Commands.ROBOT_CLASS_PATH)
    home = np.array(command.home_joints(robot_config, ARM))
    gravity_model = GravityModel(robot_config.geometry.urdf)

    min_travel, loaded = AccuracyCheck.MIN_LOADED_TRAVEL_RAD, SO101AccuracyCheck.LOADED_JOINTS
    signs = set()  # of the travel times the gravity torque of each loaded joint that travels far enough
    for joints in command.poses(robot_config).values():
        command.check_pose(robot_config, ARM, [*joints, 1.0])
        travel = np.asarray(joints) - home
        assert np.abs(travel[list(loaded)]).max() >= min_travel
        torque = gravity_model.torque(joints)
        signs |= {np.sign(travel[j] * torque[j]) for j in loaded if abs(travel[j]) >= min_travel}

    assert {-1.0, 1.0} <= signs  # with gravity and against it


def test_accuracy_check_passes_and_records_both_passes(run, tmp_path):
    code, output, error = run("--yes")

    assert code == 0, error
    assert "PASS" in output
    (path,) = (tmp_path / AccuracyCheck.OUTPUT_FOLDER).glob("*.json")
    passes = json.loads(path.read_text())["passes"]
    assert passes["uncompensated"]["compensation"] == {
        name: [0.0] * 5 for name in ("gravity_compliance", "friction_offset_rad", "friction_load_ratio")
    }
    assert passes["compensated"]["compensation"]["gravity_compliance"] == [0.0, 0.061, 0.068, 0.074, 0.0]


def test_accuracy_check_refuses_an_arm_without_its_feedforward(run):
    code, _, error = run("--yes", overrides=["robot.feedforward=null"])

    assert code == 1
    assert "run friction-sweep" in error
