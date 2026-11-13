import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.commands import AccuracyCheck
from pear.deploy.station.control.robot import ActionMode
from pear.deploy.station.control.robots.yam.commands.accuracy_check import YAMAccuracyCheck
from pear.deploy.station.control.robots.yam.commands.commands import YAMCommands
from pear.deploy.station.control.robots.yam.commands.friction_sweep import FrictionSweep
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from stand_in_robots import StandInYAMRobot

STATION_FILE = ConfigFiles.station_file("yam")  # the own file of the tests' YAM station
# the wide station of the YAM's command tests, with fast caps: each pose is reached in a moment
QUICK_STATION = (
    *ConfigFiles.WIDE_LIMITS["yam"],
    *(
        f"robot.arms.{arm}.limits.{name}={values}"
        for arm in ("left", "right")
        for name, values in (("velocity", [3] * 7), ("acceleration", [20] * 7), ("jerk", [100] * 7))
    ),
    "robot.velocity_tolerance=0.03",
    "control.settle_window_s=0.1",
    "robot.home.settle_window_s=0.1",
    "robot.home.speed_scale=1",
    "robot.home.limits=null",
    "robot.home.arrival_timeout_s=20",
    "robot.cameras=[]",
)
PACKAGE_FACTORS = [1, 1.1, 1.1, 1.2, 1, 1]  # the gravity factors of the package's config.yaml


@pytest.fixture
def run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[int, str, str]]:
    """Run `pear-station yam <arguments>` on the quick station configuration of the tests, its runs under
    ``tmp_path`` and no wait for the feedforward's fade; return exit code, stdout and stderr."""
    monkeypatch.setattr(AccuracyCheck, "FADE_WAIT_S", 0)

    def execute(*arguments: str, overrides: tuple[str, ...] = QUICK_STATION) -> tuple[int, str, str]:
        """Returns the exit code and the two outputs of the YAM command ``arguments`` with the tests' station file
        and the ``overrides``."""
        command, *command_arguments = arguments
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pear-station",
                "yam",
                command,
                "--config",
                str(STATION_FILE),
                *command_arguments,
                *overrides,
                f"run_dir={tmp_path}",
            ],
        )

        try:
            StationCommand.from_command_line().run()
            code = 0
        except SystemExit as exit_error:
            code = exit_error.code

        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return execute


def test_the_poses_of_both_sides_are_safe_and_reached_with_and_against_gravity():
    robot_config = StandInYAMRobot(YAMRobot.load_config(STATION_FILE, []), action_mode=ActionMode.JOINT).robot_config

    for side in ("left", "right"):
        command = YAMAccuracyCheck(argparse.Namespace(side=side), YAMCommands.ROBOT_CLASS_PATH)
        home = np.asarray(command.home_joints(robot_config, side))
        directions = {joint: set() for joint in YAMAccuracyCheck.LOADED_JOINTS}  # travel against gravity: > 0
        for joints in command.poses(robot_config).values():
            command.check_pose(robot_config, side, [*joints, 1.0])
            travel = np.asarray(joints) - home
            assert max(abs(travel[joint]) for joint in YAMAccuracyCheck.LOADED_JOINTS) >= 0.3
            gravity = FrictionSweep.model_gravity(list(joints))
            for joint in YAMAccuracyCheck.LOADED_JOINTS:
                if abs(travel[joint]) >= 0.3:
                    directions[joint].add(bool(travel[joint] * gravity[joint] > 0))

        assert any(len(signs) == 2 for signs in directions.values()), (side, directions)


def test_accuracy_check_passes_the_stand_in_arm_and_records_both_passes(run, tmp_path):
    code, output, error = run("accuracy-check", "--side", "left", "--yes")

    assert code == 0, error
    assert output.splitlines()[-1] == "PASS"
    [record_path] = (tmp_path / "accuracy").glob("accuracy-left-*.json")
    passes = json.loads(record_path.read_text())["passes"]
    uncompensated = passes["uncompensated"]["compensation"]
    assert uncompensated["gravity_compensation_factors"] == PACKAGE_FACTORS
    assert uncompensated["friction_feedforward_nm"] == [0] * 6
    assert uncompensated["friction_load_ratio"] == [0] * 6
    assert passes["compensated"]["compensation"]["friction_load_ratio"] == [0, 0.183, 0, 0, 0, 0]


def test_accuracy_check_refuses_an_arm_with_the_package_feedforward(run):
    package_values = (
        f"robot.arms.left.gravity_compensation_factors={PACKAGE_FACTORS}",
        f"robot.arms.left.friction_feedforward_nm={[0] * 6}",
        f"robot.arms.left.friction_load_ratio={[0] * 6}",
    )

    code, _, error = run("accuracy-check", "--side", "left", "--yes", overrides=(*QUICK_STATION, *package_values))

    assert code == 1
    assert "run friction-sweep" in error
