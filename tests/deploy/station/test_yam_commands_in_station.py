import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.yam.commands.commands import Cameras, Snapshot
from pear.deploy.station.control.robots.yam.commands.friction_sweep import FrictionSweep
from pear.deploy.station.control.robots.yam.commands.gripper_calibration import GripperCalibration
from pear.deploy.station.control.robots.yam.hardware.cameras import RealSenseCameras
from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.yam.robot import YAMRobot

# a wide station: limits wide enough for every move of the tests, a velocity tolerance of 0.03 rad/s, Home at a third
# of the move caps with no caps of its own and 20 s for its arrival, and no cameras
WIDE_STATION = (
    *ConfigFiles.WIDE_LIMITS["yam"],
    "robot.velocity_tolerance=0.03",
    "robot.home.speed_scale=0.3333333333333333",
    "robot.home.limits=null",
    "robot.home.arrival_timeout_s=20",
    "robot.cameras=[]",
)
STATION_FILE = ConfigFiles.station_file("yam")  # the own file of the tests' YAM station


@pytest.fixture
def run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[int, str, str]]:
    """Run `pear-station yam <arguments>` on the wide station configuration of the tests, its runs under
    ``tmp_path``; return exit code, stdout and stderr."""

    def execute(
        *arguments: str, config: Path = STATION_FILE, overrides: tuple[str, ...] = WIDE_STATION
    ) -> tuple[int, str, str]:
        """Returns the exit code and the two outputs of the YAM command ``arguments`` with the station's file
        ``config`` and the ``overrides``; a command that reads no configuration is given neither."""
        command, *command_arguments = arguments
        if command == Cameras.NAME:
            config_arguments, override_arguments = [], []
        else:
            config_arguments, override_arguments = ["--config", str(config)], [*overrides, f"run_dir={tmp_path}"]
        monkeypatch.setattr(
            sys, "argv", ["pear-station", "yam", command, *config_arguments, *command_arguments, *override_arguments]
        )

        try:
            StationCommand.from_command_line().run()
            code = 0
        except SystemExit as exit_error:
            code = exit_error.code

        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return execute


def test_robot_help_has_no_interactive_control(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pear-station", "yam", "--help"])

    with pytest.raises(SystemExit) as result:
        StationCommand.from_command_line()

    assert result.value.code == 0
    output = capsys.readouterr().out
    for name in (
        "check",
        "preflight",
        "calibrate-gripper",
        "cameras",
        "snapshot",
        "smoke",
        "friction-sweep",
        "accuracy-check",
    ):
        assert name in output
    assert "session" not in output and "motor-status" not in output


def test_check_reports_inventory_and_blockers(run, monkeypatch):
    monkeypatch.setattr(StartupChecks, "inventory", lambda: {"interfaces": ["stand_in_can"]})

    code, output, _ = run("check")

    assert code == 0
    assert json.loads(output)["interfaces"] == ["stand_in_can"]
    assert "motion_blockers" in json.loads(output)


def test_preflight_prints_the_report(run, monkeypatch):
    monkeypatch.setattr(StartupChecks, "preflight", lambda config: {"arms": sorted(config.arms)})

    code, output, _ = run("preflight")

    assert code == 0
    assert json.loads(output) == {"arms": ["left", "right"]}


def test_cameras_prints_the_discovery(run, monkeypatch):
    monkeypatch.setattr(RealSenseCameras, "discover", lambda: [{"serial": "1"}])

    assert run("cameras")[:2] == (0, json.dumps([{"serial": "1"}], indent=2) + "\n")


def test_calibrate_gripper_runs_for_the_side(run, monkeypatch):
    monkeypatch.setattr(GripperCalibration, "run", lambda config, side, output_dir: {"side": side})

    # --yes before the arm: the overrides that follow the arm are read only right after it
    code, output, _ = run("calibrate-gripper", "--yes", "right")

    assert code == 0
    assert json.loads(output) == {"side": "right"}


def test_calibrate_gripper_refuses_a_template_placeholder(run, monkeypatch):
    sides = []
    monkeypatch.setattr(
        GripperCalibration, "run", lambda config, side, output_dir: sides.append(side) or {"side": side}
    )
    template = YAMRobot.CONFIG_PATH

    code, _, error = run("calibrate-gripper", "--yes", "right", config=template, overrides=())

    assert code == 1
    assert f"arms.left.channel is a placeholder in {template}; see the yam setup in README.md" in error
    assert sides == []


def test_snapshot_passes_the_frame_count_and_the_output(run, monkeypatch, tmp_path):
    received = {}

    def capture(serials, output, frames, roles):
        received.update(serials=serials, output=output, frames=frames, roles=roles)
        return {"frames": frames}

    monkeypatch.setattr(RealSenseCameras, "snapshot", capture)

    code, _, error = run("snapshot", "--frames", "3")

    assert code == 0
    assert received["frames"] == 3 and received["output"] == tmp_path / Snapshot.OUTPUT_FOLDER
    assert received["serials"] == [] and received["roles"] == {}


def test_snapshot_rejects_too_many_frames(run):
    code, _, error = run("snapshot", "--frames", "18001")

    assert code == 1
    assert "Use 1..18000 frames" in error


def test_smoke_moves_a_joint_and_returns(run):
    code, output, _ = run("smoke", "--delta", "0.02", "--yes")

    assert code == 0
    report = json.loads(output)
    assert np.allclose(report["before"]["targets"]["left"], report["returned"]["targets"]["left"], atol=1e-6)
    assert not np.allclose(report["before"]["targets"]["left"], report["moved"]["targets"]["left"], atol=1e-6)


def test_smoke_rejects_a_large_delta(run):
    code, _, error = run("smoke", "--delta", "0.5", "--yes")

    assert code == 1
    assert "at most 0.05 rad" in error


def test_real_motion_needs_confirmation(run, monkeypatch):
    monkeypatch.setattr(GripperCalibration, "run", lambda config, side, output_dir: pytest.fail("moved"))

    for arguments in (("smoke",), ("calibrate-gripper", "left")):
        code, _, error = run(*arguments)

        assert code == 1
        assert f"{arguments[0]} moves the real hardware; add --yes to confirm" in error


# the keys of an arm that the friction sweep prints for the station file
FEEDFORWARD_KEYS = ("gravity_compensation_factors", "friction_feedforward_nm", "friction_load_ratio")


@pytest.fixture
def synthetic_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the friction sweep move nothing of its own and find, at every speed, a friction of 0.5 + 2 * speed N*m and
    no gravity model error, with J2's model gravity -5 N*m lifted and -11 N*m low and J3's 5 N*m."""

    def analyse(samples, joint, speed, position_gains, damping_gains, uses_velocity_feedforward):
        """Returns the synthetic result of one sweep of ``joint`` at ``speed``, with every value a real one has."""
        friction = 0.5 + 2 * speed
        return {
            "joint": joint + 1,
            "speed": speed,
            "bins": FrictionSweep.BIN_COUNT,
            "needed_torque_plus_nm": friction,
            "needed_torque_minus_nm": -friction,
            "friction_nm": friction,
            "gravity_model_error_nm": 0.0,
            "friction_spread_nm": 0.0,
            "effort_friction_nm": friction,
            "breakaway_nm": {"plus": friction, "minus": friction},
        }

    def model_gravity(pose):
        """Returns the synthetic model gravity at ``pose``, one of the sweep's poses."""
        return np.array([0.0, -5.0 if list(pose) == FrictionSweep.POSES["lifted"] else -11.0, 5.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(FrictionSweep, "sweep", lambda *args, **kwargs: [])
    monkeypatch.setattr(FrictionSweep, "analyse", staticmethod(analyse))
    monkeypatch.setattr(FrictionSweep, "model_gravity", staticmethod(model_gravity))


def paste_under_left_arm(text: str, snippet: str) -> str:
    """Returns the station file ``text`` with the left arm's feedforward keys dropped and ``snippet`` put right under
    its ``left:`` line, as the operator pastes it."""
    pasted, in_left_arm = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("    ") and not line.startswith("     "):  # an arm's line opens its section
            in_left_arm = line.strip() == "left:"
        if in_left_arm and line.strip().split(":")[0] in FEEDFORWARD_KEYS:
            continue
        pasted.append(line)
        if line.rstrip() == "    left:":
            pasted.append(snippet)

    return "".join(pasted)


def test_friction_sweep_takes_both_poses_by_default(run, monkeypatch):
    seen = []
    monkeypatch.setattr(FrictionSweep, "run", lambda sweep, is_confirmed: seen.append(sweep.poses))

    run("friction-sweep", "--side", "left")
    # --poses before --side: the overrides that follow the command's own words would be read as poses
    run("friction-sweep", "--poses", "low", "--side", "left")

    assert seen == [["lifted", "low"], ["low"]]


def test_the_sweep_runs_with_the_arms_friction_feedforward_off(run, monkeypatch):
    seen = []
    monkeypatch.setattr(
        FrictionSweep, "run", lambda sweep, is_confirmed: seen.append(sweep.robot.robot_config.arms["left"])
    )

    run("friction-sweep", "--side", "left")

    [arm] = seen
    assert list(arm.friction_feedforward_nm) == [0] * 6 and list(arm.friction_load_ratio) == [0] * 6
    assert list(arm.gravity_compensation_factors) == [1, 1.20, 1.26, 1.2, 1, 1]


def test_a_finished_sweep_prints_a_snippet_the_station_file_takes(run, synthetic_sweeps, tmp_path):
    code, output, _ = run("friction-sweep", "--side", "left", "--yes")

    assert code == 0
    snippet = output[output.index("# yam friction-sweep") :]
    path = tmp_path / "pasted.yaml"
    path.write_text(paste_under_left_arm(STATION_FILE.read_text(), snippet))
    _, robot_config = ConfigFiles.make_configs(BaseRobot.load_config(path, []))
    assert list(robot_config.arms["left"].friction_feedforward_nm)[0] == pytest.approx(0.85 * 0.5, abs=2e-3)
    assert list(robot_config.arms["right"].friction_load_ratio) == [0, 0.197, 0, 0, 0, 0]  # untouched
