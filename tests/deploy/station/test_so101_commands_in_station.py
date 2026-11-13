import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml

from config_files import ConfigFiles
from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.so101.commands.commands import SO101Commands
from pear.deploy.station.control.robots.so101.commands.friction_sweep import FrictionSweep
from pear.deploy.station.control.robots.so101.commands.home_check import HomeCheck
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.so101.layout import JOINTS
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.robots.so101.robot import SO101Robot
from stand_in_joint_driver import StandInSO101Arm, StandInSO101Driver
from stand_in_robots import StandInSO101Robot

STATION_FILE = ConfigFiles.station_file("so101")  # the own file of the SO-101's station of the tests
ZERO_HOME_OVERRIDE = "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]"  # Home: every joint at zero, the gripper open
# what the tests of the arm change of the station of the tests besides its quick settings and its Home: the
# tolerances they were written for (Home's, an arrival's, the feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)


class StandInSaggingArm(StandInSO101Arm):
    """Starts away from Home; the elbow stops short of every command, as under gravity."""

    START = (0.0, -1.0, 1.0, 0.5, 0.0, 0.0)  # where the arm starts
    SAG = 0.05  # elbow_flex off its command, beyond the Home tolerance
    made: ClassVar[list["StandInSaggingArm"]] = []  # every arm of the class, the latest last

    def __init__(self, control_config: ControlConfig, robot_config: SO101Config) -> None:
        super().__init__(control_config, robot_config)
        self.position = self.START
        StandInSaggingArm.made.append(self)

    def send(self, target: Sequence[float]) -> None:
        """Puts the arm at ``target`` but for its elbow, which stops short."""
        super().send(target)
        q = list(self.position)
        q[2] += self.SAG
        self.position = tuple(q)


class StandInBus:
    """A servo bus that reports the same raw position for every motor."""

    def sync_read(self, register: str, num_retry: int = 0) -> dict[str, float]:
        """Returns the raw value of ``register`` of every motor."""
        return {name: 0.0 for name in (*JOINTS, "gripper")}

    def disconnect(self, disable_torque: bool = True) -> None:
        """Lets the bus go."""
        self.disconnected = True


class StandInFollower:
    """A lerobot follower that records what the calibration command asks of it."""

    made: ClassVar[list["StandInFollower"]] = []  # every follower of the class, the latest last

    def __init__(self, config: Any) -> None:
        self.config = config
        self.calls: list[str] = []  # what the follower was asked, in order
        self.calibration_fpath = Path(config.calibration_dir) / f"{config.id}.json"
        self.bus = self
        StandInFollower.made.append(self)

    def connect(self) -> None:
        """Records the connection."""
        self.calls.append("connect")

    def calibrate(self) -> None:
        """Records the calibration."""
        self.calls.append("calibrate")

    def disconnect(self) -> None:
        """Records the disconnection."""
        self.calls.append("disconnect")


class StandInCamera:
    """A wrist camera that returns one flat frame."""

    def __init__(self, config: SO101Config) -> None:
        self.role = config.camera.role

    def read(self) -> dict[str, dict[str, np.ndarray]]:
        """Returns the one flat frame under the camera's role, without the stamps of a camera's frame."""
        return {self.role: {"rgb": np.full((4, 6, 3), 7, dtype=np.uint8)}}

    def close(self) -> None:
        """Lets the camera go."""
        self.closed = True


@pytest.fixture
def run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[int, str, str]]:
    """Runs `pear-station so101 <arguments>` on the station of the tests, by default with Home at zero and the
    tolerances of the tests of the arm, its run directory the test's; returns exit code, stdout and stderr."""

    def execute(
        *arguments: str,
        config: Path = STATION_FILE,
        overrides: Sequence[str] = (ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES),
    ) -> tuple[int, str, str]:
        command, *options = arguments
        command_line = [command, "--config", str(config), *options, *overrides, f"run_dir={tmp_path}"]
        monkeypatch.setattr(sys, "argv", ["pear-station", "so101", *command_line])
        try:
            StationCommand.from_command_line().run()
            code = 0
        except SystemExit as stop:
            code = stop.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return execute


def test_check_reports_inventory_and_blockers(run, monkeypatch):
    monkeypatch.setattr(StartupChecks, "inventory", lambda: {"serial_ports": []})

    code, output, error = run("check")

    assert code == 0, error
    assert json.loads(output)["serial_ports"] == []
    assert "motion_blockers" in json.loads(output)


def test_preflight_prints_the_report(run, monkeypatch):
    monkeypatch.setattr(StartupChecks, "preflight", lambda config: {"servos": 6})

    assert json.loads(run("preflight")[1]) == {"servos": 6}


def test_pose_reads_the_arm_without_torque(run, monkeypatch):
    monkeypatch.setattr(StartupChecks, "lock_bus", lambda port: type("Lease", (), {"close": lambda self: None})())
    monkeypatch.setattr(StartupChecks, "make_follower", lambda hardware: type("Robot", (), {"bus": StandInBus()})())
    monkeypatch.setattr(StartupChecks, "connect_checked", lambda robot: None)

    code, output, error = run("pose", overrides=())

    assert code == 0, error
    report = json.loads(output)
    assert set(report) == {"joints_rad", "joints_deg", "single_arm_pose"}


def test_snapshot_saves_one_frame(run, monkeypatch):
    monkeypatch.setattr(StandInSO101Robot, "CAMERA_SOURCE_CLASS", StandInCamera)

    code, output, error = run("snapshot", overrides=())

    assert code == 0, error
    assert Path(json.loads(output)["path"]).is_file()


def test_calibrate_uses_the_stations_port_and_directory(run, monkeypatch, tmp_path):
    monkeypatch.setattr("lerobot.robots.so_follower.SO101Follower", StandInFollower)

    code, output, error = run("calibrate", overrides=(f"robot.hardware.calibration_dir={tmp_path / 'out'}",))

    follower = StandInFollower.made[-1]
    assert code == 0, error
    assert follower.calls == ["connect", "calibrate", "disconnect"]
    assert follower.config.id == "so101_follower"
    assert json.loads(output)["calibration"] == str(tmp_path / "out" / "so101_follower.json")


def test_calibrate_refuses_a_template_placeholder(run, monkeypatch):
    monkeypatch.setattr("lerobot.robots.so_follower.SO101Follower", StandInFollower)
    made = len(StandInFollower.made)

    template = SO101Robot.CONFIG_PATH

    code, _, error = run("calibrate", config=template, overrides=())

    assert code == 1
    assert "hardware.port" in error
    assert len(StandInFollower.made) == made


def test_smoke_moves_one_joint_and_returns(run):
    code, output, error = run("smoke", "--yes", "--joint", "elbow_flex", "--delta", "0.05")

    assert code == 0, error
    report = json.loads(output)
    assert report["moved"]["target"][2] == pytest.approx(report["before"]["target"][2] + 0.05, abs=1e-3)
    assert report["returned"]["target"] == pytest.approx(report["before"]["target"], abs=1e-3)


def test_smoke_rejects_a_large_delta(run):
    code, _, error = run("smoke", "--yes", "--delta", "0.5")

    assert code == 1
    assert "at most 0.1 rad" in error


def test_real_motion_needs_confirmation(run):
    for command in ("smoke", "home", "steps", "safety"):
        code, _, error = run(command)

        assert code == 1
        assert f"{command} moves the arm; add --yes to confirm" in error


@pytest.mark.parametrize("command", [["home"], ["steps", "--photos"]], ids=["home", "steps"])
def test_uncommissioned_real_motion_opens_no_camera(run, monkeypatch, tmp_path, command):
    opened = []
    monkeypatch.setattr(
        StandInSO101Robot,
        "CAMERA_SOURCE_CLASS",
        staticmethod(lambda config: opened.append(config) or StandInCamera(config)),
    )
    # The station of the tests with a calibration directory that holds no calibration file.
    data = yaml.safe_load(STATION_FILE.read_text())
    data["robot"]["hardware"]["calibration_dir"] = str(tmp_path)
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump(data))

    code, _, error = run(*command, "--yes", config=station, overrides=())

    assert code == 1
    assert "Real motion is not commissioned" in error
    assert opened == []


def test_home_photographs_and_returns(run):
    code, output, error = run("home", "--yes")

    assert code == 0, error
    assert json.loads(output)["returned"]["reached"] is True


def test_steps_reports_commanded_and_measured_change(run):
    code, output, error = run("steps", "--yes", "--deltas", '[{"x": 1}]')

    assert code == 0, error
    (step,) = json.loads(output)
    assert step["commanded"] == {"x": 1.0}
    assert step["measured"]["x"] == pytest.approx(1.0, abs=0.05)
    assert step["status"] == "completed"


def test_steps_rejects_an_unknown_quantity(run):
    code, _, error = run("steps", "--yes", "--deltas", '[{"nope": 1}]')

    assert code == 1
    assert "has no quantities ['nope']" in error


def test_safety_rejects_the_unreachable_step_and_stops(run):
    code, output, error = run("safety", "--yes")

    assert code == 0, error
    report = json.loads(output)
    assert report["unreachable"]["passed"] is True, report
    assert report["stop"]["passed"] is True, report


def test_failed_home_returns_to_the_start_before_release(monkeypatch, tmp_path):
    overrides = [
        *ConfigFiles.SO101_QUICK_OVERRIDES,
        ZERO_HOME_OVERRIDE,
        *ARM_TEST_OVERRIDES,
        "robot.home.arrival_timeout_s=0.5",
        f"run_dir={tmp_path}",
    ]
    monkeypatch.setattr(StandInSO101Driver, "HARDWARE_CLASS", StandInSaggingArm)
    arguments = argparse.Namespace(config=STATION_FILE, overrides=overrides, yes=True)

    with pytest.raises(HomeFailed, match="Home feedback"):
        HomeCheck(arguments, SO101Commands.ROBOT_CLASS_PATH).run()

    arm = StandInSaggingArm.made[-1]
    assert arm.is_closed
    sent = np.array(arm.sent_targets)
    home = np.array(ConfigFiles.load("so101", *overrides).robot.home.position)
    assert np.abs(sent - home).max(axis=1).min() < 1e-6  # Home was commanded
    np.testing.assert_allclose(sent[-1], StandInSaggingArm.START, atol=1e-6)  # then the start, before release


def test_friction_sweep_receives_its_own_arguments(monkeypatch):
    received = []
    monkeypatch.setattr(FrictionSweep, "run", lambda command: received.append(command.args))
    monkeypatch.setattr(sys, "argv", ["pear-station", "so101", "friction-sweep", "--config", "station.yaml", "--yes"])

    StationCommand.from_command_line().run()

    [args] = received
    assert (args.config, args.yes, args.poses, args.feedforward, args.overrides) == (
        Path("station.yaml"),
        True,
        list(FrictionSweep.POSES),
        False,
        [],
    )


def test_the_sweep_plan_passes_its_checks_on_the_commissioned_arm():
    _, config = ConfigFiles.load_configs("so101")

    sweeps = FrictionSweep.plan(config, SO101Kinematics.shared_instance(config), list(FrictionSweep.POSES))

    assert [(s["pose"], s["joint"]) for s in sweeps] == [
        ("home", 0),
        ("home", 1),
        ("home", 2),
        ("home", 3),
        ("home", 4),
        ("reach", 1),
        ("reach", 2),
        ("reach", 3),
    ]


@pytest.fixture
def so101_run(run: Callable[..., tuple[int, str, str]]) -> Callable[..., tuple[int, str, str]]:
    """Runs `pear-station so101 <arguments>` on the station of the tests with its quick settings, its Home and its
    feedforward as they are; returns exit code, stdout and stderr."""

    def execute(*arguments: str, overrides: Sequence[str] = ConfigFiles.SO101_QUICK_OVERRIDES) -> tuple[int, str, str]:
        return run(*arguments, overrides=overrides)

    return execute


def staircase_rows(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns the steps of ``sweep`` as an arm measures them whose shoulder_lift has the compliance and the
    load-dependent friction of the synthetic fit case; every other joint lands on its targets."""
    c, f0, ratio = 0.08, 0.01, 0.02  # shoulder_lift
    joint, gravity = sweep["joint"], {"home": -0.1, "reach": -0.42}[sweep["pose"]]
    rows = []
    for i in range(1, 4):
        g = gravity * (1 + 0.05 * i)
        for direction, sign in (("up", -1), ("down", 1)):
            error, torque = [0.0] * 5, [0.0] * 5
            if joint == 1:
                error[1], torque[1] = -c * g + sign * (f0 + ratio * abs(g)), g
            rows.append({"index": i, "direction": direction, "error": error, "gravity": torque})
    return rows


def paste_feedforward_under_robot(text: str, snippet: str) -> str:
    """Returns the station file ``text`` without its feedforward section and with ``snippet`` under ``robot:``, as
    the operator pastes it."""
    lines = text.splitlines(keepends=True)
    start = lines.index("  feedforward:\n")
    del lines[start : start + 4]
    at = lines.index("robot:\n") + 1
    return "".join([*lines[:at], snippet, *lines[at:]])


def test_a_finished_sweep_prints_a_feedforward_the_station_file_takes(so101_run, monkeypatch, tmp_path):
    monkeypatch.setattr(FrictionSweep, "run_sweep", lambda command, control_loop, model, sweep: staircase_rows(sweep))

    code, output, error = so101_run("friction-sweep", "--yes")

    assert code == 0, error
    snippet = output[output.index("# so101 friction-sweep") :]
    station = paste_feedforward_under_robot(STATION_FILE.read_text(), snippet)
    path = tmp_path / "pasted.yaml"
    path.write_text(station)
    _, robot_config = ConfigFiles.make_configs(BaseRobot.load_config(path, []))
    assert robot_config.feedforward.gravity_compliance[1] == pytest.approx(0.08, abs=1e-3)
    assert robot_config.feedforward.friction_offset_rad[1] == pytest.approx(0.85 * 0.01, abs=1e-3)
    assert robot_config.feedforward.friction_load_ratio[1] == pytest.approx(0.85 * 0.02, abs=1e-4)
    assert robot_config.hardware.port == "/dev/stand-in-so101-serial"  # untouched
    (record_path,) = (tmp_path / "friction").glob("*.json")
    record = json.loads(record_path.read_text())
    assert record["suggested_feedforward"] == yaml.safe_load(snippet)["feedforward"]
    assert record["warnings"] == []


def test_a_residual_sweep_prints_no_feedforward_to_paste(so101_run, monkeypatch):
    monkeypatch.setattr(FrictionSweep, "run_sweep", lambda command, control_loop, model, sweep: staircase_rows(sweep))

    code, output, error = so101_run("friction-sweep", "--feedforward", "--yes")

    assert code == 0
    assert "feedforward:" not in output  # residuals: pasting them would replace the arm's feedforward
    assert "residual" in error
