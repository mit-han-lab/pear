import copy
import dataclasses
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import ControlRig, Robots, StandInClock, Ticker
from pear.deploy.station.control.joint.config import RobotConfig
from pear.deploy.station.control.joint.trajectory import JointMove
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.joint.grippers import YAMGripperOperation
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from stand_in_joint_driver import StandInYAMDriver
from yam_rigs import ControlHelpers, GripperRig, GripperRigs


class StandInEffortDriver(StandInYAMDriver):
    """YAM's stand-in joints whose readings carry an effort on every motor."""

    def read(self) -> dict[str, YAMArmState]:
        """Returns every arm where its last command put it, with the effort of each motor."""
        return {
            s: YAMArmState(x.position, x.velocity, x.sampled_monotonic_s, effort=(3.0,) * 7)
            for s, x in super().read().items()
        }


class JointMoves:
    """The joint move the tests of a move's dispatch run: the right arm's first two joints, from rest."""

    @staticmethod
    def fixture_move(config: RobotConfig, speed_scale: float = 1 / 3) -> JointMove:
        """Returns the move of the right arm's J1 by 0.15 rad and J2 by 0.06 rad at ``speed_scale``, grippers half
        open."""
        start = np.zeros(14)
        start[[6, 13]] = 0.5
        goal = start.copy()
        goal[7:9] = [0.15, 0.06]

        return JointMove.build(start, goal, config.joint_limits(), LAYOUT, speed_scale=speed_scale)


class ContactCandidate:
    """How a test steps a closing gripper until the SDK's force limit holds it on its object."""

    CANDIDATE_STEPS = 400  # control periods a closing gripper may take to meet an object

    @classmethod
    def reach_candidate(cls, rig: GripperRig, op: YAMGripperOperation, side: str = "right") -> float:
        """Steps ``rig`` until the gripper of ``side`` has a contact to confirm, and returns the clock's reading
        then."""
        for _ in range(cls.CANDIDATE_STEPS):
            rig.tick()
            if side in op.contacts:
                assert side not in op.outcomes
                return rig.clock.now
            assert rig.control_loop.fault is None

        pytest.fail("No contact candidate")


@pytest.fixture
def rig(gripper_rigs: GripperRigs) -> GripperRig:
    """A gripper rig whose right gripper meets an object at the opening 0.8."""
    return gripper_rigs.make(contacts={"right": 0.8})


@pytest.fixture
def control_loop_rig(stand_in_clock: StandInClock):
    """A YAM control loop of the tests' station on stand-in joints, stepped by hand: the loop, its driver, the
    stand-in clock and the step of one control period; the loop is closed at the end."""
    with ControlRig(YAMControlLoop, StandInYAMDriver, *ConfigFiles.load_configs("yam"), stand_in_clock) as (
        control_loop,
        driver,
        ticker,
    ):
        yield control_loop, driver, stand_in_clock, ticker.tick


@pytest.mark.parametrize("enabled", [(False, False), (True, False), (False, True)])
def test_velocity_feedforward_switch_is_per_arm(stand_in_clock, enabled):
    control_config, config = Robots.load("yam")
    for side, on in zip(("left", "right"), enabled):
        config.arms[side].uses_velocity_feedforward = on
    hardware = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(hardware, control_config, config)
    ticker = Ticker(runtime, stand_in_clock)

    try:
        targets = {s: list(q) for s, q in runtime.targets.items()}
        targets["left"][1] = targets["right"][1] = 0.3
        runtime.move(targets)
        sent = []
        for _ in range(1000):
            ticker.tick()
            sent.append(hardware.velocities)
            if runtime.status != "RUNNING":
                break

        assert runtime.status == "HOLD", runtime.fault
        if not any(enabled):
            # Position-only: no velocity is sent.
            assert all(v is None for v in sent)
        else:
            for side, on in zip(("left", "right"), enabled):
                peak = max(v[side][1] for v in sent)
                assert peak > 0.1 if on else all(v[side] == (0.0,) * 7 for v in sent)
    finally:
        runtime.close()


def test_runtime_retains_effort_without_affecting_target(stand_in_clock):
    control_config, config = Robots.load("yam")
    hardware = StandInEffortDriver(control_config, config)
    runtime = YAMControlLoop(hardware, control_config, config)

    try:
        snapshot = runtime.snapshot()
        assert snapshot["arms"]["left"]["effort"][2] == 3.0
        assert snapshot["targets"] == hardware.positions
        json.dumps(snapshot)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "lag,tolerance,expected",
    [(0.026, 0.02, "HOLD"), (0.026, 0.03, "HOLD"), (0.035, 0.03, "HOLD"), (0.16, 0.03, "FAULT")],
)
def test_a_tracking_lag_is_reported_against_the_arrival_tolerance_and_faults_beyond_the_tracking_bound(
    stand_in_clock, lag, tolerance, expected
):
    _, baseline = ConfigFiles.load_configs("yam")
    control_config, config = ConfigFiles.load_configs("yam", f"robot.position_tolerance={tolerance}")
    assert baseline.position_tolerance == 0.02
    assert dataclasses.replace(config, position_tolerance=baseline.position_tolerance) == baseline
    device = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(device, control_config, config)
    send = device.send
    commands = []

    def lagged_send(targets, velocities=None):
        commands.append(dict(targets))
        values = {s: list(q) for s, q in targets.items()}
        values["right"][4] -= lag
        send(values, velocities)

    device.send = lagged_send
    try:
        goal = {s: list(q) for s, q in runtime.targets.items()}
        goal["right"][4] = 0.06
        runtime.move(goal, speed_scale=1 / 3)
        for _ in range(1000):
            stand_in_clock.now += runtime.control_config.control_period_s
            runtime.step()
            if runtime.status != "RUNNING":
                break

        assert runtime.status == expected
        if expected == "HOLD":
            assert runtime.snapshot()["arrival"]["reached"] is (lag <= tolerance)
        if expected == "HOLD":
            status = BaseRobot.hold_snapshot(SimpleNamespace(control_loop=runtime, _control_config=control_config))
            assert status["targets"]["right"][4] == pytest.approx(0.06)
            assert status["arms"]["right"]["position"][4] == pytest.approx(0.06 - lag)
        else:
            assert runtime.fault
            assert commands[-1] == runtime.targets
    finally:
        runtime.close()


def test_changed_start_cannot_dispatch(stand_in_clock):
    control_config, config = ConfigFiles.load_configs("yam")
    device = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(device, control_config, config)

    try:
        before = len(device.sent_targets)
        move = JointMoves.fixture_move(runtime.robot_config)
        runtime.trajectory.input.current_position = [0.1] * 14
        with pytest.raises(ValueError, match="start"):
            runtime.dispatch_joint_move(move)
        assert len(device.sent_targets) == before and runtime.status == "HOLD"
    finally:
        runtime.close()


@pytest.mark.parametrize("event", ["finish", "stop", "stale", "tracking", "drift", "delayed_tick", "right_bus"])
@pytest.mark.parametrize("speed_scale", [1 / 3, 1.0])
def test_joint_move_runtime_keeps_fault_checks_and_controlled_stop(stand_in_clock, event, speed_scale):
    control_config, config = ConfigFiles.load_configs("yam")
    device = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(device, control_config, config)
    move = JointMoves.fixture_move(config, speed_scale)

    try:
        runtime.dispatch_joint_move(move)

        def tick():
            stand_in_clock.now += runtime.control_config.control_period_s
            runtime.step()

        for _ in range(int(move.duration_s / runtime.control_config.control_period_s / 3)):
            tick()
            assert runtime.status == "RUNNING", runtime.fault
        if event == "stop":
            assert runtime.trajectory.input.current_velocity[7] > 0
            runtime.stop()
            assert runtime.operation is None
        elif event == "stale":
            device.is_stale = True
        elif event == "right_bus":
            device.failed_arm = "right"
        elif event in ("tracking", "drift"):
            values = list(device.positions["left"])
            values[0] += 0.2 if event == "tracking" else 0.006
            device.positions["left"] = tuple(values)
        elif event == "delayed_tick":
            stand_in_clock.now += 0.2
        for _ in range(2000):
            tick()
            if runtime.status in ("HOLD", "FAULT"):
                break

        if event in ("finish", "stop", "drift", "delayed_tick"):
            assert runtime.status == "HOLD", runtime.fault
            assert np.max(np.abs(runtime.trajectory.input.current_velocity)) < 1e-8
            if event in ("finish", "drift", "delayed_tick"):
                assert runtime.trajectory.input.current_position == pytest.approx(move.goal)
                assert runtime.snapshot()["motion"]["end_reason"] == "finished"
            else:
                # A time-optimal native move may already require all remaining
                # distance to brake within its jerk bound. Reaching the endpoint
                # during that stop must still be reported as interruption.
                assert 0 < runtime.trajectory.input.current_position[7] <= move.goal[7] + 1e-12
                assert runtime.snapshot()["motion"]["end_reason"] == "stopped"
        else:
            assert runtime.status == "FAULT" and runtime.operation is None
            assert device.sent_targets[-1] == runtime.targets
    finally:
        runtime.close()


@pytest.mark.parametrize("prepared_move", [False, True])
def test_a_capped_move_and_its_stop_keep_the_caps_and_the_next_move_uses_the_station_limits(
    stand_in_clock, prepared_move
):
    control_config, config = ConfigFiles.load_configs("yam")
    hardware = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(hardware, control_config, config)
    initial = runtime.trajectory.input.current_position
    end = np.array(initial)
    end[7] += 0.3
    caps = np.array(config.joint_limits().motion_caps(caps=config.home.limits))

    try:
        if prepared_move:
            profile = copy.deepcopy(config.home.limits)
            move = JointMove.build(initial, end, config.joint_limits(), LAYOUT, caps=profile)
            profile.velocity[0] = 0.9  # The prepared move owns its copy.
            runtime.dispatch_joint_move(move)
            assert np.asarray(move.motion_caps) == pytest.approx(caps)
        else:
            runtime.move(LAYOUT.split(end), caps=config.home.limits)
        trajectory_input = runtime.trajectory.input
        assert np.array(
            [trajectory_input.max_velocity, trajectory_input.max_acceleration, trajectory_input.max_jerk]
        ) == pytest.approx(caps)
        samples = []

        def tick():
            stand_in_clock.now += runtime.control_config.control_period_s
            runtime.step()
            samples.append([runtime.trajectory.input.current_velocity, runtime.trajectory.input.current_acceleration])

        for _ in range(30):
            tick()
        assert runtime.trajectory.input.current_velocity[7] > 0.01
        runtime.stop()
        for _ in range(500):
            tick()
            if runtime.status == "HOLD":
                break
        assert runtime.status == "HOLD", runtime.fault
        samples = np.array(samples)
        assert np.all(abs(samples[:, 0]) <= caps[0] + 1e-9)
        assert np.all(abs(samples[:, 1]) <= caps[1] + 1e-9)
        assert np.all(abs(np.diff(samples[:, 1], axis=0) / runtime.control_config.control_period_s) <= caps[2] + 1e-8)

        # Subsequent continuous policies use station limits, not the last slow move's caps.
        runtime.move(LAYOUT.split(end))
        assert runtime.trajectory.input.max_velocity[7] == 1
        assert runtime.trajectory.input.max_acceleration[7] == 2
        assert runtime.trajectory.input.max_jerk[7] == 10
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["joint_move", "target", "sequence"])
def test_an_operation_reaches_the_velocity_cap_and_stops_within_the_acceleration_and_jerk_caps(stand_in_clock, mode):
    control_config, config = ConfigFiles.load_configs("yam")
    hardware = StandInYAMDriver(control_config, config)
    runtime = YAMControlLoop(hardware, control_config, config)
    end = np.array(runtime.trajectory.input.current_position)
    end[7] = 1.5

    try:
        if mode == "joint_move":
            move = JointMove.build(runtime.trajectory.input.current_position, end, config.joint_limits(), LAYOUT)
            runtime.dispatch_joint_move(move)
        elif mode == "target":
            runtime.move(LAYOUT.split(end))
        else:
            runtime.start_sequence(np.tile(end, (45, 1)), rate_hz=30)
        samples = []
        stopped = False
        for _ in range(500):
            stand_in_clock.now += runtime.control_config.control_period_s
            runtime.step()
            trajectory_input = runtime.trajectory.input
            samples.append(
                [
                    trajectory_input.current_position,
                    trajectory_input.current_velocity,
                    trajectory_input.current_acceleration,
                ]
            )
            if not stopped and trajectory_input.current_velocity[7] >= 0.998:
                runtime.stop()
                stopped = True
            if runtime.status == "HOLD":
                break

        assert stopped, "Policy never approached the requested 1 rad/s cap"
        assert runtime.status == "HOLD", runtime.fault
        assert 0 < runtime.trajectory.input.current_position[7] < end[7]
        samples = np.array(samples)
        control_period_s = runtime.control_config.control_period_s
        assert np.max(abs(samples[:, 1, 7])) <= 1 + 1e-9
        assert np.max(abs(samples[:, 2, 7])) <= 2 + 1e-9
        assert np.max(abs(np.diff(samples[:, 2, 7]) / control_period_s)) <= 10 + 1e-7
        assert np.max(abs(np.diff(samples[:, 0, 7]) / control_period_s)) <= 1 + 1e-9
        assert samples[-1, 1:, 7] == pytest.approx([0, 0], abs=1e-8)
    finally:
        runtime.close()


@pytest.mark.parametrize("phase", ["hold", "arm"])
def test_gripper_effort_alone_never_faults_a_hold_or_an_arm_move(gripper_rigs, phase):
    rig = gripper_rigs.make()
    if phase == "arm":
        q = {s: list(v) for s, v in rig.control_loop.targets.items()}
        q["right"][0] = 0.1
        rig.control_loop.move(q)
    read = rig.driver.read

    def feedback():
        return {s: replace(v, effort=(0.0,) * 6 + (1.03 if s == "left" else 0.0,)) for s, v in read().items()}

    rig.driver.read = feedback
    rig.clock.now += 0.02
    rig.control_loop.step()

    assert rig.control_loop.fault is None
    assert rig.control_loop.states["left"].effort[6] == 1.03


@pytest.mark.parametrize(
    "field,value",
    [
        ("effort", None),
        ("sdk_gripper_target_effort_nm", None),
        ("sdk_gripper_target_effort_nm", 0.0),
        ("sdk_gripper_target_effort_nm", float("nan")),
        ("sdk_gripper_target", float("nan")),
        ("sdk_gripper_target", 1.1),
    ],
)
def test_invalid_contact_telemetry_faults_before_new_targets(rig, field, value):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    read = rig.driver.read

    def feedback():
        values = read()
        values["right"] = replace(values["right"], **{field: value})
        return values

    rig.driver.read = feedback
    targets = dict(rig.control_loop.targets)
    rig.tick()

    assert rig.control_loop.status == "FAULT"
    assert "Invalid SDK gripper" in rig.control_loop.fault
    assert rig.control_loop.targets == targets
    assert not op.is_complete


@pytest.mark.parametrize("phase", ["hold", "arm", "gripper", "stop"])
def test_effort_is_checked_before_publishing_feedback_in_every_phase(control_loop_rig, monkeypatch, phase):
    runtime, hardware, _, tick = control_loop_rig
    read = hardware.read
    corrupt = [False]

    def feedback():
        return {
            side: replace(
                state,
                effort=(float("nan"),) * 7 if corrupt[0] and side == "left" else (0.0,) * 7,
            )
            for side, state in read().items()
        }

    monkeypatch.setattr(hardware, "read", feedback)
    tick(30)
    if phase in ("arm", "stop"):
        ControlHelpers.jog(runtime, "right", 1, 0.3)
        tick(5)
        if phase == "stop":
            runtime.stop()
    elif phase == "gripper":
        runtime.begin_grippers({"right": True})
        tick(5)
    before = len(hardware.sent_targets)
    targets = dict(runtime.targets)

    corrupt[0] = True
    tick()

    assert runtime.status == "FAULT"
    assert len(hardware.sent_targets) == before + 1 and hardware.sent_targets[-1] == targets
    assert runtime.states["left"].effort == (0.0,) * 7
