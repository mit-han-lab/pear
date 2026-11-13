import copy
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import pytest

from config_files import ConfigFiles
from control_rig import ControlRig, Robots, StandInClock, Ticker
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.stability import StabilityError, StabilityWindow
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import SteppedStandInYAMRobot
from yam_rigs import ControlHelpers, MotionHelpers, ReplayRobot

FED_POSITION = 0.1  # where every coordinate of a snapshot fed to a window rests, before its offset
SO101_REST = (0.0, -1.0, 1.0, 0.5, 0.0, 0.5)  # the SO-101's arm at rest, its jaw half open
PAUSE_RIG_SETTLE_S = 0.6  # how long the YAM robot of the pause tests holds before a test begins
RECORDED_STATIC_TRACKING_ERROR = "recorded_static_tracking_error.json"  # a recorded hold with a static error


class StabilityWindows:
    """Stability windows of each robot's configuration on the stand-in clock, and the feedback a test adds to
    them."""

    @staticmethod
    def window(
        robot_name: str, clock: StandInClock, *, is_strict: bool = False, checks_velocity_rms: bool = False
    ) -> tuple[StabilityWindow, StandInClock, JointLayout, ControlConfig, RobotConfig]:
        """Returns a stability window of the robot ``robot_name`` with the clock at its first reading, the clock, the
        robot's layout and its two configurations."""
        control_config, robot_config = Robots.load(robot_name)
        clock.now = StandInClock.START_S
        window = StabilityWindow(
            control_config,
            robot_config,
            duration_s=control_config.settle_window_s,
            is_strict=is_strict,
            checks_velocity_rms=checks_velocity_rms,
        )

        return window, clock, robot_config.LAYOUT, control_config, robot_config

    @classmethod
    def feed(
        cls,
        window: StabilityWindow,
        clock: StandInClock,
        layout: JointLayout,
        control_config: ControlConfig,
        *,
        velocity: float | list[float] = 0.0,
        drift: float = 0.0,
    ) -> list[bool]:
        """Adds two windows of samples, one each control period, and returns each sample's stability."""
        stable = []
        count = round(2 * control_config.settle_window_s * control_config.rate_hz)

        for sample in range(count):
            clock.now += 1 / control_config.rate_hz
            value = velocity[sample] if isinstance(velocity, list) else velocity
            speeds = (value,) * len(layout.coordinates)
            offset = drift * sample / count
            stable.append(window.add(cls.snapshot(layout, clock.now, sample_id=sample, velocity=speeds, offset=offset)))

        return stable

    @staticmethod
    def snapshot(
        layout: JointLayout,
        sampled_at: float,
        *,
        sample_id: int | None = None,
        velocity: tuple[float, ...] | None = None,
        offset: float = 0.0,
    ) -> dict[str, Any]:
        """Returns the feedback of every arm resting at FED_POSITION plus ``offset``, sampled at ``sampled_at``."""
        width = len(layout.coordinates)
        feedback = {
            "position": tuple(FED_POSITION + offset for _ in range(width)),
            "velocity": velocity or (0.0,) * width,
            "sampled_monotonic_s": sampled_at,
            "sample_id": sample_id,
        }

        return {"arms": {arm: dict(feedback) for arm in layout.arms}}


class SO101Feedback:
    """Snapshots of the SO-101's arm resting at SO101_REST, fed to a stability window on the stand-in clock."""

    @classmethod
    def feed(
        cls, window: StabilityWindow, clock: StandInClock, control_config: ControlConfig, *, velocity: float
    ) -> list[bool]:
        """Adds one control period of samples beyond the settle window, and returns each sample's stability."""
        stable = []

        for sample_id in range(round(2 * control_config.settle_window_s * control_config.rate_hz)):
            clock.now += 1 / control_config.rate_hz
            velocities = (velocity,) * len(SO101_REST)
            stable.append(window.add(cls.snapshot(sampled_at=clock.now, sample_id=sample_id, velocity=velocities)))

        return stable

    @staticmethod
    def snapshot(
        *,
        sampled_at: float = StandInClock.START_S,
        sample_id: int | None = None,
        velocity: tuple[float, ...] = (0.0,) * len(SO101_REST),
        target: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Returns a control snapshot of the arm resting at SO101_REST."""
        feedback = {
            "position": SO101_REST,
            "velocity": velocity,
            "sampled_monotonic_s": sampled_at,
            "sample_id": sample_id,
            "age_seconds": StandInClock.START_S - sampled_at,
        }

        return {"arms": {"arm": feedback}, "targets": {"arm": SO101_REST if target is None else tuple(target)}}


class IdleSnapshots:
    """Control snapshots of both YAM arms holding still at zero, written by hand."""

    @staticmethod
    def idle_snapshot() -> dict[str, Any]:
        """Returns a HOLD snapshot of both arms at rest at zero, fresh."""
        return {
            "status": "HOLD",
            "fault": None,
            "generation": 0,
            "targets": {side: [0.0] * 7 for side in ("left", "right")},
            "arms": {
                side: {"position": [0.0] * 7, "velocity": [0.0] * 7, "age_seconds": 0.0} for side in ("left", "right")
            },
        }


class YAMStationRig:
    """What the tests on both YAM arms of their station of the tests share: the end of an operation."""

    MAX_STEPS = 2000  # the most control periods an operation may take

    @classmethod
    def finish(cls, runtime: JointControlLoop, tick: Callable[..., None]) -> dict[str, Any]:
        """Steps until the operation ends in a HOLD or a FAULT, and returns the snapshot then."""
        for _ in range(cls.MAX_STEPS):
            tick()
            if runtime.status in ("HOLD", "FAULT"):
                return runtime.snapshot()

        pytest.fail("controller did not terminate")


@pytest.fixture
def yam_station_rig(
    stand_in_clock: StandInClock,
) -> Iterator[tuple[JointControlLoop, JointDriver, StandInClock, Callable[..., None]]]:
    """Both YAM arms of their station of the tests on stand-in joints, stepped by hand: the control loop, its driver,
    the clock and a step of one control period."""
    with ControlRig(YAMControlLoop, StandInYAMDriver, *ConfigFiles.load_configs("yam"), stand_in_clock) as rig:
        runtime, device, ticker = rig
        yield runtime, device, stand_in_clock, ticker.tick


@pytest.fixture
def pause_rig(
    stand_in_clock: StandInClock,
) -> Iterator[tuple[BaseRobot, JointControlLoop, JointDriver, StandInClock, Callable[[float], None]]]:
    """A YAM robot of its station of the tests in joint mode, its control loop stepped by hand, holding for
    PAUSE_RIG_SETTLE_S: the robot, its control loop, the driver, the clock and what advances them; the control loop is
    closed at the end."""
    robot = SteppedStandInYAMRobot(ConfigFiles.load("yam"), action_mode=ActionMode.JOINT).open()
    runtime = robot.control_loop
    ticker = Ticker(runtime, stand_in_clock)

    def advance(seconds: float) -> None:
        for _ in range(round(seconds / runtime.control_config.control_period_s)):
            ticker.tick()
            assert runtime.status != "FAULT", runtime.fault

    try:
        advance(PAUSE_RIG_SETTLE_S)
        yield robot, runtime, runtime.driver, stand_in_clock, advance
    finally:
        runtime.close()


@pytest.fixture
def replay(stand_in_clock: StandInClock) -> tuple[ReplayRobot, StandInClock, dict[str, Any]]:
    """A YAM robot's waits over a recorded hold with a static tracking error, the clock at zero, and the record."""
    control_config, robot_config = ConfigFiles.load_configs("yam")
    recorded = json.loads((ConfigFiles.YAM_RECORDED_CASES / RECORDED_STATIC_TRACKING_ERROR).read_text())
    stand_in_clock.now = 0.0
    station = ReplayRobot(control_config, robot_config, lambda: copy.deepcopy(recorded))

    return station, stand_in_clock, recorded


def test_still_feedback_settles_after_a_full_window(robot_name, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)

    stable = StabilityWindows.feed(window, clock, layout, control_config)

    assert stable[-1] and not stable[0]
    assert window.metrics["violations"] == []


@pytest.mark.parametrize(
    ("robot_name", "axis"), [("yam", {"side": "left", "joint": "J1"}), ("so101", {"joint": "shoulder_pan"})]
)
def test_mean_velocity_beyond_tolerance_is_unstable_and_named(robot_name, axis, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)

    stable = StabilityWindows.feed(window, clock, layout, control_config, velocity=2 * robot_config.velocity_tolerance)

    assert not any(stable)
    violation = window.metrics["violations"][0]
    assert {key: violation[key] for key in axis} == axis
    assert (violation["metric"], violation["units"]) == ("velocity_mean", "rad/s")


def test_a_strict_window_bounds_position_span_and_optionally_velocity_rms(robot_name, stand_in_clock):
    for checks_velocity_rms in (False, True):
        window, clock, layout, control_config, robot_config = StabilityWindows.window(
            robot_name, stand_in_clock, is_strict=True, checks_velocity_rms=checks_velocity_rms
        )
        # alternating velocity with a zero mean: only RMS sees it; the span moves too
        alternating = [(-1) ** sample * 2 * robot_config.velocity_tolerance for sample in range(1000)]

        StabilityWindows.feed(
            window, clock, layout, control_config, velocity=alternating, drift=4 * robot_config.position_tolerance
        )

        metrics = {violation["metric"] for violation in window.metrics["violations"]}
        assert metrics == ({"velocity_rms", "position_span"} if checks_velocity_rms else {"position_span"})


def test_stale_feedback_is_rejected_and_resets_the_window(robot_name, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)
    StabilityWindows.feed(window, clock, layout, control_config)

    with pytest.raises(RuntimeError, match="Stale/invalid stability feedback"):
        window.add(StabilityWindows.snapshot(layout, clock.now - 2 * control_config.state_timeout_s, sample_id=-1))

    assert window.metrics == {}


def test_a_sampling_gap_restarts_the_window(robot_name, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)
    StabilityWindows.feed(window, clock, layout, control_config)

    clock.now += 2 * control_config.state_timeout_s
    assert not window.add(StabilityWindows.snapshot(layout, clock.now, sample_id=10_000))


def test_target_accuracy_over_the_window_is_judged_per_arm(robot_name, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)
    StabilityWindows.feed(window, clock, layout, control_config)
    position = StabilityWindows.snapshot(layout, clock.now)["arms"][layout.arms[0]]["position"]
    missed = (position[0] + 0.1, *position[1:])

    metrics = window.evaluate(arms=layout.arms[:1], targets={layout.arms[0]: missed}, position_tolerance=0.05)

    assert not metrics["stable"]
    assert list(metrics["target_error_max"]) == [layout.arms[0]]
    assert metrics["violations"][0]["metric"] == "target_error_max"


@pytest.mark.parametrize(
    ("robot_name", "message"),
    [
        ("yam", "Unstable: left J1 velocity_mean=0.120000 > 0.060000 rad/s (window=0.480s, samples=25)"),
        ("so101", "Unstable: shoulder_pan velocity_mean=0.120000 > 0.060000 rad/s (window=0.480s, samples=25)"),
    ],
)
def test_a_failure_names_joint_metric_and_window_and_survives_a_snapshot(robot_name, message, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)
    robot_config.velocity_tolerance = 0.06
    StabilityWindows.feed(window, clock, layout, control_config, velocity=0.12)

    error = window.stability_error("Unstable")
    snapshot = {"status": "FAULT", "fault": str(error), "fault_details": error.details}

    assert str(error).startswith(message[: message.index(" (")])
    with pytest.raises(StabilityError) as rebuilt:
        SnapshotCheck.raise_for_fault(snapshot)
    assert str(rebuilt.value) == str(error)


def test_persistent_motion_during_hold_faults_after_the_settle_timeout(robot_name, stand_in_clock):
    window, clock, layout, control_config, robot_config = StabilityWindows.window(robot_name, stand_in_clock)

    with pytest.raises(StabilityError, match="Persistent motion during HOLD"):
        for sample in range(round(2 * control_config.settle_timeout_s * control_config.rate_hz)):
            clock.now += 1 / control_config.rate_hz
            velocity = (2 * robot_config.velocity_tolerance,) * len(layout.coordinates)
            window.check_hold(StabilityWindows.snapshot(layout, clock.now, sample_id=sample, velocity=velocity))


def test_still_feedback_becomes_stable(so101_config, stand_in_clock):
    control_config, config = ConfigFiles.make_configs(so101_config)
    clock = stand_in_clock
    window = StabilityWindow(
        control_config, config, duration_s=control_config.settle_window_s, is_strict=False, checks_velocity_rms=False
    )

    stable = SO101Feedback.feed(window, clock, control_config, velocity=0.0)

    assert stable[-1] and not stable[0]
    assert window.metrics["violations"] == []


def test_velocity_beyond_tolerance_is_unstable(so101_config, stand_in_clock):
    control_config, config = ConfigFiles.make_configs(so101_config)
    clock = stand_in_clock
    window = StabilityWindow(
        control_config, config, duration_s=control_config.settle_window_s, is_strict=False, checks_velocity_rms=False
    )

    stable = SO101Feedback.feed(window, clock, control_config, velocity=2 * config.velocity_tolerance)

    assert not any(stable)
    assert {v["metric"] for v in window.metrics["violations"]} == {"velocity_mean"}
    assert window.metrics["violations"][0]["joint"] == "shoulder_pan"


def test_stale_feedback_is_rejected(so101_config, stand_in_clock):
    control_config, config = ConfigFiles.make_configs(so101_config)
    clock = stand_in_clock
    window = StabilityWindow(
        control_config, config, duration_s=control_config.settle_window_s, is_strict=False, checks_velocity_rms=False
    )
    stale = SO101Feedback.snapshot(sampled_at=clock.now - 2 * control_config.state_timeout_s)

    with pytest.raises(RuntimeError, match="Stale/invalid stability feedback"):
        window.add(stale)
    with pytest.raises(RuntimeError, match="Stale station feedback"):
        SnapshotCheck.validate_snapshot({**stale, "status": "HOLD", "fault": None}, control_config)


@pytest.mark.parametrize("mode", ["spikes", "drift", "oscillation", "sustained_velocity"])
def test_settling_uses_velocity_without_position_change_limits(yam_station_rig, monkeypatch, mode):
    runtime, hardware, clock, tick = yam_station_rig
    start_s = clock.now  # the scenario's start: the control loop waited for settled feedback when it was made
    read = hardware.read
    count = [0]

    def feedback():
        states = read()
        count[0] += 1
        q, v = list(states["left"].position), [0.0] * 7
        if mode == "spikes":
            v[3] = 0.3 * (-1) ** count[0]
        elif mode == "drift":
            q[3] = (clock.now - start_s) * 0.008
        elif mode == "oscillation":
            q[3] = 0.003 * (-1) ** count[0]
        else:
            v[3] = 0.08
        states["left"] = replace(states["left"], position=tuple(q), velocity=tuple(v))
        return states

    monkeypatch.setattr(hardware, "read", feedback)
    ControlHelpers.jog(runtime, "right", 1, 0.01)
    result = YAMStationRig.finish(runtime, tick)
    assert result["status"] == ("FAULT" if mode == "sustained_velocity" else "HOLD")
    if mode in ("drift", "oscillation"):
        assert result["settling"]["position_span"][3] > 0.003
        assert result["motion"]["end_reason"] == "finished"
    if mode == "spikes":
        assert result["settling"]["velocity_rms"][3] > runtime.robot_config.velocity_tolerance


def test_sampling_gap_cannot_be_counted_as_stationary_time(yam_station_rig):
    runtime, _, clock, tick = yam_station_rig
    window = StabilityWindow(
        runtime.control_config,
        runtime.robot_config,
        duration_s=runtime.control_config.settle_window_s,
        is_strict=False,
        checks_velocity_rms=True,
    )
    for _ in range(20):
        tick()
        window.add(runtime.snapshot())
    clock.now += 0.3
    sample = runtime.snapshot()
    for arm in sample["arms"].values():
        arm["sampled_monotonic_s"] = clock.now
    assert not window.add(sample)
    assert window.metrics["samples"] == 1


def test_duplicate_batches_cannot_create_a_stability_window_and_gap_resets(yam_station_rig):
    runtime, hardware, clock, _ = yam_station_rig
    window = StabilityWindow(
        runtime.control_config,
        runtime.robot_config,
        duration_s=runtime.control_config.settle_window_s,
        is_strict=False,
        checks_velocity_rms=True,
    )

    def sample(batch, stamp):
        snap = runtime.snapshot()
        for arm in snap["arms"].values():
            arm["sample_id"] = batch
            arm["sampled_monotonic_s"] = stamp
        return snap

    for batch in range(10):
        clock.now += 0.06
        for _ in range(20):
            assert not window.add(sample(batch, clock.now))
    assert window.metrics["samples"] < 20
    for batch in range(10, 40):
        clock.now += 0.02
        stable = window.add(sample(batch, clock.now))
    assert stable
    clock.now += 0.2
    assert not window.add(sample(40, clock.now))
    assert window.metrics["samples"] == 1


@pytest.mark.parametrize("moving", ["left", "right"])
def test_one_repeated_batch_cannot_hide_other_arms_new_motion(pause_rig, monkeypatch, moving):
    station, runtime, hardware, clock, advance = pause_rig
    dt = runtime.control_config.control_period_s
    sequence = {s: 0 for s in ("left", "right")}
    stamp = {s: clock.now for s in sequence}
    velocity = {s: (0.0,) * 7 for s in sequence}
    read = hardware.read

    def batches():
        return {
            s: replace(state, sample_id=sequence[s], sampled_monotonic_s=stamp[s], velocity=velocity[s])
            for s, state in read().items()
        }

    monkeypatch.setattr(hardware, "read", batches)
    for _ in range(30):
        for side in sequence:
            sequence[side] += 1
            stamp[side] = clock.now + dt
        advance(dt)
    assert station.status()["status"] == "READY"
    velocity[moving] = (2.0,) + (0.0,) * 6
    sequence[moving] += 1
    stamp[moving] = clock.now + dt
    advance(dt)
    assert station.status()["status"] == "BUSY"
    assert any(
        v["side"] == moving and v["metric"] == "velocity_mean" for v in runtime.snapshot()["settling"]["violations"]
    )
    # the high-velocity sample remains in the window until enough fresh samples arrive
    velocity[moving] = (0.0,) * 7
    for tick in range(30):
        for side in sequence:
            sequence[side] += 1
            stamp[side] = clock.now + dt
        advance(dt)
        if tick < 24:
            assert not runtime.snapshot()["arrival"]["settled"]
    assert station.status()["status"] == "READY"


def test_position_excursion_does_not_restart_the_velocity_window(replay):
    station, clock, recorded = replay

    def status():
        result = copy.deepcopy(recorded)
        for side in ("left", "right"):
            result["arms"][side]["velocity"] = [0.0] * 7
            result["arms"][side]["position"] = result["targets"][side][:]
        if 0.3 <= clock.now < 0.4:
            result["arms"]["right"]["position"][2] -= 0.021
        return result

    station.control_loop.read_snapshot = status
    station.wait_stationary()
    result = station.control_loop.snapshot()
    assert 0.5 <= clock.now <= 0.54
    assert result["settling"]["position_span"][9] == pytest.approx(0.021)
    assert result["arrival"]["settled"] and result["arrival"]["reached"]


def test_stability_failure_identifies_joint_metric_units_and_original_window(stand_in_clock):
    control_config, config = ConfigFiles.load_configs("yam")
    clock = stand_in_clock
    monitor = StabilityWindow(
        control_config, config, duration_s=control_config.settle_window_s, is_strict=False, checks_velocity_rms=True
    )
    for tick in range(40):
        state = IdleSnapshots.idle_snapshot()
        state["arms"]["right"]["position"][2] = 0.004 * (-1) ** tick
        state["arms"]["right"]["velocity"][2] = 0.08
        state["arms"]["left"]["velocity"][6] = 0.1
        clock.now += 1 / control_config.rate_hz
        assert not monitor.add(MotionHelpers.stamp(state, clock.now))
    error = monitor.stability_error("Station lost stability")
    assert "right J3 velocity_mean=0.080000 > 0.060000 rad/s" in str(error)
    assert error.metrics["position_span"][9] == pytest.approx(0.008)
    assert all(v["metric"] == "velocity_mean" for v in error.metrics["violations"])
    assert "left gripper velocity_mean=0.100000 > 0.060000 opening/s" in str(error)
    original = copy.deepcopy(error.metrics)
    assert original["samples"] >= 25
    for _ in range(40):
        clock.now += 1 / control_config.rate_hz
        settled = monitor.add(MotionHelpers.stamp(IdleSnapshots.idle_snapshot(), clock.now))
    assert settled
    assert error.metrics == original
    json.dumps(error.metrics)
