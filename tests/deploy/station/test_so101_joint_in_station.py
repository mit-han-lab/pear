import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.hardware.arm_state import ServoFault
from pear.deploy.station.control.robots.so101.joint.control_loop import SO101ControlLoop
from pear.deploy.station.control.robots.so101.joint.feedforward import GravityModel, SO101FrictionFeedforward
from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX
from pear.deploy.station.control.schemas import ControlSnapshot
from stand_in_joint_driver import StandInSO101Arm, StandInSO101Driver
from waiting import Waiting

ZERO_HOME_OVERRIDE = "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]"  # Home: every joint at zero, the gripper open
# what the tests of the arm change of the station of the tests besides its quick settings and its Home: the
# tolerances they were written for (Home's, an arrival's, the feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)
COMPLIANCE = [0.0, 0.08, 0.10, 0.05, 0.0]  # rad per N*m, a typical task fit
FRICTION = [0.006, 0.016, 0.016, 0.008, 0.006]  # rad
WAIT_TIMEOUT_S = 2.0  # default wait for a condition on the stand-in arm
WAIT_POLL_S = 0.01  # between two looks at a condition on the stand-in arm
SETTLE_TIMEOUT_S = 3  # for a move to reach its hold
FADE_MARGIN_S = 0.1  # past the friction fade, so the offset is gone
ADVANCE_PERIOD_S = 0.02  # the 50 Hz control period
CLOSE = {"type": "gripper", "command": "close"}  # the action that closes the gripper
OPEN = {"type": "gripper", "command": "open"}  # the action that opens the gripper
TARGET = (0.1, -0.2, 0.3, -0.4, 0.5, 0.6)  # a target the stand-in arm is sent


class StandInFeedforwardDriver(StandInSO101Driver):
    """The SO-101's own driver over the stand-in arm, with the configured feedforward."""

    USES_FEEDFORWARD = True


class FeedforwardRig:
    """The SO-101's control loop on a stand-in arm that sags and sticks as the real servos do, and the steps of the
    tests of its feedforward."""

    @staticmethod
    def with_feedforward(
        config: DictConfig,
        compliance: Sequence[float] = COMPLIANCE,
        friction: Sequence[float] = FRICTION,
        load_ratio: Sequence[float] | None = None,
    ) -> tuple[ControlConfig, SO101Config]:
        """Returns the control loop's configuration and the SO-101's settings of ``config`` with the feedforward
        section set from these fits."""
        feedforward = {
            "gravity_compliance": list(compliance),
            "friction_offset_rad": list(friction),
            "friction_load_ratio": list(load_ratio or [0.0] * 5),
        }

        return ConfigFiles.make_configs(OmegaConf.merge(config, {"robot": {"feedforward": feedforward}}))

    @staticmethod
    def servo(
        config: SO101Config, compliance: Sequence[float] = COMPLIANCE, friction: Sequence[float] = FRICTION
    ) -> Callable[[tuple[float, ...], tuple[float, ...]], np.ndarray]:
        """Returns a P-only position servo: it rests `compliance x gravity` below its goal, anywhere within
        +-`friction` of that, and moves only when pushed out of that band."""
        gravity = GravityModel(config.geometry.urdf)
        c, f = np.array(compliance), np.array(friction)

        def response(goal: tuple[float, ...], previous: tuple[float, ...]) -> np.ndarray:
            goal, previous = np.array(goal), np.array(previous)
            centre = goal[:GRIPPER_INDEX] - c * gravity.torque(previous)
            out = goal.copy()
            out[:GRIPPER_INDEX] = np.clip(previous[:GRIPPER_INDEX], centre - f, centre + f)
            return out

        return response

    @staticmethod
    def started(
        control_config: ControlConfig,
        config: SO101Config,
        response: Callable[[tuple[float, ...], tuple[float, ...]], Sequence[float]] | None,
    ) -> SO101ControlLoop:
        """Returns a started control loop over the stand-in arm, which answers with ``response``, and the configured
        feedforward."""
        driver = StandInFeedforwardDriver(control_config, config)
        driver.hardware.response = response
        runtime = SO101ControlLoop(driver, control_config, config)
        runtime.start()

        return runtime

    @staticmethod
    def move_and_hold(runtime: SO101ControlLoop, delta: np.ndarray) -> tuple[np.ndarray, ControlSnapshot]:
        """Moves the arm by ``delta`` from where it settled and waits out the friction fade; returns the goal and the
        snapshot then."""
        Waiting.until(lambda: runtime.snapshot()["arrival"]["settled"], WAIT_TIMEOUT_S, WAIT_POLL_S)
        goal = np.array(runtime.targets["arm"])
        goal[:GRIPPER_INDEX] += delta
        runtime.move({"arm": goal})
        runtime.wait_for_hold(SETTLE_TIMEOUT_S)
        time.sleep(SO101FrictionFeedforward.LATCH_S + SO101FrictionFeedforward.FADE_S + FADE_MARGIN_S)

        return goal, runtime.snapshot()

    @staticmethod
    def advance(
        friction: SO101FrictionFeedforward, velocity: Sequence[float], start: float, end: float, load: Sequence[float]
    ) -> np.ndarray:
        """Updates at the control rate from ``start`` through ``end`` and returns the last offset."""
        out = None
        for now in np.arange(start, end + 1e-9, ADVANCE_PERIOD_S):
            out = friction.update(velocity, now, load)

        return out


@pytest.fixture
def so101_config() -> DictConfig:
    """The SO-101's station of the tests with its quick settings, Home at zero, the tolerances of the tests of the arm
    and no feedforward; it takes the place of the conftest's in this file."""
    return ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES)


@pytest.fixture
def arm_configs(so101_config: DictConfig) -> tuple[ControlConfig, SO101Config]:
    """The control loop's configuration and the SO-101's settings of ``so101_config``."""
    return ConfigFiles.make_configs(so101_config)


@pytest.fixture
def runtime(arm_configs: tuple[ControlConfig, SO101Config]) -> Iterator[SO101ControlLoop]:
    """A started SO-101 control loop on the stand-in arm; closed at the end."""
    control_config, config = arm_configs
    runtime = SO101ControlLoop(StandInSO101Driver(control_config, config), control_config, config)
    runtime.start()

    yield runtime

    runtime.close()


def test_gravity_torque_is_what_the_joint_must_supply(arm_configs):
    _, config = arm_configs
    home = np.array([0.0, -0.814, 0.077, 1.423, -1.613])

    torque = GravityModel(config.geometry.urdf).torque(home)

    # Matches placo's generalized_gravity at the real Home.
    assert torque == pytest.approx([0.0, -0.0974, -0.3308, -0.0906, 0.0018], abs=1e-3)


def test_friction_ramps_in_latches_past_the_stop_and_fades():
    friction = SO101FrictionFeedforward([0.01, 0.02], [0.0, 0.1])
    load = np.array([0.0, 0.5])

    assert friction.update([0.2, -0.2], 0.0, load) == pytest.approx([0, 0])
    full = FeedforwardRig.advance(friction, [0.2, -0.2], 0.02, 0.2, load)
    assert full == pytest.approx([0.01, -(0.02 + 0.1 * 0.5)])
    # The plan stops at 0.2 s; the offset stays for LATCH_S, then fades over FADE_S.
    assert FeedforwardRig.advance(
        friction, [0, 0], 0.22, 0.2 + SO101FrictionFeedforward.LATCH_S, load
    ) == pytest.approx(full)
    half = FeedforwardRig.advance(
        friction,
        [0, 0],
        0.2 + SO101FrictionFeedforward.LATCH_S + 0.02,
        0.2 + SO101FrictionFeedforward.LATCH_S + SO101FrictionFeedforward.FADE_S / 2,
        load,
    )
    assert 0 < half[0] < full[0]
    end = 0.2 + SO101FrictionFeedforward.LATCH_S + SO101FrictionFeedforward.FADE_S + 0.1
    assert FeedforwardRig.advance(friction, [0, 0], end - 0.1, end, load) == pytest.approx([0, 0])


def test_friction_reverses_through_zero_and_ignores_malformed_velocity():
    friction = SO101FrictionFeedforward([0.01], [0.0])
    FeedforwardRig.advance(friction, [0.2], 0.0, 0.2, [0.0])

    # Reversing ramps from +1 to -1 at the engage rate: two ENGAGE_S in all.
    assert FeedforwardRig.advance(friction, [-0.2], 0.2 + 0.01, 0.2 + 0.01, [0.0]) == pytest.approx([0.008])
    assert FeedforwardRig.advance(
        friction, [-0.2], 0.22, 0.2 + 2 * SO101FrictionFeedforward.ENGAGE_S + 0.02, [0.0]
    ) == pytest.approx([-0.01])
    assert friction.update([np.nan], 0.4, [0.0]) == pytest.approx([-0.01])


@pytest.mark.parametrize("sign", [1, -1])
def test_a_sagging_sticky_arm_settles_on_its_target(so101_config, sign):
    control_config, config = FeedforwardRig.with_feedforward(so101_config)
    runtime = FeedforwardRig.started(control_config, config, FeedforwardRig.servo(config))

    try:
        delta = sign * np.array([0.1, 0.2, -0.2, 0.15, 0.1])
        goal, snapshot = FeedforwardRig.move_and_hold(runtime, delta)

        error = np.array(snapshot["arms"]["arm"]["position"]) - goal
        assert np.abs(error[:GRIPPER_INDEX]).max() < 1e-3
        assert snapshot["arrival"]["reached"]
        # After the fade only the gravity offset remains, and the gripper never has one.
        gravity = np.array(COMPLIANCE) * GravityModel(config.geometry.urdf).torque(goal)
        # HOLD stops writing once the fade changes the goal by under OFFSET_EPSILON.
        assert snapshot["motion"]["feedforward"] == pytest.approx(gravity.tolist(), abs=1e-5)
        assert runtime.driver.hardware.sent_targets[-1][GRIPPER_INDEX] == goal[GRIPPER_INDEX]
        assert snapshot["targets"]["arm"] == pytest.approx(goal)
    finally:
        runtime.close()


def test_without_feedforward_the_same_arm_stops_short(arm_configs):
    control_config, config = arm_configs
    runtime = FeedforwardRig.started(control_config, config, FeedforwardRig.servo(config))

    try:
        goal, snapshot = FeedforwardRig.move_and_hold(runtime, np.array([0.1, 0.2, -0.2, 0.15, 0.1]))

        error = np.array(snapshot["arms"]["arm"]["position"]) - goal
        assert abs(error[1]) > 0.01 and abs(error[2]) > 0.01
        assert snapshot["motion"]["feedforward"] == [0.0] * GRIPPER_INDEX
        assert runtime.driver.hardware.sent_targets[-1] == pytest.approx(tuple(goal))
    finally:
        runtime.close()


def test_hold_writes_stop_once_the_friction_offset_has_faded(so101_config):
    control_config, config = FeedforwardRig.with_feedforward(so101_config)
    runtime = FeedforwardRig.started(control_config, config, FeedforwardRig.servo(config))

    try:
        FeedforwardRig.move_and_hold(runtime, np.array([0.0, 0.2, 0.0, 0.0, 0.0]))
        count = len(runtime.driver.hardware.sent_targets)
        time.sleep(0.2)

        assert len(runtime.driver.hardware.sent_targets) == count
    finally:
        runtime.close()


def test_a_fault_holds_the_measured_pose_with_gravity_only(so101_config):
    control_config, config = FeedforwardRig.with_feedforward(so101_config)
    runtime = FeedforwardRig.started(control_config, config, FeedforwardRig.servo(config))

    try:
        FeedforwardRig.move_and_hold(runtime, np.array([0.0, 0.2, 0.0, 0.0, 0.0]))
        runtime.driver.hardware.failure = ServoFault("Servo error: elbow_flex: overload")
        Waiting.until(lambda: runtime.status == "FAULT", WAIT_TIMEOUT_S, WAIT_POLL_S)

        measured = np.array(runtime.states["arm"].position)
        gravity = np.array(COMPLIANCE) * GravityModel(config.geometry.urdf).torque(measured)
        sent = np.array(runtime.driver.hardware.sent_targets[-1])
        assert sent[:GRIPPER_INDEX] == pytest.approx(measured[:GRIPPER_INDEX] + gravity)
    finally:
        runtime.close()


def test_the_offset_never_carries_a_goal_past_a_limit(so101_config):
    control_config, config = FeedforwardRig.with_feedforward(so101_config, compliance=[0.0, 0.3, 0.3, 0.3, 0.0])
    runtime = FeedforwardRig.started(control_config, config, None)

    try:
        Waiting.until(lambda: runtime.snapshot()["arrival"]["settled"], WAIT_TIMEOUT_S, WAIT_POLL_S)
        goal = np.array(runtime.targets["arm"])
        goal[3] = config.limits.lower_bounds[3]  # wrist_flex gravity pushes its goal lower here
        runtime.move({"arm": goal})
        runtime.wait_for_hold(SETTLE_TIMEOUT_S)

        lower = np.array(config.limits.lower_bounds)
        assert all((np.array(q) >= lower - 1e-9).all() for q in runtime.driver.hardware.sent_targets)
    finally:
        runtime.close()


def test_the_stand_in_driver_runs_without_the_stations_feedforward(so101_config):
    control_config, config = FeedforwardRig.with_feedforward(so101_config)

    runtime = SO101ControlLoop.open(StandInSO101Driver(control_config, config), control_config, config)

    try:
        assert runtime.driver.feedforward is None
        assert runtime.driver.hardware.sent_targets[-1] == pytest.approx(runtime.targets["arm"])
    finally:
        runtime.close()


def test_the_gripper_reaches_both_endpoints(so101_robot):
    closed = so101_robot.execute(CLOSE)
    opened = so101_robot.execute(OPEN)

    assert (closed.status, closed.gripper_outcome, closed.target_reached) == ("completed", "closed", True)
    assert closed.measured_position[5] == pytest.approx(0.0)
    assert (opened.status, opened.gripper_outcome, opened.target_reached) == ("completed", "open", True)
    assert opened.measured_position[5] == pytest.approx(1.0)


def test_blocked_jaw_is_judged_where_it_stopped(so101_robot):
    so101_robot.control_loop.driver.hardware.blocked_gripper_opening = 0.4

    blocked = so101_robot.execute(CLOSE)

    assert (blocked.status, blocked.gripper_outcome, blocked.target_reached) == (
        "completed",
        "stopped_before_closed",
        False,
    )
    with so101_robot.control_loop.lock:
        assert so101_robot.control_loop.requested_position()[5] == pytest.approx(0.4)
    assert so101_robot.control_loop.snapshot()["arrival"]["reached"]


def test_jaw_that_never_arrives_faults(so101_robot):
    so101_robot.execute(CLOSE)
    so101_robot.control_loop.driver.hardware.response = lambda goal, position: (*goal[:5], min(goal[5], 0.5))

    result = so101_robot.execute(OPEN)

    assert result.status == "failed"
    assert result.error["message"] == "Station fault: FAULT: Gripper opening endpoint was not reached"
    assert so101_robot.status()["status"] == "FAULT"
    assert so101_robot.control_loop.snapshot()["targets"]["arm"][5] == pytest.approx(0.5)


def test_the_stand_in_arm_reads_back_what_it_was_sent(arm_configs):
    hardware = StandInSO101Arm(*arm_configs)

    hardware.send(TARGET)
    state = hardware.read()

    assert state.position == TARGET
    assert state.velocity == (0.0,) * len(TARGET)
    assert hardware.sent_targets == [TARGET]
    assert hardware.read().sample_id == state.sample_id + 1


def test_the_stand_in_arm_starts_at_home(arm_configs):
    control_config, config = arm_configs

    assert StandInSO101Arm(control_config, config).read().position == tuple(config.home.position)


def test_recovery_check_reads_a_healthy_arm(arm_configs):
    control_config, config = arm_configs
    hardware = StandInSO101Arm(control_config, config)

    assert hardware.read_for_recovery().position == tuple(config.home.position)

    hardware.failure = ServoFault("Servo error: elbow_flex: overload")
    with pytest.raises(ServoFault, match="overload"):
        hardware.read_for_recovery()


def test_a_closed_stand_in_arm_refuses_reads_and_commands(arm_configs):
    hardware = StandInSO101Arm(*arm_configs)

    hardware.close()

    assert hardware.is_closed
    with pytest.raises(RuntimeError, match="Hardware is closed"):
        hardware.read()
    with pytest.raises(RuntimeError, match="Hardware is closed"):
        hardware.send(TARGET)


def test_initial_hold_is_the_measured_pose_even_past_a_limit(arm_configs):
    control_config, config = arm_configs
    driver = StandInSO101Driver(control_config, config)
    hardware = driver.hardware
    q = list(hardware.position)
    q[1] = config.limits.lower_bounds[1] - 0.03
    hardware.position = tuple(q)

    runtime = SO101ControlLoop(driver, control_config, config)

    try:
        assert runtime.targets["arm"] == tuple(q)
        assert hardware.sent_targets == [tuple(q)]
    finally:
        runtime.close()


def test_a_brief_bus_glitch_does_not_fault(runtime):
    Waiting.until(lambda: runtime.snapshot()["arrival"]["settled"], WAIT_TIMEOUT_S, WAIT_POLL_S)

    runtime.driver.hardware.failure = RuntimeError("Incorrect status packet")
    time.sleep(0.04)  # two ticks, well inside state_timeout_s
    runtime.driver.hardware.failure = None
    time.sleep(0.05)

    assert runtime.status == "HOLD"


def test_a_bus_glitch_while_enabling_settles_fails_the_enable(arm_configs):
    control_config, config = arm_configs
    driver = StandInSO101Driver(control_config, config)
    hardware = driver.hardware
    read, reads = hardware.read, []

    def glitching():
        reads.append(len(reads))
        if len(reads) == 2:
            raise RuntimeError("Incorrect status packet")
        return read()

    hardware.read = glitching
    with pytest.raises(RuntimeError, match="Incorrect status packet"):
        SO101ControlLoop(driver, control_config, config)


def test_lost_feedback_faults_after_state_timeout(runtime):
    runtime.driver.hardware.failure = RuntimeError("bus unplugged")

    Waiting.until(lambda: runtime.status == "FAULT", WAIT_TIMEOUT_S, WAIT_POLL_S)

    assert "Stale" in runtime.fault


def test_servo_fault_is_immediate_and_holds_the_measured_pose(runtime):
    Waiting.until(lambda: runtime.snapshot()["arrival"]["settled"], WAIT_TIMEOUT_S, WAIT_POLL_S)
    q = list(runtime.driver.hardware.position)
    q[0] += 0.1  # the arm was pushed away from its target, within the tracking bound
    runtime.driver.hardware.position = tuple(q)
    Waiting.until(lambda: runtime.states["arm"].position[0] == q[0], WAIT_TIMEOUT_S, WAIT_POLL_S)

    runtime.driver.hardware.failure = ServoFault("Servo error: elbow_flex: overload")
    Waiting.until(lambda: runtime.status == "FAULT", WAIT_TIMEOUT_S, WAIT_POLL_S)

    assert "overload" in runtime.fault
    assert runtime.targets["arm"][0] == pytest.approx(q[0])
    assert runtime.driver.hardware.sent_targets[-1][0] == pytest.approx(q[0])


def test_recovery_reacquires_a_hold(runtime):
    runtime.driver.hardware.failure = ServoFault("Servo torque is off: gripper")
    Waiting.until(lambda: runtime.status == "FAULT", WAIT_TIMEOUT_S, WAIT_POLL_S)

    runtime.driver.hardware.failure = ServoFault("still off")
    with pytest.raises(Exception, match="still off"):
        runtime.recover(timeout_s=1)

    runtime.driver.hardware.failure = None
    generation = runtime.generation
    snapshot = runtime.recover(timeout_s=2)
    assert snapshot["status"] == "HOLD" and runtime.fault is None
    assert runtime.generation > generation
    assert runtime.snapshot()["motion"]["end_reason"] == "recovered"


def test_recovery_holds_the_folded_rest_pose_past_a_limit(arm_configs):
    control_config, config = arm_configs
    driver = StandInSO101Driver(control_config, config)
    hardware = driver.hardware
    q = list(hardware.position)
    q[1] = config.limits.lower_bounds[1] - config.feedback_position_tolerance / 2  # folded onto the shoulder's stop
    hardware.position = tuple(q)
    runtime = SO101ControlLoop(driver, control_config, config)
    runtime.start()

    try:
        hardware.failure = ServoFault("Servo torque is off: gripper")
        Waiting.until(lambda: runtime.status == "FAULT", WAIT_TIMEOUT_S, WAIT_POLL_S)
        hardware.failure = None

        snapshot = runtime.recover(timeout_s=2)

        assert snapshot["status"] == "HOLD" and runtime.fault is None
        assert runtime.targets["arm"] == tuple(q)
    finally:
        runtime.close()
