import copy
import dataclasses
import json
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig
from omegaconf.errors import ConfigKeyError
from ruckig import InputParameter, Ruckig, Synchronization, Trajectory

from config_files import ConfigFiles
from control_rig import ControlRig, Robots, StandInClock, Ticker
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.control_loop import Braking, JointControlLoop
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.operations import OperationInterrupted, SequenceOperation
from pear.deploy.station.control.joint.recovery import (
    MeasuredHoldRecovery,
    RecoveryPolicy,
    RecoveryRequest,
    ReleaseRecovery,
)
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.tracking import Tracking, TrackingError
from pear.deploy.station.control.joint.trajectory import JointMove, JointTrajectory
from pear.deploy.station.control.robot import ActionMode
from pear.deploy.station.control.robots.so101.layout import LAYOUT as SO101_LAYOUT
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from pear.deploy.station.exceptions import OperatorPaused
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import SteppedStandInYAMRobot
from yam_rigs import ControlHelpers, MotionHelpers, ReplayRobot

# the shared control loop on a driver in memory
STEP_RAD = 0.1  # a move of every coordinate
DISPLACEMENT_RAD = 0.3  # an arm pushed beyond its tracking bound
LONG_DISPLACEMENT_RAD = 1.0  # a push whose release outlasts a recovery's default timeout
DRIFT_RAD = 0.1  # a jump of a pushed arm during the release, beyond the position tolerance
SLIP = 0.1  # how far each jaw rests short of its command, on an object
STALL_PERIODS = 20  # control periods a stalled controller misses
STALL_AFTER = 5  # control periods the thread runs before it stalls
LOOP_PERIODS = 40  # control periods the thread runs in all
THREAD_END_S = 10.0  # for the control thread of a test to end once it has run its periods
SLOW_SCALE = 0.25  # the speed scale of a move whose own caps brake gently
SEQUENCE_ROWS = 10  # of a sequence that a stall makes late
SEQUENCE_SPAN_RAD = 0.05  # how far that sequence moves the first joint
LATE_S = 1.0  # past an operation's deadline
DRIFT_RAD_S = 0.2  # a velocity reading far above the settling tolerance
RECOVERY_TIMEOUT_S = 2.0  # of a recovery a test gives a timeout

# moves and sequences of any layout
BRAKING_RATIO = 0.6  # how much shorter the stronger STOP caps must brake
SEQUENCE_RATE_HZ = 25.0  # of a sequence operation stepped by hand

# the SO-101 resting past a limit
SO101_REST = (0.0, -1.0, 1.0, 0.5, 0.0, 0.5)  # the SO-101's arm at rest, its jaw half open
LIFT = 1  # shoulder_lift, which rests past its lower limit when folded
PAST_RAD = 0.03  # how far past the limit it rests
LIFT_STEP_RAD = 0.01  # a step of shoulder_lift
PAN_STEP_RAD = 0.05  # a step of shoulder_pan

# the YAM arms enabled past a limit
J1 = 0
J4 = 3  # the joint that rests outside its lower limit
GRIPPER = 6
OUTSIDE_RAD = 0.01  # how far past the limit it rests, within the feedback tolerance
OPENING_PAST = 0.005  # how far past fully open a gripper rests, within the feedback tolerance
AWAY_RAD = 0.3  # each arm's base turn away from Home

RECORDED_STATIC_TRACKING_ERROR = "recorded_static_tracking_error.json"  # a recorded hold with a static error
ENDPOINT_TARGETS = "endpoint_targets.json"  # recorded starts and goals of moves


class MemoryDriver(JointDriver):
    """A joint driver in memory: every arm reaches its command at once; a test makes its readings stale, holds its
    arms where they are or gives its first joints a velocity."""

    STALE_S = 1.0  # the age of a stale reading

    def __init__(self, layout: JointLayout, start: tuple[float, ...]) -> None:
        self.positions = layout.split(start)  # each arm's reading, by the arm's name
        self.is_stale = False
        self.is_following = True  # whether the arms move to each command; False: held in place by contact
        self.velocity = 0.0  # every arm's first-joint velocity reading, rad/s
        self.is_oscillating = False  # whether that velocity flips sign on every reading
        # each command, with its planned velocities, the latest last
        self.sent: list[tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]] | None]] = []
        self.refreshes = 0  # how often a hold was renewed
        self.reads = 0  # the readings so far, which number them

    def read(self) -> dict[str, ArmState]:
        """Returns every arm where its last command put it, with the first joint's velocity reading."""
        self.reads += 1
        sampled_monotonic_s = time.monotonic() - (self.STALE_S if self.is_stale else 0.0)
        velocity = self.velocity * (-1) ** self.reads if self.is_oscillating else self.velocity

        return {
            arm: ArmState(q, (velocity,) + (0.0,) * (len(q) - 1), sampled_monotonic_s, self.reads)
            for arm, q in self.positions.items()
        }

    def read_for_recovery(self) -> dict[str, ArmState]:
        """Returns every arm's reading: the arms are always healthy enough to hold again."""
        return self.read()

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Keeps the command and puts every arm at its target, unless the arms are held in place."""
        self.sent.append((dict(targets), velocities))
        if self.is_following:
            self.positions = {arm: tuple(q) for arm, q in targets.items()}

    def refresh_hold(self, targets: dict[str, tuple[float, ...]]) -> None:
        """Counts the renewal of a hold."""
        self.refreshes += 1

    def close(self) -> None:
        """Ends nothing: the driver opened no device."""


class NoGripper:
    """A gripper operation the control loop tests never start."""


class PacedExit:
    """A control thread's exit event on the stand-in clock: each wait advances the clock, and it is set after
    ``periods`` waits."""

    def __init__(self, clock: StandInClock, periods: int) -> None:
        self.clock, self.periods = clock, periods
        self.waits: list[float] = []  # the timeout of each wait, in order

    def is_set(self) -> bool:
        """Whether the thread is to end."""
        return len(self.waits) >= self.periods

    def wait(self, timeout: float) -> bool:
        """Advances the clock by ``timeout`` and returns whether the thread is to end."""
        self.waits.append(timeout)
        self.clock.now += timeout

        return self.is_set()

    def set(self) -> None:
        """Ends the thread at its next check."""
        self.periods = len(self.waits)


class JointRuntimeRig:
    """A control loop on a driver in memory for each robot's layout, stepped by hand, and what the tests do to its
    arms."""

    @staticmethod
    def rig(
        robot_name: str,
        clock: StandInClock,
        *,
        start: list[float] | None = None,
        recovery: type[RecoveryPolicy] = MeasuredHoldRecovery,
        braking: Braking = Braking.STATION_LIMITS,
        checks_velocity_rms: bool = False,
    ) -> tuple[JointControlLoop, MemoryDriver, Ticker]:
        """Returns a control loop of the robot ``robot_name`` on a driver in memory at ``start`` (the middle of the
        bounds by default), its driver and its ticker."""
        control_config, robot_config = Robots.load(robot_name)
        layout = robot_config.LAYOUT
        driver = MemoryDriver(layout, Robots.rest(robot_config) if start is None else start)
        control_loop_class = type(
            "MemoryControlLoop",
            (JointControlLoop,),
            {
                "GRIPPER_OPERATION_CLASS": NoGripper,
                "RECOVERY_POLICY_CLASS": recovery,
                "BRAKING": braking,
                "CHECKS_VELOCITY_RMS": checks_velocity_rms,
            },
        )
        runtime = control_loop_class(driver, control_config, robot_config)

        return runtime, driver, Ticker(runtime, clock)

    @staticmethod
    def push(
        runtime: JointControlLoop, driver: MemoryDriver, ticker: Ticker, displacement: float = DISPLACEMENT_RAD
    ) -> dict[str, tuple[float, ...]]:
        """Pushes the first arm joint beyond its tracking bound while the arm stays where it is pushed."""
        driver.is_following = False
        first = runtime.layout.arms[0]
        pushed = {
            **driver.positions,
            first: (driver.positions[first][0] + displacement, *driver.positions[first][1:]),
        }
        driver.positions = pushed
        ticker.tick()

        return pushed

    @staticmethod
    def reach(runtime: JointControlLoop, delta: float) -> dict[str, tuple[float, ...]]:
        """Returns the held targets with every arm's first joint moved by ``delta``."""
        return {arm: (target[0] + delta, *target[1:]) for arm, target in runtime.targets.items()}

    @staticmethod
    def recover(runtime: JointControlLoop, ticker: Ticker, **criteria: Any) -> dict[str, Any]:
        """Recovers ``runtime`` with each sleep of the wait stepping it one control period, and returns the
        snapshot."""
        ticker.clock.control_loop = runtime
        try:
            return runtime.recover(**criteria)
        finally:
            ticker.clock.control_loop = None


class Sequences:
    """Moves, trajectories and sequence operations of a robot's layout, at rest in the middle of the bounds."""

    @staticmethod
    def trajectory(control_config: ControlConfig, robot_config: RobotConfig) -> JointTrajectory:
        """Returns a trajectory at rest in the middle of the bounds."""
        trajectory = JointTrajectory(robot_config.joint_limits(), robot_config.LAYOUT, control_config)
        trajectory.hold(Robots.rest(robot_config))

        return trajectory

    @staticmethod
    def sample_times(move: JointMove, rate_hz: float) -> np.ndarray:
        """Returns uniform sample times of ``move`` at ``rate_hz``, both ends included."""
        return np.linspace(0, move.duration_s, max(2, int(np.ceil(move.duration_s * rate_hz)) + 1))


class YAMStationRig:
    """What the tests on both YAM arms of their station of the tests share: the end of an operation and a chunk of
    joint targets."""

    MAX_STEPS = 2000  # the most control periods an operation may take

    @classmethod
    def finish(cls, runtime: JointControlLoop, tick: Callable[..., None]) -> dict[str, Any]:
        """Steps until the operation ends in a HOLD or a FAULT, and returns the snapshot then."""
        for _ in range(cls.MAX_STEPS):
            tick()
            if runtime.status in ("HOLD", "FAULT"):
                return runtime.snapshot()

        pytest.fail("controller did not terminate")

    @staticmethod
    def chunk(runtime: JointControlLoop, start: float, stop: float, count: int = 10) -> np.ndarray:
        """Returns ``count`` copies of the held targets, the right arm's first joint from ``start`` to ``stop``."""
        q = np.array([v for s in ("left", "right") for v in runtime.targets[s]])
        actions = np.tile(q, (count, 1))
        actions[:, 7] = np.linspace(start, stop, count)

        return actions


class ManualRecovery:
    """A fault of a YAM control loop with wide limits, and its recovery stepped by hand."""

    DISPLACED_RAD = 0.3  # where a load keeps the right arm's first joint after a fault

    @staticmethod
    def fault(runtime: JointControlLoop, hardware: StandInYAMDriver, ticker: Ticker) -> None:
        """Faults the control loop where it is, by one period of stale feedback, and makes the feedback fresh again."""
        hardware.is_stale = True
        ticker.tick()
        hardware.is_stale = False

    @staticmethod
    def recover(runtime: JointControlLoop, ticker: Ticker, **criteria: Any) -> dict[str, Any]:
        """Recovers ``runtime`` with each sleep of the wait stepping it one control period, and returns the
        snapshot."""
        return JointRuntimeRig.recover(runtime, ticker, **criteria)

    @classmethod
    def displaced_hold(
        cls,
        rig: tuple[JointControlLoop, StandInYAMDriver, Ticker],
        monkeypatch: pytest.MonkeyPatch,
        displacement: float = DISPLACED_RAD,
    ) -> tuple[JointControlLoop, StandInYAMDriver]:
        """Faults the control loop while a load keeps the right arm away from its command, which the arm no longer
        follows, and returns the loop and its driver."""
        runtime, hardware, ticker = rig
        cls.fault(runtime, hardware, ticker)
        # a load or a contact can keep a powered arm stationary away from its old command
        hardware.positions["right"] = (displacement, 0, 0, 0, 0, 0, 0.2)
        monkeypatch.setattr(
            hardware, "send", lambda targets, velocities=None: hardware.sent_targets.append(dict(targets))
        )

        return runtime, hardware


class JointMoves:
    """YAM moves of their station of the tests and the checks of a move against its caps."""

    @staticmethod
    def fixture_move(robot_config: RobotConfig, speed_scale: float = 1 / 3) -> JointMove:
        """Returns a slow move of the right arm's first two joints, both grippers half open."""
        start = np.zeros(14)
        start[[6, 13]] = 0.5
        goal = start.copy()
        goal[7:9] = [0.15, 0.06]

        return JointMove.build(start, goal, robot_config.joint_limits(), LAYOUT, speed_scale=speed_scale)

    @staticmethod
    def check_bounds(move: JointMove) -> None:
        """Checks that ``move`` keeps its caps and starts and ends at rest at its start and its goal."""
        times = np.linspace(0, move.duration_s, 1001)
        samples = np.array([move.at_time(t) for t in times])
        caps = np.asarray(move.motion_caps)
        assert np.all(abs(samples[:, 1]) <= caps[0] + 1e-9)
        assert np.all(abs(samples[:, 2]) <= caps[1] + 1e-9)
        if move.duration_s > 0:
            assert np.all(abs(np.diff(samples[:, 2], axis=0) / (move.duration_s / 1000)) <= caps[2] + 1e-7)
        assert samples[0, 0] == pytest.approx(move.start)
        assert samples[-1, 0] == pytest.approx(move.goal)
        assert samples[[0, -1], 1:] == pytest.approx(np.zeros((2, 2, 14)), abs=1e-9)


class JointBounds:
    """The SO-101 folded: shoulder_lift resting past its lower limit."""

    @staticmethod
    def resting(config: RobotConfig) -> np.ndarray:
        """Returns the folded pose: shoulder_lift resting past its lower limit."""
        q = np.zeros(SO101_LAYOUT.vector_length)
        q[LIFT] = config.limits.lower_bounds[LIFT] - PAST_RAD

        return q


@pytest.fixture
def yam_station_rig(
    stand_in_clock: StandInClock,
) -> Iterator[tuple[JointControlLoop, StandInYAMDriver, StandInClock, Callable[..., None]]]:
    """Both YAM arms of their station of the tests on stand-in joints, stepped by hand: the control loop, its driver,
    the clock and a step of one control period."""
    with ControlRig(YAMControlLoop, StandInYAMDriver, *ConfigFiles.load_configs("yam"), stand_in_clock) as rig:
        runtime, device, ticker = rig
        yield runtime, device, stand_in_clock, ticker.tick


@pytest.fixture
def startup_rig(stand_in_clock: StandInClock) -> Iterator[tuple[JointControlLoop, StandInYAMDriver, Ticker, tuple]]:
    """A YAM control loop with wide limits, enabled with the left arm's J4 resting past its lower limit: the loop, its
    driver, its ticker and that resting pose; the loop is closed at the end."""
    control_config, config = Robots.load("yam")
    hardware = StandInYAMDriver(control_config, config)
    resting = list(hardware.positions["left"])
    resting[J4] = config.arms["left"].limits.lower_bounds[J4] - OUTSIDE_RAD
    hardware.positions = {**hardware.positions, "left": tuple(resting)}
    runtime = YAMControlLoop(hardware, control_config, config)

    try:
        yield runtime, hardware, Ticker(runtime, stand_in_clock), tuple(resting)
    finally:
        runtime.close()


@pytest.fixture
def held_past_a_limit_rig(
    stand_in_clock: StandInClock, outside: str
) -> Iterator[tuple[JointControlLoop, StandInYAMDriver, Ticker]]:
    """A YAM control loop of its station of the tests, enabled with both arms turned AWAY_RAD and the left arm past a
    limit as ``outside`` says: the loop, its driver and its ticker; the loop is closed at the end."""
    control_config, config = ConfigFiles.load_configs("yam")
    hardware = StandInYAMDriver(control_config, config)
    limits = config.arms["left"].limits
    resting = {arm: list(position) for arm, position in hardware.positions.items()}
    for position in resting.values():
        position[J1] = AWAY_RAD
    if outside.startswith("joint"):
        resting["left"][J4] = limits.lower_bounds[J4] - OUTSIDE_RAD
    else:
        resting["left"][GRIPPER] = limits.upper_bounds[GRIPPER] + OPENING_PAST
    hardware.positions = {arm: tuple(position) for arm, position in resting.items()}
    runtime = YAMControlLoop(hardware, control_config, config)

    try:
        yield runtime, hardware, Ticker(runtime, stand_in_clock)
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


@pytest.fixture
def so101_configs(so101_config: DictConfig) -> tuple[ControlConfig, RobotConfig]:
    """The two configurations of the SO-101's station of the tests with its short settle windows and fast caps."""
    return ConfigFiles.make_configs(so101_config)


def test_the_control_loop_holds_the_measured_pose_even_past_a_limit(robot_name, stand_in_clock):
    control_config, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    start = list(Robots.rest(robot_config))
    start[1] = robot_config.joint_limits().lower_bounds[1] - robot_config.joint_limits().feedback_tolerance / 2

    runtime, driver, _ = JointRuntimeRig.rig(robot_name, stand_in_clock, start=start)

    assert runtime.targets == layout.split(start)
    assert driver.sent == [(layout.split(start), None)]
    assert runtime.status == "HOLD"


def test_the_control_loop_moves_to_a_target_and_holds(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    goal = {arm: tuple(q + STEP_RAD for q in target) for arm, target in runtime.targets.items()}

    runtime.move(goal)
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert runtime.layout.join(driver.positions) == pytest.approx(runtime.layout.join(goal))
    assert runtime.snapshot()["motion"]["end_reason"] == "finished"


def test_the_control_loop_is_settled_once_the_hold_passes_its_stability_window(robot_name, stand_in_clock):
    runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)

    runtime.move(JointRuntimeRig.reach(runtime, STEP_RAD))
    assert not runtime.snapshot()["arrival"]["settled"]
    ticker.until_hold()
    for _ in range(Ticker.STEPS):
        if runtime.snapshot()["arrival"]["settled"]:
            break
        ticker.tick()

    assert runtime.snapshot()["arrival"]["settled"]


def test_a_sequence_reaches_its_last_row(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    rows = np.tile(runtime.layout.join(runtime.targets), (5, 1))
    rows[:, 0] += np.linspace(0, 0.05, 5)

    runtime.start_sequence(rows, rate_hz=runtime.control_config.rate_hz / 2)
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert runtime.layout.join(driver.positions) == pytest.approx(rows[-1])


def test_station_limits_brake_a_slow_move_sooner_than_its_own_caps(robot_name, stand_in_clock):
    def stopping_steps(braking):
        runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, braking=braking)
        runtime.move(JointRuntimeRig.reach(runtime, 10 * STEP_RAD), speed_scale=SLOW_SCALE)
        ticker.tick(50)
        runtime.stop()
        steps = ticker.until_hold()
        assert runtime.snapshot()["motion"]["end_reason"] == "stopped"
        return steps

    assert stopping_steps(Braking.STATION_LIMITS) < stopping_steps(Braking.MOVE_CAPS)


def test_stale_feedback_faults(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    driver.is_stale = True

    ticker.tick()

    assert runtime.status == "FAULT"
    assert runtime.fault == runtime.layout.texts.stale_feedback.format(arm=runtime.layout.arms[0], age=driver.STALE_S)


def test_measured_hold_recovery_holds_where_the_arm_was_pushed(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=MeasuredHoldRecovery)
    pushed = JointRuntimeRig.push(runtime, driver, ticker)

    assert runtime.status == "FAULT"
    assert runtime.targets == pushed
    JointRuntimeRig.recover(runtime, ticker)

    assert runtime.status == "HOLD" and runtime.targets == pushed
    assert runtime.snapshot()["motion"]["end_reason"] == "recovered"


def test_measured_hold_recovery_never_holds_a_pose_past_the_feedback_bounds(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=MeasuredHoldRecovery)
    JointRuntimeRig.push(runtime, driver, ticker)
    first = runtime.layout.arms[0]
    position = list(runtime.states[first].position)
    position[1] = runtime.limits.lower_bounds[1] - 2 * runtime.limits.feedback_tolerance
    # a reading the control loop's own feedback check would have refused, handed straight to the policy
    runtime.states = {**runtime.states, first: dataclasses.replace(runtime.states[first], position=tuple(position))}
    request = RecoveryRequest(SimpleNamespace(reset=lambda: None), ticker.clock.now + RECOVERY_TIMEOUT_S)
    sent = len(driver.sent)

    with pytest.raises(ValueError, match="outside"):
        runtime.recovery_policy.step(runtime, request, runtime.snapshot(), True)

    assert len(driver.sent) == sent and request.phase == "Checking stability"


def test_release_recovery_releases_the_old_target_slowly(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=ReleaseRecovery)
    held = dict(runtime.targets)
    pushed = JointRuntimeRig.push(runtime, driver, ticker)

    assert runtime.status == "FAULT" and runtime.targets == held
    sent = len(driver.sent)
    JointRuntimeRig.recover(runtime, ticker)

    assert runtime.status == "HOLD"
    assert runtime.layout.join(runtime.targets) == pytest.approx(runtime.layout.join(pushed))
    steps = np.diff([runtime.layout.join(targets) for targets, _ in driver.sent[sent:]], axis=0)
    assert np.max(abs(steps)) <= ReleaseRecovery.RELEASE_CAPS[0] * runtime.control_config.control_period_s + 1e-9


def test_a_stalled_clock_advances_one_sample_per_step(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    runtime.move(JointRuntimeRig.reach(runtime, 10 * STEP_RAD))
    ticker.tick(10)
    before = np.array(runtime.layout.join(runtime.targets))

    ticker.tick(period_s=STALL_PERIODS * runtime.control_config.control_period_s)

    step = abs(np.array(runtime.layout.join(runtime.targets)) - before)
    assert runtime.status == "RUNNING", runtime.fault
    assert np.all(step <= np.array(runtime.limits.velocity) * runtime.control_config.control_period_s + 1e-9)


def test_a_hold_refreshes_the_driver_every_period(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)

    ticker.tick(7)

    assert driver.refreshes == 7
    assert len(driver.sent) == 1


def test_the_control_thread_never_bursts_after_a_stall(robot_name, stand_in_clock, monkeypatch):
    exit_event = PacedExit(stand_in_clock, LOOP_PERIODS)
    # the control loop's exit event, made with the loop
    with monkeypatch.context() as patch:
        patch.setattr(threading, "Event", lambda: exit_event)
        runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    step = runtime.step

    def stalling_step():
        step()
        if len(exit_event.waits) == STALL_AFTER:
            ticker.clock.now += STALL_PERIODS * runtime.control_config.control_period_s

    runtime.step = stalling_step
    threads, thread_class = [], threading.Thread
    # the control loop's thread, as the loop starts it
    with monkeypatch.context() as patch:
        patch.setattr(threading, "Thread", lambda **kwargs: threads.append(thread_class(**kwargs)) or threads[-1])
        runtime.start()
    [control_thread] = threads
    control_thread.join(THREAD_END_S)

    assert exit_event.waits == pytest.approx([runtime.control_config.control_period_s] * LOOP_PERIODS)


def test_a_sequence_late_after_a_stall_stops_at_its_deadline(robot_name, stand_in_clock):
    runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    rows = np.tile(runtime.layout.join(runtime.targets), (SEQUENCE_ROWS, 1))
    rows[:, 0] += np.linspace(0, SEQUENCE_SPAN_RAD, SEQUENCE_ROWS)
    runtime.start_sequence(rows, rate_hz=runtime.control_config.rate_hz / 2)
    ticker.tick(3)

    ticker.tick(period_s=STALL_PERIODS * runtime.control_config.control_period_s)
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert runtime.snapshot()["motion"]["end_reason"] == "sequence_deadline"


def test_a_cancelled_operation_brakes_into_a_hold(robot_name, stand_in_clock):
    runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    runtime.move(JointRuntimeRig.reach(runtime, 10 * STEP_RAD))
    runtime.operation_cancellation = threading.Event()
    ticker.tick(5)

    runtime.operation_cancellation.set()
    ticker.tick()

    assert runtime.status == "STOPPING"
    ticker.until_hold()
    assert runtime.snapshot()["motion"]["end_reason"] == "cancelled"


def test_an_operation_past_its_deadline_brakes_into_a_hold(robot_name, stand_in_clock):
    runtime, _, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    runtime.move(JointRuntimeRig.reach(runtime, 10 * STEP_RAD))
    ticker.tick(5)

    margin = runtime.MOVE_MARGIN_S + runtime.control_config.settle_timeout_s + LATE_S
    ticker.tick(period_s=runtime.operation.move.duration_s + margin)

    assert runtime.status == "STOPPING"
    ticker.until_hold()
    assert runtime.snapshot()["motion"]["end_reason"] == "action_timeout"


def test_persistent_motion_in_a_hold_faults_with_its_details_in_the_snapshot(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    driver.velocity = DRIFT_RAD_S

    ticker.tick(int(runtime.control_config.settle_timeout_s / runtime.control_config.control_period_s) + STALL_PERIODS)
    snapshot = runtime.snapshot()
    snapshot["fault_details"]["reason"] = "changed by a reader"

    assert snapshot["status"] == "FAULT"
    assert runtime.snapshot()["fault_details"]["code"] == "StabilityError"
    assert runtime.snapshot()["fault_details"]["reason"] == "Persistent motion during HOLD"


def test_a_failed_fault_hold_blocks_every_recovery(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock)
    send = driver.send

    def failing_send(targets, velocities=None):
        raise RuntimeError("bus write failed")

    driver.send, driver.is_stale = failing_send, True
    ticker.tick()
    driver.send, driver.is_stale = send, False

    assert runtime.status == "FAULT"
    with pytest.raises(RuntimeError, match="bus write failed"):
        JointRuntimeRig.recover(runtime, ticker)
    assert runtime.status == "FAULT"


def test_release_recovery_is_blocked_by_an_arm_moving_during_the_release(robot_name, stand_in_clock, monkeypatch):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=ReleaseRecovery)
    JointRuntimeRig.push(runtime, driver, ticker)
    first = runtime.layout.arms[0]
    jumped = []

    def sleep(_):
        recovery = runtime.snapshot()["recovery"]
        if not jumped and recovery is not None and recovery["phase"] == "Releasing old target":
            position = driver.positions[first]
            driver.positions = {**driver.positions, first: (position[0] + DRIFT_RAD, *position[1:])}
            jumped.append(True)
        ticker.tick()

    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="Recovery blocked: .*moved"):
        runtime.recover()
    assert runtime.status == "FAULT"


def test_release_recovery_extends_its_automatic_timeout_to_a_long_release(robot_name, stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=ReleaseRecovery)
    JointRuntimeRig.push(runtime, driver, ticker, LONG_DISPLACEMENT_RAD)
    started = ticker.clock.now

    JointRuntimeRig.recover(runtime, ticker)

    assert runtime.status == "HOLD"
    assert ticker.clock.now - started > ReleaseRecovery().default_timeout_s(runtime.control_config)


@pytest.mark.parametrize("recovery", [MeasuredHoldRecovery, ReleaseRecovery])
def test_recovery_keeps_each_gripper_command(robot_name, stand_in_clock, recovery):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, recovery=recovery)
    commanded = {arm: target[-1] for arm, target in runtime.targets.items()}
    JointRuntimeRig.push(runtime, driver, ticker)
    driver.positions = {arm: (*q[:-1], q[-1] - SLIP) for arm, q in driver.positions.items()}

    JointRuntimeRig.recover(runtime, ticker)

    assert runtime.status == "HOLD"
    assert {arm: target[-1] for arm, target in runtime.targets.items()} == commanded


@pytest.mark.parametrize("checks_velocity_rms", [False, True])
def test_a_recovery_window_bounds_velocity_rms_only_when_asked(robot_name, stand_in_clock, checks_velocity_rms):
    runtime, driver, ticker = JointRuntimeRig.rig(robot_name, stand_in_clock, checks_velocity_rms=checks_velocity_rms)
    driver.is_stale = True
    ticker.tick()
    driver.is_stale = False
    driver.velocity, driver.is_oscillating = DRIFT_RAD_S, True

    if checks_velocity_rms:
        with pytest.raises(RuntimeError, match="velocity_rms"):
            JointRuntimeRig.recover(runtime, ticker, timeout_s=RECOVERY_TIMEOUT_S)
    else:
        JointRuntimeRig.recover(runtime, ticker, timeout_s=RECOVERY_TIMEOUT_S)

    assert runtime.status == ("FAULT" if checks_velocity_rms else "HOLD")


def test_a_driver_reading_its_arms_in_another_order_keeps_the_layout_order(stand_in_clock):
    runtime, driver, ticker = JointRuntimeRig.rig("yam", stand_in_clock)
    read = driver.read
    driver.read = lambda: dict(reversed(read().items()))

    JointRuntimeRig.push(runtime, driver, ticker)
    snapshot = runtime.snapshot()

    assert list(runtime.targets) == list(snapshot["targets"]) == list(snapshot["arms"]) == list(runtime.layout.arms)
    assert SnapshotCheck.target_vector(snapshot) == pytest.approx(runtime.layout.join(runtime.targets))
    assert SnapshotCheck.state_vector(snapshot) == pytest.approx(runtime.layout.join(driver.positions))


def test_split_and_join_are_inverse(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    vector = tuple(float(i) for i in range(layout.vector_length))

    per_arm = layout.split(vector)

    assert list(per_arm) == list(layout.arms)
    assert layout.join(per_arm) == vector


@pytest.mark.parametrize(("robot_name", "indices"), [("yam", (6, 13)), ("so101", (5,))])
def test_each_arm_s_gripper_is_its_last_coordinate(robot_name, indices):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT

    assert layout.gripper_indices == indices
    assert [layout.axis_unit(i) for i in indices] == ["opening"] * len(indices)


@pytest.mark.parametrize(
    ("robot_name", "index", "axis"),
    [("yam", 8, {"side": "right", "joint": "J2"}), ("so101", 1, {"joint": "shoulder_lift"})],
)
def test_a_coordinate_is_named_by_its_arm_and_joint(robot_name, index, axis):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT

    assert (
        layout.arm_axis_labels(layout.arms[index // layout.coordinate_count], index % layout.coordinate_count) == axis
    )


@pytest.mark.parametrize(("robot_name", "joints"), [("yam", 6), ("so101", 5)])
def test_vector_rejects_wrong_or_nonfinite_values(robot_name, joints):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    message = f"Expected {joints} finite joint radians and 1 finite gripper opening"

    with pytest.raises(ValueError, match=message):
        layout.vector([0.0] * (layout.vector_length - 1))
    with pytest.raises(ValueError, match=message):
        layout.vector([np.nan] * layout.vector_length)


@pytest.mark.parametrize(
    ("robot_name", "message"),
    [
        (
            "yam",
            "left: joint or gripper target outside commissioned bounds: "
            "joint 3=-3.150000 rad, bounds=[-3.000000, 3.000000], feedback allowance=0.020000",
        ),
        (
            "so101",
            "Joint or gripper target outside commissioned bounds: "
            "elbow_flex=-1.800300 rad, bounds=[-1.650300, 1.650300], allowance=0.020000",
        ),
    ],
)
def test_positions_outside_the_bounds_are_named(robot_name, message):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    vector = list(Robots.rest(robot_config))
    vector[2] = limits.lower_bounds[2] - 0.15

    with pytest.raises(ValueError) as error:
        limits.check_positions(vector, layout, 0.02)

    assert str(error.value) == message


def test_a_command_outside_a_limit_moves_only_inward(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    previous = list(Robots.rest(robot_config))
    previous[1] = limits.lower_bounds[1] - 0.03
    inward, outward = list(previous), list(previous)
    inward[1] += 0.01
    outward[1] -= 0.01

    limits.check_command(previous, previous, layout)
    limits.check_command(inward, previous, layout)
    with pytest.raises(ValueError, match=layout.texts.bound_axes[1]):
        limits.check_command(outward, previous, layout)


def test_the_so101_names_the_joint_of_a_command_past_a_bound():
    _, robot_config = Robots.load("so101")
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    target = list(Robots.rest(robot_config))
    target[0] = limits.upper_bounds[0] + 0.1

    with pytest.raises(ValueError, match=r"^Command leaves the position bounds: shoulder_pan=1\.862300 rad$"):
        limits.check_command(target, Robots.rest(robot_config), layout)


def test_dynamics_apply_one_arms_caps_to_every_arm_and_scale_with_time(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    width = len(layout.coordinates)
    caps = type("Caps", (), {"velocity": [1e-3] * width, "acceleration": [1e3] * width, "jerk": [1e3] * width})

    velocity, acceleration, jerk = limits.motion_caps(0.5, caps)

    assert velocity == pytest.approx([0.5e-3] * layout.vector_length)
    assert acceleration == pytest.approx([a * 0.25 for a in limits.acceleration])
    assert jerk == pytest.approx([j * 0.125 for j in limits.jerk])
    with pytest.raises(ValueError, match="Speed scale"):
        limits.motion_caps(1.5)


def test_a_joint_move_reaches_the_goal_at_rest_within_the_caps(robot_name):
    control_config, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    start = np.array(Robots.rest(robot_config))
    goal = start + STEP_RAD

    move = JointMove.build(start, goal, limits, layout)

    q, v, _ = move.at_time(move.duration_s)
    assert q == pytest.approx(goal)
    assert v == pytest.approx(np.zeros(layout.vector_length))
    times = Sequences.sample_times(move, control_config.rate_hz)
    peak = np.max([abs(move.at_time(t)[1]) for t in times], axis=0)
    assert np.all(peak <= np.array(limits.velocity) + 1e-9)


def test_a_joint_move_crosses_a_process_boundary_without_its_native_trajectory(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    start = np.array(Robots.rest(robot_config))
    move = JointMove.build(start, start + STEP_RAD, robot_config.joint_limits(), layout)
    duration = move.duration_s

    copy = pickle.loads(pickle.dumps(move))

    assert "trajectory" not in copy.__dict__
    assert copy.duration_s == pytest.approx(duration)


def test_a_joint_move_rejects_goals_outside_the_bounds_and_nonfinite_values(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    start = np.array(Robots.rest(robot_config))
    outside = start.copy()
    outside[0] = limits.upper_bounds[0] + 0.1

    with pytest.raises(ValueError, match=layout.texts.bound_axes[0]):
        JointMove.build(start, outside, limits, layout)
    with pytest.raises(ValueError, match=f"Expected finite {layout.vector_length}D start and goal"):
        JointMove.build(start, start * np.nan, limits, layout)


def test_a_start_resting_past_a_limit_may_only_move_inward(robot_name):
    _, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    start = np.array(Robots.rest(robot_config))
    start[1] = limits.lower_bounds[1] - limits.feedback_tolerance / 2
    inward, outward = start.copy(), start.copy()
    inward[1] += STEP_RAD
    outward[1] -= STEP_RAD

    assert JointMove.build(start, inward, limits, layout).duration_s > 0
    with pytest.raises(ValueError):
        JointMove.build(start, outward, limits, layout)


def test_rejected_target_keeps_the_previous_input(robot_name):
    control_config, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    trajectory = JointTrajectory(robot_config.joint_limits(), layout, control_config)
    trajectory.hold(Robots.rest(robot_config))
    before = list(trajectory.input.target_position)
    outside = list(before)
    outside[0] = robot_config.joint_limits().upper_bounds[0] + 1

    with pytest.raises(ValueError):
        trajectory.set_target(outside)

    assert list(trajectory.input.target_position) == before


def test_stronger_braking_caps_stop_sooner(robot_name):
    control_config, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    limits = robot_config.joint_limits()
    strong = (tuple(4 * a for a in limits.acceleration), tuple(8 * j for j in limits.jerk))

    def braking_seconds(braking):
        trajectory = JointTrajectory(limits, layout, control_config)
        trajectory.hold(Robots.rest(robot_config))
        trajectory.input.current_velocity = [0.5 * v for v in limits.velocity]
        trajectory.stop(from_standstill=False, braking_caps=braking)
        return trajectory.duration_s

    assert braking_seconds(strong) < BRAKING_RATIO * braking_seconds(None)


def test_a_sequence_rejects_malformed_chunks_and_rates(robot_name):
    control_config, robot_config = Robots.load(robot_name)
    layout = robot_config.LAYOUT
    trajectory = Sequences.trajectory(control_config, robot_config)
    rows = [Robots.rest(robot_config)] * SEQUENCE_ROWS

    with pytest.raises(ValueError, match=f"Expected 1..1000 finite {layout.vector_length}D targets"):
        SequenceOperation(trajectory, [row[:-1] for row in rows], SEQUENCE_RATE_HZ)
    with pytest.raises(ValueError, match="Action rate"):
        SequenceOperation(trajectory, rows, 2 * control_config.rate_hz)


def test_a_sequence_hands_each_row_to_the_trajectory_once(robot_name, monkeypatch):
    control_config, robot_config = Robots.load(robot_name)
    trajectory = Sequences.trajectory(control_config, robot_config)
    rows = np.tile(Robots.rest(robot_config), (SEQUENCE_ROWS, 1))
    rows[:, 0] += np.linspace(0, 0.05, SEQUENCE_ROWS)
    accepted = []
    original = trajectory.set_target
    monkeypatch.setattr(trajectory, "set_target", lambda target: accepted.append(list(target)) or original(target))
    operation = SequenceOperation(trajectory, rows, SEQUENCE_RATE_HZ)

    now = 0.0
    while not operation.is_complete:
        now += trajectory.control_config.control_period_s
        trajectory.accept(operation.advance(now, {}, {}))

    assert np.array(accepted) == pytest.approx(rows)


def test_a_late_row_interrupts_the_sequence(robot_name):
    control_config, robot_config = Robots.load(robot_name)
    trajectory = Sequences.trajectory(control_config, robot_config)
    operation = SequenceOperation(trajectory, [Robots.rest(robot_config)] * SEQUENCE_ROWS, SEQUENCE_RATE_HZ)
    operation.advance(0.0, {}, {})

    with pytest.raises(OperationInterrupted, match="sequence_deadline"):
        for _ in range(SEQUENCE_ROWS):
            operation.advance(1.0, {}, {})


def test_unknown_keys_are_rejected(robot_name):
    config = ConfigFiles.load(robot_name, *ConfigFiles.WIDE_LIMITS[robot_name], "control.state_timeuot=0.5")

    with pytest.raises(ConfigKeyError):
        ConfigFiles.make_configs(config)


def test_the_settle_window_must_fit_in_the_settle_timeout(robot_name):
    config = ConfigFiles.load(robot_name, *ConfigFiles.WIDE_LIMITS[robot_name])
    config.control.settle_window_s = config.control.settle_timeout_s

    with pytest.raises(ValueError, match="settle_timeout_s must exceed settle_window_s"):
        ConfigFiles.make_configs(config)


@pytest.mark.parametrize(
    ("robot_name", "home_outside"), [("yam", ("left_pose", 6, 1.1)), ("so101", ("position", 1, 3.0))]
)
def test_home_must_lie_inside_the_position_limits(robot_name, home_outside):
    config = ConfigFiles.load(robot_name, *ConfigFiles.WIDE_LIMITS[robot_name])
    field, coordinate, value = home_outside
    config.robot.home[field][coordinate] = value

    with pytest.raises(ValueError, match="(?i)home must lie inside"):
        ConfigFiles.make_configs(config)


def test_tracking_error_beyond_its_bound_faults(so101_configs):
    _, config = so101_configs
    target = list(SO101_REST)
    target[1] += config.limits.tracking_error[1] + 0.01

    tracking = Tracking(config, config.joint_limits(), SO101_LAYOUT)

    tracking.check(SO101_REST, SO101_REST)
    with pytest.raises(TrackingError, match="Excessive tracking error: shoulder_lift error="):
        tracking.check(SO101_REST, target)


def test_gripper_arrival_uses_the_gripper_tolerance(so101_configs):
    _, config = so101_configs
    within, beyond = list(SO101_REST), list(SO101_REST)
    within[5] += 0.5 * config.gripper_tolerance
    beyond[5] += 1.5 * config.gripper_tolerance

    tracking = Tracking(config, config.joint_limits(), SO101_LAYOUT)

    arrival = tracking.arrival(SO101_REST, within)

    assert arrival["reached"]
    assert arrival["tolerance"][5] == config.gripper_tolerance
    assert not tracking.arrival(SO101_REST, beyond)["reached"]


def test_feedback_tolerance_accepts_rest_against_a_stop(so101_configs):
    _, config = so101_configs
    limits, q = config.joint_limits(), JointBounds.resting(config)

    with pytest.raises(ValueError, match="shoulder_lift"):
        limits.check_positions(q, SO101_LAYOUT)
    limits.check_positions(q, SO101_LAYOUT, config.feedback_position_tolerance)


def test_commands_may_stay_out_or_move_in_but_not_further_out(so101_configs):
    _, config = so101_configs
    limits, q = config.joint_limits(), JointBounds.resting(config)
    inward, outward = q.copy(), q.copy()
    inward[LIFT] += LIFT_STEP_RAD
    outward[LIFT] -= LIFT_STEP_RAD

    limits.check_command(q, q, SO101_LAYOUT)
    limits.check_command(inward, q, SO101_LAYOUT)
    with pytest.raises(ValueError, match="shoulder_lift"):
        limits.check_command(outward, q, SO101_LAYOUT)


def test_joint_move_from_outside_a_limit_travels_inward(so101_configs):
    _, config = so101_configs
    limits, start = config.joint_limits(), JointBounds.resting(config)
    further = start.copy()
    further[LIFT] -= LIFT_STEP_RAD

    assert JointMove.build(start, np.zeros(SO101_LAYOUT.vector_length), limits, SO101_LAYOUT).duration_s > 0
    with pytest.raises(ValueError):
        JointMove.build(start, further, limits, SO101_LAYOUT)


def test_other_joints_may_move_while_one_rests_past_a_limit(so101_configs):
    _, config = so101_configs
    start = JointBounds.resting(config)
    goal = start.copy()
    goal[0] += PAN_STEP_RAD

    assert JointMove.build(start, goal, config.joint_limits(), SO101_LAYOUT).duration_s > 0


@pytest.mark.parametrize("displacement", [0.150678, -0.3, 1.5])
def test_recovery_releases_the_old_target_to_the_measured_pose_within_the_release_caps(
    yam_rig, monkeypatch, displacement
):
    runtime, hardware = ManualRecovery.displaced_hold(yam_rig, monkeypatch, displacement)
    ticker = yam_rig[2]
    before, started = len(hardware.sent_targets), ticker.clock.now
    result = ManualRecovery.recover(runtime, ticker)
    commands = hardware.sent_targets[before:]
    assert result["status"] == "HOLD" and result["arrival"]["settled"]
    assert runtime.targets["right"][0] == pytest.approx(displacement)
    assert commands[-1] == runtime.targets
    assert ticker.clock.now - started >= 1.0  # fresh stability before and after transfer
    positions = np.array([0] + [q["right"][0] for q in commands])
    velocity = np.diff(positions) / runtime.control_config.control_period_s
    acceleration = np.diff(velocity) / runtime.control_config.control_period_s
    assert max(abs(velocity)) <= 0.1 + 1e-9
    assert max(abs(acceleration)) <= 0.2 + 1e-9
    assert max(abs(np.diff(acceleration) / runtime.control_config.control_period_s)) <= 1.0 + 1e-8
    assert all(q["right"][6] == 0.5 for q in commands)  # retain grasp, not measured opening
    assert runtime.trajectory.input.current_position == runtime.trajectory.input.target_position
    ticker.tick(30)
    assert runtime.status == "HOLD"  # does not immediately fault on the discarded target


@pytest.mark.parametrize("phase", ["Releasing old target", "Verifying new hold"])
@pytest.mark.parametrize("failure", ["cancel", "stop", "hardware", "drift", "write"])
def test_transfer_failures_never_report_recovered_or_continue_sending(yam_rig, monkeypatch, phase, failure):
    runtime, hardware = ManualRecovery.displaced_hold(yam_rig, monkeypatch)
    ticker, cancel = yam_rig[2], threading.Event()
    original, triggered = runtime.fault, False

    def fail(*args):
        raise RuntimeError("injected hardware failure")

    def sleep(_):
        nonlocal triggered
        request = runtime.snapshot()["recovery"]
        if not triggered and request and request["phase"] == phase:
            triggered = True
            if failure == "cancel":
                cancel.set()
            elif failure == "stop":
                runtime.stop()
            elif failure == "hardware":
                monkeypatch.setattr(hardware, "read_for_recovery", fail)
            elif failure == "write":
                # verification sends nothing; failure here models losing the reader
                method = "send" if phase == "Releasing old target" else "read_for_recovery"
                monkeypatch.setattr(hardware, method, fail)
            else:
                hardware.positions["right"] = (0.4, 0, 0, 0, 0, 0, 0.2)
        ticker.tick()

    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(RuntimeError):
        runtime.recover(cancellation=cancel)
    assert triggered and runtime.status == "FAULT" and runtime.fault == original
    count = len(hardware.sent_targets)
    ticker.tick(50)
    assert runtime.snapshot()["recovery"] is None and len(hardware.sent_targets) == count
    assert not hardware.is_closed


def test_explicit_timeout_is_not_extended_to_finish_a_transfer(yam_rig, monkeypatch):
    runtime, _ = ManualRecovery.displaced_hold(yam_rig, monkeypatch)
    ticker = yam_rig[2]
    started = ticker.clock.now
    with pytest.raises(RuntimeError, match="Recovery requires"):
        ManualRecovery.recover(runtime, ticker, timeout_s=0.8)
    assert ticker.clock.now - started < 0.85 and runtime.status == "FAULT"


def test_cancelled_transfer_can_be_retried_from_last_acknowledged_target(yam_rig, monkeypatch):
    runtime, hardware = ManualRecovery.displaced_hold(yam_rig, monkeypatch)
    ticker, cancel = yam_rig[2], threading.Event()
    before = len(hardware.sent_targets)

    def sleep(_):
        ticker.tick()
        if len(hardware.sent_targets) > before + 10:
            cancel.set()

    with monkeypatch.context() as patch:
        patch.setattr(time, "sleep", sleep)
        with pytest.raises(OperatorPaused):
            runtime.recover(cancellation=cancel)
    partial = runtime.targets["right"][0]
    assert 0 < partial < 0.3
    ManualRecovery.recover(runtime, ticker)
    assert runtime.status == "HOLD" and runtime.targets["right"][0] == pytest.approx(0.3)


def test_recovery_requires_new_window_discards_motion_and_invalidates_old_guards(yam_rig):
    runtime, hardware, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 0.5)
    ticker.tick(40)
    assert runtime.trajectory.input.current_velocity[0] > 0
    old_guard = ActionGuard(runtime)
    old_observation = {"control_snapshot": runtime.snapshot(), "joint_positions": [0] * 14}
    ManualRecovery.fault(runtime, hardware, ticker)
    targets, count = dict(runtime.targets), len(hardware.sent_targets)
    ticker.tick(50)
    assert runtime.status == "FAULT"  # healthy feedback never resets it automatically
    started, generation = ticker.clock.now, runtime.generation
    result = ManualRecovery.recover(runtime, ticker)
    assert ticker.clock.now - started >= 0.5
    assert result["status"] == "HOLD" and result["arrival"]["settled"]
    assert runtime.generation == generation + 1 and runtime.fault is None
    assert runtime.targets == targets and len(hardware.sent_targets) == count
    assert not hardware.is_closed and runtime.operation is None
    assert np.allclose(runtime.trajectory.input.current_velocity, 0)
    assert np.allclose(runtime.trajectory.input.current_acceleration, 0)
    assert runtime.trajectory.input.current_position == runtime.trajectory.input.target_position
    with pytest.raises(RuntimeError, match="cancelled"):
        old_guard.check()
    with pytest.raises(ValueError, match="invalidated"):
        runtime.verify_reference(old_observation)
    ControlHelpers.jog(runtime, "left", 1, 0.1)
    for _ in range(500):
        ticker.tick()
        if runtime.status == "HOLD":
            break
    assert runtime.status == "HOLD" and len(hardware.sent_targets) > count


@pytest.mark.parametrize(
    "failure",
    [
        "bus",
        "stale",
        "missing",
        "nan",
        "bounds",
        "frozen",
        "moving",
        "oscillating",
        "position_span",
    ],
)
def test_unhealthy_feedback_never_recovers_or_sends(yam_rig, monkeypatch, failure):
    runtime, hardware, ticker = yam_rig
    ManualRecovery.fault(runtime, hardware, ticker)
    original, count = runtime.fault, len(hardware.sent_targets)
    read = hardware.read

    def feedback():
        states = read()
        left = states["left"]
        if failure == "missing":
            del states["right"]
        elif failure in ("nan", "bounds"):
            value = {"nan": float("nan"), "bounds": 100}[failure]
            states["left"] = replace(left, position=(value,) + left.position[1:])
        elif failure == "frozen":
            # even fresh-looking repeated sample identities cannot fill the window
            states = {s: replace(v, sample_id=1) for s, v in states.items()}
        elif failure == "moving":
            states["left"] = replace(left, velocity=(0.2,) + (0,) * 6)
        elif failure == "oscillating":
            velocity = (
                0.2
                if round((ticker.clock.now - StandInClock.START_S) / runtime.control_config.control_period_s) % 2
                else -0.2
            )
            states["left"] = replace(left, velocity=(velocity,) + (0,) * 6)
        elif failure == "position_span":
            position = (
                0.04
                if round((ticker.clock.now - StandInClock.START_S) / runtime.control_config.control_period_s) % 2
                else -0.04
            )
            states["left"] = replace(left, position=(position,) + left.position[1:])
        return states

    hardware.is_stale = failure == "stale"
    hardware.failed_arm = "right" if failure == "bus" else None
    monkeypatch.setattr(hardware, "read", feedback)
    with pytest.raises((RuntimeError, ValueError)):
        ManualRecovery.recover(runtime, ticker, timeout_s=0.8)
    assert runtime.status == "FAULT" and runtime.fault == original
    assert len(hardware.sent_targets) == count and runtime.snapshot()["recovery"] is None
    assert not hardware.is_closed


@pytest.mark.parametrize("interrupt", ["cancel", "stop", "close"])
def test_recovery_can_be_interrupted_without_motion(yam_rig, monkeypatch, interrupt):
    runtime, hardware, ticker = yam_rig
    ManualRecovery.fault(runtime, hardware, ticker)
    before = len(hardware.sent_targets)
    cancel = threading.Event()

    def sleep(_):
        ticker.tick()
        if interrupt == "cancel":
            cancel.set()
        elif interrupt == "stop":
            runtime.stop()
        else:
            runtime.close()

    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises((OperatorPaused, RuntimeError)):
        runtime.recover(cancellation=cancel)
    assert runtime.status == ("CLOSED" if interrupt == "close" else "FAULT")
    assert runtime.snapshot()["recovery"] is None and len(hardware.sent_targets) == before


def test_recovery_only_accepts_fault_and_refuses_concurrent_checks(yam_rig, monkeypatch):
    runtime, hardware, ticker = yam_rig
    with pytest.raises(RuntimeError, match="requires FAULT"):
        ManualRecovery.recover(runtime, ticker)
    ManualRecovery.fault(runtime, hardware, ticker)

    def sleep(_):
        with pytest.raises(RuntimeError, match="already running"):
            ManualRecovery.recover(runtime, ticker)
        ticker.tick()

    monkeypatch.setattr(time, "sleep", sleep)
    runtime.recover()
    assert runtime.status == "HOLD"


def test_static_tracking_offset_is_reported_without_fault_or_target_rewrite(yam_station_rig, monkeypatch):
    runtime, hardware, _, tick = yam_station_rig
    send = hardware.send

    def biased(targets, velocities=None):
        send(targets, velocities)
        q = list(hardware.positions["right"])
        q[4] -= 0.031234
        hardware.positions["right"] = tuple(q)

    monkeypatch.setattr(hardware, "send", biased)
    ControlHelpers.jog(runtime, "right", 1, 0.05)
    result = YAMStationRig.finish(runtime, tick)
    assert result["status"] == "HOLD" and result["fault"] is None
    assert result["arrival"]["settled"] and not result["arrival"]["reached"]
    assert result["arrival"]["signed_error"][11] == pytest.approx(-0.031234)
    assert result["targets"]["right"][4] == 0
    # an excessive deviation stays a latched fault even in HOLD
    hardware.positions["right"] = (1.0,) + hardware.positions["right"][1:]
    tick()
    assert runtime.status == "FAULT"


def test_complete_sequence_never_stops_between_rows_and_lags_within_dynamics(yam_station_rig):
    runtime, _, _, tick = yam_station_rig
    runtime.start_sequence(YAMStationRig.chunk(runtime, 0.0, 0.2, count=20), rate_hz=25)
    operation = runtime.operation
    statuses, lag = [], 0.0
    for _ in range(1000):
        tick()
        statuses.append(runtime.status)
        trajectory_input = runtime.trajectory.input
        lag = max(lag, abs(trajectory_input.target_position[7] - trajectory_input.current_position[7]))
        if runtime.status in ("HOLD", "FAULT"):
            break
    result = runtime.snapshot()
    assert runtime.status == "HOLD", runtime.fault
    assert "STOPPING" not in statuses
    assert operation.index == 20
    assert result["motion"]["end_reason"] == "finished"
    assert lag > 0.01
    assert result["arrival"]["reached"]
    assert result["arms"]["right"]["position"][0] == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["stop", "fault", "deadline"])
def test_interrupted_sequence_discards_every_unconsumed_action(yam_station_rig, mode):
    runtime, hardware, clock, tick = yam_station_rig
    runtime.start_sequence(YAMStationRig.chunk(runtime, 0, 0.2, count=30), rate_hz=25)
    operation = runtime.operation
    tick(7)
    consumed = operation.index
    if mode == "stop":
        runtime.stop()
    elif mode == "fault":
        hardware.failed_arm = "left"
    else:
        clock.now += 0.2
    result = YAMStationRig.finish(runtime, tick)
    assert runtime.operation is None
    assert operation.index == consumed
    assert result["status"] == ("FAULT" if mode == "fault" else "HOLD")
    assert (
        result["motion"]["end_reason"] == {"stop": "stopped", "fault": "fault", "deadline": "sequence_deadline"}[mode]
    )
    assert result["targets"]["right"][0] < 0.1


@pytest.mark.parametrize("bad", ["bounds", "nan", "too_long", "too_slow", "too_fast"])
def test_invalid_complete_sequence_is_atomic(yam_station_rig, bad):
    runtime, hardware, _, tick = yam_station_rig
    values = YAMStationRig.chunk(runtime, 0, 0.04, count=1001 if bad == "too_long" else 10)
    if bad in ("bounds", "nan"):
        values[-1, 0] = 20 if bad == "bounds" else float("nan")
    rate = {"too_slow": 1, "too_fast": 51}.get(bad, 30)
    count = len(hardware.sent_targets)
    with pytest.raises(ValueError):
        runtime.start_sequence(values, rate_hz=rate)
    assert runtime.operation is None and runtime.status == "HOLD"
    assert len(hardware.sent_targets) == count
    good = YAMStationRig.chunk(runtime, 0, 0.04)
    runtime.start_sequence(good, rate_hz=30)
    operation = runtime.operation
    good[:] = 100
    assert not operation.positions.flags.writeable
    with pytest.raises(RuntimeError):
        runtime.start_sequence(YAMStationRig.chunk(runtime, 0, 0.5), rate_hz=30)
    YAMStationRig.finish(runtime, tick)
    assert operation.index == 10
    assert hardware.positions["right"][0] == pytest.approx(0.04)


def test_task_accuracy_cannot_relax_measured_joint_bounds():
    control_config, config = ConfigFiles.load_configs("yam")
    config.position_tolerance = 0.1
    q = (0, config.arms["left"].limits.lower_bounds[1] - 0.03, 0, 0, 0, 0, 0.5)
    states = {s: YAMArmState(q, (0.0,) * 7, 100) for s in ("left", "right")}
    with pytest.raises(ValueError, match="bounds"):
        ControlHelpers.check_states(states, control_config, config, 100)


def test_reached_refers_to_requested_goal_and_stop_is_not_completion(yam_station_rig):
    runtime, _, _, tick = yam_station_rig
    assert runtime.snapshot()["arrival"]["reached"]
    # leave enough distance that braking cannot happen to reach the goal
    ControlHelpers.jog(runtime, "right", 1, 1.5)
    tick(20)
    moving = runtime.snapshot()
    assert not moving["arrival"]["reached"]
    assert runtime.trajectory.input.target_position[7] == 1.5
    runtime.stop()
    result = YAMStationRig.finish(runtime, tick)
    assert result["arrival"]["settled"] and not result["arrival"]["reached"]
    assert result["motion"]["end_reason"] == "stopped"


def test_wait_cannot_complete_against_a_replacement_operation(yam_station_rig, monkeypatch):
    runtime, _, _, tick = yam_station_rig
    ControlHelpers.jog(runtime, "right", 1, 0.05)

    def replace_operation(_):
        runtime.stop()
        YAMStationRig.finish(runtime, tick)
        ControlHelpers.jog(runtime, "left", 1, 0.02)

    monkeypatch.setattr(time, "sleep", replace_operation)
    with pytest.raises(RuntimeError, match="cancelled"):
        runtime.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
    assert runtime.snapshot()["motion"]["end_reason"] != "finished"


def test_stale_cached_hold_is_not_ready(yam_station_rig):
    runtime, _, clock, tick = yam_station_rig
    tick(30)
    assert runtime.snapshot()["arrival"]["settled"]
    clock.now += runtime.control_config.state_timeout_s + 0.01
    snapshot = runtime.snapshot()
    assert not snapshot["arrival"]["settled"]
    with pytest.raises(RuntimeError, match="Stale"):
        SnapshotCheck.validate_snapshot(snapshot, runtime.control_config, requires_hold=True)


def test_inactive_joint_position_change_during_settling_keeps_move(yam_station_rig):
    runtime, hardware, _, tick = yam_station_rig
    start = np.asarray(runtime.trajectory.input.current_position)
    goal = start.copy()
    goal[7] += 0.05
    runtime.dispatch_joint_move(JointMove.build(start, goal, runtime.limits, LAYOUT))
    for _ in range(200):
        tick()
        if runtime.snapshot()["motion"]["settling"]:
            break
    assert runtime.status == "RUNNING" and runtime.snapshot()["motion"]["settling"]
    hardware.positions["left"] = (0.006,) + hardware.positions["left"][1:]
    tick()
    assert runtime.status == "RUNNING" and runtime.operation is not None
    result = YAMStationRig.finish(runtime, tick)
    assert result["status"] == "HOLD" and result["motion"]["end_reason"] == "finished"
    assert runtime.operation is None


@pytest.mark.parametrize("trigger", ["deadline", "cancel"])
def test_failed_controlled_stop_never_sends_another_motion_command(yam_station_rig, monkeypatch, trigger):
    runtime, hardware, clock, tick = yam_station_rig
    start = np.asarray(runtime.trajectory.input.current_position)
    goal = start.copy()
    goal[7] += 0.05
    move = JointMove.build(start, goal, runtime.limits, LAYOUT)
    runtime.dispatch_joint_move(move)
    tick(10)

    def reject_stop():
        raise ValueError("No valid stop trajectory")

    monkeypatch.setattr(runtime.trajectory, "compute_trajectory", reject_stop)
    if trigger == "deadline":
        # past the move's deadline: its duration, the settle timeout and the margin from its dispatch
        clock.now += move.duration_s + runtime.control_config.settle_timeout_s + runtime.MOVE_MARGIN_S
    else:
        runtime.operation_cancellation = threading.Event()
        runtime.operation_cancellation.set()
    sent = len(hardware.sent_targets)
    held = dict(runtime.targets)
    tick()
    assert runtime.status == "FAULT" and "No valid stop" in runtime.fault
    assert len(hardware.sent_targets) == sent + 1  # only the unchanged halt target
    assert hardware.sent_targets[-1] == held
    tick(3)
    assert len(hardware.sent_targets) == sent + 1


def test_control_ticks_do_not_build_external_reports(yam_station_rig, monkeypatch):
    runtime, _, _, tick = yam_station_rig

    def forbidden():
        raise AssertionError("The control tick built an external status report")

    monkeypatch.setattr(runtime, "snapshot", forbidden)
    tick(30)
    assert YAMControlLoop.snapshot(runtime)["arrival"]["settled"]
    ControlHelpers.jog(runtime, "right", 1, 0.01)
    for _ in range(100):
        tick()
        if runtime.status != "RUNNING":
            break
    assert runtime.status == "HOLD", runtime.fault
    assert YAMControlLoop.snapshot(runtime)["arrival"]["reached"]
    ControlHelpers.jog(runtime, "right", 1, 0.3)
    tick(10)
    runtime.stop()
    for _ in range(100):
        tick()
        if runtime.status != "STOPPING":
            break
    assert runtime.status == "HOLD", runtime.fault


def test_hold_timeout_is_controller_owned_and_observers_cannot_restart_it(yam_station_rig, monkeypatch):
    runtime, hardware, clock, tick = yam_station_rig
    tick(30)
    read = hardware.read

    def moving_feedback():
        states = read()
        states["left"] = replace(states["left"], velocity=(0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0))
        return states

    monkeypatch.setattr(hardware, "read", moving_feedback)
    unstable_at = None
    for _ in range(250):
        tick()
        if runtime.status == "FAULT":
            break
        for _ in range(3):
            state = runtime.snapshot()
            runtime.stability()
        if not state["arrival"]["settled"] and unstable_at is None:
            unstable_at = clock.now
    assert runtime.status == "FAULT" and "Persistent motion during HOLD" in runtime.fault
    assert "left J4" in runtime.fault
    assert (
        runtime.control_config.settle_timeout_s
        < clock.now - unstable_at
        <= runtime.control_config.settle_timeout_s + 0.021
    )


@pytest.mark.parametrize("failure", ["stale", "missing"])
def test_hardware_fault_cannot_be_recovered_or_restart_automatically(gripper_rigs, failure):
    rig = gripper_rigs.make()
    driver_flag = {"stale": "is_stale", "missing": "lacks_effort"}[failure]
    rig.begin({"right": False})
    for _ in range(10):
        rig.tick()
    setattr(rig.driver, driver_flag, True)
    rig.tick()
    assert rig.control_loop.status == "FAULT"
    held = dict(rig.control_loop.targets)
    before = len(rig.driver.commands)
    assert "hold_recovery" not in rig.control_loop.snapshot()
    assert not hasattr(rig.control_loop, "begin_hold_recovery")
    setattr(rig.driver, driver_flag, False)
    for _ in range(100):
        rig.tick()
    assert rig.control_loop.status == "FAULT"
    assert rig.control_loop.targets == held and len(rig.driver.commands) == before


def test_no_file_io_on_controller_clock(gripper_rigs, monkeypatch):
    rig = gripper_rigs.make()
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("control thread file IO"))
    for _ in range(250):
        rig.clock.now += 0.02
        rig.control_loop.step()
    assert rig.control_loop.status == "HOLD"


def test_bimanual_motion_and_derivative_bounds(yam_rig):
    runtime, hardware, ticker = yam_rig
    targets = {s: list(q) for s, q in runtime.targets.items()}
    targets["left"][0], targets["right"][1] = 0.2, -0.15
    runtime.move(targets)
    velocity, acceleration = [], []
    for _ in range(1000):
        ticker.tick()
        velocity.append(runtime.trajectory.output.new_velocity)
        acceleration.append(runtime.trajectory.output.new_acceleration)
        if runtime.status != "RUNNING":
            break
    trajectory_input = runtime.trajectory.input
    assert runtime.status == "HOLD", runtime.fault
    assert np.allclose(hardware.positions["left"], targets["left"])
    assert np.allclose(hardware.positions["right"], targets["right"])
    assert np.all(np.max(abs(np.array(velocity)), axis=0) <= np.array(trajectory_input.max_velocity) + 1e-9)
    assert np.all(np.max(abs(np.array(acceleration)), axis=0) <= np.array(trajectory_input.max_acceleration) + 1e-9)
    assert np.all(
        np.max(abs(np.diff(acceleration, axis=0)) / runtime.control_config.control_period_s, axis=0)
        <= np.array(trajectory_input.max_jerk) + 1e-8
    )


def test_moves_feed_planned_arm_velocity_forward_and_holds_send_none(yam_rig):
    runtime, hardware, ticker = yam_rig
    assert hardware.velocities is None  # startup hold
    targets = {s: list(q) for s, q in runtime.targets.items()}
    targets["right"][1] = 0.3
    runtime.move(targets)
    positions, velocities = [], []
    for _ in range(1000):
        ticker.tick()
        positions.append(hardware.positions["right"][1])
        velocities.append(hardware.velocities["right"][1])
        if runtime.status != "RUNNING":
            break
    assert runtime.status == "HOLD", runtime.fault
    assert max(velocities) > 0.1 and min(velocities) >= 0
    # the velocity sent each tick is the derivative of the positions sent
    finite_difference = np.diff(positions) / runtime.control_config.control_period_s
    midpoint = (np.array(velocities[1:]) + np.array(velocities[:-1])) / 2
    assert np.allclose(finite_difference, midpoint, atol=0.01)
    assert hardware.velocities["right"][1] == pytest.approx(0.0)  # plan ends at rest


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 100])
def test_rejected_target_never_reaches_hardware(yam_rig, bad):
    runtime, hardware, _ = yam_rig
    count = len(hardware.sent_targets)
    with pytest.raises(ValueError):
        ControlHelpers.jog(runtime, "left", 1, bad)
    assert len(hardware.sent_targets) == count
    assert runtime.status == "HOLD"


def test_second_move_does_not_replace_active_plan(yam_rig):
    runtime, _, _ = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 0.2)
    with pytest.raises(RuntimeError):
        ControlHelpers.jog(runtime, "right", 1, 0.2)


def test_stop_decelerates_continuously_and_holds_before_goal(yam_rig):
    runtime, hardware, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 1.0)
    ticker.tick(40)
    initial_q = hardware.positions["left"][0]
    initial_v = runtime.trajectory.input.current_velocity[0]
    assert initial_v > 0
    runtime.stop()
    velocities = [initial_v]
    for _ in range(500):
        ticker.tick()
        velocities.append(runtime.trajectory.output.new_velocity[0])
        if runtime.status != "STOPPING":
            break
    assert runtime.status == "HOLD", runtime.fault
    assert initial_q < hardware.positions["left"][0] < 1.0
    assert abs(velocities[-1]) < 1e-8
    assert max(abs(np.diff(velocities))) <= 0.5 * runtime.control_config.control_period_s + 1e-8


@pytest.mark.parametrize("gap", [0.02, 0.22])
@pytest.mark.parametrize("fault", ["stale", "right_bus", "tracking"])
def test_fault_latches_and_halts_both_arms(yam_rig, fault, gap):
    runtime, hardware, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 0.3)
    ticker.tick(10)
    if fault == "stale":
        hardware.is_stale = True
    elif fault == "right_bus":
        hardware.failed_arm = "right"
    elif fault == "tracking":
        hardware.positions["right"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    ticker.clock.now += gap
    runtime.step()
    assert runtime.status == "FAULT"
    assert runtime.fault
    assert hardware.sent_targets[-1] == runtime.targets  # one hold command for both arms
    sent = len(hardware.sent_targets)
    hardware.is_stale, hardware.failed_arm = False, None
    ticker.tick(5)
    assert runtime.status == "FAULT" and len(hardware.sent_targets) == sent
    with pytest.raises(RuntimeError):
        ControlHelpers.jog(runtime, "right", 1, 0.1)


@pytest.mark.parametrize("gap", [0.084, 0.22])
@pytest.mark.parametrize("status", ["HOLD", "RUNNING", "STOPPING"])
def test_delayed_tick_with_fresh_feedback_keeps_current_operation(yam_rig, status, gap):
    runtime, hardware, ticker = yam_rig
    if status != "HOLD":
        ControlHelpers.jog(runtime, "left", 1, 0.3)
    ticker.tick(10)
    if status == "STOPPING":
        runtime.stop()
    before = dict(runtime.targets)
    sent = len(hardware.sent_targets)
    operation, generation = runtime.operation, runtime.generation

    ticker.clock.now += gap
    runtime.step()

    assert runtime.status == status, runtime.fault
    assert runtime.fault is None
    assert runtime.generation == generation
    assert runtime.operation is operation
    assert all(state.sampled_monotonic_s == ticker.clock.now for state in runtime.states.values())
    if status == "HOLD":
        assert runtime.targets == before
        assert len(hardware.sent_targets) == sent
    else:
        assert len(hardware.sent_targets) == sent + 1


def test_partial_write_fault_attempts_hold_on_both_arms(yam_rig, monkeypatch):
    runtime, hardware, ticker = yam_rig
    previous = hardware.positions.copy()
    original = hardware.send
    attempts = 0

    def send(targets, velocities=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            hardware.positions["left"] = targets["left"]
            raise RuntimeError("right arm write failed")
        return original(targets, velocities)

    monkeypatch.setattr(hardware, "send", send)
    ControlHelpers.jog(runtime, "left", 1, 0.3)
    ticker.tick()
    assert runtime.status == "FAULT"
    assert attempts == 2
    assert hardware.positions == previous


def test_feedback_tolerance_does_not_relax_action_bounds():
    control_config, config = ConfigFiles.load_configs("yam")
    q = (0, -0.0005, 0, 0, 0, 0, 1.003)
    states = {side: YAMArmState(q, (0,) * 7, 100) for side in ("left", "right")}
    ControlHelpers.check_states(states, control_config, config, 100)
    with pytest.raises(ValueError, match="outside"):
        config.joint_limits().check_positions(LAYOUT.join({side: q for side in states}), LAYOUT)
    states["left"] = YAMArmState(q[:-1] + (1.1,), (0,) * 7, 100)
    with pytest.raises(ValueError, match="outside"):
        ControlHelpers.check_states(states, control_config, config, 100)


def test_bounds_error_identifies_axis_value_and_unchanged_allowance():
    _, config = ConfigFiles.load_configs("yam")
    targets = {side: [0.0] * 6 + [0.5] for side in ("left", "right")}
    targets["left"][2] = -0.15
    targets["left"][6] = 1.237412
    with pytest.raises(ValueError) as exc:
        config.joint_limits().check_positions(LAYOUT.join(targets), LAYOUT, tolerance=0.02)
    message = str(exc.value)
    assert "left:" in message and "joint 3=-0.150000 rad" in message
    assert "gripper=1.237412 opening" in message
    assert "bounds=[0.000000, 1.000000]" in message
    assert "feedback allowance=0.020000" in message


@pytest.mark.parametrize("settles", [True, False])
def test_enable_waits_for_stable_feedback_but_times_out_on_persistent_motion(stand_in_clock, monkeypatch, settles):
    clock = stand_in_clock
    control_config, config = ConfigFiles.load_configs(
        "yam",
        *ConfigFiles.WIDE_LIMITS["yam"],
        "control.settle_timeout_s=0.2",
        "control.settle_window_s=0.1",
        "robot.gripper_contact_confirm_s=0.1",
        "robot.velocity_tolerance=0.03",
    )
    velocities = iter([0.2, 0, 0, 0.1] + [0] * 20 if settles else [0.2] * 30)
    reads = []

    def read():
        clock.now += 1 / control_config.rate_hz  # one controller period per startup read
        velocity = next(velocities)
        reads.append(velocity)
        state = YAMArmState((0,) * 6 + (0.5,), (velocity,) + (0,) * 6, clock.now)
        return {s: state for s in ("left", "right")}

    hardware = SimpleNamespace(read=read, send=lambda targets: None, refresh_hold=lambda targets: None, telemetry=dict)
    # the reads advance the clock: the waits between them do not, as the real sleeps of the startup did not
    monkeypatch.setattr(time, "sleep", lambda _: None)
    if settles:
        runtime = YAMControlLoop(hardware, control_config, config)
        assert len(reads) >= 6  # a complete window, not a few instantaneous samples
        assert runtime.states["left"].velocity == (0,) * 7
        assert runtime.snapshot()["arrival"]["settled"]
    else:
        with pytest.raises(RuntimeError, match="startup timed out"):
            YAMControlLoop(hardware, control_config, config)


@pytest.mark.parametrize("failure", ["timeout", "fault", "interrupt"])
def test_wait_cannot_report_success_after_timeout_fault_or_interrupt(yam_rig, monkeypatch, failure):
    runtime, hardware, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 0.3)

    def sleep(dt):
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "fault":
            hardware.failed_arm = "right"
        ticker.tick()

    error = {"timeout": TimeoutError, "fault": RuntimeError, "interrupt": KeyboardInterrupt}
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(error[failure]):
        runtime.wait_for_hold(0.1)
    assert runtime.status == ("FAULT" if failure == "fault" else "STOPPING")


def test_wait_rejects_stale_hold_feedback(yam_rig):
    runtime, _, ticker = yam_rig
    ticker.clock.now += runtime.control_config.state_timeout_s + 0.01
    with pytest.raises(RuntimeError, match="[Ss]tale"):
        runtime.wait_for_hold(JointControlLoop.HOLD_WAIT_S)


def test_shutdown_waits_for_stop_and_releases_once(yam_rig, monkeypatch):
    runtime, hardware, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 1.0)
    ticker.tick(40)
    closed = []
    close = hardware.close

    def release():
        closed.append(runtime.status)
        assert max(abs(v) for v in runtime.trajectory.input.current_velocity) < 1e-8
        close()

    monkeypatch.setattr(hardware, "close", release)
    ticker.clock.control_loop = runtime
    runtime.shutdown()
    ticker.clock.control_loop = None
    runtime.shutdown()
    assert closed == ["HOLD"]
    assert runtime.status == "CLOSED"
    assert hardware.is_closed
    assert hardware.positions["left"][0] < 1.0


def test_repeated_stop_does_not_restart_settling(yam_rig):
    runtime, _, ticker = yam_rig
    ControlHelpers.jog(runtime, "left", 1, 1.0)
    ticker.tick(40)
    for _ in range(200):
        runtime.stop()
        ticker.tick()
        if runtime.status == "HOLD":
            break
    assert runtime.status == "HOLD"


@pytest.mark.parametrize("rate", [30, 50])
def test_long_finite_sequence_consumes_every_row_once_despite_jitter(yam_station_rig, rate, monkeypatch):
    runtime, hardware, clock, _ = yam_station_rig
    start_s = clock.now  # the scenario's start: the control loop waited for settled feedback when it was made
    rows = np.tile(runtime.trajectory.input.current_position, (1000, 1))
    rows[:, 0] = np.linspace(0, 0.1, 1000)
    accepted = []
    original = runtime.trajectory.set_target

    def record(targets, **kwargs):
        accepted.append(list(targets))
        return original(targets, **kwargs)

    monkeypatch.setattr(runtime.trajectory, "set_target", record)
    runtime.start_sequence(rows, rate_hz=rate)
    for tick in range(1800):
        # jitter of a fraction of a period around each tick
        jitter = (0.0005, 0.0001, 0.0006)[tick % 3]
        clock.now = start_s + (tick + 1) * 0.02 + jitter
        before = len(accepted)
        runtime.step()
        assert len(accepted) - before <= 1
        assert runtime.status not in ("FAULT", "STOPPING"), runtime.fault
        if runtime.status == "HOLD":
            break
    assert np.array(accepted) == pytest.approx(rows)
    assert runtime.snapshot()["motion"]["end_reason"] == "finished"


def test_sustained_sequence_backlog_still_requests_controlled_stop(yam_station_rig):
    runtime, hardware, clock, _ = yam_station_rig
    rows = np.tile(runtime.trajectory.input.current_position, (1000, 1))
    runtime.start_sequence(rows, rate_hz=50)
    operation = runtime.operation
    for _ in range(30):
        clock.now += 0.025
        runtime.step()
        if runtime.status != "RUNNING":
            break
    assert runtime.snapshot()["motion"]["end_reason"] == "sequence_deadline"
    assert operation.index < 30
    assert runtime.operation is None


def test_holds_measured_outside_limit(startup_rig):
    runtime, hardware, _, resting = startup_rig

    assert hardware.sent_targets[0]["left"] == resting
    assert runtime.targets["left"] == resting


def test_only_inward_motion_allowed(startup_rig):
    runtime, hardware, ticker, resting = startup_rig
    outward, inward = list(resting), list(resting)
    outward[J4] -= OUTSIDE_RAD
    inward[J4] += 10 * OUTSIDE_RAD

    with pytest.raises(ValueError, match="left: joint or gripper target outside commissioned bounds: joint 4="):
        runtime.move({**runtime.targets, "left": outward})
    runtime.move({**runtime.targets, "left": inward})
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert hardware.positions["left"][J4] == pytest.approx(inward[J4])


def test_release_recovery_moves_only_inward_to_the_limit(startup_rig):
    runtime, hardware, ticker, resting = startup_rig
    lower = runtime.limits.lower_bounds[J4]
    hardware.is_stale = True
    ticker.tick()
    assert runtime.status == "FAULT"
    hardware.is_stale = False
    sent = len(hardware.sent_targets)

    result = JointRuntimeRig.recover(runtime, ticker)

    released = [command["left"][J4] for command in hardware.sent_targets[sent:]]
    assert result["status"] == "HOLD", runtime.fault
    assert released and min(released) >= resting[J4] - runtime.limits.EPSILON
    assert runtime.targets["left"][J4] == pytest.approx(lower)


@pytest.mark.parametrize("outside", ["joint 4 below its lower limit", "gripper past fully open"])
def test_recovery_after_a_fault_reaches_a_hold(held_past_a_limit_rig):
    runtime, hardware, ticker = held_past_a_limit_rig
    hardware.is_stale = True
    ticker.tick()
    assert runtime.status == "FAULT"
    hardware.is_stale = False

    result = JointRuntimeRig.recover(runtime, ticker)

    assert result["status"] == "HOLD", runtime.fault


@pytest.mark.parametrize("readiness", [True, False, "error"])
def test_move_readiness_and_dispatch_are_atomic_with_concurrent_stop(stand_in_clock, monkeypatch, readiness):
    station = SteppedStandInYAMRobot(
        ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), action_mode=ActionMode.JOINT
    ).open()
    device = station.control_loop.driver
    if readiness is False:
        # one period of motion leaves the HOLD unsettled
        read = device.read
        with monkeypatch.context() as patch:
            patch.setattr(
                device,
                "read",
                lambda: {arm: replace(state, velocity=(2.0,) + state.velocity[1:]) for arm, state in read().items()},
            )
            Ticker(station.control_loop, stand_in_clock).tick()
    start = SnapshotCheck.target_vector(station.control_loop.snapshot())
    observation = {"joint_positions": start.copy(), "control_snapshot": station.control_loop.snapshot()}
    goal = start.copy()
    goal[7] += 0.03
    move = JointMove.build(start, goal, station.robot_config.joint_limits(), LAYOUT)
    attempted, stopped = threading.Event(), threading.Event()

    def stop():
        attempted.set()
        station.stop()
        stopped.set()

    contender = threading.Thread(target=stop, daemon=True)

    verify = station.control_loop.verify_reference

    def ready(observation):
        contender.start()
        assert attempted.wait(1)
        assert not stopped.wait(0.02), "A competing command ran during readiness checking"
        if readiness == "error":
            raise RuntimeError("observation changed")
        verify(observation)

    station.control_loop.verify_reference = ready
    try:
        if readiness == "error":
            with pytest.raises(RuntimeError, match="observation changed"):
                station.control_loop.dispatch_joint_move(move, observation)
        else:
            assert station.control_loop.dispatch_joint_move(move, observation) is readiness
        assert stopped.wait(1)
        # a stop arriving during readiness must see the dispatched move, or an untouched HOLD when readiness fails;
        # no hardware tick has run here
        assert station.control_loop.snapshot()["status"] == ("STOPPING" if readiness is True else "HOLD")
        assert len(device.sent_targets) == 1
        assert SnapshotCheck.target_vector(station.control_loop.snapshot()) == pytest.approx(start)
    finally:
        if contender.ident is not None:
            contender.join(timeout=1)
        station.control_loop.close()


@pytest.mark.parametrize("tolerance,reached", [(0.005, False), (0.02, True)])
def test_recorded_static_error_is_reported_without_turning_into_a_stability_fault(replay, tolerance, reached):
    station, clock, recorded = replay
    station.robot_config.position_tolerance = tolerance
    station.wait_stationary()
    result = station.control_loop.snapshot()
    assert result["arrival"]["settled"] and result["arrival"]["reached"] is reached
    assert 0.5 <= clock.now <= 0.54
    if not reached:
        limits = MotionHelpers.tracking(station.robot_config).arrival_tolerance
        violations = MotionHelpers.tracking(station.robot_config).metrics(*MotionHelpers.vectors(result), limits)[
            "violations"
        ]
        assert [v["joint"] for v in violations] == ["J2", "J3"]
        assert [v["error"] for v in violations] == pytest.approx([0.008579997, 0.018927828])
    limits = MotionHelpers.tracking(station.robot_config).arrival_tolerance
    assert MotionHelpers.tracking(station.robot_config).metrics(
        *MotionHelpers.vectors(result), limits
    ) == MotionHelpers.tracking(station.robot_config).metrics(*MotionHelpers.vectors(recorded), limits)
    json.dumps(result)


def test_gripper_error_is_reported_without_an_application_tracking_fault(replay):
    station, _, recorded = replay
    recorded["arms"]["right"]["position"][6] -= 0.011
    result = station.hold_snapshot()
    limits = MotionHelpers.tracking(station.robot_config).arrival_tolerance
    violations = MotionHelpers.tracking(station.robot_config).metrics(*MotionHelpers.vectors(result), limits)[
        "violations"
    ]
    assert any(v["side"] == "right" and v["joint"] == "gripper" for v in violations)
    MotionHelpers.tracking(station.robot_config).check(*MotionHelpers.vectors(result))  # the SDK owns gripper tracking
    # removing gripper regulation from the app must leave arm tracking intact
    recorded["arms"]["right"]["position"][0] += 0.2
    with pytest.raises(TrackingError, match="J1"):
        MotionHelpers.tracking(station.robot_config).check(*MotionHelpers.vectors(station.control_loop.snapshot()))


@pytest.mark.parametrize("case", json.loads((ConfigFiles.YAM_RECORDED_CASES / ENDPOINT_TARGETS).read_text())["cases"])
def test_recorded_endpoints_obey_native_limits(case):
    _, config = ConfigFiles.load_configs("yam")
    slow = JointMove.build(case["start"], case["goal"], config.joint_limits(), LAYOUT, speed_scale=1 / 3)
    move = JointMove.build(case["start"], case["goal"], config.joint_limits(), LAYOUT)
    assert slow.duration_s == pytest.approx(move.duration_s * 3, rel=1e-8)
    JointMoves.check_bounds(move)


def test_a_joint_move_equals_the_ruckig_trajectory_before_and_after_pickling():
    _, config = ConfigFiles.load_configs("yam")
    move = JointMoves.fixture_move(config)
    inp, native = InputParameter(14), Trajectory(14)
    inp.current_position, inp.target_position = move.start, move.goal
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = move.motion_caps
    inp.synchronization = Synchronization.Phase
    assert Ruckig(14).calculate(inp, native) >= 0
    restored = pickle.loads(pickle.dumps(move))
    assert restored.duration_s == native.duration == move.duration_s
    for t in np.linspace(0, move.duration_s, 101):
        assert np.asarray(restored.at_time(t)) == pytest.approx(np.asarray(native.at_time(t)), abs=1e-12)
    JointMoves.check_bounds(restored)


def test_phase_sync_holds_inactive_joints_and_keeps_dual_arm_joint_progress():
    _, config = ConfigFiles.load_configs("yam")
    start = np.zeros(14)
    start[[6, 13]] = [0.3, 0.8]
    goal = start.copy()
    goal[[0, 2, 7, 11]] = [0.4, 0.2, -0.5, 0.3]
    move = JointMove.build(start, goal, config.joint_limits(), LAYOUT)
    active = goal != start
    for t in np.linspace(0, move.duration_s, 101):
        q = move.at_time(t)[0]
        assert q[~active] == pytest.approx(start[~active], abs=1e-14)
        progress = (q[active] - start[active]) / (goal[active] - start[active])
        assert progress == pytest.approx(np.full(len(progress), progress[0]), abs=1e-12)
    JointMoves.check_bounds(move)


@pytest.mark.parametrize("delta", [0.0, 1e-12, 1e-8])
def test_noop_and_tiny_motion_are_finite(delta):
    _, config = ConfigFiles.load_configs("yam")
    start = np.zeros(14)
    start[[6, 13]] = 0.5
    goal = start.copy()
    goal[7] += delta
    move = JointMove.build(start, goal, config.joint_limits(), LAYOUT)
    assert np.isfinite(move.duration_s) and move.duration_s >= 0
    JointMoves.check_bounds(move)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 100])
def test_an_invalid_goal_cannot_be_built_into_a_move(bad):
    _, config = ConfigFiles.load_configs("yam")
    start = np.zeros(14)
    goal = start.copy()
    goal[7] = bad
    with pytest.raises(ValueError):
        JointMove.build(start, goal, config.joint_limits(), LAYOUT)


def test_per_move_caps_respect_stricter_station_joints_and_time_scaling():
    _, config = ConfigFiles.load_configs("yam")
    config.arms["right"].limits.velocity[2] = 0.02
    config.arms["left"].limits.acceleration[6] = 0.03
    config.arms["right"].limits.jerk[5] = 0.01
    actual = np.array(config.joint_limits().motion_caps(0.5, caps=config.home.limits))
    assert actual[0, 9] == 0.01
    assert actual[1, 6] == 0.03 / 4
    assert actual[2, 12] == 0.01 / 8
    assert actual[0, 2] == 0.15
    assert actual[1, 13] == pytest.approx((0.6 / 9) / 4)
    assert actual[2, 5] == pytest.approx(2 / 8)
    assert np.all(actual <= config.joint_limits().motion_caps(0.5))


def test_native_extrema_reject_interior_overshoot_even_with_legal_endpoints():
    control_config, config = ConfigFiles.load_configs("yam")
    trajectory = JointTrajectory(config.joint_limits(), LAYOUT, control_config)
    position = np.zeros(14)
    position[0] = config.arms["left"].limits.upper_bounds[0] - 0.001
    trajectory.input.current_position = position.tolist()
    trajectory.input.current_velocity = [0.4] + [0.0] * 13
    with pytest.raises(ValueError, match="position limit"):
        trajectory.set_target(position)
