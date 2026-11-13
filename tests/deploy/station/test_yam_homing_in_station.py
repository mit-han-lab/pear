import json
import math
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import ControlRig, StandInClock
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.motion.planning import HomePlanningError
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.layout import BOTH_ARMS, LAYOUT
from pear.deploy.station.control.robots.yam.motion.homing import YAMHoming
from pear.deploy.station.control.schemas import ControlSnapshot
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import StandInYAMRobot
from yam_rigs import MotionHelpers

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
# what the tests of a whole Home change of the tests' station: sequences may run late, the arms settle in a short
# window, also at Home, and Home's last wait is short
DISTANT_STATION = (
    "control.deadline_tolerance_s=0.5",
    "control.settle_window_s=0.1",
    "robot.home.settle_window_s=0.1",
    "robot.home.arrival_timeout_s=1.5",
)
LATE_TIMEOUT_S = 4.0  # Home's timeout when approved late: past planning and half the distant Home's estimate


class HomePoses:
    """The pose away from Home the arms of the tests start at."""

    @staticmethod
    def distant_pose() -> np.ndarray:
        """Returns both arms' joints away from Home: the bases turned, the grippers partly open."""
        q = np.zeros(14)
        q[[0, 7]] = [0.6, -0.6]
        q[[6, 13]] = [0.4, 0.7]
        return q


class StandInDistantYAMDriver(StandInYAMDriver):
    """YAM's stand-in joints that start away from Home."""

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        super().__init__(control_config, robot_config)
        q = HomePoses.distant_pose()
        self.positions = {s: tuple(q[i * 7 : (i + 1) * 7]) for i, s in enumerate(("left", "right"))}


class StandInDistantYAMRobot(StandInYAMRobot):
    """The stand-in YAM robot, its arms starting away from Home."""

    DRIVER_CLASS = StandInDistantYAMDriver


class HomeJudgement:
    """Home's own judgement of the arms, as a Home of some of them makes it: its wait over the stability history of a
    YAM control loop that the stand-in clock steps, and its accuracy of a snapshot."""

    @staticmethod
    def control_loop(
        clock: StandInClock,
        control_config: ControlConfig,
        robot_config: YAMConfig,
        positions: dict[str, tuple[float, ...]],
    ) -> tuple[YAMControlLoop, StandInYAMDriver]:
        """Returns a control loop and its driver, enabled with the arms at ``positions`` and stepped at each sleep of
        ``clock``, whose stability history starts with the test's scenario."""
        hardware = StandInYAMDriver(control_config, robot_config)
        hardware.positions = positions
        runtime = YAMControlLoop(hardware, control_config, robot_config)
        clock.control_loop = runtime
        # a loop settles when it is made: a gap past the age of a stale reading starts its history over at its next
        # reading, which the scenario's first step makes, at a whole second
        clock.now = float(math.ceil(clock.now + control_config.state_timeout_s))

        return runtime, hardware

    @staticmethod
    def homing(
        control_loop: YAMControlLoop,
        sides: Sequence[str],
        targets: dict[str, Sequence[float]] | None,
        open_grippers: bool,
    ) -> YAMHoming:
        """Returns a Home of ``sides`` over ``control_loop``, judged against their ``targets`` (by default their Home
        poses), which opens the grippers unless ``open_grippers`` is false."""
        home = control_loop.robot_config.home
        arm_targets = {arm: (targets or {}).get(arm, home.arm_pose(arm)) for arm in LAYOUT.arms}
        # Home's wait and its accuracy read no robot: only the control loop and the configurations of the guard
        homing = YAMHoming(None, ActionGuard(control_loop), None, BOTH_ARMS, open_grippers)
        homing.arms, homing.targets = tuple(sides), list(LAYOUT.join(arm_targets))

        return homing

    @classmethod
    def wait(
        cls,
        control_loop: YAMControlLoop,
        sides: Sequence[str],
        *,
        targets: dict[str, Sequence[float]] | None = None,
        open_grippers: bool = True,
    ) -> ControlSnapshot:
        """Waits, Home's arrival timeout at most, until the control loop's history shows ``sides`` settled at their
        targets within Home's tolerance, and returns the snapshot then."""
        homing = cls.homing(control_loop, sides, targets, open_grippers)

        return homing.wait(control_loop.robot_config.home.arrival_timeout_s)

    @classmethod
    def accuracy(
        cls,
        snapshot: ControlSnapshot,
        control_loop: YAMControlLoop,
        sides: Sequence[str],
        targets: dict[str, Sequence[float]],
        open_grippers: bool = True,
    ) -> dict[str, Any]:
        """Returns each of ``sides``' error in ``snapshot`` against its ``targets``, judged by Home's tolerance."""
        return cls.homing(control_loop, sides, targets, open_grippers).accuracy(snapshot)


@pytest.fixture
def distant_station() -> Iterator[StandInDistantYAMRobot]:
    """The open stand-in YAM robot, its arms away from Home, with the short settle windows of a whole Home's tests;
    settled at the start and closed at the end."""
    config = ConfigFiles.load("yam", *DISTANT_STATION)

    with StandInDistantYAMRobot(config) as station:
        station.wait_stationary()
        yield station


@pytest.mark.parametrize("mode", ["brief_velocity_spikes", "oscillation", "sustained_velocity"])
def test_home_uses_velocity_and_accuracy_without_position_span_limit(stand_in_clock, mode):
    control_config, config = ConfigFiles.load_configs("yam", *WIDE_STATION, "robot.home.arrival_timeout_s=1.0")
    clock = stand_in_clock
    runtime, hardware = HomeJudgement.control_loop(
        clock, control_config, config, {side: tuple(config.home.arm_pose(side)) for side in ("left", "right")}
    )
    read = hardware.read
    count = 0

    def feedback():
        nonlocal count
        states = read()
        count += 1
        state = states["left"]
        position, velocity = list(state.position), [0.0] * 7
        position[3] += (0.01 if mode == "oscillation" else 0.0004) * (-1) ** count
        velocity[3] = 0.1 if mode == "sustained_velocity" or count % 10 == 0 else 0.01
        states["left"] = replace(state, position=tuple(position), velocity=tuple(velocity))
        return states

    hardware.read = feedback
    runtime.step()  # the history's first reading, which the wait's first look needs fresh
    start = clock.now

    try:
        assert runtime.robot_config.home.arrival_timeout_s == 1.0
        if mode != "sustained_velocity":
            result = HomeJudgement.wait(runtime, ("left",))
            assert result["home_accuracy_window"]["stable"]
            if mode == "oscillation":
                assert result["home_accuracy_window"]["position_span"][3] == pytest.approx(0.02)
        else:
            with pytest.raises(TimeoutError, match="Home feedback"):
                HomeJudgement.wait(runtime, ("left",))
            assert 1.0 <= clock.now - start < 1.05
    finally:
        runtime.close()


@pytest.mark.parametrize("duration", [0.5, 0.8])
def test_home_reuses_shared_history_and_respects_longer_home_window(stand_in_clock, monkeypatch, duration):
    control_config, config = ConfigFiles.load_configs(
        "yam", *WIDE_STATION, f"robot.home.settle_window_s={duration}", "robot.home.arrival_timeout_s=2.0"
    )
    clock = stand_in_clock
    runtime, hardware = HomeJudgement.control_loop(
        clock, control_config, config, {s: tuple(config.home.arm_pose(s)) for s in ("left", "right")}
    )

    def tick(dt):
        clock.now += dt
        runtime.step()

    try:
        for _ in range(30):
            tick(runtime.control_config.control_period_s)
        assert runtime.snapshot()["arrival"]["settled"]
        before = clock.now
        HomeJudgement.wait(runtime, ("left", "right"))
        elapsed = clock.now - before
        if duration == 0.5:
            assert elapsed == 0
        else:
            assert 0.18 < elapsed < 0.3
        # Another query reads the same history and needs no new window.
        monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("Repeated stable window"))
        HomeJudgement.wait(runtime, ("left", "right"))
    finally:
        runtime.close()


def test_shared_home_history_requires_target_accuracy_for_entire_window(stand_in_clock):
    control_config, config = ConfigFiles.load_configs("yam", *WIDE_STATION)
    clock = stand_in_clock
    runtime, hardware = HomeJudgement.control_loop(
        clock, control_config, config, {s: tuple(config.home.arm_pose(s)) for s in ("left", "right")}
    )

    def pose(offset):
        q = list(config.home.left_pose)
        q[0] += offset
        hardware.positions["left"] = tuple(q)

    def tick(dt):
        clock.now += dt
        runtime.step()

    try:
        for _ in range(30):
            pose(0.0205)
            tick(runtime.control_config.control_period_s)
        assert runtime.snapshot()["arrival"]["settled"]
        # General stability alone cannot satisfy home's target accuracy.
        pose(0.0195)
        start = clock.now
        HomeJudgement.wait(runtime, ("left",))
        assert 0.48 < clock.now - start < 0.6
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "open_grippers,joint,offset,passes",
    [(True, 6, 0.3, False), (False, 6, 0.3, True), (False, 0, 0.03, False)],
    ids=["open_judges_gripper", "keep_ignores_gripper", "keep_judges_arm_joints"],
)
def test_kept_gripper_is_not_judged_but_arm_joints_are(stand_in_clock, open_grippers, joint, offset, passes):
    # A gripper closed on an object holds a target it cannot reach. Home that keeps
    # the grippers must not wait on it; the arm joints keep their Home tolerances.
    control_config, config = ConfigFiles.load_configs("yam", *WIDE_STATION, "robot.home.arrival_timeout_s=1.0")
    clock = stand_in_clock
    measured = list(config.home.left_pose)
    measured[6] = 0.3  # Object between the fingers.
    measured[joint] += offset if joint != 6 else 0.0
    runtime, hardware = HomeJudgement.control_loop(
        clock, control_config, config, {"left": tuple(measured), "right": tuple(config.home.right_pose)}
    )
    runtime.step()  # the history's first reading, which the wait's first look needs fresh
    target = list(config.home.left_pose)
    target[6] = 0.0 if joint == 6 else 0.3

    def wait():
        return HomeJudgement.wait(runtime, ("left",), targets={"left": target}, open_grippers=open_grippers)

    try:
        if not passes:
            with pytest.raises(TimeoutError, match="Home feedback"):
                wait()
            return
        result = HomeJudgement.accuracy(wait(), runtime, ("left",), {"left": np.array(target)}, False)
        assert result["left"]["reached"] and result["left"]["gripper_kept"]
        assert result["left"]["tolerance"][6] is None
        json.dumps(result, allow_nan=False)  # GUI failure records forbid infinity.
    finally:
        runtime.close()


def test_full_distant_home_executes_direct_move_and_remains_enabled(distant_station):
    station = distant_station
    owner = station.control_loop
    plans = []

    result = station.home(approve_plan=lambda plan: plans.append(plan) or True)

    assert len(plans) == 1 and result["plan"]["trajectory_seconds"] > 0
    assert "method" not in result["plan"]
    assert station.control_loop is owner and not owner.driver.is_closed
    after = SnapshotCheck.state_vector(result["after"])
    assert after[[6, 13]] == pytest.approx([1, 1])
    assert after[np.r_[0:6, 7:13]] == pytest.approx(np.zeros(12), abs=0.002)
    assert station.status()["status"] == "READY"


@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_keep_grippers_home_never_commands_grippers(distant_station, side):
    station = distant_station
    before = SnapshotCheck.target_vector(station.control_loop.snapshot())
    sent = len(station.control_loop.driver.sent_targets)

    result = station.home(side, opens_grippers=False)

    assert result["open_grippers"] is False and result["plan"]["opening_commands"] == {}
    assert [phase["stage"] for phase in result["phases"]] == ["homing"]
    commands = station.control_loop.driver.sent_targets[sent:]
    assert commands
    for command in commands:
        assert [command["left"][6], command["right"][6]] == pytest.approx(before[[6, 13]])
    after = SnapshotCheck.state_vector(result["after"])
    for i, name in enumerate(("left", "right")):
        arm = after[i * 7 : i * 7 + 6]
        if side in (name, "both"):
            assert result["results"][name]["reached"]
            assert result["results"][name]["gripper_kept"]
            assert arm == pytest.approx(np.zeros(6), abs=0.002)
        else:
            assert name not in result["results"]
            assert arm == pytest.approx(before[i * 7 : i * 7 + 6])
    json.dumps(result["results"], allow_nan=False)
    assert station.status()["status"] == "READY"


def test_cancelled_review_dispatches_no_move(distant_station):
    station = distant_station
    original = SnapshotCheck.target_vector(station.control_loop.snapshot())

    with pytest.raises(HomeFailed, match="cancelled before execution") as error:
        station.home(approve_plan=lambda plan: False)

    assert isinstance(error.value.__cause__, ValueError)
    assert SnapshotCheck.target_vector(station.control_loop.snapshot()) == pytest.approx(original)
    assert station.control_loop.operation is None
    assert not station.control_loop.driver.is_closed


def test_stop_during_planning_prevents_dispatch(distant_station, monkeypatch):
    station = distant_station
    original = SnapshotCheck.target_vector(station.control_loop.snapshot())
    plan = station.planner.plan_home

    def stop_after_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        station.stop()
        return result

    monkeypatch.setattr(station.planner, "plan_home", stop_after_plan)

    with pytest.raises(RuntimeError, match="cancelled"):
        station.home()

    assert SnapshotCheck.target_vector(station.control_loop.snapshot()) == pytest.approx(original)
    assert station.control_loop.operation is None


@pytest.mark.parametrize("bias,passes", [(0.021, True), (0.028, True), (0.03501, False)])
def test_home_tolerance_is_independent_of_policy(stand_in_clock, bias, passes):
    control_config, config = ConfigFiles.load_configs("yam")
    q = np.zeros(14)
    q[[6, 13]] = 1
    state = MotionHelpers.snapshot(q)
    for name in ("left", "right"):
        state["arms"][name]["position"][2] = bias

    # Home's judgement reads the configurations of its control loop, which does not run here
    with ControlRig(YAMControlLoop, StandInYAMDriver, control_config, config, stand_in_clock) as (runtime, _, _):
        result = HomeJudgement.accuracy(state, runtime, ["left", "right"], state["targets"])

    assert all(r["reached"] == passes for r in result.values())
    assert not MotionHelpers.tracking(config).arrival(*MotionHelpers.vectors(state))["reached"]
    if not passes:
        assert result["left"]["violations"][0]["joint"] == "J3"
        assert result["left"]["violations"][0]["limit"] == 0.035


@pytest.mark.parametrize("failure", ["opening", "recheck"])
def test_failed_open_or_invalidated_move_never_moves_arms(distant_station, monkeypatch, failure):
    station = distant_station
    before = SnapshotCheck.target_vector(station.control_loop.snapshot())
    if failure == "opening":

        def blocked(*args, **kwargs):
            return {"outcomes": {"left": "blocked", "right": "open"}}

        monkeypatch.setattr(station, "operate_grippers", blocked)
    else:

        def invalid(*args, **kwargs):
            raise HomePlanningError("fresh feedback invalidates approved move")

        monkeypatch.setattr(station.planner, "recheck_home", invalid)

    with pytest.raises(HomeFailed) as error:
        station.home()

    after = SnapshotCheck.target_vector(station.control_loop.snapshot())
    assert after[np.r_[0:6, 7:13]] == pytest.approx(before[np.r_[0:6, 7:13]])
    assert isinstance(error.value.__cause__, RuntimeError if failure == "opening" else HomePlanningError)
    assert error.value.details["stage"] == ("opening" if failure == "opening" else "checking")


@pytest.mark.parametrize("excessive", [False, True])
def test_home_accepts_a_steady_j3_bias_within_its_own_tolerance_and_fails_beyond_it(
    distant_station, monkeypatch, excessive
):
    station = distant_station
    # Inject stable encoder bias before capture; both planning and recheck see it.
    read = station.control_loop.driver.read

    def feedback():
        states = read()
        for side, bias in (("left", 0.021), ("right", 0.036 if excessive else 0.028)):
            q = list(states[side].position)
            q[2] += bias
            states[side] = replace(states[side], position=tuple(q))
        return states

    monkeypatch.setattr(station.control_loop.driver, "read", feedback)
    station.control_loop.wait_for_hold(
        3, accepts=lambda state: abs(state["arms"]["left"]["position"][2] - 0.021) < 1e-9
    )
    station.wait_stationary()

    if excessive:
        with pytest.raises(HomeFailed) as error:
            station.home()
        assert isinstance(error.value.__cause__, TimeoutError)
        details = error.value.details
        assert details["stage"] == "homing"
        assert details["results"]["right"]["violations"][0]["joint"] == "J3"
    else:
        result = station.home()
        assert all(value["reached"] for value in result["results"].values())
        assert station.status(with_snapshot=True)["arrival"]["reached"] is False


@pytest.mark.parametrize("stop_at", ["checking_after_open", "homing", "moving"])
def test_home_cancellation_never_starts_or_resumes_another_phase(distant_station, monkeypatch, stop_at):
    station = distant_station
    initial = SnapshotCheck.target_vector(station.control_loop.snapshot())
    opened = False

    def progress(stage):
        nonlocal opened
        if stage["stage"] == "opening":
            opened = True
        if (stop_at == "checking_after_open" and opened and stage["stage"] == "checking") or (
            stop_at == "homing" and stage["stage"] == "homing"
        ):
            station.stop()

    if stop_at == "moving":
        original = station.control_loop.driver.send
        stopped = False

        def send(targets, velocities=None):
            nonlocal stopped
            original(targets, velocities)
            if not stopped and abs(targets["left"][0] - initial[0]) > 0.002:
                stopped = True
                station.stop()

        monkeypatch.setattr(station.control_loop.driver, "send", send)

    with pytest.raises(RuntimeError, match="cancelled"):
        station.home(report_progress=progress)

    station.control_loop.wait_for_hold(station.control_loop.HOLD_WAIT_S)
    final = SnapshotCheck.target_vector(station.control_loop.snapshot())
    assert final[[6, 13]] == pytest.approx([1, 1])
    if stop_at != "moving":
        assert final[np.r_[0:6, 7:13]] == pytest.approx(initial[np.r_[0:6, 7:13]])
    else:
        assert final[0] > 0.1


def test_home_confirmation_revalidates_cancel_token(distant_station):
    cancel = threading.Event()
    before = len(distant_station.control_loop.driver.sent_targets)

    def review(plan):
        cancel.set()
        return True

    with pytest.raises(RuntimeError):
        distant_station.home(approve_plan=review, cancellation=cancel)

    assert len(distant_station.control_loop.driver.sent_targets) == before


def test_late_approval_sends_nothing(distant_station):
    station = distant_station
    before = len(station.control_loop.driver.sent_targets)
    started = time.monotonic()

    def slow_review(plan):
        # The operator answers once only half of the plan's estimated time is left.
        left = plan["operation_seconds_estimate"] / 2
        time.sleep(max(0.0, started + LATE_TIMEOUT_S - left - time.monotonic()))
        return True

    with pytest.raises(HomeFailed, match="confirm again"):
        station.home(approve_plan=slow_review, timeout_s=LATE_TIMEOUT_S)

    assert len(station.control_loop.driver.sent_targets) == before
    assert station.control_loop.operation is None


def test_already_home_opens_nothing_and_sends_no_motion(distant_station):
    station = distant_station
    station.home()
    before = len(station.control_loop.driver.sent_targets)
    generation = station.control_loop.generation

    result = station.home()

    assert result["phases"] == []
    assert result["plan"]["trajectory_seconds"] == 0
    assert result["plan"]["opening_commands"] == {}
    assert len(station.control_loop.driver.sent_targets) == before
    assert station.control_loop.generation == generation


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_side_home_preserves_other_side(distant_station, side):
    station = distant_station
    other = "right" if side == "left" else "left"
    before = station.control_loop.targets[other]
    sent = len(station.control_loop.driver.sent_targets)

    result = station.home(side)

    assert result["results"][side]["reached"]
    assert all(command[other] == before for command in station.control_loop.driver.sent_targets[sent:])


def test_position_change_during_home_confirmation_allows_execution(distant_station):
    station = distant_station
    before = len(station.control_loop.driver.sent_targets)

    def review(plan):
        with station.control_loop.lock:
            hardware = station.control_loop.driver
            q = list(hardware.positions["left"])
            q[0] += 0.02
            hardware.positions["left"] = tuple(q)
            station.control_loop.states = hardware.read()
        return True

    result = station.home(approve_plan=review)

    assert all(value["reached"] for value in result["results"].values())
    assert len(station.control_loop.driver.sent_targets) > before
