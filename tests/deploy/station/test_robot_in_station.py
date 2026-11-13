import copy
import json
import threading
import time
import types
import typing
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import StandInClock, Ticker
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.stability import StabilityWindow
from pear.deploy.station.control.motion.execution import Execution
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.motion.actions import JointPositions
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from pear.deploy.station.control.schemas import (
    BriefStatus,
    CameraFrame,
    ControlState,
    DetailedStatus,
    HomePlan,
    HomeReport,
    Observation,
    RobotStatus,
)
from stand_in_cameras import StandInCameras
from stand_in_robots import StandInRobots, StandInYAMRobot, SteppedStandInYAMRobot
from waiting import Waiting
from yam_rigs import ControlHelpers, MotionHelpers, ReplayRobot

SETTLE_TIMEOUT_S = 10.0  # for a stand-in arm to reach a settled HOLD
POLL_S = 0.01  # between two looks at a robot a test waits for
ZERO = {
    "yam": dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
    "so101": dict(x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0),
}  # each robot's arm delta of no motion
# what each robot's contract tests change of its station of the tests: the SO-101's short settle windows and fast caps,
# and the Home and the tolerances of the SO-101 these tests ran with
CONTRACT_OVERRIDES = {
    "yam": (),
    "so101": (
        *ConfigFiles.SO101_QUICK_OVERRIDES,
        "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
        "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
        "robot.position_tolerance=0.035",
        "robot.feedback_position_tolerance=0.05",
    ),
}
# the hooks of the robot's base class that a robot's class implements
HOOKS = ("open_planner", "close_planner", "plan", "result", "run_home")
RELAXED_DEADLINE_TOLERANCE = "control.deadline_tolerance_s=0.5"  # a sequence of targets may run this late
RECORDED_STATIC_TRACKING_ERROR = "recorded_static_tracking_error.json"  # a recorded hold with a static error


class Shape:
    """Checks that a value has the keys of a TypedDict, nested TypedDicts and per-key maps of them included."""

    @classmethod
    def check(cls, value: dict[str, Any], kind: type, path: str = "value") -> None:
        """Fails unless ``value`` has every required key of ``kind``, recursively."""
        missing = kind.__required_keys__ - set(value)
        assert not missing, f"{path} lacks {sorted(missing)}"
        for key, hint in typing.get_type_hints(kind).items():
            if key in value:
                cls.check_hint(value[key], hint, f"{path}.{key}")

    @classmethod
    def check_hint(cls, item: Any, hint: Any, path: str) -> None:
        """Checks ``item`` against one field's type when that type is, or holds, a TypedDict."""
        arguments = typing.get_args(hint)
        if typing.is_typeddict(hint):
            cls.check(item, hint, path)
        elif isinstance(hint, types.UnionType) and item is not None:
            for kind in arguments:
                if typing.is_typeddict(kind):
                    cls.check(item, kind, path)
        elif typing.get_origin(hint) is dict and typing.is_typeddict(arguments[1]):
            for key, value in item.items():
                cls.check(value, arguments[1], f"{path}[{key}]")


class RobotContract:
    """What every robot must do, run against its stand-in hardware: an action interrupted on its way."""

    @staticmethod
    def interrupted(
        station: BaseRobot,
        action: dict[str, Any],
        interrupt: Callable[[], None],
        *,
        cancel: threading.Event | None = None,
    ) -> Any:
        """Runs ``action`` in a thread, calls ``interrupt`` once the arm moves, and returns the action's result."""
        results = []
        worker = threading.Thread(target=lambda: results.append(station.execute(action, cancellation=cancel)))

        worker.start()
        Waiting.until(
            lambda: station.status(with_snapshot=True).get("control_state") == "RUNNING" or not worker.is_alive(),
            SETTLE_TIMEOUT_S,
            POLL_S,
        )
        interrupt()
        worker.join(SETTLE_TIMEOUT_S)

        assert not worker.is_alive()
        return results[0]


class ThreadCapture:
    """The threads a block of code starts, kept as the code makes them."""

    @staticmethod
    def started(monkeypatch: pytest.MonkeyPatch, start: Callable[[], Any]) -> tuple[Any, list[threading.Thread]]:
        """Calls ``start`` and returns what it returns with every thread it made."""
        threads, thread_class = [], threading.Thread
        with monkeypatch.context() as patch:
            patch.setattr(threading, "Thread", lambda **kwargs: threads.append(thread_class(**kwargs)) or threads[-1])
            result = start()

        return result, threads


@pytest.fixture
def robot_class(robot_name: str) -> type[BaseRobot]:
    """The class of the robot ``robot_name`` that its station of the tests names."""
    return StandInRobots.robot_class(robot_name)


@pytest.fixture
def station_factory(robot_name: str, robot_class: type[BaseRobot]) -> Callable[[], BaseRobot]:
    """What makes the robot ``robot_name`` of its contract tests on stand-in hardware, without cameras, not open."""
    return lambda: robot_class(ConfigFiles.load(robot_name, *CONTRACT_OVERRIDES[robot_name]))


@pytest.fixture
def station(station_factory: Callable[[], BaseRobot]) -> Iterator[BaseRobot]:
    """The robot of the contract tests, open and settled; closed at the end."""
    with station_factory() as station:
        station.wait_stationary()
        yield station


@pytest.fixture
def sample_action(robot_name: str) -> dict[str, Any]:
    """A small action of the robot ``robot_name`` that completes: a lift of every arm; the grippers hold."""
    if robot_name == "yam":
        return {
            "type": "bimanual_delta",
            "left_arm": {**ZERO["yam"], "z": 0.3},
            "right_arm": {**ZERO["yam"], "z": 0.3},
            "left_gripper": {"width": 0.0},
            "right_gripper": {"width": 0.0},
        }

    return {"type": "single_arm_delta", "arm": {**ZERO["so101"], "z": 1.0}, "gripper": {"width": 0.0}}


@pytest.fixture
def long_action(robot_name: str) -> dict[str, Any]:
    """An action of the robot ``robot_name`` long enough to be interrupted on its way; the grippers hold."""
    if robot_name == "yam":
        return {
            "type": "bimanual_delta",
            "left_arm": {**ZERO["yam"], "z": -2.0},
            "right_arm": {**ZERO["yam"], "z": -2.0},
            "left_gripper": {"width": 0.0},
            "right_gripper": {"width": 0.0},
        }

    return {"type": "single_arm_delta", "arm": {**ZERO["so101"], "x": -8.0, "z": -6.0}, "gripper": {"width": 0.0}}


@pytest.fixture
def stale_feedback(robot_name: str) -> Callable[[BaseRobot, bool], None]:
    """What makes the readings of the robot ``robot_name``'s stand-in hardware stale or fresh again."""

    def inject(station: BaseRobot, is_stale: bool) -> None:
        driver = station.control_loop.driver
        # the SO-101's driver reads its stand-in arm
        (driver if robot_name == "yam" else driver.hardware).is_stale = is_stale

    return inject


@pytest.fixture
def yam_station(stand_in_clock: StandInClock) -> Iterator[tuple[BaseRobot, Ticker]]:
    """A YAM robot of its station of the tests in joint mode, its control loop stepped by hand, and its ticker; the
    control loop is closed at the end."""
    robot = SteppedStandInYAMRobot(ConfigFiles.load("yam"), action_mode=ActionMode.JOINT).open()

    try:
        yield robot, Ticker(robot.control_loop, stand_in_clock)
    finally:
        robot.control_loop.close()


@pytest.fixture
def replay(stand_in_clock: StandInClock) -> tuple[ReplayRobot, StandInClock, dict[str, Any]]:
    """A YAM robot's waits over a recorded hold with a static tracking error, the clock at zero, and the record."""
    control_config, robot_config = ConfigFiles.load_configs("yam")
    recorded = json.loads((ConfigFiles.YAM_RECORDED_CASES / RECORDED_STATIC_TRACKING_ERROR).read_text())
    stand_in_clock.now = 0.0
    station = ReplayRobot(control_config, robot_config, lambda: copy.deepcopy(recorded))

    return station, stand_in_clock, recorded


def test_constructing_opens_nothing(station_factory):
    station = station_factory()

    assert station.status() == {"status": "OFF"}
    assert station.session_id is None
    with pytest.raises(RuntimeError, match="Open the station first"):
        station.observe()


def test_station_has_the_contract_members(station_factory):
    assert isinstance(station_factory(), BaseRobot)


def test_status_and_observation_have_the_contract_shape(station):
    brief, detailed, observation = station.status(), station.status(with_snapshot=True), station.observe()

    Shape.check(brief, BriefStatus)
    Shape.check(detailed, DetailedStatus)
    Shape.check(observation, Observation)
    assert brief["status"] in RobotStatus and detailed["status"] in RobotStatus
    assert detailed["control_state"] in ControlState
    assert observation["control_snapshot"]["status"] in ControlState


def test_stand_in_cameras_have_the_contract_frames(robot_class):
    cameras = StandInCameras(robot_class.CAMERAS)
    try:
        frames, errors = cameras.read_preview()
        for role, frame in cameras.read().items():
            Shape.check(frame, CameraFrame, role)
    finally:
        cameras.close()

    assert set(frames) == {camera.role for camera in robot_class.CAMERAS} and errors == {}


def test_a_robot_opens_ready_with_a_session_and_closes_off_without_one(station_factory):
    station = station_factory()

    station.open()
    try:
        station.wait_stationary()
        assert station.status()["status"] == "READY"
        assert station.session_id is not None
    finally:
        station.close()

    assert station.status() == {"status": "OFF"}
    assert station.session_id is None


def test_an_observation_carries_its_session_and_a_held_pose(station):
    observation = station.observe()

    assert {
        "observation_id",
        "session_id",
        "joint_positions",
        "images",
        "camera_timestamps",
        "control_snapshot",
    } <= set(observation)
    assert observation["session_id"] == station.session_id
    assert observation["control_snapshot"]["status"] == "HOLD"
    assert station.is_holding(observation)


def test_an_action_completes_settled_at_its_target(station, sample_action):
    result = station.execute(sample_action)

    assert (result.status, result.is_settled, result.target_reached, result.error) == (
        "completed",
        True,
        True,
        None,
    )
    assert result.measured_state is not None
    assert station.status()["status"] == "READY"


def test_an_action_based_on_an_observation_from_before_another_action_is_rejected(station, sample_action):
    observation = station.observe()
    station.execute(sample_action)

    result = station.execute(sample_action, reference_observation=observation)

    assert result.status == "rejected"
    assert result.error == {"code": "ValueError", "message": "Observation invalidated by another operation"}


def test_a_stop_during_an_action_fails_it_and_ends_in_a_settled_hold(station, long_action):
    result = RobotContract.interrupted(station, long_action, station.stop)

    assert result.status == "failed", result
    station.wait_settled(SETTLE_TIMEOUT_S)
    assert station.status()["status"] == "READY"
    assert station.status(with_snapshot=True)["motion"]["end_reason"] == "stopped"


def test_stale_feedback_faults_the_station_and_recover_makes_it_ready(station, stale_feedback):
    stale_feedback(station, True)
    Waiting.until(lambda: station.status()["status"] == "FAULT", SETTLE_TIMEOUT_S, POLL_S)

    with pytest.raises(RuntimeError, match="Station fault"):
        station.check_fault()
    stale_feedback(station, False)
    station.recover()

    station.check_fault()
    assert station.status()["status"] == "READY"


def test_a_refused_home_review_fails_before_anything_moves_and_an_approved_home_returns_its_plan_and_report(
    station, sample_action
):
    station.execute(sample_action)
    plans = []

    with pytest.raises(HomeFailed, match="Home cancelled before execution") as failure:
        station.home(approve_plan=lambda plan: False)
    assert failure.value.details["stage"] and failure.value.details["completed_stages"] == []
    report = station.home(approve_plan=lambda plan: plans.append(plan) or True)

    Shape.check(plans[0], HomePlan)
    Shape.check(report, HomeReport)
    assert station.status()["status"] == "READY"


def test_a_cancel_during_an_action_fails_it_as_paused_and_ends_in_a_settled_hold(station, long_action):
    cancel = threading.Event()

    result = RobotContract.interrupted(station, long_action, cancel.set, cancel=cancel)

    assert result.status == "failed", result
    assert result.error["code"] == "OperatorPaused"
    station.wait_settled(SETTLE_TIMEOUT_S)
    assert station.status()["status"] == "READY"


def test_the_model_state_has_the_parts_of_each_format_and_a_failure_is_a_failed_result_carrying_the_error(
    station, robot_class
):
    state = station.observe()["joint_positions"]

    for name, action_format in robot_class.ACTION_FORMATS.items():
        quantities = station.model_state(state, name).model_dump()
        failure = station.failure_result(RuntimeError("injected"), format_name=name)

        assert {part: tuple(values) for part, values in quantities.items()} == action_format.schema
        assert failure.status == "failed"
        assert failure.error == {"code": "RuntimeError", "message": "injected"}
        assert failure.measured_state is not None


def test_the_hooks_are_abstract():
    assert BaseRobot.__abstractmethods__ == frozenset(HOOKS)


@pytest.mark.parametrize("missing", HOOKS)
def test_a_robot_missing_a_hook_cannot_be_built(missing):
    hooks = {name: lambda self, *args, **kwargs: None for name in HOOKS if name != missing}
    partial = type("PartialStation", (BaseRobot,), hooks)

    with pytest.raises(TypeError, match=missing):
        partial("station.yaml")


def test_repeated_stationary_checks_reuse_controller_history_and_stale_is_not_ready(
    yam_station, stand_in_clock, monkeypatch
):
    station, ticker = yam_station
    now = stand_in_clock
    for _ in range(30):
        ticker.tick()
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("Repeated an already satisfied stability wait"))
    first = station.wait_stationary()
    second = station.wait_stationary()
    assert first["settling"] == second["settling"]
    assert first["settling"]["window_seconds"] >= 0.5
    assert station.status()["status"] == "READY"
    now.now += 0.11
    assert station.status()["status"] == "BUSY"
    with pytest.raises(RuntimeError, match="Stale"):
        station.wait_stationary()


@pytest.mark.parametrize("cancel", ["stop", "close", "reopen"])
def test_cancel_during_initial_wait_never_dispatches_or_affects_reopened_session(monkeypatch, cancel):
    config = ConfigFiles.load("yam", RELAXED_DEADLINE_TOLERANCE)
    entered, release = threading.Event(), threading.Event()
    original = YAMRobot.wait_for_settled_hold

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(YAMRobot, "wait_for_settled_hold", gated)
    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        old = station.control_loop
        q = station.observe()["joint_positions"].tolist()
        q[0] = 0.02
        results, errors = [], []

        def run():
            try:
                results.append(station.execute({"type": "joint_positions", "positions": [q], "rate_hz": 30.0}))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(3)
            if cancel == "stop":
                station.stop()
            else:
                station.close()
                if cancel == "reopen":
                    station.open()
                    targets = {s: list(v) for s, v in station.control_loop.targets.items()}
                    targets["right"][0] = 0.1
                    station.control_loop.move(targets)
                    generation = station.control_loop.generation
            release.set()
            thread.join(5)
            assert not thread.is_alive()
            assert not errors
            assert results[0].status == "failed"
            assert results[0].error["code"] == "RuntimeError"
            assert all(command["left"][0] == 0 for command in old.driver.sent_targets)
            if cancel == "reopen":
                assert station.control_loop.generation == generation
                assert station.control_loop.trajectory.input.target_position[7] == 0.1
            elif cancel == "close":
                assert station.status()["status"] == "OFF"
                assert old.driver.is_closed
        finally:
            release.set()
            thread.join(5)


@pytest.mark.parametrize("reopen", [False, True])
def test_close_during_submitted_action_returns_result_from_original_runtime(monkeypatch, reopen):
    config = ConfigFiles.load("yam", RELAXED_DEADLINE_TOLERANCE)
    entered, release = threading.Event(), threading.Event()
    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        old = station.control_loop
        wait = old.wait_for_hold

        def gated(*args, **kwargs):
            if kwargs.get("guard") is not None:
                entered.set()
                assert release.wait(5)
            return wait(*args, **kwargs)

        monkeypatch.setattr(old, "wait_for_hold", gated)
        q = station.observe()["joint_positions"].tolist()
        q[0] += 0.1
        results, errors = [], []

        def run():
            try:
                results.append(station.execute({"type": "joint_positions", "positions": [q], "rate_hz": 30.0}))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(3)
            assert old.generation == 1
            station.close()
            if reopen:
                station.open()
                generation = station.control_loop.generation
            release.set()
            thread.join(5)
            assert not thread.is_alive() and not errors
            assert results[0].status == "failed"
            assert results[0].error["code"] == "RuntimeError"
            assert old.status == "CLOSED" and old.driver.is_closed
            if reopen:
                assert station.control_loop.generation == generation
        finally:
            release.set()
            thread.join(5)


def test_camera_failure_from_old_observation_does_not_stop_reopened_session():
    config = ConfigFiles.load("yam", RELAXED_DEADLINE_TOLERANCE)
    stations = []

    def read():
        station = stations[0]
        station.close()
        station.open()
        targets = {s: list(q) for s, q in station.control_loop.targets.items()}
        targets["right"][0] = 0.1
        station.control_loop.move(targets)
        raise OSError("Old camera closed during observation")

    # the cameras the robot is made with: their read fails once the robot holds them, as the cameras the SOURCE gave
    # an open station did
    cameras = SimpleNamespace(read=read, close=lambda: None)
    with StandInYAMRobot(config, cameras=cameras, action_mode=ActionMode.JOINT) as station:
        stations.append(station)
        with pytest.raises(OSError, match="Old camera"):
            station.observe()
        assert station.control_loop.generation == 1
        assert station.control_loop.trajectory.input.target_position[7] == 0.1


def test_policy_chunk_reaches_final_action_and_releases_hardware():
    station = StandInYAMRobot(ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), action_mode=ActionMode.JOINT)
    with station:
        q = station.observe()["joint_positions"]
        assert q.shape == (14,)
        actions = np.repeat(q[None], 6, axis=0)
        actions[:, 0] += np.linspace(0, 0.05, 6)
        actions[:, 7] -= np.linspace(0, 0.04, 6)
        station.execute(JointPositions(positions=actions.tolist(), rate_hz=30.0))
        assert station.observe()["joint_positions"] == pytest.approx(actions[-1], abs=0.002)
        assert station.control_loop.status == "HOLD"
        hardware = station.control_loop.driver
    assert station.session_id is None
    assert hardware.is_closed


def test_an_unknown_robot_class_never_opens_hardware():
    with pytest.raises(ValueError, match="no robot's class"):
        ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"], "robot_class=builtins.dict")


def test_explicit_session_can_move_inspect_fault_and_reopen():
    station = StandInYAMRobot(ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), action_mode=ActionMode.JOINT)
    assert station.status() == {"status": "OFF"}
    with pytest.raises(RuntimeError, match="Open"):
        station.observe()
    try:
        station.open()
        original = station.control_loop.snapshot()["targets"]
        with pytest.raises(RuntimeError, match="already open"):
            station.open()
        ControlHelpers.jog(station.control_loop, "left", 1, 0.02)
        result = station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        assert result["arms"]["left"]["position"][0] == pytest.approx(0.02)
        station.control_loop.move(original)
        station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        ControlHelpers.grip(station.control_loop, "right", 0.52)
        station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        assert station.observe()["joint_positions"][13] == pytest.approx(0.52)
        station.control_loop.driver.failed_arm = "right"
        station.control_loop.step()
        assert station.control_loop.snapshot()["status"] == "FAULT"
        with pytest.raises(RuntimeError, match="FAULT"):
            station.observe()
        station.close()
        assert station.status() == {"status": "OFF"}
        station.open()
        assert station.control_loop.snapshot()["status"] == "HOLD"
    finally:
        station.close()


@pytest.mark.parametrize("failure", [TimeoutError("stop timeout"), KeyboardInterrupt()])
def test_close_releases_hardware_when_stop_wait_fails(monkeypatch, failure):
    station, [control_thread] = ThreadCapture.started(
        monkeypatch,
        StandInYAMRobot(ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), action_mode=ActionMode.JOINT).open,
    )
    runtime = station.control_loop
    released = []
    close = runtime.driver.close

    def release():
        released.append("motors")
        close()

    def fail(*_, **kwargs):
        raise failure

    monkeypatch.setattr(runtime.driver, "close", release)
    monkeypatch.setattr(runtime, "wait_for_hold", fail)
    ControlHelpers.jog(station.control_loop, "left", 1, 0.2)
    try:
        with pytest.raises(type(failure)):
            station.close()
        assert not control_thread.is_alive()
        assert runtime.driver.is_closed
        assert station.session_id is None
        station.close()
        assert released == ["motors"]
    finally:
        runtime.close()


def test_caller_owned_cameras_survive_failed_open_and_close(monkeypatch):
    released, reads = [], []
    cameras = SimpleNamespace(read=lambda: reads.append(True) or {}, close=lambda: released.append("cameras"))

    def fail(*_):
        raise RuntimeError("enable failed")

    station = StandInYAMRobot(
        ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), cameras=cameras, action_mode=ActionMode.JOINT
    )
    with monkeypatch.context() as patch:
        patch.setattr(YAMControlLoop, "open", fail)
        with pytest.raises(RuntimeError, match="enable failed"):
            station.open()
        station.close()
    # the robot still reads the caller's cameras: a session opened now observes through them
    with station:
        station.observe()
    assert not released and reads
    assert station.status() == {"status": "OFF"}


def test_public_surface_configuration_snapshot_and_old_observation_are_explicit():
    config = ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"])
    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        for name in (
            "command",
            "stream",
            "finish_stream",
            "dispatch_joint_move",
            "try_move_path",
            "move",
            "jog",
            "grip",
            "wait_for_hold",
        ):
            assert not hasattr(station, name)
        assert callable(station.home)
        assert not hasattr(station, "recover_hold")
        original = station.control_config.rate_hz
        config.control.rate_hz = 25
        station.control_config.rate_hz = 10
        station.control_loop.control_config.rate_hz = 10
        assert station.control_config.rate_hz == station.control_loop.control_config.rate_hz == original
        station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        assert station.status()["status"] == "READY"
        observation = station.observe()
        station.stop()
        result = station.execute(
            JointPositions(positions=[observation["joint_positions"].tolist()], rate_hz=30.0),
            reference_observation=observation,
        )
        assert result.status == "rejected" and "invalidated" in result.error["message"]
    station.open()
    try:
        assert station.control_config.rate_hz == 25
        result = station.execute(
            JointPositions(positions=[observation["joint_positions"].tolist()], rate_hz=20.0),
            reference_observation=observation,
        )
        assert result.status == "rejected" and "different station session" in result.error["message"]
    finally:
        station.close()


def test_timeout_only_tightens_and_requests_controlled_stop():
    with StandInYAMRobot(
        ConfigFiles.load("yam", *ConfigFiles.WIDE_LIMITS["yam"]), action_mode=ActionMode.JOINT
    ) as station:
        station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        q = station.observe()["joint_positions"]
        q[0] = 0.8
        result = station.execute(JointPositions(positions=[q.tolist()], rate_hz=30.0), timeout_s=0.05)
        assert result.status == "failed" and result.error["code"] == "TimeoutError"
        assert station.control_loop.snapshot()["status"] == "STOPPING"
        station.control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)
        assert not result.target_reached


@pytest.mark.parametrize("failure", ["stale", "fault", "nonfinite"])
def test_arrival_wait_does_not_defer_other_failures(replay, failure):
    station, clock, recorded = replay
    if failure == "stale":
        recorded["arms"]["right"]["age_seconds"] = 1
    elif failure == "fault":
        recorded.update(status="FAULT", fault="motor failure")
    else:
        recorded["arms"]["right"]["position"][0] = float("nan")
    with pytest.raises((RuntimeError, ValueError)):
        station.wait_stationary()
    assert clock.now == 0


@pytest.mark.parametrize("recovers", [True, False])
def test_dispatch_waits_for_velocity_recovery_but_stops_persistent_noise(stand_in_clock, recovers):
    now = stand_in_clock
    q = np.zeros(14)
    q[[6, 13]] = 0.5
    commands = []

    def feedback():
        state = MotionHelpers.snapshot(q)
        state["arms"]["left"]["velocity"][3] = 0.08 if not recovers or now.now - StandInClock.START_S < 0.7 else 0.0
        return state

    control_config, config = ConfigFiles.load_configs("yam")
    monitor = StabilityWindow(
        control_config,
        config,
        duration_s=control_config.settle_window_s,
        is_strict=False,
        checks_velocity_rms=True,
    )

    def dispatch(move, observation):
        if monitor.add(MotionHelpers.stamp(feedback(), now.now)):
            commands.append((now.now, move))
            return True
        return False

    station = SimpleNamespace(
        remaining_s=Execution.remaining_s,
        control_loop=SimpleNamespace(
            dispatch_joint_move=dispatch,
            lock=threading.RLock(),
            generation=0,
            status="HOLD",
            fault=None,
            control_config=control_config,
        ),
    )
    guard = ActionGuard(station.control_loop)
    if recovers:
        Execution.submit_when_settled(station, guard, station.control_loop.dispatch_joint_move, q, {}, deadline=None)
        assert len(commands) == 1
        assert 0.7 <= commands[0][0] - StandInClock.START_S < control_config.settle_timeout_s
        assert max(monitor.metrics["velocity_mean"]) <= 0.06
    else:
        with pytest.raises(TimeoutError, match="No stable window"):
            Execution.submit_when_settled(
                station, guard, station.control_loop.dispatch_joint_move, q, {}, deadline=None
            )
        assert not commands


@pytest.mark.parametrize("index,offset", [(0, 0.006), (9, -0.02), (6, 0.02), (13, -0.02)])
def test_observation_accepts_changed_arm_and_gripper_positions(index, offset):
    reference = MotionHelpers.observation(MotionHelpers.snapshot(np.zeros(14)))
    measured = np.zeros(14)
    measured[index] = offset
    SnapshotCheck.validate_reference(MotionHelpers.snapshot(measured), reference)


def test_another_operation_invalidates_the_observation():
    reference = MotionHelpers.observation(MotionHelpers.snapshot(np.zeros(14)))
    current = MotionHelpers.snapshot(np.zeros(14))
    current["generation"] += 1
    with pytest.raises(ValueError, match="another operation"):
        SnapshotCheck.validate_reference(current, reference)


@pytest.mark.parametrize("state", [np.zeros(13), np.zeros((2, 7)), [float("nan")] * 14])
def test_a_malformed_observation_is_rejected(state):
    current = MotionHelpers.snapshot(np.zeros(14))
    reference = {**MotionHelpers.observation(current), "joint_positions": state}
    with pytest.raises(ValueError, match="Invalid observation reference"):
        SnapshotCheck.validate_reference(current, reference)


def test_another_session_invalidates_the_observation(stand_in_clock):
    current = MotionHelpers.snapshot(np.zeros(14))
    control_config, robot_config = ConfigFiles.load_configs("yam", *ConfigFiles.WIDE_LIMITS["yam"])
    station = ReplayRobot(control_config, robot_config, lambda: current)
    reference = {**MotionHelpers.observation(current), "session_id": "previous-session"}
    with pytest.raises(ValueError, match="different station session"):
        station.hold_snapshot(reference)
