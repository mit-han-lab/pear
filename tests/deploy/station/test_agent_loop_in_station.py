import copy
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from config_files import ConfigFiles
from pear.deploy.station.agent_loop import agent_loop
from pear.deploy.station.agent_loop.agent_loop import ModelCall, StationAgentLoop, TurnEventHandler
from pear.deploy.station.agent_loop.clients.client import SessionLost
from pear.deploy.station.control.joint.stability import StabilityError
from pear.deploy.station.control.motion.results import MotionResult
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE
from pear.deploy.station.control.schemas import Observation
from pear.deploy.station.exceptions import OperatorPaused
from stand_in_cameras import StandInCameras
from stand_in_clients import ModelReply, StandInDemonstrationClient, StandInScriptedClient
from stand_in_robots import StandInRobots, StandInYAMRobot
from waiting import Waiting
from yam_rigs import MotionHelpers, ReplayRobot

# how long the model may take to answer a turn, where a test gives no time of its own: the session's and its
# client's, as a station gives both the model's one timeout
MODEL_TIMEOUT_S = 60.0
SYSTEM_PROMPT = "system"  # what a stand-in model is prompted with, which it never reads
WAIT_SECONDS = 3  # for the stand-in robot to reach an expected state
POLL_SECONDS = 0.005  # between checks of that state


class Steps:
    """A step of an agent loop with what a test leaves out: no pause, nobody hearing its events and a new attempt."""

    @staticmethod
    def run(
        session: StationAgentLoop,
        *,
        cancellation: threading.Event | None = None,
        on_event: TurnEventHandler | None = None,
    ) -> MotionResult:
        """Runs one step of ``session``, which ``cancellation`` pauses and ``on_event`` hears when a test gives
        them."""
        return session.step(
            cancellation=threading.Event() if cancellation is None else cancellation,
            on_event=(lambda kind, data: None) if on_event is None else on_event,
            attempt_id=uuid4().hex,
        )


class ModelCalls:
    """A model call as the agent loop makes one, with what a test leaves out: the timeout of MODEL_TIMEOUT_S, no pause
    and nothing that stops the request."""

    @staticmethod
    def make(
        robot: Any,
        reference_observation: Observation | None,
        call: Callable[[], Any],
        *,
        timeout_s: float = MODEL_TIMEOUT_S,
        cancellation: threading.Event | None = None,
    ) -> ModelCall:
        """Returns the model call that makes ``call`` while ``robot`` holds ``reference_observation``, paused by
        ``cancellation`` when a test gives it."""
        return ModelCall(
            robot,
            reference_observation,
            call,
            timeout_s=timeout_s,
            cancellation=threading.Event() if cancellation is None else cancellation,
            abort=lambda: None,
        )


class HoldSnapshots:
    """Control snapshots of both YAM arms, written by hand."""

    @staticmethod
    def idle_snapshot() -> dict[str, Any]:
        """Returns a fresh snapshot of both arms holding at zero, at rest."""
        return {
            "status": "HOLD",
            "fault": None,
            "generation": 0,
            "targets": {side: [0.0] * 7 for side in ("left", "right")},
            "arms": {
                side: {"position": [0.0] * 7, "velocity": [0.0] * 7, "age_seconds": 0.0} for side in ("left", "right")
            },
        }


@pytest.fixture
def yam_station() -> Iterator[BaseRobot]:
    """The open YAM robot of a model's session, on stand-in hardware and drawn cameras."""
    with StandInRobots.make("yam") as instance:
        yield instance


@pytest.fixture
def station() -> Iterator[BaseRobot]:
    """The open YAM robot on stand-in hardware in joint mode, without cameras."""
    # how late a sequence of targets may run, as the tests that act through a robot allow
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        yield station


@pytest.fixture
def moving_hold(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[BaseRobot, threading.Event]]:
    """The open YAM robot, settled in its hold, whose left J4 reads 0.1 rad/s once the event it comes with is set."""
    config = ConfigFiles.load(
        "yam", "control.settle_window_s=0.1", "control.settle_timeout_s=0.4", "control.deadline_tolerance_s=0.5"
    )

    with StandInYAMRobot(config, cameras=StandInCameras(StandInYAMRobot.CAMERAS)) as station:
        station.wait_stationary()
        runtime = station.control_loop
        active = threading.Event()
        read = runtime.driver.read

        def feedback():
            states = read()
            if active.is_set():
                v = [0.0] * 7
                v[3] = 0.1
                states["left"] = replace(states["left"], velocity=tuple(v))
            return states

        monkeypatch.setattr(runtime.driver, "read", feedback)
        yield station, active


def test_local_round_trip_feedback_follows_the_format(robot_name, robot, action_format, idle_action, tmp_path):
    calls = []
    client = StandInScriptedClient(
        SYSTEM_PROMPT,
        lambda obs, emit, cancel: (calls.append(obs), ModelReply.output(idle_action))[1],
        timeout_s=MODEL_TIMEOUT_S,
    )
    session = StationAgentLoop(
        robot, client, action_format=action_format, timeout_s=MODEL_TIMEOUT_S, directory=tmp_path, reference_images=()
    )

    first, second = Steps.run(session), Steps.run(session)

    assert first.status == second.status == "completed"
    assert first.target_reached and first.is_settled
    assert len(calls) == 2 and session.turn_count == 2
    assert "Previous execution" not in calls[0].text
    # YAM omits feedback for an ordinary completed action; SO-101 always reports the outcome.
    assert ("Previous execution" in calls[1].text) == (robot_name == "so101")
    assert session.previous_result is second
    assert session.last_turn.station_result is second
    assert calls[0].images[0].uuid != calls[1].images[0].uuid
    session.close()
    assert calls[0].images[0].path.exists()  # Closing does not delete task records.


@pytest.mark.parametrize("mode", ["invalid", "cancel", "session", "operation"])
def test_invalid_cancelled_or_stale_action_never_dispatches(
    robot, action_format, idle_action, monkeypatch, mode, tmp_path
):
    cancel = threading.Event()

    def infer(obs, emit, token):
        if mode == "cancel":
            cancel.set()
        elif mode == "session":
            robot.session_id = "other-session"
        elif mode == "operation":
            robot.control_loop.generation += 1
        result = ModelReply.output(idle_action)
        if mode == "invalid":
            result = type(result)("partial JSON")
        return result

    monkeypatch.setattr(robot, "execute", lambda *a, **kw: pytest.fail("Invalid action dispatched"))
    session = StationAgentLoop(
        robot,
        StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S),
        action_format=action_format,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    if mode == "cancel":
        with pytest.raises(OperatorPaused):
            Steps.run(session, cancellation=cancel)
    else:
        result = Steps.run(session, cancellation=cancel)
        assert result.status in ("rejected", "failed") and not result.target_reached
    session.close()


def test_policy_timeout_cancels_worker_and_discards_late_reply(
    robot, action_format, idle_action, monkeypatch, tmp_path
):
    finished = threading.Event()

    def infer(obs, emit, cancel):
        cancel.wait(3)
        finished.set()
        return ModelReply.output(idle_action)

    monkeypatch.setattr(robot, "execute", lambda *a, **kw: pytest.fail("Late action dispatched"))
    session = StationAgentLoop(
        robot,
        StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S),
        action_format=action_format,
        timeout_s=0.05,
        directory=tmp_path,
        reference_images=(),
    )

    result = Steps.run(session)

    assert result.status == "failed" and result.error["code"] == "TimeoutError"
    assert finished.is_set()
    assert session.last_turn.model_output is not None
    session.close()


def test_an_execution_stopped_midway_is_reported_failed(yam_station, monkeypatch, tmp_path):
    original = yam_station.control_loop.dispatch_joint_move

    def interrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        yam_station.stop()
        return value

    monkeypatch.setattr(yam_station.control_loop, "dispatch_joint_move", interrupted)
    session = StationAgentLoop(
        yam_station,
        StandInDemonstrationClient(
            SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=BIMANUAL_POSE.name, delay_s=0, is_moving=True
        ),
        action_format=BIMANUAL_POSE,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    result = Steps.run(session)

    assert result.status == "failed" and not result.target_reached
    session.close()


def test_disconnection_does_not_invent_feedback(robot, action_format, monkeypatch, tmp_path):
    def disconnected(*args, **kwargs):
        robot.close()
        raise RuntimeError("hardware session was closed")

    monkeypatch.setattr(robot, "execute", disconnected)
    session = StationAgentLoop(
        robot,
        StandInDemonstrationClient(
            SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=action_format.name, delay_s=0, is_moving=False
        ),
        action_format=action_format,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    result = Steps.run(session)

    assert result.status == "failed" and not result.target_reached and not result.is_settled
    assert result.measured_position is None and result.tracking_error is None
    session.close()


def test_cancelled_dispatch_records_measured_outcome_for_standalone_resume(
    robot, action_format, idle_action, monkeypatch, tmp_path
):
    cancel = threading.Event()
    calls = []
    execute = robot.execute

    def interrupted(*args, **kwargs):
        cancel.set()
        raise OperatorPaused("Stopped during dispatch")

    monkeypatch.setattr(robot, "execute", interrupted)
    session = StationAgentLoop(
        robot,
        StandInScriptedClient(
            SYSTEM_PROMPT,
            lambda obs, emit, token: (calls.append(obs), ModelReply.output(idle_action))[1],
            timeout_s=MODEL_TIMEOUT_S,
        ),
        action_format=action_format,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    try:
        with pytest.raises(OperatorPaused):
            Steps.run(session, cancellation=cancel)
        assert session.last_turn.is_dispatched
        assert session.previous_result.status == "failed"
        assert session.previous_result.measured_position is not None

        monkeypatch.setattr(robot, "execute", execute)
        cancel.clear()
        assert Steps.run(session, cancellation=cancel).status == "completed"
        assert '"execution_requested": true' in calls[1].text
        assert '"status": "failed"' in calls[1].text
        assert "no action was executed" not in calls[1].text
    finally:
        session.close()


def test_pause_does_not_hide_session_lost_during_cleanup(robot, action_format, tmp_path):
    cancel = threading.Event()

    def infer(obs, emit, request_cancel):
        cancel.set()
        assert request_cancel.wait(3)
        raise SessionLost("Native interrupt could not be confirmed")

    session = StationAgentLoop(
        robot,
        StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S),
        action_format=action_format,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    try:
        with pytest.raises(SessionLost, match="could not be confirmed"):
            Steps.run(session, cancellation=cancel)
        assert session.blocked_reason
        assert not session.last_turn.is_dispatched
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["joint", "unknown"])
def test_a_station_that_is_not_cartesian_is_refused_with_its_mode(action_format, mode, tmp_path):
    station = SimpleNamespace(action_mode=mode, session_id="session")

    with pytest.raises(ValueError, match=f"Cartesian Station, not one in {mode} mode"):
        StationAgentLoop(
            station,
            StandInDemonstrationClient(SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=action_format.name),
            action_format=action_format,
            timeout_s=MODEL_TIMEOUT_S,
            directory=tmp_path,
            reference_images=(),
        )


def test_a_station_that_is_not_open_is_refused(action_format, tmp_path):
    station = SimpleNamespace(action_mode="cartesian", session_id=None)

    with pytest.raises(ValueError, match="Open a Cartesian Station"):
        StationAgentLoop(
            station,
            StandInDemonstrationClient(SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=action_format.name),
            action_format=action_format,
            timeout_s=MODEL_TIMEOUT_S,
            directory=tmp_path,
            reference_images=(),
        )


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_a_timeout_must_be_finite_and_positive(action_format, timeout, tmp_path):
    station = SimpleNamespace(action_mode="cartesian", session_id="session")

    with pytest.raises(ValueError, match="finite and positive"):
        StationAgentLoop(
            station,
            StandInDemonstrationClient(SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=action_format.name),
            action_format=action_format,
            timeout_s=timeout,
            directory=tmp_path,
            reference_images=(),
        )


@pytest.mark.parametrize("joint,offset", [(0, 0.02), (6, 0.02)])
def test_position_change_during_inference_keeps_reply(station, joint, offset):
    reference = station.observe()
    original = station.control_loop.snapshot
    drift = threading.Event()

    def snapshot():
        result = copy.deepcopy(original())
        if drift.is_set():
            q = list(result["arms"]["right"]["position"])
            q[joint] += offset
            result["arms"]["right"]["position"] = q
        return result

    def call():
        drift.set()
        return {"answer": "ready"}

    # the robot reads the control loop's snapshot, and so does the loop's own wait when the robot closes: the
    # replacement ends with the call
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(station.control_loop, "snapshot", snapshot)
        assert ModelCalls.make(station, reference, call).run() == {"answer": "ready"}

    assert station.control_loop.generation == reference["control_snapshot"]["generation"]


def test_fault_during_model_call_rejects_its_reply(station):
    reference = station.observe()
    original = station.control_loop.snapshot
    changed = threading.Event()

    def snapshot():
        result = original()
        if changed.is_set():
            result["status"] = "FAULT"
        return result

    def start():
        changed.set()
        return {"session_id": "late"}

    # the robot reads the control loop's snapshot, and so does the loop's own wait when the robot closes: the
    # replacement ends with the call
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(station.control_loop, "snapshot", snapshot)
        with pytest.raises(RuntimeError):
            ModelCalls.make(station, reference, start).run()


def test_timeout_and_late_answer_never_move(station):
    before = len(station.control_loop.driver.sent_targets)

    def delayed():
        time.sleep(0.3)
        return {"answer": "too late"}

    with pytest.raises(TimeoutError):
        ModelCalls.make(station, station.observe(), delayed, timeout_s=0.05).run()
    time.sleep(0.35)

    assert len(station.control_loop.driver.sent_targets) == before
    assert station.control_loop.snapshot()["status"] == "HOLD"


def test_network_failure_never_moves(station):
    before = len(station.control_loop.driver.sent_targets)

    def failed():
        raise ConnectionError("network down")

    with pytest.raises(ConnectionError):
        ModelCalls.make(station, station.observe(), failed).run()

    assert len(station.control_loop.driver.sent_targets) == before


@pytest.mark.parametrize("reply_before_fault", [False, True])
def test_model_wait_discards_reply_on_actual_hold_fault(moving_hold, reply_before_fault):
    station, active = moving_hold
    runtime = station.control_loop
    release = threading.Event()
    before = len(runtime.driver.sent_targets)

    def call():
        active.set()
        Waiting.until(lambda: not runtime.snapshot()["arrival"]["settled"], WAIT_SECONDS, POLL_SECONDS)
        if not reply_before_fault:
            assert release.wait(3)
        return {"answer": "must not execute"}

    try:
        with pytest.raises(StabilityError, match="Persistent motion during HOLD") as caught:
            ModelCalls.make(station, station.observe(), call, timeout_s=3).run()
        assert caught.value.metrics["violations"]
        assert runtime.status == "FAULT"
    finally:
        release.set()

    assert len(runtime.driver.sent_targets) == before + 1


def test_policy_reports_stability_failure_and_emits_original_diagnostic(moving_hold, tmp_path):
    station, active = moving_hold
    runtime = station.control_loop
    before = len(runtime.driver.sent_targets)
    events = []

    def infer(request, emit, cancel):
        active.set()
        Waiting.until(lambda: not runtime.snapshot()["arrival"]["settled"], WAIT_SECONDS, POLL_SECONDS)
        return StandInDemonstrationClient(
            SYSTEM_PROMPT, timeout_s=MODEL_TIMEOUT_S, action_format=BIMANUAL_POSE.name, delay_s=0, is_moving=False
        ).infer(request, on_event=emit, cancellation=cancel)

    policy = StationAgentLoop(
        station,
        StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S),
        action_format=BIMANUAL_POSE,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    result = Steps.run(policy, on_event=lambda kind, value: events.append((kind, value)))

    assert result.status == "failed" and not result.is_settled and not result.target_reached
    assert result.error["code"] == "StabilityError"
    diagnostic = [value for kind, value in events if kind == "diagnostic"]
    assert diagnostic == [runtime.snapshot()["fault_details"]]
    assert not any(kind == "action" for kind, _ in events)
    assert len(runtime.driver.sent_targets) == before + 1


def test_operator_cancel_is_not_relabelled_as_a_hold_fault(moving_hold):
    station, _ = moving_hold
    runtime = station.control_loop
    cancel = threading.Event()
    before = len(runtime.driver.sent_targets)

    def call():
        cancel.set()
        return "cancelled reply"

    with pytest.raises(OperatorPaused):
        ModelCalls.make(station, station.observe(), call, cancellation=cancel).run()

    assert runtime.status == "HOLD" and runtime.fault is None
    assert runtime.snapshot()["fault_details"] is None
    assert len(runtime.driver.sent_targets) == before


@pytest.mark.parametrize(
    "mode",
    [
        "brief_spikes",
        "velocity_burst",
        "reply_during_burst",
        "recorded_burst",
        "sustained_velocity",
        "marginal_sustained_velocity",
        "oscillation",
        "span_during_recovery",
        "drift_during_recovery",
        "stale_during_recovery",
        "fault_during_recovery",
        "timeout_during_recovery",
    ],
)
def test_model_wait_keeps_velocity_checks_and_accepts_position_changes(monkeypatch, stand_in_clock, mode):
    config, robot_config = ConfigFiles.load_configs("yam")
    clock = stand_in_clock
    requested, stops = [], []
    recorded = json.loads((ConfigFiles.YAM_RECORDED_CASES / "j4_hold_velocity_burst.json").read_text())

    def sleep(dt):
        clock.now += dt

    def status():
        state = HoldSnapshots.idle_snapshot()
        tick = int((clock.now - 100 + 1e-8) * config.rate_hz)
        left = state["arms"]["left"]
        # Isolated encoder spikes: sampled slowly they alias into sustained velocity,
        # sampled at the control rate the window mean stays below the limit.
        left["velocity"][3] = 0.139194139194 if tick % 10 == 1 else 0.01
        elapsed = clock.now - requested[0] if requested else 0
        if requested and "sustained_velocity" in mode:
            # A marginal RMS with a tiny position span: a sustained inconsistency
            # must still fail, even near the limit.
            left["velocity"][3] = 0.062725 if mode.startswith("marginal") else 0.1
        if requested and mode == "oscillation":
            left["position"][3] = 0.004 * (-1) ** tick
        if "burst" in mode or "recovery" in mode:
            left["velocity"][3] = 0.3 if 0.2 <= elapsed < 0.32 else 0.01
        if mode == "recorded_burst":
            index = min(tick, len(recorded["samples"]) - 1)
            left["position"][3], left["velocity"][3] = recorded["samples"][index]
        if elapsed >= 0.34:
            if mode == "span_during_recovery":
                left["position"][3] = 0.004 * (-1) ** tick
            elif mode == "drift_during_recovery":
                left["position"][3] = 0.006
            elif mode == "stale_during_recovery":
                left["age_seconds"] = config.state_timeout_s + 0.01
            elif mode == "fault_during_recovery":
                state.update(status="FAULT", fault="test fault")
        return state

    class DelayedQueue:
        def __init__(self, **kwargs):
            self.value = None

        def put(self, value):
            self.value = value
            self.available_at = clock.now + (
                0.36 if mode in ("reply_during_burst", "recorded_burst", "timeout_during_recovery") else 2
            )

        def get(self, *, timeout):
            sleep(timeout)
            if clock.now < self.available_at:
                raise queue.Empty
            return self.value

        def get_nowait(self):
            if not hasattr(self, "available_at") or clock.now < self.available_at:
                raise queue.Empty
            return self.value

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    # the robot's waits and the stability window read the monotonic clock itself: the stand-in clock is every
    # module's for the test
    monkeypatch.setattr(agent_loop, "queue", SimpleNamespace(Queue=DelayedQueue, Empty=queue.Empty))
    monkeypatch.setattr(agent_loop, "threading", SimpleNamespace(Thread=InlineThread, Event=threading.Event))
    station = ReplayRobot(config, robot_config, status, on_stop=lambda: stops.append(True))

    def call():
        requested.append(clock.now)
        return {"answer": "ready"}

    if mode in (
        "brief_spikes",
        "velocity_burst",
        "reply_during_burst",
        "recorded_burst",
        "oscillation",
        "span_during_recovery",
        "drift_during_recovery",
    ):
        assert ModelCalls.make(station, MotionHelpers.observation(HoldSnapshots.idle_snapshot()), call).run() == {
            "answer": "ready"
        }
        assert not stops
        if mode not in ("brief_spikes", "recorded_burst"):
            # A completed reply must wait until the original limits pass again.
            minimum_wait = 0.7
            assert clock.now - requested[0] >= minimum_wait - 1e-8
    elif mode in ("stale_during_recovery", "fault_during_recovery"):
        with pytest.raises(RuntimeError, match="Stale|FAULT"):
            ModelCalls.make(station, MotionHelpers.observation(HoldSnapshots.idle_snapshot()), call).run()
        assert clock.now - requested[0] < 0.38  # Fault/stale feedback reject immediately.
        assert stops and all(stops)
    elif mode == "timeout_during_recovery":
        with pytest.raises(TimeoutError):
            ModelCalls.make(station, None, call, timeout_s=0.4).run()
        assert stops and all(stops)
    else:
        with pytest.raises(StabilityError, match="left J4") as failure:
            ModelCalls.make(station, None, call).run()
        assert any(v["metric"] == "velocity_mean" for v in failure.value.metrics["violations"])
        assert stops and all(stops)
        assert clock.now - requested[0] >= config.settle_timeout_s
    assert len(requested) == 1  # Recovery never reissues model requests.


@pytest.mark.parametrize("mode", ["brief_span", "sustained_span", "drift", "stale", "fault"])
def test_model_wait_accepts_position_changes_but_rejects_stale_or_faulted_feedback(monkeypatch, stand_in_clock, mode):
    config, robot_config = ConfigFiles.load_configs("yam")
    clock = stand_in_clock
    requested, stops = [], []

    def sleep(dt):
        clock.now += dt

    def status():
        state = HoldSnapshots.idle_snapshot()
        elapsed = clock.now - requested[0] if requested else 0
        right = state["arms"]["right"]
        if mode == "brief_span" and 0.18 < elapsed < 0.26:
            right["position"][5] = 0.00305176
        elif mode == "sustained_span" and elapsed > 0.18:
            right["position"][5] = 0.0016 * (-1) ** int(elapsed * 50)
        elif elapsed > 0.18:
            if mode == "drift":
                right["position"][5] = 0.006
            elif mode == "stale":
                right["age_seconds"] = 1
            elif mode == "fault":
                state.update(status="FAULT", fault="fixture")
        return state

    class DelayedQueue:
        def __init__(self, **kwargs):
            pass

        def put(self, value):
            self.value = value
            self.available_at = clock.now + 0.35

        def get(self, *, timeout):
            sleep(timeout)
            if clock.now < self.available_at:
                raise queue.Empty
            return self.value

        def get_nowait(self):
            if not hasattr(self, "available_at") or clock.now < self.available_at:
                raise queue.Empty
            return self.value

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    # the robot's waits and the stability window read the monotonic clock itself: the stand-in clock is every
    # module's for the test
    monkeypatch.setattr(agent_loop, "queue", SimpleNamespace(Queue=DelayedQueue, Empty=queue.Empty))
    monkeypatch.setattr(agent_loop, "threading", SimpleNamespace(Thread=InlineThread, Event=threading.Event))
    station = ReplayRobot(config, robot_config, status, on_stop=lambda: stops.append(True))

    def call():
        requested.append(clock.now)
        return "ready"

    def run():
        return ModelCalls.make(
            station,
            MotionHelpers.observation(HoldSnapshots.idle_snapshot()),
            call,
        ).run()

    if mode in ("brief_span", "sustained_span", "drift"):
        assert run() == "ready"
        assert 0.35 <= clock.now - requested[0] < 0.4
        assert not stops
    else:
        with pytest.raises(RuntimeError):
            run()
        assert stops and all(stops)
        assert clock.now - requested[0] < 0.23
    assert len(requested) == 1
