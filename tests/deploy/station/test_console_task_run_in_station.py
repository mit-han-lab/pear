import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop
from pear.deploy.station.agent_loop.clients.schemas import ModelOutput
from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.robots.yam.motion.actions import BimanualDelta
from pear.deploy.station.gui.console.phase import Phase
from pear.deploy.station.gui.console.task_run import TaskRun
from pear.deploy.station.records.task_events import TaskEvents
from pear.deploy.station.records.task_files import TaskFiles
from stand_in_clients import ModelReply, StandInDemonstrationClient
from waiting import Waiting

STOP_REQUEST_LIMIT_S = 0.2  # a pause or STOP must be requested within this
STEP_COUNT = 250  # steps a single session must survive
STEP_TIMEOUT_S = 120  # for all of them
CONTEXT_LIMIT = 2  # turns a session holds before it blocks
EVENT_LIMIT = 500  # events the console keeps
RELEASE_WAIT_S = 3  # a held inference waits this long to be released
STAGE_RELEASE_S = 4  # a held stage waits this long to be released
ENTER_WAIT_S = 5  # for a held inference to start
STAGE_ENTER_S = 8  # for a held stage to start
SETTLE_WINDOW_S = 0.5  # long enough to pause during settling
HELD_INFERENCE_S = 8  # a model held while the arm's reading changes waits this long to be released
HELD_START_S = 5  # for that model to be entered
POSITION_OFFSET = 0.02  # added to the left arm's sixth joint reading while the model is held, within tolerance
OFFSET_JOINT = 5  # the index of that joint in the arm's reading
FAULT_WAIT_S = 3  # for an event the hold-fault tests expect soon
FAULT_START_WAIT_S = 5  # for the model to be entered after a start
WATCHDOG_WAIT_S = 8  # a deferred watchdog STOP waits this long to be released
FAULT_SETTLE_TIMEOUT_S = 0.4  # settling limit of the console of the hold-fault tests
FAULT_SETTLE_WINDOW_S = 0.1  # settling window and Home settle time of that console
MOVING_VELOCITY = 0.1  # reported joint velocity that makes the hold unstable
REPLY_POLL_S = 0.01  # between checks of a held reply
LIFT_Z = 0.3  # z delta of the late reply
REBASED_POSITION = 0.16  # joint position the right arm is moved to before recovery
RESUME_DELAY_S = 0.03  # inference delay of the model that resumes after recovery
SESSION_TIMEOUT_S = 60  # how long the model of a session made by hand may take to answer
# the console of the hold-fault tests settles fast, so that a hold that keeps moving faults soon
HOLD_FAULT_OVERRIDES = (
    f"control.settle_timeout_s={FAULT_SETTLE_TIMEOUT_S}",
    f"control.settle_window_s={FAULT_SETTLE_WINDOW_S}",
    f"robot.home.settle_window_s={FAULT_SETTLE_WINDOW_S}",
)


@pytest.fixture
def holding_gui(robot_name, tmp_path, monkeypatch):
    """An enabled console whose left arm reports a velocity once ``moving`` is set, and whose model, once entered,
    returns a complete late answer when released or cancelled: the console and the four events."""
    with ConsoleRig(robot_name, tmp_path, *HOLD_FAULT_OVERRIDES) as c:
        c.commands.run("enable", None)
        Waiting.until(
            lambda: not c.is_busy and c.robot.status()["status"] == "READY", ConsoleRig.WAIT_S, ConsoleRig.POLL_S
        )
        runtime = c.robot.control_loop
        moving, entered, release, replied = (threading.Event() for _ in range(4))
        read = runtime.driver.read

        def feedback():
            states = read()
            if moving.is_set():
                states["left"] = replace(states["left"], velocity=(0, 0, 0, MOVING_VELOCITY, 0, 0, 0))
            return states

        class LateReply(StandInDemonstrationClient):
            def infer(self, request, *, on_event, cancellation):
                entered.set()
                try:
                    # Cooperate with cancellation but deliberately return a complete late answer.
                    while not release.wait(REPLY_POLL_S) and not cancellation.is_set():
                        pass
                    zero = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
                    return ModelReply.output(
                        BimanualDelta(
                            left_arm={**zero, "z": LIFT_Z},
                            right_arm=zero,
                            left_gripper={"width": 0.0},
                            right_gripper={"width": 0.0},
                        )
                    )
                finally:
                    replied.set()

        monkeypatch.setattr(runtime.driver, "read", feedback)
        monkeypatch.setattr(c, "new_client", StandInModel(c, LateReply))
        try:
            yield c, moving, entered, release, replied
        finally:
            release.set()
            if entered.is_set():
                assert replied.wait(FAULT_WAIT_S)


def test_many_steps_keep_one_session(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0, is_moving=False))
    c.commands.run("start", {"instruction": "Hold both arms"})
    Waiting.until(lambda: c.task_run.total_steps >= STEP_COUNT, STEP_TIMEOUT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.total_steps >= STEP_COUNT
    assert c.robot.status()["status"] == "READY"
    root = c.root / c.task_run.task["id"]
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    sessions = [e for e in events if e["type"] == "session"]
    starts = [e["data"] for e in events if e["type"] == "turn_started"]
    assert [e["turn_number"] for e in starts] == list(range(1, len(starts) + 1))
    assert len(starts) >= STEP_COUNT
    assert len(sessions) == 1
    assert c.task_run.session.turn_count >= STEP_COUNT
    assert len(c.events) <= EVENT_LIMIT


def test_context_capacity_pauses_and_preserves_history(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.02, is_moving=False, context_limit=CONTEXT_LIMIT))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.total_steps == CONTEXT_LIMIT
    assert len([e for e in c.events if e["type"] == "session"]) == 1
    assert c.task_run.session.blocked_reason
    with pytest.raises(ValueError, match="capacity"):
        c.commands.run("resume", None)
    assert c.robot.status()["status"] == "READY"


def test_cancel_keeps_only_current_final_statistics(console):
    c = console
    c.task_run.active_attempt_id = "current"
    cancel = threading.Event()
    cancel.set()
    before = len(c.events)
    c.task_run.on_event(cancel, "current", "answer_delta", {"text": "late"})
    c.task_run.on_event(cancel, "current", "phase", {"phase": "Executing"})
    c.task_run.on_event(cancel, "old", "usage", {"input": 123, "final": True})
    c.task_run.on_event(cancel, "current", "usage", {"input": 123, "final": False})
    c.task_run.on_event(cancel, "old", "performance", {"output_tokens": 12, "final": True})
    c.task_run.on_event(cancel, "current", "performance", {"output_tokens": 12, "final": False})
    assert len(c.events) == before
    c.task_run.on_event(cancel, "current", "usage", {"input": 123, "final": True})
    assert len(c.events) == before + 1
    assert c.events[-1]["type"] == "usage"
    assert c.events[-1]["data"]["attempt"] == "current"
    c.task_run.on_event(cancel, "current", "performance", {"output_tokens": 12, "final": True})
    assert len(c.events) == before + 2
    assert c.events[-1]["type"] == "performance"
    assert c.events[-1]["data"]["attempt"] == "current"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_moving_action_moves_both_arms(enabled_console, monkeypatch):
    c = enabled_console
    before = c.robot.observe()["joint_positions"].copy()
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.08, is_moving=True))
    c.commands.run("start", {"instruction": "Lift"})
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    after = c.robot.observe()["joint_positions"]
    assert np.max(abs(after[:6] - before[:6])) > 0.0001
    assert np.max(abs(after[7:13] - before[7:13])) > 0.0001
    c.commands.run("disable", {"confirm": True})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_pause_during_inference_rejects_late_reply_and_resume_is_fresh(enabled_console, monkeypatch):
    c = enabled_console
    entered, release = threading.Event(), threading.Event()

    class Slow(StandInDemonstrationClient):
        def infer(self, request, *, on_event, cancellation):
            entered.set()
            release.wait(RELEASE_WAIT_S)
            return super().infer(request, on_event=on_event, cancellation=None)

    monkeypatch.setattr(c, "new_client", StandInModel(c, Slow, delay_s=0, is_moving=True))
    initial = c.robot.observe()["joint_positions"].copy()
    c.commands.run("start", {"instruction": "Small lift"})
    assert entered.wait(ENTER_WAIT_S)
    stop = c.commands.run("pause", None)
    assert stop["stop_request_seconds"] < STOP_REQUEST_LIMIT_S
    release.set()
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"] == pytest.approx(initial)
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.05, is_moving=False))
    c.commands.run("resume", None)
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.total_steps >= 1
    observations = [e for e in c.events if e["type"] == "observation"]
    assert len({e["data"]["turn_id"] for e in observations}) >= 2


@pytest.mark.parametrize("stage", ["Observing", "Thinking", "Planning", "Executing"])
def test_pause_at_stage_boundary_revokes_dispatch_and_resume_observes_again(enabled_console, monkeypatch, stage):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.02, is_moving=True))
    entered, release = threading.Event(), threading.Event()
    original = c.task_run.on_event

    def gate(cancel, attempt, kind, data):
        if kind == "phase" and data["phase"] == stage and not entered.is_set():
            entered.set()
            release.wait(STAGE_RELEASE_S)
        return original(cancel, attempt, kind, data)

    monkeypatch.setattr(c.task_run, "on_event", gate)
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("start", {"instruction": "Lift both arms"})
    assert entered.wait(STAGE_ENTER_S)
    assert c.commands.run("pause", None)["stop_request_seconds"] < STOP_REQUEST_LIMIT_S
    release.set()
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)
    assert c.task_run.total_steps == 0
    c.commands.run("resume", None)
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.total_steps >= 1
    ids = [e["data"]["turn_id"] for e in c.events if e["type"] == "observation"]
    assert len(ids) == len(set(ids))


def test_pause_during_settling_fails_the_turn_as_paused_and_resume_steps_again(robot_name, tmp_path, monkeypatch):
    with ConsoleRig(robot_name, tmp_path, f"control.settle_window_s={SETTLE_WINDOW_S}") as c:
        c.commands.run("enable", None)
        Waiting.until(
            lambda: not c.is_busy and c.robot.status()["status"] == "READY", ConsoleRig.WAIT_S, ConsoleRig.POLL_S
        )
        monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.04, is_moving=True))
        c.commands.run("start", {"instruction": "Lift both arms"})
        Waiting.until(lambda: c.snapshot()["phase"] == "Settling", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert c.commands.run("pause", None)["stop_request_seconds"] < STOP_REQUEST_LIMIT_S
        Waiting.until(lambda: not c.is_busy and c.phase == "Paused", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert c.robot.status()["status"] == "READY"
        c.commands.run("resume", None)
        Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert c.task_run.total_steps >= 1
        records = [e for e in c.events if e["type"] == "execution"]
        assert any(e["data"]["error"]["code"] == "OperatorPaused" for e in records if e["data"]["status"] == "failed")


def test_journal_failure_prevents_model_and_motion(enabled_console, monkeypatch):
    c = enabled_console
    original = TaskFiles.write_json

    def fail(path, *args, **kwargs):
        if "-observation.json" in path.name:
            raise OSError("Simulated disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(TaskFiles, "write_json", fail)
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "disk full" in c.error
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)


def test_model_error_does_not_loop_or_disable(enabled_console, monkeypatch):
    c = enabled_console

    class Bad(StandInDemonstrationClient):
        def infer(self, *args, **kwargs):
            raise ValueError("Invalid final JSON")

    monkeypatch.setattr(c, "new_client", StandInModel(c, Bad))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.total_steps == 0 and "Invalid final JSON" in c.error
    assert c.robot.status()["status"] == "READY"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_model_phases_become_phase_members(console):
    c = console
    c.task_run.active_attempt_id = "current"

    for name in ("Observing", "Thinking", "Compacting", "Planning", "Executing"):
        c.task_run.on_event(threading.Event(), "current", "phase", {"phase": name})
        assert c.phase is Phase(name)
    with pytest.raises(ValueError):
        c.task_run.on_event(threading.Event(), "current", "phase", {"phase": "Checking right gripper"})


@pytest.mark.parametrize("robot_name", ["yam"])
@pytest.mark.parametrize("order", ["before_poll", "during_diagnostic", "after_result", "policy_first"])
def test_hold_fault_survives_gui_cancellation_at_each_boundary(holding_gui, monkeypatch, order):
    c, moving, entered, release, replied = holding_gui
    runtime = c.robot.control_loop
    let_watchdog_stop, boundary = threading.Event(), threading.Event()
    deferred_stops = []
    stop, event, holding, step = c.stop, c.task_run.on_event, c.robot.is_holding, StationAgentLoop.step

    def watchdog_stop(**kwargs):
        if threading.current_thread() is c.watchdog.periodic_thread.thread and runtime.status == "FAULT":
            if order == "after_result":
                # _watch owns the GUI lock here. Defer without holding that lock,
                # so result journaling can reach the boundary being tested.
                def deferred():
                    let_watchdog_stop.wait(WATCHDOG_WAIT_S)
                    stop(**kwargs)

                thread = threading.Thread(target=deferred)
                deferred_stops.append(thread)
                thread.start()
                return {"stop_requested": False}
            elif order == "during_diagnostic":
                assert boundary.wait(FAULT_WAIT_S)
        return stop(**kwargs)

    def emit(*args):
        if order == "during_diagnostic" and args[2] == "diagnostic":
            boundary.set()
            assert c.cancellation.wait(FAULT_WAIT_S)
        return event(*args)

    def read(*args, **kwargs):
        settled = holding(*args, **kwargs)
        if order == "before_poll" and entered.is_set() and not boundary.is_set():
            # Pause after a valid read: the next poll must see cancellation even
            # though this already captured snapshot predates the latched fault.
            boundary.set()
            moving.set()
            assert c.cancellation.wait(FAULT_WAIT_S)
        return settled

    def policy_step(*args, **kwargs):
        result = step(*args, **kwargs)
        if order == "after_result":
            boundary.set()
            let_watchdog_stop.set()
            assert c.cancellation.wait(FAULT_WAIT_S)
        return result

    monkeypatch.setattr(c, "stop", watchdog_stop)
    monkeypatch.setattr(c.task_run, "on_event", emit)
    monkeypatch.setattr(c.robot, "is_holding", read)
    monkeypatch.setattr(StationAgentLoop, "step", policy_step)
    before, targets = len(runtime.driver.sent_targets), dict(runtime.targets)
    try:
        c.commands.run("start", {"instruction": "Hold during the offline race test"})
        assert entered.wait(FAULT_START_WAIT_S)
        if order != "before_poll":
            moving.set()
        Waiting.until(lambda: runtime.status == "FAULT" and not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        if order != "policy_first":
            assert boundary.is_set()
        detail = runtime.snapshot()["fault_details"]
        result_files = list((c.root / c.task_run.task["id"]).glob("*-result.json"))
        assert len(result_files) == 1
        result = json.loads(result_files[0].read_text())
        request = c.task_run.session.last_turn
        assert result["turn_id"] == request.attempt_id
        assert result["status"] == "failed" and not result["settled"] and not result["reached"]
        assert result["error"] == {"code": "StabilityError", "message": detail["message"]}
        assert c.task_run.active_turn["phase"] == "Failed" and c.phase == "Fault"
        assert c.task_run.session.turn_count == 1
        assert all(result[k] == v for k, v in c.task_run.session.previous_result.to_record().items())
        assert c.task_run.total_steps == 0
        diagnostics = [e["data"] for e in c.events if e["type"] == "diagnostic"]
        assert len(diagnostics) == 1
        assert {key: diagnostics[0][key] for key in detail} == detail
        assert diagnostics[0]["attempt"] == c.task_run.active_attempt_id
        assert len([e for e in c.events if e["type"] == "execution"]) == 1
        assert not any(e["type"] in ("action", "interrupted") for e in c.events)
        saved = list(c.root.glob("control-fault-*.json"))
        assert saved and all(json.loads(p.read_text())["details"] == detail for p in saved)
        # Cancellation waits for the worker; its complete late reply never executes.
        assert replied.is_set()
        release.set()
        assert replied.wait(FAULT_WAIT_S)
        assert runtime.status == "FAULT"
        assert runtime.driver.sent_targets[before:] == [targets]
        assert c.task_run.session.last_turn.model_output is not None
    finally:
        let_watchdog_stop.set()
        release.set()
        for thread in deferred_stops:
            thread.join(FAULT_WAIT_S)
            assert not thread.is_alive()


@pytest.mark.parametrize("robot_name", ["yam"])
def test_operator_pause_without_fault_keeps_interrupted_result(holding_gui):
    c, _, entered, release, replied = holding_gui
    runtime = c.robot.control_loop
    before = len(runtime.driver.sent_targets)
    c.commands.run("start", {"instruction": "Hold until paused"})
    assert entered.wait(FAULT_START_WAIT_S)
    c.commands.run("pause", None)
    release.set()
    assert replied.wait(FAULT_WAIT_S)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    result = json.loads(next((c.root / c.task_run.task["id"]).glob("*-result.json")).read_text())
    assert result["error"]["code"] == "OperatorPaused"
    assert c.task_run.active_turn["phase"] == "Interrupted"
    assert runtime.status == "HOLD" and runtime.snapshot()["fault_details"] is None
    assert not any(e["type"] in ("action", "diagnostic") for e in c.events)
    assert len(runtime.driver.sent_targets) == before


@pytest.mark.parametrize("robot_name", ["yam"])
def test_fault_before_new_request_does_not_rewrite_previous_turn(holding_gui, monkeypatch, tmp_path):
    c, moving, _, _, _ = holding_gui
    c.task_run.model_client = StandInModel(c, delay_s=0, is_moving=False)(c.model, "Previous completed turn")
    session_directory = tmp_path / "previous-session"
    session_directory.mkdir()
    c.task_run.session = StationAgentLoop(
        c.robot,
        c.task_run.model_client,
        action_format=c.action_format,
        timeout_s=SESSION_TIMEOUT_S,
        directory=session_directory,
        reference_images=(),
    )
    previous = c.task_run.session.step(
        cancellation=threading.Event(), on_event=lambda kind, data: None, attempt_id="previous"
    )
    assert previous.status == "completed"
    old_request, sequence = c.task_run.session.last_turn, c.task_run.session.turn_count
    step = StationAgentLoop.step

    def before_request(*args, **kwargs):
        moving.set()
        assert c.cancellation.wait(FAULT_WAIT_S)
        return step(*args, **kwargs)

    monkeypatch.setattr(StationAgentLoop, "step", before_request)
    runtime = c.robot.control_loop
    before, targets = len(runtime.driver.sent_targets), dict(runtime.targets)
    c.commands.run("start", {"instruction": "Fault before a new observation"})
    Waiting.until(lambda: runtime.status == "FAULT" and not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.session.last_turn is old_request and c.task_run.session.turn_count == sequence
    assert c.task_run.session.previous_result == previous
    assert not list((c.root / c.task_run.task["id"]).glob("*-result.json"))
    assert c.task_run.active_turn["phase"] == "Failed" and c.phase == "Fault"
    assert any(e["type"] == "diagnostic" for e in c.events)
    assert not any(e["type"] in ("action", "execution", "interrupted") for e in c.events)
    assert runtime.driver.sent_targets[before:] == [targets]


@pytest.mark.parametrize("robot_name", ["yam"])
@pytest.mark.parametrize("rebase", [False, True])
def test_recover_retains_task_and_uses_fresh_observation(holding_gui, monkeypatch, rebase):
    c, moving, entered, release, replied = holding_gui
    runtime = c.robot.control_loop
    c.commands.run("start", {"instruction": "Keep this task after recovery"})
    assert entered.wait(FAULT_START_WAIT_S)
    session, task, station_session = c.task_run.session, c.task_run.task, c.robot.session_id
    request, old_cancel = session.last_turn, c.cancellation
    moving.set()
    Waiting.until(
        lambda: runtime.status == "FAULT" and not c.is_busy and c.phase == "Fault",
        ConsoleRig.WAIT_S,
        ConsoleRig.POLL_S,
    )
    assert c.task_run.resume_pending and replied.is_set()
    moving.clear()
    before, targets = len(runtime.driver.sent_targets), dict(runtime.targets)
    if rebase:
        runtime.driver.positions["right"] = (REBASED_POSITION,) + runtime.driver.positions["right"][1:]
        monkeypatch.setattr(
            runtime.driver,
            "send",
            lambda q, velocities=None: runtime.driver.sent_targets.append(dict(q)),
        )
    c.commands.run("recover_control", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.phase == "Paused" and c.robot.status()["status"] == "READY"
    assert c.task_run.session is session and c.task_run.task is task and c.robot.session_id == station_session
    assert c.task_run.total_steps == 0 and old_cancel.is_set() and c.task_run.resume_pending
    assert not runtime.driver.is_closed
    if rebase:
        assert len(runtime.driver.sent_targets) > before
        assert runtime.targets["right"][0] == pytest.approx(REBASED_POSITION)
        assert runtime.targets != targets
    else:
        assert len(runtime.driver.sent_targets) == before
    before, targets = len(runtime.driver.sent_targets), dict(runtime.targets)
    failed_result = json.loads(next((c.root / task["id"]).glob("*-result.json")).read_text())
    observed = []
    demo = StandInModel(c, delay_s=RESUME_DELAY_S, is_moving=False)(c.model, task["instruction"])

    def new_inference(obs, *, on_event, cancellation):
        observed.append(obs)
        return demo.infer(obs, on_event=on_event, cancellation=cancellation)

    monkeypatch.setattr(c.task_run.model_client, "infer", new_inference)
    c.commands.run("resume", None)
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert observed[0].images[0].uuid != request.model_input.images[0].uuid
    assert "StabilityError" in observed[0].text
    assert c.task_run.task is task and c.task_run.session is session
    assert runtime.targets == targets and len(runtime.driver.sent_targets) == before
    saved = [json.loads(p.read_text()) for p in (c.root / task["id"]).glob("*-result.json")]
    assert failed_result in saved


def test_live_metrics_of_the_current_attempt_stay_in_memory_and_only_final_ones_are_journaled(enabled_console):
    c = enabled_console
    c.task_run.task = {"id": "a" * 32, "instruction": "Hold", "model": "demo"}
    (c.root / c.task_run.task["id"]).mkdir()
    c.task_run.active_attempt_id = "attempt-new"
    for i in range(1000):
        c.task_run.on_event(
            c.cancellation,
            "attempt-new",
            "performance",
            dict(request_id="request-new", backend="vllm", output_tokens=i, final=False),
        )
        c.task_run.on_event(c.cancellation, "attempt-new", "usage", {"input": 100, "output": i, "final": False})
    assert len(c.events) <= 500 and c.performance["output_tokens"] == 999
    assert not (c.root / c.task_run.task["id"] / "events.jsonl").exists()
    c.task_run.on_event(c.cancellation, "attempt-old", "performance", dict(request_id="request-old", output_tokens=123))
    assert c.performance["output_tokens"] == 999
    c.task_run.on_event(
        c.cancellation,
        "attempt-new",
        "performance",
        dict(request_id="request-new", backend="vllm", output_tokens=1000, final=True),
    )
    assert len((c.root / c.task_run.task["id"] / "events.jsonl").read_text().splitlines()) == 1
    c.task_run.on_event(c.cancellation, "attempt-new", "usage", {"input": 100, "output": 1000, "final": True})
    records = [json.loads(line) for line in (c.root / c.task_run.task["id"] / "events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in records] == ["performance", "usage"]
    c.task_run.session = None  # This fixture never created a real episode.
    c.stop()
    assert c.performance is None


@pytest.mark.parametrize("robot_name", ["yam"])
@pytest.mark.parametrize("failure", ["planning", "json"])
def test_failure_resume_preserves_session_and_reports_actual_outcome(enabled_console, monkeypatch, failure):
    c = enabled_console
    calls, created = [], []

    class Client(StandInDemonstrationClient):
        def infer(self, obs, *, on_event, cancellation):
            calls.append(obs)
            if failure == "json" and len(calls) == 1:
                return ModelOutput("invalid JSON")
            return super().infer(obs, on_event=on_event, cancellation=cancellation)

    def factory(model, instruction):
        value = StandInModel(c, Client, delay_s=0.03, is_moving=failure == "planning")(model, instruction)
        created.append(value)
        return value

    monkeypatch.setattr(c, "new_client", factory)
    plan = c.robot.planner.plan
    reject = {"value": failure == "planning"}

    def rejected(*args, **kwargs):
        if reject["value"]:
            raise ValueError("Injected planning rejection")
        return plan(*args, **kwargs)

    monkeypatch.setattr(c.robot.planner, "plan", rejected)
    targets = dict(c.robot.control_loop.targets)
    c.commands.run("start", {"instruction": "Keep original context"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    original = c.task_run.session
    assert c.task_run.total_steps == 0 and c.task_run.resume_pending
    assert original.last_turn.model_output is not None
    assert original.previous_result.status == "rejected"
    reject["value"] = False
    c.task_run.model_client.is_moving = False
    c.commands.run("resume", None)
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.session is original and len(created) == 1
    assert calls[1].images[0].uuid != calls[0].images[0].uuid
    assert '"status": "rejected"' in calls[1].text
    assert c.robot.control_loop.targets == targets


@pytest.mark.parametrize("robot_name", ["yam"])
def test_position_change_during_inference_continues_without_manual_resume(enabled_console, monkeypatch):
    c = enabled_console
    entered, release = threading.Event(), threading.Event()

    class Slow(StandInDemonstrationClient):
        def infer(self, request, *, on_event, cancellation):
            entered.set()
            assert release.wait(HELD_INFERENCE_S)
            return super().infer(request, on_event=on_event, cancellation=cancellation)

    monkeypatch.setattr(c, "new_client", StandInModel(c, Slow, delay_s=0.03, is_moving=False))
    hardware = c.robot.control_loop.driver
    read = hardware.read
    bias = {"value": 0.0}

    def feedback():
        states = read()
        q = list(states["left"].position)
        q[OFFSET_JOINT] += bias["value"]
        states["left"] = replace(states["left"], position=tuple(q))
        return states

    monkeypatch.setattr(hardware, "read", feedback)
    targets = dict(c.robot.control_loop.targets)
    c.commands.run("start", {"instruction": "Hold"})
    try:
        assert entered.wait(HELD_START_S)
        episode = c.task_run.session.episode_id
        bias["value"] = POSITION_OFFSET
        Waiting.until(
            lambda: c.robot.control_loop.snapshot()["arms"]["left"]["position"][OFFSET_JOINT] == POSITION_OFFSET,
            ConsoleRig.WAIT_S,
            ConsoleRig.POLL_S,
        )
        release.set()
        Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert not c.task_run.resume_pending and c.task_run.session.episode_id == episode
        assert not c.error
        assert not any(e["type"] == "diagnostic" for e in c.events)
        assert c.robot.control_loop.targets == targets
    finally:
        release.set()
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_a_new_session_is_refused_while_a_session_exists(enabled_console):
    c = enabled_console
    old = SimpleNamespace(episode_id="old")
    c.task_run.session = old

    try:
        with pytest.raises(RuntimeError, match="Clear the existing"):
            c.task_run.new_session()
        assert c.task_run.session is old
    finally:
        c.task_run.session = c.task_run.model_client = None


@pytest.mark.parametrize("robot_name", ["yam"])
def test_ik_retry_observes_again_resets_after_success_and_numbers_every_attempt(enabled_console, monkeypatch):
    c = enabled_console
    execute = c.robot.execute
    attempts = []

    def controlled(action, **kwargs):
        attempts.append(kwargs["reference_observation"]["observation_id"])
        if len(attempts) != 3:
            raise UnreachablePoseError("IK target outside reachable workspace")
        return execute(action, **kwargs)

    monkeypatch.setattr(c.robot, "execute", controlled)
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("start", {"instruction": "Hold both arms"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    turns = 3 + TaskRun.UNREACHABLE_RETRIES + 1
    assert len(attempts) == len(set(attempts)) == turns
    assert c.task_run.total_steps == 1
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)
    events = list(TaskEvents(c.root / c.task_run.task["id"] / "events.jsonl").read_numbered())
    starts = [e["data"] for e in events if e["type"] == "turn_started"]
    assert [e["turn_number"] for e in starts] == list(range(1, turns + 1))
    assert len({e["attempt"] for e in starts}) == turns
    notices = [e["data"] for e in events if e["type"] == "notice"]
    assert [n["retry"] for n in notices] == [1, 2, *range(1, TaskRun.UNREACHABLE_RETRIES + 1)]
    assert notices[-1]["message"].endswith(f"({TaskRun.UNREACHABLE_RETRIES}/{TaskRun.UNREACHABLE_RETRIES}).")
    observations = [e["data"] for e in events if e["type"] == "observation"]
    assert observations[0]["previous_result"] is None
    assert observations[1]["previous_result"]["error"]["code"] == "UnreachablePoseError"
    assert observations[3]["previous_result"]["status"] == "completed"
    assert "Previous execution:" in observations[1]["text"]
    assert "Previous execution:" in observations[2]["text"]
    assert "Previous execution:" not in observations[3]["text"]
    assert "Previous execution:" in observations[4]["text"]
    # Omitting model-facing feedback must not remove any successful execution data.
    root = c.root / c.task_run.task["id"]
    success = starts[2]["attempt"]
    previous = observations[3]["previous_result"]
    saved_result = json.loads((root / f"{success}-result.json").read_text())
    saved_observation = json.loads((root / f"{starts[3]['attempt']}-observation.json").read_text())
    execution = next(e["data"] for e in events if e["type"] == "execution" and e["data"]["attempt"] == success)
    assert saved_observation["previous_result"] == previous
    assert previous["measured_state"] is not None and previous["measured_position"] is not None
    assert all(saved_result[k] == value == execution[k] for k, value in previous.items())
    assert "IK target" in c.error


@pytest.mark.parametrize("failure", [ValueError("Joint limit exceeded"), RuntimeError("Feedback unavailable")])
def test_a_motion_error_other_than_unreachable_is_not_retried(enabled_console, monkeypatch, failure):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(enabled_console.robot, "execute", fail)
    enabled_console.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: not enabled_console.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert len(calls) == 1 and enabled_console.task_run.turn_number == 1
    assert str(failure) in enabled_console.error
