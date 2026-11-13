import threading
from dataclasses import replace

import pytest

from console_rig import ConsoleRig, MarkedCameras, StandInModel
from stand_in_cameras import StandInCameras
from waiting import Waiting

ENTER_WAIT_S = 3  # for a held reconnect to start
RELEASE_WAIT_S = 5  # a held reconnect waits this long to be released
PREVIEW_JPEG_MAGIC = bytes([255, 216])  # first two bytes of a JPEG
RUNNING_INFERENCE_S = 3  # inference delay of a model that must still be running when the operator acts


class CameraRecovery:
    """What the tests of the cameras' reconnection put in place of what a reconnection must not call."""

    @staticmethod
    def forbid_open(*args, **kwargs):
        """Fails the test: in place of what opens a robot or a camera reader, which a reconnection must not call."""
        pytest.fail("Camera recovery must not open a robot station or a camera reader")


class ControlRecovery:
    """The fault the tests of the recovery of control start from."""

    @staticmethod
    def inject_fault(c):
        """Faults the console's control loop with a transient bus failure of the stand-in driver and returns the
        loop once the console shows the fault."""
        runtime = c.robot.control_loop
        runtime.driver.failed_arm = "left"
        Waiting.until(lambda: c.phase == "Fault" and not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        runtime.driver.failed_arm = None
        return runtime


def test_reconnect_with_motors_off_replaces_the_reader_and_enables_nothing(console, monkeypatch):
    c = console
    old_source, old_preview, station = c.camera_feed.reader, c.preview.encoding_thread, c.robot
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)
    c.commands.run("stop", None)
    Waiting.until(lambda: c.phase == "Stopped", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    Waiting.until(
        lambda: len(c.preview.status()["roles"]) == len(c.robot.CAMERAS), ConsoleRig.WAIT_S, ConsoleRig.POLL_S
    )
    assert old_source.is_closed and not old_preview.thread.is_alive()
    assert c.camera_feed.reader is not old_source and c.robot is station
    assert c.robot.status()["status"] == "OFF"
    assert c.is_stop_latched and c.phase == "Stopped"
    assert c.task_run.task is c.task_run.session is c.task_run.model_client is None
    assert not c.error and not c.preview.camera_error


def test_reconnect_keeps_the_hold_and_the_paused_task(enabled_console, monkeypatch):
    c = enabled_console
    ConsoleRig.pause_after_a_step(c)
    task, session, turn_number, steps = (
        c.task_run.task,
        c.task_run.session,
        c.task_run.turn_number,
        c.task_run.total_steps,
    )
    old_source, station = c.camera_feed.reader, c.robot
    reader = MarkedCameras(c.robot.CAMERAS)
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)
    monkeypatch.setattr(c.preview, "open_reader", lambda: reader)

    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert old_source.is_closed and c.camera_feed.reader is reader and c.robot is station
    assert c.robot.status()["status"] == "READY" and c.phase == "Paused"
    assert c.task_run.task is task and c.task_run.session is session and c.task_run.turn_number == turn_number
    assert not c.error and not c.preview.camera_error
    observation = c.robot.observe()
    assert {stamp["frame_number"] for stamp in observation["camera_timestamps"].values()} == {reader.FRAME_NUMBER}
    Waiting.until(
        lambda: (
            c.preview.get(c.preview.roles[0]) and c.preview.get(c.preview.roles[0]).frame_number == reader.FRAME_NUMBER
        ),
        ConsoleRig.WAIT_S,
        ConsoleRig.POLL_S,
    )

    c.commands.run("resume", None)
    Waiting.until(lambda: c.task_run.total_steps > steps, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.session is session and c.task_run.turn_number > turn_number
    stamps = c.task_run.session.last_turn.observation["camera_timestamps"]
    assert {stamp["frame_number"] for stamp in stamps.values()} == {reader.FRAME_NUMBER}


def test_reconnect_is_refused_while_a_turn_runs(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=RUNNING_INFERENCE_S, is_moving=False))
    old_source = c.camera_feed.reader
    monkeypatch.setattr(c.preview, "open_reader", CameraRecovery.forbid_open)
    c.commands.run("start", {"instruction": "Lift"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    try:
        with pytest.raises(ValueError, match="Wait for the current operation"):
            c.commands.run("reconnect_cameras", None)
        assert c.camera_feed.reader is old_source and not old_source.is_closed and c.phase == "Thinking"
    finally:
        c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_reconnect_is_refused_while_home_awaits_review(enabled_console, monkeypatch):
    c = enabled_console
    old_source = c.camera_feed.reader
    monkeypatch.setattr(c.preview, "open_reader", CameraRecovery.forbid_open)
    c.commands.run("home", None)
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    try:
        with pytest.raises(ValueError, match="Wait for the current operation"):
            c.commands.run("reconnect_cameras", None)
        assert c.camera_feed.reader is old_source and not old_source.is_closed and c.phase == "Confirm home"
    finally:
        c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": False})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_stop_during_reconnect_stops_while_the_arms_hold(enabled_console, monkeypatch):
    c = enabled_console
    ConsoleRig.pause_after_a_step(c)
    task, session = c.task_run.task, c.task_run.session
    entered, release = threading.Event(), threading.Event()
    reader = StandInCameras(c.robot.CAMERAS)

    def delayed():
        entered.set()
        assert release.wait(RELEASE_WAIT_S)
        return reader

    monkeypatch.setattr(c.preview, "open_reader", delayed)
    c.commands.run("reconnect_cameras", None)
    try:
        assert entered.wait(ENTER_WAIT_S)
        assert c.robot.status()["status"] == "READY"
        assert c.commands.run("stop", None)["stop_requested"]
        assert c.is_stop_latched and c.cancellation.is_set()
    finally:
        release.set()
    Waiting.until(lambda: not c.is_busy and c.phase == "Stopped", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert reader.is_closed and c.camera_feed.reader is None
    assert c.robot.status()["status"] == "READY"
    assert c.task_run.task is task and c.task_run.session is session


def test_failed_reconnect_is_visible_and_requires_explicit_retry(console, monkeypatch):
    c = console
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("Top camera still disconnected")

    monkeypatch.setattr(c.preview, "open_reader", fail)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "Top camera still disconnected" in c.error
    assert c.snapshot()["cameras"]["error"] == "Top camera still disconnected"
    assert c.camera_feed.reader is None and c.robot.status()["status"] == "OFF"
    assert len(calls) == 1
    monkeypatch.setattr(c.preview, "open_reader", lambda: StandInCameras(c.robot.CAMERAS))
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert not c.error and c.camera_feed.reader is not None


def test_failed_reconnect_keeps_the_hold_and_the_paused_task(enabled_console, monkeypatch):
    c = enabled_console
    ConsoleRig.pause_after_a_step(c)
    task, session = c.task_run.task, c.task_run.session

    def fail():
        raise RuntimeError("Top camera still disconnected")

    monkeypatch.setattr(c.preview, "open_reader", fail)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy and c.phase == "Paused", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "Top camera still disconnected" in c.error and c.camera_feed.reader is None
    assert c.robot.status()["status"] == "READY"
    assert c.task_run.task is task and c.task_run.session is session
    with pytest.raises(ValueError, match="Restore live camera feeds"):
        c.commands.run("resume", None)


@pytest.mark.parametrize("cancel_kind", ["stop", "connection_loss"])
def test_cancel_during_reconnect_discards_new_source(console, monkeypatch, cancel_kind):
    c = console
    entered, release = threading.Event(), threading.Event()
    new_source = StandInCameras(c.robot.CAMERAS)
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)

    def delayed():
        entered.set()
        assert release.wait(RELEASE_WAIT_S)
        return new_source

    monkeypatch.setattr(c.preview, "open_reader", delayed)
    c.commands.run("reconnect_cameras", None)
    try:
        assert entered.wait(ENTER_WAIT_S)
        with pytest.raises(ValueError, match="current operation"):
            c.commands.run("reconnect_cameras", None)
        if cancel_kind == "stop":
            c.commands.run("stop", None)
        else:
            c.stop(reason="Control connection lost")
    finally:
        release.set()
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert new_source.is_closed and c.camera_feed.reader is None
    assert c.robot.status()["status"] == "OFF"
    assert c.task_run.task is c.task_run.session is c.task_run.model_client is None
    assert c.is_stop_latched == (cancel_kind == "stop")


def test_old_reader_must_close_before_a_new_one_starts(console, monkeypatch):
    c = console
    source = c.camera_feed.reader
    close = source.close

    def fail_close():
        raise RuntimeError("Camera reader did not stop")

    monkeypatch.setattr(source, "close", fail_close)
    monkeypatch.setattr(c.preview, "open_reader", CameraRecovery.forbid_open)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "Camera reader did not stop" in c.error
    assert c.camera_feed.reader is source
    monkeypatch.setattr(source, "close", close)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_single_camera_error_keeps_other_previews_but_stops_policy(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=RUNNING_INFERENCE_S, is_moving=True))
    c.commands.run("start", {"instruction": "Lift"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    # the stand-in cameras draw their frames in read_preview, which read itself goes through
    read_preview = c.camera_feed.reader.read_preview

    def partial():
        values = read_preview()[0]
        values.pop("top")
        return values, {"top": "Device disconnected"}

    monkeypatch.setattr(c.camera_feed.reader, "read_preview", partial, raising=False)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.cancellation.is_set() and c.task_run.total_steps == 0
    assert c.robot.status()["status"] == "READY"
    assert c.preview.status()["errors"] == {"top": "Device disconnected"}
    assert c.preview.get("top") is None
    assert c.preview.get("left").jpeg.startswith(PREVIEW_JPEG_MAGIC)
    previous = c.preview.get("right").frame_number
    Waiting.until(lambda: c.preview.get("right").frame_number > previous, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    with pytest.raises(ValueError, match="Restore live camera feeds"):
        c.commands.run("resume", None)


def test_reconnect_publishes_progress_but_cannot_enable_until_frames_arrive(console, monkeypatch):
    c = console
    source = StandInCameras(c.robot.CAMERAS)
    original_read, original_preview = source.read, source.read_preview

    def unavailable(*args, **kwargs):
        raise RuntimeError("Camera unavailable: top reconnecting")

    monkeypatch.setattr(source, "read", unavailable)
    monkeypatch.setattr(source, "read_preview", lambda: ({}, {"top": "Reconnecting camera"}))
    monkeypatch.setattr(c.preview, "open_reader", lambda: source)
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    Waiting.until(lambda: c.preview.status()["error"] is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.camera_feed.reader is source and not source.is_closed
    assert c.robot.status()["status"] == "OFF"
    with pytest.raises((ValueError, RuntimeError), match="[Cc]amera"):
        c.commands.run("enable", None)
    monkeypatch.setattr(source, "read", original_read)
    monkeypatch.setattr(source, "read_preview", original_preview)
    Waiting.until(lambda: c.preview.status()["error"] is None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.status()["status"] == "OFF"
    assert c.task_run.task is c.task_run.session is c.task_run.model_client is None
    assert not c.is_busy


def test_browser_loss_during_camera_maintenance_keeps_recovery_running(enabled_console, monkeypatch):
    c = enabled_console
    ConsoleRig.pause_after_a_step(c)
    task, session = c.task_run.task, c.task_run.session
    entered, release = threading.Event(), threading.Event()
    source = StandInCameras(c.robot.CAMERAS)
    monkeypatch.setattr(type(c.robot), "open", CameraRecovery.forbid_open)

    def delayed():
        entered.set()
        assert release.wait(RELEASE_WAIT_S)
        return source

    monkeypatch.setattr(c.preview, "open_reader", delayed)
    c.commands.run("reconnect_cameras", None)
    try:
        assert entered.wait(ENTER_WAIT_S)
        c.lease.owner = "disconnected-client"
        c.lease.heartbeat_at = 0
        Waiting.until(lambda: c.lease.owner is None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert not c.cancellation.is_set()
    finally:
        release.set()
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.camera_feed.reader is source and not source.is_closed
    assert c.robot.status()["status"] == "READY" and c.phase == "Paused"
    assert c.task_run.task is task and c.task_run.session is session


@pytest.mark.parametrize("robot_name", ["yam"])
def test_recovery_checks_hardware_only_on_the_control_thread_and_records_each_fault(enabled_console, monkeypatch):
    c = enabled_console
    runtime = ControlRecovery.inject_fault(c)
    before = len(runtime.driver.sent_targets)
    check = runtime.driver.read_for_recovery
    callers = []

    def checked():
        callers.append(threading.current_thread().name)
        return check()

    monkeypatch.setattr(runtime.driver, "read_for_recovery", checked)
    for _ in range(2):
        c.commands.run("recover_control", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert c.phase == "Holding" and c.error is None
        assert c.robot.status()["status"] == "READY"
        assert len(runtime.driver.sent_targets) == before
        assert callers and set(callers) == {"station-control"}
        ControlRecovery.inject_fault(c)
        before += 1
    assert len(list(c.root.glob("control-fault-*.json"))) == 3


@pytest.mark.parametrize("robot_name", ["yam"])
def test_failed_recovery_preserves_fault_and_reports_hardware_cause(enabled_console, monkeypatch):
    c = enabled_console
    runtime = ControlRecovery.inject_fault(c)
    before, original = len(runtime.driver.sent_targets), runtime.fault

    def failed():
        raise RuntimeError("Recovery blocked: right motor 4: over temperature")

    monkeypatch.setattr(runtime.driver, "read_for_recovery", failed)
    c.commands.run("recover_control", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.phase == "Fault" and "over temperature" in c.error
    assert runtime.fault == original and runtime.status == "FAULT"
    assert len(runtime.driver.sent_targets) == before and not runtime.driver.is_closed
    with pytest.raises(ValueError, match="stable enabled"):
        c.commands.run("home", None)


@pytest.mark.parametrize("robot_name", ["yam"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_stop_latch_survives_recovery_and_stop_interrupts_check(enabled_console, when):
    c = enabled_console
    runtime = ControlRecovery.inject_fault(c)
    before = len(runtime.driver.sent_targets)
    if when == "before":
        c.commands.run("stop", None)
        Waiting.until(lambda: c.phase == "Fault", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("recover_control", None)
    if when == "during":
        Waiting.until(lambda: runtime.snapshot()["recovery"] is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        c.commands.run("stop", None)
        Waiting.until(lambda: not c.is_busy and c.phase == "Fault", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert runtime.status == "FAULT"
        c.commands.run("recover_control", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.phase == "Stopped" and c.is_stop_latched and runtime.status == "HOLD"
    assert len(runtime.driver.sent_targets) == before
    with pytest.raises(ValueError, match="Reset STOP"):
        c.commands.run("home", None)
    c.commands.run("reset_stop", None)
    assert c.phase == "Holding" and not c.is_stop_latched


@pytest.mark.parametrize("robot_name", ["yam"])
def test_recovery_rejected_when_not_faulted(enabled_console):
    with pytest.raises(ValueError, match="faulted robot"):
        enabled_console.commands.run("recover_control", None)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_home_after_recovery_still_requires_review_before_motion(enabled_console):
    c = enabled_console
    runtime = ControlRecovery.inject_fault(c)
    c.commands.run("recover_control", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    before = len(runtime.driver.sent_targets)
    c.commands.run("home", {"side": "both"})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert len(runtime.driver.sent_targets) == before
    c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": True})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.phase == "Home complete" and c.robot.status()["status"] == "READY"
    assert c.robot.observe()["joint_positions"][[6, 13]] == pytest.approx([1, 1])


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_faulted_robot_rejects_an_unknown_command_and_can_be_disabled(enabled_console, monkeypatch):
    c = enabled_console
    runtime = c.robot.control_loop
    read = runtime.driver.read
    monkeypatch.setattr(
        runtime.driver,
        "read",
        lambda: {s: replace(v, sampled_monotonic_s=v.sampled_monotonic_s - 1) for s, v in read().items()},
    )
    Waiting.until(lambda: runtime.status == "FAULT", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    targets = dict(runtime.targets)
    assert "hold_recovery" not in runtime.snapshot()
    with pytest.raises(ValueError, match="Unknown command"):
        c.commands.run("recover_hold", {"confirm": True})
    assert runtime.targets == targets
    c.commands.run("disable", {"confirm": True})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.status()["status"] == "OFF"
