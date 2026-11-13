import json
import threading
import time

import pytest

from config_files import ConfigFiles
from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.control.cameras import CameraFeed
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.schemas import RobotStatus
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.lease import ControlLease
from pear.deploy.station.gui.console.phase import Phase
from pear.deploy.station.gui.console.watchdog import Watchdog
from recorded_task import RecordingWaits
from stand_in_robots import StandInRobots
from waiting import Waiting

LONG_INFERENCE_S = 3  # inference delay of a model still thinking when the test acts
RACE_WAIT_S = 0.2  # long enough for a racing write that wrongly ignores the lock to land
BARRIER_S = 5  # for both racing threads to arrive
CLOSE_WAIT_S = 0.2  # long enough for a close that wrongly ignored a running read to finish
JOIN_S = 5  # for the threads of a test of the camera feed
TASK_TIMEOUT_S = 60  # for a task or a Home of the SO-101 to finish what a test waits for
THINKING_S = 2  # inference delay of the model of the recording's pause and resume
HOLDING_MODEL_S = 10  # inference delay of a model that thinks for the whole test
# a model of the station's configuration whose action format the robot lacks
UNKNOWN_FORMAT_MODEL = (
    "default_model=lab",
    "models.lab.label=Lab",
    "models.lab.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
    "models.lab.client.base_url=http://127.0.0.1:9000/v1",
    "models.lab.client.model=m",
    "models.lab.action_format=pose7",
)


class HeldReader:
    """A camera reader whose read waits for its release."""

    FRAMES = {"top": {"rgb": None, "received_monotonic_s": 0.0, "frame_number": 1}}  # what a read returns
    RELEASE_WAIT_S = 5  # a read waits this long to be released

    def __init__(self):
        self.reading, self.release = threading.Event(), threading.Event()
        self.is_closed = False

    def read(self):
        """Returns FRAMES once released; raises when the reader was closed meanwhile."""
        self.reading.set()
        assert self.release.wait(self.RELEASE_WAIT_S)
        if self.is_closed:
            raise RuntimeError("Reader closed during a read")
        return self.FRAMES

    def close(self):
        """Closes the reader."""
        self.is_closed = True


@pytest.mark.parametrize("robot_name", ["yam"])
def test_stop_ends_the_wait_even_when_the_station_stop_fails(enabled_console, monkeypatch):
    c = enabled_console
    before = c.robot.observe()["joint_positions"].copy()
    stop = c.robot.stop

    def failing_stop():
        monkeypatch.setattr(c.robot, "stop", stop)
        raise RuntimeError("Stop request failed")

    c.commands.run("home", {"side": "left"})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    monkeypatch.setattr(c.robot, "stop", failing_stop)

    with pytest.raises(RuntimeError, match="Stop request failed"):
        c.stop()
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)


def test_a_console_starts_off_and_previews_every_camera(console, robot_name):
    assert type(console.robot) is StandInRobots.robot_class(robot_name)
    assert console.robot.status()["status"] == "OFF"
    Waiting.until(
        lambda: len(console.preview.status()["roles"]) == len(console.robot.CAMERAS),
        ConsoleRig.WAIT_S,
        ConsoleRig.POLL_S,
    )
    assert console.preview.get(console.robot.CAMERAS[-1].role).jpeg.startswith(bytes([255, 216]))


@pytest.mark.parametrize("robot_name", ["yam"])
def test_failed_stop_motion_still_latches(enabled_console, monkeypatch):
    c = enabled_console
    recording_stops = []
    stop = c.robot.stop

    def failing_stop():
        monkeypatch.setattr(c.robot, "stop", stop)
        raise RuntimeError("Stop request failed")

    monkeypatch.setattr(c.robot, "stop", failing_stop)
    monkeypatch.setattr(c.recordings, "stop", lambda reason, stopped_at=None: recording_stops.append(reason))

    with pytest.raises(RuntimeError, match="Stop request failed"):
        c.commands.run("stop", None)

    assert c.is_stop_latched and c.phase == "Stopping" and c.cancellation.is_set()
    assert recording_stops == ["STOP MOTION activated"]
    with pytest.raises(ValueError, match="Reset"):
        c.commands.run("start", {"instruction": "Hold"})


def test_browser_loss_pauses_without_disabling(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=LONG_INFERENCE_S, is_moving=True))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.lease.owner = "abandoned-tab"
    c.lease.heartbeat_at = time.monotonic() - ControlLease.LEASE_S - 1
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.status()["status"] == "READY" and c.cancellation.is_set()


def test_camera_failure_pauses_the_turn_and_keeps_the_motor_hold(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=LONG_INFERENCE_S, is_moving=True))
    c.commands.run("start", {"instruction": "Lift"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)

    def broken(*args, **kwargs):
        raise RuntimeError("Camera disconnected")

    # the stand-in cameras draw their frames in read_preview, which read itself goes through
    monkeypatch.setattr(c.camera_feed.reader, "read_preview", broken)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.cancellation.is_set() and c.task_run.total_steps == 0
    assert c.robot.status()["status"] == "READY"
    assert c.preview.status()["error"] == "Camera disconnected"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_idle_phase_never_replaces_a_phase_a_command_shows_meanwhile(console, monkeypatch):
    c = console
    c.watchdog.close()  # its one check below runs on a watchdog of the test
    c.set_phase(Phase.STOPPING)
    idle_for, commands = Phase.idle_for, []

    def racing_idle_for(*args, **kwargs):
        # A command shows its phase while the watchdog chooses the idle one.
        command = threading.Thread(target=c.set_phase, args=(Phase.ENABLING,))
        command.start()
        command.join(RACE_WAIT_S)
        commands.append(command)
        return idle_for(*args, **kwargs)

    monkeypatch.setattr(Phase, "idle_for", racing_idle_for)
    watchdog = Watchdog(c)
    try:
        Waiting.until(lambda: commands, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    finally:
        watchdog.close()
    commands[0].join()

    assert c.phase == Phase.ENABLING


@pytest.mark.parametrize("robot_name", ["yam"])
def test_an_expired_holder_is_paused_before_another_tab_takes_control(enabled_console, monkeypatch):
    c = enabled_console
    c.watchdog.close()  # its one check below runs on a watchdog of the test
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=LONG_INFERENCE_S, is_moving=True))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: c.phase == Phase.THINKING, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.lease.heartbeat_at = time.monotonic() - ControlLease.LEASE_S - 1

    assert not c.lease.claim("new-tab") and not c.lease.heartbeat(ConsoleRig.CLIENT_ID)
    assert c.is_busy and not c.cancellation.is_set()
    watchdog = Watchdog(c)
    try:
        Waiting.until(lambda: c.cancellation.is_set(), ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    finally:
        watchdog.close()
    assert c.cancellation.is_set() and c.lease.claim("new-tab")
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_fault_found_by_two_threads_is_saved_once(console, monkeypatch):
    c = console
    status, arrivals = c.robot.status, threading.Barrier(2, timeout=BARRIER_S)

    def faulted(with_snapshot=False):
        if not with_snapshot:
            return status()
        arrivals.wait()  # both threads have read the fault before either saves it
        return {"status": RobotStatus.FAULT, "fault": "Injected fault", "fault_details": {}, "generation": 1}

    monkeypatch.setattr(c.robot, "status", faulted)
    savers = [threading.Thread(target=c.save_control_diagnostic) for _ in range(2)]
    for saver in savers:
        saver.start()
    for saver in savers:
        saver.join()

    assert len(list(c.root.glob("control-fault-*.json"))) == 1


@pytest.mark.parametrize("robot_name", ["yam"])
def test_failed_home_shows_its_details(enabled_console, monkeypatch):
    c = enabled_console
    details = {"stage": "homing", "error": "Joint limit", "completed_stages": ["opening"], "results": {}}

    def fail(**options):
        raise HomeFailed("Joint limit", details)

    monkeypatch.setattr(c.robot, "home", fail)
    c.commands.run("home", {"side": "left"})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)

    assert c.phase == Phase.HOME_FAILED and c.home_status == {**details, "status": "failed"}
    saved = list(c.root.glob("home-failed-*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text()) == c.home_status


@pytest.mark.parametrize("robot_name", ["yam"])
def test_polls_reuse_a_recent_check_and_a_forced_one_builds_a_client(console, monkeypatch):
    c, built = console, []

    def factory(model, instruction):
        built.append(model)
        return StandInModel(c)(model, instruction)

    monkeypatch.setattr(c, "new_client", factory)
    first = c.check_model()

    assert c.check_model() == first and built == [c.model]
    assert c.check_model(is_forced=True) == first and built == [c.model, c.model]
    assert c.snapshot()["model_health"] == first


def test_a_read_in_progress_finishes_before_its_reader_closes():
    reader = HeldReader()
    feed = CameraFeed(reader)
    frames = []
    read = threading.Thread(target=lambda: frames.append(feed.read()))
    read.start()
    assert reader.reading.wait(JOIN_S)

    close = threading.Thread(target=feed.close)
    close.start()
    close.join(CLOSE_WAIT_S)
    assert close.is_alive() and not reader.is_closed
    reader.release.set()
    read.join(JOIN_S)
    close.join(JOIN_S)
    assert frames == [HeldReader.FRAMES] and reader.is_closed and feed.reader is None


def test_reads_without_a_reader_raise():
    feed = CameraFeed(None)
    with pytest.raises(RuntimeError, match="Camera unavailable"):
        feed.read()
    with pytest.raises(RuntimeError, match="Camera unavailable"):
        feed.read_preview()


@pytest.mark.parametrize("robot_name", ["yam"])
def test_observation_after_the_reader_closed_raises(enabled_console):
    enabled_console.camera_feed.close()
    with pytest.raises(RuntimeError, match="Camera unavailable"):
        enabled_console.robot.observe()


@pytest.mark.parametrize("robot_name", ["yam"])
def test_production_opens_cameras_only_and_never_falls_back(robot_name, tmp_path, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("Test camera disconnected")

    monkeypatch.setattr(StandInRobots.robot_class(robot_name), "CAMERA_SOURCE_CLASS", unavailable)
    with ConsoleRig(robot_name, tmp_path) as c:
        state = c.snapshot()
        assert c.robot.status()["status"] == "OFF"
        assert c.camera_feed.reader is None and state["cameras"]["error"] == "Test camera disconnected"
        assert c.model == ConsoleRig.MODEL
        assert all(set(m) == {"id", "label", "action_format"} for m in state["models"])
        with pytest.raises(ValueError, match="Cameras are unavailable"):
            c.commands.run("enable", None)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_controller_pause_resume_toggle_and_no_motion_segment(enabled_console, monkeypatch):
    c = enabled_console
    c.commands.run("recording", {"enabled": True})
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=THINKING_S, is_moving=True))
    try:
        c.commands.run("start", {"instruction": "Lift both arms"})
        Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        first = c.recordings.active_segment
        label = c.models_config.models[ConsoleRig.MODEL].label
        assert c.task_run.task["model_label"] == label
        assert first.snapshot()["model_label"] == label
        saved_task = json.loads((c.root / c.task_run.task["id"] / "task.json").read_text())
        assert saved_task["model_label"] == label
        with pytest.raises(ValueError, match="pause"):
            c.commands.run("recording", {"enabled": False})
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        RecordingWaits.await_finished(first)
        assert first.snapshot()["status"] == "saved", first.snapshot()
        assert first.snapshot()["no_motion"]
        assert not (first.directory / "motion.mp4").exists()
        c.commands.run("resume", None)
        Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        second = c.recordings.active_segment
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        RecordingWaits.await_finished(second)
        assert first.manifest["id"] == "001" and second.manifest["id"] == "002"
        assert second.snapshot()["status"] == "saved", second.snapshot()
        assert second.snapshot()["motion"] == "motion.mp4"
        assert len(c.recordings.manifests_of_task(c.task_run.task["id"])) == 2
        c.commands.run("recording", {"enabled": False})
        c.commands.run("resume", None)
        Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert c.recordings.active_segment is None
        c.commands.run("pause", None)
        Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
        assert len(c.recordings.manifests_of_task(c.task_run.task["id"])) == 2
    finally:
        c.commands.run("pause", None)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_failed_encoder_pauses_the_task_and_leaves_the_robot_ready(enabled_console, monkeypatch):
    c = enabled_console
    c.commands.run("recording", {"enabled": True})
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=HOLDING_MODEL_S))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    job = c.recordings.active_segment
    job.terminate()
    Waiting.until(lambda: not c.is_busy and c.recordings.active_segment is None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "Recording failed" in c.error
    assert c.robot.status()["status"] == "READY"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_pause_during_recorder_start_cannot_launch_policy(enabled_console, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    errors = []

    def delayed_start(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(enabled_console.recordings, "start", delayed_start)

    def start():
        try:
            enabled_console.commands.run("start", {"instruction": "Hold"})
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=start)
    worker.start()
    assert entered.wait(5)
    enabled_console.commands.run("pause", None)
    release.set()
    worker.join(5)
    assert errors and not enabled_console.is_busy
    assert enabled_console.task_run.session is None and enabled_console.task_run.total_steps == 0


@pytest.mark.parametrize("robot_name", ["yam"])
def test_events_carry_the_recording_clock(console):
    before = time.monotonic()
    event = console.emit("notice", {"message": "clock"}, is_saved=False)
    assert before <= event["monotonic_s"] <= time.monotonic()
    assert abs(event["time"] - time.time()) < 1


@pytest.mark.parametrize("robot_name", ["so101"])
def test_profile_lists_one_wrist_camera_and_one_format(console):
    state = console.snapshot()
    assert state["profile"]["name"] == "so101"
    assert state["profile"]["cameras"] == [{"role": "wrist", "label": "Wrist Camera"}]
    assert state["profile"]["formats"] == ["single_arm_pose"]
    assert state["profile"]["home_sides"] == [] and not state["profile"]["home_keep_grippers"]
    demos = {m["id"]: m["action_format"] for m in state["models"] if m["id"].startswith(ConsoleRig.MODEL)}
    assert demos == {ConsoleRig.MODEL: "single_arm_pose"}
    assert state["action_format"] == "single_arm_pose"
    Waiting.until(lambda: "wrist" in console.preview.status()["roles"], ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


@pytest.mark.parametrize("robot_name", ["so101"])
def test_demo_task_runs_in_the_model_format(console):
    c = console
    c.commands.run("model", {"model": ConsoleRig.MODEL})
    c.commands.run("enable", None)
    Waiting.until(lambda: not c.is_busy and c.robot.status()["status"] == "READY", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("start", {"instruction": "Bob the arm"})
    Waiting.until(lambda: c.task_run.total_steps >= 2, TASK_TIMEOUT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    root = c.root / c.task_run.task["id"]
    task = json.loads((root / "task.json").read_text())
    session = json.loads((root / "session.json").read_text())
    assert "action_format" not in task and session["action_format"] == "single_arm_pose"
    observations = sorted(root.glob("*-observation.json"))
    shown = json.loads(json.loads(observations[0].read_text())["text"].splitlines()[0])
    assert list(shown) == ["arm", "gripper"] and list(shown["arm"]) == ["x", "y", "z", "pitch", "roll"]
    assert list(json.loads(observations[0].read_text())["images"]) == ["wrist"]


@pytest.mark.parametrize("robot_name", ["so101"])
def test_home_always_opens_the_gripper(console):
    c = console
    c.commands.run("enable", None)
    Waiting.until(lambda: not c.is_busy and c.robot.status()["status"] == "READY", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    with pytest.raises(ValueError, match="gripper choice"):
        c.commands.run("home", {"open_grippers": False})


@pytest.mark.parametrize("robot_name", ["so101"])
def test_home_runs_after_review_without_side_options(console):
    c = console
    c.commands.run("enable", None)
    Waiting.until(lambda: not c.is_busy and c.robot.status()["status"] == "READY", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("home", None)
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert "active_sides" not in c.home_review.plan
    c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": True})
    Waiting.until(lambda: not c.is_busy, TASK_TIMEOUT_S, ConsoleRig.POLL_S)
    assert c.phase == "Home complete" and c.home_status["status"] == "completed"


def test_unknown_format_is_refused(tmp_path):
    config = ConfigFiles.load("so101", *UNKNOWN_FORMAT_MODEL)
    assert config.models.lab.action_format == "pose7"
    with pytest.raises(ValueError, match="no format 'pose7'"):
        Console(config, root=tmp_path / "tasks", uploads_dir=tmp_path / "tasks-uploads")
