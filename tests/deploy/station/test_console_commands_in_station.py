import threading
import time

import pytest

from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.gui.console.phase import Phase
from stand_in_clients import StandInDemonstrationClient
from waiting import Waiting

STOP_REQUEST_LIMIT_S = 0.2  # a pause or STOP must be requested within this
LONG_INFERENCE_S = 3  # inference delay of a model still thinking when the test acts
STALE_WAKE_S = 0.2  # long enough for a review that wrongly took the wake as an answer to end
RACE_WAIT_S = 0.2  # long enough for a racing write that wrongly ignores the lock to land


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_wake_without_an_answer_keeps_waiting(enabled_console):
    c = enabled_console
    c.commands.run("home", {"side": "left"})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)

    c.home_review.answered.set()
    time.sleep(STALE_WAKE_S)

    assert c.is_busy and c.home_review.plan is not None
    c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": False})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)


def test_clear_waits_for_old_work_and_keeps_history(enabled_console, monkeypatch):
    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=LONG_INFERENCE_S, is_moving=True))
    c.commands.run("start", {"instruction": "Old task"})
    Waiting.until(lambda: c.phase == "Thinking", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    old = c.task_run.task["id"]
    c.commands.run("clear", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.task is None and c.task_run.session is None
    assert (c.root / old / "task.json").exists()
    monkeypatch.setattr(c, "new_client", StandInModel(c, delay_s=0.03, is_moving=False))
    c.commands.run("start", {"instruction": "New task"})
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.task["id"] != old and c.task_run.task["instruction"] == "New task"


def test_failed_episode_cleanup_retains_task_for_explicit_retry(enabled_console):
    c = enabled_console
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    original = c.task_run.model_client.close

    def broken():
        raise RuntimeError("Model cleanup unavailable")

    c.task_run.model_client.close = broken
    old = c.task_run.task["id"]
    c.commands.run("clear", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.task["id"] == old and "cleanup unavailable" in c.error
    c.task_run.model_client.close = original
    c.commands.run("clear", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.task_run.task is None


def test_stop_latches_and_reset_does_not_move(enabled_console):
    c = enabled_console
    c.commands.run("stop", None)
    Waiting.until(lambda: c.phase == "Stopped", ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    before = c.robot.observe()["joint_positions"].copy()
    with pytest.raises(ValueError, match="Reset"):
        c.commands.run("start", {"instruction": "Hold"})
    c.commands.run("reset_stop", None)
    assert not c.is_stop_latched
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)


@pytest.mark.parametrize("robot_name", ["yam"])
@pytest.mark.parametrize("confirm", [True, False])
def test_home_moves_nothing_until_confirmed_and_opens_the_grippers_when_confirmed(enabled_console, confirm):
    c = enabled_console
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("home", {"side": "both"})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)
    c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": confirm})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"][[6, 13]] == pytest.approx([1, 1] if confirm else before[[6, 13]])
    assert c.robot.status()["status"] == "READY"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_home_keep_grippers_leaves_gripper_targets(enabled_console):
    c = enabled_console
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("home", {"side": "both", "open_grippers": False})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.home_review.plan["open_grippers"] is False and c.home_review.plan["opening_commands"] == {}
    c.commands.run("home_confirm", {"plan_id": c.home_review.plan["plan_id"], "confirm": True})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.observe()["joint_positions"][[6, 13]] == pytest.approx(before[[6, 13]])
    assert c.home_status["status"] == "completed"
    assert all(v["reached"] and v["gripper_kept"] for v in c.home_status["results"].values())
    with pytest.raises(ValueError, match="gripper choice"):
        c.commands.run("home", {"side": "both", "open_grippers": "no"})
    assert c.robot.status()["status"] == "READY"


def test_stop_during_home_confirmation_never_dispatches(enabled_console):
    c = enabled_console
    before = c.robot.observe()["joint_positions"].copy()
    c.commands.run("home", {"side": "left"})
    Waiting.until(lambda: c.home_review.plan is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    plan = c.home_review.plan["plan_id"]
    assert c.commands.run("stop", None)["stop_request_seconds"] < STOP_REQUEST_LIMIT_S
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    with pytest.raises(ValueError, match="no longer current"):
        c.commands.run("home_confirm", {"plan_id": plan, "confirm": True})
    assert c.robot.observe()["joint_positions"] == pytest.approx(before)
    assert c.is_stop_latched


@pytest.mark.parametrize("robot_name", ["yam"])
def test_stop_during_reset_stays_latched(enabled_console, monkeypatch):
    c = enabled_console
    c.commands.run("stop", None)
    Waiting.until(lambda: c.phase == Phase.STOPPED, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    robot_status, stops = c.robot_status, []

    def racing_status():
        # STOP MOTION arrives while the reset reads the robot.
        stop = threading.Thread(target=c.stop, kwargs={"is_latching": True})
        stop.start()
        stop.join(RACE_WAIT_S)
        stops.append(stop)
        return robot_status()

    monkeypatch.setattr(c, "robot_status", racing_status)
    c.commands.run("reset_stop", None)
    stops[0].join()

    assert c.is_stop_latched


@pytest.mark.parametrize("command", ["step", "environment", "arms"])
def test_an_unknown_command_is_rejected(console, command):
    with pytest.raises(ValueError, match="Unknown command"):
        console.commands.run(command, None)
    assert console.robot.status()["status"] == "OFF"


def test_explicit_disable_releases_motors_even_when_model_cleanup_is_offline(enabled_console, monkeypatch):
    class OfflineOnCleanup(StandInDemonstrationClient):
        def close(self):
            raise ConnectionError("Model endpoint disconnected")

    c = enabled_console
    monkeypatch.setattr(c, "new_client", StandInModel(c, OfflineOnCleanup, delay_s=0.01, is_moving=False))
    c.commands.run("start", {"instruction": "Hold"})
    Waiting.until(lambda: c.task_run.total_steps >= 1, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("disable", {"confirm": True})
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    assert c.robot.status()["status"] == "OFF"
    assert c.phase == "Disabled" and c.task_run.task is not None
    assert c.task_run.session.blocked_reason
    c.task_run.model_client.close = lambda: None  # Explicit retry can finish cleanup.
    assert "Motors disabled; model session cleanup failed" in c.error
