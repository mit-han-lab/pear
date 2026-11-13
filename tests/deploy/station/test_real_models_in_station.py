import dataclasses
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.gui.console.console import Console

CLIENT = "real-model-test"  # the tab that holds control of the console
HEARTBEAT_S = 0.5  # between two heartbeats of that tab
STEPS = 3  # real answers executed and settled after the interrupted one
ANSWER_WAIT_S = 360  # for one real model answer
OPERATION_WAIT_S = 30  # for enabling, pausing, clearing or disabling
POLL_S = 0.03  # between two looks at the console
STREAMED = ("reasoning_delta", "answer_delta")  # the events that carry the model's streamed text
INSTRUCTION = "Hold the current pose. Keep every arm and gripper unchanged."  # the task the model is given


class RealModels:
    """The real models the test talks to, named by environment variables."""

    FILE = "PEAR_STATION_MODELS"  # a station's own file, such as an operator's, whose models are the real ones
    MODEL = "PEAR_STATION_MODEL"  # a model of that file; by default the file's default model

    @classmethod
    def load(cls) -> tuple[Path, ModelsConfig]:
        """Returns the station's file and its models; the test is skipped when none is named."""
        if not os.environ.get(cls.FILE):
            pytest.skip(f"Set {cls.FILE} to a station's file with the real models to test")
        path = Path(os.environ[cls.FILE]).resolve()

        return path, ModelsConfig.from_station_config(BaseRobot.load_config(path, ()))


# the test talks to the models' services, which only a machine that reaches them has
pytestmark = pytest.mark.skipif(
    RealModels.FILE not in os.environ, reason=f"needs real models: {RealModels.FILE} names a station's file"
)


class RealModelRun:
    """What the test does with its console besides the commands: the tab's heartbeat, a look at the streamed text and
    the waits."""

    @staticmethod
    def heartbeat(c: Console, beat: threading.Event) -> None:
        """Renews the tab's control of the console until ``beat`` is set."""
        while not beat.wait(HEARTBEAT_S):
            c.lease.heartbeat(CLIENT)

    @staticmethod
    def has_streamed(c: Console) -> bool:
        """Whether the current task's model has streamed any reasoning or answer text."""
        with c.lock:
            return any(event["type"] in STREAMED and event["task_id"] == c.task_run.task["id"] for event in c.events)

    @staticmethod
    def wait(c: Console, predicate: Callable[[], bool], timeout: float = OPERATION_WAIT_S) -> None:
        """Returns once ``predicate`` holds; fails the test with the console's phase and error after ``timeout``."""
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError(f"Timed out during {c.phase}: {c.error}")
            time.sleep(POLL_S)


@pytest.fixture
def models() -> tuple[Path, ModelsConfig]:
    """The station's file whose models the test talks to, and its models."""
    return RealModels.load()


@pytest.fixture
def controller(models: tuple[Path, ModelsConfig], tmp_path: Path) -> Iterator[Console]:
    """A console of the station's robot on stand-in hardware and drawn cameras with the station's real model, holding
    control with a heartbeat."""
    path, config = models
    robot = config.robot_type
    model = os.environ.get(RealModels.MODEL) or config.default_model
    # the robot's station of the tests with the station's models, whose paths are already their file's
    station_config = OmegaConf.merge(
        ConfigFiles.load(robot.NAME),
        {
            "default_model": model,
            "models": {name: dataclasses.asdict(offered_model) for name, offered_model in config.models.items()},
        },
    )
    c = Console(station_config, root=tmp_path / "tasks", uploads_dir=tmp_path / "uploads")
    c.commands.run("recording", {"enabled": False})
    c.lease.claim(CLIENT)
    beat = threading.Event()
    thread = threading.Thread(target=RealModelRun.heartbeat, args=(c, beat), daemon=True)
    thread.start()

    try:
        yield c
    finally:
        c.close()
        beat.set()
        thread.join(HEARTBEAT_S * 4)


def test_a_task_paused_mid_answer_resumes_in_one_session_and_clears(controller):
    c = controller
    assert c.check_model()["status"] == "ok"
    c.commands.run("enable", None)
    RealModelRun.wait(c, lambda: not c.is_busy)
    assert c.robot_status() == "READY"

    c.commands.run("start", {"instruction": INSTRUCTION})
    RealModelRun.wait(c, lambda: RealModelRun.has_streamed(c), ANSWER_WAIT_S)
    c.commands.run("pause", None)
    RealModelRun.wait(c, lambda: not c.is_busy)
    assert c.task_run.total_steps == 0 and c.error is None and not c.task_run.session.blocked_reason

    for step in range(1, STEPS + 1):
        c.commands.run("resume", None)
        RealModelRun.wait(c, lambda step=step: c.task_run.total_steps >= step or not c.is_busy, ANSWER_WAIT_S)
        c.commands.run("pause", None)
        RealModelRun.wait(c, lambda: not c.is_busy)
        assert c.task_run.total_steps >= step, c.error
        assert c.robot_status() == "READY"

    journal = (c.root / c.task_run.task["id"] / "events.jsonl").read_text().splitlines()
    sessions = [event for event in map(json.loads, journal) if event["type"] == "session"]
    assert len(sessions) == 1
    c.commands.run("clear", None)
    RealModelRun.wait(c, lambda: not c.is_busy)
    assert c.task_run.task is None and c.task_run.session is None
    assert c.check_model()["status"] == "ok"
    c.commands.run("disable", {"confirm": True})
    RealModelRun.wait(c, lambda: not c.is_busy)
    assert c.robot_status() == "OFF"
