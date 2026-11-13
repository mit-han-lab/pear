import os
import subprocess
import sys

import pytest

from repository import REPO

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="a process dies with its parent through prctl (Linux)")
# The check's env class: a call returns its one camera's frame and as many native actions as its action names; the
# task "broken" raises.
ENV = r"""
import os, sys

from PIL import Image

from pear.sim.env import StatelessEnv
from pear.sim.process import SimulatorProcess


class OneCameraEnv(StatelessEnv):
    START_TIMEOUT_S = 60.0  # a child imports Python and pear in seconds
    ROLLOUT_TIMEOUT_S = 60.0
    CAMERAS = ("front",)

    @classmethod
    def setup_env(cls, context):
        return [SimulatorProcess("sim", cls, sys.executable, [], dict(os.environ), os.getcwd(), context.log_dir)], 1

    @classmethod
    def task_name(cls, env_args):
        return env_args["task"]

    def __init__(self, task):
        self.task = task

    def rollout(self, history, action):
        if self.task == "broken":
            raise ValueError("the env raised")
        return {"images": [Image.new("RGB", (4, 4))], "executed": 0 if action is None else action["steps"]}
"""
TOOL_CONFIG = """
tools:
  - class_name: pear.envs.tool.EnvironmentTool
    config:
      env_class: one_camera_env.OneCameraEnv
"""


@pytest.fixture
def check(tmp_path):
    """Runs python -m pear.sim.checker on OneCameraEnv with the given env args and action; returns the finished process."""
    (tmp_path / "one_camera_env.py").write_text(ENV)
    (tmp_path / "tool_config.yaml").write_text(TOOL_CONFIG)
    environment = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{REPO}"}

    def run(env_args: str, *action: str) -> subprocess.CompletedProcess:
        command = [sys.executable, "-m", "pear.sim.checker", "--tool-config", str(tmp_path / "tool_config.yaml")]
        command += ["--env-args", env_args, "--log-dir", str(tmp_path / "logs"), *action]
        return subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True)

    return run


def test_an_episode_s_initial_observation_passes_with_the_frames_of_the_cameras(check):
    finished = check('{"task": "a"}')

    assert finished.returncode == 0, finished.stderr
    assert "frames [(4, 4)]" in finished.stdout


def test_an_action_passes_when_the_process_carried_it_out(check):
    finished = check('{"task": "a"}', "--action", '{"steps": 3}')

    assert finished.returncode == 0, finished.stderr
    assert "3 native actions" in finished.stdout


def test_an_action_that_moves_nothing_fails(check):
    finished = check('{"task": "a"}', "--action", '{"steps": 0}')

    assert finished.returncode != 0
    assert "moved nothing" in finished.stderr


def test_an_env_that_raises_fails_with_its_traceback(check):
    finished = check('{"task": "broken"}')

    assert finished.returncode != 0
    assert "the env raised" in finished.stderr
