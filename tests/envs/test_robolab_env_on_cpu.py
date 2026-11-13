from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from pear.envs.robolab import env as robolab_env
from pear.envs.robolab.env import RoboLabEnv

ENV_ARGS = {"task": "BananaInBowlTask", "instruction_type": "default"}
ACTION = {"arm": {"z": 1.0}}  # what the controller receives (the stand-in reads nothing of it)


class Frames:
    """A camera's frames as RoboLab's observation holds them: a tensor, read through ``cpu().numpy()``."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __getitem__(self, index: int) -> Frames:
        return Frames(self.array[index])

    def cpu(self) -> Frames:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class Entities(dict):
    """A RobolabEnv's scene: its entities by name, and the envs' origins."""

    env_origins = np.array([[0.0, 0.0, 0.1]])


class OfficialEnv:
    """RoboLab's RobolabEnv of a task as RoboLabEnv uses it, ended with ``success`` once ``terminated``; ``reads``
    lists what the results read of it."""

    def __init__(self) -> None:
        self.terminated, self.success, self.subtasks = False, False, []
        self.calls, self.reads = [], []
        robot = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=np.array([[0.0, 0.0, 0.3]])))
        self.scene = Entities(robot=robot)

    @property
    def all_terminated(self) -> bool:
        self.reads.append("all_terminated")

        return self.terminated

    def reset(self):
        self.calls.append("reset")
        frames = {key: Frames(np.full((1, 72, 128, 3), 7, np.uint8)) for key in RoboLabEnv.CAMERA_NAME_BY_KEY}

        return {"image_obs": frames}, {}

    def get_env_results(self):
        self.reads.append("results")

        return [{"success": self.success}]

    def subtask_infos(self):
        self.reads.append("subtasks")

        return self.subtasks

    def close(self):
        self.calls.append("close")


class Controller:
    """MotionController as RoboLabEnv calls it: each move ends with ``outcome``, each state is a new reading."""

    outcome = "reached"
    table_height_m = None  # what the last one was made with

    def __init__(self, env, observation, table_height_m) -> None:
        Controller.table_height_m = table_height_m
        self.observation, self.reading_n = observation, 0

    def move(self, action):
        return self.outcome

    def state(self):
        self.reading_n += 1

        return {"arm": {"z": float(self.reading_n)}}


@pytest.fixture
def official_env(monkeypatch) -> OfficialEnv:
    """The env RoboLabEnv builds, through stand-ins of the RoboLab modules it imports."""
    official_env = OfficialEnv()
    env_cfg = types.SimpleNamespace(instruction="put the banana in the bowl")
    world = types.SimpleNamespace(get_bbox=lambda name, env_id: ([(0.0, 0.0, 0.5), (0.0, 0.0, 0.9)], None))
    modules = {
        "robolab": {},
        "robolab.constants": {"set_output_dir": lambda path: None},
        "robolab.core": {},
        "robolab.core.environments": {},
        "robolab.core.environments.runtime": {"create_env": lambda task, **options: (official_env, env_cfg)},
        "robolab.core.logging": {},
        "robolab.core.logging.results": {"get_all_env_subtask_infos": lambda env: env.subtask_infos()},
        "robolab.core.world": {},
        "robolab.core.world.world_state": {"get_world": lambda env: world},
        "robolab.registrations": {},
        "robolab.registrations.droid": {},
        "robolab.registrations.droid.auto_env_registrations_abs_ik": {
            "auto_register_droid_abs_ik_envs": lambda task: None
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        package, _, submodule = name.rpartition(".")
        if package:  # ``import robolab.constants`` reads the submodule as its package's attribute
            setattr(sys.modules[package], submodule, module)
    monkeypatch.setattr(robolab_env, "MotionController", Controller)
    monkeypatch.setattr(Controller, "outcome", "reached")

    return official_env


def test_an_episode_runs_on_robolab_s_env_of_its_task(official_env):
    env = RoboLabEnv(**ENV_ARGS)

    open_result, step_result = env.open(), env.step(ACTION)
    env.close()

    assert official_env.calls == ["reset", "reset", "close"]  # reset twice, as RoboLab's run_episode
    assert Controller.table_height_m == pytest.approx(0.7)  # the table's top above the robot's root
    assert [frame.size for frame in open_result["images"]] == [RoboLabEnv.FRAME_SIZE] * len(RoboLabEnv.CAMERAS)
    assert open_result["actions"] == [] and open_result["rewards"] == []
    assert open_result["outcome"] is None and not open_result["done"]
    assert step_result["actions"] == [ACTION] and step_result["rewards"] == [0.0] and step_result["executed"] == 1
    assert step_result["outcome"] == "reached" and not step_result["done"] and step_result["env_end_reason"] is None
    assert step_result["state"] == {"arm": {"z": 3.0}} and step_result["instruction"] == "put the banana in the bowl"
    assert step_result["subtask_score"] is None and step_result["subtask_info"] is None  # a task without subtasks


@pytest.mark.parametrize(("success", "env_end_reason"), [(True, "success"), (False, "time_limit")])
def test_the_action_that_ends_the_episode_reads_done_with_robolab_s_end(official_env, success, env_end_reason):
    env = RoboLabEnv(**ENV_ARGS)
    env.open()
    official_env.terminated, official_env.success = True, success
    official_env.subtasks = [{"score": 0.5, "info": "one of two"}]

    result = env.step(ACTION)

    assert result["done"] and result["outcome"] == "done" and result["images"] is None
    assert result["rewards"] == [float(success)] and result["env_end_reason"] == env_end_reason
    assert result["subtask_score"] == 0.5 and result["subtask_info"] == "one of two"


def test_a_divergence_of_the_physics_ends_the_episode_at_the_state_before_the_action(official_env):
    env = RoboLabEnv(**ENV_ARGS)
    env.open()
    Controller.outcome = "diverged"
    reads = list(official_env.reads)

    result = env.step(ACTION)

    assert result["done"] and result["rewards"] == [0.0] and result["env_end_reason"] == "physics_diverged"
    assert result["actions"] == [ACTION] and result["outcome"] == "diverged" and result["images"] is None
    assert result["state"] == {"arm": {"z": 2.0}}  # the reading before the action: the open's was the first
    assert result["subtask_score"] is None and result["subtask_info"] is None
    assert official_env.reads == reads and env.controller.reading_n == 2  # nothing is read after the divergence


def test_the_initial_observation_of_an_env_that_ended_has_no_outcome(official_env):
    official_env.terminated = official_env.success = True

    open_result = RoboLabEnv(**ENV_ARGS).open()

    assert open_result["done"] and open_result["env_end_reason"] == "success" and open_result["outcome"] is None
    assert open_result["actions"] == [] and open_result["rewards"] == [] and open_result["images"] is None


def test_closing_an_env_that_never_opened_does_nothing():
    RoboLabEnv(**ENV_ARGS).close()
