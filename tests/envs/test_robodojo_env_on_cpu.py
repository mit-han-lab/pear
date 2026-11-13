from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from pear.envs.robodojo import env as robodojo_env
from pear.envs.robodojo.env import RoboDojoEnv
from pear.sim.env import EpisodeLost

ENV_ARGS = {"task_name": "stack_bowls", "config_name": "arx_x5", "layout_set": 1, "layout_id": 3, "random_seed": 0}
ACTION = {"left_arm": {"z": 1.0}}  # what the controller receives (the stand-in reads nothing of it)


class UnStableError(Exception):
    """The official stability check's error (utils.cluttered_generator)."""


class PhysXBrokenError(Exception):
    """The official PhysX warning monitor's (src.eval_client.physx_warning_monitor)."""


class Scene:
    """The official env of a scene as RoboDojoEnv uses it, its reset raising ``failure`` if set; each observation is a
    frame the camera dispatcher delivered unless ``stale``."""

    def __init__(self) -> None:
        self.failure, self.stale, self.calls, self.frames = None, False, [], 0
        self.end_flag, self.success, self.unstable_envs = [False], [False], set()
        self.take_action_cnt, self.step_lim = [0], 100

    def reset(self, seed):
        self.calls.append(("reset", seed))
        if self.failure is not None:
            raise self.failure

    def run_reward(self):
        self.calls.append("run_reward")

    def is_episode_end(self):
        return False

    def get_obs(self):
        self.frames += 0 if self.stale else 1
        return {
            "vision": {
                name: {"color": np.full((4, 6, 3), self.frames, np.uint8)} for name in RoboDojoEnv.CAMERA_NAME_BY_KEY
            },
            "instruction": "stack the bowls",
        }

    def close(self):
        self.calls.append("close")


class Controller:
    """MotionController as RoboDojoEnv calls it: each move raises ``failure`` if set, else reaches its goals."""

    failure = None

    def __init__(self, env, observation) -> None:
        pass

    def move(self, action):
        if self.failure is not None:
            raise self.failure
        return "reached"

    def state(self):
        return {"left_arm": {"z": 1.0}}


@pytest.fixture
def scene(monkeypatch) -> Scene:
    """The scene RoboDojoEnv builds, through stand-ins of the official modules it imports."""
    scene = Scene()
    modules = {
        "src": {},
        "src.eval_client": {},
        "src.eval_client.eval_env": {"WsModelClient": None, "create_eval_env": lambda config, app: scene},
        "src.eval_client.physx_warning_monitor": {"PhysXBrokenError": PhysXBrokenError},
        "utils": {},
        "utils.cluttered_generator": {"UnStableError": UnStableError},
        "env": {},
        "env.camera_manager": {},
        "env.camera_manager.capture": {},
        "env.camera_manager.capture.render_sync": {"dispatcher_time": lambda: scene.frames},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(RoboDojoEnv, "_config", lambda self: "the official config")
    monkeypatch.setattr(robodojo_env, "MotionController", Controller)
    monkeypatch.setattr(Controller, "failure", None)

    return scene


def test_an_episode_runs_on_the_official_scene_of_its_layout(scene):
    env = RoboDojoEnv(**ENV_ARGS)

    opened, stepped = env.open(), env.step(ACTION)
    env.close()

    assert scene.calls == [("reset", [3]), "run_reward", "close"]
    # the official env's policy client, made and told of the reset as the official env does: it connects to nothing
    assert sys.modules["src.eval_client.eval_env"].WsModelClient(url="ws://127.0.0.1:0").call(func_name="reset") is None
    assert [frame.getpixel((0, 0)) for frame in opened["images"]] == [(1, 1, 1)] * 3 and opened["outcome"] is None
    assert [frame.getpixel((0, 0)) for frame in stepped["images"]] == [(2, 2, 2)] * 3  # observed after the action
    assert stepped["actions"] == [ACTION] and stepped["outcome"] == "reached" and not stepped["done"]
    assert stepped["state"] == {"left_arm": {"z": 1.0}} and stepped["instruction"] == "stack the bowls"


@pytest.mark.parametrize("failure", [UnStableError("never settles"), PhysXBrokenError({0})])
def test_a_scene_that_never_settles_loses_the_episode(scene, failure):
    scene.failure = failure

    with pytest.raises(EpisodeLost):  # its trajectory starts over on a new App
        RoboDojoEnv(**ENV_ARGS).open()


def test_any_other_failure_of_the_reset_reaches_the_client_as_it_is(scene):
    scene.failure = ValueError("a scene that reads wrong")

    with pytest.raises(ValueError):
        RoboDojoEnv(**ENV_ARGS).open()


def test_physx_breaking_during_an_action_ends_the_episode_unstable(scene):
    env = RoboDojoEnv(**ENV_ARGS)
    env.open()
    Controller.failure = PhysXBrokenError({0})

    result = env.step(ACTION)

    assert result["done"] and result["rewards"] == [0.0] and result["env_end_reason"] == "unstable"
    assert result["images"] is None and result["state"] is None and result["outcome"] == "done"


def test_a_frame_the_dispatcher_did_not_deliver_fails_the_observation(scene):
    scene.stale = True

    with pytest.raises(RuntimeError, match="delivered no new frame"):
        RoboDojoEnv(**ENV_ARGS).open()


def test_closing_an_env_that_never_opened_does_nothing():
    RoboDojoEnv(**ENV_ARGS).close()
