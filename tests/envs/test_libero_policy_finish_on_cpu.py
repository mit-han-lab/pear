from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from pear.agent_loop.episode import Episode
from pear.envs.libero.controller import MotionController
from pear.envs.libero.env import LiberoEnv

ENV_ARGS = {"suite": "libero_goal", "task_id": 3, "trial_id": 0}
FINISH = {"finish": True}


def move_to(position):
    action = {part: dict.fromkeys(quantities, 0) for part, quantities in MotionController.SCHEMA.items()}
    action["arm"]["x"] = position
    return action


class Scene:
    """A LIBERO-shaped scene whose goal is satisfied at position 1."""

    def __init__(self):
        self.env = self
        self.position = 0
        self.steps = 0

    def check_success(self):
        return self.position == 1

    def _get_observations(self, *, force_update=False):
        frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
        return {"agentview_image": frame, "robot0_eye_in_hand_image": frame}

    def step(self, action):
        self.steps += 1
        self.position = int(action[0])
        success = self.check_success()
        return self._get_observations(), float(success), success, {}


@pytest.fixture
def make_env(monkeypatch):
    def open_scene(env):
        env.scene = Scene()
        return env.scene._get_observations()

    def init_motion(controller, scene, observation, action):
        controller.scene = scene
        controller.observation = observation
        controller.position = action["arm"]["x"]
        controller.start_step = scene.steps

    benchmark = SimpleNamespace(get_task=lambda _: SimpleNamespace(language="Place the bowl on the plate."))
    monkeypatch.setattr(LiberoEnv, "_bind_rendering", lambda _: None)
    monkeypatch.setattr(LiberoEnv, "_open_scene", open_scene)
    monkeypatch.setattr(LiberoEnv, "_close_scene", lambda _: None)
    monkeypatch.setattr(LiberoEnv, "_no_render", lambda _: nullcontext())
    monkeypatch.setattr(LiberoEnv, "_benchmark", lambda _: benchmark)
    monkeypatch.setattr(MotionController, "__init__", init_motion)
    monkeypatch.setattr(MotionController, "_arm_offsets", lambda self: [self.position, 0, 0, 0, 0, 0])
    monkeypatch.setattr(MotionController, "_grip_direction", lambda _: 0)
    monkeypatch.setattr(
        MotionController, "_step_outcome", lambda self: "reached" if self.scene.steps - self.start_step == 2 else None
    )
    monkeypatch.setattr(MotionController, "state", staticmethod(lambda scene, _: {"position": {"x": scene.position}}))
    return lambda **kwargs: LiberoEnv(**ENV_ARGS, **kwargs)


@pytest.mark.parametrize(("positions", "success"), [([], False), ([1], True), ([1, 0], False)])
def test_finish_scores_only_the_final_state(make_env, positions, success):
    env = make_env(policy_finish=True)
    episode = Episode(ENV_ARGS)
    for position in positions:
        result = env.rollout(episode.actions, move_to(position))
        episode.update(result)
        assert not episode.done
        assert not episode.success
        assert result["executed"] == 2
        assert all(image.getpixel((0, 0)) == (position,) * 3 for image in result["images"])

    result = env.rollout(episode.actions, FINISH)
    episode.update(result)

    assert episode.done
    assert episode.success is success
    assert (result["outcome"], result["images"], result["executed"]) == ("finish", None, 0)
    assert len(episode.actions) == 2 * len(positions)


def test_policy_finish_at_step_budget_is_a_truncation(make_env, monkeypatch):
    env = make_env(policy_finish=True)
    monkeypatch.setattr(env, "MAX_STEP_N_BY_SUITE", {"libero_goal": 2})
    episode = Episode(ENV_ARGS)

    result = env.rollout([], move_to(1))
    episode.update(result)

    assert (episode.done, episode.success) == (True, False)
    assert (result["outcome"], result["rewards"], result["images"]) == ("budget", [1.0, 1.0], None)


def test_default_mode_stops_at_first_successful_native_step(make_env):
    env = make_env()
    episode = Episode(ENV_ARGS)

    result = env.rollout([], move_to(1))
    episode.update(result)

    assert (episode.done, episode.success) == (True, True)
    assert (result["outcome"], result["executed"], result["images"]) == ("done", 1, None)
