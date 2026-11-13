from __future__ import annotations

import os
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.envs.robodojo.controller import MotionController
from pear.envs.robodojo.kit_runtime import KitRuntime
from pear.sim.env import EpisodeLost, StatefulEnv
from pear.sim.schemas import EnvResult, ServerContext

if TYPE_CHECKING:  # an annotation only
    from pear.sim.process import SimulatorProcess

# RoboDojo, OmegaConf and PIL are imported inside the methods: the verl side (the env class's facts) and the servers
# (Ray and numpy only) import this module too, and in an App RoboDojo's packages resolve only once its runtime
# (KitRuntime) bound the checkout, PIL only once Kit started (its bundled packages hold it)


class RoboDojoEnv(StatefulEnv):
    """The env class of the RoboDojo environment family: one layout's episode in this process's Isaac App, its scene
    built, reset and judged by the official evaluation env and each action of the protocol carried out by a
    MotionController."""

    # RoboDojo's cameras, in the order of a result's frames -> the names the model reads
    CAMERA_NAME_BY_KEY = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
    POLICY_NAME = "external"  # what the official env takes as the episode's policy
    # a server: the Apps of one GPU of a machine whose Ray offers its GPUs as robodojo
    # (scripts/start_ray_node.sh robodojo)
    SERVER_RESOURCES = {"num_cpus": 0, "num_gpus": 1, "resources": {"robodojo": 1}}
    # a server's start: its simulator machine joined the launch, its Kit cache warmed up and copied for each of its Apps
    START_TIMEOUT_S = 1800.0
    # the family's conversation (EnvironmentTool)
    SCHEMA = MotionController.SCHEMA
    CAMERAS = tuple(CAMERA_NAME_BY_KEY.values())
    ENVIRONMENT_PROMPT = Path(__file__).with_name("environment_prompt.md").read_text().strip()
    LOG_KEYS = ("env_end_reason",)
    # past this an App hangs in an open: a build with the official stability check, cold caches included
    OPEN_TIMEOUT_S = 660.0
    # past this an App hangs in a step: an action of at most MotionController.MAX_ACTION_STEP_N control steps
    STEP_TIMEOUT_S = 260.0

    app = None  # the process's Isaac App, once its runtime started Kit

    def __init__(self, task_name: str, config_name: str, layout_set: int, layout_id: int, random_seed: int) -> None:
        self.task = task_name
        self.config_name = config_name
        self.layout_set = int(layout_set)
        self.layout_id = int(layout_id)
        self.random_seed = int(random_seed)

        self.env: Any = None  # the official env of the episode's scene (open)
        self.controller: MotionController | None = None
        self.observation: dict[str, Any] | None = None  # the scene's last official observation
        self.scene_broken = False  # PhysX broke the scene: its bodies' states read as NaN or worse

    @classmethod
    def session_cap(cls, config: dict[str, Any]) -> int:
        # an episode holds an App from its open to its close: the Apps of every server, a GPU's each
        return int(config["server_n"]) * int(config["apps_per_gpu"])

    @classmethod
    def setup_env(cls, context: ServerContext) -> tuple[list[SimulatorProcess], int]:
        """Sets up the server's Apps with their Kit caches, mesh store, inotify shim and MPS daemons (RoboDojoApps),
        from its tool config and the simulator machine's ROBODOJO_* settings."""
        # apps.py imports this module for the class its Apps run, so this module imports it only here
        from pear.envs.robodojo.apps import RoboDojoApps

        return RoboDojoApps(context).setup_apps()

    @classmethod
    @contextmanager
    def process_context(cls, context_args: list[str]) -> Iterator[None]:
        """Kit, as every RoboDojo App runs it (KitRuntime)."""
        with KitRuntime(context_args).running() as app:
            cls.app = app
            yield

    @classmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        return env_args["task_name"]

    def open(self) -> EnvResult:
        """Builds and resets the official scene of the layout and returns the result of its initial observation;
        EpisodeLost when it never settles."""
        from src.eval_client import eval_env
        from src.eval_client.physx_warning_monitor import PhysXBrokenError
        from utils.cluttered_generator import UnStableError

        # the official env connects a policy client (WsModelClient) to its policy server as it is made and tells it of
        # each reset; RoboDojoEnv carries out the actions itself, so its client does nothing
        eval_env.WsModelClient = lambda **kwargs: types.SimpleNamespace(call=lambda **kwargs: None)
        self.env = eval_env.create_eval_env(self._config(), self.app)
        # the official env makes a video of every frame it observes
        self.env._stream_vision = lambda env_idx, frame: None

        try:
            self.env.reset(seed=[self.layout_id])
        except (PhysXBrokenError, UnStableError) as exc:
            # PhysX broke while the scene settled, or the official stability check gave up: a reset differs between
            # processes (articulations, fluids), so the layout may settle in a new App
            raise EpisodeLost(
                f"RoboDojo unstable layout {self.task} set {self.layout_set} id {self.layout_id}: {exc}"
            ) from exc

        # the official client's start of an episode (run_eval); a task may define no process score, as it allows
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        if self.env.is_episode_end():
            raise RuntimeError("reset produced a terminal or invalid initial scene")

        self.observation = self._observe()
        self.controller = MotionController(self.env, self.observation)

        return self._result(None, None)

    def step(self, action: dict[str, dict[str, float]]) -> EnvResult:
        """Carries out the model's ``action`` (the parts' quantity changes) in the episode and returns its result."""
        from src.eval_client.physx_warning_monitor import PhysXBrokenError

        try:
            outcome = self.controller.move(action)
            if outcome != "done":
                self.observation = self._observe()
        except PhysXBrokenError:  # the official checks found PhysX broken: the episode ends unstable
            outcome, self.scene_broken = "done", True

        return self._result(action, outcome)

    def close(self) -> None:
        """The scene's official env closed, if there is one."""
        # an open env keeps Isaac Lab's stop handler, which renders for as long as Kit's timeline is stopped: Kit's
        # close stops the timeline, so the App would never exit (env.close() removes the handler)
        if self.env is not None:
            self.env.close()

    def _config(self) -> Any:
        """The official evaluation config of the episode's scene, assembled as the official client's (main.py) for one
        env and one episode."""
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from omegaconf import OmegaConf
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self.config_name + ".yml"))
        eval_cfg.update(
            task_name=self.task,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            policy_name=self.POLICY_NAME,
            seed=self.layout_set,
            physx_monitor_enabled=True,
            eval_num=1,
        )

        config = {
            name: load_yaml(os.path.join(ENV_CONFIG_PATH, name, eval_cfg["config"][name] + ".yml"))
            for name in ("sim", "scene", "camera", "robot")
        }
        config["task_env"] = load_yaml(
            task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), self.task)
        )
        config["eval_cfg"] = eval_cfg
        # the official env requires its policy server's port (nothing connects to it)
        config["deploy_cfg"] = {"policy_name": self.POLICY_NAME, "port": 0, "host": "127.0.0.1"}
        config = OmegaConf.create(config)
        config.sim.scene.num_envs = 1

        config = process_randomization(config)
        config, _ = process_config(config, task_name=self.task)
        OmegaConf.update(config, "camera.default_frequency", config.eval_cfg.observation.collect_freq, force_add=True)
        config.sim.seed = [self.random_seed]

        return config

    def _observe(self) -> dict[str, Any]:
        """The scene's official observation, its frames rendered for it."""
        from env.camera_manager.capture.render_sync import dispatcher_time

        # without replicator's dispatcher the official capture counts a render as a delivered frame
        previous_frame_time = dispatcher_time()
        observation = self.env.get_obs()
        if dispatcher_time() in (None, previous_frame_time):
            raise RuntimeError("the camera dispatcher delivered no new frame")

        return observation

    def _result(self, action: dict[str, Any] | None, outcome: str | None) -> EnvResult:
        """The result of the call that carried out ``action`` with ``outcome`` (both None: the initial observation),
        with RoboDojo's ``env_end_reason``."""
        from PIL import Image

        env = self.env
        scene_unstable = self.scene_broken or 0 in env.unstable_envs  # PhysX broke the scene, or its task flagged it
        done = bool(env.end_flag[0]) or scene_unstable
        success = done and not scene_unstable and bool(env.success[0])
        if scene_unstable:
            env_end_reason = "unstable"
        elif success:
            env_end_reason = "success"
        elif done and env.take_action_cnt[0] >= env.step_lim:
            env_end_reason = "time_limit"
        elif done:
            env_end_reason = "task_failure"
        else:
            env_end_reason = None

        if action is None:  # the initial observation: no action carried out, no outcome
            actions, outcome = [], None
        else:
            actions, outcome = [action], "done" if done else outcome

        # an ended episode's last action rendered nothing; a broken scene's bodies read as NaN (or flung far away)
        return {
            "images": None
            if done
            else [Image.fromarray(self.observation["vision"][key]["color"]) for key in self.CAMERA_NAME_BY_KEY],
            "state": None if self.scene_broken else self.controller.state(),
            "actions": actions,
            "rewards": [float(success)] * len(actions),
            "done": done,
            "executed": len(actions),
            "outcome": outcome,
            "instruction": str(self.observation["instruction"]),
            "env_end_reason": env_end_reason,
        }
