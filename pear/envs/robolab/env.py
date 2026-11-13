from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.envs.robolab.controller import MotionController
from pear.envs.robolab.kit_runtime import KitRuntime
from pear.sim.env import StatefulEnv
from pear.sim.schemas import EnvResult, ServerContext

if TYPE_CHECKING:  # an annotation only
    from pear.sim.process import SimulatorProcess

# RoboLab and PIL are imported inside the methods: the verl side (the env class's facts) and the servers (Ray and numpy
# only) import this module too, and in an App RoboLab's packages import Isaac Lab's, which import only once Kit started

DEVICE = "cuda:0"  # an App's one GPU (CUDA_VISIBLE_DEVICES), RoboLab's runner default


class RoboLabEnv(StatefulEnv):
    """The env class of the RoboLab environment family: one task's episode in this process's Isaac App, built, reset and
    judged as RoboLab's evaluation runner does with its DROID robot under the absolute-IK action, each action of the
    protocol carried out by a MotionController."""

    # RoboLab's DROID cameras, in the order of a result's frames -> the names the model reads
    CAMERA_NAME_BY_KEY = {"over_shoulder_left_camera": "over_shoulder", "wrist_cam": "wrist"}
    FRAME_SIZE = (640, 360)  # the frames the model sees: RoboLab's 1280 x 720 renders, halved
    POLICY_NAME = "pear"  # what RoboLab records as the episode's policy
    # a server: the Apps of one GPU of a machine whose Ray offers its GPUs as robolab
    # (scripts/start_ray_node.sh robolab)
    SERVER_RESOURCES = {"num_cpus": 0, "num_gpus": 1, "resources": {"robolab": 1}}
    # a server's start: its simulator machine joined the launch, its Kit cache warmed up and copied for each of its Apps
    START_TIMEOUT_S = 1800.0
    # the family's conversation (EnvironmentTool)
    SCHEMA = MotionController.SCHEMA
    CAMERAS = tuple(CAMERA_NAME_BY_KEY.values())
    ENVIRONMENT_PROMPT = Path(__file__).with_name("environment_prompt.md").read_text().strip()
    LOG_KEYS = ("env_end_reason", "subtask_score", "subtask_info")
    OPEN_TIMEOUT_S = 600.0  # past this an App hangs in an open: a scene's build and reset
    # past this an App hangs in a step: an action of at most MotionController.MAX_ACTION_STEP_N control steps
    STEP_TIMEOUT_S = 300.0

    output_dir: Path | None = None  # the App's own RoboLab output directory

    def __init__(self, task: str, instruction_type: str) -> None:
        self.task = task  # a RoboLab-120 task, its class name
        self.instruction_type = instruction_type  # default, vague or specific

        self.env: Any = None  # the episode's RobolabEnv (open)
        self.env_cfg: Any = None
        self.controller: MotionController | None = None

    @classmethod
    def session_cap(cls, config: dict[str, Any]) -> int:
        # an episode holds an App from its open to its close: the Apps of every server, a GPU's each
        return int(config["server_n"]) * int(config["apps_per_gpu"])

    @classmethod
    def setup_env(cls, context: ServerContext) -> tuple[list[SimulatorProcess], int]:
        """Sets up the server's Apps with their Kit caches (RoboLabApps), from its tool config and the simulator
        machine's ROBOLAB_* settings."""
        # apps.py imports this module for the class its Apps run, so this module imports it only here
        from pear.envs.robolab.apps import RoboLabApps

        return RoboLabApps(context).setup_apps()

    @classmethod
    @contextmanager
    def process_context(cls, context_args: list[str]) -> Iterator[None]:
        """Kit, as RoboLab's evaluation runners run it (KitRuntime)."""
        runtime = KitRuntime(context_args)
        with runtime.running():
            cls.output_dir = runtime.output_dir
            yield

    @classmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        return env_args["task"]

    def open(self) -> EnvResult:
        """Builds and resets the task's scene as RoboLab's runner and run_episode do, with the DROID robot's absolute-IK
        action, and returns the result of its initial observation."""
        import robolab.constants
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

        # the env config's recorder (data.hdf5) and create_env (env_cfg.json) write into RoboLab's output directory
        robolab.constants.set_output_dir(str(self.output_dir))
        auto_register_droid_abs_ik_envs(task=[self.task])
        self.env, self.env_cfg = create_env(
            self.task, device=DEVICE, num_envs=1, instruction_type=self.instruction_type, policy=self.POLICY_NAME
        )
        self.env.reset()
        observation, _ = self.env.reset()  # (run_episode resets twice)

        # the table top stands at different heights above the robot's root from scene to scene: its top is that of
        # RoboLab's bounding box of the table, as its placement checks measure a surface
        corners, _ = get_world(self.env).get_bbox("table", env_id=0)
        root_z = float(self.env.scene["robot"].data.root_pos_w[0, 2] - self.env.scene.env_origins[0, 2])
        self.controller = MotionController(self.env, observation, max(float(corner[2]) for corner in corners) - root_z)

        return self._result(None, None)

    def step(self, action: dict[str, dict[str, float]]) -> EnvResult:
        """Carries out the model's ``action`` (the parts' quantity changes) in the episode and returns its result."""
        state = self.controller.state()  # the readings before the action, the last finite ones if the physics diverges

        return self._result(action, self.controller.move(action), state)

    def close(self) -> None:
        """The episode's RobolabEnv closed, if there is one."""
        if self.env is not None:
            self.env.close()

    def _result(
        self, action: dict[str, Any] | None, outcome: str | None, state_before: dict[str, Any] | None = None
    ) -> EnvResult:
        """The result of the call that carried out ``action`` with ``outcome`` (both None: the initial observation),
        with RoboLab's ``env_end_reason`` and subtask progress (``subtask_score``, ``subtask_info``)."""
        from PIL import Image
        from robolab.core.logging.results import get_all_env_subtask_infos

        if outcome == "diverged":
            # the episode ends there, not won, with ``state_before`` (the action's start) as its last state: nothing
            # more is read from an env whose physics diverged
            done, success, env_end_reason = True, False, "physics_diverged"
            images, state, subtask = None, state_before, {}
        else:
            # RobolabEnv freezes an env once a termination term fires: a success term (every term of RoboLab-120's but
            # its time_out) or the time limit
            done = bool(self.env.all_terminated)
            success = done and bool(self.env.get_env_results()[0]["success"])
            env_end_reason = ("success" if success else "time_limit") if done else None
            subtask_infos = get_all_env_subtask_infos(self.env)
            subtask = subtask_infos[0] if subtask_infos else {}  # none for a task without subtasks
            images = (
                None
                if done
                else [
                    Image.fromarray(self.controller.observation["image_obs"][key][0].cpu().numpy()).resize(
                        self.FRAME_SIZE, Image.BILINEAR
                    )
                    for key in self.CAMERA_NAME_BY_KEY
                ]
            )
            state = self.controller.state()
            if done and action is not None:  # the action that ended the episode
                outcome = "done"

        actions = [] if action is None else [action]

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "rewards": [float(success)] * len(actions),
            "done": done,
            "executed": len(actions),
            "outcome": outcome,
            "instruction": str(self.env_cfg.instruction),
            "env_end_reason": env_end_reason,
            "subtask_score": subtask.get("score"),
            "subtask_info": subtask.get("info"),
        }
