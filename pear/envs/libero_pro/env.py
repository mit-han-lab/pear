from __future__ import annotations

# NVIDIA GL shader cache path, set before libGL loads (before mujoco / OpenGL / robosuite / libero are imported): the
# default ~/.cache/nvidia/GLCache on the network file system makes the driver hang in the cache lookup inside
# mujoco.MjrContext; a local per-uid directory also avoids collisions between users.
import os

os.environ.setdefault("__GL_SHADER_DISK_CACHE_PATH", f"/tmp/nvidia-glcache-{os.getuid()}")

import functools
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from pear.envs.libero_pro.controller import MotionController
from pear.sim.env import StatelessEnv
from pear.sim.process import SimulatorProcess
from pear.sim.schemas import EnvResult, ServerContext

# LIBERO-PRO, robosuite and mujoco are imported inside the methods: they exist only in the simulator's Python (the verl
# side imports it for setup_env and CAMERAS), and mujoco and robosuite only after _bind_rendering chose the GL device

logger = logging.getLogger(__name__)


class LiberoProEnv(StatelessEnv):
    """The env class of the LIBERO-PRO environment family, whose tasks are LIBERO's (suite, task_id, trial_id) under
    ``perturbation``: a call replays the episode's native actions (LIBERO's 20 Hz OSC deltas) and has a MotionController
    carry out one action of the protocol."""

    ACTION_DIM = 7  # LIBERO's native action: end-effector deltas (xyz, axis-angle) + gripper
    FRAME_SIZE = 512  # square frames, pixels
    # the side an instruction names -> the side the rotated frames show it on
    MIRRORED_SIDE_BY_SIDE = {"left": "right", "right": "left"}
    # the native actions an episode of each suite may take
    MAX_STEP_N_BY_SUITE = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    SEED = 7  # the LIBERO evaluation's seed
    TRIALS_PER_TASK = 50  # LIBERO's initial states per task
    # none: the original task; lan: a paraphrased instruction; object: other object appearances; swap: two objects'
    # initial regions exchanged; task: another goal. The benchmarks {suite}_{perturbation} hold them in LIBERO's order
    PERTURBATIONS = ("none", "lan", "object", "swap", "task")
    BDDL_INSTRUCTION_PERTURBATIONS = ("lan", "task")  # the perturbations whose instruction is in the bddl file only
    STABILIZATION_STEP_N = 10  # no-op steps after the reset while the objects settle (the LIBERO evaluation's wait)
    STABILIZATION_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # no motion, the gripper open
    RENDER_THREAD_N = 4  # llvmpipe threads per simulator process
    # one BLAS/OpenMP thread per process: a rollout machine runs many servers, and more threads would oversubscribe its
    # cores
    SINGLE_THREAD_VARIABLES = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    # the protocol's cameras -> LIBERO's observation keys
    OBSERVATION_KEY_BY_CAMERA = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
    # the Ray resource a rollout machine offers per CPU slot of a server (scripts/start_ray_node.sh)
    SLOT_RESOURCE = "libero"
    # a server: one simulator process on one of those slots
    SERVER_RESOURCES = {"num_cpus": 1, "resources": {SLOT_RESOURCE: 1}}
    START_TIMEOUT_S = 600.0  # importing LIBERO/mujoco in the simulator process from a cold file cache
    # the family's conversation (EnvironmentTool)
    SCHEMA = MotionController.SCHEMA
    CAMERAS = ("agentview", "wrist")  # the frames of an observation, in this order
    ENVIRONMENT_PROMPT = Path(__file__).with_name("environment_prompt.md").read_text().strip()
    # beyond this the simulator process is hung, even in a replay of the longest episode (520 steps)
    ROLLOUT_TIMEOUT_S = 180.0

    # benchmark object per name (instruction, bddl file, init states), process-wide
    _benchmark_by_name: dict[str, Any] = {}

    def __init__(self, suite: str, perturbation: str, task_id: int, trial_id: int) -> None:
        if suite not in self.MAX_STEP_N_BY_SUITE:
            raise ValueError(f"unknown suite {suite!r}; known: {sorted(self.MAX_STEP_N_BY_SUITE)}")
        if perturbation not in self.PERTURBATIONS:
            raise ValueError(f"unknown perturbation {perturbation!r}; known: {self.PERTURBATIONS}")
        assert 0 <= int(trial_id) < self.TRIALS_PER_TASK, (
            f"trial {trial_id} out of range (0..{self.TRIALS_PER_TASK - 1})"
        )

        self.suite = suite
        self.perturbation = perturbation
        self.benchmark_name = suite if perturbation == "none" else f"{suite}_{perturbation}"  # the task's benchmark
        self.task_id = int(task_id)
        self.trial_id = int(trial_id)

        self.scene: Any = None  # the OffScreenRenderEnv of the current call
        self._render_cache: dict[tuple, Any] = {}

        self._scene_cls = self._bind_rendering()

    @classmethod
    def server_n(cls, config: dict[str, Any], cluster_resources: dict[str, float]) -> int:
        """Returns one server per slot the cluster's rollout machines offer."""
        slot_n = int(cluster_resources.get(cls.SLOT_RESOURCE, 0))
        if slot_n == 0:
            raise RuntimeError(
                f"no machine of the Ray cluster offers the resource {cls.SLOT_RESOURCE}: scripts/start_ray_node.sh "
                "gives a rollout machine LIBERO_SERVERS_PER_NODE of it, a server each"
            )

        return slot_n

    @classmethod
    def setup_env(cls, context: ServerContext) -> tuple[list[SimulatorProcess], int]:
        """Sets up the server's one simulator process: the LIBERO-PRO simulator environment's Python (LIBERO_PYTHON)
        runs it from this repository with the machine's variables, one BLAS thread and Mesa's software rendering
        (LIBERO_MESA_DIR)."""
        mesa_dir = os.environ["LIBERO_MESA_DIR"]
        variables = {
            **os.environ,
            **cls.SINGLE_THREAD_VARIABLES,
            # Mesa's EGL vendor and DRI drivers, its software device, llvmpipe with RENDER_THREAD_N threads, no CUDA
            "LD_LIBRARY_PATH": f"{mesa_dir}/lib"
            + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
            "__EGL_VENDOR_LIBRARY_FILENAMES": f"{mesa_dir}/share/glvnd/egl_vendor.d/50_mesa.json",
            "LIBGL_DRIVERS_PATH": f"{mesa_dir}/lib/dri",
            "GALLIUM_DRIVER": "llvmpipe",
            "LIBGL_ALWAYS_SOFTWARE": "1",  # without it Mesa refuses to initialize the display
            "LP_NUM_THREADS": str(cls.RENDER_THREAD_N),
            "XDG_RUNTIME_DIR": os.path.join(os.environ.get("TMPDIR", "/tmp"), f"xdg_{os.environ['USER']}"),
            "CUDA_VISIBLE_DEVICES": "",
        }

        process = SimulatorProcess(
            "sim", cls, os.environ["LIBERO_PYTHON"], [], variables, SimulatorProcess.PEAR_ROOT, context.log_dir
        )

        return [process], 1

    @classmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        # a task under one perturbation
        return f"{env_args['suite']}/task_{env_args['task_id']}/{env_args['perturbation']}"

    def rollout(self, history: list[list[float]], action: dict[str, Any] | None) -> EnvResult:
        """Replays ``history`` (native actions) unrendered from the episode's initial state and carries out ``action``
        (None: the initial observation) within the suite's step budget."""
        max_step_n = self.MAX_STEP_N_BY_SUITE[self.suite]
        history = np.asarray(history, dtype=np.float64).reshape(-1, self.ACTION_DIM)
        if len(history) > max_step_n:
            raise ValueError(f"{len(history)} history actions exceed the {self.suite} budget of {max_step_n} steps")

        try:
            observation = self._open_scene()

            for native_action in history:  # as executed: no discretization of the gripper (0 holds it)
                observation = self._step(native_action)

            actions: list[list[float]] = []
            rewards: list[float] = []
            outcome = None
            done = len(history) >= max_step_n  # nothing left to execute: the budget is already spent
            if action is not None and not done:
                with self._no_render():  # the controller steps the scene and reads only proprioception
                    observation, actions, rewards, done, outcome = MotionController(
                        self.scene, observation, action
                    ).move(max_step_n - len(history))

            result: EnvResult = {
                "images": None,  # the frames come below, while the episode goes on
                "state": MotionController.state(self.scene, observation),
                "actions": actions,
                "rewards": rewards,
                "done": done,
                "executed": len(actions),
                "outcome": outcome,  # reached | stopped | done | budget | timeout
                "instruction": re.sub(
                    r"\b(left|right)\b", lambda match: self.MIRRORED_SIDE_BY_SIDE[match.group(1)], self._instruction()
                ),
            }
            if not done:
                if len(history) or actions:  # the reset rendered its last step, the replay and the action nothing
                    observation = self.scene.env._get_observations(force_update=True)

                # each camera's RGB frame, rotated 180 degrees
                result["images"] = [
                    Image.fromarray(
                        np.ascontiguousarray(
                            observation[self.OBSERVATION_KEY_BY_CAMERA[camera]][::-1, ::-1], dtype=np.uint8
                        )
                    )
                    for camera in self.CAMERAS
                ]
        finally:
            self._close_scene()  # drop the EGL renderer and the MuJoCo scene: idle episodes hold no memory

        return result

    def _bind_rendering(self):
        """Binds this process's GL backend (Mesa's EGL software device), then imports the simulator packages."""
        # EGL is only the headless way to get a GL context; the device behind it is Mesa's software device (llvmpipe,
        # CPU): setup_env points EGL at the Mesa prefix and hides every GPU. MuJoCo binds its EGL display and robosuite
        # its GL binding at import time, so the device is chosen first.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["EGL_PLATFORM"] = "device"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # setup_env names the directory: Mesa creates its anonymous memory files there
        os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
        # with Mesa's vendor file the EGL device list holds only its software device
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

        import mujoco  # noqa: F401  (initializes the EGL display on the software device)
        import robosuite.utils.binding_utils  # noqa: F401  (binds robosuite's renderer to that display)
        from libero.libero.envs import OffScreenRenderEnv

        return OffScreenRenderEnv

    def _open_scene(self) -> dict[str, np.ndarray]:
        """A fresh ``OffScreenRenderEnv`` for the episode's task, reset to its trial's initial state and stabilized: the
        last stabilization step's observation, rendered."""
        self._close_scene()

        bddl_file = self._benchmark().get_task_bddl_file_path(self.task_id)
        with self._permissive_torch_load():  # LIBERO's init-state files predate torch.load's weights_only default
            init_state = self._benchmark().get_task_init_states(self.task_id)[self.trial_id]

        # LIBERO renders every camera at one resolution (OffScreenRenderEnv takes a single camera_heights/widths)
        self.scene = self._scene_cls(
            bddl_file_name=bddl_file, camera_heights=self.FRAME_SIZE, camera_widths=self.FRAME_SIZE, hard_reset=False
        )

        self.scene.seed(self.SEED)
        self.scene.reset()
        self.scene.set_init_state(init_state)

        # the last stabilization step is rendered: its observation is the episode's first
        for _ in range(self.STABILIZATION_STEP_N - 1):
            self._step(self.STABILIZATION_ACTION)
        observation, _, _, _ = self.scene.step(self.STABILIZATION_ACTION)

        return observation

    @staticmethod
    @contextmanager
    def _permissive_torch_load():
        """Lets ``torch.load`` deserialize numpy arrays inside (LIBERO's init-state files predate ``weights_only``)."""
        torch_load = torch.load

        @functools.wraps(torch_load)
        def permissive_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return torch_load(*args, **kwargs)

        torch.load = permissive_load

        try:
            yield
        finally:
            torch.load = torch_load

    def _step(self, native_action: np.ndarray) -> dict[str, np.ndarray]:
        """One native action without rendering; the observation's proprioception is current, its frames are not."""
        with self._no_render():
            observation, _, _, _ = self.scene.step(native_action)

        return observation

    def _instruction(self) -> str:
        """The task's instruction: the bddl file's for the perturbations that change it, else the benchmark's (from the
        bddl file's name)."""
        # the benchmark takes every instruction from the bddl file's name, which the perturbed sets keep: it would hide
        # the paraphrase (lan) and the new goal (task)
        if self.perturbation in self.BDDL_INSTRUCTION_PERTURBATIONS:
            from libero.libero.envs.bddl_utils import get_problem_info

            return str(
                get_problem_info(self._benchmark().get_task_bddl_file_path(self.task_id))["language_instruction"]
            )

        return str(self._benchmark().get_task(self.task_id).language)

    def _close_scene(self) -> None:
        if self.scene is not None:
            self.scene.close()
            self.scene = None

    @contextmanager
    def _no_render(self):
        """The scene's steps inside skip the camera renders: their observations carry stale frames."""
        sim = self.scene.sim
        render = sim.render
        sim.render = self._noop_render

        try:
            yield
        finally:
            sim.render = render

    def _benchmark(self):
        benchmark = self._benchmark_by_name.get(self.benchmark_name)
        if benchmark is None:
            from libero.libero.benchmark import get_benchmark_dict

            benchmark = self._benchmark_by_name[self.benchmark_name] = get_benchmark_dict()[self.benchmark_name]()
            logger.info("LiberoProEnv: benchmark %s, %d tasks", self.benchmark_name, benchmark.n_tasks)

        return benchmark

    def _noop_render(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        camera_name: str | None = None,
        depth: bool = False,
        mode: str = "offscreen",
        device_id: int = -1,
        segmentation: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Cached render stub that skips actual rendering."""
        h, w = height or 1, width or 1
        key = (h, w, depth)

        result = self._render_cache.get(key)
        if result is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)  # RGB
            result = (frame, np.zeros((h, w), dtype=np.float32)) if depth else frame
            self._render_cache[key] = result

        return result
