from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from pear.envs.robodojo.env import RoboDojoEnv
from pear.envs.robodojo.mps import MPSDaemon
from pear.sim.process import SimulatorProcess
from pear.sim.schemas import ServerContext


class RoboDojoApps:
    """A server's Isaac Apps (RoboDojoEnv.setup_env), set up from its tool config and the ROBODOJO_* variables of its
    machine: each with its own copy of a Kit cache template, sharing the server's store of cooked meshes, its inotify
    shim and its GPUs' MPS daemons."""

    # the Isaac Python's exec wrapper
    EXEC_ISAAC_PYTHON_PATH = Path(__file__).resolve().with_name("exec_isaac_python.sh")
    NO_INOTIFY_SOURCE_PATH = Path(__file__).with_name("no_inotify.c")  # the Apps' inotify, which never reports a change
    DEVICE = "cpu"  # what an App's Isaac Lab launcher takes as --device
    MAX_CONCURRENT_START_N = 8  # the Apps of a server that may be starting at once
    KIT_CACHE_DIR_NAME = "kit"  # an App's own Kit cache, its portable root
    # an App's other cache directories, by the variable that names each to its user
    DIRECTORY_BY_VARIABLE = {
        "XDG_CACHE_HOME": "xdg_cache",
        "CUDA_CACHE_PATH": "cuda",
        "__GL_SHADER_DISK_CACHE_PATH": "glcache",
        "OPTIX_CACHE_PATH": "optix",
    }
    DIRECTORY_NAMES = (KIT_CACHE_DIR_NAME, *DIRECTORY_BY_VARIABLE.values())  # an App's own cache directories
    XDG_RUNTIME_DIR = Path("/tmp/xdg-runtime")  # the Apps' runtime directory
    XDG_RUNTIME_DIR_MODE = 0o700  # what the runtime directory's users (NGX among them) require of it
    OMP_THREAD_N = 4  # an App's OpenMP threads
    # the scene a warm-up builds into its server's Kit cache
    WARM_UP_ENV_ARGS = {
        "task_name": "stack_bowls",
        "config_name": "arx_x5",
        "layout_set": 0,
        "layout_id": 0,
        "random_seed": 0,
    }
    APPS_PER_MPS_DAEMON = 16  # a daemon's MPS server takes 48 client contexts and an App opens 3

    def __init__(self, context: ServerContext) -> None:
        self.context = context

        self.robodojo_root = Path(os.environ["ROBODOJO_ROOT"])  # the official RoboDojo checkout
        # Kit extensions the image lacks
        self.kit_exts_dir = os.environ["ROBODOJO_KIT_EXTS_DIR"]
        # a store of cooked meshes (a server's mesh_store) its store starts from, optional
        self.mesh_store_template_dir = os.environ.get("ROBODOJO_MESH_STORE_TEMPLATE_DIR")
        self.apps_per_gpu = int(context.config["apps_per_gpu"])  # each hosting one episode
        self.host = socket.gethostname().split(".")[0]

        # the servers' Kit caches, mesh stores and MPS pipes on the machine's own disk (ROBODOJO_LOCAL_DIR): the
        # container's /tmp is an overlay held in memory, which the Apps' episodes need; named by the server's GPUs: a
        # server started on them later clears what this one leaves
        self.local_dir = Path(os.environ["ROBODOJO_LOCAL_DIR"]) / f"gpu{'-'.join(str(gpu) for gpu in context.gpu_ids)}"
        # one store of cooked collision meshes for the server's Apps: an App that starts takes over what ended Apps
        # cooked (a layout's first build cooks its objects' convex decompositions)
        self.mesh_store_dir = self.local_dir / "mesh_store"
        # the Apps' inotify, in place of two kernel instances per App (Kit's audio plugin, omniclient's file provider)
        self.no_inotify_library_path = self.local_dir / "libno_inotify.so"

        self.gpu_uuid_by_index = self.gpu_uuids()  # by the NVML index, which Ray's GPU ids are

    def setup_apps(self) -> tuple[list[SimulatorProcess], int]:
        """Sets up the server's resources and returns its Apps with how many of them may start at once."""
        self.context.log(
            f"Apps on GPUs {self.context.gpu_ids} x {self.apps_per_gpu}, {self.MAX_CONCURRENT_START_N} starting at once"
        )

        shutil.rmtree(self.local_dir, ignore_errors=True)  # an earlier server's on these GPUs: died with it
        self.local_dir.mkdir(parents=True)
        if self.mesh_store_template_dir:
            shutil.copytree(self.mesh_store_template_dir, self.mesh_store_dir)
        else:
            self.mesh_store_dir.mkdir()
        self.build_no_inotify(self.no_inotify_library_path)
        self.XDG_RUNTIME_DIR.mkdir(mode=self.XDG_RUNTIME_DIR_MODE, parents=True, exist_ok=True)

        # every App starts from its own copy of one Kit cache, which a warm-up fills by building a scene
        return self._apps(self._warm_up()), self.MAX_CONCURRENT_START_N

    def variables(self, gpu: int, app_dir: Path, mps_daemon: MPSDaemon | None) -> dict[str, str]:
        """The variables of an App on GPU ``gpu`` whose own directory is ``app_dir`` (and the MPS daemon ``mps_daemon``,
        if any)."""
        return {
            **(mps_daemon.variables() if mps_daemon else os.environ),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid_by_index[gpu],
            **{variable: str(app_dir / name) for variable, name in self.DIRECTORY_BY_VARIABLE.items()},
            "PYTHONPATH": f"{self.robodojo_root}:{self.robodojo_root}/XPolicyLab",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": str(self.OMP_THREAD_N),
            "XDG_RUNTIME_DIR": str(self.XDG_RUNTIME_DIR),
            "ACCEPT_EULA": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            # exec_isaac_python.sh preloads Kit's libcarb ahead of it
            "LD_PRELOAD": str(self.no_inotify_library_path),
        }

    def context_args(self, gpu: int, app_dir: Path) -> list[str]:
        """The Kit runtime's arguments of an App on GPU ``gpu`` whose own directory is ``app_dir``, with the server's
        store of cooked meshes."""
        # its own Kit cache, and the server's store of PhysX's cooked meshes (Kit apps share it)
        kit_args = f" --portable --portable-root {app_dir / self.KIT_CACHE_DIR_NAME}"
        kit_args += f" --/UJITSO/datastore/localCachePath={self.mesh_store_dir}"
        # an exception thrown inside a Kit task is suppressed by default and leaves whatever waits for the task waiting
        # forever: unsuppressed it ends the App, visibly
        kit_args += " --/plugins/carb.tasking.plugin/suppressTaskException=false"
        # Vulkan device order equals NVML order, so activeGpu=<NVML index>
        # renders on the same card that CUDA_VISIBLE_DEVICES=<UUID> exposes
        kit_args += f" --/renderer/multiGpu/enabled=false --/renderer/activeGpu={gpu} --/physics/cudaDevice=0"
        kit_args += " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
        # the Kit extensions the image lacks come from a local folder: pulled from
        # NVIDIA's extension registry at every App start, they would collide in the container's shared ov data folder,
        # where Apps start together; the registry stays off
        kit_args += f" --ext-folder {self.kit_exts_dir} --/app/extensions/registryEnabled=false"

        return [
            "--repo-root",
            str(self.robodojo_root),
            "--headless",
            "--device",
            self.DEVICE,
            "--enable_cameras",
            "--kit_args",
            kit_args,
        ]

    def _warm_up(self) -> Path:
        """Fills the Kit cache the Apps start from (its template) by building one scene (WARM_UP_ENV_ARGS) in a process
        of its own on the server's first GPU, and returns it; raises RuntimeError if the build failed."""
        template, gpu = self.local_dir / "template", self.context.gpu_ids[0]
        for name in self.DIRECTORY_NAMES:
            (template / name).mkdir(parents=True)
        process = SimulatorProcess(
            f"{self.host}-warm-up",
            RoboDojoEnv,
            str(self.EXEC_ISAAC_PYTHON_PATH),
            self.context_args(gpu, template),
            self.variables(gpu, template, None),
            self.robodojo_root,
            self.context.log_dir,
        )

        # the build's error comes back from the process, the App's log holds none of it
        start_time = time.monotonic()
        error = process.warm_up(self.WARM_UP_ENV_ARGS)
        if error is not None or process.poll() != 0:
            raise RuntimeError(
                f"the warm-up of the Kit cache failed (exit {process.poll()}): {error}; see {process.log_path}"
            )
        self.context.log(f"warmed up the Kit cache in {time.monotonic() - start_time:.0f}s")

        return template

    def _apps(self, template: Path) -> list[SimulatorProcess]:
        """The server's Apps, each with its own copy of ``template``, its GPUs' MPS daemons started for them."""
        apps = []
        for gpu in self.context.gpu_ids:
            for number in range(self.apps_per_gpu):
                # with PhysX on the GPU (as RoboDojo's setup_physics puts it) and without MPS the GPU time-slices
                # among the Apps' contexts and every physics step waits for its turn
                if number % self.APPS_PER_MPS_DAEMON == 0:  # each group of APPS_PER_MPS_DAEMON Apps of a GPU shares one
                    label = f"{self.host}-g{gpu}-{number // self.APPS_PER_MPS_DAEMON}"
                    mps_daemon = MPSDaemon(
                        self.local_dir / "mps" / label,
                        self.context.log_dir / f"mps-{label}.log",
                        self.gpu_uuid_by_index[gpu],
                    )
                    mps_daemon.start()
                name = f"{self.host}-g{gpu}-a{number}"
                app_dir = self.local_dir / name
                shutil.copytree(template, app_dir, symlinks=True)
                apps.append(
                    SimulatorProcess(
                        name,
                        RoboDojoEnv,
                        str(self.EXEC_ISAAC_PYTHON_PATH),
                        self.context_args(gpu, app_dir),
                        self.variables(gpu, app_dir, mps_daemon),
                        self.robodojo_root,
                        self.context.log_dir,
                    )
                )

        return apps

    @staticmethod
    def gpu_uuids() -> dict[int, str]:
        """The machine's GPUs' UUIDs by their NVML index."""
        gpu_table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )

        return {
            int(index): gpu_uuid.strip()
            for index, gpu_uuid in (line.split(",") for line in gpu_table.strip().splitlines())
        }

    @classmethod
    def build_no_inotify(cls, library_path: Path) -> None:
        """no_inotify.c built into ``library_path`` (the Isaac container carries gcc)."""
        subprocess.run(
            ["gcc", "-O2", "-fPIC", "-shared", str(cls.NO_INOTIFY_SOURCE_PATH), "-ldl", "-o", str(library_path)],
            check=True,
        )
