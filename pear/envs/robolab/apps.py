from __future__ import annotations

import glob
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

from pear.envs.robolab.env import DEVICE, RoboLabEnv
from pear.envs.robolab.kit_runtime import KitRuntime
from pear.envs.robolab.mps import MPSDaemon
from pear.sim.process import SimulatorProcess
from pear.sim.schemas import ServerContext


class RoboLabApps:
    """A server's Isaac Apps (RoboLabEnv.setup_env), set up from its tool config and the ROBOLAB_* variables of its
    machine: each with its own copy of the caches a warm-up filled."""

    MAX_CONCURRENT_START_N = 4  # the Apps of a server that may be starting at once
    KIT_CACHE_DIR_NAME = "kit"  # an App's own Kit cache, its portable root
    # an App's other directories of its own, by the variable that names each to its user
    DIRECTORY_BY_VARIABLE = {
        "HOME": "home",
        "XDG_CACHE_HOME": "xdg_cache",
        "CUDA_CACHE_PATH": "cuda",
        "__GL_SHADER_DISK_CACHE_PATH": "glcache",
        "OPTIX_CACHE_PATH": "optix",
    }
    DIRECTORY_NAMES = (KIT_CACHE_DIR_NAME, *DIRECTORY_BY_VARIABLE.values())  # an App's own, copied from the warm-up's
    XDG_RUNTIME_DIR_NAME = "xdg_runtime"  # an App's own runtime directory, made for it and not copied
    XDG_RUNTIME_DIR_MODE = 0o700  # what the runtime directory's users (NGX among them) require of it
    # where the NVIDIA driver puts its Vulkan ICD
    VULKAN_ICD_PATHS = (Path("/etc/vulkan/icd.d/nvidia_icd.json"), Path("/usr/share/vulkan/icd.d/nvidia_icd.json"))
    OMP_THREAD_N = 4  # an App's OpenMP threads
    AARCH64_PRELOAD_GLOB = "/lib/*/libgomp.so.1"  # what isaacsim requires in LD_PRELOAD on aarch64
    # the NVRTC of RoboLab's environment, which the Apps preload
    NVRTC_GLOB = "lib/python3*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
    # the scene a warm-up builds into its server's Kit cache
    WARM_UP_ENV_ARGS = {"task": "BananaInBowlTask", "instruction_type": "default"}
    APPS_PER_MPS_DAEMON = 16  # a daemon's MPS server takes 48 client contexts and an App opens 3

    def __init__(self, context: ServerContext) -> None:
        self.context = context

        self.python = os.environ["ROBOLAB_PYTHON"]  # RoboLab's uv environment
        # the directory of omni.client's omniverse.toml, whose aliases serve the remote assets from their local copy,
        # and the libraries Isaac Sim assumes the machine has
        self.omni_config_path = os.environ["ROBOLAB_OMNI_CONFIG_PATH"]
        self.host_lib_dir = os.environ["ROBOLAB_HOST_LIB_DIR"]
        self.preload = self.preload_libraries(Path(self.python).parents[1])
        self.apps_per_gpu = int(context.config["apps_per_gpu"])  # each hosting one episode
        self.host = socket.gethostname().split(".")[0]
        # the Apps' caches on the machine's own disk, named by the server's GPUs: a server started on them later clears
        # what this one leaves
        self.local_dir = Path(os.environ["ROBOLAB_LOCAL_DIR"]) / f"gpu{'-'.join(str(gpu) for gpu in context.gpu_ids)}"
        self.gpu_uuid_by_index = self.gpu_uuids()  # by the NVML index, which Ray's GPU ids are

        vulkan_icd_paths = [path for path in self.VULKAN_ICD_PATHS if path.is_file()]
        if not vulkan_icd_paths:
            raise RuntimeError(f"no NVIDIA Vulkan ICD at {[str(path) for path in self.VULKAN_ICD_PATHS]}")
        self.vulkan_icd_path = vulkan_icd_paths[0]

    def setup_apps(self) -> tuple[list[SimulatorProcess], int]:
        """Returns the server's Apps with how many of them may start at once, their caches set up first."""
        self.context.log(
            f"Apps on GPUs {self.context.gpu_ids} x {self.apps_per_gpu}, {self.MAX_CONCURRENT_START_N} starting at once"
        )

        shutil.rmtree(self.local_dir, ignore_errors=True)  # an earlier server's on these GPUs: died with it
        self.local_dir.mkdir(parents=True)

        # every App starts from its own copy of the caches a warm-up fills by building a scene
        return self._apps(self._warm_up()), self.MAX_CONCURRENT_START_N

    def variables(self, gpu: int, app_dir: Path, mps_daemon: MPSDaemon | None) -> dict[str, str]:
        """The variables of an App on GPU ``gpu`` whose own directory is ``app_dir`` (and the MPS daemon ``mps_daemon``,
        if any)."""
        variables = {
            **(mps_daemon.variables() if mps_daemon else os.environ),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid_by_index[gpu],
            **{variable: str(app_dir / name) for variable, name in self.DIRECTORY_BY_VARIABLE.items()},
            "XDG_RUNTIME_DIR": str(app_dir / self.XDG_RUNTIME_DIR_NAME),
            "OMNI_CONFIG_PATH": self.omni_config_path,
            "LD_LIBRARY_PATH": self.host_lib_dir,
            "LD_PRELOAD": self.preload,
            "VK_ICD_FILENAMES": str(self.vulkan_icd_path),
            "PYTHONPATH": "",  # nothing of the server's path: the App's Python is RoboLab's own
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": str(self.OMP_THREAD_N),
            # the user accepted the NVIDIA Omniverse EULA for RoboLab
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }

        return variables

    def context_args(self, gpu: int, app_dir: Path) -> list[str]:
        """The Kit runtime's arguments of an App on GPU ``gpu`` whose own directory is ``app_dir``."""
        kit_args = f"{KitRuntime.KIT_ARGS} --portable --portable-root {app_dir / self.KIT_CACHE_DIR_NAME}"
        # Vulkan device order equals NVML order: activeGpu=<NVML index> renders on the card that
        # CUDA_VISIBLE_DEVICES=<UUID> exposes
        kit_args += f" --/renderer/multiGpu/enabled=false --/renderer/activeGpu={gpu} --/physics/cudaDevice=0"

        return [
            "--output-dir",
            str(app_dir / "robolab_output"),
            "--headless",
            "--enable_cameras",
            "--device",
            DEVICE,
            "--kit_args",
            kit_args,
        ]

    def _warm_up(self) -> Path:
        """Fills the caches the Apps start from (their template) by building one scene (WARM_UP_ENV_ARGS) in a process
        of its own on the server's first GPU, and returns them; raises RuntimeError if the build failed."""
        template, gpu = self.local_dir / "template", self.context.gpu_ids[0]
        for name in self.DIRECTORY_NAMES:
            (template / name).mkdir(parents=True)
        (template / self.XDG_RUNTIME_DIR_NAME).mkdir(mode=self.XDG_RUNTIME_DIR_MODE)
        process = SimulatorProcess(
            f"{self.host}-warm-up",
            RoboLabEnv,
            self.python,
            self.context_args(gpu, template),
            self.variables(gpu, template, None),
            template,
            self.context.log_dir,
        )

        # the build's error comes back from the process
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
                        RoboLabEnv,
                        self.python,
                        self.context_args(gpu, app_dir),
                        self.variables(gpu, app_dir, mps_daemon),
                        app_dir,
                        self.context.log_dir,
                    )
                )

        return apps

    @classmethod
    def preload_libraries(cls, environment_dir: Path) -> str:
        """The Apps' LD_PRELOAD: the NVRTC of RoboLab's environment ``environment_dir``, and libgomp on aarch64."""
        nvrtc = glob.glob(str(environment_dir / cls.NVRTC_GLOB))
        if len(nvrtc) != 1:
            raise RuntimeError(f"expected one NVRTC 12.9 in {environment_dir}, found {nvrtc}")

        return ":".join([*(glob.glob(cls.AARCH64_PRELOAD_GLOB) if platform.machine() == "aarch64" else []), *nvrtc])

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
