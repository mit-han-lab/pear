from __future__ import annotations

import os
import pickle
import platform
import socket
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from pear.envs.robolab import apps as robolab_apps
from pear.envs.robolab.apps import RoboLabApps
from pear.envs.robolab.env import RoboLabEnv
from pear.envs.robolab.kit_runtime import KitRuntime
from pear.sim.process import SimulatorProcess
from pear.sim.schemas import ServerContext
from repository import REPO

UUIDS = {0: "GPU-a", 1: "GPU-b"}  # the node's GPUs, by NVML index
APPS_PER_GPU = 20  # the Apps of a server's GPU in the tests' tool config


class FakeMps:
    """An MPS control daemon as RoboLabApps starts it: the GPUs of those started, in order."""

    started: list[str] = []

    def __init__(self, pipe_dir: Path, log_path: Path, gpu_uuid: str) -> None:
        self.pipe_dir, self.gpu_uuid = pipe_dir, gpu_uuid

    def start(self) -> None:
        self.started.append(self.gpu_uuid)

    def variables(self) -> dict[str, str]:
        return {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir)}


class Setup:
    """RoboLab's side of a server on this machine (the GPUs, MPS, the driver's Vulkan ICD and the NVRTC of RoboLab's
    environment stood in for, the machine an x86_64 one), its node's settings in the environment."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(RoboLabApps, "gpu_uuids", staticmethod(lambda: UUIDS))
        monkeypatch.setattr(robolab_apps, "MPSDaemon", FakeMps)
        monkeypatch.setattr(FakeMps, "started", [])
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        self.vulkan_icd = tmp_path / "nvidia_icd.json"
        self.vulkan_icd.touch()
        monkeypatch.setattr(RoboLabApps, "VULKAN_ICD_PATHS", (tmp_path / "no_icd.json", self.vulkan_icd))
        self.server_log: list[str] = []  # the lines of the server's log
        # a server on the node's second GPU, with its tool config
        self.context = ServerContext({"apps_per_gpu": APPS_PER_GPU}, tmp_path / "logs", [1], self.server_log.append)
        self.python = tmp_path / "robolab" / "bin" / "python"
        self.nvrtc = tmp_path / "robolab" / "lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
        self.nvrtc.parent.mkdir(parents=True)
        self.nvrtc.touch()
        monkeypatch.setenv("ROBOLAB_PYTHON", str(self.python))
        monkeypatch.setenv("ROBOLAB_OMNI_CONFIG_PATH", str(tmp_path / "omni_config"))
        monkeypatch.setenv("ROBOLAB_HOST_LIB_DIR", str(tmp_path / "host_lib"))
        monkeypatch.setenv("ROBOLAB_LOCAL_DIR", str(tmp_path / "local"))
        self.local_dir = tmp_path / "local" / "gpu1"  # in the node's local directory, named by the server's GPU
        # the processes the setup starts (the Apps it only makes), each a warm-up, whose build replies warm_up_reply
        self.warm_ups: list[SimulatorProcess] = []
        self.warm_up_requests: list[dict] = []
        self.warm_up_reply: dict = {}
        monkeypatch.setattr(SimulatorProcess, "start", lambda process: self.warm_ups.append(process))
        monkeypatch.setattr(SimulatorProcess, "wait_ready", lambda process, timeout: True)
        monkeypatch.setattr(SimulatorProcess, "call", self.reply_to_warm_up)
        monkeypatch.setattr(SimulatorProcess, "close", lambda process: None)
        monkeypatch.setattr(SimulatorProcess, "wait_or_kill", lambda process, timeout: None)
        monkeypatch.setattr(SimulatorProcess, "poll", lambda process: 0)

    def reply_to_warm_up(self, request: dict, timeout: float | None) -> bytes:
        # (bound to the setup, not to the process: SimulatorProcess.call's stand-in gets only the call's arguments)
        self.warm_up_requests.append(request)
        return pickle.dumps(self.warm_up_reply)

    def setup_env(self):
        """RoboLabEnv's Apps of the server, its simulator processes."""
        return RoboLabEnv.setup_env(self.context)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return Setup(tmp_path, monkeypatch)


def test_every_app_has_its_gpu_its_caches_and_its_gpu_s_mps_daemon(setup, monkeypatch):
    monkeypatch.setenv("ROBOLAB_APPS_PER_GPU", "3")  # the tool config says how many Apps, not the node's environment

    apps, max_concurrent_start_n = setup.setup_env()

    host = socket.gethostname().split(".")[0]
    assert [app.name for app in apps] == [f"{host}-g1-a{number}" for number in range(APPS_PER_GPU)]
    assert setup.server_log[0] == f"Apps on GPUs [1] x 20, {RoboLabApps.MAX_CONCURRENT_START_N} starting at once"
    assert max_concurrent_start_n == RoboLabApps.MAX_CONCURRENT_START_N
    assert FakeMps.started == ["GPU-b", "GPU-b"]  # 16 Apps per server

    app = apps[17]  # the second MPS server's second App
    cache = setup.local_dir / app.name
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        [*RoboLabApps.DIRECTORY_NAMES, RoboLabApps.XDG_RUNTIME_DIR_NAME]
    )
    assert app.command == [
        str(setup.python),
        "-u",
        "-m",
        "pear.sim.process",
        "--env",
        "pear.envs.robolab.env.RoboLabEnv",
    ]
    assert (app.cwd, app.log_dir) == (cache, setup.context.log_dir)
    kit_args = app.context_args[-1]
    assert app.context_args[:-1] == [
        "--output-dir",
        f"{cache}/robolab_output",
        "--headless",
        "--enable_cameras",
        "--device",
        "cuda:0",
        "--kit_args",
    ]
    assert kit_args.startswith(KitRuntime.KIT_ARGS)
    assert f"--portable-root {cache}/kit" in kit_args and "--/renderer/activeGpu=1" in kit_args
    assert app.variables["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert app.variables["HOME"] == f"{cache}/home" and app.variables["XDG_CACHE_HOME"] == f"{cache}/xdg_cache"
    assert app.variables["XDG_RUNTIME_DIR"] == f"{cache}/xdg_runtime"
    # this repository alone: the App's Python runs pear.sim.process out of the tree of its server
    assert app.variables["PYTHONPATH"] == str(REPO)
    assert app.variables["CUDA_MPS_PIPE_DIRECTORY"].endswith(f"mps/{host}-g1-1")
    assert app.variables["LD_PRELOAD"] == str(setup.nvrtc)
    assert app.variables["VK_ICD_FILENAMES"] == str(setup.vulkan_icd)


def test_a_server_fills_its_apps_caches_template_in_a_warm_up_on_its_gpu(setup):
    apps, _ = setup.setup_env()

    template = setup.local_dir / "template"
    [warm_up] = setup.warm_ups
    host = socket.gethostname().split(".")[0]
    assert (warm_up.name, warm_up.command[0], warm_up.cwd) == (f"{host}-warm-up", str(setup.python), template)
    assert setup.warm_up_requests == [{"op": "warm", "env_args": RoboLabApps.WARM_UP_ENV_ARGS}]
    environment = warm_up.variables
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-b" and "CUDA_MPS_PIPE_DIRECTORY" not in environment
    assert environment["HOME"] == f"{template}/home"
    kit_args = warm_up.context_args[-1]
    assert f"--portable-root {template}/kit" in kit_args and "--/renderer/activeGpu=1" in kit_args
    assert sorted(path.name for path in template.iterdir()) == sorted(
        [*RoboLabApps.DIRECTORY_NAMES, RoboLabApps.XDG_RUNTIME_DIR_NAME]
    )
    runtime_dir = template / RoboLabApps.XDG_RUNTIME_DIR_NAME
    assert runtime_dir.stat().st_mode & 0o777 == RoboLabApps.XDG_RUNTIME_DIR_MODE
    assert len(apps) == APPS_PER_GPU


def test_a_failed_warm_up_fails_the_server_before_any_app(setup):
    setup.warm_up_reply = {"error": "Traceback: the scene never built"}

    with pytest.raises(RuntimeError, match="the warm-up of the Kit cache failed"):
        setup.setup_env()

    assert FakeMps.started == []


def test_the_apps_of_an_aarch64_node_preload_its_libgomp_ahead_of_the_nvrtc(setup, monkeypatch, tmp_path):
    libgomp = tmp_path / "lib" / "aarch64-linux-gnu" / "libgomp.so.1"
    libgomp.parent.mkdir(parents=True)
    libgomp.touch()
    monkeypatch.setattr(RoboLabApps, "AARCH64_PRELOAD_GLOB", str(tmp_path / "lib" / "*" / "libgomp.so.1"))
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")

    apps, _ = setup.setup_env()

    assert apps[0].variables["LD_PRELOAD"] == f"{libgomp}:{setup.nvrtc}"


def test_a_robolab_environment_without_its_one_nvrtc_fails_the_server(setup):
    setup.nvrtc.unlink()

    with pytest.raises(RuntimeError, match="expected one NVRTC"):
        setup.setup_env()


def test_robolab_s_process_context_gives_the_process_s_episodes_the_app_s_output_directory(monkeypatch):
    @contextmanager
    def running(runtime):
        yield

    monkeypatch.setattr(KitRuntime, "running", running)
    monkeypatch.setattr(RoboLabEnv, "output_dir", None)

    with RoboLabEnv.process_context(["--output-dir", "/apps/a0/robolab_output", "--headless"]):
        assert RoboLabEnv.output_dir == Path("/apps/a0/robolab_output")


def test_an_app_loads_its_entry_point_and_robolab_s_env_class_without_ray():
    # an App runs in RoboLab's Python, which has no Ray (a None in sys.modules makes every import of it fail)
    code = "import sys; sys.modules['ray'] = None; import pear.sim.process, pear.envs.robolab.env"
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)


class FakeLauncher:
    """Isaac Lab's AppLauncher as KitRuntime uses it: its add_app_launcher_args parses the process's command line with
    the caller's parser (as Isaac Lab's does) and records it."""

    command_lines: list[list[str]] = []

    def __init__(self, args) -> None:
        self.app = types.SimpleNamespace(close=lambda: None)

    @classmethod
    def add_app_launcher_args(cls, parser) -> None:
        cls.command_lines.append(list(sys.argv[1:]))
        parser.parse_known_args()  # Isaac Lab's check for name collisions, on sys.argv
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--kit_args", default="")


def test_isaac_lab_reads_the_launcher_s_arguments_as_the_process_s_command_line(monkeypatch, tmp_path):
    launcher_args = ["--headless", "--kit_args", f"{KitRuntime.KIT_ARGS} --portable"]
    context_args = ["--output-dir", str(tmp_path / "robolab_output"), *launcher_args]
    monkeypatch.setattr(sys, "argv", ["process.py", "--env", "E", "--fd", "3", "--", *context_args])
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=FakeLauncher))
    monkeypatch.setattr(FakeLauncher, "command_lines", [])

    runtime = KitRuntime(context_args)
    with runtime.running():
        pass

    assert runtime.output_dir == tmp_path / "robolab_output"
    # none of SimulatorCommand's, no -- before them, and not the runtime's own --output-dir
    assert FakeLauncher.command_lines == [launcher_args]
