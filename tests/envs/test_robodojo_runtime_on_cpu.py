from __future__ import annotations

import os
import pickle
import shutil
import socket
import stat
import subprocess
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from pear.envs.robodojo import apps as robodojo_apps
from pear.envs.robodojo.apps import RoboDojoApps
from pear.envs.robodojo.env import RoboDojoEnv
from pear.envs.robodojo.kit_runtime import KitRuntime
from pear.sim.process import SimulatorProcess
from pear.sim.schemas import ServerContext
from repository import REPO

UUIDS = {0: "GPU-a", 1: "GPU-b"}  # the node's GPUs, by NVML index
APPS_PER_GPU = 20  # the Apps of a server's GPU in the tests' tool config


class FakeMps:
    """An MPS control daemon as RoboDojoApps starts it: the GPUs of those started, in order."""

    started: list[str] = []

    def __init__(self, pipe_dir: Path, log_path: Path, gpu_uuid: str) -> None:
        self.pipe_dir, self.gpu_uuid = pipe_dir, gpu_uuid

    def start(self) -> None:
        self.started.append(self.gpu_uuid)

    def variables(self) -> dict[str, str]:
        return {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir)}


class Setup:
    """RoboDojo's side of a server on this machine (the GPUs, MPS and the inotify shim stood in for), its node's
    settings in the environment."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(RoboDojoApps, "gpu_uuids", staticmethod(lambda: UUIDS))
        monkeypatch.setattr(RoboDojoApps, "build_no_inotify", staticmethod(lambda library_path: library_path.touch()))
        monkeypatch.setattr(robodojo_apps, "MPSDaemon", FakeMps)
        monkeypatch.setattr(FakeMps, "started", [])
        # the Apps' runtime directory, which a server makes: in the test's own directory, not on the machine
        self.runtime_dir = tmp_path / "xdg-runtime"
        monkeypatch.setattr(RoboDojoApps, "XDG_RUNTIME_DIR", self.runtime_dir)
        self.server_log: list[str] = []  # the lines of the server's log
        # a server on the node's second GPU, with its tool config
        self.context = ServerContext({"apps_per_gpu": APPS_PER_GPU}, tmp_path / "logs", [1], self.server_log.append)
        self.repo = tmp_path / "RoboDojo"
        monkeypatch.setenv("ROBODOJO_ROOT", str(self.repo))
        monkeypatch.setenv("ROBODOJO_KIT_EXTS_DIR", str(tmp_path / "kit_exts"))
        monkeypatch.setenv("ROBODOJO_LOCAL_DIR", str(tmp_path / "local"))
        monkeypatch.delenv("ROBODOJO_MESH_STORE_TEMPLATE_DIR", raising=False)
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
        """RoboDojoEnv's Apps of the server, its simulator processes."""
        return RoboDojoEnv.setup_env(self.context)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return Setup(tmp_path, monkeypatch)


def test_every_app_has_its_gpu_its_kit_cache_and_its_gpu_s_mps_daemon(setup, monkeypatch):
    monkeypatch.setenv("ROBODOJO_APPS_PER_GPU", "3")  # the tool config says how many Apps, not the node's environment

    apps, max_concurrent_start_n = setup.setup_env()

    host = socket.gethostname().split(".")[0]
    assert [app.name for app in apps] == [f"{host}-g1-a{number}" for number in range(APPS_PER_GPU)]
    assert setup.server_log[0] == f"Apps on GPUs [1] x 20, {RoboDojoApps.MAX_CONCURRENT_START_N} starting at once"
    assert max_concurrent_start_n == RoboDojoApps.MAX_CONCURRENT_START_N
    assert FakeMps.started == ["GPU-b", "GPU-b"]  # 16 Apps per server
    assert sorted(path.name for path in (setup.local_dir / f"{host}-g1-a3").iterdir()) == sorted(
        RoboDojoApps.DIRECTORY_NAMES
    )

    app = apps[17]  # the second MPS server's second App
    cache = setup.local_dir / app.name
    assert app.command == [
        str(RoboDojoApps.EXEC_ISAAC_PYTHON_PATH),
        "-u",
        "-m",
        "pear.sim.process",
        "--env",
        "pear.envs.robodojo.env.RoboDojoEnv",
    ]
    assert app.log_dir == setup.context.log_dir
    kit = app.context_args[-1]
    assert app.context_args[:-1] == [
        "--repo-root",
        str(setup.repo),
        "--headless",
        "--device",
        "cpu",
        "--enable_cameras",
        "--kit_args",
    ]
    assert f"--portable-root {cache}/kit" in kit and "--/renderer/activeGpu=1" in kit
    assert app.variables["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert app.variables["XDG_CACHE_HOME"] == f"{cache}/xdg_cache"
    # the one runtime directory of the node's Apps, made with the mode its users require
    assert app.variables["XDG_RUNTIME_DIR"] == str(setup.runtime_dir)
    assert stat.S_IMODE(setup.runtime_dir.stat().st_mode) == RoboDojoApps.XDG_RUNTIME_DIR_MODE
    # this repository first: the App's Python runs pear.sim.process out of the tree of its server
    assert app.variables["PYTHONPATH"] == f"{REPO}:{setup.repo}:{setup.repo}/XPolicyLab"
    assert app.variables["CUDA_MPS_PIPE_DIRECTORY"].endswith(f"mps/{host}-g1-1")
    assert app.variables["LD_PRELOAD"] == str(setup.local_dir / "libno_inotify.so")


def test_a_server_fills_its_apps_kit_cache_template_in_a_warm_up_on_its_gpu(setup):
    apps, _ = setup.setup_env()

    template = setup.local_dir / "template"
    [warm_up] = setup.warm_ups
    host = socket.gethostname().split(".")[0]
    assert (warm_up.name, warm_up.command[0], warm_up.cwd) == (
        f"{host}-warm-up",
        str(RoboDojoApps.EXEC_ISAAC_PYTHON_PATH),
        setup.repo,
    )
    assert setup.warm_up_requests == [{"op": "warm", "env_args": RoboDojoApps.WARM_UP_ENV_ARGS}]
    environment = warm_up.variables
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-b" and "CUDA_MPS_PIPE_DIRECTORY" not in environment
    kit = warm_up.context_args[-1]
    assert f"--portable-root {template}/kit" in kit and "--/renderer/activeGpu=1" in kit
    assert sorted(path.name for path in template.iterdir()) == sorted(RoboDojoApps.DIRECTORY_NAMES)
    assert len(apps) == APPS_PER_GPU


def test_a_failed_warm_up_fails_the_server_before_any_app(setup):
    setup.warm_up_reply = {"error": "Traceback: the scene never built"}

    with pytest.raises(RuntimeError, match="the warm-up of the Kit cache failed"):
        setup.setup_env()

    assert FakeMps.started == []


def test_a_server_on_the_gpu_of_an_earlier_one_sets_up_afresh(setup):
    (setup.local_dir / "mps").mkdir(parents=True)  # what the earlier server on this GPU left on this node
    (setup.local_dir / "mps" / "control").touch()

    setup.setup_env()

    assert not (setup.local_dir / "mps" / "control").exists() and (setup.local_dir / "mesh_store").is_dir()


def test_robodojo_s_process_context_binds_its_kit_app_for_the_process_s_episodes(monkeypatch):
    @contextmanager
    def running(kit):
        yield f"the Kit app of {kit.context_args}"

    monkeypatch.setattr(KitRuntime, "running", running)
    monkeypatch.setattr(RoboDojoEnv, "app", None)

    with RoboDojoEnv.process_context(["--repo-root", "R"]):
        assert RoboDojoEnv.app == "the Kit app of ['--repo-root', 'R']"


def test_an_app_loads_its_entry_point_and_robodojo_s_env_class_without_ray():
    # an App runs in the Isaac Python, which has no Ray (a None in sys.modules makes every import of it fail)
    code = "import sys; sys.modules['ray'] = None; import pear.sim.process, pear.envs.robodojo.env"
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


def test_isaac_lab_reads_the_runtime_s_arguments_as_the_process_s_command_line(monkeypatch, tmp_path):
    context_args = ["--repo-root", str(tmp_path), "--headless", "--kit_args", " --portable"]
    monkeypatch.setattr(sys, "argv", ["process.py", "--env", "E", "--fd", "3", "--", *context_args])
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=FakeLauncher))
    render_sync = types.SimpleNamespace(add_zero_delay_kit_args=lambda args: None)
    for name in ("env", "env.camera_manager", "env.camera_manager.capture"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "env.camera_manager.capture.render_sync", render_sync)
    monkeypatch.setattr(KitRuntime, "bind_checkout", lambda kit: tmp_path)
    monkeypatch.setattr(KitRuntime, "drain_kit_output", staticmethod(lambda log_fd: None))
    monkeypatch.setattr(KitRuntime, "shrink_openexr_pools", classmethod(lambda cls: None))
    monkeypatch.setattr(KitRuntime, "wait_for_denoiser", classmethod(lambda cls, log_fd: None))
    monkeypatch.setattr(FakeLauncher, "command_lines", [])
    monkeypatch.chdir(tmp_path)  # (the runtime enters the checkout)

    # nothing is asserted inside the block: the runtime ends its process on whatever is raised there
    with KitRuntime(context_args).running():
        pass

    assert FakeLauncher.command_lines == [context_args]  # none of SimulatorCommand's, no -- before them


# a library with an OpenEXR pool as Kit's have one: it keeps the size its pool was last given
OPENEXR_SOURCE = """
static int pool_size = -1;
void SET_THREAD_COUNT(int count) { pool_size = count; }
int last_pool_size(void) { return pool_size; }
"""
# the libraries loaded as Kit's startup loads them and their pools shrunk, then the size each pool was given
OPENEXR_CHILD = """
import ctypes, sys
from pear.envs.robodojo.kit_runtime import KitRuntime
libraries = [ctypes.CDLL(path) for path in sys.argv[1:]]
KitRuntime.shrink_openexr_pools()
print(*[library.last_pool_size() for library in libraries])
"""


@pytest.fixture
def openexr_libraries(tmp_path) -> list[Path]:
    """Stand-ins for Kit's two libraries with an OpenEXR pool, under their names."""
    source = tmp_path / "openexr.c"
    source.write_text(OPENEXR_SOURCE.replace("SET_THREAD_COUNT", KitRuntime.OPENEXR_SET_THREAD_COUNT))
    libraries = [tmp_path / name for name in KitRuntime.OPENEXR_LIBRARIES]
    for library in libraries:
        subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)

    return libraries


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("gcc") is None, reason="needs Linux and gcc")
def test_the_openexr_pool_of_each_of_kit_s_libraries_is_sized_down(openexr_libraries):
    child = subprocess.run(
        [sys.executable, "-c", OPENEXR_CHILD, *map(str, openexr_libraries)],
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert child.stdout.split() == [str(KitRuntime.OPENEXR_THREAD_N)] * len(openexr_libraries)


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("gcc") is None, reason="needs Linux and gcc")
def test_an_app_without_kit_s_imaging_library_loaded_fails(openexr_libraries):
    child = subprocess.run(
        [sys.executable, "-c", OPENEXR_CHILD, str(openexr_libraries[1])],
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
    )

    assert child.returncode != 0
    assert f"expected {KitRuntime.OPENEXR_LIBRARIES[0]} loaded" in child.stderr


DENOISER_LINE_S = 0.3  # how long after the wait begins the fake Kit logs its denoiser's line
# the wait finds the App's log by its descriptor in /proc
REQUIRES_PROC = pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="the wait reads Linux's /proc")


class AppLog:
    """An App's log as KitRuntime.wait_for_denoiser reads it: a file it holds open, to which Kit's lines come later."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.write_bytes(b"[63.637s] app ready\n")
        self.fd = os.open(path, os.O_RDONLY)

    def log_later(self, line: bytes) -> None:
        def append():
            with self.path.open("ab") as out:
                out.write(line)

        threading.Timer(DENOISER_LINE_S, append).start()


@REQUIRES_PROC
def test_an_app_takes_its_episode_only_once_rtx_created_its_denoiser(tmp_path):
    log = AppLog(tmp_path / "app.log")
    log.log_later(b"[Info] [rtx.optixdenoising.plugin] [77] Creating Optix 7.5 adaptor for device 0\n")

    KitRuntime.wait_for_denoiser(log.fd)

    # the wait ended on the line, which Kit logged only after the wait began: a wait that ends before finds no line
    assert KitRuntime.DENOISER_CREATED_LINE in log.path.read_bytes()


@REQUIRES_PROC
def test_an_app_whose_denoiser_failed_fails(tmp_path):
    log = AppLog(tmp_path / "app.log")
    log.log_later(b"[Warning] [rtx.optixdenoising.plugin] [77] Failed to initialize OptixDenoiserContext.\n")

    with pytest.raises(RuntimeError, match="denoiser was not created"):
        KitRuntime.wait_for_denoiser(log.fd)


@REQUIRES_PROC
def test_an_app_without_its_denoiser_by_the_deadline_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(KitRuntime, "DENOISER_TIMEOUT_S", DENOISER_LINE_S)
    log = AppLog(tmp_path / "app.log")

    with pytest.raises(RuntimeError, match="denoiser was not created"):
        KitRuntime.wait_for_denoiser(log.fd)
