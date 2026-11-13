import argparse
import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Isaac Lab and RoboDojo are imported inside the methods: the verl side and the servers import this module with the env
# class, and in an App RoboDojo's packages resolve only once bind_checkout bound the checkout


class KitRuntime:
    """Kit in a RoboDojo App (RoboDojoEnv.process_context): started as the official client starts it, its crash reports
    kept on the machine, its output drained, its thread pools shrunk and its OptiX denoiser created before its
    episode."""

    # the status of an App that failed: Kit's shutdown ends the process with 0, so a failed App exits without it and its
    # server sees the failure (a warm-up that failed fails its server instead of starting Apps that fail too)
    FAILED_EXIT_STATUS = 1
    # Kit uploads each crash report (command line, paths, nvidia-smi) to NVIDIA unless its URL is empty (the privacy
    # settings do not stop it) and records telemetry events for its transmitter: both off. Its crash log is off too:
    # after a heap corruption it allocates inside the handler, aborts again and hangs; the dump stays
    CRASH_KIT_ARGS = " --/crashreporter/url= --/structuredLog/enable=false --/crashreporter/log=false"
    # Isaac Lab's AppLauncher replaces the reporter's handlers of these by one that closes Kit: a crash on any thread
    # then ends the App with 0, as if it had stopped by itself
    CRASH_SIGNALS = (signal.SIGSEGV, signal.SIGABRT)
    # the libraries with their own static OpenEXR and so their own global pool: carb.imaging decodes a scene's .exr
    # textures, replicator's writer encodes .exr files (RoboDojo writes none); a machine's pids.max counts every thread
    # of every App, these pools' idle ones too
    OPENEXR_LIBRARIES = ("libcarb.imaging.plugin.so", "_omni_replicator_exrwriter.so")
    OPENEXR_THREAD_N = 2  # per pool
    # OpenEXR 3.2's Imf::setGlobalThreadCount(int), by the name its libraries export it under
    OPENEXR_SET_THREAD_COUNT = "_ZN7Imf_3_220setGlobalThreadCountEi"
    # RTX creates its OptiX denoiser (the real-time renderer's direct lighting) on a worker thread after Kit's start,
    # and the creation synchronizes the device, which breaks any CUDA graph capture under way (cuRobo's, in the App's
    # first open): an App takes its episode only once this line says the creation ended
    DENOISER_CREATED_LINE = b"Creating Optix"
    DENOISER_FAILED_LINE = b"Failed to initialize OptixDenoiserContext"  # the line of a creation that failed
    # the denoiser's channel on the App's log from info up (its creation's line), every other one from warnings up
    DENOISER_KIT_ARGS = (
        " --/log/level=warn --/log/outputStreamLevel=info --/log/channels/rtx.optixdenoising.plugin=info"
    )
    DENOISER_TIMEOUT_S = 300  # an App's wait for its denoiser after "app ready"
    DENOISER_POLL_S = 0.1  # between two looks at the App's log for the denoiser's line

    def __init__(self, context_args: list[str]) -> None:
        # --repo-root and Isaac Lab's AppLauncher arguments (RoboDojoApps.context_args)
        self.context_args = context_args

    @contextmanager
    def running(self) -> Iterator[Any]:
        """Kit's app, from its start to its close, which ends the process; a failure inside ends the process without
        Kit's shutdown, with FAILED_EXIT_STATUS."""
        # Isaac Lab's AppLauncher (add_app_launcher_args) and Isaac Sim's SimulationApp parse the process's command line
        # themselves (parse_known_args of sys.argv) and hand it to Kit: from here on it holds the runtime's arguments
        # only, as an official client's holds its own, not SimulatorCommand's before them
        sys.argv[1:] = self.context_args
        robodojo_root = self.bind_checkout()
        from isaaclab.app import AppLauncher

        # --repo-root is declared again: this parse takes all of the runtime's arguments
        parser = argparse.ArgumentParser(description=self.__doc__)
        parser.add_argument("--repo-root", required=True, help="the official RoboDojo checkout")
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(self.context_args)
        os.chdir(robodojo_root)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

        # the cameras' current-frame capture needs Kit's zero-delay settings
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

        add_zero_delay_kit_args(args)
        # between requests the main thread waits for its server, outside Kit's loop: Kit's hang detector takes every
        # wait over 120 s for a hang and asks, through a dialog no machine can show (zenity), whether to crash the App
        args.kit_args += " --/app/hangDetector/enabled=false"
        args.kit_args += self.DENOISER_KIT_ARGS

        # the App's own stderr (its log): the output drain's destination, and a failure's, written past the drain
        stderr_fd = os.dup(2)
        # the official PhysX warning monitor (broken scenes end an episode as unstable) reads Kit's log output, so it
        # must own stdout and stderr before Kit starts; the official client starts it for the tasks with articulations,
        # an App learns its task only at its open
        self.drain_kit_output(stderr_fd)
        launcher = self.launch(AppLauncher, args)

        try:
            self.shrink_openexr_pools()  # Kit's startup loaded both libraries and sized their pools to the machine
            self.wait_for_denoiser(stderr_fd)
            yield launcher.app
        except BaseException:
            os.write(stderr_fd, f"App failed\n{traceback.format_exc()}".encode())
            os._exit(self.FAILED_EXIT_STATUS)

        launcher.app.close()

    def bind_checkout(self) -> Path:
        """The RoboDojo checkout of --repo-root, its top-level packages bound before Isaac and Kit extend sys.path."""
        repo_root_parser = argparse.ArgumentParser(add_help=False)
        repo_root_parser.add_argument("--repo-root", required=True)
        robodojo_root = Path(repo_root_parser.parse_known_args(self.context_args)[0].repo_root).resolve()
        sys.path[:0] = [str(robodojo_root), str(robodojo_root / "XPolicyLab")]
        import utils.load_file  # RoboDojo's own package

        # the arm64 Isaac image ships another ``utils``, which would shadow RoboDojo's
        if Path(utils.load_file.__file__).resolve() != (robodojo_root / "utils/load_file.py").resolve():
            raise RuntimeError(f"RoboDojo utils package is shadowed by {utils.load_file.__file__}")

        return robodojo_root

    @staticmethod
    def drain_kit_output(log_fd: int) -> None:
        """The official PhysX warning monitor over Kit's stdout and stderr, which a tee process drains into the App's
        log ``log_fd`` and the monitor's mirror file, where the monitor's tail thread finds the warnings."""
        # the monitor's readers are Python threads, and Kit's C++ code writes into their pipe inside calls whose Python
        # caller holds the GIL: a burst the pipe cannot take blocks that writer with the GIL, the readers never drain
        # the pipe and the App hangs; a tee process drains it whatever the App's threads do
        from src.eval_client.physx_warning_monitor import get_monitor

        monitor = get_monitor()
        monitor.start(enabled=True)
        read_fd, write_fd = os.pipe()
        # its own session: a stop signal to the App's process group leaves it to pass on the App's last output
        subprocess.Popen(
            ["tee", "-a", monitor._mirror_path], stdin=read_fd, stdout=log_fd, stderr=log_fd, start_new_session=True
        )

        for fd in (1, 2):
            # the monitor's pipe stays open and silent: its reader blocks and never reattaches (it would if Kit took
            # the fd over)
            os.dup(fd)
            os.dup2(write_fd, fd)
        os.close(read_fd)
        os.close(write_fd)

    @classmethod
    def launch(cls, launcher_class: Any, args: argparse.Namespace) -> Any:
        """Isaac Lab's AppLauncher started on ``args``, Kit's crash reporter set to keep its reports on the machine and
        its handlers kept, so that a crash ends the App with its signal."""
        args.kit_args += cls.CRASH_KIT_ARGS
        signal_function = signal.signal

        def keep(signum, handler):
            return signal.getsignal(signum) if signum in cls.CRASH_SIGNALS else signal_function(signum, handler)

        signal.signal = keep
        try:
            return launcher_class(args)
        finally:
            signal.signal = signal_function

    @classmethod
    def shrink_openexr_pools(cls) -> None:
        """The OpenEXR thread pools of Kit's libraries, each sized to every CPU of the machine, sized down through
        OpenEXR's own Imf::setGlobalThreadCount in each library's copy."""
        with open("/proc/self/maps") as maps:
            paths = sorted({line.split()[-1] for line in maps if line.rstrip().endswith(cls.OPENEXR_LIBRARIES)})
        if not any(path.endswith(cls.OPENEXR_LIBRARIES[0]) for path in paths):
            raise RuntimeError(f"OpenEXR pools: expected {cls.OPENEXR_LIBRARIES[0]} loaded, found {paths}")

        for path in paths:
            getattr(ctypes.CDLL(path, mode=os.RTLD_NOLOAD), cls.OPENEXR_SET_THREAD_COUNT)(cls.OPENEXR_THREAD_N)

    @classmethod
    def wait_for_denoiser(cls, log_fd: int) -> None:
        """RTX's OptiX denoiser created, as its line in the App's log ``log_fd`` says; RuntimeError if its creation
        failed."""
        path, deadline = Path(os.readlink(f"/proc/self/fd/{log_fd}")), time.monotonic() + cls.DENOISER_TIMEOUT_S
        while cls.DENOISER_CREATED_LINE not in (log_bytes := path.read_bytes()):
            if cls.DENOISER_FAILED_LINE in log_bytes or time.monotonic() > deadline:
                raise RuntimeError(f"RTX's OptiX denoiser was not created; see {path}")
            time.sleep(cls.DENOISER_POLL_S)
