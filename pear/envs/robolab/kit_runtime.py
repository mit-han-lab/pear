import argparse
import os
import signal
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# cv2 and Isaac Lab are imported inside running: the verl side and the servers import this module with the env class


class KitRuntime:
    """Kit in a RoboLab App (RoboLabEnv.process_context): started as RoboLab's evaluation runners start it (cv2 first,
    then Isaac Lab's AppLauncher, headless with cameras), its crash reports kept on the machine and its failures visible
    in its exit status."""

    KIT_ARGS = (
        # Kit uploads each crash report to NVIDIA unless its URL is empty and records telemetry events for its
        # transmitter: both off; its crash log too, which can hang the handler after a heap corruption
        "--/crashreporter/url= --/structuredLog/enable=false --/crashreporter/log=false "
        # between requests the main thread waits for its server outside Kit's loop, which Kit's hang detector would
        # take for a hang
        "--/app/hangDetector/enabled=false "
        # the experience's extensions all come with the pip install: NVIDIA's extension registry stays off
        "--/app/extensions/registryEnabled=false "
        # the extension folder watcher holds one of an App's inotify instances, limited per user and machine
        "--/app/extensions/fsWatcherEnabled=false "
        # Kit's present thread off, as the experience means it (it names the extension omni.renderer.core, so Kit
        # starts the thread): it keeps each App's GPU busy between requests
        "--/exts/omni.kit.renderer.core/present/enabled=false"
    )
    # the status of an App that failed: Kit's shutdown ends the process with 0, so a failed App exits without it and its
    # server sees the failure
    FAILED_EXIT_STATUS = 1
    # Isaac Lab's AppLauncher replaces the crash reporter's handlers of these by one that closes Kit: a crash on any
    # thread would end the App with 0, as if it had stopped by itself
    CRASH_SIGNALS = (signal.SIGSEGV, signal.SIGABRT)

    def __init__(self, context_args: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", required=True, help="the App's own RoboLab output directory")
        # the rest: Isaac Lab's AppLauncher arguments
        options, self.launcher_args = parser.parse_known_args(context_args)
        self.output_dir = Path(options.output_dir)

    @contextmanager
    def running(self) -> Iterator[None]:
        """Kit's app, from its start to its close, which ends the process; a failure inside ends the process without
        Kit's shutdown, with FAILED_EXIT_STATUS."""
        # Isaac Lab's AppLauncher and Isaac Sim's SimulationApp parse the process's command line themselves and hand it
        # to Kit: from here on it holds the launcher's arguments only
        sys.argv[1:] = self.launcher_args
        import cv2  # noqa: F401 -- before Isaac Lab, as RoboLab's runners import it
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description=self.__doc__)
        AppLauncher.add_app_launcher_args(parser)
        launcher = self.launch(AppLauncher, parser.parse_args(self.launcher_args))

        try:
            yield
        except BaseException:
            os.write(2, f"App failed\n{traceback.format_exc()}".encode())
            os._exit(self.FAILED_EXIT_STATUS)

        launcher.app.close()

    @classmethod
    def launch(cls, launcher_class: Any, args: argparse.Namespace) -> Any:
        """Isaac Lab's AppLauncher started on ``args`` with the crash reporter's handlers kept, so that a crash ends the
        App with its signal."""
        signal_function = signal.signal

        def keep(signum, handler):
            return signal.getsignal(signum) if signum in cls.CRASH_SIGNALS else signal_function(signum, handler)

        signal.signal = keep
        try:
            return launcher_class(args)
        finally:
            signal.signal = signal_function
