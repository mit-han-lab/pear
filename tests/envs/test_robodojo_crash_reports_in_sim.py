from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys

import pytest

from repository import REPO

KIT_ROOT = os.environ.get("ROBODOJO_KIT_ROOT")  # a warmed-up Kit root (a server's template/kit): Kit starts offline
pytestmark = pytest.mark.skipif(
    not KIT_ROOT, reason="needs a warmed-up Kit root in ROBODOJO_KIT_ROOT (the Isaac container)"
)
APP_S = 600.0  # Kit starts and the crash ends the App within this
# An App launched as KitRuntime launches it, then a write to address 0 on a second thread through a call that releases
# the GIL, as on one of Kit's native threads; the main thread returns to Python every 0.1 s, as an App at work does
# (where the handler Isaac Lab's AppLauncher would install runs)
CHILD = r"""
import argparse, ctypes, sys, threading, time
from isaaclab.app import AppLauncher
from pear.envs.robodojo.kit_runtime import KitRuntime

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--device", "cpu", "--kit_args",
                          f"--portable --portable-root {sys.argv[1]} --/physics/cudaDevice=-1"])
launcher = KitRuntime.launch(AppLauncher, args)
threading.Thread(target=ctypes.memset, args=(0, 0, 1), daemon=True).start()
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    time.sleep(0.1)
print("outlived the crash", flush=True)
"""


@pytest.fixture
def offline():
    # the crashing App runs without network, so that no report can leave whatever the settings do
    if shutil.which("unshare") is None or subprocess.run(["unshare", "-n", "true"]).returncode != 0:
        pytest.skip("needs unshare -n (the Isaac container)")


def test_a_crash_on_a_second_thread_ends_the_app_with_its_signal_and_its_dump_stays_on_the_node(tmp_path, offline):
    root = tmp_path / "kit"
    shutil.copytree(KIT_ROOT, root, symlinks=True)
    shutil.rmtree(root / "logs", ignore_errors=True)  # the warm-up's Kit logs
    log = tmp_path / "app.log"
    with log.open("wb") as out:
        status = subprocess.run(
            ["unshare", "-n", sys.executable, "-c", CHILD, str(root)],
            env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}", "CUDA_VISIBLE_DEVICES": ""},
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=APP_S,
        ).returncode
    # the App's output and Kit's own log, whose info lines (the reporter's first) the output lacks
    text = log.read_text(errors="replace") + "".join(
        path.read_text(errors="replace") for path in (root / "logs").rglob("kit_*.log")
    )
    assert "[carb.crashreporter-breakpad.plugin] upload enabled" in text, text[-3000:]
    # the reporter's handler: the minidump written and kept, no crash log, the App ended by the signal, nothing sent
    assert status == -signal.SIGSEGV, text[-3000:]
    assert "Wrote dump file" in text and "'/crashreporter/url' is empty" in text, text[-3000:]
    assert "backtrace follows" not in text and "Crash metadata" not in text, text[-3000:]
    assert "Uploading minidump" not in text and "telemetry.transmitter" not in text, text[-3000:]
