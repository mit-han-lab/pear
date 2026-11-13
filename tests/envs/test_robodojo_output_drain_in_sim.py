from __future__ import annotations

import os
import subprocess
import sys

import pytest

from repository import REPO

ROBODOJO_ROOT = os.environ.get("ROBODOJO_ROOT")  # the patched RoboDojo checkout, for its PhysX warning monitor
pytestmark = pytest.mark.skipif(not ROBODOJO_ROOT, reason="needs the RoboDojo checkout in ROBODOJO_ROOT")
BURST_BYTES = 4 * 1024 * 1024  # more than any pipe holds (the monitor asks for 1 MB, a spent user budget leaves 8 KB)
DONE_S = 60.0  # a drained child's write, the tail thread's scan and its exit
# An App's output plumbing (the official monitor behind KitRuntime's drain), then one write of a burst that holds the
# GIL throughout (ctypes.PyDLL), as Kit's C++ code writes inside calls from Python: a PhysX warning for env 3, then filler
CHILD = r"""
import ctypes, os, sys, time
from src.eval_client.physx_warning_monitor import get_monitor
from pear.envs.robodojo.kit_runtime import KitRuntime

log = os.dup(2)
KitRuntime.drain_kit_output(log)
while get_monitor()._tail_fp is None:  # the tail thread reads from the mirror's end once it opened it
    time.sleep(0.01)
warning = b"[1,000ms] [Warning] [omni.physx.plugin] Invalid PhysX transform detected for /World/envs/env_3/body\n"
filler = b"[1,000ms] [Warning] [omni.usd] " + b"x" * 200 + b"\n"
burst = warning + filler * (int(sys.argv[1]) // len(filler))
ctypes.PyDLL(None).write(1, burst, len(burst))
deadline = time.monotonic() + 10
while 3 not in get_monitor().get_broken_envs() and time.monotonic() < deadline:
    time.sleep(0.05)
os.write(log, b"broken %r\n" % sorted(get_monitor().get_broken_envs()))
"""


def test_a_burst_written_with_the_gil_held_reaches_the_log_and_the_monitor_finds_its_warning(tmp_path):
    log = tmp_path / "app.log"
    with log.open("wb") as out:
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD, str(BURST_BYTES)],
            cwd=ROBODOJO_ROOT,
            env={**os.environ, "PYTHONPATH": f"{REPO}:{ROBODOJO_ROOT}"},
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    # drained by tee: the write completes, the log has it, the tail thread found the warning in the mirror
    assert child.wait(timeout=DONE_S) == 0, log.read_text(errors="replace")[-2000:]
    text = log.read_text(errors="replace")
    assert "Invalid PhysX transform detected for /World/envs/env_3/" in text and "broken [3]" in text
