import os
import subprocess
import sys
from pathlib import Path

import pytest

from repository import REPO

SITE_DIR = REPO / "pear" / "utils" / "sitecustomize"  # the launchers put it on every process's path
# the interpreter's own descriptor table size
SIZE_PROBE = "print(next(line.split()[1] for line in open('/proc/self/status') if line.startswith('FDSize:')))"
# the table size of a program the interpreter starts (grep reads its own /proc/self/status), as ray start starts the raylet
CHILD_SIZE_PROBE = (
    "import subprocess; print(subprocess.run(['grep', 'FDSize:', '/proc/self/status'], capture_output=True,"
    " text=True, check=True).stdout.split()[1])"
)
# the size PEAR's sitecustomize grows a table to (its FD_TABLE_SIZE, or the limit on open files where that is lower),
# read in the probe's interpreter: run in this process it would presize the test's own table, which every interpreter
# it starts would inherit
TARGET_PROBE = (
    "import resource, sitecustomize; print(min(sitecustomize.FDTable.FD_TABLE_SIZE,"
    " resource.getrlimit(resource.RLIMIT_NOFILE)[0]))"
)
# an isolated interpreter (without PEAR's sitecustomize) starts the probe's: a child's table is sized by its parent's
# highest open descriptor, so the probe starts from a small table whatever this process holds
LAUNCHER = "import subprocess, sys; sys.exit(subprocess.run(sys.argv[1:]).returncode)"
REQUIRES_PROC = pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="FDSize comes from Linux's /proc")


class Interpreter:
    """A fresh Python interpreter with or without PEAR's sitecustomize on its path, running a probe that prints a number."""

    @staticmethod
    def number(has_site: bool, probe: str) -> int:
        env = {**os.environ, "PYTHONPATH": str(SITE_DIR) if has_site else ""}
        command = [sys.executable, "-I", "-c", LAUNCHER, sys.executable, "-c", probe]
        return int(subprocess.run(command, env=env, capture_output=True, text=True, check=True).stdout)


@REQUIRES_PROC
def test_an_interpreter_with_pear_site_on_its_path_starts_with_a_presized_descriptor_table():
    target = Interpreter.number(has_site=True, probe=TARGET_PROBE)

    assert Interpreter.number(has_site=True, probe=SIZE_PROBE) >= target
    assert Interpreter.number(has_site=False, probe=SIZE_PROBE) < target


@REQUIRES_PROC
def test_a_program_that_a_presized_interpreter_starts_inherits_the_presized_table():
    target = Interpreter.number(has_site=True, probe=TARGET_PROBE)

    assert Interpreter.number(has_site=True, probe=CHILD_SIZE_PROBE) >= target
    assert Interpreter.number(has_site=False, probe=CHILD_SIZE_PROBE) < target
