from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys

import pytest

from pear.envs.robodojo.apps import RoboDojoApps

pytestmark = pytest.mark.skipif(sys.platform != "linux" or shutil.which("gcc") is None, reason="needs Linux and gcc")
IN_IGNORED = 0x8000  # <sys/inotify.h>
# inotify through libc, as an App's libraries call it: the instance's file, watches of a directory, of a missing path
# and of a file under IN_ONLYDIR, then the event a removed watch queues
CHILD = r"""
import ctypes, os, struct, sys
IN_CREATE, IN_ONLYDIR = 0x100, 0x1000000
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
print(os.readlink(f"/proc/self/fd/{fd}"))
for path, mask in ((b"/", IN_CREATE), (b"/no/such/path", IN_CREATE), (sys.executable.encode(), IN_CREATE | IN_ONLYDIR)):
    print(libc.inotify_add_watch(fd, path, mask), ctypes.get_errno())
libc.inotify_rm_watch(fd, 1)
print(*struct.unpack("iIII", os.read(fd, 16)))
"""


def test_an_app_s_inotify_holds_no_kernel_instance(tmp_path):
    library = tmp_path / "libno_inotify.so"
    RoboDojoApps.build_no_inotify(library)

    out = subprocess.run(
        [sys.executable, "-c", CHILD],
        env={**os.environ, "LD_PRELOAD": str(library)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    assert out[0].startswith("pipe:")  # a kernel instance reads anon_inode:inotify
    assert out[1].split()[0] == "1"
    assert out[2].split() == ["-1", str(errno.ENOENT)]
    assert out[3].split() == ["-1", str(errno.ENOTDIR)]
    assert out[4].split() == ["1", str(IN_IGNORED), "0", "0"]
