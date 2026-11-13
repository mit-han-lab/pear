import ctypes
import signal

_PR_SET_PDEATHSIG = 1  # prctl option (linux/prctl.h): the signal a child gets when its parent dies


def die_with_parent() -> None:
    """``preexec_fn`` for child processes: SIGKILL the child when the process that spawned it disappears."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
