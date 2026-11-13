import os
import resource


class FDTable:
    """The process's file-descriptor table, grown once at interpreter start while the process has one thread."""

    FD_TABLE_SIZE = 16384  # descriptors a process's table holds before the kernel grows it

    @staticmethod
    def presize() -> None:
        """Grow the table to FD_TABLE_SIZE descriptors (or the soft limit) by keeping one open at its last slot."""
        # a multi-threaded process waits for an RCU grace period whenever its table grows (expand_fdtable); that wait
        # holds gRPC's accept thread in accept4(), so a burst of callers times out before the server's SETTINGS frame.
        # With one thread the kernel grows the table without that wait.
        last_fd = min(FDTable.FD_TABLE_SIZE, resource.getrlimit(resource.RLIMIT_NOFILE)[0]) - 1
        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            if fd < last_fd:
                # a forked child's table is sized by its parent's highest open descriptor and exec keeps that size: left
                # open (close-on-exec), this one presizes the programs the process starts, as ray start's raylet
                os.dup2(fd, last_fd, inheritable=False)
        finally:
            os.close(fd)


FDTable.presize()
