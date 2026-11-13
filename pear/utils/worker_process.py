import multiprocessing
import os
import signal
from collections.abc import Callable
from typing import Any


class WorkerProcess:
    """A daemon process that runs one function in a new interpreter of this Python environment and is ended within
    time limits."""

    # makes the process and whatever its function is given: pipes, queues, locks, events, shared arrays. Spawn: a new
    # interpreter, which takes over none of the threads, the locks and the open devices of the process that starts it
    CONTEXT = multiprocessing.get_context("spawn")

    def __init__(
        self,
        target: Callable[..., None],
        arguments: tuple[Any, ...],
        name: str,
        exit_timeout_s: float,
        is_group_leader: bool,
    ) -> None:
        # a daemon: the interpreter that started it terminates it at its own exit. Not die_with_parent, which ends a
        # process with the thread that started it: a worker may be started by a thread that ends before it
        self.process = self.CONTEXT.Process(target=target, args=arguments, name=name, daemon=True)
        self.exit_timeout_s = exit_timeout_s  # after SIGTERM, and again after SIGKILL, for the process to end
        # whether the function makes its process the leader of a group of its own (os.setsid), in which the processes
        # it starts are: a signal then goes to the whole group
        self.is_group_leader = is_group_leader

    def start(self) -> None:
        """Starts the process."""
        self.process.start()

    def stop(self, grace_s: float) -> None:
        """Waits up to ``grace_s`` seconds for the process to end by itself, then ends it with SIGTERM and, if it is
        still there ``exit_timeout_s`` seconds later, with SIGKILL."""
        if self.is_alive():
            self.process.join(grace_s)

        if self.is_alive():
            self._send(signal.SIGTERM, self.process.terminate)
            self.process.join(self.exit_timeout_s)

        if self.is_alive():
            self._send(signal.SIGKILL, self.process.kill)
            self.process.join(self.exit_timeout_s)  # alive after this too: stuck in the kernel; is_alive tells

    def is_alive(self) -> bool:
        """Whether the process runs: False before its start, when its start failed and once it ended."""
        return self.process.is_alive()

    def _send(self, signal_number: int, process_method: Callable[[], None]) -> None:
        """Sends a signal to the group the process leads or, through its own method for that signal, to it alone."""
        if not self.is_group_leader:
            process_method()
            return

        try:
            os.killpg(self.process.pid, signal_number)
        except ProcessLookupError:  # no group of its own: it has not come to os.setsid yet, or it just ended
            process_method()
