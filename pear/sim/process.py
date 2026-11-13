from __future__ import annotations

import argparse
import os
import pickle
import pkgutil
import signal
import subprocess
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from pear.sim.env import BaseEnv, StatefulEnv
from pear.utils.system_utils import die_with_parent


class SimulatorProcess:
    """One simulator process of a server: it starts the process (again each time it ended) with a pipe to it and a log
    file per start."""

    KILL_TIMEOUT_S = 10.0  # after SIGKILL, before a process stuck in the kernel is given up on
    # this repository's root, first on a process's path: its Python runs this module out of its server's tree
    PEAR_ROOT = Path(__file__).resolve().parents[2]

    def __init__(
        self,
        name: str,
        env_class: type[BaseEnv],
        python: str,
        context_args: list[str],
        variables: dict[str, str],
        cwd: str | Path,
        log_dir: Path,
    ) -> None:
        self.name = name
        self.command = [
            python,
            "-u",
            "-m",
            "pear.sim.process",
            "--env",
            f"{env_class.__module__}.{env_class.__qualname__}",
        ]  # the class the process imports
        self.context_args = context_args  # its process context's arguments, after --
        # the family's variables for it, with PEAR_ROOT ahead of the family's own entries of the path
        python_path = os.pathsep.join(filter(None, [str(self.PEAR_ROOT), variables.get("PYTHONPATH")]))
        self.variables = {**variables, "PYTHONPATH": python_path}
        self.cwd = cwd
        self.log_dir = log_dir

        self.popen: subprocess.Popen | None = None
        self.conn: Connection | None = None  # the pipe to the process, from its start until it is closed
        self.start_n = 0
        self.log_path: Path | None = None  # the current start's log file

    def warm_up(self, env_args: dict[str, Any]) -> str | None:
        """Starts the process of a stateful env, has it build the episode of ``env_args`` and end; returns the build's
        error ("no reply" when the process ended in it), None when it built."""
        self.start()
        try:  # a warm-up request: its reply holds nothing of the episode, which the caller's Python might not unpickle
            request = {"op": StatefulEnv.WARM_UP_OP, "env_args": env_args}
            reply = self.call(request, None) if self.wait_ready(None) else None
        except (EOFError, OSError):  # it ended in the build
            reply = None
        self.close()
        self.wait_or_kill(None)  # it closes the episode and exits

        return "no reply" if reply is None else pickle.loads(reply).get("error")

    def start(self) -> None:
        """Starts the process (again) with a new pipe to it, whose other end it gets as --fd."""
        conn, process_end = Pipe()
        self.start_n += 1
        self.log_path = self.log_dir / f"{self.name}-{self.start_n}.log"

        # die_with_parent's signal comes when the thread that started the process ends, not when its whole server does:
        # a server starts each of its processes on the process's daemon, a thread that lives as long as the server
        with self.log_path.open("ab") as log_file:
            self.popen = subprocess.Popen(
                [*self.command, "--fd", str(process_end.fileno()), "--", *self.context_args],
                cwd=self.cwd,
                env=self.variables,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                pass_fds=(process_end.fileno(),),
                start_new_session=True,
                preexec_fn=die_with_parent,
            )

        process_end.close()  # only the process's copy stays open: its exit closes the pipe
        self.conn = conn

    def wait_ready(self, timeout: float | None) -> bool:
        """Waits up to ``timeout`` seconds (None: as long as it takes) for the ready message the process sends once it
        got past its start; returns False if the process ended first and raises TimeoutError if neither came."""
        if not self.conn.poll(timeout):  # (an ended process leaves the pipe readable: its end comes as EOFError)
            raise TimeoutError(f"not past its start within {timeout:.0f}s")

        try:
            self.conn.recv_bytes()
        except EOFError:  # it ended in its start
            return False

        return True

    def call(self, request: Any, timeout: float | None) -> bytes:
        """Sends ``request`` and returns the process's reply as it pickled it; raises EOFError or OSError if the
        process ended and TimeoutError if no reply came within ``timeout`` seconds (None: as long as it takes)."""
        conn = self.conn  # one pipe for the whole exchange, even if the process is started again meanwhile
        conn.send(request)
        if not conn.poll(timeout):
            raise TimeoutError(f"no reply within {timeout:.0f}s")

        return conn.recv_bytes()

    def close(self) -> None:
        """Closes the pipe to the process, which tells it its work is over: it ends by itself."""
        self.conn.close()

    def wait_or_kill(self, timeout: float | None) -> None:
        """Waits up to ``timeout`` seconds (None: as long as it takes) for the process to end, then kills its group."""
        try:
            self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except ProcessLookupError:  # the group ended since the wait gave up
                pass
            try:
                self.popen.wait(timeout=self.KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # stuck in the kernel: given up on, the next start is a new process
                pass

    def poll(self) -> int | None:
        """Returns the process's exit status once it ended, None while it runs."""
        return self.popen.poll()


class SimulatorCommand:
    """The entry point of every simulator process (python -m pear.sim.process --env module.Class --fd N -- ARGS): it
    replies to its server's requests on the pipe of fd N with envs of the class, inside the class's process context."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.env_class: type[BaseEnv] = pkgutil.resolve_name(args.env)  # the simulator's packages are present here
        self.context_args: list[str] = args.context_args
        self.fd: int = args.fd

    @classmethod
    def from_command_line(cls) -> SimulatorCommand:
        """The command of the env class, the pipe and the context args the command line names."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--fd", type=int, required=True, help="its end of its server's pipe")
        parser.add_argument("--env", required=True, help="the env class, module.Class")
        parser.add_argument(
            "context_args", nargs="*", help="the arguments of the env class's process context, after --"
        )

        return cls(parser.parse_args())

    def run(self) -> None:
        """Runs the simulator process until its server closes the pipe."""
        with self.env_class.process_context(self.context_args):
            self.env_class.run(Connection(self.fd))


if __name__ == "__main__":
    SimulatorCommand.from_command_line().run()
