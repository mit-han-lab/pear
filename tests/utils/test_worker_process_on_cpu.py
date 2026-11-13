import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection

import pytest

from pear.utils.worker_process import WorkerProcess

START_TIMEOUT_S = 60.0  # for a worker to get through its start: a new interpreter imports this module in seconds
EXIT_TIMEOUT_S = 10.0  # for a worker to end once it got a signal
IGNORED_SIGTERM_S = 0.5  # what the worker that ignores SIGTERM is given before SIGKILL
GRACE_S = 0.1  # what a worker that does not end by itself is given before SIGTERM
SLEEP_S = 60.0  # how long a worker sleeps that its test ends first


class Workers:
    """The workers of the tests: each has a pipe to its test, says through it that it started, and sleeps."""

    @staticmethod
    def started(
        function: Callable[[Connection], None], exit_timeout_s: float, is_group_leader: bool
    ) -> tuple[WorkerProcess, Connection]:
        """Starts a worker that runs ``function`` and returns it, once it said it started, with the test's end of its
        pipe."""
        receiver, sender = WorkerProcess.CONTEXT.Pipe(duplex=False)
        worker = WorkerProcess(function, (sender,), function.__name__, exit_timeout_s, is_group_leader)

        worker.start()
        sender.close()  # the pipe closes once every process that holds its sending end has ended
        assert receiver.poll(START_TIMEOUT_S), "the worker did not get through its start"
        receiver.recv()

        return worker, receiver

    @staticmethod
    def sleep(sender: Connection) -> None:
        """Sleeps."""
        sender.send(None)
        time.sleep(SLEEP_S)

    @staticmethod
    def ignore_sigterm(sender: Connection) -> None:
        """Sleeps through SIGTERM."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sender.send(None)
        time.sleep(SLEEP_S)

    @staticmethod
    def lead_a_group(sender: Connection) -> None:
        """Leads a group of its own with a sleeping process it started in it, which holds the pipe's sending end too."""
        os.setsid()
        sleep_command = [sys.executable, "-c", f"import time; time.sleep({SLEEP_S})"]
        sleeper = subprocess.Popen(sleep_command, pass_fds=(sender.fileno(),))
        sender.send(None)
        sleeper.wait()


def test_a_worker_that_ends_by_itself_within_the_grace_gets_no_signal():
    worker = WorkerProcess(time.sleep, (0,), "worker", EXIT_TIMEOUT_S, is_group_leader=False)

    worker.start()
    worker.stop(START_TIMEOUT_S)

    assert worker.process.exitcode == 0


def test_a_worker_that_outlives_the_grace_is_terminated_and_can_be_stopped_again():
    worker, _ = Workers.started(Workers.sleep, EXIT_TIMEOUT_S, is_group_leader=False)

    worker.stop(GRACE_S)
    worker.stop(GRACE_S)

    assert not worker.is_alive()
    assert worker.process.exitcode == -signal.SIGTERM


def test_a_worker_that_ignores_sigterm_is_killed():
    worker, _ = Workers.started(Workers.ignore_sigterm, IGNORED_SIGTERM_S, is_group_leader=False)

    worker.stop(0)
    worker.process.join(EXIT_TIMEOUT_S)  # the wait after SIGKILL is as short as the one after SIGTERM

    assert worker.process.exitcode == -signal.SIGKILL


def test_the_processes_a_group_leader_started_end_with_it():
    worker, receiver = Workers.started(Workers.lead_a_group, EXIT_TIMEOUT_S, is_group_leader=True)

    worker.stop(0)

    assert worker.process.exitcode == -signal.SIGTERM
    # the worker has ended: the pipe closes when the process it started has ended too
    assert receiver.poll(EXIT_TIMEOUT_S)
    with pytest.raises(EOFError):
        receiver.recv()


def test_a_group_leader_that_leads_no_group_yet_is_signalled_alone():
    worker, _ = Workers.started(Workers.sleep, EXIT_TIMEOUT_S, is_group_leader=True)

    worker.stop(0)

    assert worker.process.exitcode == -signal.SIGTERM


def test_a_worker_whose_start_failed_is_stopped_without_an_error():
    # a lock of this process's threads cannot be given to another process
    worker = WorkerProcess(time.sleep, (threading.Lock(),), "worker", EXIT_TIMEOUT_S, is_group_leader=False)
    with pytest.raises(TypeError):
        worker.start()

    worker.stop(GRACE_S)

    assert not worker.is_alive()
