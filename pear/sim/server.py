from __future__ import annotations

import asyncio
import os
import queue
import socket
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError

from pear.sim.env import BaseEnv, EpisodeLost, StatefulEnv
from pear.sim.schemas import ServerContext
from pear.utils.ray_utils import ON_RAY_HEAD


class ProcessState(Enum):
    """What a server's simulator process is to the server's calls."""

    UNAVAILABLE = "unavailable"  # starting, or ended (or ending) and to be started again
    IDLE = "idle"  # past its start, for a call or an episode to acquire
    IN_USE = "in use"  # a stateless call's or a stateful episode's


class Server:
    """A Ray actor that runs the simulator processes its env class sets up and lends them out: a stateless env's for one
    call at a time (rollout), a stateful env's for one episode each, from its open to its close."""

    ROLLOUT_ATTEMPT_N = 2  # a stateless call whose process ended or hung is made once more, in another process
    FIRST_START_ATTEMPT_N = 3  # a process that ends in its first start this many times in a row means a broken setup
    FAILED_EXIT_STATUS = 1  # the exit status of a server when a process's daemon failed: Ray starts it again
    LOG_FILE_NAME = "server.log"  # the server's own log, next to its processes'

    def __init__(self, env_class: type[BaseEnv], config: dict) -> None:
        self.env_class = env_class  # Ray passes it by its module and name: the family's packages are present here
        self.name = f"{socket.gethostname()}-{os.getpid()}"  # the run's servers share a log_dir: host and pid name it
        self.log_dir = Path(str(config["log_dir"])) / self.name  # its own log and its processes'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # the processes, their states and the episode of each stateful one in use: each process's daemon (a thread)
        # keeps it running (starts it, releases it once past its start and, once it ended and nothing uses it, starts it
        # again); the calls acquire idle ones and release them, or restart them; every change under state_condition
        context = ServerContext(config, self.log_dir, [int(gpu) for gpu in ray.get_gpu_ids()], self.log)
        self.processes, max_concurrent_start_n = env_class.setup_env(context)
        self.states = [ProcessState.UNAVAILABLE] * len(self.processes)
        self.episode_uids: list[str | None] = [None] * len(self.processes)
        self.idle_indices: queue.Queue[int] = queue.Queue()  # the processes found idle, for the calls to acquire
        self.state_condition = threading.Condition()  # a process's daemon waits on it until nothing uses the process
        self.start_permits = threading.Semaphore(max_concurrent_start_n)  # each held from a start to its ready message
        self.call_threads = ThreadPoolExecutor(max_workers=len(self.processes))  # the calls' blocking waits

        # a process dies with the thread that started it (die_with_parent): its daemon, which lives as long as the
        # server and resolves the process's first start
        first_starts = [Future() for _ in self.processes]
        for index, first_start in enumerate(first_starts):
            threading.Thread(target=self._daemon, args=(index, first_start), daemon=True).start()

        # the server is up once every process got past its first start (Ray answers no call before): one that ended in
        # it FIRST_START_ATTEMPT_N times in a row means a broken setup, with which the server's creation fails at once
        for first_start in as_completed(first_starts):
            first_start.result()
        self.log(f"started {len(self.processes)} processes on {socket.gethostname()}")

    def log(self, message: str) -> None:
        """Appends ``message`` to the server's log."""
        with (self.log_dir / self.LOG_FILE_NAME).open("a") as log_file:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", file=log_file)

    def process_n(self) -> int:
        """Returns how many processes the server has: it takes one call or one episode at a time on each."""
        return len(self.processes)

    async def rollout(self, request: dict[str, Any]) -> bytes:
        """Returns an idle process's reply to a stateless env's ``request`` (a replay and one action) as the process
        pickled it."""
        # a replay needs nothing of a process that ended or hung: the call is made again in another
        for _ in range(self.ROLLOUT_ATTEMPT_N):
            index = await self._acquire(None)
            try:
                reply = await self._call(index, request, self.env_class.ROLLOUT_TIMEOUT_S)
            except EpisodeLost as exc:  # its process ended or hung: its daemon starts it again
                last_error = exc
                continue
            self._release(index)
            return reply

        raise RuntimeError(f"the simulator process failed {self.ROLLOUT_ATTEMPT_N} times in a row: {last_error}")

    async def open(self, uid: str, env_args: dict[str, Any]) -> bytes:
        """Returns an idle process's reply to the open of the episode ``uid`` of ``env_args`` as the process pickled it;
        the process keeps the episode until its close."""
        index = await self._acquire(uid)

        return await self._call(index, {"op": StatefulEnv.OPEN_OP, "env_args": env_args}, self.env_class.OPEN_TIMEOUT_S)

    async def step(self, uid: str, action: Any) -> bytes:
        """Returns the reply of the episode ``uid``'s process to ``action`` as the process pickled it."""
        index = self._index(uid)
        if index is None:  # its process ended, or the server started again since the open
            raise EpisodeLost(f"{self.name}: the episode {uid} is gone")

        return await self._call(index, {"op": StatefulEnv.STEP_OP, "action": action}, self.env_class.STEP_TIMEOUT_S)

    async def close(self, uid: str) -> None:
        """Ends the episode ``uid``: its process closes it and exits, and its daemon starts it again."""
        index = self._index(uid)
        if index is not None:  # (not when the episode was lost with its process)
            self._restart(index)

    def _daemon(self, index: int, first_start: Future) -> None:
        """Keeps process ``index`` running for the server's life and resolves ``first_start`` with the outcome of its
        first start."""
        process = self.processes[index]
        try:
            # a start, then the wait for its end, again and again: a process is started anew each time it ended
            while True:
                with self.start_permits:
                    process.start()
                    try:
                        is_ready = process.wait_ready(self.env_class.START_TIMEOUT_S)
                    except TimeoutError:  # hung in its start: the process and this start permit would never come back
                        self.log(f"{process.name} hung in start {process.start_n}, killed")
                        process.wait_or_kill(0)
                        is_ready = False
                if is_ready:  # past its start: idle, for a call to acquire
                    self._release(index)
                if is_ready and not first_start.done():
                    first_start.set_result(None)

                # until it ended (in its start, at its episode's close, in a crash or killed) and nothing uses it
                process.wait_or_kill(None)
                with self.state_condition:
                    self.state_condition.wait_for(lambda: self.states[index] != ProcessState.IN_USE)
                    self.states[index] = ProcessState.UNAVAILABLE
                process.close()
                if process.poll() != 0:  # 0: its episode ended
                    self.log(f"{process.name} exited with {process.poll()} after start {process.start_n}")

                if not first_start.done() and process.start_n == self.FIRST_START_ATTEMPT_N:  # a broken setup
                    first_start.set_exception(
                        RuntimeError(
                            f"{self.name}: {process.name} ended in its start "
                            f"{self.FIRST_START_ATTEMPT_N} times in a row; see "
                            f"{process.log_path}"
                        )
                    )
                    return
        except BaseException:  # a failure of the daemon itself: the server ends, visibly, and Ray starts it again
            self.log(f"failed\n{traceback.format_exc()}")
            os._exit(self.FAILED_EXIT_STATUS)

    async def _acquire(self, uid: str | None) -> int:
        """Acquires an idle process (waiting for one past its start) for a stateless call, or for the episode ``uid``
        if one is given, and returns its index."""
        while True:
            index = await asyncio.get_running_loop().run_in_executor(self.call_threads, self.idle_indices.get)
            with self.state_condition:  # (a process may have ended since it was found idle)
                if self.states[index] == ProcessState.IDLE:
                    self.states[index], self.episode_uids[index] = ProcessState.IN_USE, uid
                    return index

    async def _call(self, index: int, request: dict[str, Any], timeout: float) -> bytes:
        """Returns process ``index``'s reply to ``request``; if the process ended or did not reply within ``timeout``
        (it is then killed), restarts it and raises EpisodeLost."""
        process, loop = self.processes[index], asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.call_threads, process.call, request, timeout)
        except TimeoutError as exc:  # hung: killed at once
            error = exc
            await loop.run_in_executor(self.call_threads, process.wait_or_kill, 0)
        except (EOFError, OSError) as exc:  # it crashed, or it ended because its env could not go on (EpisodeLost)
            error = exc

        self._restart(index)
        raise EpisodeLost(f"{self.name} {process.name} after start {process.start_n}: {type(error).__name__}: {error}")

    def _release(self, index: int) -> None:
        """Releases process ``index`` as idle (past its start, or after a stateless call) for the next call to
        acquire."""
        with self.state_condition:
            self.states[index] = ProcessState.IDLE
            self.state_condition.notify_all()  # its daemon, if it ended meanwhile
        self.idle_indices.put(index)

    def _restart(self, index: int) -> None:
        """Restarts process ``index``: closes its pipe, which ends it (one that crashed or was killed ended already),
        and leaves it unavailable until its daemon started it again."""
        self.processes[index].close()
        with self.state_condition:
            self.states[index], self.episode_uids[index] = ProcessState.UNAVAILABLE, None
            self.state_condition.notify_all()

    def _index(self, uid: str) -> int | None:
        """Returns the index of the process that has the episode ``uid``, None if none has it."""
        with self.state_condition:
            return self.episode_uids.index(uid) if uid in self.episode_uids else None


class ServerPool:
    """A Ray actor on the cluster's head machine that starts the run's servers of one env class and lends them to the
    agent loops: the driver creates it (start) and holds it for the run, and each agent-loop process's client finds it
    by its name (name_of)."""

    NAMESPACE = "pear"  # the Ray namespace of the pools' names
    # the pool's calls in progress at once: each acquire that waits for a free server is one, and Ray's default of 1000
    # could leave no room for the releases that end those waits
    MAX_CONCURRENCY = 100_000
    SERVER_MAX_RESTARTS = -1  # of a server by Ray: without a limit

    def __init__(self, env_class: type[BaseEnv], config: dict, server_n: int) -> None:
        # Ray starts a server that went away again, wherever it finds the resources
        server_class = ray.remote(Server).options(max_restarts=self.SERVER_MAX_RESTARTS, **env_class.SERVER_RESOURCES)
        self.servers = [server_class.remote(env_class, config) for _ in range(server_n)]
        self.free_servers: asyncio.Queue[ActorHandle] = asyncio.Queue()

    @classmethod
    def start(cls, env_class: type[BaseEnv], config: dict) -> ActorHandle:
        """Starts the pool of env_class's servers, as many as the env class counts on this cluster, and returns it once
        every server started."""
        # counted before any actor exists: a cluster that cannot hold the servers ends the start here, with the
        # family's reason
        server_n = env_class.server_n(config, ray.cluster_resources())

        pool_class = ray.remote(cls).options(
            name=cls.name_of(env_class),
            namespace=cls.NAMESPACE,
            num_cpus=0,
            resources=ON_RAY_HEAD,
            max_concurrency=cls.MAX_CONCURRENCY,
        )
        pool = pool_class.remote(env_class, config, server_n)

        try:
            ray.get(pool.fill.remote(), timeout=env_class.START_TIMEOUT_S)
        except GetTimeoutError as exc:  # a server never started: no machine offers its resources, or its processes hang
            raise RuntimeError(
                f"{env_class.__qualname__}'s servers did not start within "
                f"{env_class.START_TIMEOUT_S:.0f}s (SERVER_RESOURCES "
                f"{env_class.SERVER_RESOURCES})"
            ) from exc

        return pool  # for the caller to hold through the run: Ray ends an actor nobody holds a handle of

    async def fill(self) -> None:
        """Puts each server in the free list once per process it has, which the server answers once it is up, the
        servers in turn."""
        process_ns = await asyncio.gather(*(server.process_n.remote() for server in self.servers))
        # in turn, so that the calls spread over every server's GPU and machine: one server's processes first would
        # crowd a run that uses fewer than all of them onto a few GPUs, and one lost machine would take all its episodes
        for index in range(max(process_ns)):
            for server, process_n in zip(self.servers, process_ns):
                if index < process_n:
                    self.free_servers.put_nowait(server)

    async def acquire(self) -> ActorHandle:
        """Takes a free server off the free list (waiting for one) and returns it."""
        return await self.free_servers.get()

    async def release(self, server: ActorHandle) -> None:
        """Puts ``server`` back in the free list (on the pool's event loop, like every change of the list)."""
        self.free_servers.put_nowait(server)

    @staticmethod
    def name_of(env_class: type[BaseEnv]) -> str:
        return f"servers of {env_class.__qualname__}"
