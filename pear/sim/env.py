from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

from pear.sim.schemas import EnvResult, ServerContext

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.sim.process import SimulatorProcess


class BaseEnv(ABC):
    """The base of every family's env class: how a server sets up and runs its simulator processes, what a server needs
    from Ray, and the facts of its conversation that the environment tool shows the model."""

    # the dotted name of its kind's client class (the class itself would import Ray into the simulator processes)
    CLIENT_CLASS_PATH: str
    # what Ray reserves for each server: Ray places the servers on the machines that offer it
    SERVER_RESOURCES: dict[str, Any]
    # how long the servers may take to start all their processes (past it the run's start fails), and so any one start
    # of a process (past it the process counts as hung: it is killed and started again)
    START_TIMEOUT_S: float
    # the family's conversation (EnvironmentTool)
    SCHEMA: dict[str, tuple[str, ...]]  # the parts and the quantities an action sets
    CAMERAS: tuple[str, ...]  # the frames' names, in the order of a result's images
    ENVIRONMENT_PROMPT: str  # the facts the protocol leaves out, in the family's words; empty: no ENVIRONMENT section
    LOG_KEYS: tuple[str, ...] = ()  # the keys of its results the call log records besides the protocol's

    @classmethod
    def server_n(cls, config: dict[str, Any], cluster_resources: dict[str, float]) -> int:
        """Returns how many servers a run with the tool config ``config`` starts on a Ray cluster that offers
        ``cluster_resources`` (by default the tool config's ``server_n``)."""
        return int(config["server_n"])

    @classmethod
    def session_cap(cls, config: dict[str, Any]) -> int | None:
        """Returns how many of a run's sessions the servers of the tool config ``config`` take at once (by default
        None: the family sets no cap)."""
        return None

    @classmethod
    @abstractmethod
    def setup_env(cls, context: ServerContext) -> tuple[list[SimulatorProcess], int]:
        """Sets up the env for the server of ``context`` from the machine's variables: its simulator processes (each
        logging into context.log_dir) and what they share; returns the processes with how many of them may start at
        once."""

    @classmethod
    @contextmanager
    def process_context(cls, context_args: list[str]) -> Iterator[None]:
        """Starts what a simulator process needs before its first episode and stops it after its last (by default
        nothing); ``context_args`` are the process's command-line arguments after ``--``."""
        yield

    @classmethod
    @abstractmethod
    def run(cls, conn: Connection) -> None:
        """Replies to the server's requests on ``conn`` with envs of the class until the server closes the pipe."""

    @classmethod
    @abstractmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        """Returns the task of an episode's env args, under which the progress panel counts its successes."""

    @classmethod
    def environment_prompt(cls, result: EnvResult) -> str:
        """Returns the family's facts for the episode of ``result``."""
        return cls.ENVIRONMENT_PROMPT


class StatelessEnv(BaseEnv):
    """The base of an env whose process keeps nothing of an episode between calls: each call rebuilds the episode from
    its env args, replays its native actions and carries out one more action."""

    CLIENT_CLASS_PATH = "pear.sim.client.StatelessClient"
    ROLLOUT_TIMEOUT_S: float  # how long one call (a replay and an action) may take before its process counts as hung

    @abstractmethod
    def rollout(self, history: list[Any], action: Any) -> EnvResult:
        """Replays ``history`` (the native actions so far), carries out ``action`` (None: no action, the episode's
        initial observation) and returns the result."""

    @classmethod
    def run(cls, conn: Connection) -> None:
        env_by_args: dict[str, StatelessEnv] = {}  # each built when its first call comes
        conn.send("ready")

        while True:
            try:
                request = conn.recv()
            except EOFError:  # the server closed the pipe
                break

            try:
                args_key = json.dumps(request["env_args"], sort_keys=True)
                if args_key not in env_by_args:
                    env_by_args[args_key] = cls(**request["env_args"])
                reply = env_by_args[args_key].rollout(request["history"], request["action"])
            except Exception:  # reported to the client: the trajectory fails with it
                reply = {"error": traceback.format_exc()}

            conn.send(reply)


class StatefulEnv(BaseEnv):
    """The base of an env that keeps one episode in its process from the open to the close; it raises EpisodeLost when
    the episode cannot go on."""

    CLIENT_CLASS_PATH = "pear.sim.client.StatefulClient"
    # the ops of the requests: a warm-up request (SimulatorProcess.warm_up) builds an episode as an open does, and its
    # reply holds nothing of it
    OPEN_OP = "open"
    STEP_OP = "step"
    WARM_UP_OP = "warm"
    # how long an open and a step may take before the server kills the process and the episode is lost (the wait for an
    # idle process does not count)
    OPEN_TIMEOUT_S: float
    STEP_TIMEOUT_S: float

    @abstractmethod
    def open(self) -> EnvResult:
        """Builds the episode and returns its initial observation."""

    @abstractmethod
    def step(self, action: Any) -> EnvResult:
        """Carries out ``action`` in the episode and returns its result."""

    @abstractmethod
    def close(self) -> None:
        """Releases the episode's scene, if it was built."""

    @classmethod
    def run(cls, conn: Connection) -> None:
        # one episode per process: the env its open builds; a warm-up request builds the episode, and its reply holds
        # nothing of it
        env = None
        conn.send("ready")  # past its start: its server may start the next process

        try:
            while True:
                try:
                    request = conn.recv()
                except EOFError:  # the episode's end: the server closed the pipe
                    break

                try:
                    if request["op"] == cls.STEP_OP:
                        reply = env.step(request["action"])
                    else:  # open or warm up
                        env = cls(**request["env_args"])
                        result = env.open()
                        reply = result if request["op"] == cls.OPEN_OP else {}
                except EpisodeLost:  # the env cannot go on: the process ends, and its episode is lost
                    break
                except Exception:  # reported to the client, whose trajectory fails with it
                    reply = {"error": traceback.format_exc()}

                conn.send(reply)
        finally:
            if env is not None:
                env.close()


class EpisodeLost(RuntimeError):
    """Raised when an episode cannot go on (its simulator process or server is gone, or its env gave up on it); its
    trajectory starts over in a new episode."""
