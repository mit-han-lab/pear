from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest
import ray

from pear.sim.client import StatelessClient
from pear.sim.server import Server, ServerPool
from repository import REPO
from sim_servers import INSPECTED_POOL, StartedPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="a process dies with its server through prctl (Linux)"),
]
SLOT_N = 2  # the units of sim the tests' Ray clusters offer, a server each
HUNG_ROLLOUT_TIMEOUT_S = 5.0  # the servers' call timeout where a test hangs a call: far past a reply's milliseconds
# The servers' env class: its rollout returns the process's pid, its process context's arguments, the history and
# the action; an action can ask it to raise, to exit or to hang, every time or only in the first process that meets the
# marker file it names; its servers take their call timeout from the tool config, if it gives one
ENV = r"""
import os, sys, time
from contextlib import contextmanager
from pathlib import Path

import pear
from pear.sim.env import StatelessEnv
from pear.sim.process import SimulatorProcess
from pear.sim.server import ServerPool

REPO = str(Path(pear.__file__).resolve().parents[1])


class EchoEnv(StatelessEnv):
    SERVER_RESOURCES = {"num_cpus": 0, "resources": {"sim": 1}}
    START_TIMEOUT_S = 60.0  # a child imports Python and pear in seconds
    ROLLOUT_TIMEOUT_S = 60.0  # a reply takes milliseconds, and seconds on a machine that is busy
    context_args = None  # those of the process context the process serves in, while it runs

    @classmethod
    def setup_env(cls, context):
        if context.config["rollout_timeout_s"]:  # of the test that hangs a call
            cls.ROLLOUT_TIMEOUT_S = context.config["rollout_timeout_s"]

        # one process, which finds this module on the path the test's Ray runtime env gives the server
        return [SimulatorProcess("sim", cls, sys.executable, [], dict(os.environ), REPO, context.log_dir)], 1

    @classmethod
    def task_name(cls, env_args):
        return env_args["task"]

    @classmethod
    @contextmanager
    def process_context(cls, context_args):
        cls.context_args = context_args
        yield
        cls.context_args = None

    def __init__(self, task):
        self.task = task

    def rollout(self, history, action):
        what = action.get("do") if isinstance(action, dict) else None
        if what is not None and ("once" not in action or not Path(action["once"]).exists()):
            if "once" in action:
                Path(action["once"]).touch()
            if what == "raise":
                raise ValueError("the env raised")
            if what == "exit":
                os._exit(3)
            time.sleep(3600)
        actions = [] if action is None else [action]
        return {"pid": os.getpid(), "context": self.context_args, "task": self.task, "history": list(history),
                "actions": actions}


class UnimportableEchoEnv(EchoEnv):
    @classmethod
    def setup_env(cls, context):
        # its process looks for this module on the repository's path alone, where it is not
        environment = {**os.environ, "PYTHONPATH": ""}
        return [SimulatorProcess("sim", cls, sys.executable, [], environment, REPO, context.log_dir)], 1


class SlotCountedEchoEnv(EchoEnv):
    @classmethod
    def server_n(cls, config, cluster_resources):
        # a server per unit of sim the cluster offers, whatever the config says
        return int(cluster_resources["sim"])


class UncountableEchoEnv(EchoEnv):
    @classmethod
    def server_n(cls, config, cluster_resources):
        raise RuntimeError(f"no place among {sorted(cluster_resources)}")
"""
ENV += INSPECTED_POOL


class Servers(StartedPool):
    """EchoEnv's server pool in the test's Ray cluster as the driver starts and holds it (the processes' logs in
    tmp_path/logs) and an agent-loop process's client, made after the pool."""

    def __init__(self, env_class: type, tmp_path: Path) -> None:
        self.env_class = env_class
        self.log_dir = tmp_path / "logs"
        self.server_pool = None  # the pool the test started, held as the driver holds it
        self.client: StatelessClient | None = None

    def start(self, server_n: int = 1, rollout_timeout_s: float | None = None) -> list:
        """Starts the pool of ``server_n`` servers of EchoEnv as the driver does, with the servers' ROLLOUT_TIMEOUT_S
        ``rollout_timeout_s`` if given, makes a client after it and returns the servers."""
        pool_class = importlib.import_module(self.env_class.__module__).InspectedPool
        config = {"server_n": server_n, "log_dir": str(self.log_dir), "rollout_timeout_s": rollout_timeout_s}
        self.server_pool = pool_class.start(self.env_class, config)
        self.client = StatelessClient(self.env_class)

        return ray.get(self.server_pool.server_handles.remote())


@pytest.fixture
def servers(tmp_path, monkeypatch, local_ray_cluster):
    """The test's cluster for EchoEnv's servers, which offers SLOT_N units of sim and ends with the test."""
    (tmp_path / "echo_env.py").write_text(ENV)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "echo_env", raising=False)
    path = os.pathsep.join([str(REPO), str(tmp_path)])  # pear and EchoEnv's module, for the servers and processes

    with local_ray_cluster(num_cpus=1, resources={"sim": SLOT_N}, runtime_env={"env_vars": {"PYTHONPATH": path}}):
        yield Servers(importlib.import_module("echo_env").EchoEnv, tmp_path)


def test_every_call_of_an_episode_replays_it_on_a_server(servers):
    servers.start()
    client = servers.client

    async def play():
        opened = await client.open("e", {"task": 1})
        first = await client.step("e", "a")
        second = await client.step("e", "b")
        await client.close("e")
        return opened, first, second

    opened, first, second = asyncio.run(play())

    assert [opened["history"], first["history"], second["history"]] == [[], [], ["a"]]
    assert client.replay_by_uid == {}  # the closed episode forgotten
    assert opened["task"] == 1 and len({opened["pid"], first["pid"], second["pid"]}) == 1
    assert opened["context"] == []  # the process serves inside its env class's process context, given no arguments


def test_the_episodes_share_the_servers_each_call_its_own(servers):
    servers.start(server_n=SLOT_N)
    client = servers.client

    async def play(task):
        await client.open(f"e{task}", {"task": task})
        return [await client.step(f"e{task}", f"{task}-{step}") for step in range(3)]

    async def play_all():
        return await asyncio.gather(*(play(task) for task in range(4)))

    trajectories = asyncio.run(play_all())

    for task, results in enumerate(trajectories):  # every result is its own call's
        assert [result["history"] for result in results] == [[], [f"{task}-0"], [f"{task}-0", f"{task}-1"]]
    assert len({result["pid"] for results in trajectories for result in results}) == SLOT_N


@pytest.mark.parametrize("what", ["exit", "hang"])
def test_a_call_whose_process_dies_or_hangs_is_made_again_in_a_fresh_one(servers, tmp_path, what):
    # a hung call ends at the servers' call timeout, which the case that hangs one makes short
    servers.start(rollout_timeout_s=HUNG_ROLLOUT_TIMEOUT_S if what == "hang" else None)
    client = servers.client

    async def play():
        opened = await client.open("e", {"task": 1})
        retried = await client.step("e", {"do": what, "once": str(tmp_path / "met")})
        return opened, retried, await client.step("e", "next")

    opened, retried, following = asyncio.run(play())

    assert retried["pid"] != opened["pid"] and following["pid"] == retried["pid"]
    assert following["history"] == [{"do": what, "once": str(tmp_path / "met")}]
    with pytest.raises(ProcessLookupError):  # the old process was stopped (a hung one killed)
        os.kill(opened["pid"], 0)


def test_an_exception_of_the_env_fails_its_call_and_keeps_the_process(servers):
    servers.start()
    client = servers.client

    async def play():
        opened = await client.open("e", {"task": 1})
        with pytest.raises(RuntimeError) as caught:
            await client.step("e", {"do": "raise"})
        return opened, caught.value, await client.step("e", "next")

    opened, error, following = asyncio.run(play())

    assert "ValueError: the env raised" in str(error)
    assert following["pid"] == opened["pid"] and following["history"] == []


def test_a_process_that_fails_every_time_fails_the_call_and_the_next_call_gets_a_fresh_one(servers):
    servers.start()
    client = servers.client

    async def play():
        opened = await client.open("e", {"task": 1})
        with pytest.raises(RuntimeError, match=f"failed {Server.ROLLOUT_ATTEMPT_N} times in a row"):
            await client.step("e", {"do": "exit"})
        return opened, await client.step("e", "next")

    opened, following = asyncio.run(play())

    assert following["pid"] != opened["pid"] and following["history"] == []


def test_a_server_gone_for_good_leaves_the_calls_to_the_others(servers):
    handles = servers.start(server_n=SLOT_N)
    ray.kill(handles[0])  # gone for good: met once, it stays out of the pool

    async def play():
        return [await servers.client.open(f"e{task}", {"task": task}) for task in range(3)]

    results = asyncio.run(play())

    assert len({result["pid"] for result in results}) == 1 and [result["task"] for result in results] == [0, 1, 2]
    assert servers.free_n() == 1  # the other one, released after each call


def test_a_server_that_ray_starts_again_takes_calls_again(servers):
    [handle] = servers.start()
    first = asyncio.run(servers.client.open("e1", {"task": 1}))

    ray.kill(handle, no_restart=False)  # until its new start is up, a call on it meets ActorUnavailableError
    # ray.kill returns before the server ends, and the server's process ends with it
    servers.until(lambda: not servers.is_alive(first["pid"]))

    assert asyncio.run(servers.client.open("e2", {"task": 2}))["pid"] != first["pid"]  # its new start's process
    assert servers.free_n() == 1  # back in the pool


def test_a_client_finds_the_pool_the_driver_started_and_never_makes_one(servers):
    with pytest.raises(ValueError):
        StatelessClient(servers.env_class)

    servers.start()

    assert asyncio.run(servers.client.open("e", {"task": 1}))["task"] == 1


def test_a_pool_starts_as_many_servers_as_its_env_class_counts_on_the_cluster(servers):
    echo_env = importlib.import_module("echo_env")

    config = {"server_n": 1, "log_dir": str(servers.log_dir), "rollout_timeout_s": None}
    pool = echo_env.InspectedPool.start(echo_env.SlotCountedEchoEnv, config)

    assert len(ray.get(pool.server_handles.remote())) == SLOT_N


def test_a_cluster_its_env_class_counts_no_servers_on_ends_the_start_before_any_actor(servers):
    uncountable = importlib.import_module("echo_env").UncountableEchoEnv

    # the env class's own error where the driver starts the pool, not a wait for servers Ray cannot place
    with pytest.raises(RuntimeError, match="no place among .*'sim'"):
        ServerPool.start(uncountable, {"log_dir": str(servers.log_dir)})

    with pytest.raises(ValueError):  # no pool of its name
        ray.get_actor(ServerPool.name_of(uncountable), namespace=ServerPool.NAMESPACE)


def test_a_server_whose_process_cannot_start_fails_the_run_s_start(servers):
    unimportable = importlib.import_module("echo_env").UnimportableEchoEnv

    error = f"ended in its start {Server.FIRST_START_ATTEMPT_N} times in a row"  # its creation's
    with pytest.raises(ray.exceptions.RayError, match=error):
        ServerPool.start(unimportable, {"server_n": 1, "log_dir": str(servers.log_dir)})

    [log] = servers.log_dir.glob("*/sim-1.log")  # the first start's
    assert "ModuleNotFoundError: No module named 'echo_env'" in log.read_text()
