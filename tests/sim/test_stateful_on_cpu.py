from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from pear.sim.client import StatefulClient
from pear.sim.env import EpisodeLost
from pear.sim.server import Server
from repository import REPO
from sim_servers import INSPECTED_POOL, WAIT_S, StartedPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "linux", reason="a simulator process dies with its server through prctl (Linux)"
    ),
]
GPU_N = 2  # the GPUs the tests' Ray clusters offer, as many units of sim
HUNG_START_TIMEOUT_S = 10.0  # the servers' own start timeout where a test hangs a start
HUNG_STEP_TIMEOUT_S = 1.0  # the servers' step timeout where a test hangs a step: the test waits it out once
# The env class of these tests' servers, from a module every process of the Ray cluster imports: its process context
# and episodes marked in the file $MARKS, a line each ending with the process's pid and the time; a server serves the
# processes the settings file next to the marks names (how many, how many start at once, each one's start in seconds,
# the env args of a warm-up, if any, the server's own start timeout and its steps' timeout, if any) and writes its GPUs
# next to them, under its pid; a start that finds the file "hang" next to the marks hangs; env args or an action can
# crash its process, hang it, lose its scene or fail the env's code
ENV = r"""
import json, os, sys, time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pear.sim.env import EpisodeLost, StatefulEnv
from pear.sim.process import SimulatorProcess
from pear.sim.server import ServerPool


class MarkedEnv(StatefulEnv):
    SERVER_RESOURCES = {"num_cpus": 0, "num_gpus": 1, "resources": {"sim": 1}}
    START_TIMEOUT_S = 60.0  # a server's setup takes seconds
    OPEN_TIMEOUT_S = STEP_TIMEOUT_S = 60.0  # a reply takes milliseconds, and seconds on a machine that is busy

    @classmethod
    def task_name(cls, env_args):
        return cls.__name__

    @classmethod
    @contextmanager
    def process_context(cls, context_args):
        name, start_s = context_args
        cls.mark("start", name)
        time.sleep(3600.0 if Path(os.environ["MARKS"]).with_name("hang").exists() else float(start_s))
        cls.mark("ready", name)
        yield
        cls.mark("end", name)

    @classmethod
    def setup_env(cls, context):
        marks = Path(os.environ["MARKS"])
        settings = json.loads(marks.with_name("settings.json").read_text())
        marks.with_name(f"{os.getpid()}.gpu_ids").write_text(json.dumps(context.gpu_ids))
        if settings["start_timeout_s"]:  # the server's own (its daemons'), not the driver's start of the pool
            cls.START_TIMEOUT_S = settings["start_timeout_s"]
        if settings["step_timeout_s"]:  # of the test that hangs a step
            cls.STEP_TIMEOUT_S = settings["step_timeout_s"]
        if settings["warm"]:  # an episode built in a process of its own first, as a family warms its caches up
            process = SimulatorProcess("w", cls, sys.executable, ["w", "0"], dict(os.environ), Path.cwd(),
                                       context.log_dir)
            if process.warm_up(settings["warm"]) is not None:
                raise RuntimeError("the warm-up w failed")
        processes = [SimulatorProcess(f"a{i}", cls, sys.executable, [f"a{i}", str(settings["start_s"])],
                                      dict(os.environ), Path.cwd(), context.log_dir)
                     for i in range(settings["process_n"])]
        return processes, settings["max_concurrent_start_n"]

    def __init__(self, **env_args):
        self.env_args = env_args

    def open(self):
        self.mark("open", json.dumps(self.env_args, separators=(",", ":")))
        self.act(self.env_args)
        return {"pid": os.getpid(), "env_args": self.env_args, "frame": np.full((4, 6, 3), 7, np.uint8)}

    def step(self, action):
        self.act(action)
        return {"pid": os.getpid(), "action": action}

    def close(self):
        self.mark("close")

    @staticmethod
    def act(request):
        if request.get("crash"):
            os._exit(3)
        if request.get("hang"):
            time.sleep(3600)
        if request.get("lost"):
            raise EpisodeLost("the scene never settled")
        if request.get("fail"):
            raise ValueError("the env raised")

    @staticmethod
    def mark(*words):
        with open(os.environ["MARKS"], "a") as f:
            f.write(" ".join([*words, str(os.getpid()), str(time.time())]) + "\n")
"""
ENV += INSPECTED_POOL


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    """The module of MarkedEnv, importable here and in the processes of the tests' Ray clusters."""
    directory = tmp_path_factory.mktemp("env")
    (directory / "marked_env.py").write_text(ENV)
    sys.path.insert(0, str(directory))
    yield importlib.import_module("marked_env")
    sys.path.remove(str(directory))
    del sys.modules["marked_env"]


class Servers(StartedPool):
    """MarkedEnv's server pool in the test's Ray cluster as the driver starts and holds it (the marks in
    tmp_path/marks, the logs in tmp_path/logs) and an agent-loop process's client, made after the pool."""

    def __init__(self, marked, tmp_path: Path) -> None:
        self.marks_file = tmp_path / "marks"
        self.marked = marked
        self.log_dir = tmp_path / "logs"
        self.server_pool = None  # the pool the test started, held as the driver holds it
        self.client: StatefulClient | None = None

    def start(
        self,
        process_n: int = 1,
        max_concurrent_start_n: int | None = None,
        start_s: float = 0.0,
        warm: dict | None = None,
        server_n: int = 1,
        start_timeout_s: float | None = None,
        step_timeout_s: float | None = None,
    ) -> list:
        """Starts the pool of ``server_n`` servers of MarkedEnv as the driver does, each with ``process_n`` processes,
        at most ``max_concurrent_start_n`` (default all) of them starting at once, each taking ``start_s``, after a
        warm-up of ``warm`` if given, with the servers' own START_TIMEOUT_S ``start_timeout_s`` and their STEP_TIMEOUT_S
        ``step_timeout_s`` if given; makes a client after it and returns the servers."""
        settings = {
            "process_n": process_n,
            "max_concurrent_start_n": max_concurrent_start_n or process_n,
            "start_s": start_s,
            "warm": warm,
            "start_timeout_s": start_timeout_s,
            "step_timeout_s": step_timeout_s,
        }
        self.marks_file.with_name("settings.json").write_text(json.dumps(settings))
        config = {"server_n": server_n, "log_dir": str(self.log_dir)}
        self.server_pool = self.marked.InspectedPool.start(self.marked.MarkedEnv, config)
        self.client = StatefulClient(self.marked.MarkedEnv)

        return ray.get(self.server_pool.server_handles.remote())

    def gpu_ids_by_pid(self) -> dict[int, list[int]]:
        """Returns the GPUs of each server that started, by its pid."""
        return {int(path.stem): json.loads(path.read_text()) for path in self.marks_file.parent.glob("*.gpu_ids")}

    def ready_n(self) -> int:
        """Returns how many starts of the servers' processes got past their start so far."""
        return [mark[0] for mark in self.marks()].count("ready")

    def marks(self) -> list[list[str]]:
        """Returns the marks so far, each as the words of its line."""
        return [line.split() for line in self.marks_file.read_text().splitlines()] if self.marks_file.exists() else []

    @staticmethod
    def answers(server) -> bool:
        """Whether ``server`` takes a call, which it does not while this process still sees it restarting."""
        try:
            ray.get(server.process_n.remote())
        except ActorUnavailableError:
            return False
        return True

    @staticmethod
    def parent_pid(pid: int) -> int:
        """Returns the pid of process ``pid``'s parent."""
        # /proc/<pid>/stat: the pid, the command in parentheses, the state, the parent's pid, ...
        return int(Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[1])


@pytest.fixture
def servers(marked, tmp_path, local_ray_cluster):
    # a local Ray cluster of the test's own that offers GPU_N GPUs as sim: its pool and servers end with it
    path = os.pathsep.join([str(REPO), str(Path(marked.__file__).parent)])
    environment = {"PYTHONPATH": path, "MARKS": str(tmp_path / "marks")}
    with local_ray_cluster(num_cpus=1, num_gpus=GPU_N, resources={"sim": GPU_N}, runtime_env={"env_vars": environment}):
        yield Servers(marked, tmp_path)


def test_an_episode_keeps_its_process_from_open_to_close_and_a_fresh_process_follows(servers):
    servers.start()

    opened = asyncio.run(servers.client.open("e1", {"task": 1}))
    steps = [asyncio.run(servers.client.step("e1", {"x": x})) for x in range(2)]
    asyncio.run(servers.client.close("e1"))
    following = asyncio.run(servers.client.open("e2", {"task": 2}))

    assert opened["frame"].tolist() == np.full((4, 6, 3), 7, np.uint8).tolist() and opened["env_args"] == {"task": 1}
    assert {step["pid"] for step in steps} == {opened["pid"]} != {following["pid"]}
    assert [step["action"] for step in steps] == [{"x": 0}, {"x": 1}]
    assert [mark[:-2] for mark in servers.marks()] == [
        ["start", "a0"],
        ["ready", "a0"],
        ["open", '{"task":1}'],
        ["close"],
        ["end", "a0"],
        ["start", "a0"],
        ["ready", "a0"],
        ["open", '{"task":2}'],
    ]


def test_an_open_waits_for_a_free_server(servers):
    servers.start()
    held = asyncio.run(servers.client.open("e1", {"task": 1}))
    opened = []

    # a daemon: an open that never returns fails the test and does not keep pytest from ending
    thread = threading.Thread(
        target=lambda: opened.append(asyncio.run(servers.client.open("e2", {"task": 2}))), daemon=True
    )
    thread.start()
    # the server's one process hosts the first episode: the second open's acquire waits in the pool until its close
    servers.until(lambda: servers.waiting_n() == 1)
    assert not opened
    asyncio.run(servers.client.close("e1"))
    thread.join(WAIT_S)

    assert opened[0]["env_args"] == {"task": 2} and opened[0]["pid"] != held["pid"]


@pytest.mark.parametrize("env_args, action", [({"task": 1}, {"crash": True}), ({"crash": True}, None)])
def test_a_process_that_crashes_loses_its_episode_and_starts_again(servers, env_args, action):
    servers.start()

    # the request that crashes the process: the open itself, or the step of an episode whose open went through
    if action is None:
        crashing_request = servers.client.open("e1", env_args)
    else:
        asyncio.run(servers.client.open("e1", env_args))
        crashing_request = servers.client.step("e1", action)

    with pytest.raises(EpisodeLost, match="EOFError"):
        asyncio.run(crashing_request)
    asyncio.run(servers.client.close("e1"))  # nothing to end: it releases the server

    assert asyncio.run(servers.client.open("e2", {"task": 2}))["env_args"] == {"task": 2}
    [log_dir] = servers.log_dir.iterdir()  # the server's
    assert (log_dir / "a0-2.log").exists()
    assert "a0 exited with 3 after start 1" in (log_dir / Server.LOG_FILE_NAME).read_text()


def test_an_env_that_cannot_go_on_ends_the_episode_as_a_lost_process(servers):
    servers.start()

    with pytest.raises(EpisodeLost):  # the env's EpisodeLost is the one the process catches
        asyncio.run(servers.client.open("e1", {"lost": True}))
    asyncio.run(servers.client.close("e1"))

    assert asyncio.run(servers.client.open("e2", {"task": 2}))["env_args"] == {"task": 2}
    # the episode closed, the process ended
    assert [mark[0] for mark in servers.marks()][2:5] == ["open", "close", "end"]


def test_the_close_of_an_episode_whose_open_failed_ends_its_process(servers):
    servers.start()

    with pytest.raises(RuntimeError, match="ValueError: the env raised"):
        asyncio.run(servers.client.open("e1", {"fail": True}))
    asyncio.run(servers.client.close("e1"))

    servers.until(lambda: servers.ready_n() == 2)  # the process's next start
    assert [mark[0] for mark in servers.marks()][2:7] == ["open", "close", "end", "start", "ready"]


def test_a_failure_of_the_env_s_code_fails_its_request_and_leaves_the_episode_as_it_was(servers):
    servers.start()
    opened = asyncio.run(servers.client.open("e1", {"task": 1}))

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(servers.client.step("e1", {"fail": True}))

    assert type(caught.value) is RuntimeError and "ValueError: the env raised" in str(caught.value)
    stepped = asyncio.run(servers.client.step("e1", {"x": 1}))
    assert stepped == {"pid": opened["pid"], "action": {"x": 1}}


def test_a_request_past_its_timeout_kills_its_process_and_loses_the_episode(servers):
    servers.start(step_timeout_s=HUNG_STEP_TIMEOUT_S)
    opened = asyncio.run(servers.client.open("e1", {"task": 1}))

    start_time = time.monotonic()
    with pytest.raises(EpisodeLost, match="TimeoutError"):
        asyncio.run(servers.client.step("e1", {"hang": True}))
    asyncio.run(servers.client.close("e1"))

    assert HUNG_STEP_TIMEOUT_S <= time.monotonic() - start_time < WAIT_S  # lost at the step's timeout, not before
    assert not servers.is_alive(opened["pid"])
    assert asyncio.run(servers.client.open("e2", {"task": 2}))["pid"] != opened["pid"]


def test_an_open_skips_the_processes_that_ended_while_free_and_the_server_starts_them_again(servers):
    servers.start(process_n=3)
    pids = {mark[1]: int(mark[2]) for mark in servers.marks() if mark[0] == "ready"}

    for name in ("a0", "a1"):
        os.kill(pids[name], signal.SIGKILL)
    [log_dir] = servers.log_dir.iterdir()  # the server's
    servers.until(lambda: len(list(log_dir.glob("a[01]-2.log"))) == 2)

    opened = asyncio.run(servers.client.open("e1", {"task": 1}))
    assert servers.is_alive(opened["pid"]) and opened["pid"] not in (pids["a0"], pids["a1"])


def test_a_process_that_hangs_in_its_start_is_killed_and_started_again(servers):
    servers.start(start_timeout_s=HUNG_START_TIMEOUT_S)
    asyncio.run(servers.client.open("e1", {"task": 1}))
    hang_file = servers.marks_file.with_name("hang")
    hang_file.touch()  # the process's next start hangs

    asyncio.run(servers.client.close("e1"))  # its episode ends: its daemon starts it again
    [log_dir] = servers.log_dir.iterdir()  # the server's
    servers.until(lambda: "a0 hung in start 2, killed" in (log_dir / Server.LOG_FILE_NAME).read_text())
    hang_file.unlink()

    assert asyncio.run(servers.client.open("e2", {"task": 2}))["env_args"] == {"task": 2}  # a later start got through


def test_a_server_starts_no_more_processes_at_once_than_its_env_allows(servers):
    servers.start(process_n=3, max_concurrent_start_n=1, start_s=0.3)

    events = [mark[:2] for mark in sorted(servers.marks(), key=lambda mark: float(mark[-1]))]
    assert events == [
        ["start", "a0"],
        ["ready", "a0"],
        ["start", "a1"],
        ["ready", "a1"],
        ["start", "a2"],
        ["ready", "a2"],
    ]


def test_the_servers_go_where_ray_finds_their_gpus(servers):
    servers.start(server_n=GPU_N)

    gpu_ids = sorted(gpu for server_gpu_ids in servers.gpu_ids_by_pid().values() for gpu in server_gpu_ids)
    assert gpu_ids == list(range(GPU_N))  # a GPU each
    assert servers.free_n() == GPU_N  # each in the pool once: one process each


def test_the_first_opens_spread_over_the_servers(servers):
    servers.start(process_n=2, server_n=GPU_N)

    async def play():
        return [await servers.client.open(f"e{task}", {"task": task}) for task in range(GPU_N)]

    opened = asyncio.run(play())

    # a process of each server (whose worker starts it), not the first server's processes first
    assert len({servers.parent_pid(result["pid"]) for result in opened}) == GPU_N


def test_servers_that_ray_cannot_place_fail_the_run_s_start(servers, monkeypatch):
    monkeypatch.setattr(servers.marked.MarkedEnv, "START_TIMEOUT_S", 5.0)  # what the test waits for the last one

    with pytest.raises(RuntimeError, match="did not start within 5s"):
        servers.start(server_n=GPU_N + 1)  # one more than the GPUs


def test_a_server_s_processes_die_with_it(servers):
    [server] = servers.start(process_n=2)
    pids = [int(mark[2]) for mark in servers.marks() if mark[0] == "ready"]

    ray.kill(server)

    servers.until(lambda: not any(servers.is_alive(pid) for pid in pids))


def test_an_episode_whose_server_left_is_lost(servers):
    [server] = servers.start()
    asyncio.run(servers.client.open("e1", {"task": 1}))

    ray.kill(server)

    with pytest.raises(EpisodeLost, match="Actor"):
        asyncio.run(servers.client.step("e1", {"x": 1}))
    asyncio.run(servers.client.close("e1"))  # its server left for good: nothing to end, nothing to release
    assert servers.free_n() == 0


def test_an_open_skips_a_server_gone_for_good(servers):
    [gone_server, _] = servers.start(server_n=GPU_N)
    ray.kill(gone_server)  # met once, it stays out of the pool

    assert asyncio.run(servers.client.open("e1", {"task": 1}))["env_args"] == {"task": 1}
    assert servers.free_n() == 0  # the other one is held by the episode


def test_an_open_on_a_server_that_ray_starts_again_waits_for_its_new_start(servers):
    [server] = servers.start()
    first = asyncio.run(servers.client.open("e1", {"task": 1}))
    asyncio.run(servers.client.close("e1"))

    ray.kill(server, no_restart=False)  # until its new start is up, a call on it meets ActorUnavailableError

    assert asyncio.run(servers.client.open("e2", {"task": 2}))["pid"] != first["pid"]


def test_a_server_that_went_away_starts_again_and_the_episodes_of_its_old_processes_are_lost(servers):
    [server] = servers.start()
    opened = asyncio.run(servers.client.open("e1", {"task": 1}))

    ray.kill(server, no_restart=False)
    # the process of the server's new start, and the new start taking calls: before Ray's word that it runs again
    # reaches this process, a call finds the server restarting (ActorUnavailableError), not the episode gone
    servers.until(lambda: servers.ready_n() == 2 and servers.answers(server))

    with pytest.raises(EpisodeLost, match="is gone"):  # it reaches the new start, which never heard of the episode
        asyncio.run(servers.client.step("e1", {"x": 1}))
    asyncio.run(servers.client.close("e1"))
    assert servers.free_n() == 1 and not servers.is_alive(opened["pid"])
    assert asyncio.run(servers.client.open("e2", {"task": 2}))["pid"] != opened["pid"]


def test_a_warm_up_builds_one_episode_inside_its_process_context_before_the_server_s_processes_start(servers):
    servers.start(warm={"task": 0})

    assert [mark[:-2] for mark in servers.marks()] == [
        ["start", "w"],
        ["ready", "w"],
        ["open", '{"task":0}'],
        ["close"],
        ["end", "w"],
        ["start", "a0"],
        ["ready", "a0"],
    ]


def test_a_failed_warm_up_fails_the_run_s_start_and_the_server_for_good(servers):
    # ActorDiedError is Ray's word that it does not start the server again: one it starts again is unavailable
    # (ActorUnavailableError) until it is up
    with pytest.raises(ActorDiedError, match="the warm-up w failed"):
        servers.start(warm={"fail": True})

    assert [mark[:2] for mark in servers.marks()].count(["start", "w"]) == 1
