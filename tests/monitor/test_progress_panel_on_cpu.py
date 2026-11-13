import collections
import io
import itertools
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from first_line import FirstLine
from pear.monitor import metrics as metrics_module
from pear.monitor.driver_log import DriverLog
from pear.monitor.metrics import ReplicaMetrics, ReplicaSnapshot
from pear.monitor.progress_panel import PanelView, ProgressPanel
from pear.monitor.progress_tracker import ProgressTracker
from pear.rollout.server import PearVLLMServer
from repository import REPO

COMM_CHARS = 15  # the kernel's comm field of a process, without its terminator
PANEL_EXIT_S = 60  # for the panel to leave after its signal
FIRST_LINE_S = 180  # for a cluster to start, and for a panel to reach its first refresh
REFRESH_TEXT = " | engines "  # in every refresh line of a panel whose output is not a terminal
# the sitecustomize of a cluster whose machines do not have the repository: its processes refuse every import of pear,
# whatever the environment has installed
NO_PEAR_SITECUSTOMIZE = """
import sys


class NoPear:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name.split(".")[0] == "pear":
            raise ModuleNotFoundError(f"No module named {name!r}")


sys.meta_path.insert(0, NoPear)
"""
# a local Ray cluster in a process of its own: it says its address and whether a worker of it can import pear, and ends
# when its input closes
CLUSTER_PROGRAM = """
import sys

import ray

context = ray.init(num_cpus=1, include_dashboard=False)  # with LocalRayCluster.ENVIRONMENT, from its fixture


@ray.remote
def worker_imports_pear():
    try:
        import pear
    except ModuleNotFoundError:
        return False
    return True


print(context.address_info["address"], ray.get(worker_imports_pear.remote()), flush=True)
sys.stdin.read()
"""
REPLICA = "192.0.2.1:8000"  # a replica's address, of the range reserved for documentation
SAMPLE_PERIOD_S = 10  # between two samples of the replica
# a replica's metrics at its first sample and what a sample period adds to its counters
FIRST_SAMPLE = {
    "num_requests_running": 3,
    "num_requests_waiting": 1,
    "prompt_tokens_total": 1000,
    "generation_tokens_total": 400,
    "num_preemptions_total": 0,
    "prefix_cache_queries_total": 1000,
    "prefix_cache_hits_total": 500,
    "external_prefix_cache_hits_total": 100,
}
SAMPLE_INCREASES = {
    "prompt_tokens_total": 20000,
    "generation_tokens_total": 5000,
    "num_preemptions_total": 1,
    "prefix_cache_queries_total": 20000,
    "prefix_cache_hits_total": 12000,
    "external_prefix_cache_hits_total": 4000,
}


@pytest.fixture(scope="module")
def cluster_address(local_ray_cluster):
    # a local Ray cluster, which the panel joins as a recipe's panel joins the run's
    with local_ray_cluster(num_cpus=1) as context:
        yield context.address_info["address"]


@pytest.fixture
def cluster_without_the_repository(local_ray_cluster, tmp_path):
    # the address of a cluster whose processes cannot import pear: the raylet and its workers load the sitecustomize
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(NO_PEAR_SITECUSTOMIZE)
    cluster = subprocess.Popen(
        [sys.executable, "-c", CLUSTER_PROGRAM],
        env={**os.environ, **local_ray_cluster.ENVIRONMENT, "PYTHONPATH": str(site_dir)},
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    try:
        address, worker_imports_pear = FirstLine.of(cluster, FIRST_LINE_S).split()
        assert worker_imports_pear == "False"
        yield address
    finally:
        cluster.stdin.close()
        cluster.wait(PANEL_EXIT_S)


@pytest.fixture
def sampled_replica(monkeypatch):
    # one replica sampled twice, a sample period apart, whose /metrics answers leave out the metrics named
    def sample(*absent_names: str) -> ReplicaSnapshot:
        second_sample = {name: value + SAMPLE_INCREASES.get(name, 0) for name, value in FIRST_SAMPLE.items()}
        answers = iter(
            "".join(
                f'vllm:{name}{{engine="0",model_name="m"}} {float(value)}\n'
                for name, value in metrics.items()
                if name not in absent_names
            ).encode()
            for metrics in (FIRST_SAMPLE, second_sample)
        )
        seconds = itertools.count(step=SAMPLE_PERIOD_S)
        sampled_metrics = ReplicaMetrics()
        sampled_metrics.history[REPLICA] = collections.deque(maxlen=ReplicaMetrics.HISTORY)

        # the module's clock alone: a sample's time, not a log record's
        with monkeypatch.context() as patch:
            patch.setattr(ReplicaMetrics, "_get", staticmethod(lambda address, limit=None: next(answers)))
            patch.setattr(metrics_module, "time", SimpleNamespace(time=lambda: float(next(seconds))))
            sampled_metrics.snapshot()

            return sampled_metrics.snapshot()

    return sample


def test_the_snapshot_holds_the_replicas_rates_between_two_samples(sampled_replica):
    snapshot = sampled_replica()

    assert (snapshot.live, snapshot.known, snapshot.running, snapshot.waiting) == (1, 1, 3, 1)
    assert (snapshot.gen_tps, snapshot.prompt_tps, snapshot.preempt_per_min) == (500.0, 2000.0, 6.0)
    # the GPU's and LMCache's hits among the queried tokens
    assert snapshot.cache_hit == 0.8
    assert snapshot.line() == "engines 1/1 gen 0.5k tok/s prompt 2k tok/s run 3 wait 1 cachehit 80.0% preempt 6/min"


def test_a_metric_a_replica_does_not_report_shows_as_missing_and_is_said_once(sampled_replica, monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(metrics_module.logger.handlers[0], "stream", stream)

    # as after a vLLM that renamed the metric
    snapshot = sampled_replica("generation_tokens_total")

    # not a rate of 0 tokens a second, and the numbers of the other metrics as they are
    assert snapshot.gen_tps is None
    assert (snapshot.prompt_tps, snapshot.running, snapshot.cache_hit) == (2000.0, 3, 0.8)
    assert snapshot.line() == "engines 1/1 gen n/a tok/s prompt 2k tok/s run 3 wait 1 cachehit 80.0% preempt 6/min"
    stats = {"step": None, "finished": 0, "accept_groups": 0}  # of a run without a finished episode
    rows = PanelView(target_trajectory_n=2, rollout_n=1, run_name="run", is_tty=False)._rows(
        stats, DriverLog(), snapshot, now=0.0, width=120
    )
    assert "gen n/a tok/s" in "".join(text for row in rows for text, _ in row)
    # over the two samples
    assert stream.getvalue().count("vllm:generation_tokens_total") == 1
    assert REPLICA in stream.getvalue()


def test_the_rows_of_a_run_with_finished_episodes(sampled_replica):
    stats = {
        "step": 3,
        "started": 0.0,
        "finished": 4,
        "success": 1,
        "avg_at_n": 0.25,
        "turns": 40,
        "judged": 2,
        "accept_groups": 1,
        "zero": 1,
        "one": 0,
        "end_reason": {"env": 3, "budget": 1},
        "tasks": {"task_10": [1, 0], "task_2": [3, 1]},
    }
    view = PanelView(target_trajectory_n=4, rollout_n=2, run_name="run", is_tty=False)

    rows = view._rows(stats, DriverLog(), sampled_replica(), now=10.0, width=120)

    assert ["".join(text for text, _ in row) for row in rows][2:] == [
        "episodes  4 done · 1 success (25.0%) · avg@n 0.250 · 4.0 turns/s",
        "groups    2 judged · 1 kept · 1 all-fail · 0 all-pass",
        "ended by  env 3 · budget 1",
        # in natural order: task_2 before task_10
        "tasks     task_2 1/3  task_10 0/1",
        "engines   1/1 up · gen 0.5k tok/s · prompt 2k tok/s · 3 running · 1 waiting · 6 preempt/min",
        "cache hit 80.0%",
    ]


def test_the_batch_row_counts_the_groups_ready_for_the_trainer_and_those_in_flight():
    view = PanelView(target_trajectory_n=8, rollout_n=2, run_name="run", is_tty=False)
    stats = {"accept_groups": 1}

    # before the sampler's first poll in the driver log: the groups the filter kept this step
    assert view._batch(stats, DriverLog()) == (1, "1/4 groups kept this step")
    assert view._batch(stats, DriverLog(available_group_n=3, inflight_group_n=5)) == (
        3,
        "3/4 groups ready · 5 in flight",
    )
    assert view._batch(stats, DriverLog(available_group_n=3, inflight_group_n=0)) == (3, "3/4 groups ready")


def test_the_tracker_counts_the_groups_of_the_latest_step():
    tracker = ProgressTracker()
    # an earlier step's trajectory, then two groups of two of the latest: one of mixed outcomes, one all won
    trajectories = [
        (2, "earlier", True, "task_1"),
        (3, "mixed", True, "task_1"),
        (3, "mixed", False, "task_1"),
        (3, "won", True, "task_2"),
        (3, "won", True, "task_2"),
    ]
    for step, uid, success, task in trajectories:
        record = {"step": step, "uid": uid, "success": success, "task": task, "num_chunks": 5, "end_reason": "env"}
        tracker.report(record)

    stats = tracker.stats(rollout_n=2)

    assert (stats["step"], stats["finished"], stats["success"]) == (3, 4, 3)
    # the group of mixed outcomes is the one a filter keeps
    assert (stats["judged"], stats["accept_groups"], stats["zero"], stats["one"]) == (2, 1, 0, 1)
    assert stats["avg_at_n"] == 0.75
    assert stats["tasks"] == {"task_1": [2, 1], "task_2": [2, 2]}
    # the turns of every step's trajectories
    assert stats["turns"] == 25
    # no step before a trajectory has ended
    assert ProgressTracker().stats(rollout_n=2)["step"] is None


def test_the_panel_looks_for_the_process_name_of_pears_server_actor():
    # Ray titles an actor's process ray::<class name>, and `ss -p` prints the kernel's comm field of it
    assert ReplicaMetrics.SERVER_ACTOR == f"ray::{PearVLLMServer.__name__}"[:COMM_CHARS]


def test_a_panel_reads_its_run_s_driver_log_and_writes_its_numbers_there_only_when_asked(monkeypatch, tmp_path):
    arguments = ["progress", "--target", "16", "--group", "8", "--run", str(tmp_path / "a_run")]

    monkeypatch.setattr(sys, "argv", arguments)
    live_panel = ProgressPanel.from_command_line()
    monkeypatch.setattr(sys, "argv", [*arguments, "--json"])
    recipe_panel = ProgressPanel.from_command_line()

    assert (live_panel.driver_log_path, live_panel.numbers_path) == (str(tmp_path / "a_run" / "train.log"), None)
    assert recipe_panel.numbers_path == str(tmp_path / "a_run" / "progress.json")


def test_sigterm_ends_the_panel_as_ctrl_c_does(cluster_address, tmp_path):
    log = tmp_path / "train.log"
    log.write_text("")
    command = [
        sys.executable,
        "-m",
        "pear.monitor.progress_panel",
        "--target",
        "2",
        "--group",
        "1",
        "--run",
        str(tmp_path),
    ]
    panel = subprocess.Popen(
        command,
        env={**os.environ, "RAY_ADDRESS": cluster_address},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # its first refresh, not the first line a library writes: the panel is in its loop, where a recipe's exit finds it
    first_output = FirstLine.of(panel, FIRST_LINE_S, REFRESH_TEXT)
    panel.send_signal(signal.SIGTERM)
    output = first_output + panel.communicate(timeout=PANEL_EXIT_S)[0]

    assert panel.returncode == 0, output
    # the stack Ray's handler writes for the signal
    assert "SIGTERM received" not in output


def test_a_panel_started_in_any_directory_needs_no_repository_on_the_cluster_s_workers(
    cluster_without_the_repository, tmp_path
):
    # as a panel started by hand: the repository on its own path alone, in a directory that is not the repository's
    log = tmp_path / "train.log"
    log.write_text("")
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    command = [
        sys.executable,
        "-m",
        "pear.monitor.progress_panel",
        "--target",
        "2",
        "--group",
        "1",
        "--run",
        str(tmp_path),
    ]
    panel = subprocess.Popen(
        command,
        env={**os.environ, "PYTHONPATH": str(REPO), "RAY_ADDRESS": cluster_without_the_repository},
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # its first refresh: the replicas' discovery ran on the cluster's machine and the tracker's actor answered
    first_output = FirstLine.of(panel, FIRST_LINE_S, REFRESH_TEXT)
    panel.send_signal(signal.SIGTERM)
    output = first_output + panel.communicate(timeout=PANEL_EXIT_S)[0]

    assert panel.returncode == 0, output
    assert "Traceback" not in output, output
