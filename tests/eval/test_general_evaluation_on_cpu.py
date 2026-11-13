import asyncio
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import yaml
from aiohttp import ClientSession, web

from pear.eval.general import BenchmarkRun, DatasetDirectory, DatasetDownload, GeneralEvaluation, Router, RunComparison

SAMPLES = 40  # samples of the synthetic benchmark
REPEATS = 2  # repeats of each sample, which share its group id


class Reviews:
    """evalscope's reviews of one benchmark of a run: each sample's score, the same at every repeat."""

    @staticmethod
    def write(run_dir: Path, benchmark: str, score_of) -> None:
        path = run_dir / benchmark / "reviews" / "model" / f"{benchmark}.jsonl"
        path.parent.mkdir(parents=True)
        reviews = [
            {
                "sample_score": {
                    "group_id": sample,
                    "score": {"main_score_name": "acc", "value": {"acc": score_of(sample)}},
                }
            }
            for sample in range(SAMPLES)
            for _ in range(REPEATS)
        ]
        path.write_text("".join(json.dumps(review) + "\n" for review in reviews))


class Replica:
    """An OpenAI endpoint stand-in on a local port the system picks that answers with its name and the path it was
    asked."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.runner = None

    async def start(self) -> int:
        """Starts the endpoint and returns its port."""
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.answer)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()

        return self.runner.addresses[0][1]

    async def answer(self, request: web.Request) -> web.Response:
        return web.json_response({"replica": self.name, "path": request.path_qs})


def test_the_comparison_pairs_the_samples_two_runs_share(tmp_path, capsys):
    Reviews.write(tmp_path / "a", "ifbench", lambda sample: float(sample % 2))
    Reviews.write(tmp_path / "b", "ifbench", lambda sample: float(sample % 2) * 0.5 + 0.5)  # +0.5 where a scored 0

    config = yaml.safe_load(GeneralEvaluation.CONFIG_PATH.read_text())
    RunComparison(config, str(tmp_path / "a"), str(tmp_path / "b"), ("a", "b")).run()

    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("ifbench"))
    words = line.split()
    # a means 0.5 and b 0.75; half the paired differences are +0.5, so the bootstrap interval lies above 0
    assert words[1] == "50.0" and "75.0" in words and "+25.0" in words and line.endswith(f"* {SAMPLES}")
    assert RunComparison.sample_scores(str(tmp_path / "a"), "ifbench")["ifbench.jsonl:1"] == 1.0


def test_the_router_answers_from_a_live_replica_past_one_that_refuses_the_connection():

    async def scenario(dead_port: int):
        replica = Replica("live")
        live_port = await replica.start()
        # the test serves the router's app itself, on a port the system picks
        router = Router([f"127.0.0.1:{dead_port}", f"127.0.0.1:{live_port}"], port=0)
        router.inflight_request_n[
            f"http://127.0.0.1:{dead_port}"
        ] = -1  # the dead replica looks least loaded: it is tried first
        app = web.Application()
        app.on_startup.append(router.open)
        app.on_cleanup.append(router.close)
        app.router.add_route("*", "/{tail:.*}", router.proxy)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        router_port = runner.addresses[0][1]
        try:
            async with ClientSession() as client:
                async with client.post(f"http://127.0.0.1:{router_port}/v1/chat/completions?x=1", json={}) as response:
                    return response.status, await response.json(), dict(router.inflight_request_n)
        finally:
            await runner.cleanup()
            await replica.runner.cleanup()

    # the dead replica's port: bound and not listened on, Linux refuses a connection to it and no other process gets it
    with socket.socket() as dead_socket:
        dead_socket.bind(("127.0.0.1", 0))
        status, answer, inflight = asyncio.run(scenario(dead_socket.getsockname()[1]))

    assert status == 200 and answer == {"replica": "live", "path": "/v1/chat/completions?x=1"}
    assert sorted(inflight.values()) == [-1, 0]  # every attempt gave its slot back


class Evalscope:
    """A stand-in for the evalscope package on the path: a task records the process that ran it, a dataset its fetch,
    and an NLTK resource is fetched into the home directory."""

    PACKAGE = """
import os
from pathlib import Path


class TaskConfig:
    def __init__(self, **arguments):
        self.__dict__.update(arguments)


def run_task(config):
    if config.datasets == ["broken"]:
        raise FileNotFoundError("no such dataset")
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    Path(config.work_dir, "pid").write_text(str(os.getpid()))
    names = ("MODELSCOPE_CACHE", "MODELSCOPE_CREDENTIALS_PATH", "HF_HOME", "NLTK_DATA")
    Path(config.work_dir, "directories").write_text(chr(10).join(os.environ[name] for name in names))
"""
    REGISTRY = """
import os
from pathlib import Path


class Benchmark:
    def __init__(self, name, config):
        self.name, self.config = name, config

    def load_dataset(self):
        Path(self.config.dataset_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.config.dataset_dir, "fetched"), "a") as fetched:
            fetched.write(self.name + chr(10))
        names = ("MODELSCOPE_CACHE", "MODELSCOPE_CREDENTIALS_PATH", "HF_HOME", "NLTK_DATA")
        Path(self.config.dataset_dir, "directories").write_text(chr(10).join(os.environ[name] for name in names))
        budget = os.environ.get("MODELSCOPE_REPO_TREE_PAGE_TIMEOUT_SECONDS", "")
        Path(self.config.dataset_dir, "hub_probe_budget").write_text(budget)


def get_benchmark(name, config):
    return Benchmark(name, config)
"""
    RESOURCES = """
import os
from pathlib import Path

MIRROR_MAP = {"punkt_tab": {}, "stopwords": {}}


def check_nltk_data(name):
    Path(os.path.expanduser("~/nltk_data"), name).mkdir(parents=True, exist_ok=True)
"""

    @classmethod
    def install(cls, directory: Path, monkeypatch) -> None:
        for package in ("api", "utils"):
            (directory / "evalscope" / package).mkdir(parents=True)
            (directory / "evalscope" / package / "__init__.py").write_text("")
        (directory / "evalscope" / "__init__.py").write_text(cls.PACKAGE)
        (directory / "evalscope" / "api" / "registry.py").write_text(cls.REGISTRY)
        (directory / "evalscope" / "utils" / "resource_utils.py").write_text(cls.RESOURCES)
        # a benchmark's process is started anew with this process's path
        monkeypatch.syspath_prepend(str(directory))
        # set here so that the test's end takes away what the download sets in this process
        for variable in DatasetDirectory.DIRECTORY_BY_VARIABLE:
            monkeypatch.setenv(variable, "elsewhere")
        # a home of the test's own, which nothing writes: the test's end takes away the one a download leaves
        monkeypatch.setenv("HOME", str(directory / "home"))
        # unset here, and the test's end takes away what a download sets in this process
        monkeypatch.setenv(DatasetDownload.HUB_PROBE_BUDGET_VARIABLE, "")
        monkeypatch.delenv(DatasetDownload.HUB_PROBE_BUDGET_VARIABLE)

    @staticmethod
    def benchmarks(*names: str) -> dict:
        benchmarks = {name: {"dataset": name, "dataset_args": {}, "repeats": 1, "max_tokens": 1} for name in names}
        return {"benchmarks": benchmarks, "sampling": {}, "request_timeout_s": 1, "concurrency": 1}

    @staticmethod
    def run_arguments(directory: Path) -> dict:
        """What a run of every benchmark takes besides its config, with the run's files under the directory."""
        return {
            "api_url": "http://127.0.0.1:1/v1",
            "model": "model",
            "out_dir": str(directory / "out"),
            "dataset_dir": str(directory / "data"),
            "only": None,
            "limit": None,
        }


def test_each_benchmark_runs_in_a_process_of_its_own(tmp_path, monkeypatch):
    # modelscope patches its whole process while it loads a dataset: two loads in one process undo each other's patch
    Evalscope.install(tmp_path, monkeypatch)

    BenchmarkRun(Evalscope.benchmarks("first", "second"), **Evalscope.run_arguments(tmp_path)).run()

    process_ids = {(tmp_path / "out" / name / "pid").read_text() for name in ("first", "second")}
    assert len(process_ids) == 2
    assert str(os.getpid()) not in process_ids


def test_a_failed_benchmark_fails_the_run_once_the_others_have_ended(tmp_path, monkeypatch):
    Evalscope.install(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="1 of 2 benchmarks failed: broken"):
        BenchmarkRun(Evalscope.benchmarks("broken", "second"), **Evalscope.run_arguments(tmp_path)).run()

    assert (tmp_path / "out" / "second" / "pid").exists()


def test_the_download_fetches_every_dataset_in_turn_without_an_endpoint(tmp_path, monkeypatch):
    Evalscope.install(tmp_path, monkeypatch)
    DatasetDownload(Evalscope.benchmarks("first", "second", "third"), str(tmp_path / "data"), only=None).run()

    assert (tmp_path / "data" / "fetched").read_text().split() == ["first", "second", "third"]


def test_a_benchmark_from_hugging_face_names_its_hub_and_its_dataset():
    benchmark = {"dataset": "mmmu_pro", "dataset_args": {"extra_params": {"dataset_format": "vision"}}}

    arguments = BenchmarkRun.dataset_arguments({**benchmark, "hf_dataset_id": "MMMU/MMMU_Pro"}, "/data")

    assert arguments == {
        "datasets": ["mmmu_pro"],
        "dataset_args": {"mmmu_pro": {"extra_params": {"dataset_format": "vision"}, "dataset_id": "MMMU/MMMU_Pro"}},
        "dataset_hub": "huggingface",
        "dataset_dir": "/data",
    }
    assert "dataset_hub" not in BenchmarkRun.dataset_arguments(benchmark, "/data")


def test_the_libraries_keep_their_own_directories_in_the_dataset_directory(tmp_path, monkeypatch):
    # left alone they write under the home directory, where the Hub also finds a stored login and refreshes it
    Evalscope.install(tmp_path, monkeypatch)
    names = ("modelscope", "modelscope/credentials", "huggingface", "nltk_data")
    directories = [f"{tmp_path}/data/{name}" for name in names]

    BenchmarkRun(Evalscope.benchmarks("first"), **Evalscope.run_arguments(tmp_path)).run()
    DatasetDownload(Evalscope.benchmarks("first"), f"{tmp_path}/data", only=None).run()

    assert (tmp_path / "out" / "first" / "directories").read_text().split() == directories
    assert (tmp_path / "data" / "directories").read_text().split() == directories


def test_the_download_fetches_the_scorers_nltk_data_into_the_dataset_directory(tmp_path, monkeypatch):
    # evalscope fetches a resource a scorer misses into ~/nltk_data, and NLTK_DATA moves the lookup alone
    Evalscope.install(tmp_path, monkeypatch)

    DatasetDownload(Evalscope.benchmarks("first"), f"{tmp_path}/data", only=None).run()

    assert sorted(path.name for path in (tmp_path / "data" / "nltk_data").iterdir()) == ["punkt_tab", "stopwords"]
    assert not (tmp_path / "home").exists()


def test_a_download_gives_the_hub_the_budget_of_its_probe_of_a_repository(tmp_path, monkeypatch):
    # modelscope's hub reads the budget when it is first imported, and probes a repository before every read of a fetch
    Evalscope.install(tmp_path, monkeypatch)

    DatasetDownload(Evalscope.benchmarks("first"), f"{tmp_path}/data", only=None).run()

    assert (tmp_path / "data" / "hub_probe_budget").read_text() == str(DatasetDownload.HUB_PROBE_BUDGET_S)


def test_a_caller_s_own_budget_for_the_hub_s_probe_is_kept(tmp_path, monkeypatch):
    Evalscope.install(tmp_path, monkeypatch)
    monkeypatch.setenv(DatasetDownload.HUB_PROBE_BUDGET_VARIABLE, "120")

    DatasetDownload(Evalscope.benchmarks("first"), f"{tmp_path}/data", only=None).run()

    assert (tmp_path / "data" / "hub_probe_budget").read_text() == "120"


def test_the_command_line_builds_the_command_it_names(monkeypatch):
    def command_of(*words: str):
        monkeypatch.setattr(sys, "argv", ["general", *words])
        return GeneralEvaluation.from_command_line().command

    router = command_of("router", "--port", "9000", "--backends", "first:8000,second:8001")
    run = command_of(
        "run", "--api-url", "http://head:9000/v1", "--model", "model", "--out", "/out", "--dataset-dir", "/data"
    )
    download = command_of("download", "--dataset-dir", "/data", "--only", "ifbench")
    comparison = command_of("compare", "--a", "/a", "--b", "/b", "--names", "base", "tuned")

    assert isinstance(router, Router)
    assert (router.replicas, router.port) == (["http://first:8000", "http://second:8001"], 9000)
    assert isinstance(run, BenchmarkRun)
    assert (run.api_url, run.model, run.out_dir, run.dataset_dir) == ("http://head:9000/v1", "model", "/out", "/data")
    # every benchmark of the config, and all of their samples
    assert (run.only, run.limit) == (None, None)
    assert isinstance(download, DatasetDownload)
    assert (download.dataset_dir, download.only) == ("/data", ["ifbench"])
    assert isinstance(comparison, RunComparison)
    assert (comparison.run_dir_a, comparison.run_dir_b, comparison.names) == ("/a", "/b", ("base", "tuned"))
