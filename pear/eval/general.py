from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, TCPConnector, web


class Router:
    """One OpenAI endpoint over many vLLM replicas, each request to the replica with the fewest in flight."""

    ATTEMPTS = 3  # replicas a request tries: one that drops the connection hands it to another
    CONNECT_TIMEOUT_S = 60  # to connect to a replica; long reasoning requests have no timeout but the client's
    MAX_BODY = 1 << 30  # bytes of a request (its images included)
    KEEPALIVE_S = 3600  # far above aiohttp's 75 s, so that a client never reuses a connection the router just closed

    def __init__(self, replicas: list[str], port: int) -> None:
        self.replicas = [f"http://{replica}" for replica in replicas]
        self.port = port
        self.inflight_request_n = {replica: 0 for replica in self.replicas}
        self.session = None  # the replicas' client session, opened with the app

    def run(self) -> None:
        app = web.Application(client_max_size=self.MAX_BODY)
        app.on_startup.append(self.open)
        app.on_cleanup.append(self.close)
        app.router.add_route("*", "/{tail:.*}", self.proxy)
        web.run_app(app, host="0.0.0.0", port=self.port, access_log=None, keepalive_timeout=self.KEEPALIVE_S)

    async def open(self, app: web.Application) -> None:
        self.session = ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=self.CONNECT_TIMEOUT_S), connector=TCPConnector(limit=0)
        )

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        """The answer of the replica with the fewest requests in flight, or of another one when it drops the
        connection."""
        body = await request.read()
        headers = {
            name: value for name, value in request.headers.items() if name.lower() in ("content-type", "authorization")
        }

        tried_replicas = set()
        for attempt in range(self.ATTEMPTS):
            # a replica this request found down is tried again only once every other one was
            untried_replicas = [
                candidate for candidate in self.replicas if candidate not in tried_replicas
            ] or self.replicas
            replica = min(untried_replicas, key=lambda candidate: (self.inflight_request_n[candidate], random.random()))
            tried_replicas.add(replica)
            self.inflight_request_n[replica] += 1
            try:
                async with self.session.request(
                    request.method, replica + request.path_qs, data=body, headers=headers
                ) as response:
                    payload = await response.read()
                    return web.Response(body=payload, status=response.status, content_type=response.content_type)
            except ClientConnectionError as exc:
                print(f"{replica}: {exc!r} (attempt {attempt + 1})", flush=True)
            finally:
                self.inflight_request_n[replica] -= 1

        return web.Response(status=502, text="no replica answered")

    async def close(self, app: web.Application) -> None:
        await self.session.close()


class DatasetDirectory:
    """The directory the benchmarks' datasets are kept in: evalscope's downloads, and under it what modelscope and the
    Hugging Face Hub keep for themselves (their caches, their downloads, their sessions and logins) and the NLTK data
    the scorers read."""

    # the variable a library reads its own directory from when it is first imported, and that directory under the
    # datasets'. Left alone each is under the home directory, where the Hub also finds a stored login and refreshes it
    DIRECTORY_BY_VARIABLE = {
        "MODELSCOPE_CACHE": "modelscope",
        "MODELSCOPE_CREDENTIALS_PATH": "modelscope/credentials",
        "HF_HOME": "huggingface",
        "NLTK_DATA": "nltk_data",
    }

    @classmethod
    def use(cls, path: str) -> None:
        """Points the libraries' own directories into this one, for this process."""
        for variable, name in cls.DIRECTORY_BY_VARIABLE.items():
            os.environ[variable] = os.path.join(path, name)


class BenchmarkRun:
    """The benchmarks of pear/configs/general.yaml (or a subset) against an OpenAI endpoint, each an evalscope task under
    OUT/<benchmark>; a rerun on the same OUT reuses the finished samples."""

    DONE_STATUS = "ok"  # how a benchmark that ran to its end is reported
    STATUS_LIMIT = 300  # characters of a failed benchmark's error in its report line

    def __init__(
        self,
        config: dict,
        api_url: str,
        model: str,
        out_dir: str,
        dataset_dir: str,
        only: list[str] | None,
        limit: int | None,
    ) -> None:
        self.config = config
        self.api_url = api_url
        self.model = model  # the served model name
        self.out_dir = out_dir
        self.dataset_dir = dataset_dir
        self.only = only  # the benchmarks to run; None: every one of the config
        self.limit = limit  # a smoke: the first samples of each benchmark; None: all of them

    def run(self) -> None:
        names = self.only or list(self.config["benchmarks"])
        os.makedirs(self.out_dir, exist_ok=True)
        failed_names = []

        # every benchmark at once: the replicas see one steady stream instead of each benchmark's tail. A process each:
        # while it loads a dataset, modelscope patches the datasets and huggingface_hub libraries of its whole process,
        # so two loads in one process undo each other's patch
        with ProcessPoolExecutor(len(names), max_tasks_per_child=1) as pool:
            for name, duration_s, status in pool.map(self.benchmark, names):
                print(f"{name}: {status} in {duration_s / 60:.0f} min", flush=True)
                if status != self.DONE_STATUS:
                    failed_names.append(name)

        if failed_names:
            raise SystemExit(f"{len(failed_names)} of {len(names)} benchmarks failed: {', '.join(failed_names)}")

    def benchmark(self, name: str) -> tuple[str, float, str]:
        """Runs one benchmark as an evalscope task: its name, how long it took and how it ended."""
        DatasetDirectory.use(self.dataset_dir)
        from evalscope import TaskConfig, run_task

        benchmark = self.config["benchmarks"][name]
        work_dir = os.path.join(self.out_dir, name)
        start_time = time.monotonic()
        config = TaskConfig(
            model=self.model,
            api_url=self.api_url,
            api_key="EMPTY",
            eval_type="openai_api",
            **self.dataset_arguments(benchmark, self.dataset_dir),
            generation_config={
                **self.config["sampling"],
                "max_tokens": benchmark["max_tokens"],
                "timeout": self.config["request_timeout_s"],
            },
            repeats=benchmark["repeats"],
            eval_batch_size=self.config["concurrency"],
            work_dir=work_dir,
            limit=self.limit,
            # a rerun on the same OUT reuses the finished predictions (only where there are some: evalscope rejects a
            # cache without its record of the evaluation's identity)
            use_cache=work_dir if os.path.isdir(os.path.join(work_dir, "predictions")) else None,
            no_timestamp=True,
        )

        try:
            run_task(config)
            status = self.DONE_STATUS
        except Exception as exc:  # one benchmark's failure must not stop the others
            status = f"failed: {exc!r}"[: self.STATUS_LIMIT]

        return name, time.monotonic() - start_time, status

    @staticmethod
    def dataset_arguments(benchmark: dict, dataset_dir: str) -> dict:
        """The arguments of an evalscope task that name a benchmark's dataset and where it is kept."""
        dataset, dataset_args = benchmark["dataset"], benchmark["dataset_args"]
        hf_id = benchmark.get("hf_dataset_id")
        if hf_id:
            dataset_args = {**dataset_args, "dataset_id": hf_id}

        return {
            "datasets": [dataset],
            "dataset_args": {dataset: dataset_args} if dataset_args else {},
            **({"dataset_hub": "huggingface"} if hf_id else {}),
            "dataset_dir": dataset_dir,
        }


class DatasetDownload:
    """The datasets of the benchmarks of pear/configs/general.yaml (or a subset) and the NLTK data of evalscope's
    scorers, fetched into the dataset directory one after the other, with no endpoint."""

    # the variable modelscope's hub reads its budget for one probe of a repository before a read from, and the seconds
    # a download gives it: the hub's own limit for an idle read (DOWNLOAD_TIMEOUT of modelscope-hub 0.5.1), as its
    # default ends the probe of a far endpoint. A fetched dataset is read without the probe; a caller's value is kept
    HUB_PROBE_BUDGET_VARIABLE = "MODELSCOPE_REPO_TREE_PAGE_TIMEOUT_SECONDS"
    HUB_PROBE_BUDGET_S = 60

    def __init__(self, config: dict, dataset_dir: str, only: list[str] | None) -> None:
        self.config = config
        self.dataset_dir = dataset_dir
        self.only = only  # the benchmarks whose datasets to fetch; None: every one of the config

    def run(self) -> None:
        DatasetDirectory.use(self.dataset_dir)
        os.environ.setdefault(self.HUB_PROBE_BUDGET_VARIABLE, str(self.HUB_PROBE_BUDGET_S))
        from evalscope import TaskConfig
        from evalscope.api.registry import get_benchmark
        from evalscope.utils.resource_utils import MIRROR_MAP, check_nltk_data

        for name in self.only or list(self.config["benchmarks"]):
            start_time = time.monotonic()
            task = TaskConfig(**BenchmarkRun.dataset_arguments(self.config["benchmarks"][name], self.dataset_dir))
            get_benchmark(task.datasets[0], task).load_dataset()
            print(f"{name}: fetched in {time.monotonic() - start_time:.0f} s", flush=True)

        # evalscope fetches an NLTK resource a scorer misses into ~/nltk_data and nowhere else: from here on this
        # process's home is the dataset directory, so that each lands where NLTK_DATA points a run
        os.environ["HOME"] = self.dataset_dir
        for resource in MIRROR_MAP:
            check_nltk_data(resource)
        print(f"nltk data: {', '.join(MIRROR_MAP)}", flush=True)


class RunComparison:
    """Two runs per benchmark: each one's mean score with its standard error, and a paired bootstrap of the
    difference."""

    def __init__(self, config: dict, run_dir_a: str, run_dir_b: str, names: tuple[str, str]) -> None:
        self.config = config
        self.run_dir_a = run_dir_a
        self.run_dir_b = run_dir_b
        self.names = names  # the two runs' column titles

    def run(self) -> None:
        resamples, seed, confidence = (self.config["bootstrap"][key] for key in ("samples", "seed", "confidence"))
        rng = random.Random(seed)
        tail = int(resamples * (1 - confidence) / 2)  # resamples below the interval, and above it
        name_a, name_b = self.names

        print(f"{'benchmark':22s} {name_a:>16s} {name_b:>16s} {'diff':>8s} {'95% CI (paired bootstrap)':>28s}  n")
        for name in self.config["benchmarks"]:
            scores_a = self.sample_scores(self.run_dir_a, name)
            scores_b = self.sample_scores(self.run_dir_b, name)
            shared_samples = sorted(set(scores_a) & set(scores_b))
            if not shared_samples:
                print(f"{name:22s} (missing in {'a' if not scores_a else 'b'})")
                continue

            values_a = [scores_a[sample] for sample in shared_samples]
            values_b = [scores_b[sample] for sample in shared_samples]
            sample_n = len(shared_samples)
            (mean_a, se_a), (mean_b, se_b) = self.mean_se(values_a), self.mean_se(values_b)
            differences = [values_b[i] - values_a[i] for i in range(sample_n)]
            resampled_means = sorted(
                sum(differences[rng.randrange(sample_n)] for _ in range(sample_n)) / sample_n for _ in range(resamples)
            )
            lower, upper = resampled_means[tail], resampled_means[resamples - 1 - tail]
            significance_mark = " *" if upper < 0 or lower > 0 else ""

            print(
                f"{name:22s} {100 * mean_a:9.1f} ±{100 * se_a:4.1f} {100 * mean_b:9.1f} ±{100 * se_b:4.1f} "
                f"{100 * (mean_b - mean_a):+8.1f}    [{100 * lower:+6.1f}, {100 * upper:+6.1f}]"
                f"{significance_mark:2s} {sample_n}"
            )

        print("* the 95% interval of the paired difference excludes 0")

    @staticmethod
    def sample_scores(run_dir: str, name: str) -> dict[str, float]:
        """Per-sample score of one benchmark's evalscope reviews (mean over its repeats), keyed by the sample's id."""
        scores: dict[str, list[float]] = {}
        for path in glob.glob(os.path.join(run_dir, name, "**", "reviews", "**", "*.jsonl"), recursive=True):
            with open(path) as reviews:
                for line in reviews:
                    review = json.loads(line)
                    sample = review.get("sample_score", {})
                    # the repeats of a sample share its group id
                    key = str(sample.get("group_id", review.get("index")))
                    score = sample.get("score", {})
                    value = score.get("main_score_name") and score["value"].get(score["main_score_name"])
                    if value is None:
                        value = next(iter(score.get("value", {}).values()), 0.0)
                    scores.setdefault(f"{os.path.basename(path)}:{key}", []).append(float(value))

        return {sample_key: sum(repeat_scores) / len(repeat_scores) for sample_key, repeat_scores in scores.items()}

    @staticmethod
    def mean_se(values: list[float]) -> tuple[float, float]:
        """The mean of ``values`` and its standard error."""
        sample_n = len(values)
        mean = sum(values) / sample_n

        return mean, (sum((value - mean) ** 2 for value in values) / max(sample_n - 1, 1) / sample_n) ** 0.5


class GeneralEvaluation:
    """General-capability evaluation of a model served by vLLM (OpenAI API) through evalscope, one protocol for every
    model compared: a router over the replicas, the benchmarks' run and the paired comparison of two runs."""

    CONFIG_PATH = Path(__file__).parents[1] / "configs" / "general.yaml"

    def __init__(self, command: Router | BenchmarkRun | DatasetDownload | RunComparison) -> None:
        self.command = command

    @classmethod
    def from_command_line(cls) -> GeneralEvaluation:
        config = yaml.safe_load(cls.CONFIG_PATH.read_text())
        parser = argparse.ArgumentParser(description=cls.__doc__)
        commands = parser.add_subparsers(dest="command", required=True)

        router = commands.add_parser("router", help=Router.__doc__)
        router.add_argument("--port", type=int, required=True)
        router.add_argument("--backends", required=True, help="host:port,host:port,...")

        run = commands.add_parser("run", help=BenchmarkRun.__doc__)
        run.add_argument("--api-url", required=True)
        run.add_argument("--model", required=True, help="the served model name")
        run.add_argument("--out", required=True)
        run.add_argument("--only", nargs="+", choices=sorted(config["benchmarks"]))
        run.add_argument("--limit", type=int, help="a smoke: the first N samples of each benchmark")

        download = commands.add_parser("download", help=DatasetDownload.__doc__)
        download.add_argument("--only", nargs="+", choices=sorted(config["benchmarks"]))
        for command in (run, download):
            command.add_argument("--dataset-dir", required=True, help=DatasetDirectory.__doc__)

        compare = commands.add_parser("compare", help=RunComparison.__doc__)
        compare.add_argument("--a", required=True)
        compare.add_argument("--b", required=True)
        compare.add_argument("--names", nargs=2, default=("a", "b"))

        args = parser.parse_args()
        if args.command == "router":
            command = Router(args.backends.split(","), args.port)
        elif args.command == "run":
            command = BenchmarkRun(config, args.api_url, args.model, args.out, args.dataset_dir, args.only, args.limit)
        elif args.command == "download":
            command = DatasetDownload(config, args.dataset_dir, args.only)
        else:
            command = RunComparison(config, args.a, args.b, tuple(args.names))

        return cls(command)

    def run(self) -> None:
        self.command.run()


if __name__ == "__main__":
    GeneralEvaluation.from_command_line().run()
