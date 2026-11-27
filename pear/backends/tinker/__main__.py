from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import closing
from pathlib import Path

from pear.utils.file_lock import FileLock
from pear.utils.file_utils import write_text_atomically

from .config import TinkerConfig
from .preflight import preflight
from .reporting import run_metadata, write_report
from .runtime import PearDataset, SimulatorRuntime, make_renderer
from .sampling import head_request_limit
from .state import Checkpoints
from .train import evaluate, train
from .usage import Usage, reconcile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEAR episodes and policy updates through Tinker.")
    parser.add_argument("command", choices=("train", "eval", "check", "preflight", "usage"))
    parser.add_argument("--timeout", type=float, default=900, help="preflight timeout per environment, in seconds")
    parser.add_argument("--config", required=True, help="YAML recipe; file paths are relative to the working directory")
    parser.add_argument(
        "--checkpoint", help="Tinker checkpoint for evaluation; defaults to this run's latest or the base model"
    )
    parser.add_argument("overrides", nargs="*", help="configuration overrides as key=value")
    args = parser.parse_args()
    config = TinkerConfig.load(args.config, args.overrides)
    if args.command == "train" and config.objective == "pear":
        # Fail before creating a paid client or simulator pool if the recipe needs the expensive fallback.
        head_request_limit(config.response_budget, config.topk, config.allow_head_continuations)
    if args.checkpoint and args.command != "eval":
        parser.error("--checkpoint is for evaluation; train resumes run_dir or starts from init_checkpoint")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "usage":
        result = asyncio.run(reconcile(run_dir))
        print(json.dumps({key: value for key, value in result.items() if key != "events"}), flush=True)
        return
    runtime = SimulatorRuntime(config)
    dataset = PearDataset(config, runtime.tool_configs, evaluation=args.command == "eval")
    if args.command == "check":
        print(
            json.dumps(
                {
                    "episodes": len(dataset.tasks),
                    "environments": list(runtime.tool_configs),
                    "objective": config.objective,
                    "model": config.base_model,
                },
                indent=2,
            )
        )
        return

    if args.command == "preflight":
        with closing(FileLock(run_dir / "run.lock")):
            report_path = run_dir / "preflight/report.json"
            write_report(report_path, {**run_metadata(config), "status": "starting"})
            try:
                with runtime:
                    result = asyncio.run(preflight(config, runtime, dataset, args.timeout))
            except Exception as error:
                report = json.loads(report_path.read_text())
                write_report(report_path, {**report, "status": "failed", "error": str(error)})
                raise
        print(json.dumps(result), flush=True)
        return

    import tinker

    with closing(FileLock(run_dir / "run.lock")):
        reference = Checkpoints(config).reference() if args.command == "eval" and not args.checkpoint else None
        evaluation_path = args.checkpoint or (reference["model_path"] if reference else None)
        metadata = run_metadata(config)
        if args.command == "eval":
            metadata["checkpoint"] = evaluation_path
        write_text_atomically(
            run_dir / f"{args.command}_config.json",
            json.dumps(metadata, indent=2),
        )
        renderer = make_renderer(config)
        # The SDK reads TINKER_API_KEY or its saved login; neither is written into the run configuration.
        usage = Usage(config.run_dir, args.command)
        try:
            with tinker.ServiceClient(user_metadata={"pear_run_id": usage.identity["run_id"]}) as service, runtime:
                if args.command == "train":
                    result = asyncio.run(train(config, runtime, dataset, renderer, service, usage))
                else:
                    result = asyncio.run(
                        evaluate(
                            config,
                            runtime,
                            dataset,
                            renderer,
                            service,
                            evaluation_path,
                            reference["step"] if reference else 0,
                            usage,
                        )
                    )
        except BaseException as error:
            write_report(
                run_dir / f"{args.command}_report.json",
                {
                    **metadata,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "usage": dict(usage.totals),
                },
            )
            raise
        write_report(
            run_dir / f"{args.command}_report.json",
            {**metadata, "status": "passed", "result": result, "usage": dict(usage.totals)},
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
