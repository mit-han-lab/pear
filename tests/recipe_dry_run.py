import argparse
import ast
import functools
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from pear.trainer.config import RunConfig
from repository import REPO


@dataclass(frozen=True)
class RecipeLaunch:
    """What a recipe's dry run launched: the entry module with its Hydra overrides and the driver's environment, and
    the arguments of the progress panel beside it."""

    module: str
    overrides: list[str]
    environment: dict[str, str]
    panel_arguments: list[str]

    @functools.cached_property
    def composed_config(self) -> dict:
        """The config the entry module's Hydra main composes from the overrides, as it hands it to verl's launch."""
        config_dir, config_name = self.entry_config()
        GlobalHydra.instance().clear()

        # an override that is no key of the config, or that adds a key it has, raises here
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name=config_name, overrides=self.overrides)
        RunConfig(config).prepare()

        return OmegaConf.to_container(config, resolve=True)

    def entry_config(self) -> tuple[Path, str]:
        """The config directory and the config name the entry module's Hydra main names, read from its decorator."""
        module_file = Path(importlib.util.find_spec(self.module).origin)
        functions = [node for node in ast.parse(module_file.read_text()).body if isinstance(node, ast.FunctionDef)]
        [main] = [function for function in functions if function.name == "main"]
        [hydra_main] = main.decorator_list
        arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in hydra_main.keywords}

        # from the module's file, as Hydra finds the configs of a module run with `python -m`
        return (module_file.parent / arguments["config_path"]).resolve(), arguments["config_name"]

    def override_keys(self) -> list[str]:
        """The key of every override, without the + or ~ that adds or removes it."""
        return [override.partition("=")[0].lstrip("+~") for override in self.overrides]


class RecipeDryRun:
    """A recipe run to its launch in a sandbox: a copy of a tree's recipes, with stand-ins for the programs they call and
    for the verl checkout step."""

    GPUS = ("gb300", "h100")  # what a recipe's GPU may be
    RAY_LAUNCH_MARK = "ray_kwargs.ray_init.runtime_env"  # in the text of a recipe that launches on a Ray cluster
    PANEL_MODULE = "pear.monitor.progress_panel"  # the progress panel a recipe starts beside its launch
    # a cluster of 4 GPU machines and 2 simulator machines: 2 trainer and 2 rollout machines for a training, 4 rollout
    # machines for an evaluation
    TRAINING_CLUSTER = {"TRAINER_NNODES": "2", "ROLLOUT_NNODES": "2", "SIM_NNODES": "2"}
    EVALUATION_CLUSTER = {"ROLLOUT_NNODES": "4", "SIM_NNODES": "2"}
    EVAL_STEP = "1"  # an evaluation's checkpoint step
    TIMEOUT_S = 120  # for a recipe to reach its launch
    PANEL_WAIT_S = 30  # for the panel's record to be there when a launch has made its own
    POLL_S = 0.05  # between two looks for it
    SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"  # where the recipes find bash and the coreutils, on any machine
    # variables of the driver's environment that are the sandbox's or the shell's own, not the recipe's
    OWN_VARIABLE_PREFIX = "PEAR_DRY_RUN_"
    SHELL_VARIABLES = ("PWD", "OLDPWD", "SHLVL", "_")
    # The training environment's python: a -c program runs for real, a module's launch is recorded (its arguments and
    # its environment, NUL-separated) instead of started. A launch ends after the panel beside it is recorded.
    PYTHON = """#!/bin/bash
case "$1" in
  -c)
    exec "$PEAR_DRY_RUN_PYTHON" "$@"
    ;;
  -m)
    module=$2
    record=$PEAR_DRY_RUN_DIR/$module
    shift 2
    env -0 > "$record.environment"
    printf '%s\\0' "$@" > "$record.partial"
    mv "$record.partial" "$record.arguments"
    if [ "$module" != "$PEAR_DRY_RUN_PANEL" ]; then
      until [ -e "$PEAR_DRY_RUN_DIR/$PEAR_DRY_RUN_PANEL.arguments" ]; do
        if [ "$SECONDS" -ge "$PEAR_DRY_RUN_PANEL_WAIT_S" ]; then
          echo "the recipe started no progress panel" >&2
          exit 1
        fi
        sleep "$PEAR_DRY_RUN_POLL_S"
      done
    fi
    ;;
  *)
    echo "a dry run has no stand-in for: python $*" >&2
    exit 1
    ;;
esac
"""
    DATE = "#!/bin/sh\necho 20260101T000000Z\n"  # one clock for every launch: its start time is in its paths
    CHECKOUT = '#!/usr/bin/env bash\necho "verl: a dry run checks nothing out"\n'  # no clone, no network

    def __init__(self, root: Path, sandbox: Path) -> None:
        self.sandbox = sandbox.resolve()  # resolved: the recipes resolve their own paths
        self.tree = self.sandbox / "tree"  # the recipes' ROOT
        self.work_dir = self.sandbox / "work"
        self.local_dir = self.sandbox / "local"
        self.model = self.sandbox / "model"
        self.verl_env = self.sandbox / "env"
        self.stand_ins = self.sandbox / "bin"

        shutil.copytree(root / "examples", self.tree / "examples")
        (self.tree / "pear").symlink_to(root.resolve() / "pear", target_is_directory=True)
        (self.tree / "scripts").symlink_to(root.resolve() / "scripts", target_is_directory=True)
        scripts = {
            self.tree / "3rdparty" / "verl" / "scripts" / "checkout_verl.sh": self.CHECKOUT,
            self.verl_env / "bin" / "python": self.PYTHON,
            self.stand_ins / "date": self.DATE,
        }
        for path, text in scripts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            path.chmod(0o755)

    @classmethod
    def recipes(cls, root: Path) -> list[str]:
        """The tree's recipes that launch on a Ray cluster, by their paths in it."""
        scripts = sorted((root / "examples").rglob("*.sh"))

        return [str(script.relative_to(root)) for script in scripts if cls.RAY_LAUNCH_MARK in script.read_text()]

    def run(self, recipe: str, gpu: str, user_variables: dict[str, str] | None = None) -> RecipeLaunch:
        """The launch of ``recipe`` on a cluster of ``gpu`` machines, with the ``user_variables`` a user gave it."""
        name = f"{Path(recipe).stem}.{gpu}"
        record_dir = self.sandbox / "records" / name
        record_dir.mkdir(parents=True)
        # a recipe of a model's train/ is a training
        cluster = self.TRAINING_CLUSTER if Path(recipe).parent.name == "train" else self.EVALUATION_CLUSTER
        variables = {
            "WORK_DIR": str(self.work_dir),
            "VERL_ENV": str(self.verl_env),
            "LOCAL_DIR": str(self.local_dir),
            "RUN": str(self.sandbox / "runs" / name),
            "MODEL": str(self.model),
            "GPU": gpu,
            "EVAL_STEP": self.EVAL_STEP,
            **cluster,
            **(user_variables or {}),
        }
        own_variables = {
            "PEAR_DRY_RUN_DIR": str(record_dir),
            "PEAR_DRY_RUN_PYTHON": sys.executable,
            "PEAR_DRY_RUN_PANEL": self.PANEL_MODULE,
            "PEAR_DRY_RUN_PANEL_WAIT_S": str(self.PANEL_WAIT_S),
            "PEAR_DRY_RUN_POLL_S": str(self.POLL_S),
        }

        process = subprocess.run(
            ["bash", str(self.tree / recipe)],
            env={"PATH": f"{self.stand_ins}{os.pathsep}{self.SYSTEM_PATH}", **variables, **own_variables},
            capture_output=True,
            text=True,
            timeout=self.TIMEOUT_S,
        )
        if process.returncode != 0:
            raise RuntimeError(f"{recipe} ended with {process.returncode}:\n{process.stdout}{process.stderr}")

        [launch_record] = [
            path for path in record_dir.glob("*.arguments") if path.name != f"{self.PANEL_MODULE}.arguments"
        ]
        environment = dict(item.split("=", 1) for item in self.items(launch_record.with_suffix(".environment")))

        return RecipeLaunch(
            module=launch_record.stem,
            overrides=self.items(launch_record),
            environment={
                variable: value
                for variable, value in environment.items()
                if not variable.startswith(self.OWN_VARIABLE_PREFIX) and variable not in self.SHELL_VARIABLES
            },
            panel_arguments=self.items(record_dir / f"{self.PANEL_MODULE}.arguments"),
        )

    def sanitized(self, text: str) -> str:
        """``text`` with the sandbox's paths as placeholders, the same in every sandbox."""
        placeholders = {
            self.tree: "<ROOT>",
            self.work_dir: "<WORK_DIR>",
            self.local_dir: "<LOCAL_DIR>",
            self.model: "<MODEL>",
            self.verl_env: "<VERL_ENV>",
            self.sandbox / "runs": "<RUNS>",
            self.sandbox: "<SANDBOX>",
        }
        for path, placeholder in placeholders.items():
            text = text.replace(str(path), placeholder)

        return text

    @staticmethod
    def items(record: Path) -> list[str]:
        """What a stand-in recorded in ``record``: a call's arguments or its environment, each item ended by a NUL."""
        return record.read_text().split("\0")[:-1]


class RecipeDump:
    """The command: the launches of a tree's recipes written to a directory, to compare two trees by."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root).resolve()
        self.out = Path(args.out)
        self.recipes = args.recipe or RecipeDryRun.recipes(self.root)
        self.gpus = args.gpu or RecipeDryRun.GPUS

    @classmethod
    def from_command_line(cls) -> "RecipeDump":
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--out", required=True, help="the directory to write to: a directory per recipe and GPU")
        parser.add_argument("--root", default=str(REPO), help="the tree, whose pear and verl are the ones on the path")
        parser.add_argument("--recipe", action="append", help="a recipe's path in the tree (by default every recipe)")
        parser.add_argument("--gpu", action="append", choices=RecipeDryRun.GPUS, help="by default every GPU")
        return cls(parser.parse_args())

    def run(self) -> None:
        # the entry modules compose from the pear and verl this process imports: they must be the tree's
        imported_root = Path(importlib.import_module("pear").__file__).resolve().parents[1]
        if imported_root != self.root:
            raise SystemExit(f"pear is imported from {imported_root}, not from {self.root}: set PYTHONPATH to the tree")

        with tempfile.TemporaryDirectory() as sandbox:
            dry_run = RecipeDryRun(self.root, Path(sandbox))
            for recipe in self.recipes:
                for gpu in self.gpus:
                    self.write(dry_run, recipe, gpu)

    def write(self, dry_run: RecipeDryRun, recipe: str, gpu: str) -> None:
        """One launch's files, the sandbox's paths as placeholders in them."""
        launch = dry_run.run(recipe, gpu)
        directory = self.out / f"{Path(recipe).stem}.{gpu}"
        files = {
            "overrides.txt": sorted(launch.overrides),
            "config.txt": self.leaves(launch.composed_config),
            "environment.txt": sorted(f"{name}={value}" for name, value in launch.environment.items()),
            "panel.txt": launch.panel_arguments,
        }

        directory.mkdir(parents=True)
        for name, lines in files.items():
            (directory / name).write_text(dry_run.sanitized("\n".join([f"# {launch.module}", *lines, ""])))
        print(f"{recipe} on {gpu}: {len(launch.overrides)} overrides, {directory}")

    @classmethod
    def leaves(cls, config, prefix: str = "") -> list[str]:
        """The config's leaves in the order of their keys, one ``key = value`` line each."""
        if isinstance(config, dict) and config:
            items = [(f"{prefix}.{key}" if prefix else str(key), config[key]) for key in sorted(config, key=str)]
        elif isinstance(config, list) and config:
            items = [(f"{prefix}[{index}]", value) for index, value in enumerate(config)]
        else:
            return [f"{prefix} = {json.dumps(config)}"]

        return [line for key, value in items for line in cls.leaves(value, key)]


if __name__ == "__main__":
    RecipeDump.from_command_line().run()
