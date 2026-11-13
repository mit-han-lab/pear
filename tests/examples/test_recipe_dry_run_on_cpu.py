import argparse
import filecmp
import json
import os
import pkgutil
import re
import sys
from pathlib import Path
from unittest import mock

import pytest
from omegaconf import OmegaConf

from recipe_dry_run import RecipeDryRun, RecipeDump, RecipeLaunch
from repository import REPO
from verl.utils.dataset.rl_dataset import get_dataset_class

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the recipes call GNU readlink")
# every recipe that launches on a Ray cluster, on each GPU
LAUNCHES = [(recipe, gpu) for recipe in RecipeDryRun.recipes(REPO) for gpu in RecipeDryRun.GPUS]
ENTRY_MODULES = ("pear.trainer.main", "pear.eval.embodied")  # PEAR's entry points, each with a Hydra main
CLASS_PATH = re.compile(r"(pear|verl)(\.\w+)+")  # a module or a class of PEAR's or verl's, by its dotted path
RUNTIME_ENV_KEY = "ray_kwargs.ray_init.runtime_env.env_vars"  # what a launch gives its Ray workers
TOOL_CONFIG_KEY = "actor_rollout_ref.rollout.multi_turn.tool_config_path"
EPISODE_FILE_KEYS = ("data.train_files", "data.val_files")
# What a worker reads from the environment a recipe gives it: the agent loop in every launch, then a family's server
# and simulator process by the env class of its tool: pear's code, LIBERO's own package (LIBERO_CONFIG_PATH),
# LIBERO-plus's ImageMagick (MAGICK_HOME). A tool config's variables are read from the tool config.
LAUNCH_VARIABLES = ("PEAR_LOG_DIR",)
FAMILY_VARIABLES = {
    "pear.envs.libero.env.LiberoEnv": ("LIBERO_MESA_DIR", "LIBERO_PYTHON", "LIBERO_CONFIG_PATH"),
    "pear.envs.libero_plus.env.LiberoPlusEnv": (
        "LIBERO_MESA_DIR",
        "LIBERO_PYTHON",
        "LIBERO_CONFIG_PATH",
        "MAGICK_HOME",
    ),
    "pear.envs.libero_pro.env.LiberoProEnv": ("LIBERO_MESA_DIR", "LIBERO_PYTHON", "LIBERO_CONFIG_PATH"),
    "pear.envs.robodojo.env.RoboDojoEnv": (),
    "pear.envs.robolab.env.RoboLabEnv": (),
}
# a replica's decoding slots, the agent loop's processes and the sessions kept generating. The five evaluations have
# the same machines.
SIZE_KEYS = (
    "actor_rollout_ref.rollout.max_num_seqs",
    "actor_rollout_ref.rollout.agent.num_workers",
    "trainer.v1.sampler.max_inflight_sessions",
)
TRAINING_SIZES = {
    "run_stage1.gb300": (128, 64, 256),
    "run_stage1.h100": (128, 64, 256),
    "run_stage2.gb300": (64, 32, 128),
    "run_stage2.h100": (64, 32, 128),
    "run_stage3.gb300": (64, 32, 128),
    "run_stage3.h100": (64, 32, 128),
}
EVALUATION_SIZES = {"gb300": (64, 64, 256), "h100": (64, 64, 256)}
# the recipes whose user may give the Apps of a simulator server: the variable, and the tool config's own number
APPS_VARIABLES = {
    "examples/models/qwen3_8_27b_pear/train/run_stage3.sh": ("ROBODOJO_APPS_PER_GPU", "8"),
    "examples/eval/eval_robodojo.sh": ("ROBODOJO_APPS_PER_GPU", "8"),
    "examples/eval/eval_robolab.sh": ("ROBOLAB_APPS_PER_GPU", "10"),
}


class ComposedConfig:
    """The config a launch's entry point composes, as the tests read it."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def value(self, key: str):
        """The value at a dotted key."""
        value = self.config
        for part in key.split("."):
            value = value[part]

        return value

    def named_paths(self) -> set[str]:
        """Every module or class of PEAR's or verl's the config names, as a value or as a key."""
        nodes, paths = [self.config], set()
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                nodes.extend([*node, *node.values()])
            elif isinstance(node, list):
                nodes.extend(node)
            elif isinstance(node, str) and CLASS_PATH.fullmatch(node):
                paths.add(node)

        return paths

    def tool_configs(self) -> list[dict]:
        """The configs of the launch's tools, their tool config read as a worker reads it: under the workers'
        environment alone."""
        with mock.patch.dict(os.environ, self.value(RUNTIME_ENV_KEY), clear=True):
            tools = OmegaConf.to_container(OmegaConf.load(self.value(TOOL_CONFIG_KEY)), resolve=True)["tools"]

        return [tool["config"] for tool in tools]

    def tool_env_classes(self) -> list[str]:
        """The env classes of the launch's tools."""
        return [config["env_class"] for config in self.tool_configs()]


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory) -> RecipeDryRun:
    return RecipeDryRun(REPO, tmp_path_factory.mktemp("recipes"))


@pytest.fixture(scope="module", params=LAUNCHES, ids=lambda launch: f"{Path(launch[0]).stem}-{launch[1]}")
def launch(request, dry_run) -> RecipeLaunch:
    return dry_run.run(*request.param)


@pytest.fixture(scope="module")
def composed(launch) -> ComposedConfig:
    return ComposedConfig(launch.composed_config)


def test_a_recipe_launches_an_entry_point_and_gives_no_key_twice(launch):
    keys = launch.override_keys()

    assert launch.module in ENTRY_MODULES
    assert sorted(keys) == sorted(set(keys))


def test_the_overrides_of_a_recipe_compose_in_its_entry_point(launch):
    # an override that is no key of the entry point's config, or that adds a key it has, raises here
    assert launch.composed_config


def test_every_module_and_class_a_launch_names_can_be_imported(composed, monkeypatch):
    # verl loads the dataset class from its file, by a path it takes from the working directory: a recipe's is the tree
    monkeypatch.chdir(REPO)
    worker_modules = composed.value(RUNTIME_ENV_KEY)["VERL_USE_EXTERNAL_MODULES"].split(",")

    # a path that names nothing raises
    for path in [*composed.named_paths(), *worker_modules]:
        pkgutil.resolve_name(path)
    dataset_class = get_dataset_class(OmegaConf.create(composed.value("data")))
    assert dataset_class.__name__ == composed.value("data.custom_cls.name")


def test_the_process_a_recipe_starts_has_a_gpu_machine_s_environment(launch, dry_run):
    # it imports vLLM, whose Triton builds an extension at import: without the machine's cache directory, under the home
    assert launch.environment["TRITON_CACHE_DIR"] == str(dry_run.local_dir / "triton")
    # the NVIDIA driver's cache of the kernels it compiles, under the home by default, is on the machine's disk too,
    # with the driver's largest limit: the default holds less than a trainer's kernels
    assert launch.environment["CUDA_CACHE_PATH"] == str(dry_run.local_dir / "cuda")
    assert int(launch.environment["CUDA_CACHE_MAXSIZE"]) == 4 * 1024**3
    assert launch.environment["HF_HOME"] == str(dry_run.work_dir / "cache" / "hf")


def test_the_workers_of_a_launch_get_every_variable_its_tools_read(composed):
    runtime_environment = composed.value(RUNTIME_ENV_KEY)
    env_classes = composed.tool_env_classes()
    read_variables = {
        *LAUNCH_VARIABLES,
        *(variable for env_class in env_classes for variable in FAMILY_VARIABLES[env_class]),
    }

    # Ray takes strings alone: a number Hydra read as one would end the launch
    assert all(isinstance(value, str) for value in runtime_environment.values())
    assert read_variables <= runtime_environment.keys()


def test_the_workers_of_a_launch_encode_on_the_calling_thread(composed):
    # verl turns a tokenizer's thread pool on in every worker of a launch whose runtime environment does not name it
    assert composed.value(RUNTIME_ENV_KEY)["TOKENIZERS_PARALLELISM"] == "false"


def test_the_sizes_of_a_launch_are_those_the_recipes_own_arithmetic_gave(composed):
    run = composed.value("trainer.experiment_name")  # the dry run names a launch's run <recipe>.<GPU>
    sizes = TRAINING_SIZES.get(run) or EVALUATION_SIZES[run.rpartition(".")[2]]

    assert tuple(composed.value(key) for key in SIZE_KEYS) == sizes


def test_other_machines_and_sessions_a_replica_give_a_launch_their_sizes(tmp_path):
    recipe = "examples/models/qwen3_8_27b_pear/train/run_stage3.sh"
    variables = {"TRAINER_NNODES": "4", "ROLLOUT_NNODES": "3", "TP": "4", "SESSIONS_PER_REPLICA": "40"}

    composed = ComposedConfig(RecipeDryRun(REPO, tmp_path).run(recipe, "gb300", variables).composed_config)

    # three rollout machines of 4 GPUs at TP4: three replicas of 40 sessions each
    assert tuple(composed.value(key) for key in SIZE_KEYS) == (80, 30, 120)


def test_the_episodes_of_a_launch_are_files_of_the_tree_each_played_by_one_of_its_tools(composed):
    env_classes = composed.tool_env_classes()
    episode_files = [file for key in EPISODE_FILE_KEYS for file in composed.value(key)]

    assert episode_files
    assert {json.loads(Path(file).read_text())["env_class"] for file in episode_files} <= set(env_classes)


@pytest.mark.parametrize("recipe", APPS_VARIABLES, ids=lambda recipe: Path(recipe).stem)
def test_the_workers_get_the_apps_of_a_server_only_when_the_recipe_was_given_them(tmp_path, recipe):
    variable, own_number = APPS_VARIABLES[recipe]
    gpu = RecipeDryRun.GPUS[0]

    # each launch from a sandbox of its own
    plain = ComposedConfig(RecipeDryRun(REPO, tmp_path / "plain").run(recipe, gpu).composed_config)
    given = ComposedConfig(RecipeDryRun(REPO, tmp_path / "given").run(recipe, gpu, {variable: "20"}).composed_config)

    # the tool config, which the servers and the trainer's session cap read, has the number in both launches
    assert variable not in plain.value(RUNTIME_ENV_KEY)
    assert [config["apps_per_gpu"] for config in plain.tool_configs() if "apps_per_gpu" in config] == [own_number]
    assert given.value(RUNTIME_ENV_KEY)[variable] == "20"
    assert [config["apps_per_gpu"] for config in given.tool_configs() if "apps_per_gpu" in config] == ["20"]


def test_two_dumps_of_a_recipe_are_the_same_files(tmp_path):
    # each from a sandbox of its own: a dump holds none of its sandbox's paths, so two trees' dumps differ where the
    # trees do
    recipe, gpu = LAUNCHES[0]
    for name in ("first", "second"):
        RecipeDump(argparse.Namespace(root=str(REPO), out=str(tmp_path / name), recipe=[recipe], gpu=[gpu])).run()

    directory = f"{Path(recipe).stem}.{gpu}"
    files = sorted(path.name for path in (tmp_path / "first" / directory).iterdir())
    _, mismatched_files, missing_files = filecmp.cmpfiles(
        tmp_path / "first" / directory, tmp_path / "second" / directory, files, shallow=False
    )
    assert files == ["config.txt", "environment.txt", "overrides.txt", "panel.txt"]
    assert (mismatched_files, missing_files) == ([], [])
