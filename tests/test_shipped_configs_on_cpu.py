import inspect
import json
import pkgutil
import sys
import types
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from pear.agent_loop.dataset import EpisodeDataset
from pear.eval.general import BenchmarkRun, DatasetDirectory, GeneralEvaluation
from pear.sim.env import BaseEnv
from repository import REPO
from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.tools.base_tool import BaseTool

ENVS = REPO / "pear" / "envs"
FAMILIES = sorted(path.parent for path in ENVS.glob("*/env.py"))  # a family: a directory with its env class
# the tool config of each family, and the recipes' own, which combine families
TOOL_CONFIGS = sorted([*ENVS.glob("*/configs/tool_config.yaml"), *REPO.glob("examples/**/*_tool_config.yaml")])
# the families' episode files: the JSON files of their configs that name an env class
EPISODE_FILES = sorted(path for path in ENVS.glob("*/configs/*.json") if '"env_class"' in path.read_text())
AGENT_LOOP_CONFIG = REPO / "pear" / "agent_loop" / "config.yaml"
BASE_CONFIG = REPO / "pear" / "configs" / "base.yaml"
STATION_CONFIG = REPO / "pear" / "configs" / "station.yaml"  # what every robot station starts from
ROBOTS = REPO / "pear" / "deploy" / "station" / "control" / "robots"
ROBOT_CONFIGS = sorted(ROBOTS.glob("*/config.yaml"))  # each robot's defaults, read over the station's
# what only a station knows of the keys every station has: its models and its run directory
STATION_OWN_KEYS = {"default_model", "models", "run_dir"}


class ShippedConfigs:
    """What the tests read of the tool configs and the episode files."""

    @staticmethod
    def tools(tool_config: Path) -> list[dict]:
        """A tool config's tools, its environment variables unresolved: the classes it names are none of them."""
        return OmegaConf.to_container(OmegaConf.load(tool_config), resolve=False)["tools"]

    @classmethod
    def tool_env_classes(cls) -> set[str]:
        """The env classes the tool configs name, by their paths."""
        return {tool["config"]["env_class"] for tool_config in TOOL_CONFIGS for tool in cls.tools(tool_config)}

    @staticmethod
    def episodes(episode_file: Path) -> list[dict]:
        """The env args of every episode of a file, as the dataset makes its rows of them."""
        dataset = object.__new__(EpisodeDataset)
        dataset.data_files, dataset.max_samples = [str(episode_file)], -1

        dataset._read_files_and_tokenize()

        return [json.loads(row["extra_info"]["env"]) for row in dataset.dataframe]


@pytest.fixture
def evalscope_tasks(monkeypatch) -> list[dict]:
    # an evalscope that keeps the tasks it is given and runs none
    tasks = []
    evalscope = types.ModuleType("evalscope")
    evalscope.TaskConfig, evalscope.run_task = dict, tasks.append
    monkeypatch.setitem(sys.modules, "evalscope", evalscope)
    # a benchmark points the libraries at its dataset directory: the test leaves the variables as it found them
    for variable in DatasetDirectory.DIRECTORY_BY_VARIABLE:
        monkeypatch.setenv(variable, "elsewhere")

    return tasks


def test_every_family_ships_its_tool_config_and_an_episode_file():
    assert FAMILIES
    for family in FAMILIES:
        assert family / "configs" / "tool_config.yaml" in TOOL_CONFIGS
        assert [episode_file for episode_file in EPISODE_FILES if episode_file.parents[1] == family]


@pytest.mark.parametrize("tool_config", TOOL_CONFIGS, ids=lambda path: str(path.relative_to(REPO)))
def test_a_tool_config_names_a_tool_and_the_env_class_of_a_family(tool_config):
    tools = ShippedConfigs.tools(tool_config)

    assert tools
    for tool in tools:
        env_class = pkgutil.resolve_name(tool["config"]["env_class"])
        assert issubclass(pkgutil.resolve_name(tool["class_name"]), BaseTool)
        assert issubclass(env_class, BaseEnv)
        # its kind's client, which the tool makes by this path
        pkgutil.resolve_name(env_class.CLIENT_CLASS_PATH)


@pytest.mark.parametrize("episode_file", EPISODE_FILES, ids=lambda path: str(path.relative_to(ENVS)))
def test_every_episode_of_a_file_is_arguments_of_its_env_class_which_a_tool_config_names(episode_file):
    env_class_path = json.loads(episode_file.read_text())["env_class"]
    env_class = pkgutil.resolve_name(env_class_path)
    constructor = inspect.signature(env_class.__init__)

    episodes = ShippedConfigs.episodes(episode_file)

    assert env_class_path in ShippedConfigs.tool_env_classes()
    assert episodes
    for env_args in episodes:
        # as the simulator process calls it, without the simulator: the arguments are the constructor's own
        constructor.bind(None, **env_args)
        assert isinstance(env_class.task_name(env_args), str)


def test_the_agent_loop_config_holds_the_loop_the_trainer_config_names():
    loops = yaml.safe_load(AGENT_LOOP_CONFIG.read_text())
    agent = yaml.safe_load(BASE_CONFIG.read_text())["actor_rollout_ref"]["rollout"]["agent"]

    assert all(issubclass(pkgutil.resolve_name(loop["_target_"]), AgentLoopBase) for loop in loops)
    assert agent["default_agent_loop"] in [loop["name"] for loop in loops]
    # by a path from the repository, a recipe's working directory
    assert REPO / agent["agent_loop_config_path"] == AGENT_LOOP_CONFIG


def test_every_benchmark_of_the_general_evaluation_s_config_makes_an_evalscope_task(
    evalscope_tasks, tmp_path, monkeypatch
):
    run_arguments = ["--api-url", "http://localhost", "--model", "model", "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", ["general", "run", *run_arguments, "--dataset-dir", str(tmp_path / "datasets")])
    run = GeneralEvaluation.from_command_line().command
    benchmarks = run.config["benchmarks"]

    statuses = [run.benchmark(name)[2] for name in benchmarks]

    assert benchmarks
    assert statuses == [BenchmarkRun.DONE_STATUS] * len(benchmarks)
    assert [task["datasets"] for task in evalscope_tasks] == [
        [benchmark["dataset"]] for benchmark in benchmarks.values()
    ]


def test_the_station_s_config_leaves_a_station_its_robot_its_models_its_port_and_its_run_directory():
    config = OmegaConf.load(STATION_CONFIG)

    assert OmegaConf.missing_keys(config) == {"robot_class", "robot", "port", *STATION_OWN_KEYS}
    assert len(ROBOT_CONFIGS) > 1


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS, ids=[path.parent.name for path in ROBOT_CONFIGS])
def test_a_robot_s_config_names_a_class_of_its_package_and_leaves_its_machine_s_values_missing(robot_config):
    config = OmegaConf.merge(OmegaConf.load(STATION_CONFIG), OmegaConf.load(robot_config))
    module_path = REPO.joinpath(*config.robot_class.split(".")[:-1]).with_suffix(".py")
    # over the station's config, a robot's leaves what a station knows and its own machine's devices and calibration
    machine_keys = OmegaConf.missing_keys(config) - STATION_OWN_KEYS

    assert module_path.is_file() and robot_config.parent in module_path.parents
    assert machine_keys and all(key.startswith("robot.") for key in machine_keys)
