from __future__ import annotations

import json
import os
import pkgutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

from pear.agent_loop.tasks import load_tasks
from pear.envs.session import EnvironmentSession


def resource_plan(tool_configs, resources, available, nodes):
    """Check aggregate capacity and per-server placement before creating any simulator actors."""
    required, plan = {}, {}
    for name, config in tool_configs.items():
        env_class = pkgutil.resolve_name(name)
        count = env_class.server_n(config, resources)
        options = env_class.SERVER_RESOURCES
        demand = {"CPU": options.get("num_cpus", 1), "GPU": options.get("num_gpus", 0), **options.get("resources", {})}
        demand = {key: value for key, value in demand.items() if value}
        if count < 1 or not any(all(node.get(key, 0) >= value for key, value in demand.items()) for node in nodes):
            raise ValueError(f"no node can place a {name} server requiring {demand}")
        for key, value in demand.items():
            required[key] = required.get(key, 0) + count * value
        plan[name] = {"servers": count, "per_server": demand, "session_cap": env_class.session_cap(config)}
    missing = {
        key: {"required": value, "available": available.get(key, 0)}
        for key, value in required.items()
        if available.get(key, 0) < value
    }
    if missing:
        raise ValueError(f"insufficient available simulator resources: {missing}")
    return {"required": required, "available": {key: available.get(key, 0) for key in required}, "families": plan}


def make_renderer(config):
    return _renderer(config.base_model, config.renderer, json.dumps(config.image_processor_options, sort_keys=True))


@lru_cache
def _renderer(model: str, name: str, options: str):
    from tinker_cookbook.image_processing_utils import get_image_processor
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model)
    processor = deepcopy(get_image_processor(model))
    for key, value in json.loads(options).items():
        if processor is None or not hasattr(processor, key):
            raise ValueError(f"image processor has no option {key!r}")
        setattr(processor, key, value)
    return get_renderer(name, tokenizer, processor, model_name=model)


class PearDataset:
    """Validated tasks for the rollout pool and evaluation."""

    def __init__(self, config, tool_configs: dict[str, dict], evaluation: bool = False) -> None:
        paths = config.eval_episode_files if evaluation else config.episode_files
        self.tasks = load_tasks(paths)
        if not self.tasks:
            raise ValueError("the episode files contain no tasks")
        for task in self.tasks:
            if task.env_class not in tool_configs:
                raise ValueError(f"no environment configuration for {task.env_class}")


class SimulatorRuntime:
    """The existing PEAR simulator pools, with no trainer or local model workers."""

    def __init__(self, config) -> None:
        self.config = config
        self.pools = []
        self.sessions = {}
        self.caps = {}
        self.connected = False
        log_dir = Path(config.run_dir).resolve() / "sim"
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PEAR_LOG_DIR"] = str(log_dir)
        self.tool_configs = {}
        for path in config.tool_configs:
            for tool in OmegaConf.to_container(OmegaConf.load(path), resolve=True)["tools"]:
                entry = dict(tool["config"])
                name = entry["env_class"]
                if name in self.tool_configs:
                    raise ValueError(f"duplicate environment configuration: {name}")
                self.tool_configs[name] = entry

    def __enter__(self):
        import ray

        from pear.sim.server import ServerPool

        if ray.is_initialized():
            raise RuntimeError("start the Tinker coordinator in its own process")
        kwargs = {"address": self.config.ray_address, "namespace": ServerPool.NAMESPACE}
        if self.config.simulator_env:
            kwargs["runtime_env"] = {"env_vars": self.config.simulator_env}
        if self.config.ray_address is None:
            kwargs.update(resources=self.config.ray_resources, num_cpus=self.config.ray_num_cpus)
        ray.init(**kwargs)
        self.connected = True
        try:
            self.resource_plan = resource_plan(
                self.tool_configs,
                ray.cluster_resources(),
                ray.available_resources(),
                [node["Resources"] for node in ray.nodes() if node["Alive"]],
            )
            for name, config in self.tool_configs.items():
                env_class = pkgutil.resolve_name(name)
                self.pools.append(ServerPool.start(env_class, config))
                self.sessions[name] = EnvironmentSession(config)
                self.caps[name] = env_class.session_cap(config) or self.config.max_sessions
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exception):
        import ray

        for pool in self.pools:
            ray.kill(pool, no_restart=True)
        self.pools.clear()
        if self.connected:
            ray.shutdown()
            self.connected = False
