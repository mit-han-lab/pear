from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class TinkerConfig:
    base_model: str
    renderer: str
    episode_files: list[str]
    tool_configs: list[str]
    run_dir: str
    model_revision: str | None = None
    think_end: str | None = None
    context_length: int = 262144
    max_prompt_tokens: int = 4096
    response_budget: int = 160000
    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    objective: str = "stable_reinforce"
    loss_config: dict = field(default_factory=dict)
    topk: int = 32
    allow_head_continuations: bool = False
    normalize_advantages: bool = True
    group_size: int = 8
    batch_groups: int = 128
    max_sessions: int = 64
    prefetch_groups: int | None = None
    max_staleness: int = 4
    steps: int = 1000
    seed: int = 1
    shuffle: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    sampler_kind: str = "inflight"
    train_request_tokens: int = 262144
    train_inflight_requests: int = 2
    max_groups_per_step: int = 0
    save_every: int = 1
    keep_checkpoints: int = 2
    dump_training_batch: bool = False
    init_checkpoint: str | None = None
    eval_episode_files: list[str] = field(default_factory=list)
    eval_group_size: int = 1
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    ray_address: str | None = None
    ray_resources: dict[str, float] = field(default_factory=dict)
    ray_num_cpus: int | None = None
    simulator_env: dict[str, str] = field(default_factory=dict)
    image_processor_options: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str, overrides: list[str]) -> TinkerConfig:
        recipe, options = OmegaConf.load(path), OmegaConf.from_dotlist(overrides)
        if "loss_config" in options:
            # Loss options belong to the selected objective; an explicit {} must clear the recipe's options.
            recipe.loss_config = {}
        merged = OmegaConf.merge(recipe, options)
        config = cls(**OmegaConf.to_container(merged, resolve=True))
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("model_revision must be a full Hugging Face commit hash")
        for name in (
            "context_length",
            "max_prompt_tokens",
            "response_budget",
            "rank",
            "group_size",
            "batch_groups",
            "max_sessions",
            "steps",
            "train_request_tokens",
            "train_inflight_requests",
            "save_every",
            "keep_checkpoints",
            "eval_group_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("topk", "max_staleness", "max_groups_per_step", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.prefetch_groups is not None and (type(self.prefetch_groups) is not int or self.prefetch_groups < 0):
            raise ValueError("prefetch_groups must be a nonnegative integer or null")
        for name in ("learning_rate", "adam_eps", "temperature", "eval_temperature", "grad_clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not 0 < self.top_p <= 1 or not 0 < self.eval_top_p <= 1:
            raise ValueError("top_p and eval_top_p must be in (0, 1]")
        if self.max_prompt_tokens + self.response_budget > self.context_length:
            raise ValueError("the prompt and response budgets exceed context_length")
        if self.group_size < 2:
            raise ValueError("training needs group_size >= 2")
        if self.train_request_tokens < self.context_length:
            raise ValueError("train_request_tokens must accommodate a full context")
        if self.objective not in ("stable_reinforce", "pear", "importance_sampling", "ppo"):
            raise ValueError("objective must be stable_reinforce, pear, importance_sampling, or ppo")
        if not isinstance(self.loss_config, dict):
            raise ValueError("loss_config must be a mapping")
        if self.objective == "pear" and (self.topk == 0 or self.top_p != 1 or self.temperature != 1):
            raise ValueError("PEAR score centering needs topk > 0, temperature=1 and top_p=1")
        if self.objective == "pear" and self.loss_config:
            raise ValueError("loss_config applies only to Tinker's built-in objectives")
        if self.objective == "stable_reinforce":
            if self.loss_config.keys() - {"kl_coeff", "kl_estimator"}:
                raise ValueError("stable_reinforce supports kl_coeff and kl_estimator")
            coefficient = self.loss_config.get("kl_coeff", 0.0)
            if not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("kl_coeff must be nonnegative and finite")
            if self.loss_config.get("kl_estimator", "k1") not in ("k1", "k3"):
                raise ValueError("kl_estimator must be k1 or k3")
        if self.sampler_kind not in ("snapshot", "inflight"):
            raise ValueError("sampler_kind must be snapshot or inflight")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.simulator_env.items()):
            raise ValueError("simulator_env must map environment variable names to strings")
        if not self.base_model or not self.renderer or not self.episode_files or not self.tool_configs:
            raise ValueError("base_model, renderer, episode_files and tool_configs are required")
        for file in [*self.episode_files, *self.eval_episode_files, *self.tool_configs]:
            if not Path(file).is_file():
                raise ValueError(f"missing file: {file}")

    def signature(self) -> str:
        """Recipe identity for resume; run length, placement and evaluation can change."""
        config = asdict(self)
        for key in (
            "run_dir",
            "steps",
            "ray_address",
            "ray_resources",
            "ray_num_cpus",
            "simulator_env",
            "max_sessions",
            "prefetch_groups",
            "train_inflight_requests",
            "save_every",
            "keep_checkpoints",
            "dump_training_batch",
            "allow_head_continuations",
            "eval_episode_files",
            "eval_group_size",
            "eval_temperature",
            "eval_top_p",
        ):
            config.pop(key)
        config["files"] = {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [*self.episode_files, *self.tool_configs]
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
