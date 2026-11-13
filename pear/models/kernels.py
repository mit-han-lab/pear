from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only
    import torch
    from vllm.config import VllmConfig

    from verl.workers.config import FSDPEngineConfig


class TrainingKernels(ABC):
    """A model family's training kernels for the module the FSDP engine built, named by fsdp_config.kernels as a
    class path: what the family installs on any GPU and, with fsdp_config.use_gpu_backend, a GPU generation's
    backend."""

    @classmethod
    @abstractmethod
    def check_config(cls, engine_config: FSDPEngineConfig) -> None:
        """Raise unless the engine config is one the GPU backend computes under, before the module is built."""

    @classmethod
    @abstractmethod
    def install(
        cls, module: torch.nn.Module, fp32_residual: bool, use_fused_kernels: bool, fused_kernels_backend: str | None
    ) -> None:
        """Apply what the family installs on any GPU to the module verl built and patched, before FSDP wraps it."""

    @classmethod
    @abstractmethod
    def check_model(
        cls,
        module: torch.nn.Module,
        use_remove_padding: bool,
        use_fused_kernels: bool,
        fused_kernels_backend: str | None,
    ) -> None:
        """Raise unless the module runs the paths the GPU backend replaces."""

    @classmethod
    @abstractmethod
    def install_gpu_backend(cls, module: torch.nn.Module) -> None:
        """Patch the module with the backend of the process's GPU after install(), before FSDP wraps it."""

    @classmethod
    @abstractmethod
    def install_checkpointing(cls, module: torch.nn.Module) -> None:
        """Extend the gradient checkpoints the module has enabled with what the GPU backend needs in a recompute."""

    @classmethod
    @abstractmethod
    def run_checked(cls, run: Callable[[], Any], before_rerun: Callable[[], None]) -> Any:
        """The result of run() on the GPU backend's fast path, or after before_rerun() of a second run() off it."""


class RolloutKernels(ABC):
    """A model family's vLLM classes and patches for its replicas, named by rollout.kernels as a class path."""

    # the class path, set by the server for the worker processes it starts, which inherit its environment
    ENV = "PEAR_ROLLOUT_KERNELS"

    @classmethod
    def export(cls, path: str) -> type[RolloutKernels]:
        """Name the rollout kernels to the worker processes this process starts afterwards, and return them."""
        os.environ[cls.ENV] = path

        return cls.from_env()

    @classmethod
    def from_env(cls) -> type[RolloutKernels] | None:
        """The rollout kernels the server named, None when it serves vLLM's own model."""
        path = os.environ.get(cls.ENV)
        if not path:
            return None
        module_path, _, class_name = path.rpartition(".")

        # not verl's class loader: importing verl here runs its external modules, and pear.rollout.server among them
        # imports this module back before its class exists
        return getattr(importlib.import_module(module_path), class_name)

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether the server that started this worker process named these rollout kernels."""
        return cls.from_env() is cls

    @classmethod
    @abstractmethod
    def configure_server(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """The settings of the process that builds the engine config (the server), before it builds it."""

    @classmethod
    @abstractmethod
    def install_worker(cls, vllm_config: VllmConfig) -> None:
        """Register the rollout kernels' classes and apply their patches in a worker process, before it builds its
        model."""

    @classmethod
    @abstractmethod
    def check_model(cls, model: torch.nn.Module, vllm_config: VllmConfig) -> None:
        """Raise unless a worker built the rollout kernels' model."""


class DeployKernels(ABC):
    """A model family's vLLM classes and patches for a server of one session, named by kernels as a class path."""

    # the class path, set by the server for the engine processes it starts, which inherit its environment
    ENV = "PEAR_DEPLOY_KERNELS"

    @classmethod
    def export(cls, path: str) -> type[DeployKernels]:
        """Name the deploy kernels to the engine processes this process starts afterwards, and return them."""
        os.environ[cls.ENV] = path

        return cls.from_env()

    @classmethod
    def from_env(cls) -> type[DeployKernels] | None:
        """The deploy kernels the server named, None when it serves vLLM's own model."""
        path = os.environ.get(cls.ENV)
        if not path:
            return None
        module_path, _, class_name = path.rpartition(".")

        # not verl's class loader: the server's environment has no verl
        return getattr(importlib.import_module(module_path), class_name)

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether the server that started this engine process named these deploy kernels."""
        return cls.from_env() is cls

    @classmethod
    @abstractmethod
    def configure_server(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """The settings of the process that builds the engine config (the server), before it builds it."""

    @classmethod
    @abstractmethod
    def install_worker(cls, vllm_config: VllmConfig) -> None:
        """Register the deploy kernels' classes and apply their patches in an engine process, before it builds its
        model."""
