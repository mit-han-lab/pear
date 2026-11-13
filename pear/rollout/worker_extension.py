from __future__ import annotations

import gc

import torch

from pear.models.kernels import RolloutKernels
from pear.rollout.vllm_patches import Fp32Modules
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class PearVLLMWorkerExtension(vLLMColocateWorkerExtension):
    """verl's vLLM worker extension with the weights kept in fp32, a model family's rollout kernels and the checks of
    a sync."""

    ERROR_NAME_LIMIT = 4  # a failed check's error names at most this many of the parameters it found

    def __new__(cls, **kwargs) -> PearVLLMWorkerExtension:
        instance = super().__new__(cls, **kwargs)
        # weights kept in fp32 (rollout.fp32_modules, exported by PearVLLMServer), cast as the model is built
        fp32_modules = Fp32Modules.from_env()
        if fp32_modules is not None:
            fp32_modules.install()
        # the model family's vLLM classes and patches (rollout.kernels, named by PearVLLMServer)
        kernels = RolloutKernels.from_env()
        if kernels is not None:
            kernels.install_worker(kwargs.get("vllm_config"))

        return instance

    def check_kernels(self) -> None:
        """Raise unless this worker built the rollout kernels' model."""
        RolloutKernels.from_env().check_model(self.model_runner.get_model(), self.model_runner.vllm_config)

    def check_fp32_modules(self, patterns: list[str]) -> None:
        """Raise unless this worker keeps in fp32 the weights of the fp32 modules (``patterns``) and no other."""
        # the sender leaves those in fp32 and casts every other weight to the replica's dtype
        dtype = self.model_runner.vllm_config.model_config.dtype
        fp32_modules = Fp32Modules(patterns)
        unlisted_names, unconverted_weights = [], []
        for name, param in self.model_runner.model.named_parameters():
            if not param.is_floating_point():
                continue
            if fp32_modules.matches(name):
                if param.dtype != torch.float32:
                    unconverted_weights.append(f"{name} in {param.dtype}")
            elif param.dtype != dtype:
                unlisted_names.append(name)

        # the sender's cast would round them
        if unlisted_names:
            raise ValueError(
                f"{len(unlisted_names)} parameters wider than {dtype} not in rollout.fp32_modules={list(patterns)}: "
                f"{unlisted_names[: self.ERROR_NAME_LIMIT]}"
            )
        # the cast at the model's construction did not reach them: this worker would round what the sender sends
        if unconverted_weights:
            raise ValueError(
                f"{len(unconverted_weights)} parameters of rollout.fp32_modules={list(patterns)} not in fp32: "
                f"{unconverted_weights[: self.ERROR_NAME_LIMIT]}"
            )

    def release_cached_memory(self) -> None:
        """Return the caching allocator's unused blocks before a weight sync."""
        # the receiver on this GPU needs the room for its buckets and its merge buffer
        gc.collect()
        torch.cuda.empty_cache()
