from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

logger = get_logger(__name__, default_level="INFO")  # the plan's placements belong to the server's log


class MemoryPlan:
    """Where an engine's process places a Qwen3.5 model's KV cache and weights, as its environment names it: placement
    only, no value a kernel computes changes."""

    # comma-separated options, inherited by the engine's processes: groups (KV cache groups of the least padding), embed
    # (the embedding table in pinned host memory, read through a UVA view), vision_host (the vision encoder there too)
    ENV = "PEAR_DEPLOY_MEMORY_PLAN"
    OPTIONS = ("groups", "embed", "vision_host")
    # a group size's cost: a padding full-attention layer holds a whole window, a padding sliding-window or Mamba layer
    # a few blocks, and every group adds one (its block table and manager)
    FULL_ATTENTION_PADDING_COST = 1000
    OTHER_PADDING_COST = 1
    VISION_HOST_MIN_ELEMENTS = 1 << 16  # the vision encoder's parameters that move: the small ones stay
    GIB = 1 << 30  # bytes per GiB, for the log
    host_tensors: list[torch.Tensor] = []  # the pinned host copies the UVA views read, kept alive

    @classmethod
    def install(cls) -> None:
        """Patch vLLM in this process with the plan's options."""
        options = cls.options()
        if not options:
            return

        if "groups" in options:
            cls.install_groups()
        if "embed" in options or "vision_host" in options:
            cls.install_host_weights("embed" in options, "vision_host" in options)
        cls.install_group_log()

    @classmethod
    def options(cls) -> set[str]:
        """The plan's options in this process's environment."""
        options = {option for option in os.environ.get(cls.ENV, "").split(",") if option}
        unknown_options = options - set(cls.OPTIONS)
        if unknown_options:
            raise ValueError(f"{cls.ENV}: {sorted(unknown_options)} are not of the memory plan's options {cls.OPTIONS}")

        return options

    @classmethod
    def install_groups(cls) -> None:
        """Group vLLM's KV cache layers by the group size of the least padding cost, in place of the size of the
        smallest bucket of same-type layers."""

        def group_layers(kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
            # vLLM's buckets of mergeable same-type layers (kv_cache_utils._get_kv_cache_groups_uniform_page_size)
            names_of_spec = defaultdict(list)
            for name, spec in kv_cache_spec.items():
                names_of_spec[spec].append(name)
            buckets, bucket_specs = [], []
            for spec, names in names_of_spec.items():
                for bucket, specs in zip(buckets, bucket_specs):
                    try:
                        type(specs[0]).merge([*specs, spec])
                    except (AssertionError, ValueError):
                        continue
                    bucket.extend(names)
                    specs.append(spec)
                    break
                else:
                    buckets.append(list(names))
                    bucket_specs.append([spec])

            def cost(group_size: int) -> int:
                padding_cost = 0
                for names, specs in zip(buckets, bucket_specs):
                    padding = -(-len(names) // group_size) * group_size - len(names)
                    is_full_attention = isinstance(specs[0], FullAttentionSpec) and not isinstance(specs[0], MambaSpec)
                    padding_cost += padding * (
                        cls.FULL_ATTENTION_PADDING_COST if is_full_attention else cls.OTHER_PADDING_COST
                    )
                group_count = sum(-(-len(names) // group_size) for names in buckets)

                return padding_cost + group_count

            group_sizes = range(1, max(len(names) for names in buckets) + 1)
            group_size = min(group_sizes, key=lambda size: (cost(size), -size))

            grouped_layers = []
            for names in buckets:
                group_count = -(-len(names) // group_size)
                for index in range(group_count):
                    grouped_layers.append(names[index::group_count])
            logger.info(
                "memory plan: KV cache groups of %d layers (buckets of %s layers)",
                group_size,
                [len(names) for names in buckets],
            )

            return kv_cache_utils.create_kv_cache_group_specs(kv_cache_spec, grouped_layers)

        kv_cache_utils._get_kv_cache_groups_uniform_page_size = group_layers

    @classmethod
    def install_host_weights(cls, is_embed_on_host: bool, is_vision_on_host: bool) -> None:
        """Move the target's embedding table, its vision encoder or both into pinned host memory after vLLM loads the
        model."""
        vllm_load_model = GPUModelRunner.load_model

        def load_model(self, *args: Any, **kwargs: Any) -> None:
            vllm_load_model(self, *args, **kwargs)
            allocated_bytes = torch.cuda.memory_allocated()

            language_model = (
                self.model.get_language_model() if hasattr(self.model, "get_language_model") else self.model
            )
            if is_embed_on_host:
                cls.to_host(language_model.model.embed_tokens.weight)
            # the encoder runs in prefills with images only
            vision_encoder = getattr(self.model, "visual", None)
            if is_vision_on_host and vision_encoder is not None:
                for parameter in vision_encoder.parameters():
                    if parameter.numel() >= cls.VISION_HOST_MIN_ELEMENTS:
                        cls.to_host(parameter)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            freed_bytes = allocated_bytes - torch.cuda.memory_allocated()
            logger.info("memory plan: %.2f GiB of weights moved to pinned host memory", freed_bytes / cls.GIB)
            # what vLLM's memory profile counts as the model's
            self.model_memory_usage -= freed_bytes

        GPUModelRunner.load_model = load_model

    @classmethod
    def install_group_log(cls) -> None:
        """Log each KV cache group's layers, block, page and memory at the model's length."""
        vllm_get_kv_cache_groups = kv_cache_utils.get_kv_cache_groups

        def get_kv_cache_groups(
            vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
        ) -> list[KVCacheGroupSpec]:
            groups = vllm_get_kv_cache_groups(vllm_config, kv_cache_spec)

            total_bytes = 0
            for index, group in enumerate(groups):
                spec = group.kv_cache_spec
                group_bytes = spec.max_memory_usage_bytes(vllm_config) * len(group.layer_names)
                total_bytes += group_bytes
                logger.info(
                    "memory plan: KV cache group %d: %s, block %d, page %d, %d layers, %.3f GiB",
                    index,
                    type(spec).__name__,
                    spec.block_size,
                    spec.page_size_bytes,
                    len(group.layer_names),
                    group_bytes / cls.GIB,
                )
            logger.info("memory plan: %d KV cache groups, %.3f GiB", len(groups), total_bytes / cls.GIB)

            return groups

        kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups

    @classmethod
    def to_host(cls, parameter: torch.nn.Parameter) -> None:
        """Move a parameter's storage into pinned host memory, the parameter a UVA view of it."""
        host_tensor = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu", pin_memory=True)
        host_tensor.copy_(parameter.data)
        cls.host_tensors.append(host_tensor)
        parameter.data = get_accelerator_view_from_cpu_tensor(host_tensor)
