from typing import ClassVar

import torch
import vllm.envs

from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.models.qwen3_5.rollout.rollout_kernels import Qwen3_5RolloutKernels


class RolloutGpuBackend(VllmGpuBackend):
    """The vLLM classes and patches of one GPU generation for the replicas of a Qwen3.5 model, on vLLM's V2 model
    runner."""

    # the Gated DeltaNet states the kernels read and write: the conv states in bf16, the recurrent states in fp32
    STATE_DTYPES = (torch.bfloat16, torch.float32)
    # the KV cache the attention, the Gated DeltaNet's replay and the KV cache updates are validated on
    KV_CACHE_DTYPE = "bfloat16"
    # the MTP drafter's class (Qwen3_5RolloutKernels.DRAFT_ARCHITECTURE), vLLM's when None
    draft_model_cls: ClassVar[type | None] = None

    @classmethod
    def can_implement(cls, vllm_config) -> tuple[bool, str | None]:
        is_implementable, reason = super().can_implement(vllm_config)
        if not is_implementable:
            return is_implementable, reason

        parallel_config = vllm_config.parallel_config
        # the decoder layers all-reduce in their norms and the model returns the final norm's output
        if parallel_config.pipeline_parallel_size != 1:
            return False, f"its model runs on one pipeline stage, not {parallel_config.pipeline_parallel_size}"
        # the row-parallel layers leave their reduction to the next norm, which LoRA's layers do not
        if vllm_config.lora_config is not None:
            return False, "its decoder layers run without LoRA adapters"
        # the logits gather and the fused samplers take a step's whole batch on every rank
        if parallel_config.enable_batch_sharded_sampling:
            return False, "its sampling takes a step's whole batch on every rank, not batch-sharded sampling"
        # a drafter's method (a custom one by its own name, a built-in one by vLLM's): the family's MTP method, drafting
        # on the target's M-RoPE positions (vLLM's own MTP drafts on 1-D positions)
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            drafter = speculative_config.custom_method or speculative_config.method
            method = Qwen3_5RolloutKernels.SPECULATIVE_METHOD
            if drafter != method:
                return False, f"its kernels draft with the speculative method {method}, not {drafter}"
            # the verify steps' Gated DeltaNet rows run on the backend's core alone, which a decode kernel other than
            # vLLM's fused one hands to vLLM's layer (RolloutGDNAttention.computes_core)
            gdn_decode_kernel = vllm.envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
            if gdn_decode_kernel != "cuda":
                return False, (
                    "its Gated DeltaNet kernels verify a speculative_config's drafts under "
                    f"VLLM_GDN_DECODE_KERNEL=cuda, not {gdn_decode_kernel}"
                )

        state_dtypes = cls.model_cls.get_mamba_state_dtype_from_config(vllm_config)
        if state_dtypes != cls.STATE_DTYPES:
            return False, f"its Gated DeltaNet kernels take states of {cls.STATE_DTYPES}, not {state_dtypes}"

        # vLLM's auto KV cache takes the model's dtype
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype == "auto":
            cache_dtype = str(vllm_config.model_config.dtype).removeprefix("torch.")
        if cache_dtype != cls.KV_CACHE_DTYPE:
            return False, f"its attention kernels take a KV cache of {cls.KV_CACHE_DTYPE}, not {cache_dtype}"

        return True, None
