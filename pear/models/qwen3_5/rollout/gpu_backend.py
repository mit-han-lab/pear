import torch

from pear.models.qwen3_5.gpu_backend import VllmGpuBackend


class RolloutGpuBackend(VllmGpuBackend):
    """The vLLM classes and patches of one GPU generation for the replicas of a Qwen3.5 model, on vLLM's V2 model
    runner."""

    # the Gated DeltaNet states the kernels read and write: the conv states in bf16, the recurrent states in fp32
    STATE_DTYPES = (torch.bfloat16, torch.float32)

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
        if vllm_config.speculative_config is not None:
            return False, "its kernels serve the steps without a drafter on vLLM's V2 model runner"

        state_dtypes = cls.model_cls.get_mamba_state_dtype_from_config(vllm_config)
        if state_dtypes != cls.STATE_DTYPES:
            return False, f"its Gated DeltaNet kernels take states of {cls.STATE_DTYPES}, not {state_dtypes}"

        return True, None
