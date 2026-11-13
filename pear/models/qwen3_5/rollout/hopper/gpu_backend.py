from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum, MambaAttentionBackendEnum, register_backend

from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.models.qwen3_5.gpu_generation import GpuGeneration
from pear.models.qwen3_5.rollout.hopper.all_reduce import DecodeAllReduce
from pear.models.qwen3_5.rollout.hopper.attention import HopperFlashAttentionBackend
from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperMixedStepGraphs
from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
from pear.models.qwen3_5.rollout.hopper.gdn import HopperGDNAttention, HopperGDNBackend
from pear.models.qwen3_5.rollout.hopper.model import HopperQwen3_5ForConditionalGeneration
from pear.models.qwen3_5.rollout.hopper.sampler import HopperSampler
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.mamba_copy import MambaCopy
from pear.models.qwen3_5.rollout.prompt_tail_copy import PromptTailCopy


class HopperRolloutGpuBackend(VllmGpuBackend):
    """vLLM classes and patches for the replicas of a Qwen3.5 model on Hopper at TP4: the steps with prefill rows as
    FULL graphs on MixedStepGDNCore with split decode attention, the pure decode steps on fused decode kernels."""

    generation = GpuGeneration.HOPPER
    tensor_parallel_size = DecodeAllReduce.TP_SIZE  # 6 query heads per KV head on a rank, 4-rank reductions
    model_cls = HopperQwen3_5ForConditionalGeneration
    # the Inductor settings make every fresh compile produce the same kernels (verl compiles afresh at every start);
    # FlashInfer's fused all-reduce + norm then covers the mixed steps of up to 1024 tokens (10 MB)
    INDUCTOR_COMPILE_CONFIG = {
        "deterministic": True,
        "benchmark_epilogue_fusion": False,
        "benchmark_fusion": False,
        "combo_kernels": True,
        "benchmark_combo_kernel": False,
        "test_configs.force_filter_reduction_configs": True,
    }
    ALL_REDUCE_FUSION_MAX_SIZE_MB = 10  # the largest step FlashInfer's fused all-reduce + norm takes

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        compilation_config.setdefault("inductor_compile_config", {}).update(cls.INDUCTOR_COMPILE_CONFIG)
        compilation_config.setdefault("pass_config", {})["fi_allreduce_fusion_max_size_mb"] = (
            cls.ALL_REDUCE_FUSION_MAX_SIZE_MB
        )
        HopperMixedStepGraphs.configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def install(cls, vllm_config) -> None:
        HopperExtension.load()

        # through vLLM's extension points (the model by PEAR's vLLM plugin, Qwen3_5RolloutKernels.register): the GDN layer,
        # its metadata builder and the attention backend
        QwenGatedDeltaNetAttention.register_oot(HopperGDNAttention)
        register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)(HopperGDNBackend)
        register_backend(AttentionBackendEnum.FLASH_ATTN)(HopperFlashAttentionBackend)

        # patches of what vLLM offers no extension point for
        HopperMixedStepGraphs.install()
        DecodeAllReduce.install()
        HopperSampler.install()
        MambaCopy.install()
        PromptTailCopy.install(vllm_config)
        StepInputArrays.install()
