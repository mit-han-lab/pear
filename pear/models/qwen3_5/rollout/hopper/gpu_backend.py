from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum, MambaAttentionBackendEnum, register_backend

from pear.models.qwen3_5.gpu_generation import GpuGeneration
from pear.models.qwen3_5.rollout.gpu_backend import RolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.all_reduce import AllReduceNorm
from pear.models.qwen3_5.rollout.hopper.attention import HopperFlashAttentionBackend
from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperCudaGraphManager, HopperMixedStepGraphs
from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
from pear.models.qwen3_5.rollout.hopper.gdn import HopperGDNAttention, HopperGDNBackend
from pear.models.qwen3_5.rollout.hopper.model import HopperQwen3_5ForConditionalGeneration


class HopperRolloutGpuBackend(RolloutGpuBackend):
    """vLLM classes and patches for the replicas of a Qwen3.5 model on Hopper at TP4: the steps with prefill rows as
    FULL graphs on MixedStepGDNCore with split decode attention, the pure decode steps on fused decode kernels."""

    generation = GpuGeneration.HOPPER
    tensor_parallel_size = AllReduceNorm.TP_SIZE  # 6 query heads per KV head on a rank, 4-rank reductions
    model_cls = HopperQwen3_5ForConditionalGeneration
    # under torch.compile (VLLM_USE_BREAKABLE_CUDAGRAPH=0) the Inductor settings make every fresh compile produce the
    # same kernels (verl compiles afresh at every start)
    INDUCTOR_COMPILE_CONFIG = {
        "deterministic": True,
        "benchmark_epilogue_fusion": False,
        "benchmark_fusion": False,
        "combo_kernels": True,
        "benchmark_combo_kernel": False,
        "test_configs.force_filter_reduction_configs": True,
    }

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        compilation_config.setdefault("inductor_compile_config", {}).update(cls.INDUCTOR_COMPILE_CONFIG)
        HopperMixedStepGraphs.configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def install(cls, vllm_config) -> None:
        HopperExtension.load()
        AllReduceNorm.register_op()

        # through vLLM's extension points (the model and its model state by PEAR's vLLM plugin,
        # Qwen3_5RolloutKernels.register): the GDN layer, its metadata builder and the attention backend
        QwenGatedDeltaNetAttention.register_oot(HopperGDNAttention)
        register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)(HopperGDNBackend)
        register_backend(AttentionBackendEnum.FLASH_ATTN)(HopperFlashAttentionBackend)

        # what vLLM offers no extension point for: FULL graphs of the mixed steps beside the FULL decode graphs
        HopperCudaGraphManager.install()
