from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum, MambaAttentionBackendEnum, register_backend

from pear.models.qwen3_5.gpu_generation import GpuGeneration
from pear.models.qwen3_5.rollout.blackwell.attention import BlackwellFlashInferBackend, DecodeSplit
from pear.models.qwen3_5.rollout.blackwell.cudagraph import BlackwellMixedStepGraphs
from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gdn.layer import BlackwellGDNAttention, BlackwellGDNBackend
from pear.models.qwen3_5.rollout.blackwell.gemm import DraftGemmTableLinearMethod, GemmTableLinearMethod
from pear.models.qwen3_5.rollout.blackwell.model import TP_SIZE, BlackwellQwen3_5ForConditionalGeneration
from pear.models.qwen3_5.rollout.blackwell.mtp import BlackwellQwen3_5MTP
from pear.models.qwen3_5.rollout.gpu_backend import RolloutGpuBackend
from pear.models.qwen3_5.rollout.nonfinite_logits import NonfiniteCheckedOutput


class BlackwellRolloutGpuBackend(RolloutGpuBackend):
    """vLLM classes and patches for the replicas of a Qwen3.5 model on datacenter Blackwell at TP2: the Gated
    DeltaNet's own kernels, an fp32 residual stream with fused all-reduces, mixed steps as CUDA graphs, the GEMM
    tables, the fused samplers, the MTP drafter's kernels."""

    generation = GpuGeneration.DATACENTER_BLACKWELL
    tensor_parallel_size = TP_SIZE
    model_cls = BlackwellQwen3_5ForConditionalGeneration
    draft_model_cls = BlackwellQwen3_5MTP

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        BlackwellMixedStepGraphs.configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def install(cls, vllm_config) -> None:
        BlackwellExtension.load()
        GemmTableLinearMethod.load_table()
        DraftGemmTableLinearMethod.load_table()

        # through vLLM's extension points (the model and its model state by PEAR's vLLM plugin,
        # Qwen3_5RolloutKernels.register): the GDN layer, its metadata builder and the attention backend
        QwenGatedDeltaNetAttention.register_oot(BlackwellGDNAttention)
        register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)(BlackwellGDNBackend)
        register_backend(AttentionBackendEnum.FLASHINFER)(BlackwellFlashInferBackend)

        # what vLLM offers no extension point for: FlashInfer's SM count in vLLM's own decode attention, the step
        # output that raises on the NaN and +inf logits the samplers count
        DecodeSplit.install()
        NonfiniteCheckedOutput.install()
