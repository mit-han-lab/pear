from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel, pre_process_bwd_kernel_merged, pre_process_fwd_kernel_merged

from pear.models.qwen3_5.gpu_generation import GpuGeneration
from pear.models.qwen3_5.trainer.gpu_backend import TrainingGpuBackend
from pear.models.qwen3_5.trainer.hopper.delta_rule import HopperDeltaRule
from pear.models.qwen3_5.trainer.hopper.ulysses_attention import HopperUlyssesAttention


class HopperTrainingGpuBackend(TrainingGpuBackend):
    """Hopper's kernels: FLA's delta rule, FA4's SM90 forward with FA3's backward; the vision tower keeps its SDPA."""

    generation = GpuGeneration.HOPPER
    # FLA's autotuned kernels whose 2-warp candidates spill past a thread's 255 registers at head dimension 128: one
    # launch makes the driver keep its local memory on every resident thread for the rest of the process, outside
    # PyTorch's pool
    SPILLING_FLA_KERNELS = (
        pre_process_fwd_kernel_merged,
        pre_process_bwd_kernel_merged,
        merge_fwd_bwd_kernel,
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    )
    FLA_MIN_WARPS = 4  # the fewest warps at which their candidates' local memory stays near the registers

    attention_cls = HopperUlyssesAttention
    delta_rule_cls = HopperDeltaRule
    # the recompute runs FlashAttention-3's bf16 attention and FLA's bf16 delta rule, as the backward reads them
    recompute_is_fp16 = False
    # the projections take the twin alone
    gdn_wants_remainder = False

    @classmethod
    def install(cls) -> None:
        """Drop FLA's autotune candidates that spill past a thread's registers, before any of them is launched."""
        for kernel in cls.SPILLING_FLA_KERNELS:
            autotuner = kernel.fn  # under FLA's heuristics
            autotuner.configs = [config for config in autotuner.configs if config.num_warps >= cls.FLA_MIN_WARPS]
