from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

from pear.models.qwen3_5.gpu_generation import GpuGeneration
from pear.models.qwen3_5.trainer.blackwell.delta_rule import BlackwellDeltaRule, CudnnGDNGraphs
from pear.models.qwen3_5.trainer.blackwell.ulysses_attention import BlackwellUlyssesAttention
from pear.models.qwen3_5.trainer.blackwell.vision_attention import BlackwellVisionAttention
from pear.models.qwen3_5.trainer.gpu_backend import TrainingGpuBackend


class BlackwellTrainingGpuBackend(TrainingGpuBackend):
    """Datacenter Blackwell's kernels: cuDNN's FROST delta rule, FlashAttention-4's attention, a BF16x9 vision tower."""

    generation = GpuGeneration.DATACENTER_BLACKWELL
    attention_cls = BlackwellUlyssesAttention
    delta_rule_cls = BlackwellDeltaRule
    recompute_is_fp16 = True
    gdn_wants_remainder = True

    @classmethod
    def install(cls) -> None:
        CudnnGDNGraphs.install()
        # the vision tower's attention on cuBLAS's BF16x9 emulation of fp32 GEMMs, which Blackwell's tensor cores run
        BlackwellVisionAttention.install(Qwen3_5VisionAttention)
