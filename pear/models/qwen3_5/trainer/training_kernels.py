from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionAttention

from pear.models.kernels import TrainingKernels
from pear.models.qwen3_5.trainer.attention import FullAttention
from pear.models.qwen3_5.trainer.blackwell.gpu_backend import BlackwellTrainingGpuBackend
from pear.models.qwen3_5.trainer.forward_with_torch_backend import ForwardWithTorchBackend
from pear.models.qwen3_5.trainer.gdn import GatedDeltaNet
from pear.models.qwen3_5.trainer.gpu_backend import TrainingGpuBackend
from pear.models.qwen3_5.trainer.head import Fp32Head, FusedHead
from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
from pear.models.qwen3_5.trainer.mlp import Fp32SwiGLUMLP
from pear.models.qwen3_5.trainer.operands import Fp16Operands, RecomputeFlag
from pear.models.qwen3_5.trainer.residual_stream import Fp32ResidualStream
from pear.models.qwen3_5.trainer.rms_norm import DecoderLayer, PreciseRMSNorm
from pear.models.qwen3_5.trainer.vision_attention import BatchedVisionAttention
from pear.models.qwen3_5.trainer.vision_tower import VisionTower
from verl.utils.device import get_device_capability


class Qwen3_5TrainingKernels(TrainingKernels):
    """The training kernels of a Qwen3.5 actor (fsdp_config.kernels): on any GPU a batched vision attention, the
    residual stream and the head's logits in fp32 and precise norms, and with a GPU generation's backend the fused
    training step."""

    MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")  # the dense and the mixture-of-experts architecture
    FUSED_KERNELS_BACKEND = "torch"  # the backend of verl's fused kernels whose forward the family replaces
    # the GPU generations' backends, in priority order
    GPU_BACKENDS = (BlackwellTrainingGpuBackend, HopperTrainingGpuBackend)
    GPU_BACKEND_MODEL_TYPE = "qwen3_5"  # the dense architecture, whose paths the GPU backend replaces
    VISION_TOWER = "model.visual"  # the module mixed_precision.fp32_modules must name

    @classmethod
    def check_config(cls, engine_config) -> None:
        """Raise unless the actor is FSDP2's with an fp32 residual stream and the vision tower in fp32 on TF32."""
        mixed_precision = engine_config.mixed_precision or {}
        if not (
            engine_config.strategy == "fsdp2"
            and mixed_precision.get("fp32_residual")
            and cls.VISION_TOWER in mixed_precision.get("fp32_modules", [])
            and mixed_precision.get("fp32_modules_tf32")
        ):
            raise ValueError(
                "the GPU backend of Qwen3_5TrainingKernels requires FSDP2, mixed_precision.fp32_residual "
                "and the vision tower in fp32 on TF32"
            )

    @classmethod
    def install(cls, module, fp32_residual: bool, use_fused_kernels: bool, fused_kernels_backend: str | None) -> None:
        if module.config.model_type not in cls.MODEL_TYPES:
            raise NotImplementedError(f"{cls.__name__} are for Qwen3.5, not {module.config.model_type}")

        # a fixed order: the GPU backend replaces the vision attention's forward and wraps the head's and the norms'
        BatchedVisionAttention.install(Qwen3_5VisionAttention)
        BatchedVisionAttention.install(Qwen3_5MoeVisionAttention)
        if fp32_residual:
            Fp32ResidualStream.install(module)
        if use_fused_kernels:
            Fp32Head.install(module.lm_head)
        PreciseRMSNorm.install(module.model.language_model)
        if use_fused_kernels and fused_kernels_backend == cls.FUSED_KERNELS_BACKEND:
            ForwardWithTorchBackend.install(type(module))

    @classmethod
    def check_model(
        cls, module, use_remove_padding: bool, use_fused_kernels: bool, fused_kernels_backend: str | None
    ) -> None:
        """Raise unless the module runs the paths the GPU backend replaces: dense Qwen3.5, packed, the torch fused
        head."""
        if module.config.model_type != cls.GPU_BACKEND_MODEL_TYPE:
            raise NotImplementedError(
                f"the GPU backend of Qwen3_5TrainingKernels is for the dense Qwen3.5, not {module.config.model_type}"
            )
        if not (use_remove_padding and use_fused_kernels and fused_kernels_backend == cls.FUSED_KERNELS_BACKEND):
            raise ValueError(
                "the GPU backend of Qwen3_5TrainingKernels requires use_remove_padding and the torch backend of "
                "use_fused_kernels"
            )

    @classmethod
    def install_gpu_backend(cls, module) -> None:
        """Patch the module's classes and layers with the backend of the process's GPU, before FSDP wraps the
        module."""
        gpu_backend = cls.select_gpu_backend(get_device_capability())
        gpu_backend.install()

        DecoderLayer.install(module, gpu_backend)
        GatedDeltaNet.install(module, gpu_backend)
        FullAttention.install(module, gpu_backend)
        Fp32SwiGLUMLP.install(module)
        FusedHead.install(module)
        VisionTower.install(module)

    @classmethod
    def install_checkpointing(cls, module) -> None:
        RecomputeFlag.install(module)

    @classmethod
    def run_checked(cls, run, before_rerun):
        return Fp16Operands.run_checked(run, before_rerun)

    @classmethod
    def select_gpu_backend(cls, capability: tuple[int, int]) -> type[TrainingGpuBackend]:
        """The first GPU backend that supports the GPU's compute capability."""
        for gpu_backend in cls.GPU_BACKENDS:
            if gpu_backend.supports_compute_capability(capability):
                return gpu_backend

        names = ", ".join(gpu_backend.__name__ for gpu_backend in cls.GPU_BACKENDS)
        raise NotImplementedError(
            f"no GPU backend of Qwen3_5TrainingKernels ({names}) supports compute capability {capability}"
        )
