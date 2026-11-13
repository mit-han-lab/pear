from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from pear.models.qwen3_5.gpu_generation import GpuGeneration


class TrainingGpuBackend(ABC):
    """The training kernels of one GPU generation."""

    generation: ClassVar[GpuGeneration]  # the GPUs the kernels run on
    # the Ulysses full attention (eligible, apply) and the Gated DeltaNet's delta rule (apply_packed)
    attention_cls: ClassVar[type[torch.autograd.Function]]
    delta_rule_cls: ClassVar[type[torch.autograd.Function]]
    # whether a checkpoint's recompute reruns the fp16 forward, twins included; otherwise it runs the bf16 path whose
    # tensors the backward kernels read, both passes save the bf16 handle (the checkpoint compares their metadata) and
    # verl's attention retention, which replays the first forward's attention outputs, does not apply
    recompute_is_fp16: ClassVar[bool]
    # whether the input norm's twin remainder also feeds the Gated DeltaNet's projections (a second GEMM)
    gdn_wants_remainder: ClassVar[bool]

    @classmethod
    def supports_compute_capability(cls, capability: tuple[int | None, int | None]) -> bool:
        """Whether the kernels run on a GPU of this compute capability (major, minor)."""
        return cls.generation.includes(capability)

    @classmethod
    @abstractmethod
    def install(cls) -> None:
        """Apply the process-wide patches of the backend's own kernels, before the layers are patched."""
