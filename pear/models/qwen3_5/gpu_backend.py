from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

from pear.models.qwen3_5.gpu_generation import GpuGeneration


class VllmGpuBackend(ABC):
    """The vLLM classes and patches of one GPU generation for an engine of a Qwen3.5 model; a process selects one
    backend."""

    # the HF architecture of the engine's model, which the backends serve
    ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
    generation: ClassVar[GpuGeneration]  # the GPUs the kernels run on
    tensor_parallel_size: ClassVar[int]  # the engine's tensor parallelism the kernels are written for
    model_cls: ClassVar[type]  # the model class the engine builds for ARCHITECTURE

    @staticmethod
    def select(gpu_backends: tuple[type[VllmGpuBackend], ...], vllm_config, error_message: str) -> type[VllmGpuBackend]:
        """The first of gpu_backends that supports this process's GPU and, given one, the engine config, else a
        ValueError of error_message with each one's reason."""
        # the capability through NVML: the server process does not initialize CUDA
        capability = current_platform.get_device_capability()
        reasons = []

        for gpu_backend in gpu_backends:
            if not gpu_backend.supports_compute_capability(capability):
                reasons.append(f"{gpu_backend.__name__} does not support compute capability {capability}")
                continue
            is_implementable, reason = (
                gpu_backend.can_implement(vllm_config) if vllm_config is not None else (True, None)
            )
            if is_implementable:
                return gpu_backend
            reasons.append(f"{gpu_backend.__name__} cannot serve the engine config: {reason}")

        raise ValueError(f"{error_message}:\n" + "\n".join(reasons))

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability | None) -> bool:
        """Whether the kernels run on a GPU of this compute capability (None: no GPU)."""
        return cls.generation.includes(capability)

    @classmethod
    def can_implement(cls, vllm_config) -> tuple[bool, str | None]:
        """Whether the kernels serve this engine config, and the reason when they do not."""
        architectures = vllm_config.model_config.architectures
        if cls.ARCHITECTURE not in architectures:
            return False, f"its model replaces vLLM's {cls.ARCHITECTURE}, not {architectures}"

        tp = vllm_config.parallel_config.tensor_parallel_size
        if tp != cls.tensor_parallel_size:
            return False, f"its kernels are written for tensor parallelism {cls.tensor_parallel_size}, not {tp}"

        return True, None

    @classmethod
    @abstractmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """Set the engine's compilation settings the kernels need, in the process that builds the engine config."""

    @classmethod
    @abstractmethod
    def install(cls, vllm_config) -> None:
        """Register the backend's classes with vLLM and apply its patches, in a worker before it builds its model."""
