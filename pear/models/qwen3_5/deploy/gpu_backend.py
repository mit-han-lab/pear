from __future__ import annotations

import os
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string

from pear.models.qwen3_5.gpu_backend import VllmGpuBackend

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig


class DeployGpuBackend(VllmGpuBackend):
    """The vLLM classes and patches of one GPU generation and one checkpoint format for a deploy of a Qwen3.5 model to
    one session; a process selects one backend."""

    tensor_parallel_size = 1
    model_cls = Qwen3_5ForConditionalGeneration  # vLLM's own: the backends patch its runner, not its model
    # the decode steps hook vLLM's V2 model runner, the one speculative decoding with a DFlash draft runs on
    V2_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
    COMPILE_CACHE_ENV = "VLLM_DISABLE_COMPILE_CACHE"  # set: the engine compiles its graphs anew
    # the draft's sampling the kernels implement: a sample of the draft's distribution, whose logits vLLM keeps only for
    # this method (its greedy draft, the argmax, has none)
    DRAFT_SAMPLE_METHOD = "probabilistic"
    DRAFT_METHOD = "dflash"  # the speculative method of the draft the kernels propose for
    DRAFT_TOKENS: ClassVar[int]  # the draft's tokens per verify step the kernels are built for
    KV_CACHE_DTYPES: ClassVar[tuple[str, ...]]  # the KV cache dtypes the kernels' attention reads, by vLLM's names

    @classmethod
    def can_implement(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        is_implementable, reason = super().can_implement(vllm_config)
        if not is_implementable:
            return is_implementable, reason

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        if max_num_seqs != 1:
            return False, f"its decode step serves one session (max_num_seqs 1), not {max_num_seqs}"

        if not vllm_config.use_v2_model_runner:
            return False, f"its decode step hooks vLLM's V2 model runner ({cls.V2_MODEL_RUNNER_ENV}=1)"

        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != cls.DRAFT_METHOD:
            draft_method = speculative_config.method if speculative_config is not None else "none"
            return False, f"its verify steps take a {cls.DRAFT_METHOD} draft, not {draft_method}"
        if speculative_config.num_speculative_tokens != cls.DRAFT_TOKENS:
            return (
                False,
                f"its verify steps take {cls.DRAFT_TOKENS} draft tokens, not {speculative_config.num_speculative_tokens}",
            )
        if speculative_config.draft_sample_method != cls.DRAFT_SAMPLE_METHOD:
            return (
                False,
                f"its verify steps take a {cls.DRAFT_SAMPLE_METHOD} draft, not {speculative_config.draft_sample_method}",
            )

        # "auto" as vLLM resolves it: the checkpoint's KV cache quantization, else the model's dtype
        model_config = vllm_config.model_config
        kv_cache_dtype = resolve_kv_cache_dtype_string(vllm_config.cache_config.cache_dtype, model_config)
        if kv_cache_dtype == "auto":
            kv_cache_dtype = str(model_config.dtype).removeprefix("torch.")
        if kv_cache_dtype not in cls.KV_CACHE_DTYPES:
            return False, f"its attention reads a KV cache of {cls.KV_CACHE_DTYPES}, not {kv_cache_dtype}"

        return cls.reads_checkpoint(vllm_config)

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        # the engine processes inherit them; the decode steps run outside vLLM's graphs
        os.environ[cls.V2_MODEL_RUNNER_ENV] = "1"
        cls.configure_compile_cache()

    @classmethod
    @abstractmethod
    def reads_checkpoint(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        """Whether the kernels read the checkpoint's format (its config.json), and the reason when they do not."""

    @classmethod
    def configure_compile_cache(cls) -> None:
        """Set vLLM's compile cache for the engine's processes."""
        # off: the cache does not key on the backend's patches
        os.environ[cls.COMPILE_CACHE_ENV] = "1"


class Nvfp4CheckpointMixin:
    """What a deploy GPU backend of ModelOpt's mixed NVFP4 / FP8 export does differently: it reads that checkpoint and
    picks the GEMMs of vLLM's own steps on it."""

    # the checkpoint the kernels read, by its config.json: ModelOpt's mixed export, NVFP4 MLP and LM head, FP8 attention
    # and Gated DeltaNet projections
    QUANTIZATION = ("modelopt", "MIXED_PRECISION")
    # the GEMMs of vLLM's own steps (the prefill) on this checkpoint, unless the engine names a linear backend
    LINEAR_BACKEND = "flashinfer_cutlass"

    @classmethod
    def reads_checkpoint(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        quantization_config = getattr(vllm_config.model_config.hf_config, "quantization_config", None) or {}
        quantization = (quantization_config.get("quant_method"), quantization_config.get("quant_algo"))
        if quantization != cls.QUANTIZATION:
            return False, f"its kernels read the quantization {cls.QUANTIZATION}, not {quantization}"

        return True, None

    @classmethod
    def install(cls, vllm_config: VllmConfig) -> None:
        # vLLM reads it as the worker builds the model's linear layers, after the install
        if vllm_config.kernel_config.linear_backend == "auto":
            vllm_config.kernel_config.linear_backend = cls.LINEAR_BACKEND
        super().install(vllm_config)
