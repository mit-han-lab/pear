from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import vllm.envs
from vllm.platforms import current_platform

from pear.models.qwen3_5.cuda_extension import CudaExtension
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend, Nvfp4CheckpointMixin
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtx5090.draft import DraftLaunch as Rtx5090DraftLaunch
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtx5090.extension import Rtx5090Nvfp4Fp8KvExtension
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtx5090.runner import DecodeLaunch as Rtx5090DecodeLaunch
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtxpro6000.draft import DraftLaunch as RtxPro6000DraftLaunch
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtxpro6000.extension import (
    RtxPro6000Nvfp4Bf16KvExtension,
    RtxPro6000Nvfp4Fp8KvExtension,
)
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtxpro6000.runner import DecodeLaunch as RtxPro6000DecodeLaunch
from pear.models.qwen3_5.gpu_generation import GpuGeneration

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig


class RtxBlackwellDeployGpuBackend(DeployGpuBackend):
    """The decode megakernels of RTX Blackwell for a deploy of a Qwen3.5 model: every verify step of a DFlash draft one
    megakernel launch, the draft's proposal in it. A subclass per checkpoint format, then per GPU, each GPU with a
    megakernel of its own, and per KV cache dtype."""

    generation = GpuGeneration.RTX_BLACKWELL  # the megakernels' mma.sync code, built for SM120
    DRAFT_TOKENS = 7  # a verify step's NT = 8 token rows (schedule.py): the target's token and 7 drafts
    # the GPUs, by NVML's names, whose SM count the schedules are built for and whose memory the recipe's window fits
    # (the family's GPUs share one compute capability: the name tells them apart)
    GPU_NAMES: ClassVar[tuple[str, ...]]
    # the GPU's megakernel: its library and the runner's patches that launch it and the draft's proposal
    extension: ClassVar[type[CudaExtension]]
    decode_launch: ClassVar[type]
    draft_launch: ClassVar[type]

    @classmethod
    def can_implement(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        is_implementable, reason = super().can_implement(vllm_config)
        if not is_implementable:
            return is_implementable, reason

        gpu_name = current_platform.get_device_name()
        if gpu_name not in cls.GPU_NAMES:
            return False, f"its schedules are built for the {cls.GPU_NAMES}, not the {gpu_name}"

        return True, None

    @classmethod
    def install(cls, vllm_config: VllmConfig) -> None:
        cls.extension.load()
        cls.decode_launch.install()
        cls.draft_launch.install()


class RtxBlackwellNvfp4DeployGpuBackend(Nvfp4CheckpointMixin, RtxBlackwellDeployGpuBackend):
    """The megakernels of ModelOpt's NVFP4 / FP8 export (rtx_blackwell/nvfp4/): a subclass per GPU, with the megakernel
    of its package, and per KV cache dtype of that megakernel's builds."""

    # the compiled graphs' cache root under vLLM's, keyed by the GPUs' packages' sources: the draft's projections,
    # replaced at loading by NVFP4 ones (draft.py), shape graphs that vLLM's cache key does not cover
    PACKAGE = Path(__file__).resolve().parent / "nvfp4"
    CACHE_KEY_CHARS = 12  # the hexadecimal digits of the sources' hash in the cache root's name
    # the cache root's name before the hash: the checkpoint format's, whichever GPU (the server configures before the
    # engine's config says which backend of the GPU's generation serves)
    COMPILE_CACHE_NAME = "pear_deploy_rtx_blackwell_nvfp4"

    @classmethod
    def configure_compile_cache(cls) -> None:
        sources_hash = hashlib.sha256()
        for path in sorted([*cls.PACKAGE.rglob("*.py"), *cls.PACKAGE.rglob("csrc/*")]):
            sources_hash.update(path.read_bytes())
        # (and the GPU: the graphs call the ops of its megakernel)
        sources_hash.update(current_platform.get_device_name().encode())
        cache_dir = f"{cls.COMPILE_CACHE_NAME}-{sources_hash.hexdigest()[: cls.CACHE_KEY_CHARS]}"
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(vllm.envs.VLLM_CACHE_ROOT, cache_dir)


class Fp8KvCacheMixin:
    """What a backend adds whose build of the megakernel reads an FP8 KV cache: the cache's e4m3 rows at scale 1."""

    KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")  # "fp8" is e4m3 on CUDA

    @classmethod
    def can_implement(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        is_implementable, reason = super().can_implement(vllm_config)
        if not is_implementable:
            return is_implementable, reason

        if getattr(vllm_config.cache_config, "calculate_kv_scales", False):
            return False, "its attention reads the FP8 KV cache at scale 1, not with calculated KV scales"

        return True, None


class Rtx5090Nvfp4Fp8KvDeployGpuBackend(Fp8KvCacheMixin, RtxBlackwellNvfp4DeployGpuBackend):
    """The RTX 5090's megakernel (rtx_blackwell/nvfp4/rtx5090/), on an FP8 KV cache: its one build."""

    GPU_NAMES = ("NVIDIA GeForce RTX 5090",)
    extension = Rtx5090Nvfp4Fp8KvExtension
    decode_launch = Rtx5090DecodeLaunch
    draft_launch = Rtx5090DraftLaunch


class RtxPro6000Nvfp4DeployGpuBackend(RtxBlackwellNvfp4DeployGpuBackend):
    """The RTX PRO 6000's megakernel (rtx_blackwell/nvfp4/rtxpro6000/): a subclass per KV cache dtype, each with its
    build."""

    GPU_NAMES = ("NVIDIA RTX PRO 6000 Blackwell Server Edition",)
    decode_launch = RtxPro6000DecodeLaunch
    draft_launch = RtxPro6000DraftLaunch
    MAMBA_CACHE_DTYPES = ("auto", "float32")  # the Gated DeltaNet's recurrent states the kernel reads: fp32

    @classmethod
    def can_implement(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        is_implementable, reason = super().can_implement(vllm_config)
        if not is_implementable:
            return is_implementable, reason

        mamba_cache_dtype = vllm_config.cache_config.mamba_ssm_cache_dtype
        if mamba_cache_dtype not in cls.MAMBA_CACHE_DTYPES:
            return False, f"its Gated DeltaNet reads fp32 recurrent states, not {mamba_cache_dtype}"

        return True, None


class RtxPro6000Nvfp4Bf16KvDeployGpuBackend(RtxPro6000Nvfp4DeployGpuBackend):
    """The RTX PRO 6000's megakernel on a bf16 KV cache, the dtype vLLM gives a bf16 model."""

    extension = RtxPro6000Nvfp4Bf16KvExtension
    KV_CACHE_DTYPES = ("bfloat16",)


class RtxPro6000Nvfp4Fp8KvDeployGpuBackend(Fp8KvCacheMixin, RtxPro6000Nvfp4DeployGpuBackend):
    """The RTX PRO 6000's megakernel on an FP8 KV cache."""

    extension = RtxPro6000Nvfp4Fp8KvExtension
