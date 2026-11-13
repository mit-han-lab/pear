from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import vllm.envs
from vllm.platforms import current_platform

from pear.models.qwen3_5.cuda_extension import CudaExtension
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend, Nvfp4CheckpointMixin
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.draft import DraftLaunch
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.extension import RtxBlackwellNvfp4Extension
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.runner import DecodeLaunch
from pear.models.qwen3_5.gpu_generation import GpuGeneration

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig


class RtxBlackwellDeployGpuBackend(DeployGpuBackend):
    """The decode megakernels of RTX Blackwell for a deploy of a Qwen3.5 model: every verify step of a DFlash draft one
    megakernel launch, the draft's proposal in it."""

    generation = GpuGeneration.RTX_BLACKWELL  # the megakernels' mma.sync code, built for SM120
    DRAFT_TOKENS = 7  # a verify step's NT = 8 token rows (schedule.py): the target's token and 7 drafts
    # the GPUs, by NVML's names, whose SM count the schedules are built for and whose memory the recipe's window fits
    GPU_NAMES = ("NVIDIA GeForce RTX 5090",)
    # the checkpoint format's megakernel: its library and the runner's patches that launch it and the draft's proposal
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
    """The megakernel of ModelOpt's NVFP4 / FP8 export on an FP8 KV cache (rtx_blackwell/nvfp4/)."""

    extension = RtxBlackwellNvfp4Extension
    decode_launch = DecodeLaunch
    draft_launch = DraftLaunch
    # the attention reads e4m3 k and v at scale 1 ("fp8" is e4m3 on CUDA)
    KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")
    # the compiled graphs' cache root under vLLM's, keyed by the package's sources: the draft's projections, replaced at
    # loading by NVFP4 ones (draft.py), shape graphs that vLLM's cache key does not cover
    PACKAGE = Path(__file__).resolve().parent / "nvfp4"
    CACHE_KEY_CHARS = 12  # the hexadecimal digits of the sources' hash in the cache root's name

    @classmethod
    def configure_compile_cache(cls) -> None:
        sources_hash = hashlib.sha256()
        for path in sorted([*cls.PACKAGE.glob("*.py"), *cls.PACKAGE.glob("csrc/*")]):
            sources_hash.update(path.read_bytes())
        cache_dir = f"{cls.extension.OP_NAMESPACE}-{sources_hash.hexdigest()[: cls.CACHE_KEY_CHARS]}"
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(vllm.envs.VLLM_CACHE_ROOT, cache_dir)
