from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum, register_backend

from pear.models.qwen3_5.cuda_extension import CudaExtension
from pear.models.qwen3_5.deploy.blackwell.bf16.draft.extension import BlackwellBf16DraftExtension
from pear.models.qwen3_5.deploy.blackwell.bf16.draft.speculator import ProposalLaunch as Bf16ProposalLaunch
from pear.models.qwen3_5.deploy.blackwell.bf16.runner import DecodeLaunch as Bf16DecodeLaunch
from pear.models.qwen3_5.deploy.blackwell.common.gdn import (
    DeployGDNAttention,
    DeployGDNBackend,
    DeployGDNMetadataBuilder,
)
from pear.models.qwen3_5.deploy.blackwell.common.rope import RopeStaging
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.extension import BlackwellNvfp4DraftExtension
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.speculator import ProposalLaunch as Nvfp4ProposalLaunch
from pear.models.qwen3_5.deploy.blackwell.nvfp4.runner import DecodeLaunch as Nvfp4DecodeLaunch
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend, Nvfp4CheckpointMixin
from pear.models.qwen3_5.gpu_generation import GpuGeneration

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig


class BlackwellDeployGpuBackend(DeployGpuBackend):
    """The decode megakernels of datacenter Blackwell for a deploy of a Qwen3.5 model: every verify step of a DFlash
    draft one megakernel launch, the draft's proposal in it, a megakernel per checkpoint format."""

    generation = GpuGeneration.BLACKWELL_ULTRA  # the megakernels' tcgen05 code, built for SM103
    KV_CACHE_DTYPES = ("bfloat16",)  # the attention core reads bf16 k and v
    DRAFT_TOKENS = 7  # a verify step's NT = 8 token rows (schedule.py): the target's token and 7 drafts
    # the checkpoint format's megakernel: the runner's patches that launch it, the draft's, and the draft's library
    decode_launch: ClassVar[type]
    proposal_launch: ClassVar[type]
    draft_extension: ClassVar[type[CudaExtension]]

    @classmethod
    def install(cls, vllm_config: VllmConfig) -> None:
        # the megakernel's library is loaded by the runner, once it has a draft its verify step runs
        cls.draft_extension.load()

        # through vLLM's extension points: the GDN layer and its metadata builder
        QwenGatedDeltaNetAttention.register_oot(DeployGDNAttention)
        register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)(DeployGDNBackend)

        # patches of what vLLM offers no extension point for
        DeployGDNMetadataBuilder.install()
        RopeStaging.install()
        cls.decode_launch.install()
        cls.proposal_launch.install()


class BlackwellNvfp4DeployGpuBackend(Nvfp4CheckpointMixin, BlackwellDeployGpuBackend):
    """The megakernel of ModelOpt's NVFP4 / FP8 export (blackwell/nvfp4/)."""

    draft_extension = BlackwellNvfp4DraftExtension
    decode_launch = Nvfp4DecodeLaunch
    proposal_launch = Nvfp4ProposalLaunch


class BlackwellBf16DeployGpuBackend(BlackwellDeployGpuBackend):
    """The megakernel of the unquantized checkpoint in its own bf16, reading vLLM's bf16 weights in place
    (blackwell/bf16/)."""

    draft_extension = BlackwellBf16DraftExtension
    decode_launch = Bf16DecodeLaunch
    proposal_launch = Bf16ProposalLaunch
    CHECKPOINT_DTYPE = "bfloat16"  # the checkpoint's own dtype (config.json), which the engine keeps

    @classmethod
    def reads_checkpoint(cls, vllm_config: VllmConfig) -> tuple[bool, str | None]:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config:
            return False, f"its kernels read an unquantized checkpoint, not quantization {quantization_config}"

        # the text model's config: a multimodal checkpoint keeps the dtype there
        text_config = getattr(hf_config, "text_config", hf_config)
        checkpoint_dtype = getattr(text_config, "dtype", None) or getattr(text_config, "torch_dtype", None)
        checkpoint_dtype = str(checkpoint_dtype).removeprefix("torch.") if checkpoint_dtype is not None else None
        if checkpoint_dtype != cls.CHECKPOINT_DTYPE or model_config.dtype != torch.bfloat16:
            return (
                False,
                f"its kernels read bf16 weights in bf16: checkpoint {checkpoint_dtype}, engine {model_config.dtype}",
            )

        return True, None
