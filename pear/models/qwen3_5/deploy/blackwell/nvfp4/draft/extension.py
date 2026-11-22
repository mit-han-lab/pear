from typing import ClassVar

import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellNvfp4DraftExtension(CudaExtension):
    """The DFlash 2 draft's proposal megakernel of blackwell/nvfp4/draft/csrc/: one build per KV cache dtype (the
    subclasses), each an op namespace of its own, built at first use."""

    # no fast math: the kernel reproduces the lowerings of vLLM's kernels (div.full, ex2.approx, rsqrt.approx)
    # no -std: torch adds the C++ standard its headers need
    CUDA_FLAGS = ("-O3", "-lineinfo")
    LINK_FLAGS = ()
    # the dtype of the draft's cache rows its attention reads and writes (csrc MK_KV_FP8): bf16, or e4m3 at scale 1
    KV_DTYPE: ClassVar[torch.dtype]

    @staticmethod
    def loaded() -> type["BlackwellNvfp4DraftExtension"]:
        """The build of the draft's kernel this process loaded."""
        for extension in (BlackwellNvfp4Bf16KvDraftExtension, BlackwellNvfp4Fp8KvDraftExtension):
            if extension.is_loaded:
                return extension
        raise RuntimeError("the draft's kernel library is not loaded")

    @staticmethod
    def ops() -> torch._ops._OpNamespace:
        """The ops of the build of the draft's kernel this process loaded."""
        return getattr(torch.ops, BlackwellNvfp4DraftExtension.loaded().OP_NAMESPACE)


class BlackwellNvfp4Bf16KvDraftExtension(BlackwellNvfp4DraftExtension):
    """The build on a bf16 KV cache, as torch.ops.pear_deploy_blackwell_nvfp4_draft."""

    OP_NAMESPACE = "pear_deploy_blackwell_nvfp4_draft"
    KV_DTYPE = torch.bfloat16


class BlackwellNvfp4Fp8KvDraftExtension(BlackwellNvfp4DraftExtension):
    """The build on an FP8 e4m3 KV cache (MK_KV_FP8), as torch.ops.pear_deploy_blackwell_nvfp4_fp8_kv_draft."""

    OP_NAMESPACE = "pear_deploy_blackwell_nvfp4_fp8_kv_draft"
    CUDA_FLAGS = (*BlackwellNvfp4DraftExtension.CUDA_FLAGS, "-DMK_KV_FP8=1")
    KV_DTYPE = torch.float8_e4m3fn
