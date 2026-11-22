from typing import ClassVar

import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellNvfp4Extension(CudaExtension):
    """The decode megakernel of blackwell/nvfp4/csrc/: one build per KV cache dtype (the subclasses), each an op
    namespace of its own, built at first use."""

    # no fast math: the kernel reproduces the lowerings of the kernels it replaces (div.full, ex2.approx, rsqrt.approx)
    # no -std: torch adds the C++ standard its headers need
    CUDA_FLAGS = ("-O3", "-lineinfo")
    LINK_FLAGS = ()
    # what the schedules take from the build (csrc MK_KV_FP8); the code and the programs are otherwise the same
    # the dtype of the cache rows its attention reads and writes: bf16, or e4m3 at scale 1
    KV_DTYPE: ClassVar[torch.dtype]
    # the dtypes of vLLM's cache tensor of such rows: an FP8 cache is allocated as bytes
    KV_CACHE_DTYPES: ClassVar[tuple[torch.dtype, ...]]

    @classmethod
    def kv_cache_of(cls, attention_layer: torch.nn.Module) -> torch.Tensor:
        """An attention layer's cache (attention_layer: vLLM's) as the build's attention reads it: the same storage, its
        rows in the build's dtype."""
        kv_cache = attention_layer.kv_cache
        assert kv_cache.dtype in cls.KV_CACHE_DTYPES, (
            f"the megakernel's build reads a KV cache of {cls.KV_CACHE_DTYPES}",
            kv_cache.dtype,
        )

        return kv_cache.view(cls.KV_DTYPE)

    @classmethod
    def check_kv_scales(cls, attention_layer: torch.nn.Module) -> None:
        """Refuse the scales of an FP8 cache, which the attention reads at scale 1 (attention_layer: vLLM's); a bf16
        cache has none."""
        if cls.KV_DTYPE != torch.float8_e4m3fn:
            return

        scales = [getattr(attention_layer, f"_{part}_scale_float", 1.0) for part in ("q", "k", "v")]
        assert scales == [1.0, 1.0, 1.0], ("an FP8 KV cache is read at scale 1", scales)

    @staticmethod
    def loaded() -> type["BlackwellNvfp4Extension"]:
        """The build of the megakernel this process loaded."""
        for extension in (BlackwellNvfp4Bf16KvExtension, BlackwellNvfp4Fp8KvExtension):
            if extension.is_loaded:
                return extension
        raise RuntimeError("the decode megakernel's library is not loaded")

    @staticmethod
    def ops() -> torch._ops._OpNamespace:
        """The ops of the build of the megakernel this process loaded."""
        return getattr(torch.ops, BlackwellNvfp4Extension.loaded().OP_NAMESPACE)


class BlackwellNvfp4Bf16KvExtension(BlackwellNvfp4Extension):
    """The build on a bf16 KV cache, as torch.ops.pear_deploy_blackwell_nvfp4."""

    OP_NAMESPACE = "pear_deploy_blackwell_nvfp4"
    KV_DTYPE = torch.bfloat16
    KV_CACHE_DTYPES = (torch.bfloat16,)


class BlackwellNvfp4Fp8KvExtension(BlackwellNvfp4Extension):
    """The build on an FP8 e4m3 KV cache (MK_KV_FP8), as torch.ops.pear_deploy_blackwell_nvfp4_fp8_kv."""

    OP_NAMESPACE = "pear_deploy_blackwell_nvfp4_fp8_kv"
    CUDA_FLAGS = (*BlackwellNvfp4Extension.CUDA_FLAGS, "-DMK_KV_FP8=1")
    KV_DTYPE = torch.float8_e4m3fn
    KV_CACHE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
