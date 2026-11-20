from typing import ClassVar

import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class RtxPro6000Nvfp4Extension(CudaExtension):
    """The RTX PRO 6000's decode megakernel (rtx_blackwell/nvfp4/rtxpro6000/csrc/): one build per KV cache dtype (the
    subclasses), each an op namespace of its own, built at first use."""

    # no fast math: the kernel reproduces the lowerings of the kernels it replaces
    # no -std: torch adds the C++ standard its headers need
    CUDA_FLAGS = ("-O3", "-lineinfo", "--expt-relaxed-constexpr")
    LINK_FLAGS = ()
    # what the schedules take from the build (csrc MK_KV_FP8); the code and the programs are otherwise the same
    # the dtypes of vLLM's cache tensor its attention reads and writes: e4m3 rows, which vLLM allocates as bytes, or
    # bf16 ones
    KV_CACHE_DTYPES: ClassVar[tuple[torch.dtype, ...]]
    # its attention's Q rows: the values' own bf16, or f16 (the bf16 values) for the f16 mma with e4m3 keys
    QUERY_DTYPE: ClassVar[torch.dtype]

    @classmethod
    def register_fakes(cls) -> None:
        # the draft's dequantization, in the draft model vLLM compiles
        torch.library.register_fake(f"{cls.OP_NAMESPACE}::nvfp4_dequant")(
            lambda codes, scales, gs: codes.new_empty((codes.shape[0], 2 * codes.shape[1]), dtype=torch.bfloat16)
        )

    @classmethod
    def kv_element_bytes(cls) -> int:
        """The bytes of an element of the cache the build's attention reads."""
        return cls.KV_CACHE_DTYPES[0].itemsize

    @classmethod
    def kv_cache_bytes(cls, kv_cache: torch.Tensor, num_heads: int, row_elems: int) -> torch.Tensor:
        """vLLM's cache of an attention layer, (kernel blocks, num_heads, block tokens, row_elems of k | v), as the
        build's attention reads it: its bytes in the cache's NHD memory order [kernel blocks][tokens][heads], flat."""
        assert kv_cache.dtype in cls.KV_CACHE_DTYPES, (
            f"the megakernel's build reads a KV cache of {cls.KV_CACHE_DTYPES}",
            kv_cache.dtype,
        )
        kv_bytes = kv_cache.permute(0, 2, 1, 3)
        assert kv_bytes.shape[2:] == (num_heads, row_elems) and kv_bytes.is_contiguous(), (
            tuple(kv_cache.shape),
            kv_cache.stride(),
        )

        return kv_bytes.view(torch.uint8).reshape(-1)

    @classmethod
    def check_kv_scales(cls, attention_layer: torch.nn.Module) -> None:
        """Refuse the scales of an FP8 cache, which the attention reads at scale 1 (attention_layer: vLLM's); a bf16
        cache has none."""
        if torch.float8_e4m3fn not in cls.KV_CACHE_DTYPES:
            return

        scales = [getattr(attention_layer, f"_{part}_scale_float") for part in ("q", "k", "v")]
        assert scales == [1.0, 1.0, 1.0], ("an FP8 KV cache is read at scale 1", scales)

    @staticmethod
    def loaded() -> type["RtxPro6000Nvfp4Extension"]:
        """The build of the megakernel this process loaded."""
        for extension in (RtxPro6000Nvfp4Bf16KvExtension, RtxPro6000Nvfp4Fp8KvExtension):
            if extension.is_loaded:
                return extension
        raise RuntimeError("the decode megakernel's library is not loaded")

    @staticmethod
    def ops() -> torch._ops._OpNamespace:
        """The ops of the build of the megakernel this process loaded."""
        return getattr(torch.ops, RtxPro6000Nvfp4Extension.loaded().OP_NAMESPACE)


class RtxPro6000Nvfp4Bf16KvExtension(RtxPro6000Nvfp4Extension):
    """The build on a bf16 KV cache, as torch.ops.pear_deploy_rtx_blackwell_nvfp4_rtxpro6000."""

    OP_NAMESPACE = "pear_deploy_rtx_blackwell_nvfp4_rtxpro6000"
    KV_CACHE_DTYPES = (torch.bfloat16,)
    QUERY_DTYPE = torch.bfloat16


class RtxPro6000Nvfp4Fp8KvExtension(RtxPro6000Nvfp4Extension):
    """The build on an FP8 e4m3 KV cache (MK_KV_FP8), as torch.ops.pear_deploy_rtx_blackwell_nvfp4_rtxpro6000_fp8_kv."""

    OP_NAMESPACE = "pear_deploy_rtx_blackwell_nvfp4_rtxpro6000_fp8_kv"
    CUDA_FLAGS = (*RtxPro6000Nvfp4Extension.CUDA_FLAGS, "-DMK_KV_FP8=1")
    KV_CACHE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
    QUERY_DTYPE = torch.float16
