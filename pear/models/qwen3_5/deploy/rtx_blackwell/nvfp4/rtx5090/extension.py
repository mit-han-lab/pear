import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class Rtx5090Nvfp4Fp8KvExtension(CudaExtension):
    """The RTX 5090's decode megakernel (rtx_blackwell/nvfp4/rtx5090/csrc/), on an FP8 e4m3 KV cache, as
    torch.ops.pear_deploy_rtx_blackwell_nvfp4, built at first use."""

    OP_NAMESPACE = "pear_deploy_rtx_blackwell_nvfp4"
    # no fast math: the kernel reproduces the lowerings of the kernels it replaces
    # no -std: torch adds the C++ standard its headers need
    CUDA_FLAGS = ("-O3", "-lineinfo", "--expt-relaxed-constexpr")
    LINK_FLAGS = ()
    # the dtypes of vLLM's cache tensor the attention reads and writes: e4m3 rows, which vLLM allocates as bytes
    KV_CACHE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
    QUERY_DTYPE = torch.float16  # the attention's Q rows: f16 (the bf16 values) for the f16 mma with e4m3 keys

    @classmethod
    def register_fakes(cls) -> None:
        # the draft's dequantization, in the draft model vLLM compiles
        torch.library.register_fake(f"{cls.OP_NAMESPACE}::nvfp4_dequant")(
            lambda codes, scales, gs: codes.new_empty((codes.shape[0], 2 * codes.shape[1]), dtype=torch.bfloat16)
        )

    @classmethod
    def kv_element_bytes(cls) -> int:
        """The bytes of an element of the cache the attention reads."""
        return cls.KV_CACHE_DTYPES[0].itemsize

    @classmethod
    def kv_cache_bytes(cls, kv_cache: torch.Tensor, num_heads: int, row_elems: int) -> torch.Tensor:
        """vLLM's cache of an attention layer, (kernel blocks, num_heads, block tokens, row_elems of k | v), as the
        attention reads it: its bytes in the cache's NHD memory order [kernel blocks][tokens][heads], flat."""
        assert kv_cache.dtype in cls.KV_CACHE_DTYPES, (
            f"the megakernel reads a KV cache of {cls.KV_CACHE_DTYPES}",
            kv_cache.dtype,
        )
        kv_bytes = kv_cache.permute(0, 2, 1, 3)
        assert kv_bytes.shape[2:] == (num_heads, row_elems) and kv_bytes.is_contiguous(), (
            tuple(kv_cache.shape),
            kv_cache.stride(),
        )

        return kv_bytes.view(torch.uint8).reshape(-1)

    @staticmethod
    def check_kv_scales(attention_layer: torch.nn.Module) -> None:
        """Refuse the scales of an FP8 cache, which the attention reads at scale 1 (attention_layer: vLLM's)."""
        scales = [getattr(attention_layer, f"_{part}_scale_float") for part in ("q", "k", "v")]
        assert scales == [1.0, 1.0, 1.0], ("an FP8 KV cache is read at scale 1", scales)

    @classmethod
    def loaded(cls) -> type["Rtx5090Nvfp4Fp8KvExtension"]:
        """The megakernel's build this process loaded: its one build."""
        if not cls.is_loaded:
            raise RuntimeError("the decode megakernel's library is not loaded")
        return cls

    @classmethod
    def ops(cls) -> torch._ops._OpNamespace:
        """The megakernel's ops."""
        return getattr(torch.ops, cls.loaded().OP_NAMESPACE)
