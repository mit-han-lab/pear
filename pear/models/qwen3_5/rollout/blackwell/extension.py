import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellExtension(CudaExtension):
    """The datacenter Blackwell replicas' CUDA kernels of blackwell/csrc/ as torch.ops.pear_rollout_blackwell, built at
    first use."""

    OP_NAMESPACE = "pear_rollout_blackwell"
    # no -std: torch adds the C++ standard its headers need
    CUDA_FLAGS = ("-O3", "--use_fast_math", "-lineinfo")
    LINK_FLAGS = ("-lcublasLt",)

    @staticmethod
    def register_fakes() -> None:
        # the decoder layers' fused all-reduce + norms (x a bf16 partial [T, H] or fp32 split-K partials [splits, T, H])
        # and the linear layers' GEMMs
        @torch.library.register_fake("pear_rollout_blackwell::ar_add_rmsnorm")
        def _(x, residual, weight, eps, out_f32):
            out_dtype = torch.float32 if out_f32 else torch.bfloat16
            return x.new_empty(x.shape[-2:], dtype=out_dtype), torch.empty_like(residual)

        @torch.library.register_fake("pear_rollout_blackwell::ar_rmsnorm")
        def _(x, weight, eps):
            return x.new_empty(x.shape[-2:], dtype=torch.bfloat16), x.new_empty(x.shape[-2:], dtype=torch.float32)

        @torch.library.register_fake("pear_rollout_blackwell::lt_mm")
        def _(x, w):
            return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)

        @torch.library.register_fake("pear_rollout_blackwell::sg_swiglu")
        def _(x, w):
            return x.new_empty((x.shape[0], w.shape[0] // 2), dtype=torch.bfloat16)

        @torch.library.register_fake("pear_rollout_blackwell::sg_partials")
        def _(x, w, splits):
            return x.new_empty((splits, x.shape[0], w.shape[0]), dtype=torch.float32)

        # the gated full attention's output gate
        @torch.library.register_fake("pear_rollout_blackwell::output_gate")
        def _(x, gate):
            return torch.empty_like(x)
