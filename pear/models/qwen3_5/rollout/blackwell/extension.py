import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellExtension(CudaExtension):
    """The datacenter Blackwell replicas' CUDA kernels of blackwell/csrc/ as torch.ops.pear_rollout_blackwell, built at
    first use."""

    OP_NAMESPACE = "pear_rollout_blackwell"
    CUDA_FLAGS = ("-O3", "--use_fast_math", "-std=c++17", "-lineinfo")
    LINK_FLAGS = ("-lcublasLt",)

    @staticmethod
    def register_fakes() -> None:
        # the decoder layers' fused all-reduce + norms and the linear layers' GEMMs
        @torch.library.register_fake("pear_rollout_blackwell::ar_add_rmsnorm")
        def _(x, residual, weight, eps, out_f32):
            return x.new_empty(x.shape, dtype=torch.float32 if out_f32 else torch.bfloat16), torch.empty_like(residual)

        @torch.library.register_fake("pear_rollout_blackwell::ar_rmsnorm")
        def _(x, weight, eps):
            return torch.empty_like(x), x.new_empty(x.shape, dtype=torch.float32)

        @torch.library.register_fake("pear_rollout_blackwell::lt_mm")
        def _(x, w):
            return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)
