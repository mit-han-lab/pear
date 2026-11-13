from pear.models.qwen3_5.cuda_extension import CudaExtension


class HopperExtension(CudaExtension):
    """The Hopper replicas' CUDA kernels of hopper/csrc/ as torch.ops.pear_rollout_hopper, built at first use."""

    # the kernels reproduce Triton's and FlashInfer's exact lowerings (divisions, exp2, rsqrt): no fast math
    OP_NAMESPACE = "pear_rollout_hopper"
    CUDA_FLAGS = ("-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo", "-DNDEBUG")
    LINK_FLAGS = ()
