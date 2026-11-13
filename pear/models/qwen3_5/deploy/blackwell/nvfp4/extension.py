from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellNvfp4Extension(CudaExtension):
    """The decode megakernel of blackwell/nvfp4/csrc/ as torch.ops.pear_deploy_blackwell_nvfp4, built at first use."""

    OP_NAMESPACE = "pear_deploy_blackwell_nvfp4"
    # no fast math: the kernel reproduces the lowerings of the kernels it replaces (div.full, ex2.approx, rsqrt.approx)
    CUDA_FLAGS = ("-O3", "-std=c++17", "-lineinfo")
    LINK_FLAGS = ()
