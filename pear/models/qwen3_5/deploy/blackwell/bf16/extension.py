import os

from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellBf16Extension(CudaExtension):
    """The decode megakernel of blackwell/bf16/csrc/ as torch.ops.pear_deploy_blackwell_bf16, built at first use."""

    OP_NAMESPACE = "pear_deploy_blackwell_bf16"
    # the variable of a test build's extra nvcc flags, e.g. -DMK_NEG_ACT=5, a numerical negative control
    EXTRA_FLAGS_ENV = "PEAR_DEPLOY_EXTRA_CUDA_FLAGS"
    # no fast math: the kernel reproduces the lowerings of the kernels it replaces (div.full, ex2.approx, rsqrt.approx)
    CUDA_FLAGS = ("-O3", "-std=c++17", "-lineinfo") + tuple(os.environ.get(EXTRA_FLAGS_ENV, "").split())
    LINK_FLAGS = ()
