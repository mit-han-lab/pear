from pear.models.qwen3_5.cuda_extension import CudaExtension


class BlackwellBf16DraftExtension(CudaExtension):
    """The DFlash 2 draft's proposal megakernel of blackwell/bf16/draft/csrc/ as
    torch.ops.pear_deploy_blackwell_bf16_draft, built at first use."""

    OP_NAMESPACE = "pear_deploy_blackwell_bf16_draft"
    # no fast math: the kernel reproduces the lowerings of vLLM's kernels (div.full, ex2.approx, rsqrt.approx)
    CUDA_FLAGS = ("-O3", "-std=c++17", "-lineinfo")
    LINK_FLAGS = ()
