import torch

from pear.models.qwen3_5.cuda_extension import CudaExtension


class RtxBlackwellNvfp4Extension(CudaExtension):
    """The decode megakernel of rtx_blackwell/nvfp4/csrc/ as torch.ops.pear_deploy_rtx_blackwell_nvfp4, built at first
    use."""

    OP_NAMESPACE = "pear_deploy_rtx_blackwell_nvfp4"
    # no fast math: the kernel reproduces the lowerings of the kernels it replaces
    CUDA_FLAGS = ("-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr")
    LINK_FLAGS = ()

    @staticmethod
    def register_fakes() -> None:
        # the draft's dequantization, in the draft model vLLM compiles
        torch.library.register_fake("pear_deploy_rtx_blackwell_nvfp4::nvfp4_dequant")(
            lambda codes, scales, gs: codes.new_empty((codes.shape[0], 2 * codes.shape[1]), dtype=torch.bfloat16)
        )
