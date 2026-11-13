from enum import Enum


class GpuGeneration(Enum):
    """A GPU generation the family's kernels are written for, by its CUDA compute capability (major, minor)."""

    # SM90 (H100, H200): wgmma, TMA multicast and FlashAttention-3
    HOPPER = (9, 0)
    # datacenter Blackwell, SM100 and SM103 (B200, B300 and GB300): the tcgen05 instructions of the TRT-LLM attention
    # and of the CuTe DSL kernels; None: every minor
    DATACENTER_BLACKWELL = (10, None)
    # SM103 alone (B300 and GB300): the decode megakernels' tcgen05 code is built for sm_103a
    BLACKWELL_ULTRA = (10, 3)
    # RTX Blackwell, SM120 (RTX 5090, RTX PRO 6000): mma.sync with FP4 and FP8 operands, no tcgen05
    RTX_BLACKWELL = (12, 0)

    def includes(self, capability) -> bool:
        """Whether a GPU's compute capability (major, minor) is of this generation; None or (None, None): no GPU."""
        major, minor = self.value

        return capability is not None and capability[0] == major and (minor is None or capability[1] == minor)
