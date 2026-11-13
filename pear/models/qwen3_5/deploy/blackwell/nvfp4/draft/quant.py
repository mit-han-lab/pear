import math
from dataclasses import dataclass

import torch

E2M1_MAX, E4M3_MAX = 6.0, 448.0  # the largest e2m1 and e4m3 magnitudes
BLOCK = 16  # weights per e4m3 scale
ATOM_ROWS, ATOM_BLOCKS = 128, 4  # vLLM's swizzled scale atom: 128 rows x 4 blocks (512 bytes, [32][4][4])
# e2m1's magnitudes (code = index | sign << 3) and the bounds between neighbours (ties to the even code)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass
class Fp4Weight:
    """A projection [rows, K] in NVFP4 (the target's format): weight = e2m1 * scale / global_scale."""

    codes: torch.Tensor  # uint8 [rows, K / 2]: two e2m1 codes per byte, the even K index in the low nibble
    scales: torch.Tensor  # uint8 (e4m3) [rows / 128 * K / 64 * 512]: vLLM's swizzled atoms
    global_scale: float  # a power of two (the dequantized weights are exact in bf16)

    @classmethod
    def quantize(cls, weight: torch.Tensor) -> "Fp4Weight":
        """The NVFP4 form of weight (bf16 [rows, K]): an e4m3 scale per 16 of amax / 6 (times the global scale), e2m1
        codes of weight * global scale / scale."""
        rows, K = weight.shape
        assert rows % ATOM_ROWS == 0 and K % (BLOCK * ATOM_BLOCKS) == 0

        blocks = weight.float().view(rows, K // BLOCK, BLOCK)
        amax = float(blocks.abs().amax())
        global_scale = 2.0 ** math.floor(math.log2(E4M3_MAX * E2M1_MAX / amax)) if amax > 0 else 1.0
        scales_e4m3 = (blocks.abs().amax(-1) * (global_scale / E2M1_MAX)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        scales_fp32 = scales_e4m3.float()
        codes = cls.e2m1_codes(
            blocks * torch.where(scales_fp32 > 0, global_scale / scales_fp32, 0.0).unsqueeze(-1)
        ).view(rows, K)

        code_bytes = codes[:, 0::2] | (codes[:, 1::2] << 4)

        return cls(code_bytes.contiguous(), cls.swizzle(scales_e4m3.view(torch.uint8)), global_scale)

    @property
    def shape(self) -> tuple[int, int]:
        """The weight's [rows, K]."""
        return self.codes.shape[0], 2 * self.codes.shape[1]

    def dequantize(self) -> torch.Tensor:
        """The weights in bf16 (exact)."""
        rows, K = self.shape
        even_codes, odd_codes = self.codes & 15, self.codes >> 4
        values = torch.stack((self.e2m1_values(even_codes), self.e2m1_values(odd_codes)), -1).view(
            rows, K // BLOCK, BLOCK
        )
        scales = self.unswizzle(self.scales, rows, K // BLOCK).view(torch.float8_e4m3fn).float()

        return (values * (scales / self.global_scale).unsqueeze(-1)).view(rows, K).to(torch.bfloat16)

    @staticmethod
    def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
        """The e2m1 codes (uint8, 0..15) of fp32 x: nearest value, ties to the even code, saturating at 6."""
        magnitude = x.abs()
        bounds = torch.tensor(E2M1_BOUNDS, device=x.device)
        magnitude_index = torch.bucketize(magnitude, bounds, right=False)  # magnitude <= bound: the lower neighbour
        # a tie that landed on an odd code: the even one above
        odd_tie = (magnitude_index % 2 == 1) & torch.isin(magnitude, bounds)
        magnitude_index = magnitude_index + odd_tie.to(magnitude_index.dtype)

        return (magnitude_index | (torch.signbit(x).to(magnitude_index.dtype) << 3)).to(torch.uint8)

    @staticmethod
    def swizzle(block_scales: torch.Tensor) -> torch.Tensor:
        """Scales [rows, K / 16] in vLLM's swizzled atoms [rows / 128][K / 64][32][4][4] (flat)."""
        rows, row_blocks = block_scales.shape
        # row = 128a + 32b + c, block = 4d + e
        atoms = block_scales.view(rows // ATOM_ROWS, 4, 32, row_blocks // ATOM_BLOCKS, ATOM_BLOCKS)

        return atoms.permute(0, 3, 2, 1, 4).contiguous().view(-1)

    @staticmethod
    def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
        """fp32 values of e2m1 codes."""
        magnitude = torch.tensor(E2M1_VALUES, device=codes.device)[(codes & 7).long()]

        return torch.where((codes & 8) != 0, -magnitude, magnitude)

    @staticmethod
    def unswizzle(swizzled_scales: torch.Tensor, rows: int, row_blocks: int) -> torch.Tensor:
        """swizzle's inverse: [rows, K / 16]."""
        atoms = swizzled_scales.view(rows // ATOM_ROWS, row_blocks // ATOM_BLOCKS, 32, 4, ATOM_BLOCKS)

        return atoms.permute(0, 3, 2, 1, 4).contiguous().view(rows, row_blocks)
