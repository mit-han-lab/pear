import math
from dataclasses import dataclass

import torch

E2M1_MAX, E4M3_MAX = 6.0, 448.0  # the largest e2m1 and e4m3 magnitudes
BLOCK = 16  # weights per e4m3 scale
ATOM_ROWS, ATOM_BLOCKS = 128, 4  # vLLM's swizzled scale atom: 128 rows x 4 blocks (512 bytes, [32][4][4])
# the bounds between e2m1's neighbouring magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 (ties to the even code)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass
class Fp4Weight:
    """A projection [rows, K] in NVFP4 (the target's format): weight = e2m1 * scale / gs."""

    SIGN_SHIFT = 3  # an e2m1 code's sign bit: code = magnitude index | sign << 3
    QUANT_ELEMS = 1 << 22  # values per chunk of the quantization (bounds its fp32 and int64 temporaries)
    DEQUANT_ELEMS = 1 << 24  # weights per dequantized chunk of vLLM's draft paths (bounds its bf16 temporaries)
    ATOM_BYTES = 512  # a swizzled scale atom's bytes (ATOM_ROWS rows x ATOM_BLOCKS e4m3 scales)

    codes: torch.Tensor  # uint8 [rows, K / 2]: two e2m1 codes per byte, the even K index in the low nibble
    scales: torch.Tensor  # uint8 (e4m3) [rows / 128 * K / 64 * 512]: vLLM's swizzled atoms
    gs: float  # the global scale, a power of two (the dequantized weights are exact in bf16)

    @classmethod
    def quantize(cls, weight: torch.Tensor, gs: float | None, device: torch.device) -> "Fp4Weight":
        """weight (bf16 [rows, K], any device) in NVFP4 on the device, in row chunks, under the shared global scale gs
        or its own (None)."""
        rows, K = weight.shape
        assert rows % ATOM_ROWS == 0 and K % (BLOCK * ATOM_BLOCKS) == 0
        if gs is None:
            gs = cls.global_scale(float(weight.abs().amax()))

        codes = torch.empty(rows, K // 2, dtype=torch.uint8, device=device)
        block_scales = torch.empty(rows, K // BLOCK, dtype=torch.uint8, device=device)
        chunk_rows = max(ATOM_ROWS, cls.QUANT_ELEMS // K // ATOM_ROWS * ATOM_ROWS)
        for row_start in range(0, rows, chunk_rows):
            row_end = min(rows, row_start + chunk_rows)
            chunk_weights = weight[row_start:row_end].to(device).float().view(row_end - row_start, K // BLOCK, BLOCK)
            scales_e4m3 = (chunk_weights.abs().amax(-1) * (gs / E2M1_MAX)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
            scales_fp32 = scales_e4m3.float()
            chunk_codes = cls.e2m1_codes(
                chunk_weights * torch.where(scales_fp32 > 0, gs / scales_fp32, 0.0).unsqueeze(-1)
            ).view(row_end - row_start, K)
            codes[row_start:row_end] = chunk_codes[:, 0::2] | (chunk_codes[:, 1::2] << 4)
            block_scales[row_start:row_end] = scales_e4m3.view(torch.uint8)
            del chunk_weights, chunk_codes

        return cls(codes, cls.swizzle(block_scales), gs)

    @staticmethod
    def global_scale(amax: float) -> float:
        """The power-of-two global scale that maps amax onto the largest e2m1 x e4m3 magnitude."""
        return 2.0 ** math.floor(math.log2(E4M3_MAX * E2M1_MAX / amax)) if amax > 0 else 1.0

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """F.linear(x, the bf16 values of the weight) over row chunks of DEQUANT_ELEMS dequantized weights."""
        # (bounded temporaries: vLLM compiles and captures this path, its graphs' pools hold them)
        rows, K = self.shape
        chunk_rows = max(ATOM_ROWS, self.DEQUANT_ELEMS // K // ATOM_ROWS * ATOM_ROWS)
        if rows <= chunk_rows:
            return torch.nn.functional.linear(
                x, torch.ops.pear_deploy_rtx_blackwell_nvfp4.nvfp4_dequant(self.codes, self.scales, self.gs), bias
            )

        # (each chunk of chunk_rows rows holds whole scale atoms)
        scale_bytes_per_row = K // (BLOCK * ATOM_BLOCKS) * self.ATOM_BYTES // ATOM_ROWS
        chunk_outputs = []
        for row_start in range(0, rows, chunk_rows):
            row_end = min(rows, row_start + chunk_rows)
            chunk_weight = torch.ops.pear_deploy_rtx_blackwell_nvfp4.nvfp4_dequant(
                self.codes[row_start:row_end],
                self.scales[row_start * scale_bytes_per_row : row_end * scale_bytes_per_row],
                self.gs,
            )
            chunk_outputs.append(
                torch.nn.functional.linear(x, chunk_weight, None if bias is None else bias[row_start:row_end])
            )

        return torch.cat(chunk_outputs, -1)

    @property
    def shape(self) -> tuple[int, int]:
        """The weight's [rows, K]."""
        return self.codes.shape[0], 2 * self.codes.shape[1]

    @staticmethod
    def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
        """The e2m1 codes (uint8, 0..15) of fp32 x: nearest value, ties to the even code, saturating at 6."""
        magnitude = x.abs()
        bounds = torch.tensor(E2M1_BOUNDS, device=x.device)
        magnitude_index = torch.bucketize(magnitude, bounds, right=False)  # magnitude <= bound: the lower neighbour
        # a tie that landed on an odd code: the even one above
        odd_tie = (magnitude_index % 2 == 1) & torch.isin(magnitude, bounds)
        magnitude_index = magnitude_index + odd_tie.to(magnitude_index.dtype)

        return (magnitude_index | (torch.signbit(x).to(magnitude_index.dtype) << Fp4Weight.SIGN_SHIFT)).to(torch.uint8)

    @staticmethod
    def swizzle(block_scales: torch.Tensor) -> torch.Tensor:
        """Scales [rows, K / 16] in vLLM's swizzled atoms [rows / 128][K / 64][32][4][4] (flat)."""
        rows, row_blocks = block_scales.shape
        # row = 128a + 32b + c, block = 4d + e
        atoms = block_scales.view(rows // ATOM_ROWS, 4, 32, row_blocks // ATOM_BLOCKS, ATOM_BLOCKS)

        return atoms.permute(0, 3, 2, 1, 4).contiguous().view(-1)
