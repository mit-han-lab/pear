import json
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class StreamGemm:
    """The bf16 GEMMs x @ w^T the stream GEMM (csrc/stream_gemm.cu) serves: up to a few hundred rows of x, each weight
    byte streamed from HBM once."""

    # rows of x up to which each fused epilogue beats cuBLAS: the SwiGLU saves the activation kernel up to the verify
    # steps' largest batch; the partial sums cost the next fused norm one fp32 read each, worth it up to 128 rows
    SWIGLU_MAX_ROWS = 256
    PARTIALS_MAX_ROWS = 128
    LOGITS_MAX_ROWS = 128  # logit rows up to which the lm_head's [high rows; low rows] fit the MMA's N
    LOGITS_ROW_ALIGN = 16  # logit rows in whole TMEM loads: the low rows' columns start at a multiple of it
    TILE_ROWS = 128  # rows of w per CTA (kRows of csrc/stream_gemm.cu)
    K_BLOCK = 64  # K elements per stage (kKBlock)
    MAX_SPLITS = 4  # K ranges of the partial sums at most
    MIN_SPLIT_BLOCKS = 16  # stages per K range at least

    @classmethod
    def takes(cls, x: torch.Tensor, w: torch.Tensor, max_rows: int) -> bool:
        """Whether the stream GEMM serves x @ w^T: at most max_rows bf16 rows, K in whole stages, N in whole tiles."""
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= max_rows
            and x.dtype == torch.bfloat16
            and w.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and w.is_contiguous()
            and x.shape[1] % cls.K_BLOCK == 0
            and w.shape[0] % cls.TILE_ROWS == 0
        )

    @classmethod
    def splits(cls, x: torch.Tensor, w: torch.Tensor) -> int:
        """The K ranges of x @ w^T's partial sums: as many as fill the SMs, each of MIN_SPLIT_BLOCKS stages at least."""
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        tiles = w.shape[0] // cls.TILE_ROWS

        return max(1, min(cls.MAX_SPLITS, sm_count // tiles, x.shape[1] // cls.K_BLOCK // cls.MIN_SPLIT_BLOCKS))

    @classmethod
    def takes_logits(cls, high_low: torch.Tensor, w: torch.Tensor) -> bool:
        """Whether the stream GEMM computes the logits of [high rows; low rows] through the lm_head shard w."""
        rows = high_low.shape[0] // 2
        return (
            0 < rows <= cls.LOGITS_MAX_ROWS
            and rows % cls.LOGITS_ROW_ALIGN == 0
            and w.shape[0] % cls.TILE_ROWS == 0
            and high_low.shape[1] % cls.K_BLOCK == 0
        )

    @staticmethod
    def logits(high_low: torch.Tensor, w: torch.Tensor, rank: int) -> torch.Tensor:
        """Both ranks' logits [rows, vocab] of [high rows; low rows] through this rank's lm_head shard w, each GEMM's
        epilogue writing its shard into both gather buffers: a view valid until the next step's logits."""
        return torch.ops.pear_rollout_blackwell.sg_logits(high_low, w, high_low.shape[0] // 2, rank)


class GemmTableLinearMethod(UnquantizedLinearMethod):
    """vLLM's unquantized linear method with the bf16 GEMMs through cuBLASLt on the table's heuristic candidate per
    shape."""

    # (M, N, K, candidate) rows for Qwen3.8-27B's TP2 shapes on GB300 at the mixed steps' sizes
    TABLE = Path(__file__).resolve().parent / "gemm_table.json"

    # shapes not in the table take candidate 0. The same bf16 x bf16 -> fp32-accumulated product, another tiling
    def apply(self, layer, x, bias=None):
        if bias is None and x.dim() == 2 and x.dtype == torch.bfloat16 and layer.weight.dtype == torch.bfloat16:
            return torch.ops.pear_rollout_blackwell.lt_mm(x, layer.weight)

        return super().apply(layer, x, bias)

    @staticmethod
    def load_table() -> None:
        """Hand the table's rows to the extension's lt_mm, in this process."""
        for M, N, K, candidate in json.loads(GemmTableLinearMethod.TABLE.read_text())["rows"]:
            torch.ops.pear_rollout_blackwell.lt_set(M, N, K, candidate)


class PartialSumsLinearMethod(GemmTableLinearMethod):
    """The GEMM table's method of a row-parallel layer whose output the next fused norm reduces: at decode sizes the
    stream GEMM's fp32 split-K partial sums [splits, M, N], which the norm adds."""

    def apply(self, layer, x, bias=None):
        if bias is None and StreamGemm.takes(x, layer.weight, StreamGemm.PARTIALS_MAX_ROWS):
            return torch.ops.pear_rollout_blackwell.sg_partials(x, layer.weight, StreamGemm.splits(x, layer.weight))

        return super().apply(layer, x, bias)


class DraftGemmTableLinearMethod(UnquantizedLinearMethod):
    """The drafter's linear method: the bf16 GEMMs through cuBLASLt on a table of the drafter's own, apart from the
    target's, whose shapes it shares at other batch sizes."""

    # (M, N, K, fp32 out, candidate) rows for the MTP layer's and the lm_head's TP2 shapes on GB300; shapes not in the
    # table take candidate 0
    TABLE = Path(__file__).resolve().parent / "draft_gemm_table.json"

    def apply(self, layer, x, bias=None):
        if bias is None and x.dim() == 2 and x.dtype == torch.bfloat16 and layer.weight.dtype == torch.bfloat16:
            return torch.ops.pear_rollout_blackwell.dg_mm(x, layer.weight, False)

        return super().apply(layer, x, bias)

    @staticmethod
    def load_table() -> None:
        """Hand the table's rows to the extension's dg_mm, in this process."""
        for M, N, K, is_fp32, candidate in json.loads(DraftGemmTableLinearMethod.TABLE.read_text())["rows"]:
            torch.ops.pear_rollout_blackwell.dg_set(M, N, K, is_fp32, candidate)


class DraftPartialsLinearMethod(DraftGemmTableLinearMethod):
    """The drafter's method of a row-parallel layer whose output the next fused norm reduces: at decode sizes the
    stream GEMM's fp32 split-K partial sums [splits, M, N], which the norm adds."""

    def apply(self, layer, x, bias=None):
        if bias is None and StreamGemm.takes(x, layer.weight, StreamGemm.PARTIALS_MAX_ROWS):
            return torch.ops.pear_rollout_blackwell.sg_partials(x, layer.weight, StreamGemm.splits(x, layer.weight))

        return super().apply(layer, x, bias)
