import json
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


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
