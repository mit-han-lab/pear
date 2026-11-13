from dataclasses import dataclass

import torch

from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.quant import Fp4Weight
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.schedule import TILE_M


@dataclass
class DraftLayer:
    """One draft decoder layer's tensors as vLLM holds them (qkv and gate_up stacked)."""

    qkv: torch.Tensor  # bf16 [6144, 5120]
    o: torch.Tensor  # bf16 [5120, 4096]
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    gate_up: torch.Tensor  # bf16 [34816, 5120]: gate rows, then up rows
    down: torch.Tensor  # bf16 [5120, 17408]
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    a_base: torch.Tensor  # attention_conv.base_kernel bf16 [2 sides, 2 taps, 5120]
    a_kp: torch.Tensor  # attention_conv.kernel_projection bf16 [1280, 5120]
    m_base: torch.Tensor
    m_kp: torch.Tensor


@dataclass
class DraftWeights:
    """Every tensor the proposal reads: the draft's, the target's embedding and NVFP4 LM head, the rotary cache."""

    fc: torch.Tensor  # bf16 [5120, 25600]
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    sel_proj: torch.Tensor  # bf16 [256, 5120]
    pred_cb: torch.Tensor  # bf16 [vocab, 256]
    succ_cb: torch.Tensor
    embed: torch.Tensor  # bf16 [vocab, 5120]
    cos_sin: torch.Tensor  # bf16 [max_pos, 128]
    layers: list[DraftLayer]
    lm_weight: torch.Tensor  # uint8 [vocab, 2560]: packed e2m1
    lm_block_scales: torch.Tensor  # e4m3 [vocab, 320] in vLLM's swizzled layout
    lm_alpha: float
    lm_input_global_scale_inv: float  # the input global scale (a multiplier)

    @classmethod
    def from_vllm(cls, draft_model: torch.nn.Module, target_language_model: torch.nn.Module) -> "DraftWeights":
        """The tensors of vLLM's loaded DFlash2Qwen3ForCausalLM and of its target (shared embedding and LM head)."""
        draft = draft_model.model
        lm_head = target_language_model.lm_head

        return cls(
            fc=draft.fc.weight, hidden_norm=draft.hidden_norm.weight, norm=draft.norm.weight,
            sel_proj=draft.candidate_selector.hidden_projection.weight,
            pred_cb=draft.candidate_selector.predecessor_codebook,
            succ_cb=draft.candidate_selector.successor_codebook,
            embed=draft.embed_tokens.weight, cos_sin=draft.layers[0].self_attn.rotary_emb.cos_sin_cache,
            layers=[
                DraftLayer(
                    qkv=layer.self_attn.qkv_proj.weight, o=layer.self_attn.o_proj.weight,
                    q_norm=layer.self_attn.q_norm.weight, k_norm=layer.self_attn.k_norm.weight,
                    gate_up=layer.mlp.gate_up_proj.weight, down=layer.mlp.down_proj.weight,
                    in_norm=layer.input_layernorm.weight, post_norm=layer.post_attention_layernorm.weight,
                    a_base=layer.attention_conv.base_kernel, a_kp=layer.attention_conv.kernel_projection.weight,
                    m_base=layer.mlp_conv.base_kernel, m_kp=layer.mlp_conv.kernel_projection.weight,
                )
                for layer in draft.layers
            ],
            lm_weight=lm_head.weight, lm_block_scales=lm_head.weight_scale, lm_alpha=float(lm_head.alpha),
            lm_input_global_scale_inv=float(lm_head.input_global_scale_inv),
        )  # fmt: skip

    def quantized(self) -> dict[str, Fp4Weight]:
        """Every projection in NVFP4 (the target's format), quantized once, rows in the kernel's order."""
        fp4_weights = {}
        for name, weight in self.projections().items():
            order = self.row_order(name, weight)
            fp4_weights[name] = Fp4Weight.quantize(weight if order is None else weight[order])

        return fp4_weights

    def write_back(self, fp4_weights: dict[str, Fp4Weight]) -> None:
        """The projections replaced in place by their 4-bit values (vLLM's own draft paths run the same weights)."""
        for name, weight in self.projections().items():
            values = fp4_weights[name].dequantize()
            order = self.row_order(name, weight)
            if order is None:
                weight.copy_(values)
            else:
                weight[order] = values

    def projections(self) -> dict[str, torch.Tensor]:
        """The draft's projections by name (bf16 [rows, K]): the ones its 4-bit form quantizes."""
        projections = {"fc": self.fc, "sel_proj": self.sel_proj}
        for i, layer in enumerate(self.layers):
            projections |= {
                f"L{i}.{name}": getattr(layer, name) for name in ("a_kp", "qkv", "o", "m_kp", "gate_up", "down")
            }

        return projections

    @staticmethod
    def row_order(name: str, weight: torch.Tensor) -> torch.Tensor | None:
        """The kernel's row order of a projection (gate_up: the paired tiles of draft/csrc's E_SILU), None for
        vLLM's."""
        if not name.endswith("gate_up"):
            return None

        # each M tile holds TILE_M / 2 gate rows, then the up rows of the same outputs
        num_gate_rows = weight.shape[0] // 2
        j = torch.arange(weight.shape[0], device=weight.device)
        tile, row = j // TILE_M, j % TILE_M
        gate_row = tile * (TILE_M // 2) + row % (TILE_M // 2)

        return torch.where(row < TILE_M // 2, gate_row, num_gate_rows + gate_row)
