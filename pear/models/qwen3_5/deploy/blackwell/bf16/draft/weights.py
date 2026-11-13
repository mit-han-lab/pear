from dataclasses import dataclass

import torch

# the draft's shapes the kernel is built for (csrc dk_layout.cuh): checked against the loaded model
HID, INTER, QKV_ROWS, O_K, KPROJ, SEL_ROWS = 5120, 17408, 6144, 4096, 1280, 256


@dataclass
class DraftLayer:
    """One draft decoder layer's tensors as vLLM holds them (qkv and gate_up stacked)."""

    qkv: torch.Tensor  # bf16 [6144, 5120]
    o: torch.Tensor  # bf16 [5120, 4096]
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    gate_up: torch.Tensor  # bf16 [34816, 5120]: gate rows, then up rows (the kernel's paired tiles read both halves)
    down: torch.Tensor  # bf16 [5120, 17408]
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    a_base: torch.Tensor  # attention_conv.base_kernel bf16 [2 sides, 2 taps, 5120]
    a_kp: torch.Tensor  # attention_conv.kernel_projection bf16 [1280, 5120]
    m_base: torch.Tensor
    m_kp: torch.Tensor


@dataclass
class DraftWeights:
    """Every tensor the proposal reads, in place as vLLM holds them (bf16, the checkpoint's values): the draft's, the
    target's embedding and bf16 LM head, the rotary cache."""

    NAUX = 5  # the target's aux hidden states the fc reads, of HID values each (csrc dk_layout.cuh NAUX)

    fc: torch.Tensor  # bf16 [5120, 25600]
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    sel_proj: torch.Tensor  # bf16 [256, 5120]
    pred_cb: torch.Tensor  # bf16 [vocab, 256]
    succ_cb: torch.Tensor
    embed: torch.Tensor  # bf16 [vocab, 5120]
    cos_sin: torch.Tensor  # bf16 [max_pos, 128]
    layers: list[DraftLayer]
    lm_weight: torch.Tensor  # bf16 [vocab, 5120]: the target's LM head (the same storage)

    @classmethod
    def from_vllm(cls, draft_model: torch.nn.Module, target_language_model: torch.nn.Module) -> "DraftWeights":
        """The tensors of vLLM's loaded DFlash2Qwen3ForCausalLM and of its target (shared embedding and LM head)."""
        draft = draft_model.model
        lm_head = target_language_model.lm_head
        weights = cls(
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
            lm_weight=lm_head.weight,
        )  # fmt: skip
        weights.check()

        return weights

    def projections(self) -> dict[str, torch.Tensor]:
        """The GEMVs' weights by name (bf16 [rows, K], read in place)."""
        projections = {"fc": self.fc, "sel_proj": self.sel_proj, "lm_head": self.lm_weight}
        for i, layer in enumerate(self.layers):
            projections |= {
                f"L{i}.{name}": getattr(layer, name) for name in ("a_kp", "qkv", "o", "m_kp", "gate_up", "down")
            }

        return projections

    def tensors(self) -> dict[str, torch.Tensor]:
        """Every tensor the kernel reads by name (the projections, norms, base kernels, codebooks, embedding)."""
        tensors = self.projections() | {
            "hidden_norm": self.hidden_norm,
            "norm": self.norm,
            "pred_cb": self.pred_cb,
            "succ_cb": self.succ_cb,
            "embed": self.embed,
            "cos_sin": self.cos_sin,
        }
        for i, layer in enumerate(self.layers):
            tensors |= {
                f"L{i}.{name}": getattr(layer, name)
                for name in ("q_norm", "k_norm", "in_norm", "post_norm", "a_base", "m_base")
            }

        return tensors

    def dtypes(self) -> dict[str, str]:
        """The tensors' dtypes by name (logged when the proposal is built)."""
        return {name: str(tensor.dtype).removeprefix("torch.") for name, tensor in self.tensors().items()}

    def check(self) -> None:
        """Raises a ValueError listing every tensor not in bf16 or not contiguous and every projection not of the shape
        the kernel is built for."""
        shapes = {
            "fc": (HID, self.NAUX * HID),
            "sel_proj": (SEL_ROWS, HID),
            "lm_head": (None, HID),
        }
        for i in range(len(self.layers)):
            shapes |= {
                f"L{i}.a_kp": (KPROJ, HID),
                f"L{i}.qkv": (QKV_ROWS, HID),
                f"L{i}.o": (HID, O_K),
                f"L{i}.m_kp": (KPROJ, HID),
                f"L{i}.gate_up": (2 * INTER, HID),
                f"L{i}.down": (HID, INTER),
            }

        problems = [
            f"{name}: {tensor.dtype}" for name, tensor in self.tensors().items() if tensor.dtype != torch.bfloat16
        ]
        problems += [f"{name}: not contiguous" for name, tensor in self.tensors().items() if not tensor.is_contiguous()]
        for name, (rows, k) in shapes.items():
            tensor = self.projections()[name]
            if tensor.dim() != 2 or tensor.shape[1] != k or (rows is not None and tensor.shape[0] != rows):
                problems.append(f"{name}: shape {tuple(tensor.shape)}, expected [{rows or 'vocab'}, {k}]")
        if problems:
            raise ValueError("the BF16 draft kernel needs the bf16 DFlash 2 draft as released: " + "; ".join(problems))
