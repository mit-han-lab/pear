import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision


class BatchedVisionAttention:
    """The vision attention's SDPA path with one call per run of equal-length images instead of one per image."""

    # transformers loops one attention call per image outside flash attention, which an fp32 vision tower cannot use;
    # a turn's frames share one size
    @classmethod
    def install(cls, attention_cls: type) -> None:
        """Patch the attention class's SDPA path (every other implementation stays as it is)."""
        stock_forward = attention_cls.forward

        def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
            if self.config._attn_implementation != "sdpa":
                return stock_forward(self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            return cls.forward(self, hidden_states, cu_seqlens, position_embeddings)

        attention_cls.forward = forward

    @classmethod
    def forward(cls, attention, hidden_states, cu_seqlens, position_embeddings):
        """The attention module's forward with one attention call per run of equal-size images."""
        length = hidden_states.shape[0]
        qkv = attention.qkv(hidden_states).reshape(length, 3, attention.num_heads, -1).permute(1, 0, 2, 3)
        query, key, value = qkv.unbind(0)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)

        outputs, start = [], 0

        for size, count in cls.image_runs(cu_seqlens):
            end = start + size * count
            outputs.append(cls.attend(attention, query[start:end], key[start:end], value[start:end], size, count))
            start = end

        return attention.proj(torch.cat(outputs).reshape(length, -1))

    @classmethod
    def image_runs(cls, cu_seqlens) -> list[tuple[int, int]]:
        """The runs of equal-size images between the boundaries."""
        return cls.runs((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())

    @staticmethod
    def attend(attention, query, key, value, size: int, count: int):
        """SDPA over a run of count images of size patches: [size * count, heads, d] in and out."""
        q, k, v = (t.reshape(count, size, attention.num_heads, -1).transpose(1, 2) for t in (query, key, value))
        out = F.scaled_dot_product_attention(q, k, v, scale=attention.scaling)
        return out.transpose(1, 2).reshape(size * count, attention.num_heads, -1)

    @staticmethod
    def runs(sizes: list[int]) -> list[tuple[int, int]]:
        """Consecutive equal sizes as (size, count)."""
        out = []

        for size in sizes:
            if out and out[-1][0] == size:
                out[-1] = (size, out[-1][1] + 1)
            else:
                out.append((size, 1))

        return out
