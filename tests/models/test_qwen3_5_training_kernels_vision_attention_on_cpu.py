import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

from pear.models.qwen3_5.trainer.vision_attention import BatchedVisionAttention

HIDDEN, HEADS = 64, 4
SIZES = [16, 16, 8, 16, 16, 4]  # patches per image: four runs of equal sizes


def test_batched_vision_attention_matches_the_per_image_loop(monkeypatch):
    torch.manual_seed(0)
    config = Qwen3_5VisionConfig(hidden_size=HIDDEN, num_heads=HEADS)
    config._attn_implementation = "sdpa"
    attention = Qwen3_5VisionAttention(config).eval()
    tokens = sum(SIZES)
    hidden = torch.randn(tokens, HIDDEN)
    cu_seqlens = torch.tensor([0, *torch.tensor(SIZES).cumsum(0).tolist()], dtype=torch.int32)
    angles = torch.randn(tokens, HIDDEN // HEADS // 2)
    position = (angles.cos().repeat(1, 2), angles.sin().repeat(1, 2))
    expected = attention(hidden, cu_seqlens, position_embeddings=position)  # transformers: one call per image

    monkeypatch.setattr(Qwen3_5VisionAttention, "forward", Qwen3_5VisionAttention.forward)
    BatchedVisionAttention.install(Qwen3_5VisionAttention)

    torch.testing.assert_close(attention(hidden, cu_seqlens, position_embeddings=position), expected)
    assert BatchedVisionAttention.runs(SIZES) == [(16, 2), (8, 1), (16, 2), (4, 1)]
