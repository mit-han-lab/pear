import torch
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pear.models.qwen3_5.trainer.residual_stream import Fp32ResidualStream

VOCAB, HIDDEN, HEAD_DIM, TOKENS = 64, 128, 64, 12
LAYERS = 2  # full-attention layers: the residual adds and the rotary embedding, without the Gated DeltaNet's kernels
ROPE = {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [11, 11, 10], "mrope_interleaved": True}


def test_the_residual_stream_stays_fp32_while_the_attention_takes_bf16(monkeypatch):
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        layer_types=["full_attention"] * LAYERS,
        rope_parameters={**ROPE, "partial_rotary_factor": 1.0},
    )
    config._attn_implementation = "sdpa"
    model = modeling_qwen3_5.Qwen3_5TextModel(config).to(torch.bfloat16).eval()
    monkeypatch.setattr(modeling_qwen3_5, "apply_rotary_pos_emb", modeling_qwen3_5.apply_rotary_pos_emb)
    Fp32ResidualStream.install(model)
    layer_inputs = []
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda module, args: layer_inputs.append(args[0].dtype))

    # SDPA would reject fp32 queries/keys next to bf16 values, so this forward also checks the rounding after RoPE
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = model(input_ids=torch.randint(0, VOCAB, (1, TOKENS)), use_cache=False).last_hidden_state

    assert layer_inputs == [torch.float32] * LAYERS
    assert hidden.dtype == torch.float32
