import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.rope import Fp32Rope
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend

CAPABILITY = current_platform.get_device_capability()
ON_BLACKWELL = BlackwellRolloutGpuBackend.supports_compute_capability(CAPABILITY)
# the GPUs of the rollout kernels' backends
pytestmark = pytest.mark.skipif(
    not (ON_BLACKWELL or HopperRolloutGpuBackend.supports_compute_capability(CAPABILITY)),
    reason="the rollout kernels' GPUs",
)

DEVICE = "cuda"
HEAD = 256  # the attention layers' head size
Q_HEADS, KV_HEADS = 12, 2  # their heads on one of a GB300 replica's two ranks
TOKENS = 300
EPS = 1e-6  # the q / k norms' epsilon
ROPE = {  # the model's rope_parameters
    "rope_type": "default",
    "rope_theta": 1e7,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}
ROPE_POSITIONS = 4096  # the test cache's positions (the model's: 262144)
BF16_STEP = 2**-7  # bf16's relative rounding step


@pytest.fixture
def replica_ops():
    # vLLM's custom ops as a replica's engine runs them: no torch.compile, every custom op on
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


def attention_layer(generator: torch.Generator) -> Qwen3NextAttention:
    """One full-attention layer of a replica's rank, without its projections and attention: what projects q, k and the
    gate (on Blackwell with the GPU backend's Fp32Rope)."""
    layer = Qwen3NextAttention.__new__(Qwen3NextAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads, layer.num_kv_heads, layer.head_dim = Q_HEADS, KV_HEADS, HEAD
    layer.q_size, layer.kv_size = Q_HEADS * HEAD, KV_HEADS * HEAD
    layer.attn_output_gate = True
    layer.q_norm, layer.k_norm = GemmaRMSNorm(HEAD, eps=EPS), GemmaRMSNorm(HEAD, eps=EPS)
    for norm in (layer.q_norm, layer.k_norm):
        norm.weight.data = (torch.randn(HEAD, generator=generator, device=DEVICE) * 0.1).bfloat16()
    layer.rotary_emb = get_rope(HEAD, ROPE_POSITIONS, rope_parameters=ROPE, dtype=torch.bfloat16).to(DEVICE)
    if ON_BLACKWELL:
        Fp32Rope.install(layer)

    return layer


def differing(got: torch.Tensor, want: torch.Tensor) -> str:
    """How many elements differ, and by how much at most."""
    largest = (got.float() - want.float()).abs().max().item()
    return f"{(got != want).sum().item()} of {got.numel()} differ, max |difference| {largest:.3e}"


def test_q_k_and_the_gate_come_from_the_layer_s_norms_and_rotary_embedding(replica_ops):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    layer = attention_layer(generator)
    qkv = torch.randn(TOKENS, (2 * Q_HEADS + 2 * KV_HEADS) * HEAD, generator=generator, device=DEVICE).bfloat16()
    positions = torch.randint(0, ROPE_POSITIONS, (3, TOKENS), generator=generator, device=DEVICE)  # T / H / W

    layer.use_fused_qk_norm_rope_gate = False  # as the rollout's decoder layer builds it
    q, k, _, gate = layer._project_qkv_gate(qkv, positions)
    layer.use_fused_qk_norm_rope_gate = True  # vLLM's default for this model on a GPU
    fused_q, fused_k, _, fused_gate = layer._project_qkv_gate(qkv, positions)

    # the layer's own norms, then its own rotary embedding, as the unfused path calls them
    q_gate, raw_k, _ = qkv.split([2 * layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    raw_q, want_gate = (part.reshape(TOKENS, -1) for part in q_gate.view(TOKENS, Q_HEADS, -1).chunk(2, dim=-1))
    normed_q = layer.q_norm(raw_q.reshape(-1, Q_HEADS, HEAD)).view(TOKENS, -1)
    normed_k = layer.k_norm(raw_k.reshape(-1, KV_HEADS, HEAD)).view(TOKENS, -1)
    want_q, want_k = layer.rotary_emb(positions, normed_q.clone(), normed_k.clone())
    print(f"\nfused kernel against the layer's ops: q {differing(fused_q, q)}; k {differing(fused_k, k)}")
    assert torch.equal(q, want_q) and torch.equal(k, want_k)
    assert torch.equal(gate, want_gate) and torch.equal(fused_gate, want_gate)

    if ON_BLACKWELL:
        # Fp32Rope's rotation: the normed rows rotated in fp32 on the fp32 cache, rounded once (its native path); the
        # replica's mRoPE kernel within one bf16 step of it
        fp32_q, fp32_k = layer.rotary_emb.forward_native(positions, normed_q.float(), normed_k.float())
        fp32_q, fp32_k = fp32_q.bfloat16(), fp32_k.bfloat16()
        print(f"layer against Fp32Rope's fp32 rotation: q {differing(q, fp32_q)}; k {differing(k, fp32_k)}")
        print(f"fused kernel against it: q {differing(fused_q, fp32_q)}; k {differing(fused_k, fp32_k)}")
        torch.testing.assert_close(q, fp32_q, rtol=BF16_STEP, atol=0)
        torch.testing.assert_close(k, fp32_k, rtol=BF16_STEP, atol=0)
