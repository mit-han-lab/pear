import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import qwen3_next
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.attention import Qwen3_5RolloutAttention
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.rope import Fp32Rope

CAPABILITY = current_platform.get_device_capability()
# the GPUs of the rollout kernels' backends
pytestmark = pytest.mark.skipif(
    not (
        BlackwellRolloutGpuBackend.supports_compute_capability(CAPABILITY)
        or HopperRolloutGpuBackend.supports_compute_capability(CAPABILITY)
    ),
    reason="the rollout kernels' GPUs",
)

DEVICE = "cuda"
HEAD = 256  # the attention layers' head size
# the attention's q and kv heads on one rank of a GB300 replica (TP2) and of an H200 replica (TP4)
RANK_HEADS = [pytest.param(12, 2, id="tp2"), pytest.param(6, 1, id="tp4")]
TOKENS = 300
# a step's rows: a decode step's of one and of max_num_seqs requests, a mixed step's at the mixed-step graphs' size, a
# prefill step's past every graph
STEP_TOKENS = [
    pytest.param(1, id="decode_1"),
    pytest.param(256, id="decode_256"),
    pytest.param(1024, id="mixed_1024"),
    pytest.param(4096, id="prefill_4096"),
]
# the runner's persistent T / H / W positions (max_num_batched_tokens columns), whose leading columns a step passes
POSITIONS_WIDTH = 8192
EPS = 1e-6  # the q / k norms' epsilon
ROPE = {  # the model's rope_parameters
    "rope_type": "default",
    "rope_theta": 1e7,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}
ROPE_POSITIONS = 4096  # the test cache's positions (the model's: 262144)
FP32_SLACK = 2**-18  # how far fp32 arithmetic may land from fp64, relative to the magnitude of the terms
BF16_STEP = 2**-7  # bf16's relative rounding step


@pytest.fixture
def replica_ops():
    # vLLM's custom ops as a replica's engine runs them: no torch.compile, every custom op on
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


def rollout_attention(monkeypatch, q_heads: int, kv_heads: int, generator: torch.Generator) -> Qwen3_5RolloutAttention:
    """One rank's rollout attention layer built by its own constructor over vLLM's, which is reduced to an empty module
    with its output gate unfused (as vLLM leaves it), given the model's q / k norms and the fp32 rotary embedding."""

    def unfused_module(layer, *args, **kwargs):
        torch.nn.Module.__init__(layer)
        layer.use_fused_output_gate = False

    monkeypatch.setattr(Qwen3NextAttention, "__init__", unfused_module)
    layer = Qwen3_5RolloutAttention()
    layer.num_heads, layer.num_kv_heads, layer.head_dim = q_heads, kv_heads, HEAD
    layer.q_size, layer.kv_size = q_heads * HEAD, kv_heads * HEAD
    layer.attn_output_gate = True
    layer.q_norm, layer.k_norm = GemmaRMSNorm(HEAD, eps=EPS), GemmaRMSNorm(HEAD, eps=EPS)
    for norm in (layer.q_norm, layer.k_norm):
        norm.weight.data = (torch.randn(HEAD, generator=generator, device=DEVICE) * 0.1).bfloat16()
    layer.rotary_emb = get_rope(HEAD, ROPE_POSITIONS, rope_parameters=ROPE, dtype=torch.bfloat16).to(DEVICE)
    Fp32Rope.install(layer)

    return layer


def projected_rows(
    layer: Qwen3_5RolloutAttention, tokens: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """A step's projected rows (q and gate per head, k, v) and its T / H / W positions, a view of the runner's."""
    qkv = torch.randn(tokens, 2 * layer.q_size + 2 * layer.kv_size, generator=generator, device=DEVICE).bfloat16()
    positions = torch.randint(0, ROPE_POSITIONS, (3, POSITIONS_WIDTH), generator=generator, device=DEVICE)

    return qkv, positions[:, :tokens]


def exact_heads(layer, raw: torch.Tensor, norm: GemmaRMSNorm, positions: torch.Tensor):
    """The heads' (1 + w) RMSNorm and interleaved mRoPE in fp64 on the fp32 cache, and each value's magnitude (the sum
    of its terms' absolute values)."""
    x = raw.double()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + norm.weight.double())

    # frequency i takes the H position if i % 3 == 1, the W position if i % 3 == 2 within their sections, else T's
    rope = layer.rotary_emb
    half = rope.rotary_dim // 2
    pairs = torch.arange(half, device=DEVICE)
    is_h = (pairs % 3 == 1) & (pairs < 3 * rope.mrope_section[1])
    is_w = (pairs % 3 == 2) & (pairs < 3 * rope.mrope_section[2])
    rows = torch.where(is_h, 1, torch.where(is_w, 2, 0))
    cos_sin = rope.cos_sin_cache.double()[positions]  # [T / H / W, tokens, 2 * half]
    cos, sin = cos_sin[rows, :, pairs].T[:, None], cos_sin[rows, :, half + pairs].T[:, None]

    x1, x2, tail = normed[..., :half], normed[..., half : 2 * half], normed[..., 2 * half :]
    exact = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, tail], dim=-1)
    magnitude = torch.cat([(x1 * cos).abs() + (x2 * sin).abs(), (x2 * cos).abs() + (x1 * sin).abs(), tail.abs()], -1)

    return exact.flatten(1), magnitude.flatten(1)


def beyond_one_rounding(got: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Where got lies farther from the exact value than fp32 arithmetic rounded once to bf16 can."""
    _, exponent = torch.frexp(exact)
    half_step = torch.ldexp(torch.ones_like(exact), exponent - 9)  # |exact| in [2^(e-1), 2^e): bf16's step 2^(e-8)

    return (got.double() - exact).abs() > half_step + FP32_SLACK * magnitude


@pytest.mark.usefixtures("isolated_rope_cache", "replica_ops")
@pytest.mark.parametrize("q_heads, kv_heads", RANK_HEADS)
def test_q_k_and_the_gate_are_rounded_once(monkeypatch, q_heads, kv_heads):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    layer = rollout_attention(monkeypatch, q_heads, kv_heads, generator)
    qkv, positions = projected_rows(layer, TOKENS, generator)

    q, k, v, gate = layer._project_qkv_gate(qkv, positions)

    q_gate, raw_k, want_v = qkv.split([2 * layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    raw_q, want_gate = q_gate.view(TOKENS, q_heads, 2 * HEAD).chunk(2, dim=-1)
    raw_k = raw_k.view(TOKENS, kv_heads, HEAD)
    exact_q, q_magnitude = exact_heads(layer, raw_q, layer.q_norm, positions)
    exact_k, k_magnitude = exact_heads(layer, raw_k, layer.k_norm, positions)
    assert not beyond_one_rounding(q, exact_q, q_magnitude).any()
    assert not beyond_one_rounding(k, exact_k, k_magnitude).any()
    assert torch.equal(gate, want_gate.reshape(TOKENS, -1)) and torch.equal(v, want_v)

    # the control: vLLM's unfused chain, the norms' bf16 outputs rotated by its mRoPE kernel on the same fp32 cache
    normed_q = layer.q_norm(raw_q.reshape(-1, q_heads, HEAD)).view(TOKENS, -1)
    normed_k = layer.k_norm(raw_k.reshape(-1, kv_heads, HEAD)).view(TOKENS, -1)
    chain_q, chain_k = layer.rotary_emb(positions, normed_q, normed_k)
    assert beyond_one_rounding(chain_q, exact_q, q_magnitude).any()
    assert beyond_one_rounding(chain_k, exact_k, k_magnitude).any()


@pytest.mark.usefixtures("isolated_rope_cache", "replica_ops")
@pytest.mark.parametrize("q_heads, kv_heads", RANK_HEADS)
def test_the_gated_output_is_rounded_once(monkeypatch, q_heads, kv_heads):
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    layer = rollout_attention(monkeypatch, q_heads, kv_heads, generator)
    qkv, positions = projected_rows(layer, TOKENS, generator)
    attn_output = torch.randn(TOKENS, layer.q_size, generator=generator, device=DEVICE).bfloat16()
    # the projections and the attention stand in: the layer's own forward gates what the attention returns
    layer.qkv_proj = lambda hidden_states: (qkv, None)
    layer.attn = lambda q, k, v: attn_output
    layer.o_proj = lambda gated: (gated, None)

    got = layer(positions, None)

    gate = qkv[:, : 2 * layer.q_size].view(TOKENS, q_heads, 2 * HEAD)[..., HEAD:].reshape(TOKENS, -1)
    exact = attn_output.double() * torch.sigmoid(gate.double())
    assert not beyond_one_rounding(got, exact, exact.abs()).any()
    # the control: vLLM's eager gate, sigmoid(gate) rounded to bf16 before the product
    assert beyond_one_rounding(attn_output * torch.sigmoid(gate), exact, exact.abs()).any()


@pytest.mark.usefixtures("isolated_rope_cache", "replica_ops")
@pytest.mark.parametrize("tokens", STEP_TOKENS)
@pytest.mark.parametrize("q_heads, kv_heads", RANK_HEADS)
def test_vllm_s_fused_q_k_norm_rope_kernel_never_runs(monkeypatch, q_heads, kv_heads, tokens):
    generator = torch.Generator(device=DEVICE).manual_seed(8)
    layer = rollout_attention(monkeypatch, q_heads, kv_heads, generator)
    layer.use_fused_qk_norm_rope_gate = True  # as vLLM's constructor leaves it for this model on a GPU
    qkv, positions = projected_rows(layer, tokens, generator)
    attention_inputs = []

    def attention(q, k, v):
        attention_inputs.append((q, k))
        return torch.zeros_like(q)

    def vllm_kernel(*args, **kwargs):
        raise AssertionError("vLLM's fused q / k norm + RoPE + gate kernel ran")

    # the projections and the attention stand in; vLLM's kernel, wherever its module would call it, raises
    layer.qkv_proj = lambda hidden_states: (qkv, None)
    layer.attn = attention
    layer.o_proj = lambda gated: (gated, None)
    monkeypatch.setattr(qwen3_next, "fused_qk_rmsnorm_rope_gate", vllm_kernel)

    layer(positions, None)

    # the attention took the rollout kernel's q and k
    want_q, want_k, _, _ = layer._project_qkv_gate(qkv, positions)
    ((q, k),) = attention_inputs
    assert torch.equal(q, want_q) and torch.equal(k, want_k)


@pytest.mark.usefixtures("isolated_rope_cache", "replica_ops")
@pytest.mark.parametrize("q_heads, kv_heads", RANK_HEADS)
def test_the_rotary_embedding_rotates_in_fp32(q_heads, kv_heads):
    rope = get_rope(HEAD, ROPE_POSITIONS, rope_parameters=ROPE, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.rotary_emb = rope.to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    positions = torch.randint(0, ROPE_POSITIONS, (3, TOKENS), generator=generator, device=DEVICE)  # T / H / W
    q = torch.randn(TOKENS, q_heads * HEAD, generator=generator, device=DEVICE).bfloat16()
    k = torch.randn(TOKENS, kv_heads * HEAD, generator=generator, device=DEVICE).bfloat16()
    bf16_q, _ = rope.forward_native(positions, q.clone(), k.clone())

    Fp32Rope.install(layer)

    want_q, want_k = rope.forward_native(positions, q.float(), k.float())  # fp32 throughout
    assert not torch.equal(bf16_q, want_q.bfloat16())
    # vLLM's path with custom ops off: rotated in fp32, rounded once
    got_q, got_k = rope.forward_native(positions, q.clone(), k.clone())
    assert torch.equal(got_q, want_q.bfloat16()) and torch.equal(got_k, want_k.bfloat16())
    # the mRoPE kernel (custom ops on): within one bf16 step
    got_q, got_k = rope.forward_cuda(positions, q.clone(), k.clone())
    torch.testing.assert_close(got_q, want_q.bfloat16(), rtol=BF16_STEP, atol=0)
    torch.testing.assert_close(got_k, want_k.bfloat16(), rtol=BF16_STEP, atol=0)
    assert rope.cos_sin_cache.dtype == torch.float32
