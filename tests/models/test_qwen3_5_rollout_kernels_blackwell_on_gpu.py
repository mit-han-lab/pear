import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flashinfer.utils import get_device_sm_count, get_trtllm_gen_multi_ctas_kv_counter_bytes
from vllm.config.cache import CacheConfig
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep, fused_sigmoid_gating_delta_rule_update
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.utils import select_common_block_size

from pear.models.qwen3_5.rollout.blackwell.attention import AttentionStep, BlackwellFlashInferImpl, KVCacheViews
from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.flashinfer_launchers import FlashInferChunk
from pear.models.qwen3_5.rollout.blackwell.gdn import BlackwellGDNMetadataBuilder, GDNCore
from pear.models.qwen3_5.rollout.blackwell.gemm import GemmTableLinearMethod
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.model import AllReduceBuffers, LogitsGatherBuffers
from pear.models.qwen3_5.rollout.blackwell.sampler import BlackwellSampler

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
TP = 2  # the Blackwell replicas' tensor parallelism
H, HV, K, V = 8, 24, 128, 128  # Qwen3.8-27B's Gated DeltaNet heads on one of two tensor-parallel ranks
C = 2 * H * K + HV * V  # conv channels: q, k, v
CONV_WIDTH = 4  # the Gated DeltaNet's causal conv taps
SLOTS = 8  # state blocks of the pool; block 0 is the padding rows' slot
EPS = 1e-6  # the gated norm's epsilon
HIDDEN = 5120  # the model's hidden size, which the fused all-reduce is built for
VOCAB = 248320
MAX_TOKENS = 8192  # the replicas' max_num_batched_tokens
MAX_ROWS = 128  # their max_num_seqs
L2_EPS = 1e-6  # vLLM's q / k l2norm epsilon, under the square root
CHUNK = 64  # tokens per chunk of FlashInfer's chunked Gated DeltaNet kernel
# PEAR's error against fp64 at most this many times vLLM's own path's on the same inputs: its largest and its mean
MAX_ERROR_RATIO = 2.0
MEAN_ERROR_RATIO = 1.25
HEAD = 256  # the attention layers' head size
Q_HEADS, KV_HEADS = 12, 2  # their heads on one of two ranks
KV_PAGES = 256  # pages of the attention tests' KV cache
MAX_CONTEXT = 2352  # the longest context of the attention tests, in tokens


def gdn_layer(generator: torch.Generator) -> types.SimpleNamespace:
    """A Gated DeltaNet layer's weights and state pools on one of two ranks."""
    shape = (SLOTS, C, CONV_WIDTH - 1) if is_conv_state_dim_first() else (SLOTS, CONV_WIDTH - 1, C)
    return types.SimpleNamespace(
        num_k_heads=TP * H,
        num_v_heads=TP * HV,
        head_k_dim=K,
        head_v_dim=V,
        tp_size=TP,
        kv_cache=(
            torch.randn(shape, generator=generator, device=DEVICE).to(torch.bfloat16),
            torch.randn(SLOTS, HV, V, K, generator=generator, device=DEVICE) * 0.1,
        ),
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(C, CONV_WIDTH, generator=generator, device=DEVICE) * 0.5).bfloat16()
        ),
        A_log=torch.randn(HV, generator=generator, device=DEVICE) * 0.5,
        dt_bias=torch.randn(HV, generator=generator, device=DEVICE).bfloat16(),
        norm=types.SimpleNamespace(weight=torch.randn(V, generator=generator, device=DEVICE).bfloat16(), eps=EPS),
    )


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)


def conv_window(conv_state: torch.Tensor) -> torch.Tensor:
    """A slot's conv state as [3, C], whichever layout vLLM keeps it in."""
    return conv_state.transpose(-1, -2) if is_conv_state_dim_first() else conv_state


def delta_rule_step(state, q, k, v, alpha, beta):
    """One token of the gated delta rule in fp64 on a [HV, V, K] state (q and k per v head): its output, the state."""
    state = state * alpha[:, None, None]
    u = (v - torch.einsum("hvk,hk->hv", state, k)) * beta[:, None]
    state = state + u[:, :, None] * k[:, None, :]
    return torch.einsum("hvk,hk->hv", state, q), state


def reference_core(layer, states, qkvz, ba, query_lens, slots, init):
    """The Gated DeltaNet core in fp64, token by token: the output rows, each sequence's conv window and state."""
    conv_state, ssm = states
    w = layer.conv1d.weight.double()
    decay = -layer.A_log.double().exp()
    qsl = np.concatenate([[0], np.cumsum(query_lens)])
    out = torch.zeros(int(qsl[-1]), HV, V, dtype=torch.float64, device=DEVICE)
    windows, finals = [], []
    for i, slot in enumerate(slots):
        window = conv_window(conv_state[slot]).double() * init[i]
        state = ssm[slot].double() * init[i]
        for t in range(qsl[i], qsl[i + 1]):
            x = qkvz[t, :C].double()
            acc = (window * w[:, :3].T).sum(0) + x * w[:, 3]
            window = torch.cat([window[1:], x[None]])
            conv = acc * torch.sigmoid(acc)
            q = l2norm(conv[: H * K].view(H, K)).repeat_interleave(HV // H, 0) * K**-0.5
            k = l2norm(conv[H * K : 2 * H * K].view(H, K)).repeat_interleave(HV // H, 0)
            alpha = (decay * F.softplus(ba[t, HV:].double() + layer.dt_bias.double())).exp()
            beta = torch.sigmoid(ba[t, :HV].double())
            o, state = delta_rule_step(state, q, k, conv[2 * H * K :].view(HV, V), alpha, beta)
            z = qkvz[t, C:].double().view(HV, V)
            out[t] = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * layer.norm.weight.double() * F.silu(z)
        windows.append(window)
        finals.append(state)
    return out, windows, finals


def vllm_core(layer, qkvz, ba, query_lens, slots, init, num_decodes) -> torch.Tensor:
    """vLLM's Gated DeltaNet core of a step without drafts, decode rows first, as the kernel calls of
    QwenGatedDeltaNetAttention._forward_core and its gated norm on the FlashInfer prefill backend: the output rows (the
    layer's states updated in place)."""
    conv_state, ssm = layer.kv_cache
    num_tokens = sum(query_lens)
    qsl = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32, device=DEVICE)
    slots = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    init = torch.tensor(init, dtype=torch.bool, device=DEVICE)
    b, a = (x.contiguous() for x in ba[:num_tokens].chunk(2, dim=-1))
    out = torch.empty(num_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)

    # every row's causal conv, rounded to bf16, the conv states as (..., C, 3)
    conv_states = conv_state if is_conv_state_dim_first() else conv_state.transpose(-1, -2)
    conv_out = causal_conv1d_fn(
        qkvz[:num_tokens, :C].transpose(0, 1), layer.conv1d.weight, None, conv_states, qsl, cache_indices=slots,
        has_initial_state=init, activation="silu",
    ).transpose(0, 1)  # fmt: skip

    # the decode rows' recurrent step on their conv'd q, k, v, normalized in the kernel
    q, k, v = (
        x.contiguous().view(1, num_decodes, -1, dim)
        for x, dim in zip(conv_out[:num_decodes].split([H * K, H * K, HV * V], dim=-1), (K, K, V), strict=True)
    )
    decode_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=layer.A_log, a=a[:num_decodes], b=b[:num_decodes], dt_bias=layer.dt_bias, q=q, k=k, v=v,
        initial_state=ssm, inplace_final_state=True, cu_seqlens=qsl[: num_decodes + 1], ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    out[:num_decodes] = decode_out.squeeze(0)

    # the prefill rows: vLLM's prep, FlashInfer's public prefill (its context-parallel kernel when few sequences)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_out[num_decodes:].contiguous(), a[num_decodes:], b[num_decodes:], layer.A_log, layer.dt_bias, H, K, V,
        apply_l2norm=True, output_g_exp=True,
    )  # fmt: skip
    _, final = fi_chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=gather_initial_states(ssm, slots[num_decodes:], init[num_decodes:]), output_final_state=True,
        cu_seqlens=qsl[num_decodes:] - qsl[num_decodes], use_qk_l2norm_in_kernel=False, output=out[num_decodes:],
    )  # fmt: skip
    scatter_states(ssm, final, slots[num_decodes:])

    # the gated RMSNorm, the norm before the gate
    gate = qkvz[:num_tokens, C:].contiguous().view(-1, V)
    layer_norm_fwd(
        out.view(-1, V), layer.norm.weight, None, EPS, z=gate, out=out.view(-1, V), norm_before_gate=True,
        is_rms_norm=True, activation="silu",
    )  # fmt: skip

    return out


def error_stats(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float, float]:
    """The largest and the mean |error| of got against fp64 want, and the mean signed error."""
    error = got.double() - want
    return error.abs().max().item(), error.abs().mean().item(), error.mean().item()


def report(title: str, stats: dict[str, tuple[float, float, float]]) -> None:
    """Print each path's error statistics side by side (pytest -s shows them)."""
    print(f"\n{title}")
    for path, (largest, mean, signed) in stats.items():
        print(f"  {path:<44} max |err| {largest:.3e}  mean |err| {mean:.3e}  mean err {signed:+.3e}")


def assert_within_vllms_error(stats: tuple[float, float, float], vllm_stats: tuple[float, float, float]) -> None:
    """PEAR's error against fp64 at most MAX_ERROR_RATIO (largest) and MEAN_ERROR_RATIO (mean) times vLLM's own."""
    assert stats[0] <= MAX_ERROR_RATIO * vllm_stats[0], (stats, vllm_stats)
    assert stats[1] <= MEAN_ERROR_RATIO * vllm_stats[1], (stats, vllm_stats)


def test_a_decode_step_matches_the_reference():
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    layer = gdn_layer(generator)
    rows = 4
    qkvz = torch.randn(rows + 1, C + HV * V, generator=generator, device=DEVICE).bfloat16()
    ba = torch.randn(rows + 1, 2 * HV, generator=generator, device=DEVICE).bfloat16()
    slots = torch.tensor([3, 1, 5, 2, 0], dtype=torch.int32, device=DEVICE)  # the last row pads (slot 0)
    md = types.SimpleNamespace(num_actual_tokens=rows + 1, non_spec_state_indices_tensor=slots)
    initial = [s.clone() for s in layer.kv_cache]
    want, windows, finals = reference_core(layer, initial, qkvz, ba, [1] * rows, slots[:rows].tolist(), [1] * rows)
    out = torch.empty(rows + 1, HV, V, dtype=torch.bfloat16, device=DEVICE)

    GDNCore.decode_step(layer, qkvz, ba, out, md)

    torch.cuda.synchronize()
    torch.testing.assert_close(out[:rows].double(), want, rtol=2e-2, atol=2e-2)
    assert torch.equal(out[rows], torch.zeros_like(out[rows]))
    for i, slot in enumerate(slots[:rows].tolist()):
        assert torch.equal(conv_window(layer.kv_cache[0][slot]).double(), windows[i])
        torch.testing.assert_close(layer.kv_cache[1][slot].double(), finals[i], rtol=1e-4, atol=1e-5)
    # the padding row's slot keeps its states
    assert torch.equal(layer.kv_cache[0][0], initial[0][0]) and torch.equal(layer.kv_cache[1][0], initial[1][0])


@pytest.fixture
def gdn_builder(monkeypatch) -> BlackwellGDNMetadataBuilder:
    # vLLM's GDN builder reduced to the setting the Blackwell builder's path reads: no drafter
    def vllm_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.use_spec_decode = False

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", vllm_init)
    scheduler_config = types.SimpleNamespace(max_num_seqs=MAX_ROWS, max_num_batched_tokens=MAX_TOKENS)

    return BlackwellGDNMetadataBuilder(
        None, [], types.SimpleNamespace(scheduler_config=scheduler_config), torch.device(DEVICE)
    )


def common_metadata(query_lens: list[int], contexts: list[int], slots: list[int]) -> CommonAttentionMetadata:
    """A step's common attention metadata as vLLM's V2 runner gives it to a GDN group's builder."""
    qsl = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    seq_lens = torch.tensor(np.add(contexts, query_lens), dtype=torch.int32)
    num_tokens = int(qsl[-1])

    return CommonAttentionMetadata(
        query_start_loc=qsl.to(DEVICE),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.to(DEVICE),
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.tensor(slots, dtype=torch.int32, device=DEVICE)[:, None],
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE),
        is_prefilling=torch.tensor([query_len > 1 for query_len in query_lens]),
        seq_lens_cpu_upper_bound=seq_lens,
    )


def test_a_mixed_step_s_error_is_vllms(gdn_builder):
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    layer = gdn_layer(generator)
    # three decode rows, then prompts' chunks over a cached prefix: a new turn's observation, exactly a chunk, two
    # tokens (fewer than the conv window); and a prompt's first chunk, without a state
    query_lens = [1, 1, 1, 32 * CHUNK + 86, CHUNK, 2, 37]
    contexts = [30, 7, 500, 120, 64, 9, 0]
    slots = [3, 1, 5, 2, 7, 6, 4]
    num_decodes = 3
    init = [int(context > 0) for context in contexts]
    num_tokens = sum(query_lens)
    qkvz = torch.randn(num_tokens, C + HV * V, generator=generator, device=DEVICE).bfloat16()
    ba = torch.randn(num_tokens, 2 * HV, generator=generator, device=DEVICE).bfloat16()
    initial = [s.clone() for s in layer.kv_cache]
    want, windows, finals = reference_core(layer, initial, qkvz, ba, query_lens, slots, init)
    layout = gdn_builder.build(0, common_metadata(query_lens, contexts, slots))._pear_layout

    # PEAR twice from the same states: the chunk kernel's first call compiles through FlashInfer's wrapper, the second
    # launches the compiled function as a replica's steps do; then vLLM's own path
    outs, states = {}, {}
    for path in ("PEAR, FlashInfer's wrapper", "PEAR, its compiled call", "vLLM"):
        for state, start in zip(layer.kv_cache, initial, strict=True):
            state.copy_(start)
        if path == "vLLM":
            outs[path] = vllm_core(layer, qkvz, ba, query_lens, slots, init, num_decodes)
        else:
            outs[path] = torch.empty(num_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)
            GDNCore.mixed_step(layer, qkvz, ba, outs[path], layout)
        torch.cuda.synchronize()
        states[path] = [s.clone() for s in layer.kv_cache]

    pear, vllm = "PEAR, its compiled call", "vLLM"
    want_states = torch.stack(finals)
    stats = {}
    for rows, part in ((slice(0, num_decodes), "decode rows"), (slice(num_decodes, num_tokens), "prefill rows")):
        stats[part] = {path: error_stats(out[rows], want[rows]) for path, out in outs.items()}
        report(f"a mixed step's outputs, {part}", stats[part])
    stats["states"] = {path: error_stats(pool[1][slots], want_states) for path, pool in states.items()}
    report("a mixed step's recurrent states", stats["states"])

    assert torch.equal(outs["PEAR, FlashInfer's wrapper"], outs[pear])
    for part_stats in stats.values():
        assert_within_vllms_error(part_stats[pear], part_stats[vllm])
    for i, slot in enumerate(slots):
        assert torch.equal(conv_window(states[pear][0][slot]).double(), windows[i])


def reference_chunks(q, k, v, alpha, beta, initial, query_lens, scale):
    """The gated delta rule over packed sequences in fp64, token by token: the outputs, each sequence's final state."""
    qsl = np.concatenate([[0], np.cumsum(query_lens)])
    out = torch.zeros(int(qsl[-1]), HV, V, dtype=torch.float64, device=DEVICE)
    finals = []
    for i in range(len(query_lens)):
        state = initial[i].double()
        for t in range(qsl[i], qsl[i + 1]):
            qt = q[t].double().repeat_interleave(HV // H, 0) * scale
            kt = k[t].double().repeat_interleave(HV // H, 0)
            out[t], state = delta_rule_step(state, qt, kt, v[t].double(), alpha[t].double(), beta[t].double())
        finals.append(state)
    return out, torch.stack(finals)


@pytest.mark.parametrize(
    "query_lens, init",
    [
        # one sequence: vLLM's call of FlashInfer's public API takes its context-parallel kernel (few sequences x heads
        # for the GPU's SMs), PEAR's call the chunked kernel
        ([64 * CHUNK + 50], [1]),
        # three: both take the chunked kernel; one without a cached state
        ([32 * CHUNK + 86, 16 * CHUNK, 37], [1, 1, 0]),
    ],
)
def test_the_chunk_kernel_s_error_is_vllms(query_lens, init):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    scale = K**-0.5
    num_tokens, num_seqs = sum(query_lens), len(query_lens)
    cu_seqlens = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32, device=DEVICE)
    q = l2norm(torch.randn(num_tokens, H, K, generator=generator, device=DEVICE)).bfloat16()
    k = l2norm(torch.randn(num_tokens, H, K, generator=generator, device=DEVICE)).bfloat16()
    v = torch.randn(num_tokens, HV, V, generator=generator, device=DEVICE).bfloat16()
    alpha = torch.sigmoid(torch.randn(num_tokens, HV, generator=generator, device=DEVICE) + 3)  # decays near 1
    beta = torch.sigmoid(torch.randn(num_tokens, HV, generator=generator, device=DEVICE))
    cached = torch.tensor(init, dtype=torch.bool, device=DEVICE)[:, None, None, None]
    initial = torch.where(cached, torch.randn(num_seqs, HV, V, K, generator=generator, device=DEVICE) * 0.1, 0)
    want, want_final = reference_chunks(q, k, v, alpha, beta, initial, query_lens, scale)

    # PEAR's call twice: the first through FlashInfer's wrapper, which compiles, the second on the compiled function;
    # then vLLM's own call of FlashInfer's public API
    chunk = FlashInferChunk()
    runs = {}
    for path in ("PEAR, FlashInfer's wrapper", "PEAR, its compiled call"):
        out, final = torch.empty_like(v), torch.empty_like(initial)
        chunk(q, k, v, alpha, beta, out, cu_seqlens, initial, final, scale)
        runs[path] = out, final
    vllm_out = torch.empty_like(v)
    _, vllm_final = fi_chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=alpha[None], beta=beta[None], initial_state=initial,
        output_final_state=True, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False, output=vllm_out,
    )  # fmt: skip
    runs["vLLM (use_cp auto)"] = vllm_out, vllm_final
    torch.cuda.synchronize()

    out_stats = {path: error_stats(out, want) for path, (out, _) in runs.items()}
    state_stats = {path: error_stats(final, want_final) for path, (_, final) in runs.items()}
    report(f"the chunk kernel's outputs, sequences of {query_lens} tokens", out_stats)
    report(f"the chunk kernel's final states, sequences of {query_lens} tokens", state_stats)

    (wrapper_out, wrapper_final), (out, final) = runs["PEAR, FlashInfer's wrapper"], runs["PEAR, its compiled call"]
    assert torch.equal(wrapper_out, out) and torch.equal(wrapper_final, final)
    for stats in (out_stats, state_stats):
        assert_within_vllms_error(stats["PEAR, its compiled call"], stats["vLLM (use_cp auto)"])


def attention_page() -> int:
    """The tokens per page of a GB300 replica's attention kernels as vLLM derives them for Qwen3.8-27B at TP2:
    the hybrid KV cache block that holds a Gated DeltaNet state (Platform._align_hybrid_block_size from FlashInfer's
    preferred block), split into the largest page FlashInfer's kernels take (select_common_block_size)."""
    attention_bytes = FullAttentionSpec(
        block_size=1, num_kv_heads=KV_HEADS, head_size=HEAD, dtype=torch.bfloat16
    ).page_size_bytes
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(TP, TP * H, TP * HV, K, V, CONV_WIDTH)
    mamba_bytes = MambaSpec(shapes=shapes, dtypes=(torch.bfloat16, torch.float32), block_size=-1).page_size_bytes
    preferred_block = FlashInferBackend.get_preferred_block_size(CacheConfig.DEFAULT_BLOCK_SIZE)
    hybrid_block = preferred_block * cdiv(mamba_bytes, preferred_block * attention_bytes)

    return select_common_block_size(hybrid_block, [FlashInferBackend])


def attention_impl() -> BlackwellFlashInferImpl:
    """An attention layer's impl as a replica builds it, reduced to what attend() reads."""
    impl = BlackwellFlashInferImpl.__new__(BlackwellFlashInferImpl)
    impl.views, impl.head_size = KVCacheViews(), HEAD
    impl.bmm1_scale, impl.bmm2_scale = HEAD**-0.5, 1.0
    # vLLM's multi-CTA semaphores at a replica's size (they cover the decode kernel's boosted split)
    sm_count = get_device_sm_count(torch.device(DEVICE, torch.cuda.current_device()))
    impl._multi_ctas_kv_counter_buffer = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(MAX_TOKENS, Q_HEADS, sm_count), dtype=torch.uint8, device=DEVICE
    )

    return impl


def attention_metadata(seq_lens, query_lens, num_decodes, tables, decode_max_seq_len) -> types.SimpleNamespace:
    """The FlashInfer metadata attend() reads: the decode rows (one query token each), then the prefill rows."""
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    prefill_lens = query_lens[num_decodes:]
    decode = prefill = None
    if num_decodes:
        decode = types.SimpleNamespace(
            block_tables=tables[:num_decodes], seq_lens=lens[:num_decodes], max_seq_len=decode_max_seq_len
        )
    if prefill_lens:
        prefill = types.SimpleNamespace(
            block_tables=tables[num_decodes:],
            seq_lens=lens[num_decodes:],
            max_q_len=max(prefill_lens),
            max_seq_len=max(seq_lens[num_decodes:]),
            cum_seq_lens_q=torch.tensor([0, *np.cumsum(prefill_lens)], dtype=torch.int32, device=DEVICE),
            cum_seq_lens_kv=torch.tensor([0, *np.cumsum(seq_lens[num_decodes:])], dtype=torch.int32, device=DEVICE),
        )

    return types.SimpleNamespace(
        num_decode_tokens=num_decodes,
        num_actual_tokens=sum(query_lens),
        num_prefills=len(prefill_lens),
        kv_cache_layout=KVCacheLayout.LBHNC,
        decode=decode,
        prefill=prefill,
    )


def reference_attention(query, kv_cache, tables, seq_lens, query_lens) -> torch.Tensor:
    """Causal attention in fp64 of each row's queries, its last query_len positions, over its pages."""
    page, group = kv_cache.shape[2], Q_HEADS // KV_HEADS
    out, start = [], 0
    for table, seq_len, query_len in zip(tables.tolist(), seq_lens, query_lens, strict=True):
        pages = kv_cache[table[: -(-seq_len // page)]]  # [pages, KV heads, page, K | V]
        rows = pages.transpose(1, 2).reshape(-1, KV_HEADS, 2 * HEAD)[:seq_len].double()
        q = query[start : start + query_len].double().view(query_len, KV_HEADS, group, HEAD)
        scores = torch.einsum("qhgd,khd->hgqk", q, rows[..., :HEAD]) * HEAD**-0.5
        positions = torch.arange(seq_len - query_len, seq_len, device=DEVICE)
        scores = scores.masked_fill(torch.arange(seq_len, device=DEVICE) > positions[:, None], float("-inf"))
        o = torch.einsum("hgqk,khd->qhgd", scores.softmax(-1), rows[..., HEAD:])
        out.append(o.reshape(query_len, Q_HEADS, HEAD))
        start += query_len
    return torch.cat(out)


@pytest.mark.parametrize("num_decodes", [5, AttentionStep.SORT_MIN_ROWS + 8])  # the decode rows in order, or sorted
def test_the_attention_of_a_mixed_step_matches_the_reference(num_decodes):
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    rng = np.random.default_rng(6)
    page = attention_page()
    kv_cache = torch.randn(KV_PAGES, KV_HEADS, page, 2 * HEAD, generator=generator, device=DEVICE).bfloat16()
    # decode rows: one token, a page and a token past it, other contexts; then prompts' chunks: a first chunk, over a
    # cached prefix ending inside a page and at its end
    decode_lens = [1, page, page + 1, *rng.integers(2, MAX_CONTEXT, num_decodes - 3).tolist()]
    query_lens = [1] * num_decodes + [100, 37, 64]
    seq_lens = decode_lens + [100, 900 + 37, page + 64]
    tables = torch.randint(0, KV_PAGES, (len(seq_lens), cdiv(MAX_CONTEXT, page)), generator=generator, device=DEVICE)
    query = (torch.randn(sum(query_lens), Q_HEADS, HEAD, generator=generator, device=DEVICE) * 2).bfloat16()
    md = attention_metadata(seq_lens, query_lens, num_decodes, tables.int(), max(decode_lens))
    out = torch.full_like(query, 7.0)

    attention_impl().attend(query, kv_cache, md, out)

    torch.cuda.synchronize()
    want = reference_attention(query, kv_cache, tables, seq_lens, query_lens)
    torch.testing.assert_close(out.double(), want, rtol=0, atol=2e-2)


def test_a_decode_graph_replays_the_attention_of_longer_contexts():
    # a FULL decode graph records the step's arguments: the longest-first order and its sorted lengths are computed
    # again at every replay, from the lengths vLLM's runner updates in place
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    rng = np.random.default_rng(7)
    page = attention_page()
    num_decodes = AttentionStep.SORT_MIN_ROWS + 8
    kv_cache = torch.randn(KV_PAGES, KV_HEADS, page, 2 * HEAD, generator=generator, device=DEVICE).bfloat16()
    seq_lens = rng.integers(1, MAX_CONTEXT, num_decodes).tolist()
    query_lens = [1] * num_decodes
    tables = torch.randint(0, KV_PAGES, (num_decodes, cdiv(MAX_CONTEXT, page)), generator=generator, device=DEVICE)
    query = (torch.randn(num_decodes, Q_HEADS, HEAD, generator=generator, device=DEVICE) * 2).bfloat16()
    out = torch.empty_like(query)
    impl = attention_impl()
    # captured, as vLLM captures its decode graphs, at the longest context a replay may have
    impl.attend(query, kv_cache, attention_metadata(seq_lens, query_lens, num_decodes, tables.int(), MAX_CONTEXT), out)
    md = attention_metadata(seq_lens, query_lens, num_decodes, tables.int(), MAX_CONTEXT)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.attend(query, kv_cache, md, out)

    md.decode.seq_lens.add_(1)
    query.copy_((torch.randn(num_decodes, Q_HEADS, HEAD, generator=generator, device=DEVICE) * 2).bfloat16())
    graph.replay()

    torch.cuda.synchronize()
    want = reference_attention(query, kv_cache, tables, [n + 1 for n in seq_lens], query_lens)
    torch.testing.assert_close(out.double(), want, rtol=0, atol=2e-2)


def two_rank_checks(rank: int, rendezvous: str) -> None:
    """The fused all-reduce + fp32 residual + RMSNorm, the partial-embedding norm and the logits gather on one rank."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, device_id=device)
    group = dist.group.WORLD
    BlackwellExtension.load()
    # held like the model holds them: the extension keeps only the buffers' addresses
    AllReduceBuffers.create(group, MAX_TOKENS, HIDDEN, device)
    LogitsGatherBuffers.create(group, MAX_ROWS, VOCAB, device)
    gather = LogitsGatherBuffers.instance
    generator = torch.Generator(device=device).manual_seed(1234)  # the same draws on both ranks
    w = (0.1 * torch.randn(HIDDEN, generator=generator, device=device)).bfloat16()

    # a sequence of token counts: rows a larger call wrote must not leak into the later ones
    for T in (848, 64, 1, 1000, 7, 64, MAX_TOKENS, 300, 64):
        residual = torch.randn(T, HIDDEN, generator=generator, device=device) * 3
        partials = [torch.randn(T, HIDDEN, generator=generator, device=device).bfloat16() for _ in range(2)]
        out, new_residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
            partials[rank].clone(), residual, w, EPS, False
        )
        out32, _ = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(partials[rank].clone(), residual, w, EPS, True)
        want = residual.double() + partials[0].double() + partials[1].double()
        want_out = want * torch.rsqrt(want.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.double())
        torch.testing.assert_close(new_residual.double(), want, rtol=0, atol=1e-4)
        torch.testing.assert_close(out.double(), want_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out32.double(), want_out, rtol=1e-5, atol=1e-5)
        peer = new_residual.clone()
        dist.broadcast(peer, 0)
        assert torch.equal(peer, new_residual), "the ranks' residuals differ"

    # the first layer: each token's embedding row is nonzero on one rank, so the residual is exactly the full row
    for T in (64, 848, 7):
        full = torch.randn(T, HIDDEN, generator=generator, device=device).bfloat16()
        owner = torch.randint(0, 2, (T, 1), generator=generator, device=device)
        out, residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(torch.where(owner == rank, full, 0), w, EPS)
        assert torch.equal(residual, full.float())
        want_out = full.double() * torch.rsqrt(full.double().pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.double())
        torch.testing.assert_close(out.double(), want_out, rtol=1e-2, atol=1e-2)

    # the logits: this rank's vocabulary shard as high and low rows, summed and gathered into [n, vocab]
    shard = slice(rank * VOCAB // 2, (rank + 1) * VOCAB // 2)
    for n in (65, 1, MAX_ROWS, 33):
        high = torch.randn(n, VOCAB, generator=generator, device=device)
        low = torch.randn(n, VOCAB, generator=generator, device=device) * 1e-3
        logits = torch.ops.pear_rollout_blackwell.lg_gather(
            torch.cat([high[:, shard], low[:, shard]]).contiguous(), n, gather.rank
        )
        torch.cuda.synchronize()
        assert torch.equal(logits, high + low)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the fused all-reduce exchanges between two GPUs")
def test_the_fused_all_reduce_and_the_logits_gather_on_two_ranks(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(two_rank_checks, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def test_the_linear_layers_gemms_match_an_fp32_product():
    BlackwellExtension.load()
    GemmTableLinearMethod.load_table()
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    method = GemmTableLinearMethod()
    # the MLP gate/up at 896 rows (a measured shape), a shape the table leaves to candidate 0, a row-strided input
    for M, N, K, pad in ((896, 17408, HIDDEN, 0), (7, 8192, HIDDEN, 0), (64, HIDDEN, 3072, 8)):
        x = torch.randn(M, K + pad, generator=generator, device=DEVICE).bfloat16()[:, :K]
        w = (torch.randn(N, K, generator=generator, device=DEVICE) * K**-0.5).bfloat16()

        out = method.apply(types.SimpleNamespace(weight=w), x)

        assert out.dtype == torch.bfloat16 and torch.equal(out, torch.ops.pear_rollout_blackwell.lt_mm(x, w))
        torch.testing.assert_close(out.float(), F.linear(x.float(), w.float()), rtol=1e-2, atol=1e-2)


def test_the_fused_sampler_draws_the_race_with_vllms_log_probs():
    BlackwellExtension.load()
    sampler = BlackwellSampler.__new__(BlackwellSampler)  # draw() reads its constants alone
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    k = BlackwellSampler.MAX_LOGPROBS
    # a step's rows, one row at a low temperature, all rows at a high one, integer logits (ties: radix select)
    for rows, temperature, scale, integers in (
        (65, 1.0, 3.0, False),
        (1, 0.7, 2.0, False),
        (MAX_ROWS, 1.3, 8.0, False),
        (8, 1.0, 2.0, True),
    ):
        logits = torch.randn(rows, VOCAB, generator=generator, device=DEVICE) * scale
        logits = logits.round() if integers else logits
        temperatures = torch.full((rows,), temperature, device=DEVICE)

        torch.cuda.manual_seed(rows)  # the same noise for both
        sampled, logprobs = sampler.draw(logits.clone(), temperatures, k)
        torch.cuda.manual_seed(rows)
        noise = torch.empty_like(logits).exponential_()

        # the race over torch's exponential noise as vLLM's V1 sampler ran it, vLLM's top-K of the processed logits
        processed = logits / temperature
        want_sampled = processed.softmax(-1).div(noise).argmax(-1)
        want = compute_topk_scores(processed, k, want_sampled)
        assert sampled.dtype == torch.int64 and torch.equal(sampled, want_sampled)
        assert torch.equal(logprobs.selected_token_ranks, want.selected_token_ranks)
        torch.testing.assert_close(logprobs.logprobs, want.logprobs, rtol=0, atol=1e-4)
        if not integers:  # tied log-probs may list their tokens in another order
            assert torch.equal(logprobs.logprob_token_ids.long(), want.logprob_token_ids)
