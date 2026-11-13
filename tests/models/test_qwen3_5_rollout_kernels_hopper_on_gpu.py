import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_kernel_o
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum_scalar_kernel
from vllm.third_party.flash_linear_attention.ops.solve_tril import (
    FLA_TRIL_PRECISION,
    merge_16x16_to_64x64_inverse_kernel,
)
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd_kernel

from pear.models.qwen3_5.rollout.hopper.all_reduce import DecodeAllReduce
from pear.models.qwen3_5.rollout.hopper.attention import (
    HEAD_DIM,
    QUERY_HEADS,
    HopperFlashAttentionImpl,
    HopperFlashAttentionMetadataBuilder,
)
from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperMixedStepGraphs
from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
from pear.models.qwen3_5.rollout.hopper.gdn import CHUNK, FusedGDNDecode, GDNLayout, MixedStepGDNCore
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.sampler import HopperSampler
from verl.utils.net_utils import get_free_port

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not HopperRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="HopperRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
TP = 4  # the Hopper replicas' tensor parallelism
H, HV, K, V = 4, 12, 128, 128  # Qwen3.8-27B's Gated DeltaNet heads on one of four tensor-parallel ranks
C = 2 * H * K + HV * V  # conv channels: q, k, v
QKVZ = C + HV * V  # in_proj_qkvz's outputs (q, k, v, z); in_proj_ba's b and a follow them in the merged projection
SLOTS = 8  # state slots of the pool; slot 0 is the null block, the padding rows'
EPS = 1e-6  # the gated norm's and the RMSNorm's epsilon
L2_EPS = 1e-6  # vLLM's q / k l2norm epsilon, under the square root
GRAPH_ROWS = 256  # the token count a mixed-step graph pads the test batch to
KV_BLOCKS = 200  # pages of the attention test's KV cache
PAGE = HopperFlashAttentionMetadataBuilder.PAGE  # tokens per page
HIDDEN = 5120  # the model's hidden size
VOCAB = 248320
NORM_BIAS = 1.0  # Qwen3.5's RMSNorm scales by 1 + w (FlashInfer's weight_bias)
LOCALHOST = "127.0.0.1"
SAMPLER_ROWS = 64  # a decode step's rows
MAX_LOGPROBS = HopperSampler.MAX_LOGPROBS  # the largest top-K the sampler's kernels keep
DRAWS = 50  # sampler calls of the distribution check, over 256 rows each
SAMPLER_SEED = 5  # the seed of the sampler's draws in its test


def gdn_layer(generator: torch.Generator) -> types.SimpleNamespace:
    """A Gated DeltaNet layer's weights and state pools on one of four ranks, the conv state as [slots, 3, C]."""
    return types.SimpleNamespace(
        prefix="language_model.model.layers.0.linear_attn",
        num_k_heads=TP * H,
        num_v_heads=TP * HV,
        head_k_dim=K,
        head_v_dim=V,
        tp_size=TP,
        kv_cache=(
            torch.randn(SLOTS, 3, C, generator=generator, device=DEVICE).bfloat16(),
            torch.randn(SLOTS, HV, V, K, generator=generator, device=DEVICE) * 0.1,
        ),
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(C, 1, 4, generator=generator, device=DEVICE) * 0.5).bfloat16()
        ),
        A_log=torch.randn(HV, generator=generator, device=DEVICE) * 0.5,
        dt_bias=torch.randn(HV, generator=generator, device=DEVICE).bfloat16(),
        norm=types.SimpleNamespace(weight=torch.randn(V, generator=generator, device=DEVICE).bfloat16(), eps=EPS),
    )


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)


def reference_core(layer, states, proj, qsl, slots, init):
    """The Gated DeltaNet core in fp64, token by token: the output rows, each sequence's conv window and state."""
    conv_state, ssm = states
    w = layer.conv1d.weight.view(C, 4).double()
    decay = -layer.A_log.double().exp()
    out = torch.zeros(int(qsl[-1]), HV, V, dtype=torch.float64, device=DEVICE)
    windows, finals = [], []
    for i in range(len(qsl) - 1):
        window = conv_state[slots[i]].double() if init[i] else torch.zeros(3, C, dtype=torch.float64, device=DEVICE)
        state = ssm[slots[i]].double() if init[i] else torch.zeros(HV, V, K, dtype=torch.float64, device=DEVICE)
        for t in range(qsl[i], qsl[i + 1]):
            x = proj[t, :C].double()
            acc = (window * w[:, :3].T).sum(0) + x * w[:, 3]
            window = torch.cat([window[1:], x[None]])
            conv = acc * torch.sigmoid(acc)
            q = l2norm(conv[: H * K].view(H, K)).repeat_interleave(HV // H, 0) * K**-0.5
            k = l2norm(conv[H * K : 2 * H * K].view(H, K)).repeat_interleave(HV // H, 0)
            v = conv[2 * H * K :].view(HV, V)
            b, a = proj[t, QKVZ : QKVZ + HV].double(), proj[t, QKVZ + HV :].double()
            state = state * (decay * F.softplus(a + layer.dt_bias.double())).exp()[:, None, None]
            u = (v - torch.einsum("hvk,hk->hv", state, k)) * torch.sigmoid(b)[:, None]
            state = state + u[:, :, None] * k[:, None, :]
            o = torch.einsum("hvk,hk->hv", state, q)
            z = proj[t, C:QKVZ].double().view(HV, V)
            out[t] = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * layer.norm.weight.double() * F.silu(z)
        windows.append(window)
        finals.append(state)
    return out, windows, finals


def test_a_decode_step_matches_the_reference():
    HopperExtension.load()
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    layer = gdn_layer(generator)
    rows = 5
    proj = torch.randn(rows + 1, QKVZ + 2 * HV, generator=generator, device=DEVICE).bfloat16()
    slots = torch.tensor([3, 1, 5, 2, 7, 0], dtype=torch.int32, device=DEVICE)  # the last row pads (the null block)
    md = types.SimpleNamespace(num_actual_tokens=rows + 1, non_spec_state_indices_tensor=slots)
    initial = [s.clone() for s in layer.kv_cache]
    out = torch.full((rows + 1, HV, V), 7.0, dtype=torch.bfloat16, device=DEVICE)

    FusedGDNDecode.decode_step(layer, md, proj[:, :QKVZ], proj[:, QKVZ:], out)

    torch.cuda.synchronize()
    want, windows, finals = reference_core(layer, initial, proj, np.arange(rows + 1), slots.tolist(), [1] * rows)
    torch.testing.assert_close(out[:rows].double(), want, rtol=2e-2, atol=2e-2)
    assert not out[rows].any()
    for i, slot in enumerate(slots[:rows].tolist()):
        assert torch.equal(layer.kv_cache[0][slot].double(), windows[i])
        torch.testing.assert_close(layer.kv_cache[1][slot].double(), finals[i], rtol=2e-2, atol=2e-3)
    assert torch.equal(layer.kv_cache[0][0], initial[0][0]) and torch.equal(layer.kv_cache[1][0], initial[1][0])


def test_a_mixed_step_matches_the_reference_eagerly_and_from_a_graph():
    HopperExtension.load()
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    layer = gdn_layer(generator)
    # three decode rows, then prefills: a chunk and a tail, exactly a chunk, a short one without a cached state
    lens, init, slots, decodes = [1, 1, 1, CHUNK + 86, CHUNK, 37], [1, 1, 1, 1, 1, 0], [3, 1, 5, 2, 6, 4], 3
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    T, n, P, _ = GDNLayout.needs(qsl, decodes)
    proj = torch.randn(GRAPH_ROWS, QKVZ + 2 * HV, generator=generator, device=DEVICE).bfloat16()
    initial = [s.clone() for s in layer.kv_cache]
    device = torch.device(DEVICE, torch.cuda.current_device())
    seq_slots = torch.tensor(slots, dtype=torch.int32, device=DEVICE)

    # eager: a layout of the batch's own size
    batch_layout = GDNLayout(T, n, P, H, HV, K, V, device)
    batch_layout.fill(qsl, decodes, np.array(init, dtype=np.int32))
    out = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=DEVICE)
    MixedStepGDNCore.mixed_step(
        layer, batch_layout, seq_slots, batch_layout.seq_init, proj[:T, :QKVZ], proj[:T, QKVZ:], out
    )
    torch.cuda.synchronize()
    eager_states = [s.clone() for s in layer.kv_cache]

    want, windows, finals = reference_core(layer, initial, proj, qsl, slots, init)
    torch.testing.assert_close(out.double(), want, rtol=5e-2, atol=5e-2)
    assert (out.double() - want).abs().mean() < 5e-3
    for i, slot in enumerate(slots):
        assert torch.equal(layer.kv_cache[0][slot].double(), windows[i])
        torch.testing.assert_close(layer.kv_cache[1][slot].double(), finals[i], rtol=5e-2, atol=5e-3)

    # a graph captured from an empty batch over the graph layout's capacities (room for every request as a prefill),
    # replayed with the batch's layout and state slots
    for state, start in zip(layer.kv_cache, initial, strict=True):
        state.copy_(start)
    graph_layout = GDNLayout(HopperMixedStepGraphs.MAX_TOKENS, SLOTS, SLOTS, H, HV, K, V, device)
    graph_layout.fill(np.zeros(1, dtype=np.int64), 0, np.zeros(0, dtype=np.int32))
    graph_slots = torch.zeros(SLOTS, dtype=torch.int32, device=DEVICE)
    graph_out = torch.zeros(GRAPH_ROWS, HV, V, dtype=torch.bfloat16, device=DEVICE)

    def step():
        MixedStepGDNCore.mixed_step(
            layer, graph_layout, graph_slots, graph_layout.seq_init, proj[:, :QKVZ], proj[:, QKVZ:], graph_out
        )

    step()  # the kernels' first launches compile them; an empty layout touches nothing
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph_layout.fill(qsl, decodes, np.array(init, dtype=np.int32))
    graph_slots[:n].copy_(seq_slots)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(graph_out[:T], out) and not graph_out[T:].any()
    assert all(torch.equal(got, ref) for got, ref in zip(layer.kv_cache, eager_states, strict=True))


def fla_chunks(scratch, lay, NT, initial, scale):
    """The prefill rows' outputs and final states from FLA's own kernels on MixedStepGDNCore's inputs and configs."""
    T, P = lay.T, lay.P
    cu, ci, co = lay.pf_cu, lay.chunk_indices, lay.chunk_offsets
    g_cum, A = torch.empty_like(scratch.g), torch.empty(T, HV, CHUNK, device=DEVICE)
    Ai = torch.zeros(T, HV, CHUNK, dtype=torch.bfloat16, device=DEVICE)
    w, u, v_new, o = (torch.empty_like(x) for x in (scratch.w, scratch.u, scratch.u, scratch.o))
    h, final = torch.empty(NT, HV, V, K, dtype=torch.bfloat16, device=DEVICE), torch.empty_like(initial)
    # FLA's JIT functions under their heuristics and autotuners
    chunk_local_cumsum_scalar_kernel.fn.fn[(NT, HV)](
        scratch.g, g_cum, cu, ci, T, B=1, H=HV, BT=CHUNK, REVERSE=False, IS_VARLEN=True, HEAD_FIRST=False, num_warps=1
    )
    chunk_scaled_dot_kkt_fwd_kernel.fn.fn[(NT, HV)](
        scratch.k, scratch.beta, g_cum, A, cu, ci, T, H=HV, Hg=H, K=K, BT=CHUNK, BK=128, IS_VARLEN=True, USE_G=True,
        CAST_DOT_TO_K_DTYPE=False, num_warps=8, num_stages=2,
    )  # fmt: skip
    merge_16x16_to_64x64_inverse_kernel.fn.fn[(NT, HV)](
        A, Ai, cu, ci, T, H=HV, BT=CHUNK, USE_TMA=False, DOT_PRECISION=FLA_TRIL_PRECISION, IS_VARLEN=True, num_warps=4,
        num_stages=3,
    )  # fmt: skip
    recompute_w_u_fwd_kernel.fn.fn[(NT, HV)](
        scratch.k, scratch.v, scratch.beta, w, u, Ai, g_cum, cu, ci, T, H=HV, Hg=H, K=K, V=V, BT=CHUNK, BK=64, BV=64,
        IS_VARLEN=True, num_warps=4, num_stages=3,
    )  # fmt: skip
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn[(V // 32, P * HV)](
        scratch.k, u, w, v_new, g_cum, None, h, initial, final, cu, co, T, H=HV, Hg=H, K=K, V=V, BT=CHUNK, BV=32,
        USE_G=True, USE_GK=False, USE_INITIAL_STATE=True, STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True, IS_VARLEN=True,
        USE_EXP2=False, num_warps=4, num_stages=2,
    )  # fmt: skip
    chunk_fwd_kernel_o.fn.fn[(V // 64, NT, HV)](
        scratch.q, scratch.k, v_new, h, g_cum, o, cu, ci, scale, T, H=HV, Hg=H, K=K, V=V, BT=CHUNK, BK=128, BV=64,
        USE_G=True, IS_VARLEN=True, num_warps=4, num_stages=4,
    )  # fmt: skip
    return o, final


def test_the_chunk_kernels_equal_flas():
    HopperExtension.load()
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    layer = gdn_layer(generator)
    # two decode rows, then prefills: two chunks and a tail, exactly a chunk, a short one without a cached state
    lens, init, slots, decodes = [1, 1, CHUNK + 86, CHUNK, 37], [1, 1, 1, 1, 0], [3, 1, 2, 6, 4], 2
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    T, n, P, NT = GDNLayout.needs(qsl, decodes)
    proj = torch.randn(T, QKVZ + 2 * HV, generator=generator, device=DEVICE).bfloat16()
    device = torch.device(DEVICE, torch.cuda.current_device())
    seq_slots = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    prefills = torch.tensor(slots[decodes:], device=DEVICE)
    cached = torch.tensor(init[decodes:], device=DEVICE)[:, None, None, None]
    initial = torch.where(cached.bool(), layer.kv_cache[1][prefills], 0)

    lay = GDNLayout(T, n, P, H, HV, K, V, device)
    lay.fill(qsl, decodes, np.array(init, dtype=np.int32))
    out = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=DEVICE)
    MixedStepGDNCore.mixed_step(layer, lay, seq_slots, lay.seq_init, proj[:, :QKVZ], proj[:, QKVZ:], out)
    want, final = fla_chunks(lay.scratch, lay, NT, initial, K**-0.5)
    torch.cuda.synchronize()

    assert torch.equal(lay.scratch.o[decodes:T], want[decodes:T])
    assert torch.equal(layer.kv_cache[1][prefills], final)


def test_the_split_decode_attention_matches_the_reference():
    HopperExtension.load()
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    rng = np.random.default_rng(2)
    kv = torch.randn(KV_BLOCKS, PAGE, 1, 2 * HEAD_DIM, generator=generator, device=DEVICE).bfloat16()
    state = HopperFlashAttentionImpl.device_state(kv.device)
    scale = HEAD_DIM**-0.5
    # one token, page multiples and their neighbours, a long sequence beside short ones, a pure decode step
    cases = [[1], [PAGE, 2 * PAGE, PAGE - 1, PAGE + 1], [40000, 1, 1, 7], rng.integers(1, 3000, 48).tolist()]

    for lens in cases:
        n = len(lens)
        pages = [(length + PAGE - 1) // PAGE for length in lens]
        order = rng.permutation(KV_BLOCKS)
        table = torch.zeros(n, max(pages) + 2, dtype=torch.int32)
        cache = kv.clone()
        for i, length in enumerate(lens):
            table[i, : pages[i]] = torch.from_numpy(order[sum(pages[:i]) : sum(pages[: i + 1])])
            # the rows past the sequence's end in its last page are stale
            cache[int(table[i, pages[i] - 1]), length - (pages[i] - 1) * PAGE :] = float("nan")
        table = table.to(DEVICE)
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
        q = (torch.randn(n, QUERY_HEADS + 2, HEAD_DIM, generator=generator, device=DEVICE) * 2).bfloat16()[
            :, 1 : QUERY_HEADS + 1
        ]
        want = []
        for i, length in enumerate(lens):
            rows = cache[table[i, : pages[i]].long(), :, 0].reshape(-1, 2 * HEAD_DIM)[:length].double()
            weights = torch.softmax(q[i].double() @ rows[:, :HEAD_DIM].T * scale, -1)
            want.append(weights @ rows[:, HEAD_DIM:])
        want = torch.stack(want)

        # all rows decoding, or the leading half (a mixed step's: FlashAttention-3 computes the others)
        for decodes in (n, n // 2):
            for ctas in (HopperFlashAttentionImpl.MIXED_CTAS, HopperFlashAttentionImpl.DECODE_CTAS):
                out = torch.full((n, QUERY_HEADS, HEAD_DIM), 7.0, dtype=torch.bfloat16, device=DEVICE)
                num_decode_rows = torch.tensor([decodes], dtype=torch.int32, device=DEVICE)
                torch.ops.pear_rollout_hopper.split_decode_attention(
                    q, cache, table, seq_lens, num_decode_rows, out, state["part_o"], state["part_ml"],
                    state["counters"], scale, ctas, n,
                )  # fmt: skip
                torch.cuda.synchronize()
                torch.testing.assert_close(out[:decodes].double(), want[:decodes], rtol=0, atol=2e-2)
                assert (out[decodes:] == 7.0).all() and not state["counters"].any()


def four_rank_checks(rank: int, port: int) -> None:
    """The decode all-reduce + residual + RMSNorm on one of four ranks."""
    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(TP, rank, f"tcp://{LOCALHOST}:{port}", rank)
        initialize_model_parallel(tensor_model_parallel_size=TP)
    HopperExtension.load()
    ops = torch.ops.pear_rollout_hopper
    generator = torch.Generator(device=DEVICE).manual_seed(6)  # the same draws on every rank
    text_config = types.SimpleNamespace(hidden_size=HIDDEN)
    model = types.SimpleNamespace(config=types.SimpleNamespace(get_text_config=lambda: text_config))
    DecodeAllReduce.create_workspace(model)
    workspace = DecodeAllReduce.workspace
    buffers = (
        int(workspace.uc_ptrs_dev),
        int(workspace.uc_ptr_local),
        int(workspace.mc_ptr),
        DecodeAllReduce.flags,
        int(workspace.rank),
    )
    gamma = (0.1 * torch.randn(HIDDEN, generator=generator, device=DEVICE)).bfloat16()
    scale = gamma.double() + NORM_BIAS

    # the bf16 residual: the reduced rows (fp32 sums over the ranks in rank order, rounded) added in bf16
    for T in (1, 7, 64, 128):
        residual = (3 * torch.randn(T, HIDDEN, generator=generator, device=DEVICE)).bfloat16()
        partials = [torch.randn(T, HIDDEN, generator=generator, device=DEVICE).bfloat16() for _ in range(TP)]
        want = (residual.float() + sum(p.float() for p in partials).bfloat16().float()).bfloat16()
        norm_out = torch.empty_like(residual)
        ops.ar_norm(partials[rank].clone(), residual, gamma, norm_out, *buffers, EPS, NORM_BIAS)
        torch.cuda.synchronize()
        assert torch.equal(residual, want)
        rms = want.double() * torch.rsqrt(want.double().pow(2).mean(-1, keepdim=True) + EPS) * scale
        torch.testing.assert_close(norm_out.double(), rms, rtol=1e-2, atol=1e-2)

    # a decode step's fp32 stream: the first layer starts it from the bf16 residual, which stays as it was
    for T in (7, 64):
        residual = (3 * torch.randn(T, HIDDEN, generator=generator, device=DEVICE)).bfloat16()
        before = residual.clone()
        stream = residual.float()
        norm_out = torch.empty_like(residual)
        for layer in range(3):
            partials = [torch.randn(T, HIDDEN, generator=generator, device=DEVICE).bfloat16() for _ in range(TP)]
            stream = stream + sum(p.float() for p in partials).bfloat16().float()
            ops.ar_norm(
                partials[rank].clone(), residual, gamma, norm_out, *buffers, EPS, NORM_BIAS, DecodeAllReduce.stream32,
                layer == 0,
            )  # fmt: skip
            torch.cuda.synchronize()
            assert torch.equal(DecodeAllReduce.stream32[:T], stream) and torch.equal(residual, before)
            rms = stream.double() * torch.rsqrt(stream.double().pow(2).mean(-1, keepdim=True) + EPS) * scale
            torch.testing.assert_close(norm_out.double(), rms, rtol=1e-2, atol=1e-2)

    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.skipif(torch.cuda.device_count() < TP, reason="the decode all-reduce runs over four GPUs")
def test_the_decode_all_reduce_on_four_ranks():
    torch.multiprocessing.spawn(four_rank_checks, args=(get_free_port(LOCALHOST)[0],), nprocs=TP)


def test_the_sampler_draws_from_the_softmax(monkeypatch):
    HopperExtension.load()
    # the draws' seed and counter, which the sampler keeps for the process: the same draws whatever ran before
    monkeypatch.setattr(HopperSampler, "seed", SAMPLER_SEED)
    monkeypatch.setattr(HopperSampler, "calls", 0)
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    temperature = torch.full((SAMPLER_ROWS,), 0.8, device=DEVICE)
    # rows whose mass sits on a few dozen tokens (the draw inside the top-K), flat rows (its rank counted), one row
    peaked = torch.randn(SAMPLER_ROWS, VOCAB, generator=generator, device=DEVICE) * 2
    hot = torch.randint(0, VOCAB, (SAMPLER_ROWS, 40), generator=generator, device=DEVICE)
    peaked.scatter_(1, hot, torch.randn(SAMPLER_ROWS, 40, generator=generator, device=DEVICE) * 3 + 12)
    flat = torch.randn(SAMPLER_ROWS, VOCAB, generator=generator, device=DEVICE) * 2

    for logits in (peaked, flat, peaked[:1]):
        result = HopperSampler.sample(logits, temperature, MAX_LOGPROBS)
        tensors = result.logprobs_tensors
        ids, logprobs, ranks = tensors.logprob_token_ids, tensors.logprobs, tensors.selected_token_ranks
        sampled = result.sampled_token_ids[:, 0].long()
        lp = (logits / temperature[: len(logits), None]).log_softmax(-1)
        top_lp, top_ids = lp.topk(MAX_LOGPROBS, -1)
        sampled_lp = lp.gather(1, sampled[:, None])
        assert torch.equal(ids[:, 0].long(), sampled) and torch.equal(ids[:, 1:].long(), top_ids)
        torch.testing.assert_close(logprobs[:, 0], sampled_lp[:, 0], rtol=0, atol=1e-4)
        torch.testing.assert_close(logprobs[:, 1:], top_lp, rtol=0, atol=1e-4)
        # a log-prob within an fp32 step of the draw's counts either way
        assert ((ranks - (lp >= sampled_lp).sum(-1)).abs() <= 1).all()

    # the frequencies of a five-token distribution over many draws
    probs = torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03], device=DEVICE)
    logits = torch.full((256, VOCAB), -30.0, device=DEVICE)
    logits[:, : len(probs)] = probs.log()
    counts = torch.zeros(len(probs), device=DEVICE)
    for _ in range(DRAWS):
        sampled = HopperSampler.sample(logits, torch.ones(256, device=DEVICE), MAX_LOGPROBS).sampled_token_ids[:, 0]
        assert (sampled < len(probs)).all()
        counts += torch.bincount(sampled.long(), minlength=len(probs)).float()
    draws = 256 * DRAWS
    assert ((counts / draws - probs).abs() < 4 * (probs * (1 - probs) / draws).sqrt()).all()
