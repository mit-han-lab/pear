import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flashinfer.activation import silu_and_mul
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gdn.core import GDNCore
from pear.models.qwen3_5.rollout.blackwell.gdn.layer import BlackwellGDNMetadataBuilder
from pear.models.qwen3_5.rollout.blackwell.gemm import GemmTableLinearMethod, PartialSumsLinearMethod, StreamGemm
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.model import AllReduceBuffers, BlackwellQwen3_5MLP, LogitsGatherBuffers

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
HIDDEN = 5120  # the model's hidden size, which the fused all-reduce is built for
INTERMEDIATE = 8704  # the MLP's intermediate channels on one of two ranks
MAX_TOKENS = 8192  # the replicas' max_num_batched_tokens
MAX_ROWS = 128  # their max_num_seqs
EPS = 1e-6  # the norms' epsilon
WEIGHT_SCALE = 0.05  # the random weights' scale
BF16_ULP = 2**-7  # a bf16 value's last mantissa bit, relative to its exponent
ACCUMULATION_TOLERANCE = 1e-5  # the fp32 accumulation's error, relative to sum |x w|
SPLIT_COUNTS = (2, 3)  # the stream GEMM's split-K ranges an all-reduce adds
# x rows, w rows, K of the stream GEMM's epilogues: one CTA and CTA pairs (from 65 rows of x over an even number of
# tiles), an odd number of tiles, the MLP's and the output projections' shapes at TP2
SWIGLU_SHAPES = [(16, 1024, 5120), (64, 2048, 5120), (77, 2048, 2048), (256, 1024, 5120), (128, 17408, 5120)]
PARTIALS_SHAPES = [(16, 5120, 3072), (64, 5120, 8704), (100, 384, 2048), (128, 5120, 3072), (130, 1024, 2048)]
# the stream GEMM's lm_head: logit rows of one CTA and of CTA pairs (from 65 rows of [high; low]), a rank's shard rows
HEAD_ROWS = (16, 48, 128)
HEAD_SHARD_ROWS = 1024
# Qwen3.8-27B's Gated DeltaNet heads on one of two tensor-parallel ranks
H, HV, K, V = 8, 24, 128, 128
C = 2 * H * K + HV * V  # conv channels: q, k, v
CONV_WIDTH = 4  # the Gated DeltaNet's causal conv taps
CHUNK = 64  # tokens per chunk of the chunked Gated DeltaNet kernel


def products(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ w^T in fp64 and the sum of |x w| each entry adds up, the scale of the fp32 accumulation's error."""
    return x.double() @ w.double().t(), x.double().abs() @ w.double().abs().t()


@pytest.mark.parametrize("M, N, K", SWIGLU_SHAPES)
def test_the_stream_gemm_s_swiglu_is_within_a_bf16_rounding_of_fp64(M, N, K):
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(M + N)
    x = torch.randn(M, K, generator=generator, device=DEVICE).bfloat16()
    w = (torch.randn(N, K, generator=generator, device=DEVICE) * WEIGHT_SCALE).bfloat16()
    assert StreamGemm.takes(x, w, StreamGemm.SWIGLU_MAX_ROWS)

    out = torch.ops.pear_rollout_blackwell.sg_swiglu(x, w)

    product, magnitude = products(x, w)
    gate, up = product.chunk(2, dim=-1)
    gate_magnitude, up_magnitude = magnitude.chunk(2, dim=-1)
    want = F.silu(gate) * up
    # the fp32 sums' errors through silu(gate) * up, then the bf16 rounding of the result
    scale = (gate_magnitude * up_magnitude).sqrt() * (gate.abs() + up.abs()).clamp_min(1e-3)
    ulp = want.abs().clamp_min(1e-30).log2().floor().exp2() * BF16_ULP
    bound = 0.5 * ulp + ACCUMULATION_TOLERANCE * scale
    assert out.shape == (M, N // 2) and out.dtype == torch.bfloat16
    assert ((out.double() - want).abs() / bound).max().item() <= 1


@pytest.mark.parametrize("M, N, K", PARTIALS_SHAPES)
def test_the_stream_gemm_s_partial_sums_add_up_to_fp64(M, N, K):
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(M + N)
    x = torch.randn(M, K, generator=generator, device=DEVICE).bfloat16()
    w = (torch.randn(N, K, generator=generator, device=DEVICE) * WEIGHT_SCALE).bfloat16()

    for splits in SPLIT_COUNTS:
        out = torch.ops.pear_rollout_blackwell.sg_partials(x, w, splits)

        want, magnitude = products(x, w)
        assert out.shape == (splits, M, N) and out.dtype == torch.float32
        assert ((out.double().sum(0) - want).abs() / (ACCUMULATION_TOLERANCE * magnitude)).max().item() <= 1


def test_the_row_parallel_layers_hand_their_next_norm_partial_sums_at_decode_sizes():
    BlackwellExtension.load()
    GemmTableLinearMethod.load_table()
    generator = torch.Generator(DEVICE).manual_seed(1)
    weight = (torch.randn(HIDDEN, 3072, generator=generator, device=DEVICE) * WEIGHT_SCALE).bfloat16()
    layer = types.SimpleNamespace(weight=weight)
    method = PartialSumsLinearMethod()

    for M in (1, StreamGemm.PARTIALS_MAX_ROWS, StreamGemm.PARTIALS_MAX_ROWS + 1):
        x = torch.randn(M, 3072, generator=generator, device=DEVICE).bfloat16()
        out = method.apply(layer, x)

        want = F.linear(x.float(), layer.weight.float())
        if M <= StreamGemm.PARTIALS_MAX_ROWS:
            assert out.dim() == 3 and out.dtype == torch.float32
            torch.testing.assert_close(out.sum(0), want, rtol=1e-4, atol=1e-4)
        else:
            assert torch.equal(out, torch.ops.pear_rollout_blackwell.lt_mm(x, layer.weight))


class StandInProjection(torch.nn.Module):
    """A linear layer as vLLM's return their output: (x @ w^T in bf16, no bias)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x @ self.weight.t(), None


@pytest.mark.parametrize("M", [1, StreamGemm.SWIGLU_MAX_ROWS, StreamGemm.SWIGLU_MAX_ROWS + 1])
def test_the_mlp_takes_the_stream_gemm_s_swiglu_up_to_its_rows_and_the_rollout_mlp_s_above(M):
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(M)
    x = torch.randn(M, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    weight = (torch.randn(2 * INTERMEDIATE, HIDDEN, generator=generator, device=DEVICE) * WEIGHT_SCALE).bfloat16()
    # the projections stand in: gate_up a bf16 GEMM of the weight, down hands its input back
    mlp = BlackwellQwen3_5MLP.__new__(BlackwellQwen3_5MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = StandInProjection(weight)
    mlp.down_proj = lambda activation: (activation, None)

    got = mlp(x)

    # each path rounds silu(gate) * up to bf16 once: the stream GEMM's epilogue from its fp32 sums, FlashInfer's
    # kernel from the projection's bf16 output (vLLM's own rounds silu(gate) first)
    if M <= StreamGemm.SWIGLU_MAX_ROWS:
        assert torch.equal(got, torch.ops.pear_rollout_blackwell.sg_swiglu(x, weight))
    else:
        assert torch.equal(got, silu_and_mul(x @ weight.t()))


def test_the_head_s_high_and_low_rows_are_the_torch_split():
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(2)
    # a decode step's rows, and a verify step's past the kernel's grid (its grid-stride loop)
    for rows in (1, 64, 512):
        x = torch.randn(rows, HIDDEN, generator=generator, device=DEVICE) * 3

        high_low = torch.ops.pear_rollout_blackwell.hl_split(x)

        high = x.bfloat16()
        assert torch.equal(high_low[:rows], high)
        assert torch.equal(high_low[rows:], (x - high.float()).bfloat16())


def split_k_checks(rank: int, rendezvous: str) -> None:
    """The fused all-reduce + fp32 residual + RMSNorm of the stream GEMM's fp32 split-K partials on one rank, onto a
    residual and as a layer's first sum: the partials added in their order and rounded to bf16, as the bf16 partial's
    path takes them."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, device_id=device)
    BlackwellExtension.load()
    AllReduceBuffers.create(dist.group.WORLD, MAX_TOKENS, HIDDEN, device)
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    w = (0.1 * torch.randn(HIDDEN, generator=generator, device=device)).bfloat16()

    for T in (1, 64, MAX_ROWS, 7, 300):
        for splits in SPLIT_COUNTS:
            residual = torch.randn(T, HIDDEN, generator=generator, device=device) * 3
            partials = torch.randn(splits, T, HIDDEN, generator=generator, device=device)
            summed = partials[0].clone()
            for part in partials[1:]:
                summed += part
            for out_f32 in (False, True):
                out, new_residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(partials, residual, w, EPS, out_f32)
                want_out, want_residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
                    summed.bfloat16(), residual, w, EPS, out_f32
                )
                assert torch.equal(out, want_out) and torch.equal(new_residual, want_residual), (rank, T, splits)
            out, first_residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(partials, w, EPS)
            want_out, want_residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(summed.bfloat16(), w, EPS)
            assert torch.equal(out, want_out) and torch.equal(first_residual, want_residual), (rank, T, splits)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the fused all-reduce exchanges between two GPUs")
def test_the_fused_all_reduce_adds_the_split_k_partials_as_one_bf16_partial(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(split_k_checks, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def head_checks(rank: int, rendezvous: str) -> None:
    """The stream GEMM's lm_head on one of two ranks: both ranks' logits of [high rows; low rows] in this rank's gather
    buffer, each the sum of its high and its low row's products within the fp32 accumulation's error."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, device_id=device)
    BlackwellExtension.load()
    LogitsGatherBuffers.create(dist.group.WORLD, max(HEAD_ROWS), 2 * HEAD_SHARD_ROWS, device)
    generator = torch.Generator(device=device).manual_seed(4321)  # the same weights and states on both ranks
    w = (torch.randn(2 * HEAD_SHARD_ROWS, HIDDEN, generator=generator, device=device) * WEIGHT_SCALE).bfloat16()
    shard = w[rank * HEAD_SHARD_ROWS : (rank + 1) * HEAD_SHARD_ROWS]
    for rows in HEAD_ROWS:
        x = torch.randn(rows, HIDDEN, generator=generator, device=device) * 3
        high_low = torch.ops.pear_rollout_blackwell.hl_split(x)

        assert StreamGemm.takes_logits(high_low, shard)
        logits = StreamGemm.logits(high_low, shard, rank).clone()

        want_high, scale_high = products(high_low[:rows], w)
        want_low, scale_low = products(high_low[rows:], w)
        error = (logits.double() - (want_high + want_low)).abs()
        assert (error <= ACCUMULATION_TOLERANCE * (scale_high + scale_low)).all(), (rank, rows)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the head's epilogue writes both GPUs' gather buffers")
def test_the_stream_gemm_s_head_writes_both_ranks_logits_of_the_high_and_low_rows(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(head_checks, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def gdn_layer(generator: torch.Generator, slots: int) -> types.SimpleNamespace:
    """A Gated DeltaNet layer's weights and state pools on one of two ranks."""
    shape = (slots, C, CONV_WIDTH - 1) if is_conv_state_dim_first() else (slots, CONV_WIDTH - 1, C)
    return types.SimpleNamespace(
        num_k_heads=2 * H,
        num_v_heads=2 * HV,
        head_k_dim=K,
        head_v_dim=V,
        tp_size=2,
        kv_cache=(
            torch.randn(shape, generator=generator, device=DEVICE).to(torch.bfloat16),
            torch.randn(slots, HV, V, K, generator=generator, device=DEVICE) * 0.1,
        ),
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(C, CONV_WIDTH, generator=generator, device=DEVICE) * 0.5).bfloat16()
        ),
        A_log=torch.randn(HV, generator=generator, device=DEVICE) * 0.5,
        dt_bias=torch.randn(HV, generator=generator, device=DEVICE).bfloat16(),
        norm=types.SimpleNamespace(weight=torch.randn(V, generator=generator, device=DEVICE).bfloat16(), eps=EPS),
    )


def mixed_step_builder(monkeypatch) -> BlackwellGDNMetadataBuilder:
    """The Blackwell GDN builder of an engine without a drafter: vLLM's builder state that a mixed step's build
    reads."""

    def vllm_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.use_spec_decode = False

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", vllm_init)
    scheduler_config = types.SimpleNamespace(max_num_seqs=MAX_ROWS, max_num_batched_tokens=MAX_TOKENS)

    return BlackwellGDNMetadataBuilder(
        None, [], types.SimpleNamespace(scheduler_config=scheduler_config), torch.device(DEVICE)
    )


def mixed_step_metadata(query_lens, contexts, slots, num_decodes) -> CommonAttentionMetadata:
    """A mixed step's common attention metadata as vLLM's V2 runner gives it to a GDN group's builder: its decode rows
    first, every row's state slot."""
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
        is_prefilling=torch.arange(len(query_lens)) >= num_decodes,
        seq_lens_cpu_upper_bound=seq_lens,
    )


def test_the_gdn_core_takes_the_merged_projection_s_row_strides(monkeypatch):
    # the layer's in_proj_qkvz and in_proj_ba rows come out of one GEMM: views of its [tokens, qkvz | ba] output. A
    # mixed step (decode rows, then prompt chunks) and a decode step on those views and on contiguous copies
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    prefill_lens, prefill_contexts = [3 * CHUNK + 86, 37], [120, 0]
    decode_contexts = [30, 70, 9]
    num_decodes = len(decode_contexts)
    query_lens = [1] * num_decodes + prefill_lens
    num_tokens = sum(query_lens)
    num_slots = len(query_lens) + 1  # slot 0 is the null block
    slots = (torch.randperm(num_slots - 1, generator=torch.Generator().manual_seed(5)) + 1).tolist()
    layer = gdn_layer(generator, num_slots)
    initial = [state.clone() for state in layer.kv_cache]
    projection = torch.randn(num_tokens, C + HV * V + 2 * HV, generator=generator, device=DEVICE).bfloat16()
    views = projection[:, : C + HV * V], projection[:, C + HV * V :]
    copies = tuple(view.contiguous() for view in views)
    m = mixed_step_metadata(query_lens, decode_contexts + prefill_contexts, slots, num_decodes)
    mixed_step = mixed_step_builder(monkeypatch).build(0, m)
    decode_slots = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    decode_step = types.SimpleNamespace(num_actual_tokens=len(slots), non_spec_state_indices_tensor=decode_slots)

    results = {}
    for name, (qkvz, ba) in (("views", views), ("copies", copies)):
        for state, start in zip(layer.kv_cache, initial, strict=True):
            state.copy_(start)
        mixed_out = torch.empty(num_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)
        GDNCore.mixed_step(layer, qkvz, ba, mixed_out, mixed_step._pear_layout)
        decode_out = torch.empty(len(slots), HV, V, dtype=torch.bfloat16, device=DEVICE)
        GDNCore.decode_step(layer, qkvz[: len(slots)], ba[: len(slots)], decode_out, decode_step)
        torch.cuda.synchronize()
        results[name] = (mixed_out, decode_out, *(state.clone() for state in layer.kv_cache))

    assert mixed_step._pear_layout.D == num_decodes
    assert views[0].stride(0) == views[1].stride(0) == projection.stride(0)
    for got, want in zip(results["views"], results["copies"], strict=True):
        assert torch.equal(got, want)
