import json

import pytest
import torch
import torch.distributed as dist
from vllm.kernels.triton.fused_pre_fc_norms import fused_pre_fc_norms
from vllm.platforms import current_platform
from vllm.triton_utils import triton

from nonfinite_rows import NonfiniteRows
from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.mtp import DraftArgmaxBuffers, HopperQwen3_5MTP, _draft_fc_input_kernel
from pear.models.qwen3_5.rollout.speculator import Qwen3_5MTPSpeculator

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not HopperRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="HopperRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
WORLD = DraftArgmaxBuffers.WORLD  # the Hopper replica's ranks
SHARD = 62080  # a rank's vocabulary shard of Qwen3.5 at TP4
HIDDEN = 5120  # Qwen3.5's hidden size
ARGMAX_ROWS = (1, 16, 64, 128)  # the drafter's batch sizes
CALLS = 30  # calls per size: the calls alternate the slot parities
GRAPH_ROWS = 64  # the rows of the argmax call captured in a CUDA graph
GRAPH_REPLAYS = 20  # replays of the captured call, each on new logits
FINITE_CALL_STRIDE = 5  # argmax_logits' calls of finite logits alone: every fifth
TIE = 100.0  # a logit above every random one
FC_VOCAB = WORLD * 2048  # the fc test's vocabulary of four shards (the kernel reads rows of the shard it is given)
FC_ROWS = (1, 16, 64, 256)  # the drafter's passes' rows
EPS = 1e-6  # the drafter's norms' epsilon
BF16_ULP = 2.0**-7  # a bf16 rounding's relative size, generous (2^-8 is half an ulp)
FC_TOLERANCE = 4 * BF16_ULP  # the summed fc products' error against fp64, relative to the largest output
GEMM_SHAPES = (  # (N, K, fp32 out) of the drafter's GEMMs at TP4: the lm_head, qkv_proj, gate_up_proj
    (62080, HIDDEN, True),
    (3584, HIDDEN, False),
    (8704, HIDDEN, False),
)
GEMM_ROWS = (1, 16, 32, 64, 128)  # the drafter's rows: the table's and others (candidate 0)
GEMM_TOLERANCE = 2e-5  # relative to the largest output: fp32 accumulation over 5,120 bf16 products in another order
BF16_ROUNDING = 2.0**-8  # relative to the largest output: a bf16 output's rounding (half an ulp, the binade's top)


def argmax_logits(generator: torch.Generator, rank: int, rows: int, call: int, device: torch.device) -> torch.Tensor:
    """A shard of random logits; every third call plants ties within the shard, every third one across the ranks; of
    every five calls one masks every second token and the second rank's whole shard by -inf, and three corrupt every
    rank's shard: a first row of NaN alone, a +inf column, a NaN column."""
    logits = torch.randn(rows, SHARD, generator=generator, device=device) * 3
    if call % 3 == 1:
        logits[:, 7] = TIE
        logits[:, SHARD - 5] = TIE
    elif call % 3 == 2:
        logits[:, (call * 37) % SHARD] = TIE
    if call % 5 == 1:
        logits[:, 1 :: NonfiniteRows.MASK_STRIDE] = float("-inf")
        if rank == 1:
            logits.fill_(float("-inf"))
    elif call % 5 == 2:
        logits[0] = float("nan")
    elif call % 5 == 3:
        logits[:, (call * 41) % SHARD] = float("inf")
    elif call % 5 == 4:
        logits[:, (call * 53) % SHARD] = float("nan")
    return logits


def full_vocabulary(logits: torch.Tensor) -> torch.Tensor:
    """The ranks' shards gathered into rows of the whole vocabulary."""
    shards = [torch.empty_like(logits) for _ in range(WORLD)]
    dist.all_gather(shards, logits)
    return torch.cat(shards, dim=1)


def initialize_rank(rank: int, rendezvous: str) -> torch.device:
    """One of four ranks: its GPU, the process group and the Hopper extension."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD, device_id=device)
    HopperExtension.load()
    return device


def argmax_checks(rank: int, rendezvous: str) -> None:
    """The drafter's argmax over the ranks' shards on one rank, eagerly and replayed from a CUDA graph, and the error
    the step after a proposal raises."""
    device = initialize_rank(rank, rendezvous)
    DraftArgmaxBuffers.create(dist.group.WORLD, max(ARGMAX_ROWS), device)
    buffers = DraftArgmaxBuffers.instance
    counts = buffers.nonfinite
    generator = torch.Generator(device=device).manual_seed(100 + rank)  # each rank its own shard

    # a size's calls as the draft steps of one proposal, its counts zeroed first (Qwen3_5MTPSpeculator.propose)
    for rows in ARGMAX_ROWS:
        counts.zero_()
        want_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        for call in range(CALLS):
            logits = argmax_logits(generator, rank, rows, call, device)
            tokens = buffers.argmax(logits, rank * SHARD)
            full = full_vocabulary(logits)

            # every row's token inside the vocabulary (the next draft step's input), a row free of NaN and +inf
            # logits drafting torch.argmax's token (the smallest on ties), the rows' NaN and +inf logits over every
            # shard added up
            is_corrupted = ~(full < float("inf"))
            is_clean = ~is_corrupted.any(dim=1)
            want_counts += is_corrupted.sum(dim=1, dtype=torch.int32)
            assert ((tokens >= 0) & (tokens < full.shape[1])).all(), f"rank {rank} rows {rows} call {call}"
            assert torch.equal(tokens[is_clean], full.argmax(dim=1)[is_clean]), f"rank {rank} rows {rows} call {call}"
            assert torch.equal(counts[:rows], want_counts), f"rank {rank} rows {rows} call {call}"

    # the output of the step after the proposal raises on its rows, naming the step, the drafter and the rows
    req_ids = [f"request-{row}" for row in range(rows)]
    want_rows = [(row, req_ids[row], count) for row, count in enumerate(want_counts.tolist()) if count]
    NonfiniteRows.assert_step_output(Qwen3_5MTPSpeculator.DRAFT_KERNEL, counts[:rows].clone(), req_ids, want_rows)

    # a call captured in a CUDA graph and replayed on new finite logits (ties among them), eager calls between the
    # replays
    logits = argmax_logits(generator, rank, GRAPH_ROWS, 0, device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buffers.argmax(logits, rank * SHARD)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = buffers.argmax(logits, rank * SHARD)
    for replay in range(GRAPH_REPLAYS):
        logits.copy_(argmax_logits(generator, rank, GRAPH_ROWS, FINITE_CALL_STRIDE * replay, device))
        graph.replay()
        assert torch.equal(tokens, full_vocabulary(logits).argmax(dim=1)), f"rank {rank} replay {replay}"
        assert torch.equal(buffers.argmax(logits, rank * SHARD), tokens), f"rank {rank} eager after replay {replay}"
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason="the drafter's exchanges are between four GPUs")
def test_the_drafter_s_argmax_on_four_ranks(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(argmax_checks, args=(str(tmp_path / "rendezvous"),), nprocs=WORLD)


def fc_checks(rank: int, rendezvous: str) -> None:
    """The drafter's fc on partial embedding rows on one rank: its input kernel against vLLM's fused_pre_fc_norms, and
    the ranks' summed products with DraftFcLinear's weights against fp64 on the same inputs."""
    device = initialize_rank(rank, rendezvous)
    generator = torch.Generator(device).manual_seed(7)  # the same weights and inputs on every rank
    table = torch.randn(FC_VOCAB, HIDDEN, generator=generator, device=device).bfloat16()
    w_embed = (torch.randn(HIDDEN, generator=generator, device=device) * 0.1).bfloat16()
    w_hidden = (torch.randn(HIDDEN, generator=generator, device=device) * 0.1).bfloat16()
    fc = (torch.randn(HIDDEN, 2 * HIDDEN, generator=generator, device=device) / 100).bfloat16()
    part = HIDDEN // WORLD
    start, end = rank * FC_VOCAB // WORLD, (rank + 1) * FC_VOCAB // WORLD
    shard = table[start:end].contiguous()
    # DraftFcLinear's weight on this rank: the embedding block whole, its columns of the hidden block
    weight = torch.cat([fc[:, :HIDDEN], fc[:, HIDDEN + rank * part : HIDDEN + (rank + 1) * part]], 1)

    def fc_input(ids, embeds, hidden):
        out = hidden.new_empty((hidden.shape[0], HIDDEN + part))
        is_lookup = embeds is None
        _draft_fc_input_kernel[(hidden.shape[0],)](
            ids, hidden if is_lookup else embeds, shard, hidden, hidden.stride(0), w_embed, w_hidden, out, start, end,
            EPS, H=HIDDEN, PART=part, RANK=rank, BLOCK=triton.next_power_of_2(HIDDEN), LOOKUP=is_lookup,
            num_warps=HopperQwen3_5MTP.FC_INPUT_WARPS,
        )  # fmt: skip
        return out

    def assert_close(actual, expected, what):
        # within a bf16 rounding of the expected values (the reductions' order may differ)
        error = (actual.float() - expected.float()).abs() / expected.float().abs().clamp_min(1e-6)
        assert error.max().item() <= BF16_ULP, f"rank {rank} {what}: {error.max().item()}"

    for rows in FC_ROWS:
        ids = torch.randint(0, FC_VOCAB, (rows,), generator=generator, device=device)
        hidden = torch.randn(rows, HIDDEN, generator=generator, device=device).bfloat16()

        # vLLM: the whole embedding rows, both norms
        reference_input = fused_pre_fc_norms(table[ids], hidden, w_embed, w_hidden, EPS)
        # the drafter: this rank's input (its partial rows looked up, or given)
        looked_up = fc_input(ids, None, hidden)
        in_shard = ((ids >= start) & (ids < end)).unsqueeze(-1)
        partial_rows = torch.where(in_shard, shard[(ids - start).clamp(0, end - start - 1)], 0)
        assert torch.equal(looked_up, fc_input(ids, partial_rows, hidden)), f"rank {rank} rows {rows}"

        # the input's columns against vLLM's: this rank's embedding rows and its part of the hidden half
        assert_close(looked_up[:, :HIDDEN], torch.where(in_shard, reference_input[:, :HIDDEN], 0), f"{rows} embedding")
        expected_hidden = reference_input[:, HIDDEN + rank * part : HIDDEN + (rank + 1) * part]
        assert_close(looked_up[:, HIDDEN:], expected_hidden, f"rows {rows} hidden norm")

        # the ranks' products summed, against the whole fc in fp64
        summed = (looked_up @ weight.t()).float()
        dist.all_reduce(summed)
        exact = reference_input.double() @ fc.double().t()
        error = (summed.bfloat16().double() - exact).abs().max().item() / exact.abs().max().item()
        assert error < FC_TOLERANCE, f"rank {rank} rows {rows}: fc error {error}"
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason="the drafter's fc sums over four GPUs")
def test_the_drafter_s_fc_on_partial_embedding_rows_on_four_ranks(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(fc_checks, args=(str(tmp_path / "rendezvous"),), nprocs=WORLD)


@pytest.mark.parametrize("shape", GEMM_SHAPES)
@pytest.mark.parametrize("rows", GEMM_ROWS)
def test_the_drafter_s_gemm_on_the_table_s_candidates_matches_fp64_and_torch(rows, shape):
    HopperExtension.load()
    # the table's candidates, as the drafter sets them when it is built
    for table_rows, n, k, is_fp32, candidate in json.loads(HopperQwen3_5MTP.GEMM_TABLE.read_text())["rows"]:
        torch.ops.pear_rollout_hopper.dg_set(table_rows, n, k, is_fp32, candidate)
    n, k, is_fp32 = shape
    generator = torch.Generator(DEVICE).manual_seed(rows)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) / 50).bfloat16()
    x = torch.randn(rows, k, generator=generator, device=DEVICE).bfloat16()

    out = torch.ops.pear_rollout_hopper.dg_mm(x, weight, is_fp32)
    reference = torch.mm(x, weight.t(), out_dtype=torch.float32) if is_fp32 else torch.mm(x, weight.t())
    exact = x.double() @ weight.double().t()
    torch.cuda.synchronize()

    assert out.dtype == (torch.float32 if is_fp32 else torch.bfloat16) and out.shape == (rows, n)
    scale = exact.abs().max().item()
    assert (out.double() - exact).abs().max().item() < (GEMM_TOLERANCE if is_fp32 else BF16_ROUNDING) * scale
    assert (out.double() - reference.double()).abs().max().item() < (GEMM_TOLERANCE if is_fp32 else BF16_ULP) * scale
