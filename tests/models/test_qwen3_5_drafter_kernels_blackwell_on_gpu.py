import types

import pytest
import torch
import torch.distributed as dist
from vllm.platforms import current_platform

from nonfinite_rows import NonfiniteRows
from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gemm import DraftGemmTableLinearMethod
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.model import TP_SIZE
from pear.models.qwen3_5.rollout.blackwell.mtp import BlackwellQwen3_5MTP, DraftArgmaxBuffers
from pear.models.qwen3_5.rollout.speculator import Qwen3_5MTPSpeculator

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
SHARD = 124160  # a rank's vocabulary shard of Qwen3.5 at TP2
HIDDEN = 5120  # Qwen3.5's hidden size
ARGMAX_ROWS = (1, 16, 64, 128)  # the drafter's batch sizes
FC_INPUT_ROWS = (1, 16, 64, 256, 2048)  # the drafter's token counts
EMBEDDING_SHARD = 4096  # the fc-input test's vocabulary shard: its rows, the lookup as at Qwen3.5's size
EPS = 1e-6  # the drafter's norms' epsilon
NORM_TOLERANCE = 1e-2  # a bf16 output's rounding of an fp32 norm
CALLS = 30  # calls per size: the epochs advance through both slot parities
TIE = 100.0  # a logit above every random one
CANDIDATES = 16  # the heuristic candidates the GEMM table keeps per shape (kCandidates of csrc/gemm_lt.cpp)
GEMM_TOLERANCE = 1e-2  # relative to the row's largest |out|: a bf16 output's rounding, fp32 accumulation
# the drafter's GEMMs per rank at TP2 (N, K, fp32 out): fc (its embedding block whole, half of its hidden block), qkv,
# o_proj, gate_up, down, the lm_head shard
DRAFT_SHAPES = (
    (5120, 7680, False),
    (7168, 5120, False),
    (5120, 3072, False),
    (17408, 5120, False),
    (5120, 8704, False),
    (124160, 5120, True),
)


def argmax_logits(generator: torch.Generator, rank: int, rows: int, call: int, device: torch.device) -> torch.Tensor:
    """A shard of random logits; every third call plants ties within the shard, every third one across the ranks; of
    every five calls one masks every second token and the second rank's whole shard by -inf, and three corrupt both
    ranks' shards: a first row of NaN alone, a +inf column, a NaN column."""
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
    shards = [torch.empty_like(logits) for _ in range(2)]
    dist.all_gather(shards, logits)
    return torch.cat(shards, dim=1)


def two_rank_checks(rank: int, rendezvous: str) -> None:
    """The drafter's argmax over the ranks' shards on one rank, and the error the step after a proposal raises."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, device_id=device)
    group = dist.group.WORLD
    BlackwellExtension.load()
    DraftArgmaxBuffers.create(group, max(ARGMAX_ROWS), device)
    counts = DraftArgmaxBuffers.instance.nonfinite
    generator = torch.Generator(device=device).manual_seed(100 + rank)  # each rank its own shard

    # a size's calls as the draft steps of one proposal, its counts zeroed first (Qwen3_5MTPSpeculator.propose)
    for rows in ARGMAX_ROWS:
        counts.zero_()
        want_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        for call in range(CALLS):
            logits = argmax_logits(generator, rank, rows, call, device)
            tokens = torch.ops.pear_rollout_blackwell.draft_argmax(logits, rank * SHARD)
            full = full_vocabulary(logits)

            # every row's token inside the vocabulary (the next draft step's input), a row free of NaN and +inf
            # logits drafting torch.argmax's token, the rows' NaN and +inf logits over both shards added up
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
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the drafter's exchanges are between two GPUs")
def test_the_drafter_s_argmax_on_two_ranks(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(two_rank_checks, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Qwen3.5's (1 + w) RMSNorm of x's rows in fp64."""
    x = x.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + weight.double())


@pytest.mark.parametrize("rank", range(TP_SIZE))
@pytest.mark.parametrize("is_lookup", [True, False])
@pytest.mark.parametrize("tokens", FC_INPUT_ROWS)
def test_the_drafter_s_fc_input_is_its_partial_embedding_normed_and_its_half_of_the_normed_hidden(
    tokens, is_lookup, rank
):
    generator = torch.Generator(DEVICE).manual_seed(tokens + rank)
    start = rank * EMBEDDING_SHARD
    table = torch.randn(EMBEDDING_SHARD, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    input_ids = torch.randint(0, TP_SIZE * EMBEDDING_SHARD, (tokens,), generator=generator, device=DEVICE)
    hidden_states = torch.randn(tokens, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    embedding_weight, hidden_weight = (0.1 * torch.randn(2, HIDDEN, generator=generator, device=DEVICE)).bfloat16()
    # this rank's partial rows: the tokens of its shard, zeros for the others
    in_shard = (input_ids >= start) & (input_ids < start + EMBEDDING_SHARD)
    partial_rows = torch.where(in_shard[:, None], table[(input_ids - start).clamp(0, EMBEDDING_SHARD - 1)], 0)
    mtp = types.SimpleNamespace(
        model=types.SimpleNamespace(
            embed_tokens=types.SimpleNamespace(
                weight=table,
                shard_indices=types.SimpleNamespace(
                    org_vocab_start_index=start, org_vocab_end_index=start + EMBEDDING_SHARD
                ),
            ),
            pre_fc_norm_embedding=types.SimpleNamespace(weight=embedding_weight, variance_epsilon=EPS),
            pre_fc_norm_hidden=types.SimpleNamespace(weight=hidden_weight),
            fc=types.SimpleNamespace(rank=rank),
        ),
        fc_input_kernel=BlackwellQwen3_5MTP.fc_input_kernel,
        FC_INPUT_WARPS=BlackwellQwen3_5MTP.FC_INPUT_WARPS,
    )
    half = HIDDEN // TP_SIZE
    want = torch.cat(
        [rms_norm(partial_rows, embedding_weight), rms_norm(hidden_states, hidden_weight)[:, rank * half :][:, :half]],
        dim=1,
    )

    out = BlackwellQwen3_5MTP.fc_input(mtp, input_ids, None if is_lookup else partial_rows, hidden_states)

    assert out.shape == (tokens, HIDDEN + half) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.double(), want, rtol=NORM_TOLERANCE, atol=NORM_TOLERANCE)


@pytest.mark.parametrize("N, K, is_fp32", DRAFT_SHAPES)
@pytest.mark.parametrize("M", [16, 64, 256])
def test_every_candidate_of_the_drafter_s_gemm_table_matches_fp64(M, N, K, is_fp32):
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(M + N)
    x = torch.randn(M, K, generator=generator, device=DEVICE).bfloat16()
    w = torch.randn(N, K, generator=generator, device=DEVICE).bfloat16() / K**0.5
    reference = x.double() @ w.double().t()
    scale = reference.abs().amax(dim=1, keepdim=True)

    for candidate in range(CANDIDATES):
        torch.ops.pear_rollout_blackwell.dg_set(M, N, K, is_fp32, candidate)
        out = torch.ops.pear_rollout_blackwell.dg_mm(x, w, is_fp32)
        assert out.dtype == (torch.float32 if is_fp32 else torch.bfloat16)
        error = ((out.double() - reference).abs() / scale).max().item()
        assert error < GEMM_TOLERANCE, f"candidate {candidate}: {error}"
    # candidate 0 and the table's own choices back
    torch.ops.pear_rollout_blackwell.dg_set(M, N, K, is_fp32, 0)
    DraftGemmTableLinearMethod.load_table()
