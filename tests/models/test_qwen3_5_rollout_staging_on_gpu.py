import types

import numpy as np
import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

from pear.models.qwen3_5.rollout.blackwell.gdn.layer import MixedStepMetadata
from pear.models.qwen3_5.rollout.gdn import RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.hopper.attention import AttentionStepBuffers
from pear.models.qwen3_5.rollout.hopper.gdn import GDNLayout
from pear.models.qwen3_5.rollout.staged_buffer import StagedBuffer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the uploads' device copies")

DEVICE = "cuda"
SIZE = 4096  # slots of the test's buffer
# GPU clock cycles the stream spins before the uploads: far longer than the host takes for them, so every upload is
# still queued when the host writes the next one
BUSY_CYCLES = 1 << 28
UPLOADS = 3  # the third reuses the first one's host buffer (or vLLM's slot)
MAX_NUM_SEQS = 4  # the builders' request capacity
TOKEN_BUDGET = 256  # their max_num_batched_tokens
# the Blackwell mixed steps' token budget: a decode row beside a prompt chunk of up to 16384 tokens
BLACKWELL_TOKEN_BUDGET = 32768
GDN_HEADS, GDN_HEAD_DIM = 2, 128  # the test layout's key and value heads and their dims
# what MixedStepMetadata reads of its builder: vLLM's decode rows, a run of rows' state slots in its block table's
# column 0 (a slice, as the builder's: a pageable copy to the GPU would wait for the busy stream)
BUILDER = types.SimpleNamespace(
    num_decodes=RolloutGDNMetadataBuilder.num_decodes,
    row_slots=lambda block_table, rows: block_table[int(rows[0]) : int(rows[-1]) + 1, 0],
)


def busy_stream_reads(upload, read, count: int) -> list[list[int]]:
    """Uploads 1 .. count behind a busy stream, each read on the device right after it: what each reader saw."""
    torch.cuda._sleep(BUSY_CYCLES)
    reads = []
    for k in range(1, count + 1):
        upload(k)
        reads.append(read().clone())
    torch.cuda.synchronize()

    return [r.tolist() for r in reads]


def staged_values(buffer: StagedBuffer, k: int) -> None:
    """Upload SIZE slots of value k through buffer."""
    stage = buffer.next_stage()
    stage[:] = k
    buffer.upload(SIZE)


def mixed_step(prompt_tokens: int) -> CommonAttentionMetadata:
    """A step of a decode row and a new prompt's chunk, as vLLM's V2 runner gives it to a GDN group's builder."""
    qsl = torch.tensor([0, 1, 1 + prompt_tokens], dtype=torch.int32)
    seq_lens = torch.tensor([50, prompt_tokens], dtype=torch.int32)

    return CommonAttentionMetadata(
        query_start_loc=qsl.to(DEVICE),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.to(DEVICE),
        num_reqs=2,
        num_actual_tokens=1 + prompt_tokens,
        max_query_len=prompt_tokens,
        max_seq_len=50,
        block_table_tensor=torch.tensor([[1], [2]], dtype=torch.int32, device=DEVICE),
        slot_mapping=torch.zeros(1 + prompt_tokens, dtype=torch.int64, device=DEVICE),
        is_prefilling=torch.tensor([False, True]),
        seq_lens_cpu_upper_bound=seq_lens,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8])  # the step layouts' and the verify rows'
def test_a_host_buffer_is_rewritten_only_after_its_last_upload_ran(dtype):
    buffer = StagedBuffer(SIZE, DEVICE, dtype)

    reads = busy_stream_reads(lambda k: staged_values(buffer, k), lambda: buffer.device_buffer, UPLOADS)

    assert [sorted(set(r)) for r in reads] == [[1], [2], [3]]


def test_without_the_wait_the_first_reader_sees_the_third_upload(monkeypatch):
    # the control: the host buffer of the first upload rewritten while that upload still waits on the stream
    monkeypatch.setattr(StagedBuffer, "next_stage", lambda buffer: buffer.stages[buffer.turn].numpy())
    buffer = StagedBuffer(SIZE, DEVICE)

    reads = busy_stream_reads(lambda k: staged_values(buffer, k), lambda: buffer.device_buffer, UPLOADS)

    assert sorted(set(reads[0])) == [3]


def test_vllms_two_slot_pool_hands_the_first_reader_the_third_upload():
    # vLLM's pool rewrites a slot without waiting for its last reader
    pool = UvaBufferPool(SIZE, torch.int32)
    device_buffer = torch.zeros(SIZE, dtype=torch.int32, device=DEVICE)

    reads = busy_stream_reads(
        lambda k: pool.copy_to_gpu(np.full(SIZE, k, dtype=np.int32), out=device_buffer),
        lambda: device_buffer,
        UPLOADS,
    )

    assert sorted(set(reads[0])) == [3]


def test_hopper_attention_arguments_reach_their_step_behind_a_busy_stream():
    buffers = AttentionStepBuffers(MAX_NUM_SEQS, DEVICE)
    qlens = np.array([1, 1, 5])

    reads = busy_stream_reads(
        lambda k: buffers.fill(1, np.array([0, 1, 2]), 7, qlens, np.array([10, 20, 30]) * k),
        lambda: buffers.sk_all[: len(qlens)],
        UPLOADS,
    )

    assert reads == [[10 * k, 20 * k, 30 * k] for k in range(1, UPLOADS + 1)]


def test_a_hopper_gdn_layout_reaches_its_step_behind_a_busy_stream():
    heads, dim = GDN_HEADS, GDN_HEAD_DIM
    layout = GDNLayout(TOKEN_BUDGET, MAX_NUM_SEQS, MAX_NUM_SEQS, heads, heads, dim, dim, DEVICE)

    reads = busy_stream_reads(
        lambda k: layout.fill(np.array([0, 1, 1 + 10 * k]), 1, np.array([1, 0], dtype=np.int32)),
        lambda: layout.seq_start[:3],
        UPLOADS,
    )

    assert reads == [[0, 1, 1 + 10 * k] for k in range(1, UPLOADS + 1)]


# a step's upload: 48 bytes for a short prompt, 8 KB and 16 KB for chunks of 8192 and 16384 tokens (one conv program
# per CONV_TOKENS tokens)
@pytest.mark.parametrize("chunk", [30, 8192, 16384])
def test_three_blackwell_mixed_steps_each_reach_their_own_step_behind_a_busy_stream(chunk):
    scheduler_config = types.SimpleNamespace(max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=BLACKWELL_TOKEN_BUDGET)
    metadata = MixedStepMetadata(scheduler_config, torch.device(DEVICE))
    # the steps exist before the busy stream: their pageable copies to the GPU would wait for it
    steps = [mixed_step(chunk + k) for k in range(1, UPLOADS + 1)]
    built = []

    reads = busy_stream_reads(
        lambda k: built.append(metadata.build(BUILDER, steps[k - 1])),
        lambda: built[-1].prefill_query_start_loc,
        UPLOADS,
    )

    assert reads == [[0, chunk + k] for k in range(1, UPLOADS + 1)]
