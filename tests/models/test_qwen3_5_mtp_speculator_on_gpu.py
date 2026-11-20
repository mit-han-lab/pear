import types

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    _prepare_decode_inputs_kernel,
    update_draft_inputs,
)

from pear.models.qwen3_5.rollout.speculator import Qwen3_5MTPSpeculator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the speculator's Triton kernels need a GPU")

DEVICE = "cuda"
MAX_MODEL_LEN = 164096  # stage 1's model length
MAX_REQUESTS = 128  # request-state slots
MAX_TOKENS = 2048  # the drafter's token buffers
MROPE_ROWS = 3  # Qwen3.5's M-RoPE rows: time, height, width
ROWS_PER_REQUEST = 4  # a verify step's rows of a request: its last token and three drafts
NUM_SPECULATIVE_TOKENS = 3  # a step's drafts
VOCAB = 248320  # Qwen3.5's vocabulary
HIDDEN = 5120  # Qwen3.5's hidden size: the drafter's hidden states
AT_THE_END = 5  # the columns placed at the model's last position


def speculator(generator: torch.Generator, advance_draft_positions: bool) -> types.SimpleNamespace:
    """The speculator's buffers that its draft-step preparation reads and writes, random, the first M-RoPE columns and
    1-D positions at the model's last position."""

    def randint(high: int, *shape) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=generator, device=DEVICE)

    mrope_positions = randint(MAX_MODEL_LEN, MROPE_ROWS, ROWS_PER_REQUEST * MAX_REQUESTS + 1)
    mrope_positions[:, :AT_THE_END] = MAX_MODEL_LEN - 1
    positions = randint(MAX_MODEL_LEN, MAX_TOKENS)
    positions[:AT_THE_END] = MAX_MODEL_LEN - 1
    # each request's last token among its rows: the gathered columns are other requests' destinations
    last_token_indices = torch.arange(MAX_REQUESTS, device=DEVICE) * ROWS_PER_REQUEST + randint(
        ROWS_PER_REQUEST, MAX_REQUESTS
    )
    return types.SimpleNamespace(
        draft_tokens=randint(VOCAB, MAX_REQUESTS, NUM_SPECULATIVE_TOKENS),
        input_buffers=types.SimpleNamespace(
            input_ids=randint(VOCAB, MAX_TOKENS).int(),
            positions=positions,
            query_start_loc=randint(MAX_TOKENS, MAX_REQUESTS + 1).int(),
            seq_lens=randint(MAX_MODEL_LEN, MAX_REQUESTS).int(),
        ),
        sample_src_positions=randint(MAX_TOKENS, MAX_REQUESTS),
        mrope_positions=mrope_positions,
        last_token_indices=last_token_indices,
        max_model_len=MAX_MODEL_LEN,
        max_model_len_gpu=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE),
        max_num_reqs=MAX_REQUESTS,
        advance_draft_positions=advance_draft_positions,
        PAD_BLOCK=Qwen3_5MTPSpeculator.PAD_BLOCK,
        prepare_draft_steps_kernel=Qwen3_5MTPSpeculator.prepare_draft_steps_kernel,
    )


def buffers(mtp: types.SimpleNamespace) -> list[torch.Tensor]:
    """Every buffer the draft-step preparation writes."""
    inputs = mtp.input_buffers
    return [
        inputs.input_ids,
        inputs.positions,
        inputs.query_start_loc,
        inputs.seq_lens,
        mtp.sample_src_positions,
        mtp.mrope_positions,
    ]


@pytest.mark.parametrize("num_reqs", [1, 16, 64, MAX_REQUESTS])
@pytest.mark.parametrize("advance_draft_positions", [True, False])
def test_the_draft_steps_kernel_is_vllms_decode_inputs_then_the_m_rope_advance_bit_for_bit(
    num_reqs, advance_draft_positions
):
    mtp = speculator(torch.Generator(DEVICE).manual_seed(num_reqs), advance_draft_positions)
    input_batch = types.SimpleNamespace(seq_lens=torch.randint(1, MAX_MODEL_LEN, (num_reqs,), device=DEVICE).int())
    num_rejected = torch.randint(0, NUM_SPECULATIVE_TOKENS + 1, (num_reqs,), device=DEVICE).int()
    # vLLM 0.34's kernel, then each request's M-RoPE rows one past those of its last token, on the same buffers drawn
    # again
    want = speculator(torch.Generator(DEVICE).manual_seed(num_reqs), advance_draft_positions)
    inputs = want.input_buffers
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        want.draft_tokens[:num_reqs, 0], want.draft_tokens.stride(0), input_batch.seq_lens, num_rejected,
        inputs.input_ids, inputs.positions, want.sample_src_positions, inputs.query_start_loc, inputs.seq_lens,
        MAX_MODEL_LEN, MAX_REQUESTS, BLOCK_SIZE=Qwen3_5MTPSpeculator.PAD_BLOCK,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )  # fmt: skip
    sources = want.last_token_indices[:num_reqs]
    want.mrope_positions[:, :num_reqs] = (want.mrope_positions[:, sources] + 1).clamp(max=MAX_MODEL_LEN - 1)

    Qwen3_5MTPSpeculator._prepare_decode_inputs(mtp, input_batch, num_rejected, num_reqs, False, None)

    for got, expected in zip(buffers(mtp), buffers(want), strict=True):
        assert torch.equal(got, expected)


def draft_step_speculator(
    generator: torch.Generator, advance_draft_positions: bool, num_reqs: int, step: int
) -> types.SimpleNamespace:
    """The speculator's buffers that a draft step reads and writes, random, at the step's draft; the drafter's forward
    and its argmax replaced by their outputs, random too."""
    mtp = speculator(generator, advance_draft_positions)
    mtp.input_buffers.seq_lens[:AT_THE_END] = MAX_MODEL_LEN
    mtp.hidden_states = torch.randn(MAX_TOKENS, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    drafter_output = torch.randn(ROWS_PER_REQUEST * num_reqs, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    drafts = torch.randint(0, VOCAB, (num_reqs,), generator=generator, device=DEVICE)
    mtp.current_draft_step = torch.tensor(step, dtype=torch.int64, device=DEVICE)
    mtp._prepare_eplb_forward = lambda num_reqs: None
    mtp._run_model = lambda *arguments: (drafter_output, drafter_output)
    mtp.sample_draft = lambda *arguments: drafts
    mtp.idx_mapping = torch.arange(MAX_REQUESTS, device=DEVICE)
    mtp.temperature = mtp.seeds = mtp.draft_logits = None  # an argmax drafter reads neither
    mtp.HIDDEN_BLOCK = Qwen3_5MTPSpeculator.HIDDEN_BLOCK
    mtp.update_draft_steps_kernel = Qwen3_5MTPSpeculator.update_draft_steps_kernel
    mtp.drafter_output, mtp.drafts = drafter_output, drafts
    return mtp


def draft_step_buffers(mtp: types.SimpleNamespace) -> list[torch.Tensor]:
    """Every buffer a draft step writes."""
    inputs = mtp.input_buffers
    return [
        mtp.draft_tokens,
        inputs.input_ids,
        inputs.positions,
        inputs.seq_lens,
        mtp.sample_src_positions,
        mtp.hidden_states,
        mtp.mrope_positions,
    ]


@pytest.mark.parametrize("num_reqs", [1, 16, 64, MAX_REQUESTS])
@pytest.mark.parametrize("step", range(NUM_SPECULATIVE_TOKENS))
@pytest.mark.parametrize("advance_draft_positions", [True, False])
def test_a_draft_step_is_vllms_input_update_then_the_m_rope_advance_bit_for_bit(
    num_reqs, step, advance_draft_positions
):
    seed = num_reqs * NUM_SPECULATIVE_TOKENS + step
    mtp = draft_step_speculator(torch.Generator(DEVICE).manual_seed(seed), advance_draft_positions, num_reqs, step)
    # vLLM 0.34's update of the next draft step's inputs, then each request's M-RoPE rows one past the last but after
    # the final draft step, on the same buffers drawn again
    want = draft_step_speculator(torch.Generator(DEVICE).manual_seed(seed), advance_draft_positions, num_reqs, step)
    update_draft_inputs(
        want.drafts, want.current_draft_step, want.drafter_output, want.draft_tokens, want.hidden_states,
        want.input_buffers, want.sample_src_positions, num_reqs, want.max_model_len_gpu, NUM_SPECULATIVE_TOKENS,
        advance_draft_positions=advance_draft_positions,
    )  # fmt: skip
    if step < NUM_SPECULATIVE_TOKENS - 1:
        want.mrope_positions[:, :num_reqs] = (want.mrope_positions[:, :num_reqs] + 1).clamp(max=MAX_MODEL_LEN - 1)

    Qwen3_5MTPSpeculator._decode(
        mtp, num_reqs, ROWS_PER_REQUEST * num_reqs, None, None, None, CUDAGraphMode.NONE, NUM_SPECULATIVE_TOKENS
    )

    for got, expected in zip(draft_step_buffers(mtp), draft_step_buffers(want), strict=True):
        assert torch.equal(got, expected)
