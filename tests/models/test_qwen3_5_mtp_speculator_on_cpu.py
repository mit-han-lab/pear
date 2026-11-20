import contextlib
import types

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode import speculator_config as vllm_speculator_config
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as target_dependent_ar

from pear.models.qwen3_5.rollout import speculator as speculator_module
from pear.models.qwen3_5.rollout.nonfinite_logits import NonfiniteLogits
from pear.models.qwen3_5.rollout.rollout_kernels import Qwen3_5RolloutKernels
from pear.models.qwen3_5.rollout.speculator import Qwen3_5MTPSpeculator

MROPE_ROWS = 3  # Qwen3.5's M-RoPE rows: time, height, width
NUM_SPECULATIVE_TOKENS = 3  # a step's drafts: the first pass's, then two draft steps'
MAX_REQS = 4  # the replicas' max_num_seqs
MAX_TOKENS = 16  # their max_num_batched_tokens
MAX_MODEL_LEN = 1000
HIDDEN = 4
# a step of two requests: an image's three tokens and a text token, then two text tokens, padded to eight tokens
QUERY_START_LOC = [0, 4, 6]
NUM_TOKENS_PADDED = 8
POSITIONS = [20, 21, 22, 23, 30, 31]  # the target's 1-D positions
LAST_TOKENS = [3, 5]  # the tokens the first pass samples at
# the target's M-RoPE positions of the step's tokens and padding
TARGET_POSITIONS = torch.tensor([[7, 7, 7, 8, 9, 10, 0, 0], [7, 8, 8, 8, 9, 10, 0, 0], [7, 7, 8, 8, 9, 10, 0, 0]])
# Qwen3.8-27B's rotary embedding, which its config.json names
ROPE = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


class Drafter:
    """Qwen3_5MTP reduced to what the speculator calls: each forward's positions recorded, a greedy draft."""

    def __init__(self):
        self.positions = []

    def __call__(self, input_ids, positions, hidden_states, inputs_embeds):
        self.positions.append(positions.clone())
        return torch.zeros(positions.shape[-1], HIDDEN)

    def get_top_tokens(self, hidden_states):
        return torch.zeros(hidden_states.shape[0], dtype=torch.int64)


def update_draft_steps(
    key, grid, draft_tokens, draft_tokens_stride, next_hidden, next_hidden_stride, input_ids, positions,
    sample_src_positions, seq_lens, sampled, current_draft_step, hidden, hidden_stride, hidden_size, max_model_len,
    mrope, mrope_stride, num_speculative_steps, **constexprs,
) -> None:  # fmt: skip
    """The speculator's draft-step update (a Triton kernel) in torch on the CPU, its M-RoPE advance alone (vLLM's draft
    inputs are vLLM's concern): each request's rows one on within the model's length, but after the final draft step."""
    if int(current_draft_step) < num_speculative_steps - 1:
        num_reqs = grid[0]
        mrope[:, :num_reqs] = torch.minimum(mrope[:, :num_reqs] + 1, max_model_len - 1)


def prepare_draft_steps(
    key, grid, draft_tokens, draft_tokens_stride, target_seq_lens, num_rejected, input_ids, positions,
    sample_src_positions, query_start_loc, seq_lens, mrope, mrope_stride, last_token_indices, max_model_len,
    max_num_reqs, num_reqs, **constexprs,
) -> None:  # fmt: skip
    """The speculator's draft-step kernel (Triton) in torch on the CPU, its M-RoPE program alone (vLLM's decode inputs
    are vLLM's concern): each request's rows one past those of its last token, within the model's length."""
    columns = last_token_indices[:num_reqs]
    mrope[:, :num_reqs] = (mrope[:, columns] + 1).clamp(max=max_model_len - 1)


def prepare_prefill_inputs(
    last_token_indices, current_draft_step, input_buffers, input_batch, num_sampled, num_rejected, *args
):
    """vLLM's Triton kernel launch, on the CPU for what the speculator reads next: the requests' last accepted tokens
    and the target's 1-D positions."""
    num_reqs = input_batch.num_reqs
    last_token_indices[:num_reqs] = input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected - 1
    input_buffers.positions[: input_batch.num_tokens] = input_batch.positions
    current_draft_step.zero_()

    return last_token_indices


@pytest.fixture
def drafter(monkeypatch) -> Drafter:
    # vLLM 0.34's propose without its GPU-only steps: the Triton kernels, Triton itself (vLLM's placeholder on a machine
    # without a GPU driver) and the drafter's forward context
    monkeypatch.setattr(target_dependent_ar, "prepare_prefill_inputs", prepare_prefill_inputs)
    monkeypatch.setattr(Qwen3_5MTPSpeculator, "prepare_draft_steps_kernel", staticmethod(prepare_draft_steps))
    monkeypatch.setattr(Qwen3_5MTPSpeculator, "update_draft_steps_kernel", staticmethod(update_draft_steps))
    monkeypatch.setattr(
        speculator_module, "triton", types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    )
    monkeypatch.setattr(speculator_module, "set_forward_context", lambda *args, **kwargs: contextlib.nullcontext())

    return Drafter()


def loaded_speculator(drafter: Drafter) -> Qwen3_5MTPSpeculator:
    """The family's MTP speculator as vLLM's runner holds it after load_model: before initialize_kv_cache's set_attn
    (no target model state) and init_cudagraph_manager (no graph managers)."""
    mtp = Qwen3_5MTPSpeculator.__new__(Qwen3_5MTPSpeculator)
    mtp.__dict__.update(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=MAX_MODEL_LEN), parallel_config=None
        ),
        num_speculative_steps=NUM_SPECULATIVE_TOKENS,
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_MODEL_LEN,
        advance_draft_positions=True,
        temperature=torch.zeros(MAX_REQS),
        seeds=torch.zeros(MAX_REQS, dtype=torch.int64),
        idx_mapping=torch.zeros(MAX_REQS, dtype=torch.int64),
        draft_tokens=torch.zeros(MAX_REQS, NUM_SPECULATIVE_TOKENS, dtype=torch.int64),
        last_token_indices=torch.zeros(MAX_REQS, dtype=torch.int64),
        current_draft_step=torch.tensor(0),
        sample_src_positions=torch.zeros(MAX_REQS, dtype=torch.int64),
        hidden_states=torch.zeros(MAX_TOKENS, HIDDEN),
        input_buffers=types.SimpleNamespace(
            input_ids=torch.zeros(MAX_TOKENS, dtype=torch.int32),
            positions=torch.zeros(MAX_TOKENS, dtype=torch.int64),
            query_start_loc=torch.zeros(MAX_REQS + 1, dtype=torch.int32),
            seq_lens=torch.zeros(MAX_REQS, dtype=torch.int32),
        ),
        mrope_positions=torch.zeros(MROPE_ROWS, MAX_TOKENS + 1, dtype=torch.int64),
        max_model_len_gpu=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32),
        model=drafter,
        use_local_argmax_reduction=True,
        draft_logits=None,
        acceptance_estimator=None,
        pcp_manager=None,
        eplb_state=None,
        dp_size=1,
        dp_rank=0,
        prefill_cudagraph_manager=None,
        decode_cudagraph_manager=None,
        supports_decode_metadata_update=False,
    )

    return mtp


def step_batch() -> types.SimpleNamespace:
    """The target's input batch of the step, reduced to what the speculator reads."""
    return types.SimpleNamespace(
        num_reqs=len(LAST_TOKENS),
        num_tokens=len(POSITIONS),
        num_tokens_after_padding=NUM_TOKENS_PADDED,
        num_scheduled_tokens=np.diff(QUERY_START_LOC),
        seq_lens_cpu_upper_bound=torch.tensor(POSITIONS)[LAST_TOKENS] + 1,
        idx_mapping=torch.arange(len(LAST_TOKENS)),
        decode_graph_eligible=False,
        query_start_loc=torch.tensor(QUERY_START_LOC),
        positions=torch.tensor(POSITIONS),
        seq_lens=torch.tensor(POSITIONS)[LAST_TOKENS] + 1,
    )


def profile_propose(mtp: Qwen3_5MTPSpeculator, input_batch: types.SimpleNamespace) -> None:
    """vLLM 0.34's propose as profile_run's dummy run calls it: eager, without attention."""
    num_reqs = input_batch.num_reqs

    mtp.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.zeros(NUM_TOKENS_PADDED, HIDDEN),
        aux_hidden_states=None,
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=torch.zeros(MAX_REQS, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(MAX_REQS, dtype=torch.int32),
        temperature=torch.ones(MAX_REQS),
        seeds=torch.zeros(MAX_REQS, dtype=torch.int64),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )


def test_the_memory_profile_drafts_before_the_speculator_has_the_target_s_model_state(drafter):
    # the worker profiles memory (determine_available_memory -> profile_run -> _dummy_run -> propose) before it
    # initializes the KV cache, where set_attn hands the speculator the target's model state
    mtp = loaded_speculator(drafter)
    assert not hasattr(mtp, "model_state")

    profile_propose(mtp, step_batch())

    # the first pass and each draft step on M-RoPE positions, the shape torch.compile traces the drafter with there
    shapes = [tuple(positions.shape) for positions in drafter.positions]
    assert shapes == [(MROPE_ROWS, NUM_TOKENS_PADDED)] + [(MROPE_ROWS, len(LAST_TOKENS))] * (NUM_SPECULATIVE_TOKENS - 1)


def test_a_step_drafts_at_the_target_s_m_rope_positions_then_one_past_per_draft_step(drafter):
    mtp = loaded_speculator(drafter)
    target_positions = torch.zeros(MROPE_ROWS, MAX_TOKENS + 1, dtype=torch.int64)
    target_positions[:, :NUM_TOKENS_PADDED] = TARGET_POSITIONS
    # set_attn's link: the target's model state, whose RoPE state holds the step's positions
    rope_state = types.SimpleNamespace(get_positions=lambda num_tokens: target_positions[:, :num_tokens])
    mtp.model_state = types.SimpleNamespace(rope_state=rope_state)
    input_batch = step_batch()
    num_reqs = input_batch.num_reqs

    # the runner's step: prepare_propose, then propose (here as the profile calls it: the positions take the same path)
    mtp.prepare_propose(input_batch, torch.ones(num_reqs, dtype=torch.int32), torch.zeros(num_reqs), None)
    profile_propose(mtp, input_batch)

    first_pass, *draft_steps = drafter.positions
    assert torch.equal(first_pass, TARGET_POSITIONS)
    assert len(draft_steps) == NUM_SPECULATIVE_TOKENS - 1
    for step, positions in enumerate(draft_steps, start=1):
        assert torch.equal(positions, TARGET_POSITIONS[:, LAST_TOKENS] + step)


def test_each_draft_step_moves_every_m_rope_row_one_on_within_the_model_s_length(drafter):
    mtp = loaded_speculator(drafter)
    # a text token and a token at the model's last position sampled at, then a padding request
    mtp.mrope_positions[:, :3] = torch.tensor([[40, MAX_MODEL_LEN - 2, 5]] * MROPE_ROWS)
    mtp.last_token_indices[:2] = torch.tensor([0, 1])
    num_reqs = 2

    input_batch = types.SimpleNamespace(seq_lens=torch.zeros(num_reqs, dtype=torch.int32))
    mtp._prepare_decode_inputs(input_batch, torch.zeros(num_reqs, dtype=torch.int32), num_reqs, False, None)
    assert mtp.mrope_positions[:, :3].tolist() == [[41, MAX_MODEL_LEN - 1, 5]] * MROPE_ROWS
    mtp._decode(num_reqs, num_reqs, None, None, None, None, NUM_SPECULATIVE_TOKENS)
    assert mtp.mrope_positions[:, :3].tolist() == [[42, MAX_MODEL_LEN - 1, 5]] * MROPE_ROWS


def test_a_proposal_reports_its_drafter_s_nan_and_inf_counts_for_the_next_step_and_a_dummy_run_none(monkeypatch):
    # vLLM's propose as two draft steps whose argmax adds the rows' counts, over a buffer an earlier call left non-zero
    counts = torch.full((MAX_REQS,), 7, dtype=torch.int32)
    draft_tokens = torch.zeros(len(LAST_TOKENS), NUM_SPECULATIVE_TOKENS, dtype=torch.int64)

    def propose(self, input_batch, *args, **kwargs):
        counts[: input_batch.num_reqs] += torch.tensor([0, 3], dtype=torch.int32)
        counts[: input_batch.num_reqs] += torch.tensor([0, 1], dtype=torch.int32)
        return draft_tokens

    monkeypatch.setattr(MTPSpeculator, "propose", propose)
    mtp = Qwen3_5MTPSpeculator.__new__(Qwen3_5MTPSpeculator)
    mtp.model = types.SimpleNamespace(nonfinite_counts=counts)
    input_batch = types.SimpleNamespace(num_reqs=len(LAST_TOKENS), req_ids=["a", "b"])
    NonfiniteLogits.take()  # counts an earlier test left queued

    # a dummy run reports nothing; a step's proposal reports its own counts, a copy the next proposal leaves as it is
    assert mtp.propose(input_batch, dummy_run=True) is draft_tokens
    assert NonfiniteLogits.take() == []
    assert mtp.propose(input_batch) is draft_tokens
    (report,) = NonfiniteLogits.take()
    counts.zero_()
    assert report.kernel == Qwen3_5MTPSpeculator.DRAFT_KERNEL and report.req_ids == ["a", "b"]
    assert report.counts.tolist() == [0, 4] and report.cu_num_logits is None

    # a drafter whose argmax counts nothing (vLLM's own) reports nothing
    mtp.model = types.SimpleNamespace()
    assert mtp.propose(input_batch) is draft_tokens
    assert NonfiniteLogits.take() == []


def write_checkpoint_config(directory, mtp_layers: int) -> None:
    """A Qwen3.8-27B checkpoint's config.json with the given MTP layers in its text config."""
    Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config={"mtp_num_hidden_layers": mtp_layers, "rope_parameters": ROPE},
    ).save_pretrained(directory)


def family_speculative_config(tmp_path, monkeypatch, **target_options) -> SpeculativeConfig:
    """The family's speculative config, its method as PEAR's vLLM plugin registers it: a checkpoint the actor exported
    (verl zeroes the MTP layers of an actor without them) as the target, of the given ModelConfig options, the base
    model as the drafter."""
    target_directory, draft_directory = tmp_path / "exported", tmp_path / "base"
    write_checkpoint_config(target_directory, 0)
    write_checkpoint_config(draft_directory, 1)
    monkeypatch.setattr(vllm_speculator_config, "_CUSTOM_CONFIGS", {})
    vllm_speculator_config.register_speculative_method(
        Qwen3_5RolloutKernels.SPECULATIVE_METHOD, Qwen3_5RolloutKernels.SPECULATOR_CONFIG
    )

    return SpeculativeConfig(
        model=str(draft_directory),
        method=Qwen3_5RolloutKernels.SPECULATIVE_METHOD,
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        target_model_config=ModelConfig(model=str(target_directory), **target_options),
        target_parallel_config=ParallelConfig(),
    )


def test_the_drafter_is_built_as_the_draft_checkpoint_describes_it(tmp_path, monkeypatch):
    spec = family_speculative_config(tmp_path, monkeypatch)
    target_config = spec.target_model_config

    # vLLM's draft config: the draft checkpoint's, as its MTP model with the MTP layer the checkpoint holds
    assert spec.draft_model_config.architectures == ["Qwen3_5MTP"]
    assert spec.draft_model_config.hf_text_config.mtp_num_hidden_layers == 1
    assert target_config.hf_text_config.mtp_num_hidden_layers == 0

    # the drafter built on that config (Qwen3_5MTP counts its MTP layers in the model config it is built with), the
    # engine's config left the target's
    drafter_vllm_configs = []
    monkeypatch.setattr(
        speculator_module, "load_eagle_model", lambda model, config: drafter_vllm_configs.append(config)
    )
    mtp = Qwen3_5MTPSpeculator.__new__(Qwen3_5MTPSpeculator)
    mtp.vllm_config = types.SimpleNamespace(model_config=target_config, speculative_config=spec)
    mtp.draft_model_config = spec.draft_model_config

    mtp.load_draft_model(None, set())

    (drafter_vllm_config,) = drafter_vllm_configs
    assert drafter_vllm_config.model_config is spec.draft_model_config
    assert drafter_vllm_config.speculative_config is spec and mtp.vllm_config.model_config is target_config


def test_the_drafter_s_logits_take_the_target_s_head_dtype(tmp_path, monkeypatch):
    # PEAR's engine: bf16 weights, the lm_head's logits in fp32 by config.yaml's hf_overrides, which vLLM keeps from a
    # draft's config
    spec = family_speculative_config(tmp_path, monkeypatch, dtype="bfloat16", hf_overrides={"head_dtype": "float32"})

    assert spec.draft_model_config.dtype == torch.bfloat16
    assert spec.draft_model_config.head_dtype == spec.target_model_config.head_dtype == torch.float32
