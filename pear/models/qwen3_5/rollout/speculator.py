import copy
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from pear.models.qwen3_5.rollout.nonfinite_logits import NonfiniteLogits
from pear.models.qwen3_5.rollout.triton_launcher import TritonLauncher


@triton.jit(do_not_specialize=["draft_tokens_stride", "max_model_len", "max_num_reqs", "mrope_stride", "num_reqs"])
def _prepare_draft_steps_kernel(
    DRAFT_TOKENS,
    draft_tokens_stride,
    TARGET_SEQ_LENS,
    NUM_REJECTED,
    INPUT_IDS,
    POSITIONS,
    SAMPLE_SRC_POSITIONS,
    QUERY_START_LOC,
    SEQ_LENS,
    MROPE,
    mrope_stride,
    LAST_TOKEN_INDICES,
    max_model_len,
    max_num_reqs,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    DIMS: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
):
    # vLLM 0.34's _prepare_decode_inputs_kernel (a program per request, then the padding) and a program more for the
    # first draft step's M-RoPE positions: one past those the first pass sampled at, all loads before any store (the
    # gathered columns may be other requests' destinations)
    req_idx = tl.program_id(0)
    if req_idx == num_reqs + 1:
        r = tl.arange(0, REQ_BLOCK)
        mask = r < num_reqs
        columns = tl.load(LAST_TOKEN_INDICES + r, mask=mask, other=0)
        for d in tl.static_range(DIMS):
            mrope = tl.load(MROPE + d * mrope_stride + columns, mask=mask, other=0)
            tl.debug_barrier()
            tl.store(MROPE + d * mrope_stride + r, tl.minimum(mrope + 1, max_model_len - 1), mask=mask)
        return
    if req_idx == num_reqs:
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            tl.store(QUERY_START_LOC + block, q, mask=block < max_num_reqs + 1)
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            tl.store(SEQ_LENS + block, 0, mask=block < max_num_reqs)
        return

    draft_token = tl.load(DRAFT_TOKENS + req_idx * draft_tokens_stride)
    tl.store(INPUT_IDS + req_idx, draft_token)
    sample_position = tl.load(SAMPLE_SRC_POSITIONS + req_idx)
    tl.store(SAMPLE_SRC_POSITIONS + req_idx, sample_position + 1)
    seq_len = tl.load(TARGET_SEQ_LENS + req_idx) - tl.load(NUM_REJECTED + req_idx)
    if ADVANCE_DRAFT_POSITIONS:
        position = tl.load(POSITIONS + req_idx)
        tl.store(POSITIONS + req_idx, tl.minimum(position + 1, max_model_len - 1))
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(SEQ_LENS + req_idx, seq_len)


@triton.jit(
    do_not_specialize=[
        "draft_tokens_stride",
        "next_hidden_stride",
        "hidden_stride",
        "hidden_size",
        "mrope_stride",
        "num_speculative_steps",
    ]
)
def _update_draft_steps_kernel(
    DRAFT_TOKENS,
    draft_tokens_stride,
    NEXT_HIDDEN,
    next_hidden_stride,
    INPUT_IDS,
    POSITIONS,
    SAMPLE_SRC_POSITIONS,
    SEQ_LENS,
    SAMPLED,
    CURRENT_DRAFT_STEP,
    HIDDEN,
    hidden_stride,
    hidden_size,
    MAX_MODEL_LEN,
    MROPE,
    mrope_stride,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    DIMS: tl.constexpr,
):
    # vLLM 0.34's _update_draft_inputs_kernel (a program per request) and the request's M-RoPE rows one past the last;
    # after the final draft step only its token
    req_idx = tl.program_id(0)
    draft_token = tl.load(SAMPLED + req_idx)
    step = tl.load(CURRENT_DRAFT_STEP)
    tl.store(DRAFT_TOKENS + req_idx * draft_tokens_stride + step, draft_token)
    if step >= num_speculative_steps - 1:
        return

    sample_position = tl.load(SAMPLE_SRC_POSITIONS + req_idx)
    tl.store(SAMPLE_SRC_POSITIONS + req_idx, sample_position + 1)
    tl.store(INPUT_IDS + req_idx, draft_token)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden = tl.load(HIDDEN + req_idx * hidden_stride + block, mask=mask)
        tl.store(NEXT_HIDDEN + req_idx * next_hidden_stride + block, hidden, mask=mask)
    max_model_len = tl.load(MAX_MODEL_LEN)
    if ADVANCE_DRAFT_POSITIONS:
        position = tl.load(POSITIONS + req_idx)
        tl.store(POSITIONS + req_idx, tl.minimum(position + 1, max_model_len - 1))
        seq_len = tl.load(SEQ_LENS + req_idx)
        tl.store(SEQ_LENS + req_idx, tl.minimum(seq_len + 1, max_model_len))
    last_position = max_model_len.to(tl.int64) - 1
    for d in tl.static_range(DIMS):
        mrope = tl.load(MROPE + d * mrope_stride + req_idx)
        tl.store(MROPE + d * mrope_stride + req_idx, tl.minimum(mrope + 1, last_position))


class Qwen3_5MTPSpeculator(MTPSpeculator):
    """vLLM's MTP speculator whose drafter, built as its own checkpoint describes it, rotates with the target's M-RoPE
    positions: its first pass at the positions the target gave the same tokens, each draft step every M-RoPE row one
    past the last (the slot mappings keep vLLM's 1-D positions)."""

    PAD_BLOCK = 1024  # the padding program's block (vLLM's BLOCK_SIZE)
    HIDDEN_BLOCK = 1024  # the draft inputs' update copies a hidden state in blocks of this many (vLLM's BLOCK_SIZE)
    # the drafter's argmax, as an error about its rows names it in the output of the step that verifies its drafts
    DRAFT_KERNEL = "the MTP drafter's argmax of the previous step"
    prepare_draft_steps_kernel = TritonLauncher(_prepare_draft_steps_kernel)
    update_draft_steps_kernel = TritonLauncher(_update_draft_steps_kernel)

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # the draft forward's positions, one column more than the tokens so that torch.compile takes a slice as a
        # strided view (as vLLM's RopeState keeps its own)
        self.mrope_positions = torch.zeros(
            (vllm_config.model_config.mrope_num_dims, self.max_num_tokens + 1), dtype=torch.int64, device=device
        )

    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]) -> torch.nn.Module:
        # the drafter of speculative_config.model's own config, whatever the target's says: vLLM's loader builds the
        # draft class with the engine config it is handed, the target's model config in MTPSpeculator's, and Qwen3_5MTP
        # counts its MTP layers there (MTPSpeculator's top-k index sharing is for MTP layers with an indexer)
        drafter_vllm_config = copy.copy(self.vllm_config)
        drafter_vllm_config.model_config = self.draft_model_config

        return load_eagle_model(target_model, drafter_vllm_config)

    def prepare_propose(
        self, input_batch: InputBatch, num_sampled: torch.Tensor, num_rejected: torch.Tensor, sampler: Sampler
    ) -> None:
        super().prepare_propose(input_batch, num_sampled, num_rejected, sampler)

        # the first pass's positions: the target's M-RoPE positions of the step. vLLM calls this hook before a step's
        # propose, never a dummy run's (the memory profile's runs before set_attn links the target's model state): a
        # dummy run's drafts, discarded, run on the positions left in the buffer
        num_tokens = input_batch.num_tokens_after_padding
        self.mrope_positions[:, :num_tokens] = self.model_state.rope_state.get_positions(num_tokens)

    def propose(self, input_batch: InputBatch, *args, dummy_run: bool = False, **kwargs) -> torch.Tensor:
        # a drafter whose argmax counts NaN and +inf logits (BlackwellQwen3_5MTP, HopperQwen3_5MTP): its counts over
        # this proposal's draft steps, which the next step's output (whose verify step takes these drafts) raises on
        counts = None if dummy_run else getattr(self.model, "nonfinite_counts", None)
        if counts is not None:
            counts.zero_()

        draft_tokens = super().propose(input_batch, *args, dummy_run=dummy_run, **kwargs)

        if counts is not None:
            NonfiniteLogits.report(self.DRAFT_KERNEL, counts[: input_batch.num_reqs].clone(), input_batch.req_ids)

        return draft_tokens

    def _copy_request_inputs(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dummy_run: bool = False,
    ) -> None:
        # argmax drafts read the requests' slots, never their temperatures or seeds: vLLM 0.34's copy of the slots
        # without its copies of both (from host-backed tensors, every step)
        if not (self.use_local_argmax_reduction and self.draft_logits is None):
            super()._copy_request_inputs(num_reqs, idx_mapping, temperature, seeds, dummy_run)
            return

        if dummy_run:
            self.idx_mapping.fill_(-1)
        else:
            self.idx_mapping[:num_reqs].copy_(idx_mapping)
            self.idx_mapping[num_reqs:].fill_(-1)

    def _prepare_decode_inputs(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
    ) -> None:
        # vLLM's draft-step inputs and the first draft step's M-RoPE positions, one kernel
        self.prepare_draft_steps_kernel(
            (self.advance_draft_positions,),
            (num_reqs + 2,),
            self.draft_tokens[:num_reqs, 0],
            self.draft_tokens.stride(0),
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.sample_src_positions,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            self.mrope_positions,
            self.mrope_positions.stride(0),
            self.last_token_indices,
            self.max_model_len,
            self.max_num_reqs,
            num_reqs,
            BLOCK_SIZE=self.PAD_BLOCK,
            ADVANCE_DRAFT_POSITIONS=self.advance_draft_positions,
            DIMS=self.mrope_positions.shape[0],
            REQ_BLOCK=triton.next_power_of_2(self.max_num_reqs),
        )

    def _decode(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        num_speculative_steps: int,
    ) -> None:
        # vLLM 0.34's TargetDependentARSpeculator._decode, the draft inputs' update and the M-RoPE advance one kernel
        self._prepare_eplb_forward(num_reqs)
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode
        )
        sampled = self.sample_draft(
            last_hidden_states[:num_reqs],
            self.sample_src_positions[:num_reqs],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        self.update_draft_steps_kernel(
            (self.advance_draft_positions,),
            (num_reqs,),
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self.hidden_states,
            self.hidden_states.stride(0),
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.sample_src_positions,
            self.input_buffers.seq_lens,
            sampled,
            self.current_draft_step,
            hidden_states,
            hidden_states.stride(0),
            hidden_states.shape[1],
            self.max_model_len_gpu,
            self.mrope_positions,
            self.mrope_positions.stride(0),
            num_speculative_steps,
            BLOCK_SIZE=self.HIDDEN_BLOCK,
            ADVANCE_DRAFT_POSITIONS=self.advance_draft_positions,
            DIMS=self.mrope_positions.shape[0],
        )

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        skip_sample: bool = False,
    ) -> None:
        # vLLM 0.34's TargetDependentARSpeculator._prefill without prefill context parallelism, the drafter finishing
        # past its attention only the rows it samples at (every row in piecewise graphs, captured per token count)
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        # the output at P and the token at P + 1 draft the token at P + 2, whose draw is keyed by P + 1
        sample_src_positions = positions + 1
        finishes_samples = (
            getattr(self.model, "finishes_sample_rows", False)
            and not skip_sample
            and cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        sample_rows = last_token_indices if finishes_samples else None
        hidden_states, _ = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
            sample_rows=sample_rows,
        )
        if skip_sample:
            return

        sample_hidden_states = hidden_states if sample_rows is not None else hidden_states[last_token_indices]
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        self.hidden_states[:num_reqs] = sample_hidden_states
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        sample_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vLLM 0.34's TargetDependentARSpeculator._run_model on the M-RoPE positions, without prefill context
        # parallelism (Qwen3_5MTPSpeculatorConfig.supports_pcp); Qwen3_5MTP returns its final norm's output alone
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            # the drafter looks its token rows up itself unless image rows join them
            inputs_embeds = None
            if self.supports_mm_inputs and mm_inputs is not None and len(mm_inputs[0]) > 0:
                mm_embeds, is_mm_embed = mm_inputs
                num_input_tokens = is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.mrope_positions[:, :num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                hidden_states = self.prefill_cudagraph_manager.run_pw_graph(self.model, model_inputs)
            else:
                rows = {} if sample_rows is None else {"sample_rows": sample_rows}
                hidden_states = self.model(**model_inputs, **rows)

        return hidden_states, hidden_states
