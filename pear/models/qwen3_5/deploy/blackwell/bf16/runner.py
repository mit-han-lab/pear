from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, post_update
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample import sampler as vllm_sampler
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as vllm_rejection
from vllm.v1.worker.workspace import use_workspace_lane

from pear.models.qwen3_5.deploy.blackwell.bf16.draft.speculator import ProposalLaunch
from pear.models.qwen3_5.deploy.blackwell.bf16.extension import BlackwellBf16Extension
from pear.models.qwen3_5.deploy.blackwell.bf16.schedule import NT, SS_HOST
from pear.models.qwen3_5.deploy.blackwell.bf16.step import DecodeStep, ModelView
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.worker.gpu.model_runner import BatchReqState
    from vllm.v1.worker.gpu.sample.sampler import Sampler

logger = get_logger(__name__)
launch_log = get_logger(__name__ + ".launches", default_level="INFO")  # one line per launch and fallback, routed apart

MROPE_SECTIONS = 3  # the (t, h, w) positions of a token
INT32_BYTES = 4  # of an entry of num_accepted_tokens_gpu


class TopKLogProbs:
    """vLLM's top-k log-probs of a step's logits rows (compute_topk_scores' fast path) as one launch."""

    K = 32  # the top k the kernel computes (logprobs.cuh LP_K)
    PARTIAL_BYTES = 16 + 8 * K  # a chunk's LpPartial (logprobs.cuh): m, s, count, padding and K 8-byte keys
    MAX_ROWS = 64  # rows of one call at most: the partials' and the row counters' rows
    MAX_CHUNKS = 32  # CTAs per row at most (logprobs.cuh LP_MAX_CHUNKS)
    VEC_BYTES = 16  # the kernel's loads: 16-byte vectors of a row (logprobs.cuh LP_VEC_BYTES)
    CHUNK_VECS = 256 * 8  # a CTA's vectors at most (logprobs.cuh LP_CHUNK_VECS)

    def __init__(self, device: torch.device) -> None:
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.partials = torch.empty(
            self.MAX_ROWS * self.MAX_CHUNKS * self.PARTIAL_BYTES, dtype=torch.uint8, device=device
        )
        self.row_counters = torch.zeros(self.MAX_ROWS, dtype=torch.int32, device=device)

    def is_applicable(
        self, logits: torch.Tensor, num_logprobs: int, max_per_req_token_ids: int, logits_mode: bool
    ) -> bool:
        """Whether the call is the fast path this kernel computes."""
        if not (
            num_logprobs == self.K
            and max_per_req_token_ids == 0
            and not logits_mode
            and logits.dim() == 2
            and logits.stride(1) == 1
            and logits.dtype in (torch.bfloat16, torch.float32)
            and 0 < logits.shape[0] <= self.MAX_ROWS
        ):
            return False

        values_per_vec = self.VEC_BYTES // logits.element_size()

        return (
            logits.stride(0) % values_per_vec == 0
            and logits.data_ptr() % self.VEC_BYTES == 0
            and logits.shape[1] <= self.MAX_CHUNKS * self.CHUNK_VECS * values_per_vec
        )

    def __call__(
        self, logits: torch.Tensor, sampled_token_ids: torch.Tensor, cu_num_logits: list[int] | torch.Tensor | None
    ) -> LogprobsTensors:
        """The rows' LogprobsTensors, as compute_topk_scores returns them."""
        rows = logits.shape[0]
        chunks = max(1, min(self.MAX_CHUNKS, self.num_sms // rows))
        token_ids = torch.empty((rows, self.K + 1), dtype=torch.int64, device=logits.device)
        logprobs = torch.empty((rows, self.K + 1), dtype=torch.float32, device=logits.device)
        ranks = torch.empty(rows, dtype=torch.int64, device=logits.device)
        torch.ops.pear_deploy_blackwell_bf16.topk_logprobs(
            logits,
            sampled_token_ids.to(torch.int64),
            self.partials,
            self.row_counters,
            chunks,
            token_ids,
            logprobs,
            ranks,
        )

        return LogprobsTensors(
            logprob_token_ids=token_ids,
            logprobs=logprobs,
            selected_token_ranks=ranks,
            cu_num_generated_tokens=cu_num_logits,
        )


class SpecSampler:
    """vLLM's rejection sampler over the top-p nucleus of one request's verify step, as instructions of its launch."""

    MIN_ROWS = 2  # a verify step: the token and at least one draft

    def is_applicable(self, runner: GPUModelRunner, input_batch: InputBatch, logits: torch.Tensor) -> bool:
        """Whether the launch samples this step: one request, the standard rule, temperature 1, no logits processing but
        top-p, no log-probs."""
        rejection_sampler, speculator = runner.rejection_sampler, runner.speculator
        if (
            rejection_sampler is None
            or input_batch.num_reqs != 1
            or rejection_sampler.synthetic_conditional_rates is not None
        ):
            return False
        if rejection_sampler.use_block_verification or speculator is None:
            return False
        if not self.MIN_ROWS <= logits.shape[0] == input_batch.num_draft_tokens + 1 <= NT:
            return False

        sampler, req_idx = rejection_sampler.sampler, int(input_batch.idx_mapping_np[0])
        if sampler.sampling_states.temperature.np[req_idx] != 1.0 or not self.is_only_top_p(sampler, req_idx):
            return False
        if sampler.compute_nans:
            return False

        draft_logits = speculator.draft_logits

        return (
            sampler.sampling_states.max_num_logprobs(input_batch.idx_mapping_np) == NO_LOGPROBS
            and draft_logits.dtype == torch.float32
            and draft_logits.is_contiguous()
            and draft_logits.shape[-1] == logits.shape[1]
            and input_batch.input_ids.dtype == torch.int32
            and input_batch.positions.dtype == torch.int64
            and input_batch.logits_indices.dtype == torch.int64
        )

    @staticmethod
    def is_only_top_p(sampler: Sampler, req_idx: int) -> bool:
        """Whether the request's logits processing (vLLM's Sampler.add_request) is at most top-p."""
        states, budget = sampler.sampling_states, sampler.thinking_budget_state

        return not (
            sampler.logit_bias_state.use_logit_bias[req_idx]
            or sampler.penalties_state.use_penalty[req_idx]
            or sampler.bad_words_state.num_bad_words.np[req_idx] > 0
            or (budget.enabled and budget.use_thinking_budget[req_idx])
            or states.min_p.np[req_idx] != 0.0
            or states.top_k.np[req_idx] != states.vocab_size
        )

    @staticmethod
    def inputs(runner: GPUModelRunner, input_batch: InputBatch) -> dict[str, int | float]:
        """The launch's sampling inputs: their device addresses (step.py SS_INPUTS) and the request's top_p."""
        req_idx = int(input_batch.idx_mapping_np[0])
        states = runner.rejection_sampler.sampler.sampling_states

        return dict(
            draft=runner.speculator.draft_logits[req_idx].data_ptr(),
            seed=states.seeds.gpu[req_idx : req_idx + 1].data_ptr(),
            logits_indices=input_batch.logits_indices.data_ptr(),
            input_ids=input_batch.input_ids.data_ptr(),
            positions=input_batch.positions.data_ptr(),
            top_p=float(states.top_p.np[req_idx]),
        )

    @staticmethod
    def output(launch_outputs: tuple[torch.Tensor, ...]) -> SamplerOutput:
        """vLLM's RejectionSampler output for the step, from the launch's outputs (step.py spec_outputs)."""
        sampled_token_ids, num_sampled, num_rejected = launch_outputs

        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )


class LaunchedOutput(AsyncOutput):
    """V2's AsyncOutput of a launched spec step whose outputs the next launch's OP_PREV copies into the host ring."""

    # get_output's wait for the next launch to set the slot's flag (the slot is in the host ring), else it copies the
    # slot itself
    WAIT_US = 30_000_000

    def __init__(
        self, model_runner_output: ModelRunnerOutput, sampler_output: SamplerOutput, step: DecodeStep, slot: int
    ) -> None:
        T = sampler_output.sampled_token_ids.shape[1]
        self.model_runner_output, self.sampler_output = model_runner_output, sampler_output
        self.num_sampled_tokens, self.routed_experts, self._has_fault = sampler_output.num_sampled, None, None
        self.step, self.slot, self.seq, self.is_copied = step, slot, None, False
        host_ring = step.ss_host
        self.sampled_token_ids = host_ring["sampled"][slot, :T].view(1, T).numpy()
        self.logprobs_tensors = None
        self.num_nans, self.sampling_mask_tensors, self.routed_experts_cpu = None, None, None
        self.num_sampled_tokens_np = host_ring["num_sampled"][slot : slot + 1].numpy()
        self.prompt_logprobs_dict = {
            req_id: None if logprobs is None else logprobs.to_cpu_nonblocking()
            for req_id, logprobs in model_runner_output.prompt_logprobs_dict.items()
        }
        self.copy_event = torch.cuda.Event(blocking=True)  # recorded by copy_slot only

    def get_output(self) -> ModelRunnerOutput:
        if self.seq is None or not torch.ops.pear_deploy_blackwell_bf16.wait_flag(
            self.step.ss_flags, self.slot, self.seq, self.WAIT_US
        ):
            self.copy_slot()

        return super().get_output()

    def copy_slot(self) -> None:
        """Copy the ring slot's outputs into the host ring on the current stream."""
        for k in SS_HOST:
            self.step.ss_host[k][self.slot].copy_(self.step.ss_ring[k][self.slot], non_blocking=True)
        self.is_copied = True
        self.copy_event.record()


class DecodeLaunch:
    """vLLM's V2 model runner with every decode or verify step of a single request run as one megakernel launch."""

    SEQ_MASK = 0x7FFFFFFF  # the host ring flags' sequence wraps within non-negative int32 (wait_flag's)

    # one model runner per process (TP1): the process's step and the sampling paths its launches take
    step: DecodeStep | None = None
    topk: TopKLogProbs | None = None
    spec_sampler: SpecSampler | None = None
    # the outputs of the step just launched: its spec sampler's, or its logits for vLLM's sampler
    pending_spec_outputs: tuple[torch.Tensor, ...] | None = None
    pending_logits: torch.Tensor | None = None
    const_inputs: dict[tuple[int, int], dict[str, torch.Tensor]] = {}  # (request row, T) -> own_inputs' index tensors
    # whether vLLM's state post-processing to come is the launched step's, which keeps only the committed counts
    is_post_of_launch: bool = False
    # the OP_GDNT record copy written by each request's last step if that was a launch (the next launch writes the
    # other); a vLLM step in between starts the request over from vLLM's state slots
    gdnt_parities: dict[str, int] = {}
    # the launch's outputs stand in for V2's per-step copies: the ring slot of the last spec launch, the request row
    # whose copies a vLLM step must sync first, and whether sample_tokens is the launch's own
    ring_slot: int | None = None
    stale_row: tuple[int, int] | None = None
    is_own_sample: bool = False
    # the last own-sample launch, whose post_update and host outputs the next launch's OP_PREV applies (a vLLM step or
    # its request's end applies them with V2's kernel instead), and the host ring flags' sequence
    pending_launch: dict | None = None
    seq: int = 0
    launches: int = 0  # the launch log's counts
    fallbacks: int = 0

    @classmethod
    def install(cls) -> None:
        """Patch the worker's KV cache initialization, the V2 runner's step and sampling, and vLLM's top-k log-probs."""
        vllm_initialize = gpu_worker.Worker.initialize_from_config
        vllm_execute = GPUModelRunner.execute_model
        vllm_sample = GPUModelRunner.sample
        vllm_post = MambaHybridModelState.postprocess_state
        vllm_topk = vllm_rejection.compute_topk_scores
        vllm_sample_tokens = GPUModelRunner.sample_tokens

        def initialize_from_config(self, kv_cache_config):
            vllm_initialize(self, kv_cache_config)
            runner = self.model_runner
            view = ModelView.from_runner(runner)
            reason = DecodeStep.unsupported_reason(view)
            if reason is not None:
                raise RuntimeError(f"the decode megakernel cannot run this engine: {reason}")

            torch.cuda.synchronize()
            allocated_before = torch.cuda.memory_allocated()
            # the draft's proposal in the launch, on its bf16 weights read in place as vLLM's own draft paths read them
            proposal = ProposalLaunch.ensure(runner)
            # the kernel is built for verify steps of NT rows: NT - 1 drafts
            if proposal is None or proposal.nstep + 1 != NT:
                raise RuntimeError(
                    f"the decode megakernel runs the verify steps of a DFlash 2 draft of {NT - 1} tokens"
                )
            BlackwellBf16Extension.load()
            torch.cuda.synchronize()
            allocated_after_draft = torch.cuda.memory_allocated()
            cls.step = DecodeStep(view, runner.device, proposal)
            cls.topk = TopKLogProbs(runner.device)
            torch.cuda.synchronize()
            allocated_after_step, (free_bytes, total_bytes) = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()
            # the megakernel's own buffers live outside vLLM's gpu_memory_utilization share
            logger.warning(
                "decode megakernel: its own buffers %.1f MiB (the step %.1f MiB, the draft's proposal %.1f MiB); "
                "device free %.2f of %.2f GiB",
                (allocated_after_step - allocated_before) / 2**20,
                (allocated_after_step - allocated_after_draft) / 2**20,
                (allocated_after_draft - allocated_before) / 2**20,
                free_bytes / 2**30,
                total_bytes / 2**30,
            )
            cls.spec_sampler = SpecSampler()

        def compute_topk_scores(
            logits,
            num_logprobs,
            sampled_token_ids,
            cu_num_logits=None,
            logprob_token_ids_state=None,
            expanded_idx_mapping=None,
            max_per_req_token_ids=0,
            logits_mode=False,
        ):
            if cls.topk is not None and cls.topk.is_applicable(
                logits, num_logprobs, max_per_req_token_ids, logits_mode
            ):
                return cls.topk(logits, sampled_token_ids, cu_num_logits)

            return vllm_topk(
                logits,
                num_logprobs,
                sampled_token_ids,
                cu_num_logits,
                logprob_token_ids_state,
                expanded_idx_mapping,
                max_per_req_token_ids,
                logits_mode,
            )

        @torch.inference_mode()
        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, *args, **kwargs):
            if not dummy_run and cls.step is not None:
                for req_id in scheduler_output.finished_req_ids:
                    cls.gdnt_parities.pop(req_id, None)
                if intermediate_tensors is None:
                    req_id, reason = cls.request_to_launch(self, scheduler_output)
                    if req_id is not None:
                        return cls.launch(self, scheduler_output, req_id)
                    cls.fallbacks += 1
                    launch_log.info("decode megakernel fallback %d: vLLM's step (%s)", cls.fallbacks, reason)
                for req_id in scheduler_output.num_scheduled_tokens:  # vLLM's step: the next launch starts over
                    cls.gdnt_parities.pop(req_id, None)

            cls.sync(self)

            return vllm_execute(self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)

        def sample(self, hidden_states, input_batch, grammar_output):
            if cls.step is None or hidden_states.data_ptr() != cls.step.hidden.data_ptr():
                return vllm_sample(self, hidden_states, input_batch, grammar_output)
            if cls.pending_spec_outputs is not None:  # the launch sampled its step (OP_SS_*)
                launch_outputs, cls.pending_spec_outputs = cls.pending_spec_outputs, None
                sample_output = SpecSampler.output(launch_outputs)
                return sample_output, sample_output.num_sampled, sample_output.num_rejected

            # the step's logits [T, V]: vLLM's sampler or rejection sampler
            logits, cls.pending_logits = cls.pending_logits, None
            if input_batch.num_draft_tokens == 0:
                sample_output = self.sampler(logits, input_batch)
            else:
                sample_output = self.rejection_sampler(logits, input_batch, self.speculator.draft_logits)

            return sample_output, sample_output.num_sampled, sample_output.num_rejected

        def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
            if cls.is_post_of_launch:  # the launch's step: the committed counts only (OP_GDNT keeps the states itself)
                cls.is_post_of_launch = False
                num_computed_tokens = None
                if cls.ring_slot is not None:  # the next launch reads them from the ring slot, sync before a vLLM step
                    return

            vllm_post(self, idx_mapping, num_sampled, num_computed_tokens)

        def sample_tokens(self, grammar_output):
            if not cls.is_own_sample:
                return vllm_sample_tokens(self, grammar_output)

            cls.is_own_sample = False

            return cls.launched_sample_tokens(self, grammar_output)

        gpu_worker.Worker.initialize_from_config = initialize_from_config
        vllm_rejection.compute_topk_scores = compute_topk_scores
        vllm_sampler.compute_topk_scores = compute_topk_scores
        GPUModelRunner.execute_model = execute_model
        GPUModelRunner.sample = sample
        GPUModelRunner.sample_tokens = sample_tokens
        MambaHybridModelState.postprocess_state = postprocess_state

    @classmethod
    def request_to_launch(
        cls, runner: GPUModelRunner, scheduler_output: SchedulerOutput
    ) -> tuple[str | None, str | None]:
        """The request id of a verify step the megakernel runs (one request past its prompt), else None and why."""
        if len(scheduler_output.num_scheduled_tokens) != 1:
            return None, f"{len(scheduler_output.num_scheduled_tokens)} requests in the step"
        if scheduler_output.scheduled_new_reqs or scheduler_output.scheduled_encoder_inputs:
            return None, "a new request or encoder inputs (prefill)"
        if (
            scheduler_output.kv_cache_block_copies
            or scheduler_output.has_structured_output_requests
            or scheduler_output.kv_connector_metadata is not None
        ):
            return None, "block copies, structured output or a KV connector"

        ((req_id, n),) = scheduler_output.num_scheduled_tokens.items()
        if n != 1 + len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())) or n > NT:
            return (
                None,
                f"{n} tokens scheduled ({len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))} drafts)",
            )
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if req_id not in cached_reqs.req_ids or req_id in cached_reqs.resumed_req_ids:
            return None, "a resumed or uncached request"
        req_idx = runner.req_states.req_id_to_index[req_id]
        if (
            cached_reqs.num_computed_tokens[cached_reqs.req_ids.index(req_id)]
            < runner.req_states.prefill_len.np[req_idx]
        ):
            return None, "prompt tokens (prefill)"

        return req_id, None

    @classmethod
    def launch(cls, runner: GPUModelRunner, scheduler_output: SchedulerOutput, req_id: str) -> None:
        """Run vLLM's bookkeeping of the step, then its launch, and leave the runner's state for sample_tokens."""
        cls.retire_pending(runner, scheduler_output)
        runner.update_pp_decode_requests()
        runner.finish_requests(scheduler_output)
        runner.free_states(scheduler_output)
        runner.add_requests(scheduler_output)
        runner.update_requests(scheduler_output)  # also zeroes the newly allocated blocks
        runner.block_tables.apply_staged_writes()
        batch_req_state, _ = runner.gather_batch_req_state(scheduler_output, False)
        T = batch_req_state.num_tokens
        req_idx = runner.req_states.req_id_to_index[req_id]
        is_booked = cls.can_book(runner, batch_req_state)
        prev_launch = cls.take_pending(runner, req_idx, is_booked)
        if is_booked:  # the launch writes the step's inputs itself (OP_BOOK): prepare_inputs' host-side batch only
            input_batch = cls.own_inputs(runner, scheduler_output, batch_req_state, req_idx)
        else:
            batch_desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=T, num_reqs=1)
            # the tokens: last sample, drafts
            input_batch = runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc)
            runner.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=input_batch.num_reqs_after_padding
            )

        # the record: the host's position (an upper bound under asynchronous scheduling: the kernel reads the device's),
        # M-RoPE's delta, OP_GDNT's record parity and the committed count of the step before
        pos = int(runner.req_states.num_computed_tokens_np[req_idx])
        rope_state = runner.model_state.rope_state
        delta = int(rope_state.prefill_delta.np[req_idx]) if rope_state is not None and rope_state.has_delta else 0
        parity = cls.gdnt_parities.get(req_id)
        cls.gdnt_parities[req_id] = 0 if parity is None else 1 - parity
        num_accepted = getattr(runner.model_state, "num_accepted_tokens_gpu", None)
        slot = cls.step.slot  # the ring slot this launch's spec sampler writes (step.launch advances it)
        if cls.ring_slot is not None:  # the step before was a spec launch: its count in its ring slot
            committed_ptr = cls.step.ss_ring["num_sampled"][cls.ring_slot].data_ptr()
        else:
            committed_ptr = None if num_accepted is None else num_accepted.data_ptr() + INT32_BYTES * req_idx
        logits = cls.step.logits[:T]
        spec_inputs = None  # without the spec sampler, vLLM's sampler or rejection sampler runs on the launch's logits
        if cls.spec_sampler.is_applicable(runner, input_batch, logits):
            spec_inputs = cls.spec_sampler.inputs(runner, input_batch)

        # the proposal in the launch, on the spec sampler's tokens (else after the step, its own launch or vLLM's)
        draft_args = None
        if spec_inputs is not None and ProposalLaunch.eligible_in_step(runner, input_batch):
            step = cls.step

            def draft_args(slot: int) -> int:
                return ProposalLaunch.hosted_args(runner, input_batch, step.aux, step.ss_ring, slot, step.tensors)

        launch_outputs = cls.step.launch(
            token_ptr=runner.input_buffers.input_ids.data_ptr(),
            pos=pos,
            mpos=[pos + delta] * MROPE_SECTIONS,
            n_tok=T,
            req_idx=req_idx,
            gdnt_first=parity is None,
            gdnt_parity=cls.gdnt_parities[req_id],
            committed_ptr=committed_ptr,
            spec_inputs=spec_inputs,
            book_req=req_idx if is_booked else None,
            prev_launch=prev_launch,
            draft_args=draft_args,
        )
        ProposalLaunch.is_fused = draft_args is not None
        cls.launches += 1
        launch_log.info(
            "decode megakernel launch %d: request %s, T %d, position %d, sampler %s, draft %s",
            cls.launches,
            req_id,
            T,
            pos,
            "in launch" if spec_inputs is not None else "vLLM's",
            "in launch" if draft_args else "vLLM's",
        )

        if spec_inputs is not None:
            cls.pending_spec_outputs = launch_outputs
        else:
            cls.pending_logits = logits
        cls.is_post_of_launch = True
        # a spec launch whose inputs OP_BOOK writes skips V2's draft copies and committed count in sample_tokens: the
        # next launch reads them from the speculator's output and this ring slot
        cls.ring_slot = slot if spec_inputs is not None else None
        cls.is_own_sample = spec_inputs is not None and is_booked
        cls.stale_row = (req_idx, slot) if cls.is_own_sample else None
        runner.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=cls.step.hidden[:T],
            aux_hidden_states=[residual[:T] for residual in cls.step.aux] or None,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=None,
            routed_experts=None,
        )

    @classmethod
    def retire_pending(cls, runner: GPUModelRunner, scheduler_output: SchedulerOutput) -> None:
        """Apply the pending launched step's post_update now if its request ends, before its row is reused."""
        pending = cls.pending_launch
        if pending is not None and not pending.get("posted") and pending["req_id"] in scheduler_output.finished_req_ids:
            cls.post_update(runner, pending)
            pending["posted"] = True

    @classmethod
    def can_book(cls, runner: GPUModelRunner, state: BatchReqState) -> bool:
        """Whether the launch writes the step's inputs itself (OP_BOOK), V2's prepare_inputs writing nothing else."""
        return (
            cls.step.has_book_op
            and runner.adaptive_verification is None
            and not state.has_prefill
            and not runner.use_dcp
            and not runner.use_pp
            and runner.pcp_manager is None
            and runner.model_config.rswa_window is None
            and runner.model_state.num_new_sampled_tokens_per_step == 1
        )

    @classmethod
    def take_pending(cls, runner: GPUModelRunner, req_idx: int, is_booked: bool) -> dict | None:
        """The pending launched step's post_update and outputs that this launch applies (OP_PREV), else None."""
        if cls.pending_launch is None:
            return None

        pending, cls.pending_launch = cls.pending_launch, None
        has_post = not pending.get("posted") and is_booked and pending["req"] == req_idx
        if not (has_post or pending.get("posted")):
            cls.post_update(runner, pending)
        has_output = not pending["output"].is_copied
        if not (has_post or has_output):
            return None

        cls.seq = (cls.seq + 1) & cls.SEQ_MASK
        if has_output:
            pending["output"].seq = cls.seq

        return dict(
            slot=pending["slot"],
            T=pending["T"],
            req=pending["req"],
            out=int(has_output),
            post=int(has_post),
            seq=cls.seq,
        )

    @classmethod
    def own_inputs(
        cls, runner: GPUModelRunner, scheduler_output: SchedulerOutput, state: BatchReqState, req_idx: int
    ) -> InputBatch:
        """V2's InputBatch of the step over the buffers OP_BOOK writes, with no kernel launched."""
        T = state.num_tokens
        constants = cls.const_inputs.get((req_idx, T))
        if constants is None:
            device = runner.device
            constants = dict(
                idx_mapping=torch.tensor([req_idx], dtype=torch.int64, device=device),
                expanded_idx_mapping=torch.full((T,), req_idx, dtype=torch.int64, device=device),
                expanded_local_pos=torch.arange(T, dtype=torch.int32, device=device),
                query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=device),
                logits_indices=torch.arange(T, dtype=torch.int64, device=device),
                cu_num_logits=torch.tensor([0, T], dtype=torch.int32, device=device),
            )
            cls.const_inputs[(req_idx, T)] = constants

        input_buffers = runner.input_buffers
        num_computed = runner.req_states.num_computed_tokens_np[state.idx_mapping_np]
        bounds = np.array([0, T], dtype=np.int32)

        return InputBatch(
            req_ids=state.req_ids, num_reqs=1, num_reqs_after_padding=1, idx_mapping=constants["idx_mapping"],
            idx_mapping_np=state.idx_mapping_np, expanded_idx_mapping=constants["expanded_idx_mapping"],
            expanded_local_pos=constants["expanded_local_pos"], num_scheduled_tokens=state.num_scheduled_tokens,
            num_tokens=T, num_tokens_after_padding=T, num_draft_tokens=T - 1,
            num_draft_tokens_per_req=np.array([T - 1], dtype=np.int32), query_start_loc=constants["query_start_loc"],
            query_start_loc_np=bounds, seq_lens=input_buffers.seq_lens[:1],
            seq_lens_cpu_upper_bound=torch.from_numpy((num_computed + T).astype(np.int32)), dcp_local_seq_lens=None,
            num_computed_tokens_np=num_computed, prefill_len_np=state.prefill_len_np,
            num_computed_prefill_tokens_np=state.num_computed_prefill_tokens_np,
            is_prefilling_np=state.is_prefilling_np,
            has_prefill=False, max_seq_len_np=None, input_ids=input_buffers.input_ids[:T],
            positions=input_buffers.positions[:T], is_padding=input_buffers.is_padding[:T],
            logits_indices=constants["logits_indices"], cu_num_logits=constants["cu_num_logits"],
            cu_num_logits_np=bounds, has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=None, max_query_len=None,
        )  # fmt: skip

    @classmethod
    def sync(cls, runner: GPUModelRunner) -> None:
        """Before a vLLM step, make V2's request states current with the last launched step."""
        cls.ring_slot = None
        if cls.pending_launch is not None:  # its post_update here, its outputs copied by get_output
            pending, cls.pending_launch = cls.pending_launch, None
            if not pending.get("posted"):
                cls.post_update(runner, pending)
        if cls.stale_row is None:
            return

        # the copies the own-sample launch skipped: the drafts and the accepted count of its request row
        (req_idx, slot), cls.stale_row = cls.stale_row, None
        runner.req_states.draft_tokens[req_idx] = runner.speculator.draft_tokens[0]
        runner.model_state.num_accepted_tokens_gpu[req_idx] = cls.step.ss_ring["num_sampled"][slot].clamp(min=1)

    @classmethod
    def launched_sample_tokens(
        cls, runner: GPUModelRunner, grammar_output: GrammarOutput | None
    ) -> LaunchedOutput | None:
        """V2's sample_tokens for a launched spec step, without the drafts' copy into the request states."""
        execute_state = runner.execute_model_state
        if execute_state is None:
            return None

        input_batch, hidden_states, aux_hidden_states = (
            execute_state.input_batch,
            execute_state.hidden_states,
            execute_state.aux_hidden_states,
        )
        runner.execute_model_state = None
        sampler_output, num_sampled, num_rejected = runner.sample(hidden_states, input_batch, grammar_output)
        prompt_logprobs_dict = runner.prompt_logprobs_worker.compute_prompt_logprobs(
            runner.model.compute_logits,
            hidden_states,
            input_batch,
            runner.req_states.all_token_ids.gpu,
            runner.req_states.num_computed_tokens.gpu,
            runner.req_states.prompt_len.np,
        )
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )
        async_output = LaunchedOutput(model_runner_output, sampler_output, cls.step, cls.ring_slot)
        cls.pending_launch = dict(
            slot=cls.ring_slot,
            T=input_batch.num_tokens,
            req=int(input_batch.idx_mapping_np[0]),
            req_id=input_batch.req_ids[0],
            output=async_output,
        )

        mm_inputs = None
        if runner.speculator.supports_mm_inputs:
            mm_inputs = runner.model_state.gather_mm_embeddings(input_batch, draft_lookahead=1)
        # V2's post_update is the next launch's OP_PREV (or a vLLM step's sync); the next launch reads the committed
        # counts from the ring slot
        runner.model_state.postprocess_state(
            input_batch.idx_mapping, num_sampled, runner.req_states.num_computed_tokens.gpu
        )
        states = runner.sampler.sampling_states
        spec_hidden_states = hidden_states  # as V2: a model may override the drafter's hidden states
        if hasattr(runner.model, "get_mtp_target_hidden_states"):
            spec_hidden_states = runner.model.get_mtp_target_hidden_states()[: hidden_states.shape[0]]
        with use_workspace_lane(runner._draft_workspace_lane):
            draft_tokens = runner.speculator.propose(
                input_batch,
                execute_state.attn_metadata,
                execute_state.slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                runner.req_states.last_sampled_tokens,
                runner.req_states.next_prefill_tokens,
                states.temperature.gpu,
                states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
        runner.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)
        model_runner_output.kv_connector_output = runner.kv_connector.post_forward(execute_state.finished_req_ids)
        model_runner_output.ec_connector_output = execute_state.ec_connector_output

        return async_output

    @classmethod
    def post_update(cls, runner: GPUModelRunner, pending: dict) -> None:
        """Apply V2's post_update of a launched step's request from its ring slot."""
        ring, slot, T, device = cls.step.ss_ring, pending["slot"], pending["T"], runner.device
        req_states, sampler = runner.req_states, getattr(runner, "sampler", None)
        bin_counts = (
            sampler.penalties_state.output_bin_counts if runner.is_last_pp_rank and sampler is not None else None
        )

        post_update(
            torch.tensor([pending["req"]], dtype=torch.int32, device=device),
            req_states.num_computed_tokens.gpu,
            req_states.last_sampled_tokens,
            bin_counts,
            ring["sampled"][slot : slot + 1, :T],
            ring["num_sampled"][slot : slot + 1],
            ring["num_rejected"][slot : slot + 1],
            torch.tensor([0, T], dtype=torch.int32, device=device),
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
        )
