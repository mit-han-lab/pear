from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from pear.models.qwen3_5.deploy.blackwell.bf16.draft.proposal import DraftProposal
from pear.models.qwen3_5.deploy.blackwell.bf16.draft.schedule import ARGS_WORDS, ArgBlocks
from pear.models.qwen3_5.deploy.blackwell.bf16.draft.weights import DraftWeights
from pear.models.qwen3_5.deploy.blackwell.bf16.step import RING

if TYPE_CHECKING:  # annotations only
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class ProposalLaunch:
    """vLLM's DFlash 2 speculator patched to run a single request's verify-step proposal as one launch of the draft
    megakernel (DraftProposal), every other proposal (prefill chunks, dummy runs, batches) as vLLM's."""

    # One speculator per process (tensor parallel 1): its proposal, built once the draft's KV caches exist, and its own
    # launch's StepArgs block with the key of the tensors the block points to
    proposal: DraftProposal | None = None
    step_args: torch.Tensor | None = None
    step_args_key: tuple | None = None
    # The proposal hosted in the verify step's launch (DecodeStep with the draft attached): its DkArgs blocks per ring
    # slot of the step's sampler outputs (device memory), each slot's key and block address, and whether the last
    # launch ran it (propose then returns its drafts)
    hosted_blocks: torch.Tensor | None = None
    hosted_keys: list[tuple | None] = [None] * RING
    hosted_ptrs: list[int] = [0] * RING
    is_fused: bool = False

    @classmethod
    def install(cls) -> None:
        """Patch the worker's KV cache initialization (the proposal's build) and the speculator's propose."""
        vllm_initialize = gpu_worker.Worker.initialize_from_config
        vllm_propose = DFlash2Speculator.propose

        def initialize_from_config(self, kv_cache_config):
            vllm_initialize(self, kv_cache_config)
            cls.ensure(self.model_runner)

        def propose(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=None,
            dummy_run=False,
            skip_attn_for_dummy_run=False,
            mm_inputs=None,
            is_profile=False,
        ):
            if cls.is_fused and not dummy_run:  # the step's launch ran the proposal: its drafts are in the buffers
                cls.is_fused = False
                return self.draft_tokens[: input_batch.num_reqs]
            if cls.is_eligible(self, input_batch, aux_hidden_states, dummy_run):
                cls.launch(
                    self,
                    input_batch,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                )
                return self.draft_tokens[: input_batch.num_reqs]

            return vllm_propose(
                self,
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )

        gpu_worker.Worker.initialize_from_config = initialize_from_config
        DFlash2Speculator.propose = propose

    @classmethod
    def ensure(cls, runner: GPUModelRunner) -> DraftProposal | None:
        """The process's proposal over the runner's speculator (built once its KV caches exist), or None."""
        if cls.proposal is None and isinstance(runner.speculator, DFlash2Speculator):
            cls.proposal = cls.build(runner.speculator, runner.model)
            cls.hosted_blocks = torch.zeros(RING, ARGS_WORDS, dtype=torch.int64, device=runner.speculator.device)

        return cls.proposal

    @staticmethod
    def build(speculator: DFlash2Speculator, target_model: torch.nn.Module) -> DraftProposal:
        """The proposal over the speculator's draft model, the target's embedding and LM head, the draft's KV caches."""
        layers = get_layers_from_vllm_config(speculator.vllm_config, AttentionLayerBase)
        caches = [layers[name].kv_cache for name in speculator.model.get_draft_kv_cache_layer_names()]
        kv_cache_spec = speculator.kv_cache_config.kv_cache_groups[speculator.draft_kv_cache_group_id].kv_cache_spec
        weights = DraftWeights.from_vllm(speculator.model, target_model.get_language_model())
        proposal = DraftProposal(
            weights,
            caches,
            vocab=speculator.vocab_size,
            mask_token=speculator.parallel_drafting_token_id,
            max_model_len=speculator.max_model_len,
            kv_block=kv_cache_spec.block_size,
            window=kv_cache_spec.sliding_window,
            device=speculator.device,
            nstep=speculator.num_speculative_steps,
        )

        # the bf16 draft everywhere: the kernel reads vLLM's tensors in place, the same ones vLLM's own draft paths
        # (prefill, other shapes) run on, so nothing is written back
        dtypes = sorted(set(weights.dtypes().values()))
        logging.getLogger(__name__).warning(
            "draft megakernel: the DFlash 2 draft's %d tensors read in place, dtypes %s", len(weights.tensors()), dtypes
        )

        return proposal

    @classmethod
    def eligible_in_step(cls, runner: GPUModelRunner, input_batch: InputBatch) -> bool:
        """Whether a verify step's launch hosts the proposal (is_eligible's conditions, for the step's launch)."""
        speculator = runner.speculator

        return (
            cls.proposal is not None
            and isinstance(speculator, DFlash2Speculator)
            and input_batch.num_reqs == 1
            and input_batch.num_tokens == speculator.num_query_per_req
            and cls.proposal.nstep + 1 == speculator.num_query_per_req
        )

    @classmethod
    def hosted_args(
        cls,
        runner: GPUModelRunner,
        input_batch: InputBatch,
        aux: list[torch.Tensor],
        ring: dict[str, torch.Tensor],
        slot: int,
        step_tensors: dict[str, torch.Tensor],
    ) -> int:
        """The proposal's DkArgs block's device address in a step launch whose spec sampler writes ring slot `slot`."""
        # ring: the step's SS_RING outputs; step_tensors: the step's schedule, whose lists hold the draft's instructions
        speculator, states = runner.speculator, runner.sampler.sampling_states
        block_table = speculator.block_tables.input_block_tables[speculator.draft_kv_cache_group_id]
        # a slot's block is rewritten only when one of its tensors moved: the key is their base addresses, so that a
        # hit (the steady state) builds no views
        key = (
            input_batch.num_tokens,
            aux[0].stride(0),
            input_batch.positions.data_ptr(),
            input_batch.idx_mapping.data_ptr(),
            ring["sampled"].data_ptr(),
            ring["num_rejected"].data_ptr(),
            ring["num_sampled"].data_ptr(),
            runner.req_states.next_prefill_tokens.data_ptr(),
            states.seeds.gpu.data_ptr(),
            states.temperature.gpu.data_ptr(),
            block_table.data_ptr(),
            speculator.draft_tokens.data_ptr(),
            speculator.draft_logits.data_ptr(),
            speculator._cached_candidate_ids.data_ptr(),
            speculator._selector_scores.data_ptr(),
            step_tensors["instrs"].data_ptr(),
            *[hidden_states.data_ptr() for hidden_states in aux],
        )
        if cls.hosted_keys[slot] == key:
            return cls.hosted_ptrs[slot]

        # the anchor from the slot's sampled row (bonus_row), the rest as V2 passes them to propose
        step_args = ArgBlocks.step(
            aux=aux, n_ctx=input_batch.num_tokens, positions=input_batch.positions,
            num_rejected=ring["num_rejected"][slot : slot + 1], num_sampled=ring["num_sampled"][slot : slot + 1],
            idx_mapping=input_batch.idx_mapping, last_sampled=ring["sampled"][slot],
            next_prefill=runner.req_states.next_prefill_tokens, seeds=states.seeds.gpu,
            temperature=states.temperature.gpu, block_table=block_table, draft_tokens=speculator.draft_tokens,
            draft_logits=speculator.draft_logits, cached_cand=speculator._cached_candidate_ids,
            sel_scores=speculator._selector_scores, bonus_row=True,
        )  # fmt: skip

        # the step's instruction lists with the proposal's own descriptors, counters and epoch (those of its own launch)
        proposal = cls.proposal
        schedule = {**proposal.tensors, "instrs": step_tensors["instrs"], "sm_off": step_tensors["sm_off"]}
        cls.hosted_blocks[slot].copy_(
            ArgBlocks.args(schedule=schedule, ssq=proposal.ssq, step=step_args, tab=proposal.tables)
        )
        cls.hosted_keys[slot], cls.hosted_ptrs[slot] = key, cls.hosted_blocks[slot].data_ptr()

        return cls.hosted_ptrs[slot]

    @classmethod
    def is_eligible(
        cls,
        speculator: DFlash2Speculator,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        dummy_run: bool,
    ) -> bool:
        """Whether the proposal runs as its own launch: a verify step of one request, not a dummy run."""
        return (
            cls.proposal is not None
            and not dummy_run
            and aux_hidden_states is not None
            and input_batch.num_reqs == 1
            and input_batch.num_tokens == speculator.num_query_per_req
        )

    @classmethod
    def launch(
        cls,
        speculator: DFlash2Speculator,
        input_batch: InputBatch,
        aux: list[torch.Tensor],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        """The proposal's own launch, its StepArgs block rebuilt only when one of its tensors moved."""
        block_table = speculator.block_tables.input_block_tables[speculator.draft_kv_cache_group_id]
        tensors = (
            *aux,
            input_batch.positions,
            num_rejected,
            num_sampled,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            seeds,
            temperature,
            block_table,
            speculator.draft_tokens,
            speculator.draft_logits,
            speculator._cached_candidate_ids,
            speculator._selector_scores,
        )
        key = tuple(tensor.data_ptr() for tensor in tensors) + (aux[0].stride(0), input_batch.num_tokens)
        if key != cls.step_args_key:
            for tensor, dtype in (
                (input_batch.positions, torch.int64),
                (num_rejected, torch.int32),
                (num_sampled, torch.int32),
                (input_batch.idx_mapping, torch.int64),
                (last_sampled, torch.int64),
                (next_prefill_tokens, torch.int32),
                (seeds, torch.int64),
                (temperature, torch.float32),
            ):
                assert tensor.dtype == dtype, f"the kernel reads {dtype}, vLLM passed {tensor.dtype}"
            cls.step_args = ArgBlocks.step(
                aux=aux, n_ctx=input_batch.num_tokens, positions=input_batch.positions, num_rejected=num_rejected,
                num_sampled=num_sampled, idx_mapping=input_batch.idx_mapping, last_sampled=last_sampled,
                next_prefill=next_prefill_tokens, seeds=seeds, temperature=temperature, block_table=block_table,
                draft_tokens=speculator.draft_tokens, draft_logits=speculator.draft_logits,
                cached_cand=speculator._cached_candidate_ids, sel_scores=speculator._selector_scores,
            )  # fmt: skip
            cls.step_args_key = key

        cls.proposal.launch(cls.step_args)
