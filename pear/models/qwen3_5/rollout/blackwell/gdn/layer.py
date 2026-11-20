import copy
import dataclasses

import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)

from pear.models.qwen3_5.rollout.blackwell.gdn.core import GDNCore, GDNLayout
from pear.models.qwen3_5.rollout.blackwell.gdn.kernels import CONV_TOKENS
from pear.models.qwen3_5.rollout.blackwell.gdn.replay import GDNVerifyCore, ReplayMetadata, ReplayRecord, ReplayState
from pear.models.qwen3_5.rollout.gdn import GDNProjection, RolloutGDNAttention, RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims
from pear.models.qwen3_5.rollout.staged_buffer import StagedBuffer


class MixedStepMetadata:
    """The GDN metadata of a step with prefill rows from the host-side batch layout: numpy and one copy to the GPU."""

    def __init__(self, scheduler_config, device: torch.device):
        # a step's int32 array: the rows' initial-state flags | the prefill rows' query starts | conv program -> row |
        # conv program -> chunk of its row | the rows' query starts from the first row's; a conv program per decode row
        # and per CONV_TOKENS tokens (begun) of a prefill row
        max_num_seqs = scheduler_config.max_num_seqs
        max_programs = max_num_seqs + triton.cdiv(scheduler_config.max_num_batched_tokens, CONV_TOKENS)
        size = 3 * max_num_seqs + 2 + 2 * max_programs
        self.step_arrays = StagedBuffer(size, device)

    def build(self, builder: "BlackwellGDNMetadataBuilder", m) -> GDNAttentionMetadata | None:
        """What GDNCore.mixed_step reads, None for a step without prefill rows (vLLM builds it: the decode kernels'
        FULL-graph buffers)."""
        # no prefill row when the rows behind the decode rows hold no token (none, or a FULL graph's padding rows): vLLM
        # builds the step and fills the buffers its decode graphs replay
        D = builder.num_decodes(m)
        if not np.diff(m.query_start_loc_cpu.numpy()[: m.num_reqs + 1])[D:].any():
            return None

        return self.build_rows(builder, m, np.arange(m.num_reqs), D)

    def build_rows(self, builder: "BlackwellGDNMetadataBuilder", m, rows: np.ndarray, D: int) -> GDNAttentionMetadata:
        """What GDNCore.mixed_step reads for the given rows of the step (ascending, the first D of them decode rows), on
        their tokens packed in that order."""
        # vLLM builds the conv kernel's program map with Python loops and fresh pinned tensors, the initial-state flags
        # with small device ops
        num_reqs = m.num_reqs
        num_rows = len(rows)
        query_lens = np.diff(m.query_start_loc_cpu.numpy()[: num_reqs + 1])[rows]
        qsl = np.concatenate([[0], np.cumsum(query_lens)])
        num_prefills = num_rows - D
        chunks = -(-query_lens[D:] // CONV_TOKENS)
        num_programs = D + int(chunks.sum())

        # the sections' offsets in the step's array: initial-state flags first, at 0
        query_starts_offset = num_rows
        batch_ptr_offset = query_starts_offset + num_prefills + 1
        chunk_ptr_offset = batch_ptr_offset + num_programs
        rows_offset = chunk_ptr_offset + num_programs
        size = rows_offset + num_rows + 1
        host_array = self.step_arrays.next_stage()
        host_array[:num_rows] = m.seq_lens_cpu_upper_bound.numpy()[rows] > query_lens
        host_array[query_starts_offset:batch_ptr_offset] = qsl[D:] - qsl[D]
        host_array[batch_ptr_offset : batch_ptr_offset + D] = np.arange(D)
        host_array[batch_ptr_offset + D : chunk_ptr_offset] = np.repeat(np.arange(D, num_rows), chunks)
        host_array[chunk_ptr_offset : chunk_ptr_offset + D] = 0
        first_programs = np.repeat(np.cumsum(chunks) - chunks, chunks)  # each prefill row's first program, per program
        host_array[chunk_ptr_offset + D : rows_offset] = np.arange(num_programs - D) - first_programs
        host_array[rows_offset:size] = qsl
        self.step_arrays.upload(size)
        device_array = self.step_arrays.device_buffer[:size]

        idx = builder.row_slots(m.block_table_tensor, rows)
        has_initial_state = device_array[:num_rows]
        T = int(qsl[num_rows])

        md = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=T - D,
            num_decodes=D,
            num_decode_tokens=D,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=T,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=m.query_start_loc if num_rows == num_reqs else device_array[rows_offset:size],
            non_spec_state_indices_tensor=idx,
            prefill_query_start_loc=device_array[query_starts_offset:batch_ptr_offset],
            prefill_state_indices=idx[D:],
            prefill_has_initial_state=has_initial_state[D:],
            nums_dict={CONV_TOKENS: {"tot": num_programs}},
            batch_ptr=device_array[batch_ptr_offset:chunk_ptr_offset],
            token_chunk_offset_ptr=device_array[chunk_ptr_offset:rows_offset],
        )
        md._pear_layout = GDNLayout(md)

        return md


class BlackwellGDNAttention(RolloutGDNAttention):
    """vLLM's Qwen Gated DeltaNet layer on one projection GEMM and GDNCore (registered out of tree in its place); with
    a drafter its verify rows on GDNVerifyCore under replaySSM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GDNProjection.merge(self)

        # replaySSM: each request's last verify block, for the next step to replay its committed tokens, and the
        # layer's mamba KV cache group among ReplayState's, set with the KV caches
        self.replay_record = None
        if self.num_spec > 0:
            self.replay_record = ReplayRecord.allocate(
                self.vllm_config.scheduler_config.max_num_seqs,
                GDNDims.of_layer(self),
                self.num_spec + 1,
                self.A_log.device,
            )
        self.replay_group = None

    def get_kv_cache_spec(self, vllm_config):
        # one state slot per request, from which a verify step replays the committed tokens of the request's last block,
        # in place of vLLM's slot per draft token (with a drafter; without one there are none)
        return dataclasses.replace(super().get_kv_cache_spec(vllm_config), num_speculative_blocks=0)

    @property
    def kv_cache(self):
        """The state pools vLLM bound to the layer (MambaBase.bind_kv_cache)."""
        try:
            return self.__dict__["state_pools"]
        except KeyError:
            raise AttributeError("kv_cache") from None  # none bound yet: vLLM's hasattr sees no cache

    @kv_cache.setter
    def kv_cache(self, kv_cache):
        # GDNCore's constants hold the pools and so their memory: they go when vLLM binds another cache or releases this
        # one, as it releases its CUDA graph memory profiling's cache before it allocates the real one
        self.__dict__["state_pools"] = kv_cache
        self.__dict__.pop("_pear_layer_constants", None)

    def forward_cuda(self, hidden_states):
        # vLLM's fused path zero-fills the core output of every layer; GDNCore writes every row it computes (a
        # piecewise-graph batch's padding rows are never read back), so it starts uninitialized
        if not GDNProjection.eligible(self, hidden_states):
            return super().forward_cuda(hidden_states)

        out = torch.ops.pear_rollout_blackwell.lt_mm(hidden_states, self.qkvzba_weight)
        mixed_qkvz, ba = out[:, : self.qkvz_out], out[:, self.qkvz_out :]
        core_attn_out = torch.empty(
            (hidden_states.size(0), self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        return self.fused_core(mixed_qkvz, ba, core_attn_out)

    def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
        # no metadata of the layer (a profiling run leaves the GDN groups out): vLLM's warm-up of its prefill kernels
        attn_metadata = get_forward_context().attn_metadata
        md = attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        if isinstance(md, ReplayMetadata):
            return self.replay_core(mixed_qkvz, ba, core_attn_out, md)
        if md is not None:
            layout = md.__dict__.get("_pear_layout")
            if layout is not None:
                return GDNCore.mixed_step(self, mixed_qkvz, ba, core_attn_out, layout)
            if (
                md.num_prefills == 0
                and md.num_decodes > 0
                and md.spec_sequence_masks is None
                and md.non_spec_state_indices_tensor.stride(0) == 1
            ):
                return GDNCore.decode_step(self, mixed_qkvz, ba, core_attn_out, md)

        return super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)

    def replay_core(self, mixed_qkvz, ba, core_attn_out, md: ReplayMetadata) -> None:
        """A speculative replica's step: its verify rows on GDNVerifyCore, then its prefill rows on GDNCore."""
        if md.num_rows > 0:
            GDNVerifyCore.run(self, mixed_qkvz, ba, core_attn_out, ReplayState.instance.step(), md.num_rows)
        if md.prefill is None:
            return

        tokens, layout = md.prefill_tokens, md.prefill._pear_layout
        if isinstance(tokens, slice):
            GDNCore.mixed_step(self, mixed_qkvz[tokens], ba[tokens], core_attn_out[tokens], layout)
            return
        # verify rows behind prefill rows: the prefill rows' tokens gathered, their outputs scattered back
        out = core_attn_out.new_empty((tokens.numel(), *core_attn_out.shape[1:]))
        GDNCore.mixed_step(self, mixed_qkvz.index_select(0, tokens), ba.index_select(0, tokens), out, layout)
        core_attn_out.index_copy_(0, tokens, out)


class BlackwellGDNMetadataBuilder(RolloutGDNMetadataBuilder):
    """vLLM's GDN metadata builder with a mixed step's metadata from MixedStepMetadata, shared by the GDN layers'
    groups; the other steps' as vLLM builds them (the decode steps' FULL-graph buffers are per builder). With a drafter
    every step's ReplayMetadata, also shared by the groups: each layer reads its group's slots in ReplayState."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.mixed_step_metadata = MixedStepMetadata(vllm_config.scheduler_config, device)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        # with a drafter vLLM's V2 runner hands every GDN build its draft counts (a FULL graph's capture comes through
        # build_for_cudagraph_capture); vLLM's own capture path hands them to a replica without one too
        if self.use_spec_decode:
            return self.replay_metadata(common_attn_metadata, num_decode_draft_tokens_cpu)
        md = self.mixed_step_metadata.build(self, common_attn_metadata)
        if md is not None:
            return md

        return super().build(
            common_prefix_len, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build
        )

    def replay_metadata(self, m, num_decode_draft_tokens_cpu) -> ReplayMetadata:
        """A speculative step's metadata: its verify rows (those with a draft count), its prefill rows (the others with
        tokens)."""
        num_reqs = m.num_reqs
        query_starts = m.query_start_loc_cpu.numpy()[: num_reqs + 1]
        is_prefill = (num_decode_draft_tokens_cpu.numpy()[:num_reqs] < 0) & (np.diff(query_starts) > 0)
        if not is_prefill.any():
            return ReplayMetadata(num_rows=num_reqs)

        rows = np.flatnonzero(is_prefill)
        prefill = self.mixed_step_metadata.build_rows(self, m, rows, 0)
        first = int(rows[0])
        if is_prefill[first:].all():
            # the step's last rows, as vLLM's runner sorts the rows with drafts first
            tokens = slice(int(query_starts[first]), int(query_starts[num_reqs]))
            return ReplayMetadata(num_rows=first, prefill=prefill, prefill_rows=rows, prefill_tokens=tokens)

        # vLLM's runner sorts a prompt chunk of k + 1 tokens before a decode row without drafts: the verify core takes
        # every row, GDNCore the prefill rows' tokens gathered
        tokens = np.concatenate([np.arange(query_starts[row], query_starts[row + 1]) for row in rows])
        return ReplayMetadata(
            num_rows=num_reqs,
            prefill=prefill,
            prefill_rows=rows,
            prefill_tokens=torch.from_numpy(tokens).to(m.query_start_loc.device),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        if not self.use_spec_decode:
            return super().build_for_cudagraph_capture(common_attn_metadata)

        # a FULL graph's verify step: the graph reads the rows the runner's preprocess writes before each replay (a
        # capture finds none: every step's commit ends its rows)
        return ReplayMetadata(num_rows=common_attn_metadata.num_reqs)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        if isinstance(metadata, ReplayMetadata):
            if metadata.prefill is None:
                return metadata
            prefill = self.group_metadata(metadata.prefill, blk_table, metadata.prefill_rows)
            return dataclasses.replace(metadata, prefill=prefill)

        if "_pear_layout" not in metadata.__dict__:
            return super().update_block_table(metadata, blk_table, slot_mapping)

        return self.group_metadata(metadata, blk_table, np.arange(metadata.num_decodes + metadata.num_prefills))

    def group_metadata(self, metadata, blk_table, rows: np.ndarray):
        """A mixed step's metadata and layout of the given rows with this group's state slots."""
        idx = self.row_slots(blk_table, rows)
        layout = metadata._pear_layout
        group_metadata = copy.copy(metadata)
        group_metadata.non_spec_state_indices_tensor = idx
        group_metadata.prefill_state_indices = idx[metadata.num_decodes :]
        group_layout = copy.copy(layout)
        group_layout.idx = idx
        group_layout.pf_idx = idx[layout.D :]
        group_metadata._pear_layout = group_layout

        return group_metadata

    def row_slots(self, block_table: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        """The given rows' state slots in this group (rows ascending): a slice for a run of rows, else gathered."""
        slots = self.state_slots(block_table, int(rows[-1]) + 1)
        if rows[-1] - rows[0] + 1 == len(rows):
            return slots[rows[0] :].contiguous()

        return slots[torch.from_numpy(rows).to(slots.device)].contiguous()


class BlackwellGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with BlackwellGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return BlackwellGDNMetadataBuilder
