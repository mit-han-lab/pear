import dataclasses
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)

from pear.models.qwen3_5.rollout.gdn import GDNProjection, RolloutGDNAttention, RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims
from pear.models.qwen3_5.rollout.staged_buffer import StagedBuffer

CHUNK = 64  # FLA's chunk size (csrc/gdn_chunk.cu BT)


@triton.jit
def _gated_norm_kernel(
    o_ptr,  # [R, D] rows (bf16)
    z_ptr,  # rows of z, row r at z_ptr + (r // HV) * stride_z_tok + (r % HV) * D
    w_ptr,
    out_ptr,  # [R, D]
    counts_ptr,
    stride_z_tok,
    eps,
    HV: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    R = tl.load(counts_ptr) * HV
    if tl.program_id(0) * ROWS >= R:
        return

    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    m = rows < R
    cols = tl.arange(0, D)
    x = tl.load(o_ptr + rows[:, None] * D + cols[None, :], mask=m[:, None], other=0).to(tl.float32)
    z = tl.load(
        z_ptr + (rows // HV)[:, None] * stride_z_tok + (rows % HV)[:, None] * D + cols[None, :],
        mask=m[:, None],
        other=0,
    ).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    y = y * (z * tl.sigmoid(z))
    tl.store(out_ptr + rows[:, None] * D + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=m[:, None])


class GDNLayout(StagedBuffer):
    """A mixed step's batch layout, shared by every Gated DeltaNet layer of the step, in one int32 device buffer."""

    # decode rows [0, D) first, then the prefill sequences, in absolute token coordinates; one layout per process at
    # the engine's capacities (fixed addresses, which the FULL mixed-step graphs read), filled per step: the kernels
    # launch over the capacities and loop over the step's counts
    COUNT_SLOTS = 8  # int32 slots of the counts section
    instance = None  # the process's layout

    def __init__(self, Tc: int, Nc: int, Pc: int, H: int, HV: int, K: int, V: int, device):
        # Pc sequences over Tc tokens fill Tc / CHUNK chunks and a partial one each
        self.Tc, self.Nc, self.Pc, self.NTc = Tc, Nc, Pc, triton.cdiv(Tc, CHUNK) + Pc

        # counts [T, D, N, P, NT, 0, 0, 0] | tok_seq | tok_pos | seq_start | pf_cu | chunk_offsets | chunk_indices |
        # seq_init
        sizes = [self.COUNT_SLOTS, Tc, Tc, Nc + 1, Pc + 1, Pc + 1, 2 * self.NTc, Nc]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        super().__init__(int(self.offsets[-1]), device)

        (
            self.counts,
            self.tok_seq,
            self.tok_pos,
            self.seq_start,
            self.pf_cu,
            self.chunk_offsets,
            chunk_indices,
            self.seq_init,
        ) = torch.split(self.device_buffer, sizes)
        self.chunk_indices = chunk_indices.view(self.NTc, 2)
        self.scratch = GDNScratch(Tc, self.NTc, H, HV, K, V, device)  # the layers' scratch at the capacities
        self.T = self.D = self.N = self.P = 0

    @classmethod
    def of_step(cls, vllm_config, m, num_decodes: int) -> "GDNLayout":
        """A mixed step's layout from vLLM's common metadata and its leading decode rows, in the process's layout."""
        qsl_all = m.query_start_loc_cpu[: m.num_reqs + 1].numpy().astype(np.int64)
        query_lens = np.diff(qsl_all)
        n = int(np.count_nonzero(query_lens))  # CUDA graph padding requests are empty and last
        qsl = qsl_all[: n + 1]
        layout = cls.of_engine(vllm_config, m.query_start_loc.device)

        seq_lens = m.seq_lens_cpu_upper_bound[:n].numpy()
        layout.fill(qsl, num_decodes, (seq_lens - query_lens[:n] > 0).astype(np.int32))

        return layout

    @classmethod
    def of_rows(cls, vllm_config, m, rows: np.ndarray) -> "GDNLayout":
        """The layout of a step's given prefill rows (ascending), on their tokens packed in that order."""
        query_lens = np.diff(m.query_start_loc_cpu.numpy()[: m.num_reqs + 1].astype(np.int64))[rows]
        layout = cls.of_engine(vllm_config, m.query_start_loc.device)
        seq_lens = m.seq_lens_cpu_upper_bound.numpy()[rows]
        layout.fill(np.concatenate([[0], np.cumsum(query_lens)]), 0, (seq_lens - query_lens > 0).astype(np.int32))

        return layout

    @classmethod
    def of_engine(cls, vllm_config, device) -> "GDNLayout":
        """The process's layout at the engine's capacities, made outside any capture with its scratch."""
        if cls.instance is None:
            text = vllm_config.model_config.hf_text_config
            tp = vllm_config.parallel_config.tensor_parallel_size
            scheduler = vllm_config.scheduler_config
            # any step within the token budget, its requests possibly all prefills
            cls.instance = cls(
                scheduler.max_num_batched_tokens,
                scheduler.max_num_seqs,
                scheduler.max_num_seqs,
                text.linear_num_key_heads // tp,
                text.linear_num_value_heads // tp,
                text.linear_key_head_dim,
                text.linear_value_head_dim,
                device,
            )

        return cls.instance

    def fill(self, qsl: np.ndarray, D: int, seq_init: np.ndarray) -> None:
        """Write a batch's layout (qsl: int64 query start locations of its real requests) and its sequences'
        initial-state flags."""
        T, n, P, NT = self.needs(qsl, D)
        assert T <= self.Tc and n <= self.Nc and P <= self.Pc and NT <= self.NTc

        o = self.offsets
        stage = self.next_stage()
        lens = np.diff(qsl)

        stage[o[0] : o[1]] = 0
        counts = (T, D, n, P, NT)
        stage[o[0] : o[0] + len(counts)] = counts
        stage[o[1] : o[1] + T] = np.repeat(np.arange(n, dtype=np.int32), lens)
        stage[o[2] : o[2] + T] = np.arange(T, dtype=np.int64) - np.repeat(qsl[:-1], lens)
        stage[o[3] : o[3] + n + 1] = qsl
        pf_cu = qsl[D:]
        stage[o[4] : o[4] + P + 1] = pf_cu
        nchunks = (np.diff(pf_cu) + CHUNK - 1) // CHUNK
        stage[o[5]] = 0
        stage[o[5] + 1 : o[5] + P + 1] = np.cumsum(nchunks)
        chunk_indices = stage[o[6] : o[7]].reshape(self.NTc, 2)
        if NT:
            chunk_indices[:NT, 0] = np.repeat(np.arange(P, dtype=np.int32), nchunks)
            chunk_indices[:NT, 1] = np.arange(NT) - np.repeat(np.cumsum(nchunks) - nchunks, nchunks)
        stage[o[7] : o[7] + n] = seq_init

        self.upload(self.device_buffer.numel())
        self.T, self.D, self.N, self.P = T, D, n, P

    @staticmethod
    def needs(qsl: np.ndarray, D: int) -> tuple[int, int, int, int]:
        """(T, N, P, NT) of a batch: qsl over its real requests (all non-empty), D leading decode rows."""
        n = len(qsl) - 1
        pf_lens = np.diff(qsl[D:])
        return int(qsl[-1]), n, n - D, int(((pf_lens + CHUNK - 1) // CHUNK).sum())


class GDNScratch:
    """Scratch shared by every Gated DeltaNet layer of a step (they run one after another on one stream)."""

    def __init__(self, T: int, NT: int, H: int, HV: int, K: int, V: int, device):
        bf, f32 = torch.bfloat16, torch.float32
        self.q = torch.empty((T, H, K), dtype=bf, device=device)
        self.k = torch.empty((T, H, K), dtype=bf, device=device)
        self.v = torch.empty((T, HV, V), dtype=bf, device=device)
        self.g = torch.empty((T, HV), dtype=f32, device=device)
        self.beta = torch.empty((T, HV), dtype=f32, device=device)
        self.o = torch.empty((T, HV, V), dtype=bf, device=device)
        self.g_cum = torch.empty((T, HV), dtype=f32, device=device)
        self.w = torch.empty((T, HV, K), dtype=bf, device=device)
        self.u = torch.empty((T, HV, V), dtype=bf, device=device)
        self.am = torch.empty((NT, HV, CHUNK, CHUNK), dtype=bf, device=device)  # each chunk's masked, decayed q k^T

        self.sms = torch.cuda.get_device_properties(device).multi_processor_count  # the chunk kernels' waves
        # the prefill rows' chunk kernels run on this stream beside the decode rows' recurrence
        self.side_stream = torch.cuda.Stream(device=device, priority=-1)


class MixedStepGDNCore:
    """A mixed step's Gated DeltaNet core (decode rows + prefill chunks) in five kernel launches per layer."""

    # vLLM's mixed path runs many aten ops and launches per layer behind Python wrappers; this one computes the same
    # math from the step's layout: conv + l2norm + gates, the decode recurrence, the conv states, the chunked delta
    # rule, the gated norm
    DECODE_V_ROWS = 32  # value rows per program of the decode recurrence and of the chunked state recurrence
    PREP_CTAS_PER_SM = 2  # gdn_chunk_prep's resident CTAs per SM: each takes half an SM's shared memory
    STATE_CTAS_PER_SM = 1  # gdn_chunk_state's: each takes most of it
    NORM_ROWS = 16  # head rows per program of the gated norm
    NORM_WARPS = 4  # warps per program of the gated norm

    @staticmethod
    def mixed_step(layer, layout: GDNLayout, seq_slot, seq_init, mixed_qkvz, ba, core_attn_out) -> None:
        """core_attn_out[:T] = gated_norm(delta rule(conv(qkv)), z), the conv and ssm state slots updated."""
        # the kernels launch over the layout's capacities for the step's rows (a FULL graph's padded ones) and read the
        # counts
        dims = GDNDims.of_layer(layer)
        HV, K, V, C = dims.HV, dims.K, dims.V, dims.C
        conv_state, ssm_state = layer.kv_cache[0], layer.kv_cache[1]  # [slots, 3, C], [slots, HV, V, K]
        scratch = layout.scratch

        T = mixed_qkvz.shape[0]  # the step's token count, a graph's padded one
        # every request takes a token: no more sequences of either kind than tokens
        N = D = min(T, layout.Nc)
        P = min(T, layout.Pc)
        NT = min(triton.cdiv(T, CHUNK) + P, layout.NTc)
        counts = layout.counts

        torch.ops.pear_rollout_hopper.gdn_conv_prep(
            mixed_qkvz,
            ba,
            layer.conv1d.weight,
            conv_state,
            layer.A_log,
            layer.dt_bias,
            layout.tok_seq,
            layout.tok_pos,
            seq_slot,
            seq_init,
            scratch.q,
            scratch.k,
            scratch.v,
            scratch.g,
            scratch.beta,
            counts,
            T,
        )
        scale = K**-0.5

        # the prefill chunks run on a high-priority side stream beside the decode rows' recurrence (other sequences,
        # other state slots), a CTA per decode row, value head and value block (short ones that leave the chunk kernels
        # the SMs), whose first CTAs write the conv states (read by the next step only)
        main_stream = torch.cuda.current_stream()
        scratch.side_stream.wait_stream(main_stream)
        torch.ops.pear_rollout_hopper.gdn_mixed_decode(
            scratch.q,
            scratch.k,
            scratch.v,
            scratch.g,
            scratch.beta,
            scratch.o,
            ssm_state,
            seq_slot,
            counts,
            scale,
            N + D * HV * (V // MixedStepGDNCore.DECODE_V_ROWS),
            mixed_qkvz,
            conv_state,
            layout.seq_start,
            seq_slot,
            seq_init,
            C,
        )
        with torch.cuda.stream(scratch.side_stream):
            MixedStepGDNCore.prefill(layout, scratch, ssm_state, seq_slot, seq_init, NT, P, HV, V, scale)
        main_stream.wait_stream(scratch.side_stream)

        _gated_norm_kernel[(triton.cdiv(T * HV, MixedStepGDNCore.NORM_ROWS),)](
            scratch.o,
            mixed_qkvz[:, C:],
            layer.norm.weight,
            core_attn_out,
            counts,
            mixed_qkvz.stride(0),
            layer.norm.eps,
            HV=HV,
            D=V,
            ROWS=MixedStepGDNCore.NORM_ROWS,
            num_warps=MixedStepGDNCore.NORM_WARPS,
        )

    @staticmethod
    def prefill(layout, scratch, ssm_state, seq_slot, seq_init, NT, P, HV, V, scale) -> None:
        """The prefill rows' chunked gated delta rule into scratch.o, the state slots updated."""
        # csrc/gdn_chunk.cu: per chunk (cumsum, KKT, the triangular inverse, w and u, the output's masked q k^T), then
        # per sequence (the state recurrence in place in the state slots, and the output); each runs one wave of CTAs
        # that loop over the step's real chunks and sequences, read from the layout's counts
        ops = torch.ops.pear_rollout_hopper
        cu, chunk_indices, chunk_offsets, counts = (
            layout.pf_cu,
            layout.chunk_indices,
            layout.chunk_offsets,
            layout.counts,
        )

        ops.gdn_chunk_prep(
            scratch.q,
            scratch.k,
            scratch.v,
            scratch.g,
            scratch.beta,
            scratch.g_cum,
            scratch.w,
            scratch.u,
            scratch.am,
            cu,
            chunk_indices,
            chunk_offsets,
            counts,
            min(NT * HV, scratch.sms * MixedStepGDNCore.PREP_CTAS_PER_SM),
        )

        ops.gdn_chunk_state(
            scratch.q,
            scratch.k,
            scratch.u,
            scratch.w,
            scratch.am,
            scratch.g_cum,
            scratch.o,
            ssm_state,
            seq_slot,
            seq_init,
            cu,
            chunk_indices,
            chunk_offsets,
            counts,
            scale,
            min(P * HV * (V // MixedStepGDNCore.DECODE_V_ROWS), scratch.sms * MixedStepGDNCore.STATE_CTAS_PER_SM),
        )


class FusedGDNDecode:
    """The Gated DeltaNet core of a pure decode step as one fused kernel (csrc/gdn_decode.cu)."""

    WORKSPACE_INTS = 1024 * 16 + 16  # the decode kernel's per-(sequence, key head) counters, zero between launches
    workspaces: dict = {}  # device index -> the decode kernel's counters

    @classmethod
    def decode_step(cls, layer, md, mixed_qkvz, ba, core_attn_out) -> None:
        """The fused decode kernel on the projection [qkvz | ba] (ba's columns follow qkvz's in the same rows); the
        padding rows keep their zeros."""
        n = mixed_qkvz.shape[1]
        if ba.data_ptr() != mixed_qkvz.data_ptr() + n * mixed_qkvz.element_size() or ba.stride() != mixed_qkvz.stride():
            raise RuntimeError(f"{layer.prefix}: ba must follow mixed_qkvz in the projection's rows")

        proj = mixed_qkvz.as_strided((mixed_qkvz.shape[0], n + ba.shape[1]), mixed_qkvz.stride())
        device = proj.device
        if device.index not in cls.workspaces:
            cls.workspaces[device.index] = torch.zeros(cls.WORKSPACE_INTS, dtype=torch.int32, device=device)

        torch.ops.pear_rollout_hopper.gdn_decode(
            proj,
            layer.kv_cache[0],
            layer.conv1d.weight.view(layer.conv1d.weight.size(0), -1),
            layer.kv_cache[1],
            md.non_spec_state_indices_tensor[: md.num_actual_tokens],
            layer.A_log,
            layer.dt_bias,
            layer.norm.weight,
            core_attn_out,
            cls.workspaces[device.index],
            layer.num_k_heads // layer.tp_size,
            layer.norm.eps,
            layer.head_k_dim**-0.5,
        )


class ReplayRecord:
    """The layout of a Gated DeltaNet layer's replay records: per request two copies (a verify step reads one and writes
    the other) of its last verify block, the conv inputs from the window before it on, then its b / a rows."""

    TAPS = 3  # the conv window before the block (kTaps of csrc/gdn_verify.cu)
    COPIES = 2
    ALIGNMENT = 256  # a copy's size is a multiple of it (bytes)

    @classmethod
    def ba_offset(cls, dims: GDNDims, max_tokens: int) -> int:
        """The b / a rows' offset in a copy (bytes): after the window's and the block's bf16 conv inputs."""
        return (cls.TAPS + max_tokens) * dims.C * torch.bfloat16.itemsize

    @classmethod
    def allocate(cls, num_requests: int, dims: GDNDims, max_tokens: int, device) -> torch.Tensor:
        """Zeroed records of num_requests requests for blocks of up to max_tokens tokens."""
        copy_bytes = cls.ba_offset(dims, max_tokens) + max_tokens * 2 * dims.HV * torch.bfloat16.itemsize
        copy_bytes = triton.cdiv(copy_bytes, cls.ALIGNMENT) * cls.ALIGNMENT

        return torch.zeros(num_requests, cls.COPIES, copy_bytes, dtype=torch.uint8, device=device)


class ReplayStep(NamedTuple):
    """A step's replaySSM tables: its rows and their slots in every GDN KV cache group."""

    # int32 [rows, 8]: first token, verify tokens (0: not a verify row), request, tokens computed, tokens replayed, the
    # record copy read, 2 unused
    rows: torch.Tensor
    slots: torch.Tensor  # int32 [groups, rows, 4]: source, destination, the block of a boundary inside it (-1: none)
    max_tokens: int  # tokens of a verify block at most: the records' size
    block_size: int  # the mamba block size (tokens): the prefix cache's state boundaries


class GDNVerifyCore:
    """A Gated DeltaNet layer's verify rows under replaySSM (csrc/gdn_verify.cu): from each request's slot, the replay
    of its previous block's committed tokens, this block's outputs with the gated norm, the state written once."""

    @staticmethod
    def run(layer, mixed_qkvz, ba, core_attn_out, step: ReplayStep, num_rows: int) -> None:
        """The core of the step's first num_rows rows (a row of no verify tokens is skipped) in the layer's group."""
        dims = GDNDims.of_layer(layer)
        torch.ops.pear_rollout_hopper.gdn_verify(
            mixed_qkvz,
            ba,
            core_attn_out,
            layer.conv1d.weight.view(dims.C, -1),
            layer.A_log,
            layer.dt_bias,
            layer.norm.weight,
            layer.kv_cache[1],
            layer.kv_cache[0],
            layer.replay_record,
            step.rows,
            step.slots,
            layer.replay_group,
            num_rows,
            dims.H,
            step.block_size,
            step.max_tokens,
            ReplayRecord.ba_offset(dims, step.max_tokens),
            layer.head_k_dim**-0.5,
            layer.norm.eps,
            layer.norm.activation == "sigmoid",
            layer.conv_state_dim_first,
        )


class ReplayState:
    """The replaySSM bookkeeping of a speculative replica (one per process, made by its model state with the KV caches):
    each request's replay count and record copy, the step's rows and their slots in every GDN KV cache group and the
    layers' table of the flush after sampling."""

    # a step's row and a row's slots in a group (ReplayStep; kRowInts, kSlotInts of csrc/gdn_verify.cu)
    ROW_INTS = 8
    SLOT_INTS = 4
    instance: Optional["ReplayState"] = None

    def __init__(self, groups: list[list], max_num_reqs: int, max_tokens: int, block_size: int, device):
        layers = [layer for group_layers in groups for layer in group_layers]
        self.dims = GDNDims.of_layer(layers[0])
        self.max_tokens, self.block_size = max_tokens, block_size
        self.scale = layers[0].head_k_dim ** -0.5
        self.replay = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.parity = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.rows = torch.zeros(max_num_reqs, self.ROW_INTS, dtype=torch.int32, device=device)
        self.slots = torch.full((len(groups), max_num_reqs, self.SLOT_INTS), -1, dtype=torch.int32, device=device)
        # a mixed step's verify rows, uploaded from pinned buffers
        self.verify_rows = StagedBuffer(max_num_reqs, device, torch.uint8)

        # each layer's group, the order of the slots' groups; the flush's per-layer descriptors
        group_ids = []
        for group_id, group_layers in enumerate(groups):
            for layer in group_layers:
                layer.replay_group = group_id
                group_ids.append(group_id)
        self.layer_table = torch.ops.pear_rollout_hopper.gdn_layer_table(
            [layer.kv_cache[1] for layer in layers],
            [layer.kv_cache[0] for layer in layers],
            [layer.replay_record for layer in layers],
            [layer.conv1d.weight.view(self.dims.C, -1) for layer in layers],
            [layer.A_log for layer in layers],
            [layer.dt_bias for layer in layers],
            [layer.norm.weight for layer in layers],
            group_ids,
            layers[0].conv_state_dim_first,
        )

    def preprocess(self, model_state, input_batch, num_computed_tokens: torch.Tensor, is_verify) -> None:
        """The step's rows and slots before the forward (vLLM's align bookkeeping with them), and vLLM's pre-copy of the
        other rows' states when there are any."""
        ctx = model_state._mamba_ctx
        verify_rows = None
        if is_verify is not None:
            num_rows = is_verify.shape[0]
            self.verify_rows.next_stage()[:num_rows] = is_verify
            self.verify_rows.upload(num_rows)
            verify_rows = self.verify_rows.device_buffer[:num_rows]

        torch.ops.pear_rollout_hopper.gdn_replay_meta(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            num_computed_tokens,
            verify_rows,
            self.replay,
            self.parity,
            model_state._mamba_state_idx_gpu,
            model_state._mamba_src_col_gpu,
            model_state._mamba_src_off_gpu,
            model_state.num_accepted_tokens_gpu,
            ctx.aligned_index_block_table_ptrs,
            ctx.block_table_stride_req,
            input_batch.num_reqs,
            input_batch.num_reqs_after_padding,
            self.block_size,
            self.rows,
            self.slots,
        )
        if is_verify is not None:
            ctx.run_fused_precopy(
                input_batch.num_reqs,
                model_state._mamba_state_idx_gpu,
                model_state._mamba_src_col_gpu,
                model_state._mamba_src_off_gpu,
                input_batch.idx_mapping,
            )

    def step(self) -> ReplayStep:
        """The tables of the step being run."""
        return ReplayStep(self.rows, self.slots, self.max_tokens, self.block_size)

    def postprocess(self, model_state, idx_mapping: torch.Tensor, num_sampled: torch.Tensor) -> None:
        """After sampling: the boundary states the step's committed tokens reached, the requests' replay counts and
        record copies, vLLM's accepted counts."""
        torch.ops.pear_rollout_hopper.gdn_commit(
            self.layer_table,
            self.rows,
            self.slots,
            self.replay,
            self.parity,
            model_state.num_accepted_tokens_gpu,
            idx_mapping,
            num_sampled.to(torch.int32),
            self.dims.H,
            self.dims.HV,
            self.block_size,
            self.max_tokens,
            ReplayRecord.ba_offset(self.dims, self.max_tokens),
            self.scale,
        )


@dataclasses.dataclass
class ReplayMetadata:
    """The GDN metadata of a speculative replica's step: its verify rows under replaySSM (rows of ReplayState's tables
    below num_rows; the core skips the others) and, in a mixed step, its prefill rows on MixedStepGDNCore's layout."""

    num_rows: int
    prefill: GDNAttentionMetadata | None = None
    prefill_rows: np.ndarray | None = None  # the step's prefill rows, ascending
    # their tokens: the step's last ones, or gathered when verify rows follow them
    prefill_tokens: slice | torch.Tensor | None = None


class HopperGDNAttention(RolloutGDNAttention):
    """vLLM's Qwen Gated DeltaNet layer on one projection GEMM, MixedStepGDNCore in a mixed step and the fused decode
    kernel (registered out of tree in its place); with a drafter its verify rows on GDNVerifyCore under replaySSM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GDNProjection.merge(self)
        self.conv_state_dim_first = is_conv_state_dim_first()  # the conv state's layout, fixed for the process

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

    def forward_cuda(self, hidden_states):
        if not GDNProjection.eligible(self, hidden_states):
            return super().forward_cuda(hidden_states)

        out = F.linear(hidden_states, self.qkvzba_weight)
        mixed_qkvz, ba = out[:, : self.qkvz_out], out[:, self.qkvz_out :]
        core_attn_out = torch.zeros(
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
            self.replay_core(mixed_qkvz, ba, core_attn_out, md)
            return
        layout = getattr(md, "_pear_layout", None)

        if layout is not None:
            MixedStepGDNCore.mixed_step(
                self, layout, md.non_spec_state_indices_tensor, md.has_initial_state, mixed_qkvz, ba, core_attn_out
            )
            return
        # a pure decode step without speculative tokens
        if (
            md is not None
            and md.num_spec_decodes == 0
            and md.spec_sequence_masks is None
            and md.num_prefills == 0
            and md.num_decodes > 0
            and not self.conv_state_dim_first
        ):
            FusedGDNDecode.decode_step(self, md, mixed_qkvz, ba, core_attn_out)
            return

        super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)

    def replay_core(self, mixed_qkvz, ba, core_attn_out, md: ReplayMetadata) -> None:
        """A speculative replica's step: its verify rows on GDNVerifyCore, then its prefill rows on MixedStepGDNCore."""
        if md.num_rows > 0:
            GDNVerifyCore.run(self, mixed_qkvz, ba, core_attn_out, ReplayState.instance.step(), md.num_rows)
        if md.prefill is None:
            return

        tokens, prefill = md.prefill_tokens, md.prefill
        rows_args = (prefill._pear_layout, prefill.non_spec_state_indices_tensor, prefill.has_initial_state)
        if isinstance(tokens, slice):
            MixedStepGDNCore.mixed_step(self, *rows_args, mixed_qkvz[tokens], ba[tokens], core_attn_out[tokens])
            return
        # verify rows behind prefill rows: the prefill rows' tokens gathered, their outputs scattered back
        out = core_attn_out.new_zeros((tokens.numel(), *core_attn_out.shape[1:]))
        MixedStepGDNCore.mixed_step(
            self, *rows_args, mixed_qkvz.index_select(0, tokens), ba.index_select(0, tokens), out
        )
        core_attn_out.index_copy_(0, tokens, out)


class HopperGDNMetadataBuilder(RolloutGDNMetadataBuilder):
    """vLLM's GDN metadata builder with a mixed step's layout for MixedStepGDNCore, which runs any mixed step of a
    graph's size in a FULL graph; the other GDN groups' metadata derives from the first group's (update_block_table)."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # this group's state slots in a mixed step, at the address its FULL mixed-step graphs read
        self.mixed_step_slots = torch.zeros(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        # any step on the mixed-step layout in a FULL graph; with a drafter vLLM's FULL graphs of uniform batches alone:
        # a mixed step's ReplayMetadata gives its verify rows' count and prefill tokens on the host
        if vllm_config.speculative_config is not None:
            return super().get_cudagraph_support(vllm_config, kv_cache_spec)
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        # with a drafter every step's ReplayMetadata (vLLM's V2 runner hands every GDN build the draft-token arguments);
        # without one a step with prefill rows on the mixed-step layout, whichever graph replays it, and a pure decode
        # step as vLLM builds it
        m = common_attn_metadata
        if self.use_spec_decode:
            return self.replay_metadata(m, num_decode_draft_tokens_cpu)
        num_decodes = self.num_decodes(m)
        if num_decode_draft_tokens_cpu is None and num_decodes < m.num_reqs:
            layout = GDNLayout.of_step(self.vllm_config, m, num_decodes)
            return self.mixed_step_metadata(m.block_table_tensor, m.num_actual_tokens, layout)

        return super().build(common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)

    def replay_metadata(self, m, num_decode_draft_tokens_cpu) -> ReplayMetadata:
        """A speculative step's metadata: its verify rows (those with a draft count), its prefill rows (the others with
        tokens) on the mixed-step layout."""
        num_reqs = m.num_reqs
        query_starts = m.query_start_loc_cpu.numpy()[: num_reqs + 1]
        is_prefill = (num_decode_draft_tokens_cpu.numpy()[:num_reqs] < 0) & (np.diff(query_starts) > 0)
        if not is_prefill.any():
            return ReplayMetadata(num_rows=num_reqs)

        rows = np.flatnonzero(is_prefill)
        layout = GDNLayout.of_rows(self.vllm_config, m, rows)
        prefill = self.mixed_step_metadata(m.block_table_tensor, layout.T, layout, rows)
        first = int(rows[0])
        if is_prefill[first:].all():
            # the step's last rows, as vLLM's runner sorts the rows with drafts first
            tokens = slice(int(query_starts[first]), int(query_starts[num_reqs]))
            return ReplayMetadata(num_rows=first, prefill=prefill, prefill_rows=rows, prefill_tokens=tokens)

        # vLLM's runner sorts a prompt chunk of k + 1 tokens before a decode row without drafts: the verify core takes
        # every row, MixedStepGDNCore the prefill rows' tokens gathered
        tokens = np.concatenate([np.arange(query_starts[row], query_starts[row + 1]) for row in rows])
        return ReplayMetadata(
            num_rows=num_reqs,
            prefill=prefill,
            prefill_rows=rows,
            prefill_tokens=torch.from_numpy(tokens).to(m.query_start_loc.device),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # a FULL graph's verify step reads the rows the runner's preprocess writes before each replay (a capture finds
        # none: every step's commit ends its rows); a decode graph's batch (one token per request) as vLLM builds it; a
        # mixed-step graph over the layout without tokens: a replay touches nothing until a step fills it
        m = common_attn_metadata
        if self.use_spec_decode:
            return ReplayMetadata(num_rows=m.num_reqs)
        if m.max_query_len <= 1:
            return super().build_for_cudagraph_capture(m)

        layout = GDNLayout.of_engine(self.vllm_config, m.query_start_loc.device)
        layout.fill(np.zeros(1, dtype=np.int64), 0, np.zeros(0, dtype=np.int32))

        return self.mixed_step_metadata(m.block_table_tensor, m.num_actual_tokens, layout)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        if isinstance(metadata, ReplayMetadata):
            if metadata.prefill is None:
                return metadata
            prefill = metadata.prefill
            prefill = self.mixed_step_metadata(
                blk_table, prefill.num_actual_tokens, prefill._pear_layout, metadata.prefill_rows
            )
            return dataclasses.replace(metadata, prefill=prefill)
        layout = getattr(metadata, "_pear_layout", None)
        if layout is None:
            return super().update_block_table(metadata, blk_table, slot_mapping)

        return self.mixed_step_metadata(blk_table, metadata.num_actual_tokens, layout)

    def mixed_step_metadata(
        self, block_table, num_actual_tokens: int, layout: GDNLayout, rows: np.ndarray | None = None
    ) -> GDNAttentionMetadata:
        """vLLM's GDN metadata of a mixed step with its layout and this group's state slots: of the step's first
        layout.N rows, or of the given rows (ascending)."""
        if rows is None:
            self.mixed_step_slots[: layout.N].copy_(self.state_slots(block_table, layout.N))
        else:
            slots = self.state_slots(block_table, int(rows[-1]) + 1)
            if rows[-1] - rows[0] + 1 == len(rows):
                self.mixed_step_slots[: layout.N].copy_(slots[int(rows[0]) :])
            else:
                self.mixed_step_slots[: layout.N].copy_(slots[torch.from_numpy(rows).to(slots.device)])

        md = GDNAttentionMetadata(
            num_prefills=layout.P,
            num_prefill_tokens=layout.T - layout.D,
            num_decodes=layout.D,
            num_decode_tokens=layout.D,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_actual_tokens,
            has_initial_state=layout.seq_init,
            non_spec_state_indices_tensor=self.mixed_step_slots,
        )
        md._pear_layout = layout

        return md


class HopperGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with HopperGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return HopperGDNMetadataBuilder
