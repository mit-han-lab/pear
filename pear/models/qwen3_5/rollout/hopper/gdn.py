import numpy as np
import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

from pear.models.qwen3_5.rollout.gdn import RolloutGDNAttention, RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims
from pear.models.qwen3_5.rollout.hopper.staging import StagedBuffer

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
    def of_step(cls, vllm_config, m) -> "GDNLayout":
        """A mixed step's layout from vLLM's common metadata, in the process's layout."""
        qsl_all = m.query_start_loc_cpu[: m.num_reqs + 1].numpy().astype(np.int64)
        query_lens = np.diff(qsl_all)
        n = int(np.count_nonzero(query_lens))  # CUDA graph padding requests are empty and last
        qsl = qsl_all[: n + 1]

        # the decode rows: the leading one-token requests (vLLM orders decodes first)
        prefill_rows = np.flatnonzero(query_lens[:n] != 1)
        D = int(prefill_rows[0]) if len(prefill_rows) else n
        layout = cls.of_engine(vllm_config, m.query_start_loc.device)

        seq_lens = (m._seq_lens_cpu if m._seq_lens_cpu is not None else m.seq_lens.cpu())[:n].numpy()
        layout.fill(qsl, D, (seq_lens - query_lens[:n] > 0).astype(np.int32))

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


class GDNProjection:
    """One GEMM for a Gated DeltaNet layer's in_proj_qkvz and in_proj_ba, over one weight both are views of."""

    # in_proj_ba costs a launch per layer at any batch size. The views are made as the layer is built, so the loaders
    # write into the one tensor; every consumer takes the output's row strides
    @staticmethod
    def merge(layer) -> None:
        """Make the two projections' weights views of one [qkvz; ba] tensor, before the weights load."""
        qkvz, ba = layer.in_proj_qkvz, layer.in_proj_ba
        if (
            getattr(qkvz, "bias", None) is not None
            or getattr(ba, "bias", None) is not None
            or not isinstance(qkvz.quant_method, UnquantizedLinearMethod)
            or not isinstance(ba.quant_method, UnquantizedLinearMethod)
            or layer.disable_tp_for_ba_proj
            or layer.gqa_interleaved_layout
        ):
            return

        w1, w2 = qkvz.weight, ba.weight
        if w1.dtype != w2.dtype or w1.shape[1] != w2.shape[1] or not w1.is_contiguous() or not w2.is_contiguous():
            return

        n = w1.shape[0]
        qkvzba = torch.empty((n + w2.shape[0], w1.shape[1]), dtype=w1.dtype, device=w1.device)
        w1.data = qkvzba[:n]
        w2.data = qkvzba[n:]
        layer.register_buffer("qkvzba_weight", qkvzba, persistent=False)
        layer.qkvz_out = n

    @staticmethod
    def eligible(layer, hidden_states) -> bool:
        """The fused-norm path vLLM takes for this replica, on a merged weight."""
        return (
            getattr(layer, "qkvzba_weight", None) is not None
            and layer.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and layer.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )

    @staticmethod
    def check(model) -> None:
        """Raise unless every merged layer's projections still view its merged weight (a loader replaced one)."""
        for module in model.modules():
            qkvzba = getattr(module, "qkvzba_weight", None)
            if qkvzba is None:
                continue
            qkvz, ba = module.in_proj_qkvz.weight, module.in_proj_ba.weight
            if qkvz.data_ptr() != qkvzba.data_ptr() or ba.data_ptr() != qkvzba[module.qkvz_out :].data_ptr():
                raise RuntimeError(f"{module.prefix}: in_proj_qkvz / in_proj_ba no longer view the merged weight")


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


class HopperGDNAttention(RolloutGDNAttention):
    """vLLM's Qwen Gated DeltaNet layer on one projection GEMM, MixedStepGDNCore in a mixed step and the fused decode
    kernel (registered out of tree in its place)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GDNProjection.merge(self)
        self.conv_state_dim_first = is_conv_state_dim_first()  # the conv state's layout, fixed for the process

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
        context = get_forward_context()
        md = context.attn_metadata[self.prefix] if context.attn_metadata is not None else None
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


class HopperGDNMetadataBuilder(RolloutGDNMetadataBuilder):
    """vLLM's GDN metadata builder with a mixed step's layout for MixedStepGDNCore, which runs any mixed step of a
    graph's size in a FULL graph; the other GDN groups' metadata derives from the first group's (update_block_table)."""

    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # this group's state slots in a mixed step, at the address its FULL mixed-step graphs read
        self.mixed_step_slots = torch.zeros(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device)

        # as vLLM's Mamba builders: a speculative step's draft-token arguments do not carry over to another group
        self.supports_update_block_table = not self.use_spec_decode

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        m = common_attn_metadata
        if self.use_spec_decode or m.max_query_len <= 1:
            md = super().build(
                common_prefix_len,
                m,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                fast_build=fast_build,
            )
            md._pear_layout, md._pear_common_attn_metadata = None, m
            return md

        return self.mixed_step_metadata(m, GDNLayout.of_step(self.vllm_config, m))

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = common_attn_metadata
        if self.use_spec_decode or m.max_query_len <= 1:
            return super().build_for_cudagraph_capture(m)

        # a mixed-step graph is captured over the layout without tokens: a replay touches nothing until a step fills it
        layout = GDNLayout.of_engine(self.vllm_config, m.query_start_loc.device)
        layout.fill(np.zeros(1, dtype=np.int64), 0, np.zeros(0, dtype=np.int32))

        return self.mixed_step_metadata(m, layout)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        common_attn_metadata = self.group_common_metadata(metadata._pear_common_attn_metadata, blk_table, slot_mapping)

        if metadata._pear_layout is None:
            return self.build(0, common_attn_metadata)
        return self.mixed_step_metadata(common_attn_metadata, metadata._pear_layout)

    def mixed_step_metadata(self, m, layout: GDNLayout) -> GDNAttentionMetadata:
        """vLLM's GDN metadata of a mixed step with its layout and this group's state slots."""
        # each sequence's state block (under mamba_cache_mode align, the block of its last token)
        n = layout.N
        slots = mamba_get_block_table_tensor(
            m.block_table_tensor[:n], m.seq_lens[:n], self.kv_cache_spec, self.vllm_config.cache_config.mamba_cache_mode
        )[:, 0]
        self.mixed_step_slots[:n].copy_(slots)

        md = GDNAttentionMetadata(
            num_prefills=layout.P,
            num_prefill_tokens=layout.T - layout.D,
            num_decodes=layout.D,
            num_decode_tokens=layout.D,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=layout.seq_init,
            non_spec_state_indices_tensor=self.mixed_step_slots,
        )
        md._pear_layout, md._pear_common_attn_metadata = layout, m

        return md


class HopperGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with HopperGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return HopperGDNMetadataBuilder
