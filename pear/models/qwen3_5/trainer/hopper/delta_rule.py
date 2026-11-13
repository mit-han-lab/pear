import torch
import torch.distributed as dist
import triton
import triton.language as tl
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    compress_h0,
    expand_h0,
    merge_fwd_bwd_kernel,
    pre_process_fwd_kernel_merged,
)
from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp2

from pear.models.qwen3_5.trainer.operands import Fp16Operands, RecomputeFlag


@triton.jit
def _finite_kernel(X, FLAG, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
    non_finite = tl.max(tl.where((x != x) | (tl.abs(x) == float("inf")), 1, 0), 0)
    tl.atomic_max(FLAG, non_finite.to(tl.int32))


@triton.jit
def _summary_kernel(
    KK,
    W,
    U,
    G,
    HM,
    CU,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    # The summary hm = [h | M] [HV, K, V + K] of the rank's last sequence: its local tokens map an incoming state x to
    # M x + h. Each program runs the chunk recurrence X' = e^(g_last) X + k~^T (u - w X) for one column block (h from
    # X = 0 on u, M from X = I on u = 0) in fp32 with TF32 dots; FLA forms M as a chain of K x K products.
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    bos = tl.load(CU).to(tl.int64)
    T = (tl.load(CU + 1).to(tl.int64) - bos).to(tl.int32)
    is_h = i_col * BS < V
    o_k = tl.arange(0, K)
    o_c = tl.arange(0, BS)

    k_base = KK + (bos * H + i_h // (HV // H)) * K
    w_base = W + (bos * HV + i_h) * K
    g_base = G + bos * HV + i_h
    u_base = U + (bos * HV + i_h) * V + i_col * BS

    if is_h:
        b_x = tl.zeros([K, BS], dtype=tl.float32)
    else:
        b_x = tl.where(o_k[:, None] == (i_col * BS - V + o_c)[None, :], 1.0, 0.0)

    for i_t in range(tl.cdiv(T, BT)):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T

        b_w = tl.load(w_base + o_t[:, None] * (HV * K) + o_k[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
        b_v = -tl.dot(b_w, b_x, input_precision="tf32")
        if is_h:
            b_u = tl.load(u_base + o_t[:, None] * (HV * V) + o_c[None, :], mask=m_t[:, None], other=0.0)
            b_v += b_u.to(tl.float32)

        last_row = tl.minimum((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g_base + last_row * HV)
        b_g = tl.load(g_base + o_t * HV, mask=m_t, other=0.0)
        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0.0)[:, None]
        b_x = b_x * exp2(b_g_last)

        b_k = tl.load(k_base + o_k[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0).to(tl.float32)
        b_x += tl.dot(b_k, b_v, input_precision="tf32")

    tl.store(HM + i_h * K * (V + K) + o_k[:, None] * (V + K) + i_col * BS + o_c[None, :], b_x)


class HopperDeltaRule(torch.autograd.Function):
    """The gated delta rule over packed local tokens on FLA's chunked kernels, with Ulysses context parallelism."""

    CHUNK = 64  # FLA's chunk size
    FINITE_ELEMENTS = 8192  # elements per program of the finiteness check
    FINITE_WARPS = 8  # its warps
    SUMMARY_COLUMNS = 64  # columns of [h | M] per program of the summary kernel
    SUMMARY_WARPS = 4  # its warps
    SUMMARY_STAGES = 2  # its pipeline stages
    FLA_SUMMARY_BLOCK = 64  # FLA's pre-process block for head dimensions above 64 (Qwen3.5's are 128)
    # the decoder layers below this take their boundary summary from _summary_kernel in the forward on fp16 twins
    SUMMARY_LAYER_N = 56

    @staticmethod
    def apply_packed(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context, q16, k16, v16, layer_idx):
        """(o [1, T, HV, V], its fp16 twin or None) for q, k [1, T, H, K] (L2-normalized), v, g (log decay) and beta;
        below SUMMARY_LAYER_N (layer_idx) the forward on twins takes the rank's boundary summary from
        _summary_kernel."""
        if q16 is not None:
            q16, k16, v16 = q16.contiguous(), k16.contiguous(), v16.contiguous()

        return HopperDeltaRule.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            cu_seqlens,
            cu_seqlens_cpu,
            cp_context,
            q16,
            k16,
            v16,
            layer_idx < HopperDeltaRule.SUMMARY_LAYER_N,
        )

    # Hopper's delta rule (cuDNN offers its FROST plan on Blackwell only). With fp16 twins of q, k, v the forward runs
    # FLA's fp16 kernels on them and returns (o, o16): o16 their output, o its bf16 rounding for autograd; the backward
    # is FLA's on the bf16 operands.
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context, q16, k16, v16, uses_summary_kernel):
        scale = k.shape[-1] ** -0.5
        chunk_indices = prepare_chunk_indices(cu_seqlens, HopperDeltaRule.CHUNK, cu_seqlens_cpu=cu_seqlens_cpu)
        forward_is_fp16 = q16 is not None

        # the checkpoint's recompute (no twins) keeps the WY factor w, the chunk states h and the new values v_new,
        # which FLA's backward would recompute from the same operands; the checkpointed forward saves placeholders
        # with their metadata (the checkpoint compares it with the recompute's)
        keeps_intermediates = any(ctx.needs_input_grad) and not forward_is_fp16
        g_cum, o, A, initial_state, intermediates = HopperDeltaRule.chunked_forward(
            *((q16, k16, v16) if forward_is_fp16 else (q, k, v)),
            g,
            beta,
            scale,
            cu_seqlens,
            chunk_indices,
            cp_context,
            uses_summary_kernel and forward_is_fp16,
            keeps_intermediates,
        )

        saved_tensors = [q, k, v, g_cum, beta, A.to(k.dtype), initial_state, cu_seqlens, chunk_indices]
        if keeps_intermediates:
            saved_tensors += intermediates
        elif any(ctx.needs_input_grad) and RecomputeFlag.saves_discarded():
            T, HV, K, V = k.shape[1], v.shape[2], k.shape[-1], v.shape[-1]
            metadata = (
                ((1, T, HV, K), k.dtype),
                ((1, len(chunk_indices), HV, K, V), k.dtype),
                ((1, T, HV, V), v.dtype),
            )
            saved_tensors += [RecomputeFlag.placeholder(shape, dtype, k.device) for shape, dtype in metadata]
        ctx.save_for_backward(*saved_tensors)
        ctx.scale, ctx.cp_context = scale, cp_context

        if not forward_is_fp16:
            return o, None

        # a NaN or an infinity in the fp16 output raises the range flag: the batch reruns on bf16 operands
        n = o.numel()
        if n:
            _finite_kernel[(triton.cdiv(n, HopperDeltaRule.FINITE_ELEMENTS),)](
                o,
                Fp16Operands.range_flag(o.device),
                n,
                HopperDeltaRule.FINITE_ELEMENTS,
                num_warps=HopperDeltaRule.FINITE_WARPS,
            )
        ctx.mark_non_differentiable(o)
        return o.to(q.dtype), o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, _twin):
        q, k, v, g_cum, beta, A, initial_state, cu_seqlens, chunk_indices, *intermediates = ctx.saved_tensors
        do = do.contiguous()

        if intermediates:
            dq, dk, dv, db, dg = HopperDeltaRule.reusing_backward(
                q,
                k,
                v,
                g_cum,
                beta,
                A,
                ctx.scale,
                initial_state,
                do,
                cu_seqlens,
                chunk_indices,
                ctx.cp_context,
                *intermediates,
            )
        else:
            dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                g=g_cum,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=None,
                cu_seqlens=cu_seqlens,
                cp_context=ctx.cp_context,
                chunk_indices=chunk_indices,
                chunk_size=HopperDeltaRule.CHUNK,
            )

        return dq.to(q), dk.to(k), dv.to(v), dg.to(g_cum), db.to(beta), None, None, None, None, None, None, None

    @staticmethod
    def chunked_forward(
        q, k, v, g, beta, scale, cu_seqlens, chunk_indices, cp_context, uses_summary_kernel, keeps_intermediates
    ):
        """FLA's chunk_gated_delta_rule_fwd stage by stage (no initial or final state): (g_cum, o, A, the compressed
        initial state, (w, h, v_new) if keeps_intermediates)."""
        g_cum = chunk_local_cumsum(
            g, chunk_size=HopperDeltaRule.CHUNK, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g_cum,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=HopperDeltaRule.CHUNK,
        )

        initial_state = (
            None
            if cp_context is None
            else HopperDeltaRule.pre_process(k, w, u, g_cum, cu_seqlens, cp_context, uses_summary_kernel)
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g_cum,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=HopperDeltaRule.CHUNK,
        )
        del u

        if cp_context is not None:
            initial_state = compress_h0(initial_state, context=cp_context)

        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=HopperDeltaRule.CHUNK,
        )

        return g_cum, o, A, initial_state, (w, h, v_new) if keeps_intermediates else None

    @staticmethod
    def pre_process(k, w, u, g, cu_seqlens, cp_context, uses_summary_kernel):
        """FLA's chunk_gated_delta_rule_fwd_h_pre_process: the incoming state [N, HV, K, V] of every local sequence."""
        # chained from the all-gathered summaries of the ranks' boundary sequences; this rank's comes from
        # _summary_kernel with uses_summary_kernel, else from FLA's kernel
        _, T, H, K = k.shape
        HV, V = u.shape[2], u.shape[-1]

        hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
        initial_state = k.new_zeros(len(cu_seqlens) - 1, HV, K, V, dtype=torch.float32)

        if not cp_context.is_last_rank:
            if uses_summary_kernel:
                _summary_kernel[((V + K) // HopperDeltaRule.SUMMARY_COLUMNS, HV)](
                    k,
                    w,
                    u,
                    g,
                    hm,
                    cu_seqlens[-2:],
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=HopperDeltaRule.CHUNK,
                    BS=HopperDeltaRule.SUMMARY_COLUMNS,
                    num_warps=HopperDeltaRule.SUMMARY_WARPS,
                    num_stages=HopperDeltaRule.SUMMARY_STAGES,
                )
            else:
                grid = (
                    triton.cdiv(V, HopperDeltaRule.FLA_SUMMARY_BLOCK)
                    + triton.cdiv(K, HopperDeltaRule.FLA_SUMMARY_BLOCK),
                    HV,
                )
                pre_process_fwd_kernel_merged[grid](
                    k=k,
                    v=u,
                    w=w,
                    g=g,
                    gk=None,
                    bg=None,
                    u=u,
                    hm=hm,
                    cu_seqlens=cu_seqlens[-2:],
                    T=T,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=HopperDeltaRule.CHUNK,
                    BK1=triton.next_power_of_2(K),
                    BLOCK_SIZE=HopperDeltaRule.FLA_SUMMARY_BLOCK,
                    MULTI_SEQS=False,
                )

        ag_hm, _ = all_gather_into_tensor(hm, group=cp_context.group)
        if not cp_context.is_first_rank:
            # the merge chains the ranks' summaries in fp32 with 3xTF32 dots (FLA's default is TF32): no forward
            # merge is compiled outside this scope, so Triton's cache, which does not key the precision, holds 3xTF32
            with triton.knobs.language.scope():
                triton.knobs.language.fp32_default = "tf32x3"
                merge_fwd_bwd_kernel[lambda meta: (triton.cdiv(V, meta["BV"]), HV)](
                    h=initial_state[0],
                    ag_hm=ag_hm,
                    pre_or_post_num_ranks=cp_context.pre_num_ranks,
                    rank=dist.get_rank(group=cp_context.group),
                    seq_offsets=None,
                    init_offsets=None,
                    h0_seq_ids=None,
                    h0=None,
                    HV=HV,
                    K=K,
                    V=V,
                    BK=triton.next_power_of_2(K),
                    FORWARD=True,
                    INTRACARD_MODE=False,
                    NUM_SEQ_ENTRIES=0,
                    STATE_V_FIRST=False,
                )

        return initial_state

    @staticmethod
    def reusing_backward(
        q, k, v, g, beta, A, scale, initial_state, do, cu_seqlens, chunk_indices, cp_context, w, h, v_new
    ):
        """FLA's chunk_gated_delta_rule_bwd from its dv_local stage on, with the recompute's w, h and v_new."""
        dht = None
        if cp_context is not None:
            initial_state = expand_h0(initial_state, context=cp_context)

        dv = chunk_bwd_dv_local(
            q=q,
            k=k,
            g=g,
            do=do,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=HopperDeltaRule.CHUNK,
        )

        if cp_context is not None:
            dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                g=g,
                scale=scale,
                cu_seqlens=cu_seqlens,
                dht=dht,
                initial_state=initial_state,
                context=cp_context,
                state_v_first=False,
                chunk_size=HopperDeltaRule.CHUNK,
            )

        dh, _, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_v_first=False,
            chunk_size=HopperDeltaRule.CHUNK,
        )

        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_v_first=False,
            chunk_size=HopperDeltaRule.CHUNK,
        )
        del dh

        dk2, dv, db, dg2 = prepare_wy_repr_bwd(
            k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum(
            dg, chunk_size=HopperDeltaRule.CHUNK, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        return dq, dk, dv, db, dg
