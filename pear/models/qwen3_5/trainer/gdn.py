import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, apply_mask_to_padding_states

from pear.models.qwen3_5.gdn_constants import L2_NORM_EPS
from pear.models.qwen3_5.trainer.operands import (
    FP16_LIMIT,
    Fp16Operands,
    Fp32OutputLinear,
    RecomputeFlag,
    raise_range_flag,
)
from verl.models.transformers.qwen3_5 import _build_fla_cp_context
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

CONV_WIDTH = 4  # the short convolution's width
CONV_HALO = CONV_WIDTH - 1  # the earlier tokens a token's convolution reads


@triton.jit
def _conv_pre(X, P, W, back_counts, t, c, T, D: tl.constexpr):
    # the convolution's pre-activation at rows t (int64 [BT]) and channels c ([HEAD]) in fp32; the halo rows come from P
    acc = tl.zeros((t.shape[0], c.shape[0]), dtype=tl.float32)

    for j in tl.static_range(4):
        s = t - 3 + j
        ok = ((3 - j) <= back_counts) & (t < T)
        wj = tl.load(W + c * 4 + j).to(tl.float32)
        in_x = s >= 0
        xv = tl.load(X + s[:, None] * D + c[None, :], mask=(ok & in_x)[:, None], other=0.0).to(tl.float32)
        pv = tl.load(P + (s + 3)[:, None] * D + c[None, :], mask=(ok & (s < 0))[:, None], other=0.0).to(tl.float32)
        acc += (xv + pv) * wj[None, :]

    return acc


@triton.jit
def _front_fwd_kernel(
    X,
    P,
    W,
    BACK,
    Q,
    K,
    V,
    Q16,
    K16,
    V16,
    FLAG,
    T,
    D: tl.constexpr,
    NK: tl.constexpr,
    HEAD: tl.constexpr,
    BT: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BF: tl.constexpr,
    HAS16: tl.constexpr,
    LIMIT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t.to(tl.int64) * BT + tl.arange(0, BT)
    c = h * HEAD + tl.arange(0, HEAD)

    back_counts = tl.load(BACK + t, mask=t < T, other=0)
    pre = _conv_pre(X, P, W, back_counts, t, c, T, D)
    y = pre * tl.sigmoid(pre)

    m = (t < T)[:, None]
    o = tl.arange(0, HEAD)[None, :]
    if h < 2 * NK:
        rstd = 1.0 / tl.sqrt(tl.sum(y * y, 1) + EPS)
        y = y * rstd[:, None]
        if h < NK:
            if HAS_BF:
                tl.store(Q + t[:, None] * (NK * HEAD) + h * HEAD + o, y.to(Q.dtype.element_ty), mask=m)
            if HAS16:
                tl.store(Q16 + t[:, None] * (NK * HEAD) + h * HEAD + o, y.to(tl.float16), mask=m)
        else:
            if HAS_BF:
                tl.store(K + t[:, None] * (NK * HEAD) + (h - NK) * HEAD + o, y.to(K.dtype.element_ty), mask=m)
            if HAS16:
                tl.store(K16 + t[:, None] * (NK * HEAD) + (h - NK) * HEAD + o, y.to(tl.float16), mask=m)
    else:
        nv = D // HEAD - 2 * NK
        if HAS_BF:
            tl.store(V + t[:, None] * (nv * HEAD) + (h - 2 * NK) * HEAD + o, y.to(V.dtype.element_ty), mask=m)
        if HAS16:
            tl.store(V16 + t[:, None] * (nv * HEAD) + (h - 2 * NK) * HEAD + o, y.to(tl.float16), mask=m)
            raise_range_flag(FLAG, y, m, LIMIT)


@triton.jit
def _front_bwd_dpre_kernel(
    X,
    P,
    W,
    BACK,
    DQ,
    DK,
    DV,
    DPRE,
    T,
    D: tl.constexpr,
    NK: tl.constexpr,
    HEAD: tl.constexpr,
    BT: tl.constexpr,
    EPS: tl.constexpr,
):
    # dL/d(pre-activation) in fp32 from the gradients of q, k (through the L2 norm) and v
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t.to(tl.int64) * BT + tl.arange(0, BT)
    c = h * HEAD + tl.arange(0, HEAD)

    back_counts = tl.load(BACK + t, mask=t < T, other=0)
    pre = _conv_pre(X, P, W, back_counts, t, c, T, D)
    sig = tl.sigmoid(pre)
    u = pre * sig

    m = (t < T)[:, None]
    o = tl.arange(0, HEAD)[None, :]
    if h < 2 * NK:
        if h < NK:
            dy = tl.load(DQ + t[:, None] * (NK * HEAD) + h * HEAD + o, mask=m, other=0.0).to(tl.float32)
        else:
            dy = tl.load(DK + t[:, None] * (NK * HEAD) + (h - NK) * HEAD + o, mask=m, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(u * u, 1) + EPS)
        y = u * rstd[:, None]
        du = (dy - y * tl.sum(dy * y, 1)[:, None]) * rstd[:, None]
    else:
        nv = D // HEAD - 2 * NK
        du = tl.load(DV + t[:, None] * (nv * HEAD) + (h - 2 * NK) * HEAD + o, mask=m, other=0.0).to(tl.float32)

    dpre = du * (sig * (1.0 + pre * (1.0 - sig)))
    tl.store(DPRE + t[:, None] * D + c[None, :], dpre.to(DPRE.dtype.element_ty), mask=m)


@triton.jit
def _front_bwd_dx_kernel(W, BACK, DPRE, DX, T, D: tl.constexpr, HEAD: tl.constexpr, BT: tl.constexpr):
    # dx[s] = sum_d w[3 - d] * dpre[s + d] over the tokens s + d that may read s (d <= back_counts[s + d])
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * HEAD + tl.arange(0, HEAD)
    s = pid_t.to(tl.int64) * BT + tl.arange(0, BT)
    ms = s < T
    dx = tl.zeros((BT, HEAD), dtype=tl.float32)

    for d in tl.static_range(4):
        u = s + d
        ok = ms & (u < T)
        b = tl.load(BACK + u, mask=ok, other=0)
        ok = ok & (d <= b)
        g = tl.load(DPRE + u[:, None] * D + c[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        wd = tl.load(W + c * 4 + (3 - d)).to(tl.float32)
        dx += g * wd[None, :]

    tl.store(DX + s[:, None] * D + c[None, :], dx.to(DX.dtype.element_ty), mask=ms[:, None])


@triton.jit
def _front_bwd_dw_kernel(
    X, P, BACK, DPRE, DWP, T, D: tl.constexpr, HEAD: tl.constexpr, BT: tl.constexpr, CHUNK: tl.constexpr
):
    # per chunk of tokens: dw[c, j] = sum_s x[s - 3 + j, c] * dpre[s, c] over the taps in reach
    # (3 - j <= back_counts[s]), accumulated elementwise in registers and reduced over the tokens once at the end
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * HEAD + tl.arange(0, HEAD)

    a0 = tl.zeros((BT, HEAD), dtype=tl.float32)
    a1 = tl.zeros((BT, HEAD), dtype=tl.float32)
    a2 = tl.zeros((BT, HEAD), dtype=tl.float32)
    a3 = tl.zeros((BT, HEAD), dtype=tl.float32)

    for sub in range(CHUNK // BT):
        s = pid_t.to(tl.int64) * CHUNK + sub * BT + tl.arange(0, BT)
        ms = s < T
        back_counts = tl.load(BACK + s, mask=ms, other=0)
        g = tl.load(DPRE + s[:, None] * D + c[None, :], mask=ms[:, None], other=0.0).to(tl.float32)

        for j in tl.static_range(4):
            r = s - 3 + j
            ok = ms & ((3 - j) <= back_counts)
            xv = tl.load(X + r[:, None] * D + c[None, :], mask=(ok & (r >= 0))[:, None], other=0.0).to(tl.float32)
            pv = tl.load(P + (r + 3)[:, None] * D + c[None, :], mask=(ok & (r < 0))[:, None], other=0.0).to(tl.float32)
            if j == 0:
                a0 += (xv + pv) * g
            elif j == 1:
                a1 += (xv + pv) * g
            elif j == 2:
                a2 += (xv + pv) * g
            else:
                a3 += (xv + pv) * g

    base = DWP + pid_t.to(tl.int64) * D * 4 + c * 4
    tl.store(base + 0, tl.sum(a0, 0))
    tl.store(base + 1, tl.sum(a1, 0))
    tl.store(base + 2, tl.sum(a2, 0))
    tl.store(base + 3, tl.sum(a3, 0))


@triton.jit
def _gated_norm_fwd_kernel(
    X,
    G,
    W,
    Y,
    Y16,
    REMAINDER,
    RSTD,
    FLAG,
    T,
    eps,
    D: tl.constexpr,
    BT: tl.constexpr,
    HAS_Y: tl.constexpr,
    HAS_Y16: tl.constexpr,
    HAS_REMAINDER: tl.constexpr,
    LIMIT: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.arange(0, D)
    mt = rows < T
    m = mt[:, None]

    x = tl.load(X + rows[:, None] * D + cols[None, :], mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, 1) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(RSTD + rows, rstd, mask=mt)

    w = tl.load(W + cols).to(tl.float32)
    g = tl.load(G + rows[:, None] * D + cols[None, :], mask=m, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    y = y * g * tl.sigmoid(g)

    if HAS_Y:
        tl.store(Y + rows[:, None] * D + cols[None, :], y.to(Y.dtype.element_ty), mask=m)
    if HAS_Y16:
        y16 = y.to(tl.float16)
        tl.store(Y16 + rows[:, None] * D + cols[None, :], y16, mask=m)
        if HAS_REMAINDER:
            tl.store(REMAINDER + rows[:, None] * D + cols[None, :], (y - y16.to(tl.float32)).to(tl.float16), mask=m)
        raise_range_flag(FLAG, y, m, LIMIT)


@triton.jit
def _gated_norm_bwd_kernel(DY, X, G, W, RSTD, DX, DG, DWP, T, D: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.arange(0, D)
    mt = rows < T
    m = mt[:, None]

    x = tl.load(X + rows[:, None] * D + cols[None, :], mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + rows[:, None] * D + cols[None, :], mask=m, other=0.0).to(tl.float32)
    dy = tl.load(DY + rows[:, None] * D + cols[None, :], mask=m, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=mt, other=0.0)
    w = tl.load(W + cols).to(tl.float32)

    sig = tl.sigmoid(g)
    silu = g * sig
    xhat = x * rstd[:, None]
    # y = xhat * w * silu(g)
    dg = dy * xhat * w[None, :] * (sig * (1.0 + g * (1.0 - sig)))
    dxhat = dy * w[None, :] * silu
    dx = rstd[:, None] * (dxhat - xhat * (tl.sum(dxhat * xhat, 1) / D)[:, None])

    tl.store(DX + rows[:, None] * D + cols[None, :], dx.to(DX.dtype.element_ty), mask=m)
    tl.store(DG + rows[:, None] * D + cols[None, :], dg.to(DG.dtype.element_ty), mask=m)
    tl.store(DWP + pid.to(tl.int64) * D + cols, tl.sum(tl.where(m, dy * xhat * silu, 0.0), 0))


class HostBoundaries:
    """One host copy per micro-batch of the packed boundaries verl moved to the GPU, for every host read of them."""

    HOST_COPIES = 8  # micro-batches whose host boundaries stay cached

    # every host read of a device tensor (here, in FLA's context-parallel set-up, in the kernels' launch arguments)
    # waits for the GPU to drain its queue, in every layer and recompute
    copies: list = []  # (device boundaries, host copy), the most recent micro-batches first

    @classmethod
    def of(cls, cu_seqlens: torch.Tensor, cu_seqlens_cpu) -> torch.Tensor:
        if cu_seqlens_cpu is not None and cu_seqlens_cpu.device.type == "cpu":
            return cu_seqlens_cpu

        source = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens

        for tensor, host in cls.copies:
            if tensor is source:
                return host

        host = source.to("cpu")
        cls.copies.insert(0, (source, host))
        del cls.copies[cls.HOST_COPIES :]

        return host


class GDNFrontEnd(torch.autograd.Function):
    """mixed_qkv [1, T, D] -> q, k (L2-normalized), v: the causal convolution, SiLU and q/k's L2 norm in one kernel."""

    HEAD_DIM = 128  # channels per head (q/k: linear_key_head_dim, v: linear_value_head_dim)
    FWD_TOKENS = 64  # tokens per program of the forward kernel
    FWD_WARPS = 4  # its warps
    DPRE_TOKENS = 32  # tokens per program of the backward's pre-activation kernel
    DPRE_WARPS = 4  # its warps
    DX_TOKENS = 32  # tokens per program of the backward's transposed convolution
    DX_WARPS = 4  # its warps
    DW_TOKENS = 32  # tokens per pass of the weight gradient's kernel
    DW_CHUNK_TOKENS = 512  # tokens a program of that kernel accumulates before its one reduction
    DW_WARPS = 4  # its warps
    DPRE_DTYPE = torch.bfloat16  # the gradient at the convolution's output, rounded as the stock path rounds it

    # fp32 math with one rounding; returns (q, k, v, q16, k16, v16): the bf16 operands autograd sees and, with fp16 ones
    # asked for, their fp16 twins; without autograd and with fp16 asked for, only the fp16 ones (in the first places).
    # Under Ulysses the halo is the previous rank's last 3 rows (prefix), its gradient returns to it in the backward.
    @staticmethod
    def forward(ctx, mixed_qkv, weight, back_counts, prefix, group, NK, wants_fp16):
        x = mixed_qkv[0]
        T, D = x.shape
        w = weight.reshape(D, CONV_WIDTH).contiguous()
        nv = D // GDNFrontEnd.HEAD_DIM - 2 * NK

        has_output = any(ctx.needs_input_grad) or not wants_fp16
        shapes = ((1, T, NK, GDNFrontEnd.HEAD_DIM), (1, T, NK, GDNFrontEnd.HEAD_DIM), (1, T, nv, GDNFrontEnd.HEAD_DIM))
        # the delta rule's operands are bf16 (and fp16) whatever the projection's output dtype: one rounding
        q, k, v = (torch.empty(s, device=x.device, dtype=torch.bfloat16) if has_output else None for s in shapes)
        q16, k16, v16 = (torch.empty(s, device=x.device, dtype=torch.float16) if wants_fp16 else None for s in shapes)
        placeholder = q if has_output else q16

        _front_fwd_kernel[(triton.cdiv(T, GDNFrontEnd.FWD_TOKENS), D // GDNFrontEnd.HEAD_DIM)](
            x,
            prefix,
            w,
            back_counts,
            q if has_output else placeholder,
            k if has_output else placeholder,
            v if has_output else placeholder,
            q16 if wants_fp16 else placeholder,
            k16 if wants_fp16 else placeholder,
            v16 if wants_fp16 else placeholder,
            Fp16Operands.range_flag(x.device),
            T,
            D,
            NK,
            GDNFrontEnd.HEAD_DIM,
            GDNFrontEnd.FWD_TOKENS,
            L2_NORM_EPS,
            has_output,
            wants_fp16,
            FP16_LIMIT,
            num_warps=GDNFrontEnd.FWD_WARPS,
        )

        ctx.save_for_backward(x, w, back_counts, prefix)
        ctx.group, ctx.NK, ctx.weight_shape = group, NK, weight.shape

        if not has_output:
            return q16, k16, v16, None, None, None
        if wants_fp16:
            ctx.mark_non_differentiable(q16, k16, v16)
        return q, k, v, q16, k16, v16

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dq, dk, dv, *_twins):
        x, w, back_counts, prefix = ctx.saved_tensors
        T, D = x.shape
        NK = ctx.NK
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()

        # the gradient at the convolution's output, then the transposed convolution and the weight gradient
        dpre = torch.empty(T, D, device=x.device, dtype=GDNFrontEnd.DPRE_DTYPE)
        _front_bwd_dpre_kernel[(triton.cdiv(T, GDNFrontEnd.DPRE_TOKENS), D // GDNFrontEnd.HEAD_DIM)](
            x,
            prefix,
            w,
            back_counts,
            dq,
            dk,
            dv,
            dpre,
            T,
            D,
            NK,
            GDNFrontEnd.HEAD_DIM,
            GDNFrontEnd.DPRE_TOKENS,
            L2_NORM_EPS,
            num_warps=GDNFrontEnd.DPRE_WARPS,
        )

        dx = torch.empty_like(x)
        _front_bwd_dx_kernel[(triton.cdiv(T, GDNFrontEnd.DX_TOKENS), D // GDNFrontEnd.HEAD_DIM)](
            w, back_counts, dpre, dx, T, D, GDNFrontEnd.HEAD_DIM, GDNFrontEnd.DX_TOKENS, num_warps=GDNFrontEnd.DX_WARPS
        )

        chunks = triton.cdiv(T, GDNFrontEnd.DW_CHUNK_TOKENS)
        dw_parts = torch.empty(chunks, D, CONV_WIDTH, device=x.device, dtype=torch.float32)
        _front_bwd_dw_kernel[(chunks, D // GDNFrontEnd.HEAD_DIM)](
            x,
            prefix,
            back_counts,
            dpre,
            dw_parts,
            T,
            D,
            GDNFrontEnd.HEAD_DIM,
            GDNFrontEnd.DW_TOKENS,
            GDNFrontEnd.DW_CHUNK_TOKENS,
            num_warps=GDNFrontEnd.DW_WARPS,
        )
        dw = dw_parts.sum(0)

        if ctx.group is not None:
            GDNFrontEnd.return_halo_gradient(dx, dpre, w, back_counts, ctx.group)

        return dx.unsqueeze(0), dw.reshape(ctx.weight_shape).to(w.dtype), None, None, None, None, None

    @staticmethod
    def return_halo_gradient(dx, dpre, w, back_counts, group) -> None:
        """Add the next rank's gradient of this rank's last 3 rows (its halo) to dx; send ours to the previous rank."""
        T, D = dx.shape

        # prefix row p (token p - 3) feeds the first tokens t = p - 3 + d, d <= back_counts[t]
        dprefix = torch.zeros(CONV_HALO, D, device=dx.device, dtype=torch.float32)
        head = dpre[:CONV_HALO].float()
        b = back_counts[:CONV_HALO].to(torch.int64)

        for p in range(CONV_HALO):
            for d in range(CONV_HALO - p, CONV_WIDTH):
                t = p - CONV_HALO + d
                if t < T:
                    dprefix[p] += torch.where(d <= b[t], head[t] * w[:, CONV_HALO - d].float(), 0.0)

        world, rank = dist.get_world_size(group), dist.get_rank(group)
        dprefixes = torch.empty(world, CONV_HALO, D, device=dx.device, dtype=torch.float32)
        dist.all_gather_into_tensor(dprefixes, dprefix, group=group)

        if rank + 1 < world:
            dx[-CONV_HALO:] = (dx[-CONV_HALO:].float() + dprefixes[rank + 1]).to(dx.dtype)

    @staticmethod
    def back_counts(local_cu_seqlens_cpu: torch.Tensor, valid_prefix: int, T: int, device) -> torch.Tensor:
        """Per local token, how many earlier tokens of its sequence the convolution reads (at most 3, halo included)."""
        with torch.device("cpu"):  # verl makes CUDA the default device
            cu_seqlens = local_cu_seqlens_cpu.to(device="cpu", dtype=torch.int64)
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            seq_start = torch.repeat_interleave(cu_seqlens[:-1], lengths, output_size=T)
            pos = torch.arange(T, dtype=torch.int64) - seq_start
            if valid_prefix:
                pos[: int(lengths[0])] += valid_prefix
            pos = pos.clamp_(max=CONV_HALO).to(torch.int32)

        return pos.to(device, non_blocking=True)


class GatedRMSNorm(torch.autograd.Function):
    """FLA's gated norm y = x * rstd(x) * w * silu(g) (fp32 math, one rounding): (y, its fp16 twin, the remainder)."""

    NORM_ROWS = 32  # rows per program of the gated norm's kernels
    NORM_WARPS = 4  # their warps

    # x16: the fp16 values of x when x is the bf16 handle of an fp16 delta rule, used in place of x's own values;
    # recompute_is_fp16: the GPU backend's (TrainingGpuBackend), whether the recompute has them too
    @staticmethod
    def forward(ctx, x, g, weight, eps, x16, recompute_is_fp16):
        xs = x16 if x16 is not None else x
        D = xs.shape[-1]
        x2 = xs.reshape(-1, D).contiguous()
        g2 = g.reshape(-1, D).contiguous()
        T = x2.shape[0]

        y, y16, remainder = Fp16Operands.output_buffers(
            ctx, (T, D), x.dtype, x2.device, Fp16Operands.is_enabled, Fp16Operands.wants_remainder()
        )
        rstd = torch.empty(T, device=x2.device, dtype=torch.float32)

        _gated_norm_fwd_kernel[(triton.cdiv(T, GatedRMSNorm.NORM_ROWS),)](
            x2,
            g2,
            weight,
            y if y is not None else y16,
            y16 if y16 is not None else rstd,
            remainder if remainder is not None else rstd,
            rstd,
            Fp16Operands.range_flag(x2.device),
            T,
            eps,
            D,
            GatedRMSNorm.NORM_ROWS,
            y is not None,
            y16 is not None,
            remainder is not None,
            FP16_LIMIT,
            num_warps=GatedRMSNorm.NORM_WARPS,
        )

        ctx.save_for_backward(x2 if recompute_is_fp16 else x.reshape(-1, D).contiguous(), g2, weight, rstd)
        ctx.shapes = (x.shape, g.shape, x.dtype, g.dtype)

        outputs = Fp16Operands.twin_outputs(ctx, y, y16, remainder)
        return tuple(None if output is None else output.view(x.shape) for output in outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy, *_twins):
        x2, g2, weight, rstd = ctx.saved_tensors
        x_shape, g_shape, x_dtype, g_dtype = ctx.shapes
        T, D = x2.shape
        dy = dy.reshape(T, D).contiguous()

        dx = torch.empty(T, D, device=x2.device, dtype=x_dtype)
        dg = torch.empty(T, D, device=x2.device, dtype=g_dtype)
        programs = triton.cdiv(T, GatedRMSNorm.NORM_ROWS)
        dw_parts = torch.empty(programs, D, device=x2.device, dtype=torch.float32)

        _gated_norm_bwd_kernel[(programs,)](
            dy, x2, g2, weight, rstd, dx, dg, dw_parts, T, D, GatedRMSNorm.NORM_ROWS, num_warps=GatedRMSNorm.NORM_WARPS
        )

        return dx.view(x_shape), dg.view(g_shape), dw_parts.sum(0).to(weight.dtype), None, None, None


class GatedDeltaNet:
    """Qwen3.5's Gated DeltaNet mixer on a packed batch: fp32 projections, fused front-end, delta rule, gated norm."""

    @staticmethod
    def install(model, gpu_backend) -> None:
        """Patch the mixers' forward and their projections; each mixer holds the GPU backend."""
        Qwen3_5GatedDeltaNet.forward = GatedDeltaNet.forward

        for layer in model.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                mixer = layer.linear_attn
                mixer.gpu_backend = gpu_backend

                # fp32 outputs: the front-end, the gated norm and the gates round them once; the output projection's
                # joins the fp32 residual stream
                projections = (mixer.in_proj_qkv, mixer.in_proj_z, mixer.in_proj_a, mixer.in_proj_b, mixer.out_proj)

                for projection in projections:
                    Fp32OutputLinear.install(projection)

    @staticmethod
    def forward(self, hidden_states, cache_params=None, attention_mask=None, cu_seqlens=None, cu_seqlens_cpu=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("Packed Qwen3.5 linear attention expects batch size 1.")

        cu_seqlens_cpu = HostBoundaries.of(cu_seqlens, cu_seqlens_cpu)
        total = int(cu_seqlens_cpu[-1])
        cp_context, group, valid_prefix = None, None, 0
        local_cu_seqlens, local_cu_seqlens_cpu = cu_seqlens, cu_seqlens_cpu

        if total != seq_len:
            sp = get_ulysses_sequence_parallel_world_size()
            if sp <= 1 or total % sp != 0 or seq_len != total // sp:
                raise ValueError(f"Packed cu_seqlens end {total} does not match seq_len {seq_len}.")
            cp_context = _build_fla_cp_context(cu_seqlens, cu_seqlens_cpu, self.conv_kernel_size)
            group = cp_context.group
            local_cu_seqlens, local_cu_seqlens_cpu = cp_context.cu_seqlens, cp_context.cu_seqlens_cpu
            valid_prefix = min(CONV_HALO, int(cp_context.pre_num_conv_tokens or 0))

        # the halo (every rank's last 3 rows of mixed_qkv, all-gathered) travels while the z, b, a projections run
        mixed_qkv = self.in_proj_qkv(hidden_states)
        if group is not None:
            tails = mixed_qkv[0, -CONV_HALO:].detach().contiguous()
            rank_tails = torch.empty(dist.get_world_size(group), *tails.shape, device=tails.device, dtype=tails.dtype)
            exchange = dist.all_gather_into_tensor(rank_tails, tails, group=group, async_op=True)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if group is not None:
            exchange.wait()
            rank = dist.get_rank(group)
            prefix = rank_tails[rank - 1] if rank > 0 else torch.zeros_like(rank_tails[0])
        else:
            prefix = mixed_qkv.new_zeros(CONV_HALO, mixed_qkv.shape[-1])

        # the convolution's reach per token depends on the packing only: once per micro-batch, on FLA's context
        if cp_context is None:
            back_counts = GDNFrontEnd.back_counts(local_cu_seqlens_cpu, 0, seq_len, hidden_states.device)
        else:
            if getattr(cp_context, "_pear_back_counts", None) is None:
                cp_context._pear_back_counts = GDNFrontEnd.back_counts(
                    local_cu_seqlens_cpu, valid_prefix, seq_len, hidden_states.device
                )
            back_counts = cp_context._pear_back_counts

        # the front-end's fp16 twins: in the first forward, and in the GPU backend's fp16 recompute
        gpu_backend = self.gpu_backend
        query, key, value, q16, k16, v16 = GDNFrontEnd.apply(
            mixed_qkv.contiguous(),
            self.conv1d.weight,
            back_counts,
            prefix,
            group,
            self.num_k_heads,
            Fp16Operands.is_enabled and (gpu_backend.recompute_is_fp16 or not RecomputeFlag.in_recompute.get()),
        )

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        core_attn_out, core16 = gpu_backend.delta_rule_cls.apply_packed(
            query,
            key,
            value,
            g,
            beta,
            local_cu_seqlens,
            local_cu_seqlens_cpu,
            cp_context,
            q16,
            k16,
            v16,
            self.layer_idx,
        )
        del q16, k16, v16

        # the gated norm's output with its fp16 twin and remainder for the output projection
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        if core16 is not None:
            core16 = core16.reshape(-1, self.head_v_dim)
        y, y16, remainder = GatedRMSNorm.apply(
            core_attn_out,
            z.reshape(-1, self.head_v_dim),
            self.norm.weight,
            self.norm.eps,
            core16,
            gpu_backend.recompute_is_fp16,
        )
        del core_attn_out, core16
        y = y.reshape(batch_size, seq_len, -1)
        if y16 is None and y.dtype == torch.float16:
            y16 = y  # without autograd the fp16 output is the handle
        Fp16Operands.register(
            y,
            None if y16 is None else y16.reshape(batch_size, seq_len, -1),
            None if remainder is None else remainder.reshape(batch_size, seq_len, -1),
        )

        try:
            return self.out_proj(y)
        finally:
            Fp16Operands.release(y)
