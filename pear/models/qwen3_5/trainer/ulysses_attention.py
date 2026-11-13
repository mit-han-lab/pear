import torch
import triton
import triton.language as tl

from pear.models.qwen3_5.trainer.operands import raise_range_flag

PACK_TOKENS, PACK_WARPS = 16, 4  # tokens per program and warps of the rotary pack and unpack kernels


@triton.jit
def pack_rope_kernel(
    Q,
    K,
    V,
    COS,
    SIN,
    OUT,
    FLAG,
    T,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SP: tl.constexpr,
    REP: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BT: tl.constexpr,
    CHECK: tl.constexpr,
    LIMIT: tl.constexpr,
):
    # out[t, p] for packed head p = r * heads_per_rank + j (rank-major chunks of q, k, v heads, each KV head repeated
    # REP times for the ranks that share it): the rotary embedding on q's and k's first ROT channels (x c + sign x' s,
    # x' the other half's channel), one rounding to OUT's dtype
    pid_t = tl.program_id(0)
    p = tl.program_id(1)

    hq_r: tl.constexpr = HQ // SP
    hk_r: tl.constexpr = HK * REP // SP
    heads_per_rank: tl.constexpr = hq_r + 2 * hk_r
    r = p // heads_per_rank
    j = p % heads_per_rank

    t = pid_t.to(tl.int64) * BT + tl.arange(0, BT)
    c = tl.arange(0, D)
    mt = (t < T)[:, None]

    half: tl.constexpr = ROT // 2
    partner = tl.where(c < half, c + half, tl.where(c < ROT, c - half, c))
    sign = tl.where(c < half, -1.0, 1.0)

    if j < hq_r + hk_r:
        if j < hq_r:
            base = Q + t[:, None] * (HQ * D) + (r * hq_r + j) * D
        else:
            base = K + t[:, None] * (HK * D) + ((r * hk_r + j - hq_r) // REP) * D
        x = tl.load(base + c[None, :], mask=mt, other=0.0).to(tl.float32)
        xp = tl.load(base + partner[None, :], mask=mt, other=0.0).to(tl.float32)
        rc = (c < ROT)[None, :]
        cs = tl.load(COS + t[:, None] * ROT + c[None, :], mask=mt & rc, other=1.0).to(tl.float32)
        sn = tl.load(SIN + t[:, None] * ROT + c[None, :], mask=mt & rc, other=0.0).to(tl.float32)
        y = x * cs + sign[None, :] * xp * sn
    else:
        offset = t[:, None] * (HK * D) + ((r * hk_r + j - hq_r - hk_r) // REP) * D + c[None, :]
        y = tl.load(V + offset, mask=mt, other=0.0).to(tl.float32)

    tl.store(OUT + t[:, None] * (SP * heads_per_rank * D) + p * D + c[None, :], y.to(OUT.dtype.element_ty), mask=mt)

    if CHECK:
        raise_range_flag(FLAG, y, mt, LIMIT)


@triton.jit
def unpack_rope_kernel(
    G,
    COS,
    SIN,
    DQ,
    DK,
    DV,
    T,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SP: tl.constexpr,
    REP: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BT: tl.constexpr,
):
    # the transpose of pack_rope_kernel on the packed gradients, one destination head h per program (q's heads,
    # then k's, then v's): the rotary's transposed rotation (dx = g c + sign' g' s' with the other half's g' and s') and
    # the sum of a KV head's REP copies, into dq, dk, dv (fp32)
    pid_t = tl.program_id(0)
    h = tl.program_id(1)

    hq_r: tl.constexpr = HQ // SP
    hk_r: tl.constexpr = HK * REP // SP
    heads_per_rank: tl.constexpr = hq_r + 2 * hk_r

    t = pid_t.to(tl.int64) * BT + tl.arange(0, BT)
    c = tl.arange(0, D)
    mt = (t < T)[:, None]

    half: tl.constexpr = ROT // 2
    partner = tl.where(c < half, c + half, tl.where(c < ROT, c - half, c))
    sign = tl.where(c < half, 1.0, -1.0)

    row = G + t[:, None] * (SP * heads_per_rank * D)
    if h < HQ + HK:
        rc = (c < ROT)[None, :]
        cs = tl.load(COS + t[:, None] * ROT + c[None, :], mask=mt & rc, other=1.0).to(tl.float32)
        snp = tl.load(SIN + t[:, None] * ROT + partner[None, :], mask=mt & rc, other=0.0).to(tl.float32)
        if h < HQ:
            p = (h // hq_r) * heads_per_rank + h % hq_r
            g = tl.load(row + p * D + c[None, :], mask=mt, other=0.0).to(tl.float32)
            gp = tl.load(row + p * D + partner[None, :], mask=mt, other=0.0).to(tl.float32)
            tl.store(DQ + t[:, None] * (HQ * D) + h * D + c[None, :], g * cs + sign[None, :] * gp * snp, mask=mt)
        else:
            slot = (h - HQ) * REP
            p = (slot // hk_r) * heads_per_rank + hq_r + slot % hk_r
            g = tl.load(row + p * D + c[None, :], mask=mt, other=0.0).to(tl.float32)
            gp = tl.load(row + p * D + partner[None, :], mask=mt, other=0.0).to(tl.float32)
            dx = g * cs + sign[None, :] * gp * snp
            for copy in tl.static_range(1, REP):
                p = ((slot + copy) // hk_r) * heads_per_rank + hq_r + (slot + copy) % hk_r
                g = tl.load(row + p * D + c[None, :], mask=mt, other=0.0).to(tl.float32)
                gp = tl.load(row + p * D + partner[None, :], mask=mt, other=0.0).to(tl.float32)
                dx += g * cs + sign[None, :] * gp * snp
            tl.store(DK + t[:, None] * (HK * D) + (h - HQ) * D + c[None, :], dx, mask=mt)
    else:
        slot = (h - HQ - HK) * REP
        dx = tl.load(
            row + ((slot // hk_r) * heads_per_rank + hq_r + hk_r + slot % hk_r) * D + c[None, :], mask=mt, other=0.0
        )
        dx = dx.to(tl.float32)
        for copy in tl.static_range(1, REP):
            p = ((slot + copy) // hk_r) * heads_per_rank + hq_r + hk_r + (slot + copy) % hk_r
            dx += tl.load(row + p * D + c[None, :], mask=mt, other=0.0).to(tl.float32)
        tl.store(DV + t[:, None] * (HK * D) + (h - HQ - HK) * D + c[None, :], dx, mask=mt)


class TrainingUlyssesAttention(torch.autograd.Function):
    """A GPU backend's Ulysses causal attention on q, k and v the pack kernel rotated, rounded and packed."""

    @staticmethod
    def flash_attention_4(q, k, v, cu_seqlens, max_length, scale):
        """FlashAttention-4's forward on the packed sequences of q, k, v [T, heads, d]: (out, the log-sum-exp, ...)."""
        import flash_attn.cute.interface as fa4

        return fa4._flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
            softmax_scale=scale,
            causal=True,
            return_lse=True,
        )
