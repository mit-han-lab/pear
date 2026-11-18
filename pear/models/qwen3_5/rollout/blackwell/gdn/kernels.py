from vllm.triton_utils import tl, triton

from pear.models.qwen3_5.rollout.triton_launcher import TritonLauncher

CONV_TOKENS = 8  # tokens per conv program: vLLM's causal-conv metadata (batch_ptr, token_chunk_offset_ptr) uses 8
CONV_CHANNELS = 256  # channels per conv program
DECODE_V_ROWS = 32  # V rows of a head's recurrent state per decode program
PREP_NORM_TOKENS = 16  # tokens per program of the prep and norm kernels
STATE_ELEMENTS = 8192  # elements per program of the state gather and scatter


@triton.jit(do_not_specialize=["stride_x", "stride_sb", "stride_sc", "stride_sk"])
def _conv_kernel(
    X,
    W,
    S,
    IDX,
    HAS_INIT,
    QSL,
    PSEQ,
    PCHUNK,
    OUT,
    stride_x,
    stride_sb,
    stride_sc,
    stride_sk,
    C: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
):
    # one program: BM tokens of a sequence (its chunk-th block of BM), BC channels; a sliding window of the last 3
    # inputs in registers. The chunk-0 program alone reads the old state and writes the new one.
    pid = tl.program_id(0)
    seq = tl.load(PSEQ + pid)
    if seq < 0:
        return

    chunk = tl.load(PCHUNK + pid)
    slot = tl.load(IDX + seq).to(tl.int64)
    if slot <= 0:
        return

    s0 = tl.load(QSL + seq)
    s1 = tl.load(QSL + seq + 1)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    sbase = S + slot * stride_sb + c * stride_sc
    t0 = s0 + chunk * BM
    w0 = tl.load(W + c * 4 + 0).to(tl.float32)
    w1 = tl.load(W + c * 4 + 1).to(tl.float32)
    w2 = tl.load(W + c * 4 + 2).to(tl.float32)
    w3 = tl.load(W + c * 4 + 3).to(tl.float32)

    if chunk == 0:
        has_init = tl.load(HAS_INIT + seq).to(tl.int1)
        # the window carries X's dtype (fp32 projections: the state's bf16 values are cast to fp32)
        x0 = tl.load(sbase + 0 * stride_sk, mask=has_init & (c < C), other=0.0).to(X.dtype.element_ty)
        x1 = tl.load(sbase + 1 * stride_sk, mask=has_init & (c < C), other=0.0).to(X.dtype.element_ty)
        x2 = tl.load(sbase + 2 * stride_sk, mask=has_init & (c < C), other=0.0).to(X.dtype.element_ty)
    else:
        x0 = tl.load(X + (t0 - 3) * stride_x + c)
        x1 = tl.load(X + (t0 - 2) * stride_x + c)
        x2 = tl.load(X + (t0 - 1) * stride_x + c)
    a0 = x0.to(tl.float32)
    a1 = x1.to(tl.float32)
    a2 = x2.to(tl.float32)

    n = tl.minimum(BM, s1 - t0)
    for i in range(0, n):
        xt = tl.load(X + (t0 + i) * stride_x + c)
        a3 = xt.to(tl.float32)
        acc = w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3
        tl.store(OUT + (t0 + i) * C + c, acc * tl.sigmoid(acc))
        x0 = x1
        x1 = x2
        x2 = xt
        a0 = a1
        a1 = a2
        a2 = a3

    if chunk == 0:
        # the new state: the last 3 inputs (the window if this chunk is the last, else read from x)
        if s1 - t0 > BM:
            x0 = tl.load(X + (s1 - 3) * stride_x + c)
            x1 = tl.load(X + (s1 - 2) * stride_x + c)
            x2 = tl.load(X + (s1 - 1) * stride_x + c)
        tl.debug_barrier()
        tl.store(sbase + 0 * stride_sk, x0)
        tl.store(sbase + 1 * stride_sk, x1)
        tl.store(sbase + 2 * stride_sk, x2)


@triton.jit(do_not_specialize=["stride_x", "stride_sb", "stride_sc", "stride_sk"])
def _conv_decode_kernel(
    X, W, S, IDX, OUT, stride_x, stride_sb, stride_sc, stride_sk, C: tl.constexpr, BC: tl.constexpr
):
    # one token per sequence (pure decode steps, CUDA-graph safe: padding rows have slot 0)
    d = tl.program_id(0)
    slot = tl.load(IDX + d).to(tl.int64)
    if slot <= 0:
        return

    c = tl.program_id(1) * BC + tl.arange(0, BC)
    sbase = S + slot * stride_sb + c * stride_sc
    x0 = tl.load(sbase + 0 * stride_sk)
    x1 = tl.load(sbase + 1 * stride_sk)
    x2 = tl.load(sbase + 2 * stride_sk)
    xt = tl.load(X + d * stride_x + c)

    acc = (
        tl.load(W + c * 4 + 0).to(tl.float32) * x0.to(tl.float32)
        + tl.load(W + c * 4 + 1).to(tl.float32) * x1.to(tl.float32)
        + tl.load(W + c * 4 + 2).to(tl.float32) * x2.to(tl.float32)
        + tl.load(W + c * 4 + 3).to(tl.float32) * xt.to(tl.float32)
    )
    tl.store(OUT + d * C + c, acc * tl.sigmoid(acc))

    tl.debug_barrier()
    tl.store(sbase + 0 * stride_sk, x1)
    tl.store(sbase + 1 * stride_sk, x2)
    tl.store(sbase + 2 * stride_sk, xt)


@triton.jit
def _decay_gate(x, ALOG, hv, THRESHOLD: tl.constexpr):
    # the log decay g = -exp(A_log) softplus(x) of v head hv, for x = a + dt_bias; softplus is x above THRESHOLD
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    sp = tl.where(x <= THRESHOLD, sp, x)
    return -tl.exp(tl.load(ALOG + hv).to(tl.float32)) * sp


@triton.jit(do_not_specialize=["stride_ab", "stride_pool", "scale"])
def _decode_kernel(
    CONV,
    BA,
    ALOG,
    DTB,
    POOL,
    IDX,
    OUT,
    stride_ab,
    stride_pool,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    # one program: a decode row, a v head, BV of its V rows; o (fp32) to OUT [D, HV, V]
    d = tl.program_id(0)
    hv = tl.program_id(1)
    vb = tl.program_id(2)
    h = hv // (HV // H)
    slot = tl.load(IDX + d).to(tl.int64)
    ov = vb * BV + tl.arange(0, BV)
    if slot <= 0:  # a padding row: zeros, as vLLM's kernels write
        tl.store(OUT + (d * HV + hv) * V + ov, tl.zeros([BV], dtype=tl.float32))
        return

    ok = tl.arange(0, K)
    C: tl.constexpr = 2 * H * K + HV * V
    row = CONV + d * C
    q = tl.load(row + h * K + ok)
    k = tl.load(row + H * K + h * K + ok)
    v = tl.load(row + 2 * H * K + hv * V + ov)
    q = q / tl.sqrt(tl.sum(q * q) + EPS) * scale
    k = k / tl.sqrt(tl.sum(k * k) + EPS)

    x = tl.load(BA + d * stride_ab + HV + hv).to(tl.float32) + tl.load(DTB + hv).to(tl.float32)
    g = _decay_gate(x, ALOG, hv, THRESHOLD)
    beta = tl.sigmoid(tl.load(BA + d * stride_ab + hv).to(tl.float32))

    p = POOL + slot * stride_pool + hv * V * K + ov[:, None] * K + ok[None, :]
    s = tl.load(p) * tl.exp(g)
    u = (v - tl.sum(s * k[None, :], 1)) * beta
    s += u[:, None] * k[None, :]
    o = tl.sum(s * q[None, :], 1)
    tl.store(p, s)
    tl.store(OUT + (d * HV + hv) * V + ov, o)


@triton.jit(do_not_specialize=["P", "row0", "stride_ab"])
def _prep_kernel(
    CONV,
    BA,
    ALOG,
    DTB,
    QO,
    KO,
    VO,
    GO,
    BO,
    P,
    row0,
    stride_ab,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    EPS: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    tb = tl.program_id(0)
    head = tl.program_id(1)
    C: tl.constexpr = 2 * H * K + HV * V
    t = tb * BT + tl.arange(0, BT)
    tm = t < P
    src = CONV + (row0 + t)[:, None] * C

    if head < H:
        ok = tl.arange(0, K)
        q = tl.load(src + head * K + ok[None, :], mask=tm[:, None], other=0.0)
        k = tl.load(src + H * K + head * K + ok[None, :], mask=tm[:, None], other=0.0)
        q = q * (1.0 / tl.sqrt(tl.sum(q * q, 1) + EPS))[:, None]
        k = k * (1.0 / tl.sqrt(tl.sum(k * k, 1) + EPS))[:, None]
        dst = (t[:, None] * H + head) * K + ok[None, :]
        tl.store(QO + dst, q.to(QO.dtype.element_ty), mask=tm[:, None])
        tl.store(KO + dst, k.to(KO.dtype.element_ty), mask=tm[:, None])
    else:
        hv = head - H
        ov = tl.arange(0, V)
        v = tl.load(src + 2 * H * K + hv * V + ov[None, :], mask=tm[:, None], other=0.0)
        tl.store(VO + (t[:, None] * HV + hv) * V + ov[None, :], v.to(VO.dtype.element_ty), mask=tm[:, None])

        x = tl.load(BA + (row0 + t) * stride_ab + HV + hv, mask=tm, other=0.0).to(tl.float32) + tl.load(DTB + hv).to(
            tl.float32
        )
        g = _decay_gate(x, ALOG, hv, THRESHOLD)
        beta = tl.sigmoid(tl.load(BA + (row0 + t) * stride_ab + hv, mask=tm, other=0.0).to(tl.float32))
        tl.store(GO + t * HV + hv, tl.exp(g), mask=tm)
        tl.store(BO + t * HV + hv, beta, mask=tm)


@triton.jit(do_not_specialize=["stride_pool"])
def _gather_kernel(POOL, IDX, HAS_INIT, OUT, stride_pool, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(IDX + i).to(tl.int64)
    has_init = tl.load(HAS_INIT + i).to(tl.int1)

    x = tl.load(POOL + slot * stride_pool + offs, mask=(offs < N) & has_init, other=0.0)
    tl.store(OUT + i * N + offs, x, mask=offs < N)


@triton.jit(do_not_specialize=["stride_pool"])
def _scatter_kernel(SRC, IDX, POOL, stride_pool, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(IDX + i).to(tl.int64)
    x = tl.load(SRC + i * N + offs, mask=offs < N)
    tl.store(POOL + slot * stride_pool + offs, x, mask=offs < N)


@triton.jit(do_not_specialize=["T", "D", "stride_z", "eps"])
def _norm_kernel(
    XP, O32, Z, W, Y, T, D, stride_z, eps, ZOFF: tl.constexpr, HEADS: tl.constexpr, N: tl.constexpr, BT: tl.constexpr
):
    # y = rmsnorm(x) * w * silu(z) per (token, head): rows < D from the decode rows' fp32 outputs O32 [D, HEADS, N],
    # the others from the prefill rows' outputs XP [T - D, HEADS, N]; z at column ZOFF of the packed qkvz rows
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    cols = tl.arange(0, N)
    m = rows[:, None] < T
    off = (rows[:, None] * HEADS + head) * N + cols[None, :]

    xd = tl.load(O32 + off, mask=m & (rows[:, None] < D), other=0.0)
    xp = tl.load(
        XP + ((rows[:, None] - D) * HEADS + head) * N + cols[None, :], mask=m & (rows[:, None] >= D), other=0.0
    ).to(tl.float32)
    x = xd + xp

    rstd = tl.rsqrt(tl.sum(x * x, 1) / N + eps)
    w = tl.load(W + cols).to(tl.float32)
    z = tl.load(Z + rows[:, None] * stride_z + ZOFF + head * N + cols[None, :], mask=m, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :] * (z * tl.sigmoid(z))
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


class GDNLaunchers:
    """The Gated DeltaNet kernels, each through its compiled runner."""

    conv = TritonLauncher(_conv_kernel)
    conv_decode = TritonLauncher(_conv_decode_kernel)
    decode = TritonLauncher(_decode_kernel)
    prep = TritonLauncher(_prep_kernel)
    gather = TritonLauncher(_gather_kernel)
    scatter = TritonLauncher(_scatter_kernel)
    norm = TritonLauncher(_norm_kernel)
