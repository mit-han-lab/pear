import copy
from typing import NamedTuple, Optional

import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)

from pear.models.qwen3_5.gdn_constants import L2_NORM_EPS
from pear.models.qwen3_5.rollout.blackwell.flashinfer_launchers import FlashInferChunk
from pear.models.qwen3_5.rollout.blackwell.triton_launcher import TritonLauncher
from pear.models.qwen3_5.rollout.gdn import RolloutGDNAttention, RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims

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


class LayerConstants(NamedTuple):
    """A Gated DeltaNet layer's state pools, their strides, its weights and its sizes."""

    conv_state: torch.Tensor  # the conv state pool
    state_pool: torch.Tensor  # the recurrent state pool
    stride_sb: int  # the conv state pool's strides over blocks, channels and taps
    stride_sc: int
    stride_sk: int
    stride_pool: int  # the recurrent state pool's stride over blocks
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_eps: float
    scale: float  # of q: head_k_dim ** -0.5
    dims: GDNDims


class GDNLaunchers:
    """The Gated DeltaNet kernels, each through its compiled runner."""

    conv = TritonLauncher(_conv_kernel)
    conv_decode = TritonLauncher(_conv_decode_kernel)
    decode = TritonLauncher(_decode_kernel)
    prep = TritonLauncher(_prep_kernel)
    gather = TritonLauncher(_gather_kernel)
    scatter = TritonLauncher(_scatter_kernel)
    norm = TritonLauncher(_norm_kernel)


class SideStream:
    """A second stream for the decode rows' recurrent step of a mixed step, beside the prefill rows' kernels."""

    instance: Optional["SideStream"] = None

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.fork_event = torch.cuda.Event()
        self.join_event = torch.cuda.Event()

    @classmethod
    def of_process(cls) -> Optional["SideStream"]:
        """The process's side stream, None while a CUDA graph is captured."""
        if torch.cuda.is_current_stream_capturing():
            return None
        if cls.instance is None:
            cls.instance = cls()

        return cls.instance


class GDNScratch:
    """The buffers of one step's GDN layers, shared by its KV cache groups (their layers run one after the other)."""

    IO_DTYPE = torch.bfloat16  # the q, k, v and output of FlashInfer's chunk kernel
    key = None
    instance = None

    @classmethod
    def of(cls, T: int, D: int, num_prefills: int, dims: GDNDims, device: torch.device) -> "GDNScratch":
        if cls.key != (T, D, num_prefills, dims):
            P = T - D
            scratch = cls()
            scratch.conv_out = torch.empty(T, dims.C, dtype=torch.float32, device=device)
            scratch.o32 = torch.empty(D, dims.HV, dims.V, dtype=torch.float32, device=device)
            scratch.q = torch.empty(P, dims.H, dims.K, dtype=cls.IO_DTYPE, device=device)
            scratch.k = torch.empty(P, dims.H, dims.K, dtype=cls.IO_DTYPE, device=device)
            scratch.v = torch.empty(P, dims.HV, dims.V, dtype=cls.IO_DTYPE, device=device)
            scratch.g = torch.empty(P, dims.HV, dtype=torch.float32, device=device)
            scratch.beta = torch.empty(P, dims.HV, dtype=torch.float32, device=device)
            scratch.prefill_out = torch.empty(P, dims.HV, dims.V, dtype=cls.IO_DTYPE, device=device)
            scratch.s_in = torch.empty(num_prefills, dims.HV, dims.V, dims.K, dtype=torch.float32, device=device)
            scratch.s_out = torch.empty(num_prefills, dims.HV, dims.V, dims.K, dtype=torch.float32, device=device)
            cls.key, cls.instance = (T, D, num_prefills, dims), scratch

        return cls.instance


class GDNLayout:
    """A step's Gated DeltaNet layout for GDNCore.mixed_step in a KV cache group: rows, state slots, scratch, grids."""

    def __init__(self, md: GDNAttentionMetadata):
        self.T = md.num_actual_tokens
        self.D = md.num_decode_tokens
        self.P = self.T - self.D
        self.num_prefills = md.num_prefills

        self.conv_programs = md.nums_dict[CONV_TOKENS]["tot"]
        self.batch_ptr = md.batch_ptr
        self.chunk_ptr = md.token_chunk_offset_ptr

        self.qsl = md.non_spec_query_start_loc
        self.idx = md.non_spec_state_indices_tensor.contiguous()  # a block-table column
        self.has_initial_state = md.has_initial_state

        self.pf_qsl = md.prefill_query_start_loc
        self.pf_idx = self.idx[self.D :]
        self.pf_has_initial_state = md.prefill_has_initial_state

        self.dims = None

    def bind(self, dims: GDNDims, device: torch.device) -> None:
        """The scratch and the launch grids, at the step's first layer (every GDN layer has the same sizes)."""
        self.dims = dims
        self.scratch = GDNScratch.of(self.T, self.D, self.num_prefills, dims, device)

        self.grid_conv = (self.conv_programs, dims.C // CONV_CHANNELS)
        self.grid_decode = (self.D, dims.HV, dims.V // DECODE_V_ROWS)
        self.grid_prep = (triton.cdiv(self.P, PREP_NORM_TOKENS), dims.H + dims.HV)
        self.grid_state = (self.num_prefills, triton.cdiv(dims.state_size, STATE_ELEMENTS))
        self.grid_norm = (triton.cdiv(self.T, PREP_NORM_TOKENS), dims.HV)


class StagedBuffers:
    """Two pinned host buffers and their device copies for a step's small int32 arrays, used in turn."""

    SLOTS = 1 << 16  # int32 slots of a buffer: one step's GDN metadata
    BUFFERS = 2  # one in flight, one being written
    instance: Optional["StagedBuffers"] = None

    def __init__(self, device: torch.device):
        # vLLM's input preparation waits for the previous step's copies before this step's, so two buffers suffice
        self.host_buffers = [torch.empty(self.SLOTS, dtype=torch.int32, pin_memory=True) for _ in range(self.BUFFERS)]
        self.device_buffers = [torch.empty(self.SLOTS, dtype=torch.int32, device=device) for _ in range(self.BUFFERS)]
        self.turn = 0

    @classmethod
    def next_buffers(cls, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if cls.instance is None:
            cls.instance = cls(device)
        staged_buffers = cls.instance
        staged_buffers.turn ^= 1

        return staged_buffers.host_buffers[staged_buffers.turn], staged_buffers.device_buffers[staged_buffers.turn]


class MixedStepMetadata:
    """The GDN metadata of a step with prefill rows from the host-side batch layout: numpy and one copy to the GPU."""

    @staticmethod
    def build(builder: RolloutGDNMetadataBuilder, m) -> GDNAttentionMetadata | None:
        """What GDNCore.mixed_step reads, None for a step without prefill rows (vLLM builds it: the decode kernels'
        FULL-graph buffers)."""
        # vLLM builds the conv kernel's program map with Python loops and fresh pinned tensors, the initial-state flags
        # with small device ops
        num_reqs = m.num_reqs
        D = builder.num_decodes(m)
        if D == num_reqs:
            return None

        qsl = m.query_start_loc_cpu.numpy()[: num_reqs + 1]
        query_lens = np.diff(qsl)
        num_prefills = num_reqs - D
        chunks = -(-query_lens[D:] // CONV_TOKENS)
        num_programs = D + int(chunks.sum())
        host_buffer, device_buffer = StagedBuffers.next_buffers(m.query_start_loc.device)

        # one int32 array: the requests' initial-state flags | the prefill rows' query starts | conv program ->
        # sequence | conv program -> chunk of its sequence (a conv program per decode row and per CONV_TOKENS tokens,
        # begun, of a prefill row)
        query_starts_offset = num_reqs
        batch_ptr_offset = query_starts_offset + num_prefills + 1
        chunk_ptr_offset = batch_ptr_offset + num_programs
        size = chunk_ptr_offset + num_programs
        if size > host_buffer.numel():
            return None

        host_array = host_buffer.numpy()
        host_array[:num_reqs] = m.seq_lens_cpu_upper_bound.numpy()[:num_reqs] > query_lens
        host_array[query_starts_offset:batch_ptr_offset] = qsl[D:] - qsl[D]
        host_array[batch_ptr_offset : batch_ptr_offset + D] = np.arange(D)
        host_array[batch_ptr_offset + D : chunk_ptr_offset] = np.repeat(np.arange(D, num_reqs), chunks)
        host_array[chunk_ptr_offset : chunk_ptr_offset + D] = 0
        first_programs = np.repeat(np.cumsum(chunks) - chunks, chunks)  # each prefill row's first program, per program
        host_array[chunk_ptr_offset + D : size] = np.arange(num_programs - D) - first_programs
        device_array = device_buffer[:size]
        device_array.copy_(host_buffer[:size], non_blocking=True)

        idx = builder.state_slots(m.block_table_tensor, num_reqs).contiguous()
        has_initial_state = device_array[:num_reqs]
        T = int(qsl[num_reqs])

        md = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=T - D,
            num_decodes=D,
            num_decode_tokens=D,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=T,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=m.query_start_loc,
            non_spec_state_indices_tensor=idx,
            prefill_query_start_loc=device_array[query_starts_offset:batch_ptr_offset],
            prefill_state_indices=idx[D:],
            prefill_has_initial_state=has_initial_state[D:],
            nums_dict={CONV_TOKENS: {"tot": num_programs}},
            batch_ptr=device_array[batch_ptr_offset:chunk_ptr_offset],
            token_chunk_offset_ptr=device_array[chunk_ptr_offset:size],
        )
        md._pear_layout = GDNLayout(md)

        return md


class GDNCore:
    """A Gated DeltaNet layer's core on own kernels launched directly, with fp32 intermediates."""

    CONV_WARPS = 4  # warps per conv program
    DECODE_WARPS = 1  # warps per decode program
    SOFTPLUS_THRESHOLD = 20.0  # softplus is the identity above it, as torch's
    chunk = FlashInferChunk()

    @classmethod
    def mixed_step(cls, layer, mixed_qkvz, ba, core_attn_out, layout: GDNLayout) -> None:
        """A step with prefill rows: conv, decode rows' recurrent step, prefill rows' chunked delta rule, norm."""
        constants = cls.constants(layer)
        dims = constants.dims
        if layout.dims is None:
            layout.bind(dims, mixed_qkvz.device)
        scratch = layout.scratch
        T, D, P = layout.T, layout.D, layout.P
        stride_x = mixed_qkvz.stride(0)
        stride_ab = ba.stride(0)

        GDNLaunchers.conv(
            (mixed_qkvz.dtype, dims),
            layout.grid_conv,
            mixed_qkvz,
            constants.conv_weight,
            constants.conv_state,
            layout.idx,
            layout.has_initial_state,
            layout.qsl,
            layout.batch_ptr,
            layout.chunk_ptr,
            scratch.conv_out,
            stride_x,
            constants.stride_sb,
            constants.stride_sc,
            constants.stride_sk,
            C=dims.C,
            BM=CONV_TOKENS,
            BC=CONV_CHANNELS,
            num_warps=cls.CONV_WARPS,
        )

        # the decode rows' recurrent step runs on the side stream beside the prefill rows' kernels (the chunked kernel
        # fills a few SMs: one sequence's heads); their states and outputs are disjoint
        side_stream = SideStream.of_process() if P > 0 and D > 0 else None
        if side_stream is not None:
            # the side stream may start after the work queued on the current stream so far
            main_stream = torch.cuda.current_stream()
            side_stream.fork_event.record(main_stream)

        if P > 0:
            GDNLaunchers.prep(
                (ba.dtype, dims),
                layout.grid_prep,
                scratch.conv_out,
                ba,
                constants.a_log,
                constants.dt_bias,
                scratch.q,
                scratch.k,
                scratch.v,
                scratch.g,
                scratch.beta,
                P,
                D,
                stride_ab,
                H=dims.H,
                HV=dims.HV,
                K=dims.K,
                V=dims.V,
                BT=PREP_NORM_TOKENS,
                EPS=L2_NORM_EPS,
                THRESHOLD=cls.SOFTPLUS_THRESHOLD,
            )
            GDNLaunchers.gather(
                (dims,),
                layout.grid_state,
                constants.state_pool,
                layout.pf_idx,
                layout.pf_has_initial_state,
                scratch.s_in,
                constants.stride_pool,
                N=dims.state_size,
                BLOCK=STATE_ELEMENTS,
            )
            cls.chunk(
                scratch.q,
                scratch.k,
                scratch.v,
                scratch.g,
                scratch.beta,
                scratch.prefill_out,
                layout.pf_qsl,
                scratch.s_in,
                scratch.s_out,
                constants.scale,
            )
            GDNLaunchers.scatter(
                (dims,),
                layout.grid_state,
                scratch.s_out,
                layout.pf_idx,
                constants.state_pool,
                constants.stride_pool,
                N=dims.state_size,
                BLOCK=STATE_ELEMENTS,
            )

        if D > 0:
            if side_stream is not None:
                side_stream.stream.wait_event(side_stream.fork_event)
                torch.cuda.set_stream(side_stream.stream)
            GDNLaunchers.decode(
                (ba.dtype, dims),
                layout.grid_decode,
                scratch.conv_out,
                ba,
                constants.a_log,
                constants.dt_bias,
                constants.state_pool,
                layout.idx,
                scratch.o32,
                stride_ab,
                constants.stride_pool,
                constants.scale,
                H=dims.H,
                HV=dims.HV,
                K=dims.K,
                V=dims.V,
                BV=DECODE_V_ROWS,
                EPS=L2_NORM_EPS,
                THRESHOLD=cls.SOFTPLUS_THRESHOLD,
                num_warps=cls.DECODE_WARPS,
            )
            if side_stream is not None:
                # back to the main stream, which continues after the side stream's work
                side_stream.join_event.record(side_stream.stream)
                torch.cuda.set_stream(main_stream)
                main_stream.wait_event(side_stream.join_event)

        GDNLaunchers.norm(
            (mixed_qkvz.dtype, dims),
            layout.grid_norm,
            scratch.prefill_out,
            scratch.o32,
            mixed_qkvz,
            constants.norm_weight,
            core_attn_out,
            T,
            D,
            stride_x,
            constants.norm_eps,
            ZOFF=dims.C,
            HEADS=dims.HV,
            N=dims.V,
            BT=PREP_NORM_TOKENS,
        )

    @classmethod
    def decode_step(cls, layer, mixed_qkvz, ba, core_attn_out, md: GDNAttentionMetadata) -> None:
        """A pure decode step, also in FULL graphs: every buffer from the graph's pool, padding rows in slot 0."""
        constants = cls.constants(layer)
        dims = constants.dims
        D = md.num_actual_tokens
        idx = md.non_spec_state_indices_tensor
        conv_out = torch.empty(D, dims.C, dtype=torch.float32, device=mixed_qkvz.device)
        o32 = torch.empty(D, dims.HV, dims.V, dtype=torch.float32, device=mixed_qkvz.device)
        stride_x = mixed_qkvz.stride(0)

        GDNLaunchers.conv_decode(
            (mixed_qkvz.dtype, dims),
            (D, dims.C // CONV_CHANNELS),
            mixed_qkvz,
            constants.conv_weight,
            constants.conv_state,
            idx,
            conv_out,
            stride_x,
            constants.stride_sb,
            constants.stride_sc,
            constants.stride_sk,
            C=dims.C,
            BC=CONV_CHANNELS,
        )

        GDNLaunchers.decode(
            (ba.dtype, dims),
            (D, dims.HV, dims.V // DECODE_V_ROWS),
            conv_out,
            ba,
            constants.a_log,
            constants.dt_bias,
            constants.state_pool,
            idx,
            o32,
            ba.stride(0),
            constants.stride_pool,
            constants.scale,
            H=dims.H,
            HV=dims.HV,
            K=dims.K,
            V=dims.V,
            BV=DECODE_V_ROWS,
            EPS=L2_NORM_EPS,
            THRESHOLD=cls.SOFTPLUS_THRESHOLD,
            num_warps=cls.DECODE_WARPS,
        )

        GDNLaunchers.norm(
            ("decode", mixed_qkvz.dtype, dims),
            (triton.cdiv(D, PREP_NORM_TOKENS), dims.HV),
            o32,
            o32,
            mixed_qkvz,
            constants.norm_weight,
            core_attn_out,
            D,
            D,
            stride_x,
            constants.norm_eps,
            ZOFF=dims.C,
            HEADS=dims.HV,
            N=dims.V,
            BT=PREP_NORM_TOKENS,
        )

    @staticmethod
    def constants(layer) -> LayerConstants:
        """The layer's state pools, strides, weights and sizes, kept until vLLM binds or releases its KV cache."""
        constants = layer.__dict__.get("_pear_layer_constants")
        if constants is None:
            conv_state, state_pool = layer.kv_cache[0], layer.kv_cache[1]
            if is_conv_state_dim_first():  # [blocks, C, 3]
                stride_sb, stride_sc, stride_sk = conv_state.stride()
            else:  # [blocks, 3, C]
                stride_sb, stride_sk, stride_sc = conv_state.stride()
            # the weights by reference: a weight sync updates them in place
            constants = layer._pear_layer_constants = LayerConstants(
                conv_state=conv_state,
                state_pool=state_pool,
                stride_sb=stride_sb,
                stride_sc=stride_sc,
                stride_sk=stride_sk,
                stride_pool=state_pool.stride(0),
                conv_weight=layer.conv1d.weight,
                a_log=layer.A_log,
                dt_bias=layer.dt_bias,
                norm_weight=layer.norm.weight,
                norm_eps=layer.norm.eps,
                scale=layer.head_k_dim**-0.5,
                dims=GDNDims.of_layer(layer),
            )

        return constants


class BlackwellGDNAttention(RolloutGDNAttention):
    """vLLM's Qwen Gated DeltaNet layer on GDNCore (registered out of tree in its place)."""

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
        if not self.computes_core(hidden_states):
            return super().forward_cuda(hidden_states)

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
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


class BlackwellGDNMetadataBuilder(RolloutGDNMetadataBuilder):
    """vLLM's GDN metadata builder with a mixed step's metadata from MixedStepMetadata, shared by the GDN layers'
    groups; the other steps' as vLLM builds them (the decode steps' FULL-graph buffers are per builder)."""

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        # vLLM's V2 runner hands every GDN build the draft-token arguments, None without a drafter; a FULL graph's
        # capture passes them for its decode batch
        if num_decode_draft_tokens_cpu is None:
            md = MixedStepMetadata.build(self, common_attn_metadata)
            if md is not None:
                return md

        return super().build(
            common_prefix_len, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        layout = metadata.__dict__.get("_pear_layout")
        if layout is None:
            return super().update_block_table(metadata, blk_table, slot_mapping)

        # a mixed step: the first group's metadata and layout with this group's state slots
        idx = self.state_slots(blk_table, metadata.num_decodes + metadata.num_prefills).contiguous()
        group_metadata = copy.copy(metadata)
        group_metadata.non_spec_state_indices_tensor = idx
        group_metadata.prefill_state_indices = idx[metadata.num_decodes :]
        group_layout = copy.copy(layout)
        group_layout.idx = idx
        group_layout.pf_idx = idx[layout.D :]
        group_metadata._pear_layout = group_layout

        return group_metadata


class BlackwellGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with BlackwellGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return BlackwellGDNMetadataBuilder
