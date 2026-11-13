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

    kv_cache: list  # the layer's KV cache the pools were read from
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


class FlashInferChunk:
    """FlashInfer's Blackwell chunked GDN kernel on packed states, called through its compiled function."""

    def __init__(self):
        self.compiled_kernel = None
        self.workspace = None
        self.cuda = None

    def __call__(self, q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale) -> None:
        """One call of the chunked kernel over the prefill rows."""
        if self.compiled_kernel is None:
            # the first call goes through FlashInfer's wrapper, which compiles
            from flashinfer.gdn_kernels.blackwell import gdn_prefill as fi

            fi.chunk_gated_delta_rule_sm100(q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale)

            # FlashInfer's key of this call: GQA when there are not fewer q heads than v heads; initial and final
            # states given, no checkpoints, packed states (no state indices)
            heads, v_heads = q.shape[1], v.shape[1]
            cache = fi._get_compiled_cache(
                str(q.dtype), str(torch.float32), heads, v_heads, heads >= v_heads, True, True, False, False
            )
            self.compiled_kernel = cache["compiled"]
            # the persistent kernel's TMA-descriptor workspace is sized per SM, so one serves any number of sequences
            self.workspace = cache[f"workspace_{q.device.index}"]
            self.cuda = fi.cuda
            return

        stream = self.cuda.CUstream(torch._C._cuda_getCurrentRawStream(q.device.index))
        self.compiled_kernel(
            q, k, v, g, beta, out, cu_seqlens, s_in, s_out, None, None, None, 0, scale, self.workspace, stream
        )


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
    """The GDN metadata of a step with prefill tokens from the host-side batch layout: numpy and one copy to the GPU."""

    @staticmethod
    def build(builder, m) -> GDNAttentionMetadata | None:
        """What GDNCore.mixed_step reads, None when the step does not qualify (vLLM then builds it)."""
        # vLLM builds the conv kernel's program map with Python loops and fresh pinned tensors, the state indices and
        # initial-state flags with small device ops
        num_computed, seq_lens = m._num_computed_tokens_cpu, m._seq_lens_cpu
        if num_computed is None or seq_lens is None:
            return None

        num_reqs = m.num_reqs
        qsl = m.query_start_loc_cpu.numpy()[: num_reqs + 1]
        T = int(qsl[num_reqs])
        if T != m.num_actual_tokens:
            return None

        query_lens = np.diff(qsl)
        is_prefill = query_lens > 1
        if not is_prefill.any():
            return None
        D = int(np.argmax(is_prefill))  # the batch is ordered: decodes (1 token) first
        if (query_lens[D:] < 1).any() or (query_lens[:D] != 1).any():
            return None

        num_prefills = num_reqs - D
        chunks = -(-query_lens[D:] // CONV_TOKENS)
        num_programs = D + int(chunks.sum())
        state_columns = np.maximum((seq_lens.numpy()[:num_reqs] - 1) // builder.kv_cache_spec.block_size, 0)
        initial_state_flags = num_computed.numpy()[:num_reqs] > 0
        host_buffer, device_buffer = StagedBuffers.next_buffers(m.query_start_loc.device)
        num_slots = 3 * num_reqs + 1 + 2 * num_programs
        if num_slots > host_buffer.numel():
            return None

        # one int32 array: block-table columns | initial-state flags | prefill query starts | conv program -> sequence |
        # conv program -> chunk of its sequence
        host_array = host_buffer.numpy()
        host_array[0:num_reqs] = state_columns
        host_array[num_reqs : 2 * num_reqs] = initial_state_flags
        host_array[2 * num_reqs : 3 * num_reqs - D + 1] = qsl[D:] - D
        batch_ptr_start = 3 * num_reqs - D + 1
        host_array[batch_ptr_start : batch_ptr_start + D] = np.arange(D)
        host_array[batch_ptr_start + D : batch_ptr_start + num_programs] = np.repeat(np.arange(D, num_reqs), chunks)
        chunk_ptr_start = batch_ptr_start + num_programs
        host_array[chunk_ptr_start : chunk_ptr_start + D] = 0
        host_array[chunk_ptr_start + D : chunk_ptr_start + num_programs] = np.arange(num_programs - D) - np.repeat(
            np.cumsum(chunks) - chunks, chunks
        )
        total = chunk_ptr_start + num_programs
        device_array = device_buffer[:total]
        device_array.copy_(host_buffer[:total], non_blocking=True)

        column_index = device_array[0:num_reqs].long().unsqueeze(1)
        idx = torch.gather(m.block_table_tensor[:num_reqs], 1, column_index)[:, 0]
        has_initial_state = device_array[num_reqs : 2 * num_reqs]

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
            prefill_query_start_loc=device_array[2 * num_reqs : 3 * num_reqs - D + 1],
            prefill_state_indices=idx[D:],
            prefill_has_initial_state=has_initial_state[D:],
            nums_dict={CONV_TOKENS: {"tot": num_programs}},
            batch_ptr=device_array[batch_ptr_start : batch_ptr_start + num_programs],
            token_chunk_offset_ptr=device_array[chunk_ptr_start : chunk_ptr_start + num_programs],
        )
        md._pear_common_attn_metadata = m
        md._pear_state_columns = column_index
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
        """The layer's state pools, strides, weights and sizes, kept while its KV cache stays the same."""
        constants = layer.__dict__.get("_pear_layer_constants")
        if constants is None or constants.kv_cache is not layer.kv_cache:
            conv_state, state_pool = layer.kv_cache[0], layer.kv_cache[1]
            if is_conv_state_dim_first():  # [blocks, C, 3]
                stride_sb, stride_sc, stride_sk = conv_state.stride()
            else:  # [blocks, 3, C]
                stride_sb, stride_sk, stride_sc = conv_state.stride()
            # the weights by reference: a weight sync updates them in place
            constants = layer._pear_layer_constants = LayerConstants(
                kv_cache=layer.kv_cache,
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

    def forward_cuda(self, hidden_states):
        # vLLM's fused path zero-fills the core output of every layer; GDNCore writes every row it computes (a
        # piecewise-graph batch's padding rows are never read back), so it starts uninitialized
        if not (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        ):
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
        attn_metadata = get_forward_context().attn_metadata
        if type(attn_metadata) is dict:
            md = attn_metadata[self.prefix]
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
    groups."""

    # The GDN layers' KV cache groups differ only in their block tables: in a mixed step the runner builds the first
    # group's metadata and derives the others' from it (update_block_table), swapping in the group's state indices
    # (one gather). Other steps (pure decode, whose FULL-graph buffers are per builder) are built in full.
    supports_update_block_table = True

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        if not self.use_spec_decode and not args and not kwargs:
            md = MixedStepMetadata.build(self, common_attn_metadata)
            if md is not None:
                return md

        md = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)
        md._pear_common_attn_metadata = common_attn_metadata
        if md.num_prefills > 0 and md.spec_sequence_masks is None and md.nums_dict is not None:
            md._pear_layout = GDNLayout(md)
            # the state block of every request: its block-table column (as mamba_get_block_table_tensor gathers it)
            md._pear_state_columns = (
                ((common_attn_metadata.seq_lens - 1) // self.kv_cache_spec.block_size)
                .clamp_(min=0)
                .to(torch.int64)
                .unsqueeze(1)
            )

        return md

    def update_block_table(self, metadata, blk_table, slot_mapping):
        layout = metadata.__dict__.get("_pear_layout")
        if layout is None:
            return self.build(
                0, self.group_common_metadata(metadata._pear_common_attn_metadata, blk_table, slot_mapping)
            )

        idx = torch.gather(blk_table, 1, metadata._pear_state_columns)[:, 0]
        group_metadata = copy.copy(metadata)
        group_metadata.non_spec_state_indices_tensor = idx
        group_metadata.prefill_state_indices = idx[metadata.num_decodes :]
        group_layout = copy.copy(layout)
        group_layout.idx = idx
        group_layout.pf_idx = idx[layout.D :]
        group_metadata._pear_layout = group_layout

        return group_metadata

    def _build_chunk_metadata(self, *args, **kwargs):
        # the FLA chunk metadata (built on the host, copied every step) feeds only the Triton / CuTe-DSL prefill
        # backends; FlashInfer's and GDNCore never read it
        if self.gdn_prefill_backend == "flashinfer":
            return None, None

        return super()._build_chunk_metadata(*args, **kwargs)


class BlackwellGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with BlackwellGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return BlackwellGDNMetadataBuilder
