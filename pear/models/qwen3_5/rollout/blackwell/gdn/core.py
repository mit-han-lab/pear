from typing import NamedTuple, Optional

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from pear.models.qwen3_5.gdn_constants import L2_NORM_EPS
from pear.models.qwen3_5.rollout.blackwell.flashinfer_launchers import FlashInferChunk
from pear.models.qwen3_5.rollout.blackwell.gdn.kernels import (
    CONV_CHANNELS,
    CONV_TOKENS,
    DECODE_V_ROWS,
    PREP_NORM_TOKENS,
    STATE_ELEMENTS,
    GDNLaunchers,
)
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims


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
