from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.schedule import (
    INSTR,
    K_NORM,
    NO_CTR,
    NT,
    SS_HOST,
    Kind,
    Op,
    Pro,
    Schedule,
)
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.draft import DraftStep

logger = get_logger(__name__, default_level="INFO")

# the step record (csrc/step.cuh StepRecWord): int32 words; its header, the words before REC_HDR, goes with the launch
REC_HDR = 144  # (words 112.. the draft's: draft.py SR_DR)
SR_FLAGS, SR_TOK_PTR, SR_POS, SR_MPOS, SR_RNG, SR_SEED, SR_NTOK = 0, 2, 4, 5, 10, 12, 15
SR_GDNT, SR_POS_PTR, SR_ATT_BT, SR_SS, SR_BK = 32, 80, 82, 96, 108
# SR_FLAGS: the input tokens are the int32s at SR_TOK_PTR, the step's first position the int32 at SR_POS_PTR
TOKEN_ON_DEVICE, POS_ON_DEVICE = 1, 2
MAX_ATT_GROUPS = (SR_SS - SR_ATT_BT) // 2  # the attention groups' block-table rows the header holds (uint64 each)
# OP_GDNT's step words at SR_GDNT (csrc/gdn_t.cuh GdnStepWord)
GS_T, GS_FIRST, GS_PARITY, GS_BLOCK, GS_CONV_OFF, GS_DIRECT = 0, 1, 2, 3, 4, 5
GS_C_PTR, GS_POS_PTR, GS_C_HOST, GS_BT = 6, 8, 10, 16
GS_MAX_GROUPS = 16  # gdn_t.cuh: the mamba groups' block-table rows the step words hold
GS_WORDS = GS_BT + 2 * GS_MAX_GROUPS  # the step words, the mamba groups' block-table rows last
MROPE_SECTIONS = (11, 11, 10)  # Qwen3.5's interleaved M-RoPE (t, h, w) over the first ROTARY_DIM dims
ROTARY_DIM = 64  # the rotary dims of a head
GDN_H, GDN_HV, GDN_V, GDN_K = 16, 48, 128, 128  # Gated DeltaNet key heads, value heads, their dims
GDN_CONV = 2 * GDN_H * GDN_K + GDN_HV * GDN_V  # the conv's channels: q, k, v
GT_SEQ, GT_MAXT = 11, 8  # gdn_t.cuh: a record's entries per channel (the window and the block), tokens at most
GTR_SEQ = 256  # gdn_t.cuh GTR_SEQ: a record's conv inputs (bf16 [GT_SEQ][GDN_CONV]) after its header (bytes)
GTR_AB = GTR_SEQ + GT_SEQ * GDN_CONV * 2  # gdn_t.cuh GTR_AB: then the block's b and a rows (bf16 [GT_MAXT][2 GDN_HV])
GDNT_REC_BYTES = (GTR_AB + GT_MAXT * 2 * GDN_HV * 2 + 255) // 256 * 256  # gdn_t.cuh GTR_BYTES: one record copy
GT_IN_BF16, GT_BA_OP, GT_GNORM = 2, 2, 4  # gdn_t.cuh GdnInMode, GdnBaMode and GdnTDesc::flags
GDNT = struct.Struct("<13Q2f12I4Q2f2I48x")  # GdnTDesc (csrc/gdn_t.cuh)
# the spec sampler's step words at SR_SS (csrc/spec_sample.cuh SsStepWord) and the device addresses they take
SS_DRAFT, SS_LIDX, SS_SEED, SS_IDS, SS_POSV, SS_SLOT, SS_TOP_P = 0, 2, 4, 6, 8, 10, 11
SS_INPUTS = {SS_DRAFT: "draft", SS_LIDX: "logits_indices", SS_SEED: "seed", SS_IDS: "input_ids", SS_POSV: "positions"}
RING = 4  # the spec sampler's output slots in flight
# the record's SR_BK words (csrc/step.cuh BookWord): OP_BOOK's request row and switch, OP_PREV's previous step and flag
BK_REQ, BK_ON, BK_PREV, BK_SEQ = 0, 1, 2, 3
PREV_OUT, PREV_POST = 1, 2  # BK_PREV's bits: the previous launched step's outputs into the host ring, its post_update
PREV_SLOT_SHIFT, PREV_T_SHIFT, PREV_REQ_SHIFT = 2, 4, 8  # BK_PREV's fields: its ring slot, rows, request row
NO_TRACE = 0  # the kernel's trace buffer: none


@dataclass
class Fp8Linear:
    """A per-tensor FP8 linear (ModelOpt): weight [N, K] e4m3 as bytes, the input scale, the product of the scales."""

    weight: torch.Tensor
    input_scale: float
    alpha: float

    @classmethod
    def from_vllm(cls, linear: torch.nn.Module) -> Fp8Linear:
        """vLLM's linear (its weight stored as the transposed view)."""
        weight = linear.weight.t()
        assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous(), "per-tensor FP8 weight [N, K] expected"
        assert linear.input_scale.numel() == 1 and linear.weight_scale.numel() == 1
        input_scale, weight_scale = float(linear.input_scale), float(linear.weight_scale)

        return cls(weight=weight.view(torch.uint8), input_scale=input_scale, alpha=input_scale * weight_scale)


@dataclass
class Nvfp4Linear:
    """An NVFP4 linear (ModelOpt): packed e2m1 weight [N, K / 2], swizzled e4m3 block scales, the global scales."""

    weight: torch.Tensor
    block_scales: torch.Tensor
    alpha: float
    input_global_scale_inv: float

    @classmethod
    def from_vllm(cls, linear: torch.nn.Module) -> Nvfp4Linear:
        """vLLM's ModelOpt NVFP4 linear."""
        assert linear.weight.dtype == torch.uint8 and linear.weight.is_contiguous()
        assert not getattr(linear, "weights_padding_cols", 0)

        return cls(
            weight=linear.weight,
            block_scales=linear.weight_scale.view(torch.uint8),
            alpha=float(linear.alpha),
            input_global_scale_inv=float(linear.input_global_scale_inv),
        )


@dataclass
class Norm:
    """An RMSNorm's weight and epsilon."""

    weight: torch.Tensor
    eps: float


@dataclass
class GDNCore:
    """A Gated DeltaNet layer's core: its projections, conv, gated norm and vLLM's state caches."""

    in_proj: Fp8Linear  # q, k, v, z rows
    ba_weight: torch.Tensor  # in_proj_ba, bf16 [2 * heads, K]
    conv_weight: torch.Tensor  # [conv_dim, 4]
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: Norm
    out_proj: Fp8Linear
    conv_state: torch.Tensor  # [slots][state_len][conv_dim] bf16
    ssm_state: torch.Tensor  # [slots][heads][v][k] fp32
    group: int
    scale: float


@dataclass
class AttentionCore:
    """A full-attention layer's core: its projections, q / k norms, rotary table and vLLM's FP8 KV cache."""

    qkv: Fp8Linear
    o_proj: Fp8Linear
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    eps: float
    cos_sin: torch.Tensor  # bf16 [positions][64]
    scale: float
    kv_cache: torch.Tensor  # the FP8 cache's bytes: [rows][4 heads][k 256 | v 256]
    group: int


@dataclass
class Layer:
    """A decoder layer: its core (GDNCore or AttentionCore), norms and MLP."""

    core: GDNCore | AttentionCore
    in_norm: Norm
    post_norm: Norm
    gate_up: Nvfp4Linear
    down: Nvfp4Linear


@dataclass
class ModelView:
    """What the step reads of the model and its caches."""

    layers: list[Layer]
    embed: torch.Tensor
    final_norm: Norm
    lm_head: Nvfp4Linear
    block_size: int
    # per KV cache group the V2 runner's device block table [requests][blocks] int32
    block_tables: list[torch.Tensor] | None = None
    num_computed: torch.Tensor | None = None  # the V2 runner's computed tokens per request (int32, device)
    aux_layers: tuple[int, ...] = ()
    # the V2 runner's tensors OP_BOOK reads and writes (book_tensors_of), None: no OP_BOOK / OP_PREV
    book_tensors: dict | None = None
    # the request states OP_PREV's part of V2's post_update writes (post_tensors_of)
    post_tensors: dict | None = None

    @classmethod
    def from_runner(cls, runner: GPUModelRunner, has_book_op: bool = False) -> ModelView:
        """vLLM's V2 model runner's model and caches (FP8 KV, align-mode state slots), with OP_BOOK / OP_PREV's."""
        model = runner.get_model()
        language_model = model.language_model
        decoder = language_model.model
        cache_config = runner.cache_config
        assert cache_config.mamba_cache_mode == "align" and cache_config.enable_prefix_caching, (
            "the state slots of align mode"
        )
        kv_cache_config = runner.kv_cache_config
        context = runner.compilation_config.static_forward_context
        group_of = {
            name: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }

        layers = []
        for layer in decoder.layers:
            if layer.layer_type == "full_attention":
                attention = layer.self_attn
                rotary = attention.rotary_emb
                assert getattr(rotary, "mrope_interleaved", False) and tuple(rotary.mrope_section) == MROPE_SECTIONS
                assert rotary.rotary_dim == ROTARY_DIM and rotary.is_neox_style
                name = attention.attn.layer_name
                kv_cache = context[name].kv_cache
                core = AttentionCore(
                    qkv=Fp8Linear.from_vllm(attention.qkv_proj),
                    o_proj=Fp8Linear.from_vllm(attention.o_proj),
                    q_norm=attention.q_norm.weight,
                    k_norm=attention.k_norm.weight,
                    eps=attention.q_norm.variance_epsilon,
                    cos_sin=rotary.cos_sin_cache.to(torch.bfloat16).contiguous(),
                    scale=attention.scaling,
                    kv_cache=kv_cache,
                    group=group_of[name],
                )
            else:
                gdn = layer.linear_attn
                conv_state, ssm_state = context[gdn.prefix].kv_cache[:2]
                conv_dim = gdn.conv1d.weight.shape[0]
                assert gdn.A_log.dtype == torch.float32 and gdn.dt_bias.dtype == torch.bfloat16
                assert gdn.in_proj_ba.weight.dtype == torch.bfloat16, "in_proj_ba stays bf16 (ModelOpt's recipe)"
                core = GDNCore(
                    in_proj=Fp8Linear.from_vllm(gdn.in_proj_qkvz),
                    ba_weight=gdn.in_proj_ba.weight.contiguous(),
                    conv_weight=gdn.conv1d.weight.view(conv_dim, -1).contiguous(),
                    a_log=gdn.A_log,
                    dt_bias=gdn.dt_bias,
                    norm=Norm(gdn.norm.weight, gdn.norm.eps),
                    out_proj=Fp8Linear.from_vllm(gdn.out_proj),
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    group=group_of[gdn.prefix],
                    scale=gdn.head_k_dim**-0.5,
                )
            layers.append(
                Layer(
                    core=core,
                    in_norm=Norm(layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon),
                    post_norm=Norm(
                        layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon
                    ),
                    gate_up=Nvfp4Linear.from_vllm(layer.mlp.gate_up_proj),
                    down=Nvfp4Linear.from_vllm(layer.mlp.down_proj),
                )
            )

        groups = {layer.core.group for layer in layers if isinstance(layer.core, AttentionCore)}
        block_size = kv_cache_config.kv_cache_groups[next(iter(groups))].kv_cache_spec.block_size

        return cls(
            layers=layers,
            embed=decoder.embed_tokens.weight,
            final_norm=Norm(decoder.norm.weight, decoder.norm.variance_epsilon),
            lm_head=Nvfp4Linear.from_vllm(language_model.lm_head),
            block_size=block_size,
            block_tables=[block_table.gpu for block_table in runner.block_tables.block_tables],
            num_computed=runner.req_states.num_computed_tokens.gpu,
            aux_layers=tuple(getattr(decoder, "aux_hidden_state_layers", ())),
            # OP_BOOK / OP_PREV: the host bookkeeping in the launch, for the verify steps of a draft's proposals
            book_tensors=cls.book_tensors_of(runner) if has_book_op else None,
            post_tensors=cls.post_tensors_of(runner) if has_book_op else None,
        )

    @staticmethod
    def book_tensors_of(runner: GPUModelRunner) -> dict:
        """The V2 runner's tensors OP_BOOK reads and writes (schedule.py add_book): its input buffers, the request
        states, the block tables, the drafts."""
        # the drafts from the speculator's output (its batch row 0, the launch's one request: a row stride of 0), which
        # the next launch reads in place of V2's copy into its request states (runner.py syncs it before a vLLM step)
        input_buffers, req_states, block_tables = runner.input_buffers, runner.req_states, runner.block_tables
        drafts = runner.speculator.draft_tokens[:1].expand(req_states.draft_tokens.shape[0], -1)

        return dict(
            inputs=dict(
                input_ids=input_buffers.input_ids,
                positions=input_buffers.positions,
                seq_lens=input_buffers.seq_lens,
                is_padding=input_buffers.is_padding,
            ),
            states=dict(
                last_sampled=req_states.last_sampled_tokens,
                draft_tokens=drafts,
                num_computed=req_states.num_computed_tokens.gpu,
            ),
            tables=dict(
                src=block_tables.block_table_ptrs,
                dst=block_tables.input_block_table_ptrs,
                stride=block_tables.block_table_strides,
                num_blocks=block_tables.num_blocks.gpu,
            ),
        )

    @staticmethod
    def post_tensors_of(runner: GPUModelRunner) -> dict:
        """The V2 runner's request states that OP_PREV's part of V2's post_update writes (schedule.py add_prev), as
        GPUModelRunner.postprocess_sampled passes them: the total length, the token ids, the penalties' counts."""
        req_states, sampler = runner.req_states, getattr(runner, "sampler", None)
        bin_counts = (
            sampler.penalties_state.output_bin_counts if runner.is_last_pp_rank and sampler is not None else None
        )

        return dict(
            total_len=req_states.total_len.gpu, all_token_ids=req_states.all_token_ids.gpu, bin_counts=bin_counts
        )


class DecodeStep:
    """One request's verify step of T <= NT tokens as one launch: the embedding, the layers, the final norm and the
    NVFP4 LM head (logits [T][vocab] bf16), on vLLM's caches and state slots."""

    # numerics as vLLM's compiled step: the post-attention residual h = o + resid in fp32, the next residual
    # bf16(mlp + h), FP8 and NVFP4 inputs quantized from the normed values (NVFP4 of their bf16), the cores as vLLM's
    # spec-decode kernels

    ATT_SPLITS = 42  # splits per KV head: 4 x 42 items, one round on the SMs
    EMBED_PARTS = 8  # the embedding's instructions per token row (add_embed's parts)
    NORM_PRO = Pro.NORM_Q  # the RMSNorm prologues (PRO_NORM_X: every SM normalizes its chunks itself)
    BA_EXTRA_STAGES = 2  # in_proj_ba's SMs take this many in_proj_qkvz stages fewer, beyond ba's own
    GDN_HEAD_ITEMS = 3  # OP_GDNT's items per value head (gdn_t.cuh), each with its part of ssx's sums of squares
    GDN_CONV_TAPS = 3  # the conv state's stored inputs at least (kernel width 4)
    MPOS_WORDS = 3  # the record's M-RoPE positions (t, h, w) from SR_MPOS
    SPEC_MIN_ROWS = 2  # a spec step's rows at least: the token and one draft
    WORD_MASK = 0xFFFFFFFF  # an int's low 32 bits
    SIGN_BIT = 0x80000000  # an int32's sign bit

    def __init__(self, view: ModelView, device: torch.device) -> None:
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.device = device
        assert view.embed.shape[1] == K_NORM
        self.attn_groups = sorted({layer.core.group for layer in view.layers if isinstance(layer.core, AttentionCore)})
        self.mamba_groups = sorted({layer.core.group for layer in view.layers if isinstance(layer.core, GDNCore)})
        assert 1 <= len(self.attn_groups) <= MAX_ATT_GROUPS and 1 <= len(self.mamba_groups) <= GS_MAX_GROUPS
        self.block_size = view.block_size
        self.block_tables, self.num_computed = view.block_tables, view.num_computed

        self.logits = torch.zeros(NT, view.lm_head.weight.shape[0], dtype=torch.bfloat16, device=device)
        self.hidden = torch.zeros(NT, K_NORM, dtype=torch.bfloat16, device=device)
        self.aux: list[torch.Tensor] = []
        self.committed_count = torch.ones(1, dtype=torch.int32, device=device)  # launch()'s default committed count
        self.steps = 0
        self.ss_ring = dict(
            sampled=torch.zeros(RING, NT, dtype=torch.int64, device=device),
            num_sampled=torch.zeros(RING, dtype=torch.int32, device=device),
            num_rejected=torch.zeros(RING, dtype=torch.int32, device=device),
        )
        # its outputs V2's async output reads, in pinned host memory the next launch writes (OP_PREV), and a flag per
        # slot: no copies after the launch
        self.ss_host = {
            k: torch.zeros(self.ss_ring[k].shape, dtype=self.ss_ring[k].dtype).pin_memory() for k in SS_HOST
        }
        self.ss_flags = torch.zeros(RING, dtype=torch.int32).pin_memory()
        # OP_BOOK / OP_PREV in the schedule: launch(book_req=.., prev_launch=..)
        self.has_book_op = view.book_tensors is not None
        self.slot = 0
        self.view = view
        self.draft, self.fused_tensors = None, None  # attach_draft: the draft and the fused program
        self.aux_all = None
        self._gdn_shared = None  # build_fused: the GDN layers' running states and records, shared by both programs

        self.schedule = self.build(view)
        self.tensors = self.schedule.finalize()
        n_instr = self.tensors["instrs"].numel() // INSTR.size
        logger.info("the decode megakernel: %d instructions, %d ops", n_instr, len(self.schedule.descs))

    # ---- the schedule
    def build(self, view: ModelView) -> Schedule:
        """The step's schedule: the target's ops, then the spec sampler."""
        schedule = Schedule(self.num_sms, self.device)

        def zeros(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        x0 = zeros(NT, K_NORM)
        book_ctr = NO_CTR
        # the previous launched step's outputs, then the step's inputs first (launch(prev_launch=, book_req=))
        if self.has_book_op:
            schedule.add_prev(ring=self.ss_ring, host=self.ss_host, flags=self.ss_flags, post=view.post_tensors)
            book_ctr = schedule.add_book(ring=self.ss_ring, **view.book_tensors)
        ctr, ssq = schedule.add_embed(table=view.embed, out=x0, dep=book_ctr, parts=self.EMBED_PARTS)

        # the residual stream's producer (the embedding, o_proj / out_proj, down) forms x = out + residual in its
        # finishers (fp32 for the next norm, bf16 for the next residual add) and publishes its per-tile sums of squares;
        # the norm's consumers read rstd from those and only their K chunks of x
        x_in, x_bf16, resid = x0, True, x0
        self.gdn_runs = []  # each GDN layer's running state: its start-of-block state after a launch
        self.gdn_recs = []  # and its replay records
        # the draft's aux hidden states (vLLM's _maybe_add_hidden_state, layer L: the residual stream after L layers)
        # in one tensor [aux layer][NT][K], which its fc reads as one operand: each aux layer's residual a slice of it
        aux_ids = sorted(view.aux_layers)
        if self.aux_all is None:  # (the fused program's build: the same buffers, the draft's input)
            self.aux_all = zeros(max(1, len(aux_ids)), NT, K_NORM)
        self.aux = []
        self.aux_ctr = NO_CTR
        if 0 in view.aux_layers:
            self.aux.append(x0)

        for i, layer in enumerate(view.layers):
            core, in_norm = layer.core, layer.in_norm
            norm_args = dict(pro=self.NORM_PRO, in0=x_in, x_bf16=x_bf16, ssq_in=ssq)
            # o + resid (fp32): the post-attention norm's x, the MLP's residual
            post_attn_residual = zeros(NT, K_NORM, dtype=torch.float32)
            xr_args = dict(xr=True, res_in=resid, res_bf16=True, x_out=post_attn_residual)
            if isinstance(core, AttentionCore):
                qkv_out = zeros(NT, core.qkv.weight.shape[0])
                ctr, _ = schedule.add_gemv(
                    w=core.qkv.weight,
                    kind=Kind.FP8,
                    K=K_NORM,
                    dep=ctr,
                    alpha=core.qkv.alpha,
                    gs=core.qkv.input_scale,
                    eps=in_norm.eps,
                    nw=in_norm.weight,
                    out=qkv_out,
                    **norm_args,
                )
                # o_proj's FP8 operand: K chunks of 128 values, each NT rows of 128 bytes
                act = zeros(core.o_proj.weight.shape[1] // 128 * NT * 128, dtype=torch.uint8)
                attn_group = self.attn_groups.index(core.group)
                kv_cache = core.kv_cache
                assert kv_cache.dtype in (torch.float8_e4m3fn, torch.uint8) and kv_cache.element_size() == 1
                # FlashInfer's NHD cache: logical [blocks][4 heads][block tokens][k 256 | v 256], in memory
                # [blocks][tokens][heads]
                kv_bytes = kv_cache.view(torch.uint8).permute(0, 2, 1, 3)
                assert (
                    kv_bytes.shape[2:] == (Schedule.ATT_HKV, Schedule.ATT_KV_ROW_BYTES) and kv_bytes.is_contiguous()
                ), (
                    tuple(kv_cache.shape),
                    kv_cache.stride(),
                )
                ctr = schedule.add_attention(
                    kv=kv_bytes.view(-1),
                    qkv=qkv_out,
                    q_norm_w=core.q_norm,
                    k_norm_w=core.k_norm,
                    cos_sin=core.cos_sin,
                    act=act,
                    block_tokens=self.block_size,
                    group=attn_group,
                    scale=core.scale,
                    eps=core.eps,
                    s_in=core.o_proj.input_scale,
                    n_splits=self.ATT_SPLITS,
                    dep=ctr,
                )
                o_out = zeros(NT, K_NORM)
                ctr, o_desc = schedule.add_gemv(
                    w=core.o_proj.weight,
                    kind=Kind.FP8,
                    K=core.o_proj.weight.shape[1],
                    dep=ctr,
                    pro=Pro.ACT,
                    alpha=core.o_proj.alpha,
                    gs=core.o_proj.input_scale,
                    in0=act,
                    out=o_out,
                    **xr_args,
                )
            else:
                qkvz_out = zeros(NT, core.in_proj.weight.shape[0])
                ba_weight = core.ba_weight.view(torch.uint8).view(core.ba_weight.shape[0], -1)
                # in_proj_ba runs after in_proj_qkvz on its SMs (one B slot): they take fewer of qkvz's stages, by ba's
                # and BA_EXTRA_STAGES for its B copy, MMAs and drain, so both end together
                ba_sm_stages = {
                    sm: stages + self.BA_EXTRA_STAGES
                    for sm, stages in schedule.gemv_counts(ba_weight, Kind.BF16, K_NORM).items()
                }
                qkvz_ctr, _ = schedule.add_gemv(
                    w=core.in_proj.weight,
                    kind=Kind.FP8,
                    K=K_NORM,
                    dep=ctr,
                    alpha=core.in_proj.alpha,
                    gs=core.in_proj.input_scale,
                    eps=in_norm.eps,
                    nw=in_norm.weight,
                    out=qkvz_out,
                    sm_extra=ba_sm_stages,
                    **norm_args,
                )
                ba_out = zeros(NT, 2 * GDN_HV)
                ba_ctr, _ = schedule.add_gemv(
                    w=ba_weight,
                    kind=Kind.BF16,
                    K=K_NORM,
                    dep=ctr,
                    alpha=1.0,
                    gs=1.0,
                    eps=in_norm.eps,
                    nw=in_norm.weight,
                    out=ba_out,
                    acc_pool="acc_ba",
                    **norm_args,
                )  # (runs beside in_proj_qkvz)
                # out_proj's FP8 operand: a K chunk per value head
                act = zeros(GDN_HV * NT * 128, dtype=torch.uint8)
                ctr = self.add_gdn(schedule, core, qkvz_out, ba_out, act, qkvz_ctr, ba_ctr)
                o_out = zeros(NT, K_NORM)
                ctr, o_desc = schedule.add_gemv(
                    w=core.out_proj.weight,
                    kind=Kind.FP8,
                    K=core.out_proj.weight.shape[1],
                    dep=ctr,
                    pro=Pro.ACT,
                    alpha=core.out_proj.alpha,
                    gs=core.out_proj.input_scale,
                    in0=act,
                    out=o_out,
                    **xr_args,
                )

            # the post-attention RMSNorm of h, the MLP; down's finishers form the next layer's x = down + h (fp32) and
            # its residual bf16(down + h)
            post_norm, gate_up, down_proj = layer.post_norm, layer.gate_up, layer.down
            # down's NVFP4 operand: K chunks of 256 values, each NT rows of 128 code bytes, then their 16 scale bytes
            act_d = zeros(down_proj.weight.shape[1] * 2 // 256 * (NT * 128 + NT * 16), dtype=torch.uint8)
            ctr, _ = schedule.add_gemv(
                w=gate_up.weight,
                sf=gate_up.block_scales,
                kind=Kind.NVFP4,
                K=K_NORM,
                dep=ctr,
                pro=self.NORM_PRO,
                alpha=gate_up.alpha,
                gs=gate_up.input_global_scale_inv,
                eps=post_norm.eps,
                nw=post_norm.weight,
                in0=post_attn_residual,
                ssq_in=schedule.ssq_of(o_desc),
                out=act_d,
                act_gs=down_proj.input_global_scale_inv,
            )
            down_out = zeros(NT, K_NORM)
            x_next = zeros(NT, K_NORM, dtype=torch.float32)
            r_next = self.aux_all[aux_ids.index(i + 1)] if i + 1 in view.aux_layers else zeros(NT, K_NORM)
            ctr, down_desc = schedule.add_gemv(
                w=down_proj.weight,
                sf=down_proj.block_scales,
                kind=Kind.NVFP4,
                K=down_proj.weight.shape[1] * 2,
                dep=ctr,
                pro=Pro.ACT,
                alpha=down_proj.alpha,
                gs=down_proj.input_global_scale_inv,
                in0=act_d,
                out=down_out,
                xr=True,
                res_in=post_attn_residual,
                x_out=x_next,
                r_out=r_next,
            )
            if i + 1 in view.aux_layers:
                self.aux.append(r_next)  # bf16(down + h)
                if i + 1 == aux_ids[-1]:
                    self.aux_ctr = ctr  # the last aux state written: the fused draft's context ops may start
            x_in, x_bf16, resid, ssq = x_next, False, r_next, schedule.ssq_of(down_desc)

        final_norm, lm_head = view.final_norm, view.lm_head
        ctr, _ = schedule.add_gemv(
            w=lm_head.weight,
            sf=lm_head.block_scales,
            kind=Kind.NVFP4,
            K=K_NORM,
            dep=ctr,
            pro=self.NORM_PRO,
            alpha=lm_head.alpha,
            gs=lm_head.input_global_scale_inv,
            eps=final_norm.eps,
            nw=final_norm.weight,
            in0=x_in,
            ssq_in=ssq,
            out=self.logits.view(-1),
            is_logits=True,
        )
        # the spec sampler after the LM head (its phases run when the record's top_p is set, else vLLM samples); the
        # fused draft's fc goes at the mark "pre_sampler", streaming beside the sampler's phases, behind phase A0 so
        # the prologue runs A0 while the LM head streams
        self.sampler_ctr = schedule.add_spec_sample(
            lm_ctr=ctr, logits=self.logits, ring=self.ss_ring, mark="pre_sampler"
        )

        return schedule

    def add_gdn(
        self,
        schedule: Schedule,
        core: GDNCore,
        qkvz_out: torch.Tensor,
        ba_out: torch.Tensor,
        act: torch.Tensor,
        qkvz_ctr: int,
        ba_ctr: int,
    ) -> int:
        """The Gated DeltaNet core (OP_GDNT, csrc/gdn_t.cuh) of the layer: three items per value head."""
        conv_state, ssm_state = core.conv_state, core.ssm_state
        conv_dim = core.conv_weight.shape[0]
        assert conv_state.dtype == torch.bfloat16 and ssm_state.dtype == torch.float32 and ssm_state.dim() == 4
        assert (
            conv_state.shape[1] >= self.GDN_CONV_TAPS and conv_state.shape[2] == conv_dim and conv_state.stride(2) == 1
        )
        assert ssm_state[0].is_contiguous()

        if self._gdn_shared is not None:  # the fused program: the target-only program's states, carried across steps
            run, rec = self._gdn_shared.pop(0)
        else:
            run = torch.zeros(GDN_HV, GDN_V, GDN_K, dtype=torch.float32, device=self.device)
            rec = torch.zeros(2, GDNT_REC_BYTES, dtype=torch.uint8, device=self.device)
        self.gdn_runs.append(run)
        self.gdn_recs.append(rec)
        ssx = torch.zeros(GDN_HV * self.GDN_HEAD_ITEMS * NT, dtype=torch.int64, device=self.device)
        group = self.mamba_groups.index(core.group)
        schedule.keep += [
            conv_state,
            ssm_state,
            run,
            rec,
            ssx,
            qkvz_out,
            ba_out,
            act,
            core.conv_weight,
            core.a_log,
            core.dt_bias,
            core.norm.weight,
        ]

        # GdnTDesc: qw the qkvz output (channel stride 1, token stride its row), ba in_proj_ba's output, the outputs as
        # out_proj's FP8 operand (GT_GNORM); 0: the fields of the modes not used
        schedule.descs.append(GDNT.pack(
            core.conv_weight.data_ptr(), core.a_log.data_ptr(), core.dt_bias.data_ptr(), run.data_ptr(),
            ssm_state.data_ptr(), conv_state.data_ptr(), rec.data_ptr(), qkvz_out.data_ptr(), 0, 0, 0,
            ba_out.data_ptr(), 0,
            core.scale, 1.0,
            conv_state.stride(0), ssm_state.stride(0), conv_state.shape[1], 0, 1, group, GT_IN_BF16,
            GT_BA_OP, GT_GNORM,
            0, 0, qkvz_out.shape[1],
            0, act.data_ptr(), core.norm.weight.data_ptr(), ssx.data_ptr(),
            float(core.norm.eps), 1.0 / core.out_proj.input_scale, 0, 0,
        ))  # fmt: skip
        desc_index = len(schedule.descs) - 1
        ctr = schedule.new_counter(GDN_HV * self.GDN_HEAD_ITEMS)
        for item in range(GDN_HV * self.GDN_HEAD_ITEMS):
            schedule.instr(item % self.num_sms, Op.GDNT, desc_index, qkvz_ctr, ctr, 0, 0, item, dep2=ba_ctr)

        return ctr

    # ---- the fused program
    def build_fused(self) -> dict:
        """The target's schedule built again for the fused program (mk_main<2>), for draft.py DraftStep to add its ops
        to: {schedule, aux_ctr, sampler_ctr, pre_sampler}."""
        # each program's per-op words follow its own launches (an op's flags, tags and double buffers by its program's
        # epoch); the state carried across steps is shared: the GDN layers' running states and records, the ring
        assert self.aux_ctr != NO_CTR and 0 not in self.view.aux_layers, "the fused draft's context: aux layers > 0"

        target_state = {k: getattr(self, k) for k in ("sampler_ctr", "aux_ctr", "gdn_runs", "gdn_recs", "aux")}
        self._gdn_shared = list(zip(self.gdn_runs, self.gdn_recs))
        schedule = self.build(self.view)
        assert not self._gdn_shared
        self._gdn_shared = None
        fused_parts = dict(
            schedule=schedule,
            aux_ctr=self.aux_ctr,
            sampler_ctr=self.sampler_ctr,
            pre_sampler=schedule.marks["pre_sampler"],
        )
        for k, value in target_state.items():  # the target-only program's, as before
            setattr(self, k, value)

        return fused_parts

    def attach_draft(self, draft: DraftStep, schedule: Schedule) -> None:
        """The fused program (mk_main<2>): build_fused's schedule with the draft's proposal after the target's ops."""
        self.draft = draft
        self.fused_tensors = schedule.finalize()
        logger.info(
            "the fused program: %d instructions, %d ops",
            self.fused_tensors["instrs"].numel() // INSTR.size,
            len(schedule.descs),
        )

    # ---- the launch
    def launch(
        self,
        *,
        token_ptr: int,
        pos: int,
        mpos: list[int],
        n_tok: int,
        req_idx: int,
        gdnt_first: bool,
        gdnt_parity: int,
        committed_ptr: int | None = None,
        spec_inputs: dict[str, int | float] | None = None,
        draft_words: dict[str, int] | None = None,
        book_req: int | None = None,
        prev_launch: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Fill the record's header and launch the step of n_tok tokens at positions pos.. (logits in self.logits),
        returning the spec sampler's outputs in their ring slot (None without spec_inputs)."""
        # spec_inputs: the request's sampling inputs (SS_INPUTS' device addresses, top_p); draft_words (draft.py
        # DraftStep.header's, but the slot): the fused program, the draft's proposal after the sampler
        assert 1 <= n_tok <= NT

        header = torch.zeros(REC_HDR, dtype=torch.int32)
        words = header.numpy()
        words[SR_FLAGS] = TOKEN_ON_DEVICE | POS_ON_DEVICE
        words[SR_TOK_PTR], words[SR_TOK_PTR + 1] = self.int32(token_ptr), self.int32(token_ptr >> 32)
        words[SR_POS] = pos
        words[SR_MPOS : SR_MPOS + self.MPOS_WORDS] = mpos
        words[SR_RNG], words[SR_RNG + 1] = self.int32(self.steps), self.int32(self.steps >> 32)
        self.steps += 1
        words[SR_NTOK] = n_tok

        gdn_words = words[SR_GDNT : SR_GDNT + GS_WORDS]
        gdn_words[GS_T], gdn_words[GS_FIRST], gdn_words[GS_PARITY] = n_tok, int(gdnt_first), gdnt_parity
        gdn_words[GS_BLOCK], gdn_words[GS_CONV_OFF] = self.block_size, 0
        gdn_words[GS_DIRECT] = 0
        gdn_words[GS_C_HOST] = int(gdnt_first)
        gdn_addresses = gdn_words.view(np.int64)
        gdn_addresses[GS_C_PTR // 2] = committed_ptr if committed_ptr is not None else self.committed_count.data_ptr()
        gdn_addresses[GS_POS_PTR // 2] = self.num_computed.data_ptr() + 4 * req_idx
        for j, group in enumerate(self.mamba_groups):
            block_table = self.block_tables[group]
            gdn_addresses[GS_BT // 2 + j] = (
                block_table.data_ptr() + block_table.stride(0) * block_table.element_size() * req_idx
            )
        addresses = words[SR_POS_PTR:SR_SS].view(np.int64)
        addresses[0] = self.num_computed.data_ptr() + 4 * req_idx
        for group_index, group in enumerate(self.attn_groups):
            block_table = self.block_tables[group]
            addresses[1 + group_index] = (
                block_table.data_ptr() + block_table.stride(0) * block_table.element_size() * req_idx
            )

        assert (book_req is None and prev_launch is None) or self.has_book_op
        if book_req is not None:  # OP_BOOK: the request's row of V2's request states, whose inputs the launch writes
            words[SR_BK + BK_REQ], words[SR_BK + BK_ON] = book_req, 1
        if prev_launch is not None:  # BK_PREV (csrc/step.cuh BookWord): the previous launched step for OP_BOOK, OP_PREV
            assert (
                prev_launch["slot"] < RING
                and self.SPEC_MIN_ROWS <= prev_launch["T"] <= NT
                and prev_launch["req"] < 1 << (32 - PREV_REQ_SHIFT)
            )
            assert not prev_launch["post"] or prev_launch["req"] == book_req, (
                "OP_BOOK takes the post_update of its own request"
            )
            words[SR_BK + BK_PREV] = self.int32(
                PREV_OUT * prev_launch["out"]
                | PREV_POST * prev_launch["post"]
                | prev_launch["slot"] << PREV_SLOT_SHIFT
                | prev_launch["T"] << PREV_T_SHIFT
                | prev_launch["req"] << PREV_REQ_SHIFT
            )
            words[SR_BK + BK_SEQ] = self.int32(prev_launch["seq"])

        slot = self.slot
        if spec_inputs is not None:  # SsStepWord: the request's sampling inputs, this launch's ring slot
            assert n_tok >= self.SPEC_MIN_ROWS and 0.0 < spec_inputs["top_p"] <= 1.0
            self.slot = (slot + 1) % RING
            sampler_words = words[SR_SS:REC_HDR]
            for word, key in SS_INPUTS.items():
                sampler_words[word : word + 2].view(np.int64)[0] = spec_inputs[key]
            sampler_words[SS_SLOT] = slot
            sampler_words[SS_TOP_P : SS_TOP_P + 1].view(np.float32)[0] = spec_inputs["top_p"]
        tensors = self.tensors
        if draft_words is not None:
            assert spec_inputs is not None and n_tok == NT, "the fused program: a sampled verify step of NT rows"
            self.draft.header(words, slot=slot, **draft_words)
            tensors = self.fused_tensors

        run_op = (
            torch.ops.pear_deploy_rtx_blackwell_nvfp4.run
            if draft_words is None
            else torch.ops.pear_deploy_rtx_blackwell_nvfp4.run_fused
        )
        run_op(
            tensors["instrs"],
            tensors["sm_off"],
            tensors["descs"],
            tensors["counters"],
            tensors["ctr_count"],
            tensors["epoch"],
            self.num_sms,
            header,
            NO_TRACE,
        )
        if spec_inputs is None:
            return None

        ring = self.ss_ring
        return (
            ring["sampled"][slot, :n_tok].view(1, n_tok),
            ring["num_sampled"][slot : slot + 1],
            ring["num_rejected"][slot : slot + 1],
        )

    @staticmethod
    def int32(x: int) -> int:
        """x's low 32 bits as a signed int32."""
        return ((x & DecodeStep.WORD_MASK) ^ DecodeStep.SIGN_BIT) - DecodeStep.SIGN_BIT
