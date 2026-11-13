from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from pear.models.qwen3_5.deploy.blackwell.nvfp4.schedule import (
    ATT_KV_HEADS,
    ATT_PAGE,
    ATT_ROW_BYTES,
    CHUNK_B,
    INSTR,
    K_NORM,
    NO_CTR,
    NT,
    SS_HOST,
    Kind,
    Pro,
    Schedule,
)
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.proposal import DraftProposal

logger = get_logger(__name__)

# the step record (step.cuh StepRecWord): int32 words; its header, the words before REC_HDR, goes with the launch
REC_HDR = 112
SR_FLAGS = 0
SR_TOK_PTR = 2
SR_POS = 4
SR_MPOS = 5
SR_RNG = 10
SR_SEED = 12
SR_NTOK = 15
SR_GDNT = 32  # OP_GDNT's step words (csrc/gdn_t.cuh GdnStepWord)
SR_POS_PTR = 80  # the step's first position on the device (uint64 address of an int32)
SR_ATT_BT = 82  # per attention KV cache group its block-table row on the device (uint64 addresses)
SR_SS = 96  # the spec sampler's step words (spec_sample.cuh SsStepWord)
SR_BK = 108  # the step's inputs written by the launch (step.cuh BookWord)
# BookWord: the request's row, whether OP_BOOK writes the step's inputs, the previous launched step, its host flag value
BK_REQ, BK_ON, BK_PREV, BK_SEQ = 0, 1, 2, 3
PREV_OUT, PREV_POST = 1, 2  # BK_PREV's bits: the previous launched step's outputs into the host ring, its post_update
PREV_SLOT_SHIFT, PREV_T_SHIFT, PREV_REQ_SHIFT = 2, 4, 8  # BK_PREV's fields: its ring slot, rows, request row
MAX_ATT_GROUPS = (SR_SS - SR_ATT_BT) // 2
# SsStepWord: the inputs' device addresses (uint64), the output ring slot, the request's top_p (fp32 bits; 0: the launch
# leaves the sampling to vLLM)
SS_DRAFT, SS_LIDX, SS_SEED, SS_IDS, SS_POSV, SS_SLOT, SS_TOP_P = 0, 2, 4, 6, 8, 10, 11
SS_INPUTS = {SS_DRAFT: "draft", SS_LIDX: "logits_indices", SS_SEED: "seed", SS_IDS: "input_ids", SS_POSV: "positions"}
TOKEN_ON_DEVICE = 1  # SR_FLAGS: the input tokens are the int32s at SR_TOK_PTR (one per token row)
POS_ON_DEVICE = 2  # SR_FLAGS: the step's first position is the int32 at SR_POS_PTR
# GdnStepWord
GS_T, GS_FIRST, GS_PARITY, GS_BLOCK, GS_CONV_OFF, GS_DIRECT = 0, 1, 2, 3, 4, 5
GS_C_PTR, GS_POS_PTR, GS_C_HOST, GS_BT = 6, 8, 10, 16
GS_MAX_GROUPS = 16  # gdn_t.cuh: the mamba groups' block-table rows the step words hold
GS_WORDS = GS_BT + 2 * GS_MAX_GROUPS
GDNT_REC_BYTES = 227072  # gdn_t.cuh GTR_BYTES: one copy of a layer's replay record
RING = 4  # output / record buffers in flight
MROPE_SECTIONS = (11, 11, 10)  # Qwen3.5's interleaved M-RoPE (t, h, w) over the first ROTARY_DIM dims
ROTARY_DIM = 64
GDN_CONV_TAPS = 3  # the conv state's stored inputs (kernel width 4)
GDN_HV, GDN_V, GDN_K = 48, 128, 128  # Gated DeltaNet value heads, their dim, the key dim (csrc/gdn_t.cuh)


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
        """vLLM's linear."""
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
    """A Gated DeltaNet layer's projections, core parameters and state cache (its KV cache group)."""

    in_proj: Fp8Linear  # q, k, v, z rows
    ba_weight: torch.Tensor  # in_proj_ba, bf16 [2 * heads, K]
    conv_weight: torch.Tensor  # [conv_dim, 4]
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: Norm
    out_proj: Fp8Linear
    conv_state: torch.Tensor  # [slots][taps][conv_dim] bf16
    ssm_state: torch.Tensor  # [slots][heads][v][k] fp32
    group: int
    scale: float


@dataclass
class AttentionCore:
    """A full-attention layer's projections, norms, rotary table and KV cache (its KV cache group)."""

    qkv: Fp8Linear
    o_proj: Fp8Linear
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    eps: float
    cos_sin: torch.Tensor  # bf16 [positions][64]
    scale: float
    kv_cache: torch.Tensor  # bf16 [kernel blocks][kv heads][64][k | v]
    group: int


@dataclass
class Layer:
    """A decoder layer: its core (GDNCore / AttentionCore), norms and MLP."""

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
    zero_meta: tuple | None = None  # vLLM's KVBlockZeroer segments (addresses, strides, pages), if any
    # per KV cache group the V2 runner's device block table [requests][blocks] int32
    block_tables: list[torch.Tensor] | None = None
    num_computed: torch.Tensor | None = None  # the V2 runner's computed tokens per request (int32, device)
    aux_layers: tuple[int, ...] = ()  # the layers whose input (the residual stream before them) the draft reads
    # the V2 runner's input buffers, request states and block tables OP_BOOK writes / reads
    book_tensors: dict | None = None
    # the V2 runner's request states its post_update writes (OP_PREV writes them instead)
    post_tensors: dict | None = None

    @classmethod
    def from_runner(cls, runner: GPUModelRunner) -> ModelView:
        """vLLM's V2 model runner's model and caches (one attention KV cache group, align-mode state slots)."""
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
                core = AttentionCore(
                    qkv=Fp8Linear.from_vllm(attention.qkv_proj),
                    o_proj=Fp8Linear.from_vllm(attention.o_proj),
                    q_norm=attention.q_norm.weight,
                    k_norm=attention.k_norm.weight,
                    eps=attention.q_norm.variance_epsilon,
                    cos_sin=rotary.cos_sin_cache.to(torch.bfloat16).contiguous(),
                    scale=attention.scaling,
                    kv_cache=context[name].kv_cache,
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

        groups = {layer.core.group for layer in layers}
        block_size = kv_cache_config.kv_cache_groups[next(iter(groups))].kv_cache_spec.block_size
        assert all(kv_cache_config.kv_cache_groups[group].kv_cache_spec.block_size == block_size for group in groups)
        meta = getattr(getattr(runner, "_kv_block_zeroer", None), "_meta", None)

        return cls(
            layers=layers,
            embed=decoder.embed_tokens.weight,
            final_norm=Norm(decoder.norm.weight, decoder.norm.variance_epsilon),
            lm_head=Nvfp4Linear.from_vllm(language_model.lm_head),
            block_size=block_size,
            zero_meta=meta,
            block_tables=[block_table.gpu for block_table in runner.block_tables.block_tables],
            num_computed=runner.req_states.num_computed_tokens.gpu,
            aux_layers=tuple(getattr(decoder, "aux_hidden_state_layers", ())),
            book_tensors=cls.book_tensors_of(runner),
            post_tensors=cls.post_tensors_of(runner),
        )

    @staticmethod
    def book_tensors_of(runner: GPUModelRunner) -> dict:
        """The V2 runner's tensors OP_BOOK reads and writes (schedule.py add_book)."""
        input_buffers, req_states, block_tables = runner.input_buffers, runner.req_states, runner.block_tables
        # the drafts from the speculator's output (its batch row 0, the launch's one request: a row stride of 0), which
        # the next launch reads in place of V2's copy into its request states (runner.py: synced before a vLLM step)
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
        GPUModelRunner.postprocess_sampled passes them: the total length, the token ids (UVA), the penalties' counts."""
        req_states, sampler = runner.req_states, getattr(runner, "sampler", None)
        bin_counts = (
            sampler.penalties_state.output_bin_counts if runner.is_last_pp_rank and sampler is not None else None
        )

        return dict(
            total_len=req_states.total_len.gpu, all_token_ids=req_states.all_token_ids.gpu, bin_counts=bin_counts
        )


class DecodeStep:
    """One sequence's verify step of T <= NT tokens as one megakernel launch: embedding, the decoder layers, the final
    norm, the NVFP4 LM head (logits [T, vocab]), the spec sampler and the draft's proposal, on vLLM's KV cache and
    Gated DeltaNet states."""

    # Numerics follow vLLM's compiled decode step: the post-attention residual h = o + resid stays fp32, the next
    # residual is bf16(mlp + h), the FP8 inputs are quantized from the fp32 normed value, the attention gate is not
    # rounded before its quantization; the cores follow vLLM's decode kernels, the LM head and the sampler vLLM's

    DRAFT_OP_BASE = 32  # the draft's op codes in the step's instruction lists (megakernel.cu DK_OP_BASE)
    SPEC_MIN_ROWS = 2  # a spec step's rows at least: the token and one draft
    MPOS_WORDS = 3  # the record's M-RoPE positions (t, h, w) from SR_MPOS
    INT32_BYTES = 4  # of a step record word and of an entry of num_computed
    GDN_HEAD_ITEMS = 3  # the core's work items per value head (gdn_t.cuh), each with its part of ssx's sums of squares
    SEED = 0x5EED  # the sampler's Philox seed is drawn from it
    SEED_BITS = 62  # the drawn seed's width in bits
    ACT_SENTINEL = 0xFF  # out_proj's operand bytes until the core's gated RMSNorm writes them (no FP8 code is 0xFF)

    def __init__(self, view: ModelView, device: torch.device, draft: DraftProposal) -> None:
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        assert view.embed.shape[1] == K_NORM, (
            f"the kernel's RMSNorm prologues are written for a hidden size of {K_NORM}"
        )

        # the KV cache groups: those of the attention layers, up to GS_MAX_GROUPS of the Gated DeltaNet layers
        self.attn_groups = sorted({layer.core.group for layer in view.layers if isinstance(layer.core, AttentionCore)})
        self.mamba_groups = sorted({layer.core.group for layer in view.layers if isinstance(layer.core, GDNCore)})
        assert 1 <= len(self.attn_groups) <= MAX_ATT_GROUPS and len(self.mamba_groups) >= 1
        assert len(self.mamba_groups) <= GS_MAX_GROUPS
        self.block_size = view.block_size

        # the record's header, filled in a pinned host ring slot and passed with the launch (step.cuh RecParam),
        # which the kernel reads instead of the device record the descriptors point at
        self.record = torch.zeros(REC_HDR, dtype=torch.int32, device=device)
        self.header_ring = [torch.zeros(REC_HDR, dtype=torch.int32).pin_memory() for _ in range(RING)]
        # the spec sampler's (OP_SS_*) outputs per ring slot, vLLM's SamplerOutput of the request
        self.ss_ring = self.spec_ring(device)
        # its outputs V2's async output reads, in pinned host memory the launch writes (OP_PREV): no copies after it
        self.ss_host = {
            k: torch.zeros(self.ss_ring[k].shape, dtype=self.ss_ring[k].dtype).pin_memory() for k in SS_HOST
        }
        # a flag per host ring slot: its outputs written
        self.ss_flags = torch.zeros(RING, dtype=torch.int32).pin_memory()
        self.has_book_op = view.book_tensors is not None  # (OP_BOOK in the schedule: launch(book_req=...))
        self.post_tensors = view.post_tensors
        self.logits = torch.zeros(NT, view.lm_head.weight.shape[0], dtype=torch.bfloat16, device=device)
        # what vLLM's bookkeeping holds on to
        self.hidden = torch.zeros(NT, K_NORM, dtype=torch.bfloat16, device=device)
        self.aux: list[torch.Tensor] = []  # the residual stream before each of view.aux_layers (the draft's inputs)
        self.seed = int(torch.randint(0, 2**self.SEED_BITS, (1,), generator=torch.Generator().manual_seed(self.SEED)))
        # OP_GDNT (replaySSM): its step words point into the V2 runner's device state (the mamba groups' block-table
        # rows, the computed tokens) and at the previous block's committed count (self.committed_count unless the caller
        # passes its own)
        self.committed_count = torch.ones(1, dtype=torch.int32, device=device)
        self.block_tables, self.num_computed = view.block_tables, view.num_computed
        assert self.block_tables is not None and self.num_computed is not None
        self.steps = 0
        self.slot = 0
        self.schedule = self.build(view, device)

        # the draft's proposal (draft.proposal.DraftProposal) after the step on every SM, the spec sampler's tokens its
        # anchor: its standalone schedule's lists (its descriptors, counters and epoch its own), the launch's draft
        # phase waiting for the sampler's last counter and the last aux layer's
        self.schedule.attach_draft(draft.schedule.relocated(self.DRAFT_OP_BASE), {self.sampler_ctr, self.aux_ctr})
        self.tensors = self.schedule.finalize()

        n_instr = self.tensors["instrs"].numel() // INSTR.size
        logger.info(
            "decode megakernel: %d instructions (%d per SM), %d ops",
            n_instr,
            n_instr // self.num_sms,
            len(self.schedule.descs),
        )

    @staticmethod
    def unsupported_reason(view: ModelView) -> str | None:
        """Why the step cannot run the view's model and caches, or None."""
        # the draft reads the residual stream before decoder layers 1..L, which the step's prologues hand it
        if not view.aux_layers or not all(0 < layer_index <= len(view.layers) for layer_index in view.aux_layers):
            return f"the draft's aux layers {view.aux_layers}, not of decoder layers 1..{len(view.layers)}"

        # its attention reads kernel blocks of ATT_PAGE tokens
        for layer in view.layers:
            if isinstance(layer.core, AttentionCore):
                shape = tuple(layer.core.kv_cache.shape[1:])
                if shape != (ATT_KV_HEADS, ATT_PAGE, ATT_ROW_BYTES // 2):
                    return f"attention KV cache blocks {shape} (block size {view.block_size})"

        return None

    def launch(
        self,
        *,
        token_ptr: int,
        pos: int,
        mpos: list[int],
        n_tok: int = 1,
        req_idx: int = 0,
        gdnt_first: bool = True,
        gdnt_parity: int = 0,
        committed_ptr: int | None = None,
        spec_inputs: dict[str, int | float] | None = None,
        book_req: int | None = None,
        prev_launch: dict[str, int] | None = None,
        draft_args: Callable[[int], int] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Fill the step record's header and launch the step of n_tok tokens at positions pos.., returning with
        spec_inputs the spec sampler's outputs (spec_outputs), else None (the logits in self.logits[:n_tok])."""
        # token_ptr: a device int32 tensor of n_tok; spec_inputs: the spec sampler's inputs (the SS_INPUTS addresses,
        # top_p); book_req: the request's row of V2's request states, whose inputs OP_BOOK writes; prev_launch: the
        # previous launched step (its ring slot, rows T, request row, out / post / seq), which OP_PREV completes
        assert 1 <= n_tok <= NT
        assert spec_inputs is None or n_tok >= self.SPEC_MIN_ROWS
        assert book_req is None or self.has_book_op
        assert prev_launch is None or self.has_book_op

        slot = self.slot
        self.slot = (slot + 1) % RING
        header = self.header_ring[slot]
        words = header.numpy()

        words[:] = 0
        words[SR_FLAGS] = TOKEN_ON_DEVICE
        words[SR_TOK_PTR], words[SR_TOK_PTR + 1] = self.int32(token_ptr), self.int32(token_ptr >> 32)
        words[SR_POS] = pos
        words[SR_MPOS : SR_MPOS + self.MPOS_WORDS] = mpos
        words[SR_RNG], words[SR_RNG + 1] = self.int32(self.steps), self.int32(self.steps >> 32)
        words[SR_SEED], words[SR_SEED + 1] = self.int32(self.seed), self.int32(self.seed >> 32)
        self.steps += 1
        words[SR_NTOK] = n_tok

        # (gdn_t.cuh GdnStepWord) this block's T tokens from the request's computed ones (device), the previous block's
        # committed count (device: committed_ptr, else self.committed_count), each mamba group's block-table row;
        # GS_FIRST: no record yet (the request's first launch); T = 1: direct (K = 0), nothing to replay
        gdn_words = words[SR_GDNT : SR_GDNT + GS_WORDS]
        gdn_words[GS_T], gdn_words[GS_FIRST], gdn_words[GS_PARITY] = n_tok, int(gdnt_first), gdnt_parity
        gdn_words[GS_BLOCK], gdn_words[GS_CONV_OFF] = self.block_size, 0
        gdn_words[GS_DIRECT] = int(n_tok == 1)
        # the previous block's committed count is known on the host after a direct or a first step (nothing to replay);
        # else it comes from the device (GS_C_PTR)
        gdn_words[GS_C_HOST] = int(bool(gdn_words[GS_DIRECT]) or gdnt_first)
        gdn_addresses = gdn_words.view(np.int64)
        gdn_addresses[GS_C_PTR // 2] = committed_ptr if committed_ptr is not None else self.committed_count.data_ptr()
        gdn_addresses[GS_POS_PTR // 2] = self.num_computed.data_ptr() + self.INT32_BYTES * req_idx
        for j, group in enumerate(self.mamba_groups):
            block_table = self.block_tables[group]
            gdn_addresses[GS_BT // 2 + j] = (
                block_table.data_ptr() + block_table.stride(0) * block_table.element_size() * req_idx
            )

        # the position and the attention groups' block-table rows on the device
        words[SR_FLAGS] |= POS_ON_DEVICE
        addresses = words[SR_POS_PTR:SR_SS].view(np.int64)
        addresses[0] = self.num_computed.data_ptr() + self.INT32_BYTES * req_idx
        for group_index, group in enumerate(self.attn_groups):
            block_table = self.block_tables[group]
            addresses[1 + group_index] = (
                block_table.data_ptr() + block_table.stride(0) * block_table.element_size() * req_idx
            )

        if book_req is not None:
            words[SR_BK + BK_REQ], words[SR_BK + BK_ON] = book_req, 1
        if prev_launch is not None:  # (step.cuh BookWord BK_PREV) the previous launched step for OP_BOOK and OP_PREV
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
        # (spec_sample.cuh SsStepWord) the request's sampling inputs, this launch's ring slot
        if spec_inputs is not None:
            sampler_words = words[SR_SS:REC_HDR]
            for word, key in SS_INPUTS.items():
                sampler_words[word : word + 2].view(np.int64)[0] = spec_inputs[key]
            sampler_words[SS_SLOT] = slot
            sampler_words[SS_TOP_P : SS_TOP_P + 1].view(np.float32)[0] = spec_inputs["top_p"]

        tensors = self.tensors
        torch.ops.pear_deploy_blackwell_nvfp4.run(
            tensors["instrs"],
            tensors["sm_off"],
            tensors["descs"],
            tensors["counters"],
            tensors["ctr_count"],
            tensors["epoch"],
            self.num_sms,
            self.record,
            0,  # (no trace buffers: the op's trace and strace)
            0,
            header,
            *self.draft_launch(draft_args, slot),
        )

        return self.spec_outputs(slot, n_tok) if spec_inputs is not None else None

    def build(self, view: ModelView, device: torch.device) -> Schedule:
        """The schedule of the whole step."""
        schedule = Schedule(self.num_sms, device)
        self.pre_at = None  # where the next GDN layer's replay instructions go: after the previous layer's gate_up
        record_ptr = self.record.data_ptr()
        hidden_size = K_NORM
        layers = view.layers

        # the previous launched step's outputs, then the step's inputs first (launch(prev_launch=..., book_req=...))
        if self.has_book_op:
            schedule.add_prev(
                rec=record_ptr, ring=self.ss_ring, host=self.ss_host, flags=self.ss_flags, post=self.post_tensors
            )
            schedule.add_book(rec=record_ptr, ring=self.ss_ring, **view.book_tensors)

        # newly allocated blocks: vLLM's KVBlockZeroer segments (attention layers), zeroed by the step itself
        zero_ctr = None
        if view.zero_meta is not None:
            meta = view.zero_meta
            zero_ctr = schedule.add_zero(seg_addrs=meta[0], seg_strides=meta[1], seg_pages=meta[2], rec=record_ptr)

        embeddings = torch.zeros(NT, hidden_size, dtype=torch.bfloat16, device=device)
        embed_ctr = schedule.add_embed(table=view.embed, out=embeddings, rec=record_ptr)
        dep, in_proj_outputs = self.in_proj(
            schedule, layers[0], dep=embed_ctr, pro=dict(pro=Pro.NORM_IN, in0=embeddings)
        )
        residual = embeddings
        schedule.keep.append(embeddings)

        finisher_x = None  # (shared B) the previous finisher's x copies: the next finisher's residual (link_shared_b)
        for i, layer in enumerate(layers):
            if isinstance(layer.core, AttentionCore):
                out_proj_ctr, out_proj_desc = self.attention(
                    schedule, layer, dep, in_proj_outputs, record_ptr, zero_ctr
                )
            else:
                out_proj_ctr, out_proj_desc = self.gdn(schedule, layer, dep, in_proj_outputs, record_ptr, device)

            # the post-attention residual add + RMSNorm, the MLP
            post_norm, gate_up, down = layer.post_norm, layer.gate_up, layer.down
            post_attention_residual = torch.zeros(NT, hidden_size, dtype=torch.float32, device=device)
            schedule.keep.append(post_attention_residual)
            gate_up_ctr, gate_up_desc = schedule.add_gemv(
                w=gate_up.weight,
                kind=Kind.NVFP4,
                K=hidden_size,
                dep=out_proj_ctr,
                sf=gate_up.block_scales,
                alpha=gate_up.alpha,
                gs=gate_up.input_global_scale_inv,
                eps=post_norm.eps,
                pro=Pro.NORM_RESID,
                src=out_proj_desc,
                nw=post_norm.weight,
                x_out=post_attention_residual,
                res_in=residual,
                res_bf16=True,
                act_gs=down.input_global_scale_inv,
            )
            # o_proj / out_proj's finishers build gate_up's B (the residual: the previous down's x copies)
            finisher_x = schedule.link_shared_b(
                out_proj_desc, gate_up_desc, rin=finisher_x, ssq_prev=schedule.shared_b_last_ssq
            )
            schedule.shared_b_consumer(gate_up_desc)
            self.pre_at = schedule.mark()
            down_ctr, down_desc = schedule.add_gemv(
                w=down.weight,
                kind=Kind.NVFP4,
                K=down.weight.shape[1] * 2,  # (two e2m1 values a byte)
                dep=gate_up_ctr,
                sf=down.block_scales,
                alpha=down.alpha,
                gs=down.input_global_scale_inv,
                pro=Pro.ACT,
                src=gate_up_desc,
                in0=schedule.acc[gate_up_desc]["act"],
                is_exact=True,
                val=True,
                fin=True,
            )

            if i + 1 < len(layers):
                residual = torch.zeros(NT, hidden_size, dtype=torch.bfloat16, device=device)
                schedule.keep.append(residual)
                if i + 1 in view.aux_layers:  # (the residual stream before layer i + 1: its in_proj prologue's r_out)
                    self.aux.append(residual)
                dep, in_proj_outputs = self.in_proj(
                    schedule,
                    layers[i + 1],
                    dep=down_ctr,
                    pro=dict(pro=Pro.NORM_RESID, src=down_desc, res_in=post_attention_residual, r_out=residual),
                )
                next_is_attn = isinstance(layers[i + 1].core, AttentionCore)
                # down's finishers build the next layer's in_qkv / qkv B (+ bf16(x_n) for in_proj_ba)
                # (an aux layer: the consumer's prologue writes no r_out: the finishers write the draft's residual rows,
                # and the draft waits for their count instead of in_proj's)
                finisher_x = schedule.link_shared_b(
                    down_desc,
                    in_proj_outputs[0],
                    rin=finisher_x,
                    xn=not next_is_attn,
                    aux=residual if i + 1 in view.aux_layers else None,
                    ssq_prev=schedule.shared_b_last_ssq,
                    ba_w=None if next_is_attn else in_proj_outputs[2],
                )
                schedule.shared_b_consumer(in_proj_outputs[0])
                if not next_is_attn:  # (in_proj_z: in_proj_qkv's twin, staged from the same B by its prologue)
                    schedule.shared_b_twin(in_proj_outputs[1], in_proj_outputs[0])
                if i + 1 in view.aux_layers:  # (its counter releases the residual slices: the draft's fc waits on it)
                    self.aux_ctr = schedule.draft_rows_ctr

        # the final RMSNorm + the LM head, the sampler
        final_norm, lm_head = view.final_norm, view.lm_head
        lm_head_ctr, lm_head_desc = schedule.add_gemv(
            w=lm_head.weight,
            kind=Kind.NVFP4,
            K=hidden_size,
            dep=down_ctr,
            sf=lm_head.block_scales,
            alpha=lm_head.alpha,
            gs=lm_head.input_global_scale_inv,
            eps=final_norm.eps,
            pro=Pro.NORM_RESID,
            src=down_desc,
            nw=final_norm.weight,
            res_in=post_attention_residual,
            logits=self.logits.view(-1),
        )
        # the last down's finishers build the LM head's B
        schedule.link_shared_b(down_desc, lm_head_desc, rin=finisher_x, xs=False, ssq_prev=schedule.shared_b_last_ssq)
        schedule.shared_b_consumer(lm_head_desc)
        self.sampler_ctr = schedule.add_spec_sample(
            lm_ctr=lm_head_ctr, logits=self.logits, ring=self.ss_ring, rec=record_ptr
        )
        schedule.keep += [self.record, self.logits]

        return schedule

    def in_proj(self, schedule: Schedule, layer: Layer, *, dep: int, pro: dict) -> tuple[int, list]:
        """Layer `layer`'s input RMSNorm + projections (FP8 qkv / qkvz, bf16 ba); returns (dependency, [desc, ...])."""
        in_norm = layer.in_norm
        gemv_args = dict(eps=in_norm.eps, nw=in_norm.weight, K=K_NORM, is_aligned=True, **pro)
        if isinstance(layer.core, AttentionCore):
            qkv = layer.core.qkv
            proj_ctr, proj_desc = schedule.add_gemv(
                w=qkv.weight, kind=Kind.FP8, dep=dep, alpha=qkv.alpha, gs=qkv.input_scale, **gemv_args
            )
            return proj_ctr, [proj_desc]

        core = layer.core
        qkvz, ba_weight = core.in_proj, core.ba_weight
        if pro["pro"] == Pro.NORM_IN:
            # the first layer, on the embedding: in_proj_qkvz and in_proj_ba as ops (the core reads both outputs)
            proj_ctr, proj_desc = schedule.add_gemv(
                w=qkvz.weight, kind=Kind.FP8, dep=dep, alpha=qkvz.alpha, gs=qkvz.input_scale, **gemv_args
            )
            gemv_args.pop("is_aligned")
            ba_bytes = ba_weight.view(torch.uint8).view(ba_weight.shape[0], -1)
            ba_ctr, ba_desc = schedule.add_gemv(
                w=ba_bytes, kind=Kind.BF16, dep=dep, alpha=1.0, gs=1.0, is_exact=True, **gemv_args
            )
            return ba_ctr, [proj_desc, ba_desc]  # ba_ctr completes after proj_ctr on every SM

        # in_proj_qkv (the rows the core reads; its sums as tagged words per K part, its prologue leaving the normed
        # input for the core's b / a), then in_proj_z on the same B operand: the out_proj polls its outputs as the gate
        z_rows = GDN_HV * GDN_V
        qkvz_weight = qkvz.weight
        assert qkvz_weight.shape[0] == core.conv_weight.shape[0] + z_rows
        proj_ctr, proj_desc = schedule.add_gemv(
            w=qkvz_weight[:-z_rows],
            kind=Kind.FP8,
            dep=dep,
            alpha=qkvz.alpha,
            gs=qkvz.input_scale,
            has_ll_parts=True,
            **gemv_args,
        )
        _, z_desc = schedule.add_gemv(
            w=qkvz_weight[-z_rows:],
            kind=Kind.FP8,
            K=K_NORM,
            dep=NO_CTR,
            alpha=qkvz.alpha,
            gs=qkvz.input_scale,
            pro=Pro.COPY_B,
            is_exact=True,
            balance=proj_desc,
            has_ll_out=True,
        )

        return proj_ctr, [proj_desc, z_desc, ba_weight]

    def attention(
        self, schedule: Schedule, layer: Layer, dep: int, in_proj_outputs: list, record_ptr: int, zero_ctr: int | None
    ) -> tuple[int, int]:
        """A full-attention layer's core and o_proj; returns the o_proj's (counter, desc)."""
        core = layer.core
        kv_cache = core.kv_cache
        assert isinstance(kv_cache, torch.Tensor) and kv_cache.dtype == torch.bfloat16 and kv_cache.dim() == 4, (
            "a bf16 HND KV cache"
        )
        assert self.block_size % kv_cache.shape[2] == 0
        o_proj = core.o_proj

        combine_ctr = schedule.add_attn(
            dep=dep,
            src=in_proj_outputs[0],
            kv_cache=kv_cache,
            kb_per_mb=self.block_size // kv_cache.shape[2],
            q_norm_w=core.q_norm,
            k_norm_w=core.k_norm,
            cos_sin=core.cos_sin,
            rec=record_ptr,
            scale=core.scale,
            eps=core.eps,
            sections=MROPE_SECTIONS,
            zero_ctr=zero_ctr,
            o_s_in=o_proj.input_scale,
            att_group=self.attn_groups.index(core.group),
        )

        # the combine publishes the gated FP8 operand
        return schedule.add_gemv(
            w=o_proj.weight,
            kind=Kind.FP8,
            K=o_proj.weight.shape[1],
            dep=combine_ctr,
            alpha=o_proj.alpha,
            gs=o_proj.input_scale,
            pro=Pro.ACT,
            in0=schedule.act_of_ctr[combine_ctr],
            is_exact=True,
            val=True,
            fin=True,
        )

    def gdn(
        self, schedule: Schedule, layer: Layer, dep: int, in_proj_outputs: list, record_ptr: int, device: torch.device
    ) -> tuple[int, int]:
        """A Gated DeltaNet layer's core (OP_GDNT) and out_proj; returns the out_proj's (counter, desc)."""
        core = layer.core
        conv_state, ssm_state = core.conv_state, core.ssm_state
        conv_dim = core.conv_weight.shape[0]
        assert conv_state.dtype == torch.bfloat16 and ssm_state.dtype == torch.float32 and ssm_state.dim() == 4
        # the (state_len, dim) conv state (SD): the window's 3 entries, then (with a draft) one per speculative token
        assert conv_state.shape[1] >= GDN_CONV_TAPS and conv_state.shape[2] == conv_dim, (
            "the (state_len, dim) conv state"
        )
        assert conv_state.stride(1) == conv_dim and conv_state.stride(2) == 1
        assert ssm_state[0].is_contiguous(), "a slot's recurrent state [heads][v][k]"

        core_output = torch.zeros(NT, GDN_HV * GDN_V, dtype=torch.bfloat16, device=device)
        # the core's output as tagged words (gdn_t.cuh ll: a bf16 pair each)
        tagged_output = torch.zeros(NT, GDN_HV * GDN_V // 2, dtype=torch.int64, device=device)
        # (the first layer: b / a and the qkvz sums from ops; the later ones: see in_proj)
        is_first = len(in_proj_outputs) == 2
        group = self.mamba_groups.index(core.group)
        # b / a: layer 0's in_proj_ba op; the later layers' from in_proj_qkv's prologue (its sums in smem)
        ba_source = dict(src_ba=in_proj_outputs[1]) if is_first else dict(ba_w=in_proj_outputs[2], reads_src_words=True)
        out_proj = core.out_proj
        running_state = torch.zeros(GDN_HV, GDN_V, GDN_K, dtype=torch.float32, device=device)
        replay_record = torch.zeros(2, GDNT_REC_BYTES, dtype=torch.uint8, device=device)

        # out_proj's gated RMSNorm in the core's output phase (GT_GNORM) writes out_proj's FP8 operand, 2 copies by
        # launch parity at 0xFF until written (no code is 0xFF), which out_proj copies (PRO_ACT) without a dependency,
        # re-polling the bytes still 0xFF; the gate z from in_proj_z's tagged words (layer 0: the qkvz sums)
        out_proj_operand = torch.full((2, GDN_HV * CHUNK_B), self.ACT_SENTINEL, dtype=torch.uint8, device=device)
        z_words = None if is_first else schedule.acc[in_proj_outputs[1]]["llw"]
        # (ssx: the heads' partial sums of squares, exchanged)
        gated_norm = dict(
            act=out_proj_operand,
            act_stride=GDN_HV * CHUNK_B,
            nw=core.norm.weight,
            eps=core.norm.eps,
            s_in=out_proj.input_scale,
            ssx=torch.zeros(GDN_HV * self.GDN_HEAD_ITEMS * NT, dtype=torch.int64, device=device),
            zw=z_words,
            zrs=0 if z_words is None else z_words.numel() // NT,
        )
        schedule.add_gdn_t(
            dep=dep,
            conv_w=core.conv_weight,
            conv_state=conv_state,
            ssm_state=ssm_state,
            a_log=core.a_log,
            dt_bias=core.dt_bias,
            out=core_output,
            ll=tagged_output,
            scale=core.scale,
            src_qkvz=in_proj_outputs[0],
            run=running_state,
            rec=replay_record,
            step=record_ptr + self.INT32_BYTES * SR_GDNT,
            group=group,
            gnorm=gated_norm,
            pre_at=None if is_first else self.pre_at,
            **ba_source,
        )

        out_proj_ctr, out_proj_desc = schedule.add_gemv(
            w=out_proj.weight,
            kind=Kind.FP8,
            K=out_proj.weight.shape[1],
            dep=NO_CTR,
            alpha=out_proj.alpha,
            gs=out_proj.input_scale,
            pro=Pro.ACT,
            in0=out_proj_operand,
            act_stride=GDN_HV * CHUNK_B,
            is_exact=True,
            val=True,
            fin=True,
        )
        schedule.gn_out.add(out_proj_desc)

        return out_proj_ctr, out_proj_desc

    def draft_launch(self, draft_args: Callable[[int], int] | None, slot: int) -> tuple[int, int, int, int]:
        """The launch's draft arguments (megakernel.cu MKArgs): the DkArgs address (0: no draft in the launch), the SMs'
        first draft instructions, the sampler's and the last aux layer's counters."""
        # draft_args(slot): the device address of the draft's arguments for the sampler outputs of ring slot `slot`
        args_ptr = draft_args(slot) if draft_args is not None else 0

        return args_ptr, self.tensors["sm_draft"].data_ptr(), self.sampler_ctr, self.aux_ctr

    def spec_outputs(self, slot: int, T: int) -> tuple[torch.Tensor, ...]:
        """The spec sampler's outputs of a T-row step in ring slot `slot`: sampled [1, T] int64, num_sampled /
        num_rejected [1] int32."""
        ring = self.ss_ring

        return (
            ring["sampled"][slot, :T].view(1, T),
            ring["num_sampled"][slot : slot + 1],
            ring["num_rejected"][slot : slot + 1],
        )

    @staticmethod
    def spec_ring(device: torch.device) -> dict[str, torch.Tensor]:
        """The spec sampler's outputs per ring slot (schedule.py SS_RING: NT rows each)."""
        return dict(
            sampled=torch.zeros(RING, NT, dtype=torch.int64, device=device),
            num_sampled=torch.zeros(RING, dtype=torch.int32, device=device),
            num_rejected=torch.zeros(RING, dtype=torch.int32, device=device),
        )

    @staticmethod
    def int32(x: int) -> int:
        """The int32 with the low 32 bits of x."""
        return ((x & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
