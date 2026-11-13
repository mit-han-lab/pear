from __future__ import annotations

import math
import types
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.quant import (
    ATOM_BLOCKS,
    ATOM_ROWS,
    BLOCK,
    Fp4Weight,
)
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.schedule import (
    DF,
    DRAFT,
    NO_CTR,
    NT,
    TMAP_BYTES,
    Kind,
    Op,
    Pro,
    Schedule,
)
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.step import DecodeStep, Nvfp4Linear

logger = get_logger(__name__, default_level="INFO")

HID, INTER, NQH, NKVH, HD = 5120, 17408, 32, 8, 128  # the DFlash 2 draft's sizes
NSTEP, TOPK, RANK, NL = 7, 16, 256, 5  # drafts per proposal, candidates per position, the selector's rank, layers
ACT_GS = 1.0  # the activations' NVFP4 global scale of every projection (the e4m3 block scales alone at 1)
LOG2E = 1.4426950408889634  # (the attention's softmax in base 2)
TOPK_THREADS, TOPK_VEC = 1024, 8  # FlashInfer's radix top-k: CTA threads, bf16 values per 16-byte load

SR_DR = 112  # the record header's first draft word (csrc/step.cuh SR_DR)
INT32_BYTES = 4  # of an entry of num_accepted_tokens_gpu and of the ring's num_sampled


class DraftWord(IntEnum):
    """The record header's draft words from SR_DR (csrc/step.cuh DraftWord; an address is two words)."""

    DR_SLOT = 0  # the verify step's ring slot
    DR_RING = 2  # the ring's sampled rows
    DR_RING_NS = 4  # the ring's num_sampled
    DR_RING_NR = 6  # the ring's num_rejected
    DR_POS = 8  # the verify step's positions
    DR_SEED = 10  # the request's seed
    DR_TEMP = 12  # the request's temperature
    DR_BT = 14  # the draft's KV cache group's block-table row
    DR_TOKENS = 16  # out: the draft tokens
    DR_LOGITS = 18  # in / out: the draft logits
    DR_CAND = 20  # in / out: the cached candidates
    DR_SCORES = 22  # out: the selector's rows
    DR_NSTEP = 24  # drafts per proposal
    DR_MASK = 25  # the mask token
    DR_VOCAB = 26  # the vocabulary's size
    DR_KVB = 27  # the kernel block's tokens
    DR_WINDOW = 28  # the sliding window's left keys
    DR_WORDS = 32  # the draft's words


class Nvfp4LinearMethod(LinearMethodBase):
    """vLLM's linear method of a draft projection kept only in NVFP4, installed after loading."""

    def __init__(self, nvfp4_weight: Fp4Weight) -> None:
        self.nvfp4_weight = Fp4Weight(nvfp4_weight.codes, nvfp4_weight.scales, float(nvfp4_weight.gs))

    def create_weights(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the draft's NVFP4 projection is installed after loading")

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.nvfp4_weight.linear(x, bias)


@dataclass
class DraftLayer:
    """One draft decoder layer's norms, conv base kernels, NVFP4 projections and FP8 cache."""

    in_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    a_base: torch.Tensor  # attention_conv.base_kernel bf16 [2 sides][2 taps][5120]
    m_base: torch.Tensor
    a_kp: Fp4Weight  # attention_conv.kernel_projection [1280][5120]
    qkv: Fp4Weight  # [6144][5120]: q 32 x 128, k 8 x 128, v 8 x 128
    o_proj: Fp4Weight  # [5120][4096]
    m_kp: Fp4Weight
    gate_up: Fp4Weight  # [34816][5120]: gate rows, then up rows (vLLM's)
    down: Fp4Weight  # [5120][17408]
    cache: torch.Tensor  # the layer's FP8 cache bytes [rows][8 heads][k 128 | v 128]


@dataclass
class DraftView:
    """What the proposal reads: the draft's weights in NVFP4, the target's embedding and LM head, the draft's caches."""

    fc: Fp4Weight
    ctx_kv: Fp4Weight  # the 5 layers' K / V rows of qkv (vLLM's _fused_kv_weight), the layers' own codes
    sel: Fp4Weight  # candidate_selector.hidden_projection [256][5120]
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    k_norms: torch.Tensor  # bf16 [5][128]
    pred_cb: torch.Tensor
    succ_cb: torch.Tensor
    cos_sin: torch.Tensor  # bf16 [positions][128]
    layers: list[DraftLayer]
    eps: float
    mask_token: int
    vocab: int
    window: int  # the sliding window (tokens)
    block_tokens: int  # the draft group's manager block
    kernel_block: int  # its kernel block (the block-table row's ids)
    group: int  # the draft's KV cache group
    max_len: int

    @classmethod
    def from_vllm(cls, runner: GPUModelRunner, nvfp4_projections: dict | None = None) -> DraftView:
        """The DFlash2Speculator's model at the KV cache's initialization (nvfp4_projections: its NVFP4 projections,
        or None)."""
        speculator = runner.speculator
        draft_model = speculator.model
        draft_decoder = draft_model.model
        device = runner.device
        if nvfp4_projections is None:
            nvfp4_projections = cls.quantize(runner, should_install=False)

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device).contiguous()  # (the small tensors on the GPU: the draft may live in host memory)

        context = runner.compilation_config.static_forward_context
        layer_names = draft_model.get_draft_kv_cache_layer_names()
        group_id = speculator.draft_kv_cache_group_id
        kv_cache_spec = speculator.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec

        layers = []
        for i, layer in enumerate(draft_decoder.layers):
            attention = layer.self_attn
            assert (
                attention.head_dim == HD
                and attention.num_heads == NQH
                and attention.num_kv_heads == NKVH
                and not attention.causal
            )
            kv_cache = context[layer_names[i]].kv_cache
            kv_cache = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
            kv_bytes = kv_cache.view(torch.uint8)
            # FlashInfer's NHD: logical [kernel blocks][8 heads][16 tokens][256], memory [blocks][tokens][heads]
            kv_bytes = kv_bytes.permute(0, 2, 1, 3)
            assert kv_bytes.shape[2:] == (NKVH, 2 * HD) and kv_bytes.is_contiguous(), (
                tuple(kv_cache.shape),
                kv_cache.stride(),
            )
            layers.append(DraftLayer(
                in_norm=to_device(layer.input_layernorm.weight),
                post_norm=to_device(layer.post_attention_layernorm.weight),
                q_norm=to_device(attention.q_norm.weight), k_norm=to_device(attention.k_norm.weight),
                a_base=to_device(layer.attention_conv.base_kernel), m_base=to_device(layer.mlp_conv.base_kernel),
                cache=kv_bytes.reshape(-1), **nvfp4_projections["layers"][i],
            ))  # fmt: skip

        fc, sel, ctx_kv = nvfp4_projections["fc"], nvfp4_projections["sel"], nvfp4_projections["ctx_kv"]
        rotary = draft_decoder.layers[0].self_attn.rotary_emb
        assert rotary.is_neox_style and rotary.rotary_dim == HD
        return cls(
            fc=fc, ctx_kv=ctx_kv, sel=sel, hidden_norm=to_device(draft_decoder.hidden_norm.weight),
            norm=to_device(draft_decoder.norm.weight),
            k_norms=torch.stack([to_device(draft_layer.k_norm) for draft_layer in layers]).contiguous(),
            pred_cb=draft_decoder.candidate_selector.predecessor_codebook,
            succ_cb=draft_decoder.candidate_selector.successor_codebook,
            cos_sin=rotary.cos_sin_cache.to(device=device, dtype=torch.bfloat16).contiguous(), layers=layers,
            eps=float(draft_decoder.norm.variance_epsilon), mask_token=int(speculator.parallel_drafting_token_id),
            vocab=int(speculator.vocab_size), window=int(kv_cache_spec.sliding_window),
            block_tokens=int(kv_cache_spec.block_size),
            kernel_block=int(runner.block_tables.kernel_block_sizes[group_id]), group=group_id,
            max_len=int(speculator.max_model_len),
        )  # fmt: skip

    @classmethod
    def quantize(cls, runner: GPUModelRunner, should_install: bool = True) -> dict:
        """The DFlash2Speculator's projections in NVFP4 by name, with should_install vLLM's draft's only copy too."""
        draft_decoder = runner.speculator.model.model
        device = runner.device
        # qkv of every layer under one global scale: the context projection reads their K / V rows as one weight
        qkv_gs = Fp4Weight.global_scale(
            max(float(layer.self_attn.qkv_proj.weight.abs().amax()) for layer in draft_decoder.layers)
        )

        layers, ctx_codes, ctx_scales = [], [], []
        for layer in draft_decoder.layers:
            attention = layer.self_attn
            projections = {}
            for key, linear in (
                ("a_kp", layer.attention_conv.kernel_projection),
                ("qkv", attention.qkv_proj),
                ("o_proj", attention.o_proj),
                ("m_kp", layer.mlp_conv.kernel_projection),
                ("gate_up", layer.mlp.gate_up_proj),
                ("down", layer.mlp.down_proj),
            ):
                nvfp4_weight = Fp4Weight.quantize(linear.weight, qkv_gs if key == "qkv" else None, device)
                if should_install:
                    cls.install_nvfp4(linear, nvfp4_weight)
                projections[key] = nvfp4_weight
            atom_scale_bytes = (
                projections["qkv"].shape[1] // (BLOCK * ATOM_BLOCKS) * Fp4Weight.ATOM_BYTES
            )  # per ATOM_ROWS rows
            ctx_codes.append(projections["qkv"].codes[NQH * HD :])
            ctx_scales.append(projections["qkv"].scales.view(-1, atom_scale_bytes)[NQH * HD // ATOM_ROWS :].reshape(-1))
            layers.append(projections)

        fc = Fp4Weight.quantize(draft_decoder.fc.weight, None, device)
        sel = Fp4Weight.quantize(draft_decoder.candidate_selector.hidden_projection.weight, None, device)
        ctx_kv = Fp4Weight(torch.cat(ctx_codes).contiguous(), torch.cat(ctx_scales).contiguous(), qkv_gs)
        if should_install:
            cls.install_nvfp4(draft_decoder.fc, fc)
            cls.install_nvfp4(draft_decoder.candidate_selector.hidden_projection, sel)
            cls.install_context_kv(draft_decoder, ctx_kv)
            torch.cuda.empty_cache()

        return dict(layers=layers, fc=fc, sel=sel, ctx_kv=ctx_kv)

    @staticmethod
    def install_nvfp4(linear: LinearBase, nvfp4_weight: Fp4Weight) -> None:
        """Keep vLLM's projection `linear` only in NVFP4: its bf16 weight freed, its forward on nvfp4_weight."""
        linear.quant_method = Nvfp4LinearMethod(nvfp4_weight)
        linear.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.bfloat16, device=nvfp4_weight.codes.device), requires_grad=False
        )

    @staticmethod
    def install_context_kv(draft_decoder: torch.nn.Module, ctx_kv: Fp4Weight) -> None:
        """Run vLLM's fused context K / V projection on the NVFP4 rows the megakernel's context projection reads."""

        # DFlashQwen3Model._project_context_kv with the NVFP4 GEMM over the layers' K / V rows of qkv
        def _project_context_kv(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
            normed_states = torch.empty_like(context_states)
            vllm_ops.rms_norm(normed_states, context_states, self._hidden_norm_weight, self._rms_norm_eps)
            kv_projection = ctx_kv.linear(normed_states, self._fused_kv_bias)
            all_kv = (
                kv_projection.view(num_ctx, num_layers, 2, num_kv_heads, head_dim).permute(2, 1, 0, 3, 4).contiguous()
            )
            return all_kv[0], all_kv[1]

        draft_decoder._project_context_kv = types.MethodType(_project_context_kv, draft_decoder)
        draft_decoder._fused_kv_weight = None  # (its bf16 fused copy freed)


class DraftStep:
    """The draft's proposal as ops of the verify step's fused schedule, after the target's (step.attach_draft)."""

    TOPK_CHUNKS = 24  # D_TOPK's vocabulary chunks per drafted position (at most csrc draft.cuh DQ_SEL_CHUNKS)
    EMBED_PARTS = 8  # the query block's embedding: the anchor's table row (host memory) in this many instructions
    MAX_SPLITS = 32  # D_ATTN's splits per KV head at most (csrc draft.cuh DQ_MAX_SPLITS)
    ATT_ROWS = NT * 4  # D_ATTN's query rows per KV head: its 4 query heads x the block's tokens
    ATT_TILE_KEYS = 64  # D_ATTN's tensor-map box: the cache rows of a tile (csrc draft.cuh DQ_TILE)
    KP_ROWS = 1280  # the kernel projections' rows (2 sides x 2 taps x 320 groups)
    KP_MIN_STAGES = 1  # the kernel projections' stages per instruction at least: spread over every SM
    ACT_CHUNK_VALUES = 256  # values per K chunk of a published NVFP4 operand
    ACT_CHUNK_BYTES = NT * 128 + NT * 16  # its chunk's bytes: the codes, then the e4m3 scales
    SEL_EXTRA_STAGES = 1.0  # the LM head's sm_extra on each SM of the selector projection
    NO_STATS = 0  # DraftDesc.stats: the statistics words of a diagnostic build, none
    # FlashInfer's radix top-k: its histogram, suffix sums and scalars (uint32), and cub's raking scan storage for
    # TOPK_THREADS threads (upper bound)
    TOPK_FIXED_WORDS = 256 + 256 + 5
    TOPK_SCAN_BYTES = 4 * (TOPK_THREADS + TOPK_THREADS // 32)

    def __init__(
        self, view: DraftView, step: DecodeStep, embed: torch.Tensor, lm_head: Nvfp4Linear, device: torch.device
    ) -> None:
        self.view, self.step, self.device = view, step, device
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.n_splits = min(self.MAX_SPLITS, self.num_sms // NKVH)
        self.topk_chunk = self.topk_chunk_of(view.vocab, device)
        self.chunk_len = (-(-view.vocab // self.TOPK_CHUNKS) + 7) // 8 * 8  # (DraftDesc.chunk_len: a multiple of 8)
        self.fused_parts = step.build_fused()  # (the fused program's target ops: its schedule and counters)
        self.schedule = self.build(embed, lm_head)
        step.attach_draft(self, self.schedule)  # (its finalize makes the fused program)

    def header(
        self,
        header_words: np.ndarray,
        *,
        slot: int,
        positions: int,
        seed: int,
        temperature: int,
        block_table: int,
        draft_tokens: int,
        draft_logits: int,
        cached_cand: int,
        sel_scores: int,
    ) -> None:
        """The draft's words of a record header (int32 view): the verify step's ring slot, vLLM's tensors' addresses."""
        words = header_words[SR_DR : SR_DR + DraftWord.DR_WORDS]
        addresses = words.view(np.int64)
        ring = self.step.ss_ring
        words[DraftWord.DR_SLOT] = slot
        for word, address in (
            (DraftWord.DR_RING, ring["sampled"].data_ptr()),
            (DraftWord.DR_RING_NS, ring["num_sampled"].data_ptr()),
            (DraftWord.DR_RING_NR, ring["num_rejected"].data_ptr()),
            (DraftWord.DR_POS, positions),
            (DraftWord.DR_SEED, seed),
            (DraftWord.DR_TEMP, temperature),
            (DraftWord.DR_BT, block_table),
            (DraftWord.DR_TOKENS, draft_tokens),
            (DraftWord.DR_LOGITS, draft_logits),
            (DraftWord.DR_CAND, cached_cand),
            (DraftWord.DR_SCORES, sel_scores),
        ):
            addresses[word // 2] = address

        view = self.view
        words[DraftWord.DR_NSTEP], words[DraftWord.DR_MASK], words[DraftWord.DR_VOCAB] = (
            NSTEP,
            view.mask_token,
            view.vocab,
        )
        words[DraftWord.DR_KVB], words[DraftWord.DR_WINDOW] = view.kernel_block, view.window - 1

    @classmethod
    def topk_chunk_of(cls, n: int, device: torch.device) -> int:
        """FlashInfer's per-CTA chunk of a bf16 row of n values (RadixTopKMultiCTA, deterministic): its ties order."""
        # the CTA's shared memory past its fixed words (16-byte aligned) and two scan buffers holds the chunk's values
        smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
        fixed_bytes = (4 * cls.TOPK_FIXED_WORDS + 15) // 16 * 16
        value_bytes = smem - fixed_bytes - 2 * cls.TOPK_SCAN_BYTES
        max_chunk = max((value_bytes // 2) // TOPK_VEC * TOPK_VEC, TOPK_VEC * TOPK_THREADS)

        ctas = math.ceil(n / max_chunk)
        chunk = math.ceil(n / ctas)
        return min((chunk + TOPK_VEC - 1) // TOPK_VEC * TOPK_VEC, max_chunk)

    def build(self, embed: torch.Tensor, lm_head: Nvfp4Linear) -> Schedule:
        """The fused schedule with the proposal's ops: the context, the layers, the candidates and the selector."""
        view, device = self.view, self.device
        # the draft's ops after the target's on every SM: its context ops wait for the last aux state, its query block
        # and its context's cache rows for the sampler's ring slot (the accepted rows, the anchor)
        fused_parts = self.fused_parts
        schedule = fused_parts["schedule"]
        sampler_ctr = fused_parts["sampler_ctr"]
        aux_ctr = fused_parts["aux_ctr"]
        assert aux_ctr != NO_CTR, "the fused draft: the step's last aux state's counter"

        def buffer(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        S = self.n_splits
        buffers = {}
        # the query block's embedding rows (the anchor, the mask tokens) and their sums of squares
        buffers["x0"] = buffer(NT, HID)
        # (the table in host memory: the constant row here)
        buffers["mask_row"] = embed[view.mask_token].to(device).contiguous()
        # the context: fc over the aux hidden states (its bf16 output and sums of squares), hidden_norm, the 5 layers'
        # K / V rows, then k_norm, rope and the committed tokens' cache rows
        aux = self.step.aux_all
        assert aux.shape == (NL, NT, HID) and aux.dtype == torch.bfloat16
        buffers["ctx"] = buffer(NT, HID)

        # fc's instructions before the sampler's on each SM, its slices published (qpub) after the target's LM head;
        # not on the last NT SMs: the embedding's there reads the table in host memory, and an SM's release waits
        # for its outstanding host reads (the context's ops beside layer 0's: sums of their own, acc_ctx)
        schedule.insert_positions = list(fused_parts["pre_sampler"])
        fc_ctr, fc_desc = schedule.add_gemv(
            **self.fp4(view.fc),
            K=NL * HID,
            dep=aux_ctr,
            pro=Pro.NORM_Q,
            in0=aux,
            dflags=DF.PLAIN,
            xr=True,
            r_out=buffers["ctx"],
            acc_pool="acc_ctx",
            sms=self.num_sms - NT,
            has_qpub=True,
        )
        schedule.insert_positions = None
        buffers["ctx_kv"] = buffer(NT, NL * 2 * NKVH * HD)
        ctx_kv_ctr, _ = schedule.add_gemv(
            **self.fp4(view.ctx_kv),
            K=HID,
            dep=fc_ctr,
            pro=Pro.NORM_Q,
            in0=buffers["ctx"],
            x_bf16=True,
            ssq_in=schedule.ssq_of(fc_desc),
            nw=view.hidden_norm,
            eps=view.eps,
            dflags=DF.NW_PLAIN,
            out=buffers["ctx_kv"],
            acc_pool="acc_ctx",
            sms=self.num_sms - NT,
        )
        # the embedding on the SMs without the context's GEMVs (its host read beside their streams); the mask token's
        # rows, which wait for nothing, before the sampler on their SMs
        embed_ctr, embed_ssq = schedule.add_embed(
            table=embed,
            out=buffers["x0"],
            mask_row=buffers["mask_row"],
            dep=sampler_ctr,
            sms=list(range(self.num_sms - NT, self.num_sms)),
            parts=self.EMBED_PARTS,
            early_rows_mark="pre_sampler",
        )
        ctx_ctr = schedule.add_dop(
            Op.D_CTX,
            self.ddesc(
                src=buffers["ctx_kv"],
                cache=[draft_layer.cache for draft_layer in view.layers],
                kn_w=view.k_norms,
                cos_sin=view.cos_sin,
                eps=view.eps,
            ),
            NL * NT,
            ctx_kv_ctr,
            dep2=sampler_ctr,
            keep=[buffers["ctx_kv"], view.k_norms, view.cos_sin] + [draft_layer.cache for draft_layer in view.layers],
        )

        # the layers
        x_in, x_bf16, ssq, ctr = buffers["x0"], True, embed_ssq, embed_ctr
        buffers["q"] = buffer(NKVH, self.ATT_ROWS, HD, dtype=torch.float16)
        buffers["po"] = buffer(NKVH, S, self.ATT_ROWS, HD, dtype=torch.float32)
        buffers["ml"] = buffer(NKVH, S, self.ATT_ROWS, 2, dtype=torch.float32)
        for i, layer in enumerate(view.layers):
            layer_buffers = dict(
                coef_a=buffer(NT, self.KP_ROWS),
                qkv=buffer(NT, NQH * HD + 2 * NKVH * HD),
                act_o=buffer(NQH * HD // self.ACT_CHUNK_VALUES * self.ACT_CHUNK_BYTES, dtype=torch.uint8),
                o=buffer(NT, HID),
                x2=buffer(NT, HID, dtype=torch.float32),
                coef_m=buffer(NT, self.KP_ROWS),
                act_d=buffer(INTER // self.ACT_CHUNK_VALUES * self.ACT_CHUNK_BYTES, dtype=torch.uint8),
                down=buffer(NT, HID),
                x3=buffer(NT, HID, dtype=torch.float32),
            )
            norm_args = dict(pro=Pro.NORM_Q, in0=x_in, x_bf16=x_bf16, ssq_in=ssq, nw=layer.in_norm, eps=view.eps)
            # (the kernel projections over every SM: a stage or two each, the NORM_Q slices small)
            a_kp_ctr, _ = schedule.add_gemv(
                **self.fp4(layer.a_kp),
                K=HID,
                dep=ctr,
                dflags=DF.NW_PLAIN,
                out=layer_buffers["coef_a"],
                min_stages=self.KP_MIN_STAGES,
                **norm_args,
            )
            qkv_ctr, _ = schedule.add_gemv(
                **self.fp4(layer.qkv),
                K=HID,
                dep=a_kp_ctr,
                dflags=DF.NW_PLAIN | DF.CONV,
                coef=layer_buffers["coef_a"],
                base=layer.a_base,
                out=layer_buffers["qkv"],
                **norm_args,
            )
            prep_ctr = schedule.add_dop(
                Op.D_PREP,
                self.ddesc(
                    src=layer_buffers["qkv"],
                    cache=[layer.cache, 0, 0, 0, 0],
                    qn_w=layer.q_norm,
                    kn_w=layer.k_norm,
                    cos_sin=view.cos_sin,
                    q=buffers["q"],
                    eps=view.eps,
                    max_len=view.max_len,
                ),
                NKVH * NT,
                qkv_ctr,
                keep=[layer_buffers["qkv"], layer.q_norm, layer.k_norm, buffers["q"]],
            )
            rows = layer.cache.numel() // (NKVH * 2 * HD)
            tensor_map = torch.ops.pear_deploy_rtx_blackwell_nvfp4.make_tmap3d(
                layer.cache.data_ptr(), rows, NKVH, 2 * HD, self.ATT_TILE_KEYS
            )
            tmap_index = schedule.n_tmaps
            schedule.n_tmaps += 1
            schedule.tmaps[tmap_index * TMAP_BYTES : (tmap_index + 1) * TMAP_BYTES].copy_(tensor_map)
            attn_ctr = schedule.add_dop(
                Op.D_ATTN,
                self.ddesc(
                    cache=[layer.cache, 0, 0, 0, 0],
                    tmap=schedule.tmaps.data_ptr() + tmap_index * TMAP_BYTES,
                    q=buffers["q"],
                    po=buffers["po"],
                    ml=buffers["ml"],
                    scale_log2=HD**-0.5 * LOG2E,
                    n_splits=S,
                    block_tokens=view.block_tokens,
                    prep_ctr=prep_ctr,
                    ctx_ctr=ctx_ctr,
                ),
                NKVH * S,
                prep_ctr,
                keep=[buffers["po"], buffers["ml"]],
            )
            combine_ctr = schedule.add_dop(
                Op.D_COMBINE,
                self.ddesc(po=buffers["po"], ml=buffers["ml"], act=layer_buffers["act_o"], gs=ACT_GS, n_splits=S),
                NQH,
                attn_ctr,
                keep=[layer_buffers["act_o"]],
            )
            # o_proj: x2 = conv side 1 of o + the residual (layer 0: the embedding rows; then bf16 of x3)
            residual_args = (
                dict(res_in=buffers["x0"], res_bf16=True) if i == 0 else dict(res_in=x_in, dflags_extra=DF.RES_ROUND)
            )
            extra_dflags = residual_args.pop("dflags_extra", 0)
            o_ctr, o_desc = schedule.add_gemv(
                **self.fp4(layer.o_proj),
                K=NQH * HD,
                dep=combine_ctr,
                pro=Pro.ACT,
                in0=layer_buffers["act_o"],
                xr=True,
                x_out=layer_buffers["x2"],
                out=layer_buffers["o"],
                dflags=DF.COLFIN | DF.CONV_FIN | extra_dflags,
                coef=layer_buffers["coef_a"],
                base=layer.a_base,
                **residual_args,
            )
            x2_ssq = schedule.ssq_of(o_desc)
            post_norm_args = dict(
                pro=Pro.NORM_Q, in0=layer_buffers["x2"], ssq_in=x2_ssq, nw=layer.post_norm, eps=view.eps
            )
            m_kp_ctr, _ = schedule.add_gemv(
                **self.fp4(layer.m_kp),
                K=HID,
                dep=o_ctr,
                dflags=DF.NW_PLAIN,
                out=layer_buffers["coef_m"],
                min_stages=self.KP_MIN_STAGES,
                **post_norm_args,
            )
            gate_up_ctr, _ = schedule.add_gemv(
                **self.fp4(layer.gate_up),
                K=HID,
                dep=m_kp_ctr,
                dflags=DF.NW_PLAIN | DF.CONV,
                coef=layer_buffers["coef_m"],
                base=layer.m_base,
                out=layer_buffers["act_d"],
                act_gs=ACT_GS,
                **post_norm_args,
            )
            ctr, down_desc = schedule.add_gemv(
                **self.fp4(layer.down),
                K=INTER,
                dep=gate_up_ctr,
                pro=Pro.ACT,
                in0=layer_buffers["act_d"],
                xr=True,
                x_out=layer_buffers["x3"],
                out=layer_buffers["down"],
                res_in=layer_buffers["x2"],
                dflags=DF.COLFIN | DF.CONV_FIN | DF.CONV_FMA,
                coef=layer_buffers["coef_m"],
                base=layer.m_base,
            )
            x_in, x_bf16, ssq = layer_buffers["x3"], False, schedule.ssq_of(down_desc)

        # the final norm: the selector's hidden projection and the target's LM head (8 rows: rows 1..7 the drafts')
        buffers["hsel"] = buffer(NT, RANK)
        final_norm_args = dict(pro=Pro.NORM_Q, in0=x_in, ssq_in=ssq, nw=view.norm, eps=view.eps, dflags=DF.NW_PLAIN)
        hsel_ctr, _ = schedule.add_gemv(**self.fp4(view.sel), K=HID, dep=ctr, out=buffers["hsel"], **final_norm_args)
        buffers["logits"] = buffer(NT, view.vocab)
        # (the selector projection's SMs take one LM head tile fewer: it runs before the LM head there)
        sel_sm_extra = {sm: self.SEL_EXTRA_STAGES for sm in schedule.gemv_counts(view.sel.codes, Kind.NVFP4, HID)}
        lm_head_ctr, _ = schedule.add_gemv(
            w=lm_head.weight,
            sf=lm_head.block_scales,
            kind=Kind.NVFP4,
            K=HID,
            dep=ctr,
            alpha=lm_head.alpha,
            gs=lm_head.input_global_scale_inv,
            out=buffers["logits"].view(-1),
            is_logits=True,
            sm_extra=sel_sm_extra,
            **final_norm_args,
        )
        buffers["parts"] = buffer(NSTEP, self.TOPK_CHUNKS, TOPK, dtype=torch.int64)
        topk_ctr = schedule.add_dop(
            Op.D_TOPK,
            self.ddesc(
                src=buffers["logits"],
                parts=buffers["parts"],
                chunks=self.TOPK_CHUNKS,
                chunk_len=self.chunk_len,
                topk_chunk=self.topk_chunk,
            ),
            NSTEP * self.TOPK_CHUNKS,
            lm_head_ctr,
            keep=[buffers["logits"], buffers["parts"]],
        )
        schedule.add_dop(
            Op.D_SELECT,
            self.ddesc(
                src=buffers["parts"],
                hsel=buffers["hsel"],
                pred_cb=view.pred_cb,
                succ_cb=view.succ_cb,
                chunks=self.TOPK_CHUNKS,
                topk_chunk=self.topk_chunk,
            ),
            1,
            topk_ctr,
            dep2=hsel_ctr,
            sms=[0],
            keep=[buffers["hsel"], view.pred_cb, view.succ_cb],
        )

        return schedule

    def ddesc(self, **fields: Any) -> bytes:
        """A DraftDesc (csrc/draft.cuh) of the fields, the others 0."""
        # cache: DraftDesc.cache's 5 layer pointers
        pointers = (
            [fields.get(k, 0) for k in ("src",)]
            + list(fields.get("cache", [0] * 5))
            + [
                fields.get(k, 0)
                for k in (
                    "tmap",
                    "qn_w",
                    "kn_w",
                    "cos_sin",
                    "q",
                    "po",
                    "ml",
                    "act",
                    "parts",
                    "hsel",
                    "pred_cb",
                    "succ_cb",
                )
            ]
        )
        floats = [fields.get(k, 0.0) for k in ("eps", "scale_log2", "gs", "f3")]
        words = [
            fields.get(k, 0)
            for k in ("n_splits", "block_tokens", "prep_ctr", "ctx_ctr", "chunks", "chunk_len", "topk_chunk", "max_len")
        ]

        return DRAFT.pack(*[Schedule.ptr(x) for x in pointers], *floats, *words, self.NO_STATS)

    @staticmethod
    def fp4(weight: Fp4Weight) -> dict:
        """add_gemv's weight arguments for the projection weight in NVFP4."""
        return dict(
            w=weight.codes,
            sf=weight.scales,
            kind=Kind.NVFP4,
            alpha=1.0 / (weight.gs * ACT_GS),
            gs=ACT_GS,
        )


class DraftLaunch:
    """vLLM's DFlash 2 speculator whose proposal after a verify step the megakernel sampled is made in that step's
    launch (DraftStep, the fused program mk_main<2>)."""

    draft: DraftStep | None = None  # the draft's proposal program (build)
    nvfp4_projections: dict | None = None  # the draft's NVFP4 projections (DraftView.quantize at loading)
    fused_slot: int | None = None  # the ring slot of the launch whose proposal the fused program made, until propose
    slot_ptr = 0  # the last sampled launch's num_sampled slot

    @classmethod
    def install(cls) -> None:
        """Patch the V2 runner's model loading and the DFlash 2 speculator's proposal."""
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

        vllm_propose = DFlash2Speculator.propose
        vllm_load = GPUModelRunner.load_model

        def load_model(self, *args, **kwargs):
            # the draft's projections in NVFP4 only, before vLLM's profiling run compiles the draft and its capture
            # records it (the compiled graph of the bf16 forward would read the freed weights)
            load_result = vllm_load(self, *args, **kwargs)
            speculator = getattr(self, "speculator", None)
            is_dflash2 = type(speculator).__name__ == "DFlash2Speculator"
            if is_dflash2 and speculator.num_speculative_steps + 1 == NT:
                allocated_before = torch.cuda.memory_allocated()
                cls.nvfp4_projections = DraftView.quantize(self)
                logger.info(
                    "the draft's projections kept only in NVFP4: %+.2f GiB",
                    (torch.cuda.memory_allocated() - allocated_before) / 2**30,
                )
            return load_result

        GPUModelRunner.load_model = load_model

        def propose(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=None,
            dummy_run=False,
            skip_attn_for_dummy_run=False,
            mm_inputs=None,
            is_profile=False,
        ):
            if cls.eligible(self, input_batch, aux_hidden_states, num_sampled, dummy_run):
                # the verify step's launch made the proposal: vLLM's buffers hold it
                slot = (num_sampled.data_ptr() - cls.draft.step.ss_ring["num_sampled"].data_ptr()) // INT32_BYTES
                if cls.fused_slot != slot:
                    raise RuntimeError(f"the proposal of ring slot {slot} is not the fused launch's ({cls.fused_slot})")
                cls.fused_slot = None
                return self.draft_tokens[: input_batch.num_reqs]

            return vllm_propose(
                self,
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )

        DFlash2Speculator.propose = propose

    @classmethod
    def build(cls, runner: GPUModelRunner, step: DecodeStep) -> None:
        """Build the draft's proposal program on the verify step's buffers."""
        speculator = runner.speculator
        if speculator is None or type(speculator).__name__ != "DFlash2Speculator":
            raise RuntimeError(f"the draft's proposal needs a DFlash2Speculator, not {type(speculator).__name__}")
        if speculator.num_speculative_steps + 1 != NT:
            raise RuntimeError(f"the draft's proposal makes {NT - 1} tokens, not {speculator.num_speculative_steps}")
        if speculator.draft_logits is None:
            raise RuntimeError(
                "the speculator keeps no draft logits (the proposal writes them, the spec sampler reads them)"
            )

        allocated_before = torch.cuda.memory_allocated()
        view = DraftView.from_vllm(runner, cls.nvfp4_projections)
        cls.draft = DraftStep(view, step, step.view.embed, step.view.lm_head, runner.device)
        free_bytes, _ = torch.cuda.mem_get_info()
        logger.info(
            "the draft's proposal: %d ops, NVFP4 weights and buffers %.2f GiB, %.2f GiB free",
            len(cls.draft.schedule.descs),
            (torch.cuda.memory_allocated() - allocated_before) / 2**30,
            free_bytes / 2**30,
        )

    @classmethod
    def fused_words(cls, runner: GPUModelRunner, input_batch: InputBatch, T: int) -> dict | None:
        """The draft words of a verify step's fused launch, None when eligible's conditions known before the launch
        (one request, NT rows) do not hold."""
        draft, speculator = cls.draft, runner.speculator
        if draft is None or input_batch.num_reqs != 1 or T != NT:
            return None
        if T != speculator.num_query_per_req:
            return None

        states = runner.sampler.sampling_states
        req_idx = int(input_batch.idx_mapping_np[0])
        block_table = runner.block_tables.block_tables[draft.view.group].gpu
        seeds, temperature = states.seeds.gpu, states.temperature.gpu

        return dict(
            positions=input_batch.positions.data_ptr(),
            seed=seeds.data_ptr() + seeds.element_size() * req_idx,
            temperature=temperature.data_ptr() + temperature.element_size() * req_idx,
            block_table=block_table.data_ptr() + block_table.stride(0) * block_table.element_size() * req_idx,
            draft_tokens=speculator.draft_tokens.data_ptr(),
            draft_logits=speculator.draft_logits[req_idx].data_ptr(),
            cached_cand=speculator._cached_candidate_ids[req_idx].data_ptr(),
            sel_scores=speculator._selector_scores.data_ptr(),
        )

    @classmethod
    def eligible(
        cls,
        speculator: DFlash2Speculator,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        dummy_run: bool,
    ) -> bool:
        """A verify step of one request whose launch sampled it (its ring slot) on the megakernel's aux states."""
        draft = cls.draft
        return (
            draft is not None
            and not dummy_run
            and aux_hidden_states is not None
            and aux_hidden_states[0].data_ptr() == draft.step.aux_all.data_ptr()
            and input_batch.num_reqs == 1
            and input_batch.num_tokens == NT == speculator.num_query_per_req
            and num_sampled.data_ptr() == cls.slot_ptr
        )
