import math

import torch

from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.quant import Fp4Weight
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.schedule import (
    NB,
    NO_CTR,
    QB,
    QKV_SLOTS,
    ArgBlocks,
    BSrc,
    DraftSchedule,
    Epi,
    Resid,
)
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.weights import DraftWeights

HID, INTER, NQH, NKVH, HDIM = 5120, 17408, 32, 8, 128  # the DFlash 2 head's sizes (csrc dk_layout.cuh)
NSTEP, TOPK, RANK = QB - 1, 16, 256  # drafts per proposal at most, candidates per position, the selector's rank
ATT_ROWS = (NQH // NKVH) * QB  # query rows per KV head (4 heads x the block's tokens)
ATT_MAXK, ATT_MAX_SPLITS = 112, 32  # an attention item's keys and a head's splits at most (dk_attention.cuh)
LOG2E = 1.4426950408889634  # (the attention's softmax in base 2)
TOPK_THREADS, TOPK_VEC = 1024, 8  # FlashInfer's radix top-k: CTA threads, bf16 values per 16-byte load
TOPK_FIXED_WORDS = 256 + 256 + 5  # its histogram, suffix sums and scalars (uint32)
TOPK_SCAN_BYTES = 4 * (TOPK_THREADS + TOPK_THREADS // 32)  # cub's raking scan storage for 1024 threads (upper bound)


class DraftProposal:
    """The DFlash 2 draft's proposal as one launch of the draft megakernel: the context update of the committed tokens,
    the block forward, the candidates, the selector's walk and the draft logits (csrc/)."""

    # the GEMVs whose weights stream without waiting for the release of the SM's previous GEMV (add_gemv's is_gated),
    # by their names in build: the LM head (the selector projection before it releases only to the selector)
    NOGATE = frozenset({"lm_head"})
    CONV_GROUPS = 320  # the kernel projections' groups per (side, tap) (csrc dk_layout.cuh CONV_GROUPS)
    # the projections run in 4-bit (NVFP4 weights quantized at load, NVFP4 activations): the activations' global scale
    # per projection (the e4m3 block scales alone at 1)
    ACT_GS = 1.0
    # down's bands' chunks at most: its K chunks in 2 bands of 34, so that an SM's stages touch at most 2 M tiles of at
    # most 6 split-K parts each
    DOWN_BAND_CHUNKS = 34

    # Inputs per launch: the target's aux hidden states of the step's tokens, vLLM's device tensors of the step
    # (positions, rejected count, the anchor, sampling state, the draft's block table row); outputs: the drafts and
    # their sparse logits in vLLM's speculator buffers, the committed tokens' K / V in the draft's KV cache
    def __init__(
        self,
        weights: DraftWeights,
        caches: list[torch.Tensor],
        *,
        vocab: int,
        mask_token: int,
        max_model_len: int,
        kv_block: int,
        window: int,
        device: torch.device,
        nstep: int,
    ) -> None:
        assert 1 <= nstep <= NSTEP, "drafts per proposal (the kernel's B rows: the block of nstep + 1 tokens, padded)"
        assert len(caches) == len(weights.layers)
        for cache in caches:
            assert cache.dtype == torch.bfloat16 and cache.is_contiguous()
            assert tuple(cache.shape[1:]) == (NKVH, kv_block, 2 * HDIM)

        self.weights, self.device, self.nstep = weights, device, nstep
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.n_layers = len(weights.layers)
        # (the attention: one item per SM, a KV head's keys split over n_splits)
        self.n_splits = self.num_sms // NKVH
        window_left = window - 1
        assert self.n_splits <= ATT_MAX_SPLITS and math.ceil((window_left + QB) / self.n_splits) <= ATT_MAXK

        self.fp4_weights = weights.quantized()  # the projections in NVFP4
        self.buffers()
        self.schedule = self.build(caches)
        self.tensors = self.schedule.finalize()
        self.empty_stamps = torch.empty(0, dtype=torch.int64, device=device)  # (the launch's tstamp: no role stamps)
        self.tables = ArgBlocks.tables(
            embed=weights.embed, cos_sin=weights.cos_sin, pred_cb=weights.pred_cb, succ_cb=weights.succ_cb, vocab=vocab,
            mask_token=mask_token, max_model_len=max_model_len, kv_block=kv_block, window_left=window_left,
            topk_chunk=self.topk_chunk(vocab), nstep=nstep,
        )  # fmt: skip

    def launch(self, step_args: torch.Tensor) -> None:
        """One proposal on the step's StepArgs block (ArgBlocks.step)."""
        tensors = self.tensors

        torch.ops.pear_deploy_blackwell_nvfp4_draft.run(
            tensors["instrs"],
            tensors["sm_off"],
            tensors["descs"],
            tensors["counters"],
            tensors["ctr_count"],
            tensors["epoch"],
            self.ssq,
            step_args,
            self.tables,
            tensors["progress"],
            self.empty_stamps,
            self.num_sms,
        )

    def buffers(self) -> None:
        """The proposal's intermediate tensors (persistent: the kernel's descriptors point to them)."""

        def buffer(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        num_layers = self.n_layers
        self.ctx = buffer(QB, HID)
        # sums of squares (int64 fixed point: csrc red_add_ssq): ctx, then x2 / x3 of each layer
        self.ssq = buffer(1 + 2 * num_layers, 2, QB, dtype=torch.int64)
        # the kernel projections' outputs [parity 2][side 2][tap 2][group 320][token] (the GEMV's rows): the fp32 sums
        # (csrc E_COEF), each launch's in its parity's half
        self.c_attn = [buffer(2, 2, 2, self.CONV_GROUPS, QB, dtype=torch.float32) for _ in range(num_layers)]
        self.c_mlp = [buffer(2, 2, 2, self.CONV_GROUPS, QB, dtype=torch.float32) for _ in range(num_layers)]
        self.queries = [buffer(QB, NQH, HDIM) for _ in range(num_layers)]
        # qkv's sums [parity 2][slot][tile 48][column NB][row 128] (fp32, csrc E_QKV): the attention items sum the slots
        self.qkv_sum = [
            buffer(2, QKV_SLOTS, (NQH + 2 * NKVH), NB, HDIM, dtype=torch.float32) for _ in range(num_layers)
        ]
        # the block's k | v rows (the cache's row layout)
        self.kvq = [buffer(QB, NKVH, 2, HDIM) for _ in range(num_layers)]
        self.head_ctr = [buffer(NKVH, dtype=torch.int32) for _ in range(num_layers)]
        self.part_o = [buffer(self.n_splits, NKVH, ATT_ROWS, HDIM, dtype=torch.float32) for _ in range(num_layers)]
        self.part_ml = [buffer(self.n_splits, NKVH, ATT_ROWS, 2, dtype=torch.float32) for _ in range(num_layers)]
        self.attn_out = [buffer(QB, NQH, HDIM) for _ in range(num_layers)]
        self.x2 = [buffer(QB, HID, dtype=torch.float32) for _ in range(num_layers)]
        self.x3 = [buffer(QB, HID, dtype=torch.float32) for _ in range(num_layers)]
        self.act = [buffer(QB, INTER) for _ in range(num_layers)]
        # the kernel projections' normed rows (bf16(RMSNorm(x)), the convs' inputs): the two-phase GEMVs' first phases
        self.xn_a = [buffer(QB, HID) for _ in range(num_layers)]
        self.xn_m = [buffer(QB, HID) for _ in range(num_layers)]
        self.hsel = buffer(NSTEP, RANK)
        lm_k = 2 * self.weights.lm_weight.shape[1]
        self.fp4_image = buffer(DraftSchedule.fp4_image_bytes(lm_k), dtype=torch.uint8)  # the LM head's NVFP4 B operand
        self.topk_part = buffer(self.num_sms, NSTEP, TOPK, dtype=torch.int64)
        self.candidates = buffer(NSTEP, TOPK, dtype=torch.int64)
        self.unary_scores = buffer(NSTEP, TOPK, dtype=torch.float32)
        self.scores = buffer(NSTEP, TOPK, TOPK, dtype=torch.float32)

    def build(self, caches: list[torch.Tensor]) -> DraftSchedule:
        """The schedule of the whole proposal: the fc, the draft's layers, the LM head's candidates and the selector."""
        weights, schedule = self.weights, DraftSchedule(self.num_sms, self.device)
        fp4 = self.fp4

        def is_gated(name: str) -> bool:
            return name not in self.NOGATE

        ssq_ctx = self.ssq[0]
        fc_ctr = schedule.add_gemv(**fp4("fc"), n=QB, bsrc=BSrc.AUX, epi=Epi.FC, b=[], e=[self.ctx, ssq_ctx])
        schedule.add_prep()  # after the fc's B operand: layer 0 reads it

        down_ctr = NO_CTR  # the previous layer's down projection
        for i, layer_weights in enumerate(weights.layers):
            is_first = i == 0  # (layer 0's B operands from the query block's embedding rows: xemb)
            x_in, ssq_in = (0, 0) if is_first else (self.x3[i - 1], self.ssq[2 * i])
            ssq2, ssq3 = self.ssq[1 + 2 * i], self.ssq[2 + 2 * i]
            a_kp_ctr = schedule.add_gemv(
                **fp4(f"L{i}.a_kp"),
                n=QB,
                bsrc=BSrc.NORM,
                epi=Epi.COEF,
                b=[x_in, ssq_in, layer_weights.in_norm, self.xn_a[i]],
                e=[self.c_attn[i]],
                dep=down_ctr,
                layer=i,
                xemb=is_first,
                is_gated=is_gated("kp_a0" if is_first else "kp_a"),
            )
            qkv_ctr = schedule.add_gemv(
                **fp4(f"L{i}.qkv"),
                n=NB,
                bsrc=BSrc.QKV,
                epi=Epi.QKV,
                b=[
                    self.xn_a[i],
                    ssq_in,
                    layer_weights.in_norm,
                    self.c_attn[i],
                    layer_weights.a_base,
                    self.ctx,
                    weights.hidden_norm,
                    ssq_ctx,
                ],
                e=[self.qkv_sum[i]],
                dep=a_kp_ctr,
                dep2=fc_ctr,
                layer=i,
                xemb=is_first,
                is_gated=is_gated("qkv"),
                xn=True,
            )
            attention_ctr = schedule.add_attention(
                q=self.queries[i],
                kvq=self.kvq[i],
                sums=self.qkv_sum[i],
                q_norm=layer_weights.q_norm,
                k_norm=layer_weights.k_norm,
                cache=caches[i],
                part_o=self.part_o[i],
                part_ml=self.part_ml[i],
                out=self.attn_out[i],
                head_ctr=self.head_ctr[i],
                n_splits=self.n_splits,
                layer=i,
                scale_log2=HDIM**-0.5 * LOG2E,
                dep=qkv_ctr,
            )
            o_ctr = schedule.add_gemv(
                **fp4(f"L{i}.o"),
                n=QB,
                bsrc=BSrc.COPY,
                epi=Epi.RESID,
                b=[self.attn_out[i]],
                e=[self.x2[i], ssq2, 0 if is_first else self.x3[i - 1], self.c_attn[i], layer_weights.a_base],
                dep=attention_ctr,
                layer=i,
                resid=Resid.EMBED if is_first else Resid.BUF_BF16,
                is_gated=is_gated("o"),
            )
            m_kp_ctr = schedule.add_gemv(
                **fp4(f"L{i}.m_kp"),
                n=QB,
                bsrc=BSrc.NORM,
                epi=Epi.COEF,
                b=[self.x2[i], ssq2, layer_weights.post_norm, self.xn_m[i]],
                e=[self.c_mlp[i]],
                dep=o_ctr,
                layer=i,
                is_gated=is_gated("kp_m"),
            )
            # (gate_up's rows in paired tiles: quantized so, DraftWeights.row_order)
            gate_up_ctr = schedule.add_gemv(
                **fp4(f"L{i}.gate_up"),
                n=QB,
                bsrc=BSrc.NORM_CONV,
                epi=Epi.SILU,
                b=[self.xn_m[i], ssq2, layer_weights.post_norm, self.c_mlp[i], layer_weights.m_base],
                e=[self.act[i]],
                dep=m_kp_ctr,
                layer=i,
                is_gated=is_gated("gate_up"),
                xn=True,
            )
            down_ctr = schedule.add_gemv(
                **fp4(f"L{i}.down"),
                n=QB,
                bsrc=BSrc.COPY,
                epi=Epi.RESID,
                b=[self.act[i]],
                e=[self.x3[i], ssq3, self.x2[i], self.c_mlp[i], layer_weights.m_base],
                dep=gate_up_ctr,
                layer=i,
                resid=Resid.BUF_FP32,
                conv_fma=True,
                is_gated=is_gated("down"),
                max_band=self.DOWN_BAND_CHUNKS,
            )

        x_last, ssq_last = self.x3[-1], self.ssq[2 * self.n_layers]
        lm_k = 2 * weights.lm_weight.shape[1]
        quant_ctr = schedule.add_quant(
            K=lm_k,
            b=[x_last, ssq_last, weights.norm],
            image=self.fp4_image,
            gs=weights.lm_input_global_scale_inv,
            dep=down_ctr,
        )
        hsel_ctr = schedule.add_gemv(
            **fp4("sel_proj"),
            n=QB,
            bsrc=BSrc.FINAL,
            epi=Epi.BF16,
            b=[x_last, ssq_last, weights.norm],
            e=[self.hsel],
            dep=down_ctr,
            n_out=self.nstep,
            is_gated=is_gated("hsel"),
        )
        lm_head_ctr = schedule.add_gemv(
            w=weights.lm_weight,
            K=lm_k,
            n=QB,
            bsrc=BSrc.FP4,
            epi=Epi.LOGITS,
            b=[x_last, ssq_last, weights.norm],
            e=[self.topk_part, self.fp4_image],
            dep=down_ctr,
            dep2=quant_ctr,
            sf=weights.lm_block_scales.view(torch.uint8),
            alpha=weights.lm_alpha,
            gs=weights.lm_input_global_scale_inv,
            is_gated=is_gated("lm_head"),
        )
        schedule.add_select(
            part=self.topk_part,
            cand=self.candidates,
            unary=self.unary_scores,
            hsel=self.hsel,
            scores=self.scores,
            n_part=self.num_sms,
            dep=lm_head_ctr,
            dep2=hsel_ctr,
        )

        return schedule

    def fp4(self, name: str) -> dict:
        """add_gemv's weight arguments for projection `name` in NVFP4."""
        fp4_weight: Fp4Weight = self.fp4_weights[name]

        return dict(
            w=fp4_weight.codes,
            K=fp4_weight.shape[1],
            sf=fp4_weight.scales,
            alpha=1.0 / (fp4_weight.global_scale * self.ACT_GS),
            gs=self.ACT_GS,
        )

    def topk_chunk(self, n: int) -> int:
        """FlashInfer's per-CTA chunk of a bf16 row of n values (RadixTopKMultiCTA, deterministic): its ties order."""
        # the CTA's shared memory past its fixed words (16-byte aligned) and two scan buffers holds the chunk's values
        smem_bytes = torch.cuda.get_device_properties(self.device).shared_memory_per_block_optin
        fixed_bytes = (4 * TOPK_FIXED_WORDS + 15) // 16 * 16
        value_bytes = smem_bytes - fixed_bytes - 2 * TOPK_SCAN_BYTES
        max_chunk = max((value_bytes // 2) // TOPK_VEC * TOPK_VEC, TOPK_VEC * TOPK_THREADS)

        ctas = math.ceil(n / max_chunk)
        chunk = math.ceil(n / ctas)

        return min((chunk + TOPK_VEC - 1) // TOPK_VEC * TOPK_VEC, max_chunk)
