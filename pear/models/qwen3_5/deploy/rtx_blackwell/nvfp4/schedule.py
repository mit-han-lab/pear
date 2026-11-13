import heapq
import struct
from enum import IntEnum
from typing import Any

import torch

# the packed layouts of csrc/megakernel.cuh: Instr and the 256-byte OpDesc
INSTR = struct.Struct("<4H6I")
DESC = struct.Struct("<16Q4f18I3Q4I")
ATTN = struct.Struct("<10Q4f7I132x")  # AttnDesc (csrc/attention.cuh)
SSD = struct.Struct("<12Q4I144x")  # SsDesc (csrc/spec_sample.cuh)
DRAFT = struct.Struct("<18Q4f8IQ56x")  # DraftDesc (csrc/draft.cuh)
BOOK = struct.Struct("<15Q4I120x")  # BookDesc (csrc/step.cuh)
PREV = struct.Struct("<9Q2q2I160x")  # PrevDesc (csrc/spec_sample.cuh)
SS_RING = ("sampled", "num_sampled", "num_rejected")  # the spec sampler's outputs, in SsDesc's order
SS_HOST = ("sampled", "num_sampled")  # the host ring's (V2's async output), in PrevDesc's order
SS_SLICE_ELEMS = 2048  # spec_sample.cuh: a slice's elements of the resampled row at most
SS_PART_BYTES = 80  # spec_sample.cuh SsPart
SS_HIST_WORDS = 2048  # spec_sample.cuh: a target chunk's histogram (two 16-bit counts a word)
SS_ROW_BINS = 35587  # spec_sample.cuh: a target row's histogram (a bin of every 16-bit order)
SS_STATE_BYTES = 2 * 16 * 32  # spec_sample.cuh: SS_MAX_T SsFact, then SS_MAX_T SsRow
SS_MAX_VOCAB = 1 << 18  # spec_sample.cuh SS_HIST_BINS: a chunk's histogram holds the nucleus' window below it
LP_MAX_CHUNKS = 32  # logprobs.cuh: chunks per row at most
LP_CHUNK_ELEMS = 2048 * 8  # logprobs.cuh: a chunk's logits at most (LP_CHUNK_VECS 16-byte vectors)
SS_X_VECS = 1480  # spec_sample.cuh: a sampler chunk's 16-byte vectors at most (its staging-buffer copy)
assert ATTN.size == 256 and DRAFT.size == 256
assert INSTR.size == 32 and DESC.size == 256
NO_CTR = 0xFFFF  # an instruction without a dependency / a completion counter
AUX_QPUB_DONE = 1 << 28  # Instr::aux of a GEMV: its NORM_Q slice published by an OP_QPUB (csrc/megakernel.cuh)
NT = 8  # token rows of every activation (the MMAs' N)
TILE = 128  # weight rows per M tile
STAGE_K = 128  # weight bytes per row per stage
CHUNK_B = NT * STAGE_K  # a K chunk of an operand: 8 rows x 128 bytes
SFB_CHUNK = NT * 16  # its NVFP4 scales
B_CHUNKS = 20  # an instruction's K chunks at most (csrc/megakernel.cuh B_CHUNKS)
B_GCH = 5  # B slots per group (csrc B_GCH): a rotated instruction's first stage's B group, which the kernel clamps
K_NORM = 5120  # the hidden size: the RMSNorm prologues' length (csrc K_NORM)
TMAP_BYTES = 128  # a CUtensorMap


class Op(IntEnum):
    GEMV = 0
    ATTN = 3
    ATTN_COMBINE = 4
    EMBED = 5
    ATTN_PREP = 6
    GDNT = 10
    SS_A = 12  # the spec sampler's phases (csrc/spec_sample.cuh)
    SS_B = 13
    SS_C = 14
    NOP = 15
    D_CTX = 16  # the draft's proposal (csrc/draft.cuh)
    D_PREP = 17
    D_ATTN = 18
    D_COMBINE = 19
    D_TOPK = 20
    D_SELECT = 21
    BOOK = 22  # the step's inputs into V2's buffers (csrc/step.cuh book_inputs)
    PREV = 23  # the previous launched step's outputs into the host ring (csrc/spec_sample.cuh prev_outputs)
    QPUB = 24  # a PRO_NORM_Q op's slice published ahead of its GEMV (add_gemv has_qpub)


class Kind(IntEnum):
    NVFP4 = 0
    FP8 = 1
    BF16 = 2


class Pro(IntEnum):
    NORM_IN = 0
    NORM_RESID = 1
    ACT = 3
    NORM_X = 4  # the RMSNorm of x (fp32 / bf16) whose sums of squares its producer published (ssq_in)
    NORM_Q = 5  # NORM_X in two phases: each instruction encodes a slice of a published operand, then copies its chunks


class Epi(IntEnum):
    RED = 0
    LOGITS = 1
    SILU = 2


class DF:
    """The draft's GEMV variants (csrc/megakernel.cuh DFlag)."""

    NW_PLAIN = 1  # the norm's weight as w (Qwen3's RMSNorm), not 1 + w
    PLAIN = 2  # PRO_NORM_Q without the norm: in0's bf16 values in segments of 5120 (aux_all [layer][NT][5120])
    CONV = 4  # PRO_NORM_Q: the grouped conv (side 0) of the normed rows
    COLFIN = 8  # EPI_RED: a column's 8 values finished together
    CONV_FIN = 16  # COLFIN: the conv side 1 plus the residual (x_out fp32, its sums of squares)
    CONV_FMA = 32  # CONV_FIN: fma(c0, o, tail)
    RES_ROUND = 64  # xr: the fp32 residual rounded to bf16 before the add
    DRAFT_TOK = 128  # OP_EMBED: the draft's query block


class Schedule:
    """Instructions per SM (program order = op order on every SM) and op descriptors."""

    # an EPI_RED instruction whose last unit holds only its last k <= ROT_MAX stages processes them first (Instr::aux
    # bits 16..23, csrc rot_k): the short segment's drain then runs under the rest's stream; 0: no rotation
    ROT_MAX = 3
    MAX_TMAPS = 1024  # the schedule's tensor maps at most
    MIN_STAGES = 4  # an instruction's stages at least (a small op uses fewer SMs)
    TMA_ALIGN = 16  # a tensor map's base address and row stride: multiples of 16 bytes
    VEC_ELEMS = 8  # bf16 values per 16-byte vector
    BOOK_IN = 1  # OP_BOOK's Instr::aux: the step's inputs (csrc/step.cuh BOOK_IN)
    BOOK_BT = 2  # OP_BOOK's Instr::aux: the block-table rows (csrc/step.cuh BOOK_BT)
    SS_AUX_A0 = 1  # OP_SS_A's Instr::aux: phase A0 alone (csrc/megakernel.cuh SS_AUX_A0)
    SS_AUX_NO_A0 = 2  # OP_SS_A's Instr::aux: phase A after an A0 instruction (csrc/megakernel.cuh SS_AUX_NO_A0)
    MAX_TILE_PARTS = 256  # parts per M tile below it: a u8 count each (OpDesc::parts)
    EPI_WARPS = 4  # epilogue warps per M tile: DF_COLFIN's counters per tile
    NVFP4_BLOCK = 16  # values per NVFP4 scale
    NORM_GROUP = 4  # values per prologue thread group (a float4): PRO_NORM_RESID's slices hold whole groups
    AUX_ROT_K_SHIFT = 16  # Instr::aux bits 16..23 of an EPI_RED GEMV: its rotated stages (csrc rot_k)
    AUX_ROT_GROUP_SHIFT = 24  # Instr::aux bits 24..27: the first rotated stage's B group (csrc rot_group)
    LOG2E = 1.4426950408889634  # log2(e): the softmax's exp2 scale
    ATT_HKV = 4  # the full attention's KV heads (csrc/attention.cuh ATT_HKV)
    ATT_D = 256  # its head dim (csrc ATT_D)
    ATT_ROWS = 48  # query rows per KV head: 6 query heads x NT tokens (csrc ATT_ROWS)
    ATT_QKV = 14336  # the qkv projection's outputs: queries and their gates, keys, values (csrc ATT_QKV)
    ATT_OUT = 6144  # the attention's output, o_proj's K (csrc ATT_OUT)
    ATT_KV_ROW_BYTES = 512  # a token's k | v of a KV head in the FP8 cache
    ATT_TILE = 64  # keys per tile: the cache tensor map's box rows (csrc ATT_TILE)
    ROTARY_DIM = 64  # cos_sin's columns: cos then sin of the rotary dims
    ATT_COMBINE_ITEMS = 96  # the combine's items: (query head, quarter of its dims)
    MROPE_SECTIONS = (11, 11, 10)  # M-RoPE's (t, h, w) sections: AttnDesc sec1 and sec2 take h and w
    KB_TOKENS = 16  # tokens per kernel block of the cache (AttnDesc kb_tokens)

    def __init__(self, nsm: int, device: torch.device) -> None:
        self.nsm, self.device = nsm, device
        self.logits_ops = 0  # the EPI_LOGITS ops with split tiles so far: their tag bases
        self.descs: list[bytes] = []
        self.sm_instrs: list[list[bytes]] = [[] for _ in range(nsm)]
        self.ctr_count: list[int] = []
        self.keep: list = []
        self.tmaps = torch.zeros(self.MAX_TMAPS * TMAP_BYTES, dtype=torch.uint8, device=device)
        self.n_tmaps = 0
        self.op_info: list[dict] = []  # per GEMV desc: its published sums of squares (ssq_of)
        # scratch shared by ops that never run at once (the step is a chain): the counted sums (each word back to zero
        # by its finisher), the attention's query rows and partials
        self.pools: dict[str, torch.Tensor] = {}
        self.marks: dict[str, list] = {}  # per mark: each SM's instruction count at that point of the build
        self.insert_positions: list | None = None  # per SM where instr() inserts; None: it appends

    # ---- ops
    def add_prev(self, *, ring: dict, host: dict, flags: torch.Tensor, post: dict) -> None:
        """OP_PREV, the last SM's first instruction: the previous launched step's ring slot into the host ring and its
        flag, and the rest of V2's post_update of its request."""
        # host: pinned, ring's layout; flags: int32 per slot, pinned; post: total_len, all_token_ids, bin_counts or None
        # (OP_BOOK writes the request's last sampled token and computed tokens)
        for k in SS_HOST:
            assert host[k].is_pinned() and host[k].shape == ring[k].shape and host[k].dtype == ring[k].dtype, k
        assert flags.is_pinned() and flags.dtype == torch.int32 and flags.numel() == ring["sampled"].shape[0]
        all_ids, bins = post["all_token_ids"], post["bin_counts"]
        assert post["total_len"].dtype == all_ids.dtype == torch.int32 and post["total_len"].stride(0) == 1
        assert all_ids.stride(1) == 1 and (bins is None or (bins.dtype == torch.int32 and bins.stride(1) == 1))

        ring_tensors = [ring[k] for k in SS_HOST]
        host_tensors = [host[k] for k in SS_HOST] + [flags]
        states = [post[k] for k in ("total_len", "all_token_ids")]
        self.keep += ring_tensors + host_tensors + states + ([] if bins is None else [bins])
        self.descs.append(
            PREV.pack(
                0,
                *(tensor.data_ptr() for tensor in ring_tensors + host_tensors + states),
                0 if bins is None else bins.data_ptr(),
                all_ids.stride(0),
                0 if bins is None else bins.stride(0),
                ring["sampled"].shape[1],
                0,
            )
        )

        last_sm = self.nsm - 1
        assert last_sm > 0 and not self.sm_instrs[last_sm], "OP_PREV is the last SM's first instruction"
        self.instr(last_sm, Op.PREV, len(self.descs) - 1, NO_CTR, NO_CTR)

    def add_book(self, *, inputs: dict, states: dict, tables: dict, ring: dict) -> int:
        """OP_BOOK, SM 0's first instruction: the step's inputs written into the V2 runner's buffers, its request from
        the step record, and with the record's PREV_POST the previous launched step's last sampled token and computed
        tokens from its ring slot, returning its counter, which the embedding waits for (it reads the input ids)."""
        # inputs: input_ids, positions, seq_lens, is_padding; states: last_sampled, draft_tokens, num_computed; tables:
        # src / dst (uint64 bases per KV cache group), stride, num_blocks; ring: sampled, num_sampled, num_rejected
        draft_tokens, num_blocks = states["draft_tokens"], tables["num_blocks"]
        assert (
            draft_tokens.dtype == torch.int64
            and draft_tokens.stride(1) == 1
            and num_blocks.dtype == torch.int32
            and num_blocks.stride(1) == 1
        )
        assert inputs["input_ids"].dtype == torch.int32 and inputs["positions"].dtype == torch.int64
        assert states["last_sampled"].dtype == torch.int64 and states["num_computed"].dtype == torch.int32
        assert tables["src"].dtype == tables["dst"].dtype == torch.uint64 and tables["stride"].dtype == torch.int64
        assert ring["sampled"].dtype == torch.int64 and ring["sampled"].stride(1) == 1
        assert ring["num_sampled"].dtype == ring["num_rejected"].dtype == torch.int32

        tensors = [inputs[k] for k in ("input_ids", "positions", "seq_lens", "is_padding")]
        tensors += [states[k] for k in ("last_sampled", "draft_tokens", "num_computed")]
        tensors += [tables[k] for k in ("src", "dst", "stride", "num_blocks")]
        tensors += [ring[k] for k in ("sampled", "num_sampled", "num_rejected")]
        self.keep += tensors
        self.descs.append(
            BOOK.pack(
                0,
                *(tensor.data_ptr() for tensor in tensors),
                draft_tokens.stride(0),
                tables["src"].numel(),
                num_blocks.stride(0),
                ring["sampled"].stride(0),
            )
        )

        assert not self.sm_instrs[0], "OP_BOOK is SM 0's first instruction (before the embedding)"
        ctr = self.new_counter(1)
        # the block-table rows, which no op of the launch reads: an instruction of their own on the second last SM, off
        # the embedding's path and not counted
        assert not self.sm_instrs[self.nsm - 2]
        self.instr(0, Op.BOOK, len(self.descs) - 1, NO_CTR, ctr, aux=self.BOOK_IN)
        self.instr(self.nsm - 2, Op.BOOK, len(self.descs) - 1, NO_CTR, NO_CTR, aux=self.BOOK_BT)

        return ctr

    def add_embed(
        self,
        *,
        table: torch.Tensor,
        out: torch.Tensor,
        dep: int = NO_CTR,
        mask_row: torch.Tensor | None = None,
        sms: list | None = None,
        parts: int = 1,
        early_rows_mark: str | None = None,
    ) -> tuple[int, torch.Tensor]:
        """The T input tokens' embedding rows (ids from the record) into out (bf16 [NT][K]) and their sums of squares
        (one M tile's slots), as (counter, ssq)."""
        assert table.dtype == torch.bfloat16 and table.is_contiguous() and out.shape == (NT, table.shape[1])
        assert table.shape[1] % (self.VEC_ELEMS * parts) == 0

        ssq = torch.zeros(2, 1, NT, dtype=torch.int64, device=self.device)  # [launch parity][one M tile][token]
        self.keep += [table, out, ssq] + ([mask_row] if mask_row is not None else [])
        self.descs.append(
            self.desc(
                in0=table,
                r_out=out,
                K=table.shape[1],
                ssq=ssq,
                res_in=mask_row,
                dflags=DF.DRAFT_TOK if mask_row is not None else 0,
            )
        )
        desc_index = len(self.descs) - 1

        # an instruction per part of a row: a table row (host memory) in `parts` parts, a device row (mask_row: the
        # draft's query block, the last sampled token then the mask token) whole, on SM sms[i % len(sms)]; the mask
        # token's rows, which wait for nothing, inserted at the mark early_rows_mark ahead of the op's own place
        work_items = [
            (row, part, n)
            for row in range(NT)
            for n in [parts if mask_row is None or row == 0 else 1]
            for part in range(n)
        ]
        sms = list(sms or range(len(work_items)))
        ctr = self.new_counter(len(work_items))
        for i, (row, part, n) in enumerate(work_items):
            is_early_row = early_rows_mark is not None and mask_row is not None and row > 0
            if is_early_row:  # (one instruction at the mark: insert_positions advances only the SMs it inserted on)
                assert self.insert_positions is None
                self.insert_positions = list(self.marks[early_rows_mark])
            self.instr(
                sms[i % len(sms)], Op.EMBED, desc_index, NO_CTR if is_early_row else dep, ctr, row, part, i, aux=n
            )
            if is_early_row:
                self.marks[early_rows_mark] = self.insert_positions  # (later early rows on the same SM: after this one)
                self.insert_positions = None

        return ctr, ssq

    def add_gemv(
        self,
        *,
        w: torch.Tensor,
        kind: Kind,
        K: int,
        dep: int,
        pro: Pro,
        alpha: float,
        gs: float,
        sf: torch.Tensor | None = None,
        eps: float = 0.0,
        in0: torch.Tensor | int = 0,
        res_in: torch.Tensor | int = 0,
        res_bf16: bool = False,
        nw: torch.Tensor | int = 0,
        x_out: torch.Tensor | int = 0,
        r_out: torch.Tensor | int = 0,
        out: torch.Tensor | None = None,
        act_gs: float | None = None,
        is_logits: bool = False,
        sms: int | None = None,
        xr: bool = False,
        ssq_in: torch.Tensor | None = None,
        x_bf16: bool = False,
        acc_pool: str = "acc",
        dflags: int = 0,
        coef: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
        sm_extra: dict | None = None,
        min_stages: int = MIN_STAGES,
        has_qpub: bool = False,
    ) -> tuple[int, int]:
        """A GEMV op over the rows of w (uint8 [rows, row bytes]) for NT token rows, as (counter, desc index)."""
        # pro: B from in0 (NORM_IN: bf16 [NT][K] normed; NORM_RESID: in0 + res_in normed, x_out / r_out its slices; ACT:
        # a published operand); out: EPI_RED bf16 [NT][rows], with is_logits EPI_LOGITS; act_gs (w is [gate; up]):
        # EPI_SILU publishes the consumer's NVFP4 operand into out; sm_extra: per SM the stages of its other work
        rows, kb = w.shape
        assert w.dtype == torch.uint8 and kb == (K // 2 if kind == Kind.NVFP4 else K if kind == Kind.FP8 else 2 * K)
        assert kb % STAGE_K == 0

        is_paired = act_gs is not None
        spm = kb // STAGE_K
        if is_paired:
            n_out = rows // 2
            assert n_out % TILE == 0 and kind == Kind.NVFP4 and spm <= B_CHUNKS
            units, spu, epi = n_out // TILE, 2 * spm, Epi.SILU
        else:
            n_out = rows
            units, spu = -(-rows // TILE), spm
            epi = Epi.LOGITS if is_logits else Epi.RED
        mtiles = units
        sm_count = sms or self.nsm
        num_bands = 1 if is_paired else self.bands(spm)
        bw = spm // num_bands
        if not is_paired:
            units, spu = mtiles * num_bands, bw  # band-major (band, tile) units
        ranges = self.split(
            units,
            spu,
            spm,
            sm_count,
            is_whole_tiles=is_logits,
            is_paired=is_paired,
            num_bands=num_bands,
            extra_stages=sm_extra,
            min_stages=min_stages,
        )

        # parts per unit (EPI_RED) and the paired split check
        def count_parts(ranges: list) -> torch.Tensor:
            parts = torch.zeros(max(mtiles, 1), dtype=torch.uint8)
            for _, r0, r1 in ranges:
                for unit in range(r0 // spu, (r1 - 1) // spu + 1):
                    parts[unit % mtiles] += 1
            return parts

        parts = count_parts(ranges)
        if is_logits and int(parts.max()) > 2:  # (fewer tiles than SMs: whole tiles per SM)
            ranges = self.split(
                units,
                spu,
                spm,
                sm_count,
                is_whole_tiles=True,
                is_paired=is_paired,
                num_bands=num_bands,
                extra_stages=sm_extra,
                min_stages=min_stages,
            )
            parts = count_parts(ranges)
        n_inst = len(ranges)
        if is_paired or is_logits:
            assert int(parts.max()) <= 2, "EPI_SILU / EPI_LOGITS: a tile in at most two parts"
        assert int(parts.max()) < self.MAX_TILE_PARTS

        acc = spart = sflag = ssq = 0
        if xr:  # the finishers form x = out + res_in (x_out fp32, r_out bf16), publish its sums of squares per M tile
            assert epi == Epi.RED and n_out == K_NORM
            ssq = torch.zeros(2, mtiles, NT, dtype=torch.int64, device=self.device)
        q_pub, qctr = 0, 0
        if pro in (Pro.NORM_X, Pro.NORM_Q) and not dflags & DF.PLAIN:
            assert ssq_in is not None and ssq_in.dtype == torch.int64 and ssq_in.shape[0] == 2
        if pro == Pro.NORM_Q:  # one instruction per SM (an SM's second one would wait for its own slice)
            assert len({sm for sm, _, _ in ranges}) == n_inst
            num_chunks = K // (256 if kind == Kind.NVFP4 else 128 if kind == Kind.FP8 else 64)  # K chunks of 128 bytes
            q_pub = torch.zeros(
                num_chunks * CHUNK_B + (num_chunks * SFB_CHUNK if kind == Kind.NVFP4 else 0),
                dtype=torch.uint8,
                device=self.device,
            )
            qctr = self.new_counter(n_inst)
        if epi == Epi.RED and int(parts.max()) > 1:
            acc = self.pooled(acc_pool, NT * mtiles * TILE, torch.int64)  # (ops that may run at once: other pools)
        colctr = 0
        if dflags & DF.COLFIN:
            assert epi == Epi.RED and (coef is not None and base is not None or not dflags & DF.CONV_FIN)
            colctr = torch.zeros(mtiles * self.EPI_WARPS, dtype=torch.int32, device=self.device)
        if dflags & DF.CONV:
            assert pro == Pro.NORM_Q and coef is not None and base is not None
        if is_paired:  # (the split pairs' partials: one pool, the paired ops run one after another; the flags per op)
            spart = self.pooled("spart", mtiles * TILE * 2 * NT, torch.float32)
            sflag = torch.zeros(mtiles, dtype=torch.int32, device=self.device)
        tag_base = 0  # OpDesc::pool_len of EPI_LOGITS: its tags' base
        if is_logits and int(parts.max()) > 1:  # (a split tile's later part's tagged sums, in its instruction's slot)
            assert num_bands == 1 and all(ranges[i][1] == ranges[i - 1][2] for i in range(1, n_inst))
            spart = torch.zeros(n_inst * TILE * NT, dtype=torch.int64, device=self.device)
            self.logits_ops += 1
            tag_base = 2 * self.logits_ops - 1  # (odd: 1 + the launch parity never 0, unique per op of a schedule)

        device_parts = parts.to(self.device)
        self.keep += [
            x
            for x in (
                acc,
                spart,
                sflag,
                device_parts,
                in0,
                res_in,
                nw,
                x_out,
                r_out,
                out,
                sf,
                ssq,
                ssq_in,
                q_pub,
                coef,
                base,
                colctr,
            )
            if isinstance(x, torch.Tensor)
        ]
        if sf is not None:
            assert sf.dtype == torch.uint8 and sf.numel() == -(-rows // TILE) * TILE * (K // self.NVFP4_BLOCK)
        has_resid_out = pro == Pro.NORM_RESID and (self.ptr(x_out) or self.ptr(r_out))
        slice_ = -(-(K_NORM // self.NORM_GROUP) // n_inst) * self.NORM_GROUP if pro == Pro.NORM_RESID else 0
        if pro in (Pro.NORM_IN, Pro.NORM_RESID, Pro.NORM_X, Pro.NORM_Q):
            assert K == K_NORM or pro == Pro.NORM_Q and dflags & DF.PLAIN and K % K_NORM == 0
        self.descs.append(
            self.desc(
                tmap=self.tmap(w, rows, kb), sf=sf, in0=in0, res_in=res_in, nw=nw, x_out=x_out, r_out=r_out, acc=acc,
                out=out, parts=device_parts, spart=spart, sflag=sflag, alpha=alpha, gs=gs, eps=eps,
                act_gs=act_gs or 0.0, kind=int(kind), K=K, kb=kb, rows=rows, n_out=n_out, mtiles=mtiles, spm=spm,
                pro=int(pro), epi=int(epi), n_inst=n_inst, paired=int(is_paired), res_bf16=int(res_bf16),
                slice=slice_, bw=bw, ssq=ssq, ssq_in=ssq_in, xr=int(xr),
                ssq_tiles=0 if ssq_in is None else ssq_in.shape[1], x_bf16=int(x_bf16), q_pub=q_pub, qctr=qctr,
                coef=coef, base=base, colctr=colctr, dflags=dflags, pool_len=tag_base,
            )
        )  # fmt: skip
        desc_index = len(self.descs) - 1

        # NORM_RESID with x_out / r_out: each instruction counts twice, its prologue's slices and its epilogue
        ctr = self.new_counter(n_inst * (2 if has_resid_out else 1))
        # per instruction start its rotation: k << 16 | its first stage's B group << 24 (bw-stage units: the sums' order
        # is free)
        rotations = {}
        if epi == Epi.RED and not is_paired and self.ROT_MAX > 0:
            for _, r0, r1 in ranges:
                last_unit_start = (r1 - 1) // bw * bw
                if last_unit_start > r0 and r1 - last_unit_start <= self.ROT_MAX:
                    rotations[r0] = (r1 - last_unit_start) << self.AUX_ROT_K_SHIFT | (
                        (last_unit_start - r0) % bw // B_GCH
                    ) << self.AUX_ROT_GROUP_SHIFT
        # has_qpub (NORM_Q, NVFP4): each instruction's slice published by an OP_QPUB just before it on its SM, so the
        # prologue publishes it on reaching the QPUB; the GEMV then only waits for every slice and copies
        assert not has_qpub or (pro == Pro.NORM_Q and kind == Kind.NVFP4)
        for i, (sm, r0, r1) in enumerate(ranges):
            if has_qpub:
                self.instr(sm, Op.QPUB, desc_index, dep, NO_CTR, 0, 0, i)
            self.instr(
                sm,
                Op.GEMV,
                desc_index,
                dep,
                ctr,
                r0,
                r1,
                i,
                aux=rotations.get(r0, 0) | (AUX_QPUB_DONE if has_qpub else 0),
            )
        self.op_info.append(dict(desc=desc_index, ssq=ssq))

        return ctr, desc_index

    def add_attention(
        self,
        *,
        kv: torch.Tensor,
        qkv: torch.Tensor,
        q_norm_w: torch.Tensor,
        k_norm_w: torch.Tensor,
        cos_sin: torch.Tensor,
        act: torch.Tensor,
        block_tokens: int,
        group: int,
        scale: float,
        eps: float,
        s_in: float,
        n_splits: int,
        dep: int,
    ) -> int:
        """A full-attention layer's core: OP_ATTN_PREP (an item per KV head and token), OP_ATTN (n_splits items per KV
        head), OP_ATTN_COMBINE (publishing o_proj's FP8 operand into act), returning the combine's counter."""
        rows = kv.numel() // (self.ATT_HKV * self.ATT_KV_ROW_BYTES)
        assert (
            kv.dtype == torch.uint8 and kv.is_contiguous() and rows * self.ATT_HKV * self.ATT_KV_ROW_BYTES == kv.numel()
        )
        assert qkv.dtype == torch.bfloat16 and qkv.shape == (NT, self.ATT_QKV)
        assert cos_sin.dtype == torch.bfloat16 and cos_sin.shape[1] == self.ROTARY_DIM
        # act: o_proj's FP8 operand [K chunks of 128 values][NT][128 bytes]
        assert act.dtype == torch.uint8 and act.numel() == self.ATT_OUT // 128 * NT * 128

        tensor_map = torch.ops.pear_deploy_rtx_blackwell_nvfp4.make_tmap3d(
            kv.data_ptr(), rows, self.ATT_HKV, self.ATT_KV_ROW_BYTES, self.ATT_TILE
        )
        i = self.n_tmaps
        self.n_tmaps += 1
        self.tmaps[i * TMAP_BYTES : (i + 1) * TMAP_BYTES].copy_(tensor_map)
        q = self.pooled("attn_q", self.ATT_HKV * self.ATT_ROWS * self.ATT_D, torch.float16).view(
            self.ATT_HKV, self.ATT_ROWS, self.ATT_D
        )
        po = self.pooled("attn_po", self.ATT_HKV * n_splits * self.ATT_ROWS * self.ATT_D, torch.float32).view(
            self.ATT_HKV, n_splits, self.ATT_ROWS, self.ATT_D
        )
        ml = self.pooled("attn_ml", self.ATT_HKV * n_splits * self.ATT_ROWS * 2, torch.float32).view(
            self.ATT_HKV, n_splits, self.ATT_ROWS, 2
        )  # (each row's m, l)
        self.keep += [kv, qkv, q_norm_w, k_norm_w, cos_sin, act, q, po, ml]

        prep_ctr = self.new_counter(self.ATT_HKV * NT)
        attn_ctr = self.new_counter(self.ATT_HKV * n_splits)
        combine_ctr = self.new_counter(self.ATT_COMBINE_ITEMS)
        desc = ATTN.pack(
            kv.data_ptr(), self.tmaps.data_ptr() + i * TMAP_BYTES, qkv.data_ptr(), q_norm_w.data_ptr(),
            k_norm_w.data_ptr(), cos_sin.data_ptr(), q.data_ptr(), po.data_ptr(), ml.data_ptr(), act.data_ptr(),
            scale * self.LOG2E, eps, s_in, 0.0, n_splits, block_tokens, group, prep_ctr, self.MROPE_SECTIONS[1],
            self.MROPE_SECTIONS[2], self.KB_TOKENS,
        )  # fmt: skip
        self.descs.append(desc)
        desc_index = len(self.descs) - 1
        for item in range(self.ATT_HKV * NT):  # (KV head, token)
            self.instr(item % self.nsm, Op.ATTN_PREP, desc_index, dep, prep_ctr, item, item + 1, item)
        for item in range(self.ATT_HKV * n_splits):
            self.instr(item % self.nsm, Op.ATTN, desc_index, prep_ctr, attn_ctr, item, item + 1, item)
        for item in range(self.ATT_COMBINE_ITEMS):
            self.instr(item % self.nsm, Op.ATTN_COMBINE, desc_index, attn_ctr, combine_ctr, item, item + 1, item)

        return combine_ctr

    def add_spec_sample(self, *, lm_ctr: int, logits: torch.Tensor, ring: dict, mark: str | None = None) -> int:
        """The spec sampler's phases A to C after the LM head (its counter lm_ctr, its logits [NT, V]) on the prologue
        warps of every SM, into ring's SS_RING outputs (RING slots of NT rows), returning the last phase's counter."""
        # A: item i (row, chunk) of the step's T rows (min(LP_MAX_CHUNKS, nsm // T) chunks each), a row's last chunk
        # done its nucleus and test; B: a slice of the vocabulary per SM; C: the resampled token on SM 0
        nt, V = logits.shape
        assert (self.nsm // nt) * LP_CHUNK_ELEMS >= V and self.nsm * SS_SLICE_ELEMS >= V and V < SS_MAX_VOCAB
        chunks = min(LP_MAX_CHUNKS, self.nsm // nt)  # (phase A's chunks per row at T = nt, its largest chunk)
        assert -(-V // self.VEC_ELEMS // chunks) <= SS_X_VECS, "the sampler's chunk past its staging buffer (SS_X_VECS)"

        device = self.device
        scratch = [
            torch.zeros(self.nsm * SS_PART_BYTES, dtype=torch.uint8, device=device),  # part (SsPart per item)
            torch.zeros(self.nsm * SS_HIST_WORDS, dtype=torch.int32, device=device),  # hist (per item)
            torch.zeros(nt * SS_ROW_BINS, dtype=torch.int32, device=device),  # rhist (per row; zero between steps)
            torch.zeros(nt, dtype=torch.int32, device=device),  # done (per row; zero between steps)
            torch.zeros(self.nsm * 2, dtype=torch.float32, device=device),  # dpart (float2 per item)
            torch.zeros(self.nsm, dtype=torch.int64, device=device),  # amax (a key per slice)
            torch.zeros(SS_STATE_BYTES, dtype=torch.uint8, device=device),  # state (SsFact, SsRow x SS_MAX_T)
        ]
        ring_outputs = [ring[k] for k in SS_RING]
        self.keep += scratch + ring_outputs + [logits]
        self.descs.append(
            SSD.pack(
                logits.data_ptr(),
                *(tensor.data_ptr() for tensor in scratch),
                0,
                *(tensor.data_ptr() for tensor in ring_outputs),
                V,
                LP_MAX_CHUNKS,
                nt,
                self.nsm,
            )
        )
        desc_index = len(self.descs) - 1

        phase_a_ctr, phase_b_ctr, phase_c_ctr = (
            self.new_counter(self.nsm),
            self.new_counter(self.nsm),
            self.new_counter(1),
        )
        # phase A0 (the draft rows' stats, no dependency) an instruction of its own on each SM before `mark`, so what is
        # inserted at the mark later runs between A0 and A
        for sm in range(self.nsm):
            self.instr(sm, Op.SS_A, desc_index, NO_CTR, NO_CTR, idx=sm, aux=self.SS_AUX_A0)
        if mark is not None:
            self.mark(mark)
        for sm in range(self.nsm):
            self.instr(sm, Op.SS_A, desc_index, lm_ctr, phase_a_ctr, idx=sm, aux=self.SS_AUX_NO_A0)
        for sm in range(self.nsm):
            self.instr(sm, Op.SS_B, desc_index, phase_a_ctr, phase_b_ctr, idx=sm)
        self.instr(0, Op.SS_C, desc_index, phase_b_ctr, phase_c_ctr)

        return phase_c_ctr

    def add_dop(
        self, op: Op, desc: bytes, items: int, dep: int, dep2: int = NO_CTR, sms: list | None = None, keep: list = ()
    ) -> int:
        """A draft op (csrc/draft.cuh) of `items` instructions (item i on SM sms[i % len] or i % nsm), waiting for dep
        (and dep2), returning its counter."""
        self.descs.append(desc)
        self.keep += list(keep)
        desc_index = len(self.descs) - 1

        ctr = self.new_counter(items)
        for item in range(items):
            sm = sms[item % len(sms)] if sms else item % self.nsm
            self.instr(sm, op, desc_index, dep, ctr, item, item + 1, item, dep2=dep2)

        return ctr

    def add_nop(self, dep: int) -> int:
        """An instruction on every SM that waits for dep and counts: a barrier."""
        self.descs.append(self.desc())
        ctr = self.new_counter(self.nsm)
        for sm in range(self.nsm):
            self.instr(sm, Op.NOP, len(self.descs) - 1, dep, ctr)

        return ctr

    def ssq_of(self, desc: int) -> torch.Tensor:
        """An xr op's published sums of squares (a NORM_X consumer's ssq_in)."""
        return next(info["ssq"] for info in self.op_info if info["desc"] == desc)

    def gemv_counts(self, w: torch.Tensor, kind: Kind, K: int) -> dict:
        """The stages per SM of a plain GEMV op over w (add_gemv's split, no paired / logits): {sm: stages}."""
        rows, kb = w.shape
        spm = kb // STAGE_K
        mtiles = -(-rows // TILE)
        num_bands = self.bands(spm)

        stages = {}
        for sm, r0, r1 in self.split(
            mtiles * num_bands,
            spm // num_bands,
            spm,
            self.nsm,
            is_whole_tiles=False,
            is_paired=False,
            num_bands=num_bands,
        ):
            stages[sm] = stages.get(sm, 0) + r1 - r0

        return stages

    # ---- finalize
    def finalize(self) -> dict:
        """The device tensors: instructions (per-SM lists concatenated), offsets, descriptors, counters."""
        instrs, offsets = [], [0]
        for instructions in self.sm_instrs:
            instrs += instructions
            offsets.append(offsets[-1] + len(instructions))

        device = self.device
        tensors = dict(
            instrs=torch.frombuffer(bytearray(b"".join(instrs) or bytes(INSTR.size)), dtype=torch.uint8).to(device),
            sm_off=torch.tensor(offsets, dtype=torch.int32, device=device),
            descs=torch.frombuffer(bytearray(b"".join(self.descs) or bytes(DESC.size)), dtype=torch.uint8).to(device),
            counters=torch.zeros(max(1, len(self.ctr_count)), dtype=torch.int32, device=device),
            ctr_count=torch.tensor(self.ctr_count or [0], dtype=torch.int32, device=device),
            epoch=torch.zeros(2, dtype=torch.int32, device=device),
        )
        self.keep += list(tensors.values())

        return tensors

    # ---- a GEMV's stages over the SMs
    @staticmethod
    def bands(spm: int) -> int:
        """K bands of an op with spm chunks: the fewest equal bands of at most B_CHUNKS chunks."""
        for num_bands in range(1, spm + 1):
            if spm % num_bands == 0 and spm // num_bands <= B_CHUNKS:
                return num_bands
        raise AssertionError(spm)

    def split(
        self,
        units: int,
        spu: int,
        spm: int,
        sm_count: int,
        is_whole_tiles: bool,
        is_paired: bool,
        num_bands: int,
        extra_stages: dict | None = None,
        min_stages: int = MIN_STAGES,
    ) -> list:
        """(sm, r0, r1) per instruction: each band's (tile, bw chunk) stages stream-K over its share of the SMs."""
        # paired / one band: stream-K over all stages; num_bands bands: an instruction stays in its band (its B holds
        # <= bw chunks); is_whole_tiles (EPI_LOGITS): whole tiles per SM
        ranges = []
        if is_whole_tiles:
            n = min(sm_count, units)
            start = 0
            for sm, count in zip(range(n), self.weighted(units, list(range(n)), extra_stages=extra_stages)):
                if count:
                    ranges.append((sm, start * spu, (start + count) * spu))
                start += count
            return ranges

        total = units * spu
        per_band = total // num_bands
        sms = max(1, min(sm_count, total // min_stages))
        for band in range(num_bands):
            if sms >= num_bands:
                # the bands' SMs interleaved: SMs with extra stages elsewhere (in_proj_ba's beside in_proj_qkvz) spread
                # over every band, not all in one band whose other SMs would then take more stages
                sm_ids = [sm % self.nsm for sm in range(band, sms, num_bands)]
            else:
                first_sm, end_sm = sms * band // num_bands, sms * (band + 1) // num_bands
                n = max(1, end_sm - first_sm)
                sm_ids = [(first_sm + i) % self.nsm for i in range(n)]
            start = band * per_band
            # (a paired op's pair must fall in at most two instructions: equal counts, its split's invariant)
            for sm, count in zip(
                sm_ids, self.weighted(per_band, sm_ids, is_equal_split=is_paired, extra_stages=extra_stages)
            ):
                if count:
                    ranges.append((sm, start, start + count))
                start += count

        return ranges

    def weighted(self, total: int, sms: list, is_equal_split: bool = False, extra_stages: dict | None = None) -> list:
        """total stages over the SMs: the counts that minimize the largest count plus extra stages (is_equal_split:
        total * (i + 1) // n - total * i // n, ignoring extra_stages)."""
        n = len(sms)
        if is_equal_split:
            return [total * (i + 1) // n - total * i // n for i in range(n)]

        extras = [(extra_stages or {}).get(sm, 0) for sm in sms]
        level = (total + sum(extras)) / n  # (the water level: count_i + extra_i = level)
        counts = [max(0, int(level - extras[i])) for i in range(n)]
        while sum(counts) > total:  # (rounding: take back from the latest)
            i = max(range(n), key=lambda j: counts[j] + extras[j] if counts[j] else -1)
            counts[i] -= 1
        heap = [(counts[i] + 1 + extras[i], i) for i in range(n)]
        heapq.heapify(heap)
        for _ in range(total - sum(counts)):
            _, i = heapq.heappop(heap)
            counts[i] += 1
            heapq.heappush(heap, (counts[i] + 1 + extras[i], i))

        return counts

    # ---- counters, descriptors, instructions
    def pooled(self, key: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """numel elements of the pool `key` (a larger buffer replaces it for this and later users)."""
        buf = self.pools.get(key)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            buf = torch.zeros(numel, dtype=dtype, device=self.device)
            self.pools[key] = buf
            self.keep.append(buf)

        return buf[:numel]

    def new_counter(self, count: int) -> int:
        """A completion counter that count arrivals complete per launch."""
        self.ctr_count.append(count)
        assert len(self.ctr_count) < NO_CTR

        return len(self.ctr_count) - 1

    def tmap(self, w: torch.Tensor, rows: int, row_bytes: int) -> int:
        """A device CUtensorMap over w (uint8 [rows][row_bytes]): box 128 B x 128 rows, 128-byte swizzle."""
        assert (
            w.dtype == torch.uint8
            and w.is_contiguous()
            and w.data_ptr() % self.TMA_ALIGN == 0
            and row_bytes % self.TMA_ALIGN == 0
        )

        tensor_map = torch.ops.pear_deploy_rtx_blackwell_nvfp4.make_tmap(w.data_ptr(), rows, row_bytes, TILE)
        i = self.n_tmaps
        self.n_tmaps += 1
        assert i < self.MAX_TMAPS
        self.tmaps[i * TMAP_BYTES : (i + 1) * TMAP_BYTES].copy_(tensor_map)
        self.keep.append(w)

        return self.tmaps.data_ptr() + i * TMAP_BYTES

    def instr(
        self,
        sm: int,
        op: int,
        desc: int,
        dep: int,
        done: int,
        r0: int = 0,
        r1: int = 0,
        idx: int = 0,
        aux: int = 0,
        dep2: int = NO_CTR,
    ) -> None:
        """An instruction on SM sm, appended or inserted at insert_positions, with dep2 (Instr::pr0) a second
        dependency or NO_CTR."""
        instruction = INSTR.pack(op, desc, dep, done, r0, r1, idx, aux, dep2, 0)
        if self.insert_positions is None:
            self.sm_instrs[sm].append(instruction)
        else:  # (inserting at a mark: each SM's position advances past what it inserts)
            self.sm_instrs[sm].insert(self.insert_positions[sm], instruction)
            self.insert_positions[sm] += 1

    def mark(self, name: str) -> None:
        """Every SM's instruction count at this point of the build, under `name` (for insert_positions to insert
        there)."""
        self.marks[name] = [len(instructions) for instructions in self.sm_instrs]

    @staticmethod
    def desc(**fields: Any) -> bytes:
        """An OpDesc from its fields by name (0 where not given)."""
        pointers = [
            fields.get(k, 0)
            for k in (
                "tmap",
                "sf",
                "in0",
                "res_in",
                "nw",
                "x_out",
                "r_out",
                "acc",
                "out",
                "parts",
                "spart",
                "sflag",
                "rec",
                "ssq",
                "ssq_in",
                "q_pub",
            )
        ]
        floats = [fields.get(k, 0.0) for k in ("alpha", "gs", "eps", "act_gs")]
        words = [
            fields.get(k, 0)
            for k in (
                "kind",
                "K",
                "kb",
                "rows",
                "n_out",
                "mtiles",
                "spm",
                "pro",
                "epi",
                "n_inst",
                "paired",
                "res_bf16",
                "slice",
                "bw",
                "xr",
                "ssq_tiles",
                "x_bf16",
                "qctr",
            )
        ]
        draft_pointers = [Schedule.ptr(fields.get(k, 0)) for k in ("coef", "base", "colctr")]

        return DESC.pack(
            *[Schedule.ptr(x) for x in pointers],
            *floats,
            *words,
            *draft_pointers,
            fields.get("dflags", 0),
            fields.get("pool_u", 0),
            fields.get("pool_len", 0),
            fields.get("pool_n", 0),
        )

    @staticmethod
    def ptr(t: torch.Tensor | int | None) -> int:
        """A tensor's device address, or the address given (0 for None)."""
        return 0 if t is None or isinstance(t, int) and t == 0 else (t if isinstance(t, int) else t.data_ptr())
