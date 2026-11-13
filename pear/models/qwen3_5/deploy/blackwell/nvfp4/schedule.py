import collections
import math
import struct
from enum import IntEnum

import torch

# the packed layouts of csrc/: Instr (step.cuh) and the 256-byte descriptors OpDesc (megakernel.cu), AttnDesc
# (attention.cuh)
INSTR = struct.Struct("<4H6I")
DESC = struct.Struct("<14Q4f20I2Q2I3Q")
# megakernel.cu SharedBExt: ssq, xs, rin, res_in, nw, aux, b, xn, gs, eps, kind, res_bf16, x_bf16, aux_ctr, b_stride,
# mtiles, outs, erstd, ssq_prev, cs, baw
SHARED_B_EXT = struct.Struct("<8Q2f8I3Q")
ATTN = struct.Struct("<8Q4f6IQQI132x")  # AttnDesc (attention.cuh): ... trace (0), ml, kb_shift
GDNT = struct.Struct("<13Q2f12I4Q2f2I48x")  # GdnTDesc (gdn_t.cuh)
SSD = struct.Struct("<12Q4I144x")  # SsDesc (spec_sample.cuh)
BOOK = struct.Struct("<15Q4I120x")  # BookDesc (step.cuh)
PREV = struct.Struct("<9Q2q2I160x")  # PrevDesc (spec_sample.cuh)
SS_RING = ("sampled", "num_sampled", "num_rejected")  # the spec sampler's outputs, in SsDesc's order
SS_HOST = ("sampled", "num_sampled")  # the host ring's (V2's async output), in PrevDesc's order
SS_SLICE_ELEMS = 2048  # spec_sample.cuh: a slice's elements of the resampled row at most
SS_PART_BYTES = 80  # spec_sample.cuh SsPart: a target chunk's max and sum of exp, its window's bins, micro logits
SS_HIST_WORDS = 2048  # spec_sample.cuh: a target chunk's histogram (two 16-bit counts a word)
SS_ROW_BINS = 35587  # spec_sample.cuh: a target row's histogram (a bin of every 16-bit order)
SS_STATE_BYTES = 2 * 16 * 32  # spec_sample.cuh: SS_MAX_T SsFact, then SS_MAX_T SsRow
SS_MAX_VOCAB = 1 << 18  # spec_sample.cuh SS_HIST_BINS: a chunk's histogram holds the nucleus' window below it
LP_CHUNK_ELEMS = 2048 * 8  # logprobs.cuh: a chunk's logits at most (LP_CHUNK_VECS 16-byte vectors)
LP_MAX_CHUNKS = 32  # logprobs.cuh: a row's chunks at most (a lane each in the merge)
# gdn_t.cuh GdnInMode, GdnBaMode and GdnTDesc::flags
GT_IN_LL, GT_IN_ACC = 0, 1
GT_BA_DOT, GT_BA_SMEM, GT_BA_OP = 0, 1, 2
GT_REPLAYED = 2  # the core after OP_GDNT_PRE
GT_GNORM = 4  # the output phase writes out_proj's FP8 operand (the gated RMSNorm applied)
NO_CTR = 0xFFFF  # an instruction without a dependency / a completion counter
DONE_RELAXED = 0x8000  # Instr.done of a validated op's instructions: its counts are relaxed (megakernel.cu)
STAGE_BYTES = 128  # the K bytes of a weight row one ring stage carries
TILE_ROWS = 128  # weight rows per M tile (MMA M); a paired tile holds 64 gate and 64 up rows
STG_BYTES = 40960  # the kernel's prologue staging buffer (bytes)
K_NORM = 5120  # the hidden size: the RMSNorm prologues' length (megakernel.cu K_NORM)
NVFP4_BLOCK = 16  # values per NVFP4 scale
NT = 8  # token rows of every activation (the MMA's N; megakernel.cu NT): the step's T <= NT tokens, then unused rows
CHUNK_B = NT * STAGE_BYTES  # one 128-byte K chunk of the B operand (all rows)
SFB_CHUNK = 2 * NT * 16  # bytes of NVFP4 B scales per K chunk (megakernel.cu sfb_pos)
B_CHUNKS, SFB_CHUNKS = 35, 20  # an instruction's K chunks at most: its B buffer (megakernel.cu), NVFP4 scales
MAX_PARTS = 256  # parts per M tile an ACC_CNT word counts (megakernel.cu CNT_BITS)
GDN_V_HEADS = 48  # the Gated DeltaNet's value heads
ATT_KV_HEADS = 4  # the full attention's kv heads
ATT_Q_HEADS = 24  # the full attention's query heads
ATT_QKV_ROWS = 14336  # the attention's qkv projection: the queries and their gates, the keys, the values
ATT_PAGE = 64  # tokens per attention page (attention.cuh ATT_PAGE): one kernel block of the cache, or two of 32 tokens
ATT_ROW_BYTES = 1024  # one token's k | v of a kv head (bf16 256 + 256)
ATT_NT = 8  # tokens per step at most (attention.cuh ATT_NT)
ATT_PO_BLOCK = ATT_NT * 64  # bf16 values per partial block: [tokens][64 dims] (attention.cuh ATT_PO_BLOCK)
ATT_MAX_SPLITS = 40  # splits per kv head at most (attention.cuh: the combine's blocks fill the staging buffer)
ATT_COMBINE_ITEMS = 96  # the combine's items: (query head, quarter of its dims)
ATT_OUT = ATT_Q_HEADS * 256  # the attention output (query heads x head dim)
LOG2E = 1.4426950408889634  # log2(e): the softmax's exp2 scale
# Instr.aux bits of a GEMV instruction (megakernel.cu): bit 1 the prologue's warm-up dry run, bit 2 no producer gate,
# bits 8..15 its TMEM park slots, bits 16..19 its pacing limit; of an attention combine: bit 0 the dry copy
AUX_DRY, AUX_WARM, AUX_NOGATE, AUX_PARK_SHIFT, AUX_PACE_SHIFT = 1, 2, 4, 8, 16
# bit 3 (mark_twin) on a PRO_COPY_B instruction whose B the previous instruction on its SM writes (its twin), with the
# same dependency: that prologue arms it, the prologue warps skip it
AUX_TWIN = 1 << 3
AUX_SHARED_B = 1 << 4  # (mark_shared_b) on a shared-B producer's instructions: its finishers build the consumer's B
# bit 4 (mark_fastact) on the GDN core and its AUX_FASTACT out_proj: the core's idle warps stage out_proj's descriptor
AUX_FAPRE = 1 << 4
# bit 5 (mark_shared_b) on an aux layer's down finishers: their residual rows for the draft, released by warp 6
AUX_DRAFT_ROWS = 1 << 5
# bit 6 (mark_fastact) on an out_proj instruction after the GDN core on its SM (AUX_ZNEXT; validated, no dependency):
# its prologue warps' path of its own
AUX_FASTACT = 1 << 6
# bit 20 (mark_split): its CTA pair builds the op's RMSNorm B operands together (Instr.pr0 / pr1: the other CTA's range)
AUX_SPLIT = 1 << 20
# bit 21 (mark_perm): its stages are loaded in a permuted order, the parked block first (megakernel.cu perm_stage)
AUX_PERM = 1 << 21
# bit 22 (mark_epi_relaxed): its epilogue's count is relaxed, the op's outputs being tagged words its consumer polls
# (only its prologue's count releases: the residual slices): no release fence, no producer gate for the next GEMV
AUX_EPI_RELAXED = 1 << 22
# bit 23 (mark_znext) on an OP_GDNT instruction followed on its SM by an active out_proj instruction, and on that one:
# the core's idle warps zero out_proj's next-parity accumulator slice
AUX_ZNEXT = 1 << 23
# bits 24..31 (mark_dep_early): an RMSNorm consumer of a validated op starts this many counts per token row short of its
# dependency's target (mark_shared_b: a shared-B consumer, its producer's finisher instructions)
AUX_DEP_EARLY_SHIFT = 24
MAX_PARK, MAX_PARK_NV = 15, 9  # TMEM park slots of an FP8 / NVFP4 instruction at most (megakernel.cu MAX_PARK_F8 / _NV)


class Op(IntEnum):
    GEMV = 0
    GDN = 1
    GDN_QK = 2
    ATTN = 3
    ATTN_COMBINE = 4
    EMBED = 5
    ZERO = 9
    GDNT = 10
    GDNT_PRE = 11
    SS_A = 12  # the spec sampler's phases (spec_sample.cuh)
    SS_B = 13
    SS_C = 14
    BOOK = 16  # the step's inputs into the V2 runner's buffers (step.cuh BookDesc)
    PREV = 17  # the previous launched step's outputs into the host ring, V2's post_update (spec_sample.cuh PrevDesc)


class Kind(IntEnum):
    NVFP4 = 0
    FP8 = 1
    BF16 = 2


# a shared-B consumer's published B (link_shared_b): its launch-parity stride by the consumer's kind (FP8: 40 chunks of
# 128 values; NVFP4: 20 chunks of 256 values, then their block scales)
SHARED_B_STRIDE = {Kind.FP8: K_NORM // 128 * CHUNK_B, Kind.NVFP4: K_NORM // 256 * (CHUNK_B + SFB_CHUNK)}


class Pro(IntEnum):
    NORM_IN = 0
    NORM_RESID = 1
    GDN_NORM = 2
    ACT = 3
    COPY_B = 4


class Epi(IntEnum):
    RED = 0
    LOGITS = 1
    SILU = 2


class AccFmt(IntEnum):
    F32 = 0
    CNT = 1


class InstrField(IntEnum):
    """An Instr field's index in INSTR.unpack (step.cuh Instr)."""

    OP = 0
    DESC_INDEX = 1  # Instr::desc (DESC names the OpDesc layout)
    DEP = 2
    DONE = 3
    R0 = 4
    R1 = 5
    TILE = 6
    AUX = 7
    PR0 = 8
    PR1 = 9


class OpDescField(IntEnum):
    """An OpDesc field's index in DESC.unpack (megakernel.cu OpDesc), of the fields read or set by index."""

    IN0 = 3
    RES_IN = 4
    NW = 5
    X_OUT = 6
    R_OUT = 7
    P0 = 10  # p[0]
    P2 = 12  # p[2]
    ALPHA = 14
    GS = 15
    EPS = 16
    K_DIM = 18  # OpDesc::K (K names the GEMV's own)
    N_OUT = 19
    KIND = 20
    PRO = 21
    SRC = 22
    ACC_FMT = 23
    PAIRED = 25
    MTILES = 26
    RES_BF16 = 28
    EPI = 29
    ACT_STRIDE = 36
    FIN = 37
    SHARED_B_CONS = 40
    SHARED_B_PARITY_STRIDE = 41  # OpDesc::shared_b_stride (SHARED_B_STRIDE names the strides by kind)
    SHARED_B = 42
    SHARED_XN = 43
    SHARED_B_EXT_ADDR = 44  # OpDesc::shared_b_ext (SHARED_B_EXT names the SharedBExt layout)


DESC_XN, DESC_TWIN = 31, 35  # OpDesc fields set after their op is added (DESC.unpack indices)


class Schedule:
    """The decode megakernel's instructions per SM and the operation descriptors they point to (csrc/megakernel.cu)."""

    # A GEMV op's (M tile of 128 weight rows, 128-byte K chunk) stages are split stream-K over the SMs: of its T stages,
    # instruction s of P owns [T*s/P, T*(s+1)/P). Each instruction waits on the counter of the op it depends on (all of
    # its instructions complete) and bumps its own op's counter when done.

    # the GEMV ops whose stream does not wait for the release of the SM's previous GEMV instruction (the producer gate):
    # down, so that its ring and park stages are loaded by the time its B is ready; not in_z behind in_qkv, as the GDN
    # core acquires in_qkv's counter before its own release
    NOGATE = {"down"}
    # the GEMV ops whose stages are loaded in the permuted order (AUX_PERM): their B is ready long after their ring and
    # park slots are full, and at B-ready their first stages are issued first, their slots free at once; not o_proj or
    # out_proj, whose B is ready early (the parked block would delay the first MMAs)
    PERM = {"gate_up", "down", "lm_head", "qkv", "in_qkv"}
    # TMEM park slots while an instruction waits for its B operand: FP8 input and output projections, NVFP4 ops (with
    # their weight scales)
    PARK_IN, PARK_OUT, PARK_NV = 15, 15, 9
    # the pacing limit of the other FP8 GEMVs (o_proj, in_z) until their prologue has staged its inputs: o_proj's
    # parked stream keeps few loads in flight during the attention, whose release fences then wait less
    PACE_ATT = 2
    PACE_GN = 3  # out_proj's pacing limit, its operand polled from the GDN core's output phase (AUX_FAPRE)
    DEP_EARLY = 12  # mark_dep_early's lead per token row: T * DEP_EARLY counts short of the dependency's target
    # an ACC_CNT op's split evens out stages + this penalty per extra M tile an instruction crosses: every segment is an
    # epilogue round of atomic adds, whose L2 throughput sets the epilogue's time; NVFP4 (down)
    SEG_PEN = 2.0
    SEG_PEN_FP8 = 12.0  # FP8 (out_proj, o_proj, in_z): their M tiles split in thirds, no instruction over two segments
    # gate_up's split charges max(0, LAST_E - the last segment's stages) per instruction over more than one M tile: a
    # shorter last segment does not hide the previous segment's epilogue under its MMAs
    LAST_E = 5.0
    BISECT_ROUNDS = 48  # the uneven splits' bisection of their bound: halved far below one stage
    FIN_MAXP = 8  # parts per M tile of a FIN op (megakernel.cu FIN_MAXP)
    FIN_W = 3  # 16-byte words of a FIN part per channel (megakernel.cu FIN_W)
    FIN_TAG_LIMIT = 256  # FIN ops' tags (1 + the op's index) stay below it: the words' 9-bit tag of parity and op
    FIN_MAX_TILE_PARTS = 5  # a FIN op's parts per M tile at most: the first and 4 later parts (megakernel.cu)
    FIN_NO_PART = 0xFF  # a FIN tile's first-stage entry past its parts (megakernel.cu fparts)
    AUX_PARK_MASK = 0xFF  # Instr.aux bits 8..15 (AUX_PARK_SHIFT): the TMEM park slots
    AUX_DEP_EARLY_LIMIT = 256  # Instr.aux bits 24..31 (AUX_DEP_EARLY_SHIFT): a dependency's lead below it
    SHARED_B_MAX_TILES = 64  # a shared-B producer's M tiles at most (megakernel.cu: lane l sums tiles l, l + 32)
    B_SENTINEL = 0xFF  # a published B byte not yet written (no FP8 code; NVFP4: its scales')
    NVFP4_CODE_SENTINEL = 0x88  # a published NVFP4 code byte not yet written (two 0x8 nibbles: -0, stored as +0)
    ZERO_STORE_UNITS = 4  # OP_ZERO's 16-byte stores in the segments' 4-byte units
    RMS_EPS_MIN = 2.0**-126  # the smallest normal fp32: rms_rsqrt5120's rsqrt.approx.ftz flushes a denormal eps
    ACC_WORD_BYTES = 16  # megakernel.cu zero_next_acc's 16-byte words: an accumulator copy holds one at least
    U32_LIMIT = 2**32  # zero_next_acc's 32-bit word count (the copies' words times the instruction count)
    SFB_COPY_BYTES = 512  # an NVFP4 scale copy's reach, 32 rows x 16 bytes (megakernel.cu SFB_BYTES)
    PRO_THREADS = 256  # the prologue warps' threads (megakernel.cu NPRO)
    GDN_HEAD_ITEMS = 3  # the Gated DeltaNet core's items per value head (gdn_t.cuh GT_ITEMS)
    GDN_V_DIM = 128  # a value head's dims (gdn_t.cuh GT_V)
    ATT_HEAD_QUARTERS = 4  # a query head's quarters of its dims: the partial o's blocks (and the combine's items)
    COST_SLACK = 1e-9  # split_last's float slack when it checks the last range's cost against the bound

    def __init__(self, nsm: int, device: torch.device) -> None:
        self.nsm, self.device = nsm, device
        self.descs: list[bytes] = []
        self.sm_instrs: list[list[bytes]] = [[] for _ in range(nsm)]
        self.ctr_count: list[int] = []
        self.ctr_val: dict[int, bool] = {}  # a GEMV op's counter -> whether the op is validated (val)
        self.keep: list[torch.Tensor] = []  # the tensors the descriptors point to
        self.acc: dict[int, dict] = {}  # desc index -> the op's output (format, ranges, buffers)
        self.act_of_ctr: dict[int, torch.Tensor] = {}  # a combine's counter -> the o_proj operand it publishes
        self.gn_out: set[int] = set()  # out_proj's desc under GT_GNORM (PRO_ACT on the core's operand: still out_proj)
        self.polled_epi_descs: set[int] = set()  # the GEMVs whose epilogue writes only polled tagged words

        # the FIN ops so far and the workspace of their tagged words (fin_tables)
        self.fin_n = 0
        self.fin_ws: torch.Tensor | None = None

        # shared B (link_shared_b): each consumer's producer, the aux layers' producers (their stages per M tile), the
        # counter their residual rows release, the last boundary's sums of squares (the next one's ssq_prev)
        self.shared_b_producer: dict[int, int] = {}
        self.draft_row_stages: dict[int, int] = {}
        self.draft_rows_ctr: int | None = None
        self.shared_b_last_ssq: torch.Tensor | None = None

        # the draft's proposal as instructions of the step (attach_draft): its per-SM lists, appended after the marks,
        # and the step's counters its instructions wait on
        self.draft_instrs: list[list[bytes]] | None = None
        self.draft_wait_ctrs: set[int] = set()

    def add_prev(self, *, rec: int, ring: dict, host: dict, flags: torch.Tensor, post: dict) -> None:
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
                rec,
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
        self.sm_instrs[last_sm].append(INSTR.pack(Op.PREV, len(self.descs) - 1, NO_CTR, NO_CTR, 0, 0, 0, 0, 0, 0))

    def add_book(self, *, rec: int, inputs: dict, states: dict, tables: dict, ring: dict) -> None:
        """OP_BOOK, SM 0's first instruction: the step's inputs written into the V2 runner's buffers, its request from
        the step record, and with the record's PREV_POST the previous launched step's last sampled token and computed
        tokens from its ring slot."""
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
                rec,
                *(tensor.data_ptr() for tensor in tensors),
                draft_tokens.stride(0),
                tables["src"].numel(),
                num_blocks.stride(0),
                ring["sampled"].stride(0),
            )
        )

        assert not self.sm_instrs[0], "OP_BOOK is SM 0's first instruction (before the embedding and any mark())"
        self.sm_instrs[0].append(INSTR.pack(Op.BOOK, len(self.descs) - 1, NO_CTR, NO_CTR, 0, 0, 0, 0, 0, 0))

    def add_zero(self, *, seg_addrs: torch.Tensor, seg_strides: torch.Tensor, seg_pages: torch.Tensor, rec: int) -> int:
        """Zero the step's newly allocated blocks (ids in the step record) in the attention layers' caches as vLLM's
        KVBlockZeroer does, every SM a share; returns its counter."""
        # segment s of block b: [addr[s] + b * stride[s] * 4, + page[s] * 4) bytes
        assert seg_addrs.dtype in (torch.uint64, torch.int64) and seg_strides.dtype == torch.int64
        assert bool((seg_pages % self.ZERO_STORE_UNITS == 0).all()), "16-byte stores"

        self.keep += [seg_addrs, seg_strides, seg_pages]
        self.descs.append(
            self.plain_desc(
                w=seg_addrs.data_ptr(),
                in0=seg_strides.data_ptr(),
                res_in=seg_pages.data_ptr(),
                rec=rec,
                n_out=seg_addrs.numel(),
                n_inst=self.nsm,
            )
        )

        desc = len(self.descs) - 1
        ctr = self.new_counter(2 * self.nsm)
        for sm in range(self.nsm):
            self.sm_instrs[sm].append(INSTR.pack(Op.ZERO, desc, NO_CTR, ctr, 0, 0, sm, 0, 0, 0))

        return ctr

    def add_embed(self, *, table: torch.Tensor, out: torch.Tensor, rec: int) -> int:
        """The input token's embedding row (token from the step record) into out (bf16), on SM 0."""
        assert table.dtype == torch.bfloat16 and table.is_contiguous()

        self.keep += [table, out]
        self.descs.append(self.plain_desc(w=table.data_ptr(), in0=out.data_ptr(), rec=rec, K=table.shape[1], n_inst=1))
        desc = len(self.descs) - 1
        ctr = self.new_counter(2)
        self.sm_instrs[0].append(INSTR.pack(Op.EMBED, desc, NO_CTR, ctr, 0, 0, 0, 0, 0, 0))

        return ctr

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
        src: int = NO_CTR,  # the producing op whose output the prologue reads (its desc)
        in0: torch.Tensor | int = 0,
        res_in: torch.Tensor | int = 0,
        nw: torch.Tensor | int = 0,
        x_out: torch.Tensor | int = 0,
        r_out: torch.Tensor | int = 0,
        res_bf16: bool = False,
        logits: torch.Tensor | None = None,  # the LM head: whole M tiles per instruction, bf16 logits
        act_gs: float | None = None,  # gate||up: w is [gate; up], the epilogue publishes the down op's NVFP4 operand
        is_exact: bool = False,  # order-independent sums: ACC_CNT words when an M tile has more than two parts
        is_aligned: bool = False,  # order-independent sums: a split with at most two parts per M tile (0 + a + b)
        val: bool = False,  # ACC_CNT: the validated hand-off (its consumers are RMSNorm prologues)
        balance: int | None = None,  # exact ops: a split evening out each SM's stages of this op and of op balance
        has_ll_out: bool = False,  # ACC_CNT: each output also as a tagged word {bf16, launch token & 0xffff}
        has_ll_parts: bool = False,  # ACC_F32: each part's sums as tagged words {fp32, token} a GDN core polls
        act_stride: int = 0,  # PRO_ACT: the operand validated, 2 copies act_stride bytes apart (GT_GNORM), dep NO_CTR
        fin: bool = False,  # ACC_CNT (shared B): each M tile finished by its first part from the others' tagged words
    ) -> tuple[int, int]:
        """A GEMV op over all rows of w (uint8 [rows, K bytes]) whose activations and outputs have NT token rows each;
        returns its completion counter and desc index."""
        # (shared B: an RMSNorm consumer, nw set, waits for its producer's count, then validates its published B)
        assert not act_stride or (
            pro == Pro.ACT and kind in (Kind.FP8, Kind.NVFP4) and (dep == NO_CTR or self.ptr(nw))
        ), "a validated operand"

        rows = w.shape[0]
        is_paired = act_gs is not None
        n_out = rows // 2 if is_paired else rows
        mtiles = n_out // (TILE_ROWS // 2) if is_paired else -(-rows // TILE_ROWS)
        assert not is_paired or n_out % TILE_ROWS == 0
        assert n_out % NVFP4_BLOCK == 0
        row_bytes = K // 2 if kind == Kind.NVFP4 else K if kind == Kind.FP8 else 2 * K
        assert row_bytes % STAGE_BYTES == 0
        spm = row_bytes // STAGE_BYTES
        stages = mtiles * spm
        n_inst = self.nsm
        epi = Epi.RED
        p = [0, 0, 0, 0]
        act = output = llw = qw = None

        if is_paired:  # the down op's operand from the epilogue: no accumulator
            assert kind == Kind.NVFP4 and pro in (Pro.NORM_RESID, Pro.ACT)
            epi = Epi.SILU
            ranges = self.split_last(stages, spm, self.LAST_E, [0] * n_inst)
            for r0, r1 in ranges:  # a tile has at most two parts: a prefix (ends an instruction), a suffix (starts one)
                assert r0 == r1 or r0 % spm == 0 or r1 % spm == 0 or r0 // spm != (r1 - 1) // spm, (
                    "EPI_SILU: an instruction inside one M tile"
                )
            fmt, nbytes = AccFmt.F32, 0
            # the down op's operand in its B layout by absolute K chunk: codes [n_out / 256][CHUNK_B], scales after
            act = torch.zeros(n_out // 256 * (CHUNK_B + SFB_CHUNK), dtype=torch.uint8, device=self.device)
            sums = torch.zeros(mtiles * NT * TILE_ROWS, dtype=torch.float32, device=self.device)
            tile_flags = torch.zeros(mtiles, dtype=torch.int32, device=self.device)
            self.keep += [act, sums, tile_flags]
            p = [act.data_ptr(), sums.data_ptr(), tile_flags.data_ptr(), 0]
        elif logits is not None:  # the LM head
            assert logits.dtype == torch.bfloat16 and logits.numel() >= NT * rows
            epi = Epi.LOGITS
            ranges = [(tile_start * spm, tile_end * spm) for tile_start, tile_end in self.split(mtiles, n_inst)]
            x_out, fmt, nbytes = logits, AccFmt.F32, 0
            self.keep.append(logits)
        else:
            load = None if balance is None else [r1 - r0 for r0, r1 in self.acc[balance]["ranges"]]
            ranges = self.split(stages, n_inst) if load is None else self.split_balanced(stages, load)
            assert balance is None or is_exact, "balance: an uneven split needs order-independent sums"
            parts = max(self.parts_per_tile(ranges, spm, mtiles))
            if is_aligned and parts > 2:
                ranges = self.split_aligned(spm, mtiles, n_inst)
                parts = max(self.parts_per_tile(ranges, spm, mtiles))
            if is_exact and not is_aligned and not has_ll_parts and parts > 2:  # (ACC_CNT)
                penalty = self.SEG_PEN if kind == Kind.NVFP4 else self.SEG_PEN_FP8
                ranges = self.split_seg(stages, spm, penalty, load or [0] * n_inst)
                parts = max(self.parts_per_tile(ranges, spm, mtiles))
                assert parts > 2
            fmt = AccFmt.CNT if is_exact and parts > 2 else AccFmt.F32
            fin = fin and fmt == AccFmt.CNT
            # (FIN: no accumulator, nothing to zero; ACC_F32: 4-byte sums, ACC_CNT: 8-byte counted words)
            nbytes = 0 if fin else NT * mtiles * TILE_ROWS * (4 if fmt == AccFmt.F32 else 8)
            if has_ll_parts:
                assert fmt == AccFmt.F32 and parts <= 2, "ll_parts: <= 2 parts per M tile, fp32 sums"
                qw = torch.zeros(2 * NT * mtiles * TILE_ROWS, dtype=torch.int64, device=self.device)
                self.keep.append(qw)
                p = [0, 0, 0, qw.data_ptr()]
            if fmt == AccFmt.CNT:  # the bf16 output the completing parts store, the parts per M tile
                assert parts < MAX_PARTS
                tile_parts = torch.tensor(
                    self.parts_per_tile(ranges, spm, mtiles), dtype=torch.uint8, device=self.device
                )
                if val:  # two copies by launch parity, at the sentinel 0xFFFF until written
                    assert not x_out and not r_out, (
                        "a validated op: no prologue stores (its count does not release them)"
                    )
                    output = torch.full((2, NT * mtiles * TILE_ROWS), -1, dtype=torch.int16, device=self.device).view(
                        torch.bfloat16
                    )
                    p = [output.data_ptr(), tile_parts.data_ptr(), output.stride(0) * output.element_size(), 0]
                else:
                    output = torch.zeros(NT * mtiles * TILE_ROWS, dtype=torch.bfloat16, device=self.device)
                    p = [output.data_ptr(), tile_parts.data_ptr(), 0, 0]
                if has_ll_out:
                    assert not val, "ll_out: not with the validated hand-off"
                    llw = torch.zeros(NT * mtiles * TILE_ROWS, dtype=torch.int32, device=self.device)
                    p[3] = llw.data_ptr()
                    self.keep.append(llw)
                self.keep += [output, tile_parts]
            else:
                assert not val and not has_ll_out, "the validated hand-off and the tagged words are ACC_CNT's"

        if is_paired:  # the weight scales in the paired tiles' order (one 2 KB atom group per stage)
            sf = self.paired_scales(sf, n_out, K)
            self.keep.append(sf)
        self.check_prologue(pro, K, kind, src, ranges, spm, res_bf16)
        # (megakernel.cu rms_rsqrt5120: rsqrt.approx.ftz of a mean + eps >= eps)
        if pro in (Pro.NORM_IN, Pro.NORM_RESID):
            assert math.isfinite(eps) and eps >= self.RMS_EPS_MIN, (
                f"RMSNorm eps {eps}: rms_rsqrt5120 needs a normal eps"
            )

        acc = torch.zeros(2, max(nbytes, self.ACC_WORD_BYTES), dtype=torch.uint8, device=self.device)
        self.keep.append(acc)
        fin_tag, fin_ws, fin_parts = 0, 0, 0
        if not is_paired and logits is None and fin:
            fin_tag, fin_ws, fin_parts = self.fin_tables(ranges, spm, mtiles)
        # (megakernel.cu zero_next_acc: the copies' 16-byte words times the instruction count in 32 bits)
        assert (
            max(nbytes, self.ACC_WORD_BYTES) // self.ACC_WORD_BYTES * n_inst < self.U32_LIMIT
            and NT * mtiles * TILE_ROWS * 2 // self.ACC_WORD_BYTES * n_inst < self.U32_LIMIT
        )
        # (xn and twin: set by the ops added after it; ll, gate_ll and qstash: 0)
        self.descs.append(
            DESC.pack(
                w.data_ptr(), self.tmap(w, TILE_ROWS // 2 if is_paired else TILE_ROWS), self.ptr(sf), self.ptr(in0),
                self.ptr(res_in), self.ptr(nw), self.ptr(x_out), self.ptr(r_out),
                acc.data_ptr() if epi == Epi.RED else 0, 0, *p,
                alpha, gs, eps, float(act_gs or 0.0),
                K, n_out, kind, pro, src, fmt, n_inst, int(is_paired), mtiles, nbytes,
                int(res_bf16), epi, int(val), 0, 0, 0, 0, 0, act_stride, fin_tag, fin_ws, fin_parts,
                0, 0, 0, 0, 0,
            )
        )  # fmt: skip

        desc = len(self.descs) - 1
        self.acc[desc] = dict(t=acc, fmt=fmt, rows=rows, spm=spm, ranges=ranges, act=act, val=val, llw=llw, qw=qw)
        if has_ll_parts:
            self.polled_epi_descs.add(desc)
        if pro == Pro.COPY_B:  # the previous op's prologue encodes this op's B (its twin)
            prev_op = INSTR.unpack(self.sm_instrs[0][-1])[InstrField.DESC_INDEX]
            self.set_field(prev_op, DESC_TWIN, 1)

        ctr = self.new_counter(2 * n_inst)  # every instruction's prologue and epilogue
        self.ctr_val[ctr] = val
        done_ctr = ctr | (DONE_RELAXED if val else 0)
        for sm, (r0, r1) in enumerate(ranges):
            self.sm_instrs[sm].append(INSTR.pack(Op.GEMV, desc, dep, done_ctr, r0, r1, sm, 0, 0, 0))

        return ctr, desc

    def add_gdn_t(
        self,
        *,
        dep: int,
        conv_w: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        ll: torch.Tensor,
        scale: float,
        src_qkvz: int,
        run: torch.Tensor,  # the layer's running state
        rec: torch.Tensor,  # its replay record (2 copies)
        step: int,  # the device address of the step's OP_GDNT words
        group: int,  # the layer's mamba group among the step's block-table rows
        # out_proj's gated RMSNorm in the output phase (GT_GNORM), which writes out_proj's FP8 operand: dict(act=,
        # act_stride=, nw=, eps=, s_in=, ssx=, zw= (None: layer 0's qkvz sums), zrs=)
        gnorm: dict,
        src_ba: int = NO_CTR,
        ba_w: torch.Tensor | None = None,
        reads_src_words: bool = False,
        pre_at: list[int] | None = None,  # (mark()) the replay as OP_GDNT_PRE instructions there, the core GT_REPLAYED
    ) -> int:
        """The Gated DeltaNet core of a T-token block under replaySSM (OP_GDNT, gdn_t.cuh) on the qkvz projection
        src_qkvz; returns its completion counter."""
        # inputs: in_proj_qkv's tagged words (reads_src_words, Q_LL) or its fp32 accumulator after the dependency; b / a
        # from the dot products with the normed input its prologue left (ba_w) or the in_proj_ba op src_ba's bf16 output
        qkvz_acc = self.acc[src_qkvz]
        assert qkvz_acc["fmt"] == AccFmt.F32, "the GDN core reads the qkvz sums as fp32"
        assert (ba_w is None) != (src_ba == NO_CTR), "b / a from an op or from the core's dot products"
        qkvz_desc = DESC.unpack(self.descs[src_qkvz])
        # (token-major: a channel's T tokens one row of the projection apart)
        qw_t = qkvz_desc[OpDescField.MTILES] * TILE_ROWS
        if reads_src_words:
            assert qkvz_acc["qw"] is not None
            in_mode, qw, qw_part = GT_IN_LL, qkvz_acc["qw"].data_ptr(), qkvz_acc["qw"].numel() // 2
        else:
            in_mode, qw, qw_part = GT_IN_ACC, qkvz_acc["t"].data_ptr(), qkvz_acc["t"].stride(0)

        # the dot products' sums, left in smem by in_proj_qkv's prologue (OpDesc::xn: ba is its rows)
        if ba_w is not None:
            assert ba_w.dtype == torch.bfloat16 and ba_w.is_contiguous() and ba_w.shape == (2 * GDN_V_HEADS, K_NORM)
            ba_mode, ba, ba_stride = GT_BA_SMEM, ba_w.data_ptr(), 0
        else:
            ba_acc = self.acc[src_ba]
            assert ba_acc["fmt"] == AccFmt.CNT and not ba_acc["val"]
            ba_desc = DESC.unpack(self.descs[src_ba])
            # the op's bf16 output and its parity stride (OpDesc p[0], p[2])
            ba_mode, ba, ba_stride = GT_BA_OP, ba_desc[OpDescField.P0], ba_desc[OpDescField.P2]

        alpha = qkvz_desc[OpDescField.ALPHA]
        assert conv_state.dim() == 3 and conv_state.stride(2) == 1 and ssm_state[0].is_contiguous()
        self.keep += [conv_state, ssm_state, out, ll, run, rec] + ([ba_w] if ba_w is not None else [])
        assert gnorm["act"].dtype == torch.uint8
        assert gnorm["act"].numel() >= GDN_V_HEADS * CHUNK_B + gnorm["act_stride"]
        assert gnorm["ssx"].dtype == torch.int64 and gnorm["ssx"].numel() >= GDN_V_HEADS * self.GDN_HEAD_ITEMS * NT
        self.keep += [gnorm["act"], gnorm["nw"], gnorm["ssx"]] + ([gnorm["zw"]] if gnorm["zw"] is not None else [])

        self.descs.append(
            GDNT.pack(
                conv_w.data_ptr(), a_log.data_ptr(), dt_bias.data_ptr(), run.data_ptr(), ssm_state.data_ptr(),
                conv_state.data_ptr(), rec.data_ptr(), qw, ll.data_ptr(), out.data_ptr(), step, ba, 0,
                scale, alpha,
                conv_state.stride(0), ssm_state.stride(0), conv_state.shape[1], qw_part, 1, group, in_mode, ba_mode,
                (GT_REPLAYED if pre_at is not None else 0) | GT_GNORM, ba_stride, 0, qw_t,
                self.ptr(gnorm["zw"]), self.ptr(gnorm["act"]), self.ptr(gnorm["nw"]), self.ptr(gnorm["ssx"]),
                float(gnorm["eps"]), 1.0 / gnorm["s_in"], int(gnorm["zrs"]), int(gnorm["act_stride"]),
            )
        )  # fmt: skip

        desc = len(self.descs) - 1
        if ba_w is not None:  # in_proj_qkv's prologue computes the b / a sums of this SM's item
            self.set_field(src_qkvz, DESC_XN, 1 + desc)

        ctr = self.new_counter(2 * self.nsm)  # every instruction twice: prologue warps and epilogue warps
        for sm in range(self.nsm):
            self.sm_instrs[sm].append(INSTR.pack(Op.GDNT, desc, dep, ctr, 0, 0, sm, 0, 0, 0))
            if pre_at is not None:  # (the same item on the same SM: the replay's state and window stay on it)
                self.sm_instrs[sm].insert(pre_at[sm], INSTR.pack(Op.GDNT_PRE, desc, NO_CTR, NO_CTR, 0, 0, sm, 0, 0, 0))

        return ctr

    def mark(self) -> list[int]:
        """The SMs' current instruction counts (where add_gdn_t's pre_at inserts the replay instructions)."""
        return [len(sm_list) for sm_list in self.sm_instrs]

    def add_attn(
        self,
        *,
        dep: int,
        src: int,
        kv_cache: torch.Tensor,
        kb_per_mb: int,
        q_norm_w: torch.Tensor,
        k_norm_w: torch.Tensor,
        cos_sin: torch.Tensor,
        rec: int,
        scale: float,
        eps: float,
        sections: tuple[int, int, int],
        zero_ctr: int | None,
        o_s_in: float,
        att_group: int = 0,  # the layer's attention KV cache group among the step's (its block-table row in the record)
    ) -> int:
        """A full-attention core of T <= ATT_NT tokens on the qkv projection op src: OP_ATTN (kv heads x splits) then
        OP_ATTN_COMBINE, which publishes o_proj's FP8 operand (act_of_ctr); returns the combine's counter."""
        assert kv_cache.dtype == torch.bfloat16 and kv_cache.is_contiguous()
        # tokens per kernel block: 64, or 32 when 64 does not divide vLLM's block size (800 tokens at 3 drafts per step)
        block_tokens = kv_cache.shape[2]
        assert block_tokens in (ATT_PAGE // 2, ATT_PAGE), "kernel blocks"
        assert tuple(kv_cache.shape[1:]) == (ATT_KV_HEADS, block_tokens, ATT_ROW_BYTES // 2), "kernel blocks"
        assert self.acc[src]["fmt"] == AccFmt.F32, "the attention core reads the qkv sums as fp32"
        assert cos_sin.dtype == torch.bfloat16 and cos_sin.is_contiguous()
        n_splits = max(1, self.nsm // ATT_KV_HEADS)
        assert n_splits <= ATT_MAX_SPLITS

        rows = kv_cache.shape[0] * ATT_KV_HEADS * block_tokens
        tmap = torch.ops.pear_deploy_blackwell_nvfp4.make_tmap(kv_cache, rows, ATT_ROW_BYTES, block_tokens).to(
            self.device
        )
        # the splits' partials: o, bf16 blocks [query head][quarter][split][tokens][64]; m, l [query head][split][T][2]
        po = torch.zeros(
            ATT_Q_HEADS * self.ATT_HEAD_QUARTERS * n_splits * ATT_PO_BLOCK, dtype=torch.bfloat16, device=self.device
        )
        ml = torch.zeros(ATT_Q_HEADS * n_splits * ATT_NT * 2, dtype=torch.float32, device=self.device)
        act = torch.zeros(ATT_OUT // STAGE_BYTES * CHUNK_B, dtype=torch.uint8, device=self.device)  # o_proj's B layout
        self.keep += [tmap, po, ml, act, kv_cache, q_norm_w, k_norm_w, cos_sin]

        self.descs.append(
            ATTN.pack(
                kv_cache.data_ptr(), tmap.data_ptr(), q_norm_w.data_ptr(), k_norm_w.data_ptr(), cos_sin.data_ptr(),
                po.data_ptr(), act.data_ptr(), rec, scale * LOG2E, eps, o_s_in, 0.0, src, n_splits, kb_per_mb,
                sections[1], sections[2], att_group, 0, ml.data_ptr(), block_tokens.bit_length() - 1,
            )
        )  # fmt: skip

        desc = len(self.descs) - 1
        n = ATT_KV_HEADS * n_splits
        c_split = self.new_counter(2 * n)
        zero_aux = zero_ctr + 1 if zero_ctr is not None else 0  # the cache write waits for the step's block zeroing

        # a dry copy of each combine item before the split on its SM (no dependency, no counter): the combine's code is
        # warmed while the prologue warps wait for the qkv projection
        for i in range(ATT_COMBINE_ITEMS):
            sm = (self.nsm - 1 - i) % self.nsm
            self.sm_instrs[sm].append(INSTR.pack(Op.ATTN_COMBINE, desc, NO_CTR, NO_CTR, i, 0, i, AUX_DRY, 0, 0))
        for i in range(n):
            self.sm_instrs[i % self.nsm].append(INSTR.pack(Op.ATTN, desc, dep, c_split, i, 0, i, zero_aux, 0, 0))

        c_comb = self.new_counter(2 * ATT_COMBINE_ITEMS)
        self.act_of_ctr[c_comb] = act
        for i in range(ATT_COMBINE_ITEMS):
            sm = (self.nsm - 1 - i) % self.nsm
            self.sm_instrs[sm].append(INSTR.pack(Op.ATTN_COMBINE, desc, c_split, c_comb, i, 0, i, 0, 0, 0))

        return c_comb

    def add_spec_sample(self, *, lm_ctr: int, logits: torch.Tensor, ring: dict, rec: int) -> int:
        """The spec sampler's phases A to C after the LM head (its counter lm_ctr, its logits [NT, V]) on the prologue
        warps of every SM, into ring's SS_RING outputs (RING slots of NT rows); returns the last phase's counter."""
        # A: item i (row, chunk) of the step's T rows (min(LP_MAX_CHUNKS, nsm // T) chunks each), a row's last chunk
        # done its nucleus and test; B: a slice of the vocabulary per SM; C: the resampled token on SM 0
        nt, V = logits.shape
        assert (self.nsm // nt) * LP_CHUNK_ELEMS >= V and self.nsm * SS_SLICE_ELEMS >= V and V < SS_MAX_VOCAB

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
        outs = [ring[k] for k in SS_RING]
        self.keep += scratch + outs
        self.descs.append(
            SSD.pack(
                logits.data_ptr(),
                *(tensor.data_ptr() for tensor in scratch),
                rec,
                *(tensor.data_ptr() for tensor in outs),
                V,
                LP_MAX_CHUNKS,
                nt,
                self.nsm,
            )
        )

        desc = len(self.descs) - 1
        c_a, c_b, c_c = self.new_counter(2 * self.nsm), self.new_counter(2 * self.nsm), self.new_counter(2)
        for sm in range(self.nsm):
            self.sm_instrs[sm].append(INSTR.pack(Op.SS_A, desc, lm_ctr, c_a, sm, 0, sm, 0, 0, 0))
        for sm in range(self.nsm):
            self.sm_instrs[sm].append(INSTR.pack(Op.SS_B, desc, c_a, c_b, sm, 0, sm, 0, 0, 0))
        self.sm_instrs[0].append(INSTR.pack(Op.SS_C, desc, c_b, c_c, 0, 0, 0, 0, 0, 0))

        return c_c

    def link_shared_b(
        self,
        prod: int,
        cons: int,
        *,
        rin: torch.Tensor | None = None,
        xn: bool = False,
        xs: bool = True,
        aux: torch.Tensor | None = None,
        ssq_prev: torch.Tensor | None = None,
        ba_w: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The finishers of FIN op prod build the B operand of its RMSNorm consumer cons; returns their x copies (xs)
        for the next finisher's residual."""
        prod_desc, cons_desc = list(DESC.unpack(self.descs[prod])), DESC.unpack(self.descs[cons])
        assert prod_desc[OpDescField.FIN] and prod_desc[OpDescField.ACC_FMT] == AccFmt.CNT, "shared B: a FIN producer"
        assert (
            cons_desc[OpDescField.PRO] == Pro.NORM_RESID
            and cons_desc[OpDescField.K_DIM] == K_NORM
            and cons_desc[OpDescField.SRC] == prod
        ), "shared B: its RMSNorm consumer"
        mtiles, kind = prod_desc[OpDescField.MTILES], cons_desc[OpDescField.KIND]
        assert (
            mtiles * TILE_ROWS == K_NORM
            and prod_desc[OpDescField.N_OUT] == K_NORM
            and mtiles <= self.SHARED_B_MAX_TILES
        ), "shared B: a finisher per 128 channels"
        assert kind in (Kind.FP8, Kind.NVFP4) and not (xn and kind != Kind.FP8)

        # B by launch parity: the codes [chunks][CHUNK_B], an NVFP4 operand's scales [K / 256][SFB_CHUNK] after them,
        # each word at its sentinel until written
        stride = SHARED_B_STRIDE[kind]
        b = torch.full((2, stride), self.B_SENTINEL, dtype=torch.uint8, device=self.device)
        if kind == Kind.NVFP4:
            # (the codes' sentinel nibble 0x8; the scales' byte 0xFF)
            b[:, : K_NORM // 256 * CHUNK_B] = self.NVFP4_CODE_SENTINEL

        # xn, in_proj_ba's input: the finishers' b / a partial dot products with its rows ba_w [2][mtiles][96][NT] fp32,
        # else bf16(x_n) [2][NT][K]
        baw = ba_w if xn and ba_w is not None else None
        if baw is not None:
            assert baw.dtype == torch.bfloat16 and baw.is_contiguous() and baw.shape == (2 * GDN_V_HEADS, K_NORM)
            xnb = torch.full((2, mtiles, 2 * GDN_V_HEADS, NT), -1, dtype=torch.int32, device=self.device)
        else:
            xnb = torch.full((2, NT, K_NORM), -1, dtype=torch.int16, device=self.device) if xn else None

        ssq = torch.full((2, mtiles, NT), -1, dtype=torch.int32, device=self.device)
        # (r_out: bf16(x); x_out: fp32 x; the LM head: none)
        x_bf16 = bool(cons_desc[OpDescField.R_OUT]) and not cons_desc[OpDescField.X_OUT]
        cst = torch.full((2, NT), -1, dtype=torch.int32, device=self.device)  # c_t published (sentinel ~0u)
        # xs: this finisher's x copies for the next finisher's residual; rin: the previous finisher's (None: res_in)
        xsb = None
        if (
            xs
            and (cons_desc[OpDescField.X_OUT] or cons_desc[OpDescField.R_OUT])
            and cons_desc[OpDescField.EPI] != Epi.LOGITS
        ):
            xsb = torch.full((2, NT, K_NORM), -1, dtype=torch.int16 if x_bf16 else torch.int32, device=self.device)

        # aux (an aux layer, the consumers reading B): the draft's bf16 residual rows [NT][K], written by the finishers
        # with their x copies and released to draft_rows_ctr (one arrival per tile: mtiles per launch)
        aux_ctr = 0
        if aux is not None:
            assert xsb is not None and x_bf16 and aux.dtype == torch.bfloat16 and aux.numel() >= NT * K_NORM
            if self.draft_rows_ctr is None:
                self.draft_rows_ctr = self.new_counter(0)
            aux_ctr = self.draft_rows_ctr
            self.ctr_count[aux_ctr] += mtiles  # (one finisher per tile)
            self.draft_row_stages[prod] = (
                prod_desc[OpDescField.K_DIM] // (2 if prod_desc[OpDescField.KIND] == Kind.NVFP4 else 1) // STAGE_BYTES
            )

        # (outs 0: the finishers store no bf16 outputs, the consumers reading B; erstd 1: the consumer's epilogue
        # applies the rstd, the finishers quantizing with no exchange wait)
        ext = torch.frombuffer(bytearray(SHARED_B_EXT.pack(
            ssq.data_ptr(), xsb.data_ptr() if xsb is not None else 0, rin.data_ptr() if rin is not None else 0,
            cons_desc[OpDescField.RES_IN], cons_desc[OpDescField.NW], aux.data_ptr() if aux is not None else 0,
            b.data_ptr(), xnb.data_ptr() if xnb is not None else 0,
            cons_desc[OpDescField.GS], cons_desc[OpDescField.EPS], kind, cons_desc[OpDescField.RES_BF16], int(x_bf16),
            aux_ctr, stride, mtiles,
            0, 1,
            ssq_prev.data_ptr() if ssq_prev is not None else 0, cst.data_ptr(),
            baw.data_ptr() if baw is not None else 0,
        )), dtype=torch.uint8).to(self.device)  # fmt: skip
        assert rin is None or rin.dtype == (torch.int16 if cons_desc[OpDescField.RES_BF16] else torch.int32), (
            "shared B: the residual's dtype"
        )
        self.keep += [tensor for tensor in (b, xnb, ssq, xsb, ext, cst, baw) if tensor is not None]
        self.shared_b_last_ssq = ssq

        # (shared_xn's bit 0 tells the consumer's pro_ba_pub that xn holds the finishers' b / a partials)
        (
            prod_desc[OpDescField.SHARED_B_CONS],
            prod_desc[OpDescField.SHARED_B_PARITY_STRIDE],
            prod_desc[OpDescField.SHARED_B],
            prod_desc[OpDescField.SHARED_XN],
            prod_desc[OpDescField.SHARED_B_EXT_ADDR],
        ) = (
            cons + 1,
            stride,
            b.data_ptr(),
            (xnb.data_ptr() | int(baw is not None)) if xnb is not None else 0,
            ext.data_ptr(),
        )
        self.descs[prod] = DESC.pack(*prod_desc)
        self.shared_b_producer[cons] = prod

        return xsb

    def shared_b_consumer(self, cons: int) -> None:
        """RMSNorm consumer cons takes its B operand from its producer's finishers (link_shared_b first): PRO_ACT on the
        published copies, each 16-byte word validated (megakernel.cu stage_pub)."""
        # the norm's fields stay (nw marks a shared-B consumer); its dependency stays the producer's count (a gate: no
        # polling of the copies through the producer's tail); in_proj_qkv's b / a from the producer's xn (pro_ba_pub)
        prod = self.shared_b_producer[cons]
        prod_desc, cons_desc = DESC.unpack(self.descs[prod]), DESC.unpack(self.descs[cons])
        assert (
            cons_desc[OpDescField.PRO] == Pro.NORM_RESID
            and cons_desc[OpDescField.K_DIM] == K_NORM
            and cons_desc[OpDescField.SRC] == prod
            and prod_desc[OpDescField.SHARED_B_CONS] == cons + 1
        )
        assert prod_desc[OpDescField.SHARED_B_PARITY_STRIDE] == SHARED_B_STRIDE[Kind(cons_desc[OpDescField.KIND])], (
            "shared B: the format's parity stride"
        )

        self.set_field(cons, OpDescField.PRO, Pro.ACT)
        self.set_field(cons, OpDescField.IN0, prod_desc[OpDescField.SHARED_B])
        self.set_field(cons, OpDescField.ACT_STRIDE, prod_desc[OpDescField.SHARED_B_PARITY_STRIDE])
        # (its shared_b_ext: the producer's SharedBExt, its ssq slots for the epilogue)
        self.set_field(cons, OpDescField.SHARED_B_EXT_ADDR, prod_desc[OpDescField.SHARED_B_EXT_ADDR])

    def shared_b_twin(self, twin: int, cons: int) -> None:
        """Give in_proj_qkv's twin on the same B (in_proj_z) the shared_b_ext of consumer cons, its producer's
        SharedBExt."""
        # the twin's epilogue applies the same rstd_t / c_t
        self.set_field(
            twin, OpDescField.SHARED_B_EXT_ADDR, DESC.unpack(self.descs[cons])[OpDescField.SHARED_B_EXT_ADDR]
        )

    def attach_draft(self, lists: list[list[bytes]], wait_ctrs: set[int]) -> None:
        """The draft's proposal after the step's instructions on every SM: its per-SM instruction lists (its op codes
        from the kernel's DK_OP_BASE; its descriptors and counters its own), and the step's counters it waits for."""
        assert len(lists) == self.nsm and self.draft_instrs is None
        self.draft_instrs, self.draft_wait_ctrs = lists, set(wait_ctrs)

    def finalize(self) -> dict:
        """The schedule's device tensors: its instructions marked, then the draft's lists after them."""
        self.mark_warm()
        self.mark_park()
        self.mark_perm()
        self.mark_gate()
        self.mark_epi_relaxed()
        self.mark_dead()
        self.mark_split()
        self.mark_dep_early()
        self.mark_znext()
        self.mark_twin()
        self.mark_fastact()
        self.mark_shared_b()

        # the draft's lists join after the marks, which read this kernel's instruction layout
        instrs, offsets, draft_off = [], [0], []
        for sm, sm_list in enumerate(self.sm_instrs):
            instrs.extend(sm_list)
            draft_off.append(len(instrs))
            if self.draft_instrs is not None:
                instrs.extend(self.draft_instrs[sm])
            offsets.append(len(instrs))

        return {
            "sm_draft": torch.tensor(draft_off, dtype=torch.int32).to(self.device),  # each SM's first draft instruction
            "instrs": torch.frombuffer(bytearray(b"".join(instrs)), dtype=torch.uint8).to(self.device),
            "sm_off": torch.tensor(offsets, dtype=torch.int32).to(self.device),
            "descs": torch.frombuffer(bytearray(b"".join(self.descs)), dtype=torch.uint8).to(self.device),
            "counters": torch.zeros(max(1, len(self.ctr_count)), dtype=torch.int32, device=self.device),
            "ctr_count": torch.tensor(self.ctr_count or [0], dtype=torch.int32).to(self.device),
            "epoch": torch.zeros(2, dtype=torch.int32, device=self.device),
            "keep": self.keep,  # the buffers the descriptors point to live as long as the schedule
        }

    def check_prologue(
        self, pro: Pro, K: int, kind: Kind, src: int, ranges: list[tuple[int, int]], spm: int, res_bf16: bool
    ) -> None:
        """Assert that the kernel implements this prologue for this op and its producing op."""
        max_chunks = max(min(r1 - r0, spm) for r0, r1 in ranges)
        assert max_chunks <= B_CHUNKS and (kind != Kind.NVFP4 or max_chunks <= SFB_CHUNKS), (
            "B: an instruction's K chunks"
        )

        if pro in (Pro.NORM_IN, Pro.NORM_RESID):
            assert K == K_NORM, "the RMSNorm prologues are written for the hidden size"
        if pro == Pro.NORM_RESID:
            assert self.acc[src]["val"], "an RMSNorm prologue reads a validated ACC_CNT op's output"
            assert kind == Kind.NVFP4 or not res_bf16, "an FP8 projection's residual is the fp32 stream"
            assert K * (2 if res_bf16 else 4) <= STG_BYTES, "a token row's residual in the staging buffer"

            # pro_norm_rows' stash of the residual slices (a CTA pair's: 2 tiles' 16-byte quads, every row) past the B
            # operand
            quads, n_inst = K // 4, len(ranges)
            pair_quads = max(quads * (sm + 2) // n_inst - quads * sm // n_inst for sm in range(n_inst))
            stash_off = (
                SFB_CHUNKS * (CHUNK_B + SFB_CHUNK) + self.SFB_COPY_BYTES - SFB_CHUNK // 2
                if kind == Kind.NVFP4
                else max_chunks * CHUNK_B
            )
            # (the CTA pair's per-row copy table past the stash: 8 bytes a prologue thread)
            table_bytes = self.PRO_THREADS * 8 if kind == Kind.NVFP4 else 0
            assert stash_off + pair_quads * NT * 16 + table_bytes <= B_CHUNKS * CHUNK_B, (
                "the residual slices' stash past the B operand"
            )
        elif pro == Pro.NORM_IN:
            assert kind in (Kind.FP8, Kind.BF16), "the embedding's norm feeds the first layer's projections"
        elif pro == Pro.ACT:
            assert kind in (Kind.NVFP4, Kind.FP8), "PRO_ACT: an EPI_SILU op's NVFP4 or the combine's FP8 operand"
        elif pro == Pro.COPY_B:  # the previous GEMV instruction on every SM: active, same K and kind (the same input)
            for sm, (r0, r1) in enumerate(ranges):
                if r0 == r1:
                    continue
                prev_instr = INSTR.unpack(self.sm_instrs[sm][-1])
                assert prev_instr[InstrField.OP] == Op.GEMV and prev_instr[InstrField.R0] < prev_instr[InstrField.R1], (
                    "PRO_COPY_B: an active GEMV instruction right before"
                )
                prev_desc = DESC.unpack(self.descs[prev_instr[InstrField.DESC_INDEX]])
                assert (
                    prev_desc[OpDescField.K_DIM] == K
                    and prev_desc[OpDescField.KIND] == kind == Kind.FP8
                    and (
                        prev_desc[OpDescField.PRO] == Pro.NORM_RESID
                        or (prev_desc[OpDescField.PRO] == Pro.ACT and prev_desc[OpDescField.NW])
                    )
                ), "PRO_COPY_B: the twin of an FP8 RMSNorm prologue (shared B: of its consumer's staging)"

    def fin_tables(self, ranges: list[tuple[int, int]], spm: int, mtiles: int) -> tuple[int, int, int]:
        """A FIN op's tag (1 + its index among the step's FIN ops), the FIN ops' shared workspace of tagged words and
        its parts' first stages per M tile (a tile's first part finishes it: megakernel.cu epilogue_gemv)."""
        # the workspace: [mtiles][FIN_MAXP][FIN_W][128] 16-byte words, all ones (a tag never issued), the residual's
        # tiles at least; the first stages: [mtiles][FIN_MAXP] u8, 0xFF past the tile's parts
        self.fin_n += 1
        assert self.fin_n < self.FIN_TAG_LIMIT, "FIN: 255 ops at most (the words' 9-bit tag: launch parity, op)"
        if self.fin_ws is None or self.fin_ws.shape[0] < mtiles:
            self.fin_ws = torch.full(
                (max(mtiles, K_NORM // TILE_ROWS), self.FIN_MAXP, self.FIN_W, TILE_ROWS, 4),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.keep.append(self.fin_ws)

        starts = [[] for _ in range(mtiles)]
        for r0, r1 in ranges:
            for mt in range(r0 // spm, (r1 - 1) // spm + 1) if r1 > r0 else ():
                starts[mt].append(max(r0, mt * spm) - mt * spm)

        start_table = torch.full((mtiles, self.FIN_MAXP), self.FIN_NO_PART, dtype=torch.uint8)
        for mt, tile_starts in enumerate(starts):
            tile_starts = sorted(tile_starts)
            assert tile_starts and tile_starts[0] == 0 and len(tile_starts) <= self.FIN_MAX_TILE_PARTS, (
                "FIN: a tile's parts (megakernel.cu: <= 4 later parts)"
            )
            start_table[mt, : len(tile_starts)] = torch.tensor(tile_starts, dtype=torch.uint8)
        start_table = start_table.to(self.device)
        self.keep.append(start_table)

        return self.fin_n, self.fin_ws.data_ptr(), start_table.data_ptr()

    def tmap(self, w: torch.Tensor, box_rows: int) -> int:
        """An SW128 TMA descriptor (box 128 B x box_rows) over a row-major weight matrix; returns its device address."""
        tma_desc = torch.ops.pear_deploy_blackwell_nvfp4.make_tmap(
            w, w.shape[0], w.shape[1] * w.element_size(), box_rows
        )
        tma_desc = tma_desc.to(self.device)
        self.keep.append(tma_desc)
        return tma_desc.data_ptr()

    def mark_warm(self) -> None:
        """AUX_WARM, a dry run of the prologue code before the dependency wait, on the GEMV instructions whose prologue
        warps are a whole op ahead (the SM's previous GEMV instruction of another dependency, the op that SM is still
        streaming)."""
        # the dry run warms the instruction cache
        for sm_list in self.sm_instrs:
            for k in range(1, len(sm_list)):
                fields, j = list(INSTR.unpack(sm_list[k])), k - 1
                # (the replay instructions: not a stream)
                while j > 0 and INSTR.unpack(sm_list[j])[InstrField.OP] == Op.GDNT_PRE:
                    j -= 1

                prev_fields = INSTR.unpack(sm_list[j])
                if (
                    fields[InstrField.OP] == Op.GEMV
                    and prev_fields[InstrField.OP] == Op.GEMV
                    and fields[InstrField.DEP] != prev_fields[InstrField.DEP]
                    and fields[InstrField.DEP] != NO_CTR
                    and fields[InstrField.R0] < fields[InstrField.R1]
                ):
                    fields[InstrField.AUX] |= AUX_WARM
                    sm_list[k] = INSTR.pack(*fields)

    def mark_park(self) -> None:
        """The TMEM park slots of the GEMV instructions and the FP8 output projections' pacing limits."""
        # any instruction may park: the kernel's TMEM columns past the accumulators are written only by the MMA
        # issuer's copies, in order with its MMAs
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if fields[InstrField.OP] != Op.GEMV or fields[InstrField.R0] >= fields[InstrField.R1]:
                    continue

                desc = DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]])
                if desc[OpDescField.KIND] == Kind.NVFP4:
                    n = min(self.PARK_NV, MAX_PARK_NV, fields[InstrField.R1] - fields[InstrField.R0])
                elif desc[OpDescField.KIND] == Kind.FP8:
                    # (shared B: a consumer of its producer's published B is an input projection)
                    is_input = desc[OpDescField.PRO] in (Pro.NORM_RESID, Pro.NORM_IN) or (
                        desc[OpDescField.PRO] == Pro.ACT and desc[OpDescField.NW]
                    )
                    if not is_input:
                        pace = self.PACE_GN if fields[InstrField.DESC_INDEX] in self.gn_out else self.PACE_ATT
                        fields[InstrField.AUX] |= pace << AUX_PACE_SHIFT
                    n = min(
                        self.PARK_IN if is_input else self.PARK_OUT,
                        MAX_PARK,
                        fields[InstrField.R1] - fields[InstrField.R0],
                    )
                else:
                    continue
                fields[InstrField.AUX] |= n << AUX_PARK_SHIFT
                sm_list[k] = INSTR.pack(*fields)

    def mark_perm(self) -> None:
        """AUX_PERM on the GEMV instructions of the PERM ops with more stages than park slots."""
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if (
                    fields[InstrField.OP] != Op.GEMV
                    or fields[InstrField.R0] >= fields[InstrField.R1]
                    or self.op_kind(fields[InstrField.DESC_INDEX]) not in self.PERM
                ):
                    continue

                if (
                    0
                    < (fields[InstrField.AUX] >> AUX_PARK_SHIFT) & self.AUX_PARK_MASK
                    < fields[InstrField.R1] - fields[InstrField.R0]
                ):
                    fields[InstrField.AUX] |= AUX_PERM
                    sm_list[k] = INSTR.pack(*fields)

    def mark_gate(self) -> None:
        """AUX_NOGATE on the GEMV instructions of the NOGATE ops."""
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if fields[InstrField.OP] == Op.GEMV and self.op_kind(fields[InstrField.DESC_INDEX]) in self.NOGATE:
                    fields[InstrField.AUX] |= AUX_NOGATE
                    sm_list[k] = INSTR.pack(*fields)

    def mark_epi_relaxed(self) -> None:
        """AUX_EPI_RELAXED on the instructions of the GEMVs whose epilogue writes only tagged words that their consumer
        polls (in_proj_qkv's sums per part, has_ll_parts)."""
        # the GDN core acquires the op's counter only for the residual slices its prologues wrote, released by their
        # counts
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if fields[InstrField.OP] == Op.GEMV and fields[InstrField.DESC_INDEX] in self.polled_epi_descs:
                    fields[InstrField.AUX] |= AUX_EPI_RELAXED
                    sm_list[k] = INSTR.pack(*fields)

    def mark_dead(self) -> None:
        """DONE_RELAXED on the instructions whose completion counter no instruction waits on."""
        # a relaxed count nobody acquires is as good as a release, whose fence waits for the SM's loads in flight (the
        # GDN cores of layers 1.., in_z, layer 0's in_qkvz and the LM head)
        wait_ctrs = {
            INSTR.unpack(instr)[InstrField.DEP] for sm_list in self.sm_instrs for instr in sm_list
        } | self.draft_wait_ctrs

        # (the counters waited on through aux: OP_ATTN's zeroing counter + 1)
        for sm_list in self.sm_instrs:
            for instr in sm_list:
                fields = INSTR.unpack(instr)
                if fields[InstrField.OP] == Op.ATTN and fields[InstrField.AUX]:
                    wait_ctrs.add(fields[InstrField.AUX] - 1)

        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if (
                    fields[InstrField.DONE] != NO_CTR
                    and not fields[InstrField.DONE] & DONE_RELAXED
                    and fields[InstrField.DONE] not in wait_ctrs
                ):
                    fields[InstrField.DONE] |= DONE_RELAXED
                    sm_list[k] = INSTR.pack(*fields)

    def mark_split(self) -> None:
        """AUX_SPLIT on the instructions of the residual RMSNorm prologues without in_proj_ba's dot products or a twin
        (gate_up, the attention's qkv, the LM head) whose CTA pair 2c, 2c + 1 (a cluster) builds their B together."""
        # each CTA builds half of the token rows into both B buffers, when both instructions are active (NVFP4: with
        # every K chunk, the rows copied from one B into the other); a pair's GEMV instructions match one to one (one
        # per op and CTA), so both take the same B buffer slot
        assert self.nsm % 2 == 0, "CTA pairs"

        for sm in range(0, self.nsm, 2):
            list_a, list_b = self.sm_instrs[sm], self.sm_instrs[sm + 1]
            gemvs_a = [k for k, instr in enumerate(list_a) if INSTR.unpack(instr)[InstrField.OP] == Op.GEMV]
            gemvs_b = [k for k, instr in enumerate(list_b) if INSTR.unpack(instr)[InstrField.OP] == Op.GEMV]
            assert len(gemvs_a) == len(gemvs_b), "a CTA pair's GEMV instructions"
            for index_a, index_b in zip(gemvs_a, gemvs_b, strict=True):
                fields_a, fields_b = list(INSTR.unpack(list_a[index_a])), list(INSTR.unpack(list_b[index_b]))
                assert (
                    fields_a[InstrField.DESC_INDEX] == fields_b[InstrField.DESC_INDEX]
                    and fields_a[InstrField.TILE] == sm
                    and fields_b[InstrField.TILE] == sm + 1
                ), "a CTA pair's GEMV instructions"
                desc = DESC.unpack(self.descs[fields_a[InstrField.DESC_INDEX]])
                if (
                    desc[OpDescField.PRO] != Pro.NORM_RESID
                    or desc[DESC_XN]
                    or desc[DESC_TWIN]
                    or fields_a[InstrField.R0] >= fields_a[InstrField.R1]
                    or fields_b[InstrField.R0] >= fields_b[InstrField.R1]
                ):
                    continue

                spm = desc[OpDescField.K_DIM] // 2 // STAGE_BYTES if desc[OpDescField.KIND] == Kind.NVFP4 else 0
                # NVFP4: every chunk in both B operands (the copy)
                if (
                    spm
                    and min(
                        fields_a[InstrField.R1] - fields_a[InstrField.R0],
                        fields_b[InstrField.R1] - fields_b[InstrField.R0],
                    )
                    < spm
                ):
                    continue

                assert fields_a[InstrField.DEP] != NO_CTR and fields_b[InstrField.DEP] != NO_CTR, (
                    "AUX_SPLIT: the dependency wait"
                )
                fields_a[InstrField.AUX] |= AUX_SPLIT
                fields_b[InstrField.AUX] |= AUX_SPLIT
                (
                    fields_a[InstrField.PR0],
                    fields_a[InstrField.PR1],
                    fields_b[InstrField.PR0],
                    fields_b[InstrField.PR1],
                ) = (
                    fields_b[InstrField.R0],
                    fields_b[InstrField.R1],
                    fields_a[InstrField.R0],
                    fields_a[InstrField.R1],
                )
                list_a[index_a], list_b[index_b] = INSTR.pack(*fields_a), INSTR.pack(*fields_b)

    def mark_dep_early(self) -> None:
        """AUX_DEP_EARLY (DEP_EARLY counts) on the RMSNorm GEMV instructions whose dependency is a validated op's
        count."""
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if (
                    fields[InstrField.OP] != Op.GEMV
                    or fields[InstrField.DEP] == NO_CTR
                    or not self.ctr_val.get(fields[InstrField.DEP])
                ):
                    continue
                if DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]])[OpDescField.PRO] != Pro.NORM_RESID:
                    continue

                fields[InstrField.AUX] |= self.DEP_EARLY << AUX_DEP_EARLY_SHIFT
                sm_list[k] = INSTR.pack(*fields)

    def mark_znext(self) -> None:
        """AUX_ZNEXT on each OP_GDNT instruction whose next instruction on the SM is an active out_proj GEMV
        instruction, and on that one (megakernel.cu: the core's warps 2-7 zero its next-parity accumulator slice)."""
        for sm_list in self.sm_instrs:
            for k in range(len(sm_list) - 1):
                fields, next_fields = list(INSTR.unpack(sm_list[k])), list(INSTR.unpack(sm_list[k + 1]))
                if (
                    fields[InstrField.OP] != Op.GDNT
                    or next_fields[InstrField.OP] != Op.GEMV
                    or next_fields[InstrField.DESC_INDEX] not in self.gn_out
                    or next_fields[InstrField.R0] >= next_fields[InstrField.R1]
                ):
                    continue

                fields[InstrField.AUX] |= AUX_ZNEXT
                next_fields[InstrField.AUX] |= AUX_ZNEXT
                sm_list[k], sm_list[k + 1] = INSTR.pack(*fields), INSTR.pack(*next_fields)

    def mark_twin(self) -> None:
        """AUX_TWIN on each active PRO_COPY_B GEMV instruction whose previous instruction on the SM is an active GEMV
        instruction with that twin (OpDesc.twin) and the same dependency."""
        for sm_list in self.sm_instrs:
            for k in range(1, len(sm_list)):
                prev_fields, fields = list(INSTR.unpack(sm_list[k - 1])), list(INSTR.unpack(sm_list[k]))
                if (
                    prev_fields[InstrField.OP] != Op.GEMV
                    or fields[InstrField.OP] != Op.GEMV
                    or prev_fields[InstrField.R0] >= prev_fields[InstrField.R1]
                    or fields[InstrField.R0] >= fields[InstrField.R1]
                ):
                    continue

                prev_desc, desc = (
                    DESC.unpack(self.descs[prev_fields[InstrField.DESC_INDEX]]),
                    DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]]),
                )
                if (
                    not prev_desc[DESC_TWIN]
                    or desc[OpDescField.PRO] != Pro.COPY_B
                    or (fields[InstrField.DEP] != NO_CTR and fields[InstrField.DEP] != prev_fields[InstrField.DEP])
                    or (fields[InstrField.AUX] & AUX_WARM)
                ):
                    continue

                fields[InstrField.AUX] |= AUX_TWIN
                sm_list[k] = INSTR.pack(*fields)

    def mark_fastact(self) -> None:
        """AUX_FASTACT and AUX_FAPRE on the active out_proj instructions marked AUX_ZNEXT (after the GDN core on their
        SM: validated FP8, no dependency, twin, split or warm-up; megakernel.cu stage_val), AUX_FAPRE on that core."""
        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if (
                    fields[InstrField.OP] != Op.GEMV
                    or fields[InstrField.R0] >= fields[InstrField.R1]
                    or not fields[InstrField.AUX] & AUX_ZNEXT
                    or fields[InstrField.DESC_INDEX] not in self.gn_out
                ):
                    continue

                desc = DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]])
                assert (
                    desc[OpDescField.PRO] == Pro.ACT
                    and desc[OpDescField.KIND] == Kind.FP8
                    and fields[InstrField.DEP] == NO_CTR
                    and not fields[InstrField.AUX] & (AUX_TWIN | AUX_SPLIT | AUX_WARM)
                )
                fields[InstrField.AUX] |= AUX_FASTACT | AUX_FAPRE
                sm_list[k] = INSTR.pack(*fields)

                prev_fields = list(INSTR.unpack(sm_list[k - 1]))
                assert prev_fields[InstrField.OP] == Op.GDNT and prev_fields[InstrField.AUX] & AUX_ZNEXT
                prev_fields[InstrField.AUX] |= AUX_FAPRE
                sm_list[k - 1] = INSTR.pack(*prev_fields)

    def mark_shared_b(self) -> None:
        """AUX_SHARED_B on the instructions of the ops whose finishers build a consumer's B (link_shared_b),
        AUX_DRAFT_ROWS on an aux layer's finisher instructions, and in each shared-B consumer instruction's
        AUX_DEP_EARLY bits its producer's finisher instructions."""
        # a finisher instruction ends in an M tile's first part and counts after the finisher's tail: the consumer's
        # gate is the producer's count less them, then it polls the B words
        producers, aux_stages = set(self.shared_b_producer.values()), self.draft_row_stages
        finisher_counts = collections.Counter()

        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if fields[InstrField.OP] == Op.GEMV and fields[InstrField.DESC_INDEX] in producers:
                    fields[InstrField.AUX] |= AUX_SHARED_B
                    spm = aux_stages.get(fields[InstrField.DESC_INDEX])
                    # (its last segment: a tile's first part)
                    if (
                        spm
                        and fields[InstrField.R1] > fields[InstrField.R0]
                        and (fields[InstrField.R1] - 1) // spm * spm >= fields[InstrField.R0]
                    ):
                        fields[InstrField.AUX] |= AUX_DRAFT_ROWS
                    sm_list[k] = INSTR.pack(*fields)
                    desc = DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]])
                    row_bytes = (
                        desc[OpDescField.K_DIM] // 2
                        if desc[OpDescField.KIND] == Kind.NVFP4
                        else desc[OpDescField.K_DIM]
                        if desc[OpDescField.KIND] == Kind.FP8
                        else 2 * desc[OpDescField.K_DIM]
                    )
                    spm = row_bytes // STAGE_BYTES
                    if (
                        fields[InstrField.R0] < fields[InstrField.R1]
                        and (fields[InstrField.R1] - 1) // spm * spm >= fields[InstrField.R0]
                    ):
                        finisher_counts[fields[InstrField.DESC_INDEX]] += 1

        for sm_list in self.sm_instrs:
            for k, instr in enumerate(sm_list):
                fields = list(INSTR.unpack(instr))
                if fields[InstrField.OP] != Op.GEMV or fields[InstrField.DEP] == NO_CTR:
                    continue

                desc = DESC.unpack(self.descs[fields[InstrField.DESC_INDEX]])
                if (
                    desc[OpDescField.PRO] == Pro.ACT
                    and desc[OpDescField.NW]
                    and desc[OpDescField.ACT_STRIDE]
                    and desc[OpDescField.SRC] in producers
                ):
                    assert (
                        not fields[InstrField.AUX] >> AUX_DEP_EARLY_SHIFT
                        and 0 < finisher_counts[desc[OpDescField.SRC]] < self.AUX_DEP_EARLY_LIMIT
                    )
                    fields[InstrField.AUX] |= finisher_counts[desc[OpDescField.SRC]] << AUX_DEP_EARLY_SHIFT
                    sm_list[k] = INSTR.pack(*fields)

    def op_kind(self, desc: int) -> str:
        """A GEMV op's role in the step, from its descriptor."""
        fields = DESC.unpack(self.descs[desc])
        n_out, kind, pro, is_paired, epi = (
            fields[OpDescField.N_OUT],
            fields[OpDescField.KIND],
            fields[OpDescField.PRO],
            fields[OpDescField.PAIRED],
            fields[OpDescField.EPI],
        )

        if kind == Kind.NVFP4:
            return "gate_up" if is_paired else "lm_head" if epi == Epi.LOGITS else "down"
        if kind == Kind.BF16:
            return "in_ba"
        if desc in self.gn_out:
            return "out_proj"
        # (shared B: an RMSNorm consumer taking its B from the producer: the norm's weights kept)
        if pro == Pro.ACT and fields[OpDescField.NW]:
            return "in_z" if n_out == GDN_V_HEADS * self.GDN_V_DIM else "qkv" if n_out == ATT_QKV_ROWS else "in_qkv"
        return {Pro.ACT: "o_proj", Pro.COPY_B: "in_z"}.get(pro, "qkv" if n_out == ATT_QKV_ROWS else "in_qkv")

    def set_field(self, desc: int, field: int, value: int) -> None:
        """Set an OpDesc field (DESC.unpack index) of an op already added."""
        fields = list(DESC.unpack(self.descs[desc]))
        fields[field] = value
        self.descs[desc] = DESC.pack(*fields)

    def new_counter(self, count: int) -> int:
        """A completion counter that count arrivals complete per launch."""
        self.ctr_count.append(count)
        return len(self.ctr_count) - 1

    @staticmethod
    def plain_desc(*, w: int, in0: int, rec: int, n_inst: int, res_in: int = 0, K: int = 0, n_out: int = 0) -> bytes:
        """An OpDesc of an op without weights or prologue (OP_ZERO, OP_EMBED)."""
        return DESC.pack(
            w, 0, 0, in0, res_in, 0, 0, 0, 0, rec, 0, 0, 0, 0,
            1.0, 1.0, 0.0, 0.0,
            K, n_out, 0, 0, 0, 0, n_inst, *([0] * 13), 0, 0, 0, 0, 0, 0, 0,  # (13: paired .. fin)
        )  # fmt: skip

    @staticmethod
    def ptr(tensor_or_address: torch.Tensor | int | None) -> int:
        """A tensor's device address, or the address given."""
        return (
            tensor_or_address.data_ptr() if isinstance(tensor_or_address, torch.Tensor) else int(tensor_or_address or 0)
        )

    @staticmethod
    def paired_scales(sf: torch.Tensor, n_out: int, K: int) -> torch.Tensor:
        """gate||up's NVFP4 weight scales (vLLM's swizzled layout [2 n_out / 128][K / 64][32][4][4] bytes) in the
        paired M tiles' order: paired tile t (outputs 64 t ..) takes row groups 2 (t % 2), 2 (t % 2) + 1 of gate atom
        t / 2, then the same of the up atom, as one [K / 64][32][4][4] atom row (megakernel.cu sf_atoms; datapath q
        reads row group q)."""
        assert n_out % 128 == 0 and K % 64 == 0
        # [g/u][atom][k][lane][m][q][b]
        atoms = sf.contiguous().view(torch.uint8).view(2, n_out // 128, K // 64, 32, 2, 2, 4)
        return atoms.permute(1, 4, 2, 3, 0, 5, 6).contiguous().view(-1)

    @staticmethod
    def split(stages: int, n_ranges: int) -> list[tuple[int, int]]:
        """Stream-K ranges: [stages * i / n_ranges, stages * (i + 1) / n_ranges) for i in [0, n_ranges)."""
        return [(stages * i // n_ranges, stages * (i + 1) // n_ranges) for i in range(n_ranges)]

    @staticmethod
    def split_balanced(stages: int, load: list[int]) -> list[tuple[int, int]]:
        """Contiguous ranges over the stages, one per SM, sized so that load[i] + the range of SM i is as even as
        possible."""
        n_ranges = len(load)
        level = (sum(load) + stages) / n_ranges
        shares = [max(0.0, level - x) for x in load]
        total = sum(shares)

        bounds = [0]
        cumulative = 0.0
        for share in shares:
            cumulative += share
            bounds.append(min(stages, round(cumulative / total * stages)))
        bounds[-1] = stages

        return [(bounds[i], bounds[i + 1]) for i in range(n_ranges)]

    @staticmethod
    def split_seg(stages: int, spm: int, penalty: float, load: list[int]) -> list[tuple[int, int]]:
        """Contiguous ranges over the stages, one per SM (len(load) of them), keeping load[i] + a range's stages +
        penalty * (the M tiles it crosses - 1) under the smallest bound a greedy left-to-right fill can meet
        (bisection)."""
        n_ranges = len(load)

        def longest_end(r0: int, cap: float) -> int:
            """The end of the longest range from r0 within cap: in its k-th M tile (k boundaries crossed) a range
            ending at range_end costs range_end - r0 + penalty * k."""
            best_end, k, base = r0, 0, r0 // spm
            while True:
                start, end = (r0 if k == 0 else (base + k) * spm), min(stages, (base + k + 1) * spm)
                if start >= stages:
                    break
                range_end = min(end, math.floor(r0 + cap - penalty * k))
                if range_end <= start:
                    break
                best_end = range_end
                if range_end < end:
                    break
                k += 1
            return best_end

        def fill(bound: float) -> list[tuple[int, int]] | None:
            ranges, r0 = [], 0
            for sm in range(n_ranges):
                r1 = longest_end(r0, bound - load[sm]) if r0 < stages else stages
                ranges.append((r0, r1))
                r0 = r1
            return ranges if r0 >= stages else None

        lower_bound, upper_bound = 0.0, float(max(load) + stages)
        for _ in range(Schedule.BISECT_ROUNDS):
            midpoint = (lower_bound + upper_bound) / 2
            if fill(midpoint) is None:
                lower_bound = midpoint
            else:
                upper_bound = midpoint

        ranges = fill(upper_bound)
        assert ranges is not None and len(ranges) == n_ranges and ranges[-1][1] == stages

        return ranges

    @staticmethod
    def split_last(stages: int, spm: int, epilogue_stages: float, load: list[int]) -> list[tuple[int, int]]:
        """Contiguous ranges over the stages, one per SM (len(load) of them), keeping load[i] + a range's stages +
        max(0, epilogue_stages - its last segment's stages) (ranges over more than one M tile) under the smallest bound
        any split meets (bisection on a greedy fill of the longest ranges), then within the bound each range as close
        to an even share of the stages left as the SMs after it allow."""
        n_ranges = len(load)

        def cost(r0: int, r1: int) -> float:
            last_stages = r1 - ((r1 - 1) // spm) * spm
            return r1 - r0 + (max(0.0, epilogue_stages - last_stages) if (r1 - 1) // spm != r0 // spm else 0.0)

        def longest_end(r0: int, cap: float) -> int:
            """The end of the longest range from r0 within cap: its cost is flat (tile_boundary + epilogue_stages - r0)
            over a later tile's first epilogue_stages stages, then r1 - r0, so each tile's feasible ends run up to r0 +
            cap."""
            tile_boundary = (r0 // spm + 1) * spm
            best_end = min(stages, tile_boundary, r0 + math.floor(cap))
            if best_end <= r0:
                return r0
            while best_end == tile_boundary and tile_boundary < stages and tile_boundary + epilogue_stages - r0 <= cap:
                best_end, tile_boundary = min(stages, tile_boundary + spm, r0 + math.floor(cap)), tile_boundary + spm
            return best_end

        def fill(bound: float) -> bool:
            r0 = 0
            for sm in range(n_ranges):
                r0 = longest_end(r0, bound - load[sm]) if r0 < stages else stages
            return r0 >= stages

        lower_bound, upper_bound = 0.0, float(max(load) + stages + epilogue_stages)
        for _ in range(Schedule.BISECT_ROUNDS):
            midpoint = (lower_bound + upper_bound) / 2
            lower_bound, upper_bound = (lower_bound, midpoint) if fill(midpoint) else (midpoint, upper_bound)
        bound = upper_bound

        # earliest_starts[sm]: the earliest start from which SMs sm.. cover the rest within the bound (their ends at
        # earliest_starts[sm + 1] or past it by less than a penalty's reach)
        earliest_starts = [stages] * (n_ranges + 1)
        for sm in range(n_ranges - 1, -1, -1):
            best_start = earliest_starts[sm + 1]
            for r1 in range(
                earliest_starts[sm + 1], min(stages, earliest_starts[sm + 1] + math.ceil(epilogue_stages)) + 1
            ):
                r0 = r1
                while r0 > 0 and cost(r0 - 1, r1) <= bound - load[sm]:
                    r0 -= 1
                best_start = min(best_start, r0)
            earliest_starts[sm] = best_start
        assert earliest_starts[0] == 0, "split_last: the bound's split"

        ranges, r0 = [], 0
        for sm in range(n_ranges):
            if sm == n_ranges - 1:
                ranges.append((r0, stages))
                break
            share = r0 + (stages - r0) / (n_ranges - sm)
            candidates = [
                r1
                for r1 in range(max(r0, earliest_starts[sm + 1]), longest_end(r0, bound - load[sm]) + 1)
                if r1 == r0 or cost(r0, r1) <= bound - load[sm]
            ]
            r1 = min(candidates, key=lambda end: (abs(end - share), end))
            ranges.append((r0, r1))
            r0 = r1
        assert (
            len(ranges) == n_ranges
            and ranges[-1][1] == stages
            and cost(*ranges[-1]) <= bound - load[-1] + Schedule.COST_SLACK
        )

        return ranges

    @staticmethod
    def parts_per_tile(ranges: list[tuple[int, int]], spm: int, mtiles: int) -> list[int]:
        """Number of (non-empty) ranges covering each M tile of spm stages."""
        counts = [0] * mtiles
        for r0, r1 in ranges:
            if r0 < r1:
                for mt in range(r0 // spm, (r1 - 1) // spm + 1):
                    counts[mt] += 1
        return counts

    @classmethod
    def split_aligned(cls, spm: int, mtiles: int, n_ranges: int) -> list[tuple[int, int]]:
        """n_ranges stream-K ranges over mtiles M tiles of spm stages with at most two ranges per M tile."""
        # x blocks of g tiles split evenly into g + 1 ranges and y blocks of g - 1 tiles into g ranges
        # (g x + (g - 1) y = mtiles, (g + 1) x + g y = n_ranges), g chosen for the shortest longest range, the block
        # sizes interleaved evenly
        best_split = None
        for g in range(2, mtiles + 1):
            x, y = g * mtiles - (g - 1) * n_ranges, g * n_ranges - (g + 1) * mtiles
            if x < 0 or y < 0:
                continue
            longest_len = max(-(-g * spm // (g + 1)) if x else 0, -(-(g - 1) * spm // g) if y else 0)
            if best_split is None or longest_len < best_split[0]:
                best_split = (longest_len, g, x, y)
        assert best_split is not None, f"no split of {mtiles} M tiles over {n_ranges} SMs with <= 2 parts per tile"

        _, g, x, y = best_split
        n = x + y
        sizes = [g if (i + 1) * x // n > i * x // n else g - 1 for i in range(n)]
        ranges, tile_offset = [], 0
        for block_size in sizes:
            ranges += [
                (tile_offset * spm + r0, tile_offset * spm + r1)
                for r0, r1 in cls.split(block_size * spm, block_size + 1)
            ]
            tile_offset += block_size
        assert len(ranges) == n_ranges and tile_offset == mtiles and max(cls.parts_per_tile(ranges, spm, mtiles)) <= 2

        return ranges
