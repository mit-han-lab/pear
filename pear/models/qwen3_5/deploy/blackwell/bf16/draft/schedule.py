import struct
from enum import IntEnum

import torch

# the packed layouts of csrc/: Instr, GemvDesc, StepArgs, Tables (dk_layout.cuh), the attention's and the selector's
# descriptors
INSTR = struct.Struct("<4H2I4HIHH")  # (csrc Instr: ..., flags, xn, slot)
GEMV = struct.Struct("<21Q12I2f8I")
ATTN = struct.Struct("<8Q2IfI2Q160x")
SELECT = struct.Struct("<5QI212x")
DESC_SIZE = 256
STEP_WORDS = 20  # StepArgs as int64 words
TABLE_WORDS = 8  # Tables as int64 words
ARGS_WORDS = 10 + STEP_WORDS + TABLE_WORDS  # DkArgs: its tables' and buffers' pointers, then StepArgs and Tables
MAX_REQS = 1  # Tables.max_reqs: the proposal serves one request
NO_CTR = 0xFFFF
KIND_BF16 = 0  # csrc Kind: the GEMVs' weights and B operands are bf16
F_NOGATE = 1  # a GEMV whose weights stream without waiting for the SM's previous GEMV's release
F_XN = 2  # a kernel projection writing its normed rows (bf16) / a two-phase GEMV whose first phase copies them
TILE_M = 128
ACC_FLOATS = 64  # csrc acc_slots: a split-K finisher's slot values per row at most (its slots x n)
PIECE_SLOTS = 5  # a pieced tile's accumulator slots at most (its parts: twice as many)
QKV_SLOTS = 2  # csrc QKV_SLOTS: the qkv op's slots (its tiles' parts at most twice as many)
BAND_SLOT = 1 << 15  # csrc part_slot: the instruction's slot field holds its band and the band count (a table lookup)
CHUNK = 128  # K bytes per ring stage
CHUNK_K = CHUNK // 2  # bf16: K values per chunk
STG_ROW = 2 * CHUNK_K  # a two-phase operand's staged bf16 row of a chunk (bytes)
QB = 8  # the query block (B rows of every op but qkv)
NB = 16  # B rows of qkv (8 query + 8 committed tokens)
# a B build's input staging per (chunk, row) (csrc dk_operand.cuh in_row): copies none (cp.async straight into the B
# tiles), normed rows (fp32 x 64 values, then the norm weight's 64 bf16); the conv's coefficients per chunk (CONV_REC:
# 4 groups x 2 taps x QB rows, bf16)
IN_COPY, IN_NORM, CONV_REC = 0, 64 * 4 + 64 * 2, 4 * 2 * QB * 2
NROLE = 5  # the warp roles' progress words per SM (producer, MMA, scales, epilogue, workers)
BSLOTS = 64  # B slot barriers per SM (dk_layout.cuh BBUF_CHUNKS)
# a two-phase B operand's offset (bytes) and first B slot: the previous op's B fits below
PREP_OFF, PREP_SLOT0 = 8192, 8
CTX_STEPS = 8  # csrc dk_layout.cuh CTX_STEPS: qkv's context rows' TMEM cache holds CTX_STEP_CHUNKS x CTX_STEPS chunks
CTX_STEP_CHUNKS = 4  # csrc dk_layout.cuh CPS: the chunks of a worker warp's step
SEL_SS_OFF = 81920  # csrc dk_select.cuh SEL_SS_OFF: the selector's scratch in the B buffer, past the LM head's image
HEADS_PER_LANE = 5  # csrc dk_select.cuh HEADS_PER_LANE: the SMs' list heads a lane takes in the merge (32 lanes)


class Op(IntEnum):
    PREP = 0
    GEMV = 1
    ATTN = 2
    SELECT = 3
    IMAGE = 4  # the LM head's bf16 B operand written into its image (csrc OP_IMAGE)


class BSrc(IntEnum):
    AUX = 0
    NORM = 1
    NORM_CONV = 2
    QKV = 3
    COPY = 4
    FINAL = 5
    IMAGE = 6  # the LM head's whole bf16 operand, copied from its image (OP_IMAGE)


class Epi(IntEnum):
    BF16 = 0
    FC = 1
    QKV = 2
    RESID = 3
    SILU = 4
    LOGITS = 5
    COEF = 6


class Resid(IntEnum):
    """GemvDesc.resid: the residual an E_RESID finisher adds."""

    EMBED = 0  # the query block's embedding rows (layer 0)
    BUF_BF16 = 1  # the fp32 buffer's rows rounded to bf16
    BUF_FP32 = 2  # the fp32 buffer's rows


class SizesField(IntEnum):
    """The entries of the extension's sizes() (csrc dk_kernel.cu dk_sizes), in bytes."""

    INSTR_BYTES = 0
    GEMV_DESC_BYTES = 1
    STEP_ARGS_BYTES = 2
    TABLES_BYTES = 3
    SMEM_BYTES = 4
    BBUF_BYTES = 5  # an SM's B buffer
    RING_BYTES = 6  # the stage ring (NSTAGE x STAGE_STRIDE)
    DK_ARGS_BYTES = 7


class DraftSchedule:
    """The proposal's instructions per SM and the descriptors they point to (csrc/dk_kernel.cu)."""

    # A GEMV's stages are its (M tile of 128 weight rows, 128-byte K chunk) pairs: a small op (no more tiles than SMs)
    # gives each SM one K range of one tile, a big op splits its chunks into bands whose stages stream-K over a share of
    # the SMs. The parts of a split tile are added by fp32 atomics; the last part finishes the tile.
    MAX_BAND_CHUNKS = 52  # K chunks of an SM's B build at most (the fc's 400 chunks per row in 8 bands of 50)
    SNAP = 2  # stages: a stream-K boundary that close to a tile boundary moves onto it
    DESC_PTRS = 8  # tensor addresses in GemvDesc.b and in GemvDesc.e (those not given: null)
    IMAGE_E_PAD = 6  # null GemvDesc.e addresses past the image op's two (e[0] none, e[1] its image)
    IST_N_SHIFT = 20  # GemvDesc.ist: the chunks per batch above the staging's offset (csrc ist_n)
    TILE_SLOTS_SHIFT = 8  # GemvDesc.tile_parts: a tile's accumulator slots in use above its parts
    TILE_PARTS_BOUND = 256  # a tile's parts are fewer: they fit below TILE_SLOTS_SHIFT
    N_BANDS_SHIFT = 8  # Instr.slot of a band op: BAND_SLOT, the band count from this bit, its band (csrc part_slot)
    GROUP_VALUES = 8  # bf16 values per 16-byte group of the LM head's image (csrc dk_layout.cuh GV)

    def __init__(self, nsm: int, device: torch.device) -> None:
        self.nsm, self.device = nsm, device
        self.bbuf_bytes = torch.ops.pear_deploy_blackwell_bf16_draft.sizes()[SizesField.BBUF_BYTES]
        self.last_b = (0, 0)  # the previous GEMV's B footprint (bytes, B slots) at most
        self.first_b_end: int | None = None  # the first GEMV's B footprint's end (bytes; step.py's SS_DRAFT_OFF check)
        self.descs: list[bytes] = []
        self.sm_instrs: list[list[bytes]] = [[] for _ in range(nsm)]
        self.ctr_count: list[int] = []
        self.keep: list[torch.Tensor] = []  # the tensors the descriptors point to

    def add_gemv(
        self,
        *,
        w: torch.Tensor,
        K: int,
        n: int,
        bsrc: BSrc,
        epi: Epi,
        b: list,
        e: list,
        dep: int = NO_CTR,
        dep2: int = NO_CTR,
        layer: int = 0,
        n_out: int = QB,
        resid: int = Resid.EMBED,
        xemb: bool = False,
        conv_fma: bool = False,
        is_gated: bool = True,
        xn: bool = False,
        max_band: int | None = None,
        paired: bool = False,
    ) -> int:
        """A GEMV op over all rows of the bf16 weights w ([rows, K] as vLLM holds them); returns its completion
        counter."""
        rows, row_bytes = w.shape[0], 2 * K
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.shape[1] == K
        plan = self.plan(rows, K, n, bsrc, epi, max_band)
        mtiles, kchunks, ranges, parts, slots = (
            plan["mtiles"],
            plan["kchunks"],
            plan["ranges"],
            plan["parts"],
            plan["slots"],
        )
        slot_of, slot_table, b_off, b_slot0, stg_off, ist = (
            plan[k] for k in ("slot_of", "slot_table", "b_off", "b_slot0", "stg_off", "ist")
        )
        is_two_phase = bsrc in (BSrc.NORM_CONV, BSrc.QKV)

        # the descriptor: the weights' TMA map, the split-K accumulator and tile counts, the B and finisher tensors
        # (paired: an M tile is 64 gate rows and the 64 up rows of the same outputs, two boxes of 64 rows; E_SILU)
        assert paired == (epi == Epi.SILU) and (not paired or rows % (2 * TILE_M) == 0), "paired tiles: E_SILU's"
        tmap = torch.ops.pear_deploy_blackwell_bf16_draft.make_tmap(
            w, rows, row_bytes, TILE_M // 2 if paired else TILE_M
        ).to(self.device)
        acc = torch.zeros(mtiles * (ACC_FLOATS // n) * TILE_M * n, dtype=torch.float32, device=self.device)
        tile_ctr = torch.zeros(mtiles, dtype=torch.int32, device=self.device)
        tile_words = [
            part_count | (slot_count << self.TILE_SLOTS_SHIFT)
            for part_count, slot_count in zip(parts, slots, strict=True)
        ]
        tile_tab = torch.tensor(tile_words, dtype=torch.int16).view(torch.uint8)
        tile_parts = torch.cat([tile_tab, torch.tensor(slot_table, dtype=torch.uint8)]).to(self.device)
        self.keep += [w, tmap, acc, tile_ctr, tile_parts] + [
            tensor for tensor in (*b, *e) if isinstance(tensor, torch.Tensor)
        ]
        b_ptrs = [self.ptr(tensor) for tensor in b] + [0] * (self.DESC_PTRS - len(b))
        e_ptrs = [self.ptr(tensor) for tensor in e] + [0] * (self.DESC_PTRS - len(e))
        self.descs.append(
            GEMV.pack(
                tmap.data_ptr(), 0, acc.data_ptr(), tile_ctr.data_ptr(), tile_parts.data_ptr(), *b_ptrs, *e_ptrs,
                K, rows, mtiles, kchunks, KIND_BF16, n, int(paired), bsrc, epi, layer, n_out, resid,
                1.0, 1.0, int(xemb), int(conv_fma), b_off, b_slot0, 0, stg_off, 0, ist,
            )
        )  # fmt: skip
        desc = len(self.descs) - 1

        # is_gated: the weights wait for the release of the SM's previous GEMV (a release waits for its loads in flight)
        aux = 0 if dep2 == NO_CTR else dep2 + 1
        flags = 0 if is_gated else F_NOGATE

        # the normed rows from a kernel projection to the next two-phase GEMV: a B_NORM op given b[3] (bf16 [QB][K])
        # writes them from its first tile's SMs, a two-phase op given xn copies them from b[0] in its conv phase
        writers = [
            sm
            for sm, (_, kbw, r0, r1) in enumerate(ranges)
            if bsrc == BSrc.NORM and len(b) > 3 and r0 < r1 and r0 // kbw == 0
        ]  # (the first tile's SMs: every chunk once)
        writer_chunks = sorted({chunk for sm in writers for chunk in range(ranges[sm][2], ranges[sm][3])})
        assert not writers or writer_chunks == list(range(kchunks)), "the normed rows' writers: every chunk once"

        ctr = self.new_counter(self.nsm + len(writers))  # every SM's epilogue, and the normed rows' writers
        for sm, (kb0, kbw, r0, r1) in enumerate(ranges):
            if sm in writers:
                instr_flags, xn_ctr = flags | F_XN, ctr + 1  # (its workers count on ctr once its rows are written)
            elif is_two_phase and xn:
                instr_flags, xn_ctr = flags | F_XN, 0
            else:
                instr_flags, xn_ctr = flags, 0
            instr = INSTR.pack(Op.GEMV, desc, dep, ctr, r0, r1, kb0, kbw, sm, aux, instr_flags, xn_ctr, slot_of[sm])
            self.sm_instrs[sm].append(instr)

        return ctr

    def plan(self, rows: int, K: int, n: int, bsrc: BSrc, epi: Epi, max_band: int | None = None) -> dict:
        """add_gemv's split of a [rows, K] bf16 GEMV over the SMs: the SMs' (kb0, kbw, r0, r1) ranges, the tiles' parts
        and accumulator slots, the B operand's footprint."""
        row_bytes = 2 * K
        assert row_bytes % CHUNK == 0
        assert rows % TILE_M == 0, "whole M tiles"
        mtiles, kchunks = rows // TILE_M, row_bytes // CHUNK

        # a two-phase B operand (built mostly during the previous op) keeps clear of that op's tiles and B slots; per
        # chunk position its tile [n rows][128 B] and (two-phase) its staged rows; B_IMAGE's image is the whole operand
        # (one barrier for it: not bounded by the B slots)
        is_two_phase, is_image = bsrc in (BSrc.NORM_CONV, BSrc.QKV), bsrc == BSrc.IMAGE
        b_off, b_slot0 = (PREP_OFF, PREP_SLOT0) if is_two_phase else (0, 0)
        per_chunk = n * CHUNK + (QB * STG_ROW if is_two_phase else 0)
        band_chunks = kchunks if is_image else (max_band or self.MAX_BAND_CHUNKS)  # (max_band: the op's own bound)
        bslots = kchunks if is_image else BSLOTS - b_slot0
        max_chunks = min((self.bbuf_bytes - b_off) // per_chunk, bslots, band_chunks)
        max_parts = 2 * (QKV_SLOTS if epi == Epi.QKV else min(PIECE_SLOTS, ACC_FLOATS // n))
        piece_ranges = self.pieces(mtiles, kchunks, max_chunks, max_parts)
        is_pieced = piece_ranges is not None
        ranges = piece_ranges if is_pieced else self.bands(mtiles, kchunks, max_chunks)
        b_chunks = kchunks if is_image else max(min(kbw, r1 - r0) for _, kbw, r0, r1 in ranges)
        stg_off = b_off + b_chunks * n * CHUNK
        end = stg_off + (b_chunks * QB * STG_ROW if is_two_phase else 0)
        assert end <= self.bbuf_bytes, "the B operand fits the B buffer"
        assert bsrc != BSrc.QKV or b_chunks <= CTX_STEP_CHUNKS * CTX_STEPS, "qkv's context rows' TMEM cache"

        # the build's input staging after it: batches of chunks (two-phase: every chunk's coefficients at once), its
        # offset in bits 0..19 of GemvDesc.ist and its chunks per batch above
        if is_image:
            per_in = 0
        elif is_two_phase:
            per_in = CONV_REC
        else:
            per_in = QB * (IN_COPY if bsrc in (BSrc.AUX, BSrc.COPY) else IN_NORM)
        ist_n = min(b_chunks, (self.bbuf_bytes - end) // per_in) if per_in else 0
        assert not per_in or (ist_n == b_chunks if is_two_phase else ist_n >= 1), "the B build's input staging"
        ist = end | (ist_n << self.IST_N_SHIFT)
        if is_two_phase:
            assert self.last_b[0] <= PREP_OFF and self.last_b[1] <= PREP_SLOT0, "the previous op's B overlaps"
        self.last_b = (end, b_slot0 + b_chunks)
        if self.first_b_end is None:
            self.first_b_end = end

        # each part's accumulator slot (csrc reduce_part: two parts at most add into a slot, so that its fp32 sum does
        # not depend on their order; the last part sums the slots in order): pieces put a tile's two consecutive parts
        # in a slot, bands a tile's parts in band order two per slot (csrc: slot_table [tile][band][starts mid-tile])
        band_index = {kb0: i for i, kb0 in enumerate(sorted({sm_range[0] for sm_range in ranges}))}
        n_bands = len(band_index)
        band_parts = {}
        for kb0, kbw, r0, r1 in ranges:
            for mt in range(r0 // kbw, (r1 - 1) // kbw + 1) if r0 < r1 else ():
                band_parts[mt, band_index[kb0]] = band_parts.get((mt, band_index[kb0]), 0) + 1

        parts, slots, slot_of, per_slot = [0] * mtiles, [0] * mtiles, [], {}
        slot_table = [0] * (mtiles * n_bands * 2)
        if not is_pieced:
            for mt in range(mtiles):
                k = 0
                for band in range(n_bands):
                    for j in range(band_parts.get((mt, band), 0)):
                        slot_table[(mt * n_bands + band) * 2 + j] = k // 2
                        k += 1
                slots[mt] = -(-k // 2)

        for kb0, kbw, r0, r1 in ranges:
            slot = parts[r0 // kbw] // 2 if is_pieced else BAND_SLOT | (n_bands << self.N_BANDS_SHIFT) | band_index[kb0]
            slot_of.append(slot)
            for mt in range(r0 // kbw, (r1 - 1) // kbw + 1) if r0 < r1 else ():
                if is_pieced:
                    part_slot = slot
                else:
                    part_slot = slot_table[(mt * n_bands + band_index[kb0]) * 2 + (max(r0, mt * kbw) > mt * kbw)]
                parts[mt] += 1
                if is_pieced:
                    slots[mt] = max(slots[mt], slot + 1)
                per_slot[mt, part_slot] = per_slot.get((mt, part_slot), 0) + 1
        assert max(parts) < self.TILE_PARTS_BOUND and max(per_slot.values()) <= 2 and max(slots) <= ACC_FLOATS // n, (
            "split-K slots"
        )
        assert epi != Epi.QKV or max(slots) <= QKV_SLOTS, "the qkv sums' slots (csrc qkv_part)"

        return dict(
            mtiles=mtiles,
            kchunks=kchunks,
            ranges=ranges,
            parts=parts,
            slots=slots,
            slot_of=slot_of,
            slot_table=slot_table,
            b_off=b_off,
            b_slot0=b_slot0,
            stg_off=stg_off,
            ist=ist,
        )

    def add_prep(self) -> None:
        """OP_PREP on every SM: the step's ids and layer 0's input norm (SM 0 also zeroes the next launch's sums)."""
        for sm in range(self.nsm):
            self.sm_instrs[sm].append(INSTR.pack(Op.PREP, 0, NO_CTR, NO_CTR, 0, 0, 0, 0, sm, 0, 0, 0, 0))

    def add_attention(
        self,
        *,
        q: torch.Tensor,
        kvq: torch.Tensor,
        sums: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        cache: torch.Tensor,
        part_o: torch.Tensor,
        part_ml: torch.Tensor,
        out: torch.Tensor,
        head_ctr: torch.Tensor,
        n_splits: int,
        layer: int,
        scale_log2: float,
        dep: int,
    ) -> int:
        """OP_ATTN: KV heads x n_splits items, one per SM, on the qkv op's sums [2][QKV_SLOTS][tiles][NB][128] (E_QKV);
        returns the counter of the items' merges."""
        n_items = part_o.shape[1] * n_splits
        self.keep += [q, kvq, sums, q_norm, k_norm, cache, part_o, part_ml, out, head_ctr]
        done_ctr = self.new_counter(n_items)  # (a head's items wait for each other: they merge its rows)
        self.descs.append(
            ATTN.pack(
                q.data_ptr(), kvq.data_ptr(), sums.data_ptr(), cache.data_ptr(), part_o.data_ptr(), part_ml.data_ptr(),
                out.data_ptr(), head_ctr.data_ptr(), n_splits, layer, scale_log2, sums.shape[2],
                q_norm.data_ptr(), k_norm.data_ptr(),
            )
        )  # fmt: skip

        desc = len(self.descs) - 1
        for i in range(n_items):
            self.sm_instrs[i].append(INSTR.pack(Op.ATTN, desc, dep, done_ctr, i, i + 1, 0, 0, i, 0, 0, 0, 0))

        return done_ctr

    def add_image(self, *, K: int, b: list, image: torch.Tensor, dep: int) -> int:
        """OP_IMAGE on every SM: a share of the (row, 16-byte group) pairs of the LM head's bf16 B operand (the final
        norm's rows) into its image (the B buffer's layout [chunk][QB rows][128 B] SW128, global memory); returns the
        counter of the shares."""
        assert image.dtype == torch.uint8 and image.numel() == self.image_bytes(K), "the image's bytes"

        self.keep += [image] + [tensor for tensor in b if isinstance(tensor, torch.Tensor)]
        b_ptrs = [self.ptr(tensor) for tensor in b] + [0] * (self.DESC_PTRS - len(b))
        e_ptrs = [0, image.data_ptr()] + [0] * self.IMAGE_E_PAD
        self.descs.append(
            GEMV.pack(
                0, 0, 0, 0, 0, *b_ptrs, *e_ptrs,
                K, 0, 0, 0, KIND_BF16, QB, 0, BSrc.IMAGE, Epi.BF16, 0, QB, 0,
                1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        )  # fmt: skip
        desc = len(self.descs) - 1

        groups = QB * (K // self.GROUP_VALUES)
        done_ctr = self.new_counter(self.nsm)
        for sm in range(self.nsm):
            r0, r1 = groups * sm // self.nsm, groups * (sm + 1) // self.nsm
            self.sm_instrs[sm].append(INSTR.pack(Op.IMAGE, desc, dep, done_ctr, r0, r1, 0, 0, sm, 0, 0, 0, 0))

        return done_ctr

    def add_select(
        self,
        *,
        part: torch.Tensor,
        cand: torch.Tensor,
        unary: torch.Tensor,
        hsel: torch.Tensor,
        scores: torch.Tensor,
        n_part: int,
        dep: int,
        dep2: int,
    ) -> None:
        """OP_SELECT on SM 0: the candidates' merge, the selector's scores, the walk and the proposal's outputs."""
        # the SMs' lists from the ring's start, reaching into the B buffer below the selector's scratch
        ring_bytes = torch.ops.pear_deploy_blackwell_bf16_draft.sizes()[SizesField.RING_BYTES]
        assert part.numel() * part.element_size() <= ring_bytes + SEL_SS_OFF, "the lists' smem"
        assert n_part <= 32 * HEADS_PER_LANE, "the merge's list heads per lane"

        self.keep += [part, cand, unary, hsel, scores]
        self.descs.append(
            SELECT.pack(part.data_ptr(), cand.data_ptr(), unary.data_ptr(), hsel.data_ptr(), scores.data_ptr(), n_part)
        )
        desc = len(self.descs) - 1
        self.sm_instrs[0].append(INSTR.pack(Op.SELECT, desc, dep, NO_CTR, 0, 0, 0, 0, 0, dep2 + 1, 0, 0, 0))

    def finalize(self) -> dict:
        """The schedule's device tensors."""
        sizes = torch.ops.pear_deploy_blackwell_bf16_draft.sizes()
        assert sizes[:2] == [INSTR.size, DESC_SIZE] and sizes[SizesField.DK_ARGS_BYTES] == 8 * ARGS_WORDS, sizes
        assert (
            sizes[SizesField.STEP_ARGS_BYTES] == 8 * STEP_WORDS and sizes[SizesField.TABLES_BYTES] == 8 * TABLE_WORDS
        ), sizes

        instrs, offsets = [], [0]
        for sm_list in self.sm_instrs:
            instrs.extend(sm_list)
            offsets.append(len(instrs))

        return {
            "instrs": torch.frombuffer(bytearray(b"".join(instrs)), dtype=torch.uint8).to(self.device),
            "sm_off": torch.tensor(offsets, dtype=torch.int32).to(self.device),
            "descs": torch.frombuffer(bytearray(b"".join(self.descs)), dtype=torch.uint8).to(self.device),
            "counters": torch.zeros(len(self.ctr_count), dtype=torch.int32, device=self.device),
            "ctr_count": torch.tensor(self.ctr_count, dtype=torch.int32).to(self.device),
            "epoch": torch.zeros(2, dtype=torch.int32, device=self.device),
            "progress": torch.zeros(self.nsm, NROLE, dtype=torch.int32, device=self.device),
        }

    def relocated(self, op_base: int) -> list[list[bytes]]:
        """The per-SM instruction lists with their op codes from op_base (the draft's range in a host step's lists)."""
        lists = []
        for sm_list in self.sm_instrs:
            fields = [INSTR.unpack(instr) for instr in sm_list]
            lists.append([INSTR.pack(instr_fields[0] + op_base, *instr_fields[1:]) for instr_fields in fields])
        return lists

    def pieces(
        self, mtiles: int, kchunks: int, max_chunks: int, max_parts: int
    ) -> list[tuple[int, int, int, int]] | None:
        """Per SM (0, kchunks, r0, r1): one K range of one tile, a tile's chunks over at most max_parts SMs (the SMs
        left: empty ranges); None when the tiles outnumber the SMs or a range's B tiles do not fit."""
        tile_sms, extra_tiles = divmod(self.nsm, mtiles)
        if tile_sms >= max_parts:
            tile_sms, extra_tiles = max_parts, 0
        if tile_sms == 0 or -(-kchunks // tile_sms) > max_chunks:
            return None

        ranges = []
        for mt in range(mtiles):
            k = tile_sms + (mt < extra_tiles)
            ranges += [
                (0, kchunks, mt * kchunks + kchunks * i // k, mt * kchunks + kchunks * (i + 1) // k) for i in range(k)
            ]
        return ranges + [(0, kchunks, 0, 0)] * (self.nsm - len(ranges))

    def bands(self, mtiles: int, kchunks: int, max_chunks: int) -> list[tuple[int, int, int, int]]:
        """Per SM (kb0, kbw, r0, r1): its band's chunks [kb0, kb0 + kbw) and its range of the band's (M tile, chunk)
        stages, in the fewest bands (a divisor of the SM count) of at most max_chunks chunks."""
        n_bands = next(
            divisor
            for divisor in range(1, self.nsm + 1)
            if self.nsm % divisor == 0 and -(-kchunks // divisor) <= max_chunks
        )
        edges = [kchunks * band // n_bands for band in range(n_bands + 1)]

        # each band's SMs in proportion to its chunks, by largest remainders, so that the SMs of wide and narrow bands
        # stream about as many stages each
        ideal_sms = [self.nsm * (edges[band + 1] - edges[band]) / kchunks for band in range(n_bands)]
        sms = [int(x) for x in ideal_sms]
        for band in sorted(range(n_bands), key=lambda i: ideal_sms[i] - sms[i], reverse=True)[: self.nsm - sum(sms)]:
            sms[band] += 1

        ranges = []
        for band in range(n_bands):
            kb0, kbw, band_sms = edges[band], edges[band + 1] - edges[band], sms[band]
            total = mtiles * kbw
            # a boundary within SNAP stages of a tile's moves onto it: a segment that short costs a whole reduction
            cuts = [self.snap(total * sm // band_sms, kbw) for sm in range(band_sms)] + [total]
            ranges += [(kb0, kbw, cuts[sm], cuts[sm + 1]) for sm in range(band_sms)]
        assert len(ranges) == self.nsm

        return ranges

    @classmethod
    def snap(cls, stage: int, kbw: int) -> int:
        """A stream-K boundary moved onto the nearest tile boundary when within SNAP stages of it."""
        tile_start = stage - stage % kbw
        if stage - tile_start <= cls.SNAP:
            return tile_start
        return tile_start + kbw if tile_start + kbw - stage <= cls.SNAP else stage

    def new_counter(self, count: int) -> int:
        """A counter that `count` arrivals complete per launch."""
        self.ctr_count.append(count)
        return len(self.ctr_count) - 1

    @staticmethod
    def image_bytes(K: int) -> int:
        """The LM head's bf16 B operand image (csrc image_bytes): QB rows of K values, in the B buffer's layout."""
        return QB * K * 2

    @staticmethod
    def ptr(tensor_or_address: torch.Tensor | int | None) -> int:
        """A tensor's device address, or the address given (0 for none)."""
        return (
            tensor_or_address.data_ptr() if isinstance(tensor_or_address, torch.Tensor) else int(tensor_or_address or 0)
        )


class ArgBlocks:
    """The argument blocks of csrc/dk_layout.cuh (Tables, StepArgs, DkArgs) as CPU int64 tensors."""

    @staticmethod
    def tables(
        *,
        embed: torch.Tensor,
        cos_sin: torch.Tensor,
        pred_cb: torch.Tensor,
        succ_cb: torch.Tensor,
        vocab: int,
        mask_token: int,
        max_model_len: int,
        kv_block: int,
        window_left: int,
        topk_chunk: int,
        nstep: int,
    ) -> torch.Tensor:
        """Tables: the model's constant tensors the kernel addresses directly, and its sizes."""
        ptr = DraftSchedule.ptr
        words = [
            ptr(embed), ptr(cos_sin), ptr(pred_cb), ptr(succ_cb), vocab | (mask_token << 32),
            max_model_len | (kv_block << 32), window_left | (topk_chunk << 32), MAX_REQS | (nstep << 32),
        ]  # fmt: skip
        return torch.tensor(words, dtype=torch.int64)

    @staticmethod
    def step(
        *,
        aux: list[torch.Tensor],
        n_ctx: int,
        positions: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill: torch.Tensor,
        seeds: torch.Tensor,
        temperature: torch.Tensor,
        block_table: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        cached_cand: torch.Tensor,
        sel_scores: torch.Tensor,
        bonus_row: bool = False,
    ) -> torch.Tensor:
        """StepArgs: the step's vLLM device tensors and the proposal's outputs (aux: the target's aux hidden states)."""
        ptr = DraftSchedule.ptr
        tensors = (positions, num_rejected, num_sampled, idx_mapping, last_sampled, next_prefill, seeds, temperature,
                   block_table, draft_tokens, draft_logits, cached_cand, sel_scores)  # fmt: skip
        words = [ptr(tensor) for tensor in aux] + [(aux[0].stride(0) & 0xFFFFFFFF) | (n_ctx << 32)]
        words += [ptr(tensor) for tensor in tensors]
        words.append(int(bonus_row))
        return ArgBlocks.signed(words)

    @staticmethod
    def args(*, schedule: dict, ssq: torch.Tensor, step: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """DkArgs: the schedule's lists, descriptors, counters, epoch and progress words, the sums of squares, then the
        StepArgs and Tables blocks."""
        ptr = DraftSchedule.ptr
        words = [ptr(schedule[k]) for k in ("instrs", "sm_off", "descs", "counters", "ctr_count", "epoch")]
        # n_ssq (and its padding), then DkArgs::tstamp, the kernel's stamp pointer: none
        words += [ptr(ssq), ssq.numel() // (2 * QB), ptr(schedule["progress"]), 0]
        return torch.cat([ArgBlocks.signed(words), step, tab])

    @staticmethod
    def signed(words: list[int]) -> torch.Tensor:
        """Unsigned 64-bit words as an int64 tensor."""
        return torch.tensor([word - (1 << 64) if word >= 1 << 63 else word for word in words], dtype=torch.int64)
