// The decode megakernel on sm_120a: its constants, op codes, instruction and descriptor layouts, the stage geometry.
// One persistent CTA per SM walks its instruction list (schedule.py); the MMAs are mma.sync with register accumulators.
#pragma once
#include "common.cuh"
#include "step.cuh"

namespace mk {

constexpr int NW = 16;
constexpr int NTHREADS = NW * 32;
// warp roles, each walking the CTA's instruction list on its own (synchronized by mbarriers): epilogue 0..3, producer
// 4 (TMA of the weight stages), MMA 5..6, prologue 8..15 (the B operand, and every op that is not a GEMV)
constexpr int W_EPI0 = 0, W_PROD = 4, W_MMA0 = 5, NMMA = 2, W_PRO0 = 8;
constexpr int NEPI = 128, NPRO = (NW - W_PRO0) * 32;
constexpr int NT = 8;           // B rows (the MMAs' N): the step's tokens at most
constexpr int TILE = 128;       // weight rows per M tile
constexpr int STAGE_K = 128;    // weight bytes per row per stage
// two ring stages: the GDN core, the attention, the spec sampler and the draft's ops need all of STG_BYTES
constexpr int NSTAGE = 2;
constexpr int A_BYTES = TILE * STAGE_K;
constexpr int SF_ATOM_BYTES = 512;                // an NVFP4 scale atom: 128 rows x 4 blocks (vLLM's [32][4][4])
constexpr int SFA_BYTES = 4 * SF_ATOM_BYTES;      // the stage's 4 NVFP4 scale atoms
constexpr int STAGE_BYTES = A_BYTES + SFA_BYTES;  // a multiple of 1024: the SW128 tiles stay aligned
constexpr int CHUNK_B = NT * STAGE_K;             // one K chunk of B: 8 token rows x 128 bytes
constexpr int SFB_CHUNK = NT * 16;                // its NVFP4 scales: 8 rows x 16 blocks
constexpr int B_CHUNKS = 20;                      // an instruction's K chunks at most (schedule.py)
constexpr int B_SLOT_BYTES = B_CHUNKS * (CHUNK_B + SFB_CHUNK);
// one B slot: in a chain of dependent ops the next instruction's B cannot be built before its dependency anyway
constexpr int NB = 1;
// the B slot's chunks in BG groups of B_GCH slots (in the MMA's order), each completing a barrier of its own
// (xfull[slot][g]), so the MMA starts on the first group's chunks while the rest land
constexpr int BG = 4;
constexpr int B_GCH = (B_CHUNKS + BG - 1) / BG;  // B slots per group (slot l in group l / B_GCH; past nck: empty)
static_assert(BG >= 1 && BG <= B_CHUNKS && B_GCH * BG >= B_CHUNKS, "B groups");
constexpr int ACC_COLS = NT;                      // a segment slot row: 8 sums
constexpr int ACC_SLOT_BYTES = TILE * ACC_COLS * 4;
constexpr int NACC = 1;
constexpr int STG_BYTES = 32768;                  // the prologue's staging buffer (the GDN core's state rows, ...)
constexpr int RED_FLOATS = (NPRO / 32) * NT * 2;  // prologue reductions
constexpr int BAR_BYTES = ((NSTAGE > 2 ? 160 : 144) + 8 * (BG - 1) + 15) / 16 * 16;  // the mbarriers
constexpr int SMEM_BYTES =
    NSTAGE * STAGE_BYTES + NB * B_SLOT_BYTES + NACC * ACC_SLOT_BYTES + STG_BYTES + RED_FLOATS * 4 + BAR_BYTES;
constexpr int SMEM_ALIGN = 1024;  // the dynamic shared memory's base, aligned up to this (the SW128 tiles)
constexpr int SMEM_OPTIN_BYTES = 101376;  // a CTA's shared memory at most (the 99 KB opt-in)
constexpr int STATIC_SMEM_EXTRA = 64;     // the static shared words besides rec_sh, at most
static_assert(SMEM_BYTES + SMEM_ALIGN + 4 * SR_BT + STATIC_SMEM_EXTRA <= SMEM_OPTIN_BYTES, "shared memory");
// (2 of the mbarriers and 1 of the words are reserved: the layout the kernel was validated with)
static_assert((2 * NSTAGE + NB * BG + NB + 2 * NACC + 2 + 2) * 8 + (2 * NSTAGE + 2 * NACC + 6) * 4 <= BAR_BYTES,
              "the mbarriers and the pool's words");
constexpr int BAR_PRO = 1, BAR_EPI = 2, BAR_GDNT = 3;  // named barriers (BAR_GDNT: prologue warps 6-7)
// registers per thread by warpgroup (setmaxnreg at the start; launched at 128): epilogue, producer + MMA, prologue
constexpr int REG_EPI = 104, REG_MMA = 104, REG_PRO = 152;
// the kernel variants' code (mk_main<V>): the target's ops (V 0, 2), the draft's (V 1, 2)
template <int V> constexpr bool HAS_T = V != 1;
template <int V> constexpr bool HAS_D = V != 0;
constexpr int WG_THREADS = 128;  // threads per warpgroup
constexpr int SM_REGS = 65536;   // the register file's 32-bit registers
static_assert(WG_THREADS * (REG_EPI + REG_MMA + 2 * REG_PRO) <= SM_REGS, "the register file");
static_assert(W_EPI0 == 0 && W_PROD == 4 && W_MMA0 == 5 && W_PRO0 == 8, "warpgroups");
constexpr uint32_t K_NORM = 5120;
// ACC_CNT: a 64-bit word per output = (sum of fx) * 2^CNT_BITS + parts added, fx = round(alpha * part * 2^28)
constexpr int CNT_BITS = 8;
constexpr float CNT_SCALE = 268435456.0f, CNT_INV = 1.0f / 268435456.0f;
// the published sums of squares: round(x^2 * 2^SSQ_BITS) added exactly in u64
constexpr int SSQ_BITS = 24;
constexpr float SSQ_SCALE = 16777216.0f, SSQ_INV = 1.0f / 16777216.0f;

enum Op : uint16_t {
  OP_GEMV = 0, OP_ATTN = 3, OP_ATTN_COMBINE = 4, OP_EMBED = 5, OP_ATTN_PREP = 6, OP_GDNT = 10,
  OP_SS_A = 12, OP_SS_B = 13, OP_SS_C = 14,  // the spec sampler's phases (spec_sample.cuh SsDesc), after the LM head
  OP_NOP = 15,
  // the DFlash 2 draft's proposal (draft.cuh): the committed tokens' context K / V into the draft's cache, a layer's
  // query block q / k / v (norm, rope, the cache), its attention items and their merge, the candidates, the selector
  OP_D_CTX = 16, OP_D_PREP = 17, OP_D_ATTN = 18, OP_D_COMBINE = 19, OP_D_TOPK = 20, OP_D_SELECT = 21,
  // the step's inputs into the V2 runner's buffers (step.cuh book_inputs: SM 0, first); the previous launched step's
  // outputs into the host ring and the rest of its post_update (spec_sample.cuh prev_outputs: the last SM, first)
  OP_BOOK = 22, OP_PREV = 23,
  // a PRO_NORM_Q op's slice published ahead of its GEMV (Instr idx: the slice), whose instructions then carry
  // AUX_QPUB_DONE and only wait for every slice: the fused draft's fc, during the target's LM head
  OP_QPUB = 24
};
constexpr uint32_t AUX_QPUB_DONE = 1u << 28;  // (Instr::aux of a GEMV: its NORM_Q slice published by an OP_QPUB)
constexpr uint32_t SS_AUX_A0 = 1u, SS_AUX_NO_A0 = 2u;  // (Instr::aux of OP_SS_A: A0 alone; A after such an instruction)
enum Kind : uint32_t { KIND_NVFP4 = 0, KIND_FP8 = 1, KIND_BF16 = 2 };
// PRO_NORM_X: the RMSNorm of x from its producer's published sums of squares (ssq_in), reading only the instruction's
// chunks of x. PRO_NORM_Q: each instruction (one per SM) encodes a slice of the normed x into q_pub (PRO_ACT's layout)
// and counts it (qctr); once every slice is counted, its chunks go into the B slot as PRO_ACT's
enum Pro : uint32_t { PRO_NORM_IN = 0, PRO_NORM_RESID = 1, PRO_ACT = 3, PRO_NORM_X = 4, PRO_NORM_Q = 5 };
enum Epi : uint32_t { EPI_RED = 0, EPI_LOGITS = 1, EPI_SILU = 2 };
// the draft's variants (OpDesc::dflags)
enum DFlag : uint32_t {
  DF_NW_PLAIN = 1,   // the norm's weight applied as w (Qwen3's RMSNorm), not 1 + w
  DF_PLAIN = 2,      // PRO_NORM_Q without the norm: in0's bf16 values; K = segments of 5120, segment s of token t at
                     // in0 + (s NT + t) 5120 (the fc's aux hidden states, aux_all [layer][NT][5120])
  DF_CONV = 4,       // PRO_NORM_Q: the grouped conv (side 0) of the normed rows: y_t = bf16(fma(c0, n_t, c1 n_(t-1))),
                     // c_tap = base[0][tap] + coef[t][tap * 320 + g] (Inductor's)
  DF_COLFIN = 8,     // EPI_RED: a column's 8 values finished together by the last part of its 32 columns (DF_CONV_FIN)
  DF_CONV_FIN = 16,  // EPI_RED + xr + DF_COLFIN: x = conv side 1 of o = bf16(sums) + the residual (fp32 x_out, its ssq)
  DF_CONV_FMA = 32,  // DF_CONV_FIN: fma(c0, o, tail) (the MLP's: Inductor fused it), else c0 o + tail (attention's)
  DF_RES_ROUND = 64, // xr: the residual (fp32) rounded to bf16 before the add
  DF_DRAFT_TOK = 128,// OP_EMBED: the draft's query block (the anchor from the sampler's ring slot, then mask tokens)
  DF_POOL = 256,     // GEMV: each band's last pool_len stages claimed at run time in units of pool_u (never set)
  DF_PSUM = 512      // EPI_RED: a split value's parts' tagged sums added by the tile's last part (never set)
};
constexpr uint16_t NO_CTR = 0xffff;

struct alignas(16) Instr {  // 32 bytes (schedule.py INSTR)
  uint16_t op, desc;
  uint16_t dep, done;  // the counters this instruction waits on / counts (NO_CTR: none)
  uint32_t r0, r1;     // GEMV: its stage range; other ops: their item range
  uint32_t idx;        // this instruction's index among the op's n_inst
  uint32_t aux;
  uint32_t pr0, pr1;
};
static_assert(sizeof(Instr) == 32, "Instr layout");

struct OpDesc {        // 256 bytes (schedule.py DESC)
  uint64_t tmap;       // CUtensorMap over the weights [rows][row bytes] (uint8, box 128 B x 128 rows, SWIZZLE_128B)
  uint64_t sf;         // NVFP4 weight scales, vLLM's swizzled layout [rows / 128][K / 64][32][4][4]
  uint64_t in0;        // PRO_NORM_IN: the bf16 input [NT][K]; PRO_NORM_RESID: the producer's bf16 output [NT][K];
                       // PRO_ACT: the published operand (codes [K chunks][NT][128], NVFP4 scales [K chunks][NT][16]
                       // after them); OP_EMBED: the table bf16 [V][K]
  uint64_t res_in;     // PRO_NORM_RESID: the residual [NT][K] (fp32, or bf16: res_bf16)
  uint64_t nw;         // the norm's weight (bf16 [K], applied as 1 + w)
  uint64_t x_out;      // PRO_NORM_RESID: fp32 x = in0 + residual [NT][K] (each instruction a slice); 0: none
  uint64_t r_out;      // PRO_NORM_RESID: bf16(x), the next residual [NT][K]; OP_EMBED: the output rows
  uint64_t acc;        // ACC_CNT words [2 copies][NT][rows]
  uint64_t out;        // EPI_RED: bf16 output [NT][n_out]; EPI_LOGITS: bf16 logits [NT][n_out]; EPI_SILU: the published
                       // operand of the consumer (codes [n_out / 256][NT][128], then scales [n_out / 256][NT][16])
  uint64_t parts;      // ACC_CNT: parts per M tile (u8 [mtiles])
  uint64_t spart;      // EPI_SILU: a split pair's later part's sums [npairs][TILE][ACC_COLS] fp32; EPI_LOGITS: a split
                       // tile's later part's tagged sums [n_inst][TILE][NT] u64 (tag pool_len + launch parity)
  uint64_t sflag;      // EPI_SILU: per pair, the later part's release (u32, epoch-based)
  uint64_t rec;        // the step record (int32 words)
  uint64_t ssq;        // EPI_RED with xr: this op's x sums of squares [2 parities][mtiles][NT] u64 (x^2 * 2^SSQ_BITS)
  uint64_t ssq_in;     // PRO_NORM_X / Q: the producer's ssq (its copy epoch & 1)
  uint64_t q_pub;      // PRO_NORM_Q: the published operand (codes [K chunks][NT][128], NVFP4 scales after)
  float alpha;         // output = alpha * sum
  float gs;            // B: NVFP4 input global scale (multiplier) / FP8 1 / input scale (multiplier)
  float eps;           // RMSNorm epsilon
  float act_gs;        // EPI_SILU: the consumer's NVFP4 input global scale (multiplier)
  uint32_t kind, K, kb, rows;  // weight kind, K values, row bytes, weight rows
  uint32_t n_out, mtiles, spm, pro;  // outputs (paired: gate rows), M tiles (paired: pairs), stages per tile, Pro
  uint32_t epi, n_inst, paired, res_bf16;
  uint32_t slice;      // PRO_NORM_RESID: x_out / r_out values per instruction (a slice of each row)
  uint32_t bw;         // K chunks per band: the stages are band-major, (band, M tile) units of bw chunks (bw = spm: one
                       // band); an instruction stays within one band, so its B holds at most bw chunks
  uint32_t xr;         // EPI_RED: the finisher also forms x = output + res_in (fp32 x_out, bf16 r_out) and its ssq
  uint32_t ssq_tiles;  // PRO_NORM_X: the producer's M tiles (ssq slots per token)
  uint32_t x_bf16;     // PRO_NORM_X / Q: x is bf16 (the embedding rows)
  uint32_t qctr;       // PRO_NORM_Q: the counter of the published slices (target n_inst per launch)
  // the draft's GEMVs (DF_* flags)
  uint64_t coef;       // DF_CONV / DF_CONV_FIN: the grouped conv's coefficients, the kernel projection's bf16 output
                       // [NT][1280] (column side * 640 + tap * 320 + group)
  uint64_t base;       // DF_CONV / DF_CONV_FIN: the conv's base kernel bf16 [2 sides][2 taps][5120]
  uint64_t colctr;     // DF_COLFIN: per (M tile, epilogue warp) the parts that completed their 32 columns (u32,
                       // epoch-scaled)
  uint32_t dflags;     // DF_*
  uint32_t pool_u;     // DF_POOL (never set: 0): stages per pool unit
  uint32_t pool_len;   // DF_POOL: the pool's stages at each band's end (0; also EPI_LOGITS's tag base)
  uint32_t pool_n;     // DF_POOL (never set: 0): units per band's pool
};
static_assert(sizeof(OpDesc) == 256, "OpDesc layout");

// counters live in global memory and count an op's instructions once (epilogue) or twice (prologue too); their
// targets scale with the launch epoch, which the last CTA of each launch advances, so nothing is reset
struct MKArgs {
  const Instr* instrs;
  const uint32_t* sm_off;  // [nsm + 1]
  const OpDesc* descs;
  uint32_t* counters;
  const uint32_t* ctr_count;
  uint32_t* epoch;         // [0] epoch, [1] CTAs done (all launches)
  uint64_t* trace;         // per instruction TR_N globaltimer stamps (null: none)
  RecParam rec;            // the step record's header (step.cuh), copied into each CTA's rec_sh
};
// trace stamps per instruction (globaltimer ns): the prologue's dependency met and its B ready, the MMA warps' first
// stage landed and last stage done, the epilogue's end (an op other than a GEMV: the prologue's end), the NORM_Q slice
// counted and all counted, the prologue's start, the epilogue's last segment read and its sums added
enum TraceSlot : int {
  TR_DEP = 0, TR_B = 1, TR_MMA0 = 2, TR_MMA1 = 3, TR_EPI = 4, TR_Q = 5, TR_START = 7, TR_ELAST = 8, TR_N = 10
};
__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---- stage geometry ----
// a stage: the row base of its 128 weight rows, its K chunk, its half (paired: gate 0 / up 1), its unit (segments end
// at unit boundaries) and its output tile. Paired (gate_up: vLLM's [gate; up] rows): pair p's stage q is chunk q / 2
// of the gate (q even) or up (q odd) tile p. Otherwise band-major units (band, tile) of bw chunks each.
struct StageAt {
  uint32_t row0, kc, half, unit, tile;
};
__device__ __forceinline__ StageAt stage_at(const OpDesc& d, uint32_t st) {
  StageAt a;
  if (d.paired) {
    const uint32_t pair_stages = 2 * d.spm, p = st / pair_stages, q = st - p * pair_stages;
    a.unit = a.tile = p, a.kc = q >> 1, a.half = q & 1, a.row0 = (q & 1) * d.n_out + p * TILE;
  } else {
    a.unit = st / d.bw;
    const uint32_t band = a.unit / d.mtiles;
    a.tile = a.unit - band * d.mtiles, a.kc = band * d.bw + (st - a.unit * d.bw), a.half = 0, a.row0 = a.tile * TILE;
  }
  return a;
}
// the rotation (EPI_RED, schedule.py ROT_MAX): an instruction whose last unit holds only its last k stages (0: none)
// processes them first, then [r0, r1 - k), so that the short segment's drain runs under the rest's stream (its
// counted sums are exact in any order)
constexpr uint32_t AUX_ROT_K_SHIFT = 16;   // Instr::aux bits 16..23: k
constexpr uint32_t AUX_ROT_K_MASK = 0xffu;    // (its 8 bits)
// the first stage's B group (the slot (r1 - k - r0) mod bw over B_GCH), copied first
constexpr uint32_t AUX_ROT_GROUP_SHIFT = 24;   // Instr::aux bits 24..27
constexpr uint32_t AUX_ROT_GROUP_MASK = 0xfu;  // (its 4 bits)
__device__ __forceinline__ uint32_t rot_k(const Instr& in) { return (in.aux >> AUX_ROT_K_SHIFT) & AUX_ROT_K_MASK; }
__device__ __forceinline__ uint32_t rot_group(const Instr& in) {
  return (in.aux >> AUX_ROT_GROUP_SHIFT) & AUX_ROT_GROUP_MASK;
}
// the instruction's K chunks: kc0, kc0 + 1, ... (mod its band's bw chunks, within the band), nck of them
struct ChunkRange {
  uint32_t bw, nck, kc0;
};
__device__ __forceinline__ ChunkRange chunk_range(const Instr& in, const OpDesc& d) {
  const StageAt a = stage_at(d, in.r0);
  const uint32_t n = in.r1 - in.r0, bw = d.paired ? d.spm : d.bw;
  const uint32_t span = d.paired ? (n + a.half + 1) / 2 : n;
  return ChunkRange{bw, min(span, bw), a.kc};
}
constexpr uint32_t NO_SLOT = 0xffffffffu;  // chunk_slot of a chunk of another band (>= any nck)
// the B slot of K chunk kc
__device__ __forceinline__ uint32_t chunk_slot(uint32_t kc, const ChunkRange& cr) {
  const uint32_t b0 = cr.kc0 - cr.kc0 % cr.bw;
  if (kc < b0 || kc >= b0 + cr.bw) return NO_SLOT;
  return kc >= cr.kc0 ? kc - cr.kc0 : kc + cr.bw - cr.kc0;
}
// the absolute K chunk of B slot l
__device__ __forceinline__ uint32_t slot_chunk(uint32_t l, const ChunkRange& cr) {
  const uint32_t b0 = cr.kc0 - cr.kc0 % cr.bw;  // the band's first chunk
  const uint32_t r = cr.kc0 - b0 + l;
  return b0 + (r < cr.bw ? r : r - cr.bw);
}

}  // namespace mk
