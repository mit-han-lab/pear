// The draft-proposal megakernel's shapes, warp roles, shared-memory layout, instruction and descriptor formats, and the
// per-launch arguments (schedule.py packs the same layouts).
#pragma once
#include "dk_common.cuh"

namespace dk {

// ---- the DFlash 2 head of Qwen3.8-27B (schedule.py checks the loaded model against these) ----
constexpr int HID = 5120;                          // hidden size
constexpr int INTER = 17408;                       // MLP intermediate size
constexpr int NQH = 32, NKVH = 8, HDIM = 128;      // query heads, KV heads, head dim
constexpr int GQA = NQH / NKVH;                    // query heads per KV head
constexpr int QROWS = NQH * HDIM;                  // q rows of qkv_proj (4096)
constexpr int KVROWS = NKVH * HDIM;                // k (or v) rows of qkv_proj (1024)
constexpr int QB = 8;                              // the query block: the anchor and the mask tokens
constexpr int NSTEP = QB - 1;                      // drafts per proposal at most (Tables.nstep: the engine's)
constexpr int NCTX = 8;                            // context tokens a step commits at most (the verify step's tokens)
constexpr int NAUX = 5;                            // target layers the fc reads
constexpr int CONV_GSIZE = 16;                     // channels per grouped-conv coefficient
constexpr int CONV_GROUPS = HID / CONV_GSIZE;      // 320
constexpr int KPROJ = 4 * CONV_GROUPS;             // kernel projection outputs per token (2 sides x 2 taps x groups)
// the kernel projections' outputs: coef[parity][row r = side * 640 + tap * 320 + group][token], the fp32 sums (E_COEF's
// finisher stores them into the launch's parity in the GEMV's row order), rounded to bf16 where the convs read them
constexpr int COEF_HALF = KPROJ * QB;  // floats per parity
constexpr uint32_t QKV_SLOTS = 2;      // E_QKV: a tile's slots (its parts: at most two per slot; schedule.py)
__host__ __device__ constexpr uint32_t coef_row(uint32_t side, uint32_t tap, uint32_t g) {
  return (side * 2 + tap) * CONV_GROUPS + g;
}
constexpr int TOPK = 16, RANK = 256;               // candidates per position, selector rank
constexpr int ROPE_HALF = HDIM / 2;                // neox rotation pairs (d, d + 64)
constexpr float NORM_EPS = 1e-6f;

// ---- the GEMV: M = 128 weight rows x N tokens, 128-byte K chunks per ring stage ----
constexpr int TILE_M = 128;
constexpr int CHUNK = 128;                         // K bytes per stage
constexpr int CHUNK_K = 2 * CHUNK;                 // NVFP4: K values per chunk
constexpr int FP4_BLK = 16;                        // NVFP4: values per e4m3 scale
constexpr int SF_CHUNK = CHUNK_K / FP4_BLK;        // NVFP4: a row's scales per chunk (bytes)
// NVFP4 B scales as the MMA warp copies them into TMEM (tcgen05.cp, 32 rows x 16 bytes): two planes (k-steps 0, 1 and
// 2, 3) of [chunk position][n rows][SFB_ROW], k-step j's 4 block scales at (j & 1) * 8 of its row (the MMA reads B
// scales at even columns); a plane ends with a copy's reach
constexpr int SFB_ROW = 16;
constexpr int SFB_SLACK = 32 * SFB_ROW;
__host__ __device__ constexpr uint32_t sfb_plane(uint32_t nch, uint32_t n) { return nch * n * SFB_ROW + SFB_SLACK; }
// the place of a chunk row's block b (0..15) in the planes (plane stride `plane`)
__host__ __device__ constexpr uint32_t sfb_pos(uint32_t b, uint32_t plane) { return (b >> 3) * plane + ((b >> 2) & 1) * 8 + (b & 3); }
constexpr int NB = 16;                             // B rows at most (qkv: 8 query + 8 context tokens)
constexpr int KSTEP_BYTES = 32;                    // K bytes per MMA instruction
constexpr int KSTEPS = CHUNK / KSTEP_BYTES;        // MMAs per stage

// ---- warp roles (each walks the CTA's instruction list on its own; synchronized by mbarriers) ----
constexpr int NW = 20, NTHREADS = NW * 32;
constexpr int WG_WARPS = 4;  // warps per warpgroup, one per TMEM lane quadrant (the base taps', the epilogue's)
constexpr int W_SF0 = 0;    // 0..3: the convs' base taps into TMEM (lane quadrants 0..3)
constexpr int W_EPI0 = 4;   // 4..7: epilogue (TMEM -> split-K reduction -> the op's finisher)
constexpr int W_PROD = 8;   // weight tiles by TMA into the ring (never waits on dependencies)
constexpr int W_MMA = 9;    // tcgen05.mma issuer (10, 11 idle)
constexpr int W_WRK0 = 12;  // 12..19: B operands, attention, top-k, selector, walk
constexpr int NWRK = NW - W_WRK0, NWT = NWRK * 32;
constexpr int NEPI = 128;
// setmaxnreg per warpgroup: REG_IDLE for 0 (base taps), REG_LO for 2 (producer, MMA), REG_EPI for 1 (epilogue),
// REG_HI for 3-4 (workers)
constexpr int REG_LAUNCH = 96, REG_IDLE = 32, REG_LO = 56, REG_EPI = 120, REG_HI = 136;
constexpr int SM_REGS = 65536;  // 32-bit registers per SM
static_assert(NTHREADS * REG_LAUNCH <= SM_REGS, "launch registers");
// a warpgroup's register count from the launch's (setmaxnreg: inc above it, dec below)
template <int N>
__device__ __forceinline__ void set_max_regs() {
  if constexpr (N > REG_LAUNCH) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
  else if constexpr (N < REG_LAUNCH) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
static_assert(WG_WARPS * 32 * (REG_IDLE + REG_LO + REG_EPI) + NWT * REG_HI <= NTHREADS * REG_LAUNCH, "register budget");
constexpr int BAR_EPI = 2, BAR_WRK = 3;  // named barriers (0: __syncthreads)
enum Role : int { R_PROD = 0, R_MMA = 1, R_SF = 2, R_EPI = 3, R_WRK = 4, NROLE = 5 };
constexpr int MAX_INSTR = 63;  // instructions per SM the timestamps cover
constexpr int NSUB = 8;        // timestamps per instruction and role: its start, then points within it

// ---- shared memory ----
#ifdef DK_RING_STAGES  // (a host kernel's ring: its stage count, DK_RING_STRIDE its slots' stride)
constexpr int NSTAGE = DK_RING_STAGES;
#else
constexpr int NSTAGE = 8;
#endif
constexpr int A_BYTES = TILE_M * CHUNK;              // 16 KB of weights per stage (SW128)
constexpr int SFA_BYTES = TILE_M * SF_CHUNK;         // NVFP4: the stage's weight-scale atoms, after its weights
constexpr int STAGE_BYTES = A_BYTES + SFA_BYTES;
#ifdef DK_RING_STRIDE
constexpr int STAGE_STRIDE = DK_RING_STRIDE;  // (a ring slot: the draft's stage fits it)
#else
constexpr int STAGE_STRIDE = STAGE_BYTES;
#endif
static_assert(STAGE_STRIDE >= STAGE_BYTES && STAGE_STRIDE % 1024 == 0, "a stage in a ring slot (SW128 alignment)");
static_assert(STAGE_BYTES % 1024 == 0, "SW128 stages");
// the B buffer: a GEMV's B operand for the SM's K band, built once per op: per chunk a tile of codes [n rows][128 B]
// (SW128) and the rows' scales [n rows][SF_CHUNK]; between GEMVs the workers' scratch (attention, selector)
constexpr int BBUF_BYTES = 67584;
constexpr int BBUF_CHUNKS = 64;  // B chunk tiles (BSlots' bits)
constexpr uint32_t PACE_NS = 64;                         // the producer's poll interval on the gate word (epi_rel)
constexpr uint32_t SMEM_ALIGN = 1024;  // the dynamic shared memory's base (SW128 tiles)
constexpr int SMEM_BYTES = NSTAGE * STAGE_STRIDE + BBUF_BYTES + SMEM_ALIGN;  // (+ SMEM_ALIGN: the base's alignment)
constexpr int XCHG_FLOATS = TILE_M * NB;             // the epilogue's exchange buffer

// ---- TMEM: two accumulator buffers, then two NVFP4 scale slots (stage k's in slot k % 2: written by the MMA warp's
// copies right before the stage's MMAs, executed in order with them) ----
constexpr uint32_t T_COLS = 512;
constexpr uint32_t T_ACC = 16;                       // columns per accumulator buffer (>= NB)
constexpr int N_ACC = 2;  // accumulator buffers (the MMA fills one while the epilogue reads the other)
constexpr uint32_t T_SC = 32, SC_COLS = 32, N_SC = 2;  // a slot: SFA (4 k-steps x 4 columns) at +0, SFB (4 x 2) at +16
constexpr uint32_t SC_SFB = 16;
static_assert(T_SC + N_SC * SC_COLS <= T_COLS, "TMEM columns");
// qkv's context rows (hidden_norm of the fc output, quantized: the same in every layer) cached by layer 0's first
// phase: per worker warp's lane quadrant (two warps each) and step of CPS chunks, a lane's code words and scale
constexpr uint32_t T_CTX = T_SC + N_SC * SC_COLS, CTX_STEPS = 6, CTX_WORDS = 4;
// the two-phase operands' base conv taps (side 0), loaded by warpgroup 0 ahead of the conv phase: N_BASE regions,
// per step of CPS chunks a lane's 16 words (taps 0, 1 of its block's channels)
constexpr uint32_t T_BASE = T_CTX + 2 * CTX_STEPS * CTX_WORDS, BASE_WORDS = 16, BASE_REGION = CTX_STEPS * BASE_WORDS;
constexpr int N_BASE = 2;  // base-tap regions (the SM's two-phase GEMVs alternate)
static_assert(T_BASE + N_BASE * BASE_REGION <= T_COLS, "TMEM columns");

// instruction descriptors (kind::f16 bf16 x bf16 -> f32, kind::mxf4nvf4 e2m1 x e2m1 with ue4m3 scales): M = 128
__host__ __device__ constexpr uint32_t idesc_bf16(uint32_t n) { return (1u << 4) | (1u << 7) | (1u << 10) | ((n >> 3) << 17) | (8u << 24); }
__host__ __device__ constexpr uint32_t idesc_nvf4(uint32_t n) { return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | (8u << 24); }

// ---- instructions and descriptors ----
#ifndef DK_OP_BASE  // (the draft's op codes in a host kernel's instruction lists: a range of their own)
#define DK_OP_BASE 0
#endif
enum Op : uint16_t {
  OP_PREP = DK_OP_BASE + 0, OP_GEMV = DK_OP_BASE + 1, OP_ATTN = DK_OP_BASE + 2, OP_SELECT = DK_OP_BASE + 3,
  OP_QUANT = DK_OP_BASE + 4,
};
enum Kind : uint32_t { KIND_BF16 = 0, KIND_NVFP4 = 1 };
// where a GEMV's B operand comes from (the worker warps build the SM's band of it once per op, in NVFP4)
enum BSrc : uint32_t {
  B_AUX = 0,        // fc: the target's aux hidden states of the step's tokens
  B_NORM = 1,       // RMSNorm(x) (x fp32 [8][K], or the embedding rows for layer 0)
  B_NORM_CONV = 2,  // the grouped conv (side 0) of RMSNorm(x)
  B_QKV = 3,        // rows 0..7 B_NORM_CONV, rows 8..15 hidden_norm(fc output) of the committed tokens
  B_COPY = 4,       // a bf16 [8][K] activation as is (attention output, the MLP's silu * up)
  B_FINAL = 5,      // the final norm of the 7 mask positions (the selector's hidden projection)
  B_FP4 = 6,        // the final norm of the 7 mask positions (the LM head): OP_QUANT's image (e[1]) copied whole
};
// what the finisher of a tile (the last of its split-K parts) does with the tile's sums
enum Epi : uint32_t {
  E_BF16 = 0,    // bf16 out [rows of b][n_out]
  E_FC = 1,      // the fc output (bf16) and its sums of squares
  E_QKV = 2,     // the parts' sums added into the launch's parity [slot][tile][column][128 rows] (fp32, two parts per
                 // slot), no finisher: the attention items sum the slots, norm, rope and store them
  E_RESID = 3,   // the side-1 conv of the output plus the residual (fp32) and its sums of squares
  E_SILU = 4,    // silu(gate) * up of a paired tile (bf16)
  E_LOGITS = 5,  // bf16 logits into the SM's running top-16 per position
  E_COEF = 6,    // the kernel projection's sums into its output rows (fp32, the launch's parity)
};

constexpr uint32_t BAND_SLOT = 1u << 15;  // Instr::slot of a band op (schedule.py BAND_SLOT)

struct alignas(16) Instr {  // 32 bytes (schedule.py INSTR)
  uint16_t op, desc;
  uint16_t dep, done;  // the counter this instruction waits on / counts (0xffff: none)
  uint32_t r0, r1;     // GEMV: stage range [r0, r1) of the band's (M tile, K chunk), chunk fastest; others: items
  uint16_t kb0, kbw;   // GEMV: the band's K chunks [kb0, kb0 + kbw)
  uint16_t tile;       // this instruction's index within its op
  uint16_t aux;        // GEMV: a second dependency of the B operand (counter + 1, 0: none)
  uint32_t flags;      // F_*
  uint16_t xn;         // F_XN writers: the counter + 1 their workers count on once the normed rows are written
  uint16_t slot;       // GEMV: its parts' accumulator slot in their tiles (schedule.py: a tile's two consecutive
                       // parts' one; a band op: BAND_SLOT | band count << 8 | its band, the slot in the tile table)
};
static_assert(sizeof(Instr) == 32, "Instr layout");
// GEMV: its weights stream without waiting for the release of the SM's previous GEMV (the producer gate)
constexpr uint32_t F_NOGATE = 1;
// GEMV: a kernel projection's SM that writes its chunks' normed rows (bf16, b[3]) and releases counter xn - 1, or a
// two-phase GEMV whose first phase copies them (b[0]) into its staging once that counter completes
constexpr uint32_t F_XN = 2;

// a GEMV stage's M tile and K chunk
struct StageAt {
  uint32_t mt, kc;
};
__device__ __forceinline__ StageAt stage_at(const Instr& in, uint32_t st) { return {st / in.kbw, in.kb0 + st % in.kbw}; }
// a stage's B tile in the SM's B buffer: the range's distinct chunks in first-use order (slot i: chunk of stage r0 + i)
__device__ __forceinline__ uint32_t b_slot(const Instr& in, uint32_t st) { return (st - in.r0) % in.kbw; }

// The order every role processes an instruction's stages in: the partial tile at r0, the partial tile ending at r1,
// then the whole tiles between. Its last segment is then a whole tile (if it has one), long enough for the epilogue
// to finish the shorter segments' reductions meanwhile; the B slots are still first used in slot order.
struct StageOrder {
  uint32_t r0, a1, b0, r1;  // positions: [r0, a1), then [b0, r1), then [a1, b0)
  __device__ __forceinline__ explicit StageOrder(const Instr& in)
      : r0(in.r0), a1(min(in.r1, (in.r0 / in.kbw + 1) * in.kbw)), b0(0), r1(in.r1) {
    b0 = max(a1, in.r1 / in.kbw * in.kbw);
  }
  __device__ __forceinline__ uint32_t len() const { return r1 - r0; }
  // the stage at position k
  __device__ __forceinline__ uint32_t at(uint32_t k) const {
    const uint32_t na = a1 - r0, nb = r1 - b0;
    return k < na ? r0 + k : (k < na + nb ? b0 + (k - na) : a1 + (k - na - nb));
  }
  // the stages of position k's segment (one M tile, within its run)
  __device__ __forceinline__ uint32_t seg(uint32_t k, uint32_t kbw) const {
    const uint32_t na = a1 - r0, nb = r1 - b0, st = at(k);
    const uint32_t run_end = k < na ? a1 : (k < na + nb ? r1 : b0);
    return min(run_end - st, kbw - st % kbw);
  }
};
constexpr uint16_t NO_CTR = 0xffff;

struct GemvDesc {  // 256 bytes (schedule.py GEMV)
  uint64_t tmap;        // CUtensorMap over the weights [rows][row bytes], box 128 B x 128 (paired: 64) rows, SW128
  uint64_t sf;          // NVFP4: weight scales in vLLM's swizzled layout ([rows/128][K/64][32][4][4] bytes)
  uint64_t acc;         // fp32 split-K accumulator [mtiles][acc_slots<n>()][128][n] (zero between launches)
  uint64_t tile_ctr;    // uint32 [mtiles]: parts arrived (zero between launches)
  uint64_t tile_parts;  // uint16 [mtiles]: split-K parts of each tile | its accumulator's slots in use << 8; then a
                        // band op's slot table uint8 [mtiles][bands][2] (by a part's band, whether it starts mid-tile)
  uint64_t b[8];        // the B source's tensors (by bsrc)
  uint64_t e[8];        // the finisher's tensors (by epi)
  uint32_t K, rows, mtiles, kchunks, kind, n, paired, bsrc, epi, layer;
  uint32_t n_out;       // E_BF16: rows of the output (tokens)
  uint32_t resid;       // E_RESID: the residual is 0 the embedding rows, 1 bf16(fp32 buffer), 2 the fp32 buffer
  float alpha;          // NVFP4: the GEMM's alpha (1 / (the weights' global scale * gs))
  float gs;             // NVFP4: the input's global scale (multiplier)
  uint32_t xemb;        // B_NORM*: x is the query block's embedding rows (layer 0) instead of b[0]
  uint32_t conv_fma;    // E_RESID: Inductor fuses c0 * o into the conv's add (the MLP's), else rounds it (attention's)
  uint32_t b_off;       // the op's B tiles' offset in the B buffer (bytes) and their first B slot barrier: an op whose
  uint32_t b_slot0;     // B is prepared during the previous op keeps clear of that op's tiles
  uint32_t sfb_off;     // the B scales' offset in the B buffer (its two planes, sfb_plane)
  uint32_t stg_off;     // a two-phase operand's staging (bf16 [chunk position][QB rows][CHUNK_K]): its normed rows
  uint32_t sfb_plane;   // the B scales' plane stride
  uint32_t ist;         // the B build's input staging: offset (bits 0..19), chunks per batch (20..31)
};
static_assert(sizeof(GemvDesc) == 256, "GemvDesc layout");
// the GemvDesc tensors the draft's roles read: b[] by the B source, e[] by the finisher (or the op)
constexpr int BT_X = 0;        // x: the input rows (fp32 [QB][K]), B_COPY's bf16 activation, F_XN's normed rows
constexpr int BT_SSQ = 1;      // x's sums of squares (fixed point, [2][QB])
constexpr int BT_NORM_W = 2;   // the RMSNorm's weight
constexpr int BT_XN_OUT = 3;   // B_NORM with F_XN: its normed rows out (bf16 [QB][HID])
constexpr int BT_COEF = 3;     // a two-phase operand's conv coefficients (the kernel projection's sums)
constexpr int BT_BASE = 4;     // a two-phase operand's base conv kernel (bf16 [2][HID])
constexpr int BT_CTX = 5;      // B_QKV: the fc output (the committed tokens' context rows)
constexpr int BT_CTX_W = 6;    // B_QKV: hidden_norm's weight
constexpr int BT_CTX_SSQ = 7;  // B_QKV: the fc output's sums of squares
constexpr int ET_PARTS = 0;    // E_LOGITS: the SMs' sorted top-16 lists
constexpr int ET_IMAGE = 1;    // the LM head's B operand image: OP_QUANT's output, B_FP4's source

// ---- the per-launch arguments: vLLM's device tensors of the step and the proposal's outputs ----
struct StepArgs {
  const __nv_bfloat16* aux[NAUX];  // [n_ctx][HID] each (row stride aux_stride)
  uint32_t aux_stride, n_ctx;
  const int64_t* positions;        // [n_ctx]: the step's target tokens' positions
  const int32_t* num_rejected;     // [1]
  const int32_t* num_sampled;      // [1]
  const int64_t* idx_mapping;      // [1]: the request's state slot
  const int64_t* last_sampled;     // [max_reqs][1]
  const int32_t* next_prefill;     // [max_reqs][1]
  const int64_t* seeds;            // [max_reqs]
  const float* temperature;        // [max_reqs]
  const int32_t* block_table;      // the draft's KV cache group row [blocks]
  int64_t* draft_tokens;           // out [nstep]
  float* draft_logits;             // in / out [max_reqs][nstep][vocab]: -inf but the cached candidates
  int64_t* cached_cand;            // in / out [max_reqs][nstep][TOPK]
  float* sel_scores;               // out [max_reqs][nstep][TOPK]: the realized selector rows
  uint32_t bonus_row, pad;         // the anchor: last_sampled[req] (vLLM's request state), or 1: last_sampled is the
                                   // step's sampled row and the anchor its entry num_sampled - 1 (a host's sampler)
};

struct Tables {  // the model's constant tensors the kernel addresses directly
  const __nv_bfloat16* embed;      // [vocab][HID] (the target's)
  const __nv_bfloat16* cos_sin;    // [max_pos][HDIM]: cos | sin (bf16)
  const __nv_bfloat16* pred_cb;    // [vocab][RANK]
  const __nv_bfloat16* succ_cb;    // [vocab][RANK]
  uint32_t vocab, mask_token, max_model_len, kv_block, window_left, topk_chunk;
  uint32_t max_reqs;
  uint32_t nstep;      // drafts per proposal (the engine's num_speculative_tokens): the block is nstep + 1 tokens
};

struct DkArgs {
  const Instr* instrs;
  const uint32_t* sm_off;     // [nsm + 1]: each CTA's instruction range
  const GemvDesc* descs;      // 256-byte descriptors (GemvDesc or another op's)
  uint32_t* counters;
  const uint32_t* ctr_count;  // arrivals that complete each counter per launch
  uint32_t* epoch;            // [0] epoch, [1] CTAs that have read it
  long long* ssq_zero;        // the sums-of-squares buffers' next-parity halves to zero: [n][2][QB] (fixed point)
  uint32_t n_ssq;
  uint32_t* progress;         // [nsm][NROLE]: each role's current instruction (+1), for hang diagnosis
  uint64_t* tstamp;           // [nsm][NROLE][MAX_INSTR + 1][NSUB] globaltimer within each instruction (null: off)
  StepArgs step;
  Tables tab;
};

}  // namespace dk
