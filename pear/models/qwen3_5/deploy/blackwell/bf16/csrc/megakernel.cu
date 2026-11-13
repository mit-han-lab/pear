// The decode megakernel of the bf16 checkpoint: one persistent kernel per decode step (one CTA per SM) that walks an
// on-device schedule of instructions (schedule.py), one per (operation, SM), on bf16 weights.
#include <chrono>
#include <cstdint>
#include <thread>
#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef PEAR_DEVICE_ONLY  // defined for a device-only cubin, without the torch host code
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "common.cuh"
#include "../../common/csrc/step.cuh"
#include "../../common/csrc/gdn.cuh"
#include "gdn_t.cuh"
#include "attention.cuh"
#include "../../common/csrc/logprobs.cuh"
#include "../../common/csrc/spec_sample.cuh"
// the draft's proposal as instructions of this launch, after the verify step's on every CTA (draft/csrc: the draft's
// roles, dk_roles.cuh): its op codes from DK_OP_BASE, this kernel's ring (checked below)
#define DK_OP_BASE 32
// the draft's ring: 6 stages, its 96 KB B buffer and arguments past them (over the target's later ring slots, B buffers
// and staging buffer); with another target ring its barriers are re-initialized at the transition (draft_ring_reset)
#define DK_RING_STAGES 6
#define DK_RING_STRIDE 16384
#include "../draft/csrc/dk_roles.cuh"

namespace mk {

constexpr int NW = 16;  // warps per CTA
constexpr int NTHREADS = NW * 32;
// warpgroups: 0 epilogue (warps 0..3, TMEM lane quadrant = warp % 4), 1 producer + MMA (4, 5; 6, 7: one lane each
// for release counts), 2-3 prologue (8..15); setmaxnreg moves the registers of warpgroups 0-1 to the prologue warps
// (their ops: norms, cores, sampler); no warpgroup is left idle
constexpr int W_EPI0 = 0, W_PROD = 4, W_MMA = 5, W_PRO0 = 8;
constexpr int NPRO = (NW - W_PRO0) * 32;  // prologue threads
// the epilogue warpgroup takes the budget's remainder in a setmaxnreg region of its own: below REG_EPI its instruction
// loop, FIN's poll buffer and the consumers' epilogue code spill to local memory (its NT accumulator columns need 96)
constexpr int REG_LAUNCH = 128, REG_LO = 56, REG_EPI = 120, REG_HI = 168;
static_assert(4 * 32 * REG_LO + 4 * 32 * REG_EPI + NPRO * REG_HI <= NTHREADS * REG_LAUNCH, "register budget");
// the draft's phase after the step (DREG_*): the prologue warps give registers back (setmaxnreg.dec), then the
// epilogue's finishers take DREG_EPI from the pool (setmaxnreg.inc waits for it); warpgroup 1 keeps REG_LO (an .inc
// there hangs the launch: its idle warps reach it at the start); the step keeps a split of its own
constexpr int DREG_EPI = 120, DREG_HI = 160;
static_assert(4 * 32 * REG_LO + 4 * 32 * DREG_EPI + NPRO * DREG_HI <= NTHREADS * REG_LAUNCH, "the draft's budget");
static_assert(DREG_HI <= REG_HI && DREG_EPI >= REG_EPI && NPRO * (REG_HI - DREG_HI) >= 4 * 32 * (DREG_EPI - REG_EPI),
              "the transition: the prologue warps' release covers the epilogue's increase");
static_assert(W_EPI0 == 0 && W_PROD == 4 && W_PRO0 == 8 && W_PRO0 % 4 == 0, "warpgroups (setmaxnreg), TMEM lane quadrants");
static_assert(NPRO >= 256, "the GDN core maps q, k channels to 256 prologue threads");
constexpr int NT = 8;  // B rows (the MMA's N): the step's tokens at most; T <= NT of them per launch (SR_NTOK)
static_assert(NT == 8 && ATT_NT == NT, "B rows: the MMAs' N = 8 (IDESC_*), 8 accumulator columns a segment (tc_ld_x8)");
// Instr::aux of a GEMV instruction, its low bits (schedule.py AUX_*; its bits 20.. below)
constexpr uint32_t AUX_WARM = 2;          // bit 1: its prologue's warm-up dry run (none in this package's schedule)
constexpr uint32_t AUX_NOGATE = 4u;       // bit 2: no producer gate (mark_gate)
constexpr uint32_t AUX_PARK_SHIFT = 8;    // bits 8..15: its TMEM park slots (mark_park)
constexpr uint32_t AUX_PARK_MASK = 0xffu;  // (its 8 bits)
constexpr uint32_t AUX_PACE_SHIFT = 16;   // bits 16..19: its pacing limit (mark_park; 0: PACE)
constexpr uint32_t AUX_PACE_MASK = 0xfu;   // (its 4 bits)
// Instr::aux bit 20 of a GEMV instruction (schedule.py mark_split): its CTA pair (CTAs 2c, 2c + 1: a cluster) builds
// the RMSNorm B operands of the op's two instructions together, each CTA half of the token rows into both B buffers
constexpr uint32_t AUX_SPLIT = 1u << 20;
// Instr::aux bit 21 (schedule.py mark_perm): the instruction's stages are loaded in a permuted order, the stages to be
// parked first (perm_stage), so that at B-ready its first stages are in the ring and are issued first: their slots free
// at once and the stream goes on while the parked stages are issued (the MMAs still run in stage order)
constexpr uint32_t AUX_PERM = 1u << 21;
// Instr::aux bit 22 of a GEMV instruction (schedule.py mark_epi_relaxed): its epilogue's count is relaxed (its outputs
// are tagged words the consumer polls; its prologue's count still releases the residual slices): no release fence
// behind the next op's loads in flight, and no producer gate before the next GEMV's stream
constexpr uint32_t AUX_EPI_RELAXED = 1u << 22;
// Instr::aux bits 24..31 of an RMSNorm consumer of a validated op (schedule.py mark_dep_early): its dependency wait
// ends this many counts per token row (T) short of the target, so that the first row's loads leave while the producer's
// last parts finish (every value validated: those still at the sentinel re-polled)
constexpr uint32_t AUX_DEP_EARLY_SHIFT = 24;
// Instr::aux bit 23 (schedule.py mark_znext): an OP_GDNT instruction's idle warps 2-7 zero its next instruction's slice
// of the op's next-parity accumulator / output copy (zero_next_acc) in the core's output phase, and that GEMV
// (out_proj) skips it (cold code on the path from the last core item's release to out_proj's B)
constexpr uint32_t AUX_ZNEXT = 1u << 23;
// Instr::aux bit 3 (schedule.py mark_twin) of a PRO_COPY_B GEMV whose B the previous instruction's prologue on the SM
// writes (its twin), which also arms its xfull, pacing count, prologue count and zero_next_acc once both are complete;
// the prologue warps skip it (in_proj_z: so that the GDN core, next on them, starts sooner)
constexpr uint32_t AUX_TWIN = 1u << 3;
// Instr::aux bit 6 (schedule.py mark_fastact) of an out_proj GEMV instruction after the GDN core on its SM (validated,
// AUX_ZNEXT, no dependency): its prologue warps' path of its own (fast_act: the descriptor, the B slot, stage_val, B
// complete, the counts), out of line
constexpr uint32_t AUX_FASTACT = 1u << 6;
// Instr::aux bit 4 (schedule.py mark_fastact, FAPRE) on an OP_GDNT instruction and its AUX_FASTACT out_proj: the
// core's idle warps also copy out_proj's descriptor into the prologue's shared copy (with AUX_ZNEXT's block, before the
// release's barrier); fast_act skips its copy and its barrier
constexpr uint32_t AUX_FAPRE = 1u << 4;
constexpr int NSTAGE = 6;  // ring stages of 16 KB of bf16 weights (128 rows x 64 values); even (see below)
// the attention core's two warp groups take alternate ring stages: with an even ring a slot is always reused by the
// same group, so neither group can wait on a slot a full phase ahead of the other's (a parity wait cannot tell)
static_assert(NSTAGE % 2 == 0, "attention: even ring");
// until an instruction's prologue has staged its inputs, at most PACE ring stages in flight (Instr::aux bits 16..19
// override it): a burst of ring loads queued in the SM's TMA unit would delay the staging copy
constexpr uint32_t PACE = 3;
// Instr::done of a validated op's instructions (the validated hand-off, OpDesc::val): the counter index | DONE_RELAXED
// (the flag rides in the index: the epilogue role has no register to carry the op's flag)
constexpr uint16_t DONE_RELAXED = 0x8000;
constexpr int A_BYTES = 128 * 128;         // one stage of A: 128 rows x 128 bytes (SW128)
constexpr int SFA_BYTES = 4 * 512;         // weight-scale atoms of the stage's 4 MMA k-steps
// a stage is its A tile alone: the NVFP4 scale atoms' paths below are never taken (schedule.py asserts bf16 weights)
constexpr int STAGE_BYTES = A_BYTES;  // multiple of 1024 (an attention KV stage: 16 KB)
// B operand: the instruction's K chunks, chunk kc of the op at local slot (kc - kc0) mod spm (chunk_slot), each NT rows
// x 128 bytes as no-swizzle K-major core matrices [16-byte group][row][16 bytes] (LBO 128 B)
constexpr int CHUNK_B = NT * 128;
// an instruction's K chunks at most (schedule.py B_CHUNKS), 64 bf16 values each (K = 5120 whole, part of a larger K);
// one B buffer (NB): the next instruction's B is built once this one's MMAs are done, as the step's chain of dependent
// ops allows, TMEM parking keeping the weight stream going meanwhile
constexpr int B_CHUNKS = 80, NB = 1;
constexpr int XB_BYTES = B_CHUNKS * CHUNK_B;
// the B operand's NVFP4 block scales: per K chunk two halves of [NT rows][16 bytes], row n's 16 bytes of half h holding
// k-steps 2h, 2h + 1's 4 block scales at + 0 / + 8 (sfb_pos); a tcgen05.cp of a half's 32 rows (past NT: read by no
// MMA) lands them at columns 0, 2 / 4, 6 of the stage's SFB columns, B row n at lane n of every quadrant
constexpr int SFB_HALF = NT * 16, SFB_CHUNK = 2 * SFB_HALF;
constexpr int SFB_CHUNKS = 20;  // an NVFP4 instruction's K chunks at most
constexpr int SFB_BYTES = SFB_CHUNKS * SFB_CHUNK + 512 - SFB_HALF;  // + the last copy's reach
constexpr int STG_BYTES = 40960;  // prologue / core scratch: bulk-copied inputs, the GDN core's state rows, the sampler
constexpr uint32_t K_NORM = 5120;  // the hidden size: the length the RMSNorm prologues are written for
// in_proj_qkv's prologue: its in_proj_ba rows past the fp32 residual row (pro_norm_rows BA)
constexpr int BA_OFF = K_NORM * 4;
constexpr uint32_t BA_OUT = 96;  // in_proj_ba's outputs: b of the 48 value heads, then a
static_assert(BA_OFF + 2 * K_NORM * 2 <= STG_BYTES, "in_proj_ba's rows in the staging buffer");
// a B buffer slot: the operand's chunks, an NVFP4 operand's scales at SFB_OFF, past the SFB_CHUNKS chunks it has at
// most (an FP8 / BF16 operand, which may fill the slot, has none)
constexpr int SFB_OFF = SFB_CHUNKS * CHUNK_B;
constexpr int B_SLOT_BYTES = XB_BYTES > SFB_OFF + SFB_BYTES ? XB_BYTES : SFB_OFF + SFB_BYTES;
// the spec sampler's scratch: its SsArgs, then the group's SsSmem (16-byte aligned); a dry run's in the staging buffer
// (its first ~10 KB: one vector of SsSmem::x)
constexpr int SS_ARGS_BYTES = (int)((sizeof(SsArgs) + 15) / 16 * 16);
constexpr int B_AREA_BYTES = NB * B_SLOT_BYTES;  // the B buffers, then the staging buffer
static_assert(SS_ARGS_BYTES + sizeof(SsSmem<NPRO>) <= B_AREA_BYTES, "the spec sampler's scratch");
static_assert(SS_ARGS_BYTES + offsetof(SsSmem<NPRO>, x) + LP_VEC_BYTES <= STG_BYTES, "a dry run's scratch");
constexpr int SMEM_BYTES = NSTAGE * STAGE_BYTES + B_AREA_BYTES + STG_BYTES;
// TMEM (every column past the accumulators is written by the MMA issuer's tcgen05.cp only, so a later instruction's
// copies and parked stages execute after an earlier one's MMAs that read the same columns: no other ordering needed)
constexpr uint32_t T_COLS = 512;           // TMEM: accumulators in columns 0..15
// NVFP4 scale slots (stage k of an instruction: slot k % N_SC): SFA (4 k-steps x 4 columns) at +0, SFB (4 x 2) at +16
constexpr uint32_t T_SC = 32, SC_COLS = 32, N_SC = 2;
// park slots: FP8 32 columns (A) from column 32 (up to 15); NVFP4 48 (A, then its SFA) from column 96 (up to 8)
constexpr uint32_t T_PARK_F8 = 32, PARK_F8_COLS = 32, T_PARK_NV = T_SC + N_SC * SC_COLS, PARK_NV_COLS = 48;
// (NVFP4: one more slot in the columns no slot covers: its A at the 32 columns past the 8th slot, its weight
// scales at the 16 between the accumulators and the scale slots)
constexpr uint32_t MAX_PARK_F8 = (T_COLS - T_PARK_F8) / PARK_F8_COLS,
                   MAX_PARK_NV = (T_COLS - T_PARK_NV) / PARK_NV_COLS + 1u;
constexpr uint32_t PARK_NV9 = (T_COLS - T_PARK_NV) / PARK_NV_COLS;  // the extra slot's index
static_assert(T_PARK_NV + PARK_NV9 * PARK_NV_COLS + 32 <= T_COLS && 2 * NT + 16 <= T_SC, "the extra NVFP4 park slot");
// (its weight scales' columns as the regular place - PARK9_SF_BACK: no TMEM base held for it in the MMA warp's role)
constexpr uint32_t PARK9_SF_BACK = T_PARK_NV + PARK_NV9 * PARK_NV_COLS + 32 - 2 * NT;
// the B scales' place of row n's NVFP4 block b (16 values: k-step b / 4, block b % 4) of the chunk at slot c
__device__ __forceinline__ uint32_t sfb_pos(uint32_t c, uint32_t b, uint32_t n) {
  return c * SFB_CHUNK + ((b >> 3) & 1) * SFB_HALF + n * 16 + ((b >> 2) & 1) * 8 + (b & 3);
}
constexpr uint32_t IDESC_NVF4 = (1u << 7) | (1u << 10) | (1u << 17) | (8u << 24);  // e2m1 x e2m1, ue4m3 SF, N=8, M=128
constexpr uint32_t IDESC_F8 = (1u << 4) | (1u << 17) | (8u << 24);                 // e4m3 x e4m3 -> f32, N=8, M=128
constexpr uint32_t IDESC_BF16 = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 17) | (8u << 24);  // bf16 x bf16 -> f32
constexpr int BAR_PRO = 1, BAR_EPI = 2;  // named barriers
constexpr int BAR_GDNT = 5;               // OP_GDNT: the coefficient warps (6-7 of the prologue warps)
// ACC_CNT: 64-bit word per output = (sum of fx) * 2^CNT_BITS + parts added, fx = round(alpha * part * 2^28)
constexpr int CNT_BITS = 8;
constexpr float CNT_SCALE = 268435456.0f, CNT_INV = 1.0f / 268435456.0f;

// the shared B's staging: the backoff between two re-poll rounds (ns), a default a build may set with -DMK_SB_REPOLL_NS
#ifndef MK_SB_REPOLL_NS
#define MK_SB_REPOLL_NS 64
#endif
constexpr uint32_t SHARED_B_REPOLL_NS = MK_SB_REPOLL_NS;
constexpr int BARB = 4;  // pro_ba_pub: the rows of bf16(x_n) whose loads are issued together
// shared B: the RMSNorm consumers take B from their producer's finishers (PRO_ACT on GT_GNORM's validated format)
// the prologue thread issuing the RMSNorm rows' residual copies (one per row): warp 11, SMSP 3 (pt 0 shares SMSP 0
// with the producer warp, which holds the rows back)
constexpr int RESID_COPY_PT = 96;

enum Op : uint16_t {
  OP_GEMV = 0, OP_GDN = 1, OP_GDN_QK = 2, OP_ATTN = 3, OP_ATTN_COMBINE = 4, OP_EMBED = 5, OP_ZERO = 9,
  OP_GDNT = 10,  // the Gated DeltaNet core of a T-token block under replaySSM (gdn_t.cuh)
  OP_GDNT_PRE = 11,  // its replay as an instruction of its own (the split mode: gdn_t.cuh gdn_t_replay_any)
  OP_SS_A = 12, OP_SS_B = 13, OP_SS_C = 14,  // the spec sampler's phases (spec_sample.cuh SsDesc)
  OP_BOOK = 16,  // the step's inputs into the V2 runner's buffers (step.cuh BookDesc), first on SM 0
  OP_PREV = 17,  // the previous launched step's outputs into the host ring, its post_update's rest (PrevDesc), last SM
};
// GEMV epilogue: added into the op's output / bf16 logits (whole M tiles) / the consumer's bf16 operand (EPI_SILU)
enum Epi : uint32_t { EPI_RED = 0, EPI_LOGITS = 1, EPI_SILU = 2 };
enum Kind : uint32_t { KIND_NVFP4 = 0, KIND_FP8 = 1, KIND_BF16 = 2 };
// Prologues: where the activation comes from; its B-operand encoding follows the op's kind (NVFP4: vLLM's NVFP4
// quantization of bf16(v); FP8: e4m3(clamp(v / s_in)) of the fp32 v; BF16: bf16(v)). PRO_NORM_*: v = RMSNorm(x).
// Every activation has NT rows (the step's T tokens, then unused rows): row t of a [rows] vector at + t * rows.
enum Pro : uint32_t {
  PRO_NORM_IN = 0,     // x = in0 (bf16, the embedding rows)
  PRO_NORM_RESID = 1,  // x = bf16(producing op's output) + res_in (fp32, or bf16 if res_bf16); slices of x stored to
                       // x_out (fp32) and r_out (bf16) when set
  PRO_GDN_NORM = 2,    // v = bf16(gated RMSNorm per 128-value head of a GDN core's output with gate z: in_proj_z's
                       // tagged words at res_in (gate_ll), or the qkvz accumulator's z sums (src))
  PRO_ACT = 3,         // the B operand as its producer published it at in0, in the B layout by absolute chunk
                       // (EPI_SILU, OP_ATTN_COMBINE, GT_GNORM: bf16 values [K / 64][CHUNK_B]): this instruction's
                       // chunks bulk-copied into its B buffer
  PRO_COPY_B = 4,      // the B operand written by the previous GEMV instruction's prologue on this SM (OpDesc::twin),
                       // for an op on the same input right behind it (in_proj_z behind in_proj_qkv): no dependency,
                       // no inputs
};
// ACC_CNT (order-independent, compact): every part adds (fx << CNT_BITS) + 1 to the output's counted word with an
// atomic that returns the old word; the part whose add completes the count (the tile's part count, table p[1])
// stores bf16(exact sum) into the op's bf16 output p[0], which consumers read
enum AccFmt : uint32_t { ACC_F32 = 0, ACC_CNT = 1 };

struct OpDesc {  // 256 bytes (schedule.py DESC)
  uint64_t w;       // weights [rows, K bytes] (vLLM layout); OP_EMBED: the table; OP_ZERO: the segments' addresses
  uint64_t tmap;    // CUtensorMap over w: uint8, box 128 B x 128 rows (paired: 64 rows), SWIZZLE_128B
  uint64_t sf;      // NVFP4 weight scales in vLLM's swizzled layout ([rows/128][K/64][32][4][4] bytes)
  uint64_t in0;     // PRO_NORM_IN: the bf16 input [NT][K]; PRO_GDN_NORM: the core's tagged words; PRO_ACT: the
                    // published operand; OP_EMBED: the output rows; OP_ZERO: the segments' strides (4-byte units)
  uint64_t res_in;  // PRO_NORM_RESID: the residual [NT][K]; PRO_GDN_NORM (gate_ll): in_proj_z's words; OP_ZERO: segment
                    // sizes
  uint64_t nw;      // norm weight (bf16), (1 + w) applied
  uint64_t x_out;   // PRO_NORM_RESID: fp32 x = output + residual [NT][K] (each instruction stores a slice of each
                    // row); EPI_LOGITS: the logits [NT][n_out]
  uint64_t r_out;   // PRO_NORM_RESID: bf16(x), the next residual [NT][K] (slices)
  uint64_t acc;     // this op's output accumulator, copy 0 (copy 1 at acc + acc_bytes), [NT][mtiles * 128]: ACC_F32 raw
                    // sums, ACC_CNT counted words
  uint64_t rec;     // the step record (OP_EMBED, OP_ZERO)
  uint64_t p[4];    // ACC_CNT: bf16 output [NT][mtiles * 128], parts per M tile, parity stride, tagged words (ll_out);
                    // EPI_SILU: the published operand, the split tiles' partial sums [mtiles][NT][128] fp32, flags
                    // [mtiles] u32; ACC_F32 with Q_LL: p[3] the parts' tagged words [2][NT][mtiles * 128]
  float alpha, gs, eps, act_gs;  // gs: NVFP4 input global scale (multiplier); FP8 input scale s_in (divisor); act_gs:
                                 // EPI_SILU, the consumer's NVFP4 input global scale
  uint32_t K, n_out, kind, pro, src, acc_fmt, n_inst, paired, mtiles, acc_bytes;
  uint32_t res_bf16;
  uint32_t epi;      // Epi; EPI_LOGITS: bf16(alpha * sum) into x_out[row] (instructions of whole M tiles)
  uint32_t val;      // the validated hand-off: the consumers validate the bf16 outputs instead of acquiring them
  uint32_t xn;       // in_proj_qkv: 1 + the GDN core's desc whose b / a of this SM's item its prologue computes (sm.ba)
  uint32_t ll;       // PRO_GDN_NORM: the core's output polled as tagged words at in0
  uint32_t gate_ll;  // PRO_GDN_NORM: the gate z polled as tagged words at res_in (no copy, no wait)
  uint32_t slab;     // K chunks per M tile and K slab: the stages ordered slab-major (all M tiles' chunks of slab 0,
                     // then slab 1, ...), an instruction's stages in one slab; schedule.py sets the stages per M tile:
                     // one slab, the tile-major order
  uint32_t twin;     // the prologue also encodes the next instruction's B (its PRO_COPY_B op, same K and kind)
  uint32_t act_stride;  // PRO_ACT: the operand validated, 2 copies this many bytes apart (copy epoch & 1; its producer
                        // writes no 0xFFFF halfword, resets the other copy to 0xFFFF: GT_GNORM), no dependency (0:
                        // one copy, after the dependency)
  uint32_t fin;      // ACC_CNT (shared B, schedule.py FIN): 1 + the op's index among the step's FIN ops (0: atomics)
  uint64_t fslots;   // FIN: the later parts' words [mtiles][FIN_MAXP][FIN_W][128] x 16 bytes (shared by the FIN ops)
  uint64_t fparts;   // FIN: per M tile its parts' first K chunks [mtiles][FIN_MAXP] u16 (0xFFFF: none)
  uint32_t shared_b_cons;    // (shared B) 1 + the norm consumer's desc index: this op's finishers build its B (0: none)
  uint32_t shared_b_stride;  // (shared B) the B copies' parity stride (bytes)
  uint64_t shared_b;         // (shared B) the consumer's B by parity: [chunks][CHUNK_B] (NVFP4: then the scales); its
                             // 16-byte words validated (bf16: no 0xFFFF halfword; FP8: no 0xFF byte; NVFP4: no 0x8
                             // nibble, no 0xFF scale byte), the next parity's copy reset by the finishers
  uint64_t shared_xn;        // (shared B) bf16(x_n) [2][NT][K] (in_proj_ba's input on in_qkv's layers; 0: none),
                             // sentinel 0xFFFF
  uint64_t shared_b_ext;     // (shared B) the finishers' SharedBExt
};
static_assert(sizeof(OpDesc) == 256, "OpDesc layout");
static_assert(offsetof(OpDesc, fin) == 204 && offsetof(OpDesc, fslots) == 208 && offsetof(OpDesc, shared_b_cons) == 224 &&
                  offsetof(OpDesc, shared_b_stride) == 228 && offsetof(OpDesc, shared_b) == 232 &&
                  offsetof(OpDesc, shared_xn) == 240 && offsetof(OpDesc, shared_b_ext) == 248,
              "OpDesc's FIN / shared-B fields (schedule.py DESC)");
// (shared B) a FIN op's finisher outputs and its norm consumer's parameters (schedule.py link_shared_b), 128 bytes
struct SharedBExt {
  uint64_t ssq;       // [2][mtiles][NT] fp32: each M tile's sum of squares of x per token row (sentinel ~0u)
  uint64_t xs;        // [2][NT][K] x for the next finisher's residual: fp32, or bf16(x) (x_bf16); sentinel all ones
  uint64_t rin;       // [2][NT][K] the residual: the previous finisher's xs (validated); 0: res_in
  uint64_t res_in;    // the residual rows [NT][K] when rin is 0 (layer 0's embedding rows, plain)
  uint64_t nw;        // the norm's weight (bf16 [K]; 1 + w applied)
  uint64_t aux;       // (aux layers) bf16(x) [NT][K] for the draft, plain (0: none)
  uint64_t b;         // the consumer's B (OpDesc::shared_b: the finishers read their copy here, not the descriptor's)
  uint64_t xn;        // bf16(x_n) (OpDesc::shared_xn)
  float gs, eps;      // the consumer's input scale (FP8: s_in, a divisor; NVFP4: its global scale) and the norm's eps
  uint32_t kind;      // the consumer's KIND_BF16 (KIND_FP8 / KIND_NVFP4: paths this package does not take)
  uint32_t res_bf16;  // the residual (rin / res_in) is bf16
  uint32_t x_bf16;    // xs holds bf16(x)
  uint32_t aux_ctr;   // (aux) the counter the finishers' aux rows release (+1 each)
  uint32_t b_stride;  // OpDesc::shared_b_stride
  uint32_t mtiles;    // the producer's M tiles
  uint32_t outs;      // the finishers also store the producer's bf16 outputs (the consumers read them)
  uint32_t erstd;     // the rstd in the consumer's epilogue: B = quant((x * (1 + w)) * c_t), no exchange wait
  uint64_t ssq_prev;  // (erstd) the previous boundary's slots (c_t's rstd), same layout; 0: the first boundary
  uint64_t cs;        // (erstd) c_t published [2][NT] fp32 (tile 0's finisher; sentinel ~0u), the consumer's
                      // f_t = rstd_t / c_t
  uint64_t baw;       // (xn) the consumer layer's in_proj_ba rows [96][K] bf16: the finishers store their tile's b / a
                      // partial dot products [2][mtiles][96][NT] fp32 (sentinel ~0u) at xn instead of bf16(x_n)
};
static_assert(offsetof(SharedBExt, erstd) == 100 && offsetof(SharedBExt, ssq_prev) == 104 &&
                  offsetof(SharedBExt, cs) == 112,
              "SharedBExt's erstd fields (schedule.py SHARED_B_EXT)");
static_assert(sizeof(SharedBExt) == 128 && offsetof(SharedBExt, baw) == 120,
              "SharedBExt layout (schedule.py SHARED_B_EXT)");
static_assert(offsetof(GdnDesc, ba_w) == offsetof(GdnTDesc, ba), "in_proj_ba's rows at one place in both GDN cores' descs");

struct MKArgs {
  const Instr* instrs;
  const uint32_t* sm_off;
  const OpDesc* descs;
  uint32_t* counters;
  const uint32_t* ctr_count;
  uint32_t* epoch;  // [0] epoch, [1] CTAs that have read it (all launches)
  uint64_t* trace;  // per CTA and role its current instruction {epoch, ii} (mk_progress): null in this package
  const int32_t* rec;  // the step record (its token count T)
  uint64_t* strace; // per CTA and instruction 4 global-timer stamps (mk_stamp): null in this package
  uint32_t rec_param;  // 1: the record's header is the launch's RecParam; 0: its words at rec
  const dk::DkArgs* draft;   // the draft's arguments (its instruction lists are these; null: no draft in this launch)
  const uint32_t* sm_draft;  // [nsm]: each CTA's first draft instruction (null: none in the lists)
  uint32_t draft_sampler, draft_aux;  // the step's counters the draft waits for: the spec sampler's last phase (its
                                      // tokens), the last aux layer's residual (the draft's inputs)
};

struct Smem {
  uint8_t* stages;
  uint8_t* xb;   // slot s's B operand at xb + s * B_SLOT_BYTES
  uint8_t* sfb;  // its NVFP4 scales at sfb + s * B_SLOT_BYTES
  uint8_t* stg;  // [STG_BYTES]
  uint64_t* full;
  uint64_t* empty;
  uint64_t* dfull;
  uint64_t* dempty;
  uint64_t* xfull;
  uint64_t* xempty;
  uint64_t* stg_full;
  uint64_t* stg2_full;  // in_proj_qkv's in_proj_ba rows (BA) in the staging buffer at BA_OFF
  uint64_t* pfull;      // [2] AUX_SPLIT: B slot s's token rows from the other CTA (its st.async bytes, this CTA's
                        // expect)
  uint64_t* pfree;      // [2] AUX_SPLIT: the other CTA's B slot s free for this CTA's rows (its relaxed arrival)
  float* red;   // [RED_FLOATS] prologue reductions
  float* xchg;  // [2][64][NT] epilogue exchange (EPI_SILU; shared B: each warp's 1 KB, its residual rows, then its B
                // words), then (shared B) SHARED_B_SM_BYTES at shared_b_sm(sm): no member of its own, as a 64-bit
                // field would be held by every role
  float* ba;    // [2][NT] the GDN core's b / a of this SM's item (in_proj_qkv's prologue, OpDesc::xn), in stg
  uint32_t tmem;
};
constexpr int RED_FLOATS = (NPRO / 32) * 2 * NT + 2 * (NPRO / 32);  // [warp][2 NT] (b / a sums), [2][warp] (norms)

// MKArgs::trace (null in this package): role r of this CTA starts instruction ii, {epoch, ii} into its trace word
__device__ __forceinline__ void mk_progress(uint64_t* trace, int role, uint32_t ii, uint32_t epoch) {
  if (trace) *reinterpret_cast<volatile uint64_t*>(trace + blockIdx.x * 16 + role) = ((uint64_t)epoch << 32) | ii;
}
// MKArgs::strace (null in this package): the global timer at event `slot` of this CTA's instruction i (relative): 0 the
// producer starts it, 1 the prologue starts it, 2 the epilogue has released it, 3 its prologue count released
constexpr uint32_t MK_STAMP_MAXI = 1024;
__device__ __forceinline__ void mk_stamp(uint64_t* st, uint32_t slot, uint32_t i) {
  if (st != nullptr && i < MK_STAMP_MAXI) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    st[((size_t)blockIdx.x * MK_STAMP_MAXI + i) * 4 + slot] = t;
  }
}
// weight bytes per row / 128-byte stages
__device__ __forceinline__ uint32_t stages_per_mtile(const OpDesc& d) {
  return (d.kind == KIND_NVFP4 ? d.K / 2 : d.kind == KIND_FP8 ? d.K : 2 * d.K) / 128;
}
// first weight row of M tile mt's half h. paired: an M tile is 64 gate rows (h = 0) and the 64 up rows (h = 1) of the
// same outputs
__device__ __forceinline__ uint32_t tile_row0(const OpDesc& d, uint32_t mt, int h) {
  return d.paired ? (h ? d.n_out : 0u) + mt * 64 : mt * 128;
}
// the weight-scale atoms of M tile mt's K chunk kc (4 k-steps, 2 KB): vLLM's swizzled layout
// ([rows/128][K/64][32][4][4] bytes), or for paired ops its copy in the paired tiles' order (paired tiles as rows)
__device__ __forceinline__ const uint8_t* sf_atoms(const OpDesc& d, uint32_t mt, uint32_t kc) {
  return reinterpret_cast<const uint8_t*>(d.sf) + ((size_t)mt * (d.K / 64) + kc * 4) * 512;
}
// the accumulator copy used by the launch with this epoch
__device__ __forceinline__ uint8_t* acc_copy(const OpDesc& d, uint32_t epoch) {
  return reinterpret_cast<uint8_t*>(d.acc + (epoch & 1) * d.acc_bytes);
}
// ACC_CNT: the op's bf16 output used by the launch with this epoch (val: two copies, p[2] bytes apart)
__device__ __forceinline__ uint8_t* cnt_out(const OpDesc& d, uint32_t epoch) {
  return reinterpret_cast<uint8_t*>(d.p[0] + (epoch & 1) * d.p[2]);
}
// a stage's place in the op's K-slab-major order (OpDesc::slab): its M tile, its K chunk (absolute), the stages left in
// its M tile's run of the slab (this one included)
struct StagePos {
  uint32_t mt, kc, left;
};
__device__ __forceinline__ StagePos stage_pos(const OpDesc& d, uint32_t st) {
  const uint32_t S = d.slab, per = d.mtiles * S, j = st / per, l = st - j * per, mt = l / S, c = l - mt * S;
  return StagePos{mt, j * S + c, S - c};
}
// The instruction's K chunks kc0, kc0 + 1, ... (mod spm, the stages per M tile: one slab, OpDesc::slab): nck of them,
// the first n1 before the wrap; chunk l of its B buffer is the op's chunk abs(l)
struct ChunkRange {
  uint32_t spm, nck, kc0, n1;
  __device__ __forceinline__ uint32_t abs(uint32_t l) const { return l < n1 ? kc0 + l : l - n1; }
};
__device__ __forceinline__ ChunkRange chunk_range(const Instr& in, const OpDesc& d) {
  const uint32_t spm = d.slab, nck = min(in.r1 - in.r0, spm), kc0 = in.r0 % spm;
  return ChunkRange{spm, nck, kc0, min(nck, spm - kc0)};
}
// the op's chunk kc's slot in the instruction's B buffer (>= nck: not one of its chunks)
__device__ __forceinline__ uint32_t chunk_slot(uint32_t kc, const ChunkRange& cr) {
  return kc >= cr.kc0 ? kc - cr.kc0 : kc + cr.spm - cr.kc0;
}
// the instruction's TMEM park slots (Instr::aux bits 8..15, schedule.py mark_park), at most its stages
__device__ __forceinline__ uint32_t park_max(const Instr& in, const OpDesc& d) {
  const uint32_t pslots = (in.aux >> AUX_PARK_SHIFT) & AUX_PARK_MASK;
  // (a bf16 stage's A is 128 rows x 128 bytes, as FP8's: its park slot layout)
  return min(min(pslots, d.kind == KIND_NVFP4 ? MAX_PARK_NV : MAX_PARK_F8), in.r1 - in.r0);
}
// AUX_PERM: the parked block, P stages from stage L (ring positions 0..P-1 of the instruction load stages L..L+P-1,
// positions P..P+L-1 stages 0..L-1, the rest in order); P = 0: no permutation (the parked stages are the first ones)
struct PermPL {
  uint32_t P, L;
};
__device__ __forceinline__ PermPL perm_pl(const Instr& in, const OpDesc& d) {
  const uint32_t nst = in.r1 - in.r0, pmax = park_max(in, d);
  if (!(in.aux & AUX_PERM) || pmax == 0 || pmax >= nst) return PermPL{0u, 0u};
  return PermPL{pmax, min((uint32_t)NSTAGE, nst - pmax)};
}
// the instruction's stage (relative) at its ring position q
__device__ __forceinline__ uint32_t perm_stage(uint32_t q, const PermPL& pl) {
  return q < pl.P ? pl.L + q : q < pl.P + pl.L ? q - pl.P : q;
}

constexpr uint32_t NACC = 2u;  // TMEM accumulator slots (NT columns each, from column 0)

// ------------------------------------------------------------------------------------------------ producer
// Warp W_PROD, warp-uniform, one elected lane issues. cc: ring stage position (slot, phase). Until this instruction's
// prologue has staged its inputs (stg_count >= stg_target), at most `pace` stages in flight.
__device__ __forceinline__ void produce_gemv(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& cc,
                                             const volatile uint32_t& stg_count, uint32_t stg_target,
                                             volatile uint32_t* prod_n) {
  const bool nvf4 = d.kind == KIND_NVFP4;
  const uint32_t bytes = A_BYTES + (nvf4 ? SFA_BYTES : 0);
  const uint32_t pace =
      ((in.aux >> AUX_PACE_SHIFT) & AUX_PACE_MASK) ? (in.aux >> AUX_PACE_SHIFT) & AUX_PACE_MASK : PACE;
  const void* tmap = reinterpret_cast<const void*>(d.tmap);
  const PermPL pl = perm_pl(in, d);
  // (the stage's M tile and chunk carried from stage to stage: stage_pos's two divisions only where AUX_PERM's order
  // jumps, at q = 0, P, P + L)
  const uint32_t S = d.slab;
  uint32_t mt = 0, c = 0, jS = 0;
  // (the load policy made once, the pacing's staged state remembered once met: no shared load per stage after it)
  const uint64_t pol_ef = l2_evict_first();
  bool inputs_staged = false;
  // (the barriers' and the stages' shared::cta addresses made once: no window-base rebuild per stage)
  uint32_t full_a = smem_u32(sm.full), empty_a = smem_u32(sm.empty), stages_a = smem_u32(sm.stages);
  // (opaque: ptxas would rebuild them from the generic pointers at each use rather than hold three registers)
  asm volatile("" : "+r"(full_a), "+r"(empty_a), "+r"(stages_a));
  for (uint32_t q = 0; q < in.r1 - in.r0; q++) {
    if (q == 0 || q == pl.P || q == pl.P + pl.L) {
      const uint32_t st = in.r0 + perm_stage(q, pl), per = d.mtiles * S, j = st / per, l = st - j * per;
      mt = l / S;
      c = l - mt * S;
      jS = j * S;
    } else if (++c == S) {
      c = 0;
      if (++mt == d.mtiles) mt = 0, jS += S;
    }
    const uint32_t kc = jS + c;
    if (cc.n >= NSTAGE) {
      mbar_wait_a(empty_a + 8u * cc.slot, cc.phase ^ 1);
    }
    if (!inputs_staged) {
      if (stg_count >= stg_target) {
        inputs_staged = true;
      } else if (cc.n >= pace) {
        const uint32_t cw = cc.n - pace;
        while (stg_count < stg_target && !mbar_test_a(full_a + 8u * (cw % NSTAGE), (cw / NSTAGE) & 1)) {
        }
      }
    }
    if (!nvf4) {  // (BF16 / FP8: the stage's A alone)
      if (elect_one()) {
        const uint32_t bar = full_a + 8u * cc.slot, dst = stages_a + cc.slot * (uint32_t)STAGE_BYTES;
        mbar_expect_tx_a(bar, bytes);
        if (d.paired) {
          tma_load_2d_pol_a(dst, tmap, kc * 128, tile_row0(d, mt, 0), bar, pol_ef);
          tma_load_2d_pol_a(dst + A_BYTES / 2, tmap, kc * 128, tile_row0(d, mt, 1), bar, pol_ef);
        } else {
          tma_load_2d_pol_a(dst, tmap, kc * 128, tile_row0(d, mt, 0), bar, pol_ef);
        }
      }
    } else
    if (elect_one()) {
      uint64_t* bar = &sm.full[cc.slot];
      mbar_expect_tx(bar, bytes);
      uint8_t* dst = sm.stages + cc.slot * STAGE_BYTES;
      const uint64_t pol = pol_ef;
      if (d.paired) {
        tma_load_2d_pol(dst, tmap, kc * 128, tile_row0(d, mt, 0), bar, pol);
        tma_load_2d_pol(dst + A_BYTES / 2, tmap, kc * 128, tile_row0(d, mt, 1), bar, pol);
      } else {
        tma_load_2d_pol(dst, tmap, kc * 128, tile_row0(d, mt, 0), bar, pol);
      }
      if (nvf4) bulk_g2s_pol(dst + A_BYTES, sf_atoms(d, mt, kc), SFA_BYTES, bar, pol);
    }
    __syncwarp();
    cc.next(NSTAGE);
    *prod_n = cc.n;  // ring positions issued (the MMA warp parks only issued stages)
  }
}

// ------------------------------------------------------------------------------------------------ MMA issuer
// Warp W_MMA: one elected lane issues the whole instruction in one run (the segment and stage loops, their waits, every
// tcgen05 op); the other lanes advance cc (ring stages) and seg (M-tile segments: the two accumulator buffers) past it.
// xb / sfb: its B operand and NVFP4 scales; np, t_park, pk0: its parked stages; bbar: B's barrier (phase bph)
template <int KIND>
__device__ __forceinline__ void mma_gemv2(const Instr& in, const OpDesc& d, const Smem& sm, const uint8_t* xb, const uint8_t* sfb,
                                          RingPos& cc, RingPos& seg, uint32_t np = 0, uint32_t t_park = 0, uint32_t pk0 = 0,
                                          uint64_t* bbar = nullptr, uint32_t bph = 0) {
  // (spm: the slab's chunks, kc0 the first stage's chunk within the slab: B slot l of the instruction's stage i is
  // i mod spm, its segments end at the slab's end of each M tile)
  const ChunkRange cr0 = chunk_range(in, d);
  const uint32_t spm = cr0.spm, kc0 = cr0.kc0, nst = in.r1 - in.r0;
  constexpr uint32_t PCOLS = KIND == KIND_NVFP4 ? PARK_NV_COLS : PARK_F8_COLS;
  if (elect_one()) {
    // the descriptors carried from stage to stage (B chunk slot, park column, ring slot, scale slot), not rebuilt from
    // the stage index: they stay in uniform registers (no per-stage R2UR; the catch-up after B-ready is the issuer's
    // instructions, not the tensor pipe's)
    const uint32_t a_lo0 = (smem_u32(sm.stages) >> 4) & 0x3FFF, a_hi = (1024 >> 4) | (1u << 14) | (2u << 29);
    const uint32_t b_lo0 = ((smem_u32(xb) >> 4) & 0x3FFF) | ((128u >> 4) << 16), b_hi = (1024 >> 4) | (1u << 14);
    const uint32_t s_hi = (128 >> 4) | (1u << 14), sb_lo0 = (smem_u32(sfb) >> 4) & 0x3FFF;
    const uint32_t sa_lo0 = (smem_u32(sm.stages + A_BYTES) >> 4) & 0x3FFF;
    constexpr uint32_t B_KSTEP = 256 >> 4, SB16 = STAGE_BYTES >> 4;
    RingPos r = cc, sg = seg;
    r.advance(np, NSTAGE);  // (the parked stages' ring positions)
    uint32_t l = 0;         // the stage's chunk slot in B (chunk_slot of its K chunk: stage 0 is chunk kc0)
    uint32_t sc = 0;        // the stage's NVFP4 scale slot ((instruction stage) & 1)
    uint32_t kc = kc0;      // the segment's first K chunk
    for (uint32_t k = 0; k < nst;) {  // k: the segment's first stage's index in the instruction
      const uint32_t n = min(nst - k, spm - kc);  // stages of this M-tile segment
      if (sg.n >= NACC) mbar_wait(&sm.dempty[sg.slot], sg.phase ^ 1);
      tc_fence_after();
      const uint32_t t_d = sm.tmem + sg.slot * NT;
      for (uint32_t i = 0; i < n;) {  // segment stage i = the instruction's stage k + i
        if (k + i >= pk0 && k + i < pk0 + np) {
          // parked stages pk0..pk0 + np - 1 (no waits): the instruction's first np ring positions, already released
          // (pk0 = 0, or AUX_PERM's L: perm_stage), in TMEM at t_park + p * PCOLS (A, then an NVFP4 stage's weight
          // scales); B's wait right before the first MMA
          if (bbar) {
            mbar_wait(bbar, bph);
            bbar = nullptr;
            tc_fence_after();
          }
          const uint32_t m = min(n - i, pk0 + np - k - i);
          uint32_t p = k + i - pk0, ta = t_park + p * PCOLS;
          for (uint32_t j = 0; j < m; j++) {
            const uint32_t b_lo = b_lo0 + l * (CHUNK_B >> 4);
            uint32_t ss = 0, sfa = 0;
            if (KIND == KIND_NVFP4) {
              ss = sm.tmem + T_SC + sc * SC_COLS;
              const uint32_t sb = sb_lo0 + l * (SFB_CHUNK >> 4);
              // the B scales: k-steps 0, 1 at columns 16, 18, k-steps 2, 3 at columns 20, 22
              tc_cp_32x128b_x4(ss + 16, ((uint64_t)s_hi << 32) | sb);
              tc_cp_32x128b_x4(ss + 20, ((uint64_t)s_hi << 32) | (sb + (SFB_HALF >> 4)));
              sfa = ta + 32 - (p == PARK_NV9 ? PARK9_SF_BACK : 0u);  // (the extra slot's weight scales: park_nv_sf)
            }
#pragma unroll
            for (uint32_t q = 0; q < 4; q++) {
              const uint64_t bd = ((uint64_t)b_hi << 32) | (b_lo + B_KSTEP * q);
              const uint32_t acc = (i + j > 0 || q > 0) ? 1u : 0u;
              if (KIND == KIND_NVFP4) tc_mma_nvf4_ts(t_d, ta + 8 * q, bd, IDESC_NVF4, acc, sfa + 4 * q, ss + 16 + 2 * q);
              else if (KIND == KIND_FP8) tc_mma_f8_ts(t_d, ta + 8 * q, bd, IDESC_F8, acc);
              else tc_mma_bf16_ts(t_d, ta + 8 * q, bd, IDESC_BF16, acc);
            }
            p++;
            ta += PCOLS;
            if (++l == spm) l = 0;
            sc ^= 1u;
          }
          i += m;
          continue;
        }
        // ring stages, up to the parked block if it starts later in the segment: a wait for the next one, then it and
        // every following one that has landed already (a non-blocking test per stage)
        const uint32_t lim = (np && k + i < pk0) ? min(n, pk0 - k) : n;
        mbar_wait(&sm.full[r.slot], r.phase);
        if (bbar) {
          mbar_wait(bbar, bph);
          bbar = nullptr;
        }
        do {
          tc_fence_after();
          const uint32_t a_lo = a_lo0 + r.slot * SB16, b_lo = b_lo0 + l * (CHUNK_B >> 4);
          uint32_t ss = 0;
          // NVFP4: the stage's scales into its scale slot right before its MMAs (tcgen05.cp of the ring slot's
          // weight-scale atoms and the chunk's B scales), executed in order with the MMAs: the slots need no barrier
          if (KIND == KIND_NVFP4) {
            ss = sm.tmem + T_SC + sc * SC_COLS;
            const uint32_t sb = sb_lo0 + l * (SFB_CHUNK >> 4), sa = sa_lo0 + r.slot * SB16;
            tc_cp_32x128b_x4(ss + 16, ((uint64_t)s_hi << 32) | sb);
            tc_cp_32x128b_x4(ss + 20, ((uint64_t)s_hi << 32) | (sb + (SFB_HALF >> 4)));
#pragma unroll
            for (uint32_t q = 0; q < 4; q++) tc_cp_32x128b_x4(ss + 4 * q, ((uint64_t)s_hi << 32) | (sa + 32 * q));
          }
#pragma unroll
          for (uint32_t q = 0; q < 4; q++) {
            const uint64_t ad = ((uint64_t)a_hi << 32) | (a_lo + 2 * q);
            const uint64_t bd = ((uint64_t)b_hi << 32) | (b_lo + B_KSTEP * q);
            const uint32_t acc = (i > 0 || q > 0) ? 1u : 0u;
            if (KIND == KIND_NVFP4) tc_mma_nvf4(t_d, ad, bd, IDESC_NVF4, acc, ss + 4 * q, ss + 16 + 2 * q);
            else if (KIND == KIND_FP8) tc_mma_f8(t_d, ad, bd, IDESC_F8, acc);
            else tc_mma_bf16(t_d, ad, bd, IDESC_BF16, acc);
          }
          tc_commit(&sm.empty[r.slot]);
          r.next(NSTAGE);
          if (++l == spm) l = 0;
          sc ^= 1u;
          i++;
        } while (i < lim && mbar_test(&sm.full[r.slot], r.phase));
      }
      tc_commit(&sm.dfull[sg.slot]);
      sg.next(NACC);
      k += n;
      kc = 0;
    }
  }
  __syncwarp();
  // every lane: the instruction's ring positions (np parked + nst - np in the ring) and its M-tile segments
  const uint32_t n0 = min(nst, spm - kc0);
  const uint32_t nseg = nst == 0 ? 0u : 1u + (nst - n0 + spm - 1) / spm;
  cc.advance(nst, NSTAGE);
  seg.advance(nseg, NACC);
}

// the MMA warp's last commits on a ring of `size` barriers waited: the `size` positions before the next one (slot,
// phase), position next - j at slot (slot - j) mod size, its phase flipped when j > slot (a position before the first:
// the barrier's phase before its first, complete, so the wait passes)
__device__ __forceinline__ void ring_drain(uint32_t bars_a, uint32_t size, uint32_t slot, uint32_t phase) {
  for (uint32_t j = 1; j <= size; j++)
    mbar_wait_a(bars_a + 8u * ((slot + size - j) % size), phase ^ (j > slot ? 1u : 0u));
}
// the ring's empty barriers and the B buffers' xempty ones (shared::cta addresses, slot | phase << 8 each; not inlined)
__device__ __noinline__ void mma_drain(uint32_t empty_a, uint32_t xempty_a, uint32_t cc_sp, uint32_t xg_sp) {
  ring_drain(empty_a, NSTAGE, cc_sp & 0xffu, cc_sp >> 8);
  ring_drain(xempty_a, NB, xg_sp & 0xffu, xg_sp >> 8);
}

// ------------------------------------------------------------------------------------------------ epilogue
// EPI_SILU, a paired gate||up M tile's sums v (rows 0..63 gate, 64..127 up): act = bf16(silu(bf16(alpha g)) *
// bf16(alpha u)), vLLM's SiluAndMul of the bf16 GEMM output, into down's B layout by absolute chunk (a tile's 64
// outputs: chunk mt, group o / 8, value o % 8); each half computes half the token rows (xchg: 64 x NT floats)
__device__ __forceinline__ void silu_bf16_tile(const OpDesc& d, uint32_t mt, int ept, int lane, const float (&v)[NT],
                                               float* xchg, uint32_t T, uint32_t epoch) {
  constexpr int NH = NT / 2;
  static_assert(NT % 2 == 0, "token rows in pairs");
  const bool is_up = ept >= 64;
  const int o = ept & 63;
#pragma unroll
  for (int k = 0; k < NH; k++) xchg[o * NT + 2 * k + (is_up ? 0 : 1)] = is_up ? v[2 * k] : v[2 * k + 1];
  named_sync_u(BAR_EPI, 128);
  const float alpha = d.alpha;
  // (p[3] nonzero: two launch-parity copies this far apart, down re-polling its parity's at the sentinel; schedule.py
  // sets 0)
  uint8_t* const act0 = reinterpret_cast<uint8_t*>(d.p[0]) + (size_t)mt * CHUNK_B + (o / 8) * 128 + (o % 8) * 2;
  const uint64_t astride = d.p[3];
  uint8_t* const act = act0 + (epoch & 1) * astride;
#pragma unroll
  for (int k = 0; k < NH; k++) {  // row t = 2 k + is_up
    const int t = 2 * k + (is_up ? 1 : 0);
    const float own_sum = is_up ? v[2 * k + 1] : v[2 * k], other_sum = xchg[o * NT + 2 * k + (is_up ? 1 : 0)];
    const float g = bf16_round(alpha * (is_up ? other_sum : own_sum)),
                u = bf16_round(alpha * (is_up ? own_sum : other_sum));
#ifdef MK_NEG_ACT  // a numerical negative control (-DMK_NEG_ACT=n in PEAR_DEPLOY_EXTRA_CUDA_FLAGS): every MLP
                   // activation scaled by 1 + 2^-n, deterministically, in the same dtypes
    const uint16_t hv = __bfloat16_as_ushort(__float2bfloat16(silu_f32(g) * u * (1.0f + exp2f(-(float)MK_NEG_ACT))));
#else
    const uint16_t hv = __bfloat16_as_ushort(__float2bfloat16(silu_f32(g) * u));
#endif
    // (pairs of outputs in one 4-byte store: the even lane takes its neighbour's value)
    const uint32_t nb = __shfl_xor_sync(0xffffffffu, (uint32_t)hv, 1);
    if (!(lane & 1) && t < (int)T) *reinterpret_cast<uint32_t*>(act + t * 16) = (uint32_t)hv | (nb << 16);
  }
  if (astride && !(lane & 1)) {  // the other parity's copy back to the sentinel, every row (the next launch's T varies)
    uint8_t* const oth = act0 + ((epoch + 1) & 1) * astride;
#pragma unroll
    for (int k = 0; k < NH; k++) *reinterpret_cast<uint32_t*>(oth + (2 * k + (is_up ? 1 : 0)) * 16) = 0xffffffffu;
  }
}

// rsqrtf(s / 5120.f + eps) bit for bit (every s, at eps 1e-6 and 1e-5) as a short chain: div.rn's fast path as three
// FMAs with its reciprocal 0x394ccccd (no FCHK / slow path: it differs only for denormal s, rounding the same, and s =
// +inf, selected), then rsqrt.approx.ftz unscaled (argument >= eps: schedule.py asserts it)
__device__ __forceinline__ float rms_rsqrt5120(float s, float eps) {
  const float r1 = __uint_as_float(0x394ccccdu);
  const float q0 = __fmaf_rn(r1, s, 0.f), e = __fmaf_rn(q0, -5120.f, s), q = __fmaf_rn(r1, e, q0);
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(q + eps));
  return s == __uint_as_float(0x7f800000u) ? 0.f : y;
}
constexpr uint32_t FIN_MAXP = 8;  // (FIN) parts per M tile (schedule.py)
constexpr uint32_t FIN_W = 3;     // (FIN) 16-byte words per part and row: {3 rows' sums, tag} (rows 3k .. 3k + 2)
// Instr::aux bit 4 (schedule.py AUX_SHARED_B) on a FIN op's instructions: its finishers build the norm consumer's B
// (the flag in the instruction word, which the epilogue holds anyway, not the descriptor's shared_b_cons: no register
// for it)
constexpr uint32_t AUX_SHARED_B = 1u << 4;
// (shared B) past the epilogue exchange: [4][128] each epilogue warp's SharedBExt copy, [4][NT] fp32 the tile's sums of
// squares per warp, [4][32] bf16 each warp's channels of the norm weight, the aux count
constexpr int SHARED_B_SM_BYTES = 4 * 128 + 4 * NT * 4 + 4 * 64 + 16 + 4 * NT * 4;
constexpr int SHARED_B_CT_OFF = 4 * 128 + 4 * NT * 4 + 4 * 64 + 16;  // (erstd) each warp's c_t of the NT rows
constexpr int SHARED_B_SSQ_OFF = 512;  // (shared B) the tile's sums of squares [4][NT], past the SharedBExt copies
constexpr int SHARED_B_NW_OFF = 640;   // (shared B) each warp's channels of the norm weight [4][32]
// (shared B) the aux finishers' rows written (+1 each, st.release.cta by the epilogue): warp 6 lane 1 releases them to
// the draft's counter (SharedBExt::aux_ctr), off the epilogue warps (a gpu-scope release waits behind the SM's TMA
// loads)
constexpr int SHARED_B_DRAFT_ROWS_OFF = 4 * 128 + 4 * NT * 4 + 4 * 64;
// Instr::aux bit 5 (schedule.py AUX_DRAFT_ROWS): a finisher instruction of an aux layer's down (its residual rows for
// the draft)
constexpr uint32_t AUX_DRAFT_ROWS = 1u << 5;
__device__ __forceinline__ uint8_t* shared_b_sm(const Smem& sm) { return reinterpret_cast<uint8_t*>(sm.xchg + 2 * 64 * NT); }
constexpr int FIN_PC = 4;  // (FIN) the later parts' words a first part has in flight at once (this many parts' NT rows)
__device__ __forceinline__ void st_relaxed_v4(uint32_t* p, uint4 v) {
  asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// (shared B) 8 values per lane reduced over the warp, lane l ending with the sum for row l >> 2 (one order, in 9
// shuffles where 8 butterflies take 40): halves exchanged at xor 16 (rows 4 b4 + i), 8 (rows 4 b4 + 2 b3 + j), 4 (row 4
// b4 + 2 b3 + b2), each lane's own value first, then a butterfly over xor 2, 1
__device__ __forceinline__ float rs_sum8(const float (&v)[NT], int lane) {
  static_assert(NT == 8, "rs_sum8");
  const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
  float u[4], w[2];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float keep = b4 ? v[i + 4] : v[i], send = b4 ? v[i] : v[i + 4];
    u[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
  }
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const float keep = b3 ? u[j + 2] : u[j], send = b3 ? u[j] : u[j + 2];
    w[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  float z;
  {
    const float keep = b2 ? w[1] : w[0], send = b2 ? w[0] : w[1];
    z = keep + __shfl_xor_sync(0xffffffffu, send, 4);
  }
  z += __shfl_xor_sync(0xffffffffu, z, 2);
  z += __shfl_xor_sync(0xffffffffu, z, 1);
  return z;
}
// (shared B) the maximum of 8 values per lane over each 16-lane half (an NVFP4 block), lane l ending with the maximum
// for row (l >> 1) & 7: halves exchanged at xor 8 (rows 4 b3 + i), 4, 2, then xor 1
__device__ __forceinline__ float rs_max8(const float (&v)[NT], int lane) {
  const bool b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
  float u[4], w[2];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float keep = b3 ? v[i + 4] : v[i], send = b3 ? v[i] : v[i + 4];
    u[i] = fmaxf(keep, __shfl_xor_sync(0xffffffffu, send, 8));
  }
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const float keep = b2 ? u[j + 2] : u[j], send = b2 ? u[j] : u[j + 2];
    w[j] = fmaxf(keep, __shfl_xor_sync(0xffffffffu, send, 4));
  }
  float z;
  {
    const float keep = b1 ? w[1] : w[0], send = b1 ? w[0] : w[1];
    z = fmaxf(keep, __shfl_xor_sync(0xffffffffu, send, 2));
  }
  return fmaxf(z, __shfl_xor_sync(0xffffffffu, z, 1));
}
// (shared B) c_t = 2^(floor(log2 rstd_prev_t) - 1): the previous boundary's rstd of row t (its 40 slots in the
// exchange's order) as a power of 2 (bits & 0xff800000) times the headroom 0.5 for the residual's growth (0.5 at the
// first boundary), into ct; tile 0's finisher publishes it (cs) and resets the next parity's copy
__device__ __forceinline__ void shared_b_centre(const SharedBExt& e, uint32_t mt, int q, uint32_t ct, int lane, uint32_t epoch,
                                                uint32_t T) {
  float c = 0.5f;
  if (e.ssq_prev) {
    const uint32_t* const sp = reinterpret_cast<const uint32_t*>(e.ssq_prev) + (size_t)(epoch & 1u) * e.mtiles * NT;
    const uint32_t ta = (uint32_t)lane, tb = (uint32_t)lane + 32;
    const bool ha = ta < e.mtiles, hb = tb < e.mtiles, h4 = T > 4;  // (rows 4-7's words are written only at T > 4)
    uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0, b0 = a0, b1 = a0;
    bool is_missing;
    do {  // (complete an op ago: the re-poll is for the rule, not the time)
      if (ha) {
        a0 = ld_relaxed_v4(sp + ta * NT);
        if (h4) a1 = ld_relaxed_v4(sp + ta * NT + 4);
      }
      if (hb) {
        b0 = ld_relaxed_v4(sp + tb * NT);
        if (h4) b1 = ld_relaxed_v4(sp + tb * NT + 4);
      }
      is_missing = (ha && (a0.w == ~0u || (h4 && a1.w == ~0u))) || (hb && (b0.w == ~0u || (h4 && b1.w == ~0u)));
    } while (__any_sync(0xffffffffu, is_missing));
    const uint32_t av[NT] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const uint32_t bv[NT] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float rs[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
      float v = ha ? __uint_as_float(av[t]) : 0.f;
      if (hb) v = v + __uint_as_float(bv[t]);
      rs[t] = v;
    }
    const float r = rms_rsqrt5120(rs_sum8(rs, lane), e.eps);  // (row lane >> 2's)
    c = __uint_as_float(__float_as_uint(r) & 0xff800000u) * 0.5f;
  }
  if (!(lane & 3)) sts_f32(ct + (lane >> 2) * 4, c);
  if (mt == 0 && q == 0 && !(lane & 3)) {
    uint32_t* const cs = reinterpret_cast<uint32_t*>(e.cs);
    st_relaxed_b32(cs + (epoch & 1u) * NT + (lane >> 2), __float_as_uint(c));
    cs[((epoch & 1u) ^ 1u) * NT + (lane >> 2)] = ~0u;
  }
  __syncwarp();
}
// (shared B) the next parity's copies of a finisher's outputs back to their sentinels (its tile mt, this warp's 32
// channels, all NT rows): at its instruction's start, off the tile's path (the previous launch read them; this one
// reads the other parity)
__device__ __forceinline__ void shared_b_reset(const SharedBExt& e, uint32_t mt, int q, int lane, uint32_t epoch) {
  constexpr uint32_t K = K_NORM;
  const uint32_t c = mt * 128 + q * 32 + lane, npar = (epoch & 1u) ^ 1u;
  uint8_t* const bn = reinterpret_cast<uint8_t*>(e.b) + (size_t)npar * e.b_stride;
  if (e.kind == KIND_BF16) {  // (this warp's 32 channels: chunk 2 mt + q / 2, groups 4 (q & 1) .. + 3, every row)
    *reinterpret_cast<uint4*>(bn + (size_t)(2 * mt + q / 2) * CHUNK_B + (4 * (q & 1) + lane / NT) * 128 + (lane % NT) * 16) =
        make_uint4(~0u, ~0u, ~0u, ~0u);
  } else if (e.kind == KIND_FP8) {
    if (lane < 2 * NT)
      *reinterpret_cast<uint4*>(bn + (size_t)mt * CHUNK_B + (2 * q + lane / NT) * 128 + (lane % NT) * 16) =
          make_uint4(~0u, ~0u, ~0u, ~0u);
  } else {
    const uint32_t kc = mt >> 1, h = mt & 1u;
    if (lane < NT)
      *reinterpret_cast<uint4*>(bn + (size_t)kc * CHUNK_B + (4 * h + q) * 128 + lane * 16) =
          make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
    if (lane < 2 * NT) {  // (lane: the block 8 h + 2 q + lane / NT's scale of row lane % NT)
      const uint32_t blk = h * 8 + 2 * q + lane / NT;
      bn[(size_t)(K / 256) * CHUNK_B + sfb_pos(kc, blk, lane % NT)] = 0xffu;
    }
  }
  if (e.xs) {
    if (e.x_bf16) {
      uint16_t* const xr = reinterpret_cast<uint16_t*>(e.xs) + (size_t)npar * NT * K + c;
#pragma unroll
      for (int t = 0; t < NT; t++) xr[t * K] = 0xffffu;
    } else {
      uint32_t* const xr = reinterpret_cast<uint32_t*>(e.xs) + (size_t)npar * NT * K + c;
#pragma unroll
      for (int t = 0; t < NT; t++) xr[t * K] = ~0u;
    }
  }
  if (e.xn && e.baw) {  // the tile's [96][NT] b / a partials, 192 16-byte words over the 4 warps
    uint4* const pr = reinterpret_cast<uint4*>(reinterpret_cast<uint32_t*>(e.xn) + ((size_t)npar * e.mtiles + mt) * BA_OUT * NT);
    const uint32_t ept = (uint32_t)q * 32 + lane;
    pr[ept] = make_uint4(~0u, ~0u, ~0u, ~0u);
    if (ept < BA_OUT * NT / 4 - 128) pr[128 + ept] = make_uint4(~0u, ~0u, ~0u, ~0u);
  } else if (e.xn) {
    uint16_t* const xr = reinterpret_cast<uint16_t*>(e.xn) + (size_t)npar * NT * K + c;
#pragma unroll
    for (int t = 0; t < NT; t++) xr[t * K] = 0xffffu;
  }
  if (q == 0 && lane < 2)
    *reinterpret_cast<uint4*>(reinterpret_cast<uint32_t*>(e.ssq) + ((size_t)npar * e.mtiles + mt) * NT + lane * 4) =
        make_uint4(~0u, ~0u, ~0u, ~0u);
}
// (shared B, rare) the residual values of the validated copies still at the sentinel: re-polled from L2 row by row and
// written over this lane's staged values (out of line: not in the tail's instruction stream; its name sorts after
// ss_phase's: ptxas lays the internal functions out by mangled name)
__device__ __noinline__ void xs_repoll(uint32_t wsm, const uint8_t* src, uint32_t rstride, uint32_t T, bool rb16, int lane) {
  for (uint32_t t = 0; t < T; t++) {
    const uint8_t* const p = src + (size_t)t * rstride;
    if (rb16) {
      const uint32_t* const pw = reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(p) & ~(uintptr_t)3);
      uint32_t h;
      do {
        const uint32_t w = ld_relaxed(pw);
        h = (reinterpret_cast<uintptr_t>(p) & 2) ? (w >> 16) : (w & 0xffffu);
      } while (h == 0xffffu);
      asm volatile("st.shared.u16 [%0], %1;" ::"r"(wsm + t * 128 + lane * 2), "h"((uint16_t)h) : "memory");
    } else {
      uint32_t w;
      do {
        w = ld_relaxed(reinterpret_cast<const uint32_t*>(p));
      } while (w == ~0u);
      sts_u32(wsm + t * 128 + lane * 4, w);
    }
  }
}
// in_proj_ba's partial dot products of a finisher's tile, after its ssq slot: P[o][t] = sum over the tile's
// 128 channels c of W[o][c] bf16(v_t[c]) (o < 96: b of the 48 heads, then a), by mma.sync m16n8k16 (A: W's rows,
// prefetched into L2 at the instruction's start; B: bf16(v)), into xn [par][mt][96][NT] (~0u never written: 0x7fffffff)
__device__ __forceinline__ void shared_b_ba_part(const SharedBExt& e, uint32_t xchg, uint32_t mt, int q, int lane, uint32_t par,
                                                 uint32_t T, const float (&v)[NT]) {
  constexpr uint32_t K = K_NORM;
  // bf16(v) of the 4 warps' channels into their 1 KB of the exchange buffer ([t][32] each: the B words are read before
  // the ssq barrier)
#pragma unroll
  for (int t = 0; t < NT; t++)
    sts_u16(xchg + q * 1024 + t * 64 + lane * 2, t < (int)T ? __bfloat16_as_ushort(__float2bfloat16(v[t])) : 0u);
  named_sync_u(BAR_EPI, 128);
  // warp q: the M tiles q, q + 4 over all 128 channels; per 32-channel group kq thread j = lane % 4 holds channels 8 j
  // .. 8 j + 7 (the MMAs' k slots {2j, 2j+1, 2j+8, 2j+9}: channels 8j .. 8j+3, then 8j+4 .. 8j+7, for A and B), every
  // load in flight at once
  const int g = lane >> 2, j = lane & 3;
  uint4 bf[4];
#pragma unroll
  for (int kq = 0; kq < 4; kq++) bf[kq] = lds_u4(xchg + kq * 1024 + g * 64 + j * 16);
  float* const pr = reinterpret_cast<float*>(e.xn) + ((size_t)par * e.mtiles + mt) * BA_OUT * NT;
  auto fix = [](float x) { return __float_as_uint(x) == ~0u ? __uint_as_float(0x7fffffffu) : x; };
  const bool has_second_tile = q + 4 < (int)(BA_OUT / 16);  // (warps 0, 1: a second M tile)
  uint4 a0[2][4], a1[2][4];  // W's rows 16 m + g, 16 m + g + 8 of the warp's M tiles, the 4 channel groups
#pragma unroll
  for (int u = 0; u < 2; u++) {
    const int m = q + 4 * u;
    const uint4* const w0 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(e.baw) +
                                                           (size_t)(16 * m + g) * K + mt * 128 + 8 * j);
#pragma unroll
    for (int kq = 0; kq < 4; kq++) {
      a0[u][kq] = u == 0 || has_second_tile ? __ldg(w0 + 4 * kq) : make_uint4(0u, 0u, 0u, 0u);
      a1[u][kq] = u == 0 || has_second_tile ? __ldg(w0 + K + 4 * kq) : make_uint4(0u, 0u, 0u, 0u);
    }
  }
#pragma unroll
  for (int u = 0; u < 2; u++) {
    if (u == 1 && !has_second_tile) break;
    const int m = q + 4 * u;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kq = 0; kq < 4; kq++) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0[u][kq].x), "r"(a1[u][kq].x), "r"(a0[u][kq].y), "r"(a1[u][kq].y), "r"(bf[kq].x), "r"(bf[kq].y));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0[u][kq].z), "r"(a1[u][kq].z), "r"(a0[u][kq].w), "r"(a1[u][kq].w), "r"(bf[kq].z), "r"(bf[kq].w));
    }
    *reinterpret_cast<float2*>(pr + (16 * m + g) * NT + 2 * j) = make_float2(fix(d[0]), fix(d[1]));
    *reinterpret_cast<float2*>(pr + (16 * m + g + 8) * NT + 2 * j) = make_float2(fix(d[2]), fix(d[3]));
  }
}
// (shared B, SharedBExt::erstd) the finisher's tail, the rstd left to the consumer's epilogue: x = bf16(o) + residual,
// v = (x * (1 + w)) * c_t, quantized as the consumer's prologue does; the B words and bf16(v) first (the consumer's
// path), then x's copies, the aux rows and the tile's ssq slot (read after the consumer's first accumulator)
template <int KIND>
__device__ __forceinline__ void shared_b_tail_kind(const Smem& sm, uint32_t mt, int q, int lane, uint32_t epoch, uint32_t T,
                                                   const uint32_t (&ob)[NT]) {
  const SharedBExt& e = *reinterpret_cast<const SharedBExt*>(shared_b_sm(sm) + q * 128);
  constexpr uint32_t K = K_NORM;
  const uint32_t c = mt * 128 + q * 32 + lane;  // this thread's channel
  const uint32_t par = epoch & 1u;
  const uint32_t wsm = smem_u32(sm.xchg) + q * 1024;  // the warp's buffer: [T][128 B] residual rows, then B words
  // (the consumer's variants at run time: the residual's type, x's copy (2 bf16(x), 1 fp32 x, 0 none), bf16(v))
  const bool RB16 = e.res_bf16 != 0, XN = e.xn != 0;
  const int XS = e.xs ? (e.x_bf16 ? 2 : 1) : 0;
  cp_async_wait_all();  // (this lane's residual copies)
  __syncwarp();
  uint32_t rb[NT];
  auto read_res = [&]() {
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (RB16) {
        uint32_t h;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(h) : "r"(wsm + t * 128 + lane * 2));
        rb[t] = h << 16;
      } else {
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(rb[t]) : "r"(wsm + t * 128 + lane * 4));
      }
    }
  };
  read_res();
  if (e.rin) {  // (validated copies: the values still at the sentinel re-polled, out of line)
    bool has_stale = false;
#pragma unroll
    for (int t = 0; t < NT; t++) has_stale |= t < (int)T && (RB16 ? (rb[t] >> 16) == 0xffffu : rb[t] == ~0u);
    if (__builtin_expect(has_stale, 0)) {
      const uint32_t bpv = RB16 ? 2u : 4u;
      xs_repoll(wsm, reinterpret_cast<const uint8_t*>(e.rin) + ((size_t)par * NT * K + c) * bpv, K * bpv, T, RB16, lane);
      read_res();
    }
  }
  float x[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) x[t] = t < (int)T ? __fadd_rn(__uint_as_float(ob[t] << 16), __uint_as_float(rb[t])) : 0.f;
  __syncwarp();  // (the residual rows read before the B words take the buffer)
  float v[NT];
  {
    const uint32_t ct = smem_u32(shared_b_sm(sm)) + SHARED_B_CT_OFF + q * NT * 4;
    uint32_t w1b;  // (this channel's norm weight, bf16, copied with the residual rows)
    asm volatile("ld.shared.u16 %0, [%1];"
                 : "=r"(w1b)
                 : "r"(smem_u32(shared_b_sm(sm)) + SHARED_B_NW_OFF + q * 64 + lane * 2));
    const float w1 = __uint_as_float(w1b << 16) + 1.0f;
#pragma unroll
    for (int t = 0; t < NT; t++) {
      float ctt;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ctt) : "r"(ct + t * 4));
      v[t] = __fmul_rn(__fmul_rn(x[t], w1), ctt);
    }
  }
  uint8_t* const bp = reinterpret_cast<uint8_t*>(e.b) + (size_t)par * e.b_stride;
  if (KIND == KIND_BF16) {
    // bf16(v) (0xFFFF, the sentinel, never: a conversion gives 0x7FFF), the warp's 4 groups x T rows of 16 bytes staged
    // in its buffer, then one 16-byte store a lane (group lane / NT, row lane % NT)
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const uint16_t hv = __bfloat16_as_ushort(__float2bfloat16(v[t]));
        sts_u16(wsm + (lane >> 3) * 128 + t * 16 + (lane & 7) * 2, hv == 0xffffu ? 0x7fffu : hv);
      }
    }
    __syncwarp();
    if (lane % NT < (int)T)
      *reinterpret_cast<uint4*>(bp + (size_t)(2 * mt + q / 2) * CHUNK_B + (4 * (q & 1) + lane / NT) * 128 + (lane % NT) * 16) =
          lds_u4(wsm + (lane / NT) * 128 + (lane % NT) * 16);
  } else if (KIND == KIND_FP8) {
    const float inv_s = div_full(1.0f, e.gs);
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const uint32_t cb = f32x2_to_e4m3x2(__fmul_rn(v[t], inv_s), 0.f) & 0xffu;
        sts_u8(wsm + (lane >> 4) * 128 + t * 16 + (lane & 15), cb == 0xffu ? 0x7fu : cb);
      }
    }
    __syncwarp();
    if (lane < 2 * (int)T) {
      const uint32_t g = (uint32_t)lane / T, t = (uint32_t)lane % T;
      *reinterpret_cast<uint4*>(bp + (size_t)mt * CHUNK_B + (2 * q + g) * 128 + t * 16) = lds_u4(wsm + g * 128 + t * 16);
    }
  } else {
    const float gs = e.gs, rgs = rcp_approx_ftz(gs), r6 = rcp_approx_ftz(6.0f);
    const uint32_t kc = mt >> 1, h = mt & 1u, blk = h * 8 + 2 * q + (lane >> 4);
    float vb[NT], ab[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
      vb[t] = bf16_round(v[t]);
      ab[t] = fabsf(vb[t]);
    }
    const int tr = (lane >> 1) & 7;
    const float am = rs_max8(ab, lane);
    const __nv_fp8_e4m3 qs(gs * (am * r6));
    const float sfq = float(qs);
    const float om = sfq != 0.f ? rcp_approx_ftz(sfq * rgs) : 0.f;
    if (!(lane & 1) && tr < (int)T) {
      const uint32_t sv = *reinterpret_cast<const uint8_t*>(&qs);
      bp[(size_t)(K / 256) * CHUNK_B + sfb_pos(kc, blk, tr)] = (uint8_t)(sv == 0xffu ? 0x7fu : sv);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const float vo = __fmul_rn(vb[t], __shfl_sync(0xffffffffu, om, (lane & 16) | (2 * t)));
        const float vn = __shfl_xor_sync(0xffffffffu, vo, 1);
        if (!(lane & 1)) {
          uint32_t cb = f32x2_to_e2m1x2(vo, vn) & 0xffu;
          cb ^= cb & ~((cb & 0x77u) + 0x77u) & 0x88u;  // (-0 as +0: the nibble 0x8 is the sentinel)
          sts_u8(wsm + t * 16 + (lane >> 1), cb);
        }
      }
    }
    __syncwarp();
    if (lane < (int)T)
      *reinterpret_cast<uint4*>(bp + (size_t)kc * CHUNK_B + (4 * h + q) * 128 + lane * 16) = lds_u4(wsm + lane * 16);
  }
  if (XN && !e.baw) {  // bf16(v): in_proj_ba's input (in_qkv's layers), its rstd / c_t in the consumer
    uint16_t* const xn = reinterpret_cast<uint16_t*>(e.xn) + (size_t)par * NT * K + c;
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const uint16_t hv = __bfloat16_as_ushort(__float2bfloat16(v[t]));
        xn[t * K] = hv == 0xffffu ? (uint16_t)0x7fffu : hv;
      }
    }
  }
  // x for the next finisher's residual (validated copies) and the aux layers' draft rows
  if (XS == 2) {
    uint16_t* const xs = reinterpret_cast<uint16_t*>(e.xs) + (size_t)par * NT * K + c;
    uint16_t* const ax = e.aux ? reinterpret_cast<uint16_t*>(e.aux) + c : nullptr;
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const uint16_t hv = __bfloat16_as_ushort(__float2bfloat16(x[t]));
        if (ax) ax[t * K] = hv;
        xs[t * K] = hv == 0xffffu ? (uint16_t)0x7fffu : hv;
      }
    }
  } else if (XS == 1) {
    uint32_t* const xs = reinterpret_cast<uint32_t*>(e.xs) + (size_t)par * NT * K + c;
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < (int)T) {
        const uint32_t b = __float_as_uint(x[t]);
        xs[t * K] = b == ~0u ? 0x7fffffffu : b;
      }
    }
  }
  // the tile's sums of squares (rs_sum8 per warp, the 4 warps in order) into its slot
  float s[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) s[t] = __fmul_rn(x[t], x[t]);
  const float sw = rs_sum8(s, lane);
  const uint32_t red = smem_u32(shared_b_sm(sm)) + SHARED_B_SSQ_OFF;
  if (!(lane & 3)) sts_f32(red + (q * NT + (lane >> 2)) * 4, sw);
  named_sync_u(BAR_EPI, 128);
  if (XS == 2 && e.aux && q == 0 && lane == 0) {  // (the 4 warps' aux rows, before the barrier: warp 6's release)
    const uint32_t ca = smem_u32(shared_b_sm(sm)) + SHARED_B_DRAFT_ROWS_OFF;
    uint32_t n;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(n) : "r"(ca) : "memory");
    asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(ca), "r"(n + 1) : "memory");
  }
  if (q == 0 && lane < 2) {  // (lane 0: rows 0-3, lane 1: rows 4-7)
    const float4 r0 = lds_f4(red + (0 * NT + 4 * lane) * 4), r1 = lds_f4(red + (1 * NT + 4 * lane) * 4);
    const float4 r2 = lds_f4(red + (2 * NT + 4 * lane) * 4), r3 = lds_f4(red + (3 * NT + 4 * lane) * 4);
    auto tile = [](float a, float b, float c2, float d2) {
      const uint32_t u = __float_as_uint(((a + b) + c2) + d2);
      return u == ~0u ? 0x7fffffffu : u;
    };
    const uint4 w = make_uint4(tile(r0.x, r1.x, r2.x, r3.x), tile(r0.y, r1.y, r2.y, r3.y), tile(r0.z, r1.z, r2.z, r3.z),
                               tile(r0.w, r1.w, r2.w, r3.w));
    if (lane == 0 || T > 4) st_relaxed_v4(reinterpret_cast<uint32_t*>(e.ssq) + ((size_t)par * e.mtiles + mt) * NT + 4 * lane, w);
  }
  if (XN && e.baw) shared_b_ba_part(e, smem_u32(sm.xchg), mt, q, lane, par, T, v);  // (the b / a partials)
}
// (shared B) the finisher's tail for its consumer (schedule.py link_shared_b): one instance, every consumer is bf16
__device__ __forceinline__ void shared_b_tail(const Smem& sm, uint32_t mt, int q, int lane, uint32_t epoch, uint32_t T,
                                              const uint32_t (&ob)[NT]) {
  const SharedBExt& e = *reinterpret_cast<const SharedBExt*>(shared_b_sm(sm) + q * 128);
  shared_b_tail_kind<KIND_BF16>(sm, mt, q, lane, epoch, T, ob);
}
// FIN (shared B, ACC_CNT ops without the atomics): a tile's later parts store their partial sums (fp32, alpha applied);
// its first part, the last segment of its instruction, sums them after its own in one order (deterministic, other bits
// than the atomics'), then the tile's outputs: bf16 (SharedBExt::outs, or no shared B), the consumer's B (shared B)
__device__ __forceinline__ void epilogue_fin(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& seg, int warp,
                                             int lane, uint32_t epoch, uint32_t T) {
  const uint32_t spm = stages_per_mtile(d);
  const int q = warp & 3;
  const int row = q * 32 + lane;
  const uint32_t rs = d.mtiles * 128;
  const uint32_t tag32 = ((epoch & 1u) << 8) | (d.fin - 1u);
  const float alpha = d.alpha;
  // (shared B) each warp's copy of the finishers' parameters; if the instruction's last segment is its tile's first
  // part (a finisher): the next parity's copies reset, and this warp's 32 channels of the T residual rows (into the
  // exchange buffer) and of the norm weight copied by cp.async (no registers held) while the instruction's MMAs run
  const bool builds_b = (in.aux & AUX_SHARED_B) != 0;
  bool stores_outputs = !builds_b;
  if (builds_b) {
    if (lane < 8) cp_async_16(shared_b_sm(sm) + q * 128 + lane * 16, reinterpret_cast<const uint8_t*>(d.shared_b_ext) + lane * 16);
    cp_async_wait_all();
    __syncwarp();
    const SharedBExt& e = *reinterpret_cast<const SharedBExt*>(shared_b_sm(sm) + q * 128);
    stores_outputs = e.outs != 0;
    // (the instruction's last segment is its M tile's first part: it starts at the run's start, in slab 0)
    const StagePos pl = stage_pos(d, in.r1 - 1);
    const uint32_t ml = pl.mt, lc = d.slab - pl.left;  // (lc: the last stage's chunk within its slab run)
    if (in.r1 - 1 - lc >= in.r0 && pl.kc == lc) {
      if (e.erstd) shared_b_centre(e, ml, q, smem_u32(shared_b_sm(sm)) + SHARED_B_CT_OFF + q * NT * 4, lane, epoch, T);
      if (e.xn && e.baw && q * 32 + lane < (int)BA_OUT)  // the tile's 256 bytes of in_proj_ba's row (b / a partials)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;"
                     ::"l"(reinterpret_cast<const uint8_t*>(e.baw) + ((size_t)(q * 32 + lane) * K_NORM + ml * 128) * 2) : "memory");
      shared_b_reset(e, ml, q, lane, epoch);
      const uint32_t bpv = e.res_bf16 ? 2u : 4u, cpr = 2u * bpv;  // bytes per value, 16-byte copies per row
      const uint8_t* const rsrc = e.rin ? reinterpret_cast<const uint8_t*>(e.rin) + (size_t)(epoch & 1u) * NT * K_NORM * bpv
                                        : reinterpret_cast<const uint8_t*>(e.res_in);
      uint8_t* const rdst = reinterpret_cast<uint8_t*>(sm.xchg) + q * 1024;
      for (uint32_t k = lane; k < T * cpr; k += 32) {
        const uint32_t t = k / cpr, j = k - t * cpr;
        cp_async_16(rdst + t * 128 + j * 16, rsrc + ((size_t)t * K_NORM + ml * 128 + q * 32) * bpv + j * 16);
      }
      if (lane < 4)
        cp_async_16(shared_b_sm(sm) + SHARED_B_NW_OFF + q * 64 + lane * 16,
                    reinterpret_cast<const uint8_t*>(e.nw) + (ml * 128 + q * 32) * 2 + lane * 16);
    }
  }
  uint32_t st = in.r0;
  while (st < in.r1) {
    // (off: the segment's first K chunk, absolute: 0 for the tile's first part; fparts holds the parts' first chunks)
    const StagePos sp = stage_pos(d, st);
    const uint32_t mt = sp.mt, seg_end = min(in.r1, st + sp.left);
    const uint32_t idx = mt * 128 + row, off = sp.kc;
    const uint32_t npt = reinterpret_cast<const uint8_t*>(d.p[1])[mt];
    uint32_t* const ws = reinterpret_cast<uint32_t*>(d.fslots) + ((size_t)mt * FIN_MAXP * FIN_W * 128 + row) * 4;
    // a later part: its index among the tile's parts (the table's loads before the accumulator wait, not after it)
    uint32_t p = 1;
    if (off != 0) {
      const uint16_t* const fp = reinterpret_cast<const uint16_t*>(d.fparts) + mt * FIN_MAXP;
      while (p + 1 < npt && fp[p + 1] <= off) p++;
    }
    mbar_wait(&sm.dfull[seg.slot], seg.phase);
    tc_fence_after();
    float y[NT];
    tc_ld_x8(sm.tmem + seg.slot * NT, y);
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.dempty[seg.slot]);
    if (idx < d.n_out) {  // (rows past the matrix: TMA zero fill)
#pragma unroll
      for (int t = 0; t < NT; t++) y[t] = __fmul_rn(y[t], alpha);
      // a later part's words {rows 3k .. 3k + 2, tag} at [mt][part][k][row] (tag: the launch's parity and the op; every
      // row each launch, so no word of an earlier launch of this parity is left)
      if (off != 0) {  // a later part: its words
#pragma unroll
        for (int k = 0; k < (int)FIN_W; k++)
          st_relaxed_v4(ws + ((p * FIN_W + k) * 128) * 4,
                        make_uint4(__float_as_uint(y[3 * k]), __float_as_uint(y[3 * k + 1]),
                                   3 * k + 2 < NT ? __float_as_uint(y[(3 * k + 2) % NT]) : 0u, tag32));
      } else {  // the first part: its later parts' words, the last parts first (the tile's last part ends early)
        // FIN_PC parts' words in flight at once, summed after its own in one order (its own, then the parts from the
        // last)
#pragma unroll 1
        for (int p0 = (int)npt - 1; p0 >= 1; p0 -= FIN_PC) {
          uint4 w[FIN_PC][FIN_W];
#pragma unroll
          for (int j = 0; j < FIN_PC; j++)
#pragma unroll
            for (int k = 0; k < (int)FIN_W; k++)
              w[j][k] = p0 - j >= 1 ? ld_relaxed_v4(ws + (((p0 - j) * FIN_W + k) * 128) * 4) : make_uint4(0, 0, 0, tag32);
          bool all_tagged = true;
#pragma unroll
          for (int j = 0; j < FIN_PC; j++)
#pragma unroll
            for (int k = 0; k < (int)FIN_W; k++) all_tagged &= w[j][k].w == tag32;
#pragma unroll 1
          while (!all_tagged) {  // the words not yet stored re-loaded, in rounds
            all_tagged = true;
#pragma unroll
            for (int j = 0; j < FIN_PC; j++)
#pragma unroll
              for (int k = 0; k < (int)FIN_W; k++)
                if (w[j][k].w != tag32) {
                  w[j][k] = ld_relaxed_v4(ws + (((p0 - j) * FIN_W + k) * 128) * 4);
                  all_tagged = false;
                }
          }
#pragma unroll
          for (int j = 0; j < FIN_PC; j++)
#pragma unroll
            for (int k = 0; k < (int)FIN_W; k++) {
              if (p0 - j >= 1) {
                y[3 * k] += __uint_as_float(w[j][k].x);
                y[3 * k + 1] += __uint_as_float(w[j][k].y);
                if (3 * k + 2 < NT) y[(3 * k + 2) % NT] += __uint_as_float(w[j][k].z);
              }
            }
        }
        uint32_t ob[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) ob[t] = __bfloat16_as_ushort(__float2bfloat16(y[t]));
        if (stores_outputs) {
          __nv_bfloat16* const out = reinterpret_cast<__nv_bfloat16*>(cnt_out(d, epoch));
          uint32_t* const tag_words = reinterpret_cast<uint32_t*>(d.p[3]);
#pragma unroll
          for (int t = 0; t < NT; t++) {
            if (t < (int)T) {
              out[t * rs + idx] = __ushort_as_bfloat16((unsigned short)ob[t]);
              if (tag_words) st_relaxed_b32(tag_words + t * rs + idx, ob[t] | ((epoch + 1) << 16));
            }
          }
        }
        if (builds_b) {  // (the last segment of its instruction: the loop's state dead through the tail)
          seg.next(NACC);
          shared_b_tail(sm, mt, q, lane, epoch, T, ob);
          return;
        }
      }
    }
    st = seg_end;
    seg.next(NACC);
  }
}
// (shared B) a consumer's per-token factor f_t = rstd_t / c_t: rstd_t from its boundary's 40 ssq slots in the
// finishers' exchange order (lane l: tiles l, l + 32, then rs_sum8, rms_rsqrt5120), c_t the producer's published scale;
// lane t < NT stores f_t at fsm + 4 t (the caller has seen SharedBExt::erstd set). One warp, its own loads
__device__ __forceinline__ bool shared_b_erstd(uint64_t shared_b_ext) {  // (read-only: loaded early, off the path)
  return __ldg(reinterpret_cast<const uint32_t*>(shared_b_ext + offsetof(SharedBExt, erstd))) != 0u;
}
__device__ __forceinline__ void shared_b_factors(uint64_t shared_b_ext, int lane, uint32_t epoch, uint32_t T, uint32_t fsm) {
  constexpr uint32_t FACTORS_POLL_NS = 64;  // the backoff between two polls of the slots still at their sentinel
  const SharedBExt& e = *reinterpret_cast<const SharedBExt*>(shared_b_ext);
  const uint32_t par = epoch & 1u;
  const uint32_t* const sp = reinterpret_cast<const uint32_t*>(e.ssq) + (size_t)par * e.mtiles * NT;
  const uint32_t* const cp = reinterpret_cast<const uint32_t*>(e.cs) + par * NT;
  const uint32_t ta = (uint32_t)lane, tb = (uint32_t)lane + 32;
  const bool ha = ta < e.mtiles, hb = tb < e.mtiles, h4 = T > 4;
  uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0, b0 = a0, b1 = a0;
  uint32_t c = 0u;
  for (;;) {  // (complete by now: re-polled only if a word is still at its sentinel)
    if (ha) {
      a0 = ld_relaxed_v4(sp + ta * NT);
      if (h4) a1 = ld_relaxed_v4(sp + ta * NT + 4);
    }
    if (hb) {
      b0 = ld_relaxed_v4(sp + tb * NT);
      if (h4) b1 = ld_relaxed_v4(sp + tb * NT + 4);
    }
    if (lane < (int)T) c = ld_relaxed(cp + lane);
    const bool is_missing = (ha && (a0.w == ~0u || (h4 && a1.w == ~0u))) ||
                            (hb && (b0.w == ~0u || (h4 && b1.w == ~0u))) || (lane < (int)T && c == ~0u);
    if (!__any_sync(0xffffffffu, is_missing)) break;
    __nanosleep(FACTORS_POLL_NS);
  }
  const uint32_t av[NT] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  const uint32_t bv[NT] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  float rs[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) {
    float v = ha ? __uint_as_float(av[t]) : 0.f;
    if (hb) v = v + __uint_as_float(bv[t]);
    rs[t] = v;
  }
  const float rstd_lane = rms_rsqrt5120(rs_sum8(rs, lane), e.eps);  // (row lane >> 2's rstd)
  const float rt = __shfl_sync(0xffffffffu, rstd_lane, 4 * (lane & (NT - 1)));  // (lane t: row t's)
  if (lane < NT) sts_f32(fsm + lane * 4, __fdiv_rn(rt, __uint_as_float(c)));  // (c_t a power of 2: exact)
  __syncwarp();
}
// (shared B) the factors' hand-offs in shared memory past the epilogue warps' factors, by instruction (ii & 7, tagged
// shared_b_tag: an earlier instruction's or launch's entry never matches): the prologue's B ready (pt 0; every slot the
// factors read is written before the B words), then epilogue warp 0's factors done (pro_ba_pub)
__device__ __forceinline__ uint32_t shared_b_tag(uint32_t ii, uint32_t epoch) {
  return ((epoch & 0xffu) << 24) | ((ii + 1u) & 0xffffffu);
}
constexpr uint32_t SHARED_B_TAG_MASK = 7u;         // each hand-off's 8 slots, by instruction: ii & SHARED_B_TAG_MASK
constexpr uint32_t SHARED_B_FACTORS_TAG_OFF = 32;  // the factors' slots, past the B-ready ones
constexpr int SHARED_B_TAG_WORDS = 16;             // both hand-offs' slots
__device__ __forceinline__ uint32_t shared_b_ready_slot(const Smem& sm, uint32_t ii) {
  return smem_u32(shared_b_sm(sm)) + SHARED_B_SM_BYTES + 4 * NT * 4 + (ii & SHARED_B_TAG_MASK) * 4;
}
__device__ __forceinline__ uint32_t shared_b_factors_slot(const Smem& sm, uint32_t ii) {
  return shared_b_ready_slot(sm, ii) + SHARED_B_FACTORS_TAG_OFF;
}
__device__ __forceinline__ void shared_b_wait_tag(uint32_t addr, uint32_t tag) {
  constexpr uint32_t TAG_POLL_NS = 32;  // the backoff between two polls of the tag
  uint32_t v;
  for (;;) {
    asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
    if (v == tag) break;
    __nanosleep(TAG_POLL_NS);
  }
}
__device__ __forceinline__ void shared_b_set_tag(uint32_t addr, uint32_t tag) {
  asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(addr), "r"(tag) : "memory");
}
// Warps 0..3, TMEM lane quadrant q = warp % 4: per M-tile segment, accumulator row 32q + lane (column t: token row t)
// added into the op's output, rows t < T only
__device__ void epilogue_gemv(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& seg, int warp, int lane,
                              uint32_t epoch, uint32_t T, uint32_t ii) {
  if (d.fin) {  // (FIN: a loop of its own, its registers not added to the other formats')
    epilogue_fin(in, d, sm, seg, warp, lane, epoch, T);
    return;
  }
  const uint32_t spm = stages_per_mtile(d);
  const int q = warp & 3, ept = (warp - W_EPI0) * 32 + lane;
  const int row = q * 32 + lane;
  const uint32_t rs = d.mtiles * 128;  // an output vector's row stride (per token)
  uint8_t* acc = acc_copy(d, epoch);
  // (shared B) a shared-B consumer's (or its twin's: d.shared_b_ext) per-token factors in this warp's 8 floats past
  // SHARED_B_SM_BYTES, computed once the prologue has B ready (its tag) while the MMAs run, not on the tail after the
  // first accumulator; the twin (in_proj_z, the same boundary) keeps in_proj_qkv's
  const uint32_t pfs = smem_u32(shared_b_sm(sm)) + SHARED_B_SM_BYTES + q * NT * 4;
  const bool has_factors = d.shared_b_ext && shared_b_erstd(d.shared_b_ext);
  if (__builtin_expect(has_factors, 0) && !(in.aux & AUX_TWIN)) {
    const uint32_t tag = shared_b_tag(ii, epoch);
    shared_b_wait_tag(shared_b_ready_slot(sm, ii), tag);
    shared_b_factors(d.shared_b_ext, lane, epoch, T, pfs);
    if (q == 0 && lane == 0) shared_b_set_tag(shared_b_factors_slot(sm, ii), tag);  // (pro_ba_pub: in_proj_qkv's b / a)
  }
  uint32_t st = in.r0;
  while (st < in.r1) {
    const StagePos sp = stage_pos(d, st);
    const uint32_t mt = sp.mt, seg_end = min(in.r1, st + sp.left);
    const bool is_silu = d.epi == EPI_SILU;
    // (p[2] nonzero: the LM head split stream-K, its split tiles finished as EPI_SILU's, p[1] the parts' sums, p[2] the
    // tiles' flags; schedule.py gives it whole M tiles, p[2] = 0)
    const bool is_lm_split = d.epi == EPI_LOGITS && d.p[2] != 0;
    const bool is_whole_tile = d.slab == spm && sp.kc == 0 && seg_end == st + spm;  // (the segment is the whole M tile)
    // EPI_SILU (silu_bf16_tile), a tile split between two instructions: the suffix part (the later SM's first segment,
    // early) stores its partial sums and releases the tile's flag; the prefix part (the earlier SM's last segment)
    // acquires them, adds them to its own (one fp32 addition: 0 + a + b) and finalizes the tile
    const bool prefix = (is_silu || is_lm_split) && sp.kc == 0 && !is_whole_tile,
               suffix = (is_silu || is_lm_split) && sp.kc > 0;
    float* parts = reinterpret_cast<float*>(d.p[1]) + (size_t)mt * NT * 128 + row;  // EPI_SILU split tiles
    float part[NT];  // a prefix part: the suffix part's sums (acquired before its own accumulator is complete)
#pragma unroll
    for (int t = 0; t < NT; t++) part[t] = 0.f;
    if (prefix) {
      if (ept == 0) {
        while (ctr_before(ld_acquire(reinterpret_cast<const uint32_t*>(d.p[2]) + mt), epoch + 1)) {
        }
      }
      named_sync_u(BAR_EPI, 128);
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < (int)T) part[t] = __ldcg(parts + t * 128);
    }
    mbar_wait(&sm.dfull[seg.slot], seg.phase);
    tc_fence_after();
    float v[NT];
    tc_ld_x8(sm.tmem + seg.slot * NT, v);
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.dempty[seg.slot]);
    if (__builtin_expect(has_factors, 0)) {
#pragma unroll
      for (int t = 0; t < NT; t++) {
        float fv;
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(fv) : "r"(pfs + t * 4));
        v[t] = __fmul_rn(v[t], fv);
      }
    }
    const uint32_t idx = mt * 128 + row;
    if (is_silu || suffix) {  // (suffix: an EPI_SILU or a split LM-head tile's later part)
      if (suffix) {
#pragma unroll
        for (int t = 0; t < NT; t++)
          if (t < (int)T) __stcg(parts + t * 128, v[t]);
        named_sync_u(BAR_EPI, 128);
        if (ept == 0) red_release_add(reinterpret_cast<uint32_t*>(d.p[2]) + mt, 1);
      } else {
#pragma unroll
        for (int t = 0; t < NT; t++) v[t] += part[t];
        silu_bf16_tile(d, mt, ept, lane, v, sm.xchg + (seg.slot & 1u) * 64 * NT, T, epoch);
      }
    } else if (idx >= d.n_out) {  // rows past the matrix (TMA zero fill)
    } else if (d.epi == EPI_LOGITS) {  // bf16 as vLLM's linear returns them
      if (prefix) {  // (a split tile's earlier part: the later part's sums, acquired above, added: 0 + a + b)
#pragma unroll
        for (int t = 0; t < NT; t++) v[t] += part[t];
      }
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < (int)T) reinterpret_cast<__nv_bfloat16*>(d.x_out)[(size_t)t * d.n_out + idx] = __float2bfloat16(d.alpha * v[t]);
    } else if (d.acc_fmt == ACC_F32 && d.p[3]) {
      // Q_LL (in_proj_qkv, <= 2 parts per M tile): the part's sums as tagged words {fp32, token} in the part's rows of
      // p[3] [2][NT][rs] instead of red.adds; token = (epoch + 1) << 1 | (part 0 of a two-part tile)
      const bool is_first_part = sp.kc == 0;
      const uint32_t hi = ((epoch + 1) << 1) | ((is_first_part && !is_whole_tile) ? 1u : 0u);
      uint64_t* w = reinterpret_cast<uint64_t*>(d.p[3]) + (is_first_part ? 0u : NT * rs) + idx;
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < (int)T) st_relaxed_b64(w + t * rs, ((uint64_t)hi << 32) | __float_as_uint(v[t]));
    } else if (d.acc_fmt == ACC_F32) {
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < (int)T) red_add_f32_global(reinterpret_cast<float*>(acc) + t * rs + idx, v[t]);
    } else {
      // every row's add issued before any result is used (one atomic round trip, not T: explicitly global atomics, the
      // desc's fields read before them)
      const uint32_t npt = reinterpret_cast<const uint8_t*>(d.p[1])[mt];
      const float alpha = d.alpha;
      __nv_bfloat16* const out = reinterpret_cast<__nv_bfloat16*>(cnt_out(d, epoch));
      uint32_t* const tag_words = reinterpret_cast<uint32_t*>(d.p[3]);
      // (the atomics first, their results used only after the last one: an add of a result right behind its atomic
      // waits for it, T round trips)
      unsigned long long tot[NT], inc[NT];
#pragma unroll
      for (int t = 0; t < NT; t++) {
        const long long fx = __float2ll_rn(v[t] * alpha * CNT_SCALE);
        inc[t] = (unsigned long long)fx * (1ull << CNT_BITS) + 1ull;
        tot[t] = 0ull;
        atom_add_u64_global(reinterpret_cast<unsigned long long*>(acc) + t * rs + idx, inc[t], t < (int)T, tot[t]);
      }
#pragma unroll
      for (int t = 0; t < NT; t++) tot[t] = t < (int)T ? tot[t] + inc[t] : 0ull;
#pragma unroll
      for (int t = 0; t < NT; t++) {
        if (t < (int)T && (uint32_t)(tot[t] & ((1u << CNT_BITS) - 1)) == npt) {
          const __nv_bfloat16 o = __float2bfloat16(__ll2float_rn((long long)tot[t] >> CNT_BITS) * CNT_INV);
          out[t * rs + idx] = o;
          if (tag_words)  // also as a tagged word {bf16, launch token} for a consumer that polls it (in_proj_z: Z_LL)
            st_relaxed_b32(tag_words + t * rs + idx, (uint32_t)__bfloat16_as_ushort(o) | ((epoch + 1) << 16));
        }
      }
    }
    st = seg_end;
    seg.next(NACC);
  }
}

// ------------------------------------------------------------------------------------------------ prologue
__device__ __forceinline__ void unpack16_bf16(uint4 a, uint4 b, float (&v)[16]) {
  const uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[2 * i] = bf16lo(u[i]);
    v[2 * i + 1] = bf16hi(u[i]);
  }
}
// the producing op's output as a consumer reads it: the ACC_F32 accumulator copy or the ACC_CNT bf16 output, its token
// rows rs values apart
struct SrcAcc {
  const uint8_t* acc;
  float alpha;
  uint32_t rs;
};
__device__ __forceinline__ SrcAcc src_acc(const OpDesc& s, uint32_t epoch) {
  return SrcAcc{s.acc_fmt == ACC_CNT ? cnt_out(s, epoch) : acc_copy(s, epoch), s.alpha, s.mtiles * 128};
}

// the prologue reductions' smem: the norms' two alternating areas, then the b / a sums [NT][2][warps]
constexpr int RED_NORM = 0, RED_BA = 2 * (NPRO / 32);
static_assert(RED_BA + NT * 2 * (NPRO / 32) <= RED_FLOATS, "prologue reductions");

// the B address of byte b of token row t in the chunk at slot l
__device__ __forceinline__ uint8_t* b_addr(uint8_t* xb, uint32_t l, uint32_t b, uint32_t t) {
  return xb + l * CHUNK_B + (b / 16) * 128 + t * 16 + (b % 16);
}
// value vi's chunk (the kind's values per 128-byte chunk) and byte offset in it
template <int KIND>
__device__ __forceinline__ void b_place(uint32_t vi, uint32_t& kc, uint32_t& b) {
  constexpr uint32_t VPC = KIND == KIND_NVFP4 ? 256 : KIND == KIND_FP8 ? 128 : 64;
  kc = vi / VPC;
  b = KIND == KIND_NVFP4 ? (vi % VPC) / 2 : KIND == KIND_FP8 ? vi % VPC : (vi % VPC) * 2;
}

// a halfword 0xFFFF (the bf16 operands' sentinel) in x, branch-free: h is 0xFFFF iff its bit 15 and the carry of
// (h & 0x7FFF) + 1 into bit 15 are both set (no carry crosses into the upper halfword)
__device__ __forceinline__ uint32_t ffff_bits(uint32_t x) { return x & ((x & 0x7fff7fffu) + 0x00010001u) & 0x80008000u; }
__device__ __forceinline__ bool has_ffff8(const uint4& v) {
  return (ffff_bits(v.x) | ffff_bits(v.y) | ffff_bits(v.z) | ffff_bits(v.w)) != 0u;
}
__device__ __forceinline__ bool has_ff4(uint32_t x) {  // a byte 0xFF in x
  const uint32_t y = ~x;
  return ((y - 0x01010101u) & ~y & 0x80808080u) != 0u;
}
// A validated operand's re-poll (PRO_ACT with act_stride, bf16 values): the rows < T's 16-byte words still holding a
// 0xFFFF halfword re-polled from L2; word j is row j % NT of group (j / NT) % 8 of chunk slot j / (8 NT), so a thread's
// words are of one row (NPRO % NT == 0): no division by the runtime T per word
__device__ __forceinline__ void repoll_val(uint8_t* xb, const uint8_t* act, const ChunkRange& cr, int pt, uint32_t T) {
  static_assert(NPRO % NT == 0 && CHUNK_B == 8 * NT * 16, "one row per thread, 8 groups of NT rows a chunk");
  if ((uint32_t)pt % NT < T) {
    // (every word of the thread read back at once, one mask of the stale ones: the re-polls below only for those)
    constexpr int KW = (B_CHUNKS * 8 * NT + NPRO - 1) / NPRO;
    static_assert(KW <= 32, "a thread's words in one mask");
    const uint32_t nw = cr.nck * (8 * NT), x0 = smem_u32(xb);
    uint32_t stale_words = 0u;
    {
      uint4 w[KW];
#pragma unroll
      for (int k = 0; k < KW; k++) w[k] = (uint32_t)pt + k * NPRO < nw ? lds_u4(x0 + ((uint32_t)pt + k * NPRO) * 16) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int k = 0; k < KW; k++) stale_words |= (has_ffff8(w[k]) ? 1u : 0u) << k;
    }
#pragma unroll 1
    while (stale_words) {
      const int k = __ffs(stale_words) - 1;
      stale_words &= stale_words - 1u;
      const uint32_t j = (uint32_t)pt + (uint32_t)k * NPRO, l = j / (8 * NT);
      const uint32_t* gp = reinterpret_cast<const uint32_t*>(act + cr.abs(l) * CHUNK_B + (j % (8 * NT)) * 16);
      uint4 v;
      do {
        v = ld_relaxed_v4(gp);
      } while (has_ffff8(v));
      *reinterpret_cast<uint4*>(xb + j * 16) = v;
    }
  }
}
// the shared B's validated NVFP4 operand: a code nibble 0x8 (-0, which the producer writes as +0) in x
__device__ __forceinline__ bool has_8nib(uint32_t x) {
  return (x & ~((x & 0x77777777u) + 0x77777777u) & 0x88888888u) != 0u;
}
// repoll_val for the shared B's NVFP4 operand: the codes' 16-byte words (as FP8's: one row per thread) still holding a
// 0x8 nibble, then the block scales' words of rows < T (two halves of [NT rows][16 bytes] per chunk; bytes 0-3 and 8-11
// of a row's 16 are its blocks' scales, sfb_pos) still holding a 0xFF byte, re-polled from L2 until written
__device__ __forceinline__ void repoll_val_nv(uint8_t* xb, uint8_t* sfb, const uint8_t* act, const uint8_t* sc,
                                              const ChunkRange& cr, int pt, uint32_t T) {
  static_assert(NPRO % NT == 0 && CHUNK_B == 8 * NT * 16 && SFB_CHUNK == 2 * NT * 16, "one row per thread");
  if ((uint32_t)pt % NT < T) {
#pragma unroll 1
    for (uint32_t j = pt; j < cr.nck * (8 * NT); j += NPRO) {
      uint4 v = *reinterpret_cast<const uint4*>(xb + j * 16);
      if (has_8nib(v.x) || has_8nib(v.y) || has_8nib(v.z) || has_8nib(v.w)) {
        const uint32_t l = j / (8 * NT);
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(act + cr.abs(l) * CHUNK_B +
                                                               (j % (8 * NT)) * 16);
        do {
          v = ld_relaxed_v4(gp);
        } while (has_8nib(v.x) || has_8nib(v.y) || has_8nib(v.z) || has_8nib(v.w));
        *reinterpret_cast<uint4*>(xb + j * 16) = v;
      }
    }
#pragma unroll 1
    for (uint32_t j = pt; j < cr.nck * (2 * NT); j += NPRO) {
      uint4 v = *reinterpret_cast<const uint4*>(sfb + j * 16);
      if (has_ff4(v.x) || has_ff4(v.z)) {
        const uint32_t l = j / (2 * NT);
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(sc + cr.abs(l) * SFB_CHUNK +
                                                               (j % (2 * NT)) * 16);
        do {
          v = ld_relaxed_v4(gp);
        } while (has_ff4(v.x) || has_ff4(v.z));
        *reinterpret_cast<uint4*>(sfb + j * 16) = v;
      }
    }
  }
}
// The PRO_ACT operand's chunks bulk-copied into the B buffer (slot order: kc0.. then the wrap), issued by one thread
// once the dependency is met; every prologue thread then waits on stg_full (act_stride: copy epoch & 1 without a
// dependency, then each thread re-polls its words not written yet)
__device__ __forceinline__ void stage_inputs(const Instr& in, const OpDesc& d, const Smem& sm, int pt, uint32_t& phase,
                                             uint8_t* xb, uint8_t* sfb, uint32_t epoch, uint32_t T) {
  const uint8_t* act = reinterpret_cast<const uint8_t*>(d.in0) + (epoch & 1) * d.act_stride;
  const ChunkRange cr = chunk_range(in, d);
  if (pt == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");  // the acquired data, for the async-proxy reads
    const bool nv = d.kind == KIND_NVFP4;
    mbar_expect_tx(sm.stg_full, cr.nck * (nv ? CHUNK_B + SFB_CHUNK : CHUNK_B));
    bulk_g2s(xb, act + cr.kc0 * CHUNK_B, cr.n1 * CHUNK_B, sm.stg_full);
    if (cr.nck > cr.n1) bulk_g2s(xb + cr.n1 * CHUNK_B, act, (cr.nck - cr.n1) * CHUNK_B, sm.stg_full);
    if (nv) {  // (the published scales in the B scales' layout, sfb_pos)
      const uint8_t* sc = act + (d.K / 256) * CHUNK_B;
      bulk_g2s(sfb, sc + cr.kc0 * SFB_CHUNK, cr.n1 * SFB_CHUNK, sm.stg_full);
      if (cr.nck > cr.n1) bulk_g2s(sfb + cr.n1 * SFB_CHUNK, sc, (cr.nck - cr.n1) * SFB_CHUNK, sm.stg_full);
    }
  }
  mbar_wait(sm.stg_full, phase);
  phase ^= 1;
  if (d.act_stride) {  // validated (GT_GNORM's format, or the shared B's)
    if (d.kind == KIND_NVFP4) repoll_val_nv(xb, sfb, act, act + (d.K / 256) * CHUNK_B, cr, pt, T);
    else
    repoll_val(xb, act, cr, pt, T);
  }
}

// shared B: an RMSNorm consumer's B (PRO_ACT with OpDesc::nw) staged from its producer's published copy by the prologue
// threads' own loads (a bulk copy queues in the SM's TMA unit behind the weight stream), 16-byte words validated
// (re-polled at the sentinel); rows < T only (one row per thread), up to SHARED_B_LOADS words in flight
#ifndef MK_SB_LOADS  // a default a build may override (20: an 80-chunk B's 20 words per thread in one round, not two)
#define MK_SB_LOADS 12
#endif
constexpr int SHARED_B_LOADS = MK_SB_LOADS;
static_assert(B_CHUNKS * 8 * NT <= 32 * NPRO, "a thread's words in one 32-bit mask");
template <bool NV>
__device__ __forceinline__ bool shared_b_stale(const uint4& v) {
  // (NV: the NVFP4 operand's sentinel, a path this package never takes; else the bf16 one)
  return NV ? (has_8nib(v.x) || has_8nib(v.y) || has_8nib(v.z) || has_8nib(v.w)) : has_ffff8(v);
}
template <bool NV>
__device__ __forceinline__ void shared_b_words(uint8_t* dst, const uint8_t* src, const ChunkRange& cr, int pt,
                                               bool is_dry) {
  constexpr uint32_t WPC = 8 * NT, CB = CHUNK_B;  // words per chunk, chunk bytes
  const uint32_t nw = cr.nck * WPC;
  // (bf16: every word of the thread in flight at once by cp.async.cg, L2 only and no registers held, then the ones
  // still at the sentinel copied again together, round after round: one L2 round trip for an 80-chunk B where the
  // register batches below take two)
  if (!NV && !is_dry) {
    static_assert(B_CHUNKS * 8 * NT <= 32 * NPRO, "a thread's words in one 32-bit mask");
    uint32_t stale_words = 0u;
    for (uint32_t k = 0, j = pt; j < nw; k++, j += NPRO) stale_words |= 1u << k;
    // (unrolled: the stale words' copies, then their read-back all issued before one branch-free test)
    constexpr int KW = (B_CHUNKS * 8 * NT + NPRO - 1) / NPRO;
    const uint32_t d0 = smem_u32(dst);
    for (;;) {
#pragma unroll
      for (int k = 0; k < KW; k++) {
        const uint32_t j = (uint32_t)pt + k * NPRO;
        if (stale_words & (1u << k)) cp_async_16(dst + j * 16, src + cr.abs(j / WPC) * CB + (j % WPC) * 16);
      }
      cp_async_wait_all();
      uint4 w[KW];
#pragma unroll
      for (int k = 0; k < KW; k++)
        w[k] = (stale_words & (1u << k)) ? lds_u4(d0 + ((uint32_t)pt + k * NPRO) * 16) : make_uint4(0u, 0u, 0u, 0u);
      uint32_t fresh_words = 0u;
#pragma unroll
      for (int k = 0; k < KW; k++) fresh_words |= (shared_b_stale<NV>(w[k]) ? 0u : 1u) << k;
      stale_words &= ~fresh_words;
      if (!stale_words) break;
      __nanosleep(SHARED_B_REPOLL_NS);
    }
    return;
  }
#pragma unroll 1
  for (uint32_t j0 = pt; j0 < nw; j0 += SHARED_B_LOADS * NPRO) {
    uint4 v[SHARED_B_LOADS];
#pragma unroll
    for (int k = 0; k < SHARED_B_LOADS; k++) {
      const uint32_t j = j0 + k * NPRO, l = j / WPC;
      v[k] = make_uint4(0u, 0u, 0u, 0u);
      if (j < nw)
        v[k] = ld_relaxed_v4(reinterpret_cast<const uint32_t*>(src + cr.abs(l) * CB + (j % WPC) * 16));
    }
    // the words still at the sentinel, re-loaded together in rounds (polled while the finishers write:
    // backoff between the rounds), then every word stored
    uint32_t sm_ = 0u;  // (bit k: word k stale)
    if (!is_dry) {
#pragma unroll
      for (int k = 0; k < SHARED_B_LOADS; k++)
        if (j0 + k * NPRO < nw && shared_b_stale<NV>(v[k])) sm_ |= 1u << k;
    }
#pragma unroll 1
    while (__builtin_expect(sm_ != 0u, 0)) {
      __nanosleep(SHARED_B_REPOLL_NS);
#pragma unroll
      for (int k = 0; k < SHARED_B_LOADS; k++) {
        const uint32_t j = j0 + k * NPRO, l = j / WPC;
        if (sm_ & (1u << k))
          v[k] = ld_relaxed_v4(reinterpret_cast<const uint32_t*>(src + cr.abs(l) * CB + (j % WPC) * 16));
      }
#pragma unroll
      for (int k = 0; k < SHARED_B_LOADS; k++)
        if ((sm_ & (1u << k)) && !shared_b_stale<NV>(v[k])) sm_ &= ~(1u << k);
    }
#pragma unroll
    for (int k = 0; k < SHARED_B_LOADS; k++) {
      const uint32_t j = j0 + k * NPRO;
      if (j < nw) *reinterpret_cast<uint4*>(dst + j * 16) = v[k];
    }
  }
}
// an NVFP4 operand's codes and block scales staged as one batch: a thread's code words (j = pt + k NPRO,
// k < 5: <= 20 chunks of 8 NT words) and scale words (k < 2: 2 NT a chunk) loaded together, the stale ones re-polled in
// rounds, then stored (shared_b_words<true>'s code words, then the scales' words)
__device__ __forceinline__ void shared_b_words_nv(uint8_t* xb, uint8_t* sfb, const uint8_t* act, const uint8_t* sc,
                                                  const ChunkRange& cr, int pt, bool is_dry) {
  constexpr uint32_t WC = 8 * NT, WS = 2 * NT;
  constexpr int KC = (SFB_CHUNKS * WC + NPRO - 1) / NPRO, KS = (SFB_CHUNKS * WS + NPRO - 1) / NPRO;
  const uint32_t nwc = cr.nck * WC, nws = cr.nck * WS;
  auto gcode = [&](uint32_t j) {
    const uint32_t l = j / WC;
    return reinterpret_cast<const uint32_t*>(act + cr.abs(l) * CHUNK_B + (j % WC) * 16);
  };
  auto gscale = [&](uint32_t j) {
    const uint32_t l = j / WS;
    return reinterpret_cast<const uint32_t*>(sc + cr.abs(l) * SFB_CHUNK + (j % WS) * 16);
  };
  uint4 vc[KC], vs[KS];
#pragma unroll
  for (int k = 0; k < KC; k++) {
    const uint32_t j = pt + k * NPRO;
    vc[k] = make_uint4(0u, 0u, 0u, 0u);
    if (j < nwc) vc[k] = ld_relaxed_v4(gcode(j));
  }
#pragma unroll
  for (int k = 0; k < KS; k++) {
    const uint32_t j = pt + k * NPRO;
    vs[k] = make_uint4(0u, 0u, 0u, 0u);
    if (j < nws) vs[k] = ld_relaxed_v4(gscale(j));
  }
  auto sst = [](const uint4& w) { return has_ff4(w.x) || has_ff4(w.z); };
  uint32_t mc = 0u, ms = 0u;  // (bit k: word k stale)
  if (!is_dry) {
#pragma unroll
    for (int k = 0; k < KC; k++)
      if (pt + k * NPRO < nwc && shared_b_stale<true>(vc[k])) mc |= 1u << k;
#pragma unroll
    for (int k = 0; k < KS; k++)
      if (pt + k * NPRO < nws && sst(vs[k])) ms |= 1u << k;
  }
#pragma unroll 1
  while (__builtin_expect((mc | ms) != 0u, 0)) {
    __nanosleep(SHARED_B_REPOLL_NS);
#pragma unroll
    for (int k = 0; k < KC; k++)
      if (mc & (1u << k)) vc[k] = ld_relaxed_v4(gcode(pt + k * NPRO));
#pragma unroll
    for (int k = 0; k < KS; k++)
      if (ms & (1u << k)) vs[k] = ld_relaxed_v4(gscale(pt + k * NPRO));
#pragma unroll
    for (int k = 0; k < KC; k++)
      if ((mc & (1u << k)) && !shared_b_stale<true>(vc[k])) mc &= ~(1u << k);
#pragma unroll
    for (int k = 0; k < KS; k++)
      if ((ms & (1u << k)) && !sst(vs[k])) ms &= ~(1u << k);
  }
#pragma unroll
  for (int k = 0; k < KC; k++)
    if (pt + k * NPRO < nwc) *reinterpret_cast<uint4*>(xb + (pt + k * NPRO) * 16) = vc[k];
#pragma unroll
  for (int k = 0; k < KS; k++)
    if (pt + k * NPRO < nws) *reinterpret_cast<uint4*>(sfb + (pt + k * NPRO) * 16) = vs[k];
}
// (not inlined: one copy of its code, outside mk_main's, which the dry run brings into the instruction cache; its name
// sorts after ss_phase's: ptxas lays the internal functions out by mangled name)
// (is_dry: the dry run's, one row, nothing re-polled: the copy is stale before the dependency, the B slot free then)
__device__ __noinline__ void stage_pub(const uint8_t* act, uint32_t nv, uint32_t K, uint32_t nck, uint32_t kc0, uint32_t n1,
                                       uint8_t* xb, uint8_t* sfb, int pt, uint32_t T, bool is_dry) {
  static_assert(NPRO % NT == 0, "one row per thread");
  if ((uint32_t)pt % NT >= T) return;
  const ChunkRange cr{0u, nck, kc0, n1};
  if (nv) {
    if (sfb) {
      shared_b_words_nv(xb, sfb, act, act + (K / 256) * CHUNK_B, cr, pt, is_dry);
    } else {
      shared_b_words<true>(xb, act, cr, pt, is_dry);
    }
  } else {
    shared_b_words<false>(xb, act, cr, pt, is_dry);
  }
}
__device__ __forceinline__ void stage_pub(const Instr& in, const OpDesc& d, int pt, uint8_t* xb, uint8_t* sfb,
                                          uint32_t epoch, uint32_t T, bool is_dry = false) {
  const ChunkRange cr = chunk_range(in, d);
  stage_pub(reinterpret_cast<const uint8_t*>(d.in0) + (epoch & 1) * d.act_stride, d.kind == KIND_NVFP4, d.K, cr.nck,
            cr.kc0, cr.n1, xb, sfb, pt, T, is_dry);
}
// AUX_FASTACT: a validated bf16 PRO_ACT operand (no scales, no dependency): stage_inputs' copy and re-poll
__device__ __forceinline__ void stage_val(const Instr& in, const OpDesc& d, const Smem& sm, int pt, uint32_t& phase,
                                          uint8_t* xb, uint32_t epoch, uint32_t T) {
  const uint8_t* act = reinterpret_cast<const uint8_t*>(d.in0) + (epoch & 1) * d.act_stride;
  const ChunkRange cr = chunk_range(in, d);
  if (pt == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    mbar_expect_tx(sm.stg_full, cr.nck * CHUNK_B);
    bulk_g2s(xb, act + cr.kc0 * CHUNK_B, cr.n1 * CHUNK_B, sm.stg_full);
    if (cr.nck > cr.n1) bulk_g2s(xb + cr.n1 * CHUNK_B, act, (cr.nck - cr.n1) * CHUNK_B, sm.stg_full);
  }
  mbar_wait(sm.stg_full, phase);
  phase ^= 1;
  repoll_val(xb, act, cr, pt, T);
}

// ---- the RMSNorm prologues (K = K_NORM): per token row, groups of 4 consecutive values per thread
// (q = pt + i * NPRO), bf16 rounding of pairs (one cvt); a row's loads are in flight while the previous row is encoded
constexpr int NG5 = K_NORM / 4 / NPRO;  // groups per thread
__device__ __forceinline__ void bf16r2(float& a, float& b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  const uint32_t u = *reinterpret_cast<const uint32_t*>(&h);
  a = __uint_as_float(u << 16);
  b = __uint_as_float(u & 0xffff0000u);
}
// values vi..vi+3 of token row t into the B rows of the instruction's chunks (cr; the twin's too: cr2, xb2, FP8)
template <int KIND>
__device__ __forceinline__ void enc4(float gs, float inv_s, uint32_t vi, uint32_t t, float (&v)[4], const ChunkRange& cr,
                                     uint8_t* xb, uint8_t* sfb, const ChunkRange& cr2, uint8_t* xb2) {
  uint32_t kc, b;
  b_place<KIND>(vi, kc, b);
  const uint32_t l = chunk_slot(kc, cr);
  if (KIND == KIND_NVFP4) {  // the 4 lanes of a 16-value block together (vLLM's quantization of bf16(v))
    bf16r2(v[0], v[1]);
    bf16r2(v[2], v[3]);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const __nv_fp8_e4m3 q(gs * (amax * rcp_approx_ftz(6.0f)));
    const float sfq = float(q);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rcp_approx_ftz(gs)) : 0.f;
    const uint32_t c = f32x2_to_e2m1x2(v[0] * osc, v[1] * osc) | (f32x2_to_e2m1x2(v[2] * osc, v[3] * osc) << 8);
    if (l < cr.nck) {
      *reinterpret_cast<uint16_t*>(b_addr(xb, l, b, t)) = (uint16_t)c;
      if ((vi & 15) == 0) sfb[sfb_pos(l, (vi % 256) / 16, t)] = *reinterpret_cast<const uint8_t*>(&q);
    }
  } else if (KIND == KIND_FP8) {
    const float q0 = fminf(fmaxf(v[0] * inv_s, -448.f), 448.f), q1 = fminf(fmaxf(v[1] * inv_s, -448.f), 448.f);
    const float q2 = fminf(fmaxf(v[2] * inv_s, -448.f), 448.f), q3 = fminf(fmaxf(v[3] * inv_s, -448.f), 448.f);
    const uint32_t w = f32x2_to_e4m3x2(q0, q1) | (f32x2_to_e4m3x2(q2, q3) << 16);
    if (l < cr.nck) *reinterpret_cast<uint32_t*>(b_addr(xb, l, b, t)) = w;
    if (xb2) {
      const uint32_t l2 = chunk_slot(kc, cr2);
      if (l2 < cr2.nck) *reinterpret_cast<uint32_t*>(b_addr(xb2, l2, b, t)) = w;
    }
  } else {
    const __nv_bfloat162 h0 = __floats2bfloat162_rn(v[0], v[1]), h1 = __floats2bfloat162_rn(v[2], v[3]);
    if (l < cr.nck)
      *reinterpret_cast<uint2*>(b_addr(xb, l, b, t)) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
  }
}
// the codes of 4 values (and an NVFP4 block's e4m3 scale, the 4 lanes of a block together): enc4's arithmetic without
// its stores (rgs = rcp_approx_ftz(gs)), so that a row's groups are independent chains
template <int KIND>
__device__ __forceinline__ void enc_codes(float gs, float rgs, float inv_s, float (&v)[4], uint32_t& cw, uint32_t& cs) {
  if (KIND == KIND_NVFP4) {
    bf16r2(v[0], v[1]);
    bf16r2(v[2], v[3]);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const __nv_fp8_e4m3 q(gs * (amax * rcp_approx_ftz(6.0f)));
    const float sfq = float(q);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rgs) : 0.f;
    cw = f32x2_to_e2m1x2(v[0] * osc, v[1] * osc) | (f32x2_to_e2m1x2(v[2] * osc, v[3] * osc) << 8);
    cs = *reinterpret_cast<const uint8_t*>(&q);
  } else {  // (cvt.rn.satfinite saturates to +-448 as clamping first would, for every non-NaN value)
    mul2(v[0], v[1], inv_s, inv_s), mul2(v[2], v[3], inv_s, inv_s);
    cw = f32x2_to_e4m3x2(v[0], v[1]) | (f32x2_to_e4m3x2(v[2], v[3]) << 16);
    cs = 0u;
  }
}
// enc_codes over a row's NG5 groups, step by step across the groups (each step's NG5 instructions independent: the
// shuffles', the conversions' and the reciprocals' latencies overlap instead of one group's chain after another's)
template <int KIND>
__device__ __forceinline__ void enc_codes_rows(float gs, float rgs, float inv_s, float (&v)[NG5][4], uint32_t (&cw)[NG5],
                                               uint32_t (&cs)[NG5]) {
  if (KIND == KIND_NVFP4) {
    float am[NG5];
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      bf16r2(v[i][0], v[i][1]);
      bf16r2(v[i][2], v[i][3]);
    }
#pragma unroll
    for (int i = 0; i < NG5; i++) am[i] = fmaxf(fmaxf(fabsf(v[i][0]), fabsf(v[i][1])), fmaxf(fabsf(v[i][2]), fabsf(v[i][3])));
#pragma unroll
    for (int i = 0; i < NG5; i++) am[i] = fmaxf(am[i], __shfl_xor_sync(0xffffffffu, am[i], 1));
#pragma unroll
    for (int i = 0; i < NG5; i++) am[i] = fmaxf(am[i], __shfl_xor_sync(0xffffffffu, am[i], 2));
    const float r6 = rcp_approx_ftz(6.0f);
    float osc[NG5];
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      const __nv_fp8_e4m3 q(gs * (am[i] * r6));
      cs[i] = *reinterpret_cast<const uint8_t*>(&q);
      am[i] = float(q);
    }
#pragma unroll
    for (int i = 0; i < NG5; i++) osc[i] = am[i] != 0.f ? rcp_approx_ftz(am[i] * rgs) : 0.f;
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      mul2(v[i][0], v[i][1], osc[i], osc[i]), mul2(v[i][2], v[i][3], osc[i], osc[i]);
      cw[i] = f32x2_to_e2m1x2(v[i][0], v[i][1]) | (f32x2_to_e2m1x2(v[i][2], v[i][3]) << 8);
    }
  } else {
#pragma unroll
    for (int i = 0; i < NG5; i++) enc_codes<KIND>(gs, rgs, inv_s, v[i], cw[i], cs[i]);
  }
}
// the row-invariant part of a group's B store address (values vi..vi+3 of every token row: + t * 16), -1 if its chunk
// is not one of the instruction's (cr); an NVFP4 block's scale byte (sc: the block's first lane only, else -1)
template <int KIND>
__device__ __forceinline__ int32_t b_off(uint32_t vi, const ChunkRange& cr, bool sc = false) {
  uint32_t kc, b;
  b_place<KIND>(vi, kc, b);
  const uint32_t l = chunk_slot(kc, cr);
  if (l >= cr.nck || (sc && (vi & 15) != 0)) return -1;
  return (int32_t)(sc ? sfb_pos(l, (vi % 256) / 16, 0) : l * CHUNK_B + (b / 16) * 128 + (b % 16));
}
// the bf16 sentinel 0xFFFF in a word: bit 15 / 31 of the result set iff its low / high half holds it, as the zero
// halves of ~w ((x - 0x00010001) & ~x & 0x80008000 for x = ~w: exact for 16-bit halves); the caller masks with
// 0x80008000
__device__ __forceinline__ uint32_t sent_bits(uint32_t w) { return (~w - 0x00010001u) & w; }
// AUX_SPLIT: the other CTA's B slot (shared::cluster addresses; xb 0: no split), its instruction's chunks, and its
// mbarrier that this CTA's asynchronous stores complete bytes on (Smem::pfull)
struct PeerB {
  uint32_t xb, bar;
  ChunkRange cr;
};
// RESID: x = bf16(the producing op's validated ACC_CNT output) + the residual (fp32, or bf16 when BF16IN); else x = the
// bf16 input rows. Per token row t < T: its values (loaded a row ahead), the sum of squares, the B rows (xb2: also the
// twin's chunks); this instruction's slice of x (fp32) and of bf16(x) to x_out / r_out
template <int KIND, bool RESID, bool BF16IN, bool BA, bool SPLIT>
__device__ __forceinline__ void pro_norm_rows(const Instr& in, const OpDesc& d, const Smem& sm, int pt,
                                              const uint2 (&wv)[NG5], const SrcAcc& src, uint8_t* xb, uint8_t* sfb,
                                              const ChunkRange& cr2, uint8_t* xb2, const uint2* bw, uint32_t T,
                                              uint32_t& stg_phase, bool is_dry, const PeerB& pb, uint32_t t0_,
                                              uint32_t t1_) {
  constexpr uint32_t K = K_NORM, NQ = K / 4, RES_BYTES = K * (BF16IN ? 2 : 4);
  // the CTA pair's bf16 residual rows (gate_up: at most RES_ROWS of them) all in the staging buffer, copied before the
  // dependency wait (resident_rows): one wait, no copy per row
  constexpr bool ROWS_RESIDENT = RESID && BF16IN && SPLIT;
  asm volatile("" : "+r"(pt));  // (opaque: else ptxas rebuilds it, S2R SR_TID.X and its chain, at every row's uses)
  const float gs = d.gs, eps = d.eps;
  const ChunkRange cr = chunk_range(in, d);
  // (the LM head's x_out is its logits (EPI_LOGITS): no x slice stored there)
  const uint64_t xo_p = RESID && !is_dry && d.epi != EPI_LOGITS ? d.x_out : 0, ro_p = RESID && !is_dry ? d.r_out : 0;
  const float inv_s = KIND == KIND_FP8 ? div_full(1.0f, gs) : 0.f;
  // this instruction's slice (SPLIT: the pair's, their tiles are 2c, 2c + 1); its rows (SPLIT: [t0_, t1_), the dry run
  // row t0_; otherwise every row, a dry run row 0, with no SPLIT code in the instance)
  const uint32_t tl = SPLIT ? in.tile & ~1u : in.tile, tn = SPLIT ? 2u : 1u;
  const uint32_t s0 = NQ * tl / d.n_inst, s1 = NQ * (tl + tn) / d.n_inst;
  const uint32_t t0 = SPLIT ? t0_ : 0u;
  const uint32_t te = SPLIT ? (is_dry ? t0_ + 1 : t1_) : (is_dry ? 1u : T);  // (a dry run: one row, the whole code)
  // AUX_SPLIT, NVFP4 (every chunk in both CTAs' B: schedule.py mark_split): this CTA's rows of B go into the other
  // CTA's B as 16-byte st.async words, not per group (FP8: a word per group); a thread copies one word per row: of
  // chunk pt / PEER_ROW_WORDS, its PEER_CODE_WORDS code groups, then its 2 scale halves (nck <= SFB_CHUNKS)
  constexpr int PEER_CODE_WORDS = 8;  // a chunk's 16-byte code words per token row (then its scale words)
  constexpr int PEER_ROW_WORDS = 10;  // a chunk's 16-byte words per token row
  static_assert(SFB_CHUNKS * PEER_ROW_WORDS <= NPRO, "the copy: a word per thread and row");
  static_assert(!ROWS_RESIDENT || (NT + 1) / 2 * RES_BYTES <= STG_BYTES,
                "the CTA pair's rows' residuals in the staging buffer");
  uint2 ov[NG5];  // the row's producing-op values (RESID) or its bf16 input
  // (this thread's first value of row 0 and the row stride, opaque: else ptxas rebuilds the 64-bit row address in every
  // row for each load)
  const uint8_t* vb = (RESID ? src.acc : reinterpret_cast<const uint8_t*>(d.in0)) + (uint32_t)pt * 8u;
  uint32_t vs = RESID ? src.rs * 2 : K * 2;
  asm volatile("" : "+l"(vb), "+r"(vs));
  auto row_in = [&](uint32_t t) { return reinterpret_cast<const uint2*>(vb + (size_t)t * vs); };
  auto load = [&](uint32_t t) {
    const uint2* gp = row_in(t);
#pragma unroll
    for (int i = 0; i < NG5; i++) ov[i] = RESID ? ld_relaxed_v2(gp + i * NPRO) : __ldcg(gp + i * NPRO);
  };
  // row-invariant: the norm's weights + 1 (fp32), the groups' store offsets (b_off: B, the twin's B, the NVFP4 scales,
  // the other CTA's B)
  float w1[NG5][4];
  int32_t bo[NG5], bo2[NG5], so[NG5], bop[NG5];
#pragma unroll
  for (int i = 0; i < NG5; i++) {
    w1[i][0] = bf16lo(wv[i].x) + 1.0f, w1[i][1] = bf16hi(wv[i].x) + 1.0f;
    w1[i][2] = bf16lo(wv[i].y) + 1.0f, w1[i][3] = bf16hi(wv[i].y) + 1.0f;
    const uint32_t vi = (pt + i * NPRO) * 4;
    bo[i] = b_off<KIND>(vi, cr);
    bo2[i] = KIND == KIND_FP8 && xb2 ? b_off<KIND>(vi, cr2) : -1;
    so[i] = KIND == KIND_NVFP4 ? b_off<KIND>(vi, cr, true) : -1;
    bop[i] = SPLIT && KIND == KIND_FP8 && pb.xb ? b_off<KIND>(vi, pb.cr) : -1;
    // (opaque: else ptxas rebuilds them in every row from pt and the chunk range)
    asm volatile("" : "+r"(bo[i]));
    if (KIND == KIND_FP8 && !SPLIT) asm volatile("" : "+r"(bo2[i]));
    if (KIND == KIND_NVFP4) asm volatile("" : "+r"(so[i]));
    if (SPLIT && KIND == KIND_FP8) asm volatile("" : "+r"(bop[i]));
  }
  // the shared addresses, once (opaque: kept in registers, not rebuilt per access from S2R SR_CgaCtaId): the B buffers,
  // this thread's residual values in the staging buffer, its in_proj_ba weights (BA), the reductions
  uint32_t xbs = smem_u32(xb), sfbs = smem_u32(sfb), xb2s = xb2 ? smem_u32(xb2) : 0u;
  uint32_t rss = smem_u32(sm.stg) + (uint32_t)pt * (BF16IN ? 8u : 16u), bws = BA ? smem_u32(bw) + (uint32_t)pt * 8u : 0u;
  uint32_t reds = smem_u32(sm.red);
  asm volatile("" : "+r"(xbs), "+r"(sfbs), "+r"(xb2s), "+r"(rss), "+r"(bws), "+r"(reds));
  // RESID: the instruction's slice of x (quads [s0, s1) of the CTA's rows) is stashed per row in this B slot past the
  // operand (rst: [row][quad] float4; schedule.py check_prologue: it fits) and stored to x_out / r_out after the rows
  // (one quad per thread), not by the row loop: rso, a group's place in a row's stash (-1: not in the slice)
  const uint32_t ns = s1 - s0;
  uint32_t rst = xbs + (KIND == KIND_NVFP4 ? (uint32_t)(SFB_OFF + SFB_BYTES) : cr.nck * CHUNK_B);
  int32_t rso[NG5];
#pragma unroll
  for (int i = 0; i < NG5; i++) {
    const uint32_t q = pt + i * NPRO;
    rso[i] = RESID && q >= s0 && q < s1 ? (int32_t)((q - s0) * 16) : -1;
    if (RESID) asm volatile("" : "+r"(rso[i]));
  }
  asm volatile("" : "+r"(rst));
  // AUX_SPLIT, NVFP4: a row's words go to the other CTA once the next row's reduction barrier has ordered its stores
  // (the last row's after the rows): this thread's word of every row at {src, the peer's} (0: none), kept per thread in
  // a table past the stash (cpt; no registers held across the rows: they would spill the region)
  const uint32_t cpt = rst + ns * (te - t0) * 16 + (uint32_t)pt * 8;
  if constexpr (SPLIT && KIND == KIND_NVFP4) {
    uint32_t cpo = 0u, cpp = 0u;
    if (pb.xb != 0u && (uint32_t)pt < cr.nck * PEER_ROW_WORDS) {
      const uint32_t l = (uint32_t)pt / PEER_ROW_WORDS, k = (uint32_t)pt % PEER_ROW_WORDS;
      const uint32_t kc = cr.abs(l), lp = chunk_slot(kc, pb.cr);
      const uint32_t o = k < PEER_CODE_WORDS ? k * 128 : SFB_OFF + (k - PEER_CODE_WORDS) * SFB_HALF,
                     ls = k < PEER_CODE_WORDS ? CHUNK_B : SFB_CHUNK;
      cpo = xbs + l * ls + o, cpp = pb.xb + lp * ls + o;
    }
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(cpt), "r"(cpo), "r"(cpp) : "memory");
  }
  if (!SPLIT || t0 < te) load(t0);
#pragma unroll 1
  for (uint32_t t = t0; t < te; t++) {
    if (RESID && !is_dry) {
      if (!ROWS_RESIDENT || t == t0) {
        mbar_wait(sm.stg_full, stg_phase);  // row t's residual (ROWS_RESIDENT: every row's)
        stg_phase ^= 1;
      }
      // values still at the sentinel re-loaded, in rounds (every stale one re-issued each round; rare: out of line)
      // (the sentinel 0xFFFF is the largest 16-bit value: a half of the words' per-half max holds it iff a value does)
      uint32_t mx = max_u16x2(ov[0].x, ov[0].y);
#pragma unroll
      for (int i = 1; i < NG5; i++) mx = max_u16x2(max_u16x2(mx, ov[i].x), ov[i].y);
      uint32_t stale_bits = sent_bits(mx);
      if (__builtin_expect(__any_sync(0xffffffffu, (stale_bits & 0x80008000u) != 0), 0)) {
        const uint2* gp = row_in(t);
#pragma unroll 1
        do {
#pragma unroll
          for (int i = 0; i < NG5; i++)
            if (((sent_bits(ov[i].x) | sent_bits(ov[i].y)) & 0x80008000u) != 0) ov[i] = ld_relaxed_v2(gp + i * NPRO);
          stale_bits = 0;
#pragma unroll
          for (int i = 0; i < NG5; i++) stale_bits |= sent_bits(ov[i].x) | sent_bits(ov[i].y);
        } while (__any_sync(0xffffffffu, (stale_bits & 0x80008000u) != 0));
      }
    }
    float x[NG5][4];
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      if (RESID && BF16IN) {  // (the value's bf16 half + the residual: FHADD.BF16)
        const uint2 r = lds_u2(rss + (ROWS_RESIDENT ? (t - t0) * RES_BYTES : 0u) + i * NPRO * 8);
        x[i][0] = bf16lo_add(ov[i].x, bf16lo(r.x)), x[i][1] = bf16hi_add(ov[i].x, bf16hi(r.x));
        x[i][2] = bf16lo_add(ov[i].y, bf16lo(r.y)), x[i][3] = bf16hi_add(ov[i].y, bf16hi(r.y));
      } else if (RESID) {
        const float4 r = lds_f4(rss + i * NPRO * 16);
        x[i][0] = bf16lo_add(ov[i].x, r.x), x[i][1] = bf16hi_add(ov[i].x, r.y);
        x[i][2] = bf16lo_add(ov[i].y, r.z), x[i][3] = bf16hi_add(ov[i].y, r.w);
      } else {
        x[i][0] = bf16lo(ov[i].x), x[i][1] = bf16hi(ov[i].x), x[i][2] = bf16lo(ov[i].y), x[i][3] = bf16hi(ov[i].y);
      }
    }
    if (RESID && !ROWS_RESIDENT) fence_async_smem();  // (the staging buffer's reads before the next row's copy into it)
    if (t + 1 < te) load(t + 1);  // the next row's values in flight during this row's reduction and encode
    if (RESID) {  // (a dry run's too: the code it warms; the slot is this instruction's)
      const uint32_t ra = rst + (t - t0) * ns * 16;
#pragma unroll
      for (int i = 0; i < NG5; i++)
        if (rso[i] >= 0) sts_f4(ra + rso[i], make_float4(x[i][0], x[i][1], x[i][2], x[i][3]));
    }
    float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < NG5; i++) {  // (s4[k] = fma(x, x, s4[k]) per k, in group order: FFMA2 pairs)
      fma2(s4[0], s4[1], x[i][0], x[i][1], x[i][0], x[i][1]);
      fma2(s4[2], s4[3], x[i][2], x[i][3], x[i][2], x[i][3]);
    }
    float ss[1] = {(s4[0] + s4[1]) + (s4[2] + s4[3])};
    {  // the warps' sums at the shared address, in order (alternate areas by row parity)
      const uint32_t ra = reds + (RED_NORM + (t & 1) * (NPRO / 32)) * 4;
      const float w = warp_sum(ss[0]);
      if ((pt & 31) == 0) sts_f32(ra + (pt >> 5) * 4, w);
      named_sync(BAR_PRO, NPRO);  // (every thread's staging reads before it)
      const float4 a0 = lds_f4(ra), a1 = lds_f4(ra + 16);
      static_assert(NPRO / 32 == 8, "the 8 warps' sums");
      ss[0] = ((((((0.f + a0.x) + a0.y) + a0.z) + a0.w) + a1.x) + a1.y + a1.z) + a1.w;
    }
    if (SPLIT && KIND == KIND_NVFP4 && t > t0) {  // (row t - 1's stores: before the barrier above)
      const uint2 c = lds_u2(rst + ns * (te - t0) * 16 + (uint32_t)pt * 8);
      const uint32_t ro = (t - 1) * 16;
      if (c.y) st_async_v4(c.y + ro, lds_u4(c.x + ro), pb.bar);
    }
    // the next row's residual into the staging buffer while this row is encoded (row t0's: before the dependency wait)
    if (RESID && !ROWS_RESIDENT && !is_dry && pt == RESID_COPY_PT && t + 1 < te) {
      mbar_expect_tx(sm.stg_full, RES_BYTES);
      bulk_g2s(sm.stg, reinterpret_cast<const uint8_t*>(d.res_in) + (size_t)(t + 1) * RES_BYTES, RES_BYTES, sm.stg_full);
    }
    static_assert(K == 5120, "rms_rsqrt5120");
    const float rs = rms_rsqrt5120(ss[0], eps);
    float sb = 0.f, sa = 0.f;
    // the row's groups: first every group's codes in registers (independent chains: no stores, no branches between
    // them), then their stores (BF16, layer 0's in_proj_ba: enc4 at once)
    const float rgs = KIND == KIND_NVFP4 ? rcp_approx_ftz(gs) : 0.f;
    uint32_t cw[NG5], cs[NG5];
    float v[NG5][4];
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      const uint32_t q = pt + i * NPRO;
#pragma unroll
      for (int k = 0; k < 4; k += 2) {  // (x * rs) * (1 + w): FMUL2 pairs
        v[i][k] = x[i][k], v[i][k + 1] = x[i][k + 1];
        mul2(v[i][k], v[i][k + 1], rs, rs);
        mul2(v[i][k], v[i][k + 1], w1[i][k], w1[i][k + 1]);
      }
      // BA (in_proj_qkv): the dot products of bf16(v) with the two in_proj_ba rows of this SM's GDN item (bw: b's row,
      // then a's, in the staging buffer past the fp32 residual row, bulk-copied when the instruction starts), reduced
      // after the rows into sm.ba (b [NT], a [NT])
      if (BA) {  // in_proj_ba's input is bf16(v) (before the encode: NVFP4's rounds v in place)
        const uint2 wb = lds_u2(bws + i * NPRO * 8), wa = lds_u2(bws + NQ * 8 + i * NPRO * 8);
        // (bf16(v) packed as rounded, then each product of two bf16 halves into its chain: FHFMA.BF16, no unpacking)
        const __nv_bfloat162 h01 = __floats2bfloat162_rn(v[i][0], v[i][1]), h23 = __floats2bfloat162_rn(v[i][2], v[i][3]);
        const uint32_t p01 = *reinterpret_cast<const uint32_t*>(&h01), p23 = *reinterpret_cast<const uint32_t*>(&h23);
        sb = bf16lo_fma(wb.x, p01, sb), sb = bf16hi_fma(wb.x, p01, sb), sb = bf16lo_fma(wb.y, p23, sb), sb = bf16hi_fma(wb.y, p23, sb);
        sa = bf16lo_fma(wa.x, p01, sa), sa = bf16hi_fma(wa.x, p01, sa), sa = bf16lo_fma(wa.y, p23, sa), sa = bf16hi_fma(wa.y, p23, sa);
      }
      if (KIND == KIND_BF16) enc4<KIND>(gs, inv_s, q * 4, t, v[i], cr, xb, sfb, cr2, xb2);
    }
    if (KIND != KIND_BF16) enc_codes_rows<KIND>(gs, rgs, inv_s, v, cw, cs);
    if (KIND != KIND_BF16)
#pragma unroll
      for (int i = 0; i < NG5; i++) {  // (at the offsets computed once: + 16 bytes per token row)
        if (KIND == KIND_NVFP4) {
          if (bo[i] >= 0) sts_u16(xbs + bo[i] + t * 16, cw[i]);
          if (so[i] >= 0) sts_u8(sfbs + so[i] + t * 16, cs[i]);
        } else {
          if (bo[i] >= 0) sts_u32(xbs + bo[i] + t * 16, cw[i]);
          if (bo2[i] >= 0) sts_u32(xb2s + bo2[i] + t * 16, cw[i]);
          if (SPLIT && bop[i] >= 0) st_async_b32(pb.xb + bop[i] + t * 16, cw[i], pb.bar);
        }
      }
    if (BA) {  // the row's warp sums, summed over the warps in order after the last row
      warp_sum2(sb, sa);
      if ((pt & 31) == 0) {
        sts_f32(reds + (RED_BA + (t * 2) * (NPRO / 32) + (pt >> 5)) * 4, sb);
        sts_f32(reds + (RED_BA + (t * 2 + 1) * (NPRO / 32) + (pt >> 5)) * 4, sa);
      }
    }
  }
  bool rows_synced = false;  // (one barrier after the rows for the copy, the stash, BA's sums)
  if constexpr (SPLIT && KIND == KIND_NVFP4) {
    if (pb.xb && te > t0) {
      named_sync(BAR_PRO, NPRO);  // (every warp's stores of the rows)
      rows_synced = true;
      uint32_t w = pt;
      asm volatile("" : "+r"(w));  // (the copy's addresses computed here: above the rows, they would spill the region)
      if (w < cr.nck * PEER_ROW_WORDS) {  // (the last row only: the loop above copies the others)
        const uint32_t l = w / PEER_ROW_WORDS, k = w % PEER_ROW_WORDS;
        const uint32_t kc = cr.abs(l), lp = chunk_slot(kc, pb.cr);
        const uint32_t o = k < PEER_CODE_WORDS ? k * 128 : SFB_OFF + (k - PEER_CODE_WORDS) * SFB_HALF,
                       ls = k < PEER_CODE_WORDS ? CHUNK_B : SFB_CHUNK;
#pragma unroll 1
        for (uint32_t r = te - 1; r < te; r++)
          st_async_v4(pb.xb + lp * ls + o + r * 16, *reinterpret_cast<const uint4*>(xb + l * ls + o + r * 16), pb.bar);
      }
    }
  }
  const bool has_stash = RESID && (xo_p | ro_p) != 0;  // (not a dry run)
  if ((BA || has_stash) && !rows_synced) named_sync(BAR_PRO, NPRO);
  if (has_stash) {  // the stashed slice: quad s0 + j % ns of row t0 + j / ns per thread, as the rows stored it before
    const uint32_t n = ns * (te - t0);
#pragma unroll 1
    for (uint32_t j = pt; j < n; j += NPRO) {
      const uint32_t r = j / ns, q = s0 + (j - r * ns);
      const float4 x4 = lds_f4(rst + j * 16);
      if (xo_p) reinterpret_cast<float4*>(xo_p + (size_t)(t0 + r) * K * 4)[q] = x4;
      if (ro_p) {
        const __nv_bfloat162 h0 = __floats2bfloat162_rn(x4.x, x4.y), h1 = __floats2bfloat162_rn(x4.z, x4.w);
        reinterpret_cast<uint2*>(ro_p + (size_t)(t0 + r) * K * 2)[q] =
            make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
      }
    }
  }
  if (BA) {
    // (b / a of row pt / 2): bf16, as in_proj_ba's output; not by a dry run: sm.ba lies inside a's row in the
    // staging buffer, whose bulk copy the dry run does not wait for
    if (!is_dry && pt < (int)(2 * te)) {  // (BA: t0 = 0)
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NPRO / 32; w++) s += sm.red[RED_BA + pt * (NPRO / 32) + w];
      sm.ba[(pt & 1) * NT + pt / 2] = bf16_round(s);
    }
  }
}

// in_proj_qkv's b / a of this SM's item (head hv) from the producer's partials (shared_xn's bit 0; [2][40][96][NT]
// fp32, sentinel ~0u): threads pt < 160 one 16-byte word each (4 token rows of b or a of a tile, re-polled) into scr
// [2][NT][40] (in_proj_ba's region, free with the partials); pro_ba_pub's 2 T output threads sum the 40 tiles in order
__device__ __forceinline__ void pro_ba_part(uint64_t xn_base, float* scr, float* ba, int pt, uint32_t hv, uint32_t T,
                                            uint32_t epoch, bool is_dry, uint32_t factors_slot, uint32_t ftag,
                                            uint32_t fac) {
  constexpr uint32_t BA_PART_POLL_NS = 64;  // the backoff between two polls of a partial still at its sentinel
  const bool uses_factors = !is_dry && ftag != 0u;
  constexpr uint32_t MT = K_NORM / 128;
  const uint32_t sel = (uint32_t)pt / (2 * MT), r = (uint32_t)pt % (2 * MT), mt = r >> 1, half = r & 1u;
  if (pt < (int)(4 * MT) && (half == 0 || T > 4)) {
    const uint32_t* const p = reinterpret_cast<const uint32_t*>(xn_base) +
                              (((size_t)(epoch & 1) * MT + mt) * BA_OUT + (sel ? GDN_HV + hv : hv)) * NT + 4 * half;
    uint4 w = ld_relaxed_v4(p);
    if (!is_dry) {
      const uint32_t nt = min(4u, T - 4 * half);  // (rows past T: never written)
#pragma unroll 1
      for (;;) {
        const bool is_stale =
            (w.x == ~0u) || (nt > 1 && w.y == ~0u) || (nt > 2 && w.z == ~0u) || (nt > 3 && w.w == ~0u);
        if (!is_stale) break;
        __nanosleep(BA_PART_POLL_NS);
        w = ld_relaxed_v4(p);
      }
      float* const o = scr + (sel * NT + 4 * half) * MT + mt;
      o[0] = __uint_as_float(w.x), o[MT] = __uint_as_float(w.y), o[2 * MT] = __uint_as_float(w.z), o[3 * MT] = __uint_as_float(w.w);
    }
  }
  named_sync(BAR_PRO, NPRO);
  if (!is_dry && pt < (int)(2 * T)) {  // (b / a of row pt / 2): bf16, as in_proj_ba's output
    const float* const i = scr + ((pt & 1) * NT + pt / 2) * MT;
    float s = 0.f;
#pragma unroll 8
    for (uint32_t k = 0; k < MT; k++) s += i[k];
    if (uses_factors) {
      shared_b_wait_tag(factors_slot, ftag);
      float fv;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(fv) : "r"(fac + (uint32_t)(pt / 2) * 4));
      s = __fmul_rn(s, fv);
    }
    ba[(pt & 1) * NT + pt / 2] = bf16_round(s);
  }
  if (!is_dry) named_sync(BAR_PRO, NPRO);  // (sm.ba complete before the GDN core reads it: pro_ba_pub's ordering)
}
constexpr uint32_t BA_T_HEAD_SHIFT = 8;     // pro_ba_pub's T argument: the item's value head from this bit
constexpr uint32_t BA_T_ROWS_MASK = 0xffu;  // pro_ba_pub's T argument: the token rows T
// pro_ba_pub's T argument with the b / a partials: T, the item's value head in bits 8..
__device__ __forceinline__ uint32_t ba_t(const Instr& in, uint32_t T) {
  return T | (min(in.tile, (uint32_t)GDN_ITEMS - 1) / 3 << BA_T_HEAD_SHIFT);
}
// shared B, in_proj_qkv as a PRO_ACT consumer (OpDesc::xn): b / a of this SM's GDN item from the published bf16(x_n)
// ([2][NT][K], sentinel 0xFFFF; xn_base: copy 0) as pro_norm_rows' BA: per thread its 5 quads q = pt + 256 i
// (FHFMA.BF16 with in_proj_ba's rows at bw), the warps' sums, the 8 warps in order into sm.ba (b [NT], a [NT]) as bf16
constexpr uint32_t SHARED_XN_STRIDE = NT * K_NORM * 2;  // bf16(x_n)'s launch-parity stride
// the producer's bf16(x_n) buffer (OpDesc::shared_xn; 0: none)
__device__ __forceinline__ uint64_t shared_xn_of(const OpDesc& prod) { return prod.shared_xn; }
// (is_dry: the dry run's, one row, nothing re-polled, sm.ba not written: it overlays in_proj_ba's a row, landing then)
// (factors_slot, ftag: the shared B's factors, epilogue warp 0's at fac once factors_slot holds ftag; ftag 0: none)
__device__ __noinline__ void pro_ba_pub(uint64_t xn_base, float* red, float* ba, int pt, const uint2* bw, uint32_t T,
                                     uint32_t epoch, bool is_dry, uint32_t factors_slot, uint32_t ftag, uint32_t fac) {
  if (xn_base & 1u) {  // (the producer's finishers stored the b / a partials; T's bits 8.. the head)
    pro_ba_part(xn_base & ~(uint64_t)1, reinterpret_cast<float*>(const_cast<uint2*>(bw)), ba, pt, T >> BA_T_HEAD_SHIFT,
                T & BA_T_ROWS_MASK, epoch, is_dry, factors_slot, ftag, fac);
    return;
  }
  T &= BA_T_ROWS_MASK;
  const bool uses_factors = !is_dry && ftag != 0u;
  constexpr uint32_t K = K_NORM, NQ = K / 4;
  constexpr uint32_t XN_POLL_NS = 64;  // the backoff between two re-poll rounds of the rows still at the sentinel
  constexpr int RB = BARB;  // rows whose loads are in flight together
  const uint8_t* xn = reinterpret_cast<const uint8_t*>(xn_base) + (epoch & 1) * SHARED_XN_STRIDE + (uint32_t)pt * 8u;
  const uint32_t bws = smem_u32(bw) + (uint32_t)pt * 8u, reds = smem_u32(red);
#pragma unroll 1
  for (uint32_t t0 = 0; t0 < T; t0 += RB) {
    uint2 ov[RB][NG5];
#pragma unroll
    for (int r = 0; r < RB; r++)
#pragma unroll
      for (int i = 0; i < NG5; i++) {
        ov[r][i] = make_uint2(0u, 0u);
        if (t0 + r < T) ov[r][i] = ld_relaxed_v2(xn + (size_t)(t0 + r) * K * 2 + i * NPRO * 8);
      }
#pragma unroll
    for (int r = 0; r < RB; r++) {
      const uint32_t t = t0 + r;
      if (t >= T) break;
      uint32_t stale_bits = 0;
#pragma unroll
      for (int i = 0; i < NG5; i++) stale_bits |= sent_bits(ov[r][i].x) | sent_bits(ov[r][i].y);
      if (!is_dry && __builtin_expect(__any_sync(0xffffffffu, (stale_bits & 0x80008000u) != 0), 0)) {
#pragma unroll 1
        do {
#pragma unroll
          for (int i = 0; i < NG5; i++)
            if (((sent_bits(ov[r][i].x) | sent_bits(ov[r][i].y)) & 0x80008000u) != 0)
              ov[r][i] = ld_relaxed_v2(xn + (size_t)t * K * 2 + i * NPRO * 8);
          __nanosleep(XN_POLL_NS);
          stale_bits = 0;
#pragma unroll
          for (int i = 0; i < NG5; i++) stale_bits |= sent_bits(ov[r][i].x) | sent_bits(ov[r][i].y);
        } while (__any_sync(0xffffffffu, (stale_bits & 0x80008000u) != 0));
      }
      float sb = 0.f, sa = 0.f;
#pragma unroll
      for (int i = 0; i < NG5; i++) {
        const uint2 wb = lds_u2(bws + i * NPRO * 8), wa = lds_u2(bws + NQ * 8 + i * NPRO * 8);
        const uint32_t p01 = ov[r][i].x, p23 = ov[r][i].y;
        sb = bf16lo_fma(wb.x, p01, sb), sb = bf16hi_fma(wb.x, p01, sb), sb = bf16lo_fma(wb.y, p23, sb), sb = bf16hi_fma(wb.y, p23, sb);
        sa = bf16lo_fma(wa.x, p01, sa), sa = bf16hi_fma(wa.x, p01, sa), sa = bf16lo_fma(wa.y, p23, sa), sa = bf16hi_fma(wa.y, p23, sa);
      }
      warp_sum2(sb, sa);
      if ((pt & 31) == 0) {
        sts_f32(reds + (RED_BA + (t * 2) * (NPRO / 32) + (pt >> 5)) * 4, sb);
        sts_f32(reds + (RED_BA + (t * 2 + 1) * (NPRO / 32) + (pt >> 5)) * 4, sa);
      }
    }
  }
  named_sync(BAR_PRO, NPRO);
  if (!is_dry && pt < (int)(2 * T)) {  // (b / a of row pt / 2): bf16, as in_proj_ba's output
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NPRO / 32; w++) s += red[RED_BA + pt * (NPRO / 32) + w];
    // (shared B: bf16(x * (1 + w) * c_t) published) b / a times f_t = rstd_t / c_t before their bf16 rounding: the
    // epilogue's factors (computed from B ready, while these rows load)
    if (uses_factors) {
      shared_b_wait_tag(factors_slot, ftag);
      float fv;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(fv) : "r"(fac + (uint32_t)(pt / 2) * 4));
      s = __fmul_rn(s, fv);
    }
    ba[(pt & 1) * NT + pt / 2] = bf16_round(s);
  }
  // sm.ba complete before any prologue thread leaves: the GDN core, next on these warps, reads it (gdn_t.cuh
  // GT_BA_SMEM: warp 1) with no barrier of its own before; its writers (pt < 2 T) may wait for the epilogue's factors
  if (!is_dry) named_sync(BAR_PRO, NPRO);
}

// The instruction's activation as the MMA's B operand (xb, plus NVFP4 scales in sfb), T token rows. wv (the RMSNorm's
// weights, or PRO_GDN_NORM's in wv[0..3]) are loaded before the dependency wait; bw: BA's in_proj_ba rows.
// PRO_ACT: staged already; PRO_COPY_B: written by the previous instruction's prologue (its twin).
__device__ __forceinline__ void prologue(const Instr& in, const OpDesc& d, const Smem& sm, int pt, const uint2 (&wv)[NG5],
                                         const SrcAcc& src, uint8_t* xb, uint8_t* sfb, const ChunkRange& cr2,
                                         uint8_t* xb2, const uint2* bw, uint32_t T, uint32_t& stg_phase,
                                         bool is_dry, uint32_t ll_token, const PeerB& pb, uint32_t t0, uint32_t t1) {
  if (d.pro == PRO_GDN_NORM) {  // FLA's gated RMSNorm per head of 128: y = (x * rstd) * w * (z * sigmoid(z)), bf16
    // block b = 16 values: token row b / (nck * 8), head (b / 8) % nck, its block b % 8 (a head: 8 consecutive lanes;
    // wv[0..3]: block pt % 8's norm weights); the core output's tagged words and the gate (in_proj_z's words, or the
    // qkvz sums: final once the core has its inputs) loaded at once, stale ones re-polled in rounds (not when dry)
    const ChunkRange cr = chunk_range(in, d);
    const float inv_s = div_full(1.0f, d.gs);
    const uint32_t tok16 = ll_token & 0xffffu, per_row = cr.nck * 8, nb = (is_dry ? 1u : T) * per_row;
    float w[16];
    unpack16_bf16(make_uint4(wv[0].x, wv[0].y, wv[1].x, wv[1].y), make_uint4(wv[2].x, wv[2].y, wv[3].x, wv[3].y), w);
#pragma unroll 1
    for (uint32_t b0 = 0; b0 < nb; b0 += NPRO) {
      const uint32_t b = b0 + pt;
      const bool in_range = b < nb;
      const uint32_t t = in_range ? b / per_row : 0u, j = (b / 8) % cr.nck, k = b % 8;
      const uint32_t hk = cr.kc0 + j < cr.spm ? cr.kc0 + j : cr.kc0 + j - cr.spm;  // the head
      float x[16], z[16], ss = 0.f;  // (defined on every path: a register defined under a condition only stays live
#pragma unroll                        // around the prologue's whole instruction loop)
      for (int i = 0; i < 16; i++) x[i] = z[i] = 0.f;
      if (in_range) {
        const uint64_t* lp = reinterpret_cast<const uint64_t*>(d.in0) + (size_t)t * (GDN_HV * GDN_V / 2) + hk * 64 + k * 8;
        uint64_t wd[8];
#pragma unroll
        for (int i = 0; i < 8; i++) wd[i] = ld_relaxed_b64(lp + i);
        if (d.gate_ll) {
          const uint32_t* zp = reinterpret_cast<const uint32_t*>(d.res_in) + (size_t)t * (GDN_HV * GDN_V) + hk * 128 + k * 16;
          uint4 zv[4];
#pragma unroll
          for (int i = 0; i < 4; i++) zv[i] = ld_relaxed_v4(zp + 4 * i);
          if (!is_dry) {
#pragma unroll 1
            for (;;) {
              bool all_tagged = true;
#pragma unroll
              for (int i = 0; i < 4; i++)
                if ((zv[i].x >> 16) != tok16 || (zv[i].y >> 16) != tok16 || (zv[i].z >> 16) != tok16 ||
                    (zv[i].w >> 16) != tok16) {
                  all_tagged = false;
                  zv[i] = ld_relaxed_v4(zp + 4 * i);
                }
              if (all_tagged) break;
            }
          }
#pragma unroll
          for (int i = 0; i < 4; i++) {
            z[4 * i] = bf16lo(zv[i].x), z[4 * i + 1] = bf16lo(zv[i].y);
            z[4 * i + 2] = bf16lo(zv[i].z), z[4 * i + 3] = bf16lo(zv[i].w);
          }
        } else {
          const float4* zp = reinterpret_cast<const float4*>(
              reinterpret_cast<const float*>(src.acc) + (size_t)t * src.rs + GDN_CONV_DIM + hk * 128 + k * 16);
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const float4 zz = __ldcg(zp + i);
            z[4 * i] = bf16_round(src.alpha * zz.x), z[4 * i + 1] = bf16_round(src.alpha * zz.y);
            z[4 * i + 2] = bf16_round(src.alpha * zz.z), z[4 * i + 3] = bf16_round(src.alpha * zz.w);
          }
        }
        if (!is_dry) {
#pragma unroll 1
          for (;;) {
            bool all_tagged = true;
#pragma unroll
            for (int i = 0; i < 8; i++)
              if ((uint32_t)(wd[i] >> 32) != ll_token) {
                all_tagged = false;
                wd[i] = ld_relaxed_b64(lp + i);
              }
            if (all_tagged) break;
          }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          x[2 * i] = bf16lo((uint32_t)wd[i]);
          x[2 * i + 1] = bf16hi((uint32_t)wd[i]);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) ss = fmaf(x[i], x[i], ss);
      }
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (in_range) {
        const float rstd = rsqrt_tri(ss / 128.0f + d.eps);
        uint32_t c[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float q[4];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float zz = z[4 * i + e];
            const float v = bf16_round(((x[4 * i + e] * rstd) * w[4 * i + e]) * (zz * sigmoid_triton(zz)));
            q[e] = fminf(fmaxf(v * inv_s, -448.f), 448.f);
          }
          c[i] = f32x2_to_e4m3x2(q[0], q[1]) | (f32x2_to_e4m3x2(q[2], q[3]) << 16);
        }
        *reinterpret_cast<uint4*>(b_addr(xb, j, k * 16, t)) = make_uint4(c[0], c[1], c[2], c[3]);
      }
    }
  } else if (d.pro == PRO_NORM_RESID) {
    if (d.kind == KIND_NVFP4) {
      // (AUX_SPLIT: the CTA pair's instance; the other has no SPLIT code)
      if (d.res_bf16) {
        if (in.aux & AUX_SPLIT)
          pro_norm_rows<KIND_NVFP4, true, true, false, true>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T,
                                                             stg_phase, is_dry, pb, t0, t1);
        else
          pro_norm_rows<KIND_NVFP4, true, true, false, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T,
                                                              stg_phase, is_dry, pb, t0, t1);
      } else {
        if (in.aux & AUX_SPLIT)
          pro_norm_rows<KIND_NVFP4, true, false, false, true>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T,
                                                              stg_phase, is_dry, pb, t0, t1);
        else
          pro_norm_rows<KIND_NVFP4, true, false, false, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T,
                                                               stg_phase, is_dry, pb, t0, t1);
      }
    } else if (d.xn) {
      pro_norm_rows<KIND_FP8, true, false, true, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T, stg_phase,
                                                        is_dry, pb, t0, t1);
    } else {
      if (in.aux & AUX_SPLIT)
        pro_norm_rows<KIND_FP8, true, false, false, true>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T, stg_phase,
                                                          is_dry, pb, t0, t1);
      else
        pro_norm_rows<KIND_FP8, true, false, false, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T, stg_phase,
                                                           is_dry, pb, t0, t1);
    }
  } else if (d.pro == PRO_NORM_IN) {
    if (d.kind == KIND_FP8)
      pro_norm_rows<KIND_FP8, false, true, false, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T, stg_phase,
                                                         is_dry, pb, t0, t1);
    else
      pro_norm_rows<KIND_BF16, false, true, false, false>(in, d, sm, pt, wv, src, xb, sfb, cr2, xb2, bw, T, stg_phase,
                                                          is_dry, pb, t0, t1);
  }
}


// zero this instruction's slice of the op's next-parity accumulator copy
template <int NTH = NPRO>  // (pt: 0 .. NTH - 1)
__device__ __forceinline__ void zero_next_acc(const Instr& in, const OpDesc& d, int pt, uint32_t epoch) {
  const uint32_t n16 = d.acc_bytes / 16;
  uint4* p = reinterpret_cast<uint4*>(acc_copy(d, epoch + 1));
  // (32-bit products and divisions: an op's 16-byte words times its instruction count stay below 2^32; a 64-bit
  // division by a runtime divisor is a slow-path routine, on the path of the GEMVs that start right after a non-GEMV
  // instruction: o_proj, out_proj)
  const uint32_t i0 = n16 * in.tile / d.n_inst, i1 = n16 * (in.tile + 1) / d.n_inst;
  for (uint32_t i = i0 + pt; i < i1; i += NTH) p[i] = make_uint4(0, 0, 0, 0);
  if (d.val) {  // the next launch's output copy back to the sentinel (read by no one in this launch)
    const uint32_t m16 = NT * d.mtiles * 128 * 2 / 16;  // [NT][mtiles * 128] bf16
    uint4* o = reinterpret_cast<uint4*>(cnt_out(d, epoch + 1));
    const uint32_t j0 = m16 * in.tile / d.n_inst, j1 = m16 * (in.tile + 1) / d.n_inst;
    for (uint32_t j = j0 + pt; j < j1; j += NTH) o[j] = make_uint4(~0u, ~0u, ~0u, ~0u);
  }
}


// The spec sampler's phase ph (0 .. 2: A .. C) of an item on the prologue warps, its scratch at base: the phases'
// arguments (in smem: no registers held across them), SsSmem; phase A0 first with a0. Not inlined: its code lies
// outside mk_main's (inlined, it would move the GDN core's code, whose speed depends on its placement)
__device__ __noinline__ void ss_phase(const SsDesc& sd, uint32_t T, uint8_t* base, int pt, uint32_t ph, uint32_t item,
                                      bool a0, bool is_dry) {
  SsArgs& sa = *reinterpret_cast<SsArgs*>(base);
  SsSmem<NPRO>& ssh = *reinterpret_cast<SsSmem<NPRO>*>(base + SS_ARGS_BYTES);
  if (pt == 0) sa = ss_args(sd, (int)T);
  named_sync(BAR_PRO, NPRO);
  const LpGrp<NPRO> gp{pt, BAR_PRO};
  if (a0) {
    ss_phase_a0(gp, ssh, sa, (int)item);
    named_sync(BAR_PRO, NPRO);
  }
  if (ph == 0) ss_phase_a(gp, ssh, sa, (int)item, is_dry);
  else if (ph == 1) ss_phase_b(gp, ssh, sa, (int)item, (int)sd.slices, is_dry);
  else ss_phase_c(gp, ssh, sa, (int)sd.slices, is_dry);
  named_sync(BAR_PRO, NPRO);  // (the group's smem free for the next phase)
}

// ------------------------------------------------------------------------------------------------ the draft's phase
constexpr int BAR_DRAFT = 15;                         // (the role warps: 6 and 7 idle)
constexpr int DRAFT_SYNC_THREADS = NTHREADS - 2 * 32;
constexpr uint32_t DRAFT_SH_OFF = (dk::BBUF_BYTES + 15) / 16 * 16;
constexpr uint32_t DRAFT_ARGS_OFF = (DRAFT_SH_OFF + (uint32_t)sizeof(dk::DraftShared) + 15) / 16 * 16;
static_assert(DREG_EPI >= dk::REG_EPI && DREG_HI >= dk::REG_HI && REG_LO >= dk::REG_LO, "the draft's registers");
// the draft's B buffer past its own ring (dk::NSTAGE <= NSTAGE stages: over the target's later ring slots, its B
// buffers and the staging buffer)
constexpr uint32_t DRAFT_XB_OFF = dk::NSTAGE * STAGE_BYTES;
// the spec sampler's phases in the draft's lists right after its fc (schedule.py attach_draft's ss_in_draft), so that
// the fc's weights stream while the workers run them: their scratch in the draft's B buffer past the fc's B tiles
// (step.py checks the fc's end against SS_DRAFT_OFF)
constexpr uint32_t SS_DRAFT_OFF = ((uint32_t)dk::BBUF_BYTES - (uint32_t)(SS_ARGS_BYTES + sizeof(SsSmem<NPRO>))) / 1024 * 1024;
static_assert(SS_DRAFT_OFF == 55296, "step.py SS_DRAFT_OFF");
// whether this SM's draft list holds the sampler's phases (after its first instruction, the fc)
__device__ __forceinline__ bool ss_in_draft(const MKArgs& a, uint32_t d1, uint32_t d_end) {
  return d1 + 1 < d_end && a.instrs[d1 + 1].op == OP_SS_A;
}
// the target's ring or accumulator ring differs from the draft's: their barriers re-initialized at the transition
constexpr bool DRAFT_REINIT = dk::NSTAGE != NSTAGE || NACC != 2u;
static_assert(DRAFT_XB_OFF + DRAFT_ARGS_OFF + sizeof(dk::DkArgs) <= (uint32_t)(NSTAGE * STAGE_BYTES + B_AREA_BYTES + STG_BYTES) &&
              B_SLOT_BYTES % 1024 == 0, "the draft's shared memory past its ring");
static_assert(dk::NSTAGE <= NSTAGE && dk::STAGE_STRIDE == STAGE_BYTES && dk::STAGE_BYTES <= STAGE_BYTES, "one ring");
static_assert(dk::T_COLS <= T_COLS && dk::NEPI == 4 * 32 && dk::NWT == NPRO, "TMEM, the epilogue and the workers");
static_assert(sizeof(dk::Instr) == sizeof(Instr) && sizeof(dk::GemvDesc) == sizeof(OpDesc) && DK_OP_BASE > OP_PREV,
              "one instruction list and descriptor table");
static_assert(BAR_DRAFT > BAR_GDNT && dk::BAR_EPI == BAR_EPI, "named barriers");

// the mbarriers past the ring's and the accumulators': bars[3 * NSTAGE + 2 * NACC + MBAR_*], xfull[2] the first
constexpr int MBAR_XEMPTY = 2;     // xempty[2]
constexpr int MBAR_STG_FULL = 4;   // stg_full
constexpr int MBAR_STG2_FULL = 5;  // stg2_full
constexpr int MBAR_PFULL = 6;      // pfull[2]
constexpr int MBAR_PFREE = 8;      // pfree[2]
constexpr int MBAR_TAIL = 10;      // all of them
// the draft's view of this CTA's shared memory and its arguments (in shared memory once draft_setup has run)
struct DraftCtx {
  dk::Smem sm;
  const dk::DkArgs* a;
};
// (from the shared arrays themselves, not from the roles' Smem: nothing kept live across the verify's loops)
__device__ __forceinline__ DraftCtx draft_ctx(uint8_t* smem_raw, uint64_t* bars, const uint32_t& tmem_slot,
                                              volatile uint32_t* epi_count) {
  DraftCtx c;
  uint8_t* xb = smem_raw + DRAFT_XB_OFF;
  reinterpret_cast<dk::DraftShared*>(xb + DRAFT_SH_OFF)->bind(c.sm);
  c.sm.stages = smem_raw;
  c.sm.bbuf = xb;
  c.sm.full = bars;
  c.sm.empty = bars + NSTAGE;
  c.sm.dfull = bars + 3 * NSTAGE;
  c.sm.dempty = bars + 3 * NSTAGE + NACC;
  c.sm.epi_rel = epi_count;
  c.sm.tmem = *reinterpret_cast<const volatile uint32_t*>(&tmem_slot);
  c.a = reinterpret_cast<const dk::DkArgs*>(xb + DRAFT_ARGS_OFF);
  return c;
}
// the role warps' barrier (every warp of them, convergent)
__device__ __forceinline__ void draft_sync() { named_sync(BAR_DRAFT, DRAFT_SYNC_THREADS); }
// the draft's shared state (one warp, between the two barriers): DraftShared's barriers, the arguments, the draft's
// epoch (read once per CTA, counted for its advance at the kernel's end), the step's counters the draft reads
__device__ __forceinline__ void draft_setup(const MKArgs& a, uint8_t* smem_raw, int lane, const volatile uint32_t* host_epoch,
                                            volatile uint32_t* d_epoch, volatile uint32_t* epi_count, bool ss_draft) {
  uint8_t* xb = smem_raw + DRAFT_XB_OFF;
  const uint64_t* src = reinterpret_cast<const uint64_t*>(a.draft);
  uint64_t* dst = reinterpret_cast<uint64_t*>(xb + DRAFT_ARGS_OFF);
  for (uint32_t i = lane; i < sizeof(dk::DkArgs) / 8; i += 32) dst[i] = __ldg(src + i);
  reinterpret_cast<dk::DraftShared*>(xb + DRAFT_SH_OFF)->init(lane, 32);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  // the epoch and the two waits on lanes of their own: their round trips overlap (the barrier after the setup orders
  // every role after each lane's acquire)
  const uint32_t epoch = *host_epoch;
  if (lane == 0) {
    *epi_count = 0;  // (the gate word counts the draft's instructions from its first: dk_roles.cuh's numbering)
    uint32_t* de = a.draft->epoch;
    *d_epoch = ld_acquire(de);
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(de + 1) : "memory");
  } else if (lane == 1) {
    wait_counter(a.counters + a.draft_aux, (epoch + 1) * a.ctr_count[a.draft_aux]);
  } else if (lane == 2 && !ss_draft) {
    wait_counter(a.counters + a.draft_sampler, (epoch + 1) * a.ctr_count[a.draft_sampler]);
  } else if (lane == 3 && a.draft->tstamp) {  // (profiling: the draft's frame, its start; the SM clock over the draft)
    const dk::Trace frame{nullptr, 0, a.draft->tstamp + (((size_t)blockIdx.x * dk::NROLE + dk::R_SF) * (dk::MAX_INSTR + 1) + dk::MAX_INSTR) * dk::NSUB};
    frame.stamp(1), frame.put(4, clock64());
  }
  __syncwarp();
}
// the roles' state at the transition, through shared memory (nothing of the verify's loops is kept live into the
// draft's, whose registers then follow its own code): the ring and accumulator positions (the MMA warp's, equal in
// every role at the end of the verify's instructions), the epoch
struct DraftHandoff {
  uint32_t cc[3], seg[3], epoch, host_epoch;  // (epoch: the draft's; host_epoch: this launch's)
  uint32_t xg_sp;  // (the MMA warp's B-buffer position at the target's end, slot | phase << 8: mma_drain)
};
__device__ __forceinline__ void handoff_put(volatile uint32_t* w, const RingPos& p) { w[0] = p.n, w[1] = p.slot, w[2] = p.phase; }
__device__ __forceinline__ dk::RingPos handoff_get(const volatile uint32_t* w) {
  dk::RingPos q;
  q.n = w[0], q.slot = w[1], q.phase = w[2];
  return q;
}
// DRAFT_REINIT (one warp, between the transition's barriers): the target's last ring positions' MMA commits waited on
// the empty barriers (the full and dfull ones completed before), then the ring and accumulator barriers re-initialized
// for the draft's rings and the handed-off positions zeroed
__device__ __forceinline__ void draft_ring_reset(uint64_t* bars, DraftHandoff& h, int lane) {
  if (lane == 0) {
    const uint32_t n = h.cc[0];
    for (uint32_t p = n > (uint32_t)NSTAGE ? n - NSTAGE : 0u; p < n; p++)
      mbar_wait(&bars[NSTAGE + p % NSTAGE], (p / NSTAGE) & 1);
    for (int s = 0; s < NSTAGE; s++) {
      mbar_init(&bars[s], 1);
      mbar_init(&bars[NSTAGE + s], 1);
    }
    for (int b = 0; b < (int)NACC; b++) {
      mbar_init(&bars[3 * NSTAGE + b], 1);
      mbar_init(&bars[3 * NSTAGE + NACC + b], 4);
    }
    for (int i = 0; i < 3; i++) h.cc[i] = 0u, h.seg[i] = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncwarp();
}

// ------------------------------------------------------------------------------------------------ kernel
// a cluster of 2 CTAs (the CTA pair of AUX_SPLIT; 76 clusters of 2 on GB300's 152 SMs: every SM)
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NTHREADS, 1) mk_main(const __grid_constant__ MKArgs a_param,
                                                                             const __grid_constant__ RecParam rp) {
  // the launch arguments in smem: referenced throughout (a local copy would live in local memory, ~L2 latency)
  __shared__ MKArgs a_sh;
  if (threadIdx.x < sizeof(MKArgs) / 8)
    reinterpret_cast<uint64_t*>(&a_sh)[threadIdx.x] = reinterpret_cast<const uint64_t*>(&a_param)[threadIdx.x];
  // the record's header: every read of it (step.cuh rec_i32, gdn_t.cuh gt_word) is from this copy
  if (threadIdx.x < SR_BT) rec_sh[threadIdx.x] = a_param.rec_param ? rp.w[threadIdx.x] : __ldg(a_param.rec + threadIdx.x);
  __syncthreads();
  const MKArgs& a = a_sh;
  static_assert(65536 / NTHREADS >= REG_LAUNCH, "launch registers");
  constexpr uint16_t NO_CTR = 0xffff;  // Instr::dep / done, OpDesc::src: none (schedule.py NO_CTR)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ uint64_t bars[3 * NSTAGE + 2 * NACC + MBAR_TAIL];
  __shared__ __align__(16) OpDesc pdesc_sh;  // the prologue warps' current GEMV descriptor
  __shared__ __align__(16) float red[RED_FLOATS];
  __shared__ __align__(16) float epi_x[2 * 64 * NT + SHARED_B_SM_BYTES / 4 + 4 * NT + SHARED_B_TAG_WORDS];
  __shared__ uint32_t tmem_slot;
  __shared__ volatile uint32_t epi_count;  // instructions whose epilogue has released its writes (the producer's gate)
  __shared__ volatile uint32_t stg_count;  // GEMV instructions whose prologue has staged its inputs (pacing)
  __shared__ uint32_t pro_cnt;  // GEMV prologues done whose releasing count warp 7 issues (release / acquire.cta)
  __shared__ volatile uint32_t prod_n;     // ring positions the producer has issued (TMEM parking)
  __shared__ uint32_t irange_sh[5];  // this CTA's verify instructions [0, 1), the step's token rows T, its list's end,
                                     // the start of the step list's trailing sampler phases
  __shared__ DraftHandoff dho;      // the roles' state at the draft's start
  Smem sm;
  sm.stages = smem_raw;
  sm.xb = smem_raw + NSTAGE * STAGE_BYTES;  // (draft_ctx: the same place)
  sm.sfb = sm.xb + SFB_OFF;
  sm.stg = sm.xb + B_AREA_BYTES;
  sm.full = bars;
  sm.empty = bars + NSTAGE;
  sm.dfull = bars + 3 * NSTAGE;
  sm.dempty = bars + 3 * NSTAGE + NACC;
  sm.xfull = bars + 3 * NSTAGE + 2 * NACC;
  sm.xempty = bars + 3 * NSTAGE + 2 * NACC + MBAR_XEMPTY;
  sm.stg_full = bars + 3 * NSTAGE + 2 * NACC + MBAR_STG_FULL;
  sm.stg2_full = bars + 3 * NSTAGE + 2 * NACC + MBAR_STG2_FULL;
  sm.pfull = bars + 3 * NSTAGE + 2 * NACC + MBAR_PFULL;
  sm.pfree = bars + 3 * NSTAGE + 2 * NACC + MBAR_PFREE;
  sm.red = red;
  sm.xchg = epi_x;
  sm.ba = reinterpret_cast<float*>(sm.stg + GT_BA_OFF);  // (gdn_t.cuh GT_BA_SMEM's sums)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // the launch's epoch; CTA 0 advances it at its end, once every CTA has read it (the next launch is stream-ordered
  // after this one, so no other end-of-kernel synchronization is needed)
  const uint32_t epoch = ld_acquire(a.epoch);
  if (threadIdx.x == 0) asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(a.epoch + 1) : "memory");
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSTAGE; s++) {
      mbar_init(&sm.full[s], 1);
      mbar_init(&sm.empty[s], 1);
    }
    for (int b = 0; b < (int)NACC; b++) {
      mbar_init(&sm.dfull[b], 1);
      mbar_init(&sm.dempty[b], 4);
    }
    for (int b = 0; b < 2; b++) {
      mbar_init(&sm.xfull[b], 1);
      mbar_init(&sm.xempty[b], 1);  // the MMA issuer's commit
      mbar_init(&sm.pfull[b], 1);   // this CTA's expect (the other CTA's bytes complete it)
      mbar_init(&sm.pfree[b], 1);   // the other CTA's arrival
    }
    mbar_init(sm.stg_full, 1);
    mbar_init(sm.stg2_full, 1);
    attn_init();
    epi_count = 0;
    stg_count = 0;
    pro_cnt = 0;
    *reinterpret_cast<uint32_t*>(shared_b_sm(sm) + SHARED_B_DRAFT_ROWS_OFF) = 0;
    prod_n = 0;
    irange_sh[0] = a.sm_off[blockIdx.x];
    irange_sh[1] = a.sm_draft ? a.sm_draft[blockIdx.x] : a.sm_off[blockIdx.x + 1];  // (the draft's instructions last)
    irange_sh[3] = a.sm_off[blockIdx.x + 1];
    dho.host_epoch = epoch;
    irange_sh[2] = (uint32_t)min(max(rec_i32(a.rec, SR_NTOK), 1), NT);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == W_MMA) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "n"(T_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  cluster_sync();  // (the other CTA's mbarriers initialized before any arrival on them)
  sm.tmem = tmem_slot;
// the instruction range from smem, inside each role (computed before the role split, the values would be live across
// it and spill in the REG_LO roles)
#define MK_ROLE_RANGE                                                          \
  const uint32_t i0 = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[0]); \
  const uint32_t i1 = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1])
#define MK_ROWS const uint32_t T = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[2])
// the draft's loop variables from shared memory (its first instruction, the list's end, the draft's epoch)
#define MK_DRAFT_RANGE                                                                         \
  const uint32_t d1 = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1]);              \
  const uint32_t d_end = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[3]);           \
  const uint32_t d_epoch = *reinterpret_cast<const volatile uint32_t*>(&dho.epoch)
  // ring positions, replayed identically by each role: stages, NVFP4 stages, segments, GEMV instructions (B buffers)
  RingPos cc, seg, xg;
  // (each warpgroup's setmaxnreg dominates its roles' code: ptxas allocates a region by the setmaxnreg dominating it)
  if (warp >= W_PROD && warp < W_PRO0) {  // warpgroup 1: the TMA / MMA / TMEM roles
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_LO));
  if (warp == W_PROD) {
    MK_ROLE_RANGE;
    constexpr uint32_t GATE_POLL_NS = 64;     // the backoff between two polls of the epilogue's count (the gate)
    constexpr uint32_t ATT_REL_POLL_NS = 32;  // the backoff between two polls of att_rel
    uint32_t last_gemv = 0;  // 1 + this SM's last GEMV instruction (relative to i0); 0: none yet
    bool last_gemv_relaxed = false;  // its count is relaxed (validated op, or no waiter): nothing to gate
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      if (lane == 0) mk_progress(a.trace, 0, ii, epoch), mk_stamp(a.strace, 0, ii - i0);
      if (in.op == OP_GEMV) {
        const OpDesc d = a.descs[in.desc];
        // the next op's weights only once this CTA's last GEMV has released its results (across other instructions
        // too): the release's fence waits for every TMA load in flight on this SM, and the dependent op waits for it
        // anyway (Instr::aux bit 2: no wait, schedule.py mark_gate)
        if (last_gemv && !last_gemv_relaxed && !(in.aux & AUX_NOGATE))
          while (epi_count < last_gemv) __nanosleep(GATE_POLL_NS);
        last_gemv = ii - i0 + 1;
        last_gemv_relaxed = (in.done & DONE_RELAXED) != 0 || (in.aux & AUX_EPI_RELAXED) != 0;
        produce_gemv(in, d, sm, cc, stg_count, ii - i0 + 1, &prod_n);
      } else if (in.op == OP_ATTN) {
        // the attention's loads too only once the qkv projection on this SM has released its sums (its fence waits for
        // the loads in flight, and the attention's dependency is every SM's release)
        if (last_gemv && !last_gemv_relaxed)
          while (epi_count < last_gemv) __nanosleep(GATE_POLL_NS);
        produce_attn<NSTAGE>(in, reinterpret_cast<const AttnDesc&>(a.descs[in.desc]), sm.stages, STAGE_BYTES, sm.full,
                             sm.empty, sm.xb, sm.stg, cc);
        // the next loads once the item's release fence is done (att_rel): it would wait behind them
        while (att_rel < att_cnt[0]) __nanosleep(ATT_REL_POLL_NS);
      }
    }
  } else if (warp == W_MMA) {
    MK_ROLE_RANGE;
    (void)i1;  // (the loop's end re-read from shared memory: one register fewer held across this REG_LO region)
    constexpr uint32_t PARK_POLL_NS = 32;  // the backoff between two polls of the next stage to park
    for (uint32_t ii = i0; ii < *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1]); ii++) {
      const Instr in = a.instrs[ii];
      // (MKArgs::trace) the MMA warp's instruction and its rings' places (B buffers, stages, accumulators)
      if (lane == 0 && a.trace) {
        mk_progress(a.trace, 1, ii, epoch);
        mk_progress(a.trace, 8, ii, xg.slot | (xg.phase << 1) | (cc.slot << 4) | (cc.phase << 8) | (seg.slot << 12) | (seg.phase << 16));
      }

      if (in.op == OP_ATTN) {
        // (the TMEM base opaque here: its 16 query-column addresses would be hoisted out of the instruction loop, 16
        // registers of this REG_LO role for the whole kernel)
        uint32_t tm = sm.tmem;
        asm volatile("" : "+r"(tm));
        attn_mma<NSTAGE>(in, reinterpret_cast<const AttnDesc&>(a.descs[in.desc]), sm.stages, STAGE_BYTES, sm.full, sm.empty,
                         sm.xb, cc, tm);
        continue;
      }
      if (in.op != OP_GEMV) continue;
      const OpDesc d = a.descs[in.desc];
      // TMEM parking (every kind; Instr::aux bits 8..15: the park slots, schedule.py mark_park): while B is not ready,
      // the landed stages (A, an NVFP4 stage's weight scales) are copied into TMEM and their ring slots released, so
      // the producer streams on; their MMAs read TMEM. AUX_PERM: its parked block (perm_pl's P stages), B ready or not
      const bool nv = d.kind == KIND_NVFP4;
      const uint32_t pmax = park_max(in, d);
      const bool is_perm = perm_pl(in, d).P != 0;  // (then P = pmax)
      const uint32_t t_park = sm.tmem + (nv ? T_PARK_NV : T_PARK_F8), pcols = nv ? PARK_NV_COLS : PARK_F8_COLS;
      uint32_t np = 0;
      // an AUX_PERM instruction parks its block here, its B wait is mma_gemv2's
      const bool mma_waits_b = is_perm;
      if (pmax) {
        while ((is_perm && np < pmax) || (!mma_waits_b && !mbar_test(&sm.xfull[xg.slot], xg.phase))) {
          const uint32_t q = cc.n + np;  // the next stage to park (issued: its slot's previous use is consumed)
          if (np < pmax && prod_n > q && mbar_test(&sm.full[q % NSTAGE], (q / NSTAGE) & 1)) {
            tc_fence_after();
            if (elect_one()) {
              const uint32_t a_lo = ((smem_u32(sm.stages) >> 4) & 0x3FFF) + (q % NSTAGE) * (STAGE_BYTES >> 4);
              const uint32_t a_hi = (1024 >> 4) | (1u << 14) | (2u << 29), tp = t_park + np * pcols;
#pragma unroll
              for (uint32_t j = 0; j < 4; j++) tc_cp_128x256b(tp + 8 * j, ((uint64_t)a_hi << 32) | (a_lo + 2 * j));
              if (nv) {
                const uint8_t* at = sm.stages + (q % NSTAGE) * STAGE_BYTES + A_BYTES;
#pragma unroll
                for (uint32_t j = 0; j < 4; j++)
                  tc_cp_32x128b_x4(tp + 32 - (np == PARK_NV9 ? PARK9_SF_BACK : 0u) + 4 * j, sf_desc(at + 512 * j));
              }
              tc_commit(&sm.empty[q % NSTAGE]);
            }
            __syncwarp();
            np++;
          } else {
            __nanosleep(PARK_POLL_NS);
          }
        }
      } else {
        mbar_wait(&sm.xfull[xg.slot], xg.phase);
      }
      tc_fence_after();
      const uint8_t* xb = sm.xb + xg.slot * B_SLOT_BYTES;
      const uint8_t* sfb = sm.sfb + xg.slot * B_SLOT_BYTES;
      const uint32_t pk0 = is_perm ? perm_pl(in, d).L : 0u;  // the parked block's first stage
      uint64_t* bbar = mma_waits_b ? &sm.xfull[xg.slot] : nullptr;
      if (d.kind == KIND_NVFP4) mma_gemv2<KIND_NVFP4>(in, d, sm, xb, sfb, cc, seg, np, t_park, pk0, bbar, xg.phase);
      else if (d.kind == KIND_FP8) mma_gemv2<KIND_FP8>(in, d, sm, xb, sfb, cc, seg, np, t_park, pk0, bbar, xg.phase);
      else mma_gemv2<KIND_BF16>(in, d, sm, xb, sfb, cc, seg, np, t_park, pk0, bbar, xg.phase);
      if (elect_one()) tc_commit(&sm.xempty[xg.slot]);  // B buffer free once this instruction's MMAs complete
      __syncwarp();
      xg.next(NB);
    }
    // the MMA warp's ring and B-buffer positions at the target's end: the draft's start, or without the draft
    // mma_drain's at the kernel's end
    if (lane == 0) handoff_put(dho.cc, cc), handoff_put(dho.seg, seg), dho.xg_sp = xg.slot | (xg.phase << 8);
  }
  else if (warp == W_MMA + 1 && lane == 1) {  // (shared B) the aux finishers' rows released to the draft's counter
    MK_ROLE_RANGE;
    constexpr uint32_t DRAFT_ROWS_POLL_NS = 256;  // the backoff between two polls of the finishers' rows
    uint32_t n = 0;
    const uint32_t ca = smem_u32(shared_b_sm(sm)) + SHARED_B_DRAFT_ROWS_OFF;
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      if (in.op != OP_GEMV || !(in.aux & AUX_DRAFT_ROWS)) continue;
      n++;
      uint32_t c;
      do {
        asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(c) : "r"(ca) : "memory");
        if (c < n) __nanosleep(DRAFT_ROWS_POLL_NS);
      } while (c < n);
      const SharedBExt* const e = reinterpret_cast<const SharedBExt*>(a.descs[in.desc].shared_b_ext);
      red_release_add(a.counters + e->aux_ctr, 1);
    }
  }
  else if (warp == W_MMA + 2 && lane == 0) {  // a GEMV prologue's releasing count, in instruction order (see pro_cnt)
    MK_ROLE_RANGE;
    constexpr uint32_t PRO_CNT_POLL_NS = 64;  // the backoff between two polls of the prologues' count
    uint32_t n = 0;
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      mk_progress(a.trace, 3, ii, epoch);
      if (in.op != OP_GEMV || in.done == NO_CTR || (in.done & DONE_RELAXED)) continue;
      n++;
      uint32_t c;
      do {
        asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(c) : "r"(smem_u32(&pro_cnt)) : "memory");
        if (c < n) __nanosleep(PRO_CNT_POLL_NS);
      } while (c < n);
      red_release_add(a.counters + in.done, 1);
      mk_stamp(a.strace, 3, ii - i0);
    }
  }
  // (the draft's phase checked on the parameter: the shared copy's address would be live through the step's loops,
  // spilled at their split)
  if (a_param.draft) {  // the draft's phase (warps 6, 7 idle in it; the warpgroup keeps its registers)
    uint32_t tid;  // (re-read here: the warp index is not carried through the step's loops at their split)
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    if (tid / 32 == W_PROD) {
      draft_sync();
      if (DRAFT_REINIT) draft_ring_reset(bars, dho, tid % 32);
      draft_setup(a, smem_raw, tid % 32, &dho.host_epoch, &dho.epoch, &epi_count,
                  ss_in_draft(a, *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1]),
                              *reinterpret_cast<const volatile uint32_t*>(&irange_sh[3])));
      draft_sync();
      const DraftCtx dc = draft_ctx(smem_raw, bars, tmem_slot, &epi_count);
      MK_DRAFT_RANGE;
      const int d_lane = threadIdx.x & 31;
      dk::RingPos dcc = handoff_get(dho.cc);
      dk::ProdRole role;
      for (uint32_t ii = d1; ii < d_end; ii++) {
        const dk::Instr in = dc.a->instrs[ii];  // (in registers, as the draft's roles take it: not reloaded per use)
        role.instr(in, ii, d1, *dc.a, dc.sm, dcc, d_lane);
      }
    } else if (tid / 32 == W_MMA) {
      draft_sync();
      draft_sync();
      const DraftCtx dc = draft_ctx(smem_raw, bars, tmem_slot, &epi_count);
      MK_DRAFT_RANGE;
      const int d_lane = threadIdx.x & 31;
      dk::RingPos dcc = handoff_get(dho.cc), dseg = handoff_get(dho.seg);
      dk::MmaRole role;
      for (uint32_t ii = d1; ii < d_end; ii++) {
        const dk::Instr in = dc.a->instrs[ii];
        role.instr(in, ii, d1, *dc.a, dc.sm, dcc, dseg, d_lane);
      }
    }
  }
  } else if (warp < W_PROD) {  // warpgroup 0: the epilogue
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_EPI));
    const int ept = threadIdx.x - W_EPI0 * 32;
    MK_ROLE_RANGE;
    // the instruction's done, op and aux words kept in shared memory for the count after the epilogue (in registers
    // they would be held across the epilogue, whose shared-B paths need the registers, and spill)
    constexpr uint32_t EPI_WORDS_OFF = 8;  // (their place in the aux count's 16 bytes, past the count)
    const uint32_t cst = smem_u32(shared_b_sm(sm)) + SHARED_B_DRAFT_ROWS_OFF + EPI_WORDS_OFF;
    for (uint32_t ii = i0; ii < *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1]); ii++) {
      const Instr in = a.instrs[ii];
      if (ept == 0) mk_progress(a.trace, 4, ii, epoch);
      if (ept == 0) {
        sts_u32(cst, (uint32_t)in.done | ((uint32_t)in.op << 16));
        sts_u32(cst + 4, in.aux);
      }
      if (in.op == OP_GEMV) {
        const OpDesc d = a.descs[in.desc];
        MK_ROWS;
        epilogue_gemv(in, d, sm, seg, warp, lane, epoch, T, ii);
      }
      named_sync_u(BAR_EPI, 128);
      if (ept == 0) {
        uint32_t w0, w1;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w0) : "r"(cst) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w1) : "r"(cst + 4) : "memory");
        const uint32_t i_done = w0 & 0xffffu, i_op = w0 >> 16, i_aux = w1;
        // (a validated op: the count is a hint; AUX_EPI_RELAXED: nothing of the epilogue's to release)
        if (i_done != NO_CTR && ((i_done & DONE_RELAXED) || (i_op == OP_GEMV && (i_aux & AUX_EPI_RELAXED)))) {
          red_relaxed_add(a.counters + (i_done & ~DONE_RELAXED), 1);
        } else {
          if (i_done != NO_CTR) red_release_add(a.counters + i_done, 1);  // releases the warps' adds (bar.sync before)
          __threadfence_block();  // the epilogue warps' stores before the count (OP_SS_A reads the logits)
        }
        epi_count = ii - i0 + 1;
        mk_stamp(a.strace, 2, ii - i0);
      }
    }
    if (a_param.draft) {  // the draft's phase: the finishers' registers, once the prologue warps have released theirs
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(DREG_EPI));
      draft_sync();
      draft_sync();
      const DraftCtx dc = draft_ctx(smem_raw, bars, tmem_slot, &epi_count);
      MK_DRAFT_RANGE;
      const int d_lane = threadIdx.x & 31;
      const int d_ept = threadIdx.x - W_EPI0 * 32;
      dk::RingPos dseg = handoff_get(dho.seg);
      dk::EpiRole role;
      // the conv taps of the two-phase GEMV two instructions on (its TMEM region's previous reader long done) while
      // this instruction's accumulator is due: off the kernel projection's tail, which the conv phase waits for
      dk::BaseRole base;
      for (uint32_t ii = d1; ii < d_end; ii++) {
        if (ii == d1 + 1) {
          // the step's tables after the fc (its epilogue reads none): the step's tokens waited for by the setup, or
          // with the sampler's phases in the draft's lists, here
          if (ss_in_draft(a, d1, d_end)) {
            const uint32_t ep = *reinterpret_cast<const volatile uint32_t*>(&dho.host_epoch);
            if (d_ept == 0) wait_counter(a.counters + a.draft_sampler, (ep + 1) * a.ctr_count[a.draft_sampler]);
            named_sync(BAR_EPI, 128);
          }
          role.begin(*dc.a, dc.sm, d_ept, d_epoch);
        }
        if (ii == d1 && d1 + 1 < d_end) {
          const dk::Instr nx = dc.a->instrs[d1 + 1];
          base.instr(nx, *dc.a, dc.sm, d_lane);
        }
        if (ii + 2 < d_end) {
          const dk::Instr nx = dc.a->instrs[ii + 2];
          base.instr(nx, *dc.a, dc.sm, d_lane);
        }
        const dk::Instr in = dc.a->instrs[ii];
        role.instr(in, ii, d1, *dc.a, dc.sm, dseg, d_ept, d_epoch);
      }
    }
  } else {  // warpgroups 2-3: the prologue warps, the registers of the others
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(REG_HI));
    const int pt = threadIdx.x - W_PRO0 * 32;
    MK_ROLE_RANGE;
    uint32_t stg_phase = 0, stg2_phase = 0;
    uint32_t pph = 0;  // AUX_SPLIT: bit s, the parity of B slot s's current pfull / pfree phase
    uint32_t pro_n = 0;  // (pt 0) GEMV prologues whose releasing count warp 7 issues
    // a prologue-warp op's completion: its counter counts the instruction twice (here and in the epilogue warps)
    auto release = [&](const Instr& in, uint32_t i_rel) {
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) {  // (DONE_RELAXED: no instruction waits on the counter, schedule.py)
        if (in.done & DONE_RELAXED) red_relaxed_add(a.counters + (in.done & ~DONE_RELAXED), 1);
        else red_release_add(a.counters + in.done, 1);
      }
      if (pt == 0) mk_stamp(a.strace, 3, i_rel);  // (MKArgs::strace: a prologue op's completion)
    };
    // the waits branch on the instruction's dep / aux, never on a pointer selected against nullptr: ptxas (13.0-13.4,
    // sm_103a) can test such a select's register before the CS2R zero of its untaken side has landed (the dry combine
    // copies would read it as non-null and wait on address 0)
    auto dep_target = [&](const Instr& in) -> uint32_t {
      return in.dep != NO_CTR ? (epoch + 1) * a.ctr_count[in.dep] : 0u;
    };
    // up to the trailing run of the sampler's phases, which follow the loop (irange_sh[4], re-read)
    if (pt == 0) {
      uint32_t s0 = i1;
      while (s0 > i0 && a.instrs[s0 - 1].op >= OP_SS_A && a.instrs[s0 - 1].op <= OP_SS_C) s0--;
      *reinterpret_cast<volatile uint32_t*>(&irange_sh[4]) = s0;
    }
    named_sync(BAR_PRO, NPRO);
    for (uint32_t ii = i0; ii < *reinterpret_cast<const volatile uint32_t*>(&irange_sh[4]); ii++) {
      const Instr in = a.instrs[ii];
      if (pt == 0) mk_progress(a.trace, 5, ii, epoch), mk_stamp(a.strace, 1, ii - i0);
      if (pt == 0 && a.trace) mk_progress(a.trace, 9, ii, xg.slot | (xg.phase << 1) | (cc.slot << 4) | (cc.phase << 8));
      if (in.op == OP_ATTN) {
        const AttnDesc& g = reinterpret_cast<const AttnDesc&>(a.descs[in.desc]);
        const OpDesc& sd = a.descs[g.src];
        const uint32_t zero_target = in.aux ? (epoch + 1) * a.ctr_count[in.aux - 1] : 0u;  // aux: zeroing counter + 1
        // (the B buffers hold no operand during the item: the qkv projection's MMAs are done, o_proj's prologue comes
        // later; the staging buffer follows them, sm.stg = sm.xb + B_AREA_BYTES: the query rows' slots and the
        // extension stages below the rope rows)
        static_assert(ATT_ROWS * ATT_QSLOT <= B_AREA_BYTES + ATT_RT_OFF && ATT_XS * ATT_SLOT <= B_AREA_BYTES + ATT_RT_OFF &&
                      (NSTAGE * STAGE_BYTES) % 1024 == 0 && B_AREA_BYTES % 1024 == 0,
                      "the query rows' slots, the extension stages (SW128-aligned)");
        static_assert(ATT_XM_OFF + ATT_XM_FLOATS * 4 <= STG_BYTES && ATT_MAX_SPLITS * ATT_PO_BLOCK <= STG_BYTES,
                      "attention scratch");
        attn_split<NSTAGE, NPRO>(in, g, reinterpret_cast<const float*>(acc_copy(sd, epoch)), sd.alpha, a.counters,
                                 dep_target(in), zero_target, sm.stages, STAGE_BYTES, sm.full, sm.empty, sm.stg, sm.xb, pt, cc,
                                 sm.tmem);
        release(in, ii - i0);
        if (pt == 0) att_rel = att_cnt[3];  // (the producer's gate)
        continue;
      }
      if (in.op == OP_ATTN_COMBINE) {
        const AttnDesc& g = reinterpret_cast<const AttnDesc&>(a.descs[in.desc]);
        const OpDesc& sd = a.descs[g.src];
        MK_ROWS;
        attn_combine<NPRO>(in, g, reinterpret_cast<const float*>(acc_copy(sd, epoch)), sd.alpha, a.counters, dep_target(in),
                           sm.stg, sm.xb, pt, (int)T, epoch);
        release(in, ii - i0);
        continue;
      }
      if (in.op == OP_ZERO) {  // newly allocated KV blocks (vLLM's KVBlockZeroer): this SM's share of (block, segment)
        const OpDesc d = a.descs[in.desc];
        const int32_t* rec = reinterpret_cast<const int32_t*>(d.rec);
        const uint32_t nz = (uint32_t)rec_i32(rec, SR_NZERO), nseg = d.n_out;
        const uint64_t* seg_addr = reinterpret_cast<const uint64_t*>(d.w);
        const int64_t* seg_stride = reinterpret_cast<const int64_t*>(d.in0);  // 4-byte units
        const int64_t* seg_page = reinterpret_cast<const int64_t*>(d.res_in);
        for (uint32_t j = in.tile; j < nz * nseg; j += d.n_inst) {
          const uint32_t b = (uint32_t)rec_i32(rec, SR_ZERO_IDS + j / nseg), sg = j % nseg;
          uint4* p = reinterpret_cast<uint4*>(seg_addr[sg] + (uint64_t)b * seg_stride[sg] * 4);
          const uint32_t n16 = (uint32_t)(seg_page[sg] / 4);
          for (uint32_t i = pt; i < n16; i += NPRO) p[i] = make_uint4(0, 0, 0, 0);
        }
        release(in, ii - i0);
        continue;
      }
      if (in.op == OP_EMBED) {  // the input tokens' rows of the embedding table (w, K values) into in0 [NT][K] (bf16)
        const OpDesc d = a.descs[in.desc];
        MK_ROWS;
        // (schedule.py add_embed: n_inst instructions, instruction tile i the rows i, i + n_inst, ..; one: every row)
#pragma unroll 1
        for (uint32_t t = in.tile; t < T; t += d.n_inst) {
          const int32_t tok = rec_token(reinterpret_cast<const int32_t*>(d.rec), t);
          const uint4* row = reinterpret_cast<const uint4*>(d.w + (size_t)tok * d.K * 2);
          uint4* out = reinterpret_cast<uint4*>(d.in0 + (size_t)t * d.K * 2);
          for (uint32_t i = pt; i < d.K / 8; i += NPRO) out[i] = __ldg(row + i);
        }
        release(in, ii - i0);
        continue;
      }
      if (in.op == OP_GDNT_PRE) {  // gdn_t.cuh: the split mode's replay (no outputs, no counter)
        gdn_t_replay_any(in.tile, reinterpret_cast<const GdnTDesc&>(a.descs[in.desc]), sm.stg, sm.stg_full, stg_phase, pt,
                         BAR_PRO);
        continue;
      }
      if (in.op == OP_GDNT) {  // gdn_t.cuh: the T-token core; the release; the state, record and boundary stores
        const GdnTDesc& g = reinterpret_cast<const GdnTDesc&>(a.descs[in.desc]);
        constexpr int CORE_IDLE_PT0 = 64;  // the first of the core's threads idle in its output phase (warps 2-7)
        // GT_IN_ACC / GT_BA_OP (the first layer): the dependency's counter, then the out_proj's gate copy (in.r0)
        const bool acc_dep = g.in_mode == GT_IN_ACC || g.ba_mode == GT_BA_OP;
        // the op's own counter (gdn_t_finish's first-step wait): the index without DONE_RELAXED, which mark_dead sets
        // on the cores of layers 1.. (no instruction's dependency; the reads the wait orders, the window loads of every
        // item, are consumed before the item's count, so a relaxed count serves it)
        const uint32_t done_index = in.done & ~DONE_RELAXED;
        const GtDep dep{acc_dep && in.dep != NO_CTR ? a.counters + in.dep : nullptr, dep_target(in), in.r0,
                        sm.stg2_full, in.done != NO_CTR ? a.counters + done_index : nullptr,
                        in.done != NO_CTR ? (epoch + 1) * a.ctr_count[done_index] : 0u,
                        reinterpret_cast<float*>(*reinterpret_cast<const uint64_t*>(&g.pad[0]))};  // (dbg_o: null)
        GtCarry cy;
        gdn_t_core_any(in.tile, g, sm.stg, sm.stg_full, stg_phase, epoch, pt, BAR_PRO, BAR_GDNT, dep, cy);
        // Q_LL: in_proj_qkv's words are read without its counter, acquired here (complete by now) so that this release
        // orders its prologue's writes (the residual slices) before every later reader; by the first idle thread in the
        // output phase, before the release's barrier (thread 0 after it: two L2 round trips before out_proj's B)
        if (!acc_dep && in.dep != NO_CTR && pt == CORE_IDLE_PT0)
          wait_counter(a.counters + in.dep, (epoch + 1) * a.ctr_count[in.dep]);
        if ((in.aux & AUX_ZNEXT) && pt >= CORE_IDLE_PT0) {  // (AUX_ZNEXT) out_proj's zero_next_acc by warps 2-7
          const Instr nx = a.instrs[ii + 1];
          if ((in.aux & AUX_FAPRE) && pt < CORE_IDLE_PT0 + (int)(sizeof(OpDesc) / 16))
            reinterpret_cast<uint4*>(&pdesc_sh)[pt - CORE_IDLE_PT0] =
                reinterpret_cast<const uint4*>(a.descs + nx.desc)[pt - CORE_IDLE_PT0];
          zero_next_acc<NPRO - CORE_IDLE_PT0>(nx, a.descs[nx.desc], pt - CORE_IDLE_PT0, epoch);
        }
        release(in, ii - i0);
        gdn_t_finish(in.tile, g, sm.stg, pt, BAR_PRO, dep, cy);
        continue;
      }
      if (in.op == OP_PREV) {  // the previous launched step's outputs and the rest of its post_update (spec_sample.cuh)
        const PrevDesc& pd = reinterpret_cast<const PrevDesc&>(a.descs[in.desc]);
        if (rec_i32(reinterpret_cast<const int32_t*>(pd.rec), SR_BK + BK_PREV)) prev_outputs(pd, pt, NPRO, BAR_PRO);
        release(in, ii - i0);
        continue;
      }
      if (in.op == OP_BOOK) {  // before the embedding (this SM's next instruction) reads the input ids
        const BookDesc& bd = reinterpret_cast<const BookDesc&>(a.descs[in.desc]);
        if (rec_i32(reinterpret_cast<const int32_t*>(bd.rec), SR_BK + BK_ON)) {
          MK_ROWS;
          book_inputs(bd, T, pt, NPRO, BAR_PRO);
          __threadfence();  // (the inputs for the launch's later readers on other SMs, and the draft's after it)
        }
        release(in, ii - i0);
        continue;
      }
      // OP_GEMV
      if (in.aux & AUX_TWIN) {  // staged by the previous instruction's prologue (AUX_TWIN)
        xg.next(NB);
        cc.advance(in.r1 - in.r0, NSTAGE);
        continue;
      }
      if (__builtin_expect((in.aux & AUX_FASTACT) != 0, 0)) goto fast_act;
      {  // (the generic path in a block of its own, so that fast_act's path can follow it out of line)
      // the op's descriptor from a shared-memory copy (its fields read where used: a register copy would keep them live
      // across the whole prologue, at the region's register limit)
      if (pt < (int)(sizeof(OpDesc) / 16))
        reinterpret_cast<uint4*>(&pdesc_sh)[pt] = reinterpret_cast<const uint4*>(a.descs + in.desc)[pt];
      named_sync(BAR_PRO, NPRO);
      const OpDesc& d = pdesc_sh;
      MK_ROWS;
      if (xg.n >= NB) mbar_wait(&sm.xempty[xg.slot], xg.phase ^ 1);
      // AUX_SPLIT: this CTA's token rows [t0, t1) (the first half: rank 0), the other CTA's into this B slot. Its
      // slot's previous use complete, the bytes the other CTA brings are expected on pfull and the other CTA is told it
      // may write them (relaxed: the MMAs' reads are complete when xempty completes, nothing to release)
      const bool is_split = (in.aux & AUX_SPLIT) != 0;
      const uint32_t rk = is_split ? cluster_rank() : 0u;
      const uint32_t t0 = is_split && rk ? (T + 1) / 2 : 0u, t1 = is_split && !rk ? (T + 1) / 2 : T;
      if (is_split && pt == 0) {  // (bytes per chunk and row: pro_norm_rows' NVFP4 copy, FP8's codes)
        const ChunkRange cr = chunk_range(in, d);
        constexpr uint32_t PEER_ROW_BYTES_NV = 160u;  // an NVFP4 chunk's bytes per token row (codes, scales)
        constexpr uint32_t PEER_ROW_BYTES_F8 = 128u;  // an FP8 chunk's bytes per token row
        const uint32_t row_b = d.kind == KIND_NVFP4 ? PEER_ROW_BYTES_NV : PEER_ROW_BYTES_F8;
        mbar_expect_tx(&sm.pfull[xg.slot], (T - (t1 - t0)) * cr.nck * row_b);
        mbar_arrive_peer(peer_addr(smem_u32(&sm.pfree[xg.slot]), rk ^ 1));
      }
      if (!(in.aux & AUX_ZNEXT)) zero_next_acc(in, d, pt, epoch);  // (AUX_ZNEXT: zeroed by the core's idle warps)
      // what can be loaded before the dependency is met: the norm weights, in_proj_ba's rows (BA), the producing op's
      // output location
      const bool has_stages = in.r0 < in.r1;
      const bool is_norm = has_stages && (d.pro == PRO_NORM_IN || d.pro == PRO_NORM_RESID);
      SrcAcc src{nullptr, 0.f, 0u};
      if (has_stages && d.pro != PRO_NORM_IN && d.src != NO_CTR) src = src_acc(a.descs[d.src], epoch);
      uint2 wv[NG5];
      const uint2* bw = nullptr;  // BA (OpDesc::xn): in_proj_ba's b row of this SM's GDN head (a's: GDN_HV rows on)
      // (the loads are pinned here, before the dependency wait: read-only loads may otherwise be sunk to their first
      // use, an L2 round trip under the weight stream inside the critical path)
      if (is_norm) {
#pragma unroll
        for (int i = 0; i < NG5; i++) {
          wv[i] = __ldg(reinterpret_cast<const uint2*>(d.nw) + pt + i * NPRO);
          asm volatile("" ::"r"(wv[i].x), "r"(wv[i].y));
        }
      }
      // in_proj_ba's b and a rows of this SM's GDN item's head, into the staging buffer (shared B: in_proj_qkv as a
      // PRO_ACT consumer computes b / a from the published bf16(x_n), pro_ba_pub)
      // (shared B: bf16(x_n), or 0)
      const uint64_t shared_xn = has_stages && d.pro == PRO_ACT && d.xn ? shared_xn_of(a.descs[d.src]) : 0;
      const bool bacp = (is_norm || shared_xn) && d.xn;
      if (bacp) {
        const GdnDesc& g = reinterpret_cast<const GdnDesc&>(a.descs[d.xn - 1]);
        const uint32_t hv = min(in.tile, (uint32_t)GDN_ITEMS - 1) / 3;
        bw = reinterpret_cast<const uint2*>(sm.stg + BA_OFF);
        if (shared_xn & 1u) {
          if (pt == 0) mbar_arrive(sm.stg2_full);  // (b / a partials: no rows; the barrier's phase as with the copy)
        } else if (pt == 0) {
          const uint8_t* rw = reinterpret_cast<const uint8_t*>(g.ba_w);
          mbar_expect_tx(sm.stg2_full, 2 * K_NORM * 2);
          bulk_g2s(sm.stg + BA_OFF, rw + (size_t)hv * K_NORM * 2, K_NORM * 2, sm.stg2_full);
          bulk_g2s(sm.stg + BA_OFF + K_NORM * 2, rw + (size_t)(GDN_HV + hv) * K_NORM * 2, K_NORM * 2, sm.stg2_full);
        }
      }
      if (has_stages && d.pro == PRO_GDN_NORM) {  // its norm weights in wv[0..3]: block pt % 8's 16
        const uint4* wp = reinterpret_cast<const uint4*>(d.nw) + (pt & 7) * 2;
        const uint4 w0 = __ldg(wp), w1 = __ldg(wp + 1);
        wv[0] = make_uint2(w0.x, w0.y), wv[1] = make_uint2(w0.z, w0.w);
        wv[2] = make_uint2(w1.x, w1.y), wv[3] = make_uint2(w1.z, w1.w);
      }
      // the twin (OpDesc::twin): the next GEMV instruction's B (in_proj_z's chunks of the same codes) into the other
      // buffer, by the real run once that buffer's previous use's MMAs have completed (the dry run writes none of it:
      // its wait would hold the warm-up back until this SM's previous stream ends)
      const bool has_twin = has_stages && d.twin;
      uint8_t* xb2 = nullptr;
      ChunkRange cr2{1u, 0u, 0u, 0u};
      if (has_twin) {
        const Instr nx = a.instrs[ii + 1];
        cr2 = chunk_range(nx, a.descs[nx.desc]);
        xb2 = sm.xb + ((xg.slot + 1) % NB) * B_SLOT_BYTES;  // (no twins with NB = 1: schedule.py)
      }
      asm volatile("" ::"l"(src.acc), "f"(src.alpha));
      const bool is_resid = has_stages && d.pro == PRO_NORM_RESID;  // a validated producing op (its values polled)
      if (is_resid && pt == 0 && t0 < t1) {
        // token row t0's residual into the staging buffer before the dependency wait (written two ops earlier, released
        // by a counter this CTA has already acquired); the prologue copies the later rows' itself
        const uint32_t rb = d.K * (d.res_bf16 ? 2 : 4);
        const uint32_t nrb = is_split && d.res_bf16 ? t1 - t0 : 1u;  // (pro_norm_rows ROWS_RESIDENT: the CTA's rows)
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect_tx(sm.stg_full, nrb * rb);
        bulk_g2s(sm.stg, reinterpret_cast<const uint8_t*>(d.res_in) + (size_t)t0 * rb, nrb * rb, sm.stg_full);
      }
      uint8_t* xb = sm.xb + xg.slot * B_SLOT_BYTES;
      uint8_t* sfb = sm.sfb + xg.slot * B_SLOT_BYTES;
      // warm-up (Instr::aux bit 1: the prologue warps a whole op ahead): the prologue twice from the same code, first
      // dry on stale inputs before the dependency wait (one token row, no global store, its B overwritten), bringing
      // its code into the instruction cache (the weight stream evicts it between uses, even from L2), then the real run
#pragma unroll 1
      for (int rep = (has_stages && (in.aux & AUX_WARM)) ? 0 : 1; rep < 2; rep++) {
        const bool is_dry = rep == 0;
        if (!is_dry) {
          if (has_twin && xg.n >= 1) mbar_wait(&sm.xempty[(xg.slot + 1) % NB], ((xg.n - 1) >> 1) & 1);
          if (in.dep != NO_CTR) {
            if (pt == 0) {
              if (is_resid)
                wait_counter_relaxed(a.counters + in.dep, (epoch + 1) * a.ctr_count[in.dep] - (in.aux >> AUX_DEP_EARLY_SHIFT) * T);
              else if (d.pro == PRO_ACT && d.nw) {  // (shared B: no wait, its B validated)
              }
              else if (d.act_stride)  // (a validated operand behind a relaxed count: its words re-polled)
                wait_counter_relaxed(a.counters + in.dep, (epoch + 1) * a.ctr_count[in.dep]);
              else {
                const uint32_t tgt = (epoch + 1) * a.ctr_count[in.dep];
                while (ctr_before(ld_acquire(a.counters + in.dep), tgt)) {
                }
              }
              if (is_split) mbar_wait(&sm.pfree[xg.slot], (pph >> xg.slot) & 1);  // the other CTA's slot free
            }
            named_sync(BAR_PRO, NPRO);
          }
          if (bacp) {  // (landed long before)
            mbar_wait(sm.stg2_full, stg2_phase);
            stg2_phase ^= 1;
          }
          if (has_stages && d.pro == PRO_ACT && d.nw) {  // shared B: no bulk copy to protect, the pacing released first
            if (pt == 0) stg_count = ii - i0 + 1;
            stage_pub(in, d, pt, xb, sfb, epoch, T);  // (b / a after this instruction's B-ready: pro_ba_pub below)
            if (has_twin) {  // the twin's (in_proj_z's) chunks of the same B, into its slot (its xempty waited above)
              Instr nx = a.instrs[ii + 1];
              stage_pub(nx, d, pt, xb2, nullptr, epoch, T);
            }
          } else if (has_stages && d.pro == PRO_ACT) {
            stage_inputs(in, d, sm, pt, stg_phase, xb, sfb, epoch, T);
          }
          if (pt == 0) stg_count = ii - i0 + 1;
        }
        else if (has_stages && d.pro == PRO_ACT && d.nw) {  // the dry run: the shared-B code (one copy) into the icache
          stage_pub(in, d, pt, xb, sfb, epoch, 1u, true);
          if (shared_xn) pro_ba_pub(shared_xn, sm.red, sm.ba, pt, bw, ba_t(in, 1u), epoch, true, 0u, 0u, 0u);
        }
        if (has_stages) {
          PeerB pb{0u, 0u, ChunkRange{1u, 0u, 0u, 0u}};
          if (is_split && !is_dry) {  // (the dry run stores into this CTA's B rows only, row t0: its own)
            Instr ip = in;
            ip.r0 = in.pr0, ip.r1 = in.pr1;
            pb.cr = chunk_range(ip, d);
            pb.xb = peer_addr(smem_u32(xb), rk ^ 1);
            pb.bar = peer_addr(smem_u32(&sm.pfull[xg.slot]), rk ^ 1);
          }
          prologue(in, d, sm, pt, wv, src, xb, sfb, is_dry ? ChunkRange{1u, 0u, 0u, 0u} : cr2, xb2, bw, T, stg_phase,
                   is_dry, epoch + 1, pb, t0, t1);
        }
      }
      fence_async_smem();
      named_sync(BAR_PRO, NPRO);  // B operand and activation scales complete
      if (pt == 0) {
        if (is_split) {  // the other CTA's rows (its st.async bytes), then visible to the MMAs' async proxy
          mbar_wait_cluster(&sm.pfull[xg.slot], (pph >> xg.slot) & 1);
          fence_async_smem();
        }
        mbar_arrive(&sm.xfull[xg.slot]);
        // (B ready: the epilogue's f_t)
        if (d.pro == PRO_ACT && d.nw && d.shared_b_ext) shared_b_set_tag(shared_b_ready_slot(sm, ii), shared_b_tag(ii, epoch));
        // the op's counter also counts the prologues: releases their global stores (x / residual slices; validated
        // ops have none: relaxed). The release by warp 7, after this CTA-scope release of the prologue's
        // completion (the prologue threads' stores, ordered before it by the barrier above, are in its causal past)
        if (in.done != NO_CTR) {
          if (in.done & DONE_RELAXED) red_relaxed_add(a.counters + (in.done & ~DONE_RELAXED), 1);
          else {
            pro_n++;
            asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(smem_u32(&pro_cnt)), "r"(pro_n) : "memory");
          }
        }
      }
      // (shared B: for the GDN core only, after B-ready)
      if (shared_xn) {
        // (ftag: epilogue warp 0's factors)
        const uint32_t ftag = d.shared_b_ext && shared_b_erstd(d.shared_b_ext) ? shared_b_tag(ii, epoch) : 0u;
        pro_ba_pub(shared_xn, sm.red, sm.ba, pt, bw, ba_t(in, T), epoch, false, shared_b_factors_slot(sm, ii), ftag, smem_u32(shared_b_sm(sm)) + SHARED_B_SM_BYTES);
      }
      if (has_twin) {  // AUX_TWIN: the twin's B is complete with this one (the barrier above): its prologue's work here
        const Instr nx = a.instrs[ii + 1];
        if (nx.aux & AUX_TWIN) {
          zero_next_acc(nx, a.descs[nx.desc], pt, epoch);
          if (pt == 0) {
            mbar_arrive(&sm.xfull[xg.slot ^ 1]);
            stg_count = ii + 1 - i0 + 1;
            if (nx.done != NO_CTR) {  // (warp 7's walk counts the twin as its next GEMV instruction)
              if (nx.done & DONE_RELAXED) red_relaxed_add(a.counters + (nx.done & ~DONE_RELAXED), 1);
              else {
                pro_n++;
                asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(smem_u32(&pro_cnt)), "r"(pro_n) : "memory");
              }
            }
          }
        }
      }
      if (is_split) pph ^= 1u << xg.slot;
      xg.next(NB);
      cc.advance(in.r1 - in.r0, NSTAGE);  // the ring position, for the attention ops' stages
      }
      continue;
    fast_act:
      // AUX_FASTACT: out_proj after the GDN core (validated, AUX_ZNEXT, no dependency) by a path of its own: the
      // generic one branches over the other prologues' blocks between the descriptor and B, each a cold instruction
      // fetch on the step's critical path
      {
        if (!(in.aux & AUX_FAPRE)) {  // (AUX_FAPRE: copied by the core's idle warps, before its release's barrier)
          if (pt < (int)(sizeof(OpDesc) / 16))
            reinterpret_cast<uint4*>(&pdesc_sh)[pt] = reinterpret_cast<const uint4*>(a.descs + in.desc)[pt];
          named_sync(BAR_PRO, NPRO);
        }
        const OpDesc& d = pdesc_sh;
        MK_ROWS;
        if (xg.n >= NB) mbar_wait(&sm.xempty[xg.slot], xg.phase ^ 1);
        stage_val(in, d, sm, pt, stg_phase, sm.xb + xg.slot * B_SLOT_BYTES, epoch, T);
        if (pt == 0) stg_count = ii - i0 + 1;
        fence_async_smem();
        named_sync(BAR_PRO, NPRO);  // B operand complete
        if (pt == 0) {
          mbar_arrive(&sm.xfull[xg.slot]);
          if (in.done != NO_CTR) {
            if (in.done & DONE_RELAXED) red_relaxed_add(a.counters + (in.done & ~DONE_RELAXED), 1);
            else {
              pro_n++;
              asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(smem_u32(&pro_cnt)), "r"(pro_n) : "memory");
            }
          }
        }
        xg.next(NB);
        cc.advance(in.r1 - in.r0, NSTAGE);
      }
    }
    // the spec sampler's phases after the loop, one call site for both of their places: the step list's trailing run,
    // or (ss_list) the draft list's run after its fc, before which a launch with the draft (ssd) enters the draft's
    // phase and builds the fc's B; a launch without the draft runs the draft list's run too, in the step's scratch
    const uint32_t d1_ = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[1]);
    const uint32_t de_ = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[3]);
    const bool ss_list = ss_in_draft(a, d1_, de_), ssd = a_param.draft && ss_list;
    uint32_t s0 = *reinterpret_cast<const volatile uint32_t*>(&irange_sh[4]), s1 = d1_, d_ring = 0;
    if (ss_list) {
      s0 = s1 = d1_ + 1;
      while (s1 < de_ && a.instrs[s1].op >= OP_SS_A && a.instrs[s1].op <= OP_SS_C) s1++;
    }
    bool draft_entered = !ssd;  // ssd: entered before the first real run (the dry runs during the LM head)
    {
      uint32_t tid_;  // (re-read: not carried through the step's loop)
      asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid_));
      const int pt_ = (int)tid_ - W_PRO0 * 32;
      const uint32_t ep = *reinterpret_cast<const volatile uint32_t*>(&dho.host_epoch);
      // the scratch: a dry run's the staging buffer (during the LM head: its B operand built), a real run's the step's
      // B buffers (free once the SM's last GEMV, the LM head, has completed) or with ssd the draft's B buffer past the
      // fc's B tiles
      const uint32_t ssd_m = ssd ? ~0u : 0u;
#pragma unroll 1
      for (uint32_t ii = s0; ii < s1; ii++) {
        const Instr in = a.instrs[ii];
        const SsDesc& sd = reinterpret_cast<const SsDesc&>(a.descs[in.desc]);
        const bool ss_enabled =
            rec_i32(reinterpret_cast<const int32_t*>(sd.rec), SR_SS + SS_TOP_P) != 0;  // (0: vLLM's)
        // at OP_SS_A, the SM's phases first as dry runs while the LM head streams: their code runs once per step,
        // fetched cold on the step's tail otherwise; one call site for the dry and the real runs
        uint32_t nss = ii;
        while (ss_enabled && in.op == OP_SS_A && nss < s1 && a.instrs[nss].op >= OP_SS_A && a.instrs[nss].op <= OP_SS_C)
          nss++;
#pragma unroll 1
        for (uint32_t k = min(ii, nss); k <= nss; k++) {
          const uint32_t dmask = lp_lt((int)k, (int)nss);  // (masks, not selects: ptxas' predicated select hazard)
          const bool is_dry = dmask != 0u;
          const Instr x = a.instrs[ii + ((k - ii) & dmask)];  // (the real run: in)
          const uint32_t xop = x.op, item = x.r0;
          if (pt_ == 0) mk_progress(a.trace, 5, ii, ep), mk_progress(a.trace, 6, k - ii, 0x10u | (uint32_t)is_dry);
          if (!is_dry && !draft_entered) {  // (once: every dry run precedes the first real one)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(DREG_HI));
            draft_sync();
            draft_sync();
            const DraftCtx dc = draft_ctx(smem_raw, bars, tmem_slot, &epi_count);
            MK_DRAFT_RANGE;
            dk::RingPos dcc = handoff_get(dho.cc);
            dk::WrkRole role;
            const dk::Instr fin = dc.a->instrs[d1];
            role.instr(fin, d1, d1, d_end, *dc.a, dc.sm, dcc, pt_, d_epoch);
            d_ring = fin.r1 - fin.r0;
            draft_entered = true;
          }
          if (!is_dry && in.dep != NO_CTR) {
            if (pt_ == 0) wait_counter(a.counters + in.dep, (ep + 1) * a.ctr_count[in.dep]);
            named_sync(BAR_PRO, NPRO);
          }
          if (!ss_enabled) continue;
          MK_ROWS;
          if (pt_ == 0) mk_progress(a.trace, 6, k - ii, 0x20u | (uint32_t)is_dry);
          // (first: the draft rows' stats, their inputs the launch's)
          const uint32_t off = (((DRAFT_XB_OFF + SS_DRAFT_OFF) & ssd_m) | ((uint32_t)(NSTAGE * STAGE_BYTES) & ~ssd_m)) & ~dmask |
                               ((uint32_t)(NSTAGE * STAGE_BYTES + B_AREA_BYTES) & dmask);
          ss_phase(sd, T, smem_raw + off, pt_, xop - OP_SS_A, item, xop == OP_SS_A && k == min(ii, nss), is_dry);
        }
        named_sync(BAR_PRO, NPRO);
        if (pt_ == 0 && in.done != NO_CTR) {  // (ss_list: twice, the step's epilogue warps count the step's list only)
          const uint32_t n = ss_list ? 2u : 1u;
          if (in.done & DONE_RELAXED) red_relaxed_add(a.counters + (in.done & ~DONE_RELAXED), n);
          else red_release_add(a.counters + in.done, n);
        }
        if (pt_ == 0) mk_stamp(a.strace, 3, ii - *reinterpret_cast<const volatile uint32_t*>(&irange_sh[0]));
      }
      if (ssd) {  // the sampler's consumers in the draft's list: its last counter
        if (pt_ == 0) mk_progress(a.trace, 6, s1, 0x30u);
        if (pt_ == 0) wait_counter(a.counters + a.draft_sampler, (ep + 1) * a.ctr_count[a.draft_sampler]);
        if (pt_ == 0) mk_progress(a.trace, 6, s1, 0x40u);
        named_sync(BAR_PRO, NPRO);
      }
    }
    if (a_param.draft) {  // the draft's phase: the workers' registers, the rest back to the pool (before any wait)
      if (!ssd) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(DREG_HI));
        draft_sync();
        draft_sync();
      }
      const DraftCtx dc = draft_ctx(smem_raw, bars, tmem_slot, &epi_count);
      MK_DRAFT_RANGE;
      const int d_pt = threadIdx.x - W_PRO0 * 32;
      dk::RingPos dcc = handoff_get(dho.cc);
      dcc.advance(d_ring, dk::NSTAGE);
      dk::WrkRole role;
      for (uint32_t ii = ssd ? s1 : d1; ii < d_end; ii++) {
        const dk::Instr in = dc.a->instrs[ii];
        role.instr(in, ii, d1, d_end, *dc.a, dc.sm, dcc, d_pt, d_epoch);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  // without the draft the kernel ends right after the target's last instruction: the MMA warp's commits nobody waited
  // for (the ring's last NSTAGE positions, the B buffers' last loads) are waited here, so that no tcgen05.commit
  // arrival lands in this CTA's shared memory after it has exited; the draft's transition waits its own (ring reset)
  if (!a.draft && threadIdx.x == W_MMA * 32) {
    const volatile uint32_t* h = dho.cc;
    mma_drain(smem_u32(sm.empty), smem_u32(sm.xempty), h[1] | (h[2] << 8), *reinterpret_cast<volatile uint32_t*>(&dho.xg_sp));
  }
  if (threadIdx.x == 0 && a.draft && a.draft->tstamp) {  // (profiling: the draft's frame, its end)
    const dk::Trace frame{nullptr, 0, a.draft->tstamp + (((size_t)blockIdx.x * dk::NROLE + dk::R_SF) * (dk::MAX_INSTR + 1) + dk::MAX_INSTR) * dk::NSUB};
    frame.stamp(3), frame.put(5, clock64());
  }
  cluster_sync();  // (no CTA exits while the other one may still store into its shared memory)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (ctr_before(ld_relaxed(a.epoch + 1), (epoch + 1) * gridDim.x)) {
    }
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(a.epoch), "r"(epoch + 1) : "memory");
    if (a.draft) {  // the draft's epoch, as its standalone launch advances it
      uint32_t* de = a.draft->epoch;
      const uint32_t e = *reinterpret_cast<const volatile uint32_t*>(&dho.epoch);
      while (ctr_before(ld_relaxed(de + 1), (e + 1) * gridDim.x)) {
      }
      asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(de), "r"(e + 1) : "memory");
    }
  }
  if (warp == W_MMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(sm.tmem), "n"(T_COLS));
  }
#undef MK_ROLE_RANGE
#undef MK_ROWS
#undef MK_DRAFT_RANGE
}

}  // namespace mk

#ifndef PEAR_DEVICE_ONLY
// ------------------------------------------------------------------------------------------------ host
// One verify step: the schedule's launch over nsm CTAs.
void mk_run(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters, torch::Tensor ctr_count,
            torch::Tensor epoch, int64_t nsm, torch::Tensor rec, int64_t trace, int64_t strace,
            std::optional<torch::Tensor> rec_hdr, int64_t draft, int64_t sm_draft, int64_t draft_sampler, int64_t draft_aux) {
  const at::cuda::CUDAGuard guard(instrs.device());
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(mk::mk_main, cudaFuncAttributeMaxDynamicSharedMemorySize, mk::SMEM_BYTES);
    attr_set = true;
  }
  mk::MKArgs a;
  a.instrs = reinterpret_cast<const mk::Instr*>(instrs.data_ptr());
  a.sm_off = reinterpret_cast<const uint32_t*>(sm_off.data_ptr());
  a.descs = reinterpret_cast<const mk::OpDesc*>(descs.data_ptr());
  a.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  a.ctr_count = reinterpret_cast<const uint32_t*>(ctr_count.data_ptr());
  a.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
  a.trace = reinterpret_cast<uint64_t*>(trace);
  a.rec = reinterpret_cast<const int32_t*>(rec.data_ptr());
  a.strace = reinterpret_cast<uint64_t*>(strace);
  // the record's header as a launch argument (a CPU int32 tensor of SR_BT words), else read from rec
  mk::RecParam rp{};
  a.rec_param = rec_hdr.has_value() ? 1u : 0u;
  if (rec_hdr.has_value()) {
    TORCH_CHECK(rec_hdr->device().is_cpu() && rec_hdr->scalar_type() == torch::kInt32 && rec_hdr->is_contiguous() &&
                    rec_hdr->numel() >= mk::SR_BT,
                "mk_main: the record header, a CPU int32 tensor of SR_BT words");
    memcpy(rp.w, rec_hdr->data_ptr<int32_t>(), sizeof(rp.w));
  }
  a.draft = reinterpret_cast<const dk::DkArgs*>(draft);  // (device addresses; 0: no draft)
  a.sm_draft = reinterpret_cast<const uint32_t*>(sm_draft);
  a.draft_sampler = (uint32_t)draft_sampler;
  a.draft_aux = (uint32_t)draft_aux;
  TORCH_CHECK(nsm % 2 == 0, "mk_main: clusters of 2 CTAs");
  TORCH_CHECK(!draft || sm_draft, "mk_main: the draft's first instructions");
  mk::mk_main<<<(int)nsm, mk::NTHREADS, mk::SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(a, rp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// TMA descriptor over a row-major byte matrix w[rows][row_bytes] for the MMA's A tiles: uint8 elements, box 128 bytes
// x box_rows rows, 128-byte swizzle; returned as a CPU uint8 tensor [128] to be copied into device memory.
torch::Tensor mk_make_tmap(torch::Tensor w, int64_t rows, int64_t row_bytes, int64_t box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
  if (!enc) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &q) ==
                        cudaSuccess &&
                    q == cudaDriverEntryPointSuccess,
                "cuTensorMapEncodeTiled unavailable");
  }
  TORCH_CHECK(row_bytes % 128 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "shape/alignment");
  auto out = torch::zeros({128}, torch::kUInt8);
  CUtensorMap m;
  cuuint64_t gdim[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {128, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, w.data_ptr(), gdim, gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  memcpy(out.data_ptr<uint8_t>(), &m, 128);
  return out;
}

// (host) Waits until flags[slot] (pinned host memory, a host ring slot's flag OP_PREV writes once the slot is written)
// equals seq, by acquire loads (the slot's data read after it is the device's); false after timeout_us
bool mk_wait_flag(torch::Tensor flags, int64_t slot, int64_t seq, int64_t timeout_us) {
  TORCH_CHECK(flags.is_pinned() && flags.scalar_type() == torch::kInt32 && slot >= 0 && slot < flags.numel(), "flags");
  const int32_t* p = flags.data_ptr<int32_t>() + slot;
  const int32_t want = (int32_t)seq;
  const auto t0 = std::chrono::steady_clock::now();
  constexpr uint32_t CLOCK_POLLS_MASK = 255u;  // the clock read once per 256 polls
  for (uint32_t i = 0;; i++) {
    if (__atomic_load_n(p, __ATOMIC_ACQUIRE) == want) return true;
    if ((i & CLOCK_POLLS_MASK) == CLOCK_POLLS_MASK) {
      if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(timeout_us)) return false;
      std::this_thread::yield();
    }
  }
}

TORCH_LIBRARY(pear_deploy_blackwell_bf16, m) {
  m.def("run(Tensor instrs, Tensor sm_off, Tensor descs, Tensor(a!) counters, Tensor ctr_count, Tensor(b!) epoch, int nsm, "
        "Tensor rec, int trace=0, int strace=0, Tensor? rec_hdr=None, int draft=0, int sm_draft=0, int draft_sampler=0, "
        "int draft_aux=0) -> ()");
  m.def("make_tmap(Tensor w, int rows, int row_bytes, int box_rows) -> Tensor");
  m.def("topk_logprobs(Tensor logits, Tensor sampled, Tensor(a!) part, Tensor(b!) row_done, int chunks, Tensor(c!) ids, "
        "Tensor(d!) logprobs, Tensor(e!) ranks) -> ()");
  m.def("wait_flag(Tensor flags, int slot, int seq, int timeout_us) -> bool");
}
TORCH_LIBRARY_IMPL(pear_deploy_blackwell_bf16, CUDA, m) {
  m.impl("run", &mk_run);
  m.impl("topk_logprobs", &mk_topk_logprobs);
}
TORCH_LIBRARY_IMPL(pear_deploy_blackwell_bf16, CompositeExplicitAutograd, m) {
  m.impl("make_tmap", &mk_make_tmap);
  m.impl("wait_flag", &mk_wait_flag);
}
#endif  // PEAR_DEVICE_ONLY
