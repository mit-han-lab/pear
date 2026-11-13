// Gated DeltaNet core of a step of T <= 8 tokens under replaySSM, as megakernel instructions: one instruction per
// (value head hv, third of its 128 state rows), 144 items (GDN_ITEMS), run by the 256 prologue threads: the state rows
// replay the previous block's committed tokens, then give this block's outputs by the delta rule's parallel form.
#pragma once
#include "common.cuh"
#include "step.cuh"

namespace mk {

constexpr int GT_H = 16, GT_HV = 48, GT_K = 128, GT_V = 128;
constexpr int GT_CONV = 2 * GT_H * GT_K + GT_HV * GT_V;  // 10240: q, k, v channels
constexpr int GT_QOFF = 0, GT_KOFF = GT_H * GT_K, GT_VOFF = 2 * GT_H * GT_K;
constexpr int GT_ITEMS = GT_HV * 3;
constexpr int GT_MAXT = 8;
constexpr int GT_SEQ = GT_MAXT + 3;  // record entries per channel: the window (3), the block's inputs (<= MAXT)
constexpr int GT_MAXR = 44;          // state rows per item (42, 42, 44)
constexpr int GT_DMODEL = 5120;      // in_proj_ba's K (GT_BA_DOT)
constexpr int GT_NT = 256;           // threads
// the instances compiled in: T = 8 (bit 1), T = 4 (5), any T (2), the split mode's replay (3), its T = 8 / 4 cores (6)
#define GT_INSTANCES 38
constexpr int GT_INST_T1 = 1;        // GT_INSTANCES' bits: T = 1
constexpr int GT_INST_T8 = 2;        // T = 8
constexpr int GT_INST_ANY = 4;       // any T
constexpr int GT_INST_DIR = 16;      // the direct mode's T = 1
constexpr int GT_INST_T4 = 32;       // T = 4
constexpr int GT_INST_REP = 64;      // the split mode's T = 8 / 4
constexpr int GT_ROW_WARPS = 6;      // warps 0-5: the state rows (4 threads a row), the rows' MMAs
constexpr int GT_COEF_PT0 = 192;     // warps 6-7 (from this thread): the coefficients (gt_coeffs)
constexpr int GT_PAIR_THREADS = 64;  // two warps: bar2's (warps 6-7; warps 0-1 in the output phase)
constexpr int GT_Q_THREADS = 128;    // threads pt < GT_Q_THREADS: the q channels, the others k (pt - GT_Q_THREADS)
constexpr int GT_GATE_PT0 = 32;      // this block's gates: threads GT_GATE_PT0 + t (the replayed tokens': t)
constexpr float GT_L2_EPS = 1e-6f;   // FLA's l2norm epsilon (q and k)

// the replay record of one layer, one copy (two by step parity): bytes
constexpr uint32_t GTR_HDR = 0;      // int32 [GTR_P] the block's first position P, [GTR_T] its tokens T
constexpr int GTR_P = 0;             // GTR_HDR's word P
constexpr int GTR_T = 1;             // its word T
constexpr uint32_t GTR_SEQ = 256;    // bf16 [GT_SEQ][GT_CONV]: entry e, the conv input at position P - 3 + e
constexpr uint32_t GTR_AB = GTR_SEQ + GT_SEQ * GT_CONV * 2;  // bf16 [GT_MAXT][2 * GT_HV]: head h's b at h, a at 48 + h
constexpr uint32_t GTR_BYTES = (GTR_AB + GT_MAXT * 2 * GT_HV * 2 + 255) / 256 * 256;

// the step's words for this op (int32, GdnTDesc::step)
enum GdnStepWord : int {
  GS_T = 0,         // the block's tokens (1..GT_MAXT)
  GS_FIRST = 1,     // 1: no record (the first step after a vLLM step of the request)
  GS_PARITY = 2,    // the record copy this step writes (it reads 1 - parity)
  GS_BLOCK = 3,     // vLLM's mamba block size (tokens, > GT_MAXT)
  GS_CONV_OFF = 4,  // GS_FIRST: the window's first entry in vLLM's conv slot (num_accepted - 1; 0 after a prefill)
  GS_DIRECT = 5,    // 1 (T = 1 only): the block's token is committed (a q_len = 1 step, K = 0): the state after it is
                    // written (after the release) and the record holds nothing to replay
  GS_C_PTR = 6,     // [6, 7] uint64: the previous block's committed count (int32 on the device)
  GS_POS_PTR = 8,   // [8, 9] uint64: this block's first position, the request's computed tokens (int32 on the device)
  GS_C_HOST = 10,   // 1 + the previous block's committed count when the host knows it (no dependent loads: the count
                    // and the record's length; the q_len = 1 step's: 0 after a direct step); 0: from GS_C_PTR
  GS_BT = 16,       // [16 + 2g, 17 + 2g] uint64: mamba group g's block-table row (int32 slot of each block column)
};
constexpr int GS_MAX_GROUPS = 16;  // mamba groups at most (the spec engine's model has 10)
constexpr int GS_WORDS = GS_BT + 2 * GS_MAX_GROUPS;

enum GdnInMode : uint32_t {  // this block's conv inputs x(t, ch) = bf16(q_alpha * sum)
  GT_IN_LL = 0,   // in_proj_qkv's sums per part as tagged words {fp32, (epoch + 1) << 1 | part 0 of two}, polled:
                  // word (part, t, ch) at qw + part * qw_part + ch * qw_ch + t * qw_t
  GT_IN_BF16 = 2,  // after the dependency's counter: in_proj_qkvz's bf16 output, x(t, ch) at qw + t * qw_t + ch; the
                   // gated norm's z at column GT_CONV + its row
  GT_IN_ACC = 1,  // after the dependency's counter: in_proj_qkv's fp32 accumulator, copy epoch & 1 (qw + parity *
                  // qw_part bytes), sum (t, ch) at ch * qw_ch + t * qw_t floats
};
enum GdnBaMode : uint32_t {  // b / a of this block's tokens: in_proj_ba's bf16 output
  GT_BA_DOT = 0,   // T = 1: dot products of its rows (ba_w) with the normed input bf16 at GT_XN_OFF
  GT_BA_SMEM = 1,  // fp32 sums [2][GT_MAXT] (b, then a) at GT_BA_OFF, left by in_proj_qkv's prologue on this SM
  GT_BA_OP = 2,    // after the dependency's counter: an op's bf16 output, token t's b of head h at
                   // ba + t * 2 * GT_HV + h, a at + 48 (copy epoch & 1 at + ba_stride bytes)
};
constexpr uint32_t GT_BETA_BF16 = 1;  // GdnTDesc::flags: beta rounded to bf16
// GdnTDesc::flags: the split mode's core: the replay is an instruction of its own (gdn_t_replay_any, OP_GDNT_PRE, any
// earlier instruction of this SM: it depends only on the previous step), leaving this block's start state in `run` and
// its window in its record copy (entries 0..2), where the core starts from (no replay, no previous block's record)
constexpr uint32_t GT_REPLAYED = 2;
// GdnTDesc::flags: the output phase applies out_proj's gated RMSNorm (FLA's y = (x * rstd) * w * (z * sigmoid(z)) per
// head, x = bf16(o), the head's three items' sums of squares through ssx) and writes e4m3(clamp(y * inv_s)) into act in
// out_proj's B layout, which out_proj only bulk-copies (PRO_ACT)
constexpr uint32_t GT_GNORM = 4;

struct GdnTDesc {        // an OpDesc's 256 bytes (schedule.py GDNT)
  uint64_t conv_w;       // bf16 [GT_CONV][4]
  uint64_t a_log;        // fp32 [HV]
  uint64_t dt_bias;      // bf16 [HV]
  uint64_t run;          // fp32 [HV][V][K]: the running state (the start of the previous block), each row's float4 j at
                         // j ^ (r & 7), r the row's index in its item: the swizzled layout of H, which the core's bulk
                         // copy then lands as its MMAs read it
  uint64_t ssm_state;    // vLLM's fp32 state cache, slot stride ssm_slot_stride floats
  uint64_t conv_state;   // vLLM's bf16 conv cache [slots][conv_len][GT_CONV], slot stride conv_slot_stride elements
  uint64_t rec;          // the replay record: 2 copies of GTR_BYTES
  uint64_t qw;           // this block's conv inputs (GdnInMode)
  uint64_t ll;           // the output as tagged words [GT_MAXT][HV][V / 2] {bf16 pair, launch token}
  uint64_t out;          // the output bf16 [GT_MAXT][HV * V]
  uint64_t step;         // the step's words (GdnStepWord)
  uint64_t ba;           // GT_BA_DOT: in_proj_ba's weights bf16 [2 * HV][GT_DMODEL]; GT_BA_OP: its output
  uint64_t gate_src;     // the next instruction's gate z: its fp32 sums, copy 0
  float scale;           // K^-0.5
  float q_alpha;         // in_proj_qkv's alpha (sum -> value)
  uint32_t conv_slot_stride, ssm_slot_stride;
  uint32_t conv_len;     // vLLM's conv state length (3 + speculative tokens)
  uint32_t qw_part, qw_ch;
  uint32_t group;        // the layer's mamba KV cache group
  uint32_t in_mode, ba_mode, flags;
  uint32_t ba_stride;    // GT_BA_OP: bytes between its parity copies (0: one)
  uint32_t gate_stride;  // the gate z's parity stride (bytes)
  uint32_t qw_t;         // `qw`'s stride (words / floats) between the tokens of a channel (0: 1, the tokens contiguous)
  // GT_GNORM:
  uint64_t zw;           // in_proj_z's tagged words [NT][zrs] {bf16, token16} (0: layer 0, z from the qkvz sums at qw)
  uint64_t act;          // out_proj's FP8 operand: K chunk (head) h's GT_MAXT rows at act + h * GT_CHUNK_B (B layout)
  uint64_t nw;           // the gated norm's weight bf16 [V]
  uint64_t ssx;          // the heads' partial sums of squares {fp32, token32} [HV][3][GT_MAXT]
  float eps, inv_s;      // the gated norm's epsilon; 1 / out_proj's FP8 input scale
  uint32_t zrs;          // in_proj_z's token row stride (values)
  uint32_t act_stride;   // bytes between act's parity copies: out_proj polls copy epoch & 1 for bytes still 0xFF (no
                         // code is: |q| <= 448), so the item's bytes of the other copy go back to 0xFF here (0: one
                         // copy, copied after this op's counter)
  uint32_t pad[12];
};
static_assert(sizeof(GdnTDesc) == 256, "GdnTDesc layout");
constexpr uint32_t GT_CHUNK_B = GT_MAXT * 128;  // one K chunk of a GEMV's B operand: GT_MAXT rows x 128 bytes


// smem (the prologue's 32 KB staging buffer): first H, the state rows [44][128] fp32 (bulk-copied; after the replay
// float4 j of row r at j ^ (r & 7): the MMA's fragment loads conflict-free), then GtMisc
constexpr uint32_t GT_MISC_OFF = GT_MAXR * GT_K * 4;  // 22528
// the first layer's out_proj gate copy (GT_IN_ACC, T = 1) and GT_BA_DOT's normed input (T = 1): modes the RTX
// Blackwell step does not use
constexpr uint32_t GT_GZ_OFF = 0;
constexpr uint32_t GT_XN_OFF = 0;
// V: the raw keys (slots 0..7) and queries (8..15) [16][128] bf16, element k of slot n at k ^ 8 (n & 7); VV: the raw
// values of the item's rows [8][44] bf16; RS: the rows' scratch [48][16] fp32, their dot products with the keys (0..7)
// and queries (8..15), after the outputs the corrections e
constexpr int GT_V_SLOTS = 16;      // V's vectors: the keys, then the queries
constexpr uint32_t GT_V_OFF = 24576, GT_VV_OFF = GT_V_OFF + GT_V_SLOTS * GT_K * 2,
                   GT_RS_OFF = GT_VV_OFF + GT_MAXT * GT_MAXR * 2;
constexpr int GT_RS = 2 * GT_MAXT;  // floats per row of RS
constexpr int GT_RS_ROWS = 48;      // rows of RS: the rows' MMAs' three 16-row tiles
// the replay's raw keys RP [8][128] bf16 (V's swizzle) and values RPV [8][44] bf16 in RS, apart from V / VV, which this
// block's conv overwrites while slower warps still replay (every warp is past the replay at the barrier after the conv)
constexpr uint32_t GT_RP_OFF = GT_RS_OFF, GT_RPV_OFF = GT_RP_OFF + GT_MAXT * GT_K * 2;
constexpr uint32_t GT_BA_BYTES = 64;  // GT_BA_SMEM's sums: fp32 [2][GT_MAXT]
constexpr uint32_t GT_BA_OFF = GT_RS_OFF + GT_RS_ROWS * 2 * GT_MAXT * 4;
constexpr uint32_t GT_STG_BYTES = GT_BA_OFF + GT_BA_BYTES;  // the staging buffer the core needs
constexpr uint32_t GT_STG_CAP = 32768;                       // the prologue's staging buffer
static_assert(GT_STG_BYTES <= GT_STG_CAP, "the 5090's staging buffer");
static_assert(GT_RS_OFF + GT_RS_ROWS * GT_RS * 4 <= GT_BA_OFF, "smem");
static_assert(GT_RPV_OFF + GT_MAXT * GT_MAXR * 2 <= GT_RS_OFF + GT_RS_ROWS * GT_RS * 4,
              "smem: the replay's inputs in RS");

struct GtMisc {  // at GT_MISC_OFF (<= 2 KB)
  float red_q[4][GT_MAXT], red_k[4][GT_MAXT];  // per-warp sums of squares of this block's raw q, k
  float red_p[4][GT_MAXT];                      // ... of the replayed tokens' raw keys
  float M[GT_MAXT][GT_MAXT];  // b_t exp(G_t - G_i) k_i.k_t (i < t) of the normed keys (first the raw k_t.k_i)
  float N[GT_MAXT][GT_MAXT];  // exp(G_t - G_i) k_i.q_t (i <= t), normed keys, scaled normed queries (first raw)
  float E[GT_MAXT], g[GT_MAXT], beta[GT_MAXT], rq[GT_MAXT], rk[GT_MAXT], G[GT_MAXT];  // this block's
  float gp[GT_MAXT], bp[GT_MAXT];  // the replayed tokens' log decay, beta
  float bb[GT_MAXT], aa[GT_MAXT];  // this block's b, a (bf16 values, for the record)
  float red[2][8];                 // GT_BA_DOT's per-warp sums
  int32_t prm[16];                 // the step's parameters (GP_*)
  float gn[2][GT_MAXT];            // GT_GNORM: warps 0-1's partial sums of squares
  // GT_GNORM's operands, loaded at the core's start (off the output phase's critical path: a load there is an L2 round
  // trip under the weight streams): the descriptor's fields (thread GT_NT - 1), the rows' norm weights (threads < R)
  uint64_t go_act, go_zw, go_ssx;
  float go_eps, go_inv_s;
  uint32_t go_act_stride, go_zrs;
  float go_wn[GT_MAXR];
};
static_assert(GT_MISC_OFF + sizeof(GtMisc) <= GT_V_OFF, "GtMisc");
enum GtParam : int {
  GP_T = 0, GP_C, GP_POS, GP_FIRST, GP_PARITY, GP_BLOCK, GP_BND, GP_CONVOFF, GP_SLOT, GP_BSLOT, GP_DIRECT,
  GP_HNAT,  // H in the natural layout (the T = 1 instance), else swizzled
  GP_HPOST, // H holds the state after the block (the T = 1 instance), else at its start
};

// one item's rows and key head
struct GtItem {
  int hv, rb, v0, R, ih;
  bool has_work;
};
__device__ __forceinline__ GtItem gt_item(uint32_t item) {
  GtItem it;
  it.has_work = item < GT_ITEMS;
  it.hv = it.has_work ? item / 3 : 0;
  it.rb = it.has_work ? item % 3 : 0;
  // rows [v0, v1) of the head: even boundaries (0, 42, 84, 128: the tagged words carry row pairs)
  it.v0 = 2 * ((GT_V / 2) * it.rb / 3);
  const int v1 = it.rb == 2 ? GT_V : 2 * ((GT_V / 2) * (it.rb + 1) / 3);
  it.R = it.has_work ? v1 - it.v0 : 0;
  it.ih = it.hv / (GT_HV / GT_H);
  return it;
}

#define GT_WATCH(cond) \
  while (cond) {       \
  }
__device__ __forceinline__ void gt_mbar_wait(uint64_t* bar, uint32_t parity) { GT_WATCH(!mbar_test(bar, parity)) }
__device__ __forceinline__ void gt_wait_counter(const uint32_t* p, uint32_t target) {
  GT_WATCH(ctr_before(ld_relaxed(p), target))
  (void)ld_acquire(p);
}

__device__ __forceinline__ float gt_bf(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float gt_bfu(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
// causal_conv1d's SiLU of the fp32 FMA chain over (s0, s1, s2, x), bf16
__device__ __forceinline__ float gt_conv(float s0, float s1, float s2, float x, const float (&w)[4]) {
  float acc = 0.f;
  acc = fmaf(s0, w[0], acc);
  acc = fmaf(s1, w[1], acc);
  acc = fmaf(s2, w[2], acc);
  acc = fmaf(x, w[3], acc);
  return bf16_round(div_full(acc, 1.0f + exp_tri(-acc)));
}
// The per-warp sums of v[0..7] (zeros past n): a transpose-reduce (xor 16, 8, 4 halve the vector, 2 and 1 finish), 9
// shuffles; lane l holds the sum of v[(l >> 2) & 7], stored by lanes l % 4 == 0 into out[t] (n == 1: one warp sum)
__device__ __forceinline__ void gt_warp_sums(const float (&v)[GT_MAXT], int n, float* out, int lane) {
  if (n == 1) {
    const float s = warp_sum(v[0]);
    if (lane == 0) out[0] = s;
    return;
  }
  float a[4], b[2], c;
  const bool u16 = lane & 16, u8 = lane & 8, u4 = lane & 4;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const float send = u16 ? v[k] : v[4 + k];
    a[k] = (u16 ? v[4 + k] : v[k]) + __shfl_xor_sync(0xffffffffu, send, 16);
  }
#pragma unroll
  for (int k = 0; k < 2; k++) {
    const float send = u8 ? a[k] : a[2 + k];
    b[k] = (u8 ? a[2 + k] : a[k]) + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  c = (u4 ? b[1] : b[0]) + __shfl_xor_sync(0xffffffffu, u4 ? b[0] : b[1], 4);
  c += __shfl_xor_sync(0xffffffffu, c, 2);
  c += __shfl_xor_sync(0xffffffffu, c, 1);
  const int t = (lane >> 2) & 7;
  if ((lane & 3) == 0 && t < n) out[t] = c;
}
// log decay and beta of a head from its bf16 b, a (fused_sigmoid_gating: softplus with threshold 20)
__device__ __forceinline__ void gt_gate(float b, float a, float alog, float dtb, bool beta_bf16, float& g, float& beta) {
  constexpr float SOFTPLUS_THRESHOLD = 20.0f;  // softplus(x) = x above it
  const float x = a + dtb;
  const float sp = x <= SOFTPLUS_THRESHOLD ? logf(1.0f + exp_tri(x)) : x;
  g = -exp_tri(alog) * sp;
  beta = sigmoid_triton(b);
  if (beta_bf16) beta = bf16_round(beta);
}
__device__ __forceinline__ void gt_ld2(const uint64_t* p, uint64_t& a, uint64_t& b) {  // 2 tagged words (each atomic)
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ uint2 gt_lds2(const void* p) {  // (volatile: re-read, not held in registers)
  uint2 v;
  asm volatile("ld.volatile.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(smem_u32(p)));
  return v;
}
// the swizzled smem layouts: H(r, c) (fp32 index), V(n, k) (bf16 index)
__device__ __forceinline__ int gt_h(int r, int c) { return r * GT_K + ((((c >> 2) ^ (r & 7)) << 2) | (c & 3)); }
__device__ __forceinline__ int gt_v(int n, int k) { return n * GT_K + (k ^ ((n & 7) << 3)); }
// d += a b (m16n8k8, tf32 operands, fp32 accumulators)
__device__ __forceinline__ void gt_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// x = hi + lo as tf32 operands: hi = x rounded to tf32 (half an ulp added to the magnitude, the 13 low bits dropped),
// lo = x - hi (exact in fp32, at most 13 significant bits; the MMA reads its top 11: |error| <= 2^-22 |x|); integer
// operations, as cvt.rna.tf32 is emulated on sm_103
__device__ __forceinline__ void gt_split(float x, uint32_t& hi, uint32_t& lo) {
  constexpr uint32_t TF32_HALF_ULP = 0x1000u;  // bit 12: half an ulp of tf32
  constexpr uint32_t TF32_MASK = 0xffffe000u;  // the bits above the 13 dropped
  hi = (__float_as_uint(x) + TF32_HALF_ULP) & TF32_MASK;
  lo = __float_as_uint(x - __uint_as_float(hi));
}

// The step's words, read by every thread (one broadcast load per warp: no thread waits on another's chain of loads)
struct GtStep {
  int T, first, parity, B, conv_off, direct, c_host;
  const int32_t* cp;  // -> the previous block's committed count
  const int32_t* pp;  // -> this block's first position
  const int32_t* bt;  // -> the layer's group's block-table row
};
// the step's word w: in the megakernel from the CTA's copy of the record header (step.cuh rec_sh: the step's words are
// its words from SR_GDNT on, GdnTDesc::step)
__device__ __forceinline__ int32_t gt_word(int w) {
  return rec_sh[SR_GDNT + w];
}
__device__ __forceinline__ uint64_t gt_word64(int w) {
  return (uint64_t)(uint32_t)gt_word(w) | ((uint64_t)(uint32_t)gt_word(w + 1) << 32);
}
__device__ __forceinline__ GtStep gt_step(const GdnTDesc& g) {
  GtStep s;
  s.T = gt_word(GS_T), s.first = gt_word(GS_FIRST), s.parity = gt_word(GS_PARITY);
  s.B = gt_word(GS_BLOCK), s.conv_off = gt_word(GS_CONV_OFF), s.direct = gt_word(GS_DIRECT);
  s.c_host = gt_word(GS_C_HOST);
  s.cp = reinterpret_cast<const int32_t*>(gt_word64(GS_C_PTR));
  s.pp = reinterpret_cast<const int32_t*>(gt_word64(GS_POS_PTR));
  s.bt = reinterpret_cast<const int32_t*>(gt_word64(GS_BT + 2 * g.group));
  return s;
}

// The dependency of the GT_IN_ACC / GT_BA_OP inputs: a counter and its target (none: ctr == nullptr), and the gate
// copy of the next instruction issued right after it (gate_rng = kc0 | n1 << 8 | nck << 16 | 1 << 31)
struct GtDep {
  const uint32_t* ctr;
  uint32_t target;
  const uint32_t* ctr2;  // a second counter met with the first (in_proj_ba's op), or null
  uint32_t target2;
  uint32_t gate_rng;
  uint64_t* stg2_full;
  // the op's own completion counter and its full target: on a first step the q / k window is read from vLLM's conv slot
  // by the 9 items of a key head, so a boundary candidate's window there waits until every item has released
  const uint32_t* done_ctr;
  uint32_t done_target;
  float* dbg_o;  // (debug) the outputs in fp32 [GT_MAXT][HV * V]; null: off
};

// What the core leaves for the stores after the release (the rows' corrections e, for the boundary candidate, in RS)
struct GtCarry {
  int T, P, bnd, parity, first, direct;
};

// The coefficients of this block (warps 6-7, wt = pt - GT_COEF_PT0, named barrier bar2), from the raw vectors in V:
// warp 6's 16 MMAs give the raw key-key (rows 0..7 of [keys; queries] x keys^T) and query-key (rows 8..15) products,
// warp 7 the norm factors and the prefix sums G of the log decays; then M, N scaled (the rows solve with them)
__device__ __forceinline__ void gt_coeffs(int T, const uint16_t* V, GtMisc* ms, float scale, int wt, int bar2) {
  const int lane = wt & 31, gq = lane >> 2, tg = lane & 3;
  if (wt < 32) {
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int ks = 0; ks < GT_K / 8; ks++) {  // (k slots tg, tg + 4: columns c0, c0 + 1, as the rows' dots below)
      const int c0 = 8 * ks + 2 * tg;
      const uint32_t kp = *reinterpret_cast<const uint32_t*>(V + gt_v(gq, c0));
      const uint32_t qp = *reinterpret_cast<const uint32_t*>(V + gt_v(gq + 8, c0));
      const uint32_t k0 = kp << 16, k1 = kp & 0xffff0000u, q0 = qp << 16, q1 = qp & 0xffff0000u;
      gt_mma(ks & 1 ? d1 : d0, k0, q0, k1, q1, k0, k1);  // A rows: keys 0..7, queries 8..15; B: the keys
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      ms->M[gq][2 * tg + j] = d0[j] + d1[j];
      ms->N[gq][2 * tg + j] = d0[2 + j] + d1[2 + j];
    }
  } else {
    const int t = wt - 32;
    if (t < T) {
      ms->rk[t] = rsqrtf(((ms->red_k[0][t] + ms->red_k[1][t]) + (ms->red_k[2][t] + ms->red_k[3][t])) + GT_L2_EPS);
      ms->rq[t] =
          rsqrtf(((ms->red_q[0][t] + ms->red_q[1][t]) + (ms->red_q[2][t] + ms->red_q[3][t])) + GT_L2_EPS) * scale;
      float acc = 0.f;  // G_t = sum_{j <= t} g_j
#pragma unroll
      for (int j = 0; j < GT_MAXT; j++)
        if (j <= t) acc += ms->g[j];
      ms->G[t] = acc;
      ms->E[t] = exp_tri(acc);
    }
  }
  named_sync(bar2, GT_PAIR_THREADS);
  {  // scaled by the norm factors and the decays exp(G_t - G_i): entry (t, i) by thread 8 t + i
    const int t = wt / GT_MAXT, i = wt % GT_MAXT;
    if (t < T && i <= t) {
      const float f = exp_tri(ms->G[t] - ms->G[i]);
      if (i < t) ms->M[t][i] = ms->beta[t] * f * (ms->M[t][i] * ms->rk[t] * ms->rk[i]);
      ms->N[t][i] = f * (ms->N[t][i] * ms->rk[i] * ms->rq[t]);
    }
  }
}

// The core of one item up to its outputs (the caller then releases and calls gdn_t_finish). stg: the prologue's 32 KB
// staging buffer; stg_full / stg_phase: its mbarrier; bar: a named barrier of the 256 threads, bar2 one of warps 6-7.
// TT: the block's tokens at compile time (0: from the step's words); MODE 1: the replay alone; DIR, REP: gdn_t_core_any
template <int TT, int MODE = 0, int DIR = 0, int REP = 0>
__device__ __forceinline__ void gdn_t_core(uint32_t item, const GdnTDesc& g, uint8_t* stg, uint64_t* stg_full,
                                           uint32_t& stg_phase, uint32_t epoch, int pt, int bar, int bar2, const GtDep& dep,
                                           GtCarry& cy) {
  const GtItem it = gt_item(item);
  GtMisc* ms = reinterpret_cast<GtMisc*>(stg + GT_MISC_OFF);
  // ---- before the dependency (while in_proj_qkv streams): every load at most two dependent round trips deep (the
  // step's words, then what they point to), all in flight at once; the state rows' copy as soon as its source is known
  const GtStep sp = gt_step(g);
  // (split mode: the replay is done; not in a direct step, T = 1 with nothing to replay, whose replay instruction
  // returns at once: such a step runs as without the split)
  const bool is_replayed = REP || (MODE == 0 && (g.flags & GT_REPLAYED) && !(sp.direct && sp.T == 1 && sp.c_host == 1));
  const int T = TT ? TT : sp.T, is_first = is_replayed ? 0 : sp.first, parity = sp.parity;
  const bool is_direct = DIR || (sp.direct && T == 1);  // (K = 0)
  // the previous block's record (replayed: this block's, whose window the replay instruction wrote)
  const uint8_t* rin =
      reinterpret_cast<const uint8_t*>(g.rec) + (size_t)(is_replayed ? parity : 1 - parity) * GTR_BYTES;
  const __nv_bfloat16* sin = reinterpret_cast<const __nv_bfloat16*>(rin + GTR_SEQ);
  float* H = reinterpret_cast<float*>(stg);
  uint16_t* V = reinterpret_cast<uint16_t*>(stg + GT_V_OFF);
  uint16_t* VV = reinterpret_cast<uint16_t*>(stg + GT_VV_OFF);
  float* RS = reinterpret_cast<float*>(stg + GT_RS_OFF);
  uint16_t* RP = reinterpret_cast<uint16_t*>(stg + GT_RP_OFF);
  uint16_t* RPV = reinterpret_cast<uint16_t*>(stg + GT_RPV_OFF);
  const int wp = pt >> 5, lane = pt & 31;
  // the channels: q (pt < GT_Q_THREADS) or k of the key head (its index in the head: pt & (GT_Q_THREADS - 1)); v row
  // v0 + pt (pt < R)
  const bool is_q = pt < GT_Q_THREADS;
  const int ch = is_q ? GT_QOFF + it.ih * GT_K + pt : GT_KOFF + it.ih * GT_K + (pt - GT_Q_THREADS);
  const bool has_v = pt < it.R;
  const int chv = GT_VOFF + it.hv * GT_V + it.v0 + pt;
  auto state_copy = [&](const float* src) {  // (thread 0; one copy per op: the barrier's phase)
    if (it.has_work) {
      mbar_expect_tx(stg_full, it.R * GT_K * 4);
      bulk_g2s(stg, src + ((size_t)it.hv * GT_V + it.v0) * GT_K, it.R * GT_K * 4, stg_full);
    } else {
      mbar_arrive(stg_full);
    }
  };
  if (pt == 0 && !is_first) {
    if (is_replayed) asm volatile("fence.proxy.async.global;" ::: "memory");  // (run: this SM's generic stores)
    state_copy(reinterpret_cast<const float*>(g.run));
  }
  // the conv weights; the record's entries 0 .. NE - 1 of the channels (the window and up to NE - 3 replayed inputs:
  // all a block of TT tokens can need, before its committed count is known), its length, the replayed tokens' b / a
  constexpr int NE = DIR ? 3 : (TT ? TT : GT_MAXT) + 3;
  uint2 wqp = make_uint2(0, 0), wvp = make_uint2(0, 0);  // (bf16 pairs: unpacked where used)
  float eq[NE], ev[NE];
#pragma unroll
  for (int e = 0; e < NE; e++) eq[e] = ev[e] = 0.f;
  int c = 0;
  float rb = 0.f, ra = 0.f;  // (threads pt < c: token pt's b, a)
  if (it.has_work) {
    wqp = __ldg(reinterpret_cast<const uint2*>(g.conv_w) + ch);
    if (has_v) wvp = __ldg(reinterpret_cast<const uint2*>(g.conv_w) + chv);
    if (!is_first) {  // (the committed count: constant during the launch, the same for every layer: L1)
#pragma unroll
      for (int e = 0; e < NE; e++) {
        if (e < 3 || (!is_direct && !is_replayed)) {  // (direct, replayed: the window only)
          eq[e] = __bfloat162float(__ldcg(sin + (size_t)e * GT_CONV + ch));
          if (has_v) ev[e] = __bfloat162float(__ldcg(sin + (size_t)e * GT_CONV + chv));
        }
      }
      if constexpr (!DIR && !REP) {  // (REP: nothing to replay, c = 0)
        // (the count from the host, or from the device clamped to the record's length: two more dependent loads)
        const int craw = sp.c_host ? sp.c_host - 1 : __ldg(sp.cp);
        const int tprev = is_replayed ? 0
                          : sp.c_host ? GT_MAXT
                                      : __ldcg(reinterpret_cast<const int32_t*>(rin + GTR_HDR) + GTR_T);
        if (pt < NE - 3 && !is_replayed) {
          const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(rin + GTR_AB) + pt * 2 * GT_HV;
          rb = gt_bf(ab, it.hv), ra = gt_bf(ab, GT_HV + it.hv);
        }
        c = min(max(craw, 0), tprev);
      }
    }
    if (MODE == 0 && (REP || (g.flags & GT_GNORM))) {  // the output phase's operands (in the record's round trip)
      if (pt < it.R) ms->go_wn[pt] = ld_bf16(reinterpret_cast<const void*>(g.nw), it.v0 + pt);
      if (pt == GT_NT - 1) {
        ms->go_act = g.act, ms->go_zw = g.zw, ms->go_ssx = g.ssx;
        ms->go_eps = g.eps, ms->go_inv_s = g.inv_s, ms->go_act_stride = g.act_stride, ms->go_zrs = g.zrs;
      }
    }
  }
  // Gates: the replayed tokens' (the record's b / a), this block's (GT_BA_SMEM: the dot products' sums; GT_BA_DOT:
  // computed here)
  const float alog = it.has_work ? reinterpret_cast<const float*>(g.a_log)[it.hv] : 0.f;
  const float dtb = it.has_work ? ld_bf16(reinterpret_cast<const void*>(g.dt_bias), it.hv) : 0.f;
  const bool beta_bf16 = g.flags & GT_BETA_BF16;
  // T = 1: fp32 FMAs over the bf16 normed input, warp tree, warps in order
  if (MODE == 0 && TT <= 1 && g.ba_mode == GT_BA_DOT) {
    float sb = 0.f, sa = 0.f;
    if (it.has_work) {
      const uint4* xn = reinterpret_cast<const uint4*>(stg + GT_XN_OFF);
      const uint4* wb = reinterpret_cast<const uint4*>(g.ba) + (size_t)it.hv * (GT_DMODEL / 8);
      const uint4* wa = reinterpret_cast<const uint4*>(g.ba) + (size_t)(GT_HV + it.hv) * (GT_DMODEL / 8);
      constexpr int NIT = (GT_DMODEL / 8 + GT_NT - 1) / GT_NT;
      uint4 wbv[NIT], wav[NIT];  // (defined on every path)
#pragma unroll
      for (int j = 0; j < NIT; j++) {
        const int i = pt + j * GT_NT;
        wbv[j] = wav[j] = make_uint4(0, 0, 0, 0);
        if (i < GT_DMODEL / 8) {
          wbv[j] = __ldg(wb + i);
          wav[j] = __ldg(wa + i);
        }
      }
#pragma unroll
      for (int j = 0; j < NIT; j++) {
        const int i = pt + j * GT_NT;
        if (i < GT_DMODEL / 8) {
          const uint4 x = xn[i];
          const uint32_t xu[4] = {x.x, x.y, x.z, x.w}, bu[4] = {wbv[j].x, wbv[j].y, wbv[j].z, wbv[j].w},
                         au[4] = {wav[j].x, wav[j].y, wav[j].z, wav[j].w};
#pragma unroll
          for (int k = 0; k < 4; k++) {
            sb = fmaf(bf16lo(bu[k]), bf16lo(xu[k]), sb);
            sa = fmaf(bf16lo(au[k]), bf16lo(xu[k]), sa);
            sb = fmaf(bf16hi(bu[k]), bf16hi(xu[k]), sb);
            sa = fmaf(bf16hi(au[k]), bf16hi(xu[k]), sa);
          }
        }
      }
    }
    sb = warp_sum(sb);
    sa = warp_sum(sa);
    if (lane == 0) {
      ms->red[0][wp] = sb;
      ms->red[1][wp] = sa;
    }
    named_sync(bar, GT_NT);
    if (pt == GT_GATE_PT0) {  // (the gates below: one thread per token)
      float tb = 0.f, ta = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) {
        tb += ms->red[0][w];
        ta += ms->red[1][w];
      }
      ms->bb[0] = bf16_round(tb);
      ms->aa[0] = bf16_round(ta);
    }
  } else if (MODE == 0 && (REP || g.ba_mode == GT_BA_SMEM) && pt >= GT_GATE_PT0 && pt < GT_GATE_PT0 + T) {
    const float* sums = reinterpret_cast<const float*>(stg + GT_BA_OFF);
    ms->bb[pt - GT_GATE_PT0] = bf16_round(sums[pt - GT_GATE_PT0]);  // in_proj_ba's bf16 output
    ms->aa[pt - GT_GATE_PT0] = bf16_round(sums[GT_MAXT + pt - GT_GATE_PT0]);
  }
  {  // a gate per thread: the replayed tokens' (threads 0..c-1), this block's (GT_GATE_PT0 + t; GT_BA_OP: below)
    const bool is_replay_gate = pt < c,
               is_block_gate = MODE == 0 && (REP || g.ba_mode != GT_BA_OP) && pt >= GT_GATE_PT0 && pt < GT_GATE_PT0 + T;
    if (is_replay_gate || is_block_gate) {
      float b = rb, a = ra;
      if (is_replay_gate && pt >= NE - 3) {  // (a previous block longer than this instance's TT: not prefetched)
        const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(rin + GTR_AB) + pt * 2 * GT_HV;
        b = gt_bf(ab, it.hv), a = gt_bf(ab, GT_HV + it.hv);
      }
      if (!is_replay_gate) b = ms->bb[pt - GT_GATE_PT0], a = ms->aa[pt - GT_GATE_PT0];
      float gg, be;
      gt_gate(b, a, alog, dtb, beta_bf16, gg, be);
      (is_replay_gate ? ms->gp : ms->g)[is_replay_gate ? pt : pt - GT_GATE_PT0] = gg;
      (is_replay_gate ? ms->bp : ms->beta)[is_replay_gate ? pt : pt - GT_GATE_PT0] = be;
    }
  }
  // this block's position (the record's header, the boundary candidate; a first step's slot below): loaded now, into
  // the parameters after the inputs (no barrier before them waits on its dependent loads)
  const int P = pt == 0 ? __ldg(sp.pp) : 0;
  auto store_params = [&]() {
    if (pt == 0) {
      const int j = sp.B - P % sp.B;  // (this block's tokens up to the next block boundary)
      ms->prm[GP_POS] = P;
      ms->prm[GP_BND] = it.has_work && j <= T ? j : 0;
      ms->prm[GP_HNAT] = TT == 1;
      ms->prm[GP_HPOST] = TT == 1;
    }
  };

  // the window and the replay's conv: the replayed tokens' raw keys (RP) and values (RPV), the keys' per-warp sums of
  // squares; each channel's window shifted through the c replayed inputs (this block's window)
  float w0 = eq[0], w1 = eq[1], w2 = eq[2], u0 = ev[0], u1 = ev[1], u2 = ev[2];
  if (is_first && it.has_work) {  // (CTA-uniform) the window from vLLM's conv slot of the block of position P - 1
    if (pt == 0) {
      ms->prm[GP_SLOT] = __ldcg(sp.bt + (P - 1) / sp.B);
      state_copy(reinterpret_cast<const float*>(g.ssm_state) + (size_t)ms->prm[GP_SLOT] * g.ssm_slot_stride);
    }
    named_sync(bar, GT_NT);
    const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(g.conv_state) +
                              (size_t)ms->prm[GP_SLOT] * g.conv_slot_stride + (size_t)sp.conv_off * GT_CONV;
    w0 = gt_bf(cs, ch), w1 = gt_bf(cs, GT_CONV + ch), w2 = gt_bf(cs, 2 * GT_CONV + ch);
    if (has_v) u0 = gt_bf(cs, chv), u1 = gt_bf(cs, GT_CONV + chv), u2 = gt_bf(cs, 2 * GT_CONV + chv);
  } else if (is_first && pt == 0) {
    state_copy(nullptr);  // (no item: keeps the barrier's phase sequence)
  }
  if (it.has_work && c > 0) {  // (CTA-uniform)
    const float wq[4] = {bf16lo(wqp.x), bf16hi(wqp.x), bf16lo(wqp.y), bf16hi(wqp.y)};
    const float wv[4] = {bf16lo(wvp.x), bf16hi(wvp.x), bf16lo(wvp.y), bf16hi(wvp.y)};
    float sq[GT_MAXT];
#pragma unroll
    for (int s = 0; s < GT_MAXT; s++) {
      sq[s] = 0.f;
      if (s < c) {
        const float x =
            3 + s < NE && !is_direct ? eq[3 + s < NE ? 3 + s : 0] : gt_bf(sin, (size_t)(3 + s) * GT_CONV + ch);
        if (!is_q) {  // (keys: warps 4-7)
          const float y = gt_conv(w0, w1, w2, x, wq);
          RP[gt_v(s, pt - GT_Q_THREADS)] = __bfloat16_as_ushort(__float2bfloat16(y));
          sq[s] = y * y;
        }
        w0 = w1, w1 = w2, w2 = x;
      }
    }
    if (!is_q) gt_warp_sums(sq, c, ms->red_p[wp - 4], lane);
    if (has_v) {
#pragma unroll
      for (int s = 0; s < GT_MAXT; s++) {
        if (s < c) {
          const float x =
              3 + s < NE && !is_direct ? ev[3 + s < NE ? 3 + s : 0] : gt_bf(sin, (size_t)(3 + s) * GT_CONV + chv);
          RPV[s * GT_MAXR + pt] = __bfloat16_as_ushort(__float2bfloat16(gt_conv(u0, u1, u2, x, wv)));
          u0 = u1, u1 = u2, u2 = x;
        }
      }
    }
  }
  if (MODE == 1 && it.has_work) {  // this block's window (the replay's shifted one) into its record copy, entries 0..2
    __nv_bfloat16* so =
        reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<uint8_t*>(g.rec) + (size_t)parity * GTR_BYTES + GTR_SEQ);
    if (it.hv % 3 == 0 && it.rb == 0) so[ch] = __float2bfloat16(w0), so[GT_CONV + ch] = __float2bfloat16(w1), so[2 * GT_CONV + ch] = __float2bfloat16(w2);
    if (has_v)
      so[chv] = __float2bfloat16(u0), so[GT_CONV + chv] = __float2bfloat16(u1),
          so[2 * GT_CONV + chv] = __float2bfloat16(u2);
  }
  // the state rows in (their wait; the replay, or the pass into the swizzled layout): before the inputs, or for the
  // split mode's core (REP) after this block's conv, which needs only the inputs and the window: its inputs' poll
  // then starts at the core's start, not behind the state's copy and pass
  auto state_rows = [&]() {
    gt_mbar_wait(stg_full, stg_phase);  // the state rows
    stg_phase ^= 1;
    if constexpr (!REP) named_sync(bar, GT_NT);  // (REP: each row's pass by its own warp, after its own wait)

    // ---- the replay: the previous block's c committed tokens into the state rows, token by token per row as vLLM's
    // recurrence (h *= exp(g); h += b (v - h.k) k, k normed), 4 threads per row; the result into the running state
    // (its one write; direct: after the block, gdn_t_finish) and into H (swizzled for the MMAs; TT = 1: natural)
    if (!REP) {  // (CTA-uniform; REP: H lands from run swizzled: no pass; TT = 1: run swizzled, H not: a pass always)
      const int row = pt >> 2, qd = pt & 3;
      const bool has_row = row < it.R;
      float4 h[8];
      if (wp < GT_ROW_WARPS) {  // (warp-uniform: the rows' shuffles take every lane of warps 0-5; rows >= R on zeros)
        float4* hr = reinterpret_cast<float4*>(H) + row * (GT_K / 4);
  #pragma unroll
        for (int m = 0; m < 8; m++)
          h[m] = has_row ? hr[is_first ? qd + 4 * m : (qd + 4 * m) ^ (row & 7)] : make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll 1
        for (int s = 0; s < c; s++) {
          const float r =
              rsqrtf(((ms->red_p[0][s] + ms->red_p[1][s]) + (ms->red_p[2][s] + ms->red_p[3][s])) + GT_L2_EPS);
          const float d = exp_tri(ms->gp[s]);
          const uint16_t* ks = RP + s * GT_K;
          const int sw = (s & 7) << 3;  // (V's swizzle of slot s)
          float a0 = 0.f, a1 = 0.f;
  #pragma unroll
          for (int m = 0; m < 8; m++) {  // h *= decay; the dot with the raw key (times r: the normed key's)
            const uint2 kb = *reinterpret_cast<const uint2*>(ks + ((4 * (qd + 4 * m)) ^ sw));
            h[m].x *= d, h[m].y *= d, h[m].z *= d, h[m].w *= d;
            float& acc = (m & 1) ? a1 : a0;
            acc = fmaf(h[m].x, bf16lo(kb.x), acc);
            acc = fmaf(h[m].y, bf16hi(kb.x), acc);
            acc = fmaf(h[m].z, bf16lo(kb.y), acc);
            acc = fmaf(h[m].w, bf16hi(kb.y), acc);
          }
          float hk = a0 + a1;
          hk += __shfl_xor_sync(0xffffffffu, hk, 1);
          hk += __shfl_xor_sync(0xffffffffu, hk, 2);
          // the correction times r: the update's factor of the raw key
          const float e = ms->bp[s] * ((has_row ? gt_bfu(RPV[s * GT_MAXR + row]) : 0.f) - hk * r) * r;
  #pragma unroll
          for (int m = 0; m < 8; m++) {  // (the key re-read: not held in registers across the dot's reduction)
            const uint2 kb = gt_lds2(ks + ((4 * (qd + 4 * m)) ^ sw));
            h[m].x = fmaf(e, bf16lo(kb.x), h[m].x);
            h[m].y = fmaf(e, bf16hi(kb.x), h[m].y);
            h[m].z = fmaf(e, bf16lo(kb.y), h[m].z);
            h[m].w = fmaf(e, bf16hi(kb.y), h[m].w);
          }
        }
        // the running state (the start of this block)
        if (has_row && it.has_work && !is_direct && (c > 0 || is_first)) {
          float4* run = reinterpret_cast<float4*>(g.run) + (((size_t)it.hv * GT_V + it.v0 + row) * GT_K) / 4;
  #pragma unroll
          for (int m = 0; m < 8; m++) run[(qd + 4 * m) ^ (row & 7)] = h[m];
        }
        __syncwarp();  // (the row's 4 threads, one warp, have read it: the swizzle permutes within the row)
        if (MODE == 0 && has_row) {
  #pragma unroll
          for (int m = 0; m < 8; m++) hr[TT == 1 ? qd + 4 * m : (qd + 4 * m) ^ (row & 7)] = h[m];
        }
      }
    }
  };
  if constexpr (!REP) state_rows();
  if constexpr (MODE == 1) {  // the replay alone: done (every write issued; the staging buffer free for the SM)
    named_sync(bar, GT_NT);
    return;
  }

  // ---- this block's inputs
  // (REP: the layers after the first, whose modes gdn_t_core_any checked: GT_IN_LL words, GT_BA_SMEM sums, GT_GNORM
  // with in_proj_z's words, token-major words; the first layer's paths are not compiled into its code)
  if (!REP && dep.ctr) {  // GT_IN_ACC / GT_BA_OP: the dependency's counter; then the next instruction's gate copy
    if (pt == 0) {
      gt_wait_counter(dep.ctr, dep.target);
      if (dep.ctr2) gt_wait_counter(dep.ctr2, dep.target2);
    }
    named_sync(bar, GT_NT);
    constexpr uint32_t RNG_ON_SHIFT = 31;   // gate_rng's fields: the copy's flag
    constexpr uint32_t RNG_N1_SHIFT = 8;    // n1
    constexpr uint32_t RNG_NCK_SHIFT = 16;  // nck
    constexpr uint32_t RNG_FIELD = 0xff;    // a field's bits (kc0: the lowest)
    if (pt == GT_NT - 1 && (dep.gate_rng >> RNG_ON_SHIFT)) {
      const uint32_t kc0 = dep.gate_rng & RNG_FIELD, n1 = (dep.gate_rng >> RNG_N1_SHIFT) & RNG_FIELD,
                     nck = (dep.gate_rng >> RNG_NCK_SHIFT) & RNG_FIELD;
      const uint8_t* z = reinterpret_cast<const uint8_t*>(g.gate_src + (epoch & 1) * g.gate_stride);
      asm volatile("fence.proxy.async.global;" ::: "memory");
      mbar_expect_tx(dep.stg2_full, nck * 128 * 4);
      bulk_g2s(stg + GT_GZ_OFF, z + kc0 * 128 * 4, n1 * 128 * 4, dep.stg2_full);
      if (nck > n1) bulk_g2s(stg + GT_GZ_OFF + n1 * 128 * 4, z, (nck - n1) * 128 * 4, dep.stg2_full);
    }
  }
  if (!REP && g.ba_mode == GT_BA_OP && pt >= GT_GATE_PT0 && pt < GT_GATE_PT0 + T) {
    const int t = pt - GT_GATE_PT0;
    const unsigned short* o = reinterpret_cast<const unsigned short*>(g.ba + (epoch & 1) * g.ba_stride) + t * 2 * GT_HV;
    ms->bb[t] = __uint_as_float((uint32_t)__ldcg(o + it.hv) << 16);
    ms->aa[t] = __uint_as_float((uint32_t)__ldcg(o + GT_HV + it.hv) << 16);
    gt_gate(ms->bb[t], ms->aa[t], alog, dtb, beta_bf16, ms->g[t], ms->beta[t]);
  }
  // the inputs of the q / k and the v channel: in_proj_qkv's tagged words, polled with every word in flight, or its
  // accumulator
  float xq[GT_MAXT], xv[GT_MAXT];
#pragma unroll
  for (int t = 0; t < GT_MAXT; t++) xq[t] = xv[t] = 0.f;
  if (it.has_work) {
    if (REP || g.in_mode == GT_IN_LL) {  // {fp32 part, (epoch + 1) << 1 | part 0 of two}: value = 0 + p0 (+ p1)
      const uint64_t* q0 = reinterpret_cast<const uint64_t*>(g.qw) + (size_t)ch * g.qw_ch;
      const uint64_t* v0 = reinterpret_cast<const uint64_t*>(g.qw) + (size_t)chv * g.qw_ch;
      const uint32_t tok = (epoch + 1) << 1;
      // (the descriptor's fields in registers: read inside the loop they are reloaded after every polled load)
      const uint32_t qpart = g.qw_part, qwt = g.qw_t;
      const float qalpha = g.q_alpha;
      auto check = [&](const uint64_t (&a)[GT_MAXT], const uint64_t (&b)[GT_MAXT], float (&x)[GT_MAXT],
                       bool& all_current) {
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++) {
          if (t < T) {
            const uint32_t ta = (uint32_t)(a[t] >> 32);
            const bool has_two_parts = ta & 1;
            // (bitwise: short-circuit operators branch, one reconvergence per word)
            all_current &= ((ta & ~1u) == tok) & (!has_two_parts | ((uint32_t)(b[t] >> 32) == tok));
            float v = 0.f + __uint_as_float((uint32_t)a[t]);
            if (has_two_parts) v += __uint_as_float((uint32_t)b[t]);
            x[t] = bf16_round(qalpha * v);
          }
        }
      };
      auto load = [&](const uint64_t* p, uint64_t (&a)[GT_MAXT], uint64_t (&b)[GT_MAXT]) {
        if (T == 1) {  // (one word per channel and part)
          a[0] = ld_relaxed_b64(p), b[0] = ld_relaxed_b64(p + qpart);
        } else if (!REP && qwt <= 1) {  // (the tokens contiguous: 16-byte loads of token pairs)
#pragma unroll
          for (int t = 0; t < GT_MAXT; t += 2) {
            if (t < T) {
              gt_ld2(p + t, a[t], a[t + 1]);
              gt_ld2(p + qpart + t, b[t], b[t + 1]);
            }
          }
        } else {
#pragma unroll
          for (int t = 0; t < GT_MAXT; t++) {
            if (t < T) {
              a[t] = ld_relaxed_b64(p + (size_t)t * qwt);
              b[t] = ld_relaxed_b64(p + qpart + (size_t)t * qwt);
            }
          }
        }
      };
      for (;;) {
        // (every word defined on every path, also past T and without a v channel: a register written only under a
        // condition stays live around the caller's whole instruction loop)
        uint64_t aq[GT_MAXT], bq[GT_MAXT], av[GT_MAXT], bv[GT_MAXT];
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++) aq[t] = bq[t] = av[t] = bv[t] = 0;
        load(q0, aq, bq);
        if (has_v) load(v0, av, bv);
        bool all_current = true;
        check(aq, bq, xq, all_current);
        if (has_v) check(av, bv, xv, all_current);
        if (all_current) break;
      }
    } else if (!REP && g.in_mode == GT_IN_BF16) {  // in_proj_qkvz's bf16 output (the dependency met above)
      const __nv_bfloat16* xo = reinterpret_cast<const __nv_bfloat16*>(g.qw);
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        if (t < T) {
          xq[t] = __bfloat162float(__ldcg(xo + (size_t)t * g.qw_t + ch));
          if (has_v) xv[t] = __bfloat162float(__ldcg(xo + (size_t)t * g.qw_t + chv));
        }
      }
    } else if constexpr (!REP) {  // the accumulator (the dependency met above)
      const float* acc = reinterpret_cast<const float*>(g.qw + (epoch & 1) * g.qw_part);
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        if (t < T) {
          const size_t to = (size_t)t * (g.qw_t ? g.qw_t : 1);
          xq[t] = bf16_round(g.q_alpha * __ldcg(acc + (size_t)ch * g.qw_ch + to));
          if (has_v) xv[t] = bf16_round(g.q_alpha * __ldcg(acc + (size_t)chv * g.qw_ch + to));
        }
      }
    }
  }
  // the conv over [window, inputs]: raw q / k (V: keys at slots t, queries at 8 + t) and v (VV) of the T tokens, the
  // q / k sums of squares per warp; this block's record: [window, inputs] of the q / k channels by one of the 9 items
  // of the key head, of v by its owner (copy parity: no reader in this step); direct: the window after the token only
  {
    __nv_bfloat16* so =
        reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<uint8_t*>(g.rec) + (size_t)parity * GTR_BYTES + GTR_SEQ);
    const bool wq_rec = it.has_work && it.hv % 3 == 0 && it.rb == 0, wv_rec = it.has_work && has_v;
    if (is_direct) {  // (T = 1: the window after the token, entries 0..2: inputs P - 2 .. P)
      if (wq_rec) so[ch] = __float2bfloat16(w1), so[GT_CONV + ch] = __float2bfloat16(w2), so[2 * GT_CONV + ch] = __float2bfloat16(xq[0]);
      if (wv_rec) so[chv] = __float2bfloat16(u1), so[GT_CONV + chv] = __float2bfloat16(u2), so[2 * GT_CONV + chv] = __float2bfloat16(xv[0]);
    } else if (!is_replayed) {  // (replayed: written by the replay instruction)
      if (wq_rec) so[ch] = __float2bfloat16(w0), so[GT_CONV + ch] = __float2bfloat16(w1), so[2 * GT_CONV + ch] = __float2bfloat16(w2);
      if (wv_rec) so[chv] = __float2bfloat16(u0), so[GT_CONV + chv] = __float2bfloat16(u1), so[2 * GT_CONV + chv] = __float2bfloat16(u2);
    }
    const float wq[4] = {bf16lo(wqp.x), bf16hi(wqp.x), bf16lo(wqp.y), bf16hi(wqp.y)};  // (held as bf16 pairs)
    const float wv[4] = {bf16lo(wvp.x), bf16hi(wvp.x), bf16lo(wvp.y), bf16hi(wvp.y)};
    float sq[GT_MAXT];
    const int slot0 = is_q ? GT_MAXT : 0;
#pragma unroll
    for (int t = 0; t < GT_MAXT; t++) {
      sq[t] = 0.f;
      if (t < T) {
        const float y = gt_conv(w0, w1, w2, xq[t], wq);
        V[gt_v(slot0 + t, pt & 127)] = __bfloat16_as_ushort(__float2bfloat16(y));
        sq[t] = y * y;
        w0 = w1, w1 = w2, w2 = xq[t];
        if (wq_rec && !is_direct) so[(size_t)(3 + t) * GT_CONV + ch] = __float2bfloat16(xq[t]);
      }
    }
    gt_warp_sums(sq, T, (is_q ? ms->red_q : ms->red_k)[wp & 3], lane);
    if (has_v) {
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        if (t < T) {
          // (no select on it.has_work: a branch around each token's conv kept the tokens apart; an item without work
          // has zero weights and inputs, whose conv is 0)
          VV[t * GT_MAXR + pt] = __bfloat16_as_ushort(__float2bfloat16(gt_conv(u0, u1, u2, xv[t], wv)));
          u0 = u1, u1 = u2, u2 = xv[t];
          if (wv_rec && !is_direct) so[(size_t)(3 + t) * GT_CONV + chv] = __float2bfloat16(xv[t]);
        }
      }
    }
  }
  store_params();
  if constexpr (REP) state_rows();
  named_sync(bar, GT_NT);

  // ---- the outputs
  if constexpr (TT == 1) {
    // T = 1: per row (its 4 threads, warps 0-5; H natural) h.k, h.q of the raw vectors and k.q, then u = b (v - exp(g)
    // h.k^) and o = exp(g) h.q^ + u k^.q^ (normed vectors, q^ scaled): vLLM's one-token update h <- exp(g) h + u k^, o
    // = h.q^; the state after the token into H in place (stored by one bulk copy after the release: gdn_t_finish)
    if (it.has_work && wp < GT_ROW_WARPS) {  // (warp-uniform: the rows' shuffles take every lane)
      const int row = pt >> 2, qd = pt & 3;
      const bool has_row = row < it.R;
      float4* hr = reinterpret_cast<float4*>(H) + (has_row ? row : 0) * (GT_K / 4);
      float hk = 0.f, hq = 0.f, kq = 0.f;
      float4 hs[8];
#pragma unroll
      for (int m = 0; m < 8; m++) {
        const float4 h4 = hr[qd + 4 * m];
        hs[m] = h4;
        const uint2 kb = *reinterpret_cast<const uint2*>(V + gt_v(0, 4 * (qd + 4 * m)));
        const uint2 qb = *reinterpret_cast<const uint2*>(V + gt_v(GT_MAXT, 4 * (qd + 4 * m)));
        const float k[4] = {bf16lo(kb.x), bf16hi(kb.x), bf16lo(kb.y), bf16hi(kb.y)};
        const float q[4] = {bf16lo(qb.x), bf16hi(qb.x), bf16lo(qb.y), bf16hi(qb.y)};
        const float h[4] = {h4.x, h4.y, h4.z, h4.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
          hk = fmaf(h[i], k[i], hk);
          hq = fmaf(h[i], q[i], hq);
          kq = fmaf(k[i], q[i], kq);
        }
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        hk += __shfl_xor_sync(0xffffffffu, hk, o);
        hq += __shfl_xor_sync(0xffffffffu, hq, o);
        kq += __shfl_xor_sync(0xffffffffu, kq, o);
      }
      const float rk = rsqrtf(((ms->red_k[0][0] + ms->red_k[1][0]) + (ms->red_k[2][0] + ms->red_k[3][0])) + GT_L2_EPS);
      const float rq =
          rsqrtf(((ms->red_q[0][0] + ms->red_q[1][0]) + (ms->red_q[2][0] + ms->red_q[3][0])) + GT_L2_EPS) * g.scale;
      const float d = exp_tri(ms->g[0]);
      const float u = ms->beta[0] * (gt_bfu(VV[has_row ? row : 0]) - d * (hk * rk));
      const float o = d * (hq * rq) + u * (kq * rk * rq);
      const float o1 = __shfl_down_sync(0xffffffffu, o, 4);  // (row + 1's: the next 4 lanes)
      if (has_row && qd == 0) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(g.out) + it.hv * GT_V + it.v0 + row;
        *out = __float2bfloat16(o);
        if (dep.dbg_o) dep.dbg_o[it.hv * GT_V + it.v0 + row] = o;
        if (!(row & 1)) {  // rows r (even), r + 1 as one tagged word {bf16 r | bf16 r + 1 << 16, token}
          const __nv_bfloat162 hb = __floats2bfloat162_rn(o, o1);
          st_relaxed_b64(reinterpret_cast<uint64_t*>(g.ll) + (size_t)it.hv * (GT_V / 2) + ((it.v0 + row) >> 1),
                         ((uint64_t)(epoch + 1) << 32) | *reinterpret_cast<const uint32_t*>(&hb));
        }
        RS[row * GT_RS] = u;
      }
      if (has_row) {  // h <- exp(g) h + u k^ (the key re-read: not held in registers)
        const float uk = u * rk;
#pragma unroll
        for (int m = 0; m < 8; m++) {
          const uint2 kb = gt_lds2(V + gt_v(0, 4 * (qd + 4 * m)));
          float4 h = hs[m];
          h.x = fmaf(uk, bf16lo(kb.x), h.x * d);
          h.y = fmaf(uk, bf16hi(kb.x), h.y * d);
          h.z = fmaf(uk, bf16lo(kb.y), h.z * d);
          h.w = fmaf(uk, bf16hi(kb.y), h.w * d);
          hr[qd + 4 * m] = h;
        }
      }
      if (pt == 0) ms->rk[0] = rk;
    }
    fence_async_smem();  // (H's generic writes before the async proxy's bulk store after the release)
  } else {
  // Warps 0-5: the rows' dot products with the keys and queries on the tensor cores (warp w: rows
  // 16 (w / 2).., slots 8 (w % 2)..; the state's fp32 as two tf32 parts, the raw vectors exact in tf32) into RS; warps
  // 6-7: the coefficients
  if (wp >= GT_ROW_WARPS) {
    gt_coeffs(T, V, ms, g.scale, pt - GT_COEF_PT0, bar2);
  } else {
    const int gq = lane >> 2, tg = lane & 3, mt = wp >> 1, nt = wp & 1;
    const int r0 = 16 * mt + gq, r1 = r0 + 8, n = 8 * nt + gq;
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int ks = 0; ks < GT_K / 8; ks++) {
      // k slots tg and tg + 4 of step ks: columns c0 = 8 ks + 2 tg and c0 + 1 (any bijection of the columns onto the
      // slots gives the same sums): one 8-byte load per row, one 4-byte load of the vector's bf16 pair
      const int c0 = 8 * ks + 2 * tg;
      const float2 h0 = *reinterpret_cast<const float2*>(H + gt_h(r0, c0));
      const float2 h1 = *reinterpret_cast<const float2*>(H + gt_h(r1, c0));
      const uint32_t bp = *reinterpret_cast<const uint32_t*>(V + gt_v(n, c0));
      const uint32_t b0 = bp << 16, b1 = bp & 0xffff0000u;
      uint32_t a0, a1, a2, a3, l0, l1, l2, l3;
      gt_split(h0.x, a0, l0), gt_split(h1.x, a1, l1), gt_split(h0.y, a2, l2), gt_split(h1.y, a3, l3);
      float(&d)[4] = ks & 1 ? d1 : d0;
      gt_mma(d, a0, a1, a2, a3, b0, b1);
      gt_mma(d, l0, l1, l2, l3, b0, b1);
    }
    float* rs0 = RS + r0 * GT_RS + 8 * nt + 2 * tg;
    float* rs1 = RS + r1 * GT_RS + 8 * nt + 2 * tg;
    rs0[0] = d0[0] + d1[0], rs0[1] = d0[1] + d1[1];
    rs1[0] = d0[2] + d1[2], rs1[1] = d0[3] + d1[3];
  }
  named_sync(bar, GT_NT);
  // per row (thread r < 48, warps 0-1): u_t = b_t (v_t - exp(G_t) h.k_t); the corrections
  // e_t = u_t - sum_{i<t} M[t][i] e_i by forward substitution (one FMA per token on the dependency chain);
  // o_t = exp(G_t) h.q_t + sum_{i<=t} N[t][i] e_i
  if (it.has_work && pt < GT_PAIR_THREADS) {  // (warp-uniform: the row pairs' shuffles take every lane of warps 0-1)
    const int row = pt;
    const bool has_row = row < it.R;
    const int rowc = has_row ? row : 0;  // (rows past R compute on row 0's values, unstored: no branch per token)
    const float* rs = RS + rowc * GT_RS;
    // (every token computed, the ones past T masked to +0 by a set.lt mask: no zeroed register under a predicated FMA,
    // whose readers ptxas may schedule through the predicated writer; the shared rows past T are in bounds, their
    // values discarded; at a compile-time T the masks are all ones)
    float e[GT_MAXT];
#pragma unroll
    for (int t = 0; t < GT_MAXT; t++) {
      const float4 m0 = reinterpret_cast<const float4*>(ms->M[t])[0], m1 = reinterpret_cast<const float4*>(ms->M[t])[1];
      const float m[GT_MAXT] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
      float x = ms->beta[t] * (gt_bfu(VV[t * GT_MAXR + rowc]) - ms->E[t] * (rs[t] * ms->rk[t]));
#pragma unroll
      for (int i = 0; i < t; i++) x = fmaf(-m[i], e[i], x);
      e[t] = __uint_as_float(__float_as_uint(x) & lt_mask(t, T));
    }
    float o_[GT_MAXT];
#pragma unroll
    for (int t = 0; t < GT_MAXT; t++) {
      const float4 n0 = reinterpret_cast<const float4*>(ms->N[t])[0], n1 = reinterpret_cast<const float4*>(ms->N[t])[1];
      const float nn[GT_MAXT] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
      float o = ms->E[t] * (rs[GT_MAXT + t] * ms->rq[t]);
#pragma unroll
      for (int i = 0; i <= t; i++) o = fmaf(nn[i], e[i], o);
      o_[t] = __uint_as_float(__float_as_uint(o) & lt_mask(t, T));
    }
    if (REP || (g.flags & GT_GNORM)) {
      // out_proj's gated RMSNorm and FP8 operand (GT_GNORM): x = bf16(o), the item's per-token sums of squares (warps
      // 0-1) published and the head's three items' polled (rb 0, 1, 2), z of the rows (in_proj_z's words; layer 0: the
      // qkvz sums); y = bf16((x rstd) w (z sigmoid z)), its e4m3 code of y / s_in into out_proj's B layout (row pairs)
      const uint32_t tok = epoch + 1;
      const int b = it.v0 + row;  // (the row's byte in its head's chunk, in each token row)
      // the rows' z, loaded first (in flight during the sums and the partials' exchange)
      const int chz = it.hv * GT_V + it.v0 + rowc;
      const uint32_t* zw = reinterpret_cast<const uint32_t*>(ms->go_zw);
      const uint32_t zrs = ms->go_zrs;
      uint32_t zr[GT_MAXT];
      float z_[GT_MAXT];
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) zr[t] = 0, z_[t] = 0.f;
      if (REP || zw) {
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          if (t < T) zr[t] = ld_relaxed(zw + (size_t)t * zrs + chz);
      } else if (!REP && g.in_mode == GT_IN_BF16) {  // in_proj_qkvz's bf16 output, its z columns
        const __nv_bfloat16* xo = reinterpret_cast<const __nv_bfloat16*>(g.qw) + GT_CONV + chz;
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          if (t < T) z_[t] = __bfloat162float(__ldcg(xo + (size_t)t * g.qw_t));
      } else if constexpr (!REP) {  // (layer 0: the qkvz accumulator's z sums, final after this op's dependency)
        const float* acc = reinterpret_cast<const float*>(g.qw + (epoch & 1) * g.qw_part) + GT_CONV + chz;
        const float alpha = g.q_alpha;
        const size_t ts = g.qw_t ? g.qw_t : 1;
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          if (t < T) z_[t] = bf16_round(alpha * __ldcg(acc + (size_t)t * ts));
      }
      // x = bf16(o); the item's sums of squares per token over its rows: warp_sum's butterfly, the tokens' shuffles
      // of a level issued together (one shuffle latency per level, not per token and level; the same sums)
      float x_[GT_MAXT], s_[GT_MAXT];
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        // (a set.lt mask, not a select against zero)
        x_[t] = __uint_as_float(__float_as_uint(bf16_round(o_[t])) & lt_mask(row, it.R) & lt_mask(t, T));
        s_[t] = x_[t] * x_[t];
        if (dep.dbg_o && has_row && t < T) dep.dbg_o[(size_t)t * GT_HV * GT_V + it.hv * GT_V + it.v0 + row] = o_[t];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          if (t < T) s_[t] += __shfl_xor_sync(0xffffffffu, s_[t], o);
      }
      if (lane == 0) {
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          if (t < T) ms->gn[wp][t] = s_[t];
      }
      named_sync(bar2, GT_PAIR_THREADS);  // (warps 0-1; the coefficient warps are past their use of this barrier)
      uint64_t* ssx = reinterpret_cast<uint64_t*>(ms->go_ssx) + (size_t)it.hv * 3 * GT_MAXT;
      if (pt < T)  // this item's part of the head's sum of squares for token pt
        st_relaxed_b64(ssx + it.rb * GT_MAXT + pt, ((uint64_t)tok << 32) | __float_as_uint(ms->gn[0][pt] + ms->gn[1][pt]));
      // every thread: the other two items' parts of every token and its rows' z words, polled in one loop (one round
      // trip after the later of the two, not the exchange's then z's)
      const int r1 = it.rb == 0 ? 1 : 0, r2 = it.rb == 2 ? 1 : 2;
      const uint32_t tok16 = tok & 0xffffu;
      uint64_t w1[GT_MAXT], w2[GT_MAXT];
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        w1[t] = w2[t] = (uint64_t)tok << 32;
        if (t < T) w1[t] = ld_relaxed_b64(ssx + r1 * GT_MAXT + t), w2[t] = ld_relaxed_b64(ssx + r2 * GT_MAXT + t);
      }
      for (;;) {
        bool all_current = true;
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++) {
          if (t < T) {
            if ((uint32_t)(w1[t] >> 32) != tok) all_current = false, w1[t] = ld_relaxed_b64(ssx + r1 * GT_MAXT + t);
            if ((uint32_t)(w2[t] >> 32) != tok) all_current = false, w2[t] = ld_relaxed_b64(ssx + r2 * GT_MAXT + t);
            if ((REP || zw) && (zr[t] >> 16) != tok16)
              all_current = false, zr[t] = ld_relaxed(zw + (size_t)t * zrs + chz);
          }
        }
        if (all_current) break;
      }
      constexpr float GNORM_N = 128.0f;  // the head's values: the norm's mean is over them
      constexpr float E4M3_MAX = 448.f;  // the FP8 operand's largest magnitude
      constexpr uint32_t SENTINEL_PAIR = 0xffffu;  // two of the sentinel bytes 0xFF
      const float wn = ms->go_wn[rowc], eps = ms->go_eps, inv_s = ms->go_inv_s;
      uint8_t* act = reinterpret_cast<uint8_t*>(ms->go_act) + (size_t)it.hv * GT_CHUNK_B;
      const uint32_t act_stride = ms->go_act_stride;
      uint8_t* ac = act + (epoch & 1) * act_stride;  // (token t's byte b at pub_byte(t, b): the B slot's swizzle)
      // the codes of the T tokens computed without a branch (four per word), then stored one byte per row: no row-pair
      // shuffle, whose convergence check per token kept the tokens' chains apart
      uint32_t cw[GT_MAXT / 4];
#pragma unroll
      for (int k = 0; k < GT_MAXT / 4; k++) cw[k] = 0u;
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        if (t < T) {
          // the head's sum of squares in the order rb 0, 1, 2 on every item (one rstd per token)
          const float own_sum = ms->gn[0][t] + ms->gn[1][t];
          const float v1 = __uint_as_float((uint32_t)w1[t]), v2 = __uint_as_float((uint32_t)w2[t]);
          const float p0 = it.rb == 0 ? own_sum : v1, p1 = it.rb == 0 ? v1 : it.rb == 1 ? own_sum : v2,
                      p2 = it.rb == 2 ? own_sum : v2;
          const float rstd = rsqrt_tri(((p0 + p1) + p2) / GNORM_N + eps);
          const float zz = (REP || zw) ? bf16lo(zr[t]) : z_[t];
          const float y = bf16_round(((x_[t] * rstd) * wn) * (zz * sigmoid_triton(zz)));
          const float q = fminf(fmaxf(y * inv_s, -E4M3_MAX), E4M3_MAX);
          cw[t >> 2] |= (f32x2_to_e4m3x2(q, 0.f) & 0xffu) << (8 * (t & 3));
        }
      }
      if (has_row) {
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++)
          ac[pub_byte(t, b)] = t < T ? (uint8_t)(cw[t >> 2] >> (8 * (t & 3))) : (uint8_t)0;
      }
      if (act_stride && has_row && !(row & 1)) {  // the next launch's copy back to 0xFF (every row: its T is unknown)
        uint8_t* nx = act + ((epoch + 1) & 1) * act_stride;
#pragma unroll
        for (int t = 0; t < GT_MAXT; t++) *reinterpret_cast<uint16_t*>(nx + pub_byte(t, b)) = SENTINEL_PAIR;
      }
    } else if constexpr (!REP) {
      uint64_t* ll = reinterpret_cast<uint64_t*>(g.ll) + (size_t)it.hv * (GT_V / 2) + ((it.v0 + row) >> 1);
      __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(g.out) + it.hv * GT_V + it.v0 + row;
      const uint64_t ll_token = (uint64_t)(epoch + 1) << 32;
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++) {
        if (t < T) {
          const float o = o_[t];
          // rows r (even), r + 1 as one tagged word {bf16 r | bf16 r + 1 << 16, token}
          const float o1 = __shfl_down_sync(0xffffffffu, o, 1);
          if (has_row) {
            out[(size_t)t * GT_HV * GT_V] = __float2bfloat16(o);
            if (dep.dbg_o) dep.dbg_o[(size_t)t * GT_HV * GT_V + it.hv * GT_V + it.v0 + row] = o;
            if (!(row & 1)) {
              const __nv_bfloat162 hb = __floats2bfloat162_rn(o, o1);
              st_relaxed_b64(ll + (size_t)t * GT_HV * (GT_V / 2), ll_token | *reinterpret_cast<const uint32_t*>(&hb));
            }
          }
        }
      }
    }
    if (ms->prm[GP_BND] && has_row) {  // the corrections, for the boundary candidate (over the key products)
      float* er = RS + row * GT_RS;
#pragma unroll
      for (int t = 0; t < GT_MAXT; t++)
        if (t < T) er[t] = e[t];
    }
  }
  }
  cy.T = T, cy.P = ms->prm[GP_POS], cy.parity = parity, cy.first = sp.first, cy.bnd = ms->prm[GP_BND],
      cy.direct = is_direct;
}

// The core at the step's T: compiled for T = 8 (verify steps of 7 drafts) and T = 4 (3 drafts), any other T <= 8 by
// the generic instance
__device__ __forceinline__ void gdn_t_core_any(uint32_t item, const GdnTDesc& g, uint8_t* stg, uint64_t* stg_full,
                                               uint32_t& stg_phase, uint32_t epoch, int pt, int bar, int bar2,
                                               const GtDep& dep, GtCarry& cy) {
  const int T = gt_word(GS_T);
  cy = GtCarry{0, 0, 0, 0, 0, 0};  // (defined on every path)
  // DIR: the direct mode (T = 1, K = 0: the committed count 0 from the host) without replay code (the q_len = 1 step's
  // instance: its code is fetched cold every layer, the weight streams evict it)
  const bool dir = T == 1 && gt_word(GS_DIRECT) && gt_word(GS_C_HOST) == 1;
  // REP: the split mode's core (GT_REPLAYED: no direct step at T > 1) in the modes of the layers after the first,
  // without replay, first-step or committed-count code (its code is fetched cold every layer)
  const bool rep = (GT_INSTANCES & GT_INST_REP) && (g.flags & GT_REPLAYED) && (g.flags & GT_GNORM) &&
                   g.in_mode == GT_IN_LL && g.ba_mode == GT_BA_SMEM && g.zw != 0 && g.qw_t > 1;
  if ((GT_INSTANCES & GT_INST_DIR) && dir)
    gdn_t_core<1, 0, 1>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if ((GT_INSTANCES & GT_INST_REP) && rep && T == 8)
    gdn_t_core<8, 0, 0, 1>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if ((GT_INSTANCES & GT_INST_REP) && rep && T == 4)
    gdn_t_core<4, 0, 0, 1>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if ((GT_INSTANCES & GT_INST_T8) && T == 8)
    gdn_t_core<8>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if ((GT_INSTANCES & GT_INST_T4) && T == 4)
    gdn_t_core<4>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if ((GT_INSTANCES & GT_INST_T1) && T == 1)
    gdn_t_core<1>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else if (GT_INSTANCES & GT_INST_ANY)
    gdn_t_core<0>(item, g, stg, stg_full, stg_phase, epoch, pt, bar, bar2, dep, cy);
  else __trap();  // (no instance compiled for this step: GT_INSTANCES)
}

// The replay alone (split mode, OP_GDNT_PRE): the state at the start of this step's block into `run`, this block's
// window into its record copy (entries 0..2); the core then runs with GT_REPLAYED. The staging buffer is free after it.
__device__ __forceinline__ void gdn_t_replay_any(uint32_t item, const GdnTDesc& g, uint8_t* stg, uint64_t* stg_full,
                                                 uint32_t& stg_phase, int pt, int bar) {
  const int T = gt_word(GS_T);
  if (T == 1 && gt_word(GS_DIRECT) && gt_word(GS_C_HOST) == 1) return;  // (a direct step: nothing to replay)
  GtCarry cy;
  const GtDep dep{nullptr, 0, nullptr, 0, 0, nullptr, nullptr, 0, nullptr};
  (void)T, (void)item, (void)g, (void)stg, (void)stg_full, (void)stg_phase, (void)pt, (void)bar, (void)dep, (void)cy;
  __trap();  // (the split mode's replay without its instance: GT_INSTANCES bit 3)
}

// After the release: this block's record header and b / a; the state after the first j tokens of the block: a direct
// step's (j = T = 1) into the running state, a boundary candidate's (j = bnd, a block of vLLM's ends there) into vLLM's
// slot of that block with its window. bar: the 256 threads' named barrier.
__device__ __forceinline__ void gdn_t_finish(uint32_t item, const GdnTDesc& g, uint8_t* stg, int pt, int bar,
                                             const GtDep& dep, const GtCarry& cy) {
  const GtItem it = gt_item(item);
  if (!it.has_work) return;  // (CTA-uniform)
  GtMisc* ms = reinterpret_cast<GtMisc*>(stg + GT_MISC_OFF);
  uint8_t* rout = reinterpret_cast<uint8_t*>(g.rec) + (size_t)cy.parity * GTR_BYTES;
  const bool is_direct = cy.direct;
  if (pt == 0 && item == 0) {  // (direct: the next block starts after this one, nothing to replay)
    int32_t* hdr = reinterpret_cast<int32_t*>(rout + GTR_HDR);
    hdr[GTR_P] = is_direct ? cy.P + cy.T : cy.P;
    hdr[GTR_T] = is_direct ? 0 : cy.T;
  }
  const int T = cy.T;
  if (!is_direct && it.rb == 0 && pt < T) {  // this block's b / a of the head
    __nv_bfloat16* ab = reinterpret_cast<__nv_bfloat16*>(rout + GTR_AB) + pt * 2 * GT_HV;
    ab[it.hv] = __float2bfloat16(ms->bb[pt]);
    ab[GT_HV + it.hv] = __float2bfloat16(ms->aa[pt]);
  }
  const int j = is_direct ? T : cy.bnd;
  if (j) {  // (CTA-uniform)
    if (cy.bnd) {
      if (pt == 0) {  // vLLM's slot of the block ending there (a dependent load: here, off the critical path)
        const GtStep sp = gt_step(g);
        ms->prm[GP_BSLOT] = __ldcg(sp.bt + (cy.P + cy.bnd - 1) / sp.B);
      }
      named_sync(bar, GT_NT);
    }
    const int64_t slot = ms->prm[GP_BSLOT];
    const bool hnat = ms->prm[GP_HNAT], hpost = ms->prm[GP_HPOST] && j == T;
    const int row = pt >> 2, qd = pt & 3;
    const float* H = reinterpret_cast<const float*>(stg);
    const uint16_t* V = reinterpret_cast<const uint16_t*>(stg + GT_V_OFF);
    const float* er = reinterpret_cast<const float*>(stg + GT_RS_OFF) + row * GT_RS;
    if (hpost) {  // H holds the state after the block (rows in the natural layout): bulk copies, thread 0
      if (is_direct && row < it.R) {  // (run's layout: swizzled, by the rows' threads)
        float4* run = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.run) + ((size_t)it.hv * GT_V + it.v0 + row) * GT_K);
#pragma unroll
        for (int m = 0; m < 8; m++) run[(qd + 4 * m) ^ (row & 7)] = reinterpret_cast<const float4*>(H + row * GT_K)[qd + 4 * m];
      }
      if (pt == 0) {
        const size_t off = ((size_t)it.hv * GT_V + it.v0) * GT_K;
        if (cy.bnd) bulk_s2g(reinterpret_cast<float*>(g.ssm_state) + slot * g.ssm_slot_stride + off, H, it.R * GT_K * 4);
        bulk_wait_read();  // (the staging buffer read: the SM's next instruction may overwrite it)
      }
    } else if (row < it.R) {  // h after the first j tokens: h <- exp(g_i) h + e_i (normed k_i), from the block's start
      float4 h[8];
#pragma unroll
      for (int m = 0; m < 8; m++)
        h[m] = reinterpret_cast<const float4*>(H + row * GT_K)[hnat ? qd + 4 * m : (qd + 4 * m) ^ (row & 7)];
#pragma unroll 1
      for (int i = 0; i < j; i++) {
        const float d = exp_tri(ms->g[i]), e = er[i], r = ms->rk[i];
#pragma unroll
        for (int m = 0; m < 8; m++) {
          const uint2 kb = *reinterpret_cast<const uint2*>(V + gt_v(i, 4 * (qd + 4 * m)));
          h[m].x = fmaf(e, bf16lo(kb.x) * r, h[m].x * d);
          h[m].y = fmaf(e, bf16hi(kb.x) * r, h[m].y * d);
          h[m].z = fmaf(e, bf16lo(kb.y) * r, h[m].z * d);
          h[m].w = fmaf(e, bf16hi(kb.y) * r, h[m].w * d);
        }
      }
      const size_t off = ((size_t)it.hv * GT_V + it.v0 + row) * GT_K;
      if (is_direct) {  // the running state: after the block (swizzled)
        float4* run = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.run) + off);
#pragma unroll
        for (int m = 0; m < 8; m++) run[(qd + 4 * m) ^ (row & 7)] = h[m];
      }
      if (cy.bnd) {  // (direct: bnd = j)
        float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.ssm_state) + slot * g.ssm_slot_stride + off);
#pragma unroll
        for (int m = 0; m < 8; m++) dst[qd + 4 * m] = h[m];
      }
    }
    if (cy.bnd) {
      // the window there, entries [0, 3) of the conv slot: inputs at P + j - 3 .. P + j - 1, entries j .. j + 2 of this
      // block's record (direct: its entries 0 .. 2), which this thread wrote for its channels
      if (cy.first && dep.done_ctr) {  // (the slot this step's windows are read from: every reader done)
        if (pt == 0) gt_wait_counter(dep.done_ctr, dep.done_target);
        named_sync(bar, GT_NT);
      }
      const int e0 = is_direct ? 0 : j;
      const bool is_q = pt < GT_Q_THREADS;
      const int ch = is_q ? GT_QOFF + it.ih * GT_K + pt : GT_KOFF + it.ih * GT_K + (pt - GT_Q_THREADS);
      const int chv = GT_VOFF + it.hv * GT_V + it.v0 + pt;
      const __nv_bfloat16* so = reinterpret_cast<const __nv_bfloat16*>(rout + GTR_SEQ);
      __nv_bfloat16* cs = reinterpret_cast<__nv_bfloat16*>(g.conv_state) + slot * g.conv_slot_stride;
#pragma unroll
      for (int e = 0; e < 3; e++) {
        if (it.hv % 3 == 0 && it.rb == 0) cs[(size_t)e * GT_CONV + ch] = so[(size_t)(e0 + e) * GT_CONV + ch];
        if (pt < it.R) cs[(size_t)e * GT_CONV + chv] = so[(size_t)(e0 + e) * GT_CONV + chv];
      }
    }
  }
  named_sync(bar, GT_NT);  // (every read of the staging buffer done: the SM's next instruction may overwrite it)
}

}  // namespace mk
