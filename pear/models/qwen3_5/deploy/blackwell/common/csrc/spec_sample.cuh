// The spec engine's sampling of one request's verify step as phases of its launch: vLLM's rejection sampler (the
// standard rule over probabilistic draft logits, temperature 1) over the top-p nucleus of the target rows, with vLLM's
// Philox noise for the acceptance and noise of its own for the resampled token.
#pragma once
#include "logprobs.cuh"

namespace mk {

// vLLM's sampling noise as its Triton kernels compute it: Triton's Philox4x32-10 (tl.randint / tl.rand), libdevice's
// logf and log1pf as Triton inlines them (ftz), Gumbel noise of vLLM's gumbel_noised_argmax (the draft extension's
// dk_select.cuh holds the same lowerings; this extension builds from its own sources)
constexpr float TL_RAND_SCALE = 4.6566127342e-10f;  // tl.rand: 2^-31 per step of the int32 draw
constexpr float TL_RAND_MIN = 4.6566127342e-10f;    // vLLM's _TL_RAND_MIN (includes_zero=False)

constexpr int PHILOX_ROUNDS = 10;            // Philox4x32-10
constexpr uint32_t PHILOX_M0 = 0xD2511F53u;  // the rounds' multipliers
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;  // the key's increments per round
constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;
// tl.randint(seed, offset): the first word of philox(seed, offset_lo, offset_hi, 0, 0)
__device__ __forceinline__ uint32_t philox_randint(uint64_t seed, uint64_t offset) {
  uint32_t c0 = (uint32_t)offset, c1 = (uint32_t)(offset >> 32), c2 = 0u, c3 = 0u;
  uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
  for (int r = 0; r < PHILOX_ROUNDS; r++) {
    const uint32_t hi_b = __umulhi(PHILOX_M1, c2), lo_b = PHILOX_M1 * c2;
    const uint32_t hi_a = __umulhi(PHILOX_M0, c0), lo_a = PHILOX_M0 * c0;
    c0 = hi_b ^ c1 ^ k0;
    c2 = hi_a ^ c3 ^ k1;
    c1 = lo_b;
    c3 = lo_a;
    k0 += PHILOX_W0;
    k1 += PHILOX_W1;
  }
  return c0;
}
// vLLM's tl_rand32(seed, offset, includes_zero=False): tl.rand's uniform from the int32 draw, at least TL_RAND_MIN
__device__ __forceinline__ float tl_rand32(uint64_t seed, uint64_t offset) {
  const int32_t x = (int32_t)philox_randint(seed, offset);
  const int32_t y = x ^ (x >> 31);  // (x < 0: -x - 1)
  return fmaxf(__fmul_rn((float)y, TL_RAND_SCALE), TL_RAND_MIN);
}
// libdevice's log1pf as Triton inlines it (ftz), on a in (-1, 0]
__device__ __forceinline__ float log1p_tri(float a) {
  const float t = __fadd_rz(a, 1.0f);
  const int e = (__float_as_int(t) - 0x3F400000) & 0xFF800000;  // add.s32 -1061158912; and -8388608
  const float m = __int_as_float(__float_as_int(a) - e);
  const float s4 = __int_as_float(0x40800000 - e);  // 1082130432 - e
  const float r = __fadd_rn(__fmaf_rn(0.25f, s4, -1.0f), m);
  const float ie = __fmul_rn((float)e, 1.1920928955078125e-07f);  // 0f34000000
  float p = __fmaf_rn(__int_as_float(0xBD39BF78), r, __int_as_float(0x3DD80012));
  p = __fmaf_rn(p, r, __int_as_float(0xBE0778E0));
  p = __fmaf_rn(p, r, __int_as_float(0x3E146475));
  p = __fmaf_rn(p, r, __int_as_float(0xBE2A68DD));
  p = __fmaf_rn(p, r, __int_as_float(0x3E4CAF9E));
  p = __fmaf_rn(p, r, __int_as_float(0xBE800042));
  p = __fmaf_rn(p, r, __int_as_float(0x3EAAAAE6));
  p = __fmaf_rn(p, r, -0.5f);
  p = __fmul_rn(r, p);
  p = __fmaf_rn(p, r, r);
  float y = __fmaf_rn(ie, __int_as_float(0x3F317218), p);
  if (!((uint32_t)__float_as_int(a) < 0x7F800000u)) {  // a < 0 or not finite (setp.lt.u32 a, 0x7f800000)
    if (!(__float_as_int(a) < (int)0xBF800001)) y = __fmaf_rn(a, INFINITY, INFINITY);
    if (a == 0.f) y = -0.0f;
  }
  return y;
}
// libdevice's logf as Triton inlines it (ftz), on a > 0
__device__ __forceinline__ float log_tri(float a) {
  const bool is_subnormal = a < 1.17549435e-38f;  // 0f00800000
  const float x = is_subnormal ? a * 8388608.0f : a;
  const float bias = is_subnormal ? -23.0f : 0.0f;
  const int e = (__float_as_int(x) - 0x3F2AAAAB) & 0xFF800000;  // add.s32 -1059760811
  const float m = __int_as_float(__float_as_int(x) - e);
  const float ie = __fmaf_rn((float)e, 1.1920928955078125e-07f, bias);
  const float r = __fadd_rn(m, -1.0f);
  float p = __fmaf_rn(__int_as_float(0xBE055027), r, __int_as_float(0x3E1039F6));
  p = __fmaf_rn(p, r, __int_as_float(0xBDF8CDCC));
  p = __fmaf_rn(p, r, __int_as_float(0x3E0F2955));
  p = __fmaf_rn(p, r, __int_as_float(0xBE2AD8B9));
  p = __fmaf_rn(p, r, __int_as_float(0x3E4CED0B));
  p = __fmaf_rn(p, r, __int_as_float(0xBE7FFF22));
  p = __fmaf_rn(p, r, __int_as_float(0x3EAAAA78));
  p = __fmaf_rn(p, r, -0.5f);
  p = __fmul_rn(r, p);
  p = __fmaf_rn(p, r, r);
  float y = __fmaf_rn(ie, __int_as_float(0x3F317218), p);
  if (!((uint32_t)__float_as_int(x) < 0x7F800000u)) y = __fmaf_rn(x, INFINITY, INFINITY);
  return y;
}
// the Gumbel noise of token t under a Philox key: -log(-log1p(-u)), u = tl_rand32 of the key at offset t (vLLM's
// gumbel_noised_argmax: the key tl.randint(seed, pos))
__device__ __forceinline__ float ss_gumbel(uint64_t key, int64_t t) {
  return -log_tri(-log1p_tri(-tl_rand32(key, (uint64_t)t)));
}


constexpr int SS_MAX_T = 16;  // a step's logits rows at most (1 + the drafts)
constexpr int SS_F4 = 4;      // floats per 16-byte vector of the draft logits
constexpr int SS_VEC = LP_VEC_BYTES / (int)sizeof(__nv_bfloat16);  // logits per 16-byte vector
constexpr int SS_SLICE_ELEMS = 2048;  // a slice's elements of the resampled row at most (slices >= V / 2048)
// the resampled token's Philox key: the draft's noise key (tl.randint(seed, pos)) with high word 1, so that its noise
// is apart from the draft's (the draft's noise is conditioned by the rejected draft's draw: reused, as vLLM's resample
// does, its sample of the residual leans away from the tokens the draft favored)
constexpr uint64_t SS_NOISE_KEY_HI = 1ull << 32;

// The nucleus of a target row (top_p < 1; 1: every token): its tokens by logit, of equal logits the higher index first,
// each kept while the mass ranked above it is below top_p of the row's sum of exp; masses exp(x - max) in integer units
// of 2^-40 of the max's, counted by bin (a logit each) over the window within log(V / (1 - top_p)) + 1 of the max.
constexpr float SS_MASS_ONE = 1099511627776.0f;  // 2^40
constexpr float SS_WINDOW_MARGIN = 1.0f;         // the window past log(V / (1 - top_p))
constexpr int SS_HIST_BINS = 4096;               // a chunk's bins below its max's (a window spans <= 3843: V < 2^18)
constexpr int SS_HIST_WORDS = SS_HIST_BINS / 2;  // (16-bit counts, two a word)
constexpr int SS_MICRO_EXP = 117;                // the bf16 exponent of 2^-10
constexpr int SS_MANT = 128;                     // bf16 mantissas per exponent
// the orders (ss_order) of zero, of the smallest positive logit with a bin of its own and of the largest negative one
constexpr int SS_ORDER_ZERO = 0x8000, SS_ORDER_POS = SS_ORDER_ZERO + SS_MICRO_EXP * SS_MANT,
              SS_ORDER_NEG = SS_ORDER_ZERO - 1 - SS_MICRO_EXP * SS_MANT;
// a row's histogram: every bin of a 16-bit order (place SS_BIN_TOP - bin), the window from the max's place on
constexpr int SS_BIN_TOP = 0xFFFF - SS_ORDER_POS + 2, SS_ROW_BINS = 2 * SS_BIN_TOP + 1;
// a chunk's list of its micro logits (logits of unit spread hold about V / 2^11 of them in a row); past the list the
// row's masses of them come from the row itself
constexpr int SS_MICRO_CAP = 32;
constexpr int SS_BATCH = 8;  // 16-byte loads a thread has in flight in a sweep over a row
constexpr int SS_RUN = 4;    // a thread's bins per round of the crossing's scan (their exps at once)
static_assert(LP_CHUNK_VECS * SS_VEC < 1 << 16, "a chunk's counts in 16 bits");

// a == b, a >= b (64-bit) as all-ones / zero masks (set.*)
__device__ __forceinline__ uint32_t ss_eq(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("set.eq.u32.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t ss_ge64(uint64_t a, uint64_t b) {
  uint32_t r;
  asm("set.ge.u32.u64 %0, %1, %2;" : "=r"(r) : "l"(a), "l"(b));
  return r;
}
constexpr uint32_t SS_SIGN16 = 0x8000u;  // a bf16's sign bit
constexpr uint32_t SS_MAG16 = 0x7FFFu;   // its magnitude bits
// a bf16 logit's order: its 16 bits b as an increasing uint16 (negative: inverted, else the sign set), -0 as 0
__device__ __forceinline__ uint32_t ss_order(uint32_t b) {
  b &= ~(ss_eq(b, SS_SIGN16) & SS_SIGN16);
  return (b ^ ((uint32_t)((int32_t)(b << 16) >> 31) & SS_MAG16)) ^ SS_SIGN16;
}
// the logit of an order
__device__ __forceinline__ float ss_logit(uint32_t k) {
  return __uint_as_float((k ^ SS_SIGN16 ^ (((k >> 15) - 1u) & SS_MAG16)) << 16);
}
// an order's bin: one per logit of magnitude 2^-10 or more (increasing with it), the smaller ones 1 (positive) and -1
// (negative), zero 0
__device__ __forceinline__ int ss_bin(uint32_t k) {
  const int o = (int)k;
  return (max(o - SS_ORDER_POS + 2, 1) & (int)lp_lt(SS_ORDER_ZERO, o)) |
         (min(o - SS_ORDER_NEG - 2, -1) & (int)lp_lt(o, SS_ORDER_ZERO));
}
// a bin's order (a bin of one logit: not 1 or -1)
__device__ __forceinline__ uint32_t ss_bin_order(int j) {
  return ((uint32_t)(j + SS_ORDER_POS - 2) & lp_lt(1, j)) | ((uint32_t)(j + SS_ORDER_NEG + 2) & lp_lt(j, -1)) |
         ((uint32_t)SS_ORDER_ZERO & lp_lt(-1, j) & lp_lt(j, 1));
}
// a logit's mass in its row of max m
__device__ __forceinline__ unsigned long long ss_mass(float x, float m) {
  return __float2ull_rn(exp_tri(x - m) * SS_MASS_ONE);
}

// what a row's phases need of its tokens (CTA (row, 0) in A): its draft token (row T - 1: 0), the position, the
// target's and the draft's logits of the draft token
struct SsFact {
  int64_t tok, pos;
  float xt, xd, pad[2];
};
// a target chunk's max and sum of exp, its window's bins and its micro logits in the window (count, bf16 bits) (A)
struct alignas(16) SsPart {
  float m, s;
  uint32_t bins, micro;
  uint16_t mlist[SS_MICRO_CAP];
};
// a row's nucleus and test (A's last chunk of the row): whether its draft token is accepted, the nucleus' threshold
// (the tokens at or above (order, index) kept), the log of the target's nucleus mass and of the draft's sum of exp
struct SsRow {
  uint32_t accept, key, idx;
  float tlse, dlse;
  uint32_t pad[3];
};

// the spec sampler's step words at SR_SS (step.py SS_*): the request's draft logits [T - 1, V], V2's logits indices,
// its seed, the step's input ids and positions (uint64 addresses), the output ring slot, the request's top_p (fp32
// bits; 0: the launch leaves the sampling to vLLM)
enum SsStepWord : int { SS_DRAFT = 0, SS_LIDX = 2, SS_SEED = 4, SS_IDS = 6, SS_POSV = 8, SS_SLOT = 10, SS_TOP_P = 11 };

struct SsArgs {
  const __nv_bfloat16* logits;     // [T, V] the target's
  int64_t stride;
  const float* draft;              // [T - 1, V] the request's draft logits (pre-temperature: temperature 1)
  int64_t draft_stride;
  const int32_t* input_ids;        // V2's input batch: the step's tokens (logit i + 1's: the draft verified by row i)
  const int64_t* positions;
  const int64_t* logits_indices;
  const int64_t* seed;             // the request's sampling seed
  float top_p;                     // the request's
  int V, T, chunks;                // chunks per row
  SsPart* part;                    // [T * chunks] the target chunks' (scratch, A)
  uint32_t* hist;                  // [T * chunks][SS_HIST_WORDS] their histograms (scratch, A)
  uint32_t* rhist;                 // [T][SS_ROW_BINS] the rows' (scratch: A's chunks add, the row's last reads, zeroes)
  float2* dpart;                   // [(T - 1) * chunks] the draft rows' chunk (max, sum of exp) (scratch, A0)
  unsigned long long* amax;        // [slices] the resample's per-slice argmax keys (scratch, B)
  SsFact* facts;                   // [T] per row its draft token's facts (scratch, A)
  SsRow* rows;                     // [T] per row its nucleus and test (scratch, A's last chunk of the row)
  uint32_t* chunks_done;           // [T] per row its chunks done (scratch, A; the last one back to 0)
  int64_t* sampled;                // [T] out: the request's row of sampled_token_ids
  int32_t* num_sampled;            // [1] out
  int32_t* num_rejected;           // [1] out (the step's logits less num_sampled: a decode step, no chunked prefill)
};

// A group's shared memory for the phases; the small fields first: a dry run touches the first few KB only
template <int NTH>
struct SsSmem {
  static constexpr int W = LpGrp<NTH>::WARPS;
  uint64_t bar;                          // A: the chunk's bulk copy
  uint32_t mn;                           // A: the chunk's micro logits in the window
  uint32_t is_last_chunk;                // A: whether the chunk is its row's last done
  uint16_t ml[SS_MICRO_CAP];             // A: their bf16 bits (the first SS_MICRO_CAP)
  int list_overflowed;                   // row: whether a chunk's list overflowed
  float f[W];                            // per warp: a max or a sum
  unsigned long long u[W];               // per warp: a key or an integer sum
  float m, s, dlse;                      // row: its max and sum of exp, the draft row's log of its sum of exp
  unsigned long long micro[2];           // row: the masses of its logits below 2^-10 in magnitude (positive, negative)
  unsigned long long goal, above;        // row: the nucleus' mass to reach, the mass above the crossing
  int cross;                             // row: the crossing's bin below the row max's
  uint32_t key, n, tie, rank;            // row: crossing logit's order and count, a walk's index or exponent, a rank
  uint32_t holder, rest;                 // row: the thread whose share holds a walk's logit, the logits left to pass
  int delta[LP_MAX_CHUNKS];              // row: each chunk's bins below the row max's
  int rend;                              // row: where the chunks' bins in its histogram end (below the max's place)
  unsigned long long em[SS_MICRO_EXP];   // row: the masses of its micro logits of a sign by exponent
  uint32_t mc[SS_MANT];                  // row: the counts of an exponent's micro logits by mantissa
  alignas(16) uint32_t hist[SS_HIST_WORDS];  // A: the chunk's counts by bin below its max's (16-byte copies)
  uint4 x[LP_CHUNK_VECS];                // A: the chunk; row: its counts by bin (uint32)
};

// a row's (max, sum of exp) over its chunks' (m, s) (lane c: chunk c's; dead lanes (-inf, 0)) by an xor butterfly:
// partners combine the same two values (lp_combine commutes), so every lane, of any CTA, ends with the same result
__device__ __forceinline__ void ss_row_reduce(float& m, float& s) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lp_combine(m, s, __shfl_xor_sync(FULL_MASK, m, o), __shfl_xor_sync(FULL_MASK, s, o));
}
// logit `tok` of target row r / draft row r
__device__ __forceinline__ float ss_target(const SsArgs& a, int r, int64_t tok) {
  return __bfloat162float(a.logits[(int64_t)r * a.stride + tok]);
}
__device__ __forceinline__ float ss_draft(const SsArgs& a, int r, int64_t tok) { return a.draft[(int64_t)r * a.draft_stride + tok]; }
// the draft token logit row r verifies (row T - 1: none, 0)
__device__ __forceinline__ int64_t ss_draft_token(const SsArgs& a, int r) {
  return (int64_t)a.input_ids[a.logits_indices[min(r + 1, a.T - 1)]] & (int64_t)(int32_t)lp_lt(r, a.T - 1);
}
// the depth below a row's max past which its tokens are never kept (top_p < 1)
__device__ __forceinline__ float ss_window(const SsArgs& a) {
  return logf((float)a.V) - logf(1.0f - a.top_p) + SS_WINDOW_MARGIN;
}
// the lowest bin of a logit above lo (bins 1, 0, -1 whole when lo falls in them)
__device__ __forceinline__ int ss_bottom(float lo) {
  const int low_bin = ss_bin(ss_order(__bfloat16_as_ushort(__float2bfloat16_rd(lo))));
  return low_bin >= -1 && low_bin <= 1 ? -1 : low_bin + 1;
}
// a row's fact / outcome as another SM published it (two 16-byte loads past L1)
__device__ __forceinline__ SsFact ss_load_fact(const SsFact* p) {
  static_assert(sizeof(SsFact) == 2 * sizeof(uint4), "SsFact: two 16-byte words");
  const uint4 w[2] = {__ldcg(reinterpret_cast<const uint4*>(p)), __ldcg(reinterpret_cast<const uint4*>(p) + 1)};
  SsFact f;
  memcpy(&f, w, sizeof(f));
  return f;
}
__device__ __forceinline__ SsRow ss_load_row(const SsRow* p) {
  static_assert(sizeof(SsRow) == 2 * sizeof(uint4), "SsRow: two 16-byte words");
  const uint4 w[2] = {__ldcg(reinterpret_cast<const uint4*>(p)), __ldcg(reinterpret_cast<const uint4*>(p) + 1)};
  SsRow o;
  memcpy(&o, w, sizeof(o));
  return o;
}
// the step's accepted prefix (a warp, every lane): the first row whose test failed, K = T - 1 when none did (the bonus
// row is sampled)
__device__ __forceinline__ int ss_accepted(const SsArgs& a, int lane) {
  const int K = a.T - 1;
  const bool is_accepted = lane < K && __ldcg(&a.rows[min(lane, K - 1)].accept) != 0u;
  return __ffs(~__ballot_sync(FULL_MASK, is_accepted)) - 1;
}

// The group's reductions of every thread's v (all threads at the barriers, every thread gets the result; deterministic
// trees): the max of floats, the sum of floats, the max of keys, the exclusive prefix sum of integers in thread order
// and their total
template <int NTH>
__device__ __forceinline__ float ss_cta_max_f(const LpGrp<NTH>& g, SsSmem<NTH>& sh, float v) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const int warp = g.tid >> 5, lane = g.tid & 31;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
  g.sync();  // (sh.f's earlier readers done)
  if (lane == 0) sh.f[warp] = v;
  g.sync();
  v = sh.f[0];
#pragma unroll
  for (int w = 1; w < W; w++) v = fmaxf(v, sh.f[w]);
  return v;
}
template <int NTH>
__device__ __forceinline__ float ss_cta_sum_f(const LpGrp<NTH>& g, SsSmem<NTH>& sh, float v) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const int warp = g.tid >> 5, lane = g.tid & 31;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
  g.sync();
  if (lane == 0) sh.f[warp] = v;
  g.sync();
  v = sh.f[0];
#pragma unroll
  for (int w = 1; w < W; w++) v += sh.f[w];
  return v;
}
template <int NTH>
__device__ __forceinline__ unsigned long long ss_cta_max(const LpGrp<NTH>& g, SsSmem<NTH>& sh, unsigned long long v) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const int warp = g.tid >> 5, lane = g.tid & 31;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long ov = __shfl_xor_sync(FULL_MASK, v, o);
    v = ov > v ? ov : v;
  }
  g.sync();
  if (lane == 0) sh.u[warp] = v;
  g.sync();
  v = sh.u[0];
#pragma unroll
  for (int w = 1; w < W; w++) v = sh.u[w] > v ? sh.u[w] : v;
  return v;
}
template <int NTH>
__device__ __forceinline__ unsigned long long ss_cta_exscan(const LpGrp<NTH>& g, SsSmem<NTH>& sh, unsigned long long v,
                                                            unsigned long long& total) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const int warp = g.tid >> 5, lane = g.tid & 31;
  unsigned long long inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) inc += __shfl_up_sync(FULL_MASK, inc, o) & lp_wide(~lp_lt(lane, o));
  g.sync();
  if (lane == 31) sh.u[warp] = inc;
  g.sync();
  unsigned long long base = 0ull, all_sum = 0ull;
#pragma unroll
  for (int w = 0; w < W; w++) base += sh.u[w] & lp_wide(lp_lt(w, warp)), all_sum += sh.u[w];
  total = all_sum;
  return base + inc - v;
}

// The phases, each run by a group of NTH threads on one item once the phase before is complete. is_dry_run: the phase's
// code without its global stores, on one vector of its chunk (a warm-up while the LM head streams). A phase's registers
// are the prologue loop's (ss_phase: an internal call): holding more spills the loop's state

// a's item i: its (row, chunk) and the chunk's vectors [v0, v1) of the row's bf16 logits
__device__ __forceinline__ void ss_item(const SsArgs& a, int i, int& row, int& chunk, int& v0, int& v1) {
  const int nvec = (a.V + SS_VEC - 1) / SS_VEC;
  row = i / a.chunks, chunk = i % a.chunks;
  v0 = (int)((int64_t)nvec * chunk / a.chunks), v1 = (int)((int64_t)nvec * (chunk + 1) / a.chunks);
}
// The group's (max, sum of exp) of draft row r's elements [e0, e1) into *dst: rolled sweeps over the row's 16-byte
// vectors (the max, then the sum of exp; few registers, one copy of the code)
template <int NTH>
__device__ __forceinline__ void ss_draft_chunk(const LpGrp<NTH>& g, const SsArgs& a, int r, int e0, int e1, SsSmem<NTH>& sh,
                                               float2* dst) {
  const int tid = g.tid;
  const float4* d = reinterpret_cast<const float4*>(a.draft + (int64_t)r * a.draft_stride);
  const int f0 = e0 / SS_F4, f1 = (e1 + SS_F4 - 1) / SS_F4;
  float m = -INFINITY;
#pragma unroll 1
  for (int f = f0 + tid; f < f1; f += NTH) {
    const float4 q = __ldg(d + f);
    const float v[SS_F4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int e = 0; e < SS_F4; e++) m = fmaxf(m, lp_pick(lp_lt(f * SS_F4 + e, e1), v[e], -INFINITY));
  }
  m = ss_cta_max_f(g, sh, m);
  float s = 0.f;
#pragma unroll 1
  for (int f = f0 + tid; f < f1; f += NTH) {
    const float4 q = __ldg(d + f);
    const float v[SS_F4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int e = 0; e < SS_F4; e++) s += __uint_as_float(__float_as_uint(lp_exp_diff(v[e], m)) & lp_lt(f * SS_F4 + e, e1));
  }
  s = ss_cta_sum_f(g, sh, s);
  if (tid == 0) *dst = make_float2(m, s);
}
// A0 (the megakernel: while the LM head streams, its inputs are the launch's): item i = (row, chunk) of the draft rows,
// the chunk's (max, sum of exp) into dpart
template <int NTH>
__device__ void ss_phase_a0(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int item) {
  int row, chunk, v0, v1;
  ss_item(a, item, row, chunk, v0, v1);
  if (row < a.T - 1) ss_draft_chunk(g, a, row, v0 * SS_VEC, min(v1 * SS_VEC, a.V), sh, &a.dpart[row * a.chunks + chunk]);
}

// thread 0: the chunk's bulk copy (row x's vectors [v0, v1)) into sh.x on sh.bar (initialized here: one use per launch)
template <int NTH>
__device__ __forceinline__ void ss_chunk_issue(const __nv_bfloat16* x, int v0, int v1, SsSmem<NTH>& sh) {
  mbar_init(&sh.bar, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const uint32_t bytes = (uint32_t)(v1 - v0) * LP_VEC_BYTES;
  mbar_expect_tx(&sh.bar, bytes);
  bulk_g2s(sh.x, reinterpret_cast<const uint4*>(x) + v0, bytes, &sh.bar);
}

// The window's crossing (cnt [span]: its counts by bin below the max's bin top): the first bin whose mass, with the
// masses above it, reaches the goal P, into sh.cross and sh.above, the goal into sh.goal (every thread); rounds of
// SS_RUN bins a thread from the top, and a second pass to the window's mass when P lies beyond it
template <int NTH>
__device__ void ss_crossing(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const uint32_t* cnt, int span, int top, float M,
                            unsigned long long P) {
  const int tid = g.tid;
  if (tid == 0) sh.cross = 0, sh.above = 0ull;
  unsigned long long goal = P, carry = 0ull;
#pragma unroll 1
  for (int pass = 0; pass < 2 && carry < goal; pass++) {
    goal = pass == 0 ? P : carry;
    carry = 0ull;
#pragma unroll 1
    for (int base = 0; base < span && carry < goal; base += SS_RUN * NTH) {
      const int j0 = base + tid * SS_RUN;
      unsigned long long w[SS_RUN], own_mass = 0ull, total;
#pragma unroll
      for (int q = 0; q < SS_RUN; q++) {
        const int j = min(j0 + q, span - 1), bin = top - j;
        const unsigned long long mass =
            (bin == 1 || bin == -1) ? sh.micro[bin < 0] : cnt[j] * ss_mass(ss_logit(ss_bin_order(bin)), M);
        w[q] = mass & lp_wide(lp_lt(j0 + q, span));
        own_mass += w[q];
      }
      unsigned long long above = carry + ss_cta_exscan(g, sh, own_mass, total);
      if (above < goal && goal <= above + own_mass) {
#pragma unroll
        for (int q = 0; q < SS_RUN; q++) {
          if (above < goal && goal <= above + w[q]) sh.cross = j0 + q, sh.above = above;
          above += w[q];
        }
      }
      carry += total;
    }
  }
  if (tid == 0) sh.goal = goal;
  g.sync();
}
// every thread's vectors f = tid, tid + NTH, .. of a row's V logits, SS_BATCH loads at once, each passed to fn(q, f)
template <int NTH, typename Fn>
__device__ __forceinline__ void ss_sweep(const uint4* x, int V, int tid, Fn&& fn) {
  const int nvec = (V + SS_VEC - 1) / SS_VEC;
#pragma unroll 1
  for (int f = tid; f < nvec; f += SS_BATCH * NTH) {
    uint4 q[SS_BATCH];
#pragma unroll
    for (int k = 0; k < SS_BATCH; k++) q[k] = __ldcg(x + min(f + k * NTH, nvec - 1));
#pragma unroll
    for (int k = 0; k < SS_BATCH; k++)
      if (f + k * NTH < nvec) fn(q[k], f + k * NTH);
  }
}
// The masses of row r's logits below 2^-10 in magnitude (not zero) above the window's bottom lo, by sign, into
// sh.micro (every thread): from the chunks' lists (lane c: chunk c's), or with a list short of its chunk's from the row
template <int NTH>
__device__ void ss_micro_mass(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int r, float M, float lo) {
  const int lane = g.tid & 31;
  if (g.tid < 32) {
    const SsPart& p = a.part[r * a.chunks + min(lane, a.chunks - 1)];
    const uint32_t n = __ldcg(&p.micro) & lp_lt(lane, a.chunks);
    uint4 w[SS_MICRO_CAP / SS_VEC];
#pragma unroll
    for (int k = 0; k < SS_MICRO_CAP / SS_VEC; k++) w[k] = __ldcg(reinterpret_cast<const uint4*>(p.mlist) + k);
    unsigned long long pos_mass = 0ull, neg_mass = 0ull;
#pragma unroll
    for (int k = 0; k < SS_MICRO_CAP / SS_VEC; k++)
#pragma unroll 4
      for (int e = 0; e < SS_VEC; e++) {
        const float f = lp_elem<__nv_bfloat16>(w[k], e);
        if ((uint32_t)(k * SS_VEC + e) < n && f > lo) {
          if (__float_as_uint(f) >> 31) neg_mass += ss_mass(f, M);
          else pos_mass += ss_mass(f, M);
        }
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      pos_mass += __shfl_xor_sync(FULL_MASK, pos_mass, o), neg_mass += __shfl_xor_sync(FULL_MASK, neg_mass, o);
    const bool list_overflowed = __any_sync(FULL_MASK, n > (uint32_t)SS_MICRO_CAP);
    if (lane == 0) sh.micro[0] = pos_mass, sh.micro[1] = neg_mass, sh.list_overflowed = list_overflowed;
  }
  g.sync();
  if (!sh.list_overflowed) return;
  unsigned long long pos_mass = 0ull, neg_mass = 0ull, t;
  ss_sweep<NTH>(reinterpret_cast<const uint4*>(a.logits + (int64_t)r * a.stride), a.V, g.tid, [&](const uint4& q, int f) {
#pragma unroll 1
    for (int e = 0; e < SS_VEC; e++) {
      const float x = lp_elem<__nv_bfloat16>(q, e);
      const uint32_t b = __float_as_uint(x) >> 16, mag = b & SS_MAG16;
      if (f * SS_VEC + e < a.V && mag != 0u && mag < SS_MICRO_EXP * SS_MANT && x > lo) {
        if (b >> 15) neg_mass += ss_mass(x, M);
        else pos_mass += ss_mass(x, M);
      }
    }
  });
  ss_cta_exscan(g, sh, pos_mass, t);
  if (g.tid == 0) sh.micro[0] = t;
  ss_cta_exscan(g, sh, neg_mass, t);
  if (g.tid == 0) sh.micro[1] = t;
  g.sync();
}
// The crossing among row r's micro logits of one sign (is_neg) above lo, sh.above the mass above them: their masses by
// exponent, then the counts of the crossing exponent's by mantissa (a logit each); the crossing logit's order and count
// into sh.key and sh.n, the mass above it into sh.above (every thread)
template <int NTH>
__device__ void ss_micro_crossing(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int r, float M, float lo,
                                  bool is_neg) {
  const uint4* x = reinterpret_cast<const uint4*>(a.logits + (int64_t)r * a.stride);
  const uint32_t sign = is_neg ? SS_SIGN16 : 0u;
  for (int i = g.tid; i < SS_MICRO_EXP; i += NTH) sh.em[i] = 0ull;
  for (int i = g.tid; i < SS_MANT; i += NTH) sh.mc[i] = 0u;
  g.sync();
  ss_sweep<NTH>(x, a.V, g.tid, [&](const uint4& q, int f) {
#pragma unroll 1
    for (int e = 0; e < SS_VEC; e++) {
      const float v = lp_elem<__nv_bfloat16>(q, e);
      const uint32_t b = __float_as_uint(v) >> 16, mag = b & SS_MAG16;
      if (f * SS_VEC + e < a.V && (b & SS_SIGN16) == sign && mag != 0u && mag < SS_MICRO_EXP * SS_MANT && v > lo)
        atomicAdd(&sh.em[mag / SS_MANT], ss_mass(v, M));
    }
  });
  g.sync();
  if (g.tid == 0) {  // the exponents by value (negative logits: the smaller magnitudes first)
    unsigned long long above = sh.above;
    int ex = 0;
    for (int i = 0; i < SS_MICRO_EXP; i++) {
      ex = is_neg ? i : SS_MICRO_EXP - 1 - i;
      if (above + sh.em[ex] >= sh.goal) break;
      above += sh.em[ex];
    }
    sh.above = above, sh.tie = (uint32_t)ex;
  }
  g.sync();
  const uint32_t ex = sh.tie;
  ss_sweep<NTH>(x, a.V, g.tid, [&](const uint4& q, int f) {
#pragma unroll 1
    for (int e = 0; e < SS_VEC; e++) {
      const float v = lp_elem<__nv_bfloat16>(q, e);
      const uint32_t b = __float_as_uint(v) >> 16, mag = b & SS_MAG16;
      if (f * SS_VEC + e < a.V && (b & SS_SIGN16) == sign && mag != 0u && mag / SS_MANT == ex && v > lo)
        atomicAdd(&sh.mc[mag % SS_MANT], 1u);
    }
  });
  g.sync();
  if (g.tid == 0) {  // the exponent's logits by value
    unsigned long long above = sh.above;
    uint32_t b = 0u;
    for (int i = 0; i < SS_MANT; i++) {
      b = sign | ex * SS_MANT | (uint32_t)(is_neg ? i : SS_MANT - 1 - i);
      const unsigned long long w = (unsigned long long)sh.mc[b % SS_MANT] * ss_mass(__uint_as_float(b << 16), M);
      if (above + w >= sh.goal) break;
      above += w;
    }
    sh.key = ss_order(b), sh.n = sh.mc[b % SS_MANT], sh.above = above;
  }
  g.sync();
}
// the count of the logits of bits b1 or b2 (a logit's: zero has two) in vector f, of its elements below e1 (half-word
// compares of its words, a match 16 bits)
__device__ __forceinline__ uint32_t ss_ties(const uint4& q, int f, uint32_t b1, uint32_t b2, int e1) {
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
  const int n = e1 - f * SS_VEC;  // (the vector's elements below e1: all of them but in the row's last vector)
  uint32_t c = 0u;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const uint32_t in = (0xFFFFu & lp_lt(2 * k, n)) | (0xFFFF0000u & lp_lt(2 * k + 1, n));
    c += __popc((__vcmpeq2(w[k], b1 * 0x10001u) | __vcmpeq2(w[k], b2 * 0x10001u)) & in);
  }
  return c / 16;
}
// The index of the rho-th highest index among the logits of order k in elements [e0, e1) of row r (e0: a vector's
// first; every thread): each thread's count over its share of the vectors, the counts' suffix over the threads, then
// warp 0 walks the share holding it down, a vector a lane (nothing held across the barriers)
template <int NTH>
__device__ uint32_t ss_tie_index(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int r, int e0, int e1, uint32_t k,
                                 uint32_t rho) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const int tid = g.tid, warp = tid >> 5, lane = tid & 31;
  const uint4* x = reinterpret_cast<const uint4*>(a.logits + (int64_t)r * a.stride);
  const int f0 = e0 / SS_VEC, f1 = (e1 + SS_VEC - 1) / SS_VEC, per = (f1 - f0 + NTH - 1) / NTH;
  const int t0 = min(f0 + tid * per, f1), t1 = min(t0 + per, f1);
  const uint32_t b1 = __float_as_uint(ss_logit(k)) >> 16, b2 = b1 | (ss_eq(k, SS_ORDER_ZERO) & SS_SIGN16);  // (-0 too)
  uint32_t c = 0u;
#pragma unroll 1
  for (int f = t0; f < t1; f += SS_BATCH) {
    uint4 q[SS_BATCH];
#pragma unroll
    for (int i = 0; i < SS_BATCH; i++) q[i] = __ldcg(x + min(f + i, t1 - 1));
#pragma unroll
    for (int i = 0; i < SS_BATCH; i++) c += ss_ties(q[i], f + i, b1, b2, e1) & lp_lt(f + i, t1);
  }
  uint32_t suf = c;  // (the warp's from this lane up, then the later warps')
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) suf += __shfl_down_sync(FULL_MASK, suf, o) & lp_lt(lane, 32 - o);
  g.sync();
  if (lane == 0) sh.u[warp] = suf;
  g.sync();
#pragma unroll
  for (int w = 0; w < W; w++) suf += (uint32_t)sh.u[w] & lp_lt(warp, w);
  if (suf - c < rho && rho <= suf) sh.holder = (uint32_t)tid, sh.rest = rho - (suf - c);
  g.sync();
  if (warp == 0) {  // the holder's share from its top: lane i its vector i below the top, 32 at once
    const int h0 = min(f0 + (int)sh.holder * per, f1), h1 = min(h0 + per, f1);
    uint32_t rest = sh.rest;
#pragma unroll 1
    for (int top = h1; top > h0 && rest != 0u; top -= 32) {
      const int f = top - 1 - lane;
      const uint4 q = __ldcg(x + max(f, h0));
      const uint32_t vc = ss_ties(q, f, b1, b2, e1) & ~lp_lt(f, h0);
      uint32_t inc = vc;  // (the counts of the lanes from the top through this one)
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) inc += __shfl_up_sync(FULL_MASK, inc, o) & ~lp_lt(lane, o);
      if (inc - vc < rest && rest <= inc) {  // (this lane's vector holds it: its elements from the top)
        uint32_t ties_left = rest - (inc - vc);
#pragma unroll 1
        for (int e = SS_VEC - 1; e >= 0; e--) {
          const int t = f * SS_VEC + e;
          const uint32_t b = __float_as_uint(lp_elem<__nv_bfloat16>(q, e)) >> 16;
          if (ties_left != 0u && t < e1 && (b == b1 || b == b2) && --ties_left == 0u) sh.tie = (uint32_t)t;
        }
      }
      rest -= min(rest, __shfl_sync(FULL_MASK, inc, 31));
    }
  }
  g.sync();
  return sh.tie;
}
// Row r's nucleus (top_p < 1; M, S: its max and sum of exp): its threshold (order, index) and its mass (every thread):
// the window's counts by bin from the chunks' histograms, the crossing over their masses, the kept count of the
// crossing's logits and, short of all of them, the index of the last one kept (ss_tie_index)
template <int NTH>
__device__ unsigned long long ss_nucleus(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int r, float M, float S,
                                         uint32_t& key, uint32_t& idx, bool is_dry_run) {
  const int tid = g.tid, warp = tid >> 5, lane = tid & 31;
  const float lo = M - ss_window(a);
  const unsigned long long P = max(__float2ull_rn(__fmul_rn(a.top_p, S) * SS_MASS_ONE), 1ull);
  const int top = ss_bin(ss_order(__float_as_uint(M) >> 16));
  const int span = is_dry_run ? 1 : top - ss_bottom(lo) + 1;
  uint32_t* cnt = reinterpret_cast<uint32_t*>(sh.x);  // (A's chunk is done with)
  const uint32_t* hist = a.hist + (int64_t)r * a.chunks * SS_HIST_WORDS;
  // the row's counts in the window by bin (every load at once), then the row's histogram back to zero past the chunks'
  constexpr int PER = (SS_HIST_BINS + NTH - 1) / NTH;
  uint32_t* rh = a.rhist + (int64_t)r * SS_ROW_BINS + (SS_BIN_TOP - top);
  uint32_t n[PER];
#pragma unroll
  for (int q = 0; q < PER; q++) n[q] = __ldcg(rh + min(tid + q * NTH, span - 1)) & lp_lt(tid + q * NTH, span);
#pragma unroll
  for (int q = 0; q < PER; q++)
    if (tid + q * NTH < span) cnt[tid + q * NTH] = n[q];
  if (tid == 0) sh.micro[0] = sh.micro[1] = 0ull;
  g.sync();
  if (!is_dry_run)
    for (int j = tid; j < sh.rend; j += NTH) rh[j] = 0u;
  ss_crossing(g, sh, cnt, span, top, M, P);
  if (is_dry_run) return 0ull;
  const int jp = top - 1, jn = top + 1;  // (the micro bins': 1 and -1)
  const bool has_micro_bins = (jp >= 0 && jp < span && cnt[jp] != 0u) || (jn >= 0 && jn < span && cnt[jn] != 0u);
  if (top - sh.cross <= 1 && has_micro_bins) {  // (the crossing not above them, or none found: their masses)
    ss_micro_mass(g, sh, a, r, M, lo);
    ss_crossing(g, sh, cnt, span, top, M, P);
  }
  const int j = sh.cross, bin = top - j;
  if (bin == 1 || bin == -1) {
    ss_micro_crossing(g, sh, a, r, M, lo, bin < 0);
    key = sh.key;
  } else {
    key = ss_bin_order(bin);
    if (tid == 0) sh.n = cnt[j];
    g.sync();
  }
  const unsigned long long one = ss_mass(ss_logit(key), M), above = sh.above;
  const unsigned long long kept_count = min((unsigned long long)sh.n, (sh.goal - above + one - 1) / one);
  idx = 0u;
  if (kept_count < sh.n) {
    if (bin == 1 || bin == -1) {
      idx = ss_tie_index(g, sh, a, r, 0, a.V, key, (uint32_t)kept_count);
    } else {
      if (warp == 0) {  // the chunk of the kept-th highest index (lane c: chunk c's count of the bin; a suffix)
        const int c = min(lane, a.chunks - 1), i = j - sh.delta[c], ic = max(i, 0);
        const uint32_t nc =
            (__ldcg(hist + c * SS_HIST_WORDS + ic / 2) >> (ic % 2 * 16)) & 0xFFFFu & ~lp_lt(i, 0) & lp_lt(lane, a.chunks);
        uint32_t suf = nc;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) suf += __shfl_down_sync(FULL_MASK, suf, o) & lp_lt(lane, 32 - o);
        const uint32_t hit = __ballot_sync(FULL_MASK, suf - nc < (uint32_t)kept_count && (uint32_t)kept_count <= suf);
        if (lane == __ffs(hit) - 1) sh.tie = (uint32_t)lane, sh.rank = (uint32_t)kept_count - (suf - nc);
      }
      g.sync();
      int row, chunk, v0, v1;
      ss_item(a, r * a.chunks + (int)sh.tie, row, chunk, v0, v1);
      idx = ss_tie_index(g, sh, a, r, v0 * SS_VEC, min(v1 * SS_VEC, a.V), key, sh.rank);
    }
  }
  return above + kept_count * one;
}
// Row r (its chunks' outputs complete): its max and sum of exp from its chunks, its nucleus (top_p < 1) and the test
// of the draft token it verifies (rejection_sample's _rejection_kernel: log p > log u + log q, u at the row's
// position), into rows[r]
template <int NTH>
__device__ void ss_row(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int r, bool is_dry_run) {
  const int tid = g.tid, warp = tid >> 5, lane = tid & 31, K = a.T - 1;
  const SsFact f = ss_load_fact(&a.facts[r]);  // (with the chunks' partials: the test at the end)
  if (warp == 0) {  // (lane c: chunk c's, the target's and the draft row's; every load first)
    const uint32_t has_chunk_mask = lp_lt(lane, a.chunks);
    const int c = min(lane, a.chunks - 1);
    const uint4 p = __ldcg(reinterpret_cast<const uint4*>(&a.part[r * a.chunks + c]));
    const float2 d = __ldcg(&a.dpart[min(r, K - 1) * a.chunks + c]);
    float m = lp_pick(has_chunk_mask, __uint_as_float(p.x), -INFINITY), s = __uint_as_float(p.y & has_chunk_mask);
    float dm = lp_pick(has_chunk_mask, d.x, -INFINITY), ds = __uint_as_float(__float_as_uint(d.y) & has_chunk_mask);
    const float mc = m;
    ss_row_reduce(m, s);
    ss_row_reduce(dm, ds);
    const int delta = ss_bin(ss_order(__float_as_uint(m) >> 16)) - ss_bin(ss_order(__float_as_uint(mc) >> 16));
    int end = (delta + (int)p.z) & (int)has_chunk_mask;  // (the chunk's bins in the row's histogram end there)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) end = max(end, __shfl_xor_sync(FULL_MASK, end, o));
    sh.delta[lane] = delta;
    if (lane == 0) sh.m = m, sh.s = s, sh.dlse = dm + log_tri(ds), sh.rend = end;
  }
  g.sync();
  const float M = sh.m, S = sh.s;
  uint32_t key = 0u, idx = 0u;  // (top_p 1: every token at or above (0, 0))
  float tlse = M + log_tri(S);
  if (a.top_p < 1.0f)
    tlse = M + log_tri(__ull2float_rn(ss_nucleus(g, sh, a, r, M, S, key, idx, is_dry_run)) * (1.0f / SS_MASS_ONE));
  if (tid != 0 || is_dry_run) return;
  const uint64_t tk = ((uint64_t)ss_order(__float_as_uint(f.xt) >> 16) << 32) | (uint32_t)f.tok;
  const bool is_kept = f.tok >= 0 && tk >= (((uint64_t)key << 32) | idx);
  const float tlp = f.xt - tlse, dlp = f.xd - sh.dlse;
  const float u = tl_rand32((uint64_t)*a.seed, (uint64_t)f.pos);
  a.rows[r] = SsRow{(uint32_t)(r < K && is_kept && tlp > log_tri(u) + dlp), key, idx, tlse, sh.dlse, {0u, 0u, 0u}};
}

// A: item i = (row, chunk) of the target rows: the chunk's max and sum of exp, with top_p < 1 its histogram (the counts
// of its logits in the window below its max by bin) and its micro logits, the row's facts (chunk 0): rolled sweeps over
// the chunk in smem; the row's last chunk to finish (a count per row, back to 0 for the next launch) then runs the row
template <int NTH>
__device__ void ss_phase_a(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int item, bool is_dry_run) {
  int row, chunk, v0, v1;
  ss_item(a, item, row, chunk, v0, v1);
  if (row >= a.T) return;  // (T rows of a.chunks items: the items past them idle)
  v1 = is_dry_run ? v0 + 1 : v1;
  const int tid = g.tid, K = a.T - 1;
  const bool is_top_p_masked = a.top_p < 1.0f;
  if (tid == 0) ss_chunk_issue(a.logits + (int64_t)row * a.stride, v0, v1, sh), sh.mn = 0u;
  if (is_top_p_masked)
    for (int w = tid; w < SS_HIST_WORDS / 4; w += NTH) reinterpret_cast<uint4*>(sh.hist)[w] = make_uint4(0u, 0u, 0u, 0u);
  if (chunk == 0 && tid == 0 && !is_dry_run) {
    const int64_t dtok = ss_draft_token(a, row), t = max(dtok, (int64_t)0);
    a.facts[row] = SsFact{dtok, a.positions[a.logits_indices[row]], ss_target(a, row, t), ss_draft(a, min(row, K - 1), t),
                          {0.f, 0.f}};
  }
  g.sync();  // (sh.bar initialized)
  mbar_wait(&sh.bar, 0);
  float m = -INFINITY;
#pragma unroll 1
  for (int v = v0 + tid; v < v1; v += NTH) {
    const uint4 q = sh.x[v - v0];
#pragma unroll
    for (int e = 0; e < SS_VEC; e++)
      m = fmaxf(m, lp_pick(lp_lt(v * SS_VEC + e, a.V), lp_elem<__nv_bfloat16>(q, e), -INFINITY));
  }
  m = ss_cta_max_f(g, sh, m);  // (its barriers also order the histogram's zeroing before the counts)
  const float lo = m - ss_window(a);
  const int top = ss_bin(ss_order(__float_as_uint(m) >> 16));
  float s = 0.f;
#pragma unroll 1
  for (int v = v0 + tid; v < v1; v += NTH) {
    const uint4 q = sh.x[v - v0];
#pragma unroll
    for (int e = 0; e < SS_VEC; e++) {
      const float f = lp_elem<__nv_bfloat16>(q, e);
      const uint32_t in_row_mask = lp_lt(v * SS_VEC + e, a.V);
      s += __uint_as_float(__float_as_uint(lp_exp_diff(f, m)) & in_row_mask);
      if (is_top_p_masked && in_row_mask && f > lo) {
        const uint32_t b = __float_as_uint(f) >> 16, mag = b & SS_MAG16;
        const int i = top - ss_bin(ss_order(b));
        atomicAdd(&sh.hist[i >> 1], 1u << ((i & 1) << 4));
        if (mag != 0u && mag < SS_MICRO_EXP * SS_MANT) {
          const uint32_t slot = atomicAdd(&sh.mn, 1u);
          if (slot < SS_MICRO_CAP) sh.ml[slot] = (uint16_t)b;
        }
      }
    }
  }
  s = ss_cta_sum_f(g, sh, s);  // (its barriers also complete the counts)
  if (!is_dry_run) {
    const int bins = is_top_p_masked ? top - ss_bottom(lo) + 1 : 0;
    SsPart& p = a.part[item];
    if (tid == 0) p.m = m, p.s = s, p.bins = (uint32_t)bins, p.micro = sh.mn;
    if (is_top_p_masked) {
      if (tid < min((int)sh.mn, SS_MICRO_CAP)) p.mlist[tid] = sh.ml[tid];
      for (int w = tid; w < SS_HIST_WORDS / 4; w += NTH)
        reinterpret_cast<uint4*>(a.hist + (int64_t)item * SS_HIST_WORDS)[w] = reinterpret_cast<const uint4*>(sh.hist)[w];
      uint32_t* rh = a.rhist + (int64_t)row * SS_ROW_BINS + (SS_BIN_TOP - top);  // (the row's, at this chunk's max)
      for (int w = tid; 2 * w < bins; w += NTH) {
        const uint32_t c = sh.hist[w];
        if (c & 0xFFFFu) atomicAdd(rh + 2 * w, c & 0xFFFFu);
        if (c >> 16) atomicAdd(rh + 2 * w + 1, c >> 16);
      }
    }
    __threadfence();  // (the chunk's outputs before its count)
    g.sync();
    if (tid == 0) sh.is_last_chunk = atomicAdd(&a.chunks_done[row], 1u) == (uint32_t)a.chunks - 1;
    g.sync();
    if (!sh.is_last_chunk) return;
    if (tid == 0) a.chunks_done[row] = 0u;
    __threadfence();  // (the row's other chunks' outputs after their counts)
  }
  ss_row(g, sh, a, row, is_dry_run);
}

// B: slice s of G: the resampled row's residual log(max(p - q, 0)) over its nucleus (the bonus row and a rejected
// placeholder draft: the nucleus' logits) + the Gumbel noise of the row's own key, as vLLM's resample, its argmax over
// the slice (the first index of the max) into amax[s]; slice 0 also the step's outputs but the resampled token
template <int NTH>
__device__ void ss_phase_b(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int sl, int G, bool is_dry_run) {
  constexpr int SL = SS_SLICE_ELEMS / NTH;  // a thread's elements of the slice at most
  const int K = a.T - 1, ra = ss_accepted(a, g.tid & 31), rd = min(ra, K - 1);  // (the bonus row: draft row K - 1's)
  const SsRow o = ss_load_row(&a.rows[ra]);
  const SsFact f = ss_load_fact(&a.facts[ra]);
  const uint32_t direct_draw_mask = ~lp_lt(ra, K) | lp_lt((int)f.tok, 0);
  const uint64_t key = (uint64_t)philox_randint((uint64_t)*a.seed, (uint64_t)f.pos) | SS_NOISE_KEY_HI;
  const uint64_t thr = ((uint64_t)o.key << 32) | o.idx;
  const int e0 = (int)((int64_t)a.V * sl / G), e1 = (int)((int64_t)a.V * (sl + 1) / G);
  float xt[SL], xd[SL];  // (loaded at once, then a rolled loop: one copy of the noise's code, which runs cold)
#pragma unroll
  for (int j = 0; j < SL; j++) {
    const int t = min(e0 + g.tid + j * NTH, e1 - 1);
    xt[j] = ss_target(a, ra, t), xd[j] = ss_draft(a, rd, t);
  }
  unsigned long long best_key = 0ull;
#pragma unroll 1
  for (int j = 0; j < (is_dry_run ? 1 : SL); j++) {  // (element j in xt[0] / xd[0]: the arrays shift down after each)
    const int t = e0 + g.tid + j * NTH;
    const uint32_t kept_mask = ss_ge64(((uint64_t)ss_order(__float_as_uint(xt[0]) >> 16) << 32) | (uint32_t)t, thr);
    const float tlp = lp_pick(kept_mask, xt[0] - o.tlse, -INFINITY), dlp = xd[0] - o.dlse;
    const float ratio = exp_tri(dlp - tlp);
    const uint32_t below_one_mask = ~lp_ge(ratio, 1.0f) & lp_ge(1.0f, ratio);  // ratio < 1 (NaN: no)
    const float rl = lp_pick(direct_draw_mask, lp_pick(kept_mask, xt[0], -INFINITY),
                             lp_pick(below_one_mask, tlp + log1p_tri(-ratio), -INFINITY));
    const unsigned long long k = lp_key(fmaxf(rl + ss_gumbel(key, t), -INFINITY), (uint32_t)t) & lp_wide(lp_lt(t, e1));
    best_key = k > best_key ? k : best_key;
#pragma unroll
    for (int q = 0; q + 1 < SL; q++) xt[q] = xt[q + 1], xd[q] = xd[q + 1];
  }
  best_key = ss_cta_max(g, sh, best_key);
  if (is_dry_run) return;
  if (g.tid == 0) a.amax[sl] = best_key;
  if (sl == 0 && g.tid < a.T && g.tid != ra)  // the rows' tokens but row ra's: their drafts below it, 0 above
    a.sampled[g.tid] = ss_load_fact(&a.facts[g.tid]).tok & (int64_t)(int32_t)lp_lt(g.tid, ra);
  if (sl == 0 && g.tid == 0) a.num_sampled[0] = ra + 1, a.num_rejected[0] = a.T - (ra + 1);
}

// C (one group): the resampled token, the largest of the slices' keys, into row ra
template <int NTH>
__device__ void ss_phase_c(const LpGrp<NTH>& g, SsSmem<NTH>& sh, const SsArgs& a, int G, bool is_dry_run) {
  const int ra = ss_accepted(a, g.tid & 31);
  unsigned long long best_key = 0ull;
#pragma unroll 1
  for (int b = g.tid; b < G; b += NTH) {
    const unsigned long long k = __ldcg(&a.amax[b]);
    best_key = k > best_key ? k : best_key;
  }
  best_key = ss_cta_max(g, sh, best_key);
  if (g.tid == 0 && !is_dry_run) a.sampled[ra] = (int64_t)lp_index(best_key);
}

// The megakernel's OP_SS_* descriptor (an OpDesc's 256 bytes, schedule.py SSD): the LM head's logits, the scratch, the
// step record (its SR_SS words), the output ring (step.py RING slots of NT rows), the sizes
struct SsDesc {
  uint64_t logits, part, hist, rhist, done, dpart, amax, state, rec;
  uint64_t sampled, num_sampled, num_rejected;
  uint32_t V, chunks, nt, slices;  // chunks: per row at most (LP_MAX_CHUNKS); slices: the SMs
  uint32_t pad[36];
};
static_assert(sizeof(SsDesc) == 256, "SsDesc layout");
// a phase's SsArgs from the descriptor and the step's record words (T: the step's rows)
__device__ __forceinline__ SsArgs ss_args(const SsDesc& d, int T) {
  const int32_t* rec = reinterpret_cast<const int32_t*>(d.rec);
  const int64_t r = rec_i32(rec, SR_SS + SS_SLOT);
  SsArgs a;
  a.logits = reinterpret_cast<const __nv_bfloat16*>(d.logits), a.stride = d.V;
  a.draft = reinterpret_cast<const float*>(rec_u64(SR_SS + SS_DRAFT)), a.draft_stride = d.V;
  a.input_ids = reinterpret_cast<const int32_t*>(rec_u64(SR_SS + SS_IDS));
  a.positions = reinterpret_cast<const int64_t*>(rec_u64(SR_SS + SS_POSV));
  a.logits_indices = reinterpret_cast<const int64_t*>(rec_u64(SR_SS + SS_LIDX));
  a.seed = reinterpret_cast<const int64_t*>(rec_u64(SR_SS + SS_SEED));
  a.top_p = __int_as_float(rec_i32(rec, SR_SS + SS_TOP_P));
  a.V = (int)d.V, a.T = T, a.chunks = min((int)d.chunks, (int)d.slices / T);  // (phase A: the SMs over the T rows)
  a.part = reinterpret_cast<SsPart*>(d.part), a.hist = reinterpret_cast<uint32_t*>(d.hist);
  a.rhist = reinterpret_cast<uint32_t*>(d.rhist), a.chunks_done = reinterpret_cast<uint32_t*>(d.done);
  a.dpart = reinterpret_cast<float2*>(d.dpart), a.amax = reinterpret_cast<unsigned long long*>(d.amax);
  a.facts = reinterpret_cast<SsFact*>(d.state);
  a.rows = reinterpret_cast<SsRow*>(d.state + sizeof(SsFact) * SS_MAX_T);
  a.sampled = reinterpret_cast<int64_t*>(d.sampled) + r * d.nt, a.num_sampled = reinterpret_cast<int32_t*>(d.num_sampled) + r;
  a.num_rejected = reinterpret_cast<int32_t*>(d.num_rejected) + r;
  return a;
}

// OP_PREV's descriptor (an OpDesc's 256 bytes, schedule.py PREV): the step record, the spec sampler's output ring, its
// host ring (pinned host memory, the same layout: the device writes it over NVLink-C2C) and the host ring slots' flags,
// V2's request states (bin_counts: the penalties' output counts, 0 when V2 has none; strides in elements)
struct PrevDesc {
  uint64_t rec;
  uint64_t sampled, num_sampled;
  uint64_t h_sampled, h_num_sampled, h_flags;
  uint64_t total_len, all_token_ids, bin_counts;
  int64_t all_stride, bin_stride;
  uint32_t nt, pad0;
  uint32_t pad[40];
};
static_assert(sizeof(PrevDesc) == 256, "PrevDesc layout");

// OP_PREV, the first instruction of the last SM's prologue warps (nth threads, which wait for the embedding otherwise;
// nothing in the launch reads what it writes): the previous launched step's ring slot as the host left it (its sampler
// output, after anything its caller enqueued on it before this launch); not inlined: its code out of mk_main's
__device__ __noinline__ void prev_outputs(const PrevDesc& d, int pt, int nth, int bar) {
  const int32_t* rec = reinterpret_cast<const int32_t*>(d.rec);
  const uint32_t w = (uint32_t)rec_i32(rec, SR_BK + BK_PREV);
  const bool out_pending = (w & PREV_OUT) != 0u, post_pending = (w & PREV_POST) != 0u;
  const int64_t r = (w >> PREV_SLOT_SHIFT) & PREV_SLOT_MASK, T = (w >> PREV_T_SHIFT) & PREV_T_MASK;
  const int64_t req = w >> PREV_REQ_SHIFT;
  const int64_t row0 = r * d.nt;
  // (every load first, in parallel)
  const int32_t ns = __ldcg(reinterpret_cast<const int32_t*>(d.num_sampled) + r);
  const int64_t tok = pt < T ? __ldcg(reinterpret_cast<const int64_t*>(d.sampled) + row0 + pt) : 0;
  int32_t* total = reinterpret_cast<int32_t*>(d.total_len) + req;
  const int32_t len = post_pending ? *total : 0;
  // PREV_POST: the rest of V2's post_update (OP_BOOK takes the last sampled token and the computed tokens): the sampled
  // tokens appended to the token ids at the total length, with their output counts (the sums commute)
  if (post_pending && pt < ns) {
    reinterpret_cast<int32_t*>(d.all_token_ids)[req * d.all_stride + len + pt] = (int32_t)tok;
    if (d.bin_counts) atomicAdd(reinterpret_cast<int32_t*>(d.bin_counts) + req * d.bin_stride + tok, 1);
  }
  // PREV_OUT: its outputs into the host ring, then the slot's flag (V2's async output waits for it: no copies)
  if (out_pending) {
    if (pt < T) reinterpret_cast<int64_t*>(d.h_sampled)[row0 + pt] = tok;
    if (pt == 0) reinterpret_cast<int32_t*>(d.h_num_sampled)[r] = ns;
    __threadfence_system();  // (each thread's host stores before the flag: system scope)
  }
  named_sync(bar, nth);  // (the total length written once every thread has read it)
  if (pt == 0) {
    if (post_pending && ns > 0) *total = len + ns;
    if (out_pending) {
      __threadfence_system();
      asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(reinterpret_cast<uint32_t*>(d.h_flags) + r),
                   "r"((uint32_t)rec_i32(rec, SR_BK + BK_SEQ)) : "memory");
    }
  }
}

}  // namespace mk
