// The candidates and the path: the LM head's tiles into each SM's running top-16 per mask position (E_LOGITS), the SMs'
// lists merged in FlashInfer's radix top-k order (value, CTA chunk, the index's place in the chunk's strided scan), the
// selector's edge scores, the walk (vLLM's _selector_walk_kernel: Gumbel-max), the sparse draft logits.
#pragma once
#include "dk_gemv.cuh"
#include "../../../common/csrc/dk_finish.cuh"

namespace dk {

constexpr int TOPK_THREADS = 1024;  // FlashInfer's radix top-k CTA: its thread-strided scan within a chunk
constexpr uint64_t KEY_NONE = 0;    // below every key

struct SelectDesc {  // 256 bytes (schedule.py SELECT)
  uint64_t part;       // uint64 [nsm][NSTEP][TOPK]: each SM's sorted top-16 keys per mask position
  uint64_t cand;       // int64 [NSTEP][TOPK]: the merged candidates (also vLLM's candidate order)
  uint64_t unary;      // fp32 [NSTEP][TOPK]
  uint64_t hsel;       // bf16 [NSTEP][RANK]: the selector's hidden projection of the mask positions
  uint64_t scores;     // fp32 [NSTEP][TOPK][TOPK]: scores[s][p][c]
  uint32_t n_part;     // lists to merge (the LM head's instructions)
  uint32_t pad[53];
};
static_assert(sizeof(SelectDesc) == 256, "SelectDesc layout");

// ---- keys: (order-preserving bf16 logit) << 32 | ~(FlashInfer's tie rank of the index) ----
constexpr uint32_t BF16_SIGN = 0x8000u;  // a bf16's sign bit
constexpr uint32_t BF16_MAG = 0x7fffu;   // its magnitude bits
constexpr uint32_t BF16_BITS = 0xffffu;  // all its bits
__device__ __forceinline__ uint32_t bf16_order(uint32_t b) {
  return (b & BF16_SIGN) ? (~b & BF16_BITS) : (b | BF16_SIGN);
}
__device__ __forceinline__ float order_value(uint32_t k) {
  const uint32_t b = (k & BF16_SIGN) ? (k & BF16_MAG) : (~k & BF16_BITS);
  return __uint_as_float(b << 16);
}
__device__ __forceinline__ uint32_t tie_rank(uint32_t id, uint32_t chunk) {
  const uint32_t c = id / chunk, l = id % chunk;
  return (c * TOPK_THREADS + l % TOPK_THREADS) * (chunk / TOPK_THREADS + 1) + l / TOPK_THREADS;
}
__device__ __forceinline__ uint64_t umax64(uint64_t a, uint64_t b) { return a > b ? a : b; }
__device__ __forceinline__ uint64_t umin64(uint64_t a, uint64_t b) { return a > b ? b : a; }
__device__ __forceinline__ uint64_t shfl64(uint64_t v, int o) {
  return ((uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o) << 32) | __shfl_xor_sync(0xffffffffu, (uint32_t)v, o);
}
// the warp's largest key by two redux maxima: the high words, then the low words of the lanes holding that high word
// (not five shuffle rounds of 64-bit keys, which every round of the tournament waits for)
__device__ __forceinline__ uint64_t warp_max64(uint64_t k) {
  const uint32_t hi_word = __reduce_max_sync(0xffffffffu, (uint32_t)(k >> 32));
  const uint32_t lo_word = __reduce_max_sync(0xffffffffu, (uint32_t)(k >> 32) == hi_word ? (uint32_t)k : 0u);
  return ((uint64_t)hi_word << 32) | lo_word;
}
// descending bitonic sort of one key per lane over the warp
__device__ __forceinline__ uint64_t warp_sort_desc(uint64_t c, int lane) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const uint64_t p = shfl64(c, j);
      c = (((lane & j) == 0) == ((lane & k) == 0)) ? umax64(c, p) : umin64(c, p);
    }
  }
  return c;
}
// lanes 0..15 hold sorted (desc) list a, lanes 16..31 sorted list b: returns in lanes 0..15 the sorted top 16 of both
__device__ __forceinline__ uint64_t warp_merge16(uint64_t c, int lane) {
  const uint64_t rev = __shfl_sync(0xffffffffu, (uint32_t)(c >> 32), 31 - lane);
  const uint64_t rev_lo = __shfl_sync(0xffffffffu, (uint32_t)c, 31 - lane);
  const uint64_t other = (rev << 32) | rev_lo;  // lane i < 16: b[15 - i]
  uint64_t x = umax64(c, other);                // bitonic: the top 16 of a and b
#pragma unroll
  for (int j = 8; j > 0; j >>= 1) {
    const uint64_t p = shfl64(x, j);
    x = (lane & j) == 0 ? umax64(x, p) : umin64(x, p);
  }
  return x;
}

// ---- E_LOGITS: the tile's logits bf16(alpha * sums) (vocab rows mt * 128 + ept) into the SM's running top-16 per mask
// position (`run` [NSTEP][TOPK], sorted desc; keys unique): per (position, warp) its few keys above the running 16th,
// or its sorted top 16 (more than CAND_FEW); then one insertion per key or a bitonic merge per list (none: skipped).
constexpr uint32_t CAND_FEW = 4;          // a warp's keys above the threshold inserted one by one
constexpr uint32_t CAND_SORTED = 0x100;   // the warp's list is its sorted top 16 (padded with KEY_NONE)
// k into the sorted (descending) list of lanes 0..15 if it ranks among its 16 (keys unique; lanes 16.. undefined)
__device__ __forceinline__ uint64_t list_insert(uint64_t c, uint64_t k, int lane) {
  const int pos = __popc(__ballot_sync(0xffffffffu, lane < TOPK && c > k));  // entries above k
  const uint64_t prev_lane_key = ((uint64_t)__shfl_up_sync(0xffffffffu, (uint32_t)(c >> 32), 1) << 32) |
                                 __shfl_up_sync(0xffffffffu, (uint32_t)c, 1);
  return lane < pos ? c : (lane == pos ? k : prev_lane_key);
}
// ncand: [NSTEP][WG_WARPS] the warps' candidate counts (| CAND_SORTED)
__device__ __forceinline__ void finish_logits(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[QB], const Smem& sm,
                                              uint64_t* run, uint64_t* cand, uint32_t* ncand, uint32_t chunk, uint32_t ns) {
  const int w = ept >> 5, lane = ept & 31;
  const uint32_t id = mt * TILE_M + ept;
  const uint32_t rank = ~tie_rank(id, chunk);  // (the key's low word: the same at every position)
#pragma unroll 1
  for (int t = 0; t < (int)ns; t++) {  // (ns: the proposal's drafts)
    float vt = v[0];  // v[t] by selects (an indexed read would put v in local memory for every op's epilogue)
#pragma unroll
    for (int j = 1; j < NSTEP; j++) vt = t == j ? v[j] : vt;
    const uint32_t b = __bfloat16_as_ushort(__float2bfloat16(d.alpha * vt));
    const uint64_t key = id < d.rows ? ((uint64_t)bf16_order(b) << 32) | rank : KEY_NONE;
    const bool is_above = key > run[t * TOPK + TOPK - 1];
    const uint32_t above_mask = __ballot_sync(0xffffffffu, is_above), n = __popc(above_mask);
    uint64_t* slot = cand + (t * WG_WARPS + w) * TOPK;
    if (n > CAND_FEW) {
      const uint64_t sorted_key = warp_sort_desc(is_above ? key : KEY_NONE, lane);
      if (lane < TOPK) slot[lane] = sorted_key;
    } else if (is_above) {
      slot[__popc(above_mask & ((1u << lane) - 1u))] = key;
    }
    if (lane == 0) ncand[t * WG_WARPS + w] = n > CAND_FEW ? (CAND_SORTED | min(n, (uint32_t)TOPK)) : n;
  }
  named_sync(BAR_EPI, NEPI);
  // warp w updates positions w, w + WG_WARPS
  for (int t = w; t < (int)ns; t += WG_WARPS) {
    uint32_t nq[WG_WARPS];
#pragma unroll
    for (int q = 0; q < WG_WARPS; q++) nq[q] = ncand[t * WG_WARPS + q];
    if ((nq[0] | nq[1] | nq[2] | nq[3]) == 0) continue;
    uint64_t c = lane < TOPK ? run[t * TOPK + lane] : KEY_NONE;
#pragma unroll 1
    for (int q = 0; q < WG_WARPS; q++) {
      const uint64_t* lst = cand + (t * WG_WARPS + q) * TOPK;
      const uint32_t n = q == 0 ? nq[0] : q == 1 ? nq[1] : q == 2 ? nq[2] : nq[3];  // (selects: nq stays in registers)
      if (n & CAND_SORTED) {
        if (lane >= TOPK) c = lst[lane - TOPK];
        c = warp_merge16(c, lane);
      } else {
        for (uint32_t i = 0; i < n; i++) c = list_insert(c, lst[i], lane);
      }
    }
    if (lane < TOPK) run[t * TOPK + lane] = c;
  }
  named_sync(BAR_EPI, NEPI);
}

// ---- Triton's Philox4x32-10 (tl.randint: the first word of philox(seed, offset_lo, offset_hi, 0, 0)) ----
__device__ __forceinline__ uint32_t philox_randint(uint64_t seed, uint64_t offset) {
  uint32_t c0 = (uint32_t)offset, c1 = (uint32_t)(offset >> 32), c2 = 0u, c3 = 0u;
  constexpr int PHILOX_ROUNDS = 10;            // Philox4x32-10
  constexpr uint32_t PHILOX_M0 = 0xD2511F53u;  // the rounds' multiplier of c0
  constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;  // and of c2
  constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;  // the key's increment per round: k0's
  constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;  // and k1's
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
// libdevice's log1pf as Triton inlines it (ftz), on a in (-1, 0]
__device__ __forceinline__ float log1p_tri(float a) {
  const float t = __fadd_rz(a, 1.0f);
  const int e = (__float_as_int(t) - 0x3F400000) & 0xFF800000;  // add.s32 -1061158912; and -8388608
  const float m = __int_as_float(__float_as_int(a) - e);
  const float s4 = __int_as_float(0x40800000 - e);                // 1082130432 - e
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
// vLLM's gumbel_noised_argmax noise of candidate `key` at the step's (seed, position): u = tl.rand(tl.randint(seed,
// pos), key) clamped at Triton's smallest rand, -log(-log1p(-u))
__device__ __forceinline__ float gumbel_noise(uint64_t seed, int64_t pos, int64_t key) {
  const uint32_t gseed = philox_randint(seed, (uint64_t)pos);
  const uint32_t r = philox_randint((uint64_t)gseed, (uint64_t)key);
  int32_t x = (int32_t)r;
  x = x < 0 ? -x - 1 : x;
  constexpr float TL_RAND_SCALE = 4.6566127342e-10f;  // tl.rand: 2^-31 per step of the int32 draw
  constexpr float TL_RAND_MIN = 4.6566127342e-10f;    // vLLM's _TL_RAND_MIN (includes_zero=False)
  const float u = fmaxf(__fmul_rn((float)x, TL_RAND_SCALE), TL_RAND_MIN);
  return -log_tri(-log1p_tri(-u));
}


// ---- OP_SELECT (one CTA, the worker warps): the merged candidates, the selector's scores, the walk, the outputs ----
constexpr int SEL_ROW = RANK * 2 + 16;  // a codebook row's bytes in shared memory, padded (conflict-free ldmatrix)
constexpr int SEL_ROWS = NSTEP * TOPK;  // the candidates of every step (and as many predecessor rows)
constexpr int ROW_PIECES = RANK * 2 / 16;  // 16-byte copies per row

// the selector's shared-memory scratch (in the B buffer)
struct SelectSmem {
  int64_t cand[SEL_ROWS];
  int64_t old_cand[SEL_ROWS];   // the cached candidates of the previous proposal (their draft logits are cleared)
  float unary[SEL_ROWS];
  float noise[SEL_ROWS];        // the walk's Gumbel noise of each (step, candidate)
  float scores[NSTEP * TOPK * TOPK];
  int32_t pick[SEL_ROWS];       // the pick of step s after the previous pick p: [s][p]
  int32_t path[NSTEP];          // the walk's pick per step
  uint64_t rows_bar;            // the codebook / projection rows landed (bulk copies)
};

constexpr int HEADS_PER_LANE = 5;  // the SMs' list heads a lane takes in the merge (32 lanes: up to 160 SMs)

// SelectSmem past the LM head's operand in the B buffer, which follows the ring: the selector's lists and rows reach
// from the ring into the B buffer below it
constexpr uint32_t SEL_SS_OFF = 81920;  // (past the LM head's bf16 image: its dry run runs during the LM head's MMAs)
constexpr uint32_t SEL_MAX_PARTS = 160;  // the SMs' lists at most
// scratch: the B buffer (SelectSmem); ring: the stage ring (free: every MMA is done), the SMs' lists, then the codebook
// rows. is_dry_run: the same code on stale inputs while the LM head streams (it runs once per launch: cold
// otherwise), no waits, copies, barriers or global stores, the ring only read.
__device__ void select_op(const Instr& in, const SelectDesc& g_ref, const DkArgs& a, uint8_t* scratch, uint8_t* ring,
                          int pt, uint32_t epoch, const Trace& tr, bool is_dry_run) {
  constexpr int TS_DEPS = 1;    // profiling (Trace::ts): the dependencies met
  constexpr int TS_MERGE = 2;   // the candidates merged
  constexpr int TS_SCORES = 3;  // the selector's scores
  constexpr int TS_OUTPUT = 4;  // the outputs stored
  constexpr int TS_LISTS = 5;   // the SMs' lists in shared memory
  constexpr int TS_WALK = 6;    // the walk
  const SelectDesc g = g_ref;  // (registers: read through the global reference, a field is re-read after every store)
  const int w = pt >> 5, lane = pt & 31;
  SelectSmem& ss = *reinterpret_cast<SelectSmem*>(scratch + SEL_SS_OFF);
  // the step's inputs loaded before the dependencies and stored after them, once the LM head's MMAs leave the B buffer
  const StepState st = step_state(a);
  const uint32_t ns = a.tab.nstep, nr = ns * TOPK;  // the proposal's drafts, their candidates
  const int64_t old_cand = pt < (int)nr ? __ldcg(a.step.cached_cand + (size_t)st.req * nr + pt) : 0;
  const float temp = __ldcg(a.step.temperature + st.req);
  const uint64_t seed = (uint64_t)__ldcg(a.step.seeds + st.req);
  if (pt == 0 && !is_dry_run) {
    if (in.dep != NO_CTR) wait_counter(a.counters + in.dep, (epoch + 1) * a.ctr_count[in.dep]);
    if (in.aux) wait_counter(a.counters + in.aux - 1, (epoch + 1) * a.ctr_count[in.aux - 1]);
  }
  named_sync(BAR_WRK, NWT);
  if (pt < (int)nr) ss.old_cand[pt] = old_cand;
  if (pt == 0 && !is_dry_run) tr.stamp(TS_DEPS);
  // every SM's sorted lists into shared memory at once, then warp t < NSTEP merges position t's: only the 16 lists with
  // the largest best keys can hold the top 16 (16 keys exceed every key of any other list; keys are unique)
  const uint64_t* part = reinterpret_cast<const uint64_t*>(g.part);
  uint64_t* lists = reinterpret_cast<uint64_t*>(ring);
  constexpr int KEYS_PER_COPY = 2;  // 16-byte copies
  for (uint32_t i = pt * KEYS_PER_COPY; i < g.n_part * NSTEP * TOPK; i += NWT * KEYS_PER_COPY)
    if (!is_dry_run) cp_async16(lists + i, part + i);
  cp_async_commit();
  cp_async_wait<0>();
  named_sync(BAR_WRK, NWT);
  if (pt == 0 && !is_dry_run) tr.stamp(TS_LISTS);
  if (w < (int)ns) {
    // (KEY_NONE past the lists by a mask over a load inside the lists' area, q < SEL_MAX_PARTS: no select against zero)
    static_assert(KEY_NONE == 0 && 32 * HEADS_PER_LANE <= SEL_MAX_PARTS, "the masked heads");
    auto list_head = [&](uint32_t q) { return lists[((size_t)q * NSTEP + w) * TOPK] & lt_mask64(q, g.n_part); };
    // the top 16 of the SMs' sorted lists by a tournament: lane l holds the current heads of lists l, l + 32, ..; each
    // round the warp's largest head (keys are unique) is the next candidate and its list moves on one key
    uint64_t h[HEADS_PER_LANE];
    uint32_t at[HEADS_PER_LANE];
#pragma unroll
    for (int i = 0; i < HEADS_PER_LANE; i++) h[i] = list_head(lane + 32 * i), at[i] = 0;
    uint64_t c = KEY_NONE;
#pragma unroll 1
    for (int r = 0; r < TOPK; r++) {
      uint64_t lm = h[0];
      int li = 0;
#pragma unroll
      for (int i = 1; i < HEADS_PER_LANE; i++)
        if (h[i] > lm) lm = h[i], li = i;
      const uint64_t mx = warp_max64(lm);
      if (lane == r) c = mx;
      if (lm == mx && mx != KEY_NONE) {
#pragma unroll
        for (int i = 0; i < HEADS_PER_LANE; i++)
          if (i == li) {
            at[i]++;
            h[i] = at[i] < TOPK ? lists[((size_t)(lane + 32 * i) * NSTEP + w) * TOPK + at[i]] : KEY_NONE;
          }
      }
    }
    if (lane < TOPK) {
      const uint32_t rank = ~(uint32_t)c;
      // the rank back to the index: rank = (chunk * 1024 + local % 1024) * minor + local / 1024
      const uint32_t chunk = a.tab.topk_chunk, minor = chunk / TOPK_THREADS + 1;
      const uint32_t scan_thread = rank / minor, scan_iter = rank % minor;
      const uint32_t id = (scan_thread / TOPK_THREADS) * chunk + scan_iter * TOPK_THREADS + scan_thread % TOPK_THREADS;
      ss.cand[w * TOPK + lane] = id;
      ss.unary[w * TOPK + lane] = order_value((uint32_t)(c >> 32));
    }
  }
  named_sync(BAR_WRK, NWT);
  if (pt == 0 && !is_dry_run) tr.stamp(TS_MERGE);
  // the rows: predecessor p of step s (the anchor for step 0, else candidate p of step s - 1), candidate c of step s,
  // the hidden projection of step s; the walk's noise meanwhile (it does not depend on the path)
  uint8_t* prow = ring;  // (the lists are merged: the ring is free again)
  uint8_t* srow = prow + SEL_ROWS * SEL_ROW;
  uint8_t* hrow = srow + SEL_ROWS * SEL_ROW;
  const __nv_bfloat16* hsel = reinterpret_cast<const __nv_bfloat16*>(g.hsel);
  // one bulk copy per row (a row is contiguous): far fewer requests than 16-byte copies
  constexpr int ROW_BYTES = RANK * 2;
  const int n_rows = 2 * nr + ns;
  if (pt == 0 && !is_dry_run) {
    mbar_init(&ss.rows_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(&ss.rows_bar, n_rows * ROW_BYTES);
  }
  named_sync(BAR_WRK, NWT);
  fence_async_smem();  // (the merge's reads of the lists in the ring before the copies overwrite them)
  for (int r = is_dry_run ? n_rows : pt; r < n_rows; r += NWT) {
    const void* src;
    uint8_t* dst;
    if (r < (int)nr) {  // the predecessors of step r / TOPK
      const int64_t pid = r < TOPK ? st.bonus : ss.cand[r - TOPK];
      src = a.tab.pred_cb + (size_t)pid * RANK, dst = prow + r * SEL_ROW;
    } else if (r < 2 * (int)nr) {  // the candidates
      src = a.tab.succ_cb + (size_t)ss.cand[r - nr] * RANK, dst = srow + (r - nr) * SEL_ROW;
    } else {  // the steps' hidden projections
      src = hsel + (size_t)(r - 2 * nr) * RANK, dst = hrow + (r - 2 * nr) * SEL_ROW;
    }
    bulk_g2s(dst, src, ROW_BYTES, &ss.rows_bar);
  }
  if (pt < (int)nr) ss.noise[pt] = gumbel_noise(seed, st.last_pos + 1 + pt / TOPK, ss.cand[pt]);
  if (!is_dry_run) mbar_wait(&ss.rows_bar, 0);
  named_sync(BAR_WRK, NWT);
  // the selector graph: P = bf16(pred * hsel) (in place), then per step bf16(P x succ^T) in fp32 (cuBLAS's bmm) plus
  // the unary logits in fp32; warp s the 16 x 16 scores of step s
  for (int i = pt; i < (int)nr * RANK / 2; i += NWT) {
    const int r = i / (RANK / 2), k2 = i % (RANK / 2);
    uint32_t* pw = reinterpret_cast<uint32_t*>(prow + r * SEL_ROW) + k2;
    const uint32_t h = *(reinterpret_cast<const uint32_t*>(hrow + (r / TOPK) * SEL_ROW) + k2);
    const uint32_t pv = pack_bf16x2(__fmul_rn(bf16lo(*pw), bf16lo(h)), __fmul_rn(bf16hi(*pw), bf16hi(h)));
    if (!is_dry_run) *pw = pv;
  }
  named_sync(BAR_WRK, NWT);
  if (w < (int)ns) {
    float d[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const int key = (lane % 8) + (lane / 16) * 8;
#pragma unroll 4
    for (int ks = 0; ks < RANK / 16; ks++) {
      uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
      ldsm_x4(prow + (w * TOPK + (lane % 16)) * SEL_ROW + (ks * 16 + (lane / 16) * 8) * 2, a0, a1, a2, a3);
      ldsm_x4(srow + (w * TOPK + key) * SEL_ROW + (ks * 16 + ((lane / 8) % 2) * 8) * 2, b0, b1, b2, b3);
      mma16816(d[0], a0, a1, a2, a3, b0, b1);
      mma16816(d[1], a0, a1, a2, a3, b2, b3);
    }
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int p = lane / 4 + (e >= 2 ? 8 : 0), c = nt * 8 + (lane % 4) * 2 + (e & 1);
        ss.scores[(w * TOPK + p) * TOPK + c] = __fadd_rn(ss.unary[w * TOPK + c], bf16_round(d[nt][e]));
      }
  }
  named_sync(BAR_WRK, NWT);
  if (pt == 0 && !is_dry_run) tr.stamp(TS_SCORES);
  // the walk: Gumbel-max over the realized row of each step (the row of the previous pick, the anchor's for step 0); a
  // row's pick does not depend on the path, so every (step, previous pick) row's argmax at once (the lowest index of
  // equal maxima), then the path by lookups; the selector's scores, candidates and unary logits stored meanwhile
  if (pt < (int)nr) {
    const int st_i = pt / TOPK;
    const float* row = ss.scores + pt * TOPK;
    float bx = -INFINITY;
    int bi = 0;
    for (int c = 0; c < TOPK; c++) {
      const float x = temp != 0.0f ? __fadd_rn(div_full(row[c], temp), ss.noise[st_i * TOPK + c]) : row[c];
      if (x > bx) bx = x, bi = c;
    }
    ss.pick[pt] = bi;
  }
  for (int i = is_dry_run ? nr * TOPK : pt; i < (int)(nr * TOPK); i += NWT) {
    reinterpret_cast<float*>(g.scores)[i] = ss.scores[i];
    if (i < (int)nr) {
      reinterpret_cast<int64_t*>(g.cand)[i] = ss.cand[i];
      reinterpret_cast<float*>(g.unary)[i] = ss.unary[i];
    }
  }
  named_sync(BAR_WRK, NWT);
  if (pt == 0) {
    int prev_pick = 0;
    for (int st_i = 0; st_i < (int)ns; st_i++) ss.path[st_i] = prev_pick = ss.pick[st_i * TOPK + prev_pick];
  }
  named_sync(BAR_WRK, NWT);
  if (is_dry_run) return;
  if (pt == 0) tr.stamp(TS_WALK);
  // the outputs: the previous candidates' draft logits cleared first (an id can be old and new), then the realized
  // rows at the new candidates, the cached candidates, the rows, the drafts
  const size_t vocab = a.tab.vocab;
  float* dl = a.step.draft_logits + (size_t)st.req * ns * vocab;
  if (pt < (int)nr) dl[(size_t)(pt / TOPK) * vocab + ss.old_cand[pt]] = -INFINITY;
  named_sync(BAR_WRK, NWT);
  if (pt < (int)nr) {
    const int s = pt / TOPK, prev_pick = s == 0 ? 0 : ss.path[s - 1];
    const float row = ss.scores[(s * TOPK + prev_pick) * TOPK + pt % TOPK];
    dl[(size_t)s * vocab + ss.cand[pt]] = row;
    a.step.cached_cand[(size_t)st.req * nr + pt] = ss.cand[pt];
    a.step.sel_scores[pt] = row;  // (batch row 0, as the drafts)
  }
  if (pt < (int)ns) a.step.draft_tokens[pt] = ss.cand[pt * TOPK + ss.path[pt]];
  if (pt == 0) tr.stamp(TS_OUTPUT);
}

}  // namespace dk
