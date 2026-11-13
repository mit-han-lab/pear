// Full-attention core of a step of T <= ATT_NT tokens (Qwen3.5: 24 query heads, 4 KV heads of 256) on vLLM's paged
// bf16 KV cache, as megakernel instructions: OP_ATTN, a split-KV flash attention on the tensor cores, and
// OP_ATTN_COMBINE, the splits' merge into the o_proj's FP8 operand.
#pragma once
#include "common.cuh"
#include "../../common/csrc/step.cuh"

namespace mk {

constexpr int ATT_HQ = 24, ATT_HKV = 4, ATT_D = 256, ATT_G = ATT_HQ / ATT_HKV;
constexpr int ATT_NT = 8;                              // tokens per step at most
constexpr int ATT_ROWS = ATT_G * ATT_NT;               // query rows per kv head at most
constexpr int ATT_KOFF = ATT_HQ * 2 * ATT_D;           // 12288: k in the qkv projection (q|gate per head before it)
constexpr int ATT_VOFF = ATT_KOFF + ATT_HKV * ATT_D;   // 13312
constexpr int ATT_QKV = ATT_VOFF + ATT_HKV * ATT_D;    // 14336: the projection's outputs per token
constexpr int ATT_OUT = ATT_HQ * ATT_D;                // 6144: the attention's output per token (the o_proj's K)
constexpr int ATT_PAGE = 64;                           // tokens per page (4 ring stages): 1 or 2 kernel blocks
constexpr uint32_t ATT_SLOTS = 4;                      // ring stages per page
// extension stages: ATT_XS more 16 KB stages, slot e at xb + e * ATT_SLOT (the B buffers' region, the last reaching
// into the staging buffer); one below the query rows' slots streams once the earlier MMAs are done (AB_XB), any other
// once Q is in TMEM (AB_Q). An item's stage k is the ring's when k % (NST + ATT_XS) < NST, else an extension stage's
constexpr uint32_t ATT_XS = 5;
constexpr uint32_t ATT_SLOT = 16384;                   // bytes per page stage: two boxes
constexpr uint32_t ATT_BOX = 8192;                     // one TMA box: 64 keys x 128 bytes
constexpr uint32_t ATT_PO_BLOCK = ATT_NT * 64 * 2;    // a partial block: [NT tokens][64 dims] bf16 (1 KB)
constexpr uint32_t ATT_T_O = 0, ATT_T_S = 256, ATT_T_Q = 384;  // TMEM columns
constexpr float ATT_RESCALE = 8.0f;                    // O is rescaled once a row's max grows by more (base 2)
constexpr int ATT_MAX_SPLITS = 40;                     // splits per kv head at most (schedule.py; the combine's blocks)
constexpr int ATT_BAR_PRO = 1;                         // the prologue warps' named barrier (megakernel.cu BAR_PRO)
// the query rows' fp32 sums, copied by cp.async (one round trip) into slots of ATT_QSLOT bytes, normed in place; the
// slots end at the rope rows (stg + ATT_RT_OFF): in the staging buffer at T <= 4, into the B buffers' region at T = 8
constexpr uint32_t ATT_QSLOT = 1024 + 16;              // (padded: conflict-free row reads by the TMEM copy)

// the item's smem scratch (stg offsets): rope rows [NT][64] fp32, the norm weights, the softmax exchange
constexpr uint32_t ATT_RT_OFF = 27648, ATT_NW_OFF = 29696, ATT_XM_OFF = 30720;
// the softmax exchange at ATT_XM_OFF (floats): the page maxima [2 page parities][4 quadrants][2 halves][32 lanes], then
// the rows' l halves [4 quadrants][32 lanes] at ATT_XM_L
constexpr uint32_t ATT_XM_L = 2 * 4 * 2 * 32, ATT_XM_FLOATS = ATT_XM_L + 4 * 32;
// the query rows' slots per lane quadrant (row r at slot (r % 4) * stride + r / 4), their bytes below stg + ATT_RT_OFF
__device__ __forceinline__ uint32_t attn_q_stride(int T) { return ((uint32_t)(ATT_G * T) + 3) / 4; }
__device__ __forceinline__ uint32_t attn_q_bytes(int T) { return 4 * attn_q_stride(T) * ATT_QSLOT; }
// kind::f16 instruction descriptors (bf16 x bf16 -> f32, M = 128): S = Q K^T (N 64, K-major keys), O += P V (N 128, V
// MN-major)
constexpr uint32_t ATT_ID_QK = (1u << 4) | (1u << 7) | (1u << 10) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
constexpr uint32_t ATT_ID_PV = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);

struct AttnDesc {  // an OpDesc's 256 bytes for OP_ATTN / OP_ATTN_COMBINE (schedule.py ATTN)
  uint64_t kv;          // the layer's bf16 KV cache [kernel blocks][kv heads][64 tokens][k 256 | v 256]
  uint64_t tmap;        // TMA map over kv as [kernel blocks * kv heads * 64 rows][1024 bytes], box 128 B x 64, SW128
  uint64_t qn_w, kn_w;  // q_norm / k_norm weights (bf16 [256], applied as 1 + w)
  uint64_t cos_sin;     // bf16 [positions][64]: cos 0..31, sin 32..63
  uint64_t po;          // the splits' partial o (bf16): blocks [query head][quarter][split] of ATT_PO_BLOCK bytes
  uint64_t act;         // the o_proj's FP8 operand [NT][ATT_OUT]
  uint64_t rec;         // the step record
  float scale_log2, eps, s_in, f3;  // softmax scale * log2(e); q/k norm eps; the o_proj's input scale
  uint32_t src;         // the qkv projection's desc
  uint32_t n_splits;    // splits per kv head
  uint32_t kb_per_mb;   // kernel blocks per block-table entry
  uint32_t sec1, sec2;  // M-RoPE interleaving: section h at i % 3 == 1, i < 3 * sec1; w at i % 3 == 2, i < 3 * sec2
  uint32_t group;       // the layer's attention KV cache group: its block-table row in the record (rec_att_bt)
  uint64_t trace;       // (unused: null)
  uint64_t ml;          // the splits' m, l (fp32) [query head][split][NT][2]
  uint32_t kb_shift;    // log2 of the cache's kernel block tokens: 6 (64-token blocks) or 5 (32: two per page)
  uint32_t pad[33];
};
static_assert(sizeof(AttnDesc) == 256, "AttnDesc layout");

// The CTA's attention state in smem (namespace scope: addresses are immediates, no register is live across the roles'
// instruction loops): the mbarriers, the producer's gate and each role's counters
enum AttBar : int {
  AB_Q = 0,    // Q in TMEM (the 8 prologue warps)
  AB_S = 1,    // [2] S_b complete (MMA commit)
  AB_P = 3,    // [2] P_b in TMEM (the 8 prologue warps)
  AB_PV = 5,   // [2] P_b V complete (MMA commit)
  AB_O = 7,    // every MMA of the item complete (MMA commit)
  AB_OT = 8,   // O read: TMEM free (the prologue's thread 0, after its barrier)
  AB_C = 9,    // the combine's bulk copy of its partial blocks
  AB_QS = 10,  // (unused: the query rows come by cp.async, attn_split)
  AB_XF = 11,                    // [ATT_XS] an extension stage landed (TMA)
  AB_XE = AB_XF + (int)ATT_XS,   // [ATT_XS] an extension stage consumed (MMA commit)
  AB_XB = AB_XE + (int)ATT_XS,   // the B buffers' region free (a commit behind the earlier instructions' MMAs)
  AB_N = AB_XB + 1,
};
__shared__ uint64_t att_bar[AB_N];
__shared__ volatile uint32_t att_kvw;  // attention instructions whose prologue has written the step's k, v (this SM)
// attention instructions released by the prologue (its counter's release fence done): the producer issues the next
// instruction's loads only then, so that the fence does not wait behind them
__shared__ volatile uint32_t att_rel;
// per role: att_cnt's words
constexpr int AC_PROD_INSTRS = 0;   // the producer's attention instructions
constexpr int AC_MMA_PAGES = 1;     // the MMA warp's pages
constexpr int AC_MMA_ITEMS = 2;     // its items with pages
constexpr int AC_PRO_INSTRS = 3;    // the prologue's attention instructions
constexpr int AC_PRO_PAGES = 4;     // its pages
constexpr int AC_PRO_ITEMS = 5;     // its items with pages
constexpr int AC_PRO_COMBINES = 6;  // its combine copies
constexpr int AC_PROD_XS = 7;       // extension stages so far: the producer's
constexpr int AC_MMA_XS = 8;        // the MMA warp's
constexpr int AC_PRO_XS = 9;        // the prologue's
constexpr int AC_PROD_ITEMS = 10;   // the producer's items with pages
constexpr int ATT_CNT_WORDS = 12;
__shared__ uint32_t att_cnt[ATT_CNT_WORDS];
__device__ __forceinline__ void attn_init() {  // (one thread, before the CTA's barrier-init fence)
  constexpr int PRO_WARPS = 8;  // AB_Q's and AB_P's arrivals: the prologue warps'
  mbar_init(&att_bar[AB_Q], PRO_WARPS);
  for (int b = 0; b < 2; b++) {
    mbar_init(&att_bar[AB_S + b], 1);
    mbar_init(&att_bar[AB_P + b], PRO_WARPS);
    mbar_init(&att_bar[AB_PV + b], 1);
  }
  mbar_init(&att_bar[AB_O], 1);
  mbar_init(&att_bar[AB_OT], 1);
  mbar_init(&att_bar[AB_C], 1);
  mbar_init(&att_bar[AB_QS], 1);
  mbar_init(&att_bar[AB_XB], 1);
  for (int e = 0; e < (int)ATT_XS; e++) {
    mbar_init(&att_bar[AB_XF + e], 1);
    mbar_init(&att_bar[AB_XE + e], 1);
  }
  att_kvw = 0;
  att_rel = 0;
  for (int i = 0; i < ATT_CNT_WORDS; i++) att_cnt[i] = 0;
}

// the step's tokens: T, the first token's position P (the device's under asynchronous scheduling: step.cuh rec_pos),
// the attended length L = P + T
struct AttnStep {
  int32_t T, L, P;
};
__device__ __forceinline__ AttnStep attn_step(const int32_t* rec) {
  AttnStep a;
  a.T = min(max(rec_i32(rec, SR_NTOK), 1), ATT_NT);
  a.P = rec_pos(rec);
  a.L = a.P + a.T;
  return a;
}
// the item's pages [pg0, pg1)
__device__ __forceinline__ void attn_range(const AttnDesc& g, const int32_t* rec, uint32_t s, uint32_t& pg0, uint32_t& pg1) {
  const uint32_t L = (uint32_t)attn_step(rec).L;
  const uint32_t np = (L + ATT_PAGE - 1) / ATT_PAGE;  // (32-bit: np * n_splits < 2^32; no 64-bit division routine)
  pg0 = np * s / g.n_splits;
  pg1 = np * (s + 1) / g.n_splits;
}
// The desc's fields and the block table are read into registers before the item's dependency wait: after it every
// global load is an L2 round trip on the critical path, and the barriers' memory clobbers make the compiler reload a
// desc field at each use.

// the first row of logical kernel block kb (1 << kbs tokens), kv head h, in the cache viewed as [rows][1024 bytes]: its
// physical block from group grp's device row (the V2 runner's, kernel-block ids) or the host-copied table (kbm kernel
// blocks per entry)
__device__ __forceinline__ uint32_t attn_kb_row(uint32_t kbm, uint32_t grp, const int32_t* rec, uint32_t kb, uint32_t h,
                                                uint32_t kbs) {
  uint32_t pkb;
  if (rec_i32(rec, SR_FLAGS) & REC_FLAG_POS_ON_DEVICE) {
    pkb = (uint32_t)__ldg(rec_att_bt(rec, (int)grp) + kb);
  } else {
    const uint32_t mb = kb / kbm, sub = kb % kbm;
    pkb = (uint32_t)rec_i32(rec, SR_BT + mb) * kbm + sub;
  }
  return (pkb * ATT_HKV + h) << kbs;
}
// the partial block of (query head qh, quarter qd of its dims, split s), and the m, l of (query head, split, token)
__device__ __forceinline__ uint8_t* attn_po(uint8_t* po, uint32_t ns, uint32_t qh, uint32_t qd, uint32_t s) {
  return po + ((size_t)(qh * 4 + qd) * ns + s) * ATT_PO_BLOCK;
}
__device__ __forceinline__ float* attn_ml(float* ml, uint32_t ns, uint32_t qh, uint32_t s, uint32_t t) {
  return ml + ((size_t)(qh * ns + s) * ATT_NT + t) * 2;
}
// slot and phase of the ring position k past cc
__device__ __forceinline__ void attn_ring_at(const RingPos& cc, uint32_t k, uint32_t nst, uint32_t& slot, uint32_t& phase) {
  const uint32_t p = cc.slot + k;
  slot = p % nst;
  phase = cc.phase ^ ((p / nst) & 1);
}
// the item's stages of the ring and of the extension, of its S stages
template <int NST>
__device__ __forceinline__ uint32_t attn_ring_count(uint32_t S) {
  constexpr uint32_t NS = NST + ATT_XS;
  return (S / NS) * NST + min(S % NS, (uint32_t)NST);
}
// stage k of an item whose first ring position is cc and first extension stage x0 (counted per role): its smem, its
// full / empty barriers, the phase to wait on, and whether its slot has a previous use (the producer's empty wait)
struct AttStage {
  uint8_t* p;
  uint64_t* full;
  uint64_t* empty;
  uint32_t phase;
  bool reuse;
};
template <int NST>
__device__ __forceinline__ AttStage attn_stage(const RingPos& cc, uint32_t x0, uint32_t k, uint8_t* stages, uint32_t stage_bytes,
                                               uint64_t* full, uint64_t* empty, uint8_t* xb) {
  constexpr uint32_t NS = NST + ATT_XS;
  const uint32_t c = k / NS, r = k % NS;
  AttStage a;
  if (r < (uint32_t)NST) {
    const uint32_t i = c * NST + r;
    uint32_t slot;
    attn_ring_at(cc, i, NST, slot, a.phase);
    a.p = stages + slot * stage_bytes;
    a.full = &full[slot];
    a.empty = &empty[slot];
    a.reuse = cc.n + i >= (uint32_t)NST;
  } else {
    const uint32_t x = x0 + c * ATT_XS + (r - NST), e = x % ATT_XS;
    a.phase = (x / ATT_XS) & 1;
    a.p = xb + e * ATT_SLOT;
    a.full = &att_bar[AB_XF + e];
    a.empty = &att_bar[AB_XE + e];
    a.reuse = x >= ATT_XS;
  }
  return a;
}

// ---- tcgen05 helpers ----
// a 128-byte-swizzled smem matrix: K-major (SBO 1024 = 8 rows of 128 bytes; lbo unused) or MN-major (lbo = the next
// 64-element chunk along MN; SBO 1024 = 8 rows along K)
__device__ __forceinline__ uint64_t attn_desc(uint32_t addr, uint32_t lbo) {
  constexpr uint32_t SBO_FIELD = 1024u >> 4;    // bits 32-45: the stride byte offset in 16-byte units
  constexpr uint32_t VERSION_FIELD = 1u << 14;  // bits 46-48: the descriptor version, 1
  constexpr uint32_t SW128_FIELD = 2u << 29;    // bits 61-63: the 128-byte swizzle
  constexpr uint32_t ADDR_MASK = 0x3FFFu;       // bits 0-13: the start address in 16-byte units
  constexpr uint32_t LBO_SHIFT = 16;            // bits 16-29: the leading byte offset in 16-byte units
  return ((uint64_t)(SBO_FIELD | VERSION_FIELD | SW128_FIELD) << 32) |
         (((addr >> 4) & ADDR_MASK) | ((lbo >> 4) << LBO_SHIFT));
}
__device__ __forceinline__ void tc_mma_bf16_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
// 32 consecutive columns of this thread's TMEM lane (the warp's quadrant; wait with tc_wait_ld)
__device__ __forceinline__ void tc_ld_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"
      "%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
        "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
        "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_st_x32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
      "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
      "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]),
      "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]),
      "r"(r[31])
      : "memory");
}
__device__ __forceinline__ void tc_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
      "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float exp2_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// The inputs of qk_norm_rope that do not depend on the step's projections, staged in smem before the dependency wait:
// per token t the bf16 cos / sin of the 32 rotary indices at their sections' positions (token 0's + t) as floats,
// rt[t][c 32 | s 32], and the q_norm, k_norm weights nw (bf16 [2][256]).
template <int NPRO_T>
__device__ __forceinline__ void qk_rope_stage(const AttnDesc& g, const int32_t* rec, int T, int pt, float* rt, uint4* nw) {
  const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(g.cos_sin);
  const int32_t p0 = rec_pos(rec), host_p0 = rec_i32(rec, SR_POS);  // (M-RoPE's host delta on the device position)
  for (int i = pt; i < T * 32; i += NPRO_T) {
    const int t = i >> 5, k = i & 31;
    const int sec = (k % 3 == 1 && k < 3 * (int)g.sec1) ? 1 : (k % 3 == 2 && k < 3 * (int)g.sec2) ? 2 : 0;
    const size_t pos = (size_t)(p0 + (rec_i32(rec, SR_MPOS + sec) - host_p0) + t);
    rt[t * 64 + k] = __bfloat162float(cs[pos * 64 + k]);
    rt[t * 64 + 32 + k] = __bfloat162float(cs[pos * 64 + 32 + k]);
  }
  if (pt < 64) nw[pt] = __ldg(reinterpret_cast<const uint4*>(pt < 32 ? g.qn_w : g.kn_w) + (pt & 31));
}
// q_norm / k_norm (wsel 0 / 1, GemmaRMSNorm) of one 256-value head and the interleaved M-RoPE of its first 64 dims, as
// the compiled baseline (inductor) computes them: x bf16, y = (x * rsqrt(sum x^2 / 256 + eps)) * (1 + w) in fp32 (not
// rounded), rotate-half with the token's bf16 cos / sin rt[64], bf16 out. One warp, lane l holds dims 8l..8l+7.
__device__ __forceinline__ void qk_norm_rope(const float (&x)[8], const float* rt, const uint4* nw, int wsel, float eps,
                                             int lane, uint32_t (&out)[4]) {
  constexpr float NORM_N = 256.0f;  // the values the norm's mean is over: the head's
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(x[i], x[i], ss);
  ss = warp_sum(ss);
  const float rs = rsqrt_tri(ss / NORM_N + eps);
  const uint4 wv = nw[wsel * 32 + lane];
  const uint32_t wu[4] = {wv.x, wv.y, wv.z, wv.w};
  float y[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    y[2 * i] = (x[2 * i] * rs) * (bf16lo(wu[i]) + 1.0f);
    y[2 * i + 1] = (x[2 * i + 1] * rs) * (bf16hi(wu[i]) + 1.0f);
  }
  float p[8];
#pragma unroll
  for (int i = 0; i < 8; i++) p[i] = __shfl_xor_sync(0xffffffffu, y[i], 4);  // dims d + 32 / d - 32 for lanes 0..7
  if (lane < 8) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t idx = (lane & 3) * 8 + i;  // rotary index 0..31
      const float c = rt[idx], s = rt[32 + idx];
      y[i] = lane < 4 ? y[i] * c - p[i] * s : y[i] * c + p[i] * s;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) out[i] = pack_bf16x2(y[2 * i], y[2 * i + 1]);
}

// ---- producer (W_PROD): the item's pages into the ring, 4 stages each; the page holding the step's tokens once
// the item's prologue has written their k, v into the cache (att_kvw reaches this attention instruction's number) ----
template <int NST>
__device__ __forceinline__ void produce_attn(const Instr& in, const AttnDesc& g, uint8_t* stages, uint32_t stage_bytes,
                                             uint64_t* full, uint64_t* empty, uint8_t* xb, uint8_t* stg, RingPos& cc) {
  const uint32_t seq = att_cnt[AC_PROD_INSTRS] + 1, x0 = att_cnt[AC_PROD_XS], ni = att_cnt[AC_PROD_ITEMS];
  // the extension slots wholly in the B buffers' region below the query rows' slots: 0 .. xq - 1
  const uint32_t xbs = (uint32_t)(stg - xb);
  const uint32_t xq = min(xbs + ATT_RT_OFF - attn_q_bytes(attn_step(reinterpret_cast<const int32_t*>(g.rec)).T), xbs) / ATT_SLOT;
  __syncwarp();
  if ((threadIdx.x & 31) == 0) att_cnt[AC_PROD_INSTRS] = seq;
  const int32_t* rec = reinterpret_cast<const int32_t*>(g.rec);
  const uint32_t h = in.r0 / g.n_splits;
  uint32_t pg0, pg1;
  attn_range(g, rec, in.r0 % g.n_splits, pg0, pg1);
  const int32_t P = attn_step(rec).P;
  const void* tmap = reinterpret_cast<const void*>(g.tmap);
  // the page's first rows: of its kernel block, or (32-token blocks, kb2) of its two
  constexpr uint32_t PAGE_SHIFT = 6;  // log2(ATT_PAGE): a kernel block of the page's tokens (kb_shift)
  constexpr int STAGE_X = 256;       // a stage's bytes of a key's 1024 (its two boxes)
  constexpr int BOX_X = 128;         // a box's
  const uint32_t kbm = g.kb_per_mb, grp = g.group, kbs = g.kb_shift;
  const bool kb2 = kbs < PAGE_SHIFT;
  uint32_t ra_next = 0u, rb_next = 0u;
  if (pg0 < pg1) {
    ra_next = attn_kb_row(kbm, grp, rec, kb2 ? 2 * pg0 : pg0, h, kbs);
    if (kb2) rb_next = attn_kb_row(kbm, grp, rec, 2 * pg0 + 1, h, kbs);
  }
  for (uint32_t pg = pg0; pg < pg1; pg++) {
    const int ra = (int)ra_next, rb = (int)rb_next;
    if (pg + 1 < pg1) {  // (their loads in flight during this page's waits)
      ra_next = attn_kb_row(kbm, grp, rec, kb2 ? 2 * pg + 2 : pg + 1, h, kbs);
      if (kb2) rb_next = attn_kb_row(kbm, grp, rec, 2 * pg + 3, h, kbs);
    }
    if ((int32_t)((pg + 1) * ATT_PAGE) > P)
      while (att_kvw < seq) {
      }
#pragma unroll 1
    for (uint32_t part = 0; part < ATT_SLOTS; part++) {
      const uint32_t k = ATT_SLOTS * (pg - pg0) + part;
      const AttStage a = attn_stage<NST>(cc, x0, k, stages, stage_bytes, full, empty, xb);
      // an extension slot's first stage in the item: once its bytes are free (below the query rows' slots: the MMAs of
      // the instructions before the item complete; else Q in TMEM)
      if (k - NST < ATT_XS) mbar_wait(&att_bar[(x0 + k - NST) % ATT_XS < xq ? AB_XB : AB_Q], ni & 1);
      if (a.reuse) mbar_wait(a.empty, a.phase ^ 1);
      if (elect_one()) {
        mbar_expect_tx(a.full, ATT_SLOT);
        const uint64_t pol = l2_evict_first();
        if (!kb2) {
          tma_load_2d_pol(a.p, tmap, (int)(part * STAGE_X), ra, a.full, pol);
          tma_load_2d_pol(a.p + ATT_BOX, tmap, (int)(part * STAGE_X + BOX_X), ra, a.full, pol);
        } else {  // (boxes of 32 rows: keys 0..31 from the first kernel block, 32..63 from the second)
          tma_load_2d_pol(a.p, tmap, (int)(part * STAGE_X), ra, a.full, pol);
          tma_load_2d_pol(a.p + ATT_BOX / 2, tmap, (int)(part * STAGE_X), rb, a.full, pol);
          tma_load_2d_pol(a.p + ATT_BOX, tmap, (int)(part * STAGE_X + BOX_X), ra, a.full, pol);
          tma_load_2d_pol(a.p + ATT_BOX + ATT_BOX / 2, tmap, (int)(part * STAGE_X + BOX_X), rb, a.full, pol);
        }
      }
      __syncwarp();
    }
  }
  const uint32_t S = ATT_SLOTS * (pg1 - pg0), nr = attn_ring_count<NST>(S);
  cc.advance(nr, NST);
  if ((threadIdx.x & 31) == 0) att_cnt[AC_PROD_XS] = x0 + (S - nr), att_cnt[AC_PROD_ITEMS] = ni + (S > 0 ? 1u : 0u);
  __syncwarp();
}

// ---- the MMA warp's part of an OP_ATTN item: per page j (ring positions cc + 4j ..; the S / P buffer of the page's
// parity in the SM's sequence of attention pages) S = Q K^T once its K stages land (their commit frees them), and once
// the softmax warps have written P, O += P V (freeing the V stages) ----
template <int NST>
__device__ void attn_mma(const Instr& in, const AttnDesc& g, uint8_t* stages, uint32_t stage_bytes, uint64_t* full,
                         uint64_t* empty, uint8_t* xb, RingPos& cc, uint32_t tmem) {
  uint32_t pg0, pg1;
  attn_range(g, reinterpret_cast<const int32_t*>(g.rec), in.r0 % g.n_splits, pg0, pg1);
  const uint32_t n = pg1 - pg0;
  if (n == 0) return;
  constexpr uint32_t LBO_KMAJOR = 16;  // (K-major keys: the leading byte offset is not read)
  constexpr uint32_t KG_BYTES = 2048;  // 16 keys of V: 2 swizzle atoms of 8 rows x 128 bytes
  const uint32_t np = att_cnt[AC_MMA_PAGES], ni = att_cnt[AC_MMA_ITEMS], x0 = att_cnt[AC_MMA_XS];
  if (elect_one()) tc_commit(&att_bar[AB_XB]);  // the B buffers' region, once the earlier MMAs complete
  __syncwarp();
  mbar_wait(&att_bar[AB_Q], ni & 1);  // Q in TMEM (every earlier user of TMEM is done)
  tc_fence_after();
#pragma unroll 1
  for (uint32_t j = 0; j <= n; j++) {
    if (j < n) {
      const AttStage k0 = attn_stage<NST>(cc, x0, ATT_SLOTS * j, stages, stage_bytes, full, empty, xb);
      const AttStage k1 = attn_stage<NST>(cc, x0, ATT_SLOTS * j + 1, stages, stage_bytes, full, empty, xb);
      mbar_wait(k0.full, k0.phase);
      mbar_wait(k1.full, k1.phase);
      tc_fence_after();
      const uint32_t b = (np + j) & 1;
      if (elect_one()) {
        const uint32_t a0 = smem_u32(k0.p), a1 = smem_u32(k1.p), d = tmem + ATT_T_S + 64 * b;
#pragma unroll
        for (uint32_t kk = 0; kk < 16; kk++) {  // dims 16 kk..: box kk / 4 (2 per stage), 32 bytes per k-step
          const uint32_t a = (kk < 8 ? a0 : a1) + ((kk >> 2) & 1) * ATT_BOX + (kk & 3) * 32;
          tc_mma_bf16_ts(d, tmem + ATT_T_Q + 8 * kk, attn_desc(a, LBO_KMAJOR), ATT_ID_QK, kk > 0 ? 1u : 0u);
        }
        tc_commit(&att_bar[AB_S + b]);
        tc_commit(k0.empty);
        tc_commit(k1.empty);
      }
      __syncwarp();
    }
    if (j >= 1) {
      const uint32_t jj = j - 1, b = (np + jj) & 1;
      const AttStage k2 = attn_stage<NST>(cc, x0, ATT_SLOTS * jj + 2, stages, stage_bytes, full, empty, xb);
      const AttStage k3 = attn_stage<NST>(cc, x0, ATT_SLOTS * jj + 3, stages, stage_bytes, full, empty, xb);
      mbar_wait(&att_bar[AB_P + b], ((np + jj) >> 1) & 1);
      mbar_wait(k2.full, k2.phase);
      mbar_wait(k3.full, k3.phase);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t v0 = smem_u32(k2.p), v1 = smem_u32(k3.p), pa = tmem + ATT_T_S + 64 * b;
#pragma unroll
        for (uint32_t kg = 0; kg < 4; kg++) {  // keys 16 kg..: 2 swizzle atoms of 8 keys each
          const uint32_t acc = (jj > 0 || kg > 0) ? 1u : 0u;  // (P: bf16 pairs in front of each softmax half's S)
          const uint32_t pk = pa + 32 * (kg >> 1) + 8 * (kg & 1);
          tc_mma_bf16_ts(tmem + ATT_T_O, pk, attn_desc(v0 + kg * KG_BYTES, ATT_BOX), ATT_ID_PV, acc);
          tc_mma_bf16_ts(tmem + ATT_T_O + 128, pk, attn_desc(v1 + kg * KG_BYTES, ATT_BOX), ATT_ID_PV, acc);
        }
        tc_commit(&att_bar[AB_PV + b]);
        tc_commit(k2.empty);
        tc_commit(k3.empty);
      }
      __syncwarp();
    }
  }
  if (elect_one()) tc_commit(&att_bar[AB_O]);
  __syncwarp();
  mbar_wait(&att_bar[AB_OT], ni & 1);  // the softmax warps have read O: TMEM is the next instructions'
  tc_fence_after();
  const uint32_t nr = attn_ring_count<NST>(ATT_SLOTS * n);
  cc.advance(nr, NST);
  if ((threadIdx.x & 31) == 0)
    att_cnt[AC_MMA_PAGES] = np + n, att_cnt[AC_MMA_ITEMS] = ni + 1, att_cnt[AC_MMA_XS] = x0 + ATT_SLOTS * n - nr;
  __syncwarp();
}

// the causal mask of a half page's 32 scores (key tok0 + c visible to the row while c <= vis): the others -inf, so that
// one select-free softmax serves every page (stale rows past L: masked too, before any use)
__device__ __forceinline__ void attn_mask(uint32_t (&sv)[32], int32_t vis) {
  constexpr uint32_t NEG_INF_BITS = 0xff800000u;  // -inf as fp32 bits
#pragma unroll
  for (int c = 0; c < 32; c++)
    if (c > vis) sv[c] = NEG_INF_BITS;
}

// ---- OP_ATTN: the prologue warps' part of an item (kv head h, split s) ----
// stg: >= 33 KB of smem scratch (rope rows, norm weights, softmax exchange at ATT_*_OFF; the query rows' slots below);
// xb: the B buffers' region, followed by stg (the extension stages); cc: the item's first ring position, then past it
template <int NST, int NPRO_T>
__device__ void attn_split(const Instr& in, const AttnDesc& g, const float* qkv_acc, float qkv_alpha, const uint32_t* counters,
                           uint32_t dep_target, uint32_t zero_target, uint8_t* stages, uint32_t stage_bytes, uint64_t* full,
                           uint64_t* empty, uint8_t* stg, uint8_t* xb, int pt, RingPos& cc, uint32_t tmem) {
  const uint32_t seq = att_cnt[AC_PRO_INSTRS] + 1, np = att_cnt[AC_PRO_PAGES], ni = att_cnt[AC_PRO_ITEMS],
                 x0 = att_cnt[AC_PRO_XS];
  const int32_t* rec = reinterpret_cast<const int32_t*>(g.rec);
  const uint32_t ns = g.n_splits;
  const uint32_t h = in.r0 / ns, s = in.r0 % ns;
  uint32_t pg0, pg1;
  attn_range(g, rec, s, pg0, pg1);
  const uint32_t n = pg1 - pg0;
  const AttnStep st = attn_step(rec);
  const int T = st.T, R = ATT_G * T;
  const int wp = pt >> 5, lane = pt & 31;
  constexpr int NW = NPRO_T / 32;
  constexpr int KV_ROW_ELEMS = 512;      // a cache row's bf16 values: k 256 | v 256
  constexpr int QUAD_ROWS = 12;          // query rows per TMEM lane quadrant at most (ATT_ROWS / 4)
  constexpr int BAR_PAIR = 3;            // the softmax warp pair (q, q + 4)'s named barrier: BAR_PAIR + q
  constexpr int PAIR_THREADS = 64;       // its threads
  // the desc's fields this item uses after the dependency, and the cache rows of the step's (token t, k or v) pairs
  // wp, wp + NW that lie in this item's pages (reshape_and_cache)
  const float eps = g.eps;
  const uint32_t kbs = g.kb_shift;
  __nv_bfloat16* const kvc = reinterpret_cast<__nv_bfloat16*>(g.kv);
  uint8_t* const pob = reinterpret_cast<uint8_t*>(g.po);
  float* const mlb = reinterpret_cast<float*>(g.ml);
  bool in_pages[2];
  uint32_t kvrow[2];
#pragma unroll
  for (int m = 0; m < 2; m++) {
    const int pr = wp + m * NW;
    const int32_t p = st.P + (pr >> 1);
    in_pages[m] = pr < 2 * T && p >= (int32_t)(pg0 * ATT_PAGE) && p < (int32_t)(pg1 * ATT_PAGE);
    kvrow[m] = in_pages[m] ? attn_kb_row(g.kb_per_mb, g.group, rec, (uint32_t)p >> kbs, h, kbs) +
                                 ((uint32_t)p & ((1u << kbs) - 1u))
                           : 0u;
  }
  // the query rows' slots: row r at slot (r % 4) * qst + r / 4 (a lane quadrant's rows consecutive), its fp32 sums
  // (1 KB) bulk-copied in, then normed in place to bf16 (the slot's first 512 bytes); below the rope rows
  float* rt = reinterpret_cast<float*>(stg + ATT_RT_OFF);  // [NT][64]
  uint4* nw = reinterpret_cast<uint4*>(stg + ATT_NW_OFF);  // bf16 [2][256]
  qk_rope_stage<NPRO_T>(g, rec, T, pt, rt, nw);
  const uint32_t qst = attn_q_stride(T);
  uint8_t* const qsl = stg + ATT_RT_OFF - attn_q_bytes(T);
  // the softmax, by every prologue warp: warp w holds the rows of lane quadrant q = w % 4 (thread i: row 4 i + q, TMEM
  // lane 32 q + i) and half k = w / 4 of a page's keys (32 k ..; their P in S columns 32 k .. 32 k + 15, in front of
  // the S columns it reads); the pair (q, q + 4) exchanges its halves' maxima per page in smem (barrier BAR_PAIR + q)
  const int q = wp & 3, hk = wp >> 2;
  const int r = 4 * lane + q;
  const bool has_row = lane < QUAD_ROWS && r < R;
  const uint32_t tl = tmem + ((uint32_t)(32 * q) << 16);
  float* xm = reinterpret_cast<float*>(stg + ATT_XM_OFF);  // [2 page parities][4 quadrants][2 halves][32 lanes]
  const int32_t lim = st.P + (has_row ? r / ATT_G : 0);  // keys visible to the row: <= lim
  const float sc = g.scale_log2;
  float m_run = -INFINITY, l_run = 0.f;  // (l_run: this half's keys)
  // the start's code run twice from one instance: first dry, before the dependency wait while the qkv projection
  // streams (one stale row's norm, page 0's softmax on stale S; no global, slot or TMEM store, barrier or O rescale),
  // so that the real pass runs from the instruction cache
#pragma unroll 1
  for (int rep = n > 0 ? 0 : 1; rep < 2; rep++) {
    const bool is_dry_run = rep == 0;
    if (!is_dry_run) {
      if (in.dep != INSTR_NO_CTR && pt == 0) wait_counter(counters + in.dep, dep_target);
      named_sync(ATT_BAR_PRO, NPRO_T);
      if (n > 0) {
        // the query rows' fp32 sums (1 KB each) into their slots, by the warps that norm them (warp w: rows w, w + 8,
        // ..), through the LSU (cp.async: not queued behind the KV stages' bulk loads in the SM's TMA unit); lane l
        // copies the 32 bytes of each row it reads (one round trip)
#pragma unroll 1
        for (int rq = wp; rq < R; rq += NW) {
          uint8_t* slot = qsl + ((rq & 3) * qst + (rq >> 2)) * ATT_QSLOT + lane * 32;
          const float* src = qkv_acc + (size_t)(rq / ATT_G) * ATT_QKV + (h * ATT_G + rq % ATT_G) * 2 * ATT_D + lane * 8;
          cp_async_16(slot, src);
          cp_async_16(slot + 16, src + 4);
        }
        // the step's k, v in this item's pages: loads first, then the cache rows
        float4 xa[2][2];
        const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int m = 0; m < 2; m++) {
          const int pr = wp + m * NW, t = pr >> 1, is_v = pr & 1;
          const float4* src = reinterpret_cast<const float4*>(qkv_acc + (size_t)(in_pages[m] ? t : 0) * ATT_QKV +
                                                              (is_v ? ATT_VOFF : ATT_KOFF) + h * ATT_D);
          xa[m][0] = in_pages[m] ? __ldcg(src + lane * 2) : z4;
          xa[m][1] = in_pages[m] ? __ldcg(src + lane * 2 + 1) : z4;
        }
#pragma unroll
        for (int m = 0; m < 2; m++) {
          if (!in_pages[m]) continue;
          const int pr = wp + m * NW, t = pr >> 1;
          const bool is_v = pr & 1;
          const float av[8] = {xa[m][0].x, xa[m][0].y, xa[m][0].z, xa[m][0].w, xa[m][1].x, xa[m][1].y, xa[m][1].z, xa[m][1].w};
          float x[8];
#pragma unroll
          for (int i = 0; i < 8; i++) x[i] = bf16_round(qkv_alpha * av[i]);
          uint32_t o[4];
          if (is_v) {
#pragma unroll
            for (int i = 0; i < 4; i++) o[i] = pack_bf16x2(x[2 * i], x[2 * i + 1]);
          } else {
            qk_norm_rope(x, rt + t * 64, nw, 1, eps, lane, o);
          }
          if (in.aux) {  // aux: the block zeroing's counter + 1 (0: none): the row may lie in a block it zeroes
            if (lane == 0) wait_counter(counters + in.aux - 1, zero_target);
            __syncwarp();
          }
          reinterpret_cast<uint4*>(kvc + (size_t)kvrow[m] * KV_ROW_ELEMS + (is_v ? ATT_D : 0))[lane] =
              make_uint4(o[0], o[1], o[2], o[3]);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");  // the cache rows, for the producer's TMA loads
        cp_async_wait_all();  // (this lane's own bytes)
      }
    }
    if (n > 0) {
      // the query rows: bf16, q_norm, M-RoPE (warp w: rows w, w + 8, ...), over their slot's first 512 bytes (the
      // warp's loads precede its stores: the norm's shuffles order them); dry: row w only (stale), no store
#pragma unroll 1
      for (int rq = wp; rq < (is_dry_run ? wp + 1 : R); rq += NW) {
        uint8_t* slot = qsl + ((rq & 3) * qst + (rq >> 2)) * ATT_QSLOT;
        const float4 a0 = reinterpret_cast<const float4*>(slot)[lane * 2], a1 = reinterpret_cast<const float4*>(slot)[lane * 2 + 1];
        const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float x[8];
#pragma unroll
        for (int i = 0; i < 8; i++) x[i] = bf16_round(qkv_alpha * av[i]);
        uint32_t o[4];
        qk_norm_rope(x, rt + (rq / ATT_G) * 64, nw, 0, eps, lane, o);
        __syncwarp();
        if (!is_dry_run) reinterpret_cast<uint4*>(slot)[lane] = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
    if (!is_dry_run) {
      named_sync(ATT_BAR_PRO, NPRO_T);
      if (pt == 0) att_kvw = seq;  // the producer may load the page holding the step's tokens
      if (n == 0) {  // no pages: an empty partial (m = -inf)
        if (pt < R) *reinterpret_cast<float2*>(attn_ml(mlb, ns, h * ATT_G + pt % ATT_G, s, pt / ATT_G)) = make_float2(-INFINITY, 0.f);
        named_sync(ATT_BAR_PRO, NPRO_T);
        if (pt == 0) att_cnt[AC_PRO_INSTRS] = seq;
        return;
      }
      // Q into TMEM: this warp's 128 dims (64 columns), 32 dims per store
#pragma unroll 1
      for (int ch = 0; ch < 4; ch++) {
        uint32_t v[16];
        const uint4* q4 = reinterpret_cast<const uint4*>(qsl + (q * qst + min(lane, QUAD_ROWS - 1)) * ATT_QSLOT +
                                                         (128 * hk + ch * 32) * 2);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const uint4 u = has_row ? q4[i] : make_uint4(0, 0, 0, 0);
          v[4 * i] = u.x, v[4 * i + 1] = u.y, v[4 * i + 2] = u.z, v[4 * i + 3] = u.w;
        }
        tc_st_x16(tl + ATT_T_Q + 64 * hk + 16 * ch, v);
      }
      tc_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&att_bar[AB_Q]);
    }
#pragma unroll 1
    for (uint32_t j = 0; j < (is_dry_run ? 1u : n); j++) {
      const uint32_t gp = np + j, b = gp & 1;
      if (!is_dry_run) mbar_wait(&att_bar[AB_S + b], (gp >> 1) & 1);
      tc_fence_after();
      const int32_t tok0 = (int32_t)((pg0 + j) * ATT_PAGE) + 32 * hk;  // this half's first key
      const bool all_visible = tok0 + 32 <= st.P;  // keys before the step's tokens: visible to every row
      const uint32_t ts = tl + ATT_T_S + 64 * b + 32 * hk;
      // pass 1: the half's max (S in base 2, the causal mask: key <= lim), the pair's max from the partner's half
      float mx = -INFINITY;
      {
        uint32_t sv[32];
        tc_ld_x32(ts, sv);
        tc_wait_ld();
        float mx4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        if (!all_visible) attn_mask(sv, lim - tok0);
        // the raw scores' max, scaled once (sc > 0: the same value as the scaled scores' max)
#pragma unroll
        for (int c = 0; c < 32; c++) mx4[c & 3] = fmaxf(mx4[c & 3], __uint_as_float(sv[c]));
        if (has_row) mx = __fmul_rn(fmaxf(fmaxf(mx4[0], mx4[1]), fmaxf(mx4[2], mx4[3])), sc);
      }
      xm[((b * 4 + q) * 2 + hk) * 32 + lane] = mx;
      named_sync(BAR_PAIR + q, PAIR_THREADS);
      mx = fmaxf(mx, xm[((b * 4 + q) * 2 + (hk ^ 1)) * 32 + lane]);  // (the same in both warps of the pair)
      // the row's max: kept until a key exceeds it by more than ATT_RESCALE (then O and l are rescaled); the first
      // finite max sets it (O's row holds no term yet)
      const float m_new = fmaxf(m_run, mx);
      const bool raises_max = m_new > m_run + ATT_RESCALE && !is_dry_run;
      const bool rescales_o = raises_max && m_run != -INFINITY;
      const float al = rescales_o ? exp2_approx(m_run - m_new) : 1.f;
      if (raises_max) {
        l_run *= al;
        m_run = m_new;
      }
      // O's rows *= al (this warp's half of the dims), once the last P V completed
      if (__any_sync(0xffffffffu, rescales_o)) {
        if (j > 0) mbar_wait(&att_bar[AB_PV + ((gp - 1) & 1)], ((gp - 1) >> 1) & 1);
        tc_fence_after();
#pragma unroll 1
        for (int ch = 0; ch < 4; ch++) {
          uint32_t v[32];
          tc_ld_x32(tl + ATT_T_O + 128 * hk + 32 * ch, v);
          tc_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; i++) v[i] = __float_as_uint(__uint_as_float(v[i]) * al);
          tc_st_x32(tl + ATT_T_O + 128 * hk + 32 * ch, v);
        }
        tc_wait_st();
      }
      // pass 2: P = 2^(s - m) of the half's keys (bf16, over its S columns), the half's sum in fp32
      {
        const bool has_max = m_run != -INFINITY;
        float sum4[4] = {0.f, 0.f, 0.f, 0.f};
        uint32_t sv[32], pk[16];
        tc_ld_x32(ts, sv);
        tc_wait_ld();
        if (!all_visible) attn_mask(sv, lim - tok0);
        // (a row with no finite max yet: 2^(x - inf) = 0; a masked key: 2^(-inf) = 0)
        const float mz = has_max ? m_run : INFINITY;
#pragma unroll
        for (int c2 = 0; c2 < 16; c2++) {
          const float p0 = exp2_approx(__fmul_rn(__uint_as_float(sv[2 * c2]), sc) - mz);
          const float p1 = exp2_approx(__fmul_rn(__uint_as_float(sv[2 * c2 + 1]), sc) - mz);
          sum4[c2 & 3] += p0 + p1;
          pk[c2] = pack_bf16x2(p0, p1);
        }
        if (!is_dry_run) {
          tc_st_x16(ts, pk);
          l_run += (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
          tc_wait_st();
        }
      }
      if (is_dry_run) continue;
      if ((int32_t)((pg0 + j) * ATT_PAGE) + ATT_PAGE > st.L) {
        // the sequence's last page: its rows past L (stale bytes of a reused page could be NaN, and P V multiplies them
        // by 0) cleared in the V stages; this warp's share of their 16-byte chunks
        const AttStage k2 = attn_stage<NST>(cc, x0, ATT_SLOTS * j + 2, stages, stage_bytes, full, empty, xb);
        const AttStage k3 = attn_stage<NST>(cc, x0, ATT_SLOTS * j + 3, stages, stage_bytes, full, empty, xb);
        mbar_wait(k2.full, k2.phase);
        mbar_wait(k3.full, k3.phase);
        const int r0 = st.L - (int32_t)((pg0 + j) * ATT_PAGE);
        for (int i = pt; i < (ATT_PAGE - r0) * 32; i += NPRO_T) {
          const int rr = r0 + i / 32, c = i % 32;  // chunk c: stage c / 16, box (c / 8) % 2, 16-byte chunk c % 8
          uint8_t* bx = ((c >> 4) ? k3.p : k2.p) + ((c >> 3) & 1) * ATT_BOX;
          *reinterpret_cast<uint4*>(bx + rr * 128 + (((c & 7) ^ (rr & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        }
        fence_async_smem();
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&att_bar[AB_P + b]);
    }
  }
  {
    // the row's m, l (fp32; o below): l of the pair's halves, added in key order, exchanged in slots of their own past
    // the maxima's (ATT_XM_L): the pairs synchronize only among themselves, and xm[q * 32 + lane] is, for q > 0, the
    // even-page maximum slot of another pair's warp, which that pair may still be writing or reading
    if (hk) xm[ATT_XM_L + q * 32 + lane] = l_run;
    named_sync(BAR_PAIR + q, PAIR_THREADS);
    if (!hk && has_row)
      *reinterpret_cast<float2*>(attn_ml(mlb, ns, h * ATT_G + r % ATT_G, s, r / ATT_G)) =
          make_float2(m_run, l_run + xm[ATT_XM_L + q * 32 + lane]);
  }
  // the split's partial o (unnormalized, relative to the row's m) in bf16, laid out for the combine: block (query head,
  // quarter of the dims, split) = [ATT_NT tokens][64 dims], 1 KB; 16-byte piece p of token t at (p ^ t) % 8 (the
  // combine's reads: at most 2-way bank conflicts); global stores, which the caller's release covers
  {
    mbar_wait(&att_bar[AB_O], ni & 1);
    tc_fence_after();
    // warp w: its lane quadrant's rows (warps w, w + 4 share it) from TMEM, the quarters 2 (w / 4) and 2 (w / 4) + 1
    const int q = wp & 3, rr = 4 * lane + q, jr = rr % ATT_G, tr = rr / ATT_G;
    const bool has_o_row = lane < QUAD_ROWS && rr < R;
    const uint32_t tq = tmem + ((uint32_t)(32 * q) << 16) + ATT_T_O;
#pragma unroll 1
    for (int k2 = 0; k2 < 2; k2++) {  // quarter qd = 2 (wp / 4) + k2: its two 32-dim halves loaded together
      const int qd = 2 * (wp >> 2) + k2;
      uint32_t v[2][32];
      tc_ld_x32(tq + 64 * qd, v[0]);
      tc_ld_x32(tq + 64 * qd + 32, v[1]);
      tc_wait_ld();
      if (has_o_row) {
        uint8_t* blk = attn_po(pob, ns, h * ATT_G + jr, qd, s) + tr * 128;
#pragma unroll
        for (int p = 0; p < 8; p++) {  // 16-byte piece p: dims 8 p ..
          const uint32_t* w = &v[p >> 2][8 * (p & 3)];
          *reinterpret_cast<uint4*>(blk + ((p ^ tr) & 7) * 16) =
              make_uint4(pack_bf16x2(__uint_as_float(w[0]), __uint_as_float(w[1])),
                         pack_bf16x2(__uint_as_float(w[2]), __uint_as_float(w[3])),
                         pack_bf16x2(__uint_as_float(w[4]), __uint_as_float(w[5])),
                         pack_bf16x2(__uint_as_float(w[6]), __uint_as_float(w[7])));
        }
      }
    }
    tc_fence_before();
    named_sync(ATT_BAR_PRO, NPRO_T);
    if (pt == 0) mbar_arrive(&att_bar[AB_OT]);  // TMEM read by every prologue warp (before the barrier)
  }
  const uint32_t nr = attn_ring_count<NST>(ATT_SLOTS * n);
  cc.advance(nr, NST);
  named_sync(ATT_BAR_PRO, NPRO_T);  // the caller releases the counter
  if (pt == 0) {
    att_cnt[AC_PRO_INSTRS] = seq, att_cnt[AC_PRO_PAGES] = np + n, att_cnt[AC_PRO_ITEMS] = ni + 1,
        att_cnt[AC_PRO_XS] = x0 + ATT_SLOTS * n - nr;
  }
}

// ---- OP_ATTN_COMBINE: item (query head, 64-dim quarter), every token. Its partial blocks (one per split, 1 KB,
// contiguous) by one bulk copy into stg, the splits' m, l loaded meanwhile (lane i: splits i, i + 32; the weights
// through smem); thread: token t = warp, dims 2 lane, 2 lane + 1 of the quarter; the merge in split order ----
static_assert(ATT_MAX_SPLITS <= 64 && ATT_MAX_SPLITS % 4 == 0, "the splits' m, l in two registers per lane; groups of 4");
template <int NPRO_T>
__device__ void attn_combine(const Instr& in, const AttnDesc& g, const float* qkv_acc, float qkv_alpha, const uint32_t* counters,
                             uint32_t dep_target, uint8_t* stg, uint8_t* xb, int pt, int T) {
  const uint32_t qh = in.r0 >> 2, qd = in.r0 & 3;
  const int t = pt >> 5, lane = pt & 31;
  const uint32_t dim = qd * 64 + 2 * lane, ns = g.n_splits;
  const bool has_token = t < T;
  // the desc's fields, before the dependency wait (attn_kb_row's comment)
  uint8_t* const po = attn_po(reinterpret_cast<uint8_t*>(g.po), ns, qh, qd, 0);
  float* const mlb = reinterpret_cast<float*>(g.ml);
  const float inv_s = div_full(1.0f, g.s_in);
  constexpr float E4M3_MAX = 448.f;  // the FP8 operand's largest magnitude
  uint8_t* const act = reinterpret_cast<uint8_t*>(g.act);
  // is_dry_run (Instr::aux's AUX_DRY, schedule.py add_attn): a copy of the item before the split on this SM, while the
  // prologue warps wait for the qkv projection: no dependency, copy, global load or store, no weights stored (its smem
  // loads read whatever is there), so that the real item runs from the instruction cache (the weight stream evicts it)
  constexpr uint32_t AUX_DRY = 1;
  const bool is_dry_run = in.aux & AUX_DRY;
  // the gate, independent of the splits (its producer is the qkv projection, which this SM's split already waited for):
  // loaded before the wait
  float g0 = 0.f, g1 = 0.f;
  if (has_token && !is_dry_run) {
    const float2 gg = __ldcg(reinterpret_cast<const float2*>(qkv_acc + (size_t)t * ATT_QKV + qh * 2 * ATT_D + ATT_D + dim));
    g0 = bf16_round(qkv_alpha * gg.x);
    g1 = bf16_round(qkv_alpha * gg.y);
  }
  const uint32_t cn = att_cnt[AC_PRO_COMBINES];  // this SM's combine copies so far (the copy barrier's phase)
  if (in.dep != INSTR_NO_CTR) {  // (a dry copy has none)
    if (pt == 0) {
      wait_counter(counters + in.dep, dep_target);
      asm volatile("fence.proxy.async.global;" ::: "memory");  // the acquired blocks, for the bulk copy
      mbar_expect_tx(&att_bar[AB_C], ns * ATT_PO_BLOCK);
      bulk_g2s(stg, po, ns * ATT_PO_BLOCK, &att_bar[AB_C]);
    }
    named_sync(ATT_BAR_PRO, NPRO_T);
  }
  float Ls = 0.f, O0 = 0.f, O1 = 0.f;
  if (has_token) {
    // lane i: the m, l of splits i, i + 32 for this warp's token; the token's max M over the splits, the weights
    // w = 2^(m - M) (0 for an empty split) and L = sum of w l, in parallel over the lanes; then the merge, independent
    // FMAs over the splits (the weights broadcast by shuffles)
    float ma = -INFINITY, la = 0.f, mb = -INFINITY, lb = 0.f;
    if ((uint32_t)lane < ns && !is_dry_run) {
      const float2 v = __ldcg(reinterpret_cast<const float2*>(attn_ml(mlb, ns, qh, lane, t)));
      ma = v.x, la = v.y;
    }
    if ((uint32_t)lane + 32 < ns && !is_dry_run) {
      const float2 v = __ldcg(reinterpret_cast<const float2*>(attn_ml(mlb, ns, qh, lane + 32, t)));
      mb = v.x, lb = v.y;
    }
    float M = fmaxf(ma, mb);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    // (an empty split's weight 0, not exp2(-inf - M), NaN when M is -inf too: masks, not selects)
    const float wa = __uint_as_float(__float_as_uint(exp2_approx(ma - M)) & ne_mask(ma, -INFINITY));
    const float wb = __uint_as_float(__float_as_uint(exp2_approx(mb - M)) & ne_mask(mb, -INFINITY));
    Ls = warp_sum(wa * la + wb * lb);
    // the warp's weights in smem (the B buffers' region: no operand there between the split and o_proj's prologue; the
    // dry copy before the split leaves it alone), split p at ws[p], 0 past ns (those lanes' m is -inf): read four per
    // broadcast load
    float* const ws = reinterpret_cast<float*>(xb) + t * ATT_MAX_SPLITS;
    if (!is_dry_run) {
      ws[lane] = wa;
      if (lane < ATT_MAX_SPLITS - 32) ws[32 + lane] = wb;
    }
    __syncwarp();
    if (!is_dry_run) mbar_wait(&att_bar[AB_C], cn & 1);
    // this thread's bf16 pair of split p: block p, token t, 16-byte piece lane / 4 (at (piece ^ t) % 8), word lane % 4
    const uint8_t* src = stg + t * 128 + (((lane >> 2) ^ t) & 7) * 16 + (lane & 3) * 4;
    float P0 = 0.f, P1 = 0.f;  // (two accumulators per value: even / odd splits)
    // groups of 4 splits up to ns rounded up, each group's loads first (the blocks past ns: within stg, not used)
#pragma unroll 1
    for (uint32_t p0 = 0; p0 < ns; p0 += 4) {
      const float4 w4 = *reinterpret_cast<const float4*>(ws + p0);
      const float w[4] = {w4.x, w4.y, w4.z, w4.w};
      uint32_t ow[4];
#pragma unroll
      for (int k = 0; k < 4; k++) ow[k] = *reinterpret_cast<const uint32_t*>(src + (p0 + k) * ATT_PO_BLOCK);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        if (w[k] == 0.f) continue;  // (an empty split's blocks are not written)
        if (k & 1) {
          P0 = fmaf(w[k], bf16lo(ow[k]), P0);
          P1 = fmaf(w[k], bf16hi(ow[k]), P1);
        } else {
          O0 = fmaf(w[k], bf16lo(ow[k]), O0);
          O1 = fmaf(w[k], bf16hi(ow[k]), O1);
        }
      }
    }
    O0 += P0;
    O1 += P1;
  } else if (!is_dry_run) {
    mbar_wait(&att_bar[AB_C], cn & 1);  // (every thread: the barrier's phase)
  }
  if (pt == 0 && !is_dry_run) att_cnt[AC_PRO_COMBINES] = cn + 1;
  if (has_token && !is_dry_run) {
    // the o_proj's operand, 2 dims per 16-bit store, in its B layout by absolute chunk (megakernel.cu PRO_ACT): value
    // D of row t at chunk D / 128 ([16-byte group][NT rows][16 bytes])
    const float v0 = fminf(fmaxf(bf16_round(O0 / Ls) * sigmoid_triton(g0) * inv_s, -E4M3_MAX), E4M3_MAX);
    const float v1 = fminf(fmaxf(bf16_round(O1 / Ls) * sigmoid_triton(g1) * inv_s, -E4M3_MAX), E4M3_MAX);
    const uint32_t D = qh * ATT_D + dim, b = D % 128;
    *reinterpret_cast<uint16_t*>(act + (D / 128) * (ATT_NT * 128) + (b / 16) * (ATT_NT * 16) + t * 16 + b % 16) =
        (uint16_t)f32x2_to_e4m3x2(v0, v1);
  }
  named_sync(ATT_BAR_PRO, NPRO_T);
}

}  // namespace mk
