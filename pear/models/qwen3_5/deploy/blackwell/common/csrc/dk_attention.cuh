// The draft's attention as a worker-warp op (OP_ATTN): per (KV head, key split) item, the head's 32 query rows against
// the split's keys within each query's sliding window, by mma.sync bf16 -> fp32 with an online softmax in exp2 and P
// in bf16 for PV (FlashAttention's arithmetic); per item (O, m, l), merged by the head.
#pragma once
#include "dk_finish.cuh"

namespace dk {

constexpr int ATT_ROWS = GQA * QB;               // 32 query rows per KV head: row = g * QB + t
constexpr int ATT_KC = 16;                       // keys per chunk (two mma n-tiles)
constexpr int ATT_KG = 2;                        // key groups: warp w the chunks c % 2 == (w / 2) % 2
constexpr int ATT_DH = 2;                        // dim halves: warp w the output dims of half w / 4
constexpr int ATT_MAXK = 112;                    // keys per item at most (the window and the block over the splits)
constexpr int ATT_MAX_SPLITS = 32;               // items per KV head at most
constexpr int ATT_KROW = 2 * HDIM * 2;           // a key's K | V bytes in smem (16-byte pieces swizzled by the key)
constexpr int ATT_QROW = HDIM * 2;               // a query row's bytes in smem (swizzled likewise)
constexpr int ATT_PIECES = ATT_KROW / 16;        // 16-byte pieces of a key's K | V
constexpr int ATT_Q_OFF = ATT_MAXK * ATT_KROW;   // the query rows after every key of the item
constexpr int ATT_SMEM = ATT_Q_OFF + ATT_ROWS * ATT_QROW;
constexpr int ATT_NT = HDIM / ATT_DH / 8;        // output n-tiles of a warp (its half of the dims)
constexpr int ATT_XW = ATT_NT * 4 + 4;           // a lane's exchange words in the key groups' combine: O, then m, l

struct AttnDesc {  // 256 bytes (schedule.py ATTN)
  uint64_t q, kvq;      // out (split 0 of each head): the query block's rope'd q [QB][NQH][HDIM] and its
                        // k | v rows [QB][NKVH][2 * HDIM] (bf16)
  uint64_t sums;        // the qkv op's sums (E_QKV: fp32 [2 parity][mtiles][NB][TILE_M])
  uint64_t cache;       // the layer's draft KV cache [block][NKVH][kv_block][k | v]
  uint64_t part_o;      // fp32 [n_splits][NKVH][ATT_ROWS][HDIM]: unnormalized O of each item
  uint64_t part_ml;     // fp32 [n_splits][NKVH][ATT_ROWS][2]: row max (scaled log2 units), row sum
  uint64_t out;         // bf16 [QB][NQH][HDIM]: the attention output (the last split of a head merges the head's)
  uint64_t head_ctr;    // uint32 [NKVH]: the head's items stored (epoch-scaled)
  uint32_t n_splits, layer;
  float scale_log2;     // softmax scale * log2(e)
  uint32_t mtiles;      // the qkv op's M tiles (the sums' parity stride)
  uint64_t q_norm, k_norm;  // bf16 [HDIM]: q_norm's and k_norm's weights
  uint32_t pad[40];
};
static_assert(sizeof(AttnDesc) == 256, "AttnDesc layout");

struct AttnSmem {  // the item's static shared state
  const uint8_t* rows[ATT_MAXK];  // each key's K | V source (16-byte aligned)
};

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// the attended keys of the step: [lo, L] from the cache, then the block's 8 (key index j); keys below old_end: written
// before this step (the rest: the step's committed tokens and the block, by the qkv op)
struct AttnKeys {
  int64_t lo, L;
  uint32_t nc, nk, old_end;
};
__device__ __forceinline__ AttnKeys attn_keys(const DkArgs& a, const StepState& s) {
  AttnKeys k;
  k.L = s.last_pos;
  k.lo = max((int64_t)0, s.last_pos + 1 - (int64_t)a.tab.window_left);
  k.nc = (uint32_t)(k.L - k.lo + 1);
  k.nk = k.nc + a.tab.nstep + 1;  // (the block's tokens; the query rows past them pad the GEMVs' B)
  const int64_t first_new_pos = __ldcg(a.step.positions);  // the step's first token: positions >= it are new
  k.old_end = (uint32_t)max((int64_t)0, min((int64_t)k.nc, first_new_pos - k.lo));
  return k;
}

// 16-byte piece p of row j (a key's K | V, or a query row) in shared memory: pieces XOR-swizzled by j % 8 within each
// 128 bytes (ldmatrix reads 8 consecutive rows' same piece: conflict-free)
__device__ __forceinline__ uint32_t att_piece(uint32_t j, uint32_t p, uint32_t row_bytes) {
  return j * row_bytes + ((p ^ (j & 7)) * 16);
}

// The step's rows of KV head h from the qkv op's sums (the launch's parity): 32 query rows (q head GQA h + g, token t
// at row g QB + t), then 16 k and 16 v rows (the block's tokens, then the committed ones). The query rows into qs, this
// item's keys among the step's ([k0, k1)) into kv, the committed ones also into the draft KV cache.
constexpr int ATT_IN_DIMS = HDIM / 2 / 4;  // dims per thread in each half
static_assert((ATT_ROWS + 2 * NB) * 4 == NWT, "four threads per row");
static_assert(QKV_SLOTS == 2, "attn_inputs: two slots of the qkv sums");
__device__ void attn_inputs(const AttnDesc& g, const DkArgs& a, const StepState& s, const AttnKeys& ks, uint32_t h,
                            uint32_t sp, uint32_t k0, uint32_t k1, uint8_t* kv, uint8_t* qs, int pt, uint32_t epoch) {
  // four threads per row, each the 16 dims of a quarter of the first half and their rope partners
  const int r = pt >> 2, qd = pt & 3, d0 = qd * ATT_IN_DIMS;
  constexpr int ROW_Q = 0;  // the row's kind: q
  constexpr int ROW_K = 1;  // k
  constexpr int ROW_V = 2;  // v
  const int kind = r < ATT_ROWS ? ROW_Q : (r < ATT_ROWS + NB ? ROW_K : ROW_V);
  const int t = kind == ROW_Q ? r % QB : (r - ATT_ROWS) % NB;
  const uint32_t tile = kind == ROW_Q ? GQA * h + r / QB : (kind == ROW_K ? NQH + h : NQH + NKVH + h);
  // the row's key index among the item's (the committed token c = t - QB at its window position, the block's after the
  // window) and its position
  const bool is_ctx_row = kind != ROW_Q && t >= QB;
  const int c = t - QB;
  const bool has_token = kind == ROW_Q || (is_ctx_row ? (uint32_t)c < s.n_valid : t <= (int)a.tab.nstep);
  const int64_t pos = is_ctx_row ? (has_token ? __ldcg(a.step.positions + c) : ks.L) : query_pos(a, s, t);
  const uint32_t j = is_ctx_row ? (uint32_t)(pos - ks.lo) : ks.nc + (uint32_t)t;
  // (every row computed: the norm's shuffles take each warp)
  const bool is_item_key = kind != ROW_Q && has_token && j >= k0 && j < k1;
  const size_t plane = (size_t)g.mtiles * NB * TILE_M;  // (a slot's: the qkv op's sums, qkv_part)
  const float* src = reinterpret_cast<const float*>(g.sums) + (epoch & 1) * QKV_SLOTS * plane +
                     ((size_t)tile * NB + (size_t)t) * TILE_M;
  // the two slots' sums, rounded to bf16 as the GEMM's output
  float x[2][ATT_IN_DIMS];
#pragma unroll
  for (int half = 0; half < 2; half++) {
#pragma unroll
    for (int i = 0; i < ATT_IN_DIMS; i += 4) {
      const float4 f = __ldcg(reinterpret_cast<const float4*>(src + half * ROPE_HALF + d0 + i));
      const float4 f1 = __ldcg(reinterpret_cast<const float4*>(src + plane + half * ROPE_HALF + d0 + i));
      x[half][i] = bf16_round(f.x + f1.x), x[half][i + 1] = bf16_round(f.y + f1.y);
      x[half][i + 2] = bf16_round(f.z + f1.z), x[half][i + 3] = bf16_round(f.w + f1.w);
    }
  }
  // q / k: RMSNorm over the head dim (the squares per thread, then over the four) and neox rope in fp32, one rounding
  if (kind != ROW_V) {
    float ss = 0.f;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int i = 0; i < ATT_IN_DIMS; i++) ss += x[half][i] * x[half][i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rstd = rsqrt_tri(div_full(ss, (float)HDIM) + NORM_EPS);
    const __nv_bfloat16* wn = reinterpret_cast<const __nv_bfloat16*>(kind == ROW_Q ? g.q_norm : g.k_norm);
    const __nv_bfloat16* cs = a.tab.cos_sin + (size_t)pos * HDIM;
    float wv[2][ATT_IN_DIMS], cv[ATT_IN_DIMS], sv[ATT_IN_DIMS];
#pragma unroll
    for (int i = 0; i < ATT_IN_DIMS; i += 8) {
      bf16x8(__ldg(reinterpret_cast<const uint4*>(wn + d0 + i)), &wv[0][i]);
      bf16x8(__ldg(reinterpret_cast<const uint4*>(wn + ROPE_HALF + d0 + i)), &wv[1][i]);
      bf16x8(__ldg(reinterpret_cast<const uint4*>(cs + d0 + i)), &cv[i]);
      bf16x8(__ldg(reinterpret_cast<const uint4*>(cs + ROPE_HALF + d0 + i)), &sv[i]);
    }
#pragma unroll
    for (int i = 0; i < ATT_IN_DIMS; i++) {
      const float xn_lo = __fmul_rn(__fmul_rn(x[0][i], rstd), wv[0][i]),
                  xn_hi = __fmul_rn(__fmul_rn(x[1][i], rstd), wv[1][i]);
      // Inductor: fma(xn, cos, -(partner * sin)) for the first half, fma(xn, cos, partner * sin) for the second
      x[0][i] = __fmaf_rn(xn_lo, cv[i], -__fmul_rn(xn_hi, sv[i]));
      x[1][i] = __fmaf_rn(xn_hi, cv[i], __fmul_rn(xn_lo, sv[i]));
    }
  }
  uint4 o[2][ATT_IN_DIMS / 8];
#pragma unroll
  for (int half = 0; half < 2; half++)
#pragma unroll
    for (int i = 0; i < ATT_IN_DIMS / 8; i++) o[half][i] = pack_bf16x8(&x[half][i * 8]);
  // 16-byte pieces of the row: dims d0.. at d0 / 8, 64 + d0.. at 8 + d0 / 8 (a k row's v is 16 pieces further)
  const uint32_t p0 = d0 / 8, vp = kind == ROW_V ? HDIM * 2 / 16 : 0u;
  if (kind == ROW_Q) {
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int i = 0; i < ATT_IN_DIMS / 8; i++) sts128(qs + att_piece(r, half * 8 + p0 + i, ATT_QROW), o[half][i]);
  } else if (is_item_key) {
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int i = 0; i < ATT_IN_DIMS / 8; i++) sts128(kv + att_piece(j - k0, vp + half * 8 + p0 + i, ATT_KROW), o[half][i]);
    if (is_ctx_row) {  // the committed token's k / v into the draft KV cache (its block; the null block takes none)
      const uint32_t kvb = a.tab.kv_block;
      const int32_t blk = __ldcg(a.step.block_table + pos / kvb);
      if (blk != 0) {
        uint4* row = reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(g.cache) +
                                              (((size_t)blk * NKVH + h) * kvb + (size_t)(pos % kvb)) * (4 * HDIM / 2));
#pragma unroll
        for (int half = 0; half < 2; half++)
#pragma unroll
          for (int i = 0; i < ATT_IN_DIMS / 8; i++) row[vp + half * 8 + p0 + i] = o[half][i];
      }
    }
  }
  if (sp == 0 && !is_ctx_row) {  // the block's q and k | v rows
    uint4* dst = kind == ROW_Q
        ? reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(g.q) + ((size_t)t * NQH + tile) * HDIM)
        : reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(g.kvq) + ((size_t)t * NKVH + h) * (2 * HDIM)) + vp;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int i = 0; i < ATT_IN_DIMS / 8; i++) dst[half * 8 + p0 + i] = o[half][i];
  }
}

// the head's merge, distributed over its items once all are stored: item sp merges rows [32 sp / n, 32 (sp + 1) / n),
// out = sum_s O_s 2^(m_s - M) / sum_s l_s 2^(m_s - M); 4 lanes per 4 dims of a row, each a quarter of the splits
// (unrolled: every load of a pass, m, l and O, in flight at once), combined by shuffles
__device__ void attn_merge_rows(const AttnDesc& g, uint32_t h, uint32_t sp, int pt) {
  constexpr int UNITS = HDIM / 4, LANES = 4, PER = ATT_MAX_SPLITS / LANES;
  const uint32_t r0 = ATT_ROWS * sp / g.n_splits, r1 = ATT_ROWS * (sp + 1) / g.n_splits;
  const int n_units = (int)(r1 - r0) * UNITS, j = pt % LANES;
  const size_t ml_stride = (size_t)NKVH * ATT_ROWS * 2, o_stride = (size_t)NKVH * ATT_ROWS * HDIM;
  for (int u0 = 0; u0 < n_units; u0 += NWT / LANES) {  // whole warps iterate together (the shuffles)
    const int unit = u0 + pt / LANES;
    const bool has_unit = unit < n_units;
    const uint32_t r = r0 + (has_unit ? unit / UNITS : 0);
    const int d4 = (unit % UNITS) * 4;
    const float* pm = reinterpret_cast<const float*>(g.part_ml) + ((size_t)h * ATT_ROWS + r) * 2;
    const float* po = reinterpret_cast<const float*>(g.part_o) + ((size_t)h * ATT_ROWS + r) * HDIM + d4;
    float m[PER], l[PER];  // every split's (m, l) and O in one round trip
    float4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
      const uint32_t s = j + i * LANES;
      const bool has_split = has_unit && s < g.n_splits;
      m[i] = has_split ? __ldcg(pm + s * ml_stride) : -INFINITY;
      l[i] = has_split ? __ldcg(pm + s * ml_stride + 1) : 0.f;
      v[i] = has_split ? __ldcg(reinterpret_cast<const float4*>(po + s * o_stride)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float mx = m[0];
#pragma unroll
    for (int i = 1; i < PER; i++) mx = fmaxf(mx, m[i]);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float ms = mx == -INFINITY ? 0.f : mx;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float lsum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
      const uint32_t s = j + i * LANES;
      if (!has_unit || s >= g.n_splits) continue;
      const float w = ex2_approx(m[i] - ms);  // (0 at m = -inf: ex2.approx.ftz(-inf) = +0, no select against zero)
      lsum = fmaf(l[i], w, lsum);
      acc.x = fmaf(v[i].x, w, acc.x), acc.y = fmaf(v[i].y, w, acc.y), acc.z = fmaf(v[i].z, w, acc.z), acc.w = fmaf(v[i].w, w, acc.w);
    }
#pragma unroll
    for (int o = 1; o < LANES; o <<= 1) {
      lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
      acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o), acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
      acc.z += __shfl_xor_sync(0xffffffffu, acc.z, o), acc.w += __shfl_xor_sync(0xffffffffu, acc.w, o);
    }
    if (!has_unit || j != 0) continue;
    const float inv = lsum > 0.f ? 1.0f / lsum : 0.f;
    const int gq = r / QB, t = r % QB;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(g.out) + ((size_t)t * NQH + GQA * h + gq) * HDIM + d4;
    *reinterpret_cast<uint2*>(out) = make_uint2(pack_bf16x2(acc.x * inv, acc.y * inv), pack_bf16x2(acc.z * inv, acc.w * inv));
  }
}

// OP_ATTN: item = h * n_splits + s. Every key of the item into the B buffer once the MMA is done with it (cc), the old
// ones before the qkv op's counter, then the query rows; warp w: m-tile w % 2, the 16-key chunks c % 2 == (w / 2) % 2,
// the dims of half w / 4; the 2 key groups combine, then the head's items merge a share of its rows each.
__device__ void attn_item(const Instr& in, const AttnDesc& g_ref, const DkArgs& a, const Smem& sm, const RingPos& cc,
                          AttnSmem& as, int pt, uint32_t epoch, const Trace& tr) {
  constexpr int TS_DEP = 1;      // the trace's points (Trace::stamp): the dependency's wait
  constexpr int TS_DEP_MET = 2;  // its end
  constexpr int TS_INPUTS = 6;   // the item's keys and query rows in
  constexpr int TS_KEYS = 3;     // the keys done
  constexpr int TS_STORED = 4;   // the partial stored
  constexpr int TS_HEAD = 5;     // every item of the head stored
  const AttnDesc g = g_ref;  // (registers: read through the global reference, a field is re-read after every store)
  const StepState s = step_state(a);
  const AttnKeys ks = attn_keys(a, s);
  const uint32_t h = in.r0 / g.n_splits, sp = in.r0 % g.n_splits;
  const uint32_t k0 = (uint32_t)((uint64_t)ks.nk * sp / g.n_splits), k1 = (uint32_t)((uint64_t)ks.nk * (sp + 1) / g.n_splits);
  const uint32_t nk = k1 - k0, nk_pad = (nk + ATT_KC - 1) / ATT_KC * ATT_KC;
  uint8_t* kv = sm.bbuf;
  uint8_t* qs = sm.bbuf + ATT_Q_OFF;
  const int w = pt >> 5, lane = pt & 31;
  const int mt = w & 1, kg = (w >> 1) & 1, dh = w >> 2;
  // the rows of the keys written before this step (block-table lookups in parallel): before the dependency
  const uint32_t n_old = ks.old_end > k0 ? min(nk, ks.old_end - k0) : 0u;
  for (uint32_t j = k0 + pt; j < k0 + n_old; j += NWT) {
    const int64_t pos = ks.lo + j;
    const int32_t blk = __ldcg(a.step.block_table + pos / a.tab.kv_block);
    const size_t row = ((size_t)blk * NKVH + h) * a.tab.kv_block + (size_t)(pos % a.tab.kv_block);
    as.rows[j - k0] = reinterpret_cast<const uint8_t*>(g.cache) + row * (4 * HDIM);
  }
  if (pt == 0) wait_ring_drained(sm, cc);
  named_sync(BAR_WRK, NWT);
  // the last chunk's rows past the item zeroed (P is 0 there, but stale bytes could be NaN), the old keys' loads
  for (uint32_t i = nk * ATT_PIECES + pt; i < nk_pad * ATT_PIECES; i += NWT)
    *reinterpret_cast<uint4*>(kv + att_piece(i / ATT_PIECES, i % ATT_PIECES, ATT_KROW)) = make_uint4(0u, 0u, 0u, 0u);
  for (uint32_t i = pt; i < n_old * ATT_PIECES; i += NWT)
    cp_async16(kv + att_piece(i / ATT_PIECES, i % ATT_PIECES, ATT_KROW), as.rows[i / ATT_PIECES] + (i % ATT_PIECES) * 16);
  cp_async_commit();
  if (pt == 0) {
    tr.stamp(TS_DEP);
    if (in.dep != NO_CTR) wait_counter(a.counters + in.dep, (epoch + 1) * a.ctr_count[in.dep]);
    tr.stamp(TS_DEP_MET);
  }
  named_sync(BAR_WRK, NWT);
  // the step's query rows and keys from the qkv op's sums
  attn_inputs(g, a, s, ks, h, sp, k0, k1, kv, qs, pt, epoch);
  // this thread's rows (lane / 4, + 8 of the m-tile): their tokens' window starts
  const int r_lo = mt * 16 + lane / 4, r_hi = r_lo + 8;
  const int64_t win_lo = ks.L + 1 + (r_lo % QB) - (int64_t)a.tab.window_left;
  const int64_t win_hi = ks.L + 1 + (r_hi % QB) - (int64_t)a.tab.window_left;
  float o[ATT_NT][4];
#pragma unroll
  for (int i = 0; i < ATT_NT; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;
  const float sl2 = g.scale_log2;
  cp_async_wait<0>();
  named_sync(BAR_WRK, NWT);
  if (pt == 0) tr.stamp(TS_INPUTS);
  for (uint32_t j0 = kg * ATT_KC; j0 < nk; j0 += ATT_KG * ATT_KC) {
    float sc[ATT_KC / 8][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int kk = 0; kk < HDIM / 16; kk++) {
      uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
      ldsm_x4(qs + att_piece(mt * 16 + (lane % 16), kk * 2 + lane / 16, ATT_QROW), a0, a1, a2, a3);
      // two n-tiles: matrices (keys 0.., dims k), (keys 0.., dims k + 8), (keys 8.., dims k), (keys 8.., dims k + 8)
      ldsm_x4(kv + att_piece(j0 + (lane % 8) + (lane / 16) * 8, kk * 2 + (lane / 8) % 2, ATT_KROW), b0, b1, b2, b3);
      mma16816(sc[0], a0, a1, a2, a3, b0, b1);
      mma16816(sc[1], a0, a1, a2, a3, b2, b3);
    }
    float cmax_lo = -INFINITY, cmax_hi = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < ATT_KC / 8; nt++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const uint32_t j = j0 + nt * 8 + (lane % 4) * 2 + e;  // the key (item index)
        const uint32_t kj = k0 + j;
        const int64_t pos = kj < ks.nc ? ks.lo + kj : ks.L + 1 + (kj - ks.nc);
        const bool in_item = j < nk;
        if (!in_item || pos < win_lo) sc[nt][e] = -INFINITY;
        if (!in_item || pos < win_hi) sc[nt][2 + e] = -INFINITY;
        cmax_lo = fmaxf(cmax_lo, sc[nt][e]);
        cmax_hi = fmaxf(cmax_hi, sc[nt][2 + e]);
      }
    }
#pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
      cmax_lo = fmaxf(cmax_lo, __shfl_xor_sync(0xffffffffu, cmax_lo, o2));
      cmax_hi = fmaxf(cmax_hi, __shfl_xor_sync(0xffffffffu, cmax_hi, o2));
    }
    const float mn_lo = fmaxf(m_lo, cmax_lo), mn_hi = fmaxf(m_hi, cmax_hi);
    const float ms_lo = mn_lo == -INFINITY ? 0.f : mn_lo, ms_hi = mn_hi == -INFINITY ? 0.f : mn_hi;
    const float al_lo = ex2_approx((m_lo == -INFINITY ? ms_lo : m_lo) * sl2 - ms_lo * sl2);
    const float al_hi = ex2_approx((m_hi == -INFINITY ? ms_hi : m_hi) * sl2 - ms_hi * sl2);
    m_lo = mn_lo, m_hi = mn_hi;
    l_lo *= al_lo, l_hi *= al_hi;
#pragma unroll
    for (int i = 0; i < ATT_NT; i++) {
      o[i][0] *= al_lo, o[i][1] *= al_lo;
      o[i][2] *= al_hi, o[i][3] *= al_hi;
    }
    // P = exp2(s * sl2 - m * sl2): row sums from fp32 P, the PV product from bf16 P
    uint32_t pa[ATT_KC / 8][2];
    const float b_lo = -ms_lo * sl2, b_hi = -ms_hi * sl2;
#pragma unroll
    for (int nt = 0; nt < ATT_KC / 8; nt++) {
      const float p0 = ex2_approx(fmaf(sc[nt][0], sl2, b_lo)), p1 = ex2_approx(fmaf(sc[nt][1], sl2, b_lo));
      const float p2 = ex2_approx(fmaf(sc[nt][2], sl2, b_hi)), p3 = ex2_approx(fmaf(sc[nt][3], sl2, b_hi));
      l_lo += p0 + p1, l_hi += p2 + p3;
      pa[nt][0] = pack_bf16x2(p0, p1);
      pa[nt][1] = pack_bf16x2(p2, p3);
    }
    // O += P (16 x 16 keys) V (16 keys x 64 dims): V^T fragments by transposed loads, two n-tiles per load
#pragma unroll
    for (int nt = 0; nt < ATT_NT; nt += 2) {
      uint32_t b0, b1, b2, b3;
      const uint32_t piece = HDIM * 2 / 16 + dh * ATT_NT + nt + lane / 16;  // (V's pieces follow K's)
      ldsm_x4_t(kv + att_piece(j0 + (lane % 8) + ((lane / 8) % 2) * 8, piece, ATT_KROW), b0, b1, b2, b3);
      mma16816(o[nt], pa[0][0], pa[0][1], pa[1][0], pa[1][1], b0, b1);
      mma16816(o[nt + 1], pa[0][0], pa[0][1], pa[1][0], pa[1][1], b2, b3);
    }
  }
#pragma unroll
  for (int o2 = 1; o2 < 4; o2 <<= 1) {
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, o2);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, o2);
  }
  if (pt == 0) tr.stamp(TS_KEYS);
  // the key groups of each (m-tile, dim half) combined through shared memory (no key read after this point): groups
  // [half, 2 half) hand their (O, m, l) to groups [0, half)
  float* xch = reinterpret_cast<float*>(kv);
  for (int half = ATT_KG / 2; half >= 1; half /= 2) {
    named_sync(BAR_WRK, NWT);
    float* slot = xch + ((size_t)((mt * ATT_DH + dh) * ATT_KG + (kg % half)) * 32 + lane) * ATT_XW;
    if (kg >= half && kg < 2 * half) {
#pragma unroll
      for (int i = 0; i < ATT_NT; i++) reinterpret_cast<float4*>(slot)[i] = make_float4(o[i][0], o[i][1], o[i][2], o[i][3]);
      reinterpret_cast<float4*>(slot)[ATT_NT] = make_float4(m_lo, m_hi, l_lo, l_hi);
    }
    named_sync(BAR_WRK, NWT);
    if (kg < half) {
      const float4 ml = reinterpret_cast<const float4*>(slot)[ATT_NT];
      const float mx_lo = fmaxf(m_lo, ml.x), mx_hi = fmaxf(m_hi, ml.y);
      const float mo_lo = mx_lo == -INFINITY ? 0.f : mx_lo, mo_hi = mx_hi == -INFINITY ? 0.f : mx_hi;
      // (0 where m = -inf: ex2.approx.ftz(-inf) = +0, so no select against zero)
      const float a_lo = ex2_approx(m_lo * sl2 - mo_lo * sl2), a_hi = ex2_approx(m_hi * sl2 - mo_hi * sl2);
      const float b_lo = ex2_approx(ml.x * sl2 - mo_lo * sl2), b_hi = ex2_approx(ml.y * sl2 - mo_hi * sl2);
#pragma unroll
      for (int i = 0; i < ATT_NT; i++) {
        const float4 v = reinterpret_cast<const float4*>(slot)[i];
        o[i][0] = fmaf(o[i][0], a_lo, v.x * b_lo), o[i][1] = fmaf(o[i][1], a_lo, v.y * b_lo);
        o[i][2] = fmaf(o[i][2], a_hi, v.z * b_hi), o[i][3] = fmaf(o[i][3], a_hi, v.w * b_hi);
      }
      l_lo = fmaf(l_lo, a_lo, ml.z * b_lo), l_hi = fmaf(l_hi, a_hi, ml.w * b_hi);
      m_lo = mx_lo, m_hi = mx_hi;
    }
  }
  // the item's partial (key group 0 of each m-tile and dim half): O over its dims, (m, l) per row
  if (kg == 0) {
    float* po = reinterpret_cast<float*>(g.part_o) + (((size_t)sp * NKVH + h) * ATT_ROWS) * HDIM;
#pragma unroll
    for (int nt = 0; nt < ATT_NT; nt++) {
      const int dim = dh * (HDIM / ATT_DH) + nt * 8 + (lane % 4) * 2;
      __stcg(reinterpret_cast<float2*>(po + (size_t)r_lo * HDIM + dim), make_float2(o[nt][0], o[nt][1]));
      __stcg(reinterpret_cast<float2*>(po + (size_t)r_hi * HDIM + dim), make_float2(o[nt][2], o[nt][3]));
    }
    if (dh == 0 && (lane % 4) == 0) {
      float* pm = reinterpret_cast<float*>(g.part_ml) + (((size_t)sp * NKVH + h) * ATT_ROWS) * 2;
      __stcg(reinterpret_cast<float2*>(pm + r_lo * 2), make_float2(m_lo * sl2, l_lo));
      __stcg(reinterpret_cast<float2*>(pm + r_hi * 2), make_float2(m_hi * sl2, l_hi));
    }
  }
  // every item of the head stored (the head's counter, acquired before the merge's loads): this item's rows merged
  named_sync(BAR_WRK, NWT);
  tr.stamp(TS_STORED);
  if (pt == 0) {
    uint32_t* hc = reinterpret_cast<uint32_t*>(g.head_ctr) + h;
    red_release_add(hc, 1);
    wait_counter(hc, (epoch + 1) * g.n_splits);
  }
  named_sync(BAR_WRK, NWT);
  tr.stamp(TS_HEAD);
  attn_merge_rows(g, h, sp, pt);
}

}  // namespace dk
