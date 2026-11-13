// The DFlash 2 draft's proposal on sm_120a: the ops of its schedule that are not GEMVs (draft.py), as vLLM's
// DFlash2Speculator / DFlash2Qwen3ForCausalLM, over the FP8 e4m3 KV cache of the draft's group (FlashInfer's NHD rows
// [kernel block][16 tokens][8 heads][k 128 | v 128] bytes). Included by kernel.cuh after the prologue's enc_pub4.
#pragma once
#include "common.cuh"
#include "step.cuh"
#include "attention.cuh"
#include "spec_sample.cuh"

namespace mk {
template <int KIND>
__device__ __forceinline__ void enc_pub4(uint8_t* pub, uint32_t nch, uint32_t vi, uint32_t t, float (&v)[4], float gs,
                                         float rgs, float inv_s, bool st);  // (kernel.cuh)

constexpr int DQ_NQH = 32, DQ_NKVH = 8, DQ_HD = 128, DQ_G = DQ_NQH / DQ_NKVH, DQ_QB = 8, DQ_NSTEP = 7;
constexpr int DQ_TOPK = 16, DQ_RANK = 256, DQ_NL = 5;
constexpr int DQ_QKV = 6144, DQ_KOFF = 4096, DQ_VOFF = 5120;  // qkv_proj's rows: q 32 x 128, k 8 x 128, v 8 x 128
constexpr int DQ_CTX_ROW = DQ_NL * 2 * DQ_NKVH * DQ_HD;     // the context projection's outputs per token (10240)
constexpr int DQ_ROWS = DQ_G * DQ_QB;                       // query rows per KV head (row 4t + j: token t, head 4h + j)
constexpr int DQ_ROWB = DQ_NKVH * 2 * DQ_HD;                // a cache row's bytes (2048)
constexpr int DQ_TILE = 64, DQ_BOX = DQ_TILE * 128;         // keys per tile, a TMA box (64 keys x 128 bytes)
constexpr int DQ_MAX_SPLITS = 32;                           // D_ATTN's splits per KV head at most (draft.py MAX_SPLITS)
constexpr int DQ_NPRO = 256;
constexpr int TOPK_THREADS_FI = 1024;                       // FlashInfer's radix top-k CTA (its tie order)
constexpr int DQ_WARPS = 8;                                 // the prologue warps of an item
constexpr int DQ_CTX_LAYER = 2048;                          // a layer's k | v in a context projection row
constexpr int DQ_CTX_V = 1024;                              // v's offset in it
constexpr int DQ_HEAD_ROW_B = 256;                          // a KV head's cache row: k 128 | v 128 bytes
constexpr int DQ_V_OFF_B = 128;                             // v in a KV head's cache row (bytes)
constexpr int DQ_CS_ROW = 128;                              // cos_sin's row: cos 0..63, then sin
constexpr int DQ_CS_SIN = 64;                               // sin's offset in the row
// the attention item's staging (stg): Q [32][128] f16 (16-byte unit u of row r at u ^ (r & 7)), P [32][64] f16 (same
// swizzle over its 8 units), the row maxima exchange [8 warps][32]; at the end the l partials in P's place
constexpr uint32_t DQ_Q_OFF = 0, DQ_P_OFF = DQ_ROWS * DQ_HD * 2, DQ_XM_OFF = DQ_P_OFF + DQ_ROWS * DQ_TILE * 2;
constexpr uint32_t DQ_XL_OFF = DQ_P_OFF;
static_assert(DQ_XM_OFF + DQ_WARPS * DQ_ROWS * 4 <= STG_BYTES, "the draft attention's staging");

struct DraftDesc {   // 256 bytes (schedule.py DRAFT)
  uint64_t src;      // D_CTX: the context K / V bf16 [NT][10240] (layer l: k at 2048 l, v at 2048 l + 1024); D_PREP:
                     // qkv bf16 [NT][6144]; D_TOPK: the logits bf16 [NT][V]; D_SELECT: D_TOPK's lists
  uint64_t cache[5]; // D_CTX: the 5 layers' caches; D_PREP / D_ATTN: [0] the layer's (uint8 [rows][8][256])
  uint64_t tmap;     // D_ATTN: CUtensorMap 3D over the layer's cache {256 B, 8 heads, rows}, box {128 B, 1, 64}
  uint64_t qn_w;     // D_PREP: q_norm's weight (bf16 [128], applied as w)
  uint64_t kn_w;     // D_CTX: the 5 layers' k_norm weights (bf16 [5][128]); D_PREP: the layer's
  uint64_t cos_sin;  // bf16 [positions][128]: cos 0..63, sin 64..127 (neox over the whole head)
  uint64_t q;        // D_PREP out, D_ATTN in: Q f16 [8 KV heads][32 rows][128]
  uint64_t po, ml;   // D_ATTN out, D_COMBINE in: partial O fp32 [8][S][32][128], m (base 2), l fp32 [8][S][32][2]
  uint64_t act;      // D_COMBINE: o_proj's NVFP4 operand (codes [16 chunks][NT][128], scales [16][NT][16] after)
  uint64_t parts;    // D_TOPK out, D_SELECT in: u64 keys [NSTEP][chunks][TOPK] (sorted descending)
  uint64_t hsel;     // D_SELECT: the selector's hidden projection bf16 [NT][256] (row s + 1: step s)
  uint64_t pred_cb, succ_cb;  // D_SELECT: the codebooks bf16 [V][256]
  float eps, scale_log2, gs, f3;  // norms' epsilon; the softmax scale * log2(e); o_proj's operand global scale
  uint32_t n_splits;     // D_ATTN: splits per KV head
  uint32_t block_tokens; // the draft group's manager block (its kernel blocks: consecutive rows)
  uint32_t prep_ctr;     // D_ATTN: D_PREP's counter (the tiles from the step's first position wait for it ..
  uint32_t ctx_ctr;      // .. and for D_CTX's)
  uint32_t chunks;       // D_TOPK / D_SELECT: vocabulary chunks per step
  uint32_t chunk_len;    // D_TOPK: values per chunk (a multiple of 8)
  uint32_t topk_chunk;   // FlashInfer's radix top-k chunk (its tie order)
  uint32_t max_len;      // max_model_len (vLLM clamps the query block's rope positions below it)
  uint64_t stats;        // unused (0)
  uint32_t pad[14];
};
static_assert(sizeof(DraftDesc) == 256, "DraftDesc layout");

__device__ __forceinline__ int64_t dr_pos(uint32_t t) {
  return reinterpret_cast<const volatile int64_t*>(rec_u64(SR_DR + DR_POS))[t];
}
// the draft cache row of position p (the group's block-table row: kernel-block ids), -1: the null block (no write)
__device__ __forceinline__ int64_t dr_row(int64_t p) {
  const int32_t* bt = reinterpret_cast<const int32_t*>(rec_u64(SR_DR + DR_BT));
  const uint32_t kvb = (uint32_t)rec_i32(nullptr, SR_DR + DR_KVB);
  constexpr int32_t NULL_BLOCK = 0;  // vLLM's null block id
  const int32_t blk = __ldcg(bt + p / kvb);
  return blk == NULL_BLOCK ? -1 : (int64_t)blk * kvb + p % kvb;
}

// a head row (128 dims, 4 per lane: dims 4 lane ..) normed (x rstd) w and neox-rotated at position pos in fp32, one
// rounding to bf16 (Inductor's fused norm and rope); the partner dims (d + 64) in lane ^ 16
__device__ __forceinline__ void dq_norm_rope(float (&x)[4], const __nv_bfloat16* w, const __nv_bfloat16* cs, int lane,
                                             float eps) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) ss = fmaf(x[i], x[i], ss);
  ss = warp_sum(ss);
  const float rstd = rsqrt_tri(div_full(ss, (float)DQ_HD) + eps);
  const uint2 wu = __ldg(reinterpret_cast<const uint2*>(w + 4 * lane));
  const float wv[4] = {bf16lo(wu.x), bf16hi(wu.x), bf16lo(wu.y), bf16hi(wu.y)};
  float n[4];
#pragma unroll
  for (int i = 0; i < 4; i++) n[i] = __fmul_rn(__fmul_rn(x[i], rstd), wv[i]);
  const int dl = (4 * lane) & 63;  // the rotation index
  const uint2 cu = __ldg(reinterpret_cast<const uint2*>(cs + dl)),
              su = __ldg(reinterpret_cast<const uint2*>(cs + DQ_CS_SIN + dl));
  const float c[4] = {bf16lo(cu.x), bf16hi(cu.x), bf16lo(cu.y), bf16hi(cu.y)};
  const float s[4] = {bf16lo(su.x), bf16hi(su.x), bf16lo(su.y), bf16hi(su.y)};
  const bool is_upper = lane >= 16;  // (the head's second half)
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float p = __shfl_xor_sync(0xffffffffu, n[i], 16);
    // Inductor: fma(xn, cos, -(partner sin)) for the first half, fma(xn, cos, partner sin) for the second
    x[i] = bf16_round(is_upper ? __fmaf_rn(n[i], c[i], __fmul_rn(p, s[i]))
                               : __fmaf_rn(n[i], c[i], -__fmul_rn(p, s[i])));
  }
}
__device__ __forceinline__ uint32_t e4m3x4(const float (&v)[4]) {
  return f32x2_to_e4m3x2(v[0], v[1]) | (f32x2_to_e4m3x2(v[2], v[3]) << 16);
}

// ---- OP_D_CTX, item (layer l, token t), warp w = KV head w: the committed token t's (t < n_valid) k (k_norm, rope at
// its position) and v from the context projection (hidden_norm(fc(aux)) x the 5 layers' K / V rows) into the cache
__device__ __noinline__ void d_ctx(uint32_t item, const DraftDesc& g, int pt) {
  const uint32_t l = item / NT, t = item % NT;
  const DraftState st = draft_state();
  if (t >= st.n_valid) return;
  const int64_t pos = dr_pos(t);
  const int64_t row = dr_row(pos);
  if (row < 0) return;
  const int w = pt >> 5, lane = pt & 31;
  const __nv_bfloat16* src =
      reinterpret_cast<const __nv_bfloat16*>(g.src) + (size_t)t * DQ_CTX_ROW + l * DQ_CTX_LAYER + w * DQ_HD;
  const uint2 ku = __ldcg(reinterpret_cast<const uint2*>(src) + lane),
              vu = __ldcg(reinterpret_cast<const uint2*>(src + DQ_CTX_V) + lane);
  float k[4] = {bf16lo(ku.x), bf16hi(ku.x), bf16lo(ku.y), bf16hi(ku.y)};
  const float v[4] = {bf16lo(vu.x), bf16hi(vu.x), bf16lo(vu.y), bf16hi(vu.y)};
  const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(g.cos_sin) + (size_t)pos * DQ_CS_ROW;
  dq_norm_rope(k, reinterpret_cast<const __nv_bfloat16*>(g.kn_w) + l * DQ_HD, cs, lane, g.eps);
  uint8_t* dst = reinterpret_cast<uint8_t*>(g.cache[l]) + (size_t)row * DQ_ROWB + w * DQ_HEAD_ROW_B;
  reinterpret_cast<uint32_t*>(dst)[lane] = e4m3x4(k);
  reinterpret_cast<uint32_t*>(dst + DQ_V_OFF_B)[lane] = e4m3x4(v);
}

// ---- OP_D_PREP, item (KV head h, token t) of a layer: the query block's token t (position L + 1 + t), warps 0..3 its
// query heads 4 h + j (q_norm, rope; f16) into Q, warp 4 its key (k_norm, rope), warp 5 its value into the cache
__device__ __noinline__ void d_prep(uint32_t item, const DraftDesc& g, int pt) {
  constexpr int KEY_WARP = 4;    // the warp of the key, after the 4 query heads'
  constexpr int VALUE_WARP = 5;  // the warp of the value
  const uint32_t h = item / NT, t = item % NT;
  const int w = pt >> 5, lane = pt & 31;
  if (w > VALUE_WARP) return;
  const DraftState st = draft_state();
  const int64_t pos = st.L + 1 + t;
  const int64_t rpos = min(pos, (int64_t)g.max_len - 1);  // (vLLM's clamped query positions)
  const __nv_bfloat16* qkv = reinterpret_cast<const __nv_bfloat16*>(g.src) + (size_t)t * DQ_QKV;
  const uint32_t off = w < KEY_WARP ? (h * DQ_G + w) * DQ_HD : (w == KEY_WARP ? DQ_KOFF : DQ_VOFF) + h * DQ_HD;
  const uint2 xu = __ldcg(reinterpret_cast<const uint2*>(qkv + off) + lane);
  float x[4] = {bf16lo(xu.x), bf16hi(xu.x), bf16lo(xu.y), bf16hi(xu.y)};
  const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(g.cos_sin) + (size_t)rpos * DQ_CS_ROW;
  if (w < KEY_WARP) {
    dq_norm_rope(x, reinterpret_cast<const __nv_bfloat16*>(g.qn_w), cs, lane, g.eps);
    uint2 o;
    o.x = pack_f16x2(x[0], x[1]), o.y = pack_f16x2(x[2], x[3]);
    reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(g.q) + ((size_t)h * DQ_ROWS + t * DQ_G + w) * DQ_HD * 2)[lane] = o;
    return;
  }
  if (w == KEY_WARP) dq_norm_rope(x, reinterpret_cast<const __nv_bfloat16*>(g.kn_w), cs, lane, g.eps);
  if (pos >= (int64_t)g.max_len) return;
  const int64_t row = dr_row(pos);
  if (row < 0) return;
  uint8_t* dst = reinterpret_cast<uint8_t*>(g.cache[0]) + (size_t)row * DQ_ROWB + h * DQ_HEAD_ROW_B +
                 (w == KEY_WARP ? 0 : DQ_V_OFF_B);
  reinterpret_cast<uint32_t*>(dst)[lane] = e4m3x4(x);
}

// ---- OP_D_ATTN, item (KV head h, split): its 32 query rows (4 heads x 8 tokens) over the keys [lo, hi] =
// [max(0, L + 1 - W), L + 8], token t's from L + 1 + t - W (FlashInfer's bound; non-causal), as split-KV partials;
// tiles of 64 keys within manager blocks (tile i: block i / per, keys from (i % per) 64); positions below max_model_len
struct DqTiles {
  uint32_t t0, t1, per;
  int32_t lo, hi, L;
};
__device__ __forceinline__ DqTiles dq_tiles(const DraftDesc& g, uint32_t s) {
  const DraftState st = draft_state();
  DqTiles r;
  const int32_t W = rec_i32(nullptr, SR_DR + DR_WINDOW);
  r.L = (int32_t)st.L;
  r.lo = max(0, r.L + 1 - W);
  r.hi = r.L + DQ_QB;
  const uint32_t B = g.block_tokens;
  r.per = (B + DQ_TILE - 1) / DQ_TILE;
  const uint32_t a = ((uint32_t)r.lo / B) * r.per + ((uint32_t)r.lo % B) / DQ_TILE;
  const uint32_t n = ((uint32_t)r.hi / B) * r.per + ((uint32_t)r.hi % B) / DQ_TILE - a + 1;
  r.t0 = a + n * s / g.n_splits;
  r.t1 = a + n * (s + 1) / g.n_splits;
  return r;
}
__device__ __forceinline__ int32_t dq_key0(const DraftDesc& g, uint32_t i, uint32_t per) {
  return (int32_t)((i / per) * g.block_tokens + (i % per) * DQ_TILE);
}
__device__ __forceinline__ uint32_t dq_stages(const Instr& in, const DraftDesc& g) {
  const DqTiles tt = dq_tiles(g, in.r0 % g.n_splits);
  return tt.t1 - tt.t0;
}
// the producer: each tile as one ring stage (K box at 0, V box at DQ_BOX); the tiles from the step's first position on
// wait for D_PREP and D_CTX (their rows written this launch)
template <int NST>
__device__ __forceinline__ void produce_dattn(const Instr& in, const DraftDesc& g, uint8_t* stages, uint32_t stage_bytes,
                                              uint64_t* full, uint64_t* empty, RingPos& cc, const uint32_t* counters,
                                              uint32_t prep_target, uint32_t ctx_target, int lane) {
  const uint32_t h = in.r0 / g.n_splits, s = in.r0 % g.n_splits;
  const DqTiles tt = dq_tiles(g, s);
  const int32_t first_new = (int32_t)dr_pos(0);
  const int32_t* bt = reinterpret_cast<const int32_t*>(rec_u64(SR_DR + DR_BT));
  const uint32_t kvb = (uint32_t)rec_i32(nullptr, SR_DR + DR_KVB);
  const void* tmap = reinterpret_cast<const void*>(g.tmap);
  bool has_waited = false;
  for (uint32_t i = tt.t0; i < tt.t1; i++) {
    const int32_t key0 = dq_key0(g, i, tt.per);
    if (!has_waited && key0 + DQ_TILE > first_new) {
      if (lane == 0) wait_counter(counters + g.prep_ctr, prep_target), wait_counter(counters + g.ctx_ctr, ctx_target);
      __syncwarp();
      has_waited = true;
    }
    const int32_t row0 = __ldcg(bt + (uint32_t)key0 / kvb) * (int32_t)kvb + (int32_t)((uint32_t)key0 % kvb);
    if (cc.n >= NST) mbar_wait(&empty[cc.slot], cc.phase ^ 1);
    if (lane == 0) {
      uint64_t* bar = &full[cc.slot];
      mbar_expect_tx(bar, 2 * DQ_BOX);
      uint8_t* dst = stages + cc.slot * stage_bytes;
      const uint64_t pol = l2_evict_first();
      asm volatile("fence.proxy.async.global;" ::: "memory");  // (the generic stores of the step's rows)
#pragma unroll
      for (int part = 0; part < 2; part++)
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
            ::"r"(smem_u32(dst + part * DQ_BOX)), "l"(tmap), "r"(part * DQ_V_OFF_B), "r"((int)h), "r"(row0),
              "r"(smem_u32(bar)), "l"(pol)
            : "memory");
    }
    __syncwarp();
    cc.next(NST);
  }
}

// the prologue warps' part of an item (8 warps)
template <int NST>
__device__ __forceinline__ void dattn_item(const Instr& in, const DraftDesc& g, uint8_t* stages, uint32_t stage_bytes,
                                           uint64_t* full, uint64_t* empty, RingPos& cc, uint8_t* stg, int pt, int bar) {
  constexpr uint32_t Q_ROW_B = 256;  // a Q row in smem: 128 f16
  constexpr uint32_t Q_UNITS = 16;   // its 16-byte units
  constexpr uint32_t P_ROW_B = 128;  // a P row in smem: 64 f16
  const uint32_t h = in.r0 / g.n_splits, s = in.r0 % g.n_splits;
  const DqTiles tt = dq_tiles(g, s);
  const int32_t W = rec_i32(nullptr, SR_DR + DR_WINDOW);
  const int warp = pt >> 5, lane = pt & 31, gq = lane >> 2, tq = lane & 3;
  const uint32_t qs = smem_u32(stg + DQ_Q_OFF), ps = smem_u32(stg + DQ_P_OFF);
  float* xm = reinterpret_cast<float*>(stg + DQ_XM_OFF);
  float* xl = reinterpret_cast<float*>(stg + DQ_XL_OFF);
  {  // Q into smem
    const uint4* q = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(g.q) + (size_t)h * DQ_ROWS * DQ_HD * 2);
    for (uint32_t e = pt; e < DQ_ROWS * Q_UNITS; e += DQ_NPRO) {
      const uint32_t r = e / Q_UNITS, u = e % Q_UNITS;
      const uint4 v = __ldcg(q + e);
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(qs + r * Q_ROW_B + ((u ^ (r & 7)) << 4)),
                   "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  }
  named_sync(bar, DQ_NPRO);
  float o[2][2][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 2; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  float m_run[2][2], l_run[2][2];
#pragma unroll
  for (int a = 0; a < 2; a++) m_run[a][0] = m_run[a][1] = -INFINITY, l_run[a][0] = l_run[a][1] = 0.f;
  const float sl2 = g.scale_log2;
  for (uint32_t i = tt.t0; i < tt.t1; i++) {
    const int32_t key0 = dq_key0(g, i, tt.per);
    const int tile_len = min(DQ_TILE, (int)(g.block_tokens - (uint32_t)key0 % g.block_tokens));
    // the tile's valid keys [klo, khi) (the window's and the block's; past them any bytes, e4m3 NaN too); row r's
    // FlashInfer bound: keys >= L + 1 + t - W, t = r / 4 (rlo + t relative to the tile)
    const int klo = max(0, tt.lo - key0), khi = min(tile_len, tt.hi - key0 + 1), rlo = tt.L + 1 - W - key0;
    mbar_wait(&full[cc.slot], cc.phase);
    const uint32_t ks = smem_u32(stages + cc.slot * stage_bytes), vs = ks + DQ_BOX;
    // ---- S for keys 8 w .. 8 w + 7, all 32 rows
    float sacc[2][4];
#pragma unroll
    for (int a = 0; a < 2; a++) sacc[a][0] = sacc[a][1] = sacc[a][2] = sacc[a][3] = 0.f;
    const uint32_t key = warp * 8 + gq;
#pragma unroll 4
    for (int kk = 0; kk < 8; kk++) {
      const uint32_t d0 = kk * 16 + 2 * tq;
      uint32_t b[2];
      b[0] = e4m3x2_to_f16x2(lds_u16(box_byte(ks, key, d0)));
      b[1] = e4m3x2_to_f16x2(lds_u16(box_byte(ks, key, d0 + 8)));
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        uint32_t a[4];
        const int q4 = lane >> 3, r8 = lane & 7;
        const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
        ldsm_x4(a, qs + r * Q_ROW_B + ((u ^ (r & 7)) << 4));
        mma_f16(sacc[mt], a, b);
      }
    }
    // ---- the mask (per row: keys >= L + 1 + t - W, <= hi, within the tile's), the tile maxima exchanged
    float tmax[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const uint32_t r = mt * 16 + gq + 8 * (e >> 1);
        const int kl = warp * 8 + 2 * tq + (e & 1);
        const bool is_visible = kl >= klo && kl < khi && kl >= rlo + (int)(r / DQ_G);
        sacc[mt][e] = is_visible ? sacc[mt][e] * sl2 : -INFINITY;
      }
      tmax[mt][0] = fmaxf(sacc[mt][0], sacc[mt][1]);
      tmax[mt][1] = fmaxf(sacc[mt][2], sacc[mt][3]);
#pragma unroll
      for (int k = 0; k < 2; k++) {
        tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 1));
        tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 2));
      }
      if (tq == 0) xm[warp * DQ_ROWS + mt * 16 + gq] = tmax[mt][0], xm[warp * DQ_ROWS + mt * 16 + gq + 8] = tmax[mt][1];
    }
    named_sync(bar, DQ_NPRO);
    float scale[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint32_t r = mt * 16 + gq + 8 * k;
        float mx = m_run[mt][k];
#pragma unroll
        for (int w = 0; w < DQ_WARPS; w++) mx = fmaxf(mx, xm[w * DQ_ROWS + r]);
        const float mn = mx == -INFINITY ? 0.f : mx;
        scale[mt][k] = exp2f(m_run[mt][k] - mn);
        m_run[mt][k] = mx;
        const float p0 = exp2f(sacc[mt][2 * k] - mn), p1 = exp2f(sacc[mt][2 * k + 1] - mn);
        l_run[mt][k] = l_run[mt][k] * scale[mt][k] + (p0 + p1);
        const uint32_t c = warp * 8 + 2 * tq;
        asm volatile("st.shared.u32 [%0], %1;"
                     ::"r"(ps + r * P_ROW_B + ((((c >> 3) ^ (r & 7)) & 7) << 4) + (c & 7) * 2), "r"(pack_f16x2(p0, p1))
                     : "memory");
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        o[mt][nt][0] *= scale[mt][0], o[mt][nt][1] *= scale[mt][0];
        o[mt][nt][2] *= scale[mt][1], o[mt][nt][3] *= scale[mt][1];
      }
    named_sync(bar, DQ_NPRO);  // (P complete)
    // ---- O[:, 16 w ..] += P V (V bytes of keys outside [klo, khi) zeroed: their P is 0, but 0 x NaN is not)
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        const int q4 = lane >> 3, r8 = lane & 7;
        const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
        ldsm_x4(a[mt], ps + r * P_ROW_B + ((u ^ (r & 7)) << 4));
      }
      const int k0 = kk * 16 + 2 * tq;
      const uint32_t m0 = (lt_mask(k0, khi) & ~lt_mask(k0, klo)) & 0xFFu,
                     m1 = (lt_mask(k0 + 1, khi) & ~lt_mask(k0 + 1, klo)) & 0xFFu;
      const uint32_t m8 = (lt_mask(k0 + 8, khi) & ~lt_mask(k0 + 8, klo)) & 0xFFu,
                     m9 = (lt_mask(k0 + 9, khi) & ~lt_mask(k0 + 9, klo)) & 0xFFu;
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const uint32_t d = warp * 16 + nt * 8 + gq;
        uint32_t b[2];
        b[0] = e4m3x2_to_f16x2((lds_u8(box_byte(vs, k0, d)) & m0) | ((lds_u8(box_byte(vs, k0 + 1, d)) & m1) << 8));
        b[1] = e4m3x2_to_f16x2((lds_u8(box_byte(vs, k0 + 8, d)) & m8) | ((lds_u8(box_byte(vs, k0 + 9, d)) & m9) << 8));
#pragma unroll
        for (int mt = 0; mt < 2; mt++) mma_f16(o[mt][nt], a[mt], b);
      }
    }
    __syncwarp();
    named_sync(bar, DQ_NPRO);  // (every warp's reads of the stage and of P)
    if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
    cc.next(NST);
  }
  // ---- the split's partials
  float* po = reinterpret_cast<float*>(g.po) + ((size_t)h * g.n_splits + s) * DQ_ROWS * DQ_HD;
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const uint32_t r = mt * 16 + gq, d = warp * 16 + nt * 8 + 2 * tq;
      *reinterpret_cast<float2*>(po + r * DQ_HD + d) = make_float2(o[mt][nt][0], o[mt][nt][1]);
      *reinterpret_cast<float2*>(po + (r + 8) * DQ_HD + d) = make_float2(o[mt][nt][2], o[mt][nt][3]);
    }
  named_sync(bar, DQ_NPRO);
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int k = 0; k < 2; k++) {
      float l = l_run[mt][k];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      if (tq == 0) xl[warp * DQ_ROWS + mt * 16 + gq + 8 * k] = l;
    }
  named_sync(bar, DQ_NPRO);
  float* ml = reinterpret_cast<float*>(g.ml) + ((size_t)h * g.n_splits + s) * DQ_ROWS * 2;
  if (warp == 0 && tq == 0) {
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int k = 0; k < 2; k++) ml[(mt * 16 + gq + 8 * k) * 2] = tt.t0 < tt.t1 ? m_run[mt][k] : -INFINITY;
  }
  if (pt < DQ_ROWS) {
    float l = 0.f;
#pragma unroll
    for (int w = 0; w < DQ_WARPS; w++) l += xl[w * DQ_ROWS + pt];
    ml[pt * 2 + 1] = tt.t0 < tt.t1 ? l : 0.f;
  }
}

// ---- OP_D_COMBINE, item = query head qh, thread (token t = pt / 32, dims 4 (pt % 32) ..): the splits merged, the
// bf16 output quantized as o_proj's NVFP4 operand
__device__ __noinline__ void d_combine(uint32_t item, const DraftDesc& g, int pt, uint8_t* stg, int bar) {
  constexpr int LOADS = 8;                 // splits' partial outputs in flight
  constexpr int NVFP4_CHUNK = 256;        // o_proj's operand: K chunks of 256 values
  const uint32_t qh = item, h = qh / DQ_G, j = qh % DQ_G, S = g.n_splits;
  const float* ml = reinterpret_cast<const float*>(g.ml) + (size_t)h * S * DQ_ROWS * 2;
  const float* po = reinterpret_cast<const float*>(g.po) + (size_t)h * S * DQ_ROWS * DQ_HD;
  float2* wsm = reinterpret_cast<float2*>(stg);  // [8 tokens][S]
  for (uint32_t e = pt; e < DQ_QB * S; e += DQ_NPRO) {
    const uint32_t tt = e / S, s = e % S;
    wsm[e] = __ldcg(reinterpret_cast<const float2*>(ml + (s * DQ_ROWS + tt * DQ_G + j) * 2));
  }
  named_sync(bar, DQ_NPRO);
  const uint32_t t = pt / 32, d0 = (pt % 32) * 4, r = t * DQ_G + j;
  float mx = -INFINITY;
  for (uint32_t s = 0; s < S; s++) mx = fmaxf(mx, wsm[t * S + s].x);
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, l = 0.f;
#pragma unroll 1
  for (uint32_t s = 0; s < S; s += LOADS) {
    float4 v[LOADS];
#pragma unroll
    for (int u = 0; u < LOADS; u++)
      v[u] = __ldcg(reinterpret_cast<const float4*>(po + ((size_t)min(s + u, S - 1) * DQ_ROWS + r) * DQ_HD + d0));
#pragma unroll
    for (int u = 0; u < LOADS; u++) {
      if (s + u >= S) break;
      const float2 ms = wsm[t * S + s + u];
      if (ms.x == -INFINITY) continue;
      const float f = exp2f(ms.x - mx);
      l = fmaf(ms.y, f, l);
      acc[0] = fmaf(v[u].x, f, acc[0]), acc[1] = fmaf(v[u].y, f, acc[1]), acc[2] = fmaf(v[u].z, f, acc[2]),
      acc[3] = fmaf(v[u].w, f, acc[3]);
    }
  }
  const float inv = l > 0.f ? 1.0f / l : 0.f;
  float y[4];
#pragma unroll
  for (int i = 0; i < 4; i++) y[i] = bf16_round(acc[i] * inv);  // (the attention's bf16 output)
  enc_pub4<KIND_NVFP4>(reinterpret_cast<uint8_t*>(g.act), DQ_NQH * DQ_HD / NVFP4_CHUNK, qh * DQ_HD + d0, t, y, g.gs,
                       rcp_approx_ftz(g.gs), 0.f, true);
}

// ---- keys of the candidates: (order-preserving bf16 logit) << 32 | ~(FlashInfer's tie rank of the index)
__device__ __forceinline__ uint32_t dq_bf16_order(uint32_t b) { return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u); }
__device__ __forceinline__ float dq_order_value(uint32_t k) {
  const uint32_t b = (k & 0x8000u) ? (k & 0x7fffu) : (~k & 0xffffu);
  return __uint_as_float(b << 16);
}
__device__ __forceinline__ uint32_t dq_tie_rank(uint32_t id, uint32_t chunk) {
  const uint32_t c = id / chunk, l = id % chunk;
  return (c * TOPK_THREADS_FI + l % TOPK_THREADS_FI) * (chunk / TOPK_THREADS_FI + 1) + l / TOPK_THREADS_FI;
}
__device__ __forceinline__ uint32_t dq_rank_id(uint32_t rank, uint32_t chunk) {
  const uint32_t minor = chunk / TOPK_THREADS_FI + 1, hi = rank / minor, lo = rank % minor;
  return (hi / TOPK_THREADS_FI) * chunk + lo * TOPK_THREADS_FI + hi % TOPK_THREADS_FI;
}
__device__ __forceinline__ uint64_t dq_umax(uint64_t a, uint64_t b) { return a > b ? a : b; }
__device__ __forceinline__ uint64_t dq_umin(uint64_t a, uint64_t b) { return a > b ? b : a; }
__device__ __forceinline__ uint64_t dq_shfl64(uint64_t v, int o) {
  return ((uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o) << 32) | __shfl_xor_sync(0xffffffffu, (uint32_t)v, o);
}
__device__ __forceinline__ uint64_t dq_warp_max(uint64_t k) {
  const uint32_t hi = __reduce_max_sync(0xffffffffu, (uint32_t)(k >> 32));
  const uint32_t lo = __reduce_max_sync(0xffffffffu, (uint32_t)(k >> 32) == hi ? (uint32_t)k : 0u);
  return ((uint64_t)hi << 32) | lo;
}
// descending bitonic sort of one key per lane
__device__ __forceinline__ uint64_t dq_sort_desc(uint64_t c, int lane) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const uint64_t p = dq_shfl64(c, j);
      c = (((lane & j) == 0) == ((lane & k) == 0)) ? dq_umax(c, p) : dq_umin(c, p);
    }
  }
  return c;
}
// lanes 0..15 sorted (desc) list a, lanes 16..31 sorted list b: the sorted top 16 of both in lanes 0..15
__device__ __forceinline__ uint64_t dq_merge16(uint64_t c, int lane) {
  const uint64_t rev = __shfl_sync(0xffffffffu, (uint32_t)(c >> 32), 31 - lane);
  const uint64_t rev_lo = __shfl_sync(0xffffffffu, (uint32_t)c, 31 - lane);
  uint64_t x = dq_umax(c, (rev << 32) | rev_lo);
#pragma unroll
  for (int j = 8; j > 0; j >>= 1) {
    const uint64_t p = dq_shfl64(x, j);
    x = (lane & j) == 0 ? dq_umax(x, p) : dq_umin(x, p);
  }
  return x;
}

// ---- OP_D_TOPK, item (step s, vocabulary chunk c): the top 16 keys of the chunk's logits of token s + 1; each warp a
// running top 16 over its vectors (a batch of 32 keys merged only when one beats the list's 16th), then the 8 lists
// merged by warp 0
__device__ __noinline__ void d_topk(uint32_t item, const DraftDesc& g, int pt, uint8_t* stg, int bar, uint64_t* tr) {
  constexpr uint32_t TK_LISTS_B = 1024;  // the warps' lists at the staging buffer's start (8 x 16 keys)
  constexpr int TK_TR_STAGED = 0;           // trace stamps: the chunk staged,
  constexpr int TK_TR_LISTS = 1;            // the warps' lists written,
  constexpr int TK_TR_MERGED = 2;           // the item's list written,
  constexpr int TK_TR_BOUND = 3;            // warp 0's thr0,
  constexpr int TK_TR_COMPACTED = 5;        // warp 0's candidates compacted,
  constexpr int TK_TR_CANDIDATES = 6;       // and their count
  const uint32_t s = item / g.chunks, c = item % g.chunks;
  const uint32_t V = (uint32_t)rec_i32(nullptr, SR_DR + DR_VOCAB);
  const uint32_t v0 = c * g.chunk_len, v1 = min(V, v0 + g.chunk_len);
  const uint4* row = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(g.src) + (size_t)(s + 1) * V);
  const int w = pt >> 5, lane = pt & 31;
  uint64_t list = 0ull;  // lanes 0..15: the warp's sorted top 16 (0: none)
  const uint32_t nvec = (v1 - v0 + 7) / 8, vb = v0 / 8;
  // the chunk staged in smem first (after the warps' lists' 1 KB), every load in flight at once, not a round trip
  // from L2 per pass
  uint4* xs = reinterpret_cast<uint4*>(stg + TK_LISTS_B);
  constexpr uint32_t TK_STAGE = 8;  // vectors a thread stages
  const bool is_staged = nvec <= TK_STAGE * DQ_NPRO && TK_LISTS_B + nvec * 16u <= STG_BYTES;
  if (is_staged) {
    uint4 t8[TK_STAGE];
#pragma unroll
    for (uint32_t j = 0; j < TK_STAGE; j++) {
      const uint32_t vi = pt + j * DQ_NPRO;
      t8[j] = vi < nvec ? __ldcg(row + vb + vi) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (uint32_t j = 0; j < TK_STAGE; j++) {
      const uint32_t vi = pt + j * DQ_NPRO;
      if (vi < nvec) xs[vi] = t8[j];
    }
    named_sync(bar, DQ_NPRO);
  }
  if (tr && pt == 0) tr[TK_TR_STAGED] = gtimer();
  // a lower bound on the warp's 16th largest key: the 16th largest of its lanes' maxima (16 distinct lanes hold a key
  // at least that large), so only a batch with a key at or above it can enter the list: the same top 16, without the
  // sort and merge (a long chain of shuffles) for nearly every batch while the list fills up from 0
  uint64_t thr0 = 0ull;
  // (a key's tie rank, two divisions by a runtime value, only where the element's logit order reaches the running
  // bound's: the ranks order equal logits only, so the keys compared are the same)
  auto key_of = [&](uint32_t id, uint32_t hi) -> uint64_t {
    return ((uint64_t)hi << 32) | (uint32_t)~dq_tie_rank(id, g.topk_chunk);
  };
  if (is_staged) {
    // the lanes' maxima by logit order alone (no tie ranks), thr0 = (the 16th largest, rank 0): at or below the full
    // keys' bound, so the candidates are a superset holding the same top 16
    uint32_t lhi = 0u;
    for (uint32_t base = w * 32; base < nvec; base += DQ_WARPS * 32) {
      const uint32_t vi = base + lane;
      if (vi >= nvec) continue;
      const uint4 u = xs[vi];
      const uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t hi = dq_bf16_order((wd[e / 2] >> (16 * (e & 1))) & 0xffffu);
        lhi = (vb + vi) * 8 + e < v1 ? max(lhi, hi) : lhi;
      }
    }
    const uint64_t ms = dq_sort_desc((uint64_t)lhi << 32, lane);
    thr0 = (uint64_t)__shfl_sync(0xffffffffu, (uint32_t)(ms >> 32), 15) << 32;
  }
  if (tr && pt == 0) tr[TK_TR_BOUND] = gtimer();
  // the warp's keys at or above thr0 (and above 0: none) compacted into its candidate buffer, then their top 16 by a
  // sort per 32 of them; more than TK_CAP: the batch loop below. The same top 16 in the same order (the keys distinct)
  constexpr uint32_t TK_CAP = 128;
  bool use_candidates = is_staged && TK_LISTS_B + nvec * 16u + DQ_WARPS * TK_CAP * 8u <= STG_BYTES;
  if (use_candidates) {
    uint64_t* cand = reinterpret_cast<uint64_t*>(stg + TK_LISTS_B + nvec * 16) + w * TK_CAP;
    const uint32_t thi = (uint32_t)(thr0 >> 32);
    uint32_t nc = 0;
    for (uint32_t base = w * 32; base < nvec; base += DQ_WARPS * 32) {
      const uint32_t vi = base + lane;
      const bool in_range = vi < nvec;
      const uint4 u = in_range ? xs[vi] : make_uint4(0, 0, 0, 0);
      const uint32_t wd[4] = {u.x, u.y, u.z, u.w};
      // (the lane's elements whose logit order reaches thr0's: one vote per vector, the per-element compaction only
      // for the few vectors holding one)
      uint32_t reach_mask = 0u;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t id = (vb + vi) * 8 + e;
        const uint32_t hi = dq_bf16_order((wd[e / 2] >> (16 * (e & 1))) & 0xffffu);
        reach_mask |= (uint32_t)(in_range && id < v1 && hi >= thi) << e;
      }
      if (__ballot_sync(0xffffffffu, reach_mask != 0u) == 0u) continue;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t id = (vb + vi) * 8 + e;
        const uint32_t hi = dq_bf16_order((wd[e / 2] >> (16 * (e & 1))) & 0xffffu);
        const uint64_t k = (reach_mask >> e) & 1u ? key_of(id, hi) : 0ull;
        const bool is_hit = k != 0ull && k >= thr0;
        const uint32_t m = __ballot_sync(0xffffffffu, is_hit);
        const uint32_t pos = nc + __popc(m & ((1u << lane) - 1u));
        if (is_hit && pos < TK_CAP) cand[pos] = k;
        nc += __popc(m);
      }
    }
    __syncwarp();
    if (tr && pt == 0) tr[TK_TR_COMPACTED] = gtimer();
    if (tr && pt == 0) tr[TK_TR_CANDIDATES] = nc;
    if (nc <= TK_CAP) {
      for (uint32_t b = 0; b < nc; b += 32) {
        const uint64_t k = b + lane < nc ? cand[b + lane] : 0ull;
        const uint64_t sorted_key = dq_sort_desc(k, lane);
        const uint64_t batch_top = ((uint64_t)__shfl_sync(0xffffffffu, (uint32_t)(sorted_key >> 32), lane & 15) << 32) |
                                   __shfl_sync(0xffffffffu, (uint32_t)sorted_key, lane & 15);
        list = dq_merge16(lane < 16 ? list : batch_top, lane);
      }
    } else {
      use_candidates = false;
    }
  }
  for (uint32_t base = w * 32; !use_candidates && base < nvec; base += DQ_WARPS * 32) {
    const uint32_t vi = base + lane;
    uint4 u = make_uint4(0, 0, 0, 0);
    const bool in_range = vi < nvec;
    if (in_range) u = is_staged ? xs[vi] : __ldcg(row + vb + vi);
    const uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const uint32_t id = (vb + vi) * 8 + e;
      const uint32_t bits = (wd[e / 2] >> (16 * (e & 1))) & 0xffffu;
      const uint64_t k =
          in_range && id < v1 ? ((uint64_t)dq_bf16_order(bits) << 32) | (uint32_t)~dq_tie_rank(id, g.topk_chunk) : 0ull;
      const uint64_t thr = ((uint64_t)__shfl_sync(0xffffffffu, (uint32_t)(list >> 32), 15) << 32) |
                           __shfl_sync(0xffffffffu, (uint32_t)list, 15);
      if (__ballot_sync(0xffffffffu, k > thr && k >= thr0) == 0u) continue;
      const uint64_t sorted_key = dq_sort_desc(k, lane);
      // lanes 16..31 take the batch's top 16 (sorted), lanes 0..15 keep the list
      const uint64_t batch_top = ((uint64_t)__shfl_sync(0xffffffffu, (uint32_t)(sorted_key >> 32), lane & 15) << 32) |
                                 __shfl_sync(0xffffffffu, (uint32_t)sorted_key, lane & 15);
      list = dq_merge16(lane < 16 ? list : batch_top, lane);
    }
  }
  uint64_t* lists = reinterpret_cast<uint64_t*>(stg);
  if (lane < 16) lists[w * 16 + lane] = list;
  named_sync(bar, DQ_NPRO);
  if (tr && pt == 0) tr[TK_TR_LISTS] = gtimer();
  if (w == 0) {
    uint64_t acc = lane < 16 ? lists[lane] : 0ull;
    for (int k = 1; k < DQ_WARPS; k++) {
      const uint64_t next_list = lists[k * 16 + (lane & 15)];
      acc = dq_merge16(lane < 16 ? acc : next_list, lane);
    }
    if (lane < 16) reinterpret_cast<uint64_t*>(g.parts)[((size_t)s * g.chunks + c) * DQ_TOPK + lane] = acc;
    if (tr && lane == 0) tr[TK_TR_MERGED] = gtimer();
  }
}

// ---- OP_D_SELECT (SM 0's prologue warps): the chunks' lists merged per step, the selector's edge scores, the walk
// (Gumbel-max keyed by candidate id), the draft tokens and the sparse draft logits
constexpr int DQ_SEL_CHUNKS = 24;  // D_TOPK's chunks per step the selector stages (draft.py TOPK_CHUNKS)
struct DqSelSmem {
  int64_t cand[DQ_NSTEP * DQ_TOPK];
  int64_t cached_cand[DQ_NSTEP * DQ_TOPK];
  float unary[DQ_NSTEP * DQ_TOPK];
  float noise[DQ_NSTEP * DQ_TOPK];
  float scores[DQ_NSTEP * DQ_TOPK * DQ_TOPK];
  int32_t best[DQ_NSTEP * DQ_TOPK];
  int32_t path[DQ_NSTEP];
  uint64_t lists[DQ_NSTEP * DQ_SEL_CHUNKS * DQ_TOPK];  // D_TOPK's chunks' lists, loaded at once
};
static_assert(sizeof(DqSelSmem) <= STG_BYTES, "the selector's staging");
__device__ __noinline__ void d_select(uint32_t item, const DraftDesc& g, int pt, uint8_t* stg, int bar, uint64_t* tr) {
  constexpr int SEL_TR_MERGED = 0;  // trace stamps: the chunks' lists merged,
  constexpr int SEL_TR_SCORES = 1;  // the scores,
  constexpr int SEL_TR_PATH = 2;    // the path,
  constexpr int SEL_TR_LOADED = 3;  // the lists and the draft state loaded
  (void)item;
  DqSelSmem& ss = *reinterpret_cast<DqSelSmem*>(stg);
  const int w = pt >> 5, lane = pt & 31;
  const uint32_t ns = (uint32_t)rec_i32(nullptr, SR_DR + DR_NSTEP), nr = ns * DQ_TOPK;
  // the chunks' lists of step w merged: lane l holds the head of list l; 16 rounds of the warp's largest head. Every
  // list staged in smem first, all loads at once, issued before the draft state's dependent loads (they run under them)
  constexpr uint32_t SEL_PER = (DQ_NSTEP * DQ_SEL_CHUNKS * DQ_TOPK + DQ_NPRO - 1) / DQ_NPRO;
  const uint64_t* src = reinterpret_cast<const uint64_t*>(g.src);
  const uint32_t n = ns * min(g.chunks, (uint32_t)DQ_SEL_CHUNKS) * DQ_TOPK;
  uint64_t lv[SEL_PER];
#pragma unroll
  for (uint32_t j = 0; j < SEL_PER; j++) {
    const uint32_t e = pt + j * DQ_NPRO;
    lv[j] = e < n ? __ldcg(src + e) : 0ull;
  }
  const DraftState st = draft_state();
  const size_t V = (size_t)rec_i32(nullptr, SR_DR + DR_VOCAB);
  int64_t* cached = reinterpret_cast<int64_t*>(rec_u64(SR_DR + DR_CAND));
  if (pt < (int)nr) ss.cached_cand[pt] = cached[pt];
  const float temp = *reinterpret_cast<const volatile float*>(rec_u64(SR_DR + DR_TEMP));
  const uint64_t seed = (uint64_t)*reinterpret_cast<const volatile int64_t*>(rec_u64(SR_DR + DR_SEED));
#pragma unroll
  for (uint32_t j = 0; j < SEL_PER; j++) {
    const uint32_t e = pt + j * DQ_NPRO;
    if (e < n) ss.lists[e] = lv[j];
  }
  named_sync(bar, DQ_NPRO);
  if (tr && pt == 0) tr[SEL_TR_LOADED] = gtimer();
  if (w < (int)ns) {
    const uint64_t* lists = ss.lists + (size_t)w * g.chunks * DQ_TOPK;
    uint64_t head = lane < (int)g.chunks ? lists[(size_t)lane * DQ_TOPK] : 0ull;
    uint64_t lane_key = 0ull;  // the lane's rank of the merged top 16
    uint32_t head_pos = 0;
    // the 16 rounds of the warp's largest head (each a dependent load and two reductions)
#pragma unroll 1
    for (int r = 0; r < DQ_TOPK; r++) {
      const uint64_t mx = dq_warp_max(head);
      if (lane == r) lane_key = mx;
      if (head == mx && mx != 0ull) {
        head_pos++;
        head = head_pos < DQ_TOPK ? lists[(size_t)lane * DQ_TOPK + head_pos] : 0ull;
      }
    }
    if (lane < DQ_TOPK) {
      ss.cand[w * DQ_TOPK + lane] = dq_rank_id(~(uint32_t)lane_key, g.topk_chunk);
      ss.unary[w * DQ_TOPK + lane] = dq_order_value((uint32_t)(lane_key >> 32));
    }
  }
  named_sync(bar, DQ_NPRO);
  if (tr && pt == 0) tr[SEL_TR_MERGED] = gtimer();
  // the walk's noise (independent of the path)
  if (pt < (int)nr) ss.noise[pt] = ss_gumbel(philox_randint(seed, (uint64_t)(st.L + 1 + pt / DQ_TOPK)), ss.cand[pt]);
  // the scores of step w: P = bf16(pred * h) (h: the hidden projection of token w + 1), bf16(P succ^T) (fp32 sums),
  // plus the unary logits in fp32; A and B fragments straight from the codebooks (L2 / host)
  if (w < (int)ns) {
    const uint32_t* hs = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(g.hsel) + (size_t)(w + 1) * DQ_RANK);
    const int gq = lane >> 2, tq = lane & 3;
    const int64_t p0 = w == 0 ? st.bonus : ss.cand[(w - 1) * DQ_TOPK + gq];
    const int64_t p1 = w == 0 ? st.bonus : ss.cand[(w - 1) * DQ_TOPK + gq + 8];
    const uint32_t* pr0 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(g.pred_cb) + (size_t)p0 * DQ_RANK);
    const uint32_t* pr1 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(g.pred_cb) + (size_t)p1 * DQ_RANK);
    const uint32_t* sc0 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(g.succ_cb) + (size_t)ss.cand[w * DQ_TOPK + gq] * DQ_RANK);
    const uint32_t* sc1 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(g.succ_cb) + (size_t)ss.cand[w * DQ_TOPK + 8 + gq] * DQ_RANK);
    float d[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    auto pmul = [](uint32_t p, uint32_t h) { return pack_bf16x2(__fmul_rn(bf16lo(p), bf16lo(h)), __fmul_rn(bf16hi(p), bf16hi(h))); };
#pragma unroll 4
    for (int ks = 0; ks < DQ_RANK / 16; ks++) {
      const int k0 = ks * 8 + tq, k1 = k0 + 4;  // (u32 words: dims 16 ks + 2 tq, + 8)
      const uint32_t h0 = __ldg(hs + k0), h1 = __ldg(hs + k1);
      uint32_t a[4];
      a[0] = pmul(__ldg(pr0 + k0), h0), a[1] = pmul(__ldg(pr1 + k0), h0);
      a[2] = pmul(__ldg(pr0 + k1), h1), a[3] = pmul(__ldg(pr1 + k1), h1);
      const uint32_t b0[2] = {__ldg(sc0 + k0), __ldg(sc0 + k1)}, b1[2] = {__ldg(sc1 + k0), __ldg(sc1 + k1)};
      mma_bf16(d[0], a, b0);
      mma_bf16(d[1], a, b1);
    }
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int p = gq + (e >= 2 ? 8 : 0), c = nt * 8 + 2 * tq + (e & 1);
        ss.scores[(w * DQ_TOPK + p) * DQ_TOPK + c] = __fadd_rn(ss.unary[w * DQ_TOPK + c], bf16_round(d[nt][e]));
      }
  }
  named_sync(bar, DQ_NPRO);
  if (tr && pt == 0) tr[SEL_TR_SCORES] = gtimer();
  // every (step, previous pick) row's Gumbel-max (the lowest index of equal maxima), then the path by lookups
  if (pt < (int)nr) {
    const float* row = ss.scores + pt * DQ_TOPK;
    const float* nz = ss.noise + (pt / DQ_TOPK) * DQ_TOPK;
    float best_score = -INFINITY;
    int best_col = 0;
    for (int c = 0; c < DQ_TOPK; c++) {
      const float x = temp != 0.0f ? __fadd_rn(div_full(row[c], temp), nz[c]) : row[c];
      if (x > best_score) best_score = x, best_col = c;
    }
    ss.best[pt] = best_col;
  }
  named_sync(bar, DQ_NPRO);
  if (pt == 0) {
    int prev_pick = 0;
    for (uint32_t s = 0; s < ns; s++) ss.path[s] = prev_pick = ss.best[s * DQ_TOPK + prev_pick];
  }
  named_sync(bar, DQ_NPRO);
  if (tr && pt == 0) tr[SEL_TR_PATH] = gtimer();
  // the outputs: the previous candidates' draft logits cleared first (an id can be old and new), then the realized rows
  // at the new candidates, the cached candidates, the selector's rows, the drafts
  float* dl = reinterpret_cast<float*>(rec_u64(SR_DR + DR_LOGITS));
  if (pt < (int)nr) dl[(size_t)(pt / DQ_TOPK) * V + ss.cached_cand[pt]] = -INFINITY;
  named_sync(bar, DQ_NPRO);
  if (pt < (int)nr) {
    const int s = pt / DQ_TOPK, prev_pick = s == 0 ? 0 : ss.path[s - 1];
    const float rv = ss.scores[(s * DQ_TOPK + prev_pick) * DQ_TOPK + pt % DQ_TOPK];
    dl[(size_t)s * V + ss.cand[pt]] = rv;
    cached[pt] = ss.cand[pt];
    reinterpret_cast<float*>(rec_u64(SR_DR + DR_SCORES))[pt] = rv;
  }
  if (pt < (int)ns)
    reinterpret_cast<int64_t*>(rec_u64(SR_DR + DR_TOKENS))[pt] = ss.cand[pt * DQ_TOPK + ss.path[pt]];
}

}  // namespace mk
