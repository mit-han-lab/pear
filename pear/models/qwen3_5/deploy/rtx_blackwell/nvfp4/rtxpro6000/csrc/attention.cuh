// Full attention of a verify step of T <= 8 tokens on sm_120a (Qwen3.5: 24 query heads, 4 KV heads of 256, q/k
// GemmaRMSNorm, M-RoPE on the first 64 dims, sigmoid output gate) over vLLM's KV cache, memory NHD [kernel
// blocks][token][4 heads][k 256 | v 256]: BF16 in FLASH_ATTN's layout (the kernel block is the manager block; 1024 bytes
// a (token, head)), or with MK_KV_FP8 FP8 e4m3 (scale 1) in FlashInfer's (kernel blocks of 16 tokens, a manager block's
// consecutive; 512 bytes a (token, head)). Either way a manager block's tokens are rows of one [tokens][4][row] view.
//
// OP_ATTN_PREP, item = (KV head h, token t): the new token's k (k_norm, RoPE) and v into the cache, its 6 query heads' q
//   (q_norm, RoPE) into Q [h][48 rows][256] (row r = 6 t + j: token t, head 6 h + j). Numerics: x bf16, (x rsqrt(sum
//   x^2 / 256 + eps)) (1 + w) in fp32, rotate-half with the token's bf16 cos / sin, bf16 out; the cache holds the rows
//   as vLLM's reshape_and_cache_flash writes them (FP8: e4m3 by cvt.rn.satfinite of the bf16 values; Q in f16).
// OP_ATTN, item = (h, split s): split-KV flash attention over the split's tiles of 64 keys (tiles never cross a manager
//   block); a tile holding the step's new tokens waits for OP_ATTN_PREP. The producer streams a tile through the weight
//   ring as 4 stages (BF16: K dims 0..127, K 128..255, V 0..127, V 128..255) or 2 (FP8: K, V), each two TMA boxes
//   [64 keys][128 B] (128-byte swizzle). The 8 prologue warps: S = Q K^T by mma.sync (BF16: bf16, warp w keys 8 w ..
//   8 w + 7 of the tile; FP8: f16 with the e4m3 bytes converted exactly), the causal mask, an online softmax in base 2,
//   P in smem, O += P V; at the end the split's unnormalized O (fp32) and its rows' m, l into the partials.
// OP_ATTN_COMBINE, item = (query head, quarter of its dims) (96 items): the splits merged, bf16 (the attention's output),
//   times sigmoid(gate) (bf16), quantized for o_proj: e4m3(clamp(y / s_in)) into its published operand [K chunk][NT][128].
#pragma once
#include "common.cuh"
#include "../../common/csrc/step.cuh"

namespace mk {

constexpr int ATT_HQ = 24, ATT_HKV = 4, ATT_D = 256, ATT_G = ATT_HQ / ATT_HKV;
constexpr int ATT_ROWS = ATT_G * 8;                    // query rows per KV head (6 heads x 8 tokens)
constexpr uint32_t ATT_AUX_QSELF = 1u;                  // (OP_ATTN's Instr aux: the item computes its Q itself)
constexpr int ATT_KOFF = ATT_HQ * 2 * ATT_D;           // 12288: k in the qkv projection (q | gate per head before it)
constexpr int ATT_VOFF = ATT_KOFF + ATT_HKV * ATT_D;   // 13312
constexpr int ATT_QKV = ATT_VOFF + ATT_HKV * ATT_D;    // 14336
constexpr int ATT_OUT = ATT_HQ * ATT_D;                // 6144: o_proj's K
constexpr int ATT_TILE = 64;                           // keys per tile
constexpr int ATT_BOX_ROW_B = 128;                     // a TMA box's row: 128 bytes (SWIZZLE_128B)
constexpr int ATT_BOX = ATT_TILE * ATT_BOX_ROW_B;      // one TMA box: 64 keys x 128 bytes (8 KB)
#if MK_KV_FP8
constexpr int ATT_KV_B = 1;                            // a cache element's bytes: e4m3
constexpr int ATT_PARTS = 2;                           // ring stages per tile: K, V (2 boxes each)
#else
constexpr int ATT_KV_B = 2;                            // a cache element's bytes: bf16
constexpr int ATT_PARTS = 4;                           // ring stages per tile: K lo, K hi, V lo, V hi (2 boxes each)
#endif
constexpr int ATT_HEAD_ROW_B = 2 * ATT_D * ATT_KV_B;   // a KV head's cache row: k 256 | v 256 elements
constexpr int ATT_V_OFF_B = ATT_D * ATT_KV_B;          // v in a KV head's cache row (bytes)
constexpr int ATT_PART_B = 2 * ATT_BOX_ROW_B;          // a ring stage's bytes of a cache row: its two boxes'
constexpr uint32_t ATT_PREP_KEY = 6;                   // OP_ATTN_PREP's row j after the 6 query heads': the key
constexpr int ATT_WARPS = 8;                           // the warps of an OP_ATTN item
constexpr uint32_t ATT_U = ATT_TILE;                   // a split's unit of keys
constexpr uint32_t ATT_UPT = ATT_TILE / ATT_U;         // units per tile
#if MK_KV_FP8
// stage s of an item of n tiles in the ring order K0, K1, V0, K2, V1, .., K(n - 1), V(n - 2), V(n - 1) (K one tile
// ahead): its tile (from the item's first) and part (0 K, 1 V)
__device__ __forceinline__ void attn_swp_stage(uint32_t s, uint32_t n, uint32_t& ti, uint32_t& part) {
  if (s == 0) {
    ti = 0, part = 0;
  } else if (s & 1u) {
    const uint32_t k = (s + 1) / 2;
    ti = k < n ? k : n - 1, part = k < n ? 0u : 1u;
  } else {
    ti = s / 2 - 1, part = 1;
  }
}
#endif
constexpr int ATT_MAX_SPLITS = 48;
static_assert(ATT_MAX_SPLITS - ATT_MAX_SPLITS / 2 <= 32, "the combine's split mask (32 bits a half)");
constexpr int NPRO_ATT = 256;                          // the prologue threads that run OP_ATTN
// the item's staging buffer (stg): Q [48][256] (rows swizzled: 16-byte unit u of row r at u ^ (r & 7)), P [48][64] (same
// swizzle over its 8 units), both bf16 (FP8: f16), the row maxima exchange [8 warps][48] fp32; at the end the rows' l
// partials [8][48] in P's place
constexpr uint32_t ATT_Q_OFF = 0, ATT_P_OFF = ATT_ROWS * ATT_D * 2, ATT_XM_OFF = ATT_P_OFF + ATT_ROWS * ATT_TILE * 2;
constexpr uint32_t ATT_XL_OFF = ATT_P_OFF, ATT_STG_BYTES = ATT_XM_OFF + ATT_WARPS * ATT_ROWS * 4;
static_assert(ATT_STG_BYTES <= STG_BYTES, "the attention's staging");

struct AttnDesc {       // an OpDesc's 256 bytes (schedule.py ATTN)
  uint64_t kv;          // the layer's cache (uint8): [tokens rows][4 heads][k | v] (512 B each bf16, 256 B each e4m3)
  uint64_t tmap;        // CUtensorMap 3D over kv: {a head's row bytes, 4 heads, rows}, box {128 B, 1, 64}, SWIZZLE_128B
  uint64_t qkv;         // the qkv projection's bf16 output [NT][ATT_QKV]
  uint64_t qn_w, kn_w;  // q_norm / k_norm weights (bf16 [256], applied as 1 + w)
  uint64_t cos_sin;     // bf16 [positions][64]: cos 0..31, sin 32..63
  uint64_t q;           // Q bf16 (the FP8 cache's builds: f16) [4][48][256] (OP_ATTN_PREP's output)
  uint64_t po;          // the splits' partial O fp32 [4][S][48][256]
  uint64_t ml;          // the splits' m, l fp32 [4][S][48][2]
  uint64_t act;         // o_proj's FP8 operand: codes [ATT_OUT / 128][NT][128], fp32 scales [ATT_OUT / 128][SFB_CHUNK / 4]
  float scale_log2, eps, inv_s, f3;
  uint32_t n_splits;    // splits per KV head
  uint32_t block_tokens;  // vLLM's manager block (tokens)
  uint32_t group;       // the layer's attention KV cache group (its block-table row: rec_att_bt)
  uint32_t prep_ctr;    // OP_ATTN: OP_ATTN_PREP's counter (the tile with the new tokens waits for it)
  uint32_t sec1, sec2;  // M-RoPE interleaving: section h at i % 3 == 1, i < 3 * sec1; w at i % 3 == 2, i < 3 * sec2
  uint32_t kb_tokens;   // tokens per kernel block: the block-table row holds kernel-block ids (V2: a manager block of
                        // block_tokens is block_tokens / kb_tokens consecutive ids), row of position p = id * kb + p % kb
  uint32_t book_ctr;    // OP_BOOK's counter (the step's positions; NO_CTR: none): the pacer's one wait
  uint32_t pad[32];
};
static_assert(sizeof(AttnDesc) == 256, "AttnDesc layout");

// ---- the split's tiles: every manager block's tiles of 64 keys (the last one shorter), split s of n_splits takes
// tiles [n * s / S, n * (s + 1) / S); L = the step's first position + T keys
struct AttTiles {
  uint32_t t0, t1;  // tile range
  uint32_t per_mb;  // tiles per manager block
};
__device__ __forceinline__ AttTiles attn_tiles(const AttnDesc& g, uint32_t L, uint32_t s) {
  const uint32_t B = g.block_tokens, per = (B + ATT_TILE - 1) / ATT_TILE;
  const uint32_t nmb = (L + B - 1) / B;
  // tiles that hold keys < L: the full manager blocks' and the last one's first ceil((L - (nmb - 1) B) / 64)
  const uint32_t n = (nmb - 1) * per + (L - (nmb - 1) * B + ATT_TILE - 1) / ATT_TILE;
  return AttTiles{(uint32_t)((uint64_t)n * s / g.n_splits), (uint32_t)((uint64_t)n * (s + 1) / g.n_splits), per};
}
// the item's tiles
__device__ __forceinline__ uint32_t attn_ntiles(const AttTiles& at) { return (at.t1 - at.t0 + ATT_UPT - 1) / ATT_UPT; }
// the cache row of position p (the block table's kernel-block ids)
__device__ __forceinline__ uint32_t attn_row(const AttnDesc& g, const int32_t* bt, uint32_t p) {
  return (uint32_t)__ldcg(bt + p / g.kb_tokens) * g.kb_tokens + p % g.kb_tokens;
}
// tile i's first key position and its row (a tile's keys: consecutive kernel blocks of one manager block, so
// consecutive rows)
__device__ __forceinline__ void attn_tile(const AttnDesc& g, const int32_t* bt, uint32_t i, uint32_t per, uint32_t& key0,
                                          uint32_t& row0) {
  const uint32_t m = i / per, j = i - m * per;
  key0 = m * g.block_tokens + j * ATT_TILE;
  row0 = attn_row(g, bt, key0);
}
__device__ __forceinline__ uint32_t attn_L() { return (uint32_t)(rec_pos(nullptr) + rec_i32(nullptr, SR_NTOK)); }

// ---- producer: the item's tiles as ring stages of two TMA boxes [64 keys][128 B] (128-byte swizzle) each: 4 a tile
// (BF16: K dims 0..127, K 128..255, V 0..127, V 128..255) or 2 (FP8: K, V, in attn_swp_stage's order), each behind the
// pacer's gate (pace_w: kernel.cuh role_pacer); the stages count as weight stages (empty: 2 arrivals)
template <int NST>
__device__ __forceinline__ void produce_attn(const Instr& in, const AttnDesc& g, uint8_t* stages, uint32_t stage_bytes,
                                             uint64_t* full, uint64_t* empty, RingPos& cc, const uint32_t* counters,
                                             uint32_t prep_target, int lane, uint32_t* pace_w) {
  const uint32_t h = in.r0 / g.n_splits, s = in.r0 % g.n_splits;
  const uint32_t L = attn_L(), P = (uint32_t)rec_pos(nullptr);
  const AttTiles at = attn_tiles(g, L, s);
  const int32_t* bt = rec_att_bt(nullptr, (int)g.group);
  const void* tmap = reinterpret_cast<const void*>(g.tmap);
  bool prep_seen = false;
#if MK_KV_FP8
  // (the stages in attn_swp_stage's order, K one tile ahead; each stage as below)
#pragma unroll 1
  for (uint32_t sx = 0, ns = 2 * (at.t1 - at.t0); sx < ns; sx++) {
    uint32_t ti, part;
    attn_swp_stage(sx, at.t1 - at.t0, ti, part);
    uint32_t key0, row0;
    attn_tile(g, bt, at.t0 + ti, at.per_mb, key0, row0);
#else
  for (uint32_t i = at.t0; i < at.t1; i++) {
    uint32_t key0, row0;
    attn_tile(g, bt, i, at.per_mb, key0, row0);
#endif
    if (!prep_seen && key0 + ATT_TILE > P) {  // the tile holds a new token: its k, v written by OP_ATTN_PREP first
      if (lane == 0) wait_counter(counters + g.prep_ctr, prep_target);
      __syncwarp();
      prep_seen = true;
    }
#if MK_KV_FP8
    {
#else
#pragma unroll 1
    for (int part = 0; part < ATT_PARTS; part++) {  // K dims 0..127, K 128..255, V 0..127, V 128..255
#endif
      if (cc.n >= NST) mbar_wait(&empty[cc.slot], cc.phase ^ 1);
      if (lane == 0) {
        {  // (the gate: this SM's g-th paced stage once the pacer's probes of it landed, as produce_gemv's)
          volatile uint32_t* pw = pace_w;
          const uint32_t gs = pw[1];
          while (pw[0] <= gs) {
          }
          pw[1] = gs + 1;
        }
        uint64_t* bar = &full[cc.slot];
        mbar_expect_tx(bar, 2 * ATT_BOX);
        uint8_t* dst = stages + cc.slot * stage_bytes;
        const uint64_t pol = l2_evict_first();
        asm volatile("fence.proxy.async.global;" ::: "memory");  // (the prep's generic stores of the new rows)
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
            ::"r"(smem_u32(dst)), "l"(tmap), "r"((int)part * ATT_PART_B), "r"((int)h), "r"((int)row0),
              "r"(smem_u32(bar)), "l"(pol)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
            ::"r"(smem_u32(dst + ATT_BOX)), "l"(tmap), "r"((int)part * ATT_PART_B + ATT_BOX_ROW_B), "r"((int)h),
              "r"((int)row0), "r"(smem_u32(bar)), "l"(pol)
            : "memory");
      }
      __syncwarp();
      cc.next(NST);
    }
  }
}
// the ring stages an attention instruction takes (every role advances its ring position by them)
__device__ __forceinline__ uint32_t attn_stages(const Instr& in, const AttnDesc& g) {
  const AttTiles at = attn_tiles(g, attn_L(), in.r0 % g.n_splits);
  return ATT_PARTS * attn_ntiles(at);
}

// ---- helpers
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16x2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
// two e4m3 bytes (low byte -> low half) as f16x2 (exact)
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two) {
  uint32_t r;
  asm("{\n .reg .b16 lo;\n cvt.u16.u32 lo, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(two));
  return r;
}
#if MK_KV_FP8
// the FP8 paths' P for P V (the target's item and the draft's): rounded to bf16 as vLLM's XQA decode stores it, then
// held in f16 (exact for P >= 2^-14 of its row's maximum), V and the mma f16
__device__ __forceinline__ uint32_t pack_p_f16x2(float a, float b) {
  const float2 f = __bfloat1622float2(__floats2bfloat162_rn(a, b));
  return pack_f16x2(f.x, f.y);
}
#endif
// D += A B: f16 x f16 -> f32, m16n8k16
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// a byte of a swizzled [rows][128 B] box (16-byte unit u of row r at u ^ (r & 7))
__device__ __forceinline__ uint32_t box_byte(uint32_t base, uint32_t r, uint32_t c) {
  return base + r * 128 + ((((c >> 4) ^ (r & 7)) & 7) << 4) + (c & 15);
}
__device__ __forceinline__ uint32_t lds_u16(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
// two 8x8 b16 matrices transposed (rows from lanes 0..7 and 8..15): the B fragment of m16n8k16 from k-major rows
__device__ __forceinline__ void ldsm_x2_trans(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ uint32_t lds_u8(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

// ---- OP_ATTN_PREP: the new tokens' k (k_norm, RoPE) and v into the cache (e4m3, saturating: vLLM's conversion), their
// q (q_norm, RoPE, bf16 -> f16: exact) into Q [h][48 rows][256] (row r = 6 t + j: token t, head 6 h + j)

// the rows' norm and M-RoPE in parts (OP_ATTN_PREP's, and OP_ATTN's own Q: the same code, the same bits): token t's
// rotary (cos, sin) of lane < 8's 8 values, the same for every head of the token
__device__ __forceinline__ void attn_rope_cs(const AttnDesc& g, uint32_t t, int lane, int32_t P, int32_t host_p0,
                                             float (&cv)[8], float (&sv)[8]) {
  constexpr int ROPE_ROW = 64;  // cos_sin's row: cos 0..31, then sin
  constexpr int ROPE_SIN = 32;  // sin's offset in the row
  const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(g.cos_sin);
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = 0.f, sv[i] = 0.f;
  if (lane < 8) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t k = (lane & 3) * 8 + i;  // rotary index 0..31
      const int sec = (k % 3 == 1 && k < 3 * g.sec1) ? 1 : (k % 3 == 2 && k < 3 * g.sec2) ? 2 : 0;
      const size_t pos = (size_t)(P + (rec_i32(nullptr, SR_MPOS + sec) - host_p0) + (int)t);
      cv[i] = __bfloat162float(cs[pos * ROPE_ROW + k]), sv[i] = __bfloat162float(cs[pos * ROPE_ROW + ROPE_SIN + k]);
    }
  }
}
// a row's 8 values of lane l (dims 8 l ..: the bf16 projection xv, the norm's weight wv) as the compiled qk_norm_rope:
// (x rsqrt(sum x^2 / 256 + eps)) (1 + w) in fp32, the rotate-half of its first 64 dims (lanes < 8) with the token's
// bf16 cos / sin, bf16-rounded (a warp, every lane)
__device__ __forceinline__ void attn_norm_rope_row(const AttnDesc& g, uint4 xv, uint4 wv, const float (&cv)[8],
                                                   const float (&sv)[8], int lane, float (&y)[8]) {
  constexpr float HEAD_DIMS = 256.0f;  // the norm's mean is over a head's dims
  const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w};
  float x[8];
#pragma unroll
  for (int i = 0; i < 4; i++) x[2 * i] = bf16lo(xu[i]), x[2 * i + 1] = bf16hi(xu[i]);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(x[i], x[i], ss);
  ss = warp_sum(ss);
  const float rs = rsqrt_tri(ss / HEAD_DIMS + g.eps);
  const uint32_t wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    y[2 * i] = (x[2 * i] * rs) * (bf16lo(wu[i]) + 1.0f);
    y[2 * i + 1] = (x[2 * i + 1] * rs) * (bf16hi(wu[i]) + 1.0f);
  }
  float p[8];
#pragma unroll
  for (int i = 0; i < 8; i++) p[i] = __shfl_xor_sync(0xffffffffu, y[i], 4);
  if (lane < 8) {
#pragma unroll
    for (int i = 0; i < 8; i++) y[i] = lane < 4 ? y[i] * cv[i] - p[i] * sv[i] : y[i] * cv[i] + p[i] * sv[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) y[i] = bf16_round(y[i]);
}
// one query head's (j < 6) or the key's (j == 6) row of token t, KV head h (a warp)
__device__ __forceinline__ void attn_norm_rope(const AttnDesc& g, uint32_t h, uint32_t j, uint32_t t, int lane, int32_t P,
                                               int32_t host_p0, float (&y)[8]) {
  const __nv_bfloat16* qkv = reinterpret_cast<const __nv_bfloat16*>(g.qkv);
  const bool is_key = j == ATT_PREP_KEY;
  const uint32_t off = is_key ? ATT_KOFF + h * ATT_D : (h * ATT_G + j) * 2 * ATT_D;
  const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(qkv + (size_t)t * ATT_QKV + off) + lane);
  const uint4 wv = __ldg(reinterpret_cast<const uint4*>(is_key ? g.kn_w : g.qn_w) + lane);
  float cv[8], sv[8];
  attn_rope_cs(g, t, lane, P, host_p0, cv, sv);
  attn_norm_rope_row(g, xv, wv, cv, sv, lane, y);
}

__device__ __forceinline__ void attn_prep(const Instr& in, const AttnDesc& g, int pt, int nthreads) {
  // item (KV head h, token t1): warps 0..5 its 6 query heads, warp 6 its key, warp 7 its value row (32 items a layer:
  // the work of a token per SM, no warp serial over the step's tokens)
  const uint32_t h = in.r0 / NT, t1 = in.r0 % NT, T = (uint32_t)rec_i32(nullptr, SR_NTOK);
  if (t1 >= T) return;
  const int32_t P = rec_pos(nullptr), host_p0 = rec_i32(nullptr, SR_POS);
  const __nv_bfloat16* qkv = reinterpret_cast<const __nv_bfloat16*>(g.qkv);
  const int32_t* bt = rec_att_bt(nullptr, (int)g.group);
  uint8_t* kv = reinterpret_cast<uint8_t*>(g.kv);
  constexpr uint32_t PREP_JOBS = 7;  // the 6 query heads and the key
  const int warp = pt >> 5, lane = pt & 31, nwarps = nthreads >> 5;
  for (uint32_t job = warp; job < PREP_JOBS; job += nwarps) {
    const uint32_t t = t1, j = job;  // j < 6: query head 6 h + j; 6: the key
    const bool is_key = j == ATT_PREP_KEY;
    float y[8];
    attn_norm_rope(g, h, j, t, lane, P, host_p0, y);
#if MK_KV_FP8
    if (is_key) {  // the key's e4m3 row in the cache: (row, h)'s first 256 bytes
      const size_t row = attn_row(g, bt, (uint32_t)P + t);
      uint2 c;
      c.x = f32x2_to_e4m3x2(y[0], y[1]) | (f32x2_to_e4m3x2(y[2], y[3]) << 16);
      c.y = f32x2_to_e4m3x2(y[4], y[5]) | (f32x2_to_e4m3x2(y[6], y[7]) << 16);
      *reinterpret_cast<uint2*>(kv + (row * ATT_HKV + h) * ATT_HEAD_ROW_B + lane * 8) = c;
    } else {  // the query row 6 t + j, f16 (exact from bf16)
      uint4 o;
      o.x = pack_f16x2(y[0], y[1]), o.y = pack_f16x2(y[2], y[3]), o.z = pack_f16x2(y[4], y[5]), o.w = pack_f16x2(y[6], y[7]);
      reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(g.q) + ((size_t)h * ATT_ROWS + t * ATT_G + j) * ATT_D * 2)[lane] = o;
    }
#else
    uint4 o;  // (bf16 of the normed, rotated row: the y values are bf16-rounded already)
    o.x = pack_bf16x2(y[0], y[1]), o.y = pack_bf16x2(y[2], y[3]), o.z = pack_bf16x2(y[4], y[5]), o.w = pack_bf16x2(y[6], y[7]);
    if (is_key) {  // the key's bf16 row in the cache: (row, h)'s first 512 bytes
      const size_t row = attn_row(g, bt, (uint32_t)P + t);
      reinterpret_cast<uint4*>(kv + (row * ATT_HKV + h) * ATT_HEAD_ROW_B)[lane] = o;
    } else {  // the query row 6 t + j
      reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(g.q) + ((size_t)h * ATT_ROWS + t * ATT_G + j) * ATT_D * 2)[lane] = o;
    }
#endif
  }
  // the value row: the bf16 projection as it is, or (FP8) its e4m3 (thread per 8 values), the row's second half
  if (warp == nwarps - 1) {
    const uint32_t t = t1, l = lane;
    const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(qkv + (size_t)t * ATT_QKV + ATT_VOFF + h * ATT_D) + l);
#if MK_KV_FP8
    const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w};
    uint2 c;
    c.x = f32x2_to_e4m3x2(bf16lo(xu[0]), bf16hi(xu[0])) | (f32x2_to_e4m3x2(bf16lo(xu[1]), bf16hi(xu[1])) << 16);
    c.y = f32x2_to_e4m3x2(bf16lo(xu[2]), bf16hi(xu[2])) | (f32x2_to_e4m3x2(bf16lo(xu[3]), bf16hi(xu[3])) << 16);
    const size_t row = attn_row(g, bt, (uint32_t)P + t);
    *reinterpret_cast<uint2*>(kv + (row * ATT_HKV + h) * ATT_HEAD_ROW_B + ATT_V_OFF_B + l * 8) = c;
#else
    const size_t row = attn_row(g, bt, (uint32_t)P + t);
    reinterpret_cast<uint4*>(kv + (row * ATT_HKV + h) * ATT_HEAD_ROW_B + ATT_V_OFF_B)[l] = xv;
#endif
  }
}

#if MK_KV_FP8
// ---- the FP8 item's phases of a tile, software-pipelined by attn_item (inlined: the softmax and the previous tile's
// P V one basic block). S for keys 8 w .. 8 w + 7 of the K stage at ks, all 48 rows
__device__ __forceinline__ void swp_s(float (&sacc)[3][4], uint32_t ks, uint32_t qs, int warp, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int a = 0; a < 3; a++) sacc[a][0] = sacc[a][1] = sacc[a][2] = sacc[a][3] = 0.f;
  const uint32_t key = warp * 8 + gq;  // this lane's B column (key) for the k-steps
#pragma unroll 4
  for (int kk = 0; kk < 16; kk++) {  // dims 16 kk ..: box kk / 8
    const uint32_t box = ks + (kk >> 3) * ATT_BOX;
    const uint32_t d0 = (kk & 7) * 16 + 2 * tq;
    uint32_t b[2];
    b[0] = e4m3x2_to_f16x2(lds_u16(box_byte(box, key, d0)));
    b[1] = e4m3x2_to_f16x2(lds_u16(box_byte(box, key, d0 + 8)));
#pragma unroll
    for (int mt = 0; mt < 3; mt++) {
      uint32_t a[4];
      const int q4 = lane >> 3, r8 = lane & 7;
      const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
      ldsm_x4(a, qs + r * 512 + ((u ^ (r & 7)) << 4));
      mma_f16(sacc[mt], a, b);
    }
  }
}
// the mask and the tile maxima into xm ([row][warp])
__device__ __forceinline__ void swp_mask_max(float (&sacc)[3][4], float* xm, uint32_t key0, uint32_t tile_len, uint32_t T,
                                             uint32_t P, uint32_t L, float sl2, int warp, int lane) {
  // (the lane through an opaque copy: its rows' tokens r / 6 are recomputed a tile, not hoisted out of the prologue's
  // loop and held (spilled) across every op's code)
  int lo;
  asm volatile("mov.b32 %0, %1;" : "=r"(lo) : "r"(lane));
  const int gq = lo >> 2, tq = lo & 3;
  float tmax[3][2];
#pragma unroll
  for (int mt = 0; mt < 3; mt++) {
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const uint32_t r = mt * 16 + gq + 8 * (e >> 1), kp = key0 + warp * 8 + 2 * tq + (e & 1);
      const uint32_t tok = r / ATT_G;
      const bool vis = tok < T && kp <= P + tok && kp < L && kp - key0 < tile_len;
      sacc[mt][e] = vis ? sacc[mt][e] * sl2 : -INFINITY;
    }
    tmax[mt][0] = fmaxf(sacc[mt][0], sacc[mt][1]);
    tmax[mt][1] = fmaxf(sacc[mt][2], sacc[mt][3]);
#pragma unroll
    for (int k = 0; k < 2; k++) {
      tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 1));
      tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 2));
    }
    if (tq == 0) xm[(mt * 16 + gq) * 8 + warp] = tmax[mt][0], xm[(mt * 16 + gq + 8) * 8 + warp] = tmax[mt][1];
  }
}
// the softmax of the lane's rows: the running maxima, the rescale factors, P and l; no store:
// P goes out after the previous tile's P V (swp_p_store)
__device__ __forceinline__ void swp_softmax(const float (&sacc)[3][4], const float* xm, float (&m_run)[3][2],
                                            float (&l_run)[3][2], float (&scale)[3][2], uint32_t (&pp)[3][2], int lane) {
  const int gq = lane >> 2;
#pragma unroll
  for (int mt = 0; mt < 3; mt++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      const uint32_t r = mt * 16 + gq + 8 * k;
      float mx = m_run[mt][k];
      {
        const float4 xa = lds_f4(smem_u32(xm) + r * 32), xb = lds_f4(smem_u32(xm) + r * 32 + 16);
        mx = fmaxf(mx, fmaxf(fmaxf(fmaxf(xa.x, xa.y), fmaxf(xa.z, xa.w)), fmaxf(fmaxf(xb.x, xb.y), fmaxf(xb.z, xb.w))));
      }
      const float mn = mx == -INFINITY ? 0.f : mx;  // (a row with no visible key yet: no NaN)
      scale[mt][k] = exp2f(m_run[mt][k] - mn);
      m_run[mt][k] = mx;
      const float p0 = exp2f(sacc[mt][2 * k] - mn), p1 = exp2f(sacc[mt][2 * k + 1] - mn);
      l_run[mt][k] = l_run[mt][k] * scale[mt][k] + (p0 + p1);
      pp[mt][k] = pack_p_f16x2(p0, p1);
    }
  }
}
// P of the lane's rows and keys into the buffer at pb ([48][64], 16-byte units swizzled by row)
__device__ __forceinline__ void swp_p_store(const uint32_t (&pp)[3][2], uint32_t pb, int warp, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 3; mt++)
#pragma unroll
    for (int k = 0; k < 2; k++) {
      const uint32_t r = mt * 16 + gq + 8 * k, c = warp * 8 + 2 * tq;  // (row, key column)
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(pb + r * 128 + ((((c >> 3) ^ (r & 7)) & 7) << 4) + (c & 7) * 2),
                   "r"(pp[mt][k]) : "memory");
    }
}
// the O rescale (rows of this lane)
__device__ __forceinline__ void swp_rescale(float (&o)[3][4][4], const float (&scale)[3][2]) {
#pragma unroll
  for (int mt = 0; mt < 3; mt++)
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      o[mt][nt][0] *= scale[mt][0], o[mt][nt][1] *= scale[mt][0];
      o[mt][nt][2] *= scale[mt][1], o[mt][nt][3] *= scale[mt][1];
    }
}
// O[:, 32 w ..] += P V: P from the buffer at pb, V from the stage at vs, its rows from vlen on zeroed
__device__ __forceinline__ void swp_pv(float (&o)[3][4][4], uint32_t pb, uint32_t vs, uint32_t vlen, int warp, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {  // keys 16 kk ..
    uint32_t a[3][4];
#pragma unroll
    for (int mt = 0; mt < 3; mt++) {
      const int q4 = lane >> 3, r8 = lane & 7;
      const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
      ldsm_x4(a[mt], pb + r * 128 + ((u ^ (r & 7)) << 4));
    }
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const uint32_t d = warp * 32 + nt * 8 + gq;  // the lane's B column (dim)
      const uint32_t box = vs + (d >> 7) * ATT_BOX, dc = d & 127;
      const uint32_t k0 = kk * 16 + 2 * tq;
      const uint32_t m0 = lt_mask((int)k0, (int)vlen) & 0xFFu, m1 = lt_mask((int)k0 + 1, (int)vlen) & 0xFFu;
      const uint32_t m8 = lt_mask((int)k0 + 8, (int)vlen) & 0xFFu, m9 = lt_mask((int)k0 + 9, (int)vlen) & 0xFFu;
      uint32_t b[2];
      b[0] = e4m3x2_to_f16x2((lds_u8(box_byte(box, k0, dc)) & m0) | ((lds_u8(box_byte(box, k0 + 1, dc)) & m1) << 8));
      b[1] = e4m3x2_to_f16x2((lds_u8(box_byte(box, k0 + 8, dc)) & m8) | ((lds_u8(box_byte(box, k0 + 9, dc)) & m9) << 8));
#pragma unroll
      for (int mt = 0; mt < 3; mt++) mma_f16(o[mt][nt], a[mt], b);
    }
  }
}
#endif
// ---- OP_ATTN, item (KV head h, split s), the prologue warps' part (cc: the ring position of its first stage): S = Q
// K^T, the causal mask, an online softmax in base 2, P in smem, O += P V, the split's partials. vb (FP8): the B slot,
// free during the item, P's second buffer
template <int NST>
__device__ void attn_item(const Instr& in, const AttnDesc& g, uint8_t* stages, uint32_t stage_bytes, uint64_t* full,
                          uint64_t* empty, RingPos& cc, uint8_t* stg, int pt, int bar, uint8_t* vb = nullptr) {
  constexpr uint32_t Q_ROW_B = 512;  // a Q row in smem: 256 f16 (BF16: bf16)
  constexpr uint32_t Q_UNITS = 32;   // its 16-byte units
  constexpr uint32_t P_ROW_B = 128;  // a P row in smem: 64 f16 (BF16: bf16)
  const uint32_t h = in.r0 / g.n_splits, s = in.r0 % g.n_splits;
  const uint32_t L = attn_L(), P = (uint32_t)rec_pos(nullptr), T = (uint32_t)rec_i32(nullptr, SR_NTOK);
  const AttTiles at = attn_tiles(g, L, s);
  const int32_t* bt = rec_att_bt(nullptr, (int)g.group);
  const int warp = pt >> 5, lane = pt & 31, gq = lane >> 2, tq = lane & 3;
  const uint32_t qs = smem_u32(stg + ATT_Q_OFF), ps = smem_u32(stg + ATT_P_OFF);
  float* xm = reinterpret_cast<float*>(stg + ATT_XM_OFF);
  float* xl = reinterpret_cast<float*>(stg + ATT_XL_OFF);
  // Q into smem (rows swizzled by 16-byte units: conflict-free ldmatrix). Instr aux ATT_AUX_QSELF: computed here from
  // the qkv projection (warp w: token w's 6 query rows, OP_ATTN_PREP's code: the same bits), so the item waits for the
  // projection, not for OP_ATTN_PREP (whose K / V rows only the tiles holding the new tokens wait for, in the producer)
  if (in.aux & ATT_AUX_QSELF) {
    const uint32_t t = warp;
    const int32_t host_p0 = rec_i32(nullptr, SR_POS);
    static_assert(NPRO_ATT / 32 == NT, "a warp a token");
    // (every load first: the token's 6 rows, the norm's weight, its rotary values once for the 6 heads)
    const __nv_bfloat16* qkv = reinterpret_cast<const __nv_bfloat16*>(g.qkv);
    uint4 xv[ATT_G];
#pragma unroll
    for (uint32_t j = 0; j < ATT_G; j++)
      xv[j] = t < T ? __ldcg(reinterpret_cast<const uint4*>(qkv + (size_t)t * ATT_QKV + (h * ATT_G + j) * 2 * ATT_D) + lane)
                    : make_uint4(0u, 0u, 0u, 0u);
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(g.qn_w) + lane);
    float cv[8], sv[8];
    attn_rope_cs(g, t < T ? t : 0u, lane, (int32_t)P, host_p0, cv, sv);
#pragma unroll
    for (uint32_t j = 0; j < ATT_G; j++) {
      const uint32_t r = t * ATT_G + j;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (t < T) {
        float y[8];
        attn_norm_rope_row(g, xv[j], wv, cv, sv, lane, y);
#if MK_KV_FP8
        o.x = pack_f16x2(y[0], y[1]), o.y = pack_f16x2(y[2], y[3]), o.z = pack_f16x2(y[4], y[5]), o.w = pack_f16x2(y[6], y[7]);
#else
        o.x = pack_bf16x2(y[0], y[1]), o.y = pack_bf16x2(y[2], y[3]), o.z = pack_bf16x2(y[4], y[5]), o.w = pack_bf16x2(y[6], y[7]);
#endif
      }
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(qs + r * Q_ROW_B + ((lane ^ (r & 7)) << 4)),
                   "r"(o.x), "r"(o.y), "r"(o.z), "r"(o.w) : "memory");
    }
  } else {
    const uint4* q = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(g.q) + (size_t)h * ATT_ROWS * ATT_D * 2);
    for (uint32_t e = pt; e < ATT_ROWS * Q_UNITS; e += NPRO_ATT) {
      const uint32_t r = e / Q_UNITS, u = e % Q_UNITS;
      const uint4 v = __ldcg(q + e);
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(qs + r * Q_ROW_B + ((u ^ (r & 7)) << 4)),
                   "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  }
  named_sync(bar, NPRO_ATT);
  // per warp: O[48 rows][its dims] (3 m-tiles x 4 n-tiles; FP8: dims 32 w .. 32 w + 31; BF16: dims 128 vh + 16 w ..
  // + 15, vh = 0, 1, n-tile 2 vh + n), the rows' running maxima (rows of the lane: 16 mt + gq (+8)), its keys' l partials
  float o[3][4][4];
#pragma unroll
  for (int a = 0; a < 3; a++)
#pragma unroll
    for (int b = 0; b < 4; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  float m_run[3][2], l_run[3][2];
#pragma unroll
  for (int a = 0; a < 3; a++) m_run[a][0] = m_run[a][1] = -INFINITY, l_run[a][0] = l_run[a][1] = 0.f;
  const float sl2 = g.scale_log2;
#if MK_KV_FP8
  // (the ring's stages K0, K1, V0, K2, V1, .. (produce_attn); per tile i: S(i) from its K stage, the mask and the
  // maxima, B0 (the K stage's reads, the maxima exchanged), the K slot released; then the previous tile's V stage, i's
  // softmax, i - 1's P V (P(i - 1) from its buffer), i's P into the other buffer and the O rescale, B1 (P(i) complete;
  // the V stage's and P(i - 1)'s reads), the V slot released. Tile j of the item has its P in buffer j & 1: the staging
  // buffer's (ps) or the B slot (vb, its last MMA reads waited for by the caller). pcur: this tile's P buffer, pprev:
  // the previous tile's, swapped a tile; pvlen: the previous tile's V rows, >= 1, so 0 before the item's first tile)
  uint32_t pcur = ps, pprev = smem_u32(vb), pvlen = 0;
#pragma unroll 1
#endif
  for (uint32_t i = at.t0; i < at.t1; i++) {
    uint32_t key0, row0;
    attn_tile(g, bt, i, at.per_mb, key0, row0);
    (void)row0;
#if MK_KV_FP8
    mbar_wait(&full[cc.slot], cc.phase);
    float sacc[3][4];
    swp_s(sacc, smem_u32(stages + cc.slot * stage_bytes), qs, warp, lane);
    const uint32_t tile_len = min((uint32_t)ATT_TILE, g.block_tokens - key0 % g.block_tokens);
    swp_mask_max(sacc, xm, key0, tile_len, T, P, L, sl2, warp, lane);
    __syncwarp();
    named_sync(bar, NPRO_ATT);  // B0
    if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
    cc.next(NST);
    float scale[3][2];
    uint32_t pp[3][2];
    if (pvlen == 0) {  // (the item's first tile)
      swp_softmax(sacc, xm, m_run, l_run, scale, pp, lane);
      swp_p_store(pp, pcur, warp, lane);
      swp_rescale(o, scale);
    } else {
      mbar_wait(&full[cc.slot], cc.phase);  // (V of tile i - 1)
      // (P(i) stored before i - 1's P V: only the 6 rescale factors live across it (with the stores after it the
      // prologue spilled the item's bounds); ptxas sinks the factors' exp2 into P V's MMAs)
      swp_softmax(sacc, xm, m_run, l_run, scale, pp, lane);
      swp_p_store(pp, pcur, warp, lane);
      swp_pv(o, pprev, smem_u32(stages + cc.slot * stage_bytes), pvlen, warp, lane);
      swp_rescale(o, scale);
    }
    __syncwarp();
    named_sync(bar, NPRO_ATT);  // B1
    if (pvlen != 0) {
      if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
      cc.next(NST);
    }
    pvlen = min(tile_len, L - key0);  // (>= 1: the item's tiles hold keys < L)
    const uint32_t pt_ = pcur;
    pcur = pprev, pprev = pt_;
  }
  if (pvlen != 0) {  // the last tile's P V (pprev: its P buffer)
    mbar_wait(&full[cc.slot], cc.phase);
    swp_pv(o, pprev, smem_u32(stages + cc.slot * stage_bytes), pvlen, warp, lane);
    __syncwarp();
    named_sync(bar, NPRO_ATT);  // (every warp's reads of the V stage and of P)
    if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
    cc.next(NST);
#else
    // ---- the two K stages (dims 0..127, 128..255): S for keys 8 w .. 8 w + 7, bf16 mma over 16 k-steps
    float sacc[3][4];
#pragma unroll
    for (int a = 0; a < 3; a++) sacc[a][0] = sacc[a][1] = sacc[a][2] = sacc[a][3] = 0.f;
    const uint32_t key = warp * 8 + gq;  // this lane's B column (key)
#pragma unroll 1
    for (int kh = 0; kh < 2; kh++) {
      mbar_wait(&full[cc.slot], cc.phase);
      const uint32_t ks = smem_u32(stages + cc.slot * stage_bytes);
      // this warp's K fragments of the stage into registers first, the slot released, then the MMAs: the slot is
      // held for the loads only, not for the tensor work (the next TMA fill starts that much sooner)
      uint32_t bk[8][2];
#pragma unroll
      for (int k8 = 0; k8 < 8; k8++) {  // dims 128 kh + 16 k8 ..: box k8 / 4, bytes 32 (k8 % 4) + 4 tq (+ 16: dims + 8)
        const uint32_t box = ks + (k8 >> 2) * ATT_BOX;
        const uint32_t c0 = (k8 & 3) * 32 + 4 * tq;
        bk[k8][0] = lds_u32(box_byte(box, key, c0));
        bk[k8][1] = lds_u32(box_byte(box, key, c0 + 16));
      }
      named_sync(bar, NPRO_ATT);  // (every warp's reads of the K stage)
      if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
      cc.next(NST);
#pragma unroll
      for (int k8 = 0; k8 < 8; k8++) {
        const int kk = kh * 8 + k8;
#pragma unroll
        for (int mt = 0; mt < 3; mt++) {
          uint32_t a[4];
          const int q4 = lane >> 3, r8 = lane & 7;
          const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
          ldsm_x4(a, qs + r * Q_ROW_B + ((u ^ (r & 7)) << 4));
          mma_bf16(sacc[mt], a, bk[k8]);
        }
      }
    }
    // ---- the mask, the tile maxima (rows 16 mt + gq, + 8; keys 8 w + 2 tq, + 1), exchanged. A manager block's last
    // tile holds fewer keys (block_tokens % 64): its box rows past them belong to other blocks, masked
    const uint32_t tile_len = min((uint32_t)ATT_TILE, g.block_tokens - key0 % g.block_tokens);
    float tmax[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; mt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const uint32_t r = mt * 16 + gq + 8 * (e >> 1), kp = key0 + warp * 8 + 2 * tq + (e & 1);
        const uint32_t tok = r / ATT_G;
        const bool is_visible = tok < T && kp <= P + tok && kp < L && kp - key0 < tile_len;
        sacc[mt][e] = is_visible ? sacc[mt][e] * sl2 : -INFINITY;
      }
      tmax[mt][0] = fmaxf(sacc[mt][0], sacc[mt][1]);
      tmax[mt][1] = fmaxf(sacc[mt][2], sacc[mt][3]);
#pragma unroll
      for (int k = 0; k < 2; k++) {
        tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 1));
        tmax[mt][k] = fmaxf(tmax[mt][k], __shfl_xor_sync(0xffffffffu, tmax[mt][k], 2));
      }
      if (tq == 0) xm[warp * ATT_ROWS + mt * 16 + gq] = tmax[mt][0], xm[warp * ATT_ROWS + mt * 16 + gq + 8] = tmax[mt][1];
    }
    named_sync(bar, NPRO_ATT);
    float scale[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; mt++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint32_t r = mt * 16 + gq + 8 * k;
        float mx = m_run[mt][k];
#pragma unroll
        for (int w = 0; w < ATT_WARPS; w++) mx = fmaxf(mx, xm[w * ATT_ROWS + r]);
        const float mn = mx == -INFINITY ? 0.f : mx;  // (a row with no visible key yet: no NaN)
        scale[mt][k] = exp2f(m_run[mt][k] - mn);
        m_run[mt][k] = mx;
        // P = exp2(s - m) of this warp's keys, bf16 (FlashAttention 2's P V operand), into P [48][64]; l of the fp32 P
        const float p0 = exp2f(sacc[mt][2 * k] - mn), p1 = exp2f(sacc[mt][2 * k + 1] - mn);
        l_run[mt][k] = l_run[mt][k] * scale[mt][k] + (p0 + p1);
        const uint32_t c = warp * 8 + 2 * tq;  // key column
        asm volatile("st.shared.u32 [%0], %1;"
                     ::"r"(ps + r * P_ROW_B + ((((c >> 3) ^ (r & 7)) & 7) << 4) + (c & 7) * 2), "r"(pack_bf16x2(p0, p1))
                     : "memory");
      }
    }
    // the O rescale (rows of this lane)
#pragma unroll
    for (int mt = 0; mt < 3; mt++)
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        o[mt][nt][0] *= scale[mt][0], o[mt][nt][1] *= scale[mt][0];
        o[mt][nt][2] *= scale[mt][1], o[mt][nt][3] *= scale[mt][1];
      }
    named_sync(bar, NPRO_ATT);  // (P complete)
    // ---- the two V stages (dims 0..127, 128..255): every warp w, O[:, 128 vh + 16 w ..] += P V. Its V fragments
    // (ldmatrix.trans) into registers, the slot released, then the MMAs. The box rows past the tile's keys (a manager
    // block's last tile: the next physical block's memory, another group's page; the context's last tile: rows not
    // written yet) can hold any bytes, bf16 NaN too: their P is 0 but 0 x NaN is not, so those keys' values are zeroed
    // in the fragments (lane: keys 16 kk + 2 tq, + 1 in b[0], + 8, + 9 in b[1])
    const uint32_t vlen = min(tile_len, L - key0);  // (>= 1: the item's tiles hold keys < L)
#pragma unroll  // (o's index 2 vh + nt: a runtime vh would put o in local memory)
    for (int vh = 0; vh < 2; vh++) {
      mbar_wait(&full[cc.slot], cc.phase);
      const uint32_t vs = smem_u32(stages + cc.slot * stage_bytes);
      uint32_t bv[4][2][2];
#pragma unroll
      for (int kk = 0; kk < 4; kk++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
          // B (keys x dims): ldmatrix.x2.trans of keys 16 kk .. 16 kk + 15 at the n-tile's 8 dims (one 16-byte unit)
          const uint32_t d0 = warp * 16 + nt * 8;  // dims within the stage's half
          const uint32_t kr = kk * 16 + (lane & 15), u = (d0 & 63) >> 3;
          ldsm_x2_trans(bv[kk][nt], vs + (d0 >> 6) * ATT_BOX + kr * 128 + ((u ^ (kr & 7)) << 4));
        }
      named_sync(bar, NPRO_ATT);  // (every warp's reads of the V stage)
      if (pt == 0) mbar_arrive(&empty[cc.slot]), mbar_arrive(&empty[cc.slot]);
      cc.next(NST);
      if (vlen < (uint32_t)ATT_TILE) {  // (uniform)
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          const int k0 = kk * 16 + 2 * tq;
          const uint32_t m0 = (k0 < (int)vlen ? 0x0000ffffu : 0u) | (k0 + 1 < (int)vlen ? 0xffff0000u : 0u);
          const uint32_t m1 = (k0 + 8 < (int)vlen ? 0x0000ffffu : 0u) | (k0 + 9 < (int)vlen ? 0xffff0000u : 0u);
#pragma unroll
          for (int nt = 0; nt < 2; nt++) bv[kk][nt][0] &= m0, bv[kk][nt][1] &= m1;
        }
      }
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {  // keys 16 kk ..
#pragma unroll
        for (int mt = 0; mt < 3; mt++) {
          uint32_t a[4];
          const int q4 = lane >> 3, r8 = lane & 7;
          const uint32_t r = mt * 16 + r8 + 8 * (q4 & 1), u = 2 * kk + (q4 >> 1);
          ldsm_x4(a, ps + r * P_ROW_B + ((u ^ (r & 7)) << 4));
#pragma unroll
          for (int nt = 0; nt < 2; nt++) mma_bf16(o[mt][2 * vh + nt], a, bv[kk][nt]);
        }
      }
    }
    // (P's next writer, the next tile's softmax, runs after that tile's K stage barriers: every warp's reads of P here
    // come first in its program order)
#endif
  }
  // ---- the split's partials: O (unnormalized, fp32), and the rows' m (base 2) and l (the warps' l partials summed)
  float* po = reinterpret_cast<float*>(g.po) + ((size_t)h * g.n_splits + s) * ATT_ROWS * ATT_D;
#pragma unroll
  for (int mt = 0; mt < 3; mt++)
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#if MK_KV_FP8
      const uint32_t r = mt * 16 + gq, d = warp * 32 + nt * 8 + 2 * tq;
#else
      const uint32_t r = mt * 16 + gq, d = (nt >> 1) * 128 + warp * 16 + (nt & 1) * 8 + 2 * tq;  // (o[mt][2 vh + n])
#endif
      *reinterpret_cast<float2*>(po + r * ATT_D + d) = make_float2(o[mt][nt][0], o[mt][nt][1]);
      *reinterpret_cast<float2*>(po + (r + 8) * ATT_D + d) = make_float2(o[mt][nt][2], o[mt][nt][3]);
    }
  named_sync(bar, NPRO_ATT);  // (every warp past its last use of P: the l partials take its place)
#pragma unroll
  for (int mt = 0; mt < 3; mt++)
#pragma unroll
    for (int k = 0; k < 2; k++) {
      float l = l_run[mt][k];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      if (tq == 0) xl[warp * ATT_ROWS + mt * 16 + gq + 8 * k] = l;
    }
  named_sync(bar, NPRO_ATT);
  // the running max of each row from the warp that holds it (lanes tq == 0 of every warp hold the same values)
  float* ml = reinterpret_cast<float*>(g.ml) + ((size_t)h * g.n_splits + s) * ATT_ROWS * 2;
  if (warp == 0 && tq == 0) {
#pragma unroll
    for (int mt = 0; mt < 3; mt++)
#pragma unroll
      for (int k = 0; k < 2; k++) ml[(mt * 16 + gq + 8 * k) * 2] = at.t0 < at.t1 ? m_run[mt][k] : -INFINITY;
  }
  if (pt < ATT_ROWS) {
    float l = 0.f;
#pragma unroll
    for (int w = 0; w < ATT_WARPS; w++) l += xl[w * ATT_ROWS + pt];
    ml[pt * 2 + 1] = at.t0 < at.t1 ? l : 0.f;
  }
}

// ---- OP_ATTN_COMBINE, item (query head qh, quarter c of its dims), a thread per (half of the splits, token, 4 dims):
// the splits merged, the bf16 output times sigmoid(gate), quantized for o_proj (e4m3(clamp(y / s_in))) into its
// published operand [K chunk][NT][128]
__device__ __forceinline__ void attn_combine(const Instr& in, const AttnDesc& g, int pt, uint8_t* stg, int bar) {
  constexpr uint32_t QUARTERS = 4;       // the items of a query head
  constexpr uint32_t QUARTER_DIMS = 64;  // an item's dims
  constexpr int HALF_THREADS = 128;       // the threads of half the splits
  constexpr uint32_t TOKEN_THREADS = 16;  // a token's threads in a half (4 dims each)
  constexpr uint32_t ITEM_ROWS = 8;       // the item's rows: a token each
  constexpr int LOADS = 8;                // splits' partial outputs in flight
  constexpr uint32_t NO_SPLIT = 32u;      // su's sentinel: no split left (0..31: a half's splits)
  constexpr float E4M3_MAX = 448.f;       // the largest e4m3 magnitude
  constexpr uint32_t OPERAND_CHUNK = 128;  // o_proj's operand: K chunks of 128 values (a byte each)
  const uint32_t qh = in.r0 / QUARTERS, c = in.r0 % QUARTERS, h = qh / ATT_G, j = qh % ATT_G;
  const uint32_t T = (uint32_t)rec_i32(nullptr, SR_NTOK), S = g.n_splits;
  const float* ml = reinterpret_cast<const float*>(g.ml) + (size_t)h * S * ATT_ROWS * 2;
  const float* po = reinterpret_cast<const float*>(g.po) + (size_t)h * S * ATT_ROWS * ATT_D;
  // the item's 8 rows' (m, l) of every split into smem (all threads, loads at once), then per thread (half, token,
  // 4 dims) its half of the splits; the halves merged through smem
  float2* wsm = reinterpret_cast<float2*>(stg);  // [8 rows][S]
  float4* hsm = reinterpret_cast<float4*>(stg + ITEM_ROWS * S * sizeof(float2));  // [128]: half 1's sums, then its l
  float* hl = reinterpret_cast<float*>(hsm + HALF_THREADS);
  const uint32_t half = pt / HALF_THREADS, q = pt % HALF_THREADS, t = q / TOKEN_THREADS,
                 d0 = c * QUARTER_DIMS + (q % TOKEN_THREADS) * 4;
  const uint32_t r = t * ATT_G + j;
  // (half 0's gate values with the (m, l) loads: not a round trip of their own after the merge)
  const __nv_bfloat16* gate = reinterpret_cast<const __nv_bfloat16*>(g.qkv) + (size_t)t * ATT_QKV + qh * 2 * ATT_D + ATT_D + d0;
  uint2 gv = make_uint2(0u, 0u);
  if (!half) gv = __ldcg(reinterpret_cast<const uint2*>(gate));
  for (uint32_t e = pt; e < ITEM_ROWS * S; e += NPRO_ATT) {
    const uint32_t tt = e / S, s = e % S;
    wsm[e] = __ldcg(reinterpret_cast<const float2*>(ml + (s * ATT_ROWS + tt * ATT_G + j) * 2));
  }
  named_sync(bar, NPRO_ATT);
  float mx = -INFINITY;
  for (uint32_t s = 0; s < S; s++) mx = fmaxf(mx, wsm[t * S + s].x);
  const uint32_t s0 = half * (S / 2), s1 = half ? S : S / 2;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, l = 0.f;
  // only the half's splits with keys (m > -inf, by a mask from smem) loaded, 8 in flight at once, unconditional from
  // valid addresses: an empty split's stale partials never count; the rest added in ascending order
  uint32_t pending_splits = 0u;
  for (uint32_t s = s0; s < s1; s++) pending_splits |= (wsm[t * S + s].x != -INFINITY ? 1u : 0u) << (s - s0);
#pragma unroll 1
  while (pending_splits) {
    uint32_t su[LOADS];
#pragma unroll
    for (int u = 0; u < LOADS; u++) {
      su[u] = pending_splits ? (uint32_t)__ffs(pending_splits) - 1u : NO_SPLIT;
      pending_splits &= pending_splits - 1u;
    }
    float4 v[LOADS];
#pragma unroll
    for (int u = 0; u < LOADS; u++)
      v[u] = __ldcg(
          reinterpret_cast<const float4*>(po + ((size_t)(s0 + min(su[u], s1 - 1 - s0)) * ATT_ROWS + r) * ATT_D + d0));
#pragma unroll
    for (int u = 0; u < LOADS; u++) {
      if (su[u] == NO_SPLIT) break;
      const float2 ms = wsm[t * S + s0 + su[u]];
      const float f = exp2f(ms.x - mx);
      l = fmaf(ms.y, f, l);
      acc[0] = fmaf(v[u].x, f, acc[0]), acc[1] = fmaf(v[u].y, f, acc[1]), acc[2] = fmaf(v[u].z, f, acc[2]),
      acc[3] = fmaf(v[u].w, f, acc[3]);
    }
  }
  if (half) hsm[q] = make_float4(acc[0], acc[1], acc[2], acc[3]), hl[q] = l;
  named_sync(bar, NPRO_ATT);
  if (half) return;
  {
    const float4 o = hsm[q];
    acc[0] += o.x, acc[1] += o.y, acc[2] += o.z, acc[3] += o.w, l += hl[q];
  }
  const float inv = l > 0.f ? 1.0f / l : 0.f;
  const float inv_s = div_full(1.0f, g.inv_s);  // (inv_s holds o_proj's input scale s_in)
  const __nv_bfloat16* gq = reinterpret_cast<const __nv_bfloat16*>(&gv);
  uint32_t code = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    // the attention's bf16 output times sigmoid(gate), not rounded before o_proj's quantization (vLLM's compiled step)
    const float o = bf16_round(acc[i] * inv);
    const float gt = __bfloat162float(gq[i]);
    const float q = fminf(fmaxf((o * sigmoid_triton(gt)) * inv_s, -E4M3_MAX), E4M3_MAX);
    code |= (f32x2_to_e4m3x2(q, 0.f) & 0xffu) << (8 * i);
  }
  const uint32_t k = qh * ATT_D + d0;  // o_proj's input index
  uint8_t* act = reinterpret_cast<uint8_t*>(g.act) + (size_t)(k / OPERAND_CHUNK) * NT * OPERAND_CHUNK +
                 pub_byte(t, k % OPERAND_CHUNK);
  *reinterpret_cast<uint32_t*>(act) = t < T ? code : 0u;
}

}  // namespace mk
