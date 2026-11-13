// The GEMV B operands in bf16, built by the worker warps into the SM's B buffer once per op: per 64-value K chunk a
// tile [n rows][128 B] (SW128) of the bf16 values of vLLM's compiled draft (Inductor's arithmetic); worker warp w
// builds row w of every chunk.
#pragma once
#include "dk_gemv.cuh"
#include "../../../common/csrc/dk_finish.cuh"

namespace dk {

// what the worker warps keep per op (static shared memory)
struct WorkerConst {
  float rstd[NB];      // per B row: the RMSNorm's 1 / rms of its x
  float rstd_emb[QB];  // the query block's embedding rows (layer 0's input norm), from OP_PREP
  int32_t ids[QB];     // the query block's token ids
};

constexpr uint32_t STG_ROW = CHUNK_K * 2;  // bytes per staged bf16 row of a chunk
constexpr uint4 ZERO4 = {0u, 0u, 0u, 0u};

// chunk position i's tile [n rows][128 B] (SW128)
struct ChunkOut {
  uint8_t* tile;
  __device__ __forceinline__ ChunkOut(const GemvDesc& d, const Smem& sm, uint32_t i) : tile(sm.bbuf + d.b_off + i * (d.n * CHUNK)) {}
  // row t's group g (8 bf16) into the chunk
  __device__ __forceinline__ void store(uint32_t t, uint32_t g, uint4 q) const { sts128(tile + sw128(t, g), q); }
  __device__ __forceinline__ void put(uint32_t t, uint32_t g, const float (&v)[GV]) const { store(t, g, pack_bf16x8(v)); }
  __device__ __forceinline__ uint8_t* at(uint32_t t, uint32_t g) const { return tile + sw128(t, g); }
};

// the inputs of one (chunk, row)'s group for a lane: x (fp32 in u[0..1], bf16 in u[0]) and the norm weight's (bf16)
struct RowIn {
  uint4 u[2];
  uint4 w;
};

// the RMSNorm of 8 values: bf16((x * r) * w) per value (two products, one rounding), as fp32 (emb: x is the embedding
// row's bf16)
__device__ __forceinline__ void norm8(const RowIn& in, bool emb, float r, float (&v)[GV]) {
  float xv[GV], wv[GV];
  if (emb) {
    bf16x8(in.u[0], xv);
  } else {
#pragma unroll
    for (int q = 0; q < 2; q++) {
      xv[4 * q] = __uint_as_float(in.u[q].x), xv[4 * q + 1] = __uint_as_float(in.u[q].y);
      xv[4 * q + 2] = __uint_as_float(in.u[q].z), xv[4 * q + 3] = __uint_as_float(in.u[q].w);
    }
  }
  bf16x8(in.w, wv);
#pragma unroll
  for (int e = 0; e < GV; e++) v[e] = bf16_round(__fmul_rn(__fmul_rn(xv[e], r), wv[e]));
}
// the 1 / rms of token t of an op's normed input (the embedding rows' for layer 0)
__device__ __forceinline__ float row_rstd(bool emb, const WorkerConst& wc, int t) { return emb ? wc.rstd_emb[t] : wc.rstd[t]; }

// x's 8 values of token t from K index k: the fp32 residual stream, or the query block's embedding row (layer 0)
__device__ __forceinline__ void load_x8(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t t, uint32_t k,
                                        RowIn& in) {
  if (d.xemb) {
    in.u[0] = __ldg(reinterpret_cast<const uint4*>(a.tab.embed + (size_t)wc.ids[t] * HID + k));
  } else {
    const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const float*>(d.b[BT_X]) + (size_t)t * HID + k);
    in.u[0] = __ldcg(p), in.u[1] = __ldcg(p + 1);
  }
}

// the B source's token of B row t (B_FINAL: rows 0..6 are tokens 1..7), or -1 for a zero row
template <int SRC>
__device__ __forceinline__ int row_token(const DkArgs& a, int t) {
  if constexpr (SRC == B_FINAL) return t + 1 < QB ? t + 1 : -1;
  if constexpr (SRC == B_AUX) return (uint32_t)t < a.step.n_ctx ? t : -1;
  return t;
}

// B_AUX / B_COPY: row t's group g of chunk kc straight into its place in the tile (cp.async; a zero row stored)
template <int SRC>
__device__ __forceinline__ void copy_group(const GemvDesc& d, const DkArgs& a, uint32_t kc, int t, uint32_t g, uint8_t* dst) {
  const uint32_t k = kc * CHUNK_K + GV * g;
  const int tok = row_token<SRC>(a, t);
  if (tok < 0) {
    sts128(dst, ZERO4);
  } else if constexpr (SRC == B_AUX) {
    cp_async16(dst, a.step.aux[k / HID] + (size_t)tok * a.step.aux_stride + k % HID);
  } else {
    cp_async16(dst, reinterpret_cast<const __nv_bfloat16*>(d.b[BT_X]) + (size_t)tok * d.K + k);
  }
}

// The normed operands' inputs (B_NORM, B_FINAL) staged per batch of chunks by cp.async (every row's loads in flight
// at once): (chunk i of the batch, row t) at ist + (i * QB + t) * IN_ROW, x's 64 values (fp32, or the embedding rows'
// bf16), then the norm weight's 64 bf16; lanes 0..15 copy x (0..7 a bf16 x), lanes 16..23 the weight.
constexpr uint32_t IN_X = CHUNK_K * 4, IN_W = CHUNK_K * 2, IN_ROW = IN_X + IN_W;
constexpr uint32_t IST_OFF_MASK = 0xFFFFFu;  // GemvDesc::ist: the staging's offset,
constexpr uint32_t IST_N_SHIFT = 20;         // then its chunks per batch
__device__ __forceinline__ uint32_t ist_off(const GemvDesc& d) { return d.ist & IST_OFF_MASK; }
__device__ __forceinline__ uint32_t ist_n(const GemvDesc& d) { return d.ist >> IST_N_SHIFT; }

template <int SRC>
__device__ __forceinline__ void stage_row(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t kc, int t, int lane,
                                          uint8_t* row) {
  const uint32_t k = kc * CHUNK_K;
  const int tok = row_token<SRC>(a, t);
  if (lane >= 16 && lane < 24) {
    cp_async16(row + IN_X + 16 * (lane - 16),
               reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k + 8 * (lane - 16));
  } else if (lane < 16 && tok >= 0) {
    if (d.xemb) {
      if (lane < 8) cp_async16(row + 16 * lane, a.tab.embed + (size_t)wc.ids[tok] * HID + k + 8 * lane);
    } else {
      cp_async16(row + 16 * lane, reinterpret_cast<const float*>(d.b[BT_X]) + (size_t)tok * HID + k + 4 * lane);
    }
  }
}
// group g's inputs from a staged row
__device__ __forceinline__ RowIn staged_row(const uint8_t* row, uint32_t g, bool emb) {
  RowIn in{{ZERO4, ZERO4}, ZERO4};
  if (emb) {
    in.u[0] = lds128(row + 16 * g);
  } else {
    in.u[0] = lds128(row + 32 * g), in.u[1] = lds128(row + 32 * g + 16);
  }
  in.w = lds128(row + IN_X + 16 * g);
  return in;
}
// row t's group from its inputs (emb, r: the normed row's source and 1 / rms); a row past the source's tokens: zeros
template <int SRC>
__device__ __forceinline__ void row_values(const DkArgs& a, const RowIn& in, int t, bool emb, float r, float (&v)[GV]) {
  if (row_token<SRC>(a, t) < 0) {
#pragma unroll
    for (int e = 0; e < GV; e++) v[e] = 0.f;
  } else {
    norm8(in, emb, r, v);
  }
}

// B_NORM_CONV / B_QKV in two phases, so most of the B build overlaps the previous op (the kernel projection that makes
// the conv's coefficients): the first writes rows 0..7 as bf16(norm(x)) into the op's staging and B_QKV's rows 8..15
// into the tiles; the second computes the conv over the staged rows once the coefficients exist.
__host__ __device__ constexpr bool is_two_phase(uint32_t bsrc) { return bsrc == B_NORM_CONV || bsrc == B_QKV; }

// the first phase's inputs of (chunk kc, row t) for group g: x's values and the norm weight's; B_QKV's context row
// QB + t (hidden_norm of the fc output of committed token t: u the fc output's bf16, w hidden_norm's)
__device__ __forceinline__ RowIn load_prep(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t kc, int t, uint32_t g) {
  const uint32_t k = kc * CHUNK_K + GV * g;
  RowIn in{{ZERO4, ZERO4}, ZERO4};
  in.w = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k));
  load_x8(d, a, wc, t, k, in);
  return in;
}
__device__ __forceinline__ RowIn load_ctx(const GemvDesc& d, const DkArgs& a, uint32_t kc, int t, uint32_t g) {
  const uint32_t k = kc * CHUNK_K + GV * g;
  RowIn in{{ZERO4, ZERO4}, ZERO4};
  in.w = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_CTX_W]) + k));
  if ((uint32_t)t < a.step.n_ctx)
    in.u[0] = __ldcg(
        reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_CTX]) + (size_t)t * HID + k));
  return in;
}
// vLLM's _C rms_norm of the context row: bf16((ctx * rstd) * w), zero past n_ctx (bf16 words)
__device__ __forceinline__ uint4 ctx_values(const RowIn& in, bool has_ctx, float r) {
  float v[GV], hv[GV];
  bf16x8(in.u[0], v);
  bf16x8(in.w, hv);
#pragma unroll
  for (int e = 0; e < GV; e++) v[e] = bf16_round(__fmul_rn(__fmul_rn(v[e], r), hv[e]));
  const uint4 q = pack_bf16x8(v);
  const uint32_t ctx_mask = (uint32_t)lt_mask64(0u, (uint32_t)has_ctx);  // (zero by a mask: no select against zero)
  return make_uint4(q.x & ctx_mask, q.y & ctx_mask, q.z & ctx_mask, q.w & ctx_mask);
}

// the conv's inputs of (chunk, row t) for group g (half a conv group): token t's coefficients (tap 0 low, tap 1 high)
// from the staged records (a conv group's side-0 coef[cg][0][tap][token] in bf16, 32 bytes; CONV_REC a chunk's), the
// base kernel's taps (side 0: words 0..3 tap 0, 4..7 tap 1) from TMEM column tcol (warpgroup 0's copy)
static_assert(CONV_GSIZE == 2 * GV, "a conv group is two lanes' values");
constexpr uint32_t CONV_GROUPS_CHUNK = CHUNK_K / CONV_GSIZE;
constexpr uint32_t CONV_REC = CONV_GROUPS_CHUNK * 2 * QB * 2;
struct ConvIn {
  uint32_t c;
  uint32_t base[BASE_WORDS];
};
__device__ __forceinline__ void tc_ld8_u32(uint32_t taddr, uint32_t (&v)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
               : "r"(taddr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_st8_u32(uint32_t taddr, uint4 a, uint4 b) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr), "r"(a.x),
               "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w)
               : "memory");
}
__device__ __forceinline__ ConvIn load_conv(const uint8_t* coefs, uint32_t tcol, int t, uint32_t g) {
  const uint8_t* rec = coefs + (g >> 1) * (2 * QB * 2);  // the conv group's record in its chunk's
  ConvIn in;
  in.c = lds16(rec + 2 * t) | (lds16(rec + 2 * (QB + t)) << 16);
  tc_ld8_u32(tcol, in.base);
  return in;
}
// a lane's base taps of chunk kc's group g (both taps of its 8 channels) from the base kernel (bf16 [2][HID])
__device__ __forceinline__ void store_base(const __nv_bfloat16* base, uint32_t kc, uint32_t g, uint32_t tcol) {
  const uint32_t k = kc * CHUNK_K + GV * g;
  tc_st8_u32(tcol, __ldg(reinterpret_cast<const uint4*>(base + k)), __ldg(reinterpret_cast<const uint4*>(base + HID + k)));
}
// Inductor's conv of the normed input (side 0): y = fma(c0, xn[t], (c1 * xn[t - 1]) * (t & 7 >= 1)),
// c_tap = base[0][tap] + coef[t][0][tap][cg] in fp32, rounded to bf16; xn[t], xn[t - 1] read back from the staging
// (the block is the query block: row 0's previous value is the zero pad, so the mask's product is c1 * 0 already)
__device__ __forceinline__ void conv_values(const ConvIn& in, const uint8_t* stg, int t, uint32_t g, float (&v)[GV]) {
  static_assert(QB == 8, "the conv's block is the query block");
  float b0[GV], b1[GV], n[GV], p[GV];
  const uint32_t* w = in.base;
  bf16x8(make_uint4(w[0], w[1], w[2], w[3]), b0), bf16x8(make_uint4(w[4], w[5], w[6], w[7]), b1);
  const uint8_t* row = stg + t * STG_ROW + 16 * g;
  bf16x8(lds128(row), n);
  if (t > 0) {
    bf16x8(lds128(row - STG_ROW), p);
  } else {
#pragma unroll
    for (int e = 0; e < GV; e++) p[e] = 0.f;
  }
  const float d0 = bf16lo(in.c), d1 = bf16hi(in.c);
#pragma unroll
  for (int e = 0; e < GV; e++) v[e] = bf16_round(__fmaf_rn(__fadd_rn(b0[e], d0), n[e], __fmul_rn(__fadd_rn(b1[e], d1), p[e])));
}

// The LM head's B operand: the final norm of the mask positions (rows 0..6 = tokens 1..7, row 7 zero) in bf16, laid
// out as the B buffer holds it: the chunk tiles [K / 64][QB rows][128 B] (SW128).
__host__ __device__ constexpr uint32_t image_bytes(uint32_t K) { return QB * K * 2; }

// OP_IMAGE's part: (row, 16-byte group) i of the operand into the image at `img` (global memory: every SM writes a
// share once, then copies the whole image)
__device__ __forceinline__ void image_group(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t i, uint8_t* img) {
  const uint32_t ngr = d.K / GV;
  const uint32_t t = i / ngr, G = i % ngr, k0 = G * GV;
  uint4 q = ZERO4;
  if (t + 1 < (uint32_t)QB) {
    RowIn in{{ZERO4, ZERO4}, ZERO4};
    in.w = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k0));
    load_x8(d, a, wc, t + 1, k0, in);
    float v[GV];
    norm8(in, false, wc.rstd[t + 1], v);
    q = pack_bf16x8(v);
  }
  const uint32_t chunk = G / LPR, g = G % LPR;
  *reinterpret_cast<uint4*>(img + chunk * (QB * CHUNK) + sw128(t, g)) = q;
}

}  // namespace dk
