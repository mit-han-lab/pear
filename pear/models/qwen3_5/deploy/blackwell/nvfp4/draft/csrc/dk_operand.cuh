// The GEMV B operands in NVFP4, built by the worker warps into the SM's B buffer once per op: per 256-value K chunk a
// tile of e2m1 codes [n rows][128 B] (SW128) and the rows' e4m3 scales, by vLLM's NVFP4 input quantization of the bf16
// tensors of vLLM's compiled draft (Inductor's arithmetic); worker warp w builds row w of every chunk.
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

constexpr int BLK_K = FP4_BLK;  // values per lane per chunk row: one scale block
constexpr int HALF = 16;        // lanes per chunk row
constexpr int CPS = 2;          // chunks per step of a worker warp (a half warp each)
static_assert(CHUNK_K == HALF * BLK_K, "a half warp per chunk row");
constexpr uint32_t STG_ROW = CHUNK_K * 2;  // bytes per staged bf16 row of a chunk
constexpr uint4 ZERO4 = {0u, 0u, 0u, 0u};


constexpr float E2M1_MAX = 6.0f;  // e2m1's largest magnitude: a block's amax maps to it

// an op's NVFP4 input quantization (vLLM's: a block's e4m3 scale of gs * (amax * rcp(6)), the codes of
// v * rcp(scale * rcp(gs))), its constants in registers
struct Fp4Q {
  float gs, inv_gs, sixth;
  __device__ __forceinline__ explicit Fp4Q(float g)
      : gs(g), inv_gs(rcp_approx_ftz(g)), sixth(rcp_approx_ftz(E2M1_MAX)) {}
};

// chunk position i's tile of codes and its rows' scales (sfb_pos)
struct ChunkOut {
  uint8_t* codes;
  uint8_t* sf;
  uint32_t plane;
  __device__ __forceinline__ ChunkOut(const GemvDesc& d, const Smem& sm, uint32_t i)
      : codes(sm.bbuf + d.b_off + i * (d.n * CHUNK)), sf(sm.bbuf + d.sfb_off + i * (d.n * SFB_ROW)), plane(d.sfb_plane) {}
  // row t's block b in NVFP4 (its 8 code bytes, its e4m3 scale) into the chunk
  __device__ __forceinline__ void store(uint32_t t, uint32_t b, const uint32_t (&q)[3]) const {
    sts64(codes + sw128(t, b >> 1) + (b & 1) * 8, q[0], q[1]);
    sts8(sf + t * SFB_ROW + sfb_pos(b, plane), q[2]);
  }
  // row t's block b (this lane's 16 values), quantized
  __device__ __forceinline__ void put(uint32_t t, uint32_t b, const float (&v)[BLK_K], const Fp4Q& f) const {
    uint32_t q[3];
    quant16(v, f, q);
    store(t, b, q);
  }
  // 16 values in NVFP4: q[0], q[1] the codes, q[2] the e4m3 scale
  __device__ __forceinline__ static void quant16(const float (&v)[BLK_K], const Fp4Q& f, uint32_t (&q)[3]) {
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < BLK_K; e++) amax = fmaxf(amax, fabsf(v[e]));
    const __nv_fp8_e4m3 s8(f.gs * (amax * f.sixth));
    const float sfq = float(s8);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * f.inv_gs) : 0.f;
    q[0] = q[1] = 0u;
#pragma unroll
    for (int e = 0; e < BLK_K; e += 2) q[e / 8] |= f32x2_to_e2m1x2(v[e] * osc, v[e + 1] * osc) << (4 * (e % 8));
    q[2] = *reinterpret_cast<const uint8_t*>(&s8);
  }
};

// the inputs of one (chunk, row)'s block for a lane: the values' sources (fp32 x in u[0..3], bf16 in u[0..1]) and the
// norm weight's (bf16 in w[0..1])
struct RowIn {
  uint4 u[4];
  uint4 w[2];
};

// the RMSNorm of 16 values: bf16((x * r) * w) per value (two products, one rounding), as fp32 (emb: x is the embedding
// row's bf16)
__device__ __forceinline__ void norm16(const RowIn& in, bool emb, float r, float (&v)[BLK_K]) {
  float xv[BLK_K], wv[BLK_K];
  if (emb) {
    bf16x8(in.u[0], xv);
    bf16x8(in.u[1], xv + 8);
  } else {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      xv[4 * q] = __uint_as_float(in.u[q].x), xv[4 * q + 1] = __uint_as_float(in.u[q].y);
      xv[4 * q + 2] = __uint_as_float(in.u[q].z), xv[4 * q + 3] = __uint_as_float(in.u[q].w);
    }
  }
  bf16x8(in.w[0], wv);
  bf16x8(in.w[1], wv + 8);
#pragma unroll
  for (int e = 0; e < BLK_K; e++) v[e] = bf16_round(__fmul_rn(__fmul_rn(xv[e], r), wv[e]));
}
// the 1 / rms of token t of an op's normed input (the embedding rows' for layer 0)
__device__ __forceinline__ float row_rstd(bool emb, const WorkerConst& wc, int t) { return emb ? wc.rstd_emb[t] : wc.rstd[t]; }

// x's 16 values of token t from K index k: the fp32 residual stream, or the query block's embedding row (layer 0)
__device__ __forceinline__ void load_x16(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t t, uint32_t k,
                                         RowIn& in) {
  if (d.xemb) {
    const uint4* p = reinterpret_cast<const uint4*>(a.tab.embed + (size_t)wc.ids[t] * HID + k);
    in.u[0] = __ldg(p), in.u[1] = __ldg(p + 1);
  } else {
    const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const float*>(d.b[BT_X]) + (size_t)t * HID + k);
#pragma unroll
    for (int q = 0; q < 4; q++) in.u[q] = __ldcg(p + q);
  }
}

// the B source's token of B row t (B_FINAL: rows 0..6 are tokens 1..7), or -1 for a zero row
template <int SRC>
__device__ __forceinline__ int row_token(const DkArgs& a, int t) {
  if constexpr (SRC == B_FINAL) return t + 1 < QB ? t + 1 : -1;
  if constexpr (SRC == B_AUX) return (uint32_t)t < a.step.n_ctx ? t : -1;
  return t;
}

// The one-phase operands' inputs staged per batch of chunks by cp.async (every row's loads in flight at once, the code
// compact): (chunk i of the batch, row t) at ist + (i * QB + t) * in_row: the row's 256 values as they are (bf16: 512
// bytes; fp32: 1 KB), then the norm weight's (512 bytes, normed rows). The 32 lanes copy, the half warps read back.
constexpr uint32_t IN_X = CHUNK_K * 4, IN_W = CHUNK_K * 2;
template <int SRC>
__host__ __device__ constexpr uint32_t in_row() { return SRC == B_AUX || SRC == B_COPY ? CHUNK_K * 2 : IN_X + IN_W; }
constexpr uint32_t IST_OFF_MASK = 0xFFFFFu;  // GemvDesc::ist: the staging's offset,
constexpr uint32_t IST_N_SHIFT = 20;         // then its chunks per batch
__device__ __forceinline__ uint32_t ist_off(const GemvDesc& d) { return d.ist & IST_OFF_MASK; }
__device__ __forceinline__ uint32_t ist_n(const GemvDesc& d) { return d.ist >> IST_N_SHIFT; }

// row t of chunk kc into its staging row (lane j copies values 8 j .. 8 j + 7)
template <int SRC>
__device__ __forceinline__ void stage_row(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t kc, int t, int lane,
                                          uint8_t* row) {
  const uint32_t k = kc * CHUNK_K + 8 * lane;
  const int tok = row_token<SRC>(a, t);
  if constexpr (SRC == B_AUX) {
    if (tok >= 0) cp_async16(row + 16 * lane, a.step.aux[k / HID] + (size_t)tok * a.step.aux_stride + k % HID);
  } else if constexpr (SRC == B_COPY) {
    cp_async16(row + 16 * lane, reinterpret_cast<const __nv_bfloat16*>(d.b[BT_X]) + (size_t)tok * d.K + k);
  } else {
    cp_async16(row + IN_X + 16 * lane, reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k);
    if (tok >= 0) {
      if (d.xemb) {
        cp_async16(row + 16 * lane, a.tab.embed + (size_t)wc.ids[tok] * HID + k);
      } else {
        const float* x = reinterpret_cast<const float*>(d.b[BT_X]) + (size_t)tok * HID + k;
        cp_async16(row + 32 * lane, x);
        cp_async16(row + 32 * lane + 16, x + 4);
      }
    }
  }
}
// block b's inputs from a staged row
template <int SRC>
__device__ __forceinline__ RowIn staged_row(const uint8_t* row, uint32_t b, bool emb) {
  RowIn in{{ZERO4, ZERO4, ZERO4, ZERO4}, {ZERO4, ZERO4}};
  const uint8_t* x = row + (SRC == B_AUX || SRC == B_COPY || emb ? 32 : 64) * b;
  in.u[0] = lds128(x), in.u[1] = lds128(x + 16);
  if constexpr (!(SRC == B_AUX || SRC == B_COPY)) {
    if (!emb) in.u[2] = lds128(x + 32), in.u[3] = lds128(x + 48);
    const uint8_t* w = row + IN_X + 32 * b;
    in.w[0] = lds128(w), in.w[1] = lds128(w + 16);
  }
  return in;
}
// row t's block from its inputs (emb, r: the normed row's source and 1 / rms)
template <int SRC>
__device__ __forceinline__ void row_values(const DkArgs& a, const RowIn& in, int t, bool emb, float r, float (&v)[BLK_K]) {
  if (row_token<SRC>(a, t) < 0) {
#pragma unroll
    for (int e = 0; e < BLK_K; e++) v[e] = 0.f;
  } else if constexpr (SRC == B_AUX || SRC == B_COPY) {
    bf16x8(in.u[0], v);
    bf16x8(in.u[1], v + 8);
  } else {
    norm16(in, emb, r, v);
  }
}

// B_NORM_CONV / B_QKV in two phases, so most of the B build overlaps the previous op (the kernel projection that makes
// the conv's coefficients): the first writes rows 0..7 as bf16(norm(x)) into the op's staging and B_QKV's rows 8..15
// into the tiles; the second quantizes the conv over the staged rows once the coefficients exist.
__host__ __device__ constexpr bool is_two_phase(uint32_t bsrc) { return bsrc == B_NORM_CONV || bsrc == B_QKV; }

// the first phase's inputs of (chunk kc, row t) for block b: x's values and the norm weight's; B_QKV's context row
// QB + t (hidden_norm of the fc output of committed token t: u the fc output's bf16, w hidden_norm's)
__device__ __forceinline__ RowIn load_prep(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t kc, int t, uint32_t b) {
  const uint32_t k = kc * CHUNK_K + BLK_K * b;
  RowIn in{{ZERO4, ZERO4, ZERO4, ZERO4}, {ZERO4, ZERO4}};
  const uint4* w = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k);
  in.w[0] = __ldg(w), in.w[1] = __ldg(w + 1);
  load_x16(d, a, wc, t, k, in);
  return in;
}
__device__ __forceinline__ RowIn load_ctx(const GemvDesc& d, const DkArgs& a, uint32_t kc, int t, uint32_t b) {
  const uint32_t k = kc * CHUNK_K + BLK_K * b;
  RowIn in{{ZERO4, ZERO4, ZERO4, ZERO4}, {ZERO4, ZERO4}};
  const uint4* w = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_CTX_W]) + k);
  in.w[0] = __ldg(w), in.w[1] = __ldg(w + 1);
  if ((uint32_t)t < a.step.n_ctx) {
    const uint4* p =
        reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_CTX]) + (size_t)t * HID + k);
    in.u[0] = __ldcg(p), in.u[1] = __ldcg(p + 1);
  }
  return in;
}
// vLLM's _C rms_norm of the context row: bf16((ctx * rstd) * w), zero past n_ctx
__device__ __forceinline__ void ctx_values(const RowIn& in, bool has_ctx, float r, float (&v)[BLK_K]) {
  float hv[BLK_K];
  bf16x8(in.u[0], v);
  bf16x8(in.u[1], v + 8);
  bf16x8(in.w[0], hv);
  bf16x8(in.w[1], hv + 8);
  const uint32_t ctx_mask = (uint32_t)lt_mask64(0u, (uint32_t)has_ctx);  // (zero by a mask: no select against zero)
#pragma unroll
  for (int e = 0; e < BLK_K; e++)
    v[e] = __uint_as_float(__float_as_uint(bf16_round(__fmul_rn(__fmul_rn(v[e], r), hv[e]))) & ctx_mask);
}

// the conv's inputs of (chunk, row t) for block b (a conv group): token t's coefficients (tap 0 low, tap 1 high) from
// the staged records (each group's side-0 coef[g][0][tap][token] in bf16, 32 bytes: CONV_REC a chunk), the base
// kernel's taps (side 0: words 0..7 tap 0, 8..15 tap 1) from TMEM column tcol (warpgroup 0's copy)
static_assert(CONV_GSIZE == BLK_K, "a scale block is a conv group");
constexpr uint32_t CONV_REC = HALF * 2 * QB * 2;
struct ConvIn {
  uint32_t c;
  uint32_t base[BASE_WORDS];
};
__device__ __forceinline__ ConvIn load_conv(const uint8_t* coefs, uint32_t tcol, int t, uint32_t b) {
  const uint8_t* rec = coefs + b * (2 * QB * 2);  // the group's record in its chunk's
  ConvIn in;
  in.c = lds16(rec + 2 * t) | (lds16(rec + 2 * (QB + t)) << 16);
  tc_ld16_u32(tcol, in.base);
  return in;
}
// a lane's base taps of chunk kc's block b (both taps of its 16 channels) from the base kernel (bf16 [2][HID])
__device__ __forceinline__ void store_base(const __nv_bfloat16* base, uint32_t kc, uint32_t b, uint32_t tcol) {
  const uint32_t k = kc * CHUNK_K + BLK_K * b;
  const uint4* t0 = reinterpret_cast<const uint4*>(base + k);
  const uint4* t1 = reinterpret_cast<const uint4*>(base + HID + k);
  tc_st16_u32(tcol, __ldg(t0), __ldg(t0 + 1), __ldg(t1), __ldg(t1 + 1));
}
// Inductor's conv of the normed input (side 0): y = fma(c0, xn[t], (c1 * xn[t - 1]) * (t & 7 >= 1)),
// c_tap = base[0][tap] + coef[t][0][tap][g] in fp32, rounded to bf16; xn[t], xn[t - 1] read back from the staging (the
// block is the query block: row 0's previous value is the zero pad, so the mask's product is c1 * 0 already)
__device__ __forceinline__ void conv_values(const ConvIn& in, const uint8_t* stg, int t, uint32_t b, float (&v)[BLK_K]) {
  static_assert(QB == 8, "the conv's block is the query block");
  float b0[BLK_K], b1[BLK_K], n[BLK_K], p[BLK_K];
  const uint32_t* w = in.base;
  bf16x8(make_uint4(w[0], w[1], w[2], w[3]), b0), bf16x8(make_uint4(w[4], w[5], w[6], w[7]), b0 + 8);
  bf16x8(make_uint4(w[8], w[9], w[10], w[11]), b1), bf16x8(make_uint4(w[12], w[13], w[14], w[15]), b1 + 8);
  const uint8_t* row = stg + t * STG_ROW + 32 * b;
  bf16x8(lds128(row), n), bf16x8(lds128(row + 16), n + 8);
  if (t > 0) {
    const uint8_t* prev_row = row - STG_ROW;
    bf16x8(lds128(prev_row), p), bf16x8(lds128(prev_row + 16), p + 8);
  } else {
#pragma unroll
    for (int e = 0; e < BLK_K; e++) p[e] = 0.f;
  }
  const float d0 = bf16lo(in.c), d1 = bf16hi(in.c);
#pragma unroll
  for (int e = 0; e < BLK_K; e++) v[e] = bf16_round(__fmaf_rn(__fadd_rn(b0[e], d0), n[e], __fmul_rn(__fadd_rn(b1[e], d1), p[e])));
}

// The LM head's B operand: the final norm of the mask positions (rows 0..6 = tokens 1..7, row 7 zero) quantized as
// vLLM's NVFP4 input quantization, laid out as the B buffer holds it: the chunk tiles [K / 256][QB rows][128 B]
// (SW128), then their scales' planes (sfb_plane of K / 256 chunk positions).
__host__ __device__ constexpr uint32_t fp4_image_bytes(uint32_t K) { return QB * (K / 2) + 2 * sfb_plane(K / CHUNK_K, QB); }

// OP_QUANT's part: (row, 16-value block) i of the operand into the image at `img` (global memory: every SM quantizes a
// share once, then copies the whole image)
__device__ __forceinline__ void quant_block(const GemvDesc& d, const DkArgs& a, const WorkerConst& wc, uint32_t i, uint8_t* img) {
  const uint32_t nblk = d.K / FP4_BLK;
  const uint32_t t = i / nblk, b = i % nblk, k0 = b * FP4_BLK;
  uint32_t code[2] = {0u, 0u};
  uint8_t sfb = 0;
  if (t + 1 < (uint32_t)QB) {
    RowIn in{{ZERO4, ZERO4, ZERO4, ZERO4}, {ZERO4, ZERO4}};
    const uint4* w = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(d.b[BT_NORM_W]) + k0);
    in.w[0] = __ldg(w), in.w[1] = __ldg(w + 1);
    load_x16(d, a, wc, t + 1, k0, in);
    float v[BLK_K];
    norm16(in, false, wc.rstd[t + 1], v);
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < BLK_K; e++) amax = fmaxf(amax, fabsf(v[e]));
    const float gs = d.gs;
    const __nv_fp8_e4m3 q(gs * (amax * rcp_approx_ftz(E2M1_MAX)));
    const float sfq = float(q);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rcp_approx_ftz(gs)) : 0.f;
#pragma unroll
    for (int j = 0; j < FP4_BLK; j += 2) code[j / 8] |= f32x2_to_e2m1x2(v[j] * osc, v[j + 1] * osc) << (8 * ((j / 2) % 4));
    sfb = *reinterpret_cast<const uint8_t*>(&q);
  }
  const uint32_t byte = k0 / 2, chunk = byte / CHUNK;  // the block's first byte in the row's packed codes, its chunk
  uint8_t* tile = img + chunk * (QB * CHUNK);
  *reinterpret_cast<uint2*>(tile + sw128(t, (byte % CHUNK) >> 4) + (byte & 15)) = make_uint2(code[0], code[1]);
  img[QB * (d.K / 2) + chunk * (QB * SFB_ROW) + t * SFB_ROW + sfb_pos(b % SF_CHUNK, sfb_plane(d.K / CHUNK_K, QB))] = sfb;
}

}  // namespace dk
