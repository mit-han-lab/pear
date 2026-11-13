// The finishers of the GEMV tiles: what the last split-K part of a tile does with its sums (thread ept = the tile's row
// ept, v[t] = the row's sum for B row t), with the arithmetic of the kernels vLLM runs for the draft (Inductor's fused
// kernels of the compiled draft: fp32 intermediates, one bf16 rounding per stored tensor; cuBLAS's bf16 GEMM outputs).
#pragma once
// (after the checkpoint format's draft/csrc/dk_gemv.cuh, which every includer includes first)

namespace dk {

// 8 bf16 values as fp32
__device__ __forceinline__ void bf16x8(uint4 u, float* v) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; j++) v[2 * j] = bf16lo(w[j]), v[2 * j + 1] = bf16hi(w[j]);
}
// 8 fp32 values as bf16 (rounded)
__device__ __forceinline__ uint4 pack_bf16x8(const float* v) {
  return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

// ---- the step's scalars, read from vLLM's device tensors ----
struct StepState {
  uint32_t n_ctx, n_valid, req;
  int64_t last_pos;  // L: the last committed token's position; the query block sits at L + 1 .. L + QB
  int32_t bonus;     // the anchor token
};
__device__ __forceinline__ StepState step_state(const DkArgs& a) {
  StepState s;
  s.n_ctx = a.step.n_ctx;
  s.n_valid = s.n_ctx - (uint32_t)__ldcg(a.step.num_rejected);
  s.req = (uint32_t)__ldcg(a.step.idx_mapping);
  s.last_pos = __ldcg(a.step.positions + s.n_valid - 1);
  const int32_t ns = __ldcg(a.step.num_sampled);
  s.bonus = ns > 0 ? (int32_t)__ldcg(a.step.last_sampled + (a.step.bonus_row ? (uint32_t)(ns - 1) : s.req))
                   : __ldcg(a.step.next_prefill + s.req);
  return s;
}
// query token t's id and its rotary position (clamped as prepare_dflash_inputs does)
__device__ __forceinline__ int32_t query_id(const DkArgs& a, const StepState& s, int t) { return t == 0 ? s.bonus : (int32_t)a.tab.mask_token; }
__device__ __forceinline__ int64_t query_pos(const DkArgs& a, const StepState& s, int t) {
  return min(s.last_pos + 1 + t, (int64_t)a.tab.max_model_len - 1);
}

// the step's tables the finishers read, filled once per launch by the epilogue warps (fill_step) before their first op
struct alignas(16) StepSmem {  // (16: the rotary rows are written in 16-byte pieces)
  StepState st;
  int64_t ctx_pos[NCTX];  // the committed tokens' positions (0 past n_valid)
  int64_t ctx_row[NCTX];  // their draft-cache row of KV head 0 (+ kvh * kv_block for head kvh), -1: no write
  uint16_t cs[NB][HDIM];  // the rotary cos | sin rows (bf16) of the query block's positions, then the committed tokens'
};
__device__ void fill_step(const DkArgs& a, StepSmem& ss, int ept) {
  if (ept == 0) ss.st = step_state(a);
  named_sync(BAR_EPI, NEPI);
  const StepState s = ss.st;
  const uint32_t kvb = a.tab.kv_block;
  if (ept < NCTX) {
    int64_t pos = 0, row = -1;
    if ((uint32_t)ept < s.n_valid) {
      pos = __ldcg(a.step.positions + ept);
      const int32_t blk = __ldcg(a.step.block_table + pos / kvb);
      if (blk != 0) row = (int64_t)blk * NKVH * kvb + pos % kvb;  // the null block takes no write
    }
    ss.ctx_pos[ept] = pos;
    ss.ctx_row[ept] = row;
  }
  named_sync(BAR_EPI, NEPI);
  constexpr int PIECES = HDIM * 2 / 16;  // 16-byte pieces of a row
  for (int i = ept; i < NB * PIECES; i += NEPI) {
    const int r = i / PIECES, piece = i % PIECES;
    const int64_t pos = r < QB ? query_pos(a, s, r) : ss.ctx_pos[r - QB];
    *reinterpret_cast<uint4*>(&ss.cs[r][piece * 8]) = __ldg(reinterpret_cast<const uint4*>(a.tab.cos_sin + (size_t)pos * HDIM) + piece);
  }
  named_sync(BAR_EPI, NEPI);
}

// a warp's per-token sums of T values per lane over its 32 lanes by a transposing butterfly: each step pairs lanes l
// and l ^ o, each keeping half of its values plus the partner's same half (T - 1 shuffles), then the low lane bits;
// every lane returns token lane / (32 / T)'s sum (T separate warp sums take 5 T shuffles)
template <int T>
__device__ __forceinline__ float warp_sum_t(const float (&x)[T], int lane) {
  static_assert(T >= 1 && T <= 32 && (T & (T - 1)) == 0, "a power of two");
  float v[T];
#pragma unroll
  for (int i = 0; i < T; i++) v[i] = x[i];
  int o = 16;
#pragma unroll
  for (int n = T; n > 1; n >>= 1, o >>= 1) {
    const bool is_high_lane = (lane & o) != 0;
#pragma unroll
    for (int i = 0; i < n / 2; i++) {
      const float keep = is_high_lane ? v[i + n / 2] : v[i], send = is_high_lane ? v[i] : v[i + n / 2];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
  }
#pragma unroll
  for (; o >= 1; o >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
  return v[0];
}
// the warps' per-token sums into the epilogue's reduction buffer [warp][NB] (one lane per token)
template <int T>
__device__ __forceinline__ void tile_warp_sums(const float (&x)[T], const Smem& sm, int ept) {
  constexpr int LPT = 32 / T;  // lanes per token after the butterfly
  const int w = ept >> 5, lane = ept & 31;
  const float s = warp_sum_t<T>(x, lane);
  if (lane % LPT == 0) sm.ered[w * NB + lane / LPT] = s;
}

// per-token sums over the tile's 128 rows (fixed order); thread t < T returns token t's sum, the others 0
template <int T>
__device__ __forceinline__ float tile_sum_tok(const float (&x)[T], const Smem& sm, int ept, int t_out) {
  tile_warp_sums<T>(x, sm, ept);
  named_sync(BAR_EPI, NEPI);
  float r = 0.f;
  if (t_out < T) r = (sm.ered[t_out] + sm.ered[NB + t_out]) + (sm.ered[2 * NB + t_out] + sm.ered[3 * NB + t_out]);
  named_sync(BAR_EPI, NEPI);
  return r;
}

// ---- the convs' coefficients of group g, side s, every token: c[t] = tap 0 (low) | tap 1 (high) as bf16 ----
// (the launch's half of the fp32 sums, rounded to bf16 as the convs read them)
__device__ __forceinline__ void load_coefs(const float* coef, uint32_t g, uint32_t side, uint32_t (&c)[QB]) {
  const float4* p0 = reinterpret_cast<const float4*>(coef + coef_row(side, 0, g) * QB);  // tap 0's 8 tokens
  const float4* p1 = reinterpret_cast<const float4*>(coef + coef_row(side, 1, g) * QB);  // tap 1's
  const float4 a0 = __ldcg(p0), a1 = __ldcg(p0 + 1), b0 = __ldcg(p1), b1 = __ldcg(p1 + 1);
  const float t0[QB] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, t1[QB] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
  for (int t = 0; t < QB; t++) c[t] = pack_bf16x2(t0[t], t1[t]);
}

// ---- E_COEF: the tile's sums (row r's 8 tokens) into the launch's parity of the kernel projection's output (fp32,
// 32 contiguous bytes; a tile's rows contiguous): the readers round them to bf16 ----
__device__ __forceinline__ void finish_coef(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[QB],
                                            uint32_t epoch) {
  const uint32_t r = mt * TILE_M + ept;
  if (r >= d.rows) return;
  float4* coef_dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(d.e[0]) + (epoch & 1) * COEF_HALF + r * QB);
  __stcg(coef_dst, make_float4(v[0], v[1], v[2], v[3])), __stcg(coef_dst + 1, make_float4(v[4], v[5], v[6], v[7]));
}

// ---- E_BF16: the sums as bf16 (cuBLAS's output), out [n_out][rows] ----
template <int N>
__device__ __forceinline__ void finish_bf16(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[N]) {
  const uint32_t r = mt * TILE_M + ept, rows = d.rows, n_out = d.n_out;
  if (r >= rows) return;
  __nv_bfloat16* const out = reinterpret_cast<__nv_bfloat16*>(d.e[0]);
#pragma unroll
  for (uint32_t t = 0; t < (uint32_t)N; t++)  // (a constant bound: v stays in registers)
    if (t < n_out) out[(size_t)t * rows + r] = __float2bfloat16(v[t]);
}

// ---- E_FC: the fc output (bf16 [QB][HID]) and its per-token sums of squares (e[1]: fp32 [2 parity][QB]) ----
template <int N>
__device__ __forceinline__ void finish_fc(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[N], const Smem& sm,
                                          uint32_t epoch) {
  const uint32_t r = mt * TILE_M + ept;
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(d.e[0]);
  float sq[QB];
#pragma unroll
  for (int t = 0; t < QB; t++) {
    const float c = bf16_round(v[t]);
    out[(size_t)t * HID + r] = __float2bfloat16(c);
    sq[t] = c * c;
  }
  const float s = tile_sum_tok<QB>(sq, sm, ept, ept);
  if (ept < QB) red_add_ssq(reinterpret_cast<long long*>(d.e[1]) + (epoch & 1) * QB + ept, s);
}

// ---- E_QKV: tile mt = head mt of qkv_proj (q, k, v heads), columns 0..7 the query block, 8..15 the committed tokens:
// a part's sums added into its slot of e[0]'s launch parity (at most two parts a slot: one sum in either order), no
// reduction round; the tile's first part zeroes the next launch's. The attention items sum the slots (attn_inputs).
__device__ __forceinline__ void qkv_part(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[NB], uint32_t epoch,
                                         bool is_first_part, uint32_t slot) {
  // e[0]: fp32 [2][QKV_SLOTS][mtiles][NB][TILE_M] (a column's 128 rows contiguous: a warp's adds coalesced)
  const size_t plane = (size_t)d.mtiles * NB * TILE_M, half = QKV_SLOTS * plane;
  float* rec = reinterpret_cast<float*>(d.e[0]) + (size_t)mt * NB * TILE_M + ept;
  float* slot_sums = rec + (epoch & 1) * half + slot * plane;
#pragma unroll
  for (int t = 0; t < NB; t++) red_add_f32(slot_sums + t * TILE_M, v[t]);
  if (is_first_part) {
    float* next_sums = rec + ((epoch + 1) & 1) * half;
#pragma unroll
    for (uint32_t k = 0; k < QKV_SLOTS; k++)
#pragma unroll
      for (int t = 0; t < NB; t++) __stcg(next_sums + k * plane + t * TILE_M, 0.f);
  }
}

// a finisher's inputs, loaded before its tile's sums arrive (load_fin: once the op's dependencies are complete)
// the finisher's inputs as loaded (raw bits): converted where the finisher uses them, so the loads stay in flight
// until then (a conversion right after a load waits for it)
struct FinIn {
  uint32_t b;      // E_RESID: the row's side-1 base kernel (bf16: tap 0 low, tap 1 high)
  uint32_t c[QB];  // E_RESID: the tokens' side-1 coefficients of the row's group (bf16: tap 0 low, tap 1 high)
  uint32_t r[QB];  // E_RESID: the residual of the row (fp32; resid 0: the embedding's bf16)
  __device__ __forceinline__ float base0() const { return bf16lo(b); }
  __device__ __forceinline__ float base1() const { return bf16hi(b); }
  // the residual as the finisher adds it (resid: 0 the embedding rows, 1 bf16 of the fp32 buffer, 2 the fp32 buffer)
  __device__ __forceinline__ float resid(int t, uint32_t mode) const {
    return mode == 0 ? bf16lo(r[t]) : mode == 1 ? bf16_round(__uint_as_float(r[t])) : __uint_as_float(r[t]);
  }
};
__device__ __forceinline__ FinIn load_fin(const GemvDesc& d, uint32_t mt, int ept, const Smem& sm, const DkArgs& a,
                                          uint32_t epoch) {
  constexpr int RESID_IN = 2;  // e[RESID_IN]: E_RESID's residual (finish_resid)
  constexpr int COEF_IN = 3;   // e[COEF_IN]: its coefficients
  constexpr int BASE_IN = 4;   // e[BASE_IN]: its base_kernel
  FinIn f;
  f.b = 0;
  if (d.epi == E_RESID) {
    const uint32_t c = mt * TILE_M + ept, g = c / CONV_GSIZE;
    const float* coef = reinterpret_cast<const float*>(d.e[COEF_IN]) + (epoch & 1) * COEF_HALF;
    const unsigned short* base = reinterpret_cast<const unsigned short*>(d.e[BASE_IN]) + 2 * HID;  // side 1
    const unsigned int* rf = reinterpret_cast<const unsigned int*>(d.e[RESID_IN]);
    f.b = (uint32_t)__ldg(base + c) | ((uint32_t)__ldg(base + HID + c) << 16);
    load_coefs(coef, g, 1, f.c);
#pragma unroll
    for (int t = 0; t < QB; t++) {
      if (d.resid == 0)
        f.r[t] = __ldg(reinterpret_cast<const unsigned short*>(a.tab.embed) + (size_t)query_id(a, sm.step->st, t) * HID + c);
      else
        f.r[t] = __ldcg(rf + (size_t)t * HID + c);
    }
  }
  return f;
}

// ---- E_RESID: o = bf16(sums); x = c0 * o[t] + (c1 * o[t - 1]) * (t & 7 >= 1) + r in fp32, c_tap = base[1][tap] +
// coef[t][1][tap][g] (Inductor's), into e[0] (fp32 [QB][HID]), its per-token sums of squares into e[1] (fp32 [2][QB]);
// e[2] the residual (FinIn::resid), e[3] the coefficients (COEF_HALF a parity), e[4] base_kernel (bf16 [2][2][HID]).
__device__ __forceinline__ void finish_resid(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[QB], const Smem& sm,
                                             const FinIn& f, uint32_t epoch) {
  const uint32_t c = mt * TILE_M + ept;
  float* xo = reinterpret_cast<float*>(d.e[0]);
  float sq[QB], prev_o = 0.f;
#pragma unroll
  for (int t = 0; t < QB; t++) {
    const float o = bf16_round(v[t]);
    const float c0 = __fadd_rn(f.base0(), bf16lo(f.c[t])), c1 = __fadd_rn(f.base1(), bf16hi(f.c[t]));
    const float tail = __fmul_rn(__fmul_rn(c1, prev_o), (t & (QB - 1)) >= 1 ? 1.0f : 0.0f);
    const float y = d.conv_fma ? __fmaf_rn(c0, o, tail) : __fadd_rn(__fmul_rn(c0, o), tail);
    prev_o = o;
    const float x = __fadd_rn(y, f.resid(t, d.resid));
    xo[(size_t)t * HID + c] = x;
    sq[t] = x * x;
  }
  const float tot = tile_sum_tok<QB>(sq, sm, ept, ept);
  if (ept < QB) red_add_ssq(reinterpret_cast<long long*>(d.e[1]) + (epoch & 1) * QB + ept, tot);
}

// ---- E_SILU: a paired tile's rows 0..63 are gate rows, 64..127 the up rows of the same outputs j = mt * 64 + r:
// a = bf16((g / (exp(-g) + 1)) * u) of g = bf16(gate sums), u = bf16(up sums) (Inductor's kernel), out e[0]
// (bf16 [QB][INTER]).
__device__ __forceinline__ void finish_silu(const GemvDesc& d, uint32_t mt, int ept, const float (&v)[QB], const Smem& sm) {
  constexpr int HALF = TILE_M / 2;
  if (ept >= HALF) {
#pragma unroll
    for (int t = 0; t < QB; t++) sm.xchg[(ept - HALF) * QB + t] = bf16_round(v[t]);
  }
  named_sync(BAR_EPI, NEPI);
  if (ept < HALF) {
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(d.e[0]);
    const uint32_t j = mt * HALF + ept;
#pragma unroll
    for (int t = 0; t < QB; t++) {
      const float gt = bf16_round(v[t]), u = sm.xchg[ept * QB + t];
      out[(size_t)t * INTER + j] = __float2bfloat16(silu_f32(gt) * u);
    }
  }
  named_sync(BAR_EPI, NEPI);
}

}  // namespace dk
