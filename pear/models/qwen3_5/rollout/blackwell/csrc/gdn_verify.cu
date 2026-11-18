// The Gated DeltaNet core of verify steps under replaySSM: a request keeps one state slot (the state at the start of its
// last verify block) and a record of that block; a step replays the block's committed tokens from the slot, gives its
// own block's outputs and writes the state once, one CTA per row and value head on effective vectors (3xTF32 MMAs).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sync.cuh"

namespace {

constexpr int kDim = 128;                   // the key and the value head dimension
constexpr int kTaps = 3;                    // the conv window: the width-4 kernel's earlier inputs
constexpr int kThreads = 256;               // a CTA: eight warps of 16 state rows; the conv's q / k and v channels
constexpr int kWarps = kThreads / 32;
constexpr int kColChunks = kDim / 16;       // a thread's 16-column chunks of a state row
constexpr int kStateChunks = 2 * kColChunks;  // a thread's float4 of the state: two rows of kColChunks
constexpr int kVecStride = kDim + 16;       // smem row stride of the products' vectors (conflict-free fragment loads)
constexpr int kValStride = kDim + 4;        // smem row stride of the per-row corrections (conflict-free fragment loads)
constexpr int kGramSplit = 2;               // threads per product of the step's vectors
constexpr int kRowInts = 4;                 // a batch row: first token, tokens (0: not a verify row), request, position
constexpr int kSlotInts = 4;                // a row's slots in a group: source, destination, boundary block (-1: none)
constexpr int kMetaThreads = 128;           // threads of the per-step bookkeeping kernels
constexpr float kL2Eps = 1e-6f;             // FLA's l2norm epsilon of q and k, under the square root
constexpr float kSoftplusThreshold = 20.f;  // softplus is the identity above it, as torch's
constexpr uint32_t kTf32Half = 0x1000u;     // half an ulp of tf32 (bit 12)
constexpr uint32_t kTf32Mask = 0xffffe000u; // the bits of a tf32 value
constexpr uint32_t kFullMask = 0xffffffffu;

// The sizes of a step whose verify blocks have at most MT tokens
template <int MT>
struct Sizes {
  static constexpr int L = 2 * MT;          // replayed and current tokens of a step
  static constexpr int N = L + MT;          // a state row's products: effective keys, then effective queries
  static constexpr int NT = (N + 7) / 8;    // n-tiles of the products' MMAs
  static constexpr int GRAM = (L + MT) * L + MT;  // the step's vector products: keys' and queries' with the keys, q.q
  // the unroll of the loops over the vectors' and the values' rows: full at MT <= 4, rolled at MT = 8 (code size)
  static constexpr int UNROLL_VEC = MT <= 4 ? NT * 8 : 1, UNROLL_VAL = MT <= 4 ? N : 1;
};

struct Layer {                       // one Gated DeltaNet layer: its caches, its record, its weights
  float* state;                      // fp32 [slots][HV][V][K]
  __nv_bfloat16* conv;               // bf16 conv window cache
  uint8_t* record;                   // [requests][2][record bytes]
  const __nv_bfloat16* conv_w;       // bf16 [C][4]
  const float* a_log;                // fp32 [HV]
  const void* dt_bias;               // [HV], fp32 or bf16
  const void* norm_w;                // [V], fp32 or bf16
  long long state_slot;              // the state cache's slot stride (floats)
  long long conv_slot, conv_ch, conv_tap;  // the conv cache's strides (elements)
  long long record_req, record_copy;       // the record's request and copy strides (bytes)
  int group;                         // the layer's mamba KV cache group: its slots in the step's table
  int dt_bias_f32, norm_w_f32;
  int pad;
};

struct Step {                        // a step's batch
  const __nv_bfloat16* qkvz;         // [tokens][C + HV V]: q, k, v (the conv inputs), z
  const __nv_bfloat16* ba;           // [tokens][2 HV]: b, a
  __nv_bfloat16* out;                // [tokens][HV][V]: the gated norm's output
  const int* rows;                   // [rows][kRowInts]
  const int* slots;                  // [groups][max rows][kSlotInts]
  const int* replay;                 // [requests] the committed tokens of each request's last verify block
  const int* parity;                 // [requests] the record copy that holds that block
  long long qkvz_row, ba_row;        // row strides (elements)
  long long record_ba;               // the record's b / a offset (bytes): after its (kTaps + max T) x C inputs
  int max_rows, H, HV, R, block_size;  // R: value heads per key head
  float scale, norm_eps;
  int gate_sigmoid, is_flush;
};

__device__ __forceinline__ float bf(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }

// the low and the high bf16 of a 32-bit word as floats
__device__ __forceinline__ float bf_lo(uint32_t bits) { return __uint_as_float(bits << 16); }

__device__ __forceinline__ float bf_hi(uint32_t bits) { return __uint_as_float(bits & 0xffff0000u); }

__device__ __forceinline__ float param(const void* p, int i, int is_f32) {
  return is_f32 ? reinterpret_cast<const float*>(p)[i]
                : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float softplus(float x) {
  return x > kSoftplusThreshold ? x : fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));
}

// x = hi + lo: hi rounded to tf32 (integer operations: cvt.rna.tf32 is emulated on sm_103), lo exact in fp32 (the MMA
// reads its top bits: |error| <= 2^-22 |x|)
__device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
  hi = (__float_as_uint(x) + kTf32Half) & kTf32Mask;
  lo = __float_as_uint(x - __uint_as_float(hi));
}

// d += a b, m16n8k8, tf32 operands, fp32 accumulators
__device__ __forceinline__ void mma_tf32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void fma4(float4& acc, float w, const float4& x) {
  acc.x = fmaf(w, x.x, acc.x);
  acc.y = fmaf(w, x.y, acc.y);
  acc.z = fmaf(w, x.z, acc.z);
  acc.w = fmaf(w, x.w, acc.w);
}

__device__ __forceinline__ float4 scale4(const float4& a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// f(integral_constant<K>) for the runtime count k in 0..M (one instance per count)
template <int M, int K = 0, typename F>
__device__ __forceinline__ void dispatch_count(int k, F&& f) {
  if constexpr (K <= M) {
    if (k == K) {
      f(std::integral_constant<int, K>{});
      return;
    }
    dispatch_count<M, K + 1>(k, f);
  }
}

// A row's step: tokens replayed (c), current (T), the record copy it reads
struct RowStep {
  int tok0, block_tokens, req, pos, c, T, L, copy;
};

__device__ __forceinline__ RowStep row_step(const Step& sp, int row) {
  const int* ri = sp.rows + (long long)row * kRowInts;
  RowStep r;
  r.tok0 = ri[0], r.block_tokens = ri[1], r.req = ri[2], r.pos = ri[3];
  r.c = sp.is_flush ? r.block_tokens : sp.replay[r.req];
  r.T = sp.is_flush ? 0 : r.block_tokens;
  r.copy = sp.is_flush ? 1 - sp.parity[r.req] : sp.parity[r.req];
  r.L = r.c + r.T;
  return r;
}

// The shared memory of one row's value head
template <int MT>
struct alignas(16) CoreSmem {
  float4 state[kStateChunks][kThreads];      // each thread's chunks of the head's state (its two rows, 16 columns each)
  float k[Sizes<MT>::L][kDim];               // the keys' conv outputs
  float q[MT][kDim];                         // the queries' conv outputs; then the outputs
  float v[Sizes<MT>::L + MT][kValStride];    // the values' conv outputs; then the corrections' and outputs' parts
  float vec_hi[Sizes<MT>::NT * 8][kVecStride];  // the products' vectors (effective keys, queries, zero rows), tf32 hi
  float vec_lo[Sizes<MT>::NT * 8][kVecStride];  // and lo
  float gram[Sizes<MT>::GRAM];               // key j's products with keys i <= j, query t's with keys i <= c + t, q.q
  float m[Sizes<MT>::L][Sizes<MT>::L];       // the keys' interactions in the delta rule (strictly lower)
  float nq[MT][Sizes<MT>::L];                // the queries' weights of the keys' corrections
  alignas(16) float ck[Sizes<MT>::N][Sizes<MT>::L];  // each vector's coefficients of the keys' conv outputs
  alignas(16) float cv[Sizes<MT>::N][Sizes<MT>::L];  // each vector's coefficients of the values (the rows' own parts)
  float cqn[Sizes<MT>::N];                   // each vector's coefficient of its query's conv outputs (0: none)
  int qi[Sizes<MT>::N];                      // and that query
  float wc[Sizes<MT>::L], wb[Sizes<MT>::L];  // the written and the boundary state's weights of each key's correction
  float g[Sizes<MT>::L], beta[Sizes<MT>::L]; // the log decays' prefix sums, the betas
  float rk[Sizes<MT>::L], rq[MT];            // the keys' inverse norms, the queries' scaled inverse norms
  float decay_c, decay_b;                    // the written and the boundary state's decays of the slot's state
  alignas(16) float red[MT][kWarps];         // the outputs' squared sums per warp
};

// ---- one row's value head: the replay of its c committed tokens, its block's T outputs (none in a flush), the state
// written once (and a boundary candidate's), this block's record
template <int MT>
__device__ __forceinline__ void core_item(const Layer& ly, const Step& sp, int row, int hv, int ih,
                                          CoreSmem<MT>& sm) {
  using S = Sizes<MT>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, tig = lane & 3;
  const RowStep r = row_step(sp, row);
  const int H = sp.H, HV = sp.HV;
  const int C = 2 * H * kDim + HV * kDim;
  const int c = r.c, T = r.T, L = r.L;
  const int* sl = sp.slots + ((long long)ly.group * sp.max_rows + row) * kSlotInts;
  const int dst = sl[1];
  const int src = sp.is_flush ? dst : sl[0];
  const int bnd_slot = sp.is_flush ? -1 : sl[2];
  // the boundary inside this block (not at its end) whose state is written: its token among the L
  const int bnd_j = bnd_slot >= 0 ? c + (r.pos / sp.block_size + 1) * sp.block_size - r.pos - 1 : -1;
  const int nk = max(c, bnd_j + 1);  // the effective keys: the tokens whose corrections a written state needs
  const int N = nk + T;
  const int nts = (N + 7) / 8;

  // ---- the head's state rows of this thread into shared memory, asynchronously: rows 16 warp + g8 (+ 8), columns
  // 16 i + 4 tig .. + 3 (the MMA's A fragments of the k-steps 2 i, 2 i + 1: the products are invariant to that
  // permutation of the columns, which the vectors' fragments follow)
  const long long rows_off = ((long long)hv * kDim + 16 * warp + g8) * kDim + 4 * tig;
  {
    const float* base = ly.state + (long long)src * ly.state_slot + rows_off;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < kColChunks; i++) cp_async16(&sm.state[kColChunks * h + i][tid], base + 8 * h * kDim + 16 * i);
    cp_async_commit();
  }
  // ---- the conv: threads < kDim take the key head's q and k channels, the others the value head's v channels; the
  // inputs of a channel: the window (the record's, or the slot's before any record), replayed, current. The window,
  // the replayed inputs and the weights load before the wait for the projections (a programmatic dependent overlaps
  // them with the GEMM), the current inputs after it
  const __nv_bfloat16* win = ly.conv + (long long)src * ly.conv_slot;
  const uint8_t* rec_in = ly.record + (long long)r.req * ly.record_req + (long long)r.copy * ly.record_copy;
  uint8_t* rec_out = ly.record + (long long)r.req * ly.record_req + (long long)(1 - r.copy) * ly.record_copy;
  const __nv_bfloat16* rin = reinterpret_cast<const __nv_bfloat16*>(rec_in);
  __nv_bfloat16* rout = reinterpret_cast<__nv_bfloat16*>(rec_out);
  const bool is_qk = tid < kDim;
  const int ct = tid % kDim;
  const int ch = is_qk ? (H + ih) * kDim + ct : 2 * H * kDim + hv * kDim + ct;  // k, or v
  const int ch_q = ih * kDim + ct;
  const bool has_record = !sp.is_flush && (!is_qk || hv == ih * sp.R);  // the key head's q / k by its first value head
  __nv_bfloat16 x[kTaps + S::L], xq[kTaps + MT];
  if (c > 0) {
#pragma unroll
    for (int e = 0; e < kTaps; e++) {
      x[e] = rin[(long long)e * C + ch];
      xq[e] = is_qk ? rin[(long long)(c + e) * C + ch_q] : __float2bfloat16(0.f);
    }
  } else {
#pragma unroll
    for (int e = 0; e < kTaps; e++) {
      x[e] = win[ch * ly.conv_ch + e * ly.conv_tap];
      xq[e] = is_qk ? win[ch_q * ly.conv_ch + e * ly.conv_tap] : __float2bfloat16(0.f);
    }
  }
#pragma unroll
  for (int e = kTaps; e < kTaps + S::L; e++) x[e] = e < kTaps + c ? rin[(long long)e * C + ch] : __float2bfloat16(0.f);
  const uint2 wbits = *reinterpret_cast<const uint2*>(ly.conv_w + 4 * ch);
  const uint2 wqbits = is_qk ? *reinterpret_cast<const uint2*>(ly.conv_w + 4 * ch_q) : make_uint2(0, 0);
  // the gate inputs of the replayed tokens (the last warp's lanes j < c), from the record
  __nv_bfloat16 gate_b = __float2bfloat16(0.f), gate_a = __float2bfloat16(0.f);
  if (warp == kWarps - 1 && lane < c) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(rec_in + sp.record_ba) + lane * 2 * HV;
    gate_b = p[hv], gate_a = p[HV + hv];
  }

  // a programmatic dependent (the verify core) waits for the projections' GEMM here, its state already loading (a
  // plain launch returns at once)
  pdl_wait();

  // the gate inputs of the outputs this thread normalizes (row tid % kDim of tokens tid / kDim + k kThreads / kDim)
  constexpr int kNormPasses = (MT * kDim + kThreads - 1) / kThreads;
  float z[kNormPasses];
#pragma unroll
  for (int k = 0; k < kNormPasses; k++) {
    const int p = tid + k * kThreads, t = p / kDim;
    z[k] = t < T ? bf(sp.qkvz, (long long)(r.tok0 + t) * sp.qkvz_row + C + hv * kDim + p % kDim) : 0.f;
  }
  // the current inputs: entry e >= kTaps + c is the step's row e - kTaps - c
  const __nv_bfloat16* cur = sp.qkvz + (long long)(r.tok0 - kTaps - c) * sp.qkvz_row;
#pragma unroll
  for (int e = kTaps; e < kTaps + S::L; e++)
    if (e >= kTaps + c && e < kTaps + L) x[e] = cur[(long long)e * sp.qkvz_row + ch];
  if (is_qk) {
#pragma unroll
    for (int e = kTaps; e < kTaps + MT; e++)
      xq[e] = e < kTaps + T ? cur[(long long)(c + e) * sp.qkvz_row + ch_q] : __float2bfloat16(0.f);
  }
  if (warp == kWarps - 1 && lane >= c && lane < L) {
    const __nv_bfloat16* p = sp.ba + (long long)(r.tok0 + lane - c) * sp.ba_row;
    gate_b = p[hv], gate_a = p[HV + hv];
  }

  {
    auto conv = [](const float (&w)[4], __nv_bfloat16 x0, __nv_bfloat16 x1, __nv_bfloat16 x2, __nv_bfloat16 x3) {
      float acc = w[0] * __bfloat162float(x0);
      acc = fmaf(w[1], __bfloat162float(x1), acc);
      acc = fmaf(w[2], __bfloat162float(x2), acc);
      acc = fmaf(w[3], __bfloat162float(x3), acc);
      return acc * sigmoid(acc);
    };
    const float w[4] = {bf_lo(wbits.x), bf_hi(wbits.x), bf_lo(wbits.y), bf_hi(wbits.y)};
    const float wq[4] = {bf_lo(wqbits.x), bf_hi(wqbits.x), bf_lo(wqbits.y), bf_hi(wqbits.y)};
    float* dst_kv = is_qk ? &sm.k[0][ct] : &sm.v[0][ct];
    const int stride_kv = is_qk ? kDim : kValStride;
#pragma unroll
    for (int j = 0; j < S::L; j++) dst_kv[j * stride_kv] = j < L ? conv(w, x[j], x[j + 1], x[j + 2], x[j + 3]) : 0.f;
    // the rows past the block's zero: every vector's build reads query 0's row (a flush has no query)
    if (is_qk) {
#pragma unroll
      for (int t = 0; t < MT; t++) sm.q[t][ct] = t < T ? conv(wq, xq[t], xq[t + 1], xq[t + 2], xq[t + 3]) : 0.f;
    }
    // this block's record: the inputs from the window after the replayed tokens on
    if (has_record) {
      __nv_bfloat16* rout_c = rout - (long long)c * C;  // input e's entry e - c
#pragma unroll
      for (int e = 0; e < kTaps + S::L; e++)
        if (e >= c && e < c + kTaps + T) rout_c[(long long)e * C + ch] = x[e];
      if (is_qk) {
#pragma unroll
        for (int e = 0; e < kTaps + MT; e++)
          if (e < kTaps + T) rout[(long long)e * C + ch_q] = xq[e];
      }
    }
  }
  // the gates of the L tokens (the last warp's lanes): log decays' prefix sums, betas; this block's b / a into its
  // record
  if (warp == kWarps - 1) {
    const int j = lane;
    float gj = 0.f;
    if (j < L) {
      const __nv_bfloat16 b = gate_b, a = gate_a;
      gj = -__expf(ly.a_log[hv]) * softplus(__bfloat162float(a) + param(ly.dt_bias, hv, ly.dt_bias_f32));
      sm.beta[j] = sigmoid(__bfloat162float(b));
      if (!sp.is_flush && j >= c) {
        __nv_bfloat16* rout_ba = reinterpret_cast<__nv_bfloat16*>(rec_out + sp.record_ba) + (j - c) * 2 * HV;
        rout_ba[hv] = b;
        rout_ba[HV + hv] = a;
      }
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float up = __shfl_up_sync(kFullMask, gj, o);
      if (lane >= o) gj += up;
    }
    if (j < L) sm.g[j] = gj;
  }
  __syncthreads();

  // ---- the step's vector products (kGramSplit threads each): key a with keys b <= a, query t with keys b <= c + t,
  // each query with itself
  for (int base = 0; base < kGramSplit * S::GRAM; base += kThreads) {
    const int p = base + tid, e = p / kGramSplit, part = p % kGramSplit;
    int a = 0, b = 0;
    bool is_needed = false;
    if (e < S::GRAM - MT) {
      a = e / S::L, b = e % S::L;
      is_needed = a < S::L ? a < L && b <= a : a - S::L < T && b <= c + a - S::L;
    } else if (e < S::GRAM) {
      a = S::L + e - (S::GRAM - MT), b = -1;
      is_needed = a - S::L < T;
    }
    float acc = 0.f;
    if (is_needed) {
      const float* x = a < S::L ? sm.k[a] : sm.q[a - S::L];
      const float* y = b >= 0 ? sm.k[b] : x;
      constexpr int kSpan = kDim / kGramSplit / 4;  // a part's float4
#pragma unroll 8
      for (int s = 0; s < kSpan; s++) {
        const int col = kDim / kGramSplit * part + 4 * ((s + p) % kSpan);
        const float4 xv = *reinterpret_cast<const float4*>(x + col);
        const float4 yv = *reinterpret_cast<const float4*>(y + col);
        acc = fmaf(xv.x, yv.x, acc);
        acc = fmaf(xv.y, yv.y, acc);
        acc = fmaf(xv.z, yv.z, acc);
        acc = fmaf(xv.w, yv.w, acc);
      }
    }
#pragma unroll
    for (int o = 1; o < kGramSplit; o <<= 1) acc += __shfl_xor_sync(kFullMask, acc, o);
    if (is_needed && part == 0) sm.gram[e] = acc;
  }
  __syncthreads();

  // ---- the head's coefficients (the first warp): the norms, the keys' interactions M (j > i) and the queries'
  // weights; A = (I + M)^-1 diag(beta) column by column (lane l: column l); the effective vectors' coefficients
  if (warp == 0) {
    auto gram_kk = [&](int j, int i) { return sm.gram[j * S::L + i]; };
    auto gram_qk = [&](int t, int i) { return sm.gram[(S::L + t) * S::L + i]; };
    if (lane < L) sm.rk[lane] = rsqrtf(gram_kk(lane, lane) + kL2Eps);
    if (lane < T) sm.rq[lane] = sp.scale * rsqrtf(sm.gram[S::GRAM - MT + lane] + kL2Eps);
    __syncwarp();
    for (int p = lane; p < S::L * S::L; p += 32) {
      const int j = p / S::L, i = p % S::L;
      if (i < j && j < L)
        sm.m[j][i] = sm.beta[j] * __expf(sm.g[j] - sm.g[i]) * sm.rk[j] * sm.rk[i] * gram_kk(j, i);
    }
    for (int p = lane; p < MT * S::L; p += 32) {
      const int t = p / S::L, i = p % S::L;
      if (t < T) sm.nq[t][i] = i <= c + t ? __expf(sm.g[c + t] - sm.g[i]) * sm.rq[t] * sm.rk[i] * gram_qk(t, i) : 0.f;
    }
    __syncwarp();

    const int l = lane;
    // M's strictly lower part in registers (its loads issued together), then the forward substitution
    float mr[S::L][S::L];
#pragma unroll
    for (int j = 1; j < S::L; j++)
#pragma unroll
      for (int i = 0; i < j; i++) mr[j][i] = j < L ? sm.m[j][i] : 0.f;
    float col[S::L], acc[S::L];
#pragma unroll
    for (int j = 0; j < S::L; j++) acc[j] = j == l ? sm.beta[l] : 0.f;
#pragma unroll
    for (int i = 0; i < S::L; i++) {
      col[i] = i < L ? acc[i] : 0.f;
#pragma unroll
      for (int j = i + 1; j < S::L; j++) acc[j] = fmaf(-mr[j][i], col[i], acc[j]);
    }
    if (l < S::L) {
      // key l's decay from the slot's state, with its norm; the columns past the step's tokens and the rows past its
      // vectors zero (the vectors' and the own parts' loops run over all of them)
      const float ek = l < L ? __expf(sm.g[l]) * sm.rk[l] : 0.f;
#pragma unroll
      for (int n = 0; n < S::L; n++) {
        if (n < nk) {
          sm.ck[n][l] = col[n] * ek;
          sm.cv[n][l] = l < L ? col[n] : 0.f;
        }
      }
#pragma unroll
      for (int t = 0; t < MT; t++) {
        if (t < T) {
          float na = 0.f;
#pragma unroll
          for (int i = 0; i < S::L; i++) na = fmaf(sm.nq[t][i], col[i], na);
          sm.ck[nk + t][l] = -na * ek;
          sm.cv[nk + t][l] = l < L ? na : 0.f;
        }
      }
#pragma unroll
      for (int n = 0; n < S::N; n++)
        if (n >= N) sm.ck[n][l] = sm.cv[n][l] = 0.f;
      if (l < c) sm.wc[l] = __expf(sm.g[c - 1] - sm.g[l]) * sm.rk[l];
      if (l <= bnd_j) sm.wb[l] = __expf(sm.g[bnd_j] - sm.g[l]) * sm.rk[l];
    }
    if (l < S::N) {
      // vector l's query term: e^{G_{c+t}} rq_t q_t for the effective query t = l - nk, none for the others
      const int t = l - nk;
      sm.cqn[l] = t >= 0 && t < T ? __expf(sm.g[c + t]) * sm.rq[t] : 0.f;
      sm.qi[l] = t >= 0 && t < T ? t : 0;
    }
    if (l == 0) {
      sm.decay_c = c > 0 ? __expf(sm.g[c - 1]) : 1.f;
      sm.decay_b = bnd_j >= 0 ? __expf(sm.g[bnd_j]) : 0.f;
    }
  }
  __syncthreads();

  // ---- the effective vectors split to tf32 (threads < kDim: one column each) and the rows' own parts of the
  // corrections and the outputs (the others: one value channel each, in place): every row, zero coefficients past the
  // step's (no branch between the loads)
  if (is_qk) {
    float kv[S::L];
#pragma unroll
    for (int j = 0; j < S::L; j++) kv[j] = sm.k[j][ct];
    // the n-tiles' rows of the products' vectors (zero coefficients past the step's): one instance per n-tile count
    auto build = [&](auto tiles) {
      constexpr int ROWS = 8 * decltype(tiles)::value;
#pragma unroll S::UNROLL_VEC
      for (int n = 0; n < ROWS; n++) {
        float acc = 0.f;
        if (n < S::N) {
#pragma unroll
          for (int j = 0; j < S::L; j += 4) {
            const float4 w = *reinterpret_cast<const float4*>(&sm.ck[n][j]);
            acc = fmaf(w.x, kv[j], fmaf(w.y, kv[j + 1], fmaf(w.z, kv[j + 2], fmaf(w.w, kv[j + 3], acc))));
          }
          acc = fmaf(sm.cqn[n], sm.q[sm.qi[n]][ct], acc);
        }
        uint32_t hi, lo;
        split_tf32(acc, hi, lo);
        sm.vec_hi[n][ct] = __uint_as_float(hi);
        sm.vec_lo[n][ct] = __uint_as_float(lo);
      }
    };
    if (nts == 1) {
      build(std::integral_constant<int, 1>{});
    } else {
      build(std::integral_constant<int, S::NT>{});
    }
  } else {
    float vv[S::L];
#pragma unroll
    for (int j = 0; j < S::L; j++) vv[j] = j < L ? sm.v[j][ct] : 0.f;
    // the rows' own parts of the step's vectors: one instance per n-tile count
    auto own = [&](auto tiles) {
      constexpr int ROWS = 8 * decltype(tiles)::value < S::N ? 8 * decltype(tiles)::value : S::N;
#pragma unroll S::UNROLL_VAL
      for (int n = 0; n < ROWS; n++) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < S::L; j += 4) {
          const float4 w = *reinterpret_cast<const float4*>(&sm.cv[n][j]);
          acc = fmaf(w.x, vv[j], fmaf(w.y, vv[j + 1], fmaf(w.z, vv[j + 2], fmaf(w.w, vv[j + 3], acc))));
        }
        if (n < N) sm.v[n < nk ? n : S::L + n - nk][ct] = acc;
      }
    };
    if (nts == 1) {
      own(std::integral_constant<int, 1>{});
    } else {
      own(std::integral_constant<int, S::NT>{});
    }
  }
  __syncthreads();

  // ---- the rows' products with the vectors: 3xTF32 MMAs (hi hi + hi lo + lo hi), the state's chunks from shared
  // memory
  cp_async_wait_all();
  // (one n-tile at a time, the even and the odd k-steps in chains of their own)
  float pv[S::NT][4];
#pragma unroll
  for (int nt = 0; nt < S::NT; nt++) {
    float acc[2][4] = {};
    if (nt < nts) {
#pragma unroll
      for (int i = 0; i < kColChunks; i++) {
        uint32_t ah[8], al[8];
        const float4 s0 = sm.state[i][tid], s1 = sm.state[kColChunks + i][tid];
        split_tf32(s0.x, ah[0], al[0]);
        split_tf32(s1.x, ah[1], al[1]);
        split_tf32(s0.y, ah[2], al[2]);
        split_tf32(s1.y, ah[3], al[3]);
        split_tf32(s0.z, ah[4], al[4]);
        split_tf32(s1.z, ah[5], al[5]);
        split_tf32(s0.w, ah[6], al[6]);
        split_tf32(s1.w, ah[7], al[7]);
        const float4 bhv = *reinterpret_cast<const float4*>(&sm.vec_hi[8 * nt + g8][16 * i + 4 * tig]);
        const float4 blv = *reinterpret_cast<const float4*>(&sm.vec_lo[8 * nt + g8][16 * i + 4 * tig]);
        const uint32_t bh[4] = {__float_as_uint(bhv.x), __float_as_uint(bhv.y), __float_as_uint(bhv.z),
                                __float_as_uint(bhv.w)};
        const uint32_t bl[4] = {__float_as_uint(blv.x), __float_as_uint(blv.y), __float_as_uint(blv.z),
                                __float_as_uint(blv.w)};
        mma_tf32(acc[0], al[0], al[1], al[2], al[3], bh[0], bh[1]);
        mma_tf32(acc[1], al[4], al[5], al[6], al[7], bh[2], bh[3]);
        mma_tf32(acc[0], ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
        mma_tf32(acc[1], ah[4], ah[5], ah[6], ah[7], bl[2], bl[3]);
        mma_tf32(acc[0], ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
        mma_tf32(acc[1], ah[4], ah[5], ah[6], ah[7], bh[2], bh[3]);
      }
    }
    // D: rows 16 warp + g8 (+ 8 for e >= 2), vectors 8 nt + 2 tig (+ 1 for odd e); the corrections u (n < nk) and the
    // outputs (nk <= n < N) from the rows' own parts
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int n = 8 * nt + 2 * tig + (e & 1), rr = 16 * warp + g8 + 8 * (e >> 1);
      const float a = acc[0][e] + acc[1][e];
      pv[nt][e] = n < nk ? sm.v[n][rr] - a : n < N ? sm.v[S::L + n - nk][rr] + a : 0.f;
    }
  }

  // ---- the corrections of the thread's two rows (from the lanes of its quad that hold them)
  float u[2][S::L];
#pragma unroll
  for (int i = 0; i < S::L; i++) {
    const int src_lane = (lane & ~3) | ((i & 7) >> 1);
    u[0][i] = __shfl_sync(kFullMask, pv[i >> 3][i & 1], src_lane);
    u[1][i] = __shfl_sync(kFullMask, pv[i >> 3][2 + (i & 1)], src_lane);
  }

  // ---- the boundary block's state, then the state at the start of this block, each written once
  if (bnd_j >= 0) {
    float wu[2][S::L];
#pragma unroll
    for (int i = 0; i < S::L; i++) {
      const float w = i <= bnd_j ? sm.wb[i] : 0.f;
      wu[0][i] = u[0][i] * w, wu[1][i] = u[1][i] * w;
    }
    const float eb = sm.decay_b;
    float* bdst = ly.state + (long long)bnd_slot * ly.state_slot + rows_off;
#pragma unroll
    for (int i = 0; i < kColChunks; i++) {
      float4 sb[2] = {scale4(sm.state[i][tid], eb), scale4(sm.state[kColChunks + i][tid], eb)};
#pragma unroll
      for (int j = 0; j < S::L; j++) {
        const float4 kv = *reinterpret_cast<const float4*>(&sm.k[j][16 * i + 4 * tig]);
        fma4(sb[0], wu[0][j], kv);
        fma4(sb[1], wu[1][j], kv);
      }
      __stcs(reinterpret_cast<float4*>(bdst + 16 * i), sb[0]);
      __stcs(reinterpret_cast<float4*>(bdst + 8 * kDim + 16 * i), sb[1]);
    }
  }
  if (c > 0 || src != dst) {
    // the running state, one instance per replay count (the corrections' loop without padding or branches)
    const float ec = sm.decay_c;
    float* sdst = ly.state + (long long)dst * ly.state_slot + rows_off;
    auto write_running = [&](auto count) {
      constexpr int K = decltype(count)::value;
      float wu[2][K > 0 ? K : 1];
#pragma unroll
      for (int i = 0; i < K; i++) wu[0][i] = u[0][i] * sm.wc[i], wu[1][i] = u[1][i] * sm.wc[i];
#pragma unroll
      for (int i = 0; i < kColChunks; i++) {
        float4 sc[2] = {scale4(sm.state[i][tid], ec), scale4(sm.state[kColChunks + i][tid], ec)};
#pragma unroll
        for (int j = 0; j < K; j++) {
          const float4 kv = *reinterpret_cast<const float4*>(&sm.k[j][16 * i + 4 * tig]);
          fma4(sc[0], wu[0][j], kv);
          fma4(sc[1], wu[1][j], kv);
        }
        __stcs(reinterpret_cast<float4*>(sdst + 16 * i), sc[0]);
        __stcs(reinterpret_cast<float4*>(sdst + 8 * kDim + 16 * i), sc[1]);
      }
    };
    dispatch_count<MT>(c, write_running);
  }
  if (T == 0) return;

  // ---- the gated RMSNorm of each output token over the head's rows: the outputs to shared memory (over the queries)
  // with each warp's squared sums, then each token's row
#pragma unroll
  for (int nt = 0; nt < S::NT; nt++) {
#pragma unroll
    for (int odd = 0; odd < 2; odd++) {
      const int n = 8 * nt + 2 * tig + odd;
      float s = pv[nt][odd] * pv[nt][odd] + pv[nt][2 + odd] * pv[nt][2 + odd];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) s += __shfl_xor_sync(kFullMask, s, o);
      if (n >= nk && n < N) {
        sm.q[n - nk][16 * warp + g8] = pv[nt][odd];
        sm.q[n - nk][16 * warp + g8 + 8] = pv[nt][2 + odd];
        if (g8 == 0) sm.red[n - nk][warp] = s;
      }
    }
  }
  __syncthreads();
  const float nw = param(ly.norm_w, ct, ly.norm_w_f32);  // the weight of this thread's output row
#pragma unroll
  for (int k = 0; k < kNormPasses; k++) {
    const int p = tid + k * kThreads, t = p / kDim, rr = p % kDim;
    if (t >= T) break;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; w += 4) {
      const float4 part = *reinterpret_cast<const float4*>(&sm.red[t][w]);
      s += (part.x + part.y) + (part.z + part.w);
    }
    const float rstd = rsqrtf(s / kDim + sp.norm_eps);
    const float gate = sp.gate_sigmoid ? sigmoid(z[k]) : z[k] * sigmoid(z[k]);
    sp.out[((long long)(r.tok0 + t) * HV + hv) * kDim + rr] = __float2bfloat16(sm.q[t][rr] * rstd * nw * gate);
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads, 2) gdn_verify_kernel(const Layer ly, const Step sp) {
  extern __shared__ float4 smem_raw[];
  CoreSmem<MT>& sm = *reinterpret_cast<CoreSmem<MT>*>(smem_raw);
  const int row = blockIdx.x, ih = blockIdx.z, hv = ih * sp.R + blockIdx.y;
  // the next kernel (out_proj's GEMM, a programmatic dependent) launches once every CTA has started: it streams its
  // weights into the SMs this grid's tail frees and waits for this grid's end before it reads the output
  pdl_launch_next();
  if (sp.rows[(long long)row * kRowInts + 1] == 0) return;  // not a verify row (a padding or prefill row)
  core_item<MT>(ly, sp, row, hv, ih, sm);
}

// ---- verify blocks of up to kItemT tokens: a prep kernel per row and key head writes each value head's item (the
// delta rule's effective vectors and coefficients), then a persistent stream kernel runs the items: each value head's
// state in registers, its products, own parts and rank update on tensor cores, the next item's state loading meanwhile
constexpr int kItemT = 4;                   // tokens of a verify block at most on this path
constexpr int kItemL = 2 * kItemT;          // the replayed and the current tokens of a step at most
constexpr int kItemTiles = kItemL / 4;      // n-tiles of the products: tile i holds keys 4 i .. 4 i + 3 in its even slots
constexpr int kSlots = 8;                   // vectors of an n-tile
constexpr int kItemSlots = kItemTiles * kSlots;
constexpr int kMaxRatio = 3;                // value heads per key head on this path
constexpr int kPrepPairs = kDim / 2;        // the prep's threads per head of channels: two adjacent channels each
constexpr int kPrepThreads = kPrepPairs * (1 + kMaxRatio);
constexpr int kPrepGram = (kItemL + kItemT) * kItemL + kItemT;  // the Gram's entries (as Sizes<kItemT>::GRAM)
constexpr int kGramLanes = 4;               // threads per Gram entry
constexpr int kStreamCtasPerSm = 2;
constexpr int kItemVecStride = kDim + 16;   // smem row stride of the vectors (conflict-free float4 fragment loads)
constexpr int kItemValStride = kDim + 8;    // smem row stride of the values (conflict-free fragment loads)

// A row's value head, written by the prep kernel for the stream kernel
struct alignas(16) VerifyItem {
  float vec[kItemSlots][kDim];   // the products' vectors by slot: -effective key i at 8 (i / 4) + 2 (i % 4), effective
                                 // query t at 2 t + 1, zero in the other slots of the tiles in use
  float v[kItemL][kDim];         // the values' conv outputs by token (zero past the step's tokens)
  float cvf[kItemTiles][32][2];  // the own parts' B fragments: tile, lane 4 g + tig: values tig and tig + 4 in slot g
  float wc[kItemL], wb[kItemL];  // the running and the boundary state's weights of each key's correction
  float decay_c, decay_b;        // the running and the boundary state's decays of the slot's state
  int c, nts, bnd_j, pad[3];     // replayed tokens, n-tiles in use, the boundary's token (-1: none)
};

// A row's key head: the rank update's B fragments, normalized keys (group i of 16 columns, lane 4 g + tig: keys tig and
// tig + 4 of n-tiles 2 i and 2 i + 1 at column 16 i + 4 (g / 2) + 2 (tile % 2) + g % 2)
struct alignas(16) KeyHeadItem {
  float kb[kColChunks][32][4];
};

template <int MT>
struct alignas(16) PrepSmem {
  float k[kItemL][kDim];                          // the keys' conv outputs
  float q[kItemT][kDim];                          // the queries' conv outputs
  float gram[kPrepGram];                          // key j's products with keys i <= j, query t's with keys i <= c + t, q.q
  float g[kMaxRatio][kItemL], beta[kMaxRatio][kItemL];  // each value head's log decays' prefix sums and betas
  float rk[kItemL];                               // the keys' inverse norms
  float m[kMaxRatio][kItemL][kItemL];             // each value head's key interactions in the delta rule (strictly lower)
  float nq[kMaxRatio][kItemT][kItemL];            // each value head's query weights of the keys' corrections
  alignas(16) float ck[kMaxRatio][kItemSlots][kItemL];  // each slot's coefficients of the keys' conv outputs
  float cq[kMaxRatio][kItemSlots];                // each slot's coefficient of its query's conv outputs (0: none)
};

// An item's step data in the stream kernel's shared memory (cvf to pad as in VerifyItem: one copy)
struct alignas(16) StreamData {
  float vec[kItemSlots][kItemVecStride];
  float v[kItemL][kItemValStride];
  float kb[kColChunks][32][4];
  float cvf[kItemTiles][32][2];
  float wc[kItemL], wb[kItemL];
  float decay_c, decay_b;
  int c, nts, bnd_j, pad[3];
};

struct alignas(16) StreamSmem {
  float4 state[kStateChunks][kThreads];  // the next item's state: each thread's chunks (its two rows, 16 columns each)
  StreamData data[2];                    // the current and the next item's step data
  float red[kItemT][kWarps];             // the outputs' squared sums per warp
};

constexpr int kItemTailBytes = sizeof(VerifyItem) - offsetof(VerifyItem, cvf);  // cvf to pad: one 16-byte-chunk copy

__device__ __forceinline__ void mma_tf32_k4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}

// d += a b in 3xTF32 (lo hi + hi lo + hi hi), m16n8k8
__device__ __forceinline__ void mma3_tf32(float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                          uint32_t bh1, uint32_t bl0, uint32_t bl1) {
  mma_tf32(d, al[0], al[1], al[2], al[3], bh0, bh1);
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

// One row and key head of a verify step: the conv of the key head's q / k channels and its value heads' v channels
// (this block's record written), the value heads' gates, the Gram of the step's keys and queries; per value head the
// delta rule's coefficients and effective vectors into its item; the normalized keys as the rank update's fragments
__global__ void __launch_bounds__(kPrepThreads) gdn_prep_kernel(const Layer ly, const Step sp, VerifyItem* items,
                                                                KeyHeadItem* key_items) {
  extern __shared__ float4 smem_raw[];
  PrepSmem<kItemT>& sm = *reinterpret_cast<PrepSmem<kItemT>*>(smem_raw);
  constexpr int L8 = kItemL;
  const int row = blockIdx.x, ih = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // the stream kernel loads its first states while this grid runs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int4 ri = *reinterpret_cast<const int4*>(sp.rows + (long long)row * kRowInts);
  const int T = ri.y;
  if (T == 0) return;
  const int tok0 = ri.x, req = ri.z, pos = ri.w;
  const int4 sl = *reinterpret_cast<const int4*>(sp.slots + ((long long)ly.group * sp.max_rows + row) * kSlotInts);
  const int src = sl.x, bnd_slot = sl.z;
  const int c = sp.replay[req], copy = sp.parity[req];
  const int L = c + T;
  const int H = sp.H, HV = sp.HV, R = sp.R;
  const int C = 2 * H * kDim + HV * kDim, C2 = C / 2;
  const int bnd_j = bnd_slot >= 0 ? c + (pos / sp.block_size + 1) * sp.block_size - pos - 1 : -1;
  const int nk = max(c, bnd_j + 1);
  const int nts = nk > 4 ? 2 : 1;

  // ---- the conv inputs of the thread's channel pair: kind 0 the key head's k (and q) channels, kind 1 + h value head
  // h's v channels; the window (the record's, or the slot's before any record), replayed, current
  const int kind = tid / kPrepPairs, ct = 2 * (tid % kPrepPairs);
  const int hv_kind = ih * R + kind - 1;
  const int ch = kind == 0 ? (H + ih) * kDim + ct : 2 * H * kDim + hv_kind * kDim + ct;
  const int ch_q = ih * kDim + ct;
  const uint8_t* rec_in = ly.record + (long long)req * ly.record_req + (long long)copy * ly.record_copy;
  uint8_t* rec_out = ly.record + (long long)req * ly.record_req + (long long)(1 - copy) * ly.record_copy;
  const uint32_t* rin = reinterpret_cast<const uint32_t*>(rec_in);
  uint32_t* rout = reinterpret_cast<uint32_t*>(rec_out);
  const __nv_bfloat16* win = ly.conv + (long long)src * ly.conv_slot;
  auto win_pair = [&](int chan, int e) {
    const __nv_bfloat16 lo = win[chan * ly.conv_ch + e * ly.conv_tap], hi = win[(chan + 1) * ly.conv_ch + e * ly.conv_tap];
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
  };
  uint32_t x[kTaps + kItemL], xq[kTaps + kItemT];
#pragma unroll
  for (int e = 0; e < kTaps; e++) {
    x[e] = c > 0 ? rin[e * C2 + ch / 2] : win_pair(ch, e);
    xq[e] = kind != 0 ? 0u : c > 0 ? rin[(c + e) * C2 + ch_q / 2] : win_pair(ch_q, e);
  }
#pragma unroll
  for (int e = kTaps; e < kTaps + kItemL; e++) x[e] = e < kTaps + c ? rin[e * C2 + ch / 2] : 0u;
  const uint4 wbits = *reinterpret_cast<const uint4*>(ly.conv_w + 4 * ch);
  const uint4 wqbits = kind == 0 ? *reinterpret_cast<const uint4*>(ly.conv_w + 4 * ch_q) : make_uint4(0, 0, 0, 0);
  // the gates of value head h by the first warp of kind 1 + h (lane j: token j), the replayed tokens' from the record
  const bool is_gate_warp = kind >= 1 && kind <= R && tid % kPrepPairs < 32;
  const int hv_gate = hv_kind;
  __nv_bfloat16 gate_b = __float2bfloat16(0.f), gate_a = __float2bfloat16(0.f);
  if (is_gate_warp && lane < c) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(rec_in + sp.record_ba) + lane * 2 * HV;
    gate_b = p[hv_gate], gate_a = p[HV + hv_gate];
  }

  // the projections' GEMM (this kernel is its programmatic dependent) ends here
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const __nv_bfloat16* cur = sp.qkvz + (long long)(tok0 - kTaps - c) * sp.qkvz_row;  // entry e >= kTaps + c
#pragma unroll
  for (int e = kTaps; e < kTaps + kItemL; e++)
    if (e >= kTaps + c && e < kTaps + L) x[e] = *reinterpret_cast<const uint32_t*>(cur + (long long)e * sp.qkvz_row + ch);
  if (kind == 0) {
#pragma unroll
    for (int e = kTaps; e < kTaps + kItemT; e++)
      xq[e] = e < kTaps + T ? *reinterpret_cast<const uint32_t*>(cur + (long long)(c + e) * sp.qkvz_row + ch_q) : 0u;
  }
  if (is_gate_warp && lane >= c && lane < L) {
    const __nv_bfloat16* p = sp.ba + (long long)(tok0 + lane - c) * sp.ba_row;
    gate_b = p[hv_gate], gate_a = p[HV + hv_gate];
  }

  {
    auto conv = [](const float (&w)[4], uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, bool is_hi) {
      auto in = [is_hi](uint32_t v) { return is_hi ? bf_hi(v) : bf_lo(v); };
      float acc = w[0] * in(x0);
      acc = fmaf(w[1], in(x1), acc);
      acc = fmaf(w[2], in(x2), acc);
      acc = fmaf(w[3], in(x3), acc);
      return acc * sigmoid(acc);
    };
    const float w0[4] = {bf_lo(wbits.x), bf_hi(wbits.x), bf_lo(wbits.y), bf_hi(wbits.y)};
    const float w1[4] = {bf_lo(wbits.z), bf_hi(wbits.z), bf_lo(wbits.w), bf_hi(wbits.w)};
    auto pair = [&](const float (&wa)[4], const float (&wb)[4], const uint32_t* xs) {
      return make_float2(conv(wa, xs[0], xs[1], xs[2], xs[3], false), conv(wb, xs[0], xs[1], xs[2], xs[3], true));
    };
    if (kind == 0) {
      const float wq0[4] = {bf_lo(wqbits.x), bf_hi(wqbits.x), bf_lo(wqbits.y), bf_hi(wqbits.y)};
      const float wq1[4] = {bf_lo(wqbits.z), bf_hi(wqbits.z), bf_lo(wqbits.w), bf_hi(wqbits.w)};
#pragma unroll
      for (int j = 0; j < kItemL; j++)
        *reinterpret_cast<float2*>(&sm.k[j][ct]) = j < L ? pair(w0, w1, x + j) : make_float2(0.f, 0.f);
#pragma unroll
      for (int t = 0; t < kItemT; t++)
        *reinterpret_cast<float2*>(&sm.q[t][ct]) = t < T ? pair(wq0, wq1, xq + t) : make_float2(0.f, 0.f);
    } else if (kind <= R) {
      VerifyItem* it = items + (long long)row * HV + hv_kind;
#pragma unroll
      for (int j = 0; j < kItemL; j++)
        *reinterpret_cast<float2*>(&it->v[j][ct]) = j < L ? pair(w0, w1, x + j) : make_float2(0.f, 0.f);
    }
    // this block's record: the inputs from its window on
    if (kind <= R) {
      uint32_t* rout_c = rout - c * C2;  // input e's entry e - c
#pragma unroll
      for (int e = 0; e < kTaps + kItemL; e++)
        if (e >= c && e < c + kTaps + T) rout_c[e * C2 + ch / 2] = x[e];
      if (kind == 0) {
#pragma unroll
        for (int e = 0; e < kTaps + kItemT; e++)
          if (e < kTaps + T) rout[e * C2 + ch_q / 2] = xq[e];
      }
    }
  }
  if (is_gate_warp) {
    const int h = kind - 1, j = lane;
    float gj = 0.f;
    if (j < L) {
      gj = -__expf(ly.a_log[hv_gate]) * softplus(__bfloat162float(gate_a) + param(ly.dt_bias, hv_gate, ly.dt_bias_f32));
      sm.beta[h][j] = sigmoid(__bfloat162float(gate_b));
      if (j >= c) {
        __nv_bfloat16* rout_ba = reinterpret_cast<__nv_bfloat16*>(rec_out + sp.record_ba) + (j - c) * 2 * HV;
        rout_ba[hv_gate] = gate_b;
        rout_ba[HV + hv_gate] = gate_a;
      }
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float up = __shfl_up_sync(kFullMask, gj, o);
      if (lane >= o) gj += up;
    }
    if (j < kItemL) sm.g[h][j] = gj;
  }
  __syncthreads();

  // ---- the step's vector products (kGramLanes threads each): key a with keys b <= a, query t with keys b <= c + t,
  // each query with itself
  for (int base = 0; base < kGramLanes * kPrepGram; base += kPrepThreads) {
    const int p = base + tid, e = p / kGramLanes, part = p % kGramLanes;
    int a = 0, b = 0;
    bool is_needed = false;
    if (e < kPrepGram - kItemT) {
      a = e / L8, b = e % L8;
      is_needed = a < L8 ? a < L && b <= a : a - L8 < T && b <= c + a - L8;
    } else if (e < kPrepGram) {
      a = L8 + e - (kPrepGram - kItemT), b = -1;
      is_needed = a - L8 < T;
    }
    float acc = 0.f;
    if (is_needed) {
      const float* xv = a < L8 ? sm.k[a] : sm.q[a - L8];
      const float* yv = b >= 0 ? sm.k[b] : xv;
      constexpr int kSpan = kDim / kGramLanes / 4;  // a part's float4
#pragma unroll
      for (int s = 0; s < kSpan; s++) {
        const int col = kDim / kGramLanes * part + 4 * ((s + p) % kSpan);
        const float4 u = *reinterpret_cast<const float4*>(xv + col);
        const float4 w = *reinterpret_cast<const float4*>(yv + col);
        acc = fmaf(u.x, w.x, acc);
        acc = fmaf(u.y, w.y, acc);
        acc = fmaf(u.z, w.z, acc);
        acc = fmaf(u.w, w.w, acc);
      }
    }
#pragma unroll
    for (int o = 1; o < kGramLanes; o <<= 1) acc += __shfl_xor_sync(kFullMask, acc, o);
    if (is_needed && part == 0) sm.gram[e] = acc;
  }
  __syncthreads();

  // ---- each value head's coefficients (warp h): the norms, the keys' interactions M (j > i) and the queries'
  // weights N; A = (I + M)^-1 diag(beta) column by column (lane l: column l); the slots' coefficients
  if (warp < R) {
    const int h = warp, l = lane;
    VerifyItem* it = items + (long long)row * HV + ih * R + h;
    auto gram_kk = [&](int j, int i) { return sm.gram[j * L8 + i]; };
    auto gram_qk = [&](int t, int i) { return sm.gram[(L8 + t) * L8 + i]; };
    const float* gh = sm.g[h];
    const float* bh = sm.beta[h];
    const float rk_l = l < L ? rsqrtf(gram_kk(l, l) + kL2Eps) : 0.f;
    const float rq_l = l < T ? sp.scale * rsqrtf(sm.gram[kPrepGram - kItemT + l] + kL2Eps) : 0.f;
    if (h == 0 && l < kItemL) sm.rk[l] = rk_l;
    // M's strictly lower part (two entries per lane) and N (one per lane), the norms by shuffles
#pragma unroll
    for (int p = l; p < kItemL * kItemL; p += 32) {
      const int j = p / kItemL, i = p % kItemL;
      const float rk_j = __shfl_sync(kFullMask, rk_l, j), rk_i = __shfl_sync(kFullMask, rk_l, i);
      sm.m[h][j][i] = i < j && j < L ? bh[j] * __expf(gh[j] - gh[i]) * rk_j * rk_i * gram_kk(j, i) : 0.f;
    }
    {
      const int t = l / kItemL, i = l % kItemL;
      const float rq_t = __shfl_sync(kFullMask, rq_l, t), rk_i = __shfl_sync(kFullMask, rk_l, i);
      sm.nq[h][t][i] = t < T && i <= c + t ? __expf(gh[c + t] - gh[i]) * rq_t * rk_i * gram_qk(t, i) : 0.f;
    }
    const float rq_slot = __shfl_sync(kFullMask, rq_l, (l >> 1) & (kItemT - 1));
    __syncwarp();
    float mr[kItemL][kItemL];
#pragma unroll
    for (int j = 1; j < kItemL; j++)
#pragma unroll
      for (int i = 0; i < j; i++) mr[j][i] = sm.m[h][j][i];
    float col[kItemL], acc[kItemL];
#pragma unroll
    for (int j = 0; j < kItemL; j++) acc[j] = j == l && j < L ? bh[j] : 0.f;
#pragma unroll
    for (int i = 0; i < kItemL; i++) {
      col[i] = i < L ? acc[i] : 0.f;
#pragma unroll
      for (int j = i + 1; j < kItemL; j++) acc[j] = fmaf(-mr[j][i], col[i], acc[j]);
    }
    if (l < kItemL) {
      // token l's decay from the slot's state with its norm; slot n's coefficients of key l (ck) and of value l (cv)
      const float ek = l < L ? __expf(gh[l]) * rk_l : 0.f;
      float cv[kItemSlots];
#pragma unroll
      for (int n = 0; n < kItemSlots; n++) {
        sm.ck[h][n][l] = 0.f;
        cv[n] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < kItemL; i++) {
        if (i < nk) {
          const int n = 8 * (i / 4) + 2 * (i % 4);
          sm.ck[h][n][l] = -col[i] * ek;
          cv[n] = l < L ? col[i] : 0.f;
        }
      }
#pragma unroll
      for (int t = 0; t < kItemT; t++) {
        if (t < T) {
          float na = 0.f;
#pragma unroll
          for (int i = 0; i < kItemL; i++) na = fmaf(sm.nq[h][t][i], col[i], na);
          sm.ck[h][2 * t + 1][l] = -na * ek;
          cv[2 * t + 1] = l < L ? na : 0.f;
        }
      }
#pragma unroll
      for (int n = 0; n < kItemSlots; n++)
        if (n < kSlots * nts) it->cvf[n / kSlots][4 * (n % kSlots) + (l & 3)][l >> 2] = cv[n];
      it->wc[l] = l < c ? __expf(gh[c - 1] - gh[l]) : 0.f;
      it->wb[l] = l <= bnd_j ? __expf(gh[bnd_j] - gh[l]) : 0.f;
    }
    if (l < kItemSlots) {
      // slot l's query term: e^{G_{c+t}} rq_t q_t for query t = l / 2 in an odd slot of tile 0
      const int t = l >> 1;
      sm.cq[h][l] = l < kSlots && (l & 1) && t < T ? __expf(gh[c + t]) * rq_slot : 0.f;
    }
    if (l == 0) {
      it->decay_c = c > 0 ? __expf(gh[c - 1]) : 1.f;
      it->decay_b = bnd_j >= 0 ? __expf(gh[bnd_j]) : 0.f;
      it->c = c, it->nts = nts, it->bnd_j = bnd_j;
    }
  }
  __syncthreads();

  // ---- the vectors of every value head (thread: a column; half 0 the even slots, half 1 the odd), then the normalized
  // keys as the rank update's fragments (half 0 keys 0..3, half 1 keys 4..7)
  {
    const int col = tid % kDim, half = tid / kDim;
    if (half < 2) {
      float kv[kItemL], qv[kItemT];
#pragma unroll
      for (int j = 0; j < kItemL; j++) kv[j] = sm.k[j][col];
#pragma unroll
      for (int t = 0; t < kItemT; t++) qv[t] = sm.q[t][col];
      for (int h = 0; h < R; h++) {
        VerifyItem* it = items + (long long)row * HV + ih * R + h;
#pragma unroll
        for (int n = half; n < kItemSlots; n += 2) {
          if (n < kSlots * nts) {
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < kItemL; j += 4) {
              const float4 w = *reinterpret_cast<const float4*>(&sm.ck[h][n][j]);
              acc = fmaf(w.x, kv[j], fmaf(w.y, kv[j + 1], fmaf(w.z, kv[j + 2], fmaf(w.w, kv[j + 3], acc))));
            }
            if (n < kSlots && (n & 1)) acc = fmaf(sm.cq[h][n], qv[n / 2], acc);
            it->vec[n][col] = acc;
          }
        }
      }
      KeyHeadItem* kh = key_items + (long long)row * H + ih;
      const int r16 = col & 15, gk = 2 * (r16 >> 2) + (r16 & 1), tile = (r16 >> 1) & 1;
#pragma unroll
      for (int jj = 0; jj < kItemL / 2; jj++) {
        const int j = half * (kItemL / 2) + jj;
        kh->kb[col >> 4][4 * gk + (j & 3)][2 * tile + (j >> 2)] = j < L ? kv[j] * sm.rk[j] : 0.f;
      }
    }
  }
}

// One item (a row's value head) of the stream kernel, its state in registers: the products with the effective vectors
// and the own parts (3xTF32 MMAs; lane 4 g + tig holds key tig's correction and query tig's output of rows g, g + 8),
// the boundary block's state, the running state (rank update on the state's registers as the MMA's accumulators), the
// gated norm
__device__ __forceinline__ void stream_item(const Layer& ly, const Step& sp, int item, int4 ri, int4 sl,
                                            float4 (&s)[2][kColChunks], const StreamData& d, StreamSmem& sm,
                                            float nw0, float nw1) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, tig = lane & 3;
  const int HV = sp.HV, hv = item % HV;
  const int C = 2 * sp.H * kDim + HV * kDim;
  const int tok0 = ri.x, T = ri.y;
  const int src = sl.x, dst = sl.y, bnd_slot = sl.z;
  const int c = d.c;
  const int r0 = 16 * warp + g8;
  const long long rows_off = ((long long)hv * kDim + r0) * kDim + 4 * tig;

  // the gate inputs of this thread's outputs (token tig, rows r0 and r0 + 8)
  float z0 = 0.f, z1 = 0.f;
  if (tig < T) {
    const __nv_bfloat16* zp = sp.qkvz + (long long)(tok0 + tig) * sp.qkvz_row + C + hv * kDim + r0;
    z0 = __bfloat162float(zp[0]), z1 = __bfloat162float(zp[8]);
  }

  // ---- products: D = S0 vec + V cv (rows 16 warp .. + 15, the tiles' slots)
  float u0, u1, u4a, u4b, o0, o1;
  auto products = [&](auto tiles) {
    constexpr int NTS = decltype(tiles)::value;
    float acc[NTS][2][4] = {};
#pragma unroll
    for (int i = 0; i < kColChunks; i++) {
      uint32_t ah[8], al[8];
      split_tf32(s[0][i].x, ah[0], al[0]);
      split_tf32(s[1][i].x, ah[1], al[1]);
      split_tf32(s[0][i].y, ah[2], al[2]);
      split_tf32(s[1][i].y, ah[3], al[3]);
      split_tf32(s[0][i].z, ah[4], al[4]);
      split_tf32(s[1][i].z, ah[5], al[5]);
      split_tf32(s[0][i].w, ah[6], al[6]);
      split_tf32(s[1][i].w, ah[7], al[7]);
#pragma unroll
      for (int nt = 0; nt < NTS; nt++) {
        const float4 b = *reinterpret_cast<const float4*>(&d.vec[kSlots * nt + g8][16 * i + 4 * tig]);
        uint32_t bh[4], bl[4];
        split_tf32(b.x, bh[0], bl[0]);
        split_tf32(b.y, bh[1], bl[1]);
        split_tf32(b.z, bh[2], bl[2]);
        split_tf32(b.w, bh[3], bl[3]);
        mma_tf32(acc[nt][0], al[0], al[1], al[2], al[3], bh[0], bh[1]);
        mma_tf32(acc[nt][1], al[4], al[5], al[6], al[7], bh[2], bh[3]);
        mma_tf32(acc[nt][0], ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
        mma_tf32(acc[nt][1], ah[4], ah[5], ah[6], ah[7], bl[2], bl[3]);
        mma_tf32(acc[nt][0], ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
        mma_tf32(acc[nt][1], ah[4], ah[5], ah[6], ah[7], bh[2], bh[3]);
      }
    }
    // the own parts: values tig and tig + 4 of rows r0, r0 + 8 with the slots' coefficients
    uint32_t vh[4], vl[4];
    split_tf32(d.v[tig][r0], vh[0], vl[0]);
    split_tf32(d.v[tig][r0 + 8], vh[1], vl[1]);
    split_tf32(d.v[tig + 4][r0], vh[2], vl[2]);
    split_tf32(d.v[tig + 4][r0 + 8], vh[3], vl[3]);
#pragma unroll
    for (int nt = 0; nt < NTS; nt++) {
      const float2 b = *reinterpret_cast<const float2*>(&d.cvf[nt][lane][0]);
      uint32_t bh0, bh1, bl0, bl1;
      split_tf32(b.x, bh0, bl0);
      split_tf32(b.y, bh1, bl1);
      mma3_tf32(acc[nt][0], vh, vl, bh0, bh1, bl0, bl1);
    }
    u0 = acc[0][0][0] + acc[0][1][0], o0 = acc[0][0][1] + acc[0][1][1];
    u1 = acc[0][0][2] + acc[0][1][2], o1 = acc[0][0][3] + acc[0][1][3];
    if constexpr (NTS > 1) {
      u4a = acc[1][0][0] + acc[1][1][0], u4b = acc[1][0][2] + acc[1][1][2];
    } else {
      u4a = u4b = 0.f;
    }
  };
  if (d.nts == 1) {
    products(std::integral_constant<int, 1>{});
  } else {
    products(std::integral_constant<int, kItemTiles>{});
  }

  // ---- the boundary block's state: e^{G_b} S0 + sum_{i <= b} e^{G_b - G_i} u_i k_i (keys 0..7: one k8 step)
  if (bnd_slot >= 0) {
    const float wl = d.wb[tig], wh = d.wb[tig + 4], eb = d.decay_b;
    uint32_t ah[4], al[4];
    split_tf32(wl * u0, ah[0], al[0]);
    split_tf32(wl * u1, ah[1], al[1]);
    split_tf32(wh * u4a, ah[2], al[2]);
    split_tf32(wh * u4b, ah[3], al[3]);
    float* bdst = ly.state + (long long)bnd_slot * ly.state_slot + rows_off;
#pragma unroll
    for (int i = 0; i < kColChunks; i++) {
      float t0[4] = {eb * s[0][i].x, eb * s[0][i].y, eb * s[1][i].x, eb * s[1][i].y};
      float t1[4] = {eb * s[0][i].z, eb * s[0][i].w, eb * s[1][i].z, eb * s[1][i].w};
      const float4 kb = *reinterpret_cast<const float4*>(&d.kb[i][lane][0]);
      uint32_t bh[4], bl[4];
      split_tf32(kb.x, bh[0], bl[0]);
      split_tf32(kb.y, bh[1], bl[1]);
      split_tf32(kb.z, bh[2], bl[2]);
      split_tf32(kb.w, bh[3], bl[3]);
      mma3_tf32(t0, ah, al, bh[0], bh[1], bl[0], bl[1]);
      mma3_tf32(t1, ah, al, bh[2], bh[3], bl[2], bl[3]);
      __stcs(reinterpret_cast<float4*>(bdst + 16 * i), make_float4(t0[0], t0[1], t1[0], t1[1]));
      __stcs(reinterpret_cast<float4*>(bdst + 8 * kDim + 16 * i), make_float4(t0[2], t0[3], t1[2], t1[3]));
    }
  }
  // ---- the running state: e^{G_{c-1}} S0 + sum_{i < c} e^{G_{c-1} - G_i} u_i k_i (keys 0..3: one k4 step), written once
  if (c > 0 || src != dst) {
    const float ec = d.decay_c, wl = d.wc[tig];
    uint32_t ah0, al0, ah1, al1;
    split_tf32(wl * u0, ah0, al0);
    split_tf32(wl * u1, ah1, al1);
    float* sdst = ly.state + (long long)dst * ly.state_slot + rows_off;
#pragma unroll
    for (int i = 0; i < kColChunks; i++) {
      float t0[4] = {ec * s[0][i].x, ec * s[0][i].y, ec * s[1][i].x, ec * s[1][i].y};
      float t1[4] = {ec * s[0][i].z, ec * s[0][i].w, ec * s[1][i].z, ec * s[1][i].w};
      if (c > 0) {
        const float4 kb = *reinterpret_cast<const float4*>(&d.kb[i][lane][0]);
        uint32_t bh0, bl0, bh2, bl2;
        split_tf32(kb.x, bh0, bl0);
        split_tf32(kb.z, bh2, bl2);
        mma_tf32_k4(t0, al0, al1, bh0);
        mma_tf32_k4(t0, ah0, ah1, bl0);
        mma_tf32_k4(t0, ah0, ah1, bh0);
        mma_tf32_k4(t1, al0, al1, bh2);
        mma_tf32_k4(t1, ah0, ah1, bl2);
        mma_tf32_k4(t1, ah0, ah1, bh2);
      }
      __stcs(reinterpret_cast<float4*>(sdst + 16 * i), make_float4(t0[0], t0[1], t1[0], t1[1]));
      __stcs(reinterpret_cast<float4*>(sdst + 8 * kDim + 16 * i), make_float4(t0[2], t0[3], t1[2], t1[3]));
    }
  }

  // ---- the gated RMSNorm of each output token over the head's rows (lanes tig: token tig)
  float sq = tig < T ? o0 * o0 + o1 * o1 : 0.f;
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) sq += __shfl_xor_sync(kFullMask, sq, o);
  if (g8 == 0) sm.red[tig][warp] = sq;
  __syncthreads();
  if (tig < T) {
    const float4 p0 = *reinterpret_cast<const float4*>(&sm.red[tig][0]);
    const float4 p1 = *reinterpret_cast<const float4*>(&sm.red[tig][4]);
    const float rstd = rsqrtf((((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w))) / kDim + sp.norm_eps);
    const float gate0 = sp.gate_sigmoid ? sigmoid(z0) : z0 * sigmoid(z0);
    const float gate1 = sp.gate_sigmoid ? sigmoid(z1) : z1 * sigmoid(z1);
    __nv_bfloat16* op = sp.out + ((long long)(tok0 + tig) * HV + hv) * kDim + r0;
    op[0] = __float2bfloat16(o0 * rstd * nw0 * gate0);
    op[8] = __float2bfloat16(o1 * rstd * nw1 * gate1);
  }
}

// The items of a verify step (rows x value heads, row-major; the verify rows come first) over persistent CTAs: each
// CTA's next state and step data load into shared memory while it computes the current item from registers
__global__ void __launch_bounds__(kThreads, kStreamCtasPerSm) gdn_stream_kernel(const Layer ly, const Step sp,
                                                                                const VerifyItem* items,
                                                                                const KeyHeadItem* key_items,
                                                                                int num_items) {
  extern __shared__ float4 smem_raw[];
  StreamSmem& sm = *reinterpret_cast<StreamSmem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, tig = lane & 3;
  const int HV = sp.HV, R = sp.R, stride = gridDim.x;
  const long long rows_off = (16 * warp + g8) * kDim + 4 * tig;
  auto load_meta = [&](int item, int4& ri, int4& sl) {
    const int row = item / HV;
    ri = *reinterpret_cast<const int4*>(sp.rows + (long long)row * kRowInts);
    sl = *reinterpret_cast<const int4*>(sp.slots + ((long long)ly.group * sp.max_rows + row) * kSlotInts);
  };
  auto load_state = [&](int item, int src) {
    const float* base = ly.state + (long long)src * ly.state_slot + (long long)(item % HV) * kDim * kDim + rows_off;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < kColChunks; i++) cp_async16(&sm.state[kColChunks * h + i][tid], base + 8 * h * kDim + 16 * i);
  };
  auto load_data = [&](int item, StreamData& d) {
    const int row = item / HV, hv = item % HV;
    const VerifyItem* it = items + item;
    const KeyHeadItem* kh = key_items + (long long)row * sp.H + hv / R;
#pragma unroll
    for (int k = 0; k < kItemSlots * kDim / 4 / kThreads; k++) {
      const int chunk = tid + k * kThreads, n = chunk / (kDim / 4), col = 4 * (chunk % (kDim / 4));
      cp_async16(&d.vec[n][col], &it->vec[n][col]);
    }
    {
      const int n = tid / (kDim / 4), col = 4 * (tid % (kDim / 4));
      cp_async16(&d.v[n][col], &it->v[n][col]);
    }
    cp_async16(reinterpret_cast<float4*>(&d.kb[0][0][0]) + tid, reinterpret_cast<const float4*>(&kh->kb[0][0][0]) + tid);
    if (tid < kItemTailBytes / 16)
      cp_async16(reinterpret_cast<float4*>(&d.cvf[0][0][0]) + tid, reinterpret_cast<const float4*>(&it->cvf[0][0][0]) + tid);
  };

  int item = blockIdx.x;
  int4 ri = make_int4(0, 0, 0, 0), sl = make_int4(0, 0, 0, 0);
  if (item < num_items) load_meta(item, ri, sl);
  if (item >= num_items || ri.y == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
  const float nw0 = param(ly.norm_w, 16 * warp + g8, ly.norm_w_f32);
  const float nw1 = param(ly.norm_w, 16 * warp + g8 + 8, ly.norm_w_f32);
  // the first state does not depend on the step's kernels: it loads while the prep kernel runs
  load_state(item, sl.x);
  cp_async_commit();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  load_data(item, sm.data[0]);
  cp_async_commit();
  int next = item + stride;
  int4 ri_next = make_int4(0, 0, 0, 0), sl_next = make_int4(0, 0, 0, 0);
  if (next < num_items) load_meta(next, ri_next, sl_next);

  for (int buf = 0;; buf ^= 1) {
    cp_async_wait_all();
    __syncthreads();
    float4 s[2][kColChunks];
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < kColChunks; i++) s[h][i] = sm.state[kColChunks * h + i][tid];
    const bool has_next = next < num_items && ri_next.y > 0;
    const int after = next + stride;
    int4 ri_after = make_int4(0, 0, 0, 0), sl_after = make_int4(0, 0, 0, 0);
    if (has_next) {
      load_state(next, sl_next.x);
      load_data(next, sm.data[buf ^ 1]);
      cp_async_commit();
      if (after < num_items) load_meta(after, ri_after, sl_after);
    } else {
      // the next kernel (out_proj's GEMM) launches with this CTA's last item
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    stream_item(ly, sp, item, ri, sl, s, sm.data[buf], sm, nw0, nw1);
    if (!has_next) break;
    item = next, ri = ri_next, sl = sl_next;
    next = after, ri_next = ri_after, sl_next = sl_after;
  }
}

// ---- after sampling: the window at a committed boundary inside a verify row's block into the boundary block's conv
// slot, from the record the step wrote (its entry 0 holds the input at position pos - 2): the head's v channels and,
// from its key head's first value head, the key head's q / k channels
__device__ __forceinline__ void boundary_window(const Layer& ly, const Step& sp, int row, int hv, int slot, int pos,
                                                int boundary) {
  const int tid = threadIdx.x;
  if (tid >= kDim) return;
  const int req = sp.rows[(long long)row * kRowInts + 2];
  const int H = sp.H, HV = sp.HV, R = HV / H, ih = hv / R;
  const int C = 2 * H * kDim + HV * kDim;
  const __nv_bfloat16* rec = reinterpret_cast<const __nv_bfloat16*>(
      ly.record + (long long)req * ly.record_req + (long long)(1 - sp.parity[req]) * ly.record_copy);
  const int first = boundary - pos;  // the window's first entry: the input at position boundary - 2
#pragma unroll
  for (int which = 0; which < 3; which++) {
    if (which < 2 && hv % R != 0) continue;
    const int ch = which == 0 ? ih * kDim + tid : which == 1 ? (H + ih) * kDim + tid : 2 * H * kDim + hv * kDim + tid;
    __nv_bfloat16* w = ly.conv + (long long)slot * ly.conv_slot + ch * ly.conv_ch;
#pragma unroll
    for (int e = 0; e < kTaps; e++) w[e * ly.conv_tap] = rec[(long long)(first + e) * C + ch];
  }
}

// After sampling, per verify row whose committed tokens reach a block boundary: the window there into the boundary
// block (the forward wrote its state); a block ending at the boundary, committed whole, is replayed into its running
// slot (the next step finds no record). One CTA per value head and layer.
template <int MT>
__global__ void __launch_bounds__(kThreads) gdn_flush_kernel(const Layer* layers, const Step sp, const int* sampled,
                                                             int num_rows) {
  extern __shared__ float4 smem_raw[];
  CoreSmem<MT>& sm = *reinterpret_cast<CoreSmem<MT>*>(smem_raw);
  __shared__ int found[kThreads];
  __shared__ int num_found;
  const Layer ly = layers[blockIdx.y];
  const int hv = blockIdx.x;
  for (int first = 0; first < num_rows; first += kThreads) {
    if (threadIdx.x == 0) num_found = 0;
    __syncthreads();
    const int row = first + threadIdx.x;
    if (row < num_rows) {
      const int* ri = sp.rows + (long long)row * kRowInts;
      const int T = ri[1], pos = ri[3];
      const int boundary = (pos / sp.block_size + 1) * sp.block_size;
      if (T > 0 && boundary <= pos + sampled[row]) found[atomicAdd(&num_found, 1)] = row;
    }
    __syncthreads();
    for (int f = 0; f < num_found; f++) {
      const int row = found[f];
      const int* ri = sp.rows + (long long)row * kRowInts;
      const int T = ri[1], pos = ri[3];
      const int boundary = (pos / sp.block_size + 1) * sp.block_size;
      const int* sl = sp.slots + ((long long)ly.group * sp.max_rows + row) * kSlotInts;
      const bool is_block_end = boundary == pos + T;
      boundary_window(ly, sp, row, hv, is_block_end ? sl[1] : sl[2], pos, boundary);
      if (is_block_end) core_item<MT>(ly, sp, row, hv, hv / sp.R, sm);
      __syncthreads();
    }
  }
}

// The step's rows and slots (before the forward): each row's first token, verify tokens, request and position, and in
// every group its source slot (the previous running block), destination (this step's running block, vLLM's align
// column) and the block of a boundary inside its block; vLLM's align bookkeeping for the precopy of the other rows
__global__ void __launch_bounds__(kMetaThreads) replay_meta_kernel(
    const int64_t* idx_mapping, const int* query_start, const int* computed, const uint8_t* is_verify, int* state_col,
    int* src_col, int* src_off, int* accepted, const int64_t* tables, long long table_stride, int num_groups,
    int num_reqs, int num_rows, int max_rows, int block_size, int* rows, int* slots) {
  const int row = blockIdx.x * kMetaThreads + threadIdx.x;
  if (row >= num_rows) return;
  int* ri = rows + (long long)row * kRowInts;
  if (row >= num_reqs) {
    ri[0] = ri[1] = ri[2] = ri[3] = 0;
    return;
  }
  const int req = (int)idx_mapping[row];
  const int tok0 = query_start[row], T = query_start[row + 1] - tok0;
  const int pos = computed[req];
  const bool verify = is_verify == nullptr || is_verify[row];
  const int old_col = state_col[req];
  const int new_col = (pos + T + block_size - 1) / block_size - 1;
  src_off[req] = max(accepted[req] - 1, 0);
  src_col[req] = verify ? -1 : old_col;
  state_col[req] = new_col;
  if (old_col >= 0 && old_col != new_col) accepted[req] = 1;
  ri[0] = tok0;
  ri[1] = verify ? T : 0;
  ri[2] = req;
  ri[3] = pos;
  const int boundary = (pos / block_size + 1) * block_size;
  for (int g = 0; g < num_groups; g++) {
    const int* table = reinterpret_cast<const int*>(tables[g]) + (long long)row * table_stride;
    int* s = slots + ((long long)g * max_rows + row) * kSlotInts;
    s[0] = table[old_col >= 0 ? old_col : new_col];
    s[1] = table[new_col];
    s[2] = boundary < pos + T ? table[boundary / block_size - 1] : -1;
  }
}

// The step's bookkeeping after sampling (after the flush): each verify row's committed tokens are its request's
// replay count (none after a flush) and its record copy flips; vLLM's accepted counts as its own postprocess sets them.
// The rows end with the step: a forward without the runner's preprocess (a graph capture, a dummy run) has none.
__global__ void __launch_bounds__(kMetaThreads) replay_commit_kernel(const int64_t* idx_mapping, const int* sampled,
                                                                     int* rows, int* replay, int* parity,
                                                                     int* accepted, int num_rows, int block_size) {
  const int row = blockIdx.x * kMetaThreads + threadIdx.x;
  if (row >= num_rows) return;
  int* ri = rows + (long long)row * kRowInts;
  const int T = ri[1];
  ri[1] = 0;
  const int req = (int)idx_mapping[row];
  if (req < 0) return;
  const int n = sampled[row];
  accepted[req] = max(n, 1);
  if (T == 0) {
    replay[req] = 0;
    return;
  }
  const bool is_flushed = n == T && (ri[3] + T) % block_size == 0;
  replay[req] = is_flushed ? 0 : max(n, 1);
  parity[req] ^= 1;
}

// ---- host side
constexpr int kMaxT = 8;  // tokens of a verify block at most: the kernels' instances below

Layer make_layer(torch::Tensor state, torch::Tensor conv, torch::Tensor record, torch::Tensor conv_w,
                 torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor norm_w, int64_t group, bool conv_dim_first) {
  TORCH_CHECK(state.scalar_type() == at::kFloat && conv.scalar_type() == at::kBFloat16, "gdn_verify: state dtypes");
  TORCH_CHECK(state.size(2) == kDim && state.size(3) == kDim && state.stride(3) == 1 && state.stride(2) == kDim &&
                  state.stride(1) == kDim * kDim,
              "gdn_verify: the state cache must be [slots][HV][128][128] with contiguous slots");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && conv_w.scalar_type() == at::kBFloat16 && conv_w.is_contiguous());
  Layer ly{};
  ly.state = state.data_ptr<float>();
  ly.conv = reinterpret_cast<__nv_bfloat16*>(conv.data_ptr());
  ly.record = record.data_ptr<uint8_t>();
  ly.conv_w = reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr());
  ly.a_log = a_log.data_ptr<float>();
  ly.dt_bias = dt_bias.data_ptr();
  ly.norm_w = norm_w.data_ptr();
  ly.state_slot = state.stride(0);
  ly.conv_slot = conv.stride(0);
  ly.conv_ch = conv_dim_first ? conv.stride(1) : conv.stride(2);
  ly.conv_tap = conv_dim_first ? conv.stride(2) : conv.stride(1);
  ly.record_req = record.stride(0);
  ly.record_copy = record.stride(1);
  ly.group = group;
  ly.dt_bias_f32 = dt_bias.scalar_type() == at::kFloat;
  ly.norm_w_f32 = norm_w.scalar_type() == at::kFloat;
  return ly;
}

Step make_step(torch::Tensor rows, torch::Tensor slots, torch::Tensor replay, torch::Tensor parity, int64_t num_heads,
               int64_t num_v_heads, int64_t block_size, int64_t record_ba, double scale) {
  TORCH_CHECK(rows.scalar_type() == at::kInt && slots.scalar_type() == at::kInt && replay.scalar_type() == at::kInt &&
                  parity.scalar_type() == at::kInt && rows.is_contiguous() && slots.is_contiguous(),
              "gdn_verify: int32 rows, slots, replay counts and parities");
  TORCH_CHECK(num_v_heads % num_heads == 0 && num_v_heads / num_heads <= 32, "gdn_verify: value heads per key head");
  Step sp{};
  sp.rows = rows.data_ptr<int>();
  sp.slots = slots.data_ptr<int>();
  sp.replay = replay.data_ptr<int>();
  sp.parity = parity.data_ptr<int>();
  sp.record_ba = record_ba;
  sp.max_rows = slots.size(1);
  sp.H = num_heads;
  sp.HV = num_v_heads;
  sp.R = num_v_heads / num_heads;
  sp.block_size = block_size;
  sp.scale = scale;
  return sp;
}

template <typename F>
void set_smem(F* kernel, size_t bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// the kernels' instance of a block of max_tokens tokens at most
template <typename F>
void dispatch_tokens(int64_t max_tokens, F&& f) {
  TORCH_CHECK(max_tokens >= 1 && max_tokens <= kMaxT, "gdn_verify: blocks of 1..", kMaxT, " tokens");
  if (max_tokens <= 2)
    f(std::integral_constant<int, 2>{});
  else if (max_tokens <= 4)
    f(std::integral_constant<int, 4>{});
  else
    f(std::integral_constant<int, 8>{});
}

// a programmatic dependent's launch of kernel over grid x block with smem bytes of dynamic shared memory
template <typename F, typename... Args>
void launch_dependent(F* kernel, dim3 grid, int block, size_t smem, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(block);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

// The scratch of steps of max_rows rows: the value heads' items, then the key heads'
int64_t gdn_verify_scratch_bytes(int64_t max_rows, int64_t num_heads, int64_t num_v_heads) {
  return max_rows * (num_v_heads * (int64_t)sizeof(VerifyItem) + num_heads * (int64_t)sizeof(KeyHeadItem));
}

// Verify blocks of up to kItemT tokens: the prep kernel per row and key head, then the stream kernel over the items
void verify_items(const Layer& ly, const Step& sp, torch::Tensor scratch, int num_rows, cudaStream_t stream) {
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    set_smem(gdn_prep_kernel, sizeof(PrepSmem<kItemT>));
    set_smem(gdn_stream_kernel, sizeof(StreamSmem));
    int ctas = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, gdn_stream_kernel, kThreads, sizeof(StreamSmem)));
    TORCH_CHECK(ctas >= kStreamCtasPerSm, "gdn_verify: ", ctas, " stream CTAs per SM");
  }
  TORCH_CHECK(sp.R <= kMaxRatio && sp.R * sp.H == sp.HV, "gdn_verify: at most ", kMaxRatio, " value heads per key head");
  TORCH_CHECK(scratch.is_cuda() && scratch.nbytes() >= gdn_verify_scratch_bytes(sp.max_rows, sp.H, sp.HV) &&
                  reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 16 == 0,
              "gdn_verify: a scratch of gdn_verify_scratch_bytes");
  TORCH_CHECK(sp.qkvz_row % 2 == 0, "gdn_verify: even qkvz rows (bf16 pairs)");
  auto* items = reinterpret_cast<VerifyItem*>(scratch.data_ptr());
  auto* key_items = reinterpret_cast<KeyHeadItem*>(items + (int64_t)sp.max_rows * sp.HV);
  const int num_items = num_rows * sp.HV;
  launch_dependent(gdn_prep_kernel, dim3(num_rows, sp.H), kPrepPairs * (1 + sp.R), sizeof(PrepSmem<kItemT>), stream,
                   ly, sp, items, key_items);
  launch_dependent(gdn_stream_kernel, dim3(std::min(num_items, kStreamCtasPerSm * sms)), kThreads, sizeof(StreamSmem),
                   stream, ly, sp, (const VerifyItem*)items, (const KeyHeadItem*)key_items, num_items);
}

// One layer's verify core over the step's first num_rows rows: blocks of up to kItemT tokens on the prep and stream
// kernels, longer ones one CTA per row and value head
void gdn_verify(torch::Tensor qkvz, torch::Tensor ba, torch::Tensor out, torch::Tensor conv_w, torch::Tensor a_log,
                torch::Tensor dt_bias, torch::Tensor norm_w, torch::Tensor state, torch::Tensor conv,
                torch::Tensor record, torch::Tensor rows, torch::Tensor slots, torch::Tensor replay,
                torch::Tensor parity, torch::Tensor scratch, int64_t group, int64_t num_rows, int64_t num_heads,
                int64_t block_size, int64_t max_tokens, int64_t record_ba, double scale, double norm_eps,
                bool gate_sigmoid, bool conv_dim_first) {
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(qkvz.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const Layer ly = make_layer(state, conv, record, conv_w, a_log, dt_bias, norm_w, group, conv_dim_first);
  const int64_t num_v_heads = state.size(1);
  TORCH_CHECK(qkvz.scalar_type() == at::kBFloat16 && ba.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16 && qkvz.stride(1) == 1 && ba.stride(1) == 1 && out.is_contiguous(),
              "gdn_verify: bf16 inputs with contiguous rows");
  Step sp = make_step(rows, slots, replay, parity, num_heads, num_v_heads, block_size, record_ba, scale);
  sp.qkvz = reinterpret_cast<const __nv_bfloat16*>(qkvz.data_ptr());
  sp.ba = reinterpret_cast<const __nv_bfloat16*>(ba.data_ptr());
  sp.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  sp.qkvz_row = qkvz.stride(0);
  sp.ba_row = ba.stride(0);
  sp.norm_eps = norm_eps;
  sp.gate_sigmoid = gate_sigmoid;
  if (max_tokens <= kItemT) {
    verify_items(ly, sp, scratch, num_rows, stream);
    return;
  }
  dispatch_tokens(max_tokens, [&](auto mt) {
    constexpr int MT = decltype(mt)::value;
    static bool is_set = false;
    if (!is_set) {
      set_smem(gdn_verify_kernel<MT>, sizeof(CoreSmem<MT>));
      is_set = true;
    }
    // a programmatic dependent of the projections' GEMM: each CTA loads its state before it waits for the GEMM; the
    // grid: rows x value heads of a key head x key heads
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_rows, num_v_heads / num_heads, num_heads);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = sizeof(CoreSmem<MT>);
    config.stream = stream;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, gdn_verify_kernel<MT>, ly, sp));
  });
}

// The per-layer descriptors of the flush (built once per KV cache binding): a uint8 tensor of Layer structs
torch::Tensor gdn_layer_table(std::vector<torch::Tensor> states, std::vector<torch::Tensor> convs,
                              std::vector<torch::Tensor> records, std::vector<torch::Tensor> conv_ws,
                              std::vector<torch::Tensor> a_logs, std::vector<torch::Tensor> dt_biases,
                              std::vector<torch::Tensor> norm_ws, std::vector<int64_t> groups, bool conv_dim_first) {
  const size_t n = states.size();
  auto host = torch::empty({(int64_t)(n * sizeof(Layer))}, torch::dtype(torch::kUInt8));
  Layer* p = reinterpret_cast<Layer*>(host.data_ptr<uint8_t>());
  for (size_t i = 0; i < n; i++)
    p[i] = make_layer(states[i], convs[i], records[i], conv_ws[i], a_logs[i], dt_biases[i], norm_ws[i], groups[i],
                      conv_dim_first);
  return host.to(states[0].device());
}

// The flush of every layer (the grid: value heads x layers), then the step's bookkeeping
void gdn_commit(torch::Tensor layers, torch::Tensor rows, torch::Tensor slots, torch::Tensor replay,
                torch::Tensor parity, torch::Tensor accepted, torch::Tensor idx_mapping, torch::Tensor sampled,
                int64_t num_heads, int64_t num_v_heads, int64_t block_size, int64_t max_tokens, int64_t record_ba,
                double scale) {
  const int num_rows = idx_mapping.size(0);
  if (num_rows == 0) return;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && sampled.scalar_type() == at::kInt &&
              accepted.scalar_type() == at::kInt, "gdn_commit: int64 idx_mapping, int32 counts");
  const at::cuda::OptionalCUDAGuard guard(rows.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_layers = layers.size(0) / sizeof(Layer);
  Step sp = make_step(rows, slots, replay, parity, num_heads, num_v_heads, block_size, record_ba, scale);
  sp.is_flush = 1;
  dispatch_tokens(max_tokens, [&](auto mt) {
    constexpr int MT = decltype(mt)::value;
    static bool is_set = false;
    if (!is_set) {
      set_smem(gdn_flush_kernel<MT>, sizeof(CoreSmem<MT>));
      is_set = true;
    }
    gdn_flush_kernel<MT><<<dim3(num_v_heads, num_layers), kThreads, sizeof(CoreSmem<MT>), stream>>>(
        reinterpret_cast<const Layer*>(layers.data_ptr<uint8_t>()), sp, sampled.data_ptr<int>(), num_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  replay_commit_kernel<<<(num_rows + kMetaThreads - 1) / kMetaThreads, kMetaThreads, 0, stream>>>(
      idx_mapping.data_ptr<int64_t>(), sampled.data_ptr<int>(), rows.data_ptr<int>(), replay.data_ptr<int>(),
      parity.data_ptr<int>(), accepted.data_ptr<int>(), num_rows, block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The step's rows and slots before the forward
void gdn_replay_meta(torch::Tensor idx_mapping, torch::Tensor query_start, torch::Tensor computed,
                     std::optional<torch::Tensor> is_verify, torch::Tensor state_col, torch::Tensor src_col,
                     torch::Tensor src_off, torch::Tensor accepted, torch::Tensor tables, int64_t table_stride,
                     int64_t num_reqs, int64_t num_rows, int64_t block_size, torch::Tensor rows, torch::Tensor slots) {
  if (num_rows == 0) return;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && query_start.scalar_type() == at::kInt &&
                  computed.scalar_type() == at::kInt && tables.scalar_type() == at::kLong,
              "gdn_replay_meta: int64 idx_mapping and tables, int32 positions");
  const at::cuda::OptionalCUDAGuard guard(rows.device());
  replay_meta_kernel<<<(num_rows + kMetaThreads - 1) / kMetaThreads, kMetaThreads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      idx_mapping.data_ptr<int64_t>(), query_start.data_ptr<int>(), computed.data_ptr<int>(),
      is_verify.has_value() ? is_verify->data_ptr<uint8_t>() : nullptr, state_col.data_ptr<int>(),
      src_col.data_ptr<int>(), src_off.data_ptr<int>(), accepted.data_ptr<int>(), tables.data_ptr<int64_t>(),
      table_stride, tables.size(0), num_reqs, num_rows, slots.size(1), block_size, rows.data_ptr<int>(),
      slots.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def(
      "gdn_verify(Tensor qkvz, Tensor ba, Tensor(a!) out, Tensor conv_w, Tensor a_log, Tensor dt_bias, Tensor norm_w, "
      "Tensor(b!) state, Tensor(c!) conv, Tensor(d!) record, Tensor rows, Tensor slots, Tensor replay, Tensor parity, "
      "Tensor(e!) scratch, int group, int num_rows, int num_heads, int block_size, int max_tokens, int record_ba, "
      "float scale, float norm_eps, bool gate_sigmoid, bool conv_dim_first) -> ()");
  m.def("gdn_verify_scratch_bytes(int max_rows, int num_heads, int num_v_heads) -> int");
  m.def(
      "gdn_layer_table(Tensor[] states, Tensor[] convs, Tensor[] records, Tensor[] conv_ws, Tensor[] a_logs, "
      "Tensor[] dt_biases, Tensor[] norm_ws, int[] groups, bool conv_dim_first) -> Tensor");
  m.def(
      "gdn_commit(Tensor layers, Tensor rows, Tensor slots, Tensor(a!) replay, Tensor(b!) parity, "
      "Tensor(c!) accepted, Tensor idx_mapping, Tensor sampled, int num_heads, int num_v_heads, int block_size, "
      "int max_tokens, int record_ba, float scale) -> ()");
  m.def(
      "gdn_replay_meta(Tensor idx_mapping, Tensor query_start, Tensor computed, Tensor? is_verify, "
      "Tensor(a!) state_col, Tensor(b!) src_col, Tensor(c!) src_off, Tensor(d!) accepted, Tensor tables, "
      "int table_stride, int num_reqs, int num_rows, int block_size, Tensor(e!) rows, Tensor(f!) slots) -> ()");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("gdn_verify", &gdn_verify);
  m.impl("gdn_commit", &gdn_commit);
  m.impl("gdn_replay_meta", &gdn_replay_meta);
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CompositeExplicitAutograd, m) {
  m.impl("gdn_layer_table", &gdn_layer_table);
  m.impl("gdn_verify_scratch_bytes", &gdn_verify_scratch_bytes);
}
