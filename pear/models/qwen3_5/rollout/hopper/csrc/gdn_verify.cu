// The Gated DeltaNet core of verify steps under replaySSM: a request keeps one state slot (the state at the start of its
// last verify block) and a record of that block; a step replays the block's committed tokens from the slot, gives its
// own block's outputs and writes the state once, with the rounding points of the fused decode kernel (gdn_decode.cu).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <type_traits>

namespace gdn_verify {

constexpr int kDim = 128;                     // the key and the value head dimension
constexpr int kTaps = 3;                      // the conv window: the width-4 kernel's earlier inputs
constexpr int kThreads = 256;                 // a CTA: eight warps of 16 state rows; the conv's q / k and v channels
constexpr int kWarps = kThreads / 32;
constexpr int kColChunks = kDim / 16;         // a thread's 16-column chunks of a state row
constexpr int kColTiles = kDim / 8;           // the 8-column MMA tiles of a head's columns
constexpr int kGramWarps = 4;                 // the warps that split the gram's k-steps
constexpr int kGramSteps = kColTiles / kGramWarps;
constexpr int kKQStride = kDim + 4;           // smem row strides: conflict-free (row g8, column tig) fragment loads of
constexpr int kVStride = kDim + 8;            // the [keys; queries], (row tig, column g8) ones of the values' transpose
constexpr int kVecStride = kDim + 16;         // and float4 ones of the vectors (two k-steps of a chunk)
constexpr int kHalfRows = kDim / 2;          // the state's halves in smem, the second shifted by kHalfShift floats: a
constexpr int kHalfShift = 16;               // quarter-warp's float4 loads of a row of each half are conflict-free
constexpr uint32_t kHalfBytes = kHalfRows * kDim * sizeof(float);  // a half: one bulk copy
constexpr int kRowInts = 8;  // a batch row: first token, tokens (0: not a verify row), request, position, replayed
                             // tokens, record copy, 2 unused
constexpr int kSlotInts = 4;                  // a row's slots in a group: source, destination, boundary block (-1: none)
constexpr int kMetaThreads = 128;             // threads of the per-step bookkeeping kernels
constexpr int kMaxT = 8;                      // tokens of a verify block at most: the kernels' instances
constexpr float kL2Eps = 1e-6f;               // FLA's l2norm epsilon of q and k, under the square root
constexpr float kSoftplusThreshold = 20.f;    // softplus is the identity above it, as in vLLM's gating kernel
constexpr float kLog2e = 1.4426950408889634f;
constexpr uint32_t kTf32Half = 0x1000u;       // half an ulp of tf32 (bit 12)
constexpr uint32_t kTf32Mask = 0xffffe000u;   // the bits of a tf32 value
constexpr uint32_t kFullMask = 0xffffffffu;

// The sizes of a step whose verify blocks have at most MT tokens
template <int MT>
struct Sizes {
  static constexpr int L = 2 * MT;                    // replayed and current tokens of a step
  static constexpr int J = L + MT;                    // the step's keys, then its queries
  static constexpr int JP = (J + 7) / 8 * 8;          // J in whole MMA k-steps
  static constexpr int KQ_ROWS = (J + 15) / 16 * 16;  // the [keys; queries] rows in whole MMA m-tiles
  static constexpr int N = L + MT;                    // a state row's products: minus the effective keys, the queries
  static constexpr int NT = (N + 7) / 8;              // n-tiles of the products
  static constexpr int NM = (N + 15) / 16;            // m-tiles of the vectors
  static constexpr int GRAM_COLS = L + 8;             // the gram's columns: the keys, then a tile of the queries
};

struct Layer {                             // one Gated DeltaNet layer: its caches, its record, its weights
  float* state;                            // fp32 [slots][HV][V][K]
  __nv_bfloat16* conv;                     // bf16 conv window cache
  uint8_t* record;                         // [requests][2][record bytes]
  const __nv_bfloat16* conv_w;             // bf16 [C][4]
  const float* a_log;                      // fp32 [HV]
  const void* dt_bias;                     // [HV], fp32 or bf16
  const void* norm_w;                      // [V], fp32 or bf16
  long long state_slot;                    // the state cache's slot stride (floats)
  long long conv_slot, conv_ch, conv_tap;  // the conv cache's strides (elements)
  long long record_req, record_copy;       // the record's request and copy strides (bytes)
  int group;                               // the layer's mamba KV cache group: its slots in the step's table
  int dt_bias_f32, norm_w_f32;
  int pad;
};

struct Step {                           // a step's batch
  const __nv_bfloat16* qkvz;            // [tokens][C + HV V]: q, k, v (the conv inputs), z
  const __nv_bfloat16* ba;              // [tokens][2 HV]: b, a
  __nv_bfloat16* out;                   // [tokens][HV][V]: the gated norm's output
  const int* rows;                      // [rows][kRowInts]
  const int* slots;                     // [groups][max rows][kSlotInts]
  long long qkvz_row, ba_row, out_row;  // row strides (elements)
  long long record_ba;                  // the record's b / a offset (bytes): after its (kTaps + max T) x C inputs
  int max_rows, H, HV, R, block_size;   // R: value heads per key head
  float scale, norm_eps;
  int gate_sigmoid, is_flush;
};

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float param(const void* p, int i, int is_f32) {
  return is_f32 ? reinterpret_cast<const float*>(p)[i] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
}

// Triton's lowerings in vLLM's kernels (their PTX): tl.exp = ex2.approx of x * log2(e), tl.log = libdevice logf, fp32
// '/' and tl.sigmoid's = div.full, tl.sqrt = sqrt.approx.ftz, tl.rsqrt = rsqrt.approx.ftz
__device__ __forceinline__ float tl_exp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * kLog2e));
  return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float sqrt_approx(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float tl_sigmoid(float x) { return div_full(1.f, 1.f + tl_exp(0.f - x)); }

// y = bf16(silu(sum_j bf16(x_j w_j))) over the taps, the sum in fp32: the decode kernel's conv (vLLM's Triton kernel)
__device__ __forceinline__ float conv_tap(const __nv_bfloat16 (&w)[4], __nv_bfloat16 x0, __nv_bfloat16 x1,
                                          __nv_bfloat16 x2, __nv_bfloat16 x3) {
  float acc = 0.f;
  acc += __bfloat162float(__hmul(x0, w[0]));
  acc += __bfloat162float(__hmul(x1, w[1]));
  acc += __bfloat162float(__hmul(x2, w[2]));
  acc += __bfloat162float(__hmul(x3, w[3]));
  return bf(div_full(acc, 1.f + tl_exp(0.f - acc)));
}

// x = hi + lo: hi rounded to tf32 (integer operations), lo exact in fp32 (the MMA reads its top bits: |error| <=
// 2^-22 |x|)
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

__device__ __forceinline__ uint32_t u32(float x) { return __float_as_uint(x); }

__device__ __forceinline__ void fma4(float4& acc, float w, const float4& x) {
  acc.x = fmaf(w, x.x, acc.x);
  acc.y = fmaf(w, x.y, acc.y);
  acc.z = fmaf(w, x.z, acc.z);
  acc.w = fmaf(w, x.w, acc.w);
}

__device__ __forceinline__ float4 scale4(const float4& a, float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// an mbarrier of one arrival that also expects bytes of bulk copies, made visible to the async proxy
__device__ __forceinline__ void bulk_barrier_init(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// a bulk copy global -> shared by the tensor memory accelerator, completing its bytes on the barrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// wait for the barrier's phase 0: every arrival and every expected byte
__device__ __forceinline__ void bulk_barrier_wait(uint32_t bar) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}\n" ::"r"(
          bar)
      : "memory");
}

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

// A row of the step and its slots in a layer's group, read with one load each
struct RowItem {
  int4 head, tail, slots;  // the row's ints 0..3 and 4..7, its slots
};

__device__ __forceinline__ RowItem load_row(const Layer& ly, const Step& sp, int row) {
  const int4* ri = reinterpret_cast<const int4*>(sp.rows + (long long)row * kRowInts);
  return {ri[0], ri[1], *reinterpret_cast<const int4*>(sp.slots + ((long long)ly.group * sp.max_rows + row) * kSlotInts)};
}

// The shared memory of one row's value head
template <int MT>
struct alignas(16) CoreSmem {
  using S = Sizes<MT>;
  alignas(16) float state[kDim * kDim + kHalfShift];  // the head's state: rows of its halves, the second shifted
  float kq[S::KQ_ROWS][kKQStride];            // the keys' then the queries' conv outputs (zero past the step's); then
                                              // the outputs in the queries' rows
  float v[S::L][kVStride];                    // the values' conv outputs (zero past the step's)
  float vec_hi[16 * S::NM][kVecStride];       // the products' vectors (minus the effective keys, the effective
  float vec_lo[16 * S::NM][kVecStride];       // queries, zero rows), split to tf32
  float gram[S::KQ_ROWS][S::GRAM_COLS];       // [keys; queries] . [keys | queries]
  float m[S::L][S::L];                        // the keys' interactions in the delta rule (strictly lower)
  float nq[MT][S::L];                         // the queries' weights of the keys' corrections
  float ck[16 * S::NM][S::JP + 4];            // the vectors' coefficients of the keys' and queries' conv outputs
  float cv[8 * S::NT][S::L + 4];              // the vectors' coefficients of the values (the rows' own parts)
  float wc[S::L], wb[S::L];                   // the written and the boundary state's weights of each correction
  float g[S::L], beta[S::L];                  // the log decays' prefix sums, the betas
  float rk[S::L], rq[MT];                     // the keys' inverse norms, the queries' scaled inverse norms
  float decay_c, decay_b;                     // the written and the boundary state's decays of the slot's state
  alignas(16) float red[MT][kWarps];          // the outputs' squared sums per warp
  uint64_t state_bar;                         // the state's bulk copies complete on it
};

// ---- one row's value head: the replay of its c committed tokens, its block's T outputs (none in a flush), the state
// written once (and a boundary candidate's), this block's record
template <int MT>
__device__ __forceinline__ void core_item(const Layer& ly, const Step& sp, const RowItem& it, int hv, int ih,
                                          CoreSmem<MT>& sm) {
  using S = Sizes<MT>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, tig = lane & 3;
  const int H = sp.H, HV = sp.HV;
  const int C = 2 * H * kDim + HV * kDim;
  const int tok0 = it.head.x, req = it.head.z, pos = it.head.w;
  // replayed tokens and the record copy read: in a flush the block's own tokens, from the copy the step wrote
  const int c = sp.is_flush ? it.head.y : it.tail.x;
  const int T = sp.is_flush ? 0 : it.head.y;
  const int copy = sp.is_flush ? 1 - it.tail.y : it.tail.y;
  const int L = c + T;
  const int dst = it.slots.y;
  const int src = sp.is_flush ? dst : it.slots.x;
  const int bnd_slot = sp.is_flush ? -1 : it.slots.z;
  // the boundary inside this block (not at its end) whose state is written: its token among the L
  const int bnd_j = bnd_slot >= 0 ? c + (pos / sp.block_size + 1) * sp.block_size - pos - 1 : -1;
  const int nk = max(c, bnd_j + 1);  // the effective keys: the tokens whose corrections a written state needs
  const int N = nk + T;
  const int nts = (N + 7) / 8;

  // ---- a thread's state rows (the MMA's rows g8, g8 + 8 of the warp's m-tile): row_a, row_a + 4 of the first half
  // (g8 even) or the second; columns 16 i + 4 tig .. + 3 (the A fragments of the k-steps 2 i, 2 i + 1, a permutation
  // the vectors' fragments follow)
  const int row_a = (g8 & 1) * kHalfRows + 8 * warp + (g8 >> 1), row_b = row_a + 4;
  const long long rows_off = ((long long)hv * kDim + row_a) * kDim + 4 * tig;
  const float* srow0 = sm.state + row_a * kDim + (g8 & 1) * kHalfShift + 4 * tig;
  const float* srow1 = srow0 + 4 * kDim;
  const uint32_t state_bar = smem_u32(&sm.state_bar);
  // ---- the conv: threads < kDim take the key head's q and k channels, the others the value head's v channels; the
  // inputs of a channel: the window (the record's, or the slot's before any record), replayed, current. What the
  // projections' GEMM does not write loads before the wait for it (a programmatic dependent overlaps it with the GEMM)
  const __nv_bfloat16* win = ly.conv + (long long)src * ly.conv_slot;
  const uint8_t* rec_in = ly.record + (long long)req * ly.record_req + (long long)copy * ly.record_copy;
  uint8_t* rec_out = ly.record + (long long)req * ly.record_req + (long long)(1 - copy) * ly.record_copy;
  const __nv_bfloat16* rin = reinterpret_cast<const __nv_bfloat16*>(rec_in);
  __nv_bfloat16* rout = reinterpret_cast<__nv_bfloat16*>(rec_out);
  const bool is_qk = tid < kDim;
  const int ct = tid % kDim;
  const int ch = is_qk ? (H + ih) * kDim + ct : 2 * H * kDim + hv * kDim + ct;  // k, or v
  const int ch_q = ih * kDim + ct;
  const bool has_record = !sp.is_flush && (!is_qk || hv == ih * sp.R);  // the key head's q / k by its first value head
  const __nv_bfloat16 zero = __float2bfloat16(0.f);
  __nv_bfloat16 x[kTaps + S::L], xq[kTaps + MT];
  if (c > 0) {
#pragma unroll
    for (int e = 0; e < kTaps; e++) {
      x[e] = rin[(long long)e * C + ch];
      xq[e] = is_qk ? rin[(long long)(c + e) * C + ch_q] : zero;
    }
  } else {
#pragma unroll
    for (int e = 0; e < kTaps; e++) {
      x[e] = win[ch * ly.conv_ch + e * ly.conv_tap];
      xq[e] = is_qk ? win[ch_q * ly.conv_ch + e * ly.conv_tap] : zero;
    }
  }
#pragma unroll
  for (int e = kTaps; e < kTaps + S::L; e++) x[e] = e < kTaps + c ? rin[(long long)e * C + ch] : zero;
  __nv_bfloat16 w[4], wq[4];
  {
    const uint2 wbits = *reinterpret_cast<const uint2*>(ly.conv_w + 4 * ch);
    const uint2 wqbits = is_qk ? *reinterpret_cast<const uint2*>(ly.conv_w + 4 * ch_q) : make_uint2(0, 0);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wbits);
    const __nv_bfloat16* wqp = reinterpret_cast<const __nv_bfloat16*>(&wqbits);
#pragma unroll
    for (int e = 0; e < 4; e++) w[e] = wp[e], wq[e] = wqp[e];
  }
  // the gate inputs of the replayed tokens (the last warp's lanes j < c), from the record
  __nv_bfloat16 gate_b = zero, gate_a = zero;
  if (warp == kWarps - 1 && lane < c) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(rec_in + sp.record_ba) + lane * 2 * HV;
    gate_b = p[hv], gate_a = p[HV + hv];
  }

  // a programmatic dependent (the verify core) waits for the projections' GEMM here, its state already loading (a
  // plain launch returns at once)
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // the gate inputs of the outputs this thread normalizes (row tid % kDim of tokens tid / kDim + k kThreads / kDim)
  constexpr int kNormPasses = (MT * kDim + kThreads - 1) / kThreads;
  float z[kNormPasses];
#pragma unroll
  for (int k = 0; k < kNormPasses; k++) {
    const int p = tid + k * kThreads, t = p / kDim;
    z[k] = t < T ? __bfloat162float(sp.qkvz[(long long)(tok0 + t) * sp.qkvz_row + C + hv * kDim + p % kDim]) : 0.f;
  }
  // the current inputs: entry e >= kTaps + c is the step's row e - kTaps - c
  const __nv_bfloat16* cur = sp.qkvz + (long long)(tok0 - kTaps - c) * sp.qkvz_row;
#pragma unroll
  for (int e = kTaps; e < kTaps + S::L; e++)
    if (e >= kTaps + c && e < kTaps + L) x[e] = cur[(long long)e * sp.qkvz_row + ch];
  if (is_qk) {
#pragma unroll
    for (int e = kTaps; e < kTaps + MT; e++)
      xq[e] = e < kTaps + T ? cur[(long long)(c + e) * sp.qkvz_row + ch_q] : zero;
  }
  if (warp == kWarps - 1 && lane >= c && lane < L) {
    const __nv_bfloat16* p = sp.ba + (long long)(tok0 + lane - c) * sp.ba_row;
    gate_b = p[hv], gate_a = p[HV + hv];
  }

  // the conv outputs (bf16 values) into their rows, zero past the step's tokens and in the padding rows
  {
    if (is_qk) {
#pragma unroll
      for (int j = 0; j < S::L; j++) sm.kq[j][ct] = j < L ? conv_tap(w, x[j], x[j + 1], x[j + 2], x[j + 3]) : 0.f;
#pragma unroll
      for (int t = 0; t < MT; t++) sm.kq[S::L + t][ct] = t < T ? conv_tap(wq, xq[t], xq[t + 1], xq[t + 2], xq[t + 3]) : 0.f;
#pragma unroll
      for (int p = S::J; p < S::KQ_ROWS; p++) sm.kq[p][ct] = 0.f;
    } else {
#pragma unroll
      for (int j = 0; j < S::L; j++) sm.v[j][ct] = j < L ? conv_tap(w, x[j], x[j + 1], x[j + 2], x[j + 3]) : 0.f;
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
  // ---- the head's state into shared memory by the tensor memory accelerator, after the conv's loads (its traffic
  // would delay them) and off the warps' load path; complete on the state barrier
  if (tid == 0) {
    const float* head = ly.state + (long long)src * ly.state_slot + (long long)hv * kDim * kDim;
    bulk_barrier_init(state_bar, 2 * kHalfBytes);
    bulk_load(sm.state, head, kHalfBytes, state_bar);
    bulk_load(sm.state + kHalfRows * kDim + kHalfShift, head + kHalfRows * kDim, kHalfBytes, state_bar);
  }
  // the gates of the L tokens (the last warp's lanes): log decays' prefix sums, betas (bf16, as the decode kernel's);
  // this block's b / a into its record
  if (warp == kWarps - 1) {
    const int j = lane;
    float gj = 0.f;
    if (j < L) {
      const float xx = __bfloat162float(gate_a) + param(ly.dt_bias, hv, ly.dt_bias_f32);
      const float softplus = xx <= kSoftplusThreshold ? logf(1.f + tl_exp(xx)) : xx;
      gj = (0.f - tl_exp(ly.a_log[hv])) * softplus;
      sm.beta[j] = bf(tl_sigmoid(__bfloat162float(gate_b)));
      if (!sp.is_flush && j >= c) {
        __nv_bfloat16* rout_ba = reinterpret_cast<__nv_bfloat16*>(rec_out + sp.record_ba) + (j - c) * 2 * HV;
        rout_ba[hv] = gate_b;
        rout_ba[HV + hv] = gate_a;
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

  // ---- the gram [keys; queries] . [keys | queries] on tensor cores (exact products of bf16 values): its k-steps split
  // over kGramWarps warps, their partial sums (in the vectors' buffer, not yet in use) added up by every thread
  using GramPart = float[S::KQ_ROWS][S::GRAM_COLS];
  static_assert(sizeof(sm.vec_hi) >= kGramWarps * sizeof(GramPart), "the gram's partial sums fit the vectors' buffer");
  GramPart* part = reinterpret_cast<GramPart*>(&sm.vec_hi[0][0]);
  if (warp < kGramWarps) {
#pragma unroll
    for (int gm = 0; gm < S::KQ_ROWS / 16; gm++) {
      float acc[S::GRAM_COLS / 8][4] = {};
#pragma unroll
      for (int k = 0; k < kGramSteps; k++) {
        const float* a = &sm.kq[16 * gm + g8][8 * (warp * kGramSteps + k) + tig];
        const uint32_t a0 = u32(a[0]), a1 = u32(a[8 * kKQStride]), a2 = u32(a[4]), a3 = u32(a[8 * kKQStride + 4]);
#pragma unroll
        for (int gn = 0; gn < S::GRAM_COLS / 8; gn++) {
          const float* b = &sm.kq[8 * gn + g8][8 * (warp * kGramSteps + k) + tig];
          mma_tf32(acc[gn], a0, a1, a2, a3, u32(b[0]), u32(b[4]));
        }
      }
#pragma unroll
      for (int gn = 0; gn < S::GRAM_COLS / 8; gn++) {
        *reinterpret_cast<float2*>(&part[warp][16 * gm + g8][8 * gn + 2 * tig]) = make_float2(acc[gn][0], acc[gn][1]);
        *reinterpret_cast<float2*>(&part[warp][16 * gm + g8 + 8][8 * gn + 2 * tig]) = make_float2(acc[gn][2], acc[gn][3]);
      }
    }
  }
  __syncthreads();
  for (int e = tid; e < S::KQ_ROWS * S::GRAM_COLS; e += kThreads) {
    const int r = e / S::GRAM_COLS, q = e % S::GRAM_COLS;
    float sum = part[0][r][q];
#pragma unroll
    for (int w = 1; w < kGramWarps; w++) sum += part[w][r][q];
    sm.gram[r][q] = sum;
  }
  __syncthreads();

  // ---- the head's coefficients (the first warp): the norms, the keys' interactions M (j > i) and the queries'
  // weights; A = (I + M)^-1 diag(beta) column by column (lane l: column l); the vectors' coefficients
  if (warp == 0) {
    if (lane < L) sm.rk[lane] = div_full(1.f, sqrt_approx(sm.gram[lane][lane] + kL2Eps));
    if (lane < T) sm.rq[lane] = div_full(1.f, sqrt_approx(sm.gram[S::L + lane][S::L + lane] + kL2Eps)) * sp.scale;
    __syncwarp();
    for (int p = lane; p < S::L * S::L; p += 32) {
      const int j = p / S::L, i = p % S::L;
      if (i < j && j < L) sm.m[j][i] = sm.beta[j] * tl_exp(sm.g[j] - sm.g[i]) * sm.rk[j] * sm.rk[i] * sm.gram[j][i];
    }
    for (int p = lane; p < MT * S::L; p += 32) {
      const int t = p / S::L, i = p % S::L;
      if (t < T)
        sm.nq[t][i] = i <= c + t ? tl_exp(sm.g[c + t] - sm.g[i]) * sm.rq[t] * sm.rk[i] * sm.gram[S::L + t][i] : 0.f;
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
      // key l's column: its decay from the slot's state with its norm; the effective keys negated (the products then
      // give the corrections u = own - S0 ek); zero coefficients past the step's tokens and vectors
      const float ek = l < L ? tl_exp(sm.g[l]) * sm.rk[l] : 0.f;
#pragma unroll
      for (int n = 0; n < S::L; n++) {
        if (n < nk) {
          sm.ck[n][l] = -col[n] * ek;
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
      for (int n = 0; n < 16 * S::NM; n++)
        if (n >= N) sm.ck[n][l] = 0.f;
#pragma unroll
      for (int n = 0; n < 8 * S::NT; n++)
        if (n >= N) sm.cv[n][l] = 0.f;
      if (l < c) sm.wc[l] = tl_exp(sm.g[c - 1] - sm.g[l]) * sm.rk[l];
      if (l <= bnd_j) sm.wb[l] = tl_exp(sm.g[bnd_j] - sm.g[l]) * sm.rk[l];
    } else if (l < S::JP) {
      // query column t: the effective query t's term e^{G_{c+t}} rq_t q_t, in its vector nk + t only
      const int t = l - S::L;
      const float cq = t < T ? tl_exp(sm.g[c + t]) * sm.rq[t] : 0.f;
#pragma unroll
      for (int n = 0; n < 16 * S::NM; n++) sm.ck[n][l] = t < T && n == nk + t ? cq : 0.f;
    }
    if (l == 0) {
      sm.decay_c = c > 0 ? tl_exp(sm.g[c - 1]) : 1.f;
      sm.decay_b = bnd_j >= 0 ? tl_exp(sm.g[bnd_j]) : 0.f;
    }
  }
  __syncthreads();

  // ---- the vectors on tensor cores, [vectors x (keys; queries)] . [(keys; queries) x columns]: the coefficients split
  // to tf32, the conv outputs exact; each warp two column tiles of every vector, split to tf32 for the products
#pragma unroll
  for (int nm = 0; nm < S::NM; nm++) {
    float acc[2][4] = {};
#pragma unroll
    for (int ks = 0; ks < S::JP / 8; ks++) {
      const float* a = &sm.ck[16 * nm + g8][8 * ks + tig];
      uint32_t ah[4], al[4];
      split_tf32(a[0], ah[0], al[0]);
      split_tf32(a[8 * (S::JP + 4)], ah[1], al[1]);
      split_tf32(a[4], ah[2], al[2]);
      split_tf32(a[8 * (S::JP + 4) + 4], ah[3], al[3]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const float* b = &sm.kq[8 * ks + tig][8 * (2 * warp + h) + g8];
        const uint32_t b0 = u32(b[0]), b1 = u32(b[4 * kKQStride]);
        mma_tf32(acc[h], al[0], al[1], al[2], al[3], b0, b1);
        mma_tf32(acc[h], ah[0], ah[1], ah[2], ah[3], b0, b1);
      }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int col0 = 8 * (2 * warp + h) + 2 * tig;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        uint32_t h0, l0, h1, l1;
        split_tf32(acc[h][2 * half], h0, l0);
        split_tf32(acc[h][2 * half + 1], h1, l1);
        const int n = 16 * nm + g8 + 8 * half;
        *reinterpret_cast<float2*>(&sm.vec_hi[n][col0]) = make_float2(__uint_as_float(h0), __uint_as_float(h1));
        *reinterpret_cast<float2*>(&sm.vec_lo[n][col0]) = make_float2(__uint_as_float(l0), __uint_as_float(l1));
      }
    }
  }
  __syncthreads();

  // ---- the rows' own parts (values' transposes, exact, by the value coefficients split to tf32), then their products
  // with the vectors (3xTF32: hi hi + hi lo + lo hi), the state's chunks from shared memory: per n-tile the even and
  // the odd k-steps in chains of their own
  float acc[S::NT][2][4] = {};
#pragma unroll
  for (int ks = 0; ks < S::L / 8; ks++) {
    const float* a = &sm.v[8 * ks + tig][row_a];
    const uint32_t a0 = u32(a[0]), a1 = u32(a[4]), a2 = u32(a[4 * kVStride]), a3 = u32(a[4 * kVStride + 4]);
#pragma unroll
    for (int nt = 0; nt < S::NT; nt++) {
      if (nt < nts) {
        const float* b = &sm.cv[8 * nt + g8][8 * ks + tig];
        uint32_t bh0, bl0, bh1, bl1;
        split_tf32(b[0], bh0, bl0);
        split_tf32(b[4], bh1, bl1);
        mma_tf32(acc[nt][0], a0, a1, a2, a3, bl0, bl1);
        mma_tf32(acc[nt][0], a0, a1, a2, a3, bh0, bh1);
      }
    }
  }
  bulk_barrier_wait(state_bar);
#pragma unroll
  for (int i = 0; i < kColChunks; i++) {
    uint32_t ah[8], al[8];
    const float4 s0 = *reinterpret_cast<const float4*>(srow0 + 16 * i);
    const float4 s1 = *reinterpret_cast<const float4*>(srow1 + 16 * i);
    split_tf32(s0.x, ah[0], al[0]);
    split_tf32(s1.x, ah[1], al[1]);
    split_tf32(s0.y, ah[2], al[2]);
    split_tf32(s1.y, ah[3], al[3]);
    split_tf32(s0.z, ah[4], al[4]);
    split_tf32(s1.z, ah[5], al[5]);
    split_tf32(s0.w, ah[6], al[6]);
    split_tf32(s1.w, ah[7], al[7]);
#pragma unroll
    for (int nt = 0; nt < S::NT; nt++) {
      if (nt < nts) {
        const float4 bh = *reinterpret_cast<const float4*>(&sm.vec_hi[8 * nt + g8][16 * i + 4 * tig]);
        const float4 bl = *reinterpret_cast<const float4*>(&sm.vec_lo[8 * nt + g8][16 * i + 4 * tig]);
        mma_tf32(acc[nt][0], al[0], al[1], al[2], al[3], u32(bh.x), u32(bh.y));
        mma_tf32(acc[nt][1], al[4], al[5], al[6], al[7], u32(bh.z), u32(bh.w));
        mma_tf32(acc[nt][0], ah[0], ah[1], ah[2], ah[3], u32(bl.x), u32(bl.y));
        mma_tf32(acc[nt][1], ah[4], ah[5], ah[6], ah[7], u32(bl.z), u32(bl.w));
        mma_tf32(acc[nt][0], ah[0], ah[1], ah[2], ah[3], u32(bh.x), u32(bh.y));
        mma_tf32(acc[nt][1], ah[4], ah[5], ah[6], ah[7], u32(bh.z), u32(bh.w));
      }
    }
  }
  // D: rows 16 warp + g8 (+ 8 for e >= 2), vectors 8 nt + 2 tig (+ 1 for odd e): the corrections u (n < nk), the
  // outputs (nk <= n < N)
  float pv[S::NT][4];
#pragma unroll
  for (int nt = 0; nt < S::NT; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) pv[nt][e] = acc[nt][0][e] + acc[nt][1][e];

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
      const float wgt = i <= bnd_j ? sm.wb[i] : 0.f;
      wu[0][i] = u[0][i] * wgt, wu[1][i] = u[1][i] * wgt;
    }
    const float eb = sm.decay_b;
    float* bdst = ly.state + (long long)bnd_slot * ly.state_slot + rows_off;
#pragma unroll
    for (int i = 0; i < kColChunks; i++) {
      float4 sb[2] = {scale4(*reinterpret_cast<const float4*>(srow0 + 16 * i), eb),
                       scale4(*reinterpret_cast<const float4*>(srow1 + 16 * i), eb)};
#pragma unroll
      for (int j = 0; j < S::L; j++) {
        const float4 kv = *reinterpret_cast<const float4*>(&sm.kq[j][16 * i + 4 * tig]);
        fma4(sb[0], wu[0][j], kv);
        fma4(sb[1], wu[1][j], kv);
      }
      __stcs(reinterpret_cast<float4*>(bdst + 16 * i), sb[0]);
      __stcs(reinterpret_cast<float4*>(bdst + 4 * kDim + 16 * i), sb[1]);
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
        float4 sc[2] = {scale4(*reinterpret_cast<const float4*>(srow0 + 16 * i), ec),
                         scale4(*reinterpret_cast<const float4*>(srow1 + 16 * i), ec)};
#pragma unroll
        for (int j = 0; j < K; j++) {
          const float4 kv = *reinterpret_cast<const float4*>(&sm.kq[j][16 * i + 4 * tig]);
          fma4(sc[0], wu[0][j], kv);
          fma4(sc[1], wu[1][j], kv);
        }
        __stcs(reinterpret_cast<float4*>(sdst + 16 * i), sc[0]);
        __stcs(reinterpret_cast<float4*>(sdst + 4 * kDim + 16 * i), sc[1]);
      }
    };
    dispatch_count<MT>(c, write_running);
  }
  if (T == 0) return;

  // ---- the gated RMSNorm of each output token over the head's rows: the bf16 outputs into the queries' rows (no
  // longer read) with each warp's squared sums, then each token's row
#pragma unroll
  for (int nt = 0; nt < S::NT; nt++) {
#pragma unroll
    for (int odd = 0; odd < 2; odd++) {
      const int n = 8 * nt + 2 * tig + odd;
      const float o0 = bf(pv[nt][odd]), o1 = bf(pv[nt][2 + odd]);
      float s = o0 * o0 + o1 * o1;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) s += __shfl_xor_sync(kFullMask, s, o);
      if (n >= nk && n < N) {
        sm.kq[S::L + n - nk][row_a] = o0;
        sm.kq[S::L + n - nk][row_b] = o1;
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
    for (int wp = 0; wp < kWarps; wp += 4) {
      const float4 part = *reinterpret_cast<const float4*>(&sm.red[t][wp]);
      s += (part.x + part.y) + (part.z + part.w);
    }
    const float rstd = rsqrt_approx(div_full(s, (float)kDim) + sp.norm_eps);
    const float gate = sp.gate_sigmoid ? tl_sigmoid(z[k]) : z[k] * tl_sigmoid(z[k]);
    float y = sm.kq[S::L + t][rr] * rstd * nw;
    y *= gate;
    sp.out[(long long)(tok0 + t) * sp.out_row + hv * kDim + rr] = __float2bfloat16(y);
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads, 2) gdn_verify_kernel(const Layer ly, const Step sp) {
  extern __shared__ float4 smem_raw[];
  CoreSmem<MT>& sm = *reinterpret_cast<CoreSmem<MT>*>(smem_raw);
  const int row = blockIdx.x, ih = blockIdx.z, hv = ih * sp.R + blockIdx.y;
  // the next kernel, if a programmatic dependent, launches once every CTA has started (out_proj's GEMM is not)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const RowItem it = load_row(ly, sp, row);
  if (it.head.y == 0) return;  // not a verify row (a padding or prefill row)
  core_item<MT>(ly, sp, it, hv, ih, sm);
}

// ---- after sampling: the window at a committed boundary inside a verify row's block into the boundary block's conv
// slot, from the record the step wrote (its entry 0 holds the input at position pos - 3): the head's v channels and,
// from its key head's first value head, the key head's q / k channels
__device__ __forceinline__ void boundary_window(const Layer& ly, const Step& sp, const RowItem& it, int hv, int slot,
                                                int boundary) {
  const int tid = threadIdx.x;
  if (tid >= kDim) return;
  const int req = it.head.z, pos = it.head.w;
  const int H = sp.H, HV = sp.HV, R = sp.R, ih = hv / R;
  const int C = 2 * H * kDim + HV * kDim;
  const __nv_bfloat16* rec = reinterpret_cast<const __nv_bfloat16*>(
      ly.record + (long long)req * ly.record_req + (long long)(1 - it.tail.y) * ly.record_copy);
  const int first = boundary - pos;  // the window's first entry: the input at position boundary - 3
#pragma unroll
  for (int which = 0; which < 3; which++) {
    if (which < 2 && hv % R != 0) continue;
    const int ch = which == 0 ? ih * kDim + tid : which == 1 ? (H + ih) * kDim + tid : 2 * H * kDim + hv * kDim + tid;
    __nv_bfloat16* wdst = ly.conv + (long long)slot * ly.conv_slot + ch * ly.conv_ch;
#pragma unroll
    for (int e = 0; e < kTaps; e++) wdst[e * ly.conv_tap] = rec[(long long)(first + e) * C + ch];
  }
}

// After sampling, for each verify row whose block's committed tokens reach a block boundary: the window there into the
// boundary block (the forward wrote its state); a block that ends at the boundary, committed whole, has its state in
// the running slot, which this replays to the boundary (the next step then finds no record). One CTA per value head
// and layer; its threads first find the rows that reach a boundary.
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
      const RowItem it = load_row(ly, sp, found[f]);
      const int T = it.head.y, pos = it.head.w;
      const int boundary = (pos / sp.block_size + 1) * sp.block_size;
      const bool is_block_end = boundary == pos + T;
      boundary_window(ly, sp, it, hv, is_block_end ? it.slots.y : it.slots.z, boundary);
      if (is_block_end) core_item<MT>(ly, sp, it, hv, hv / sp.R, sm);
      __syncthreads();
    }
  }
}

// The step's rows and slots (before the forward): each row's first token, verify tokens, request, position, replayed
// tokens and record copy, and in every group its source slot (the previous running block), destination (this step's
// running block, vLLM's align column) and the block of a boundary inside its block; vLLM's align bookkeeping for the
// precopy of the other rows
__global__ void __launch_bounds__(kMetaThreads) replay_meta_kernel(
    const int64_t* idx_mapping, const int* query_start, const int* computed, const uint8_t* is_verify,
    const int* replay, const int* parity, int* state_col, int* src_col, int* src_off, int* accepted,
    const int64_t* tables, long long table_stride, int num_groups, int num_reqs, int num_rows, int max_rows,
    int block_size, int* rows, int* slots) {
  const int row = blockIdx.x * kMetaThreads + threadIdx.x;
  if (row >= num_rows) return;
  int4* ri = reinterpret_cast<int4*>(rows + (long long)row * kRowInts);
  if (row >= num_reqs) {
    ri[0] = ri[1] = make_int4(0, 0, 0, 0);
    return;
  }
  const int req = (int)idx_mapping[row];
  const int tok0 = query_start[row], T = query_start[row + 1] - tok0;
  const int pos = computed[req];
  const bool is_verify_row = is_verify == nullptr || is_verify[row];
  const int old_col = state_col[req];
  const int new_col = (pos + T + block_size - 1) / block_size - 1;
  src_off[req] = max(accepted[req] - 1, 0);
  src_col[req] = is_verify_row ? -1 : old_col;
  state_col[req] = new_col;
  if (old_col >= 0 && old_col != new_col) accepted[req] = 1;
  ri[0] = make_int4(tok0, is_verify_row ? T : 0, req, pos);
  ri[1] = make_int4(replay[req], parity[req], 0, 0);
  const int boundary = (pos / block_size + 1) * block_size;
  for (int g = 0; g < num_groups; g++) {
    const int* table = reinterpret_cast<const int*>(tables[g]) + (long long)row * table_stride;
    *reinterpret_cast<int4*>(slots + ((long long)g * max_rows + row) * kSlotInts) =
        make_int4(table[old_col >= 0 ? old_col : new_col], table[new_col],
                  boundary < pos + T ? table[boundary / block_size - 1] : -1, 0);
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
Layer make_layer(const torch::Tensor& state, const torch::Tensor& conv, const torch::Tensor& record,
                 const torch::Tensor& conv_w, const torch::Tensor& a_log, const torch::Tensor& dt_bias,
                 const torch::Tensor& norm_w, int64_t group, bool conv_dim_first) {
  TORCH_CHECK(state.scalar_type() == at::kFloat && conv.scalar_type() == at::kBFloat16, "gdn_verify: state dtypes");
  TORCH_CHECK(state.size(2) == kDim && state.size(3) == kDim && state.stride(3) == 1 && state.stride(2) == kDim &&
                  state.stride(1) == kDim * kDim,
              "gdn_verify: the state cache must be [slots][HV][128][128] with contiguous slots");
  TORCH_CHECK(state.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
              "gdn_verify: 16-byte aligned state slots");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && conv_w.scalar_type() == at::kBFloat16 && conv_w.is_contiguous() &&
                  conv_w.numel() == conv_w.size(0) * 4,
              "gdn_verify: fp32 A_log, bf16 [C][4] conv weights");
  TORCH_CHECK(record.scalar_type() == at::kByte && record.dim() == 3 && record.size(1) == 2 && record.stride(2) == 1,
              "gdn_verify: uint8 records [requests][2][bytes]");
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

Step make_step(const torch::Tensor& rows, const torch::Tensor& slots, int64_t num_heads, int64_t num_v_heads,
               int64_t block_size, int64_t record_ba, double scale) {
  TORCH_CHECK(rows.scalar_type() == at::kInt && slots.scalar_type() == at::kInt && rows.is_contiguous() &&
                  slots.is_contiguous(),
              "gdn_verify: int32 rows and slots");
  TORCH_CHECK(rows.size(1) == kRowInts && slots.size(2) == kSlotInts, "gdn_verify: row and slot widths");
  TORCH_CHECK(num_v_heads % num_heads == 0 && num_v_heads / num_heads <= 32, "gdn_verify: value heads per key head");
  Step sp{};
  sp.rows = rows.data_ptr<int>();
  sp.slots = slots.data_ptr<int>();
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
  if (max_tokens <= 4)
    f(std::integral_constant<int, 4>{});
  else
    f(std::integral_constant<int, 8>{});
}

// One layer's verify core over the step's first num_rows rows: one CTA per row and value head
void verify(const torch::Tensor& qkvz, const torch::Tensor& ba, torch::Tensor& out, const torch::Tensor& conv_w,
            const torch::Tensor& a_log, const torch::Tensor& dt_bias, const torch::Tensor& norm_w, torch::Tensor& state,
            torch::Tensor& conv, torch::Tensor& record, const torch::Tensor& rows, const torch::Tensor& slots,
            int64_t group, int64_t num_rows, int64_t num_heads, int64_t block_size, int64_t max_tokens,
            int64_t record_ba, double scale, double norm_eps, bool gate_sigmoid, bool conv_dim_first) {
  if (num_rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(qkvz.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const Layer ly = make_layer(state, conv, record, conv_w, a_log, dt_bias, norm_w, group, conv_dim_first);
  const int64_t num_v_heads = state.size(1);
  TORCH_CHECK(qkvz.scalar_type() == at::kBFloat16 && ba.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16 && qkvz.stride(1) == 1 && ba.stride(1) == 1 &&
                  out.dim() == 3 && out.stride(2) == 1 && out.stride(1) == kDim,
              "gdn_verify: bf16 inputs with contiguous rows, out [tokens][HV][V]");
  TORCH_CHECK(qkvz.size(1) >= 2 * num_heads * kDim + 2 * num_v_heads * kDim && ba.size(1) >= 2 * num_v_heads &&
                  out.size(1) == num_v_heads && out.size(2) == kDim,
              "gdn_verify: projection and output widths");
  TORCH_CHECK(num_rows <= slots.size(1) && num_rows <= rows.size(0), "gdn_verify: rows past the step's tables");
  Step sp = make_step(rows, slots, num_heads, num_v_heads, block_size, record_ba, scale);
  sp.qkvz = reinterpret_cast<const __nv_bfloat16*>(qkvz.data_ptr());
  sp.ba = reinterpret_cast<const __nv_bfloat16*>(ba.data_ptr());
  sp.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  sp.qkvz_row = qkvz.stride(0);
  sp.ba_row = ba.stride(0);
  sp.out_row = out.stride(0);
  sp.norm_eps = norm_eps;
  sp.gate_sigmoid = gate_sigmoid;
  dispatch_tokens(max_tokens, [&](auto mt) {
    constexpr int MT = decltype(mt)::value;
    static bool is_set = false;
    if (!is_set) {
      set_smem(gdn_verify_kernel<MT>, sizeof(CoreSmem<MT>));
      is_set = true;
    }
    // a programmatic dependent of the projections' GEMM: each CTA reads its row and record before it waits for the
    // GEMM; the grid: rows x value heads of a key head x key heads
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
torch::Tensor layer_table(std::vector<torch::Tensor> states, std::vector<torch::Tensor> convs,
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
void commit(const torch::Tensor& layers, torch::Tensor& rows, const torch::Tensor& slots, torch::Tensor& replay,
            torch::Tensor& parity, torch::Tensor& accepted, const torch::Tensor& idx_mapping,
            const torch::Tensor& sampled, int64_t num_heads, int64_t num_v_heads, int64_t block_size,
            int64_t max_tokens, int64_t record_ba, double scale) {
  const int num_rows = idx_mapping.size(0);
  if (num_rows == 0) return;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && sampled.scalar_type() == at::kInt &&
                  accepted.scalar_type() == at::kInt && replay.scalar_type() == at::kInt &&
                  parity.scalar_type() == at::kInt,
              "gdn_commit: int64 idx_mapping, int32 counts");
  TORCH_CHECK(num_rows <= rows.size(0), "gdn_commit: rows past the step's tables");
  const at::cuda::OptionalCUDAGuard guard(rows.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_layers = layers.size(0) / sizeof(Layer);
  Step sp = make_step(rows, slots, num_heads, num_v_heads, block_size, record_ba, scale);
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
void replay_meta(const torch::Tensor& idx_mapping, const torch::Tensor& query_start, const torch::Tensor& computed,
                 const std::optional<torch::Tensor>& is_verify, const torch::Tensor& replay,
                 const torch::Tensor& parity, torch::Tensor& state_col, torch::Tensor& src_col, torch::Tensor& src_off,
                 torch::Tensor& accepted, const torch::Tensor& tables, int64_t table_stride, int64_t num_reqs,
                 int64_t num_rows, int64_t block_size, torch::Tensor& rows, torch::Tensor& slots) {
  if (num_rows == 0) return;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && query_start.scalar_type() == at::kInt &&
                  computed.scalar_type() == at::kInt && tables.scalar_type() == at::kLong &&
                  replay.scalar_type() == at::kInt && parity.scalar_type() == at::kInt,
              "gdn_replay_meta: int64 idx_mapping and tables, int32 positions and counts");
  TORCH_CHECK(num_rows <= rows.size(0) && num_rows <= slots.size(1) && rows.size(1) == kRowInts &&
                  slots.size(2) == kSlotInts,
              "gdn_replay_meta: the step's tables");
  const at::cuda::OptionalCUDAGuard guard(rows.device());
  replay_meta_kernel<<<(num_rows + kMetaThreads - 1) / kMetaThreads, kMetaThreads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      idx_mapping.data_ptr<int64_t>(), query_start.data_ptr<int>(), computed.data_ptr<int>(),
      is_verify.has_value() ? is_verify->data_ptr<uint8_t>() : nullptr, replay.data_ptr<int>(),
      parity.data_ptr<int>(), state_col.data_ptr<int>(), src_col.data_ptr<int>(), src_off.data_ptr<int>(),
      accepted.data_ptr<int>(), tables.data_ptr<int64_t>(), table_stride, tables.size(0), num_reqs, num_rows,
      slots.size(1), block_size, rows.data_ptr<int>(), slots.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_verify

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "gdn_verify(Tensor qkvz, Tensor ba, Tensor(a!) out, Tensor conv_w, Tensor a_log, Tensor dt_bias, Tensor norm_w, "
      "Tensor(b!) state, Tensor(c!) conv, Tensor(d!) record, Tensor rows, Tensor slots, int group, int num_rows, "
      "int num_heads, int block_size, int max_tokens, int record_ba, float scale, float norm_eps, bool gate_sigmoid, "
      "bool conv_dim_first) -> ()");
  m.def(
      "gdn_layer_table(Tensor[] states, Tensor[] convs, Tensor[] records, Tensor[] conv_ws, Tensor[] a_logs, "
      "Tensor[] dt_biases, Tensor[] norm_ws, int[] groups, bool conv_dim_first) -> Tensor");
  m.def(
      "gdn_commit(Tensor layers, Tensor(a!) rows, Tensor slots, Tensor(b!) replay, Tensor(c!) parity, "
      "Tensor(d!) accepted, Tensor idx_mapping, Tensor sampled, int num_heads, int num_v_heads, int block_size, "
      "int max_tokens, int record_ba, float scale) -> ()");
  m.def(
      "gdn_replay_meta(Tensor idx_mapping, Tensor query_start, Tensor computed, Tensor? is_verify, Tensor replay, "
      "Tensor parity, Tensor(a!) state_col, Tensor(b!) src_col, Tensor(c!) src_off, Tensor(d!) accepted, "
      "Tensor tables, int table_stride, int num_reqs, int num_rows, int block_size, Tensor(e!) rows, "
      "Tensor(f!) slots) -> ()");
}

TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) {
  m.impl("gdn_verify", &gdn_verify::verify);
  m.impl("gdn_commit", &gdn_verify::commit);
  m.impl("gdn_replay_meta", &gdn_verify::replay_meta);
}

TORCH_LIBRARY_IMPL(pear_rollout_hopper, CompositeExplicitAutograd, m) {
  m.impl("gdn_layer_table", &gdn_verify::layer_table);
}
