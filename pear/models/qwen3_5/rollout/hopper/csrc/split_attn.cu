// Split-KV paged decode attention for sm_90: the decode rows of a step (one query token per sequence), a GQA group of
// HQ query heads per KV head, head dim HD, over the KV cache in vLLM's FlashAttention layout (per page of PAGE tokens:
// [token][kv head][K | V] bf16)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/all.h>

#include <mutex>
#include <unordered_map>

#include "tensor_map.cuh"

namespace split_attn {

constexpr int HD = 256;            // head dim
constexpr int TILE = 64;           // tokens per tile
constexpr int MAX_NCW = 8;         // consumer warps (at most)
constexpr int BOX_COLS = 64;       // bf16 columns per TMA box (128 B: the 128B swizzle span)
constexpr int NBOX = 2 * HD / BOX_COLS;         // 8 boxes per tile: K then V
constexpr int MAX_SEQS = 256;
constexpr int MAX_CTAS_ALLOWED = 160;

// everything the kernel reads sits in device memory (the number of decode rows, their lengths and block tables): the
// launch is fixed, so it can live in a CUDA graph
struct Params {
  const __nv_bfloat16* q;
  int64_t q_tok, q_head;  // strides (elements)
  __nv_bfloat16* out;
  int64_t o_tok, o_head;
  const int* block_table;
  int64_t bt_stride;
  const int* seq_lens;
  const int* num_dec;  // device scalar: decode rows (the first num_dec sequences)
  float* part_o;       // [slots][HQ][HD]
  float* part_ml;      // [slots][HQ][2]
  int* counters;       // [MAX_SEQS] partials merged per sequence (zero between calls)
  int page;            // tokens per page
  int tiles_per_page;
  float scale_log2;    // softmax scale * log2(e)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n }\n" ::"r"(smem_u32(bar)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int PAGE, int TILE_>
__device__ __forceinline__ int tiles_of(int len) {
  constexpr int TPP = (PAGE + TILE_ - 1) / TILE_;
  const int full = len / PAGE, rem = len - full * PAGE;
  return full * TPP + (rem + TILE_ - 1) / TILE_;
}

// prefix sums of the decode rows' tile counts into s_start[0..D] (all NTHREADS threads of the block take part)
template <int PAGE, int TILE_, int NTHREADS>
__device__ __forceinline__ void tile_prefix(const Params& p, int D, int* s_start, int* s_warp) {
  const int tid = threadIdx.x;
  constexpr int PER = (MAX_SEQS + NTHREADS - 1) / NTHREADS;
  int v[PER];
  int tot = 0;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    int i = tid * PER + j;
    v[j] = i < D ? tiles_of<PAGE, TILE_>(__ldg(p.seq_lens + i)) : 0;
    tot += v[j];
  }
  // inclusive warp scan of tot
  const int lane = tid & 31, warp = tid >> 5;
  int x = tot;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  int base = 0;
  for (int w = 0; w < warp; ++w) base += s_warp[w];
  int run = base + x - tot;  // exclusive prefix of this thread's first sequence
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    int i = tid * PER + j;
    if (i <= D) s_start[i] = run;
    run += v[j];
  }
  if (tid == 0 && D == 0) s_start[0] = 0;
  __syncthreads();
}

__device__ __forceinline__ int seq_of_tile(const int* s_start, int D, int t) {
  // largest s with s_start[s] <= t (s < D)
  int lo = 0, hi = D - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (s_start[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t movmatrix_t(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}

// a persistent grid streams the batch's KV with TMA into a STAGES-deep ring of TILE-token tiles (16, 32 or 64), the
// batch's tiles split evenly over the CTAs in sequence order: one producer warp issues the loads, NCW consumer warps
// own 16-row slices (with TILE = 16 every warp takes every fourth tile and one warp consumes each stage)
template <int HQ, int PAGE, int TILE_, int STAGES, int NCW, int NS, bool PLO>
__global__ void __launch_bounds__((NCW + 1) * 32, 1) decode_attn_kernel(const __grid_constant__ CUtensorMap tmap, Params p) {
  static_assert(HQ <= 8, "heads of a GQA group in the 8 columns of an n8 fragment");
  constexpr int NTHREADS = (NCW + 1) * 32;
  constexpr int SL = 16 * NS;             // rows per warp and tile (NS 16-row slices, interleaved for ILP)
  constexpr int WPT = TILE_ / SL;         // warps per tile
  constexpr int NGROUPS = NCW / WPT;      // tile groups (a warp takes the tiles j = its group mod NGROUPS)
  constexpr int TPP = (PAGE + TILE_ - 1) / TILE_;
  constexpr int BOXB = TILE_ * BOX_COLS * 2;  // bytes per box
  constexpr int STB = NBOX * BOXB;            // bytes per stage
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* stages = smem;
  constexpr int NACC = NCW < 4 ? NCW : 4;                        // the flush's accumulators (warp w adds into w % 4)
  float* acc_s = reinterpret_cast<float*>(smem + STAGES * STB);  // [NACC][HQ][HD] the flush's merged O
  float* red = acc_s + NACC * HQ * HD;                           // [NCW][8][2] warps' (m, l) per head
  float* fin = red + MAX_NCW * 16;                               // [8][2] merged (m, l) per head
  int* s_start = reinterpret_cast<int*>(fin + 16);               // [MAX_SEQS + 1]
  int* s_warp = s_start + MAX_SEQS + 1;                          // [MAX_NCW + 1]
  volatile int* last_flag = s_warp + MAX_NCW + 1;                // [1]
  float* mw = reinterpret_cast<float*>(s_warp + MAX_NCW + 2);    // [HQ][MAX_CTAS_ALLOWED + 1] the merge's weights
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(
      (reinterpret_cast<uintptr_t>(mw + HQ * (MAX_CTAS_ALLOWED + 1)) + 7) & ~uintptr_t(7));  // [STAGES]
  uint64_t* empty_bar = full_bar + STAGES;                                         // [STAGES]
  // the tile each stage holds or is loading (a group's wait must not take another group's older phase for its own:
  // the parity of a stage's phase repeats every two uses, and interleaved groups can be two uses apart)
  volatile int* stage_tile = reinterpret_cast<volatile int*>(empty_bar + STAGES);  // [STAGES]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int D = min(__ldg(p.num_dec), MAX_SEQS);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], WPT);
      stage_tile[s] = -1;
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tile_prefix<PAGE, TILE_, NTHREADS>(p, D, s_start, s_warp);  // includes __syncthreads
  if (D == 0) return;
  const int T = s_start[D];
  const int P = min((int)gridDim.x, T), c = blockIdx.x;  // every working CTA gets at least one tile
  if (c >= P) return;
  const int t0 = (int)((int64_t)c * T / P), t1 = (int)((int64_t)(c + 1) * T / P);
  if (t0 >= t1) return;

  if (warp == NCW) {
    // producer: one thread issues every tile's TMA boxes, in order; a tile's page is looked up before its stage frees
    if (lane == 0) {
      int s = seq_of_tile(s_start, D, t0);
      int lt = t0 - s_start[s];
      int nxt_block = 0, nxt_row = 0;
      auto locate = [&](int t) {
        while (t >= s_start[s + 1]) { ++s; lt = 0; }
        const int pg = lt / TPP;
        nxt_row = (lt - pg * TPP) * TILE_;
        nxt_block = __ldg(p.block_table + (int64_t)s * p.bt_stride + pg);
        ++lt;
      };
      locate(t0);
      for (int t = t0, j = 0; t < t1; ++t, ++j) {
        const int block = nxt_block, row = nxt_row;
        if (t + 1 < t1) locate(t + 1);
        const int st = j % STAGES;
        if (j >= STAGES) mbar_wait(&empty_bar[st], ((j / STAGES) & 1) ^ 1);
        uint8_t* dst = stages + st * STB;
        stage_tile[st] = j;
        mbar_expect_tx(&full_bar[st], STB);
#pragma unroll
        for (int b = 0; b < NBOX; ++b) tma_load_3d(dst + b * BOXB, &tmap, &full_bar[st], b * BOX_COLS, row, block);
      }
    }
    return;
  }

  // consumers: S^T = K Q^T (16 tokens x 8 heads per warp and tile) and O^T += V^T P^T (dims x heads), so the heads fill
  // the n8 side of the mma; every warp keeps its own running max and sum, merged per sequence in flush
  const int gw = warp / WPT, sub = warp % WPT;
  const int g = lane >> 2;          // fragment row group
  const int qc = (lane & 3) * 2;    // fragment column pair: this thread's heads qc, qc + 1
  const int mi = lane >> 3, ri = lane & 7;  // ldmatrix: this thread's matrix and row
  uint32_t qf[16][2];               // Q^T B fragments per k-step (head g, dims 16 kk + qc (+8))
  float o[16][4];                   // O^T: m16-tile i = dims 16 i + g (+8) x heads qc, qc + 1
  float m_run[2], l_run[2];         // heads qc, qc + 1 (l: this thread's tokens only)

  const int wr0 = sub * SL;         // this warp's first row in its tiles

  int s = seq_of_tile(s_start, D, t0);
  for (; s < D && s_start[s] < t1; ++s) {
    const int a = max(t0, s_start[s]), b = min(t1, s_start[s + 1]);
    if (a >= b) continue;
    const int len = __ldg(p.seq_lens + s);
    {  // Q^T fragments of this sequence
      const __nv_bfloat16* qp = p.q + (int64_t)s * p.q_tok + (int64_t)g * p.q_head;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        if (g < HQ) {
          qf[kk][0] = *reinterpret_cast<const uint32_t*>(qp + kk * 16 + qc);
          qf[kk][1] = *reinterpret_cast<const uint32_t*>(qp + kk * 16 + 8 + qc);
        } else {
          qf[kk][0] = qf[kk][1] = 0u;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    m_run[0] = m_run[1] = -INFINITY;
    l_run[0] = l_run[1] = 0.f;
    // this warp's tiles of the segment: global index t = a + k with (t - t0) % NGROUPS == gw
    int t = a + (((gw - (a - t0)) % NGROUPS) + NGROUPS) % NGROUPS;
    for (; t < b; t += NGROUPS) {
      const int jj = t - t0;
      const int lt = t - s_start[s];
      const int pg = lt / TPP;
      const int row0 = (lt - pg * TPP) * TILE_;
      const int nvalid = min(TILE_, min(PAGE - row0, len - pg * PAGE - row0)) - wr0;  // valid rows of this warp's SL
      const int st = jj % STAGES;
      while (stage_tile[st] != jj) {
      }
      mbar_wait(&full_bar[st], (jj / STAGES) & 1);
      uint8_t* tile = stages + st * STB;
      if (nvalid > 0) {
        const uint32_t kb = smem_u32(tile);
        const uint32_t vb = kb + (HD / BOX_COLS) * BOXB;
        if (nvalid < SL) {
          // V rows past the sequence's end hold whatever the page held: zero them (P is 0 there, 0 * NaN is not)
          for (int idx = lane; idx < SL * 32; idx += 32) {
            const int row = wr0 + idx / 32, ch = idx % 32;  // 32 16-byte chunks per V row (4 boxes x 8)
            if (row - wr0 >= nvalid)
              *reinterpret_cast<uint4*>(tile + (HD / BOX_COLS + ch / 8) * BOXB + row * 128 + (((ch & 7) ^ (row & 7)) << 4)) =
                  make_uint4(0, 0, 0, 0);
          }
          __syncwarp();
        }
        // S^T (SL tokens x 8 heads) = K Q^T: per 16-row slice two interleaved accumulators
        float sacc[NS][2][4];
#pragma unroll
        for (int x = 0; x < NS; ++x)
#pragma unroll
          for (int y = 0; y < 2; ++y) sacc[x][y][0] = sacc[x][y][1] = sacc[x][y][2] = sacc[x][y][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
#pragma unroll
          for (int x = 0; x < NS; ++x) {
            const int krow = wr0 + x * 16 + (mi & 1) * 8 + ri;
            uint32_t a0, a1, a2, a3;
            // dims 16 kk + (mi >> 1) * 8: box kk / 4, chunk (kk % 4) * 2 + (mi >> 1)
            ldsm_x4(kb + (kk / 4) * BOXB + (krow * 128 + ((((kk % 4) * 2 + (mi >> 1)) ^ (krow & 7)) << 4)), a0, a1, a2, a3);
            mma_bf16(sacc[x][kk & 1], a0, a1, a2, a3, qf[kk][0], qf[kk][1]);
          }
        }
        // the softmax as FlashAttention 3 computes it (the reference's kernel): maxima of the raw scores, then
        // p = ex2.approx(s * scale_log2 - max * scale_log2) as one fma, rescale ex2.approx((max_old - max_new) * scale_log2)
        float sv[NS][4];  // slice x: (token x 16 + g, heads qc, qc + 1), (token x 16 + g + 8, heads qc, qc + 1)
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int x = 0; x < NS; ++x)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float v = sacc[x][0][e] + sacc[x][1][e];
            sv[x][e] = x * 16 + g + (e >> 1) * 8 < nvalid ? v : -INFINITY;
            mx[e & 1] = fmaxf(mx[e & 1], sv[x][e]);
          }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 4));
          mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 8));
          mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 16));
        }
        float alpha[2], ms[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float mn = fmaxf(m_run[h], mx[h]);
          ms[h] = mn == -INFINITY ? 0.f : mn * p.scale_log2;
          alpha[h] = mn == -INFINITY ? 1.f : ex2_approx((m_run[h] - mn) * p.scale_log2);
          m_run[h] = mn;
        }
        // P^T as bf16 (the PV mma's operand); with PLO also the residual P - bf16(P) as a second bf16 term, so the
        // product carries P to ~16 bits (FA3 rounds P to bf16: the attention's largest rounding error)
        uint32_t pb[NS][2], pl[NS][2];
        float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
        for (int x = 0; x < NS; ++x) {
          const float p0 = ex2_approx(__fmaf_rn(sv[x][0], p.scale_log2, -ms[0]));
          const float p1 = ex2_approx(__fmaf_rn(sv[x][1], p.scale_log2, -ms[1]));
          const float p2 = ex2_approx(__fmaf_rn(sv[x][2], p.scale_log2, -ms[0]));
          const float p3 = ex2_approx(__fmaf_rn(sv[x][3], p.scale_log2, -ms[1]));
          ls0 += p0 + p2;
          ls1 += p1 + p3;
          // P^T B fragments: transpose the (token, head) pairs into (head, token) pairs
          pb[x][0] = movmatrix_t(pack_bf16(p0, p1));
          pb[x][1] = movmatrix_t(pack_bf16(p2, p3));
          if (PLO) {
            const __nv_bfloat162 h01 = __floats2bfloat162_rn(p0, p1), h23 = __floats2bfloat162_rn(p2, p3);
            pl[x][0] = movmatrix_t(pack_bf16(p0 - __low2float(h01), p1 - __high2float(h01)));
            pl[x][1] = movmatrix_t(pack_bf16(p2 - __low2float(h23), p3 - __high2float(h23)));
          }
        }
        l_run[0] = l_run[0] * alpha[0] + ls0;
        l_run[1] = l_run[1] * alpha[1] + ls1;
        if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
          for (int i2 = 0; i2 < 16; ++i2) {
            o[i2][0] *= alpha[0];
            o[i2][1] *= alpha[1];
            o[i2][2] *= alpha[0];
            o[i2][3] *= alpha[1];
          }
        }
#pragma unroll
        for (int i2 = 0; i2 < 16; ++i2) {
#pragma unroll
          for (int x = 0; x < NS; ++x) {
            const int vrow = wr0 + x * 16 + (mi >> 1) * 8 + ri;
            uint32_t a0, a1, a2, a3;
            // dims 16 i2 + (mi & 1) * 8: box i2 / 4, chunk (i2 % 4) * 2 + (mi & 1)
            ldsm_x4_t(vb + (i2 / 4) * BOXB + (vrow * 128 + ((((i2 % 4) * 2 + (mi & 1)) ^ (vrow & 7)) << 4)), a0, a1, a2, a3);
            mma_bf16(o[i2], a0, a1, a2, a3, pb[x][0], pb[x][1]);
            if (PLO) mma_bf16(o[i2], a0, a1, a2, a3, pl[x][0], pl[x][1]);
          }
        }
        if (nvalid < SL) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the zeroing before the next TMA
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty_bar[st]);
    }
    // merge the warps' states in smem, then write the normalized output (the CTA has the whole sequence) or an fp32
    // partial (unnormalized O, max, sum) to slot c + s
    float lw[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float l = l_run[h];
      l += __shfl_xor_sync(0xffffffffu, l, 4);
      l += __shfl_xor_sync(0xffffffffu, l, 8);
      l += __shfl_xor_sync(0xffffffffu, l, 16);
      lw[h] = l;
    }
    if (g == 0) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        red[(warp * 8 + qc + h) * 2] = m_run[h];
        red[(warp * 8 + qc + h) * 2 + 1] = lw[h];
      }
    }
    named_bar_sync(1, NCW * 32);
    float sc[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int hh = qc + h;
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < NCW; ++w) M = fmaxf(M, red[(w * 8 + hh) * 2]);
      const float Mu = M == -INFINITY ? 0.f : M;  // raw score units (the scale applies inside the exponent)
      float L = 0.f;
#pragma unroll
      for (int w = 0; w < NCW; ++w) L += red[(w * 8 + hh) * 2 + 1] * exp2f((red[(w * 8 + hh) * 2] - Mu) * p.scale_log2);
      sc[h] = exp2f((m_run[h] - Mu) * p.scale_log2);
      if (warp == 0 && g == 0) {
        fin[hh * 2] = M;
        fin[hh * 2 + 1] = L;
      }
    }
    // the warps' scaled states merged in a fixed order (the result must not depend on which warp gets there first):
    // warps 0..3 store into accumulators 0..3, warps 4..7 then add into accumulators 0..3, the reader sums the four
    {
      float* acc = acc_s + (warp % NACC) * HQ * HD;
      for (int round = 0; round < NCW / NACC; ++round) {
        if (warp / NACC == round) {
#pragma unroll
          for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int hh = qc + (e & 1);
              const int at = hh * HD + i * 16 + g + (e >> 1) * 8;
              if (hh < HQ) acc[at] = (round ? acc[at] : 0.f) + o[i][e] * sc[e & 1];
            }
        }
        named_bar_sync(1, NCW * 32);
      }
    }
    auto merged = [&](int idx) {
      float v = acc_s[idx];
#pragma unroll
      for (int k = 1; k < NACC; ++k) v += acc_s[k * HQ * HD + idx];
      return v;
    };
    const bool whole = s_start[s] >= t0 && s_start[s + 1] <= t1;
    if (whole) {
      for (int idx = tid; idx < HQ * HD; idx += NCW * 32) {
        const int h = idx / HD, d = idx - h * HD;
        p.out[(int64_t)s * p.o_tok + (int64_t)h * p.o_head + d] = __float2bfloat16(merged(idx) / fin[h * 2 + 1]);
      }
    } else {
      const int slot = c + s;
      for (int idx = tid; idx < HQ * HD; idx += NCW * 32) {
        p.part_o[(int64_t)slot * HQ * HD + idx] = merged(idx);
      }
      if (tid < HQ)
        *reinterpret_cast<float2*>(p.part_ml + ((int64_t)slot * HQ + tid) * 2) = make_float2(fin[tid * 2], fin[tid * 2 + 1]);
      // the last of the sequence's CTAs to get here merges its partials (CTAs c_lo..c_hi, slots c + s)
      __threadfence();
      named_bar_sync(1, NCW * 32);
      const int c_lo = (int)((((int64_t)s_start[s] + 1) * P - 1) / T);
      const int c_hi = (int)((((int64_t)s_start[s + 1]) * P - 1) / T);
      if (tid == 0) last_flag[0] = atomicAdd(p.counters + s, 1) == c_hi - c_lo;
      named_bar_sync(1, NCW * 32);
      if (last_flag[0]) {
        __threadfence();
        const int n = c_hi - c_lo + 1;
        // per head: the partials' weights exp2(m - max m) and the inverse total (into red: free after the merge above)
        if (tid < HQ) {
          float mmax = -INFINITY;
          for (int k = 0; k < n; ++k) mmax = fmaxf(mmax, __ldcg(p.part_ml + ((int64_t)(c_lo + k + s) * HQ + tid) * 2));
          float tot = 0.f;
          for (int k = 0; k < n; ++k) {
            const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml + ((int64_t)(c_lo + k + s) * HQ + tid) * 2));
            const float w = exp2f((ml.x - mmax) * p.scale_log2);
            mw[tid * (MAX_CTAS_ALLOWED + 1) + k] = w;
            tot += w * ml.y;
          }
          mw[tid * (MAX_CTAS_ALLOWED + 1) + MAX_CTAS_ALLOWED] = 1.f / tot;
        }
        named_bar_sync(1, NCW * 32);
        for (int idx = tid; idx < HQ * HD / 4; idx += NCW * 32) {
          const int h = idx / (HD / 4), d4 = idx - h * (HD / 4);
          float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
          for (int k = 0; k < n; ++k) {
            const float w = mw[h * (MAX_CTAS_ALLOWED + 1) + k];
            const float4 v = __ldcg(reinterpret_cast<const float4*>(p.part_o + (int64_t)(c_lo + k + s) * HQ * HD) + idx);
            acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
          }
          const float inv = mw[h * (MAX_CTAS_ALLOWED + 1) + MAX_CTAS_ALLOWED];
          __nv_bfloat16* op = p.out + (int64_t)s * p.o_tok + (int64_t)h * p.o_head + d4 * 4;
          *reinterpret_cast<uint2*>(op) = make_uint2(pack_bf16(acc.x * inv, acc.y * inv), pack_bf16(acc.z * inv, acc.w * inv));
        }
        if (tid == 0) p.counters[s] = 0;  // for the next call
      }
    }
    // acc_s is rewritten by the next merge only after its first barrier, which every reader here passes first
  }
}

// ------------------------------------------------------------------------------------------------------------------
// host side

struct MapKey {
  uintptr_t ptr;
  int64_t nb, page, row_stride, block_stride;
  bool operator==(const MapKey& o) const {
    return ptr == o.ptr && nb == o.nb && page == o.page && row_stride == o.row_stride && block_stride == o.block_stride;
  }
};
struct MapKeyHash {
  size_t operator()(const MapKey& k) const {
    return std::hash<uintptr_t>()(k.ptr) ^ (std::hash<int64_t>()(k.nb) * 31) ^ (std::hash<int64_t>()(k.block_stride) * 17);
  }
};

// 128B-swizzled boxes from a 3D tensor map over [K|V columns, page rows, pages]: a tile never crosses a page, and
// rows past a page's end are zero-filled
static CUtensorMap make_map(const at::Tensor& kv, int box_rows) {
  // kv: [num_blocks, page, 1, 2 * HD] bf16 view (K | V per token), unit stride in the last dim
  static std::mutex mu;
  static std::unordered_map<MapKey, CUtensorMap, MapKeyHash> cache;
  MapKey key{reinterpret_cast<uintptr_t>(kv.data_ptr()), kv.size(0), kv.size(1) * 1000 + box_rows, kv.stride(1), kv.stride(0)};
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map;
  cuuint64_t dims[3] = {(cuuint64_t)(2 * HD), (cuuint64_t)kv.size(1), (cuuint64_t)kv.size(0)};
  cuuint64_t strides[2] = {(cuuint64_t)kv.stride(1) * 2, (cuuint64_t)kv.stride(0) * 2};  // bytes, dims 1 and 2
  cuuint32_t box[3] = {BOX_COLS, (cuuint32_t)box_rows, 1};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult res = encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, kv.data_ptr(), dims, strides, box, estr,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)res);
  cache.emplace(key, map);
  return map;
}

template <int HQ, int TILE_, int STAGES>
static size_t smem_bytes() {
  return 1024 + STAGES * NBOX * TILE_ * BOX_COLS * 2 + (4 * HQ * HD + MAX_NCW * 16 + 16) * 4 + (MAX_SEQS + 1 + MAX_NCW + 2) * 4 +
         HQ * (MAX_CTAS_ALLOWED + 1) * 4 +
         16 + 2 * STAGES * 8 + STAGES * 4;
}

// the variant the kernel runs: 3 stages of 64-token tiles, 8 consumer warps of one 16-row slice each, P as bf16 hi + lo
constexpr int STAGES = 3;
constexpr int NCW = 8;
constexpr int NS = 1;
constexpr int HQ = 6;      // query heads per KV head (Qwen3.5 at TP4)
constexpr int PAGE = 784;  // tokens per KV cache page (the replica's block size)

void decode_attention(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& block_table,
                      const at::Tensor& seq_lens, const at::Tensor& num_dec, at::Tensor& out, at::Tensor& part_o,
                      at::Tensor& part_ml, at::Tensor& counters, double scale, int64_t nctas, int64_t max_dec) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == HD && q.stride(2) == 1, "q [T, HQ, 256]");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == HD && out.stride(2) == 1, "out [T, HQ, 256]");
  TORCH_CHECK(kv.dim() == 4 && kv.size(2) == 1 && kv.size(3) == 2 * HD && kv.stride(3) == 1, "kv [blocks, page, 1, 512]");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && num_dec.scalar_type() == at::kInt);
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_ml.scalar_type() == at::kFloat);
  TORCH_CHECK(part_o.numel() >= (nctas + max_dec) * q.size(1) * HD && part_ml.numel() >= (nctas + max_dec) * q.size(1) * 2);
  TORCH_CHECK(max_dec <= MAX_SEQS && max_dec >= 1 && nctas >= 1 && nctas <= MAX_CTAS_ALLOWED);
  TORCH_CHECK(q.size(1) == HQ && kv.size(1) == PAGE, "HQ 6, page 784 only");
  const at::cuda::CUDAGuard guard(q.device());
  const CUtensorMap map = make_map(kv, TILE);
  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.q_tok = q.stride(0);
  p.q_head = q.stride(1);
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.o_tok = out.stride(0);
  p.o_head = out.stride(1);
  p.block_table = block_table.data_ptr<int>();
  p.bt_stride = block_table.stride(0);
  p.seq_lens = seq_lens.data_ptr<int>();
  p.num_dec = num_dec.data_ptr<int>();
  p.part_o = part_o.data_ptr<float>();
  p.part_ml = part_ml.data_ptr<float>();
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= MAX_SEQS, "counters [MAX_SEQS] int32, zeroed");
  p.counters = counters.data_ptr<int>();
  p.page = PAGE;
  p.tiles_per_page = (PAGE + TILE - 1) / TILE;
  p.scale_log2 = (float)(scale * 1.4426950408889634);

  auto kern = decode_attn_kernel<HQ, PAGE, TILE, STAGES, NCW, NS, true>;
  const size_t smem = smem_bytes<HQ, TILE, STAGES>();
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                "smem attribute");
    configured = true;
  }
  kern<<<(int)nctas, (NCW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(map, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace split_attn

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def("split_decode_attention(Tensor q, Tensor kv, Tensor block_table, Tensor seq_lens, Tensor num_dec, "
        "Tensor(a!) out, Tensor(b!) part_o, Tensor(c!) part_ml, Tensor(d!) counters, float scale, int nctas, "
        "int max_dec) -> ()");
}
TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("split_decode_attention", &split_attn::decode_attention); }
