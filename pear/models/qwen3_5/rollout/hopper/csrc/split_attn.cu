// Split-KV paged attention for sm_90 of a step's decode rows: T query tokens per sequence (one, or a verify step's,
// causal within the block), a GQA group of HQ query heads per KV head, head dim HD, over the KV cache in vLLM's
// FlashAttention layout (per page of the cache's block size: [token][kv head][K | V] bf16)
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

constexpr int HD = 256;       // head dim
constexpr int HQ = 6;         // query heads per KV head (Qwen3.5 at TP4)
constexpr int TILE = 64;      // tokens per tile
constexpr int BOX_COLS = 64;  // bf16 columns per TMA box (128 B: the 128B swizzle span)
constexpr int MAX_SEQS = 256;
constexpr int MAX_CTAS_ALLOWED = 160;
constexpr int MAX_WARPS = 8;  // warps of a CTA of the kernels, at most

// everything the kernel reads sits in device memory (the number of decode rows, their lengths and block tables): the
// launch is fixed, so it can live in a CUDA graph
struct Params {
  __nv_bfloat16* out;
  int64_t o_tok, o_head;  // strides (elements)
  const int* block_table;
  int64_t bt_stride;
  const int* seq_lens;
  const int* num_dec;  // device scalar: decode rows (the first num_dec sequences)
  float* part_o;       // [slots][rows][HD]
  float* part_ml;      // [slots][rows][2]
  int page;            // tokens per page: the KV cache's block size
  int tiles_per_page;  // page / TILE
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
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int TILE_>
__device__ __forceinline__ int tiles_of(int len, int page, int tiles_per_page) {
  const int full = len / page, rem = len - full * page;
  return full * tiles_per_page + (rem + TILE_ - 1) / TILE_;
}

// prefix sums of the decode rows' tile counts into s_start[0..D] (all NTHREADS threads of the block take part)
template <int TILE_, int NTHREADS>
__device__ __forceinline__ void tile_prefix(const Params& p, int D, int* s_start, int* s_warp) {
  const int tid = threadIdx.x;
  constexpr int PER = (MAX_SEQS + NTHREADS) / NTHREADS;  // (MAX_SEQS + 1 prefix entries)
  int v[PER];
  int tot = 0;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    int i = tid * PER + j;
    v[j] = i < D ? tiles_of<TILE_>(__ldg(p.seq_lens + i), p.page, p.tiles_per_page) : 0;
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

// ------------------------------------------------------------------------------------------------------------------
// verify rows: T query tokens per sequence, its N = HQ T query rows (row = HQ t + head) on the N side of the wgmma: per
// 64-token tile S^T = K Q^T, then O^T += V^T P^T with P^T handed over in smem; one consumer warpgroup per CTA over the
// decode kernel's persistent split, a producer warp loading each piece's query rows and its tiles' K and V

constexpr int BOX_BYTES = TILE * BOX_COLS * 2;           // a box of a tile: 64 tokens x 128 bytes
constexpr int SLOT_BYTES = (HD / BOX_COLS) * BOX_BYTES;  // a ring slot: a tile's K or its V
constexpr int RING = 5;                                  // ring slots
constexpr int CONSUMER_THREADS = 128;                    // the consumer warpgroup
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int VERIFY_THREADS = CONSUMER_THREADS + 32;  // and the producer warp
constexpr int CONSUMER_BARRIER = 1;                    // the consumer warpgroup's named barrier
constexpr int ROW_BYTES = 128;                         // a swizzled operand row: 64 bf16
constexpr int GROUP_BYTES = 8 * ROW_BYTES;             // the 128-byte swizzle's period: an operand's 8-row groups
constexpr int K_STEP = 16;                             // a wgmma's K: dims of S^T's, tokens of O^T's
constexpr float RAISE_LOG2 = 16.f;  // a column's running max is raised once a tile's max exceeds it by more (base 2)
constexpr int MERGE_THREADS = 256;  // the merge kernel's CTA
constexpr int MERGE_CTAS_PER_SM = 2;
constexpr int MERGE_ITEMS = 2;  // items (a column's 4 dims) a merge thread loads at once
constexpr int MERGE_CHUNK = 4;  // pieces an item loads at once

// a wgmma operand in smem, 128-byte swizzled, rows of 128 bytes in 8-row groups: K-major (rows along M or N, the 16
// elements of K at the start address) or, transposed, MN-major (64 elements of M per row, rows along K)
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  constexpr uint64_t LBO = 1;                 // bits 16-29 (16-byte units): unused by these layouts
  constexpr uint64_t SBO = GROUP_BYTES >> 4;  // bits 32-45 (16-byte units): the 8-row groups' stride
  constexpr uint64_t SWIZZLE_128B = 1;        // bits 62-63
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (LBO << 16) | (SBO << 32) | (SWIZZLE_128B << 62);
}

// D (64 x N f32, over the warpgroup) [+]= A (64 x 16 bf16 in smem: K-major, or MN-major with TRANS_A) x B (16 x N bf16
// in smem, K-major)
template <int N>
struct Wgmma;

template <>
struct Wgmma<8> {
  template <int TRANS_A>
  __device__ __forceinline__ static void mma(float (&d)[4], uint64_t a, uint64_t b, uint32_t accumulate) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, %7, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a), "l"(b), "r"(accumulate), "n"(TRANS_A));
  }
};

template <>
struct Wgmma<24> {
  template <int TRANS_A>
  __device__ __forceinline__ static void mma(float (&d)[12], uint64_t a, uint64_t b, uint32_t accumulate) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %14, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},"
        " %12, %13, p, 1, 1, %15, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
          "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
        : "l"(a), "l"(b), "r"(accumulate), "n"(TRANS_A));
  }
};

template <>
struct Wgmma<48> {
  template <int TRANS_A>
  __device__ __forceinline__ static void mma(float (&d)[24], uint64_t a, uint64_t b, uint32_t accumulate) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %26, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12,"
        " %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, %24, %25, p, 1, 1, %27, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
          "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
          "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
        : "l"(a), "l"(b), "r"(accumulate), "n"(TRANS_A));
  }
};

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait_all() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }

// pins an accumulator's registers in place across a wgmma's issue and its wait (the compiler sees them change here)
template <int K>
__device__ __forceinline__ void fence_registers(float (&r)[K]) {
#pragma unroll
  for (int i = 0; i < K; ++i) asm volatile("" : "+f"(r[i])::"memory");
}

__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void stsm_x4_t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3)
               : "memory");
}
__device__ __forceinline__ void stsm_x2_t(uint32_t addr, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};" ::"r"(addr), "r"(r0), "r"(r1) : "memory");
}

// a named barrier over count threads that returns whether any of them passed pred
__device__ __forceinline__ bool bar_red_or(int id, int count, bool pred) {
  uint32_t any;
  asm volatile(
      "{\n .reg .pred p, q;\n setp.ne.u32 p, %1, 0;\n bar.red.or.pred q, %2, %3, p;\n selp.u32 %0, 1, 0, q;\n}\n"
      : "=r"(any)
      : "r"((uint32_t)pred), "r"(id), "r"(count)
      : "memory");
  return any != 0;
}

template <int N>
struct alignas(1024) VerifySmem {
  uint8_t ring[RING][SLOT_BYTES];
  uint8_t q[2][HD / BOX_COLS][N * ROW_BYTES];  // two pieces' query rows: per 64 dims N rows of 128 bytes
  uint8_t p_hi[N * ROW_BYTES];                 // P^T as bf16: N rows of the tile's 64 tokens
  uint8_t p_lo[N * ROW_BYTES];                 // P^T - bf16(P^T)
  float red[CONSUMER_WARPS][N];                // the warps' column maxima (a raise) or sums (a piece's end)
  int start[MAX_SEQS + 1];
  int warp_sums[MAX_WARPS + 1];
  uint64_t full[RING], empty[RING], q_full[2], q_empty[2];
};

// P = 2^(s scale_log2 - m) per column (0 for a masked score; against 0 while a column has no key), its per-column sums
// into lsum, and P^T as bf16 and its residual P - bf16(P) stored transposed into the P^T buffers: the stmatrix of
// column group f and token half h writes P^T rows 8 f + r, the warp's tokens 16 warp + 8 h..
template <int NF>
__device__ __forceinline__ void store_p(const float (&sacc)[NF * 4], const float (&m_run)[NF][2], float scale_log2,
                                        float (&lsum)[NF][2], uint32_t p_hi, uint32_t p_lo, int warp, int lane) {
  uint32_t hi[NF][2], lo[NF][2];  // the pairs (token g + 8 h, columns 8 f + 2 c4 (+1))
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const float m0 = m_run[f][0] == -INFINITY ? 0.f : m_run[f][0];
    const float m1 = m_run[f][1] == -INFINITY ? 0.f : m_run[f][1];
    lsum[f][0] = lsum[f][1] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float p0 = ex2_approx(__fmaf_rn(sacc[4 * f + 2 * h], scale_log2, -m0));
      const float p1 = ex2_approx(__fmaf_rn(sacc[4 * f + 2 * h + 1], scale_log2, -m1));
      lsum[f][0] += p0;
      lsum[f][1] += p1;
      const __nv_bfloat162 x = __floats2bfloat162_rn(p0, p1);
      hi[f][h] = *reinterpret_cast<const uint32_t*>(&x);
      lo[f][h] = pack_bf16(p0 - __low2float(x), p1 - __high2float(x));
    }
  }
  const int r = lane & 7, i = lane >> 3;  // lane 8 i + r: the address of row r of the instruction's matrix i
#pragma unroll
  for (int q = 0; q < NF / 2; ++q) {
    const int f = 2 * q + (i >> 1), h = i & 1;
    const uint32_t off = (8 * f + r) * ROW_BYTES + (((2 * warp + h) ^ r) << 4);
    stsm_x4_t(p_hi + off, hi[2 * q][0], hi[2 * q][1], hi[2 * q + 1][0], hi[2 * q + 1][1]);
    stsm_x4_t(p_lo + off, lo[2 * q][0], lo[2 * q][1], lo[2 * q + 1][0], lo[2 * q + 1][1]);
  }
  if constexpr (NF % 2 == 1) {
    const int f = NF - 1, h = i & 1;
    const uint32_t off = (8 * f + r) * ROW_BYTES + (((2 * warp + h) ^ r) << 4);
    stsm_x2_t(p_hi + off, hi[f][0], hi[f][1]);
    stsm_x2_t(p_lo + off, lo[f][0], lo[f][1]);
  }
  fence_proxy_async();  // the stores, for the wgmma that reads them
}

// a tile's maxima per column over the warp's 16 tokens (base 2), from its masked scores
template <int NF>
__device__ __forceinline__ void column_maxima(const float (&sacc)[NF * 4], float scale_log2, float (&mx)[NF][2]) {
#pragma unroll
  for (int f = 0; f < NF; ++f)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float m = fmaxf(sacc[4 * f + e], sacc[4 * f + 2 + e]);
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
      mx[f][e] = m * scale_log2;
    }
}

template <int T_>
__global__ void __launch_bounds__(VERIFY_THREADS, 1)
    verify_attn_kernel(const __grid_constant__ CUtensorMap kv_map, const __grid_constant__ CUtensorMap q_map, Params p) {
  constexpr int ROWS = HQ * T_;          // query rows per sequence
  constexpr int N = (ROWS + 7) / 8 * 8;  // the wgmma's N: the rows and zero rows up to a multiple of 8
  constexpr int NF = N / 8;              // its groups of 8 columns
  extern __shared__ uint8_t smem_raw[];
  VerifySmem<N>& sm =
      *reinterpret_cast<VerifySmem<N>*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if constexpr (N > ROWS) {
    // the query buffers' rows past the sequence's (never loaded): zero, so the padding columns' scores are finite
    for (int i = tid; i < 2 * (HD / BOX_COLS) * (N - ROWS) * (ROW_BYTES / 16); i += VERIFY_THREADS) {
      const int chunk = i % (ROW_BYTES / 16), row = ROWS + (i / (ROW_BYTES / 16)) % (N - ROWS);
      const int box = i / ((ROW_BYTES / 16) * (N - ROWS));  // the 8 boxes of the two buffers
      *reinterpret_cast<uint4*>(&sm.q[box / (HD / BOX_COLS)][box % (HD / BOX_COLS)][row * ROW_BYTES + chunk * 16]) =
          make_uint4(0, 0, 0, 0);
    }
    fence_proxy_async();  // (the tile prefix's barriers order them before the first wgmma)
  }
  const int page = p.page, tpp = p.tiles_per_page;
  const int D = min(__ldg(p.num_dec), MAX_SEQS);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // the merge kernel's CTAs may start as CTAs finish
  if (tid == 0) {
    for (int k = 0; k < RING; ++k) {
      mbar_init(&sm.full[k], 1);
      mbar_init(&sm.empty[k], CONSUMER_WARPS);
    }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&sm.q_full[b], 1);
      mbar_init(&sm.q_empty[b], CONSUMER_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tile_prefix<TILE, VERIFY_THREADS>(p, D, sm.start, sm.warp_sums);  // includes __syncthreads
  if (D == 0) return;
  const int total = sm.start[D];
  const int G = min((int)gridDim.x, total), c = blockIdx.x;  // every working CTA gets at least one tile
  if (c >= G) return;
  const int t0 = (int)((int64_t)c * total / G), t1 = (int)((int64_t)(c + 1) * total / G);

  if (warp == CONSUMER_WARPS) {
    // the producer: per piece (a sequence's tiles in this CTA) its query rows into the piece's buffer, then per tile its
    // K and its V into the ring (a tile's page looked up before its slots free)
    if (lane == 0) {
      int u = 0, piece = 0;
      for (int s = seq_of_tile(sm.start, D, t0); s < D && sm.start[s] < t1; ++s) {
        const int a = max(t0, sm.start[s]), b = min(t1, sm.start[s + 1]);
        if (a >= b) continue;
        const int qb = piece & 1;
        if (piece >= 2) mbar_wait(&sm.q_empty[qb], ((piece >> 1) - 1) & 1);
        mbar_expect_tx(&sm.q_full[qb], ROWS * HD * 2);
#pragma unroll
        for (int box = 0; box < HD / BOX_COLS; ++box)
          tma_load_3d(sm.q[qb][box], &q_map, &sm.q_full[qb], box * BOX_COLS, 0, s * T_);
        const int* table = p.block_table + (int64_t)s * p.bt_stride;
        for (int t = a; t < b; ++t) {
          const int lt = t - sm.start[s], pg = lt / tpp;
          const int row = (lt - pg * tpp) * TILE, block = __ldg(table + pg);
#pragma unroll
          for (int half = 0; half < 2; ++half, ++u) {  // the tile's K, then its V
            const int slot = u % RING;
            if (u >= RING) mbar_wait(&sm.empty[slot], ((u / RING) - 1) & 1);
            mbar_expect_tx(&sm.full[slot], SLOT_BYTES);
#pragma unroll
            for (int box = 0; box < HD / BOX_COLS; ++box)
              tma_load_3d(sm.ring[slot] + box * BOX_BYTES, &kv_map, &sm.full[slot], half * HD + box * BOX_COLS, row,
                          block);
          }
        }
        ++piece;
      }
    }
    return;
  }

  // the consumer warpgroup: S^T in registers (tokens 16 warp + g (+8) of the tile x columns 8 f + 2 c4 (+1)), the online
  // softmax per column against a running max raised only past RAISE_LOG2 (the vote rides on the barrier that hands P^T
  // to the wgmma), O^T in registers (dims 64 mb + 16 warp + g (+8) x the same columns)
  const int g = lane >> 2, c4 = lane & 3;
  const uint32_t ring = smem_u32(sm.ring), p_hi = smem_u32(sm.p_hi), p_lo = smem_u32(sm.p_lo);
  float o[HD / 64][N / 2];
  float sacc[N / 2];
  float m_run[NF][2], l_run[NF][2], lsum[NF][2], mx[NF][2];
  int u = 0, piece = 0;
  for (int s = seq_of_tile(sm.start, D, t0); s < D && sm.start[s] < t1; ++s) {
    const int a = max(t0, sm.start[s]), b = min(t1, sm.start[s + 1]);
    if (a >= b) continue;
    const int len = __ldg(p.seq_lens + s), last_key = len - T_;  // query token t sees the keys up to last_key + t
    const int qb = piece & 1;
    const uint32_t qbase = smem_u32(sm.q[qb]);
#pragma unroll
    for (int mb = 0; mb < HD / 64; ++mb)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) o[mb][i] = 0.f;
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int e = 0; e < 2; ++e) m_run[f][e] = -INFINITY, l_run[f][e] = 0.f;
    mbar_wait(&sm.q_full[qb], (piece >> 1) & 1);

    for (int t = a; t < b; ++t, u += 2) {
      const int lt = t - sm.start[s], pg = lt / tpp;
      const int key0 = pg * page + (lt - pg * tpp) * TILE;  // the tile's first key
      const int ks = u % RING, vs = (u + 1) % RING;
      const uint32_t kbase = ring + ks * SLOT_BYTES, vbase = ring + vs * SLOT_BYTES;

      // S^T = K Q^T in 16 steps of 16 dims: box kk / 4 of K and of Q, 32 bytes per step into their rows
      mbar_wait(&sm.full[ks], (u / RING) & 1);
#pragma unroll
      for (int i = 0; i < N / 2; ++i) sacc[i] = 0.f;
      fence_registers(sacc);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < HD / K_STEP; ++kk) {
        const uint32_t off = (kk & 3) * 32;
        Wgmma<N>::template mma<0>(sacc, sw128_desc(kbase + (kk >> 2) * BOX_BYTES + off),
                                  sw128_desc(qbase + (kk >> 2) * N * ROW_BYTES + off), kk > 0);
      }
      wgmma_commit();
      wgmma_wait_all();  // and the previous tile's O^T
      fence_registers(sacc);
#pragma unroll
      for (int mb = 0; mb < HD / 64; ++mb) fence_registers(o[mb]);
      if (lane == 0) {
        if (t > a) mbar_arrive(&sm.empty[(u - 1) % RING]);  // the previous tile's V
        mbar_arrive(&sm.empty[ks]);
        if (t == b - 1) mbar_arrive(&sm.q_empty[qb]);
      }

      // the scores past each query token's position masked (in a tile reaching past the first one's); P with the
      // running max, a raise voted by any column whose tile max exceeds it by more than RAISE_LOG2
      if (key0 + TILE - 1 > last_key) {
        const int key = key0 + 16 * warp + g;  // this thread's first token
#pragma unroll
        for (int f = 0; f < NF; ++f)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int lim = last_key + (8 * f + 2 * c4 + e) / HQ - key;  // its column's last key, from the token
            if (lim < 0) sacc[4 * f + e] = -INFINITY;
            if (lim < 8) sacc[4 * f + 2 + e] = -INFINITY;
          }
      }
      if constexpr (N > ROWS) {
#pragma unroll
        for (int f = 0; f < NF; ++f)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (8 * f + 2 * c4 + e >= ROWS) sacc[4 * f + e] = sacc[4 * f + 2 + e] = -INFINITY;  // padding columns
      }
      column_maxima<NF>(sacc, p.scale_log2, mx);
      bool raises = false;
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int e = 0; e < 2; ++e) raises |= mx[f][e] > m_run[f][e] + RAISE_LOG2;
      store_p<NF>(sacc, m_run, p.scale_log2, lsum, p_hi, p_lo, warp, lane);
      if (bar_red_or(CONSUMER_BARRIER, CONSUMER_THREADS, raises)) {
        // a raise: the tile's maxima over the warpgroup raise the running max (O^T and l rescaled), P^T rewritten
        if (g == 0) {
#pragma unroll
          for (int f = 0; f < NF; ++f)
#pragma unroll
            for (int e = 0; e < 2; ++e) sm.red[warp][8 * f + 2 * c4 + e] = mx[f][e];
        }
        named_bar_sync(CONSUMER_BARRIER, CONSUMER_THREADS);
#pragma unroll
        for (int f = 0; f < NF; ++f)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int col = 8 * f + 2 * c4 + e;
            const float m_new = fmaxf(fmaxf(m_run[f][e], fmaxf(sm.red[0][col], sm.red[1][col])),
                                      fmaxf(sm.red[2][col], sm.red[3][col]));
            const float alpha = m_run[f][e] == -INFINITY ? 0.f : ex2_approx(m_run[f][e] - m_new);
            m_run[f][e] = m_new;
            l_run[f][e] *= alpha;
#pragma unroll
            for (int mb = 0; mb < HD / 64; ++mb) {
              o[mb][4 * f + e] *= alpha;
              o[mb][4 * f + 2 + e] *= alpha;
            }
          }
        store_p<NF>(sacc, m_run, p.scale_log2, lsum, p_hi, p_lo, warp, lane);
        named_bar_sync(CONSUMER_BARRIER, CONSUMER_THREADS);
      }
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int e = 0; e < 2; ++e) l_run[f][e] += lsum[f][e];

      // V rows past the sequence's end hold whatever the page held: zeroed (P is 0 there, 0 * NaN is not)
      const int valid = len - key0;
      mbar_wait(&sm.full[vs], ((u + 1) / RING) & 1);
      if (valid < TILE) {
        for (int idx = tid; idx < (TILE - valid) * 32; idx += CONSUMER_THREADS) {
          const int row = valid + idx / 32, ch = idx % 32;  // 32 chunks of 16 bytes per row: 4 boxes x 8
          *reinterpret_cast<uint4*>(sm.ring[vs] + (ch / 8) * BOX_BYTES + row * ROW_BYTES + (ch % 8) * 16) =
              make_uint4(0, 0, 0, 0);
        }
        fence_proxy_async();
        named_bar_sync(CONSUMER_BARRIER, CONSUMER_THREADS);
      }

      // O^T += V^T P^T: per 64 dims mb (V's box mb, MN-major) 4 steps of 16 tokens, P^T's bf16 and its residual
#pragma unroll
      for (int mb = 0; mb < HD / 64; ++mb) fence_registers(o[mb]);
      wgmma_fence();
#pragma unroll
      for (int mb = 0; mb < HD / 64; ++mb)
#pragma unroll
        for (int kk = 0; kk < TILE / K_STEP; ++kk) {
          const uint64_t v_desc = sw128_desc(vbase + mb * BOX_BYTES + kk * K_STEP * ROW_BYTES);
          Wgmma<N>::template mma<1>(o[mb], v_desc, sw128_desc(p_hi + kk * 32), 1);
          Wgmma<N>::template mma<1>(o[mb], v_desc, sw128_desc(p_lo + kk * 32), 1);
        }
      wgmma_commit();
    }
    wgmma_wait_all();
#pragma unroll
    for (int mb = 0; mb < HD / 64; ++mb) fence_registers(o[mb]);
    if (lane == 0) mbar_arrive(&sm.empty[(u - 1) % RING]);  // the last tile's V

    // each column's sum over the warpgroup's tokens
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float l = l_run[f][e];
        l += __shfl_xor_sync(0xffffffffu, l, 4);
        l += __shfl_xor_sync(0xffffffffu, l, 8);
        l += __shfl_xor_sync(0xffffffffu, l, 16);
        if (g == 0) sm.red[warp][8 * f + 2 * c4 + e] = l;
      }
    named_bar_sync(CONSUMER_BARRIER, CONSUMER_THREADS);
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = 8 * f + 2 * c4 + e;
        l_run[f][e] = (sm.red[0][col] + sm.red[1][col]) + (sm.red[2][col] + sm.red[3][col]);
      }

    const bool is_whole = sm.start[s] >= t0 && sm.start[s + 1] <= t1;
    if (is_whole) {
      // the normalized output: column HQ t + head is query token t's head, dims 64 mb + 16 warp + g (+8)
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = 8 * f + 2 * c4 + e, qt = col / HQ;
          if (col >= ROWS) continue;
          const float inv = 1.f / l_run[f][e];
          __nv_bfloat16* dst =
              p.out + (int64_t)(s * T_ + qt) * p.o_tok + (int64_t)(col - qt * HQ) * p.o_head + 16 * warp + g;
#pragma unroll
          for (int mb = 0; mb < HD / 64; ++mb)
#pragma unroll
            for (int h = 0; h < 2; ++h) dst[64 * mb + 8 * h] = __float2bfloat16(o[mb][4 * f + 2 * h + e] * inv);
        }
    } else {
      // an fp32 partial (unnormalized O, the running max, the sum) to slot c + s, for the merge kernel
      const int64_t slot = c + s;
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = 8 * f + 2 * c4 + e;
          if (col >= ROWS) continue;
          float* dst = p.part_o + (slot * N + col) * HD + 16 * warp + g;
#pragma unroll
          for (int mb = 0; mb < HD / 64; ++mb)
#pragma unroll
            for (int h = 0; h < 2; ++h) dst[64 * mb + 8 * h] = o[mb][4 * f + 2 * h + e];
          if (warp == 0 && g == 0)
            *reinterpret_cast<float2*>(p.part_ml + (slot * N + col) * 2) = make_float2(m_run[f][e], l_run[f][e]);
        }
    }
    ++piece;
  }
}

// the merge of the sequences split over CTAs (pieces in CTAs c_lo..c_hi, slots c + s), the attention kernel's
// programmatic dependent: per item (a sequence's column and 4 of its dims) MERGE_CHUNK pieces' (m, l) and O loaded at
// once and merged against their running max, MERGE_ITEMS items per thread at once
template <int T_>
__global__ void __launch_bounds__(MERGE_THREADS) verify_merge_kernel(Params p, int grid_main) {
  constexpr int ROWS = HQ * T_, N = (ROWS + 7) / 8 * 8;  // a partial's rows, its row stride
  constexpr int ITEMS_PER_COLUMN = HD / 4;
  __shared__ int start[MAX_SEQS + 1];
  __shared__ int warp_sums[MAX_WARPS + 1];
  const int D = min(__ldg(p.num_dec), MAX_SEQS);
  tile_prefix<TILE, MERGE_THREADS>(p, D, start, warp_sums);
  const int total = D > 0 ? start[D] : 0;
  const int G = min(grid_main, total);
  asm volatile("griddepcontrol.wait;" ::: "memory");  // the attention kernel complete, its partials visible
  if (G == 0) return;

  const int stride = gridDim.x * MERGE_THREADS, items = D * ROWS * ITEMS_PER_COLUMN;
  for (int item = blockIdx.x * MERGE_THREADS + threadIdx.x; item < items; item += MERGE_ITEMS * stride) {
    int s[MERGE_ITEMS], col[MERGE_ITEMS], d4[MERGE_ITEMS], c_lo[MERGE_ITEMS], k[MERGE_ITEMS];
    float m_max[MERGE_ITEMS], l_sum[MERGE_ITEMS];
    float4 acc[MERGE_ITEMS];
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x) {
      const int it = item + x * stride;
      s[x] = it / (ROWS * ITEMS_PER_COLUMN);
      col[x] = (it / ITEMS_PER_COLUMN) % ROWS;
      d4[x] = it % ITEMS_PER_COLUMN;
      const bool has_tiles = it < items && start[s[x]] < start[s[x] + 1];
      c_lo[x] = has_tiles ? (int)((((int64_t)start[s[x]] + 1) * G - 1) / total) : 0;
      k[x] = has_tiles ? (int)((((int64_t)start[s[x] + 1]) * G - 1) / total) - c_lo[x] + 1 : 0;  // 1: a whole piece
      m_max[x] = -INFINITY, l_sum[x] = 0.f, acc[x] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // the items' chunks of pieces: loads first, then the merge against the chunk's max
    int k_max = 0;
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x) k_max = max(k_max, k[x]);
    for (int base = 0; base < k_max; base += MERGE_CHUNK) {
      float2 ml[MERGE_ITEMS][MERGE_CHUNK];
      float4 v[MERGE_ITEMS][MERGE_CHUNK];
#pragma unroll
      for (int x = 0; x < MERGE_ITEMS; ++x)
#pragma unroll
        for (int j = 0; j < MERGE_CHUNK; ++j) {
          const bool has_piece = k[x] > 1 && base + j < k[x];
          const int64_t at = ((int64_t)(c_lo[x] + base + j + s[x]) * N + col[x]);
          ml[x][j] = has_piece ? __ldcg(reinterpret_cast<const float2*>(p.part_ml + at * 2)) : make_float2(-INFINITY, 0.f);
          v[x][j] = has_piece ? __ldcg(reinterpret_cast<const float4*>(p.part_o + at * HD) + d4[x])
                              : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
      for (int x = 0; x < MERGE_ITEMS; ++x) {
        float m_new = m_max[x];
#pragma unroll
        for (int j = 0; j < MERGE_CHUNK; ++j) m_new = fmaxf(m_new, ml[x][j].x);
        if (m_new == -INFINITY) continue;  // (no key in these pieces yet)
        const float a = exp2f(m_max[x] - m_new);
        l_sum[x] *= a;
        acc[x].x *= a, acc[x].y *= a, acc[x].z *= a, acc[x].w *= a;
#pragma unroll
        for (int j = 0; j < MERGE_CHUNK; ++j) {
          const float w = ml[x][j].x == -INFINITY ? 0.f : exp2f(ml[x][j].x - m_new);
          l_sum[x] += w * ml[x][j].y;
          acc[x].x += w * v[x][j].x, acc[x].y += w * v[x][j].y, acc[x].z += w * v[x][j].z, acc[x].w += w * v[x][j].w;
        }
        m_max[x] = m_new;
      }
    }
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x) {
      if (k[x] <= 1) continue;
      const float inv = 1.f / l_sum[x];
      const int qt = col[x] / HQ;
      __nv_bfloat16* dst =
          p.out + (int64_t)(s[x] * T_ + qt) * p.o_tok + (int64_t)(col[x] - qt * HQ) * p.o_head + 4 * d4[x];
      *reinterpret_cast<uint2*>(dst) =
          make_uint2(pack_bf16(acc[x].x * inv, acc[x].y * inv), pack_bf16(acc[x].z * inv, acc[x].w * inv));
    }
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

// a step's query rows as a 3D map {64 dims, HQ heads, T tokens} per box, 128-byte swizzled: a sequence's box per 64
// dims lands as its N = HQ T rows of 128 bytes (row = HQ t + head), the K-major operand the wgmma reads
static CUtensorMap make_query_map(const at::Tensor& q, int query_len) {
  CUtensorMap map;
  cuuint64_t dims[3] = {(cuuint64_t)HD, (cuuint64_t)HQ, (cuuint64_t)q.size(0)};
  cuuint64_t strides[2] = {(cuuint64_t)q.stride(1) * 2, (cuuint64_t)q.stride(0) * 2};  // bytes, dims 1 and 2
  cuuint32_t box[3] = {BOX_COLS, HQ, (cuuint32_t)query_len};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult res = encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, q.data_ptr(), dims, strides, box, estr,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)res);
  return map;
}

template <int T_>
static void verify_attention(const at::Tensor& q, const CUtensorMap& kv_map, const Params& p, int64_t nctas,
                             bool is_alone) {
  const CUtensorMap q_map = make_query_map(q, T_);
  const size_t smem = sizeof(VerifySmem<(HQ * T_ + 7) / 8 * 8>) + 1024;  // and the 1024-byte alignment of the buffers
  static bool is_configured = false;
  if (!is_configured) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(verify_attn_kernel<T_>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    is_configured = true;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  verify_attn_kernel<T_><<<(int)nctas, VERIFY_THREADS, smem, stream>>>(kv_map, q_map, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // the merge; alone on the GPU, as the attention kernel's programmatic dependent (its CTAs start as SMs free up),
  // else after it (beside another kernel its waiting CTAs would hold that kernel's SMs)
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3((unsigned)(MERGE_CTAS_PER_SM * nctas));
  config.blockDim = dim3(MERGE_THREADS);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = is_alone ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, verify_merge_kernel<T_>, p, (int)nctas));
}

// q, out: [tokens, HQ, HD] bf16 (sequence s's T query tokens from row s T); kv: [blocks, page, 1, K | V] bf16
void decode_attention(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& block_table,
                      const at::Tensor& seq_lens, const at::Tensor& num_dec, at::Tensor& out, at::Tensor& part_o,
                      at::Tensor& part_ml, double scale, int64_t nctas, int64_t max_dec, int64_t query_len,
                      bool is_alone) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == HQ && q.size(2) == HD && q.stride(2) == 1, "q [tokens, 6, 256]");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "q rows 16-byte aligned (TMA)");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == HD && out.stride(2) == 1, "out [tokens, 6, 256]");
  TORCH_CHECK(kv.dim() == 4 && kv.size(2) == 1 && kv.size(3) == 2 * HD && kv.stride(3) == 1, "kv [blocks, page, 1, 512]");
  TORCH_CHECK(kv.size(1) % TILE == 0, "pages of whole 64-token tiles");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && num_dec.scalar_type() == at::kInt);
  TORCH_CHECK(query_len == 1 || query_len == 4 || query_len == 8, "query_len 1, 4 or 8");
  TORCH_CHECK(max_dec <= MAX_SEQS && max_dec >= 1 && nctas >= 1 && nctas <= MAX_CTAS_ALLOWED);
  const int64_t rows = (HQ * query_len + 7) / 8 * 8;  // a partial's rows: the query rows, to the MMA's n8
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_ml.scalar_type() == at::kFloat);
  TORCH_CHECK(part_o.numel() >= (nctas + max_dec) * rows * HD && part_ml.numel() >= (nctas + max_dec) * rows * 2,
              "partials [slots c + s][rows][HD]");
  const at::cuda::CUDAGuard guard(q.device());
  const CUtensorMap map = make_map(kv, TILE);
  Params p;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.o_tok = out.stride(0);
  p.o_head = out.stride(1);
  p.block_table = block_table.data_ptr<int>();
  p.bt_stride = block_table.stride(0);
  p.seq_lens = seq_lens.data_ptr<int>();
  p.num_dec = num_dec.data_ptr<int>();
  p.part_o = part_o.data_ptr<float>();
  p.part_ml = part_ml.data_ptr<float>();
  p.page = (int)kv.size(1);
  p.tiles_per_page = p.page / TILE;
  p.scale_log2 = (float)(scale * 1.4426950408889634);

  if (query_len == 1) {
    verify_attention<1>(q, map, p, nctas, is_alone);
  } else if (query_len == 4) {
    verify_attention<4>(q, map, p, nctas, is_alone);
  } else {
    verify_attention<8>(q, map, p, nctas, is_alone);
  }
}

}  // namespace split_attn

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def("split_decode_attention(Tensor q, Tensor kv, Tensor block_table, Tensor seq_lens, Tensor num_dec, "
        "Tensor(a!) out, Tensor(b!) part_o, Tensor(c!) part_ml, float scale, int nctas, int max_dec, int query_len, "
        "bool is_alone) -> ()");
}
TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("split_decode_attention", &split_attn::decode_attention); }
