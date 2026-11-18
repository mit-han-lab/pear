// bf16 GEMMs x @ w^T for the few hundred rows of x of a decode or verify step, each weight byte streamed from HBM once:
// a CTA takes 128 rows of w (the MMA's A operand) over a K range and all of x's rows (its B operand), four warps load
// through TMA, two issue the tcgen05 MMAs (alternate K blocks); a CTA pair shares x past kPairFromRows rows. Launched
// as a programmatic dependent (the weights load before the wait), into SwiGLU of [gate; up], fp32 split-K sums, or
// the lm_head's logits of [high rows; low rows] summed into both tensor-parallel ranks' gather buffers (NVLink stores).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "sync.cuh"
#include "tcgen05.cuh"

namespace {

// warps 0-3: the loads (one thread each), then the epilogue (warp w: TMEM lanes 32 w)
constexpr int kProducers = 4;
constexpr int kMmaWarps = 2;           // warps 4-5: the MMAs of even and odd K blocks (an odd ring: warp 4 alone)
constexpr int kThreads = 32 * (kProducers + kMmaWarps);
constexpr int kEpilogueThreads = 32 * kProducers;
constexpr int kRows = 128;             // rows of w per CTA (the MMA's M)
constexpr int kKBlock = 64;            // K elements per stage: one 128-byte swizzle row
constexpr int kRowBytes = kKBlock * 2;
constexpr int kKStep = 16;             // K per MMA instruction (bf16)
constexpr int kMaxTokens = 256;        // rows of x at most (the MMA's N)
constexpr int kAlign = 1024;           // alignment of the stages (the 128-byte swizzle's 8-row atoms)
constexpr int kMaxStages = 16;
constexpr int kEpilogueBarrier = 1;    // named barrier of the epilogue warps
constexpr int kPairFromRows = 65;      // rows of x from which a CTA pair shares them: more stages fit
constexpr int kLoadsPerWait = 2;       // kSwiGLU's epilogue: 16-column TMEM loads per accumulator before a wait
constexpr int kHalfChunk = 8;          // kSwiGLU's epilogue: tokens of a 16-column chunk each lane of a pair computes

enum Epilogue : int {
  kSwiGLU = 0,    // out[t, j] = silu(x w_gate[j]^T) * x w_up[j]^T in bf16 (w = [gate; up], a tile's rows interleaved)
  kPartials = 1,  // partials[split, t, n] = this split's fp32 sum (its K range); the consumer adds the splits
  kLogits = 2,    // logits[t, col0 + n] = (x w^T)[t, n] + (x w^T)[rows + t, n] in fp32, into both ranks' buffers
};

// ---- device helpers ----
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void named_sync(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ void sts_u16(uint32_t a, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
}
__device__ __forceinline__ void sts_f32(uint32_t a, float v) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ uint16_t bf16_bits(float v) {
  __nv_bfloat16 h = __float2bfloat16_rn(v);
  return *reinterpret_cast<uint16_t*>(&h);
}

struct Params {
  int n_tok;       // the MMA's N: a CTA's rows of x rounded up to 16
  int k_stages;    // K / kKBlock
  int splits;      // K ranges per tile
  int stages;      // ring stages
  int producers;   // warps issuing loads: a divisor of stages
  int out_cols;    // columns of out per tile: kRows (kPartials, kLogits) or kRows / 2 (kSwiGLU)
  // kLogits: the logit rows (x's rows are [high rows; low rows]), this rank's first vocabulary column, the row stride,
  // both ranks' [rows, ld] buffers
  int rows;
  int col0;
  int ld;
  float* local;
  float* peer;
};

// after the lm_head's grid (its memory operations performed, its stores into the peer's buffer included): this rank's
// launch count to the peer, then the wait for the peer's
__global__ void logits_signal_kernel(uint32_t* launches, uint32_t* done_local, uint32_t* done_peer) {
  pdl_wait();
  const uint32_t e = *launches + 1;
  *launches = e;
  st_release_sys(done_peer, e);
  while (ld_acquire_sys(done_local) < e) {
  }
}

// stage k of a tile's w: 128 rows (kSwiGLU: gate row j and up row j at rows 2j and 2j + 1)
template <int EPI, bool PAIR>
__device__ __forceinline__ void load_w(const CUtensorMap& w_map, uint32_t dst, uint64_t* bar, int tile, int k,
                                       uint64_t policy) {
  if (EPI == kSwiGLU)
    tma_load_4d<PAIR>(dst, &w_map, 0, 0, tile * (kRows / 2), k, bar, policy);
  else
    tma_load_3d<PAIR>(dst, &w_map, 0, tile * kRows, k, bar, policy);
}

template <int EPI, bool PAIR>
__global__ void __launch_bounds__(kThreads, 1)
    stream_gemm_kernel(const __grid_constant__ CUtensorMap w_map, const __grid_constant__ CUtensorMap x_map,
                       const __grid_constant__ CUtensorMap out_map, Params p) {
  extern __shared__ uint8_t smem[];
  __shared__ uint64_t full_barriers[kMaxStages], empty_barriers[kMaxStages], tmem_full_barrier[1];
  __shared__ uint32_t tmem_slot[1];
  const uint32_t stage0 = (smem_u32(smem) + kAlign - 1) & ~(uint32_t)(kAlign - 1);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // CTA b: split b / tiles, tile b % tiles (a pair: tiles 2c, 2c + 1, its leader the even one)
  const int tiles = gridDim.x / p.splits;
  const int split = blockIdx.x / tiles;
  const int tile = blockIdx.x % tiles;
  const uint32_t rank = PAIR ? cluster_rank() : 0;
  const bool is_leader = rank == 0;
  // x's rows in this CTA's stages: a pair's CTAs half of the MMA's N each, the leader the first half
  const int x_rows = PAIR ? p.n_tok / 2 : p.n_tok;
  const int tok0 = rank * x_rows;
  const int s0 = (int)((long)split * p.k_stages / p.splits);
  const int n_stages = (int)((long)(split + 1) * p.k_stages / p.splits) - s0;
  constexpr uint32_t w_bytes = kRows * kRowBytes;
  const uint32_t stage_bytes = w_bytes + x_rows * kRowBytes;
  // the bytes a stage's full barrier counts: a pair's leader both CTAs'
  const uint32_t full_bytes = PAIR ? 2 * stage_bytes : stage_bytes;
  const uint32_t acc_cols = p.n_tok <= 32 ? 32 : (p.n_tok <= 64 ? 64 : (p.n_tok <= 128 ? 128 : 256));
  const uint32_t tmem_cols = 2 * acc_cols;

  if (threadIdx.x == 0) {
    for (int s = 0; s < p.stages; s++) {
      mbar_init(&full_barriers[s], 1);
      mbar_init(&empty_barriers[s], 1);
    }
    mbar_init(tmem_full_barrier, kMmaWarps);
    mbar_init_fence();
    tma_prefetch_map(&w_map);
    tma_prefetch_map(&x_map);
    tma_prefetch_map(&out_map);
  }
  if (warp == kProducers) tc_alloc<PAIR>(tmem_slot, tmem_cols);
  tc_fence_before();
  if (PAIR)
    cluster_sync();  // both CTAs' barriers initialized before the other's loads and commits land on them
  else
    __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;

  if (warp >= kProducers) {
    // the MMAs (A: 128 rows of w, B: the n_tok rows of x; a pair's leader issues them for both CTAs, M = 256): K
    // block i into accumulator i % 2 in order, an even ring's blocks alternating between the two MMA warps, an odd
    // ring's all on warp 0 (each slot's barrier keeps one waiter)
    const int m = warp - kProducers;
    const int mma_warps = p.stages % 2 == 0 ? kMmaWarps : 1;
    if (m < mma_warps && is_leader && elect_one()) {
      // the instruction descriptor: an fp32 accumulator (bit 4), bf16 A and B (bits 7, 10), N / 8 (bits 17-22), M / 16
      // (bits 24-28)
      const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(p.n_tok >> 3) << 17) |
                             ((uint32_t)((PAIR ? 2 : 1) * kRows >> 4) << 24);
      int s = m;
      uint32_t phase = 0;
      for (int i = m; i < n_stages; i += mma_warps) {
        mbar_wait(&full_barriers[s], phase);
        tc_fence_after();
        const uint32_t st = stage0 + s * stage_bytes;
        // K-major operands: the leading byte offset unused
        const uint64_t a_desc = sw128_desc(st, 0), b_desc = sw128_desc(st + w_bytes, 0);
        const uint32_t d = tmem + (i % kMmaWarps) * acc_cols;
#pragma unroll
        for (int k = 0; k < kKBlock / kKStep; k++)
          tc_mma<PAIR>(d, a_desc + 2 * k, b_desc + 2 * k, idesc, (i >= kMmaWarps || k > 0) ? 1u : 0u);
        tc_commit<PAIR>(&empty_barriers[s]);
        s += mma_warps;
        if (s >= p.stages) {
          s -= p.stages;
          phase ^= 1;
        }
      }
      // every accumulator has stages: the host gives every split two stages at least; the barrier counts one arrival
      // per MMA warp, so a single MMA warp arrives for both
      tc_commit<PAIR>(tmem_full_barrier);
      if (mma_warps == 1) tc_commit<PAIR>(tmem_full_barrier);
    }
  } else {
    // producer w < p.producers: the stages i = w (mod p.producers), which it alone fills (a parity wait must never
    // run more than one phase ahead). Its first stages' weights load before the wait for the previous kernel (w does
    // not depend on it), their x after
    if (warp < p.producers && elect_one()) {
      const uint64_t w_policy = l2_evict_first(), x_policy = l2_evict_last();
      const int first_stages = p.stages / p.producers;  // this producer's stages before the ring wraps
      int s = warp;
      for (int j = 0, i = warp; j < first_stages && i < n_stages; j++, i += p.producers, s += p.producers) {
        if (is_leader) mbar_expect_tx(&full_barriers[s], full_bytes);
        load_w<EPI, PAIR>(w_map, stage0 + s * stage_bytes, &full_barriers[s], tile, s0 + i, w_policy);
      }
      pdl_wait();
      s = warp;
      for (int j = 0, i = warp; j < first_stages && i < n_stages; j++, i += p.producers, s += p.producers)
        tma_load_3d<PAIR>(stage0 + s * stage_bytes + w_bytes, &x_map, 0, tok0, (s0 + i), &full_barriers[s], x_policy);
      s = warp;
      uint32_t phase = 1;
      for (int i = warp + first_stages * p.producers; i < n_stages; i += p.producers) {
        mbar_wait(&empty_barriers[s], phase ^ 1);
        const uint32_t st = stage0 + s * stage_bytes;
        uint64_t* const bar = &full_barriers[s];
        if (is_leader) mbar_expect_tx(bar, full_bytes);
        load_w<EPI, PAIR>(w_map, st, bar, tile, s0 + i, w_policy);
        tma_load_3d<PAIR>(st + w_bytes, &x_map, 0, tok0, (s0 + i), bar, x_policy);
        s += p.producers;
        if (s >= p.stages) {
          s -= p.stages;
          phase ^= 1;
        }
      }
    }
    __syncwarp();
    // every epilogue thread past the previous kernel before it writes (out may reuse memory that kernel reads)
    pdl_wait();

    // the epilogue: thread r = 32 w + lane holds row r of the tile (a row of w), n_tok tokens per accumulator; the tile
    // goes through shared memory (the stages are free once the MMAs are done) transposed to [token][column] and out by
    // one TMA store, kLogits' from the registers into both ranks' buffers
    if (lane == 0) mbar_wait(tmem_full_barrier, 0);
    __syncwarp();
    tc_fence_after();
    pdl_launch_next();
    const int r = warp * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(warp * 32) << 16);
    const uint32_t tile_smem = stage0;
    if constexpr (EPI == kSwiGLU) {
      // TMEM lane 2j holds gate row j, lane 2j + 1 up row j: in each 16-token chunk the pair trades halves, the gate
      // lane computing the first kHalfChunk tokens, the up lane the rest
      const int is_up = lane & 1;
      const int col = r >> 1;
      for (int c = 0; c < p.n_tok; c += kLoadsPerWait * 16) {
        uint32_t a[kLoadsPerWait][16], b[kLoadsPerWait][16];
#pragma unroll
        for (int h = 0; h < kLoadsPerWait; h++) {
          tc_ld_x16(trow + c + 16 * h, a[h]);
          tc_ld_x16(trow + acc_cols + c + 16 * h, b[h]);
        }
        tc_wait_ld();
#pragma unroll
        for (int h = 0; h < kLoadsPerWait; h++)
          if (c + 16 * h < p.n_tok) {
#pragma unroll
            for (int i = 0; i < kHalfChunk; i++) {
              const float first = __uint_as_float(a[h][i]) + __uint_as_float(b[h][i]);
              const float second = __uint_as_float(a[h][kHalfChunk + i]) + __uint_as_float(b[h][kHalfChunk + i]);
              const float other = __shfl_xor_sync(0xffffffffu, is_up ? first : second, 1);
              const float gate = is_up ? other : first, up = is_up ? second : other;
              const int t = c + 16 * h + kHalfChunk * is_up + i;
              sts_u16(tile_smem + (t * (kRows / 2) + col) * 2, bf16_bits(silu(gate) * up));
            }
          }
      }
    } else if constexpr (EPI == kLogits) {
      // token t's logit: its high row t plus its low row rows + t, each the sum of the two accumulators; a warp's
      // lanes store 32 adjacent columns of a row into this rank's buffer and the peer's
      const size_t col = (size_t)p.col0 + (size_t)tile * kRows + r;
      for (int c = 0; c < p.rows; c += 16) {
        uint32_t high0[16], high1[16], low0[16], low1[16];
        tc_ld_x16(trow + c, high0);
        tc_ld_x16(trow + acc_cols + c, high1);
        tc_ld_x16(trow + p.rows + c, low0);
        tc_ld_x16(trow + acc_cols + p.rows + c, low1);
        tc_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; j++) {
          const float v = (__uint_as_float(high0[j]) + __uint_as_float(high1[j])) +
                          (__uint_as_float(low0[j]) + __uint_as_float(low1[j]));
          const size_t o = (size_t)(c + j) * p.ld + col;
          p.local[o] = v;
          p.peer[o] = v;
        }
      }
    } else {
      for (int c = 0; c < p.n_tok; c += 16) {
        uint32_t a[16], b[16];
        tc_ld_x16(trow + c, a);
        tc_ld_x16(trow + acc_cols + c, b);
        tc_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; j++)
          sts_f32(tile_smem + ((c + j) * kRows + r) * 4, __uint_as_float(a[j]) + __uint_as_float(b[j]));
      }
    }
    if constexpr (EPI != kLogits) {
      fence_async_smem();
      named_sync(kEpilogueBarrier, kEpilogueThreads);
      if (threadIdx.x == 0) {
        // rows beyond M are clipped by the map
        if (EPI == kPartials)
          tma_store_3d(&out_map, tile_smem, tile * kRows, 0, split);
        else
          tma_store_2d(&out_map, tile_smem, tile * p.out_cols, 0);
        tma_store_commit_wait();
      }
    }
  }

  tc_fence_before();
  if (PAIR)
    cluster_sync();  // the leader's commits and the other CTA's loads all landed before either CTA leaves
  else
    __syncthreads();
  if (warp == kProducers) {
    tc_fence_after();
    tc_dealloc<PAIR>(tmem, tmem_cols);
  }
}

// ---- host side ----
CUtensorMap encode(CUtensorMapDataType type, int rank, void* ptr, const cuuint64_t* dims, const cuuint64_t* strides,
                   const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  const cuuint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  const CUresult res = encode_tiled()(&map, type, rank, ptr, dims, strides, box, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)res);
  return map;
}

// a row-major bf16 operand [rows, cols] (row stride ld) as {kKBlock, rows, cols / kKBlock}, boxes of box_rows x one
// atom (128-byte swizzled, zeros past its rows); halves: SwiGLU's [gate; up] as {kKBlock, 2, rows / 2, cols / kKBlock},
// so that a box's rows interleave gate row j and up row j
CUtensorMap operand_map(const torch::Tensor& t, int box_rows, bool halves) {
  const int64_t rows = t.size(0), cols = t.size(1), ld = t.stride(0);
  if (halves) {
    const cuuint64_t dims[4] = {(cuuint64_t)kKBlock, 2, (cuuint64_t)(rows / 2), (cuuint64_t)(cols / kKBlock)};
    const cuuint64_t strides[3] = {(cuuint64_t)(rows / 2) * ld * 2, (cuuint64_t)ld * 2, (cuuint64_t)kRowBytes};
    const cuuint32_t box[4] = {(cuuint32_t)kKBlock, 2, (cuuint32_t)box_rows, 1};
    return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  const cuuint64_t dims[3] = {(cuuint64_t)kKBlock, (cuuint64_t)rows, (cuuint64_t)(cols / kKBlock)};
  const cuuint64_t strides[2] = {(cuuint64_t)ld * 2, (cuuint64_t)kRowBytes};
  const cuuint32_t box[3] = {(cuuint32_t)kKBlock, (cuuint32_t)box_rows, 1};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
}

template <int EPI, bool PAIR>
void launch(const CUtensorMap& w_map, const CUtensorMap& x_map, const CUtensorMap& out_map, const Params& p, int grid,
            int smem_bytes, int smem_optin, cudaStream_t stream) {
  auto kernel = stream_gemm_kernel<EPI, PAIR>;
  // the dynamic shared memory the kernel may take: the device's opt-in limit less the kernel's static shared memory
  static const int max_dynamic = [&] {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    const int bytes = smem_optin - (int)attributes.sharedSizeBytes;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    return bytes;
  }();
  TORCH_CHECK(smem_bytes <= max_dynamic, "stream_gemm: ", smem_bytes, " bytes of shared memory, ", max_dynamic,
              " at most");
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[2];
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim.x = PAIR ? 2 : 1;
  attributes[1].val.clusterDim.y = 1;
  attributes[1].val.clusterDim.z = 1;
  config.attrs = attributes;
  config.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, w_map, x_map, out_map, p));
}

// a launch's ring of stages over n_tok rows of x (the MMA's N) and `tiles` tiles of w
struct Ring {
  bool pair;        // a CTA pair shares x: from kPairFromRows rows over an even number of tiles
  int x_rows;       // rows of x in a CTA's stage
  int stage_bytes;  // a stage: 128 rows of w and the CTA's rows of x, one atom each
  int stages;       // the most that fit shared memory
  int producers;    // the most loading warps (at most kProducers) that divide the stages: each owns its stages
  int smem_optin;   // the device's shared memory per CTA
};

Ring ring_of(int n_tok, int64_t tiles) {
  Ring ring;
  ring.pair = n_tok >= kPairFromRows && tiles % 2 == 0;
  ring.x_rows = ring.pair ? n_tok / 2 : n_tok;
  ring.stage_bytes = (kRows + ring.x_rows) * kRowBytes;
  ring.smem_optin = (int)at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  ring.stages = std::min(kMaxStages, ring.smem_optin / ring.stage_bytes);
  ring.producers = ring.stages % 4 == 0 ? 4 : (ring.stages % 3 == 0 ? 3 : (ring.stages % 2 == 0 ? 2 : 1));
  return ring;
}

// x [M, K] @ w [N, K]^T into kSwiGLU's [M, N / 2] bf16 or kPartials' fp32 sums of `splits` K ranges [splits, M, N]
template <int EPI>
torch::Tensor stream_gemm(const torch::Tensor& x, const torch::Tensor& w, int64_t splits) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "stream_gemm: x [M, K], w [N, K]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "stream_gemm: bf16");
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous() && x.stride(0) % 8 == 0, "stream_gemm: row-major operands");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxTokens, "stream_gemm: 1 to ", kMaxTokens, " rows");
  const int out_cols = EPI == kSwiGLU ? kRows / 2 : kRows;
  const int64_t n_out = EPI == kSwiGLU ? N / 2 : N;
  TORCH_CHECK(K % kKBlock == 0 && n_out % out_cols == 0, "stream_gemm: K a multiple of the stage, N of the tile");
  const int64_t tiles = n_out / out_cols;
  const int64_t k_stages = K / kKBlock;
  TORCH_CHECK(splits >= 1 && k_stages >= 2 * splits, "stream_gemm: two stages per K range at least");
  const at::cuda::OptionalCUDAGuard guard(x.device());

  // the MMA's N: x's rows rounded to 16 (a pair's CTAs hold half of them each, whole 8-row swizzle atoms)
  const int n_tok = (int)((M + 15) / 16 * 16);
  const Ring ring = ring_of(n_tok, tiles);

  torch::Tensor out;
  CUtensorMap out_map;
  if (EPI == kPartials) {
    out = torch::empty({splits, M, N}, x.options().dtype(torch::kFloat));
    const cuuint64_t dims[3] = {(cuuint64_t)N, (cuuint64_t)M, (cuuint64_t)splits};
    const cuuint64_t strides[2] = {(cuuint64_t)N * 4, (cuuint64_t)(M * N * 4)};
    const cuuint32_t box[3] = {(cuuint32_t)kRows, (cuuint32_t)n_tok, 1};
    out_map =
        encode(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, out.data_ptr(), dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    out = torch::empty({M, n_out}, x.options());
    const cuuint64_t dims[2] = {(cuuint64_t)n_out, (cuuint64_t)M};
    const cuuint64_t strides[1] = {(cuuint64_t)n_out * 2};
    const cuuint32_t box[2] = {(cuuint32_t)out_cols, (cuuint32_t)n_tok};
    out_map =
        encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, out.data_ptr(), dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  Params p;
  p.n_tok = n_tok;
  p.k_stages = (int)k_stages;
  p.splits = (int)splits;
  p.stages = ring.stages;
  p.producers = ring.producers;
  p.out_cols = out_cols;
  const CUtensorMap w_map = operand_map(w, EPI == kSwiGLU ? kRows / 2 : kRows, EPI == kSwiGLU);
  const CUtensorMap x_map = operand_map(x, ring.x_rows, false);
  // the epilogue reuses the stages: fp32 [n_tok][128] at most (kSwiGLU: the bf16 tile [n_tok][64])
  const int epilogue_bytes = EPI == kSwiGLU ? n_tok * (kRows / 2) * 2 : n_tok * kRows * 4;
  const int ring_bytes = std::max(ring.stages * ring.stage_bytes, epilogue_bytes);
  const int smem_bytes = kAlign + ring_bytes;
  const int grid = (int)(tiles * splits);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (ring.pair)
    launch<EPI, true>(w_map, x_map, out_map, p, grid, smem_bytes, ring.smem_optin, stream);
  else
    launch<EPI, false>(w_map, x_map, out_map, p, grid, smem_bytes, ring.smem_optin, stream);
  return out;
}

// the logits' gather buffers (symmetric memory), this rank's launch count and both ranks' counts of the other's, set
// once per process by sg_logits_setup
struct LogitsTargets {
  float* local = nullptr;
  float* peer = nullptr;
  uint32_t* launches = nullptr;
  uint32_t* done_local = nullptr;
  uint32_t* done_peer = nullptr;
  int64_t max_rows = 0;
  int64_t vocab = 0;
};
LogitsTargets g_logits;

}  // namespace

// silu(x @ w_gate^T) * (x @ w_up^T) for w = [gate; up]: [M, N / 2] bf16
torch::Tensor sg_swiglu(torch::Tensor x, torch::Tensor w) { return stream_gemm<kSwiGLU>(x, w, 1); }

// x @ w^T as the fp32 partial sums of `splits` K ranges, [splits, M, N]
torch::Tensor sg_partials(torch::Tensor x, torch::Tensor w, int64_t splits) {
  return stream_gemm<kPartials>(x, w, splits);
}

void sg_logits_setup(torch::Tensor buffer, int64_t peer_buffer, torch::Tensor launches, torch::Tensor done,
                     int64_t peer_done) {
  TORCH_CHECK(buffer.dim() == 2 && buffer.is_contiguous() && buffer.scalar_type() == torch::kFloat,
              "sg_logits_setup: fp32 [max_rows, V] buffer");
  TORCH_CHECK(launches.scalar_type() == torch::kInt && done.scalar_type() == torch::kInt, "sg_logits_setup: counters");
  g_logits.local = buffer.data_ptr<float>();
  g_logits.peer = reinterpret_cast<float*>(peer_buffer);
  g_logits.launches = reinterpret_cast<uint32_t*>(launches.data_ptr());
  g_logits.done_local = reinterpret_cast<uint32_t*>(done.data_ptr());
  g_logits.done_peer = reinterpret_cast<uint32_t*>(peer_done);
  g_logits.max_rows = buffer.size(0);
  g_logits.vocab = buffer.size(1);
}

// x = [high rows; low rows] of `rows` hidden states in bf16 [2 rows, K], w this rank's lm_head shard [V / 2, K]: the
// logits [rows, V] in fp32, both shards, as a view of the gather buffer valid until the next call
torch::Tensor sg_logits(torch::Tensor x, torch::Tensor w, int64_t rows, int64_t rank) {
  TORCH_CHECK(g_logits.local != nullptr, "sg_logits: sg_logits_setup was not called");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && x.size(0) == 2 * rows, "sg_logits: shapes");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "sg_logits: bf16");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "sg_logits: contiguous operands");
  TORCH_CHECK(rows >= 16 && rows % 16 == 0 && 2 * rows <= kMaxTokens && rows <= g_logits.max_rows, "sg_logits: rows");
  TORCH_CHECK(2 * w.size(0) == g_logits.vocab && w.size(0) % kRows == 0, "sg_logits: the shard's rows");
  TORCH_CHECK(w.size(1) % kKBlock == 0 && w.size(1) / kKBlock >= 2, "sg_logits: two stages of K at least");
  const int64_t tiles = w.size(0) / kRows;
  const at::cuda::OptionalCUDAGuard guard(x.device());

  const int n_tok = (int)(2 * rows);
  const Ring ring = ring_of(n_tok, tiles);
  Params p;
  p.n_tok = n_tok;
  p.k_stages = (int)(w.size(1) / kKBlock);
  p.splits = 1;
  p.stages = ring.stages;
  p.producers = ring.producers;
  p.out_cols = kRows;
  p.rows = (int)rows;
  p.col0 = (int)(rank * w.size(0));
  p.ld = (int)g_logits.vocab;
  p.local = g_logits.local;
  p.peer = g_logits.peer;
  const CUtensorMap w_map = operand_map(w, kRows, false);
  const CUtensorMap x_map = operand_map(x, ring.x_rows, false);
  // the epilogue stores from registers: the ring alone
  const int smem_bytes = kAlign + ring.stages * ring.stage_bytes;
  auto stream = at::cuda::getCurrentCUDAStream();
  // the epilogue stores through plain pointers: w's map stands in for the unused output map
  if (ring.pair)
    launch<kLogits, true>(w_map, x_map, w_map, p, (int)tiles, smem_bytes, ring.smem_optin, stream);
  else
    launch<kLogits, false>(w_map, x_map, w_map, p, (int)tiles, smem_bytes, ring.smem_optin, stream);

  // the signal after the grid, a programmatic dependent of it
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(1);
  config.blockDim = dim3(1);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(
      cudaLaunchKernelEx(&config, logits_signal_kernel, g_logits.launches, g_logits.done_local, g_logits.done_peer));
  return torch::from_blob(g_logits.local, {rows, g_logits.vocab}, x.options().dtype(torch::kFloat));
}

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("sg_swiglu(Tensor x, Tensor w) -> Tensor");
  m.def("sg_partials(Tensor x, Tensor w, int splits) -> Tensor");
  m.def("sg_logits_setup(Tensor buffer, int peer_buffer, Tensor launches, Tensor done, int peer_done) -> ()");
  m.def("sg_logits(Tensor x, Tensor w, int rows, int rank) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("sg_swiglu", &sg_swiglu);
  m.impl("sg_partials", &sg_partials);
  m.impl("sg_logits_setup", &sg_logits_setup);
  m.impl("sg_logits", &sg_logits);
}
