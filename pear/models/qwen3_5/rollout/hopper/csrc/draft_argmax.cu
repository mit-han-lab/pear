// The drafter's greedy token over the tensor-parallel ranks: each row's best (value, index) of this rank's fp32
// vocabulary-shard logits, a chunk per CTA (da_chunks); then one CTA (da_exchange) merges every row's chunks, sends the
// row's best to each rank over NVLink (symmetric memory, a release flag per rank and call) and takes the ranks' best:
// the largest value, the smallest token on ties (vLLM's local max and argmax over the ranks), the same on every rank.
// Each row's NaN and +inf logits over every shard, corrupted ones the rollout raises on, are added to its count.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <climits>
#include <cstdint>

namespace draft_argmax {

constexpr int kWorld = 4;                    // tensor-parallel ranks exchanging their best
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVec = 4;                      // float4 per thread
constexpr int kChunk = kThreads * kVec * 4;  // floats of a row per CTA
constexpr int kMaxChunks = 32;               // a row's chunks, merged by a warp's lanes
constexpr int kExchangeThreads = 1024;       // the exchange CTA: a warp per row at a time
constexpr int kExchangeWarps = kExchangeThreads / 32;
constexpr int kSlotWords = 2;                // a row's slot: (value bits, token), then its NaN and +inf count

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// (value, index) order: larger value first, then smaller index
__device__ __forceinline__ bool better(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

// the warp's best (value, index), in every lane
__device__ __forceinline__ void warp_best(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, v, o);
    const int j = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(w, j, v, i)) v = w, i = j;
  }
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// the exchange's buffers: this rank's slots and flags (every rank writes them), every rank's (mapped into this one),
// the calls so far, the chunks' partials and the rows' counts of NaN and +inf logits
struct Exchange {
  uint64_t* slots;  // [2][kWorld][max_rows][kSlotWords] by the call's parity and the sending rank
  uint32_t* flags;  // [kWorld]: the last call whose slots each rank sent
  uint64_t* rank_slots[kWorld];
  uint32_t* rank_flags[kWorld];
  uint32_t* calls;  // [1]
  float* part_val;  // [max_rows][kMaxChunks]
  int* part_idx;
  int* part_nonfinite;
  int* nonfinite;  // [max_rows], added to by every call
  int rank, max_rows;
};

// a row chunk per CTA: its best (value, shard index) and count of NaN and +inf logits
__global__ void __launch_bounds__(kThreads) da_chunks(const float* __restrict__ logits, int V,
                                                      float* __restrict__ part_val, int* __restrict__ part_idx,
                                                      int* __restrict__ part_nonfinite) {
  const int c = blockIdx.x, r = blockIdx.y;
  const int start = c * kChunk;
  const int n = min(kChunk, V - start);
  const float* x = logits + (int64_t)r * V + start;
  __shared__ float red_v[kWarps];
  __shared__ int red_i[kWarps], red_n[kWarps];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;

  float4 v[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    v[k] = b < n ? *reinterpret_cast<const float4*>(x + b) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  }
  // the race from this thread's first token (no index left unset); a NaN or +inf logit leaves the sum NaN or +inf
  float best = -INFINITY, value_sum = 0.f;
  int index = threadIdx.x * 4 < n ? start + threadIdx.x * 4 : INT_MAX;
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = start + (k * kThreads + threadIdx.x) * 4;
    const float e[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (better(e[q], b + q, best, index)) best = e[q], index = b + q;
      value_sum += e[q];
    }
  }
  // the thread's NaN and +inf logits, which it reads again (rare), summed over the warp
  int nonfinite_count = 0;
  if (!(value_sum < INFINITY)) {
    for (int b = threadIdx.x * 4; b < n; b += kThreads * 4) {
      for (int q = 0; q < 4; ++q) nonfinite_count += !(x[b + q] < INFINITY);
    }
  }
  if (__any_sync(0xffffffffu, nonfinite_count != 0)) nonfinite_count = warp_sum(nonfinite_count);
  warp_best(best, index);
  if (l == 0) red_v[w] = best, red_i[w] = index, red_n[w] = nonfinite_count;
  __syncthreads();
  if (w == 0) {
    best = l < kWarps ? red_v[l] : -INFINITY;
    index = l < kWarps ? red_i[l] : INT_MAX;
    nonfinite_count = warp_sum(l < kWarps ? red_n[l] : 0);
    warp_best(best, index);
    if (l == 0) {
      part_val[r * kMaxChunks + c] = best, part_idx[r * kMaxChunks + c] = index;
      part_nonfinite[r * kMaxChunks + c] = nonfinite_count;
    }
  }
}

// one CTA: each row's best of its chunks and its count sent to every rank, then the ranks' best and the sum of their
// counts once every rank's slots arrived
__global__ void __launch_bounds__(kExchangeThreads) da_exchange(int rows, int S, int vocab_start,
                                                                int64_t* __restrict__ tokens, Exchange ex) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  const uint32_t call = *ex.calls + 1;
  const int parity = call & 1;

  for (int r = w; r < rows; r += kExchangeWarps) {
    float best = l < S ? ex.part_val[r * kMaxChunks + l] : -INFINITY;
    int index = l < S ? ex.part_idx[r * kMaxChunks + l] : INT_MAX;
    const int nonfinite_count = warp_sum(l < S ? ex.part_nonfinite[r * kMaxChunks + l] : 0);
    warp_best(best, index);
    const uint64_t packed = (uint64_t)__float_as_uint(best) << 32 | (uint32_t)(index + vocab_start);
    if (l < kWorld) {
      uint64_t* slot = ex.rank_slots[l] + ((parity * kWorld + ex.rank) * ex.max_rows + r) * kSlotWords;
      slot[0] = packed;
      slot[1] = (uint64_t)(uint32_t)nonfinite_count;
    }
  }
  // the slots before the flags: each writer's fence, then the CTA's barrier, then a release per rank
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x < kWorld) {
    st_release_sys(ex.rank_flags[threadIdx.x] + ex.rank, call);
    while (ld_acquire_sys(ex.flags + threadIdx.x) < call) {
    }
  }
  __syncthreads();

  for (int r = w; r < rows; r += kExchangeWarps) {
    float best = -INFINITY;
    int token = INT_MAX, nonfinite_count = 0;
    if (l < kWorld) {
      const unsigned long long* slot = reinterpret_cast<const unsigned long long*>(
          ex.slots + ((parity * kWorld + l) * ex.max_rows + r) * kSlotWords);
      const uint64_t packed = __ldcv(slot);
      best = __uint_as_float((uint32_t)(packed >> 32));
      token = (int)(uint32_t)packed;
      nonfinite_count = (int)(uint32_t)__ldcv(slot + 1);
    }
    warp_best(best, token);
    nonfinite_count = warp_sum(nonfinite_count);
    if (l == 0) {
      tokens[r] = token;
      ex.nonfinite[r] += nonfinite_count;
    }
  }
  if (threadIdx.x == 0) *ex.calls = call;
}

// logits [rows, V] fp32 contiguous, this rank's shard (tokens from vocab_start); the exchange's buffers (slots
// [2, kWorld, max_rows, kSlotWords] int64 and flags [kWorld] int32 in symmetric memory, every rank's addresses, calls
// [1] int32, partials [max_rows, kMaxChunks], counts [max_rows] int32). Returns the drafts int64 [rows], the same on
// every rank; the rows' NaN and +inf logits over every shard are added to nonfinite [:rows].
torch::Tensor draft_argmax(torch::Tensor logits, int64_t vocab_start, torch::Tensor slots, torch::Tensor flags,
                           std::vector<int64_t> rank_slots, std::vector<int64_t> rank_flags, torch::Tensor calls,
                           torch::Tensor part_val, torch::Tensor part_idx, torch::Tensor part_nonfinite,
                           torch::Tensor nonfinite, int64_t rank) {
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32, "logits");
  const int rows = logits.size(0), V = logits.size(1);
  const int S = (V + kChunk - 1) / kChunk;
  const int max_rows = slots.numel() / (2 * kWorld * kSlotWords);
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && flags.scalar_type() == torch::kInt32 && flags.numel() == kWorld,
              "draft_argmax: slots, flags");
  TORCH_CHECK((int)rank_slots.size() == kWorld && (int)rank_flags.size() == kWorld && rank >= 0 && rank < kWorld,
              "draft_argmax: ranks");
  TORCH_CHECK(calls.scalar_type() == torch::kInt32 && part_val.numel() == (int64_t)max_rows * kMaxChunks &&
                  part_idx.numel() == (int64_t)max_rows * kMaxChunks,
              "draft_argmax: calls, partials");
  TORCH_CHECK(part_nonfinite.scalar_type() == torch::kInt32 &&
                  part_nonfinite.numel() == (int64_t)max_rows * kMaxChunks &&
                  nonfinite.scalar_type() == torch::kInt32 && nonfinite.numel() == max_rows,
              "draft_argmax: counts");
  TORCH_CHECK(V % 4 == 0 && S <= kMaxChunks && rows <= max_rows, "draft_argmax: shape");
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  auto tokens = torch::empty({rows}, logits.options().dtype(torch::kInt64));
  if (rows == 0) return tokens;

  Exchange ex;
  ex.slots = reinterpret_cast<uint64_t*>(slots.data_ptr());
  ex.flags = reinterpret_cast<uint32_t*>(flags.data_ptr());
  for (int p = 0; p < kWorld; ++p) {
    ex.rank_slots[p] = reinterpret_cast<uint64_t*>(rank_slots[p]);
    ex.rank_flags[p] = reinterpret_cast<uint32_t*>(rank_flags[p]);
  }
  ex.calls = reinterpret_cast<uint32_t*>(calls.data_ptr());
  ex.part_val = part_val.data_ptr<float>();
  ex.part_idx = part_idx.data_ptr<int>();
  ex.part_nonfinite = part_nonfinite.data_ptr<int>();
  ex.nonfinite = nonfinite.data_ptr<int>();
  ex.rank = rank;
  ex.max_rows = max_rows;
  auto stream = at::cuda::getCurrentCUDAStream();
  da_chunks<<<dim3(S, rows), kThreads, 0, stream>>>(logits.data_ptr<float>(), V, ex.part_val, ex.part_idx,
                                                    ex.part_nonfinite);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  da_exchange<<<1, kExchangeThreads, 0, stream>>>(rows, S, (int)vocab_start, tokens.data_ptr<int64_t>(), ex);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tokens;
}

}  // namespace draft_argmax

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "draft_argmax(Tensor logits, int vocab_start, Tensor(a!) slots, Tensor(b!) flags, int[] rank_slots, "
      "int[] rank_flags, Tensor(c!) calls, Tensor(d!) part_val, Tensor(e!) part_idx, Tensor(f!) part_nonfinite, "
      "Tensor(g!) nonfinite, int rank) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("draft_argmax", &draft_argmax::draft_argmax); }
