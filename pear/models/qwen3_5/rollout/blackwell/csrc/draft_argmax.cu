// The drafter's greedy token over two tensor-parallel ranks: each row's argmax of this rank's fp32 vocabulary-shard
// logits (a chunk per CTA, the row's last CTA merging), swapped with the peer over NVLink as one 64-bit word of value,
// epoch and token (symmetric memory); the larger value, the smaller vocabulary index on ties, on both ranks. Each row's
// NaN and +inf logits over both shards, corrupted ones the rollout raises on, are added to its count (swapped in a
// second word of the epoch).
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sampler_common.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVec = 4;                      // float4 per thread
constexpr int kChunk = kThreads * kVec * 4;  // floats of a row per CTA
constexpr int kMaxChunks = 32;               // a row's chunks, merged by a warp's lanes
constexpr int kTokenBits = 18;               // a vocabulary id in the exchanged word
constexpr uint64_t kTokenMask = (1ull << kTokenBits) - 1;
constexpr uint64_t kEpochMask = (1ull << (32 - kTokenBits)) - 1;  // the epoch's low bits in the word
constexpr uint64_t kTagMask = kEpochMask << kTokenBits;

// the exchanged words, (value, epoch, token) and (count, epoch): two morally strong 64-bit accesses at system scope in
// one instruction, each word never torn
__device__ __forceinline__ ulonglong2 ld_relaxed_sys_pair(const uint64_t* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_relaxed_sys_pair(uint64_t* p, uint64_t x, uint64_t y) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(x), "l"(y) : "memory");
}

// this rank's exchange state: the slots the peer writes, the peer's (UVA), the per-row epochs, the per-row counts of
// finished chunks, the chunks' partials and the rows' counts of NaN and +inf logits
struct ArgmaxState {
  uint64_t* slots = nullptr;  // [2, max_rows, 2] by the epoch's parity: (value bits, epoch, token), (count, epoch)
  uint64_t* peer_slots = nullptr;
  uint32_t* epochs = nullptr;  // [max_rows]
  uint32_t* done = nullptr;    // [max_rows]
  float* part_val = nullptr;   // [max_rows, kMaxChunks]
  int* part_idx = nullptr;
  int* part_nonfinite = nullptr;
  int* nonfinite = nullptr;  // [max_rows], added to by every call
  int max_rows = 0;
};
ArgmaxState g_da;

// a row chunk per CTA: its best (value, index) and count of NaN and +inf logits; the row's last CTA merges the chunks,
// swaps the row's best and count with the peer and writes the better of the two and the sum
__global__ void __launch_bounds__(kThreads) da_argmax(const float* __restrict__ logits, int V, int vocab_start,
                                                      int64_t* __restrict__ tokens, ArgmaxState state) {
  const int c = blockIdx.x, r = blockIdx.y, S = gridDim.x;
  const int start = c * kChunk;
  const int n = min(kChunk, V - start);
  const float* x = logits + (int64_t)r * V + start;
  __shared__ float red_v[kWarps];
  __shared__ int red_i[kWarps], red_n[kWarps];
  __shared__ bool is_last;
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
  if (__any_sync(0xffffffffu, nonfinite_count != 0)) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) nonfinite_count += __shfl_xor_sync(0xffffffffu, nonfinite_count, o);
  }
  warp_best(best, index);
  if (l == 0) red_v[w] = best, red_i[w] = index, red_n[w] = nonfinite_count;
  __syncthreads();
  if (w == 0) {
    best = l < kWarps ? red_v[l] : -INFINITY;
    index = l < kWarps ? red_i[l] : INT_MAX;
    nonfinite_count = l < kWarps ? red_n[l] : 0;
    warp_best(best, index);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) nonfinite_count += __shfl_xor_sync(0xffffffffu, nonfinite_count, o);
    if (l == 0) {
      state.part_val[r * kMaxChunks + c] = best;
      state.part_idx[r * kMaxChunks + c] = index;
      state.part_nonfinite[r * kMaxChunks + c] = nonfinite_count;
      __threadfence();
      is_last = atomicAdd(state.done + r, 1u) == (uint32_t)(S - 1);
    }
  }
  __syncthreads();
  if (!is_last || w != 0) return;

  // the row's last CTA: the chunks' best and count, then the ranks'
  __threadfence();
  best = l < S ? __ldcg(state.part_val + r * kMaxChunks + l) : -INFINITY;
  index = l < S ? __ldcg(state.part_idx + r * kMaxChunks + l) : INT_MAX;
  nonfinite_count = l < S ? __ldcg(state.part_nonfinite + r * kMaxChunks + l) : 0;
  warp_best(best, index);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) nonfinite_count += __shfl_xor_sync(0xffffffffu, nonfinite_count, o);
  if (l == 0) {
    state.done[r] = 0;
    const int token = index + vocab_start;
    const uint32_t epoch = state.epochs[r] + 1;
    state.epochs[r] = epoch;
    const int slot = 2 * ((epoch & 1) * state.max_rows + r);
    const uint64_t tag = (uint64_t)(epoch & kEpochMask) << kTokenBits;
    st_relaxed_sys_pair(state.peer_slots + slot, (uint64_t)__float_as_uint(best) << 32 | tag | (uint32_t)token,
                        (uint64_t)(uint32_t)nonfinite_count << 32 | tag);
    ulonglong2 peer;
    do {
      peer = ld_relaxed_sys_pair(state.slots + slot);
    } while ((peer.x & kTagMask) != tag || (peer.y & kTagMask) != tag);
    const float peer_best = __uint_as_float((uint32_t)(peer.x >> 32));
    const int peer_token = (int)(peer.x & kTokenMask);
    tokens[r] = better(peer_best, peer_token, best, token) ? peer_token : token;
    state.nonfinite[r] += nonfinite_count + (int)(uint32_t)(peer.y >> 32);
  }
}

}  // namespace

void da_setup(torch::Tensor slots, int64_t peer_slots, torch::Tensor epochs, torch::Tensor done, torch::Tensor part_val,
              torch::Tensor part_idx, torch::Tensor part_nonfinite, torch::Tensor nonfinite) {
  const int64_t max_rows = epochs.numel();
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.numel() == 4 * max_rows);
  TORCH_CHECK(epochs.scalar_type() == torch::kInt32 && done.numel() == max_rows);
  TORCH_CHECK(part_val.numel() == max_rows * kMaxChunks && part_idx.numel() == max_rows * kMaxChunks);
  TORCH_CHECK(part_nonfinite.scalar_type() == torch::kInt32 && part_nonfinite.numel() == max_rows * kMaxChunks);
  TORCH_CHECK(nonfinite.scalar_type() == torch::kInt32 && nonfinite.numel() == max_rows);
  g_da.slots = reinterpret_cast<uint64_t*>(slots.data_ptr());
  g_da.peer_slots = reinterpret_cast<uint64_t*>(peer_slots);
  g_da.epochs = reinterpret_cast<uint32_t*>(epochs.data_ptr());
  g_da.done = reinterpret_cast<uint32_t*>(done.data_ptr());
  g_da.part_val = part_val.data_ptr<float>();
  g_da.part_idx = part_idx.data_ptr<int>();
  g_da.part_nonfinite = part_nonfinite.data_ptr<int>();
  g_da.nonfinite = nonfinite.data_ptr<int>();
  g_da.max_rows = max_rows;
}

// logits [n, V] fp32 contiguous, this rank's shard (vocabulary ids from vocab_start). Returns the drafts int64 [n]; the
// rows' NaN and +inf logits over both shards are added to da_setup's nonfinite [:n].
torch::Tensor draft_argmax(torch::Tensor logits, int64_t vocab_start) {
  TORCH_CHECK(g_da.slots != nullptr, "draft_argmax: da_setup was not called");
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32);
  const int n = logits.size(0), V = logits.size(1);
  const int S = (V + kChunk - 1) / kChunk;
  TORCH_CHECK(V % 4 == 0 && S <= kMaxChunks && n <= g_da.max_rows, "draft_argmax: shape");
  TORCH_CHECK(vocab_start + V <= (1 << kTokenBits), "draft_argmax: vocabulary ids beyond the exchanged word");
  auto tokens = torch::empty({n}, logits.options().dtype(torch::kInt64));
  if (n == 0) return tokens;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  da_argmax<<<dim3(S, n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), V, (int)vocab_start, tokens.data_ptr<int64_t>(), g_da);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tokens;
}

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def(
      "da_setup(Tensor slots, int peer_slots, Tensor epochs, Tensor done, Tensor part_val, Tensor part_idx, "
      "Tensor part_nonfinite, Tensor nonfinite) -> ()");
  m.def("draft_argmax(Tensor logits, int vocab_start) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("da_setup", &da_setup);
  m.impl("draft_argmax", &draft_argmax);
}
