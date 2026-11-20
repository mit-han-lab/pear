// The samplers' shared device helpers (the fused sampler, the verify sampler, the drafter's argmax): block reductions,
// the (value, index) order, bitonic sorts of a chunk's top candidates, the temperature's exact division.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kTopK = 32;  // candidates per chunk: >= the requested log-probs
constexpr int kCap = 512;  // candidates a chunk sorts for its exact top-kTopK (more: the radix-select path)

__device__ __forceinline__ uint32_t fkey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// (value, index) order: larger value first, then smaller index
__device__ __forceinline__ bool better(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

template <int NT>
__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = l < NT / 32 ? red[l] : -INFINITY;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = l < NT / 32 ? red[l] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// the warp's best (value, index), in every lane
__device__ __forceinline__ void warp_best(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, v, o);
    const int j = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(w, j, v, i)) v = w, i = j;
  }
}

template <int NT>
__device__ __forceinline__ void block_best(float& v, int& i, float* redv, int* redi) {
  warp_best(v, i);
  const int wp = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) redv[wp] = v, redi[wp] = i;
  __syncthreads();
  v = l < NT / 32 ? redv[l] : -INFINITY;
  i = l < NT / 32 ? redi[l] : INT_MAX;
  warp_best(v, i);
}

// x / t, exact (IEEE division even under --use_fast_math); t == 1 skips it
__device__ __forceinline__ float scaled(float x, float t) { return t == 1.f ? x : __fdiv_rn(x, t); }

// warp-wide bitonic sort (descending) of n2 (power of two, <= 2 * kCap) smem pairs; __syncwarp between stages
__device__ __forceinline__ void warp_sort_desc(float* sv, int* si, int n2) {
  const int l = threadIdx.x & 31;
  for (int size = 2; size <= n2; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int k = l; k < n2 / 2; k += 32) {
        const int lo = 2 * k - (k & (stride - 1));
        const int hi = lo + stride;
        const bool desc = (lo & size) == 0;
        const bool swap = desc ? better(sv[hi], si[hi], sv[lo], si[lo]) : better(sv[lo], si[lo], sv[hi], si[hi]);
        if (swap) {
          const float tv = sv[lo]; sv[lo] = sv[hi]; sv[hi] = tv;
          const int ti = si[lo]; si[lo] = si[hi]; si[hi] = ti;
        }
      }
      __syncwarp();
    }
  }
}

// one value per lane, sorted descending across the warp (shuffle bitonic)
__device__ __forceinline__ float warp_sort_lanes_desc(float v) {
  const int l = threadIdx.x & 31;
#pragma unroll
  for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const float w = __shfl_xor_sync(0xffffffffu, v, stride);
      const bool lower = (l & stride) == 0;      // this lane holds the pair's first position
      const bool desc = (l & size) == 0;         // the block sorts descending
      // descending block: the first position keeps the larger value
      const bool keep_max = lower == desc;
      v = keep_max ? fmaxf(v, w) : fminf(v, w);
    }
  }
  return v;
}

}  // namespace
