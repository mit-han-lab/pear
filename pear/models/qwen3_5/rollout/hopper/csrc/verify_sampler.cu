// vLLM's rejection sampling of greedy drafts at a temperature alone (x = logits / t, lp = log_softmax(x)): the lossless
// rule lp(d) > log u or the synthetic u < rate on vLLM's tl.rand(seed, pos), the recovered or bonus token by inverse
// transform on another Philox word of (seed, pos), the top-K lp of every row and the kept tokens' ranks; each row's
// count of NaN and +inf x, corrupted logits the rollout raises on.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace verify_sampler {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVec = 8;                      // float4 per thread: a chunk of up to 8192 floats held in registers
constexpr int kChunk = kThreads * kVec * 4;  // floats of a row per CTA
constexpr int kMaxChunks = 32;               // a row's chunks, merged by a warp's lanes
constexpr int kTopK = 32;                    // candidates per chunk: >= the requested log-probs
constexpr int kCap = 512;                    // candidates a chunk sorts for its exact top-kTopK (more: radix select)
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kMaxRows = kWarps;                  // a request's rows (its last token and its drafts), a warp each
constexpr int kPhiloxRounds = 10;                 // Triton's tl.rand
constexpr float kRandScale = 4.6566127342e-10f;  // Triton's uint_to_uniform_float scale for 32-bit words
constexpr float kRandMin = 4.6566127342e-10f;    // the smallest u vLLM's rejection sampler draws (_TL_RAND_MIN)

// Triton's Philox4x32 (tl.randint4x) of a 64-bit seed at a 64-bit offset
__device__ __forceinline__ uint4 philox(uint64_t seed, uint64_t offset) {
  uint32_t c0 = (uint32_t)offset, c1 = (uint32_t)(offset >> 32), c2 = 0, c3 = 0;
  uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
  for (int round = 0; round < kPhiloxRounds; ++round) {
    const uint32_t a = c0, b = c2;
    c0 = __umulhi(0xCD9E8D57u, b) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53u, a) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * b;
    c3 = 0xD2511F53u * a;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}

// a Philox word as Triton's uniform float in (0, 1], at least kRandMin (vLLM's tl_rand32 without zero)
__device__ __forceinline__ float uniform(uint32_t word) {
  const int x = (int)word;
  return fmaxf((float)(x < 0 ? ~x : x) * kRandScale, kRandMin);
}

// x / t, exact (IEEE division); t == 1 skips it
__device__ __forceinline__ float scaled(float x, float t) { return t == 1.f ? x : __fdiv_rn(x, t); }

// a float's order as an unsigned key
__device__ __forceinline__ uint32_t fkey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// (value, index) order: larger value first, then smaller index
__device__ __forceinline__ bool better(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = l < kWarps ? red[l] : -INFINITY;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// one value per lane, sorted descending across the warp (shuffle bitonic)
__device__ __forceinline__ float warp_sort_lanes_desc(float v) {
  const int l = threadIdx.x & 31;
#pragma unroll
  for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const float w = __shfl_xor_sync(0xffffffffu, v, stride);
      // a descending block: the pair's first position keeps the larger value
      const bool keep_max = ((l & stride) == 0) == ((l & size) == 0);
      v = keep_max ? fmaxf(v, w) : fminf(v, w);
    }
  }
  return v;
}

// a bitonic sequence over the warp's lanes, sorted descending
__device__ __forceinline__ float warp_merge_desc(float v) {
  const int l = threadIdx.x & 31;
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, v, stride);
    v = (l & stride) == 0 ? fmaxf(v, w) : fminf(v, w);
  }
  return v;
}

// warp-wide bitonic sort (descending) of n2 (a power of two, <= kCap) shared-memory pairs
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
          const float tv = sv[lo];
          sv[lo] = sv[hi];
          sv[hi] = tv;
          const int ti = si[lo];
          si[lo] = si[hi];
          si[hi] = ti;
        }
      }
      __syncwarp();
    }
  }
}

// the chunk kernel's shared memory
struct ChunkScratch {
  float red[3][kWarps];  // the max's scratch, then the warps' exp-sums and counts
  float wmax[kWarps * 32];
  uint32_t hist[256];
  uint32_t prefix, krem;
  int n, eq;
  int nonfinite;  // the chunk's NaN and +inf x
  float sv[kCap];
  int si[kCap];
};

// a row chunk per CTA, in registers: its max and exp-sum of x, its count of x >= xd (the logit of the draft d the row
// verifies, for the draft's rank), its count of NaN and +inf x and its sorted top-kTopK from the candidates >= theta
// (the kTopK-th largest of the threads' maxima); a radix select when ties leave more than kCap candidates. kIsFull: a
// chunk of kChunk floats, unguarded
template <bool kIsFull>
__device__ __forceinline__ void chunk_partials(const float* __restrict__ x, int start, int n, float t, int d,
                                               float xd, int p,
                                               float* __restrict__ part_max, float* __restrict__ part_sum,
                                               float* __restrict__ part_count, int* __restrict__ part_nonfinite,
                                               float* __restrict__ cand_val, int* __restrict__ cand_idx,
                                               ChunkScratch& sh) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (threadIdx.x == 0) sh.nonfinite = 0;  // before block_max's barriers

  // the chunk's float4s scaled by the temperature, -inf past its end
  float4 v[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    v[k] = kIsFull || b < n ? *reinterpret_cast<const float4*>(x + start + b)
                            : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  }
  if (t != 1.f) {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      v[k] = make_float4(__fdiv_rn(v[k].x, t), __fdiv_rn(v[k].y, t), __fdiv_rn(v[k].z, t), __fdiv_rn(v[k].w, t));
    }
  }
  float tmax = -INFINITY;
#pragma unroll
  for (int k = 0; k < kVec; ++k) tmax = fmaxf(tmax, fmaxf(fmaxf(v[k].x, v[k].y), fmaxf(v[k].z, v[k].w)));
  float count = 0.f;
  if (d >= 0) {
    int above_count = 0;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      if (kIsFull || (k * kThreads + threadIdx.x) * 4 < n) {
        above_count += (v[k].x >= xd) + (v[k].y >= xd) + (v[k].z >= xd) + (v[k].w >= xd);
      }
    }
    count = above_count;
  }

  // exp(x - tmax) as one FFMA and one exp2 each (every x -inf: tmax -inf, a shift of 0, every term 0); a NaN or +inf x
  // makes the sum NaN
  const float shift = tmax == -INFINITY ? 0.f : -tmax * kLog2e;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    s += exp2f(fmaf(v[k].x, kLog2e, shift)) + exp2f(fmaf(v[k].y, kLog2e, shift)) +
         exp2f(fmaf(v[k].z, kLog2e, shift)) + exp2f(fmaf(v[k].w, kLog2e, shift));
  }
  const bool is_nonfinite = isnan(s);
  const float M = block_max(tmax, sh.red[0]);
  // the thread's NaN and +inf x (rare)
  if (is_nonfinite) {
    int nonfinite_count = 0;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      nonfinite_count += !(v[k].x < INFINITY) + !(v[k].y < INFINITY) + !(v[k].z < INFINITY) + !(v[k].w < INFINITY);
    }
    atomicAdd(&sh.nonfinite, nonfinite_count);
  }
  s = tmax == -INFINITY ? 0.f : s * exp2f((tmax - M) * kLog2e);
  // the exp-sum and the count together, into rows block_max's last reads do not share
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
    count += __shfl_xor_sync(0xffffffffu, count, o);
  }
  if (l == 0) sh.red[1][w] = s, sh.red[2][w] = count;
  if (threadIdx.x == 0) sh.n = 0, sh.eq = 0;

  // theta: the warps' lane maxima sorted, then merged pairwise to the top kTopK (bitonic merges of two sorted lists)
  sh.wmax[w * 32 + l] = warp_sort_lanes_desc(tmax);
  __syncthreads();
  if (threadIdx.x == 0) {
    float chunk_sum = 0.f, chunk_count = 0.f;
#pragma unroll
    for (int q = 0; q < kWarps; ++q) chunk_sum += sh.red[1][q], chunk_count += sh.red[2][q];
    part_max[p] = M;
    part_sum[p] = chunk_sum;
    part_count[p] = chunk_count;
    part_nonfinite[p] = sh.nonfinite;
  }
  for (int lists = kWarps; lists > 1; lists >>= 1) {
    const bool is_merging = w < lists / 2;
    float merged_max = 0.f;
    if (is_merging) merged_max = warp_merge_desc(fmaxf(sh.wmax[2 * w * 32 + l], sh.wmax[(2 * w + 1) * 32 + 31 - l]));
    __syncthreads();
    if (is_merging) sh.wmax[w * 32 + l] = merged_max;
    __syncthreads();
  }
  const float theta = sh.wmax[kTopK - 1];

  // the candidates >= theta: each element slot's lanes appended at the warp's offset in the list
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    const float e[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const bool is_candidate = (kIsFull || b < n) && e[q] >= theta;
      const unsigned lanes = __ballot_sync(0xffffffffu, is_candidate);
      if (lanes) {
        int base = 0;
        if (l == 0) base = atomicAdd(&sh.n, __popc(lanes));
        const int slot = __shfl_sync(0xffffffffu, base, 0) + __popc(lanes & ((1u << l) - 1u));
        if (is_candidate && slot < kCap) sh.sv[slot] = e[q], sh.si[slot] = start + b + q;
      }
    }
  }
  __syncthreads();
  const int nc = sh.n;
  float* cv = cand_val + (int64_t)p * kTopK;
  int* ci = cand_idx + (int64_t)p * kTopK;
  if (nc <= kThreads) {
    // a thread per candidate: its position is the count of candidates before it in the (value, index) order
    if (threadIdx.x < nc) {
      const float mv = sh.sv[threadIdx.x];
      const int mi = sh.si[threadIdx.x];
      int position = 0;
      for (int j = 0; j < nc; ++j) position += better(sh.sv[j], sh.si[j], mv, mi);
      if (position < kTopK) cv[position] = mv, ci[position] = mi;
    } else if (threadIdx.x < kTopK) {
      cv[threadIdx.x] = -INFINITY, ci[threadIdx.x] = INT_MAX;  // a chunk of fewer than kTopK elements
    }
    return;
  }
  if (nc <= kCap) {
    if (w == 0) {
      int n2 = kTopK;
      while (n2 < nc) n2 <<= 1;
      for (int k = nc + l; k < n2; k += 32) sh.sv[k] = -INFINITY, sh.si[k] = INT_MAX;
      __syncwarp();
      warp_sort_desc(sh.sv, sh.si, n2);
      cv[l] = sh.sv[l];
      ci[l] = sh.si[l];
    }
    return;
  }

  // many ties above theta: radix select of the kTopK-th largest key over the chunk (8 bits per pass)
  if (threadIdx.x == 0) sh.prefix = 0, sh.krem = kTopK;
  uint32_t mask = 0;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    sh.hist[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = sh.prefix;
    for (int b = threadIdx.x; b < n; b += kThreads) {
      const uint32_t key = fkey(scaled(x[start + b], t));
      if ((key & mask) == prefix) atomicAdd(&sh.hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      uint32_t cnt[8], tot = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) cnt[b] = sh.hist[255 - 8 * l - b], tot += cnt[b];
      uint32_t incl = tot;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
        if (l >= o) incl += y;
      }
      const uint32_t krem = sh.krem;
      const uint32_t excl = incl - tot;
      if (excl < krem && incl >= krem) {
        uint32_t acc = excl;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          if (acc + cnt[b] >= krem) {
            sh.prefix = prefix | ((uint32_t)(255 - 8 * l - b) << shift);
            sh.krem = krem - acc;
            break;
          }
          acc += cnt[b];
        }
      }
    }
    mask |= 255u << shift;
    __syncthreads();
  }
  const uint32_t T = sh.prefix;
  const int krem = (int)sh.krem;
  const int gt = kTopK - krem;
  if (threadIdx.x == 0) sh.n = 0;
  __syncthreads();
  for (int b = threadIdx.x; b < n; b += kThreads) {
    const float xs = scaled(x[start + b], t);
    const uint32_t key = fkey(xs);
    if (key > T) {
      const int slot = atomicAdd(&sh.n, 1);
      sh.sv[slot] = xs, sh.si[slot] = start + b;
    } else if (key == T) {
      const int slot = atomicAdd(&sh.eq, 1);
      if (slot < krem) sh.sv[gt + slot] = xs, sh.si[gt + slot] = start + b;
    }
  }
  __syncthreads();
  if (w == 0) {
    warp_sort_desc(sh.sv, sh.si, kTopK);
    cv[l] = sh.sv[l];
    ci[l] = sh.si[l];
  }
}

// a thread per row: its temperature, position, the draft it verifies and that draft's logit; per request its seed
// and whether it is still in its prefill. The request states may live in host memory mapped into the GPU (vLLM's UVA
// tensors), where each read crosses PCIe: they are read here once, into device memory for the other kernels
__global__ void __launch_bounds__(kThreads) vs_rows(
    const float* __restrict__ logits, int V, int R, int n, const int* __restrict__ input_ids,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ logits_indices, const int64_t* __restrict__ row_states,
    const int* __restrict__ row_local, const int64_t* __restrict__ idx_mapping, const int* __restrict__ seq_lens,
    const int* __restrict__ prefill_len, const float* __restrict__ temperature, const int64_t* __restrict__ seeds,
    float* __restrict__ row_temperature, int* __restrict__ drafts, float* __restrict__ draft_logits,
    int64_t* __restrict__ row_positions, int64_t* __restrict__ request_seeds, int* __restrict__ is_prefilling) {
  const int r = blockIdx.x * kThreads + threadIdx.x;
  if (r < R) {
    // the next row's input token when it is the same request's; none for a placeholder of a request still in its
    // prefill (vLLM's gather_draft_sampled)
    int d = -1;
    if (r + 1 < R && row_local[r + 1] != 0) {
      const int64_t token = logits_indices[r + 1];
      const bool is_placeholder = positions[token] - row_local[r + 1] < prefill_len[row_states[r + 1]];
      d = is_placeholder ? -1 : input_ids[token];
    }
    const float t = temperature[row_states[r]];
    drafts[r] = d;
    draft_logits[r] = d >= 0 ? scaled(logits[(int64_t)r * V + d], t) : 0.f;
    row_positions[r] = positions[logits_indices[r]];
    row_temperature[r] = t;
  }
  if (r < n) {
    const int64_t state = idx_mapping[r];
    request_seeds[r] = seeds[state];
    is_prefilling[r] = seq_lens[r] < prefill_len[state];
  }
}

// a row chunk per CTA (kChunk floats; the row's last chunk the rest): the chunk's partials
__global__ void __launch_bounds__(kThreads) vs_chunks(const float* __restrict__ logits, int V,
                                                      const float* __restrict__ row_temperature,
                                                      const int* __restrict__ drafts,
                                                      const float* __restrict__ draft_logits,
                                                      float* __restrict__ part_max,
                                                      float* __restrict__ part_sum, float* __restrict__ part_count,
                                                      int* __restrict__ part_nonfinite, float* __restrict__ cand_val,
                                                      int* __restrict__ cand_idx) {
  const int c = blockIdx.x, r = blockIdx.y;
  const int start = c * kChunk;
  const int n = min(kChunk, V - start);
  const int d = drafts[r];
  const float t = row_temperature[r], xd = draft_logits[r];
  const float* x = logits + (int64_t)r * V;
  const int p = r * gridDim.x + c;
  __shared__ ChunkScratch scratch;
  if (n == kChunk) {
    chunk_partials<true>(x, start, n, t, d, xd, p, part_max, part_sum, part_count, part_nonfinite, cand_val, cand_idx,
                         scratch);
  } else {
    chunk_partials<false>(x, start, n, t, d, xd, p, part_max, part_sum, part_count, part_nonfinite, cand_val,
                          cand_idx, scratch);
  }
}

// a request per CTA: each row's max, log-sum, count of NaN and +inf x and top-kTopK merged from its chunks (a warp per
// row, lane = chunk), the drafts' acceptance (a lane per draft), the recovered or bonus token by inverse transform over
// the first rejected row's chunks, the sampled tokens and counts, and every row's log-probs
__global__ void __launch_bounds__(kThreads) vs_requests(
    const float* __restrict__ logits, int V, int S, int K, const int* __restrict__ cu_num_logits,
    const int64_t* __restrict__ row_positions, const float* __restrict__ draft_logits,
    const float* __restrict__ row_temperature, const int64_t* __restrict__ request_seeds,
    const int* __restrict__ is_prefilling, const float* __restrict__ rates, const float* __restrict__ part_max,
    const float* __restrict__ part_sum, const float* __restrict__ part_count, const int* __restrict__ part_nonfinite,
    const float* __restrict__ cand_val, const int* __restrict__ cand_idx, const int* __restrict__ drafts,
    int64_t* __restrict__ sampled, int sampled_width, int* __restrict__ num_sampled, int* __restrict__ num_rejected,
    int* __restrict__ out_ids, float* __restrict__ out_lp, int64_t* __restrict__ ranks, int* __restrict__ rank_rows,
    float* __restrict__ rank_values, int* __restrict__ nonfinite) {
  extern __shared__ float stage[];  // the rows' candidate lists (values, then ids); then the drawn chunk's weights
  __shared__ float row_m[kMaxRows], row_logs[kMaxRows], row_count[kMaxRows];
  __shared__ float row_top[kMaxRows][kTopK];
  __shared__ float warp_sums[kWarps];
  __shared__ int s_accepted, s_excluded_token, s_chunk, s_token;
  __shared__ float s_target;
  const int i = blockIdx.x;
  const int first_row = cu_num_logits[i], count = cu_num_logits[i + 1] - first_row;
  const float t = row_temperature[first_row];
  const uint64_t seed = (uint64_t)request_seeds[i];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;

  // row w: its partials, lane l holding chunk l
  if (w < count) {
    const int r = first_row + w;
    const bool has_chunk = l < S;
    const float pm = has_chunk ? part_max[r * S + l] : -INFINITY;
    float m = pm;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float z = has_chunk && pm != -INFINITY ? part_sum[r * S + l] * __expf(pm - m) : 0.f;
    float cnt = has_chunk ? part_count[r * S + l] : 0.f;
    int nonfinite_count = has_chunk ? part_nonfinite[r * S + l] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      z += __shfl_xor_sync(0xffffffffu, z, o);
      cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
      nonfinite_count += __shfl_xor_sync(0xffffffffu, nonfinite_count, o);
    }
    if (l == 0) nonfinite[r] = nonfinite_count;
    const float logs = __logf(z);
    float* cvs = stage + w * S * kTopK;
    int* cis = reinterpret_cast<int*>(stage + kMaxRows * S * kTopK) + w * S * kTopK;
    const float4* row_val = reinterpret_cast<const float4*>(cand_val + (int64_t)r * S * kTopK);
    const int4* row_idx = reinterpret_cast<const int4*>(cand_idx + (int64_t)r * S * kTopK);
#pragma unroll 8
    for (int k = l; k < S * kTopK / 4; k += 32) {
      reinterpret_cast<float4*>(cvs)[k] = row_val[k];
      reinterpret_cast<int4*>(cis)[k] = row_idx[k];
    }
    __syncwarp();

    // kTopK rounds: the best of the lanes' list heads
    const float* cv = cvs + l * kTopK;
    const int* ci = cis + l * kTopK;
    int h = 0;
    float hv = has_chunk ? cv[0] : -INFINITY;
    int hi = has_chunk ? ci[0] : INT_MAX;
    for (int k = 0; k < kTopK; ++k) {
      float bv = hv;
      int j = hi;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
        const int j2 = __shfl_xor_sync(0xffffffffu, j, o);
        if (better(v2, j2, bv, j)) bv = v2, j = j2;
      }
      if (l == 0) {
        row_top[w][k] = bv;
        if (k < K) {
          out_ids[(int64_t)r * (K + 1) + 1 + k] = j;
          out_lp[(int64_t)r * (K + 1) + 1 + k] = (bv - m) - logs;
        }
      }
      if (j == hi && bv == hv && has_chunk) {  // this lane's head won: advance
        ++h;
        hv = h < kTopK ? cv[h] : -INFINITY;
        hi = h < kTopK ? ci[h] : INT_MAX;
      }
    }
    if (l == 0) row_m[w] = m, row_logs[w] = logs, row_count[w] = cnt;
  }
  __syncthreads();

  // lane j tests draft j; the first rejected one (or the bonus row) is drawn from
  if (w == 0) {
    bool is_stop = true;
    int d = -1;
    if (l < count - 1) {
      d = drafts[first_row + l];
      if (d >= 0) {
        const float u = uniform(philox(seed, (uint64_t)row_positions[first_row + l]).x);
        const float lp = draft_logits[first_row + l] - row_m[l] - row_logs[l];
        is_stop = rates != nullptr ? !(u < rates[l]) : !(lp > __logf(u));
      }
    }
    const int a = __ffs(__ballot_sync(0xffffffffu, is_stop)) - 1;
    const int excluded_token = __shfl_sync(0xffffffffu, d, a);  // the rejected draft; none at the bonus row

    // the drawn row's chunk masses without the excluded token, and the chunk its draw falls in
    const int r = first_row + a;
    const float m = row_m[a];
    const bool has_chunk = l < S;
    const float pm = has_chunk ? part_max[r * S + l] : -INFINITY;
    float z = has_chunk && pm != -INFINITY ? part_sum[r * S + l] * __expf(pm - m) : 0.f;
    if (excluded_token >= 0 && excluded_token / kChunk == l) {
      z = fmaxf(z - __expf(draft_logits[r] - m), 0.f);
    }
    float incl = z;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, incl, o);
      if (l >= o) incl += y;
    }
    const float row_mass = __shfl_sync(0xffffffffu, incl, 31);
    // an independent word of the drawn row's (seed, pos): the acceptance tests read the first
    const float target = uniform(philox(seed, (uint64_t)row_positions[r]).y) * row_mass;
    const unsigned crossing_lanes = __ballot_sync(0xffffffffu, has_chunk && z > 0.f && incl > target);
    const unsigned mass_lanes = __ballot_sync(0xffffffffu, has_chunk && z > 0.f);
    const int c = crossing_lanes ? __ffs(crossing_lanes) - 1 : (mass_lanes ? 31 - __clz(mass_lanes) : 0);
    const float mass_before = __shfl_sync(0xffffffffu, incl - z, c);
    const float mass = __shfl_sync(0xffffffffu, z, c);
    if (l == 0) {
      s_accepted = a;
      s_excluded_token = excluded_token;
      s_chunk = c;
      s_token = -1;
      // the fraction of the chunk's mass below the draw, carried to the chunk's recomputed weights
      s_target = mass > 0.f ? fminf(fmaxf((target - mass_before) / mass, 0.f), 1.f) : 0.f;
    }
  }
  __syncthreads();

  // the drawn chunk's weights exp(x - m) in index order, the excluded token's 0; a contiguous segment per thread
  const int a = s_accepted, excluded_token = s_excluded_token, c = s_chunk;
  const int r = first_row + a;
  const float m = row_m[a];
  const int base = c * kChunk;
  const int n = min(kChunk, V - base);
  const float* xr = logits + (int64_t)r * V + base;
  float4 v[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    if (b < n) v[k] = *reinterpret_cast<const float4*>(xr + b);
  }
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    if (b < n) {
      const float e[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) stage[b + q] = base + b + q == excluded_token ? 0.f : __expf(scaled(e[q], t) - m);
    }
  }
  __syncthreads();
  const int segment = (n + kThreads - 1) / kThreads;
  const int lo = min(threadIdx.x * segment, n), hi = min(lo + segment, n);
  float sum = 0.f;
  for (int j = lo; j < hi; ++j) sum += stage[j];
  float incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, incl, o);
    if (l >= o) incl += y;
  }
  if (l == 31) warp_sums[w] = incl;
  __syncthreads();
  float offset = 0.f, total = 0.f;
#pragma unroll
  for (int k = 0; k < kWarps; ++k) {
    offset += k < w ? warp_sums[k] : 0.f;
    total += warp_sums[k];
  }
  incl += offset;
  const float target = s_target * total;
  if (sum > 0.f && incl - sum <= target && target < incl) {
    float acc = incl - sum;
    for (int j = lo; j < hi; ++j) {
      acc += stage[j];
      if (acc > target && stage[j] > 0.f) {
        s_token = base + j;
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && s_token < 0) {
    // the draw at the chunk's whole mass (rounding): its last weighted token
    int j = n - 1;
    while (j > 0 && stage[j] == 0.f) --j;
    s_token = base + j;
  }
  __syncthreads();
  const int token = s_token;

  if (threadIdx.x == 0) {
    for (int j = 0; j < a; ++j) sampled[(int64_t)i * sampled_width + j] = drafts[first_row + j];
    sampled[(int64_t)i * sampled_width + a] = token;
    num_sampled[i] = is_prefilling[i] ? 0 : a + 1;
    num_rejected[i] = is_prefilling[i] ? 0 : count - (a + 1);
  }
  // column 0 of every row: its kept token (the accepted drafts, then the drawn one), token 0 past them as in vLLM
  if (threadIdx.x < count) {
    const int j = threadIdx.x, rr = first_row + j;
    const int kept = j < a ? drafts[rr] : (j == a ? token : 0);
    const float xs = j < a ? draft_logits[rr] : scaled(logits[(int64_t)rr * V + kept], t);
    out_ids[(int64_t)rr * (K + 1)] = kept;
    out_lp[(int64_t)rr * (K + 1)] = (xs - row_m[j]) - row_logs[j];
    int64_t rank = j < a ? (int64_t)row_count[j] : 0;
    if (j == a) {
      // decided when the token is above the top list's last entry (nothing outside the list ties it)
      const bool is_decided = xs > row_top[j][kTopK - 1];
      int above_count = 0;
      for (int k = 0; k < kTopK; ++k) above_count += row_top[j][k] >= xs;
      rank = is_decided ? above_count : 0;
      rank_rows[i] = is_decided ? -1 : rr;
      rank_values[i] = xs;
    }
    ranks[rr] = rank;
  }
}

// a row chunk per CTA: the drawn token's rank, counted on x, for the requests whose top list left it undecided
__global__ void __launch_bounds__(kThreads) vs_ranks(const float* __restrict__ logits, int V,
                                                     const float* __restrict__ row_temperature,
                                                     const int* __restrict__ rank_rows,
                                                     const float* __restrict__ rank_values,
                                                     int64_t* __restrict__ ranks) {
  const int c = blockIdx.x, i = blockIdx.y;
  const int r = rank_rows[i];
  if (r < 0) return;
  const int start = c * kChunk;
  const int n = min(kChunk, V - start);
  const float t = row_temperature[r];
  const float xs = rank_values[i];
  const float* x = logits + (int64_t)r * V + start;
  float4 v[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    if (b < n) v[k] = *reinterpret_cast<const float4*>(x + b);
  }
  int cnt = 0;
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    if ((k * kThreads + threadIdx.x) * 4 < n) {
      cnt += (scaled(v[k].x, t) >= xs) + (scaled(v[k].y, t) >= xs) + (scaled(v[k].z, t) >= xs) +
             (scaled(v[k].w, t) >= xs);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  __shared__ int wsum[kWarps];
  if (threadIdx.x % 32 == 0) wsum[threadIdx.x / 32] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int k = 0; k < kWarps; ++k) total += wsum[k];
    atomicAdd(reinterpret_cast<unsigned long long*>(ranks + r), (unsigned long long)total);
  }
}

// logits [R, V] fp32 contiguous (V % 4 == 0), the step's input batch (vLLM's dtypes) and the samplers' request states
// [max_num_reqs]; rates [k] the synthetic conditional acceptance, or empty for the lossless rule; K <= kTopK.
// Returns (sampled int64 [n, width], num_sampled int32 [n], num_rejected int32 [n], ids int32 [R, K+1], logprobs fp32
// [R, K+1], ranks int64 [R], NaN and +inf counts int32 [R]).
std::vector<torch::Tensor> verify_sample(torch::Tensor logits, torch::Tensor input_ids, torch::Tensor positions,
                                         torch::Tensor logits_indices, torch::Tensor row_states,
                                         torch::Tensor row_local, torch::Tensor cu_num_logits,
                                         torch::Tensor idx_mapping, torch::Tensor seq_lens, torch::Tensor prefill_len,
                                         torch::Tensor temperature, torch::Tensor seeds, torch::Tensor rates,
                                         int64_t K, int64_t width) {
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32);
  TORCH_CHECK(input_ids.scalar_type() == torch::kInt32 && positions.scalar_type() == torch::kInt64);
  TORCH_CHECK(logits_indices.scalar_type() == torch::kInt64 && row_states.scalar_type() == torch::kInt64);
  TORCH_CHECK(row_local.scalar_type() == torch::kInt32 && cu_num_logits.scalar_type() == torch::kInt32);
  TORCH_CHECK(idx_mapping.scalar_type() == torch::kInt64 && seq_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(prefill_len.scalar_type() == torch::kInt32 && temperature.scalar_type() == torch::kFloat32);
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64 && rates.scalar_type() == torch::kFloat32);
  const int R = logits.size(0), V = logits.size(1), n = idx_mapping.numel();
  TORCH_CHECK(logits_indices.numel() == R && row_states.numel() == R && row_local.numel() == R);
  TORCH_CHECK(cu_num_logits.numel() == n + 1 && seq_lens.numel() >= n);
  TORCH_CHECK(K >= 1 && K <= kTopK && width >= 1 && width <= kMaxRows && rates.numel() < width);
  const int S = (V + kChunk - 1) / kChunk;
  TORCH_CHECK(V % 4 == 0 && S <= kMaxChunks, "rows of a vocabulary of at most kMaxChunks x kChunk floats");
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  auto f32 = logits.options();
  auto i32 = logits.options().dtype(torch::kInt32);
  auto i64 = logits.options().dtype(torch::kInt64);
  auto sampled = torch::empty({n, width}, i64);
  auto num_sampled = torch::empty({n}, i32);
  auto num_rejected = torch::empty({n}, i32);
  auto ids = torch::empty({R, K + 1}, i32);
  auto lp = torch::empty({R, K + 1}, f32);
  auto ranks = torch::empty({R}, i64);
  auto nonfinite = torch::empty({R}, i32);
  if (R == 0 || n == 0) return {sampled, num_sampled, num_rejected, ids, lp, ranks, nonfinite};

  auto part = torch::empty({4, R * S}, f32);  // max, exp-sum, count of x >= x_d, NaN and +inf count (as int bits)
  auto cand_val = torch::empty({R * S * kTopK}, f32);
  auto cand_idx = torch::empty({R * S * kTopK}, i32);
  auto drafts = torch::empty({R}, i32);
  auto row_temperature = torch::empty({R}, f32);
  auto draft_logits = torch::empty({R}, f32);
  auto row_positions = torch::empty({R}, i64);
  auto request_seeds = torch::empty({n}, i64);
  auto is_prefilling = torch::empty({n}, i32);
  auto rank_rows = torch::empty({n}, i32);
  auto rank_values = torch::empty({n}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  float* pm = part.data_ptr<float>();
  int* part_nonfinite = reinterpret_cast<int*>(pm + 3 * R * S);
  vs_rows<<<(std::max(R, n) + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      logits.data_ptr<float>(), V, R, n, input_ids.data_ptr<int>(), positions.data_ptr<int64_t>(),
      logits_indices.data_ptr<int64_t>(),
      row_states.data_ptr<int64_t>(), row_local.data_ptr<int>(), idx_mapping.data_ptr<int64_t>(),
      seq_lens.data_ptr<int>(), prefill_len.data_ptr<int>(), temperature.data_ptr<float>(), seeds.data_ptr<int64_t>(),
      row_temperature.data_ptr<float>(), drafts.data_ptr<int>(), draft_logits.data_ptr<float>(),
      row_positions.data_ptr<int64_t>(), request_seeds.data_ptr<int64_t>(), is_prefilling.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  vs_chunks<<<dim3(S, R), kThreads, 0, stream>>>(logits.data_ptr<float>(), V, row_temperature.data_ptr<float>(),
                                                 drafts.data_ptr<int>(), draft_logits.data_ptr<float>(), pm,
                                                 pm + R * S, pm + 2 * R * S, part_nonfinite,
                                                 cand_val.data_ptr<float>(), cand_idx.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // the candidate lists of a request's rows, staged; at least the drawn chunk's weights
  const int stage_bytes = std::max<int>(kMaxRows * S * kTopK * 2, kChunk) * sizeof(float);
  static int stage_limit = 0;
  if (stage_bytes > stage_limit) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vs_requests, cudaFuncAttributeMaxDynamicSharedMemorySize, stage_bytes));
    stage_limit = stage_bytes;
  }
  vs_requests<<<n, kThreads, stage_bytes, stream>>>(
      logits.data_ptr<float>(), V, S, (int)K, cu_num_logits.data_ptr<int>(), row_positions.data_ptr<int64_t>(),
      draft_logits.data_ptr<float>(), row_temperature.data_ptr<float>(), request_seeds.data_ptr<int64_t>(),
      is_prefilling.data_ptr<int>(), rates.numel() ? rates.data_ptr<float>() : nullptr, pm, pm + R * S,
      pm + 2 * R * S, part_nonfinite, cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), drafts.data_ptr<int>(),
      sampled.data_ptr<int64_t>(), (int)width, num_sampled.data_ptr<int>(), num_rejected.data_ptr<int>(),
      ids.data_ptr<int>(), lp.data_ptr<float>(), ranks.data_ptr<int64_t>(), rank_rows.data_ptr<int>(),
      rank_values.data_ptr<float>(), nonfinite.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  vs_ranks<<<dim3(S, n), kThreads, 0, stream>>>(logits.data_ptr<float>(), V, row_temperature.data_ptr<float>(),
                                                rank_rows.data_ptr<int>(), rank_values.data_ptr<float>(),
                                                ranks.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sampled, num_sampled, num_rejected, ids, lp, ranks, nonfinite};
}

}  // namespace verify_sampler

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "verify_sample(Tensor logits, Tensor input_ids, Tensor positions, Tensor logits_indices, Tensor row_states, "
      "Tensor row_local, Tensor cu_num_logits, Tensor idx_mapping, Tensor seq_lens, Tensor prefill_len, "
      "Tensor temperature, Tensor seeds, Tensor rates, int K, int width) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("verify_sample", &verify_sampler::verify_sample); }
