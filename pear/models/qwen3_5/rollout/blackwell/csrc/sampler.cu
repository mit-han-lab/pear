// vLLM's sampling of random rows with processed log-probs (temperature only): x = logits / t, the sampled token
// argmax_j x_j - log q_j (the exponential race on torch's noise q), lp = log_softmax(x), the top-K lp sorted
// descending with their ids, and the sampled token's rank #{j : lp_j >= lp_sampled}.
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <climits>

namespace {

constexpr int kTopK = 32;      // candidates per chunk: >= the requested log-probs
constexpr int k1Threads = 256;
constexpr int k1Vec = 8;       // float4 per thread: chunks up to 8192 floats

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

template <int NT>
__device__ __forceinline__ void block_best(float& v, int& i, float* redv, int* redi) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, v, o);
    const int j = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(w, j, v, i)) v = w, i = j;
  }
  const int wp = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) redv[wp] = v, redi[wp] = i;
  __syncthreads();
  v = l < NT / 32 ? redv[l] : -INFINITY;
  i = l < NT / 32 ? redi[l] : INT_MAX;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, v, o);
    const int j = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(w, j, v, i)) v = w, i = j;
  }
}

// x / t, exact (IEEE division even under --use_fast_math); t == 1 skips it
__device__ __forceinline__ float scaled(float x, float t) { return t == 1.f ? x : __fdiv_rn(x, t); }

constexpr int kCap = 512;  // candidates a chunk sorts for its exact top-kTopK (more: the radix-select path)

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

// a row chunk per CTA, one streaming pass: the chunk's max / exp-sum (online) and best race score, then its top-kTopK
// from the candidates >= theta (the kTopK-th largest of the threads' maxima), sorted; a radix select when ties leave
// more than kCap candidates
__global__ void __launch_bounds__(k1Threads) k1_chunks(
    const float* __restrict__ logits, const float* __restrict__ q, const float* __restrict__ temp, int V, int chunk,
    float* __restrict__ part_max, float* __restrict__ part_sum, float* __restrict__ part_best,
    int* __restrict__ part_best_idx, float* __restrict__ cand_val, int* __restrict__ cand_idx) {
  const int c = blockIdx.x, r = blockIdx.y, S = gridDim.x;
  const int start = c * chunk;
  const int n = min(chunk, V - start);
  const float t = temp[r];
  const float* x = logits + (int64_t)r * V + start;
  const float* qq = q + (int64_t)r * V + start;
  constexpr int NW = k1Threads / 32;
  __shared__ float red[32];
  __shared__ int redi[32];
  __shared__ float wmax[NW * 32];
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_prefix, s_krem;
  __shared__ int s_n, s_eq;
  __shared__ float s_theta;
  __shared__ float sv[kCap];
  __shared__ int si[kCap];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;

  // pass 1: online max / exp-sum, the race's best score, this thread's max
  float m = -INFINITY, s = 0.f, best = -INFINITY;
  int best_i = INT_MAX;
  for (int b = threadIdx.x * 4; b < n; b += k1Threads * 4) {
    float xv[4], qv[4];
    if (b + 3 < n) {
      const float4 a = *reinterpret_cast<const float4*>(x + b);
      const float4 e = *reinterpret_cast<const float4*>(qq + b);
      xv[0] = a.x, xv[1] = a.y, xv[2] = a.z, xv[3] = a.w;
      qv[0] = e.x, qv[1] = e.y, qv[2] = e.z, qv[3] = e.w;
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xv[e] = b + e < n ? x[b + e] : -INFINITY;
        qv[e] = b + e < n ? qq[b + e] : 1.f;
      }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (b + e >= n) continue;
      const float xs = scaled(xv[e], t);
      if (xs > m) {
        s = (m == -INFINITY ? 0.f : s * expf(m - xs)) + 1.f;
        m = xs;
      } else if (xs != -INFINITY) {
        s += expf(xs - m);
      }
      const float score = xs - __logf(qv[e]);
      if (better(score, start + b + e, best, best_i)) best = score, best_i = start + b + e;
    }
  }
  const float tmax = m;
  const float M = block_max<k1Threads>(m, red);
  s = m == -INFINITY ? 0.f : s * expf(m - M);
  s = block_sum<k1Threads>(s, red);
  block_best<k1Threads>(best, best_i, red, redi);
  const int p = r * S + c;
  if (threadIdx.x == 0) {
    part_max[p] = M;
    part_sum[p] = s;
    part_best[p] = best;
    part_best_idx[p] = best_i;
    s_n = 0;
    s_eq = 0;
  }

  // theta = the kTopK-th largest of the threads' maxima (kTopK elements are >= theta, so the top kTopK all are):
  // every warp sorts its lanes' maxima, warp 0 merges the NW sorted lists
  wmax[w * 32 + l] = warp_sort_lanes_desc(tmax);
  __syncthreads();
  if (w == 0) {
    int h = 0;  // lane l < NW: head of warp l's list
    float hv = l < NW ? wmax[l * 32] : -INFINITY;
    float th = -INFINITY;
    for (int k = 0; k < kTopK; ++k) {
      float v = hv;
      int j = l;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int j2 = __shfl_xor_sync(0xffffffffu, j, o);
        if (v2 > v || (v2 == v && j2 < j)) v = v2, j = j2;
      }
      th = v;
      if (j == l) {
        ++h;
        hv = h < 32 ? wmax[l * 32 + h] : -INFINITY;
      }
    }
    if (l == 0) s_theta = th;
  }
  __syncthreads();
  const float theta = s_theta;
  // pass 2 (the chunk again, from L2): the candidates >= theta
  for (int b = threadIdx.x * 4; b < n; b += k1Threads * 4) {
    float xv[4];
    if (b + 3 < n) {
      const float4 a = *reinterpret_cast<const float4*>(x + b);
      xv[0] = a.x, xv[1] = a.y, xv[2] = a.z, xv[3] = a.w;
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) xv[e] = b + e < n ? x[b + e] : -INFINITY;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float xs = b + e < n ? scaled(xv[e], t) : -INFINITY;
      if (b + e < n && xs >= theta) {
        const int slot = atomicAdd(&s_n, 1);
        if (slot < kCap) sv[slot] = xs, si[slot] = start + b + e;
      }
    }
  }
  __syncthreads();
  const int nc = s_n;
  float* cv = cand_val + (int64_t)p * kTopK;
  int* ci = cand_idx + (int64_t)p * kTopK;
  if (nc <= kCap) {
    if (w == 0) {
      int n2 = kTopK;
      while (n2 < nc) n2 <<= 1;
      for (int k = nc + l; k < n2; k += 32) sv[k] = -INFINITY, si[k] = INT_MAX;
      __syncwarp();
      warp_sort_desc(sv, si, n2);
      cv[l] = sv[l];
      ci[l] = si[l];
    }
    return;
  }
  // many ties above theta: radix select of the kTopK-th largest key over the chunk (8 bits per pass)
  if (threadIdx.x == 0) s_prefix = 0, s_krem = kTopK;
  uint32_t mask = 0;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    for (int b = threadIdx.x; b < n; b += k1Threads) {
      const uint32_t key = fkey(scaled(x[b], t));
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      uint32_t cnt[8], tot = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) cnt[b] = hist[255 - 8 * l - b], tot += cnt[b];
      uint32_t incl = tot;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
        if (l >= o) incl += y;
      }
      const uint32_t krem = s_krem;
      const uint32_t excl = incl - tot;
      if (excl < krem && incl >= krem) {
        uint32_t acc = excl;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          if (acc + cnt[b] >= krem) {
            s_prefix = prefix | ((uint32_t)(255 - 8 * l - b) << shift);
            s_krem = krem - acc;
            break;
          }
          acc += cnt[b];
        }
      }
    }
    mask |= 255u << shift;
    __syncthreads();
  }
  const uint32_t T = s_prefix;
  const int krem = (int)s_krem;
  const int gt = kTopK - krem;
  if (threadIdx.x == 0) s_n = 0;
  __syncthreads();
  for (int b = threadIdx.x; b < n; b += k1Threads) {
    const float xs = scaled(x[b], t);
    const uint32_t key = fkey(xs);
    if (key > T) {
      const int slot = atomicAdd(&s_n, 1);
      sv[slot] = xs, si[slot] = start + b;
    } else if (key == T) {
      const int slot = atomicAdd(&s_eq, 1);
      if (slot < krem) sv[gt + slot] = xs, si[gt + slot] = start + b;
    }
  }
  __syncthreads();
  if (w == 0) {
    warp_sort_desc(sv, si, kTopK);
    cv[l] = sv[l];
    ci[l] = si[l];
  }
}

// one warp per row: the row's max / sum and sampled token from the chunks' partials, the top-K by merging the chunks'
// sorted candidate lists (lane = chunk), and the sampled token's rank when the top kTopK decide it (else k3 counts)
__global__ void __launch_bounds__(32) k2_rows(
    const float* __restrict__ logits, const float* __restrict__ temp, int V, int S, int K,
    const float* __restrict__ part_max, const float* __restrict__ part_sum, const float* __restrict__ part_best,
    const int* __restrict__ part_best_idx, const float* __restrict__ cand_val, const int* __restrict__ cand_idx,
    int* __restrict__ sampled32, int* __restrict__ out_ids,
    float* __restrict__ out_lp, float* __restrict__ row_stats, int64_t* __restrict__ ranks) {
  const int r = blockIdx.x, l = threadIdx.x;
  const bool has = l < S;
  const float pm = has ? part_max[r * S + l] : -INFINITY;
  float m = pm;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float s = has && pm != -INFINITY ? part_sum[r * S + l] * expf(pm - m) : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  float best = has ? part_best[r * S + l] : -INFINITY;
  int bi = has ? part_best_idx[r * S + l] : INT_MAX;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor_sync(0xffffffffu, best, o);
    const int j = __shfl_xor_sync(0xffffffffu, bi, o);
    if (better(w, j, best, bi)) best = w, bi = j;
  }
  const float logs = logf(s);
  const float t = temp[r];
  const float lps = (scaled(logits[(int64_t)r * V + bi], t) - m) - logs;
  // merge (lane = chunk), the lists staged in shared memory
  __shared__ float cvs[32 * kTopK];
  __shared__ int cis[32 * kTopK];
  for (int k = l; k < S * kTopK; k += 32) {
    cvs[k] = cand_val[(int64_t)r * S * kTopK + k];
    cis[k] = cand_idx[(int64_t)r * S * kTopK + k];
  }
  __syncwarp();
  const float* cv = cvs + l * kTopK;
  const int* ci = cis + l * kTopK;
  int h = 0;
  float hv = has ? cv[0] : -INFINITY;
  int hi = has ? ci[0] : INT_MAX;
  int above = 0;   // top entries with lp >= lps
  float last = 0.f;
  for (int k = 0; k < kTopK; ++k) {
    float w = hv;
    int j = hi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float w2 = __shfl_xor_sync(0xffffffffu, w, o);
      const int j2 = __shfl_xor_sync(0xffffffffu, j, o);
      if (better(w2, j2, w, j)) w = w2, j = j2;
    }
    const float lp = (w - m) - logs;
    if (k < K && l == 0) {
      out_ids[(int64_t)r * (K + 1) + 1 + k] = j;
      out_lp[(int64_t)r * (K + 1) + 1 + k] = lp;
    }
    above += lp >= lps;
    last = lp;
    if (j == hi && w == hv && has) {  // this lane's head won: advance
      ++h;
      hv = h < kTopK ? cv[h] : -INFINITY;
      hi = h < kTopK ? ci[h] : INT_MAX;
    }
  }
  if (l == 0) {
    sampled32[r] = bi;
    out_ids[(int64_t)r * (K + 1)] = bi;
    out_lp[(int64_t)r * (K + 1)] = lps;
    row_stats[4 * r + 0] = m;
    row_stats[4 * r + 1] = logs;
    row_stats[4 * r + 2] = lps;
    // decided when the sampled token is strictly above the kTopK-th entry (no element outside the list ties it)
    const bool decided = lps > last;
    row_stats[4 * r + 3] = decided ? 1.f : 0.f;
    ranks[r] = decided ? above : 0;
  }
}

// a row chunk per CTA: the sampled token's rank, counted on the log-probs, for the rows k2 left undecided
__global__ void __launch_bounds__(k1Threads) k3_rank(const float* __restrict__ logits, const float* __restrict__ temp,
                                                     int V, int chunk, const float* __restrict__ row_stats,
                                                     int64_t* __restrict__ ranks) {
  const int c = blockIdx.x, r = blockIdx.y;
  if (row_stats[4 * r + 3] != 0.f) return;
  const int start = c * chunk;
  const int n = min(chunk, V - start);
  const float t = temp[r];
  const float m = row_stats[4 * r], logs = row_stats[4 * r + 1], lps = row_stats[4 * r + 2];
  const float* x = logits + (int64_t)r * V + start;
  int cnt = 0;
  for (int b = threadIdx.x * 4; b < n; b += k1Threads * 4) {
    if (b + 3 < n) {
      const float4 a = *reinterpret_cast<const float4*>(x + b);
      cnt += ((scaled(a.x, t) - m) - logs >= lps) + ((scaled(a.y, t) - m) - logs >= lps) +
             ((scaled(a.z, t) - m) - logs >= lps) + ((scaled(a.w, t) - m) - logs >= lps);
    } else {
      for (int e = 0; e < 4 && b + e < n; ++e) cnt += ((scaled(x[b + e], t) - m) - logs >= lps);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  __shared__ int wsum[k1Threads / 32];
  if (threadIdx.x % 32 == 0) wsum[threadIdx.x / 32] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < k1Threads / 32; ++w) total += wsum[w];
    atomicAdd(reinterpret_cast<unsigned long long*>(ranks + r), (unsigned long long)total);
  }
}

// logits [R, V] fp32 contiguous, q [R, V] fp32 (exponential noise), temperature [>= R] fp32; K <= kTopK log-probs.
// Returns (sampled int32 [R], ids int32 [R, K+1], logprobs fp32 [R, K+1], ranks int64 [R]).
std::vector<torch::Tensor> fused_sample(torch::Tensor logits, torch::Tensor q, torch::Tensor temperature, int64_t K,
                                        int64_t chunks) {
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32);
  TORCH_CHECK(q.sizes() == logits.sizes() && q.is_contiguous() && q.scalar_type() == torch::kFloat32);
  TORCH_CHECK(temperature.scalar_type() == torch::kFloat32 && temperature.is_contiguous());
  TORCH_CHECK(K >= 1 && K <= kTopK && chunks >= 1 && chunks <= 32);
  const int R = logits.size(0), V = logits.size(1);
  TORCH_CHECK(V % 4 == 0 && temperature.numel() >= R);
  int chunk = (V + chunks - 1) / chunks;
  chunk = (chunk + 3) / 4 * 4;
  TORCH_CHECK(chunk <= k1Threads * k1Vec * 4, "chunk too large");
  const int S = (V + chunk - 1) / chunk;
  TORCH_CHECK(S <= 32, "at most 32 chunks per row");
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  auto f32 = logits.options();
  auto i32 = logits.options().dtype(torch::kInt32);
  auto i64 = logits.options().dtype(torch::kInt64);
  auto part = torch::empty({4, R * S}, f32);  // max, sum, best, best index (as int bits)
  auto cand_val = torch::empty({R * S * kTopK}, f32);
  auto cand_idx = torch::empty({R * S * kTopK}, i32);
  auto sampled32 = torch::empty({R}, i32);
  auto ids = torch::empty({R, K + 1}, i32);
  auto lp = torch::empty({R, K + 1}, f32);
  auto stats = torch::empty({R, 4}, f32);
  auto ranks = torch::empty({R}, i64);
  if (R == 0) return {sampled32, ids, lp, ranks};
  auto stream = at::cuda::getCurrentCUDAStream();
  float* pm = part.data_ptr<float>();
  k1_chunks<<<dim3(S, R), k1Threads, 0, stream>>>(
      logits.data_ptr<float>(), q.data_ptr<float>(), temperature.data_ptr<float>(), V, chunk, pm, pm + R * S,
      pm + 2 * R * S, reinterpret_cast<int*>(pm + 3 * R * S), cand_val.data_ptr<float>(), cand_idx.data_ptr<int>());
  k2_rows<<<R, 32, 0, stream>>>(
      logits.data_ptr<float>(), temperature.data_ptr<float>(), V, S, (int)K, pm, pm + R * S, pm + 2 * R * S,
      reinterpret_cast<int*>(pm + 3 * R * S), cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(),
      sampled32.data_ptr<int>(), ids.data_ptr<int>(), lp.data_ptr<float>(),
      stats.data_ptr<float>(), ranks.data_ptr<int64_t>());
  k3_rank<<<dim3(S, R), k1Threads, 0, stream>>>(logits.data_ptr<float>(), temperature.data_ptr<float>(), V, chunk,
                                                 stats.data_ptr<float>(), ranks.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sampled32, ids, lp, ranks};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("fused_sample(Tensor logits, Tensor q, Tensor temperature, int K, int chunks) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("fused_sample", &fused_sample); }
