// vLLM's rejection sampling of greedy drafts at a temperature alone (x = logits / t, lp = log_softmax(x)): the lossless
// rule lp(d) > log u or the synthetic u < rate on vLLM's tl.rand(seed, pos), the recovered or bonus token by inverse
// transform on another Philox word of (seed, pos), the top-K lp of every row and the kept tokens' ranks; each row's
// count of NaN and +inf x, corrupted logits the rollout raises on.
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sampler_common.cuh"
#include "sync.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVec = 8;              // float4 per thread: a chunk of up to 8192 floats held in registers
constexpr int kChunk = kThreads * kVec * 4;  // floats of a row per chunk CTA
constexpr int kMaxChunks = 32;       // a row's chunks: a lane each, and a segment maximum each in a row CTA's threads
constexpr int kSegCap = 512;         // segments a row CTA ranks for its kTopK-th largest segment maximum
constexpr int kElemCap = 1024;       // elements of the selected segments a row CTA sorts (more: the radix path)
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kMaxRows = kWarps;     // a request's rows (its last token and its drafts): a lane each
constexpr int kPhiloxRounds = 10;    // Triton's tl.rand
constexpr float kRandScale = 4.6566127342e-10f;  // Triton's uint_to_uniform_float scale for 32-bit words
constexpr float kRandMin = 4.6566127342e-10f;    // the smallest u vLLM's rejection sampler draws (_TL_RAND_MIN)
constexpr int kOutcomeWords = 8;     // a row's outcome: 32-bit words, two 16-byte stores

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

// the float of an fkey
__device__ __forceinline__ float key_float(uint32_t key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
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

// a row's outcome, left by its CTA for the request's last one: the draft it verifies (-1: none), whether it stops
// the request (its draft rejected, or none), the token it draws then, the kept token's rank (the accepted draft's or
// the drawn token's; -1: the drawn token's, left to vs_ranks), the log-probs of the draft, the drawn token and token 0,
// and the drawn token's x
struct alignas(16) RowOutcome {
  int draft, is_stop, token, rank;
  float lp_draft, lp_token, lp_zero, x_token;
};
static_assert(sizeof(RowOutcome) == kOutcomeWords * 4);

// the row kernel's shared memory
struct RowScratch {
  float wmax[kThreads];
  float cv[kElemCap];   // candidates: segment maxima (value, segment), then elements (value, index)
  int ci[kElemCap];
  int sel[kSegCap];     // the segments reaching theta1
  float top_v[kTopK];
  int top_i[kTopK];
  float red[kWarps];
  float warp_sums[kWarps];
  int wcount[kWarps];
  uint32_t hist[256];
  uint32_t prefix, krem, theta_key;
  int nc, ns, chunk, token, is_last;
  float frac;
};

// a thread's NaN and +inf x of a chunk at x (its kVec float4s before len), read again: vs_chunks's cold path, out of
// line so that the hot loop keeps its registers
__device__ __noinline__ int thread_nonfinite(const float* __restrict__ x, int len, float t) {
  int nonfinite_count = 0;
#pragma unroll 1
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    if (b < len) {
      const float4 e = *reinterpret_cast<const float4*>(x + b);
      nonfinite_count += !(scaled(e.x, t) < INFINITY) + !(scaled(e.y, t) < INFINITY) + !(scaled(e.z, t) < INFINITY) +
                         !(scaled(e.w, t) < INFINITY);
    }
  }
  return nonfinite_count;
}

// a row chunk per CTA (kChunk floats; the row's last chunk the rest): its max and exp-sum of x, its count of
// x >= x_d (the draft the row verifies, for the draft's rank), its count of NaN and +inf x and the maximum of each
// thread's segment (its kVec float4s); the first CTA clears the requests' arrival counts for vs_rows
__global__ void __launch_bounds__(kThreads) vs_chunks(
    const float* __restrict__ logits, int R, int V, const int* __restrict__ input_ids,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ logits_indices,
    const int64_t* __restrict__ row_states, const int* __restrict__ row_local, const int* __restrict__ prefill_len,
    const float* __restrict__ temperature, float* __restrict__ part_max, float* __restrict__ part_sum,
    float* __restrict__ part_count, int* __restrict__ part_nonfinite, float* __restrict__ seg_max,
    int* __restrict__ arrivals, int n) {
  const int c = blockIdx.x, r = blockIdx.y, S = gridDim.x;
  const int start = c * kChunk;
  const int len = min(kChunk, V - start);
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ int nonfinite;  // the chunk's NaN and +inf x
  if (threadIdx.x == 0) nonfinite = 0;  // before block_max's barriers
  pdl_wait();
  if (c == 0 && r == 0) {
    for (int i = threadIdx.x; i < n; i += kThreads) arrivals[i] = 0;
  }

  // the draft this row verifies: the next row's input token when it is the same request's; none for a placeholder
  // of a request still in its prefill (vLLM's gather_draft_sampled)
  int d = -1;
  if (r + 1 < R && row_local[r + 1] != 0) {
    const int64_t token = logits_indices[r + 1];
    const bool is_placeholder = positions[token] - row_local[r + 1] < prefill_len[row_states[r + 1]];
    d = is_placeholder ? -1 : input_ids[token];
  }

  // the chunk's float4s scaled by the temperature, -inf past its end
  const float t = temperature[row_states[r]];
  const float* x = logits + (int64_t)r * V + start;
  float4 v[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const int b = (k * kThreads + threadIdx.x) * 4;
    v[k] = b < len ? *reinterpret_cast<const float4*>(x + b)
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
  seg_max[((int64_t)r * S + c) * kThreads + threadIdx.x] = tmax;
  float count = 0.f;
  if (d >= 0) {
    const float xd = scaled(logits[(int64_t)r * V + d], t);
    int above_count = 0;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      if ((k * kThreads + threadIdx.x) * 4 < len) {
        above_count += (v[k].x >= xd) + (v[k].y >= xd) + (v[k].z >= xd) + (v[k].w >= xd);
      }
    }
    count = above_count;
  }

  // exp(x - tmax) as one FFMA and one exp2 each, then at the chunk's max
  float s = 0.f;
  if (tmax != -INFINITY) {
    const float shift = -tmax * kLog2e;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      s += exp2f(fmaf(v[k].x, kLog2e, shift)) + exp2f(fmaf(v[k].y, kLog2e, shift)) +
           exp2f(fmaf(v[k].z, kLog2e, shift)) + exp2f(fmaf(v[k].w, kLog2e, shift));
    }
  }
  // the thread's NaN and +inf x, read again where its sum is NaN (a NaN or +inf x) or its max -inf (a NaN among -inf
  // x leaves the sum 0), in a call of its own
  int nonfinite_count = 0;
  if (isnan(s) || tmax == -INFINITY) nonfinite_count = thread_nonfinite(x, len, t);
  __shared__ float red[kWarps];
  __shared__ float sums[2][kWarps];
  const float M = block_max<kThreads>(tmax, red);
  if (nonfinite_count != 0) atomicAdd(&nonfinite, nonfinite_count);
  s = tmax == -INFINITY ? 0.f : s * exp2f((tmax - M) * kLog2e);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
    count += __shfl_xor_sync(0xffffffffu, count, o);
  }
  if (l == 0) sums[0][w] = s, sums[1][w] = count;
  __syncthreads();
  if (threadIdx.x == 0) {
    float chunk_sum = 0.f, chunk_count = 0.f;
#pragma unroll
    for (int q = 0; q < kWarps; ++q) chunk_sum += sums[0][q], chunk_count += sums[1][q];
    part_max[r * S + c] = M;
    part_sum[r * S + c] = chunk_sum;
    part_count[r * S + c] = chunk_count;
    part_nonfinite[r * S + c] = nonfinite;
  }
}

// the row's top kTopK in (value, index) order by a radix select of the kTopK-th largest key (8 bits a pass), for the
// rows whose ties defeat the segment selection
__device__ void row_top_radix(const float* __restrict__ x, int V, float t, RowScratch& sh) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (threadIdx.x == 0) sh.prefix = 0, sh.krem = kTopK;
  uint32_t mask = 0;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    sh.hist[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = sh.prefix;
    for (int b = threadIdx.x; b < V; b += kThreads) {
      const uint32_t key = fkey(scaled(x[b], t));
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
  const int krem = (int)sh.krem, gt = kTopK - krem;

  // the keys above T (gt of them) in any order, then the krem first keys equal to T in index order: a thread's
  // contiguous span, the spans ordered by a block scan of their counts
  const int span = (V + kThreads - 1) / kThreads;
  const int lo = min(threadIdx.x * span, V), hi = min(lo + span, V);
  int equal_count = 0;
  for (int b = lo; b < hi; ++b) equal_count += fkey(scaled(x[b], t)) == T;
  int incl = equal_count;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (l >= o) incl += y;
  }
  if (l == 31) sh.wcount[w] = incl;
  if (threadIdx.x == 0) sh.nc = 0;
  __syncthreads();
  int slot = incl - equal_count;
  for (int q = 0; q < w; ++q) slot += sh.wcount[q];
  for (int b = lo; b < hi; ++b) {
    const float xs = scaled(x[b], t);
    const uint32_t key = fkey(xs);
    if (key > T) {
      const int s2 = atomicAdd(&sh.nc, 1);
      sh.cv[s2] = xs, sh.ci[s2] = b;
    } else if (key == T) {
      if (slot < krem) sh.cv[gt + slot] = xs, sh.ci[gt + slot] = b;
      ++slot;
    }
  }
  __syncthreads();
  if (w == 0) {
    warp_sort_desc(sh.cv, sh.ci, kTopK);
    sh.top_v[l] = sh.cv[l];
    sh.top_i[l] = sh.ci[l];
  }
}

// a verify row per CTA (grid: the request's rows x the requests): the row's count of NaN and +inf x, its top-K
// log-probs from the segments whose maximum reaches the kTopK-th largest segment maximum, its draft's acceptance, the
// token it draws if it stops the request, the kept token's rank; the request's last row CTA to finish writes the
// request's sampled tokens, counts and every row's kept token
__global__ void __launch_bounds__(kThreads) vs_rows(
    const float* __restrict__ logits, int R, int V, int S, int K, const int* __restrict__ input_ids,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ logits_indices,
    const int64_t* __restrict__ row_states, const int* __restrict__ row_local, const int* __restrict__ cu_num_logits,
    const int64_t* __restrict__ idx_mapping, const int* __restrict__ seq_lens, const int* __restrict__ prefill_len,
    const float* __restrict__ temperature, const int64_t* __restrict__ seeds, const float* __restrict__ rates,
    const float* __restrict__ part_max, const float* __restrict__ part_sum, const float* __restrict__ part_count,
    const int* __restrict__ part_nonfinite, const float* __restrict__ seg_max, RowOutcome* __restrict__ outcomes,
    int* __restrict__ arrivals, int64_t* __restrict__ sampled, int sampled_width, int* __restrict__ num_sampled,
    int* __restrict__ num_rejected, int* __restrict__ out_ids, float* __restrict__ out_lp, int64_t* __restrict__ ranks,
    int* __restrict__ rank_rows, float* __restrict__ rank_values, int* __restrict__ nonfinite) {
  const int j = blockIdx.x, i = blockIdx.y;
  const int first_row = cu_num_logits[i], count = cu_num_logits[i + 1] - first_row;
  if (j >= count) return;
  const int r = first_row + j;
  const int64_t state = idx_mapping[i];
  const float t = temperature[state];
  const uint64_t seed = (uint64_t)seeds[state];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  const float* x = logits + (int64_t)r * V;
  __shared__ RowScratch sh;

  // the draft this row verifies: the next row's input token when it is the same request's; none for a placeholder
  // of a request still in its prefill (vLLM's gather_draft_sampled)
  int d = -1;
  if (r + 1 < R && row_local[r + 1] != 0) {
    const int64_t token = logits_indices[r + 1];
    const bool is_placeholder = positions[token] - row_local[r + 1] < prefill_len[row_states[r + 1]];
    d = is_placeholder ? -1 : input_ids[token];
  }

  const uint4 word = philox(seed, (uint64_t)positions[logits_indices[r]]);
  pdl_wait();

  // the row's max and log-sum from its chunks' partials (lane = chunk), in every warp; its NaN and +inf x in warp 0
  const bool has_chunk = l < S;
  const float pm = has_chunk ? part_max[r * S + l] : -INFINITY;
  const int chunk_nonfinite = has_chunk ? part_nonfinite[r * S + l] : 0;
  float m = pm;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float chunk_mass = has_chunk && pm != -INFINITY ? part_sum[r * S + l] * __expf(pm - m) : 0.f;
  float z = chunk_mass;
  float draft_count = has_chunk ? part_count[r * S + l] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    z += __shfl_xor_sync(0xffffffffu, z, o);
    draft_count += __shfl_xor_sync(0xffffffffu, draft_count, o);
  }
  const float logs = __logf(z);
  if (w == 0) {
    const int nonfinite_count = __reduce_add_sync(0xffffffffu, chunk_nonfinite);
    if (l == 0) nonfinite[r] = nonfinite_count;
  }

  // each thread's segment maxima (its segment of every chunk) and their maximum
  float tm[kMaxChunks];
  float g = -INFINITY;
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    tm[c] = c < S ? seg_max[((int64_t)r * S + c) * kThreads + threadIdx.x] : -INFINITY;
    g = fmaxf(g, tm[c]);
  }

  // theta0: the kTopK-th largest of the threads' maxima (the warps' lane values sorted, then merged pairwise)
  sh.wmax[w * 32 + l] = warp_sort_lanes_desc(g);
  if (threadIdx.x == 0) sh.nc = 0, sh.ns = 0, sh.theta_key = 0xffffffffu;
  __syncthreads();
  for (int lists = kWarps; lists > 1; lists >>= 1) {
    const bool is_merging = w < lists / 2;
    float merged_max = 0.f;
    if (is_merging) merged_max = warp_merge_desc(fmaxf(sh.wmax[2 * w * 32 + l], sh.wmax[(2 * w + 1) * 32 + 31 - l]));
    __syncthreads();
    if (is_merging) sh.wmax[w * 32 + l] = merged_max;
    __syncthreads();
  }
  const float theta0 = sh.wmax[kTopK - 1];

  // the segments whose maximum reaches theta0: at least kTopK (the threads' maxima above it)
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    if (tm[c] >= theta0) {
      const int slot = atomicAdd(&sh.nc, 1);
      if (slot < kSegCap) sh.cv[slot] = tm[c], sh.ci[slot] = c * kThreads + threadIdx.x;
    }
  }
  __syncthreads();
  const int nc = sh.nc;

  // theta1, the kTopK-th largest segment maximum: the segments reaching it hold every element of the row that does,
  // at least kTopK
  if (nc <= kSegCap) {
    for (int e = threadIdx.x; e < nc; e += kThreads) {
      const float v = sh.cv[e];
      int above_count = 0;
      for (int q = 0; q < nc; ++q) above_count += sh.cv[q] > v;
      if (above_count < kTopK) {
        sh.sel[atomicAdd(&sh.ns, 1)] = sh.ci[e];
        atomicMin(&sh.theta_key, fkey(v));
      }
    }
  }
  __syncthreads();
  const int ns = sh.ns;
  const float theta1 = key_float(sh.theta_key);
  if (threadIdx.x == 0) sh.nc = 0;
  __syncthreads();
  if (nc <= kSegCap) {
    for (int e = threadIdx.x; e < ns * kVec; e += kThreads) {
      const int seg = sh.sel[e / kVec], k = e % kVec;
      const int off = (seg / kThreads) * kChunk + (k * kThreads + seg % kThreads) * 4;
      if (off < V) {
        const float4 q4 = *reinterpret_cast<const float4*>(x + off);
        const float e4[4] = {scaled(q4.x, t), scaled(q4.y, t), scaled(q4.z, t), scaled(q4.w, t)};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (e4[q] >= theta1) {
            const int slot = atomicAdd(&sh.nc, 1);
            if (slot < kElemCap) sh.cv[slot] = e4[q], sh.ci[slot] = off + q;
          }
        }
      }
    }
  }
  __syncthreads();
  const int ne = sh.nc;

  // the row's top kTopK: the candidates sorted (a thread per candidate counts those before it), or the radix path
  if (nc > kSegCap || ne > kElemCap) {
    row_top_radix(x, V, t, sh);
  } else if (ne <= kThreads) {
    if (threadIdx.x < ne) {
      const float mv = sh.cv[threadIdx.x];
      const int mi = sh.ci[threadIdx.x];
      int position = 0;
      for (int q = 0; q < ne; ++q) position += better(sh.cv[q], sh.ci[q], mv, mi);
      if (position < kTopK) sh.top_v[position] = mv, sh.top_i[position] = mi;
    }
  } else if (w == 0) {
    int n2 = kTopK;
    while (n2 < ne) n2 <<= 1;
    for (int q = ne + l; q < n2; q += 32) sh.cv[q] = -INFINITY, sh.ci[q] = INT_MAX;
    __syncwarp();
    warp_sort_desc(sh.cv, sh.ci, n2);
    sh.top_v[l] = sh.cv[l];
    sh.top_i[l] = sh.ci[l];
  }
  __syncthreads();
  if (threadIdx.x < K) {
    out_ids[(int64_t)r * (K + 1) + 1 + threadIdx.x] = sh.top_i[threadIdx.x];
    out_lp[(int64_t)r * (K + 1) + 1 + threadIdx.x] = (sh.top_v[threadIdx.x] - m) - logs;
  }

  // the draft's acceptance by vLLM's u = tl.rand(seed, pos): the lossless lp(d) > log u or the synthetic u < rate
  float xd = 0.f, lp_draft = 0.f;
  bool is_stop = true;
  if (d >= 0) {
    xd = scaled(x[d], t);
    lp_draft = xd - m - logs;
    const float u = uniform(word.x);
    is_stop = rates != nullptr ? !(u < rates[j]) : !(lp_draft > __logf(u));
  }

  // the token this row draws if it stops the request (an independent word of its (seed, pos)): inverse transform
  // over the chunks' masses without the rejected draft, then over the drawn chunk's weights in index order
  int token = 0;
  float x_token = 0.f;
  if (is_stop) {
    if (w == 0) {
      float mass_l = chunk_mass;
      if (d >= 0 && d / kChunk == l) mass_l = fmaxf(mass_l - __expf(xd - m), 0.f);
      float incl = mass_l;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, incl, o);
        if (l >= o) incl += y;
      }
      const float target = uniform(word.y) * __shfl_sync(0xffffffffu, incl, 31);
      const unsigned crossing_lanes = __ballot_sync(0xffffffffu, has_chunk && mass_l > 0.f && incl > target);
      const unsigned mass_lanes = __ballot_sync(0xffffffffu, has_chunk && mass_l > 0.f);
      const int c = crossing_lanes ? __ffs(crossing_lanes) - 1 : (mass_lanes ? 31 - __clz(mass_lanes) : 0);
      const float mass_before = __shfl_sync(0xffffffffu, incl - mass_l, c);
      const float mass = __shfl_sync(0xffffffffu, mass_l, c);
      if (l == 0) {
        sh.chunk = c;
        sh.token = -1;
        // the fraction of the chunk's mass below the draw, carried to the chunk's recomputed weights
        sh.frac = mass > 0.f ? fminf(fmaxf((target - mass_before) / mass, 0.f), 1.f) : 0.f;
      }
    }
    __syncthreads();
    const int base = sh.chunk * kChunk;
    const int n = min(kChunk, V - base);
    const float* xr = x + base;
    const int segment = (n + kThreads - 1) / kThreads;
    const int lo = min(threadIdx.x * segment, n), hi = min(lo + segment, n);
    float weights[kVec * 4];  // this thread's contiguous weights (segment <= kVec * 4), 0 past hi
#pragma unroll
    for (int q = 0; q < kVec * 4; ++q) {
      const int p = lo + q;
      weights[q] = p < hi && base + p != d ? __expf(scaled(xr[p], t) - m) : 0.f;
    }
    float sum = 0.f;
#pragma unroll
    for (int q = 0; q < kVec * 4; ++q) sum += weights[q];
    float incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, incl, o);
      if (l >= o) incl += y;
    }
    if (l == 31) sh.warp_sums[w] = incl;
    __syncthreads();
    float offset = 0.f, total = 0.f;
#pragma unroll
    for (int k = 0; k < kWarps; ++k) {
      offset += k < w ? sh.warp_sums[k] : 0.f;
      total += sh.warp_sums[k];
    }
    incl += offset;
    const float target = sh.frac * total;
    if (sum > 0.f && incl - sum <= target && target < incl) {
      float acc = incl - sum;
      int token_offset = -1;
#pragma unroll
      for (int q = 0; q < kVec * 4; ++q) {
        acc += weights[q];
        if (token_offset < 0 && acc > target && weights[q] > 0.f) token_offset = lo + q;
      }
      if (token_offset >= 0) sh.token = base + token_offset;
    }
    __syncthreads();
    if (threadIdx.x == 0 && sh.token < 0) {
      // the draw at the chunk's whole mass (rounding): its last weighted token
      int q = n - 1;
      while (q > 0 && (base + q == d || __expf(scaled(xr[q], t) - m) == 0.f)) --q;
      sh.token = base + q;
    }
    __syncthreads();
    token = sh.token;
    x_token = scaled(x[token], t);
  }

  // the kept token's rank: the accepted draft's from the chunks' counts; the drawn token's from the top list when it is
  // above the list's last entry (nothing outside the list ties it), else counted by vs_ranks (-1)
  int rank = (int)draft_count;
  if (is_stop) {
    rank = -1;
    if (x_token > sh.top_v[kTopK - 1]) {
      rank = 0;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) rank += sh.top_v[k] >= x_token;
    }
  }

  // the outcome, then the request's last row CTA to arrive decides the request
  if (threadIdx.x == 0) {
    const RowOutcome outcome{d, is_stop, token, rank, lp_draft, (x_token - m) - logs, (scaled(x[0], t) - m) - logs,
                             x_token};
    const int4* words = reinterpret_cast<const int4*>(&outcome);
    int4* dst = reinterpret_cast<int4*>(outcomes + r);
    dst[0] = words[0];
    dst[1] = words[1];
    __threadfence();
    sh.is_last = atomicAdd(arrivals + i, 1) == count - 1;
  }
  __syncthreads();
  if (!sh.is_last || w != 0) return;
  __threadfence();
  RowOutcome o{-1, 1, 0, 0, 0.f, 0.f, 0.f, 0.f};
  if (l < count) {
    const int4* src = reinterpret_cast<const int4*>(outcomes + first_row + l);
    int4* words = reinterpret_cast<int4*>(&o);
    words[0] = __ldcg(src);
    words[1] = __ldcg(src + 1);
  }
  // the first row that stops (the last row always does: it verifies no draft) draws; the rows before keep their drafts
  const int a = __ffs(__ballot_sync(0xffffffffu, l < count && o.is_stop != 0)) - 1;
  const int drawn_token = __shfl_sync(0xffffffffu, o.token, a);
  if (l < count) {
    const int64_t rr = first_row + l;
    out_ids[rr * (K + 1)] = l < a ? o.draft : (l == a ? drawn_token : 0);
    out_lp[rr * (K + 1)] = l < a ? o.lp_draft : (l == a ? o.lp_token : o.lp_zero);
    ranks[rr] = l <= a ? max(o.rank, 0) : 0;
    if (l < a) sampled[(int64_t)i * sampled_width + l] = o.draft;
    if (l == a) {
      sampled[(int64_t)i * sampled_width + a] = drawn_token;
      rank_rows[i] = o.rank < 0 ? (int)rr : -1;
      rank_values[i] = o.x_token;
    }
  }
  if (l == 0) {
    const bool is_prefilling = seq_lens[i] < prefill_len[state];
    num_sampled[i] = is_prefilling ? 0 : a + 1;
    num_rejected[i] = is_prefilling ? 0 : count - (a + 1);
  }
}

// a row chunk per CTA: the drawn token's rank, counted on x, for the requests whose top list left it undecided
__global__ void __launch_bounds__(kThreads) vs_ranks(const float* __restrict__ logits, int V,
                                                     const int64_t* __restrict__ row_states,
                                                     const float* __restrict__ temperature,
                                                     const int* __restrict__ rank_rows,
                                                     const float* __restrict__ rank_values,
                                                     int64_t* __restrict__ ranks) {
  const int c = blockIdx.x, i = blockIdx.y;
  pdl_wait();
  const int r = rank_rows[i];
  if (r < 0) return;
  const int start = c * kChunk;
  const int n = min(kChunk, V - start);
  const float t = temperature[row_states[r]];
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

// a kernel launch as a programmatic dependent of the stream's previous kernel
template <typename Kernel, typename... Args>
void launch_dependent(Kernel kernel, dim3 grid, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
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
  auto seg_max = torch::empty({(int64_t)R * S * kThreads}, f32);
  auto outcomes = torch::empty({R, kOutcomeWords}, i32);
  auto arrivals = torch::empty({n}, i32);
  auto rank_rows = torch::empty({n}, i32);
  auto rank_values = torch::empty({n}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  float* pm = part.data_ptr<float>();
  int* part_nonfinite = reinterpret_cast<int*>(pm + 3 * R * S);
  const float* x = logits.data_ptr<float>();
  launch_dependent(vs_chunks, dim3(S, R), stream, x, R, V, (const int*)input_ids.data_ptr<int>(),
                   (const int64_t*)positions.data_ptr<int64_t>(), (const int64_t*)logits_indices.data_ptr<int64_t>(),
                   (const int64_t*)row_states.data_ptr<int64_t>(), (const int*)row_local.data_ptr<int>(),
                   (const int*)prefill_len.data_ptr<int>(), (const float*)temperature.data_ptr<float>(), pm,
                   pm + R * S, pm + 2 * R * S, part_nonfinite, seg_max.data_ptr<float>(), arrivals.data_ptr<int>(), n);
  launch_dependent(vs_rows, dim3(width, n), stream, x, R, V, S, (int)K, (const int*)input_ids.data_ptr<int>(),
                   (const int64_t*)positions.data_ptr<int64_t>(), (const int64_t*)logits_indices.data_ptr<int64_t>(),
                   (const int64_t*)row_states.data_ptr<int64_t>(), (const int*)row_local.data_ptr<int>(),
                   (const int*)cu_num_logits.data_ptr<int>(), (const int64_t*)idx_mapping.data_ptr<int64_t>(),
                   (const int*)seq_lens.data_ptr<int>(), (const int*)prefill_len.data_ptr<int>(),
                   (const float*)temperature.data_ptr<float>(), (const int64_t*)seeds.data_ptr<int64_t>(),
                   (const float*)(rates.numel() ? rates.data_ptr<float>() : nullptr), (const float*)pm,
                   (const float*)(pm + R * S), (const float*)(pm + 2 * R * S), (const int*)part_nonfinite,
                   (const float*)seg_max.data_ptr<float>(), reinterpret_cast<RowOutcome*>(outcomes.data_ptr<int>()),
                   arrivals.data_ptr<int>(), sampled.data_ptr<int64_t>(), (int)width, num_sampled.data_ptr<int>(),
                   num_rejected.data_ptr<int>(), ids.data_ptr<int>(), lp.data_ptr<float>(), ranks.data_ptr<int64_t>(),
                   rank_rows.data_ptr<int>(), rank_values.data_ptr<float>(), nonfinite.data_ptr<int>());
  launch_dependent(vs_ranks, dim3(S, n), stream, x, V, (const int64_t*)row_states.data_ptr<int64_t>(),
                   (const float*)temperature.data_ptr<float>(), (const int*)rank_rows.data_ptr<int>(),
                   (const float*)rank_values.data_ptr<float>(), ranks.data_ptr<int64_t>());
  return {sampled, num_sampled, num_rejected, ids, lp, ranks, nonfinite};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def(
      "verify_sample(Tensor logits, Tensor input_ids, Tensor positions, Tensor logits_indices, Tensor row_states, "
      "Tensor row_local, Tensor cu_num_logits, Tensor idx_mapping, Tensor seq_lens, Tensor prefill_len, "
      "Tensor temperature, Tensor seeds, Tensor rates, int K, int width) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("verify_sample", &verify_sample); }
