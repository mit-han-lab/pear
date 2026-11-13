// Fused random sampling without top-k / top-p (vLLM's TopKTopPSampler native path) with processed log-probs: per row
// of fp32 logits x at temperature t, y = x / t: logprobs = log_softmax(y), sampled ~ softmax(y), the top-K log-probs
// with their ids, the sampled token's log-prob and its rank (the tokens whose log-prob is >= its own)
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cfloat>
#include <cstdint>

namespace sampler {

constexpr int K = 32;      // list length (top-K, K <= 32)
constexpr int NT1 = 256;   // pass-1 threads
constexpr int NW1 = NT1 / 32;
constexpr int NT3 = 512;   // pass-3 threads

struct ChunkRec {
  float m, s;
  float topv[K];
  int topi[K];
};

struct RowRec {
  float logz;     // log Z of the row
  int chunk;      // the chunk holding the draw
  float m;        // that chunk's max
  float r;        // the remaining mass inside the chunk, in units of exp(y - m)
  float ys;       // the sampled token's y
  int need_rank;  // pass4 counts the rank
};

// Philox4x32-10
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// descending order with ties broken by the lower index
__device__ __forceinline__ bool before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// merges a sorted (descending) list (lv, li) with a sorted (descending) list (bv, bi): the top 32, sorted descending
__device__ __forceinline__ void warp_merge(float& lv, int& li, float bv, int bi) {
  const int lane = threadIdx.x & 31;
  // lane j pairs list[j] with other[31 - j]: the larger of each pair are the top 32 of the union, a bitonic sequence
  const float rv = __shfl_sync(0xffffffffu, bv, 31 - lane);
  const int ri = __shfl_sync(0xffffffffu, bi, 31 - lane);
  if (before(rv, ri, lv, li)) {
    lv = rv;
    li = ri;
  }
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, lv, stride);
    const int oi = __shfl_xor_sync(0xffffffffu, li, stride);
    const bool lower = (lane & stride) == 0;
    const bool mine_first = before(lv, li, ov, oi);
    if (lower != mine_first) {
      lv = ov;
      li = oi;
    }
  }
}

// inserts (v, i) into a warp's sorted (descending) 32-entry list if it beats the last entry
__device__ __forceinline__ void warp_insert(float& lv, int& li, float v, int i) {
  const int lane = threadIdx.x & 31;
  const int pos = __popc(__ballot_sync(0xffffffffu, before(lv, li, v, i)));  // entries staying ahead of the new one
  if (pos >= 32) return;
  const float up_v = __shfl_up_sync(0xffffffffu, lv, 1);
  const int up_i = __shfl_up_sync(0xffffffffu, li, 1);
  if (lane == pos) {
    lv = v;
    li = i;
  } else if (lane > pos) {
    lv = up_v;
    li = up_i;
  }
}

struct Params {
  const float* logits;
  int64_t stride;
  const float* temperature;  // [B]
  int V, B, chunks, chunk_len;
  uint64_t seed, offset;
  ChunkRec* recs;  // [B][chunks]
  RowRec* rows;    // [B]
  int64_t* sampled;
  int32_t* out_ids;  // [B][k + 1]: sampled, then the top k
  float* out_lp;     // [B][k + 1]
  int64_t* ranks;    // [B]
  int k;             // returned top count (<= K)
};

__device__ __forceinline__ float temp_of(const Params& p, int b) { return p.temperature[b]; }

// y = x / t exactly as vLLM's in-place division (skipped for t == 1, where it is the identity)
__device__ __forceinline__ float scaled(float x, float t, bool scale) { return scale ? x / t : x; }

// grid chunks x rows: per chunk the max m and the sum s of exp(y - m), and a top-K list (a sorted 32-entry list per
// warp, fed by single insertions of the values above the CTA's running threshold; the lists merged)
__global__ void __launch_bounds__(NT1) pass1(const Params p) {
  const int c = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = p.logits + (int64_t)b * p.stride;
  const float t = temp_of(p, b);
  const bool scale = t != 1.f;
  const int begin = c * p.chunk_len, end = min(p.V, begin + p.chunk_len);

  float m = -FLT_MAX, s = 0.f;
  float lv = -FLT_MAX;  // this warp's sorted top-32 list: lane j holds entry j
  int li = 0x7fffffff;
  float last_v = -FLT_MAX;  // the list's last entry
  int last_i = 0x7fffffff;
  float cta_thr = -FLT_MAX;
  __shared__ float thr_s[NW1];

  // every thread runs the same number of sweeps (chunk_len is a multiple of 4 * NT1) for the warp votes
  const int sweeps = p.chunk_len / (4 * NT1);
  for (int it = 0; it < sweeps; ++it) {
    const int base = begin + it * 4 * NT1 + 4 * tid;
    float x[4];
    if (base + 3 < end) {
      const float4 v = __ldcs(reinterpret_cast<const float4*>(row + base));
      x[0] = v.x;
      x[1] = v.y;
      x[2] = v.z;
      x[3] = v.w;
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] = base + j < end ? row[base + j] : -FLT_MAX;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (base + j < end) {
        const float y = scaled(x[j], t, scale);
        if (y > m) {
          s = s * __expf(m - y) + 1.f;
          m = y;
        } else {
          s += __expf(y - m);
        }
        x[j] = y;
      }
    }
    // top-K candidates: values at or above the CTA's running threshold that beat this warp's last entry
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool cand = base + j < end && x[j] >= cta_thr && before(x[j], base + j, last_v, last_i);
      unsigned mask = __ballot_sync(0xffffffffu, cand);
      if (mask) {
        while (mask) {
          const int src = __ffs(mask) - 1;
          mask &= mask - 1;
          warp_insert(lv, li, __shfl_sync(0xffffffffu, x[j], src), __shfl_sync(0xffffffffu, base + j, src));
        }
        last_v = __shfl_sync(0xffffffffu, lv, 31);
        last_i = __shfl_sync(0xffffffffu, li, 31);
      }
    }
    // the CTA's threshold: the largest of the warps' last entries (each bounds the CTA's 32nd value from below)
    if ((it & 3) == 3) {
      if (lane == 0) thr_s[warp] = last_v;
      __syncthreads();
      float t_max = thr_s[0];
#pragma unroll
      for (int w = 1; w < NW1; ++w) t_max = fmaxf(t_max, thr_s[w]);
      cta_thr = fmaxf(cta_thr, t_max);
      __syncthreads();
    }
  }

  __shared__ float sm_m[NW1], sm_s[NW1];
  __shared__ float sm_lv[NW1][32];
  __shared__ int sm_li[NW1][32];
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, m, o), os = __shfl_xor_sync(0xffffffffu, s, o);
    const float nm = fmaxf(m, om);
    s = (m == -FLT_MAX ? 0.f : s * __expf(m - nm)) + (om == -FLT_MAX ? 0.f : os * __expf(om - nm));
    m = nm;
  }
  if (lane == 0) {
    sm_m[warp] = m;
    sm_s[warp] = s;
  }
  sm_lv[warp][lane] = lv;
  sm_li[warp][lane] = li;
  __syncthreads();
  if (warp == 0) {
    float M = -FLT_MAX, S = 0.f;
    for (int w = 0; w < NW1; ++w) {
      const float nm = fmaxf(M, sm_m[w]);
      S = (M == -FLT_MAX ? 0.f : S * __expf(M - nm)) + (sm_m[w] == -FLT_MAX ? 0.f : sm_s[w] * __expf(sm_m[w] - nm));
      M = nm;
    }
    float v = sm_lv[0][lane];
    int i = sm_li[0][lane];
    for (int w = 1; w < NW1; ++w) warp_merge(v, i, sm_lv[w][lane], sm_li[w][lane]);
    ChunkRec* rec = p.recs + (int64_t)b * p.chunks + c;
    if (lane == 0) {
      rec->m = M;
      rec->s = S;
    }
    rec->topv[lane] = v;
    rec->topi[lane] = i;
  }
}

// one warp per row: log Z, the top-K, and the first level of the draw, exact inverse-CDF sampling from one uniform per
// row (Philox, counter = (offset, row)): the chunk that holds it by the chunks' masses, and the mass left inside it
__global__ void __launch_bounds__(32) pass2(const Params p) {
  const int b = blockIdx.x, lane = threadIdx.x;
  const ChunkRec* recs = p.recs + (int64_t)b * p.chunks;
  float M = -FLT_MAX;
  for (int c = 0; c < p.chunks; ++c) M = fmaxf(M, recs[c].m);
  double S = 0.0;
  for (int c = 0; c < p.chunks; ++c) S += (double)recs[c].s * exp((double)recs[c].m - M);
  const float logz = M + (float)log(S);
  float v = recs[0].topv[lane];
  int i = recs[0].topi[lane];
  for (int c = 1; c < p.chunks; ++c) warp_merge(v, i, recs[c].topv[lane], recs[c].topi[lane]);
  if (lane < p.k) {
    p.out_ids[(int64_t)b * (p.k + 1) + 1 + lane] = i;
    p.out_lp[(int64_t)b * (p.k + 1) + 1 + lane] = v - logz;
  }
  if (lane == 0) {
    // one uniform in [0, 1) with 53 bits
    const uint2 key = make_uint2((uint32_t)p.seed, (uint32_t)(p.seed >> 32));
    const uint4 r = philox(make_uint4((uint32_t)b, 0u, (uint32_t)p.offset, (uint32_t)(p.offset >> 32)), key);
    const double u = ((double)(r.x >> 5) * 67108864.0 + (double)(r.y >> 6)) * (1.0 / 9007199254740992.0);
    double target = u * S, cum = 0.0;
    int chosen = p.chunks - 1;
    for (int c = 0; c < p.chunks; ++c) {
      const double mass = (double)recs[c].s * exp((double)recs[c].m - M);
      if (mass > 0.0) chosen = c;  // the last chunk with mass, if rounding leaves the target past the end
      if (target < cum + mass) {
        chosen = c;
        break;
      }
      cum += mass;
    }
    RowRec& rr = p.rows[b];
    rr.logz = logz;
    rr.chunk = chosen;
    rr.m = recs[chosen].m;
    rr.r = (float)((target - cum) / exp((double)recs[chosen].m - M));
  }
}

// one CTA per row: the token of the chosen chunk where the running mass passes r, by a scan of that chunk only; its
// rank from the top-K list when it is in it
__global__ void __launch_bounds__(NT3) pass3(const Params p) {
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  RowRec& rr = p.rows[b];
  const float* row = p.logits + (int64_t)b * p.stride;
  const float t = temp_of(p, b);
  const bool scale = t != 1.f;
  const int begin = rr.chunk * p.chunk_len, end = min(p.V, begin + p.chunk_len);
  const float m = rr.m, r = rr.r;
  // thread segments of contiguous tokens
  const int len = end - begin, seg = (len + NT3 - 1) / NT3;
  const int s0 = begin + tid * seg, s1 = min(end, s0 + seg);
  float local = 0.f;
  for (int j = s0; j < s1; ++j) local += __expf(scaled(row[j], t, scale) - m);
  // block exclusive scan of the segment sums
  __shared__ float wsum[NT3 / 32];
  float incl = local;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  float before_w = 0.f;
  for (int w = 0; w < warp; ++w) before_w += wsum[w];
  const float excl = before_w + incl - local;
  float total = 0.f;
  for (int w = 0; w < NT3 / 32; ++w) total += wsum[w];
  __shared__ int found;
  if (tid == 0) found = -1;
  __syncthreads();
  const float rc = fminf(r, total);  // rounding between the passes: stay inside the chunk
  if (s0 < s1 && excl <= rc && (rc < excl + local || (rc >= total && s1 == end))) {
    float cum = excl;
    int j = s0;
    for (; j < s1 - 1; ++j) {
      cum += __expf(scaled(row[j], t, scale) - m);
      if (rc < cum) break;
    }
    atomicMax(&found, j);  // one thread qualifies; max guards the boundary case
  }
  __syncthreads();
  if (tid == 0) {
    int j = found;
    if (j < 0) j = s0 < end ? begin : 0;
    const float ys = scaled(row[j], t, scale);
    rr.ys = ys;
    p.sampled[b] = j;
    p.out_ids[(int64_t)b * (p.k + 1)] = j;
    p.out_lp[(int64_t)b * (p.k + 1)] = ys - rr.logz;
    // rank from the returned top-k when the token is in it and a smaller entry follows (no ties past the list)
    int ge = 0, in_list = 0;
    for (int q = 0; q < p.k; ++q) {
      const float lpq = p.out_lp[(int64_t)b * (p.k + 1) + 1 + q] + rr.logz;
      in_list |= p.out_ids[(int64_t)b * (p.k + 1) + 1 + q] == j;
      ge += lpq >= ys;
    }
    const int need = !(in_list && ge < p.k);
    rr.need_rank = need;
    p.ranks[b] = need ? 0 : ge;
  }
}

// counts y >= y_sampled per chunk for the rows that need it
__global__ void __launch_bounds__(NT1) pass4(const Params p) {
  const int c = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const RowRec& rr = p.rows[b];
  if (!rr.need_rank) return;
  const float* row = p.logits + (int64_t)b * p.stride;
  const float t = temp_of(p, b);
  const bool scale = t != 1.f;
  const float ys = rr.ys;
  const int begin = c * p.chunk_len, end = min(p.V, begin + p.chunk_len);
  int n = 0;
  for (int base = begin + 4 * tid; base < end; base += 4 * NT1) {
    if (base + 3 < end) {
      const float4 v = __ldcs(reinterpret_cast<const float4*>(row + base));
      n += (scaled(v.x, t, scale) >= ys) + (scaled(v.y, t, scale) >= ys) + (scaled(v.z, t, scale) >= ys) +
           (scaled(v.w, t, scale) >= ys);
    } else {
      for (int j = base; j < end; ++j) n += scaled(row[j], t, scale) >= ys;
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
  if ((tid & 31) == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(p.ranks + b), (unsigned long long)n);
}

void fused_sample(const torch::Tensor& logits, const torch::Tensor& temperature, int64_t seed, int64_t offset,
                  int64_t k, torch::Tensor& workspace, torch::Tensor& sampled, torch::Tensor& out_ids,
                  torch::Tensor& out_lp, torch::Tensor& ranks) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1, "logits");
  TORCH_CHECK(logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0, "align");
  TORCH_CHECK(k >= 1 && k <= K, "k");
  const int B = logits.size(0), V = logits.size(1);
  TORCH_CHECK(V >= K, "vocabulary");
  Params p;
  p.logits = logits.data_ptr<float>();
  p.stride = logits.stride(0);
  TORCH_CHECK(temperature.scalar_type() == at::kFloat && temperature.numel() >= B && temperature.is_contiguous(),
              "temperature");
  p.temperature = temperature.data_ptr<float>();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int chunks = std::max(1, std::min(64, (4 * sms + B - 1) / std::max(B, 1)));
  int chunk_len = (V + chunks - 1) / chunks;
  chunk_len = (chunk_len + 4 * NT1 - 1) / (4 * NT1) * (4 * NT1);  // whole sweeps of the CTA: uniform loops
  chunks = (V + chunk_len - 1) / chunk_len;
  const int64_t need = (int64_t)B * chunks * sizeof(ChunkRec) + (int64_t)B * sizeof(RowRec) + 256;
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= need, "workspace");
  TORCH_CHECK(sampled.scalar_type() == at::kLong && ranks.scalar_type() == at::kLong &&
                  out_ids.scalar_type() == at::kInt && out_lp.scalar_type() == at::kFloat && out_ids.is_contiguous() &&
                  out_lp.is_contiguous() && out_ids.size(1) == k + 1 && out_lp.size(1) == k + 1,
              "outputs");
  p.V = V;
  p.B = B;
  p.chunks = chunks;
  p.chunk_len = chunk_len;
  p.seed = (uint64_t)seed;
  p.offset = (uint64_t)offset;
  p.recs = reinterpret_cast<ChunkRec*>(workspace.data_ptr());
  p.rows = reinterpret_cast<RowRec*>(reinterpret_cast<char*>(workspace.data_ptr()) +
                                     ((int64_t)B * chunks * sizeof(ChunkRec) + 255) / 256 * 256);
  p.sampled = sampled.data_ptr<int64_t>();
  p.out_ids = out_ids.data_ptr<int32_t>();
  p.out_lp = out_lp.data_ptr<float>();
  p.ranks = ranks.data_ptr<int64_t>();
  p.k = k;
  if (B == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  pass1<<<dim3(chunks, B), NT1, 0, stream>>>(p);
  pass2<<<B, 32, 0, stream>>>(p);
  pass3<<<B, NT3, 0, stream>>>(p);
  pass4<<<dim3(chunks, B), NT1, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sampler

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "fused_sample(Tensor logits, Tensor temperature, int seed, int offset, int k, Tensor(a!) workspace, "
      "Tensor(b!) sampled, Tensor(c!) out_ids, Tensor(d!) out_lp, Tensor(e!) ranks) -> ()");
}
TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("fused_sample", &sampler::fused_sample); }
