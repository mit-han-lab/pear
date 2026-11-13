// The top-k log-probs of a step's logits rows in one launch, as vLLM's compute_topk_scores computes them on its fast
// path: the sampled token, then the k largest logits by (logit desc, index asc) as torch.topk breaks ties; logprob =
// x - max - log(sum exp(x - max)) in fp32; the sampled token's rank, the count of logits >= its own.
#pragma once
#include "common.cuh"

namespace mk {

constexpr int LP_K = 32;  // the head: one key per lane of a warp
constexpr int LP_THREADS = 512;  // a standalone kernel's CTA
constexpr int LP_VEC_BYTES = 16;     // the chunk in smem as 16-byte vectors of logits
constexpr int LP_CHUNK_VECS = 2048;  // a CTA's chunk at most
constexpr int LP_MAX_CHUNKS = 32;  // CTAs per row at most
constexpr uint32_t FULL_MASK = 0xffffffffu;
constexpr uint32_t SIGN_BIT = 0x80000000u;

// A thread group running a chunk reduction: NTH threads, its thread index and named barrier (barrier 0 in a standalone
// kernel's CTA, their own in the megakernel's prologue warps and the spec sampler's phases); its shares follow from NTH
template <int NTH>
struct LpGrp {
  static constexpr int THREADS = NTH, WARPS = NTH / 32;
  static constexpr int THREAD_VECS = LP_CHUNK_VECS / NTH;            // a thread's vectors of a chunk at most
  static constexpr int PARTS_PER_WARP = LP_MAX_CHUNKS / WARPS;       // a row's partials a warp merges
  // the keys at or above the threshold lie in the k threads whose maxima reach it: k threads' logits (bf16) at most
  static constexpr int CAND = LP_K * THREAD_VECS * (LP_VEC_BYTES / (int)sizeof(__nv_bfloat16));
  static_assert(NTH % 32 == 0 && LP_CHUNK_VECS % NTH == 0 && LP_MAX_CHUNKS % WARPS == 0, "the group's shares");
  int tid, bar;
  __device__ __forceinline__ void sync() const { named_sync(bar, NTH); }
};

struct LpPartial {  // one chunk's reduction
  float m, s;       // max, sum of exp(x - m)
  uint32_t count;   // logits >= the sampled token's
  unsigned long long key[LP_K];  // the chunk's top k, descending
};

// a logit and its index as one key ordered as the head is: (order-preserving bits of x) << 32 | (~index)
__device__ __forceinline__ unsigned long long lp_key(float x, uint32_t idx) {
  const uint32_t u = __float_as_uint(x);
  const uint32_t o = u ^ ((uint32_t)((int32_t)u >> 31) | SIGN_BIT);  // (negative: ~u, else u | sign)
  return ((unsigned long long)o << 32) | (unsigned long long)(~idx);
}
__device__ __forceinline__ float lp_value(unsigned long long k) {
  const uint32_t o = (uint32_t)(k >> 32);
  return __uint_as_float(o ^ ((uint32_t)((int32_t)~o >> 31) | SIGN_BIT));
}
__device__ __forceinline__ uint32_t lp_index(unsigned long long k) { return ~(uint32_t)k; }
__device__ __forceinline__ float lp_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float lp_float(float v) { return v; }

// logit e of a 16-byte vector (e at run time: selects, no local memory)
template <typename T>
__device__ __forceinline__ float lp_elem(const uint4& q, int e);
template <>
__device__ __forceinline__ float lp_elem<__nv_bfloat16>(const uint4& q, int e) {
  const uint32_t w = (e & 4) ? ((e & 2) ? q.w : q.z) : ((e & 2) ? q.y : q.x);
  return (e & 1) ? bf16hi(w) : bf16lo(w);
}
template <>
__device__ __forceinline__ float lp_elem<float>(const uint4& q, int e) {
  return __uint_as_float((e & 2) ? ((e & 1) ? q.w : q.z) : ((e & 1) ? q.y : q.x));
}

// Comparisons as all-ones / zero masks (set.*), and exp of a difference with -inf - -inf (NaN) as exp(-inf) = 0: no
// selects against a zero (ptxas: a CS2R zero then a predicated write, read sooner than the zero lands)
__device__ __forceinline__ uint32_t lp_ge(float a, float b) {
  uint32_t r;
  asm("set.ge.u32.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t lp_lt(int a, int b) {
  uint32_t r;
  asm("set.lt.u32.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ unsigned long long lp_wide(uint32_t mask) { return ((unsigned long long)mask << 32) | mask; }
// a where the mask is set, else b (bitwise: no select against a constant)
__device__ __forceinline__ float lp_pick(uint32_t mask, float a, float b) {
  return __uint_as_float((__float_as_uint(a) & mask) | (__float_as_uint(b) & ~mask));
}
__device__ __forceinline__ uint32_t lp_pick(uint32_t mask, uint32_t a, uint32_t b) { return (a & mask) | (b & ~mask); }
__device__ __forceinline__ float lp_exp_diff(float a, float b) { return exp_tri(fmaxf(a - b, -INFINITY)); }

// (max, sum of exp(x - max)) of two parts (exp as vLLM's log-softmax: tl.exp); a part with no finite logit adds nothing
__device__ __forceinline__ void lp_combine(float& m, float& s, float m2, float s2) {
  const float mm = fmaxf(m, m2);
  s = s * lp_exp_diff(m, mm) + s2 * lp_exp_diff(m2, mm);
  m = mm;
}

// a warp's 32 keys (one per lane) sorted descending by lane: the bitonic network over shuffles (rolled: small code)
__device__ __forceinline__ unsigned long long warp_sort_desc(unsigned long long v, int lane) {
#pragma unroll 1
  for (int size = 2; size <= 32; size <<= 1)
#pragma unroll 1
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const unsigned long long o = __shfl_xor_sync(FULL_MASK, v, stride);
      const bool keeps_larger = ((lane & size) == 0) == ((lane & stride) == 0);  // (of the pair)
      v = keeps_larger ? (v > o ? v : o) : (v < o ? v : o);
    }
  return v;
}
// two descending lists a, b (lane i: the i-th) into the top 32 of their union, descending
__device__ __forceinline__ unsigned long long warp_merge_desc(unsigned long long a, unsigned long long b, int lane) {
  const unsigned long long r = __shfl_sync(FULL_MASK, b, 31 - lane);  // b reversed: a | r is bitonic
  unsigned long long v = a > r ? a : r;                               // the top 32, as a bitonic sequence
#pragma unroll 1
  for (int stride = 16; stride > 0; stride >>= 1) {
    const unsigned long long o = __shfl_xor_sync(FULL_MASK, v, stride);
    v = ((lane & stride) == 0) ? (v > o ? v : o) : (v < o ? v : o);
  }
  return v;
}
// the warps' descending lists sh[w] merged pairwise into sh[0], the top k of their union (all threads at the barriers)
template <int NTH>
__device__ __forceinline__ void lp_tree_merge(const LpGrp<NTH>& g, unsigned long long (*sh)[LP_K]) {
  const int warp = g.tid >> 5, lane = g.tid & 31;
#pragma unroll 1
  for (int d = 1; d < LpGrp<NTH>::WARPS; d <<= 1) {
    if ((warp & (2 * d - 1)) == 0) sh[warp][lane] = warp_merge_desc(sh[warp][lane], sh[warp + d][lane], lane);
    g.sync();
  }
}

// A group's shared memory for its chunk's reduction (topk_logprobs_kernel), the small fields before the chunk
template <int NTH>
struct LpSmem {
  static constexpr int W = LpGrp<NTH>::WARPS;
  uint64_t bar;
  uint32_t n;
  uint32_t is_last_cta;
  float m[W], s[W];                             // per warp: max, sum of exp
  uint32_t c[W];                                // per warp: the logits >= the threshold
  unsigned long long key[W][LP_K];
  unsigned long long cand[LpGrp<NTH>::CAND];    // the keys >= the threshold
  uint4 x[LP_CHUNK_VECS];                       // the chunk, 16-byte vectors
};

// thread 0: the chunk's bulk copy (row x's vectors [v0, v1)) into sh.x on sh.bar (initialized here: one use per launch)
template <typename T, int NTH>
__device__ __forceinline__ void lp_chunk_issue(const T* x, int v0, int v1, LpSmem<NTH>& sh) {
  mbar_init(&sh.bar, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const uint32_t bytes = (uint32_t)(v1 - v0) * LP_VEC_BYTES;
  mbar_expect_tx(&sh.bar, bytes);
  bulk_g2s(sh.x, reinterpret_cast<const uint4*>(x) + v0, bytes, &sh.bar);
  sh.n = 0u;
}

// The chunk [v0, v1) of a row of V logits (lp_chunk_issue's copy, then a barrier), every thread: each thread's max
// (first index at a tie), sum of exp (vLLM's tl.exp) and count of logits >= xs; the threshold, the k-th largest of the
// threads' maxima (only those k threads hold keys >= it); the top k of the candidates at or above it into sh.key[0]
template <typename T, int NTH>
__device__ void lp_chunk(const LpGrp<NTH>& g, int V, int v0, int v1, float xs, LpSmem<NTH>& sh) {
  constexpr int E = LP_VEC_BYTES / sizeof(T), NT_ = LpGrp<NTH>::THREADS;  // logits per vector, the group's threads
  const int tid = g.tid, warp = tid >> 5, lane = tid & 31;
  mbar_wait(&sh.bar, 0);
  float m = -INFINITY;
  int mi = -1;
#pragma unroll 1
  for (int v = v0 + tid; v < v1; v += NT_) {
    const uint4 q = sh.x[v - v0];
#pragma unroll
    for (int e = 0; e < E; e++) {
      const float f = lp_elem<T>(q, e);
      if (v * E + e < V && (f > m || mi < 0)) m = f, mi = v * E + e;
    }
  }
  const unsigned long long kmax = lp_key(m, (uint32_t)mi) & lp_wide(lp_lt(-1, mi));  // (0: no logit)
  float s = 0.f;
  uint32_t cnt = 0u;
#pragma unroll 1
  for (int v = v0 + tid; v < v1; v += NT_) {
    const uint4 q = sh.x[v - v0];
#pragma unroll
    for (int e = 0; e < E; e++) {
      const float f = lp_elem<T>(q, e);
      const uint32_t in_row_mask = lp_lt(v * E + e, V);
      s += __uint_as_float(__float_as_uint(lp_exp_diff(f, m)) & in_row_mask);
      cnt += lp_ge(f, xs) & in_row_mask & 1u;
    }
  }
  // the warp's stats (one shuffle tree: a deterministic order); the threshold: the k-th largest of the threads' maxima
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lp_combine(m, s, __shfl_xor_sync(FULL_MASK, m, o), __shfl_xor_sync(FULL_MASK, s, o));
    cnt += __shfl_xor_sync(FULL_MASK, cnt, o);
  }
  if (lane == 0) sh.m[warp] = m, sh.s[warp] = s, sh.c[warp] = cnt;
  sh.key[warp][lane] = warp_sort_desc(kmax, lane);
  g.sync();
  lp_tree_merge(g, sh.key);
  const unsigned long long theta = sh.key[0][LP_K - 1];
  const float theta_x = fmaxf(lp_value(theta), -INFINITY);  // (0, fewer than k threads hold logits: NaN -> -inf)
#pragma unroll 1
  for (int v = v0 + tid; v < v1; v += NT_) {
    const uint4 q = sh.x[v - v0];
    uint32_t hit = 0u;
#pragma unroll
    for (int e = 0; e < E; e++) hit |= lp_lt(v * E + e, V) & lp_ge(lp_elem<T>(q, e), theta_x) & (1u << e);
#pragma unroll 1
    for (; hit != 0u; hit &= hit - 1u) {
      const int e = __ffs(hit) - 1;
      const unsigned long long k = lp_key(lp_elem<T>(q, e), (uint32_t)(v * E + e));
      if (k >= theta) sh.cand[atomicAdd(&sh.n, 1u)] = k;
    }
  }
  g.sync();
  // the candidates' top k: warp w merges batches w, w + the warps, ... of k keys, then the warps' heads in a tree
  const uint32_t n = sh.n;
  unsigned long long h = 0ull;
#pragma unroll 1
  for (uint32_t b = (uint32_t)warp * LP_K; b < n; b += NT_) {
    const unsigned long long k = sh.cand[min(b + lane, n - 1u)] & lp_wide(lp_lt((int)(b + lane), (int)n));
    h = warp_merge_desc(h, warp_sort_desc(k, lane), lane);
  }
  sh.key[warp][lane] = h;
  g.sync();
  lp_tree_merge(g, sh.key);
}

// a warp: the CTA's (max, sum of exp) and count from the warps' in sh (lane w: warp w's; a deterministic shuffle tree)
template <int NTH>
__device__ __forceinline__ void lp_cta_stats(const LpSmem<NTH>& sh, int lane, float& m, float& s, uint32_t& c) {
  constexpr int W = LpGrp<NTH>::WARPS;
  const uint32_t has_warp_mask = lp_lt(lane, W);
  const int w = min(lane, W - 1);
  m = __uint_as_float((__float_as_uint(sh.m[w]) & has_warp_mask) | (__float_as_uint(-INFINITY) & ~has_warp_mask));
  s = __uint_as_float(__float_as_uint(sh.s[w]) & has_warp_mask);
  c = sh.c[w] & has_warp_mask;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lp_combine(m, s, __shfl_xor_sync(FULL_MASK, m, o), __shfl_xor_sync(FULL_MASK, s, o));
    c += __shfl_xor_sync(FULL_MASK, c, o);
  }
}

// A row's partials (part[0 .. chunks)) merged by the CTA (every thread, one order): its top k into sh.key[0], warp 0
// its (max, sum of exp) and count (lp_cta_stats over the warps' merges)
template <int NTH>
__device__ __forceinline__ void lp_merge_row(const LpGrp<NTH>& g, const LpPartial* part, int chunks, LpSmem<NTH>& sh,
                                             float& M, float& S, uint32_t& C) {
  constexpr int PW = LpGrp<NTH>::PARTS_PER_WARP, W = LpGrp<NTH>::WARPS;
  const int warp = g.tid >> 5, lane = g.tid & 31;
  unsigned long long pk[PW];
  float qm[PW], qs[PW];
  uint32_t qc[PW];
#pragma unroll
  for (int q = 0; q < PW; q++) {
    const int c = warp + q * W;
    const uint32_t has_chunk_mask = lp_lt(c, chunks);  // (a dead slot: no keys, max -inf, sum and count 0)
    const LpPartial& pq = part[min(c, chunks - 1)];
    pk[q] = __ldcg(&pq.key[lane]) & lp_wide(has_chunk_mask);
    qm[q] = __uint_as_float((__float_as_uint(__ldcg(&pq.m)) & has_chunk_mask) |
                            (__float_as_uint(-INFINITY) & ~has_chunk_mask));
    qs[q] = __uint_as_float(__float_as_uint(__ldcg(&pq.s)) & has_chunk_mask);
    qc[q] = __ldcg(&pq.count) & has_chunk_mask;
  }
  unsigned long long h = pk[0];
  float m = qm[0], s = qs[0];
  uint32_t c = qc[0];
#pragma unroll
  for (int q = 1; q < PW; q++) {
    h = warp_merge_desc(h, pk[q], lane);
    lp_combine(m, s, qm[q], qs[q]);
    c += qc[q];
  }
  sh.key[warp][lane] = h;
  if (lane == 0) sh.m[warp] = m, sh.s[warp] = s, sh.c[warp] = c;
  g.sync();
  lp_tree_merge(g, sh.key);
  if (warp == 0) lp_cta_stats(sh, lane, M, S, C);
}

// warp 0: a row's outputs from its merged head (sh.key[0]) and (M, S): the sampled token sel (logit xs, rank C) first
template <int NTH>
__device__ __forceinline__ void lp_row_out(const LpSmem<NTH>& sh, int lane, float M, float S, int64_t sel, float xs, int64_t C,
                                           int64_t* rid, float* rlp, int64_t* rank) {
  const unsigned long long h = sh.key[0][lane];
  const float lse = logf(S);
  rid[1 + lane] = (int64_t)lp_index(h);
  rlp[1 + lane] = (lp_value(h) - M) - lse;
  if (lane == 0) {
    rid[0] = sel;
    rlp[0] = (xs - M) - lse;
    *rank = C;
  }
}

// CTA (row, chunk) reduces its chunk into a partial; the row's last CTA to finish (a counter) merges the row's partials
// and writes the outputs (the counter back to 0 for the next launch)
template <typename T>
__global__ void __launch_bounds__(LP_THREADS) topk_logprobs_kernel(const T* logits, int64_t stride, const int64_t* sampled,
                                                                    int V, int chunks, LpPartial* part, uint32_t* row_done,
                                                                    int64_t* ids, float* logprobs, int64_t* ranks) {
  constexpr int E = LP_VEC_BYTES / sizeof(T);
  __shared__ LpSmem<LP_THREADS> sh;
  const LpGrp<LP_THREADS> g{(int)threadIdx.x, 0};
  const int row = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const T* x = logits + (int64_t)row * stride;
  const int nvec = (V + E - 1) / E;
  const int v0 = (int)((int64_t)nvec * chunk / chunks), v1 = (int)((int64_t)nvec * (chunk + 1) / chunks);
  if (threadIdx.x == 0) lp_chunk_issue(x, v0, v1, sh);
  const int64_t sel = sampled[row];
  const float xs = lp_float(x[sel]);
  __syncthreads();
  lp_chunk<T>(g, V, v0, v1, xs, sh);
  LpPartial& p = part[blockIdx.x];
  if (warp == 0) {
    float pm, ps;
    uint32_t pc;
    lp_cta_stats(sh, lane, pm, ps, pc);
    p.key[lane] = sh.key[0][lane];
    if (lane == 0) p.m = pm, p.s = ps, p.count = pc;
    __threadfence();  // the partial before the count
    __syncwarp();
    if (lane == 0) sh.is_last_cta = atomicAdd(&row_done[row], 1u) == (uint32_t)chunks - 1;
  }
  __syncthreads();
  if (!sh.is_last_cta) return;
  // the row's last CTA: every partial's loads first, merged in one order (deterministic)
  __threadfence();
  float M, S;
  uint32_t C;
  lp_merge_row(g, part + (int64_t)row * chunks, chunks, sh, M, S, C);
  if (warp != 0) return;
  lp_row_out(sh, lane, M, S, sel, xs, (int64_t)C, ids + (int64_t)row * (LP_K + 1), logprobs + (int64_t)row * (LP_K + 1),
             ranks + row);
  if (lane == 0) row_done[row] = 0u;
}

}  // namespace mk

#ifdef TORCH_CHECK  // the torch host code (in the torch extension's translation unit only)
// The op: logits [B, V] (bf16 or fp32, 16-byte aligned rows), the sampled token per row, the workspace (partials, row
// counters at zero) and the CTAs per row asked for (raised to what a chunk's smem holds); its outputs ids and logprobs
// [B, 1 + k], ranks [B]
inline void mk_topk_logprobs(torch::Tensor logits, torch::Tensor sampled, torch::Tensor part, torch::Tensor row_done, int64_t chunks,
                      torch::Tensor ids, torch::Tensor logprobs, torch::Tensor ranks) {
  const at::cuda::CUDAGuard guard(logits.device());
  TORCH_CHECK(logits.dim() == 2 && logits.stride(1) == 1 && sampled.scalar_type() == torch::kInt64, "logits / sampled");
  const int64_t per_vec = mk::LP_VEC_BYTES / logits.element_size();
  TORCH_CHECK(logits.stride(0) % per_vec == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % mk::LP_VEC_BYTES == 0,
              "logits rows 16-byte aligned");
  const int B = (int)logits.size(0), V = (int)logits.size(1);
  const int64_t nvec = (V + per_vec - 1) / per_vec;
  chunks = std::min(std::max(chunks, (nvec + mk::LP_CHUNK_VECS - 1) / mk::LP_CHUNK_VECS), nvec);
  TORCH_CHECK(chunks <= mk::LP_MAX_CHUNKS && (int64_t)mk::LP_CHUNK_VECS * chunks >= nvec, "a row's chunks");
  TORCH_CHECK(part.numel() * part.element_size() >= (int64_t)sizeof(mk::LpPartial) * B * chunks, "workspace");
  TORCH_CHECK(row_done.numel() >= B, "row counters");
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* pp = reinterpret_cast<mk::LpPartial*>(part.data_ptr());
  auto* rd = reinterpret_cast<uint32_t*>(row_done.data_ptr());
  if (logits.scalar_type() == torch::kBFloat16)
    mk::topk_logprobs_kernel<__nv_bfloat16><<<B * (int)chunks, mk::LP_THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), logits.stride(0), sampled.data_ptr<int64_t>(), V, (int)chunks,
        pp, rd, ids.data_ptr<int64_t>(), logprobs.data_ptr<float>(), ranks.data_ptr<int64_t>());
  else
    mk::topk_logprobs_kernel<float><<<B * (int)chunks, mk::LP_THREADS, 0, stream>>>(
        logits.data_ptr<float>(), logits.stride(0), sampled.data_ptr<int64_t>(), V, (int)chunks, pp, rd,
        ids.data_ptr<int64_t>(), logprobs.data_ptr<float>(), ranks.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif  // TORCH_CHECK
