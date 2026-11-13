// TP all-reduce + residual add + RMSNorm for decode-sized batches as FlashInfer's MNNVL pair (twoshotAllreduceKernel +
// rmsNormLamport) computes them, with its arithmetic (flashinfer/comm/trtllm_mnnvl_allreduce.cuh, built with
// -use_fast_math), on an MNNVL Lamport workspace of its own
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

namespace allreduce {

constexpr int WS = 4;                       // ranks
constexpr int ELTS = 8;                     // bf16 per thread (one 16-byte packet)
constexpr uint32_t NEGZERO32 = 0x80000000u;  // Lamport sentinel word (fp32 -0)
constexpr int SCATTER = 0, BROADCAST = 1, NUM_STAGES = 2;

__device__ __forceinline__ uint4 ld_volatile(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ bool is_dirty(const uint4& v) {
  return (v.x == NEGZERO32) | (v.y == NEGZERO32) | (v.z == NEGZERO32) | (v.w == NEGZERO32);
}
// bf16 -0 elements to +0: a payload never holds the sentinel
__device__ __forceinline__ uint4 sanitize(uint4 v) {
  uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if ((w[i] & 0xffffu) == 0x8000u) w[i] &= 0xffff0000u;
    if ((w[i] >> 16) == 0x8000u) w[i] &= 0x0000ffffu;
  }
  return v;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
// fast-math lowerings (FlashInfer's -use_fast_math)
__device__ __forceinline__ float add_ftz(float a, float b) {
  float y;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float y;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float y;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
  return y;
}
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
  float y;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}
__device__ __forceinline__ float warp_sum_ftz(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = add_ftz(v, __shfl_xor_sync(0xffffffffu, v, m));
  return v;
}

// FlashInfer's LamportFlags<float4, false> state: [cur idx, dirty idx, bytes per buffer, dirty num stages,
// bytes to clear x4, access count]
struct Flags {
  uint32_t* base;
  uint32_t cur, dirty, bytes_per_buffer, dirty_stages;
  uint32_t to_clear[4];
  __device__ explicit Flags(uint32_t* f) : base(f) {
    const uint4 a = reinterpret_cast<const uint4*>(f)[0], b = reinterpret_cast<const uint4*>(f)[1];
    cur = a.x;
    dirty = a.y;
    bytes_per_buffer = a.z;
    dirty_stages = a.w;
    to_clear[0] = b.x;
    to_clear[1] = b.y;
    to_clear[2] = b.z;
    to_clear[3] = b.w;
  }
  __device__ char* stage(void* buf, uint32_t idx, uint32_t stages, int s) const {
    const size_t per = bytes_per_buffer / stages;
    return reinterpret_cast<char*>(buf) + (size_t)(idx * stages + s) * per;
  }
  __device__ __nv_bfloat16* cur_buf(void* buf, int s) const {
    return reinterpret_cast<__nv_bfloat16*>(stage(buf, cur, NUM_STAGES, s));
  }
  // the previous call's buffer back to the sentinel (every thread of the grid takes 16-byte packets)
  __device__ void clear_dirty(void* buf, int s) const {
    if (s >= (int)dirty_stages) return;
    const uint32_t cta = blockIdx.x * gridDim.y + blockIdx.y;
    const uint32_t gtid = cta * blockDim.x + threadIdx.x, nthreads = gridDim.x * gridDim.y * blockDim.x;
    const uint32_t packets = (to_clear[s] + 15) / 16;
    uint4* p = reinterpret_cast<uint4*>(stage(buf, dirty, dirty_stages, s));
    const uint4 init = make_uint4(NEGZERO32, NEGZERO32, NEGZERO32, NEGZERO32);
    for (uint32_t i = gtid; i < packets; i += nthreads) p[i] = init;
  }
};

struct Params {
  const __nv_bfloat16* input;  // [T, D] this rank's partial sums (overwritten by norm_out when aliased)
  __nv_bfloat16* residual;     // [T, D] in: residual, out: input + residual
  const __nv_bfloat16* gamma;  // [D]
  __nv_bfloat16* norm_out;     // [T, D]
  void* const* uc_ptrs;        // [WS] the ranks' workspace buffers (device array)
  void* uc_local;
  void* mc;                    // multicast address of the workspace
  uint32_t* flags;
  int rank, T, D;
  float eps, weight_bias;
  int vb, nvb;                 // rmsNormLamport's block size (threads) and blocks per token
  float* r32;                  // [T, D] or null: the residual stream kept in fp32 (decode steps)
  int r32_init;                // this call starts the stream from the bf16 residual
};

// FlashInfer's grid (per token D/8/128 CTAs of 128 threads, resident beside the preceding GEMM under programmatic
// launch): token t is scattered, reduced by rank t % WS (the fp32 sum over the ranks in rank order, rounded to bf16, -0
// sanitized) and multicast to every rank
__global__ void __launch_bounds__(128) ar_twoshot_kernel(const Params p) {
  const int t = blockIdx.x;
  const int off = (blockIdx.y * blockDim.x + threadIdx.x) * ELTS;
  const bool active = off < p.D;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const Flags flag(p.flags);
  const int owner = t % WS;
  __nv_bfloat16* scatter_dst = flag.cur_buf(p.uc_ptrs[owner], SCATTER);
  __nv_bfloat16* scatter_loc = flag.cur_buf(p.uc_local, SCATTER);
  __nv_bfloat16* bcast_w = flag.cur_buf(p.mc, BROADCAST);
  if (active) {
    const uint4 v = sanitize(*reinterpret_cast<const uint4*>(p.input + (int64_t)t * p.D + off));
    *reinterpret_cast<uint4*>(scatter_dst + ((int64_t)(t / WS) * WS + p.rank) * p.D + off) = v;
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  flag.clear_dirty(p.uc_local, SCATTER);
  if (owner == p.rank && active) {
    float acc[ELTS];
    while (true) {
      bool valid = true;
#pragma unroll
      for (int i = 0; i < ELTS; ++i) acc[i] = 0.f;
#pragma unroll
      for (int r = 0; r < WS; ++r) {
        const uint4 v = ld_volatile(scatter_loc + ((int64_t)(t / WS) * WS + r) * p.D + off);
        valid &= !is_dirty(v);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[2 * i] = add_ftz(acc[2 * i], bf_lo(w[i]));
          acc[2 * i + 1] = add_ftz(acc[2 * i + 1], bf_hi(w[i]));
        }
      }
      if (valid) break;
    }
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162 h = __floats2bfloat162_rn(add_ftz(0.f, acc[2 * i]), add_ftz(0.f, acc[2 * i + 1]));
      w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(bcast_w + (int64_t)t * p.D + off) = sanitize(make_uint4(w[0], w[1], w[2], w[3]));
  }
  flag.clear_dirty(p.uc_local, BROADCAST);
}

// ar_twoshot_kernel's programmatic dependent, one CTA per token, one thread per 8 elements: polls the broadcast row,
// adds the residual (bf16, or the fp32 stream of a decode step) and normalizes; the Lamport flags then rotate (its
// first CTA's thread 0) once ar_twoshot_kernel is done, as FlashInfer's kernels do
__global__ void __launch_bounds__(1024) ar_rms_kernel(const Params p) {
  __shared__ float warp_sums[32];
  __shared__ float vb_sums[8];
  const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int off = tid * ELTS;
  const bool active = off < p.D;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const Flags flag(p.flags);
  __nv_bfloat16* bcast_r = flag.cur_buf(p.uc_local, BROADCAST);
  uint32_t* count = p.flags + 8;
  // the residual (fp32 stream, or bf16) and gamma before the reduced row arrives
  float rr[ELTS];
  uint4 gv = make_uint4(0, 0, 0, 0);
  const bool stream32 = p.r32 != nullptr;
  if (active) {
    gv = *reinterpret_cast<const uint4*>(p.gamma + off);
    if (stream32 && !p.r32_init) {
      const float4* rp = reinterpret_cast<const float4*>(p.r32 + (int64_t)t * p.D + off);
      const float4 a = rp[0], b = rp[1];
      rr[0] = a.x; rr[1] = a.y; rr[2] = a.z; rr[3] = a.w; rr[4] = b.x; rr[5] = b.y; rr[6] = b.z; rr[7] = b.w;
    } else {
      const uint4 r = *reinterpret_cast<const uint4*>(p.residual + (int64_t)t * p.D + off);
      const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        rr[2 * i] = bf_lo(w[i]);
        rr[2 * i + 1] = bf_hi(w[i]);
      }
    }
  }
  // arrive early (as rmsNormLamport): the flags rotate once every CTA has read them
  __syncthreads();
  if (tid == 0) asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(count), "r"(1) : "memory");
  float x[ELTS];
  float thread_sum = 0.f;
  if (active) {
    uint4 v = ld_volatile(bcast_r + (int64_t)t * p.D + off);
    while (is_dirty(v)) v = ld_volatile(bcast_r + (int64_t)t * p.D + off);
    const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
    if (stream32) {  // r += reduced in fp32; the stream is the next call's residual (the bf16 tensor is not read)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[2 * i] = rr[2 * i] + bf_lo(vw[i]);
        x[2 * i + 1] = rr[2 * i + 1] + bf_hi(vw[i]);
      }
      float4* rp = reinterpret_cast<float4*>(p.r32 + (int64_t)t * p.D + off);
      rp[0] = make_float4(x[0], x[1], x[2], x[3]);
      rp[1] = make_float4(x[4], x[5], x[6], x[7]);
    } else {  // bf16: FlashInfer's PackedVec add, the residual output
      uint4 sum;
      __nv_bfloat162* sh = reinterpret_cast<__nv_bfloat162*>(&sum);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        sh[i] = __hadd2(__floats2bfloat162_rn(bf_lo(vw[i]), bf_hi(vw[i])),
                        __floats2bfloat162_rn(rr[2 * i], rr[2 * i + 1]));
      *reinterpret_cast<uint4*>(p.residual + (int64_t)t * p.D + off) = sum;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[2 * i] = __low2float(sh[i]);
        x[2 * i + 1] = __high2float(sh[i]);
      }
    }
    // rmsNormLamport's reduction: per thread 8 squares as an FMA chain, a warp xor-tree, per virtual block (its
    // blockSize from adjustGridConfig) a second xor-tree over the warp sums, the blocks' sums in block order
#pragma unroll
    for (int j = 0; j < ELTS; ++j) thread_sum = fma_ftz(x[j], x[j], thread_sum);
  }
  const float ws = warp_sum_ftz(thread_sum);
  if (lane == 0) warp_sums[warp] = ws;
  __syncthreads();
  const int nw = p.vb / 32, g = tid / p.vb;
  {
    float v = lane < nw ? warp_sums[g * nw + lane] : 0.f;
    v = warp_sum_ftz(v);
    if ((tid % p.vb) == 0) vb_sums[g] = v;
  }
  __syncthreads();
  float full = vb_sums[0];
  if (p.nvb > 1) {
    full = 0.f;
    for (int i = 0; i < p.nvb; ++i) full = add_ftz(full, vb_sums[i]);
  }
  const float rcp = rsqrt_approx_ftz(add_ftz(div_approx_ftz(full, (float)p.D), p.eps));
  if (active) {
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    uint32_t ow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float n0 = mul_ftz(mul_ftz(add_ftz(p.weight_bias, bf_lo(gw[i])), x[2 * i]), rcp);
      const float n1 = mul_ftz(mul_ftz(add_ftz(p.weight_bias, bf_hi(gw[i])), x[2 * i + 1]), rcp);
      const __nv_bfloat162 h = __floats2bfloat162_rn(n0, n1);
      ow[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(p.norm_out + (int64_t)t * p.D + off) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
  if (t == 0 && tid == 0) {
    // a programmatic dependent orders itself after its primary only by this wait (PTX griddepcontrol): the flags rotate
    // after ar_twoshot_kernel's last clear of the previous call's buffers, which follow the flags it read
    asm volatile("griddepcontrol.wait;" ::: "memory");
    while (*reinterpret_cast<volatile uint32_t*>(count) < gridDim.x) {
    }
    uint4* fp = reinterpret_cast<uint4*>(p.flags);
    fp[0] = make_uint4((flag.cur + 1) % 3, flag.cur, flag.bytes_per_buffer, NUM_STAGES);
    fp[1] = make_uint4((uint32_t)(((p.T + WS - 1) / WS * WS) * p.D * 2), (uint32_t)(p.T * p.D * 2), 0, 0);
    *count = 0;
  }
}


// rmsNormLamport's (blockSize, clusterSize) from FlashInfer's adjustGridConfig(numTokens, dim, 8, true), SM90; a block
// of partial warps rounds up to whole warps, the extra threads idle: the same squares summed in another order (with
// whole warps and the bf16 residual the outputs are bitwise the pair's)
static void norm_config(int T, int D, int sms, int& vb, int& nvb) {
  const int max_cluster = 8;
  int cluster = max_cluster;
  const int needed = (D + ELTS - 1) / ELTS;
  int block = (needed + cluster - 1) / cluster;
  while (needed % cluster != 0 && cluster > 1) cluster /= 2;
  const int max_div = cluster;
  block = (needed + cluster - 1) / cluster;
  while (block < 128 && cluster >= 2) {
    block *= 2;
    cluster /= 2;
  }
  while (T * cluster > sms && cluster > 1 && block <= 512) {
    block *= 2;
    cluster /= 2;
  }
  while (cluster < max_div) {
    const int cc = cluster * 2, cb = (needed + cc - 1) / cc;
    if (cb < 64 || T * cc > sms) break;
    cluster = cc;
    block = cb;
  }
  vb = (block + 31) / 32 * 32;
  nvb = cluster;
}

void ar_norm(const torch::Tensor& input, torch::Tensor& residual, const torch::Tensor& gamma, torch::Tensor& norm_out,
             int64_t uc_ptrs_dev, int64_t uc_local, int64_t mc, torch::Tensor& flags, int64_t rank, double eps,
             double weight_bias, const c10::optional<torch::Tensor>& r32, bool r32_init) {
  const int T = input.size(0), D = input.size(1);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 &&
                  gamma.scalar_type() == at::kBFloat16 && norm_out.scalar_type() == at::kBFloat16,
              "ar_norm: bf16");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() && gamma.is_contiguous() && norm_out.is_contiguous(),
              "ar_norm: contiguous");
  TORCH_CHECK(D % ELTS == 0 && D / ELTS <= 1024 && T >= 1, "ar_norm: shape");
  TORCH_CHECK(flags.scalar_type() == at::kInt && flags.numel() >= 9, "ar_norm: flags (uint32 as int32)");
  static int sms = 0;
  if (!sms) sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  Params p;
  p.input = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  p.residual = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
  p.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  p.norm_out = reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr());
  p.uc_ptrs = reinterpret_cast<void* const*>(uc_ptrs_dev);
  p.uc_local = reinterpret_cast<void*>(uc_local);
  p.mc = reinterpret_cast<void*>(mc);
  p.flags = reinterpret_cast<uint32_t*>(flags.data_ptr());
  p.rank = rank;
  p.T = T;
  p.D = D;
  p.eps = (float)eps;
  p.weight_bias = (float)weight_bias;
  norm_config(T, D, sms, p.vb, p.nvb);
  p.r32 = nullptr;
  p.r32_init = r32_init;
  if (r32.has_value() && r32->defined()) {
    TORCH_CHECK(r32->scalar_type() == at::kFloat && r32->is_contiguous() && r32->numel() >= (int64_t)T * D, "r32");
    p.r32 = r32->data_ptr<float>();
  }
  TORCH_CHECK(p.vb % 32 == 0 && p.vb * p.nvb >= D / ELTS && p.nvb <= 8, "ar_norm: norm config");
  cudaLaunchConfig_t c1 = {};
  c1.gridDim = dim3(T, (D / ELTS + 127) / 128);
  c1.blockDim = dim3(128);
  c1.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute a1[1];
  a1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a1[0].val.programmaticStreamSerializationAllowed = 1;
  c1.attrs = a1;
  c1.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&c1, ar_twoshot_kernel, p) == cudaSuccess, "ar_twoshot launch");
  cudaLaunchConfig_t c2 = c1;
  c2.gridDim = dim3(T);
  c2.blockDim = dim3(p.vb * p.nvb);
  TORCH_CHECK(cudaLaunchKernelEx(&c2, ar_rms_kernel, p) == cudaSuccess, "ar_rms launch");
}

}  // namespace allreduce

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "ar_norm(Tensor input, Tensor(a!) residual, Tensor gamma, Tensor(b!) norm_out, int uc_ptrs_dev, int uc_local, "
      "int mc, Tensor(c!) flags, int rank, float eps, float weight_bias, Tensor(d!)? r32=None, bool r32_init=False) "
      "-> ()");
}
TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("ar_norm", &allreduce::ar_norm); }
