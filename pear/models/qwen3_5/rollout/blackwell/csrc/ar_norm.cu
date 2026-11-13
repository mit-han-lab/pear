// The all-reduce of a row-parallel GEMM's bf16 partial sums over two tensor-parallel ranks, fused with the fp32
// residual add and the (1 + w) RMSNorm: residual[t] += x[t] + x_peer[t], out[t] = residual[t] *
// rsqrt(mean(residual[t]^2) + eps) * (1 + w), in bf16 (or fp32).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

struct alignas(16) bf16x8 {
  __nv_bfloat162 v[4];
};

template <int NTHREADS>
__device__ __forceinline__ float block_sum(float v, float* smem) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  constexpr int NW = NTHREADS / 32;
  float t = lane < NW ? smem[lane] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  __syncthreads();
  return t;
}

// Lamport style: the receive buffers hold the sentinel -0.0 (0x8000) wherever nothing has arrived; a call pushes its
// rows into the peer's buffer [epoch % 2] (posted NVLink stores, no fence, no flag) and polls its own buffer until
// none of its elements is the sentinel. VPT: 8-element vectors per thread per row (H = NTHREADS * VPT * 8).
template <int NTHREADS, int VPT, bool OUT_F32, bool HAS_RES = true>
__global__ void __launch_bounds__(NTHREADS) ar_add_rmsnorm_kernel(
    const bf16x8* __restrict__ x,       // [T, H/8] local partial
    const float* residual,              // [T, H] fp32 in
    float* res_out,                     // [T, H] fp32 out (may alias residual)
    const __nv_bfloat16* __restrict__ w,  // [H]
    void* __restrict__ out,             // [T, H] bf16 or fp32
    uint4* __restrict__ recv_local,     // [2, maxT, H/8] this rank's receive buffers (the peer writes them)
    uint4* __restrict__ recv_peer,      // [2, maxT, H/8] the peer's receive buffers (this rank writes them)
    uint32_t* __restrict__ counters,    // [nctas] local epochs
    int T, int max_t, float eps) {
  constexpr int H8 = NTHREADS * VPT;  // row length in 8-vectors
  __shared__ float smem[32];
  __shared__ uint32_t s_epoch;
  // the epoch in device memory: no host scalar changes between CUDA graph replays
  if (threadIdx.x == 0) {
    uint32_t e = counters[blockIdx.x] + 1;
    counters[blockIdx.x] = e;
    s_epoch = e;
  }
  __syncthreads();
  const uint32_t epoch = s_epoch;
  const size_t cur = (size_t)(epoch & 1u) * max_t * H8;
  const uint4 sentinel = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);

  // push this CTA's rows to the peer (-0.0 -> +0.0: -0.0 is the sentinel)
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * H8);
    uint4* dst = recv_peer + cur + (size_t)t * H8;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      uint4 v = src[threadIdx.x + i * NTHREADS];
      uint32_t* u = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if ((u[j] & 0xffffu) == 0x8000u) u[j] &= 0xffff0000u;
        if ((u[j] >> 16) == 0x8000u) u[j] &= 0x0000ffffu;
      }
      dst[threadIdx.x + i * NTHREADS] = v;
    }
  }

  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const bf16x8* mine = x + (size_t)t * H8;
    uint4* theirs = recv_local + cur + (size_t)t * H8;
    const float* res = residual + (size_t)t * H8 * 8;
    float* res_o = res_out + (size_t)t * H8 * 8;
    float r[VPT][8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int c = threadIdx.x + i * NTHREADS;
      bf16x8 a = mine[c];
      uint4 braw;
      while (true) {  // volatile loads: the peer writes these lines over NVLink
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(braw.x), "=r"(braw.y), "=r"(braw.z), "=r"(braw.w)
                     : "l"(theirs + c));
        const uint32_t u[4] = {braw.x, braw.y, braw.z, braw.w};
        bool ready = true;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          ready &= ((u[j] & 0xffffu) != 0x8000u) && ((u[j] >> 16) != 0x8000u);
        if (ready) break;
      }
      // cleared by the thread that read it, so every row ever written is cleared whatever later calls' token counts;
      // the peer writes this buffer again two calls later, which needs this rank's next push (this call done)
      theirs[c] = sentinel;
      bf16x8 b = *reinterpret_cast<bf16x8*>(&braw);
      float4 r0 = make_float4(0.f, 0.f, 0.f, 0.f), r1 = r0;
      if constexpr (HAS_RES) {
        r0 = reinterpret_cast<const float4*>(res)[2 * c];
        r1 = reinterpret_cast<const float4*>(res)[2 * c + 1];
      }
      float rv[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 fa = __bfloat1622float2(a.v[j]);
        float2 fb = __bfloat1622float2(b.v[j]);
        r[i][2 * j] = rv[2 * j] + (fa.x + fb.x);
        r[i][2 * j + 1] = rv[2 * j + 1] + (fa.y + fb.y);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) ss += r[i][j] * r[i][j];
      reinterpret_cast<float4*>(res_o)[2 * c] = make_float4(r[i][0], r[i][1], r[i][2], r[i][3]);
      reinterpret_cast<float4*>(res_o)[2 * c + 1] = make_float4(r[i][4], r[i][5], r[i][6], r[i][7]);
    }
    ss = block_sum<NTHREADS>(ss, smem);
    const float rstd = rsqrtf(ss / (float)(H8 * 8) + eps);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int c = threadIdx.x + i * NTHREADS;
      const bf16x8 wv = reinterpret_cast<const bf16x8*>(w)[c];
      float y[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 fw = __bfloat1622float2(wv.v[j]);
        y[2 * j] = (r[i][2 * j] * rstd) * (1.f + fw.x);
        y[2 * j + 1] = (r[i][2 * j + 1] * rstd) * (1.f + fw.y);
      }
      if constexpr (OUT_F32) {
        float4* o = reinterpret_cast<float4*>(reinterpret_cast<float*>(out) + (size_t)t * H8 * 8);
        o[2 * c] = make_float4(y[0], y[1], y[2], y[3]);
        o[2 * c + 1] = make_float4(y[4], y[5], y[6], y[7]);
      } else {
        bf16x8 ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) ov.v[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
        reinterpret_cast<bf16x8*>(out)[(size_t)t * H8 + c] = ov;
      }
    }
  }
}

constexpr int kThreads = 320;  // H = 5120 = 320 threads x 2 vectors x 8
constexpr int kVpt = 2;

}  // namespace

// The fused all-reduce's process-wide state, set once by ar_setup and kept out of the ops' signatures, so compiled
// graphs never bake buffer addresses.
struct ARState {
  uint4* recv_local = nullptr;
  uint4* recv_peer = nullptr;
  uint32_t* counters = nullptr;
  int max_t = 0;
  int ctas = 0;
};
ARState g_ar;

void ar_setup(torch::Tensor recv, int64_t peer_recv_ptr, torch::Tensor counters, int64_t ctas) {
  TORCH_CHECK(recv.dim() == 3 && recv.size(0) == 2 && recv.scalar_type() == at::kBFloat16);
  TORCH_CHECK(counters.numel() >= ctas && counters.scalar_type() == at::kInt);
  g_ar.recv_local = reinterpret_cast<uint4*>(recv.data_ptr());
  g_ar.recv_peer = reinterpret_cast<uint4*>(peer_recv_ptr);
  g_ar.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  g_ar.max_t = recv.size(1);
  g_ar.ctas = ctas;
}

// x: this rank's bf16 partial [T, H]; residual: fp32 [T, H]. Returns (out, new residual): new residual = residual +
// x + x_peer, out = rmsnorm(new residual) * (1 + w), in bf16 (or fp32 with out_f32).
std::tuple<torch::Tensor, torch::Tensor> ar_add_rmsnorm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
                                                         double eps, bool out_f32) {
  const int T = x.size(0);
  const int H = x.size(1);
  TORCH_CHECK(H == kThreads * kVpt * 8, "ar_add_rmsnorm: hidden size ", H, " unsupported");
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kFloat &&
              weight.scalar_type() == at::kBFloat16);
  TORCH_CHECK(g_ar.recv_local != nullptr, "ar_add_rmsnorm: ar_setup was not called");
  TORCH_CHECK(T <= g_ar.max_t, "ar_add_rmsnorm: ", T, " tokens > buffer ", g_ar.max_t);
  auto out = torch::empty({T, H}, x.options().dtype(out_f32 ? at::kFloat : at::kBFloat16));
  auto res_out = torch::empty_like(residual);
  if (T == 0) return {out, res_out};
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // in place is fine too: each element is read before it is written by the same thread
  // a fixed grid (rows strided over it): every CTA takes part in every call, so the per-CTA epochs agree and a row
  // always maps to the same CTA; all CTAs must be co-resident (each spins on its peer CTA)
#define AR_LAUNCH(F32)                                                                                          \
  ar_add_rmsnorm_kernel<kThreads, kVpt, F32, true><<<g_ar.ctas, kThreads, 0, stream>>>(                            \
      reinterpret_cast<const bf16x8*>(x.data_ptr()), residual.data_ptr<float>(), res_out.data_ptr<float>(),        \
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), out.data_ptr(), g_ar.recv_local, g_ar.recv_peer,  \
      g_ar.counters, T, g_ar.max_t, (float)eps)
  if (out_f32)
    AR_LAUNCH(true);
  else
    AR_LAUNCH(false);
#undef AR_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, res_out};
}

// The first layer with each rank's partial embeddings (the vocab-parallel lookup without its all-reduce: every token's
// row is nonzero on one rank only, so the sum is exact): residual = x + x_peer (fp32), out = rmsnorm(residual) * (1 + w).
std::tuple<torch::Tensor, torch::Tensor> ar_rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  const int T = x.size(0);
  const int H = x.size(1);
  TORCH_CHECK(H == kThreads * kVpt * 8, "ar_rmsnorm: hidden size ", H, " unsupported");
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16);
  TORCH_CHECK(g_ar.recv_local != nullptr, "ar_rmsnorm: ar_setup was not called");
  TORCH_CHECK(T <= g_ar.max_t, "ar_rmsnorm: ", T, " tokens > buffer ", g_ar.max_t);
  auto out = torch::empty_like(x);
  auto res_out = torch::empty({T, H}, x.options().dtype(at::kFloat));
  if (T == 0) return {out, res_out};
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  ar_add_rmsnorm_kernel<kThreads, kVpt, false, false><<<g_ar.ctas, kThreads, 0, stream>>>(
      reinterpret_cast<const bf16x8*>(x.data_ptr()), nullptr, res_out.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), out.data_ptr(), g_ar.recv_local, g_ar.recv_peer,
      g_ar.counters, T, g_ar.max_t, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, res_out};
}

TORCH_LIBRARY(pear_rollout_blackwell, m) {
  m.def("ar_setup(Tensor recv, int peer_recv, Tensor counters, int ctas) -> ()");
  m.def("ar_add_rmsnorm(Tensor x, Tensor residual, Tensor weight, float eps, bool out_f32) -> (Tensor, Tensor)");
  m.def("ar_rmsnorm(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("ar_setup", &ar_setup);
  m.impl("ar_add_rmsnorm", &ar_add_rmsnorm);
  m.impl("ar_rmsnorm", &ar_rmsnorm);
}
