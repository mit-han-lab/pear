// The vision tower's all-reduces of fp32 row-parallel partial sums over two tensor-parallel ranks, one-shot over
// symmetric memory like ar_norm.cu: fused with the residual add and the next LayerNorm (residual += x + x_peer, out =
// layernorm(residual) * w + b), or alone (out = x + x_peer), all in fp32. -0.0f marks "not arrived" in the receive
// buffers; a pushed -0.0f becomes +0.0f, which changes no sum and changes it the same way on both ranks.
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sync.cuh"

namespace vision_ar {

constexpr uint32_t kSentinel = 0x80000000u;  // -0.0f
constexpr int kWarps = 8;                    // rows in flight per CTA of the fused kernel: one warp per row
constexpr int kThreads = kWarps * 32;
constexpr int kCtasPerSm = 2;  // co-resident CTAs per SM: the grid is this many CTAs per SM (VisionAllReduceBuffers)
constexpr int kHidden4 = 288;  // the tower's hidden size (1152) in float4s: 9 per lane

// a value as pushed to the peer: -0.0f (the sentinel) as +0.0f
__device__ __forceinline__ float4 pushed(float4 v) {
  uint4 u = *reinterpret_cast<uint4*>(&v);
  u.x = u.x == kSentinel ? 0u : u.x;
  u.y = u.y == kSentinel ? 0u : u.y;
  u.z = u.z == kSentinel ? 0u : u.z;
  u.w = u.w == kSentinel ? 0u : u.w;
  return *reinterpret_cast<float4*>(&u);
}

// the peer's 16 bytes at p once all four of its words have arrived (volatile loads: the peer writes them over NVLink),
// then the sentinel back in their place: the peer writes this buffer again two calls later, after this rank's next push
__device__ __forceinline__ float4 take_arrived(float4* p) {
  uint4 r;
  while (true) {
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    if (r.x != kSentinel && r.y != kSentinel && r.z != kSentinel && r.w != kSentinel) break;
  }
  *reinterpret_cast<uint4*>(p) = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
  return *reinterpret_cast<float4*>(&r);
}

// this CTA's epoch of this call, in device memory: no host scalar changes between CUDA graph replays
__device__ __forceinline__ uint32_t next_epoch(uint32_t* counters) {
  __shared__ uint32_t s_epoch;
  if (threadIdx.x == 0) {
    const uint32_t e = counters[blockIdx.x] + 1;
    counters[blockIdx.x] = e;
    s_epoch = e;
  }
  __syncthreads();
  return s_epoch;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// residual[t] += x[t] + x_peer[t]; out[t] = (residual[t] - mean) * rstd * w + b. One warp per row, rows strided over a
// fixed grid of co-resident CTAs (each pushes all its rows before it waits for the peer's); launched as a programmatic
// dependent, the epoch read after the wait for the previous kernel.
__global__ void __launch_bounds__(kThreads, kCtasPerSm) ar_add_layernorm_kernel(
    const float4* __restrict__ x, const float4* __restrict__ residual, float4* __restrict__ res_out,
    const float4* __restrict__ w, const float4* __restrict__ b, float4* __restrict__ out, float4* recv_local,
    float4* recv_peer, uint32_t* counters, int T, size_t capacity4, float eps) {
  constexpr int V = kHidden4 / 32;
  const int lane = threadIdx.x % 32;
  const int first_row = blockIdx.x * kWarps + threadIdx.x / 32, stride = gridDim.x * kWarps;
  pdl_wait();
  pdl_launch_next();
  const uint32_t epoch = next_epoch(counters);
  float4* inbox = recv_local + (epoch & 1u) * capacity4;
  float4* peer_inbox = recv_peer + (epoch & 1u) * capacity4;

  for (int t = first_row; t < T; t += stride) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const size_t c = (size_t)t * kHidden4 + lane + 32 * i;
      peer_inbox[c] = pushed(x[c]);
    }
  }

  for (int t = first_row; t < T; t += stride) {
    float4 h[V];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const size_t c = (size_t)t * kHidden4 + lane + 32 * i;
      const float4 partial = x[c], r = residual[c];
      h[i] = add4(r, add4(partial, take_arrived(inbox + c)));
      res_out[c] = h[i];
      sum += (h[i].x + h[i].y) + (h[i].z + h[i].w);
    }
    const float mean = warp_sum(sum) / (float)(kHidden4 * 4);
    float squares = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float4 d = make_float4(h[i].x - mean, h[i].y - mean, h[i].z - mean, h[i].w - mean);
      squares += (d.x * d.x + d.y * d.y) + (d.z * d.z + d.w * d.w);
    }
    const float rstd = rsqrtf(warp_sum(squares) / (float)(kHidden4 * 4) + eps);
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const int k = lane + 32 * i;
      const float4 wk = __ldg(w + k), bk = __ldg(b + k);
      out[(size_t)t * kHidden4 + k] =
          make_float4(wk.x * (rstd * (h[i].x - mean)) + bk.x, wk.y * (rstd * (h[i].y - mean)) + bk.y,
                      wk.z * (rstd * (h[i].z - mean)) + bk.z, wk.w * (rstd * (h[i].w - mean)) + bk.w);
    }
  }
}

// out = x + x_peer over n4 float4s, strided over the same fixed grid
__global__ void __launch_bounds__(kThreads, kCtasPerSm) all_reduce_kernel(const float4* __restrict__ x,
                                                                         float4* __restrict__ out, float4* recv_local,
                                                                         float4* recv_peer, uint32_t* counters,
                                                                         size_t n4, size_t capacity4) {
  pdl_wait();
  pdl_launch_next();
  const uint32_t epoch = next_epoch(counters);
  float4* inbox = recv_local + (epoch & 1u) * capacity4;
  float4* peer_inbox = recv_peer + (epoch & 1u) * capacity4;
  const size_t first_index = (size_t)blockIdx.x * kThreads + threadIdx.x, stride = (size_t)gridDim.x * kThreads;
  for (size_t i = first_index; i < n4; i += stride) peer_inbox[i] = pushed(x[i]);
  for (size_t i = first_index; i < n4; i += stride) out[i] = add4(x[i], take_arrived(inbox + i));
}

// The process-wide state, set once by vision_ar_setup and kept out of the ops' signatures, so compiled graphs never
// bake buffer addresses.
struct State {
  float4* recv_local = nullptr;
  float4* recv_peer = nullptr;
  uint32_t* counters = nullptr;
  size_t capacity4 = 0;  // float4s of one rotation
  int ctas = 0;
};
State g_state;

template <typename Kernel, typename... Args>
void launch_dependent(Kernel kernel, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(g_state.ctas);
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

}  // namespace vision_ar

void vision_ar_setup(torch::Tensor recv, int64_t peer_recv_ptr, torch::Tensor counters, int64_t ctas) {
  using namespace vision_ar;
  TORCH_CHECK(recv.dim() == 2 && recv.size(0) == 2 && recv.size(1) % 4 == 0 && recv.scalar_type() == at::kFloat);
  TORCH_CHECK(counters.numel() >= ctas && counters.scalar_type() == at::kInt);
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, recv.get_device()));
  TORCH_CHECK(ctas <= (int64_t)sms * kCtasPerSm, "vision_ar_setup: ", ctas, " CTAs are not co-resident");
  g_state.recv_local = reinterpret_cast<float4*>(recv.data_ptr());
  g_state.recv_peer = reinterpret_cast<float4*>(peer_recv_ptr);
  g_state.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  g_state.capacity4 = recv.size(1) / 4;
  g_state.ctas = ctas;
}

// x: this rank's fp32 partial [T, 1152], residual: fp32 [T, 1152]. Returns (layernorm(new residual) * w + b, new
// residual), new residual = residual + (x + x_peer); calls larger than a rotation run in chunks of whole rows.
std::tuple<torch::Tensor, torch::Tensor> vision_ar_add_layernorm(torch::Tensor x, torch::Tensor residual,
                                                                 torch::Tensor weight, torch::Tensor bias,
                                                                 double eps) {
  using namespace vision_ar;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kHidden4 * 4, "vision_ar_add_layernorm: hidden size ", x.size(1));
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && residual.sizes() == x.sizes());
  TORCH_CHECK(x.scalar_type() == at::kFloat && residual.scalar_type() == at::kFloat &&
              weight.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat);
  TORCH_CHECK(g_state.recv_local != nullptr, "vision_ar_add_layernorm: vision_ar_setup was not called");
  const int T = x.size(0);
  auto out = torch::empty_like(x);
  auto res_out = torch::empty_like(residual);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = (int)(g_state.capacity4 / kHidden4);  // rows of one rotation
  for (int t0 = 0; t0 < T; t0 += rows) {
    const size_t offset = (size_t)t0 * kHidden4;
    launch_dependent(ar_add_layernorm_kernel, stream, reinterpret_cast<const float4*>(x.data_ptr<float>()) + offset,
                     reinterpret_cast<const float4*>(residual.data_ptr<float>()) + offset,
                     reinterpret_cast<float4*>(res_out.data_ptr<float>()) + offset,
                     reinterpret_cast<const float4*>(weight.data_ptr<float>()),
                     reinterpret_cast<const float4*>(bias.data_ptr<float>()),
                     reinterpret_cast<float4*>(out.data_ptr<float>()) + offset, g_state.recv_local, g_state.recv_peer,
                     g_state.counters, std::min(rows, T - t0), g_state.capacity4, (float)eps);
  }
  return {out, res_out};
}

// x: this rank's fp32 partial, any shape with a multiple of 4 elements. Returns x + x_peer.
torch::Tensor vision_all_reduce(torch::Tensor x) {
  using namespace vision_ar;
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kFloat && x.numel() % 4 == 0);
  TORCH_CHECK(g_state.recv_local != nullptr, "vision_all_reduce: vision_ar_setup was not called");
  auto out = torch::empty_like(x);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t n4 = x.numel() / 4;
  for (size_t i0 = 0; i0 < n4; i0 += g_state.capacity4) {
    launch_dependent(all_reduce_kernel, stream, reinterpret_cast<const float4*>(x.data_ptr<float>()) + i0,
                     reinterpret_cast<float4*>(out.data_ptr<float>()) + i0, g_state.recv_local, g_state.recv_peer,
                     g_state.counters, std::min(g_state.capacity4, n4 - i0), g_state.capacity4);
  }
  return out;
}

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("vision_ar_setup(Tensor recv, int peer_recv, Tensor counters, int ctas) -> ()");
  m.def(
      "vision_ar_add_layernorm(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps) -> (Tensor, Tensor)");
  m.def("vision_all_reduce(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("vision_ar_setup", &vision_ar_setup);
  m.impl("vision_ar_add_layernorm", &vision_ar_add_layernorm);
  m.impl("vision_all_reduce", &vision_all_reduce);
}
