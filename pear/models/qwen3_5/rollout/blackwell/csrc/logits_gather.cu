// The lm_head's fp32 logits over two tensor-parallel ranks: the fp32 hidden states split into the head's high and low
// bf16 rows, and the logits gathered fused with the sum of the two rows' products (every CTA writes its part of this
// rank's vocabulary shard into both ranks' [n, V] buffers over NVLink, then signals the peer's same-index CTA).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sync.cuh"

namespace {

struct LGState {
  float* local = nullptr;      // [max_rows, V] this rank's gathered logits
  float* peer = nullptr;       // the peer's buffer (UVA)
  uint32_t* flag_local = nullptr;  // [ctas] written by the peer
  uint32_t* flag_peer = nullptr;   // the peer's flags (UVA)
  uint32_t* epochs = nullptr;  // [ctas] local
  int max_rows = 0;
  int V = 0;
  int ctas = 0;
};
LGState g_lg;

constexpr int kThreads = 512;
constexpr int kSplitThreads = 256;
constexpr int kSplitCtasPerSm = 4;  // the split's CTAs per SM at most, a grid-stride loop past them

// x [n, H] fp32 -> out [2n, H] bf16: rows i the high parts (x rounded to nearest), rows n + i the low parts (the
// remainder, exact in fp32, rounded to nearest)
__global__ void __launch_bounds__(kSplitThreads) split_kernel(const float4* __restrict__ x, int64_t n4,
                                                              __nv_bfloat162* __restrict__ out) {
  for (int64_t i = blockIdx.x * (int64_t)kSplitThreads + threadIdx.x; i < n4; i += (int64_t)gridDim.x * kSplitThreads) {
    const float4 v = x[i];
    const __nv_bfloat162 high_xy = __floats2bfloat162_rn(v.x, v.y), high_zw = __floats2bfloat162_rn(v.z, v.w);
    const float2 xy = __bfloat1622float2(high_xy), zw = __bfloat1622float2(high_zw);
    out[2 * i] = high_xy;
    out[2 * i + 1] = high_zw;
    out[2 * (n4 + i)] = __floats2bfloat162_rn(v.x - xy.x, v.y - xy.y);
    out[2 * (n4 + i) + 1] = __floats2bfloat162_rn(v.z - zw.x, v.w - zw.y);
  }
}

// the flags only grow (per-CTA epochs, no reset); one buffer suffices: the peer's next gather comes after its next
// forward, whose all-reduces need this rank past this step's sampling
__global__ void __launch_bounds__(kThreads) gather_kernel(const float4* __restrict__ both, int n, int V2, int col0,
                                                          float* __restrict__ local, float* __restrict__ peer,
                                                          int V, uint32_t* __restrict__ flag_local,
                                                          uint32_t* __restrict__ flag_peer, uint32_t* __restrict__ epochs) {
  const int W = V2 / 4;  // float4 per shard row
  const int total = n * W;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < total; i += gridDim.x * kThreads) {
    const int r = i / W, c = i - r * W;
    float4 v = both[i];
    const float4 low = both[(size_t)n * W + i];
    v.x += low.x, v.y += low.y, v.z += low.z, v.w += low.w;
    const size_t o = (size_t)r * V + col0 + 4 * c;
    *reinterpret_cast<float4*>(local + o) = v;
    *reinterpret_cast<float4*>(peer + o) = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const uint32_t e = epochs[blockIdx.x] + 1;
    epochs[blockIdx.x] = e;
    st_release_sys(flag_peer + blockIdx.x, e);
    while (ld_acquire_sys(flag_local + blockIdx.x) < e) {
    }
  }
}

}  // namespace

// x [n, H] fp32 (contiguous) as the lm_head's [high rows; low rows] in bf16, [2n, H]
torch::Tensor hl_split(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == at::kFloat && x.size(1) % 4 == 0,
              "hl_split: contiguous fp32 [n, H], H a multiple of 4");
  auto out = torch::empty({2 * x.size(0), x.size(1)}, x.options().dtype(at::kBFloat16));
  const int64_t n4 = x.numel() / 4;
  if (n4 == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int64_t max_blocks = (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * kSplitCtasPerSm;
  const int blocks = (int)std::min<int64_t>((n4 + kSplitThreads - 1) / kSplitThreads, max_blocks);
  split_kernel<<<blocks, kSplitThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float4*>(x.data_ptr()), n4, reinterpret_cast<__nv_bfloat162*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void lg_setup(torch::Tensor buf, int64_t peer_buf, torch::Tensor flags, int64_t peer_flags, torch::Tensor epochs,
              int64_t ctas) {
  TORCH_CHECK(buf.dim() == 2 && buf.scalar_type() == at::kFloat && buf.is_contiguous());
  TORCH_CHECK(flags.numel() >= ctas && epochs.numel() >= ctas);
  g_lg.local = buf.data_ptr<float>();
  g_lg.peer = reinterpret_cast<float*>(peer_buf);
  g_lg.flag_local = reinterpret_cast<uint32_t*>(flags.data_ptr());
  g_lg.flag_peer = reinterpret_cast<uint32_t*>(peer_flags);
  g_lg.epochs = reinterpret_cast<uint32_t*>(epochs.data_ptr());
  g_lg.max_rows = buf.size(0);
  g_lg.V = buf.size(1);
  g_lg.ctas = ctas;
}

// both: [2n, V/2] fp32 (high rows, low rows), this rank's shard (columns rank * V/2 ...). Returns [n, V]: a view of the
// gather buffer, valid until the next call.
torch::Tensor lg_gather(torch::Tensor both, int64_t n, int64_t rank) {
  TORCH_CHECK(g_lg.local != nullptr, "lg_gather: lg_setup was not called");
  TORCH_CHECK(both.dim() == 2 && both.is_contiguous() && both.scalar_type() == at::kFloat);
  const int V2 = both.size(1);
  TORCH_CHECK(2 * V2 == g_lg.V && V2 % 4 == 0, "lg_gather: shard width");
  TORCH_CHECK(both.size(0) == 2 * n && n <= g_lg.max_rows, "lg_gather: rows");
  auto out = torch::from_blob(g_lg.local, {n, (int64_t)g_lg.V}, both.options());
  if (n == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(both.device());
  gather_kernel<<<g_lg.ctas, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float4*>(both.data_ptr()), (int)n, V2, (int)(rank * V2), g_lg.local, g_lg.peer,
      g_lg.V, g_lg.flag_local, g_lg.flag_peer, g_lg.epochs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("hl_split(Tensor x) -> Tensor");
  m.def("lg_setup(Tensor buf, int peer_buf, Tensor flags, int peer_flags, Tensor epochs, int ctas) -> ()");
  m.def("lg_gather(Tensor both, int n, int rank) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("hl_split", &hl_split);
  m.impl("lg_setup", &lg_setup);
  m.impl("lg_gather", &lg_gather);
}
