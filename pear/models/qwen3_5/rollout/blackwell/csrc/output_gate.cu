// The output gate of a gated full-attention layer, out = x * sigmoid(gate) in bf16 with torch's roundings (the sigmoid
// rounded to bf16, then the product), as a programmatic dependent that lets the next kernel launch at once: o_proj's
// GEMM streams its weights while this kernel waits for the attention and gates its output.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "sync.cuh"

extern "C" __device__ float __nv_expf(float);  // libdevice's expf, torch's, whatever the fast-math flags

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;        // bf16 per 16-byte access
constexpr int kCtasPerSm = 2;  // the grid at most: every CTA resident at once, so the next kernel launches early

struct alignas(16) Bf16x8 {
  __nv_bfloat16 v[kVec];
};

// torch's bf16 sigmoid, 1 / (1 + exp(-g)) in fp32 with IEEE addition and division (no flush to zero), rounded to bf16
__device__ __forceinline__ float sigmoid_bf16(float g) {
  float d, s;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(1.f), "f"(__nv_expf(-g)));
  asm("div.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(1.f), "f"(d));
  return __bfloat162float(__float2bfloat16_rn(s));
}

// torch's bf16 product: in fp32 (no flush to zero), rounded to bf16
__device__ __forceinline__ __nv_bfloat16 mul_bf16(float a, float b) {
  float p;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(p) : "f"(a), "f"(b));
  return __float2bfloat16_rn(p);
}

__global__ void __launch_bounds__(kThreads) output_gate_kernel(const Bf16x8* __restrict__ x,
                                                               const __nv_bfloat16* __restrict__ gate,
                                                               int64_t gate_stride, Bf16x8* __restrict__ out, int rows,
                                                               int row_vecs) {
  pdl_launch_next();
  pdl_wait();

  const int total = rows * row_vecs;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < total; i += gridDim.x * kThreads) {
    const int r = i / row_vecs, c = i - r * row_vecs;
    const Bf16x8 xv = x[i];
    const Bf16x8 gv = *reinterpret_cast<const Bf16x8*>(gate + r * gate_stride + c * kVec);
    Bf16x8 o;
#pragma unroll
    for (int e = 0; e < kVec; e++)
      o.v[e] = mul_bf16(__bfloat162float(xv.v[e]), sigmoid_bf16(__bfloat162float(gv.v[e])));
    out[i] = o;
  }
}

// x [T, D] contiguous bf16, gate [T, D] bf16 with unit column stride: x * sigmoid(gate) as torch computes it
torch::Tensor output_gate(torch::Tensor x, torch::Tensor gate) {
  TORCH_CHECK(x.dim() == 2 && gate.dim() == 2 && x.sizes() == gate.sizes(), "output_gate: x and gate [T, D]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && gate.scalar_type() == torch::kBFloat16, "output_gate: bf16");
  TORCH_CHECK(x.is_contiguous() && gate.stride(1) == 1 && gate.stride(0) % kVec == 0 && x.size(1) % kVec == 0,
              "output_gate: rows of whole 16-byte vectors");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0, "output_gate: gate 16-byte aligned");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  const int rows = (int)x.size(0), row_vecs = (int)(x.size(1) / kVec);
  if (rows == 0) return out;

  static const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int needed = (rows * row_vecs + kThreads - 1) / kThreads;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(std::min(needed, kCtasPerSm * sm_count));
  config.blockDim = dim3(kThreads);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, output_gate_kernel, reinterpret_cast<const Bf16x8*>(x.data_ptr()),
                                    reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()), (int64_t)gate.stride(0),
                                    reinterpret_cast<Bf16x8*>(out.data_ptr()), rows, row_vecs));
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) { m.def("output_gate(Tensor x, Tensor gate) -> Tensor"); }

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("output_gate", &output_gate); }
