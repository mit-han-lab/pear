// The driver's cuTensorMapEncodeTiled for the TMA kernels, looked up once through the runtime (no libcuda link).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

inline PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    cudaDriverEntryPointQueryResult q;
    void* ptr = nullptr;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}
