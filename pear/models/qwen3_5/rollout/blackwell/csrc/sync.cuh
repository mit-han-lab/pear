// The kernels' ordering with other grids: programmatic dependent launch on this GPU, and the system-scope release and
// acquire of the flags the two tensor-parallel GPUs exchange over NVLink.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// wait for the previous kernel's completion and its writes
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// let the next kernel, a programmatic dependent, launch
__device__ __forceinline__ void pdl_launch_next() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

}  // namespace
