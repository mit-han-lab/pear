// The PTX of the kernels' tensor-core pipelines on this GPU: the mbarriers their asynchronous copies and MMAs complete
// on, the TMA's copies between global and shared memory, the tcgen05 MMAs, fences and tensor-memory accesses; and the
// driver's tensor-map encoder on the host. Shared-memory data are 32-bit shared addresses, barriers pointers.
#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <cstdint>

namespace {

// a shared::cluster address's CTA-rank bit cleared: the CTA pair leader's (cta_group::2)
constexpr uint32_t kLeaderBarrierMask = 0xFEFFFFFF;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- mbarriers ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
// the barriers' initialization visible to the async proxy and the cluster, before copies and commits complete on them
__device__ __forceinline__ void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// ---- TMA ----
__device__ __forceinline__ uint64_t l2_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t l2_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// a box into this CTA's shared memory, its bytes completing on bar; PAIR (cta_group::2): on the pair leader's barrier
// at the same offset (the leader's MMA reads both CTAs' halves)
template <bool PAIR = false>
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar,
                                            uint64_t policy) {
  if constexpr (PAIR)
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar) & kLeaderBarrierMask), "l"(policy)
        : "memory");
  else
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar)), "l"(policy)
        : "memory");
}
template <bool PAIR = false>
__device__ __forceinline__ void tma_load_4d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                            uint64_t* bar, uint64_t policy) {
  if constexpr (PAIR)
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar) & kLeaderBarrierMask), "l"(policy)
        : "memory");
  else
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar)), "l"(policy)
        : "memory");
}
// bytes of global memory into this CTA's shared memory, completing on bar
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
// shared -> global through a tensor map (bulk async group), then the wait until shared memory has been read
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(map), "r"(c0),
               "r"(c1), "r"(src)
               : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(map), "r"(c0),
               "r"(c1), "r"(c2), "r"(src)
               : "memory");
}
__device__ __forceinline__ void tma_store_commit_wait() {
  asm volatile("cp.async.bulk.commit_group;\n cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tma_prefetch_map(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// ---- tcgen05 ----
__device__ __forceinline__ bool elect_one() {
  uint32_t p;
  asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.b32 %0, 1, 0, P;\n}\n" : "=r"(p));
  return p != 0;
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// cols columns of tensor memory for the CTA (a pair: for both), their address into *slot (a whole warp)
template <bool PAIR = false>
__device__ __forceinline__ void tc_alloc(uint32_t* slot, uint32_t cols) {
  if constexpr (PAIR) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  } else {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}
template <bool PAIR = false>
__device__ __forceinline__ void tc_dealloc(uint32_t tmem, uint32_t cols) {
  if constexpr (PAIR)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(cols));
  else
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(cols));
}
// D (TMEM) [+]= A x B, both from shared-memory descriptors
template <bool PAIR = false>
__device__ __forceinline__ void tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                       uint32_t accumulate) {
  if constexpr (PAIR)
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
  else
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
}
// D (TMEM) [+]= A (TMEM: 128 lanes = rows, 8 columns per 16-element k-step) x B (smem descriptor)
__device__ __forceinline__ void tc_mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc,
                                          uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}
// the arrival on bar once this thread's MMAs completed; a pair's leader: on the barrier at bar's offset in both CTAs
template <bool PAIR = false>
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  if constexpr (PAIR)
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
            smem_u32(bar)),
        "h"((uint16_t)0x3)
        : "memory");
  else
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
                 : "memory");
}
// 16 or 32 consecutive columns of this thread's TMEM lane (its warp's lane quadrant)
__device__ __forceinline__ void tc_ld_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tc_ld_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ void tc_st_x32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,"
      "%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// a 128-byte-swizzled smem matrix: K-major (SBO 1024 = 8 rows of 128 bytes) or MN-major (lbo: the next 64-element
// chunk along MN; SBO 1024 = 8 rows along K)
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr, uint32_t lbo) {
  constexpr uint32_t SBO_FIELD = 1024u >> 4;    // bits 32-45: the stride byte offset in 16-byte units
  constexpr uint32_t VERSION_FIELD = 1u << 14;  // bits 46-48: the descriptor version, 1
  constexpr uint32_t SW128_FIELD = 2u << 29;    // bits 61-63: the 128-byte swizzle
  constexpr uint32_t ADDR_MASK = 0x3FFFu;       // bits 0-13: the start address in 16-byte units
  constexpr uint32_t LBO_SHIFT = 16;            // bits 16-29: the leading byte offset in 16-byte units
  return ((uint64_t)(SBO_FIELD | VERSION_FIELD | SW128_FIELD) << 32) |
         (((addr >> 4) & ADDR_MASK) | ((lbo >> 4) << LBO_SHIFT));
}

// ---- host ----
// the driver's tensor-map encoder, looked up once
PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    cudaDriverEntryPointQueryResult q;
    void* ptr = nullptr;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

}  // namespace
