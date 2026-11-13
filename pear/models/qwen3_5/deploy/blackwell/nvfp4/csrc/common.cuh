// Device helpers of the decode megakernel: mbarriers, bulk and TMA copies, tcgen05, dependency counters, and the
// numeric conversions with the lowerings of the kernels whose arithmetic the megakernel reproduces.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace mk {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// stores to a 32-bit shared::cta address (a base held in a register: through a generic pointer ptxas may rebuild the
// window's base, S2R SR_CgaCtaId and its chain, at every store)
__device__ __forceinline__ void sts_u8(uint32_t a, uint32_t v) { asm volatile("st.shared.u8 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void sts_u16(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"((uint16_t)v) : "memory");
}
__device__ __forceinline__ void sts_u32(uint32_t a, uint32_t v) { asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void sts_f32(uint32_t a, float v) { asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory"); }
__device__ __forceinline__ void sts_f4(uint32_t a, float4 v) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
// loads from a 32-bit shared::cta address (volatile: kept in order with the barrier waits around them)
__device__ __forceinline__ float4 lds_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds_u4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds_u2(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}

// ---- mbarriers (CTA scope) ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// Waits for the completion of the phase with the given parity. The caller must never be more than one phase ahead of
// the barrier (a parity wait cannot tell phase k from phase k-2).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
// non-blocking: has the phase with this parity completed?
__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity) {
  uint32_t p;
  asm volatile(
      "{\n .reg .pred P;\n mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.b32 %0, 1, 0, P;\n}\n"
      : "=r"(p)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return p != 0;
}

// ---- the CTA pair (a cluster of 2 CTAs): the other CTA's shared memory, written by asynchronous stores (st.async:
// bytes completed on the receiving CTA's mbarrier) with relaxed arrivals, since a cluster-scope release fence is a
// GPU-scope membar on sm_103a, which waits for every load in flight on the SM (the weight stream) ----
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// the other CTA's shared::cluster address of this CTA's shared-memory address a (both CTAs have the same layout)
__device__ __forceinline__ uint32_t peer_addr(uint32_t a, uint32_t peer) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(peer));
  return r;
}
__device__ __forceinline__ void st_async_b32(uint32_t raddr, uint32_t v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr), "r"(v), "r"(rbar)
               : "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t raddr, uint4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(raddr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rbar)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive_peer(uint32_t rbar) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(rbar) : "memory");
}
// a phase completed by the other CTA's asynchronous stores (their complete_tx releases at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
// every thread of both CTAs (the kernel's start and end only)
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// ---- bulk copies global -> shared, completing on an mbarrier ----
__device__ __forceinline__ void bulk_g2s(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(smem)),
               "l"(gmem), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
// ---- bulk copies shared -> global (bulk async-group): issue, commit, wait until the source smem has been read ----
__device__ __forceinline__ void bulk_s2g(void* gmem, const void* smem, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gmem), "r"(smem_u32(smem)), "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }

// The weight and KV streams are read once per step: loaded with an L2 evict-first policy, they do not push the small,
// reused data out of L2 (the kernel's code, norm weights, accumulators, counters)
__device__ __forceinline__ uint64_t l2_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// TMA tensor-map load (2D box: {x elements of dim 0, y rows}) global -> shared with an L2 cache policy
__device__ __forceinline__ void tma_load_2d_pol(void* smem, const void* tmap, int x, int y, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], "
      "%5;" ::"r"(smem_u32(smem)),
      "l"(tmap), "r"(x), "r"(y), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s_pol(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
                   smem_u32(smem)),
               "l"(gmem), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
               : "memory");
}

// position in a ring of `size` slots: slot index and the parity of the slot's current use
struct RingPos {
  uint32_t n = 0, slot = 0, phase = 0;
  __device__ __forceinline__ void next(uint32_t size) {
    n++;
    if (++slot == size) {
      slot = 0;
      phase ^= 1;
    }
  }
  __device__ __forceinline__ void advance(uint32_t k, uint32_t size) {  // k positions
    n += k;
    slot += k;
    phase ^= (slot / size) & 1;
    slot %= size;
  }
};

// one lane of a converged warp (for single-thread tcgen05 / TMA issue from warp-uniform code)
__device__ __forceinline__ bool elect_one() {
  uint32_t p;
  asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.b32 %0, 1, 0, P;\n}\n" : "=r"(p));
  return p != 0;
}
__device__ __forceinline__ void named_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory"); }

// ---- tcgen05 (5th-generation tensor cores, tensor memory) ----
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_mma_nvf4(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t acc,
                                            uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
      : "memory");
}
__device__ __forceinline__ void tc_mma_f8(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
__device__ __forceinline__ void tc_mma_bf16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
// the same with A from tensor memory (a parked stage: 128 lanes = rows, 8 columns per 32-byte k-step)
__device__ __forceinline__ void tc_mma_nvf4_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t acc,
                                               uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, %3, [%5], [%6], p;\n}\n" ::"r"(d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
      : "memory");
}
// 32 rows x 16 bytes of shared memory into 4 TMEM columns of every lane quadrant (row i -> lane 32 q + i of each q):
// a 512-byte scale atom ([32 lanes][4 row groups][4 blocks], vLLM's swizzled layout) lands as the MMA reads it
// (datapath q: column + q of its lanes); executed in issue order with the thread's other tcgen05 operations
__device__ __forceinline__ void tc_cp_32x128b_x4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
// descriptor of a no-swizzle [32 rows][16 bytes] shared-memory matrix (core matrices of 8 rows x 16 B, SBO 128 B)
__device__ __forceinline__ uint64_t sf_desc(const void* p) {
  constexpr uint32_t SBO_FIELD = 128 >> 4;      // bits 32-45: the stride byte offset in 16-byte units
  constexpr uint32_t VERSION_FIELD = 1u << 14;  // bits 46-48: the descriptor version, 1
  constexpr uint32_t ADDR_MASK = 0x3FFF;        // bits 0-13: the start address in 16-byte units
  return ((uint64_t)(SBO_FIELD | VERSION_FIELD) << 32) | ((smem_u32(p) >> 4) & ADDR_MASK);
}
__device__ __forceinline__ void tc_mma_f8_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
// 128 rows x 32 bytes of a shared-memory matrix (descriptor: e.g. one k-step of an SW128 K-major tile) into 8 TMEM
// columns (row r -> lane r); executed in issue order with the issuing thread's other tcgen05 operations
__device__ __forceinline__ void tc_cp_128x256b(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// 8 consecutive accumulator columns of this thread's TMEM lane (a warp's own lane quadrant)
__device__ __forceinline__ void tc_ld_x8(uint32_t taddr, float (&v)[8]) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// 16 bytes global -> shared through the LSU (cp.async, L2 only), and the wait for this thread's copies
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// ---- cross-CTA dependency counters ----
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// 8-byte relaxed accesses at gpu scope: single-copy atomic (aligned), for tagged words {data, token}
__device__ __forceinline__ uint64_t ld_relaxed_b64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_b64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// 4-byte tagged words {bf16, 16-bit token} (in_proj_z -> out_proj's gate): a vector load is one access per element
__device__ __forceinline__ void st_relaxed_b32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_v4(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
// 8 bytes from L2 (strong: never a stale L1 line), for validated reads of data written by other CTAs
__device__ __forceinline__ uint2 ld_relaxed_v2(const void* p) {
  uint2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
// whether a counter has not yet reached its target, in serial-number order: the counters and their epoch-scaled
// targets ((epoch + 1) * count, never reset) wrap at 2^32, and within a launch a counter is at most a few hundred
// arrivals from its target, so the signed difference orders them across the wrap
__device__ __forceinline__ bool ctr_before(uint32_t v, uint32_t target) { return (int32_t)(v - target) < 0; }
// spin with relaxed loads until *p >= target, then one acquiring load of the counter (it synchronizes with every
// red.release of the count: they form one release sequence); a fence.acq_rel.gpu instead would wait for the SM's
// in-flight TMA loads (the weight stream) to land
__device__ __forceinline__ void wait_counter(const uint32_t* p, uint32_t target) {
  while (ctr_before(ld_relaxed(p), target)) {
  }
  (void)ld_acquire(p);
}
// relaxed polling only (the consumer validates the data itself)
__device__ __forceinline__ void wait_counter_relaxed(const uint32_t* p, uint32_t target) {
  while (ctr_before(ld_relaxed(p), target)) {
  }
}
__device__ __forceinline__ void red_relaxed_add(uint32_t* p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// Explicitly global atomics for the GEMV epilogues' accumulators (atomicAdd's semantics: relaxed, gpu scope; f32: ftz,
// round to nearest): through a generic address the compiler checks the address space after each atomic (a branch, then
// a shared-memory fallback), so each waits for its L2 round trip before the next is issued (T rows: T round trips)
__device__ __forceinline__ void red_add_f32_global(float* p, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
// (predicated: no branch around it, so that a row loop issues every row's atomic before the first result is used; r
// keeps its value when !is_enabled)
__device__ __forceinline__ void atom_add_u64_global(unsigned long long* p, unsigned long long v, bool is_enabled,
                                                    unsigned long long& r) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q atom.relaxed.gpu.global.add.u64 %0, [%1], %2;\n}"
               : "+l"(r)
               : "l"(p), "l"(v), "r"((uint32_t)is_enabled)
               : "memory");
}
__device__ __forceinline__ void red_release_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// does any of the 4 bf16 values hold the sentinel 0xFFFF
__device__ __forceinline__ bool has_sent4(uint2 v) {
  return (~v.x & 0xffffu) == 0u || (~v.x >> 16) == 0u || (~v.y & 0xffffu) == 0u || (~v.y >> 16) == 0u;
}

// ---- numeric conversions ----
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
// packed fp32 pairs (FADD2 / FMUL2 / FFMA2: per lane the scalar IEEE operation, round to nearest even) on two scalars
__device__ __forceinline__ void add2(float& a, float& b, float c, float d) {
  const float2 r = __fadd2_rn(make_float2(a, b), make_float2(c, d));
  a = r.x, b = r.y;
}
__device__ __forceinline__ void mul2(float& a, float& b, float c, float d) {
  const float2 r = __fmul2_rn(make_float2(a, b), make_float2(c, d));
  a = r.x, b = r.y;
}
// a = x * u + a, b = y * v + b
__device__ __forceinline__ void fma2(float& a, float& b, float x, float y, float u, float v) {
  const float2 r = __ffma2_rn(make_float2(x, y), make_float2(u, v), make_float2(a, b));
  a = r.x, b = r.y;
}
// float(a bf16 half of w) + c in one instruction (FHADD.BF16: the exact conversion, one rounding)
__device__ __forceinline__ float bf16lo_add(uint32_t w, float c) {
  float d;
  asm("{\n .reg .b16 l, h;\n mov.b32 {l, h}, %1;\n add.rn.f32.bf16 %0, l, %2;\n}" : "=f"(d) : "r"(w), "f"(c));
  return d;
}
__device__ __forceinline__ float bf16hi_add(uint32_t w, float c) {
  float d;
  asm("{\n .reg .b16 l, h;\n mov.b32 {l, h}, %1;\n add.rn.f32.bf16 %0, h, %2;\n}" : "=f"(d) : "r"(w), "f"(c));
  return d;
}
// fma(float(a bf16 half of a), float(the same half of b), c) in one instruction (FHFMA.BF16)
__device__ __forceinline__ float bf16lo_fma(uint32_t a, uint32_t b, float c) {
  float d;
  asm("{\n .reg .b16 al, ah, bl, bh;\n mov.b32 {al, ah}, %1;\n mov.b32 {bl, bh}, %2;\n fma.rn.f32.bf16 %0, al, bl, %3;\n}"
      : "=f"(d) : "r"(a), "r"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float bf16hi_fma(uint32_t a, uint32_t b, float c) {
  float d;
  asm("{\n .reg .b16 al, ah, bl, bh;\n mov.b32 {al, ah}, %1;\n mov.b32 {bl, bh}, %2;\n fma.rn.f32.bf16 %0, ah, bh, %3;\n}"
      : "=f"(d) : "r"(a), "r"(b), "f"(c));
  return d;
}
// the per-half unsigned max of two words (16-bit lanes)
__device__ __forceinline__ uint32_t max_u16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float ld_bf16(const void* p, size_t i) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
}
// e2m1x2 from two floats (first -> low nibble), round-to-nearest-even, saturating: as in vLLM's cvt_fp16_to_fp4
__device__ __forceinline__ uint32_t f32x2_to_e2m1x2(float lo, float hi) {
  uint32_t r;
  asm("{\n .reg .b8 q;\n cvt.rn.satfinite.e2m1x2.f32 q, %2, %1;\n cvt.u32.u8 %0, q;\n}" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
// e4m3 codes of two floats (first -> low byte), round-to-nearest-even, saturating
__device__ __forceinline__ uint32_t f32x2_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}
// Triton's lowerings (the PTX of vLLM's kernels): fp32 '/' = div.full, tl.exp = ex2.approx of x * log2(e),
// tl.sqrt / tl.rsqrt = approx.ftz, tl.sigmoid = 1 / (1 + tl.exp(0 - x)) with the mul and add not contracted
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
constexpr float LOG2E_F32 = 1.44269504f;  // log2(e) in fp32 (0f3FB8AA3B): tl.exp's factor
__device__ __forceinline__ float exp_tri(float x) {
  float m, e;
  asm("mul.f32 %0, %1, %2;" : "=f"(m) : "f"(x), "f"(LOG2E_F32));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(m));
  return e;
}
__device__ __forceinline__ float sqrt_tri(float x) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float rsqrt_tri(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float sigmoid_triton(float x) {
  float m, e;
  asm("mul.f32 %0, %1, %2;" : "=f"(m) : "f"(0.0f - x), "f"(LOG2E_F32));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(m));
  float d;
  asm("add.f32 %0, %1, %2;" : "=f"(d) : "f"(e), "f"(1.0f));
  return div_full(1.0f, d);
}
// x / (1 + exp(-x)) as the Triton kernel of vLLM's Inductor-compiled model computes it: libdevice expf, div.full.f32
__device__ __forceinline__ float silu_f32(float x) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(1.0f + expf(-x)));
  return r;
}

// a < b as an all-ones / zero mask (set.lt): no select against zero (ptxas: a CS2R zero, then a predicated write, read
// sooner than the zero lands)
__device__ __forceinline__ uint32_t lt_mask(int a, int b) {
  uint32_t r;
  asm("set.lt.u32.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// a != b (or unordered) as an all-ones / zero mask (set.neu), the same reason
__device__ __forceinline__ uint32_t ne_mask(float a, float b) {
  uint32_t r;
  asm("set.neu.u32.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// two warp_sums at once (FADD2 lanes, each in warp_sum's order)
__device__ __forceinline__ void warp_sum2(float& a, float& b) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) add2(a, b, __shfl_xor_sync(0xffffffffu, a, o), __shfl_xor_sync(0xffffffffu, b, o));
}

}  // namespace mk
