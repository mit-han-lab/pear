// Device helpers of the draft-proposal megakernel: mbarriers, bulk and TMA copies, tcgen05, dependency counters, and
// the numeric conversions with the lowerings of the kernels whose arithmetic the proposal reproduces.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace dk {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

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
// a < b (unsigned) as an all-ones / zero 64-bit mask (set.lt): no select against zero (ptxas: a CS2R zero, then a
// predicated write, read sooner than the zero lands)
__device__ __forceinline__ uint64_t lt_mask64(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("set.lt.u32.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return ((uint64_t)r << 32) | r;
}
// b, hidden from the compiler (a loop testing it is not split into a copy per value: one code path, warm for both)
__device__ __forceinline__ bool opaque_flag(bool b) {
  uint32_t v = b;
  asm volatile("" : "+r"(v));
  return v != 0;
}
// shared-memory loads / stores by 32-bit shared addresses (through the generic pointers of a struct the compiler
// emits generic LD / ST: an address-space check on every access)
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ uint32_t lds16(const void* p) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ void sts128(void* p, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(smem_u32(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// n arrivals at once
__device__ __forceinline__ void mbar_arrive_n(uint64_t* bar, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(n) : "memory");
}
// an fp32 add to global memory (red.global: the generic atomicAdd checks the address space first)
__device__ __forceinline__ void red_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
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

// ---- bulk copies global -> shared, completing on an mbarrier ----
__device__ __forceinline__ void bulk_g2s(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(smem)),
               "l"(gmem), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}

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

// 16-byte asynchronous copies global -> shared (L2 only), their groups
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// ldmatrix fragments (x4, plain / transposed) and the warp-level bf16 MMA (mma.sync m16n8k16, fp32 accumulators)
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
// D = A (16x16 bf16, row) x B (16x8 bf16, col) + D (fp32)
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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
// a named barrier of nthreads threads (whole warps): bar.sync is aligned, so the warp converges first (bar.warp.sync),
// as the host's named_sync (csrc/common.cuh)
__device__ __forceinline__ void named_sync(int id, int nthreads) {
  asm volatile("bar.warp.sync -1;\n\tbar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// ---- tcgen05 (5th-generation tensor cores, tensor memory) ----
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_mma_bf16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

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
// whether a counter has not yet reached its target, in serial-number order: the counters and their epoch-scaled
// targets ((epoch + 1) * count, never reset) wrap at 2^32, and within a launch a counter is at most a few hundred
// arrivals from its target, so the signed difference orders them across the wrap
__device__ __forceinline__ bool ctr_before(uint32_t v, uint32_t target) { return (int32_t)(v - target) < 0; }
// spin with relaxed loads until *p >= target, then one acquiring load of the counter (it synchronizes with every
// red.release of the count: they form one release sequence); a fence.acq_rel.gpu instead would also wait until the
// SM's in-flight TMA loads have landed
__device__ __forceinline__ void wait_counter(const uint32_t* p, uint32_t target) {
  while (ctr_before(ld_relaxed(p), target)) {
  }
  (void)ld_acquire(p);
}
__device__ __forceinline__ void red_relaxed_add(uint32_t* p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---- numeric conversions ----
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
// Triton's lowerings (the PTX of vLLM's kernels): fp32 '/' = div.full, tl.rsqrt = rsqrt.approx.ftz
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_tri(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// x / (1 + exp(-x)) as the Triton kernel of vLLM's compiled draft computes it: libdevice expf, div.full.f32
__device__ __forceinline__ float silu_f32(float x) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(1.0f + expf(-x)));
  return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}


// ---- TMEM words, swizzled tiles, fixed-point sums, atomics ----
// 8 consecutive TMEM columns of this lane (one fp32 accumulator per token column)
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float (&v)[8]) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tc_st4_u32(uint32_t taddr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// the completion of the thread's TMEM stores (a tcgen05.st issued without its wait)
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_ld4_u32(uint32_t taddr, uint32_t (&v)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(taddr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// byte offset of 16-byte chunk c of row r in a 128-byte-swizzled K-major tile (8-row atoms of 1024 bytes)
__device__ __forceinline__ uint32_t sw128(uint32_t r, uint32_t c) { return (r >> 3) * 1024u + (r & 7u) * 128u + ((c ^ (r & 7u)) << 4); }
// shared-memory matrix descriptor of a 128-byte-swizzled K-major tile (SBO 1024: 8-row atoms), sm_100 encoding
__device__ __forceinline__ uint64_t sw128_desc(const void* tile) {
  constexpr uint32_t ADDR_MASK = 0x3FFFu;           // bits 0-13: the start address in 16-byte units
  constexpr uint32_t SBO_FIELD = (1024u >> 4);      // bits 32-45: the stride byte offset (an 8-row atom), 16-byte units
  constexpr uint32_t VERSION_FIELD = (1u << 14);    // bits 46-48: the descriptor version, 1
  constexpr uint32_t SWIZZLE_FIELD = (2u << 29);    // bits 61-63: the layout, the 128-byte swizzle
  const uint32_t lo_word = (smem_u32(tile) >> 4) & ADDR_MASK;
  const uint32_t hi_word = SBO_FIELD | VERSION_FIELD | SWIZZLE_FIELD;
  return ((uint64_t)hi_word << 32) | lo_word;
}
// The sums of squares exact in any order: int64 fixed point at 2^-16 (a row's 5120 squares stay below 2^47)
constexpr float SSQ_ONE = 65536.0f;  // 2^16
__device__ __forceinline__ void red_add_ssq(long long* p, float v) {
  asm volatile("red.global.add.u64 [%0], %1;" ::"l"(p), "l"(__float2ll_rn(v * SSQ_ONE)) : "memory");
}
__device__ __forceinline__ float ssq_val(long long s) { return __ll2float_rn(s) * (1.0f / SSQ_ONE); }
// four fp32 atomically added to global memory (no return)
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
// atomic add with acquire-release semantics at gpu scope; returns the old value
__device__ __forceinline__ uint32_t atom_add_acq_rel(uint32_t* p, uint32_t v) {
  uint32_t old_value;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old_value) : "l"(p), "r"(v) : "memory");
  return old_value;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

}  // namespace dk
