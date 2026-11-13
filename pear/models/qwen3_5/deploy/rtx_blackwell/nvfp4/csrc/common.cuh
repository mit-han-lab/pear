// Device helpers of the decode megakernel on sm_120a: mbarriers, bulk and TMA copies, warp-level MMAs (mma.sync,
// ldmatrix), dependency counters, and the numeric conversions with the lowerings of the kernels whose arithmetic the
// megakernel reproduces.
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

// a published operand's byte b (0..127) of token row t within its K chunk: the B slot's swizzle (16-byte unit u at
// u ^ t), so that the consumer's B build is a bulk copy of whole chunks
__host__ __device__ __forceinline__ uint32_t pub_byte(uint32_t t, uint32_t b) {
  return t * 128 + ((((b >> 4) ^ t) & 7u) << 4) + (b & 15u);
}

// ---- mbarriers (CTA scope) ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
// (before its memory serves any other purpose: PTX requires it of an mbarrier in reused shared memory)
__device__ __forceinline__ void mbar_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
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
// TMA tensor-map load (2D box: {x elements of dim 0, y rows}) global -> shared with an L2 cache policy (the
// .shared::cta destination: .shared::cluster costs a cluster-address check with a call and a stack frame per site)
__device__ __forceinline__ void tma_load_2d_pol(void* smem, const void* tmap, int x, int y, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], "
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

__device__ __forceinline__ void named_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory"); }

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
// 8-byte relaxed accesses at gpu scope: single-copy atomic (aligned), for tagged words {data, token}
__device__ __forceinline__ uint64_t ld_relaxed_b64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_b64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// whether a counter has not yet reached its target, in serial-number order: the counters and their epoch-scaled
// targets ((epoch + 1) * count, never reset) wrap at 2^32, and within a launch a counter is at most a few hundred
// arrivals from its target, so the signed difference orders them across the wrap
__device__ __forceinline__ bool ctr_before(uint32_t v, uint32_t target) {
  return (int32_t)(v - target) < 0;
}
// spin with relaxed loads until *p >= target, then one acquiring load of the counter (it synchronizes with every
// red.release of the count: they form one release sequence); a fence.acq_rel.gpu instead would wait for the SM's
// in-flight TMA loads (the weight stream) to land
__device__ __forceinline__ void wait_counter(const uint32_t* p, uint32_t target) {
  while (ctr_before(ld_relaxed(p), target)) {
  }
  (void)ld_acquire(p);
}
// two tagged words to L2 (each 8-byte word single-copy atomic: its value and tag together), and back (strong)
__device__ __forceinline__ void st_cg_v2_u64(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.global.cg.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld_relaxed_v2_u64(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t atom_relaxed_add(uint32_t* p, uint32_t v) {
  uint32_t r;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}
__device__ __forceinline__ void red_release_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---- numeric conversions ----
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
// packed fp32 pairs (FMUL2 / FFMA2: per lane the scalar IEEE operation, round to nearest even) on two scalars
__device__ __forceinline__ void mul2(float& a, float& b, float c, float d) {
  const float2 r = __fmul2_rn(make_float2(a, b), make_float2(c, d));
  a = r.x, b = r.y;
}
__device__ __forceinline__ void fma2(float& a, float& b, float x, float y, float u, float v) {  // a = x * u + a, b = y * v + b
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
__device__ __forceinline__ float exp_tri(float x) {
  float m, e;
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(m) : "f"(x));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(m));
  return e;
}
__device__ __forceinline__ float rsqrt_tri(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float sigmoid_triton(float x) {
  float m, e;
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(m) : "f"(0.0f - x));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(m));
  float d;
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
  return div_full(1.0f, d);
}
// x / (1 + exp(-x)) as the compiled baseline's Triton kernel computes it: libdevice expf, div.full.f32
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
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- warp-level MMAs (sm_120a) ----
// ldmatrix: four / two 8x8 b16 matrices, row addresses from lanes 8i..8i+7 (matrix i); an 8-bit (e4m3) or 4-bit (e2m1)
// 16 x 32-byte tile loads exactly as the A fragment of mma m16n8k32 e4m3 / m16n8k64 e2m1 (bytes 4t.. of rows g, g + 8,
// then bytes 16 + 4t..), an 8 x 32-byte tile as the B fragment of n = 8 (bytes 4t.. and 16 + 4t.. of row g)
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
// D += A B: e4m3 x e4m3 -> f32, m16n8k32
__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D += (A sfa) (B sfb): e2m1 x e2m1 with ue4m3 scales per 16 values, m16n8k64. sfa: lane 4g holds row g's 4 scales,
// lane 4g + 1 row g + 8's; sfb: lane 4g holds column g's
__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}
// D += A B: bf16 x bf16 -> f32, m16n8k16
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace mk
