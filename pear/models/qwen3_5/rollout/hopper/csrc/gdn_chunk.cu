// The prefill rows' chunked gated delta rule (FLA's chunk kernels as vLLM 0.28 runs them for Qwen3.5) as two persistent
// kernels over a static layout's real chunks, bit for bit as FLA's Triton kernels compute it on Hopper; a third kernel
// runs the decode rows' recurrence of a mixed step, a fourth the step's conv + l2norm + gates
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

namespace gdn_chunk {

using bf16 = __nv_bfloat16;
constexpr int BT = 64;   // chunk size
constexpr int DK = 128;  // key head dim
constexpr int DV = 128;  // value head dim
constexpr int LD128 = 136;  // padded row stride (bf16) of 128-wide tiles: conflict-free ldmatrix
constexpr int LD64 = 72;    // of 64-wide tiles
constexpr int LD32 = 40;    // of 32-wide tiles
constexpr int LDA = 68;     // fp32 row stride of the 64x64 A tile

// exact scalar operations, one PTX instruction each (no contraction into FMAs, no reassociation by the compiler or
// ptxas): the kernels repeat FLA's scalar operations in its order, the FMA contractions LLVM chose there included, and
// its tensor-core products in the same k order (mma.sync m16n8k16 and wgmma give the bits of the wgmma Triton emits)
__device__ __forceinline__ float f_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float f_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float f_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float f_fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Triton's fp32 exp on NVIDIA: ex2.approx(x * log2(e))
__device__ __forceinline__ float t_exp(float x) { return ex2_approx(f_mul(x, __int_as_float(0x3FB8AA3B))); }
__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float lo_f(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float hi_f(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// (no memory clobber: every shared-memory write a ldmatrix reads crosses a barrier, which carries one; the loops
// below load the next k step's fragments before the current step's MMAs, which the compiler would not do itself)
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 4 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ldmatrix addresses (lane l) of the m16n8k16 fragments of a bf16 tile with row stride ld:
// A (16 x 16 at (r0, c0) of a row-major [M][K] tile): a0..a3
__device__ __forceinline__ const bf16* a_addr(const bf16* t, int ld, int r0, int c0, int lane) {
  return t + (r0 + (lane & 7) + ((lane >> 3) & 1) * 8) * ld + c0 + (lane >> 4) * 8;
}
// B of one n-tile from a [N][K] tile (x2: lanes 0-15 give the addresses)
__device__ __forceinline__ const bf16* bn1_addr(const bf16* t, int ld, int n0, int k0, int lane) {
  return t + (n0 + (lane & 7)) * ld + k0 + ((lane >> 3) & 1) * 8;
}
// B of two n-tiles from a [K][N] tile (rows k, N contiguous), with .trans
__device__ __forceinline__ const bf16* bk_addr(const bf16* t, int ld, int k0, int n0, int lane) {
  return t + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * ld + n0 + (lane >> 4) * 8;
}

// cp.async rows [0, rows) of a 64 x 128 bf16 tile (row stride src_ld elements) into smem (row stride LD128); the
// rest zero-filled
template <int NTH>
__device__ __forceinline__ void cp_tile64x128(bf16* dst, const bf16* src, int64_t src_ld, int rows, int tid) {
#pragma unroll
  for (int i = 0; i < 1024 / NTH; i++) {
    const int idx = tid + i * NTH, r = idx >> 4, c = (idx & 15) * 8;
    const bool ok = r < rows;
    cp_async16(dst + r * LD128 + c, src + (ok ? r : 0) * src_ld + c, ok);
  }
}

// the chunk kernels loop over the work items the layout's counts name (a grid of about the SM count, captured once in
// a FULL CUDA graph): a step launches no program for the static layout's unused capacity
struct Layout {
  const int* cu;             // [P + 1] prefill sequence starts (absolute tokens)
  const int* chunk_indices;  // [NTc, 2] (sequence, chunk within it)
  const int* chunk_offsets;  // [P + 1] first chunk of each prefill sequence
  const int* counts;         // [T, D, N, P, NT, ...]
};

// ================================================================================================================
// gdn_chunk_prep (per chunk and value head): g_cum, A = beta k k^T (tf32) masked and decayed, Ai = (I + A)^-1 (bf16),
// w = Ai (beta exp(g) k), u = Ai (beta v), and the output's masked decayed q k^T (bf16)
constexpr int PREP_THREADS = 256;

struct PrepParams {
  const bf16* q;      // [T, H, DK]
  const bf16* k;      // [T, H, DK]
  const bf16* v;      // [T, HV, DV]
  const float* g;     // [T, HV]
  const float* beta;  // [T, HV]
  float* g_cum;       // [T, HV]
  bf16* w;            // [T, HV, DK]
  bf16* u;            // [T, HV, DV]
  bf16* am;           // [NTc, HV, BT, BT]: the output's masked, decayed q k^T
  Layout lay;
  int H, HV;
};

struct PrepSmem {
  bf16 q[BT * LD128];
  bf16 k[BT * LD128];
  bf16 v[BT * LD128];   // then beta v (bf16)
  // kb, ai, dg, od, y hold, until they are written, the KKT's B operand: k (fp32) as wgmma core matrices
  bf16 kb[BT * LD128];  // beta exp(g) k (bf16)
  bf16 ai[BT * LD64];   // (I + A)^-1 (bf16)
  float dg[4][256];     // the inverse's diagonal blocks (fp32, with the identity)
  float od[6][256];     // its blocks below: Ai21, Ai32, Ai43, Ai31, Ai42, Ai41 (fp32)
  float y[3][256];      // Y31, Y42, Y41 (the sums Ai31, Ai42, Ai41 are products of)
  float A[BT * LDA];    // beta k k^T, masked and decayed (fp32)
  float g[BT];
  float gc[BT];
  float eg[BT];
  float beta[BT];
};
static_assert(offsetof(PrepSmem, A) - offsetof(PrepSmem, kb) >= BT * DK * 4, "the KKT's fp32 B operand fits");
static_assert(offsetof(PrepSmem, kb) % 128 == 0, "core matrices aligned");

// wgmma (sm_90a) descriptors: an operand without swizzle in 8-row x 16-byte core matrices (LBO the byte stride between
// core matrices along K, SBO along M/N) and a 128B-swizzled one (K-major: SBO the 8-row group stride; MN-major: SBO the
// 8-K-row group stride, LBO the 64-element MN block stride)
__device__ __forceinline__ uint64_t desc_core(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16) |
         ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return desc_core(addr, lbo, sbo) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// d[64 x 32] (+)= a (tf32, registers as mma.sync's m16n8k8 A per warp) x B (tf32, smem descriptor)
__device__ __forceinline__ void wgmma_tf32_m64n32k8(float (&d)[16], const uint32_t (&a)[4], uint64_t db, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(acc));
}

// a row segment of a 16x16 fp32 product as Triton's FMA dot computes each element: fma over k = 0..15 in order from
// acc; a_row: 16 values (registers), b: the B block in smem (row stride ldb), columns n0..n0+3
__device__ __forceinline__ void fma_dot_row4(const float (&a_row)[16], const float* b, int ldb, int n0,
                                             float (&acc)[4]) {
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    const float4 bv = *reinterpret_cast<const float4*>(b + kk * ldb + n0);
    acc[0] = f_fma(a_row[kk], bv.x, acc[0]);
    acc[1] = f_fma(a_row[kk], bv.y, acc[1]);
    acc[2] = f_fma(a_row[kk], bv.z, acc[2]);
    acc[3] = f_fma(a_row[kk], bv.w, acc[3]);
  }
}
__device__ __forceinline__ void load_row16(float (&r)[16], const float* p) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 v = *reinterpret_cast<const float4*>(p + 4 * i);
    r[4 * i] = v.x;
    r[4 * i + 1] = v.y;
    r[4 * i + 2] = v.z;
    r[4 * i + 3] = v.w;
  }
}
// the full 16-value row of a product whose 4-column quarters are held by 4 consecutive lanes (quarter = lane & 3)
__device__ __forceinline__ void gather_row16(float (&r)[16], const float (&x)[4], int lane) {
  const int base = lane & ~3;
#pragma unroll
  for (int qd = 0; qd < 4; qd++)
#pragma unroll
    for (int e = 0; e < 4; e++) r[4 * qd + e] = __shfl_sync(0xffffffffu, x[e], base + qd);
}
__device__ __forceinline__ void store_block_row4(PrepSmem& s, float* od, int bi, int bj, int m, int n0,
                                                 const float (&x)[4]) {
  *reinterpret_cast<float4*>(od + m * 16 + n0) = make_float4(x[0], x[1], x[2], x[3]);
  uint2 pk = make_uint2(pack2(x[0], x[1]), pack2(x[2], x[3]));
  *reinterpret_cast<uint2*>(&s.ai[(16 * bi + m) * LD64 + 16 * bj + n0]) = pk;
}

// chunk_local_cumsum of one chunk and head (a warp): Triton's scan of 64 values over 32 lanes, two Kogge-Stone scans,
// the first one's total added to the second
__device__ __forceinline__ void cumsum64(float& x0, float& x1, int lane) {
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const float y0 = __shfl_up_sync(0xffffffffu, x0, d);
    const float y1 = __shfl_up_sync(0xffffffffu, x1, d);
    if (lane >= d) {
      x0 = f_add(x0, y0);
      x1 = f_add(x1, y1);
    }
  }
  const float tot = __shfl_sync(0xffffffffu, x0, 31);
  x1 = f_add(x1, tot);
}

// u = Ai (beta v), w = Ai (beta exp(g) k) of row block MI, columns 16 cg .. 16 cg + 15 (a warp): the k16 steps right of
// the diagonal multiply zeros and are skipped; staged (bf16) in s.q (u) and s.k (w)
template <int MI>
__device__ __forceinline__ void wu_block(PrepSmem& s, int cg, int lane) {
  const int g8 = lane >> 2, t4 = lane & 3;
  float au[2][4], aw[2][4];
#pragma unroll
  for (int j = 0; j < 2; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) au[j][e] = aw[j][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks <= MI; ks++) {
    uint32_t a[4], b[4];
    ldsm_x4(a, a_addr(s.ai, LD64, 16 * MI, 16 * ks, lane));
    ldsm_x4_t(b, bk_addr(s.v, LD128, 16 * ks, 16 * cg, lane));
    mma_bf16(au[0], a, b[0], b[1]);
    mma_bf16(au[1], a, b[2], b[3]);
    ldsm_x4_t(b, bk_addr(s.kb, LD128, 16 * ks, 16 * cg, lane));
    mma_bf16(aw[0], a, b[0], b[1]);
    mma_bf16(aw[1], a, b[2], b[3]);
  }
#pragma unroll
  for (int nj = 0; nj < 2; nj++) {
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
      const int i = 16 * MI + g8 + 8 * h2, c = 16 * cg + 8 * nj + 2 * t4;
      *reinterpret_cast<uint32_t*>(&s.q[i * LD128 + c]) = pack2(au[nj][2 * h2], au[nj][2 * h2 + 1]);
      *reinterpret_cast<uint32_t*>(&s.k[i * LD128 + c]) = pack2(aw[nj][2 * h2], aw[nj][2 * h2 + 1]);
    }
  }
}

__global__ void __launch_bounds__(PREP_THREADS, 2) prep_kernel(PrepParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  PrepSmem& s = *reinterpret_cast<PrepSmem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t4 = lane & 3;
  const int H = p.H, HV = p.HV;
  const int n_items = p.lay.counts[4] * HV;

  // Ai's blocks above the diagonal stay zero
  for (int idx = tid; idx < BT * BT; idx += PREP_THREADS) {
    const int i = idx >> 6, j = idx & 63;
    if ((j >> 4) > (i >> 4)) s.ai[i * LD64 + j] = f2bf(0.f);
  }

  for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
    const int i_tg = item / HV, hv = item % HV;
    const int i_n = p.lay.chunk_indices[2 * i_tg], i_t = p.lay.chunk_indices[2 * i_tg + 1];
    const int bos = p.lay.cu[i_n], eos = p.lay.cu[i_n + 1];
    const int t0 = bos + i_t * BT;
    const int Tc = min(BT, eos - t0);
    const int kh = hv / (HV / H);

    __syncthreads();  // the previous item's smem reads are done
    // group 0: k, g, beta, q; group 1: v
    cp_tile64x128<PREP_THREADS>(s.k, p.k + ((int64_t)t0 * H + kh) * DK, H * DK, Tc, tid);
    cp_tile64x128<PREP_THREADS>(s.q, p.q + ((int64_t)t0 * H + kh) * DK, H * DK, Tc, tid);
    if (tid < 2 * BT) {
      const int r = tid & 63;
      const bool ok = r < Tc;
      const int64_t off = (int64_t)(t0 + (ok ? r : 0)) * HV + hv;
      if (tid < BT)
        cp_async4(&s.g[r], p.g + off, ok);
      else
        cp_async4(&s.beta[r], p.beta + off, ok);
    }
    cp_async_commit();
    cp_tile64x128<PREP_THREADS>(s.v, p.v + ((int64_t)t0 * HV + hv) * DV, HV * DV, Tc, tid);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    // ---- chunk_local_cumsum (warp 7, as Triton's scan of 64 values over 32 lanes: two Kogge-Stone scans, the first
    // one's total added to the second); its exp(g) for w
    if (warp == 7) {
      float x0 = s.g[lane], x1 = s.g[lane + 32];
      cumsum64(x0, x1, lane);
      s.gc[lane] = x0;
      s.gc[lane + 32] = x1;
      s.eg[lane] = t_exp(lane < Tc ? x0 : 0.f);
      s.eg[lane + 32] = t_exp(lane + 32 < Tc ? x1 : 0.f);
      if (lane < Tc) p.g_cum[(int64_t)(t0 + lane) * HV + hv] = x0;
      if (lane + 32 < Tc) p.g_cum[(int64_t)(t0 + lane + 32) * HV + hv] = x1;
    }

    // ---- A = (beta k) k^T in tf32 as Triton's dot did it: wgmma m64n32k8 per warpgroup (8 warps: columns 32wg..),
    // 16 k8 steps, A (beta k, fp32) from registers, B (k, fp32) from shared memory core matrices
    {
      float* kf = reinterpret_cast<float*>(s.kb);
      // k -> fp32 core matrices: element (n, c) at (n / 8) * 1024 + (c / 4) * 32 + (n % 8) * 4 + c % 4 floats
#pragma unroll
      for (int i = 0; i < 2048 / PREP_THREADS; i++) {
        const int qd = tid + i * PREP_THREADS, n8 = qd >> 8, rem = qd & 255, c4 = rem >> 3, r = rem & 7;
        const int n = 8 * n8 + r;
        const uint2 kk = *reinterpret_cast<const uint2*>(&s.k[n * LD128 + 4 * c4]);
        *reinterpret_cast<float4*>(&kf[n8 * 1024 + c4 * 32 + r * 4]) =
            make_float4(lo_f(kk.x), hi_f(kk.x), lo_f(kk.y), hi_f(kk.y));
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncthreads();  // kf; g_cum (warp 7)
      const int wg = warp >> 2, wl = warp & 3;
      const int r0 = 16 * wl + g8, r1 = r0 + 8;
      const float be0 = s.beta[r0], be1 = s.beta[r1];
      const uint32_t kf_base = smem_u32(kf) + wg * 4 * 4096;
      float acc[16], am_acc[4][4];
#pragma unroll
      for (int e = 0; e < 16; e++) acc[e] = am_acc[e >> 2][e & 3] = 0.f;
      // the output's q k^T (bf16, 8 k16 steps) by mma.sync in the wgmma accumulator's layout (warp (wg, wl): rows
      // 16 wl .., columns 32 wg + 8 jj ..), only the m16n8 tiles at or left of the diagonal, beside the wgmma
      auto am_steps = [&](int ks0) {
#pragma unroll
        for (int ks = ks0; ks < ks0 + 4; ks++) {
          uint32_t a4[4];
          ldsm_x4(a4, a_addr(s.q, LD128, 16 * wl, 16 * ks, lane));
#pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            if (4 * wg + jj <= 2 * wl + 1) {
              uint32_t b2[2];
              ldsm_x2(b2, bn1_addr(s.k, LD128, 32 * wg + 8 * jj, 16 * ks, lane));
              mma_bf16(am_acc[jj], a4, b2[0], b2[1]);
            }
          }
        }
      };
      // 8 k steps' A fragments in their own registers at a time: a wgmma reads its A registers asynchronously, so
      // reusing them makes the next step wait for it
#pragma unroll
      for (int half = 0; half < 2; half++) {
        uint32_t a[8][4];
#pragma unroll
        for (int kq = 0; kq < 8; kq++) {
          const int c = (8 * half + kq) * 8 + t4;
          a[kq][0] = __float_as_uint(f_mul(bf2f(s.k[r0 * LD128 + c]), be0));
          a[kq][1] = __float_as_uint(f_mul(bf2f(s.k[r1 * LD128 + c]), be1));
          a[kq][2] = __float_as_uint(f_mul(bf2f(s.k[r0 * LD128 + c + 4]), be0));
          a[kq][3] = __float_as_uint(f_mul(bf2f(s.k[r1 * LD128 + c + 4]), be1));
        }
        wgmma_fence();
#pragma unroll
        for (int kq = 0; kq < 8; kq++) {
          const int ks = 8 * half + kq;
          wgmma_tf32_m64n32k8(acc, a[kq], desc_core(kf_base + 2 * ks * 128, 128, 4096), ks);
        }
        wgmma_commit();
        am_steps(4 * half);
        wgmma_wait<0>();
      }
      wgmma_wait<0>();
      // exp(g_i - g_j) once for A = beta k k^T (strictly lower) and the output's Am = q k^T (lower incl. the diagonal),
      // both masked to the chunk's rows; Am to global (bf16, the whole 64 x 64 tile, zeros above the diagonal)
      bf16* am = p.am + ((int64_t)i_tg * HV + hv) * BT * BT;
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        float amv[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int i = 16 * wl + g8 + (e >> 1) * 8, j = 32 * wg + 8 * jj + 2 * t4 + (e & 1);
          float val = 0.f;
          amv[e] = 0.f;
          if (j <= i && i < Tc && j < Tc) {
            const float ex = t_exp(f_sub(s.gc[i], s.gc[j]));
            if (i > j) val = f_mul(acc[4 * jj + e], ex);
            amv[e] = f_mul(am_acc[jj][e], ex);
          }
          // the diagonal blocks (read by the forward substitution only) negated once here: -A as it computes it
          s.A[i * LDA + j] = (i >> 4) == (j >> 4) ? f_sub(0.f, val) : val;
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
          const int i = 16 * wl + g8 + h2 * 8, j = 32 * wg + 8 * jj + 2 * t4;
          *reinterpret_cast<uint32_t*>(am + i * BT + j) = pack2(amv[2 * h2], amv[2 * h2 + 1]);
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    if (warp < 2) {
      // ---- (I + A)^-1's diagonal blocks as FLA's merge_16x16_to_64x64_inverse_kernel computes them: forward
      // substitution, each row step reducing 16 products over the rows in Triton's order (within the 4-row groups of
      // its warps fma(own, partner two rows down) then + the next row's, across the 4 groups (W0 + W2) + (W1 + W3))
      const int b = tid >> 4, j = tid & 15;
      const float* Ab = s.A + (16 * b) * LDA + 16 * b;
      float col[16];
#pragma unroll
      for (int kk = 0; kk < 16; kk++) col[kk] = kk > j ? Ab[kk * LDA + j] : 0.f;  // (the block is -A)
      const int iend = Tc - 16 * b;
#pragma unroll
      for (int i = 2; i < 16; i++) {
        float a[16];
        load_row16(a, Ab + i * LDA);  // -A
        const float aj = Ab[i * LDA + j];
        float W[4];
#pragma unroll
        for (int w4 = 0; w4 < 4; w4++) {
          const int k0 = 4 * w4;
          const float p2 = f_mul(col[k0 + 2], a[k0 + 2]);
          const float p3 = f_mul(col[k0 + 3], a[k0 + 3]);
          const float s0 = f_fma(col[k0], a[k0], p2);
          const float s1 = f_fma(col[k0 + 1], a[k0 + 1], p3);
          W[w4] = f_add(s0, s1);
        }
        const float S = f_add(f_add(W[0], W[2]), f_add(W[1], W[3]));
        const float nv = f_add(aj, S);
        col[i] = i < iend ? nv : col[i];
      }
#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        const float d = f_add(col[kk], kk == j ? 1.f : 0.f);
        s.dg[b][kk * 16 + j] = d;
        s.ai[(16 * b + kk) * LD64 + 16 * b + j] = f2bf(d);
      }
    } else {
      const int tq = tid - 64;
      // beta v (bf16, in place) and beta exp(g) k (bf16)
      for (int idx = tq; idx < BT * 16; idx += 192) {
        const int r = idx >> 4, c = (idx & 15) * 8;
        const float be = s.beta[r], eg = s.eg[r];
        uint4 vv = *reinterpret_cast<const uint4*>(&s.v[r * LD128 + c]);
        uint4 kk = *reinterpret_cast<const uint4*>(&s.k[r * LD128 + c]);
        uint32_t* vp = reinterpret_cast<uint32_t*>(&vv);
        uint32_t* kp = reinterpret_cast<uint32_t*>(&kk);
#pragma unroll
        for (int e = 0; e < 4; e++) {
          vp[e] = pack2(f_mul(lo_f(vp[e]), be), f_mul(hi_f(vp[e]), be));
          kp[e] = pack2(f_mul(f_mul(lo_f(kp[e]), be), eg), f_mul(f_mul(hi_f(kp[e]), be), eg));
        }
        *reinterpret_cast<uint4*>(&s.v[r * LD128 + c]) = vv;
        *reinterpret_cast<uint4*>(&s.kb[r * LD128 + c]) = kk;
      }
    }
    __syncthreads();

    // ---- the blocks below the diagonal by fp32 FMA dots, a row quarter (4 columns) per thread (5 phases); the idle
    // threads compute u = Ai (beta v), w = Ai (beta exp(g) k) of the row blocks whose Ai rows are final (0 now, 1 after
    // Ai21, 2 after Ai31, 3 after Ai41), staged in shared memory (q and k are read by now), stored one phase later
    const int m = (tid >> 2) & 15, n0 = (tid & 3) * 4, blk = tid >> 6;
    // rows 16 mi .. 16 mi + 15 of u and w: 16-byte stores of whole 256-byte rows (rows past the sequence not stored)
    auto store_rows = [&](int mi, int t, int nt) {
      for (int x = t; x < 512; x += nt) {
        const int which = x >> 8, r = 16 * mi + ((x >> 4) & 15), c = (x & 15) * 8;
        if (r < Tc) {
          const int64_t row = (int64_t)(t0 + r) * HV + hv;
          const uint4 val = *reinterpret_cast<const uint4*>(which ? &s.k[r * LD128 + c] : &s.q[r * LD128 + c]);
          *reinterpret_cast<uint4*>(which ? p.w + row * DK + c : p.u + row * DV + c) = val;
        }
      }
    };
    // Ai21 = -((D2 A21) D1), Ai32 = -((D3 A32) D2), Ai43 = -((D4 A43) D3) | u/w row block 0 (warps 6, 7)
    if (tid < 192) {
      float dr[16], x[4] = {0.f, 0.f, 0.f, 0.f}, xr[16];
      load_row16(dr, &s.dg[blk + 1][m * 16]);
      fma_dot_row4(dr, s.A + (16 * (blk + 1)) * LDA + 16 * blk, LDA, n0, x);
      gather_row16(xr, x, lane);
      float r[4] = {0.f, 0.f, 0.f, 0.f};
      fma_dot_row4(xr, s.dg[blk], 16, n0, r);
#pragma unroll
      for (int e = 0; e < 4; e++) r[e] = f_sub(0.f, r[e]);
      store_block_row4(s, s.od[blk], blk + 1, blk, m, n0, r);
    } else {
      for (int cg = 4 * (warp - 6); cg < 4 * (warp - 5); cg++) wu_block<0>(s, cg, lane);
    }
    __syncthreads();
    // Y31 = A31 D1 + A32 Ai21, Y42 = A42 D2 + A43 Ai32, Y41 = A41 D1 + A42 Ai21 (one fma chain each) | u/w row block 1
    // (warps 6, 7), row block 0 stored
    if (tid < 192) {
      const float* A = s.A;
      const float* ax = blk == 0 ? A + 32 * LDA : blk == 1 ? A + 48 * LDA + 16 : A + 48 * LDA;
      const float* ay = blk == 0 ? A + 32 * LDA + 16 : blk == 1 ? A + 48 * LDA + 32 : A + 48 * LDA + 16;
      const float* mx = blk == 0 ? s.dg[0] : blk == 1 ? s.dg[1] : s.dg[0];
      const float* my = blk == 0 ? s.od[0] : blk == 1 ? s.od[1] : s.od[0];
      float ar[16], y[4] = {0.f, 0.f, 0.f, 0.f};
      load_row16(ar, ax + m * LDA);
      fma_dot_row4(ar, mx, 16, n0, y);
      load_row16(ar, ay + m * LDA);
      fma_dot_row4(ar, my, 16, n0, y);
      *reinterpret_cast<float4*>(&s.y[blk][m * 16 + n0]) = make_float4(y[0], y[1], y[2], y[3]);
    } else {
      for (int cg = 4 * (warp - 6); cg < 4 * (warp - 5); cg++) wu_block<1>(s, cg, lane);
      store_rows(0, tid - 192, 64);
    }
    __syncthreads();
    // Ai31 = -(D3 Y31), Ai42 = -(D4 Y42) | row block 1 stored (warps 4-7)
    if (tid < 128) {
      float dr[16], r[4] = {0.f, 0.f, 0.f, 0.f};
      load_row16(dr, &s.dg[blk + 2][m * 16]);
      fma_dot_row4(dr, s.y[blk], 16, n0, r);
#pragma unroll
      for (int e = 0; e < 4; e++) r[e] = f_sub(0.f, r[e]);
      store_block_row4(s, s.od[3 + blk], blk + 2, blk, m, n0, r);
    } else {
      store_rows(1, tid - 128, 128);
    }
    __syncthreads();
    // Y41 += A43 Ai31 | u/w row block 2 (warps 2-7)
    if (tid < 64) {
      float ar[16], y[4];
      const float4 y0 = *reinterpret_cast<const float4*>(&s.y[2][m * 16 + n0]);
      y[0] = y0.x;
      y[1] = y0.y;
      y[2] = y0.z;
      y[3] = y0.w;
      load_row16(ar, s.A + (48 + m) * LDA + 32);
      fma_dot_row4(ar, s.od[3], 16, n0, y);
      __syncwarp();
      *reinterpret_cast<float4*>(&s.y[2][m * 16 + n0]) = make_float4(y[0], y[1], y[2], y[3]);
    } else {
      for (int cg = warp - 2; cg < 8; cg += 6) wu_block<2>(s, cg, lane);
    }
    __syncthreads();
    // Ai41 = -(D4 Y41) | row block 2 stored (warps 2-7)
    if (tid < 64) {
      float dr[16], r[4] = {0.f, 0.f, 0.f, 0.f};
      load_row16(dr, &s.dg[3][m * 16]);
      fma_dot_row4(dr, s.y[2], 16, n0, r);
#pragma unroll
      for (int e = 0; e < 4; e++) r[e] = f_sub(0.f, r[e]);
      store_block_row4(s, s.od[5], 3, 0, m, n0, r);
    } else {
      store_rows(2, tid - 64, 192);
    }
    __syncthreads();
    // u/w row block 3 (warp w column group w), stored
    wu_block<3>(s, warp, lane);
    __syncthreads();
    store_rows(3, tid, PREP_THREADS);
  }
}

// ================================================================================================================
// gdn_chunk_state (per prefill sequence, value head and 32-row value block): the state recurrence and the output
constexpr int SBV = 32;             // value rows per state program

struct StateParams {
  const bf16* q;       // [T, H, DK]
  const bf16* k;       // [T, H, DK]
  const bf16* u;       // [T, HV, DV]
  const bf16* w;       // [T, HV, DK]
  const bf16* am;      // [NTc, HV, BT, BT]
  const float* g_cum;  // [T, HV]
  bf16* o;             // [T, HV, DV]
  float* state;        // [slots, HV, DV, DK]
  int64_t state_stride;
  const int* seq_slot;
  const int* seq_init;
  Layout lay;
  int H, HV;
  float scale;
};

__device__ __forceinline__ void bar_math() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

// wgmma (warpgroup 0) fed by a load warpgroup (warpgroup 1): the products are FLA's k16 steps in its order (wgmma gives
// mma.sync's bits), the scalar epilogues its instructions; the load warpgroup keeps up to STATE_STAGES - 1 chunks
// ahead behind full/empty mbarriers and stores the output tiles.
constexpr int STATE_THREADS = 256;

struct StateStage {
  bf16 w[2 * BT * 64];  // the chunk's w [t][kd] as two 64-column blocks of 128-byte rows, 128B-swizzled (TMA's layout)
  bf16 q[2 * BT * 64];  // q, the same
  bf16 k[2 * BT * 64];  // k, the same (the MN-major A of k^T vs: M = kd)
  bf16 am[BT * 64];     // the output's masked q k^T [t][s], 128B-swizzled
  bf16 u[BT * LD32];    // the value block's u [t][v]; then (read early in the step) the step's output tile [t][v]
  float gc[BT];
  float pad[192];       // stages stay 1024-byte aligned (the swizzle atoms)
};
static_assert(sizeof(StateStage) % 1024 == 0, "StateStage alignment");

constexpr int STATE_STAGES = 3;  // tile stages: the load warpgroup runs up to STATE_STAGES - 1 chunks ahead
struct StateSmem {
  StateStage st[STATE_STAGES];
  bf16 hb[2][SBV * DK];  // the state (bf16) as the MN-major B (N = v, K = kd) of w h^T and q h^T, core matrices:
                         // (v / 8) * 2048 + (kd / 8) * 128 + (kd % 8) * 16 + (v % 8) * 2 bytes
  bf16 vs[SBV * BT];     // decayed v_new as the MN-major B (N = v, K = t) of k^T vs: (v / 8) * 1024 + (t / 8) * 128 ..
  bf16 vn[SBV * BT];     // v_new, the same layout (the B of Am v_new)
  unsigned long long full[STATE_STAGES], empty[STATE_STAGES];
};

// byte offset of 16-byte chunk j of row r in a 128B-swizzled block of 128-byte rows (1024-byte aligned)
__device__ __forceinline__ int sw128(int r, int j) { return r * 128 + ((j ^ (r & 7)) << 4); }

// d[64 x 32] += A (bf16, shared memory) x B (bf16, shared memory); TA / TB: the operand is MN-major
template <int TA, int TB>
__device__ __forceinline__ void wgmma_bf16_m64n32k16(float (&d)[16], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15}, %16, %17, p, 1, 1, %18, %19;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
// pins accumulator registers around asynchronous wgmma (no instruction)
__device__ __forceinline__ void fence_acc(float (&d)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+f"(d[i])::"memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void mbar_init(unsigned long long* bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, int parity) {
  asm volatile(
      "{\n.reg .pred done;\nwait_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
      "@!done bra wait_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// per (prefill sequence, value head, 32-row value block): over the sequence's chunks c, with h the state before c,
//   v_new = u - w h^T, o = (q h^T exp(g) + Am v_new) scale, h <- h exp(g_last) + k^T (v_new exp(g_last - g))
// as FLA's chunk_gated_delta_rule_fwd_kernel_h_blockdim64 and chunk_fwd_kernel_o compute them.
__global__ void __launch_bounds__(STATE_THREADS, 1) state_kernel(StateParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  // 1024-byte aligned (the swizzle atoms), offset from smem_raw itself so that accesses stay in the shared space
  StateSmem& s = *reinterpret_cast<StateSmem*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t4 = lane & 3;
  const int H = p.H, HV = p.HV;
  constexpr int NVB = DV / SBV;
  const int D = p.lay.counts[1];
  const int n_items = p.lay.counts[3] * HV * NVB;
  // the warpgroup, provably warp-uniform (else ptxas serializes the wgmma in the "divergent" math path)
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);
  if (tid == 0) {
    for (int i = 0; i < STATE_STAGES; i++) {
      mbar_init(&s.full[i], 128);
      mbar_init(&s.empty[i], 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  int n = 0;  // the CTA's chunk counter over its items: stage n % STATE_STAGES, phase (n / STATE_STAGES) & 1
  for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
    const int vb = item % NVB, i_nh = item / NVB;
    const int i_n = i_nh / HV, hv = i_nh % HV;
    const int kh = hv / (HV / H);
    const int bos = p.lay.cu[i_n], eos = p.lay.cu[i_n + 1];
    const int T = eos - bos, NT = (T + BT - 1) / BT;
    const int boh = p.lay.chunk_offsets[i_n];

    if (wg == 1) {
      // ---- the load warpgroup: chunk c's tiles into stage n % STATE_STAGES once the math warps released it (rows past the
      // sequence zero-filled)
      const int ht = tid - 128;
      // the output tile of chunk c (staged in its stage's u tile by the math warps before they released the stage):
      // rows past the sequence not stored, 16-byte stores
      bf16* ob = p.o + ((int64_t)bos * HV + hv) * DV + SBV * vb;
      auto store_o = [&](int c, const bf16* o) {
        const int Tc = min(BT, T - c * BT);
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const int x = ht + 128 * i, r = x >> 2, q = (x & 3) * 8;
          if (r < Tc)
            *reinterpret_cast<uint4*>(ob + (int64_t)(c * BT + r) * HV * DV + q) =
                *reinterpret_cast<const uint4*>(&o[r * LD32 + q]);
        }
      };
      for (int c = 0; c < NT; c++, n++) {
        const int sidx = n % STATE_STAGES;
        StateStage& sg = s.st[sidx];
        if (n >= STATE_STAGES) mbar_wait(&s.empty[sidx], ((n / STATE_STAGES) - 1) & 1);
        // the output of the stage's previous chunk, before its tiles are overwritten (the loads below issue after these
        // stores' shared memory reads)
        if (c >= STATE_STAGES) store_o(c - STATE_STAGES, sg.u);
        const int Tc = min(BT, T - c * BT);
        const int64_t t0 = bos + c * BT;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int x = ht + 128 * i, r = x >> 4, c16 = x & 15;
          const bool ok = r < Tc;
          const int so = (c16 >> 3) * 8192 + sw128(r, c16 & 7);
          const int64_t tr = t0 + (ok ? r : 0);
          cp_async16(reinterpret_cast<char*>(sg.w) + so, p.w + (tr * HV + hv) * DK + c16 * 8, ok);
          cp_async16(reinterpret_cast<char*>(sg.q) + so, p.q + (tr * H + kh) * DK + c16 * 8, ok);
          cp_async16(reinterpret_cast<char*>(sg.k) + so, p.k + (tr * H + kh) * DK + c16 * 8, ok);
        }
        const bf16* amb = p.am + ((int64_t)(boh + c) * HV + hv) * BT * BT;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int x = ht + 128 * i, r = x >> 3, j = x & 7;
          cp_async16(reinterpret_cast<char*>(sg.am) + sw128(r, j), amb + r * BT + j * 8, true);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const int x = ht + 128 * i, r = x >> 2, j = x & 3;
          const bool ok = r < Tc;
          cp_async16(&sg.u[r * LD32 + j * 8], p.u + ((t0 + (ok ? r : 0)) * HV + hv) * DV + SBV * vb + j * 8, ok);
        }
        if (ht < BT) cp_async4(&sg.gc[ht], p.g_cum + (t0 + (ht < Tc ? ht : 0)) * HV + hv, ht < Tc);
        // the stage's barrier completes once every load thread's copies have landed (as CUTLASS's cp.async
        // pipelines feed wgmma: no wait here, no proxy fence)
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(&s.full[sidx]))
                     : "memory");
      }
      // the item's last output tiles, once their steps are done (the next item's first waits pass at once)
      for (int c = max(0, NT - STATE_STAGES); c < NT; c++) {
        const int m = n - NT + c;
        mbar_wait(&s.empty[m % STATE_STAGES], (m / STATE_STAGES) & 1);
        store_o(c, s.st[m % STATE_STAGES].u);
      }
      continue;
    }

    // ---- the math warpgroup
    const int seq = D + i_n;
    const int64_t slot = p.seq_slot[seq];
    const int init = p.seq_init[seq];
    float* st = p.state + slot * p.state_stride + (int64_t)hv * DV * DK + (int64_t)(SBV * vb) * DK;
    // the state block h^T [kd][v] in the accumulator layout of k^T vs: half hm, element 4j + e at kd = 64hm + 16warp +
    // g8 + 8(e >> 1), v = 8j + 2t4 + (e & 1)
    float hs[2][16];
#pragma unroll
    for (int hm = 0; hm < 2; hm++)
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const int j = i >> 2, e = i & 3;
        const int kd = 64 * hm + 16 * warp + g8 + (e >> 1) * 8, v = 8 * j + 2 * t4 + (e & 1);
        hs[hm][i] = init ? st[(int64_t)v * DK + kd] : 0.f;
      }
    auto store_hb = [&](int buf) {
      char* hb = reinterpret_cast<char*>(s.hb[buf]);
#pragma unroll
      for (int hm = 0; hm < 2; hm++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h2 = 0; h2 < 2; h2++)
            *reinterpret_cast<uint32_t*>(hb + j * 2048 + (8 * hm + 2 * warp + h2) * 128 + g8 * 16 + t4 * 4) =
                pack2(hs[hm][4 * j + 2 * h2], hs[hm][4 * j + 2 * h2 + 1]);
    };
    store_hb(0);
    fence_proxy_async();
    bar_math();
    const float scale = p.scale;
    const bool hi_half = vb & 1;
    bf16* ob = p.o + ((int64_t)bos * HV + hv) * DV + SBV * vb;
    // descriptors (their start address field is addr >> 4: byte offsets >> 4 add to it)
    const uint64_t d_vs = desc_core(smem_u32(s.vs), 128, 1024), d_vn = desc_core(smem_u32(s.vn), 128, 1024);
    const uint64_t d_hb0 = desc_core(smem_u32(s.hb[0]), 128, 2048), d_hb1 = desc_core(smem_u32(s.hb[1]), 128, 2048);
    const int t_r0 = 16 * warp + g8;  // the thread's rows t_r0 and t_r0 + 8 of a chunk
    // the output: q h^T and Am v_new of the step, exp(g) of its rows, staged at the end of the step in its stage's u
    // tile (read by then); the load warpgroup stores it once the step releases the stage, before it refills it (the
    // math warps' proxy fences would wait for their own global stores, slow beside the decode rows' memory traffic)
    float qh[16], av[16];
    auto out_stage = [&](bf16* o, float eg0, float eg1) {
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        const int t = t_r0 + 8 * h2;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          float r[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            // in each of FLA's 64-column output tiles, columns 0-31 fma(scale, Am v, scale (q h^T exp(g))), columns
            // 32-63 fma(scale, q h^T exp(g), scale (Am v)) (as LLVM contracted chunk_fwd_kernel_o): vb & 1 the half
            const float qe = f_mul(qh[4 * j + 2 * h2 + e], h2 ? eg1 : eg0), x = av[4 * j + 2 * h2 + e];
            r[e] = hi_half ? f_fma(scale, qe, f_mul(scale, x)) : f_fma(scale, x, f_mul(scale, qe));
          }
          *reinterpret_cast<uint32_t*>(&o[t * LD32 + 8 * j + 2 * t4]) = pack2(r[0], r[1]);
        }
      }
    };
    for (int c = 0; c < NT; c++, n++) {
      const int sidx = n % STATE_STAGES;
      StateStage& sg = s.st[sidx];
      const int Tc = min(BT, T - c * BT);
      const uint64_t d_w = desc_sw128(smem_u32(sg.w), 16, 1024), d_q = desc_sw128(smem_u32(sg.q), 16, 1024);
      const uint64_t d_k = desc_sw128(smem_u32(sg.k), 8192, 1024), d_am = desc_sw128(smem_u32(sg.am), 16, 1024);
      const uint64_t d_hb = (c & 1) ? d_hb1 : d_hb0;
      mbar_wait(&s.full[sidx], (n / STATE_STAGES) & 1);

      // group 1: w h^T (8 k16 steps over kd, B = the state)
      float acc[16];
#pragma unroll
      for (int i = 0; i < 16; i++) acc[i] = 0.f;
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 8; ks++)
        wgmma_bf16_m64n32k16<0, 1>(acc, d_w + (ks >> 2) * 512 + (ks & 3) * 2, d_hb + ks * 16);
      wgmma_commit();
      fence_acc(acc);
      // the step's scalars from shared memory while it runs
      const float g_last = sg.gc[Tc - 1];
      const float g0 = sg.gc[t_r0], g1 = sg.gc[t_r0 + 8];
      uint32_t uu[2][4];
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          uu[h2][j] = *reinterpret_cast<const uint32_t*>(&sg.u[(t_r0 + 8 * h2) * LD32 + 8 * j + 2 * t4]);
      const float dec0 = t_r0 < Tc ? t_exp(f_sub(g_last, g0)) : 0.f;
      const float dec1 = t_r0 + 8 < Tc ? t_exp(f_sub(g_last, g1)) : 0.f;
      const float egl = t_exp(g_last);
      const float eg0 = t_exp(g0), eg1 = t_exp(g1);

      // group 2: q h^T (the output's)
#pragma unroll
      for (int i = 0; i < 16; i++) qh[i] = 0.f;
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 8; ks++)
        wgmma_bf16_m64n32k16<0, 1>(qh, d_q + (ks >> 2) * 512 + (ks & 3) * 2, d_hb + ks * 16);
      wgmma_commit();
      fence_acc(qh);
      wgmma_wait<1>();
      fence_acc(acc);

      // v_new = u - w h^T and its decayed copy, as the B operands of the next products
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        const float dec = h2 ? dec1 : dec0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float vn0 = f_sub(lo_f(uu[h2][j]), acc[4 * j + 2 * h2]);
          const float vn1 = f_sub(hi_f(uu[h2][j]), acc[4 * j + 2 * h2 + 1]);
          const int off = j * 1024 + (2 * warp + h2) * 128 + g8 * 16 + t4 * 4;
          *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(s.vn) + off) = pack2(vn0, vn1);
          *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(s.vs) + off) = pack2(f_mul(vn0, dec), f_mul(vn1, dec));
        }
      }
      fence_proxy_async();
      bar_math();  // vn, vs

      // group 3: k^T vs (two 64-row halves of kd, 4 k16 steps over t); group 4: Am v_new
      float d[2][16];
#pragma unroll
      for (int i = 0; i < 16; i++) d[0][i] = d[1][i] = av[i] = 0.f;
      wgmma_fence();
#pragma unroll
      for (int hm = 0; hm < 2; hm++)
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
          wgmma_bf16_m64n32k16<1, 1>(d[hm], d_k + hm * 512 + ks * 128, d_vs + ks * 16);
      wgmma_commit();
#pragma unroll
      for (int ks = 0; ks < 4; ks++) wgmma_bf16_m64n32k16<0, 1>(av, d_am + ks * 2, d_vn + ks * 16);
      wgmma_commit();
      fence_acc(d[0]);
      fence_acc(d[1]);
      fence_acc(av);
      wgmma_wait<1>();
      fence_acc(d[0]);
      fence_acc(d[1]);

      // h = h exp(g_last) + k^T vs (one fma per element), into the other state buffer
#pragma unroll
      for (int hm = 0; hm < 2; hm++)
#pragma unroll
        for (int i = 0; i < 16; i++) hs[hm][i] = f_fma(hs[hm][i], egl, d[hm][i]);
      store_hb((c + 1) & 1);
      wgmma_wait<0>();
      fence_acc(qh);
      fence_acc(av);
      out_stage(sg.u, eg0, eg1);
      fence_proxy_async();          // the state buffer (wgmma)
      mbar_arrive(&s.empty[sidx]);  // this stage's tiles are read, the output tile staged
      bar_math();
    }

    // the final state to the sequence's slot
#pragma unroll
    for (int hm = 0; hm < 2; hm++)
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const int j = i >> 2, e = i & 3;
        const int kd = 64 * hm + 16 * warp + g8 + (e >> 1) * 8, v = 8 * j + 2 * t4 + (e & 1);
        st[(int64_t)v * DK + kd] = hs[hm][i];
      }
  }
}

// ================================================================================================================
// gdn_mixed_decode (per decode row, value head and 32-row value block): the one-token recurrence of a mixed step; its
// first programs also write the step's conv states
constexpr int DEC_THREADS = 128;  // 4 warps: warp w the value rows 32q + 8w.. of an item (decode row, value head,
                                  // 32-row quarter q); short CTAs, so that the chunk kernels beside them get SMs

// the step's conv states (each sequence's last 3 inputs; conv + l2norm + gates read the old ones before)
struct ConvStateParams {
  const bf16* x;  // [T, stride_x] mixed rows (the C qkv channels first)
  int64_t stride_x;
  bf16* state;  // [slots, 3, C]
  int64_t stride_slot, stride_row;
  const int* seq_start;  // [N + 1]
  const int* seq_slot;
  const int* seq_init;
  int C;  // a multiple of 8 (16-byte vectors)
};

// sequence sq's new conv state rows: row r = its input at relative position n - 3 + r (from x when >= 0, else the old
// row r + n, zero without an initial state), as _conv_state_kernel copies them, 16-byte vectors over the CTA's threads
__device__ __forceinline__ void conv_state_seq(const ConvStateParams& cs, int sq) {
  const int start = cs.seq_start[sq], n = cs.seq_start[sq + 1] - start;
  const int64_t slot = cs.seq_slot[sq];
  const int init = cs.seq_init[sq];
  for (int vi = threadIdx.x; vi < cs.C / 8; vi += blockDim.x) {
    bf16* base = cs.state + slot * cs.stride_slot + vi * 8;
    const bf16* xs = cs.x + vi * 8;
    uint4 nr[3];
#pragma unroll
    for (int r = 0; r < 3; r++) {
      const int rel = n - 3 + r;
      if (rel >= 0)
        nr[r] = *reinterpret_cast<const uint4*>(xs + (int64_t)(start + rel) * cs.stride_x);
      else if (init)
        nr[r] = *reinterpret_cast<const uint4*>(base + (int64_t)(r + n) * cs.stride_row);
      else
        nr[r] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int r = 0; r < 3; r++) *reinterpret_cast<uint4*>(base + (int64_t)r * cs.stride_row) = nr[r];
  }
}

struct DecParams {
  const bf16* q;      // [T, H, DK]
  const bf16* k;      // [T, H, DK]
  const bf16* v;      // [T, HV, DV]
  const float* g;     // [T, HV]
  const float* beta;  // [T, HV]
  bf16* o;            // [T, HV, DV]
  float* state;       // [slots, HV, DV, DK]
  int64_t state_stride;
  const int* seq_slot;
  const int* counts;
  int H, HV;
  float scale;
  ConvStateParams cs;  // the first counts[2] jobs update the conv states
};

// a warp's sum over the 32 lanes as Triton's tl.sum over a [rows, 128] fp32 tile reduces axis 1 with 4 columns per
// lane: shfl xor 16, 8, 4, 2, 1, each adding the partner's value
__device__ __forceinline__ float warp_sum_bfly(float t) {
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) t = f_add(t, __shfl_xor_sync(0xffffffffu, t, d));
  return t;
}

// the decode rows' states stream through L2 once: evict them first, so that the chunk path's tiles (written by the
// per-chunk kernel, read by the state kernel beside them) stay in L2
__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ float4 ld_stream(const float* p, uint64_t pol) {
  float4 v;
  asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], %5;\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void st_stream(float* p, float4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;\n" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z),
               "f"(v.w), "l"(pol)
               : "memory");
}

// per (decode row, value head, 32-row value block): the state S [v][kd] (fp32) <- exp(g) S, then v_new = (v - S k) beta,
// S <- S + v_new k^T, o = S (scale q); in each value row lane l holds columns 4l..4l+3, each dot an in-lane chain
// (column 1's product, fma of columns 0, 2, 3) and a butterfly
__global__ void __launch_bounds__(DEC_THREADS) dec_kernel(DecParams p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint64_t pol = evict_first_policy();
  const int H = p.H, HV = p.HV;
  const int n_cs = p.counts[2];
  const int n_jobs = n_cs + p.counts[1] * HV * 4;
  const float scale = p.scale;
  for (int job = blockIdx.x; job < n_jobs; job += gridDim.x) {
    if (job < n_cs) {
      // first: the conv states (they run beside the start of the decode rows' state streaming)
      conv_state_seq(p.cs, job);
      continue;
    }
    const int item = job - n_cs;
    const int quarter = item & 3, rh = item >> 2;
    const int r = rh / HV, hv = rh % HV, kh = hv / (HV / H);
    const int64_t slot = p.seq_slot[r];
    float* st = p.state + slot * p.state_stride + (int64_t)hv * DV * DK + 4 * lane;
    const uint2 kk = *reinterpret_cast<const uint2*>(p.k + ((int64_t)r * H + kh) * DK + 4 * lane);
    const uint2 qq = *reinterpret_cast<const uint2*>(p.q + ((int64_t)r * H + kh) * DK + 4 * lane);
    const float k0 = lo_f(kk.x), k1 = hi_f(kk.x), k2 = lo_f(kk.y), k3 = hi_f(kk.y);
    const float q0 = f_mul(scale, lo_f(qq.x)), q1 = f_mul(scale, hi_f(qq.x));
    const float q2 = f_mul(scale, lo_f(qq.y)), q3 = f_mul(scale, hi_f(qq.y));
    const float e = t_exp(p.g[(int64_t)r * HV + hv]);
    const float beta = p.beta[(int64_t)r * HV + hv];
    const bf16* vrow = p.v + ((int64_t)r * HV + hv) * DV;
    bf16* orow = p.o + ((int64_t)r * HV + hv) * DV;
    {
      const int v0 = 32 * quarter + 8 * warp;
      float4 sv[8];
#pragma unroll
      for (int j = 0; j < 8; j++) sv[j] = ld_stream(st + (int64_t)(v0 + j) * DK, pol);
      const uint4 vb = *reinterpret_cast<const uint4*>(vrow + v0);
      const uint32_t vw[4] = {vb.x, vb.y, vb.z, vb.w};
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float s0 = f_mul(e, sv[j].x), s1 = f_mul(e, sv[j].y), s2 = f_mul(e, sv[j].z), s3 = f_mul(e, sv[j].w);
        float t = f_mul(s1, k1);
        t = f_fma(s0, k0, t);
        t = f_fma(s2, k2, t);
        t = f_fma(s3, k3, t);
        t = warp_sum_bfly(t);
        const float vj = (j & 1) ? hi_f(vw[j >> 1]) : lo_f(vw[j >> 1]);
        const float bv = f_mul(f_sub(vj, t), beta);
        const float n0 = f_fma(bv, k0, s0), n1 = f_fma(bv, k1, s1), n2 = f_fma(bv, k2, s2), n3 = f_fma(bv, k3, s3);
        st_stream(st + (int64_t)(v0 + j) * DK, make_float4(n0, n1, n2, n3), pol);
        float to = f_mul(q1, n1);
        to = f_fma(q0, n0, to);
        to = f_fma(q2, n2, to);
        to = f_fma(q3, n3, to);
        to = warp_sum_bfly(to);
        if (lane == j) out = to;
      }
      if (lane < 8) orow[v0 + lane] = f2bf(out);
    }
  }
}

// ================================================================================================================
// gdn_conv_prep (per 16 tokens and head, before the others): the short convolution + silu, q and k's l2norm and the
// gates of a step's tokens
constexpr int CONV_T = 16;         // tokens per program
constexpr int CONV_THREADS = 128;  // 4 warps: a thread 2 of the tokens x 8 channels of a head
constexpr float L2_EPS = 1e-6f;    // FLA's l2norm epsilon, under the square root
constexpr float SOFTPLUS_THRESHOLD = 20.f;  // softplus(x) = x above it

struct ConvPrepParams {
  const bf16* x;  // [T, stride_x] mixed rows (the qkv channels first)
  int64_t stride_x;
  const bf16* ba;  // [T, stride_ba]: b in [0, HV), a in [HV, 2 HV)
  int64_t stride_ba;
  const bf16* conv_w;  // [C, 4]
  const bf16* conv_state;  // [slots, 3, C]
  int64_t stride_cs_slot, stride_cs_row;
  const float* A_log;
  const bf16* dt_bias;
  const int* tok_seq;
  const int* tok_pos;
  const int* seq_slot;
  const int* seq_init;
  bf16* q;  // [T, H, DK]
  bf16* k;
  bf16* v;  // [T, HV, DV]
  float* g;  // [T, HV]
  float* beta;
  const int* counts;
  int H, HV;
  int x_rows;    // rows of x (the padded batch)
  int tok_rows;  // of the layout's token arrays
};

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sqrt_approx_ftz(float a) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
// libdevice's logf (the ftz variant Triton inlines), with the caller's zero check: log(0) = -inf
__device__ __forceinline__ float t_logf(float x) {
  const bool den = x < __int_as_float(0x00800000);
  const float x1 = den ? f_mul(x, __int_as_float(0x4B000000)) : x;
  const float eb = den ? __int_as_float(0xC1B80000) : 0.f;
  const int e = (__float_as_int(x1) - 0x3F2AAAAB) & 0xFF800000;
  const float m = __int_as_float(__float_as_int(x1) - e);
  const float fe = fma_ftz((float)e, __int_as_float(0x34000000), eb);
  const float mm = f_add(m, -1.f);
  float pp = fma_ftz(__int_as_float(0xBE055027), mm, __int_as_float(0x3E1039F6));
  pp = fma_ftz(pp, mm, __int_as_float(0xBDF8CDCC));
  pp = fma_ftz(pp, mm, __int_as_float(0x3E0F2955));
  pp = fma_ftz(pp, mm, __int_as_float(0xBE2AD8B9));
  pp = fma_ftz(pp, mm, __int_as_float(0x3E4CED0B));
  pp = fma_ftz(pp, mm, __int_as_float(0xBE7FFF22));
  pp = fma_ftz(pp, mm, __int_as_float(0x3EAAAA78));
  pp = fma_ftz(pp, mm, __int_as_float(0xBF000000));
  float qq = f_mul(mm, pp);
  qq = fma_ftz(qq, mm, mm);
  float r = fma_ftz(fe, __int_as_float(0x3F317218), qq);
  if (!((unsigned)__float_as_int(x1) < 0x7F800000u))
    r = fma_ftz(x1, __int_as_float(0x7F800000), __int_as_float(0x7F800000));
  return x1 == 0.f ? __int_as_float(0xFF800000) : r;
}

__device__ __forceinline__ void bf8_to_f(const uint4& v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    f[2 * i] = lo_f(w[i]);
    f[2 * i + 1] = hi_f(w[i]);
  }
}

// op for op as Triton lowers that arithmetic (the taps as one mul and three fma, silu with div.full, the squares'
// butterfly over 16 lanes, sqrt.approx.ftz, libdevice's ftz logf), with 16-byte loads issued early
__global__ void __launch_bounds__(CONV_THREADS) conv_prep_kernel(ConvPrepParams p) {
  // program (token block, column block y): y < H the q head y, y < 2 H the k head y - H, else value head y - 2 H (and
  // its gates)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t_base = blockIdx.x * CONV_T;
  const int y = blockIdx.y, H = p.H, HV = p.HV;
  const bool qk = y < 2 * H;
  const int ch = (qk ? y * DK : 2 * H * DK + (y - 2 * H) * DV) + 8 * (lane & 15);
  int t[2];
  t[0] = t_base + 2 * warp + (lane >> 4);
  t[1] = t[0] + 8;
  // every load that does not depend on the step's layout first: the inputs of the 4 taps (rows of the padded batch;
  // taps that cross a sequence's first token are replaced below), the weights, the tokens' positions and sequences
  uint4 xv[2][4];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int j = 0; j < 4; j++)
      xv[r][j] = (t[r] < p.x_rows && t[r] - j >= 0)
                     ? *reinterpret_cast<const uint4*>(p.x + (int64_t)(t[r] - j) * p.stride_x + ch)
                     : make_uint4(0, 0, 0, 0);
  uint4 wv[4];
#pragma unroll
  for (int i = 0; i < 4; i++) wv[i] = *reinterpret_cast<const uint4*>(p.conv_w + (int64_t)ch * 4 + 8 * i);
  int pos[2], seq[2];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    pos[r] = t[r] < p.tok_rows ? p.tok_pos[t[r]] : 0;
    seq[r] = t[r] < p.tok_rows ? p.tok_seq[t[r]] : 0;
  }
  const int T = p.counts[0];
  if (t_base >= T) return;
  // taps before a sequence's first token: its conv state (row 3 + pos - j holds the input j - pos steps before it)
#pragma unroll
  for (int r = 0; r < 2; r++) {
    if (t[r] < T && pos[r] < 3) {
      const int64_t slot = p.seq_slot[seq[r]];
      const int init = p.seq_init[seq[r]];
#pragma unroll
      for (int j = 1; j < 4; j++)
        if (pos[r] < j)
          xv[r][j] = init ? *reinterpret_cast<const uint4*>(p.conv_state + slot * p.stride_cs_slot +
                                                             (int64_t)(3 + pos[r] - j) * p.stride_cs_row + ch)
                          : make_uint4(0, 0, 0, 0);
    }
  }
  float w[4][8];
  {
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
      for (int tap = 0; tap < 4; tap++) w[tap][c] = bf2f(wb[c * 4 + tap]);
  }
  float yv[2][8];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    float xf[4][8];
#pragma unroll
    for (int j = 0; j < 4; j++) bf8_to_f(xv[r][j], xf[j]);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      float acc = f_mul(xf[0][c], w[3][c]);
      acc = f_fma(xf[1][c], w[2][c], acc);
      acc = f_fma(xf[2][c], w[1][c], acc);
      acc = f_fma(xf[3][c], w[0][c], acc);
      yv[r][c] = div_full(acc, f_add(t_exp(f_sub(0.f, acc)), 1.f));
    }
  }
  if (qk) {
#pragma unroll
    for (int r = 0; r < 2; r++) {
      float ss = f_mul(yv[r][1], yv[r][1]);
      ss = f_fma(yv[r][0], yv[r][0], ss);
#pragma unroll
      for (int c = 2; c < 8; c++) ss = f_fma(yv[r][c], yv[r][c], ss);
#pragma unroll
      for (int d = 8; d >= 1; d >>= 1) ss = f_add(ss, __shfl_xor_sync(0xffffffffu, ss, d));
      const float rn = div_full(1.f, sqrt_approx_ftz(f_add(ss, L2_EPS)));
#pragma unroll
      for (int c = 0; c < 8; c++) yv[r][c] = f_mul(yv[r][c], rn);
    }
  }
  bf16* dst = y < H ? p.q + (int64_t)y * DK : y < 2 * H ? p.k + (int64_t)(y - H) * DK : p.v + (int64_t)(y - 2 * H) * DV;
  const int64_t dst_stride = qk ? (int64_t)H * DK : (int64_t)HV * DV;
  const int c0 = 8 * (lane & 15);
#pragma unroll
  for (int r = 0; r < 2; r++)
    if (t[r] < T)
      *reinterpret_cast<uint4*>(dst + (int64_t)t[r] * dst_stride + c0) =
          make_uint4(pack2(yv[r][0], yv[r][1]), pack2(yv[r][2], yv[r][3]), pack2(yv[r][4], yv[r][5]),
                     pack2(yv[r][6], yv[r][7]));
  if (!qk && tid < CONV_T) {
    // the gates of value head hv for token tt: g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)
    const int tt = t_base + tid, hv = y - 2 * H;
    if (tt < T) {
      const bf16* bar = p.ba + (int64_t)tt * p.stride_ba;
      const float a = bf2f(bar[HV + hv]), b = bf2f(bar[hv]);
      const float xg = f_add(a, bf2f(p.dt_bias[hv]));
      const float s1 = f_add(xg, t_logf(f_add(t_exp(f_sub(0.f, xg)), 1.f)));
      const float l2 = t_logf(f_add(t_exp(xg), 1.f));
      float sp = xg > 0.f ? s1 : l2;
      sp = xg <= SOFTPLUS_THRESHOLD ? sp : xg;
      p.g[(int64_t)tt * HV + hv] = f_mul(f_sub(0.f, t_exp(p.A_log[hv])), sp);
      p.beta[(int64_t)tt * HV + hv] = div_full(1.f, f_add(t_exp(f_sub(0.f, b)), 1.f));
    }
  }
}

// ================================================================================================================
// host
// ================================================================================================================
static Layout layout_of(const torch::Tensor& cu, const torch::Tensor& chunk_indices, const torch::Tensor& chunk_offsets,
                        const torch::Tensor& counts) {
  return Layout{cu.data_ptr<int>(), chunk_indices.data_ptr<int>(), chunk_offsets.data_ptr<int>(),
                counts.data_ptr<int>()};
}

template <typename Kernel>
static void set_smem(Kernel kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

void prep(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor g_cum,
          torch::Tensor w, torch::Tensor u, torch::Tensor am, torch::Tensor cu,
          torch::Tensor chunk_indices, torch::Tensor chunk_offsets, torch::Tensor counts, int64_t grid) {
  TORCH_CHECK(q.size(-1) == DK && v.size(-1) == DV, "gdn_chunk_prep: head dims 128");
  PrepParams p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr());
  p.k = reinterpret_cast<const bf16*>(k.data_ptr());
  p.v = reinterpret_cast<const bf16*>(v.data_ptr());
  p.g = g.data_ptr<float>();
  p.beta = beta.data_ptr<float>();
  p.g_cum = g_cum.data_ptr<float>();
  p.w = reinterpret_cast<bf16*>(w.data_ptr());
  p.u = reinterpret_cast<bf16*>(u.data_ptr());
  p.am = reinterpret_cast<bf16*>(am.data_ptr());
  p.lay = layout_of(cu, chunk_indices, chunk_offsets, counts);
  p.H = q.size(1);
  p.HV = v.size(1);
  const int smem = sizeof(PrepSmem);
  set_smem(prep_kernel, smem);
  prep_kernel<<<grid, PREP_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void state(torch::Tensor q, torch::Tensor k, torch::Tensor u, torch::Tensor w, torch::Tensor am,
           torch::Tensor g_cum, torch::Tensor o, torch::Tensor ssm_state, torch::Tensor seq_slot,
           torch::Tensor seq_init, torch::Tensor cu, torch::Tensor chunk_indices, torch::Tensor chunk_offsets,
           torch::Tensor counts, double scale, int64_t grid) {
  StateParams p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr());
  p.k = reinterpret_cast<const bf16*>(k.data_ptr());
  p.u = reinterpret_cast<const bf16*>(u.data_ptr());
  p.w = reinterpret_cast<const bf16*>(w.data_ptr());
  p.am = reinterpret_cast<const bf16*>(am.data_ptr());
  p.g_cum = g_cum.data_ptr<float>();
  p.o = reinterpret_cast<bf16*>(o.data_ptr());
  p.state = ssm_state.data_ptr<float>();
  p.state_stride = ssm_state.stride(0);
  p.seq_slot = seq_slot.data_ptr<int>();
  p.seq_init = seq_init.data_ptr<int>();
  p.lay = layout_of(cu, chunk_indices, chunk_offsets, counts);
  p.H = k.size(1);
  p.HV = u.size(1);
  p.scale = static_cast<float>(scale);
  const int smem = sizeof(StateSmem) + 1024;
  set_smem(state_kernel, smem);
  state_kernel<<<grid, STATE_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void conv_prep(torch::Tensor x, torch::Tensor ba, torch::Tensor conv_w, torch::Tensor conv_state, torch::Tensor A_log,
               torch::Tensor dt_bias, torch::Tensor tok_seq, torch::Tensor tok_pos, torch::Tensor seq_slot,
               torch::Tensor seq_init, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
               torch::Tensor beta, torch::Tensor counts, int64_t T) {
  TORCH_CHECK(q.size(-1) == DK && v.size(-1) == DV && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  conv_state.stride(2) == 1 && conv_state.stride(1) % 8 == 0 && conv_state.stride(0) % 8 == 0 &&
                  conv_w.is_contiguous() && conv_w.numel() == conv_w.size(0) * 4,
              "gdn_conv_prep: 16-byte channel vectors, head dims 128, a [C, 4] weight");
  TORCH_CHECK(T <= x.size(0) && T <= tok_seq.size(0) && T <= tok_pos.size(0), "gdn_conv_prep: rows");
  ConvPrepParams p{};
  p.x = reinterpret_cast<const bf16*>(x.data_ptr());
  p.stride_x = x.stride(0);
  p.ba = reinterpret_cast<const bf16*>(ba.data_ptr());
  p.stride_ba = ba.stride(0);
  p.conv_w = reinterpret_cast<const bf16*>(conv_w.data_ptr());
  p.conv_state = reinterpret_cast<const bf16*>(conv_state.data_ptr());
  p.stride_cs_slot = conv_state.stride(0);
  p.stride_cs_row = conv_state.stride(1);
  p.A_log = A_log.data_ptr<float>();
  p.dt_bias = reinterpret_cast<const bf16*>(dt_bias.data_ptr());
  p.tok_seq = tok_seq.data_ptr<int>();
  p.tok_pos = tok_pos.data_ptr<int>();
  p.seq_slot = seq_slot.data_ptr<int>();
  p.seq_init = seq_init.data_ptr<int>();
  p.q = reinterpret_cast<bf16*>(q.data_ptr());
  p.k = reinterpret_cast<bf16*>(k.data_ptr());
  p.v = reinterpret_cast<bf16*>(v.data_ptr());
  p.g = g.data_ptr<float>();
  p.beta = beta.data_ptr<float>();
  p.counts = counts.data_ptr<int>();
  p.H = q.size(1);
  p.HV = v.size(1);
  p.x_rows = x.size(0);
  p.tok_rows = std::min(tok_seq.size(0), tok_pos.size(0));
  const dim3 grid((T + CONV_T - 1) / CONV_T, 2 * p.H + p.HV);
  if (grid.x == 0) return;
  conv_prep_kernel<<<grid, CONV_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mixed_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                  torch::Tensor o, torch::Tensor ssm_state, torch::Tensor seq_slot, torch::Tensor counts, double scale,
                  int64_t grid, torch::Tensor x, torch::Tensor conv_state, torch::Tensor seq_start,
                  torch::Tensor cs_slot, torch::Tensor cs_init, int64_t C) {
  DecParams p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr());
  p.k = reinterpret_cast<const bf16*>(k.data_ptr());
  p.v = reinterpret_cast<const bf16*>(v.data_ptr());
  p.g = g.data_ptr<float>();
  p.beta = beta.data_ptr<float>();
  p.o = reinterpret_cast<bf16*>(o.data_ptr());
  p.state = ssm_state.data_ptr<float>();
  p.state_stride = ssm_state.stride(0);
  p.seq_slot = seq_slot.data_ptr<int>();
  p.counts = counts.data_ptr<int>();
  p.H = q.size(1);
  p.HV = v.size(1);
  p.scale = static_cast<float>(scale);
  TORCH_CHECK(C % 8 == 0 && x.stride(0) % 8 == 0 && x.stride(1) == 1 && conv_state.stride(0) % 8 == 0 &&
                  conv_state.stride(1) % 8 == 0 && conv_state.stride(2) == 1,
              "gdn_mixed_decode: conv states need 16-byte channel vectors");
  p.cs.x = reinterpret_cast<const bf16*>(x.data_ptr());
  p.cs.stride_x = x.stride(0);
  p.cs.state = reinterpret_cast<bf16*>(conv_state.data_ptr());
  p.cs.stride_slot = conv_state.stride(0);
  p.cs.stride_row = conv_state.stride(1);
  p.cs.seq_start = seq_start.data_ptr<int>();
  p.cs.seq_slot = cs_slot.data_ptr<int>();
  p.cs.seq_init = cs_init.data_ptr<int>();
  p.cs.C = C;
  dec_kernel<<<grid, DEC_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_chunk

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "gdn_conv_prep(Tensor x, Tensor ba, Tensor conv_w, Tensor conv_state, Tensor A_log, Tensor dt_bias, "
      "Tensor tok_seq, Tensor tok_pos, Tensor seq_slot, Tensor seq_init, Tensor(a!) q, Tensor(b!) k, Tensor(c!) v, "
      "Tensor(d!) g, Tensor(e!) beta, Tensor counts, int T) -> ()");
  m.def(
      "gdn_chunk_prep(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) g_cum, Tensor(b!) w, "
      "Tensor(c!) u, Tensor(d!) am, Tensor cu, Tensor chunk_indices, Tensor chunk_offsets, Tensor counts, "
      "int grid) -> ()");
  m.def(
      "gdn_mixed_decode(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) o, Tensor(b!) ssm_state, "
      "Tensor seq_slot, Tensor counts, float scale, int grid, Tensor x, Tensor(c!) conv_state, Tensor seq_start, "
      "Tensor cs_slot, Tensor cs_init, int C) -> ()");
  m.def(
      "gdn_chunk_state(Tensor q, Tensor k, Tensor u, Tensor w, Tensor am, Tensor g_cum, Tensor(a!) o, "
      "Tensor(b!) ssm_state, Tensor seq_slot, Tensor seq_init, Tensor cu, Tensor chunk_indices, "
      "Tensor chunk_offsets, Tensor counts, float scale, int grid) -> ()");
}

TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) {
  m.impl("gdn_conv_prep", &gdn_chunk::conv_prep);
  m.impl("gdn_chunk_prep", &gdn_chunk::prep);
  m.impl("gdn_chunk_state", &gdn_chunk::state);
  m.impl("gdn_mixed_decode", &gdn_chunk::mixed_decode);
}
