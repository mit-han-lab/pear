// The decode megakernel on sm_120a: its device code (megakernel.cuh: the layouts; megakernel.cu: the torch ops).
#pragma once
#include "megakernel.cuh"
#include "gdn_t.cuh"
#include "attention.cuh"
#include "spec_sample.cuh"

namespace mk {

// ------------------------------------------------------------------------------------------------ shared memory
struct Smem {
  uint8_t* stages;  // NSTAGE x STAGE_BYTES (1024-aligned)
  uint8_t* xb;      // NB x B_SLOT_BYTES: codes [B_CHUNKS][CHUNK_B], then NVFP4 scales [B_CHUNKS][SFB_CHUNK]
  float* accb;      // NACC x [TILE][ACC_COLS]
  uint8_t* stg;     // [STG_BYTES] the prologue's staging buffer
  float* red;       // prologue reductions
  uint64_t* full;   // [NSTAGE] ring stage landed (TMA bytes)
  uint64_t* empty;  // [NSTAGE] ring stage consumed (NMMA warps)
  uint64_t* xfull;  // [NB][BG] B slot built (prologue), by chunk group
  uint64_t* xempty; // [NB] B slot consumed (NMMA warps)
  uint64_t* dfull;  // [NACC] segment slot written (NMMA warps)
  uint64_t* dempty; // [NACC] segment slot read (4 epilogue warps)
  uint64_t* stg_full;  // the staging buffer's bulk copies (the GDN core's state rows)
  uint64_t* bis;       // reserved (initialized, unused: the layout the kernel was validated with)
  uint64_t* xeo;       // [2] reserved (initialized, unused)
  uint32_t* stage_st;  // [NSTAGE][2] per ring slot its stage and its range's end, written before the slot's arrival
  uint32_t* seg_info;  // [NACC][2] reserved (unused)
  uint32_t* dyn_total;  // [2] reserved (zeroed, unused)
  uint32_t* epi_word;  // reserved (unused)
  uint32_t* rel_cnt;   // the GEMV instructions whose op counter this SM's epilogue has released
  uint32_t* stg_ph;    // the prologue's stg_full parity (the GDN core's state rows; a word, not a loop register)
  uint32_t* q_rel;     // reserved (zeroed, unused)
};
// the layout from the dynamic shared memory's base, aligned up to SMEM_ALIGN
__device__ __forceinline__ Smem smem_layout(uint8_t* raw) {
  Smem s;
  uint8_t* p =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(raw) + SMEM_ALIGN - 1) & ~uintptr_t(SMEM_ALIGN - 1));
  s.stages = p, p += NSTAGE * STAGE_BYTES;
  s.xb = p, p += NB * B_SLOT_BYTES;
  s.accb = reinterpret_cast<float*>(p), p += NACC * ACC_SLOT_BYTES;
  s.stg = p, p += STG_BYTES;
  s.red = reinterpret_cast<float*>(p), p += RED_FLOATS * 4;
  uint64_t* b = reinterpret_cast<uint64_t*>(p);
  s.full = b, s.empty = b + NSTAGE, s.xfull = b + 2 * NSTAGE, s.xempty = s.xfull + NB * BG, s.dfull = s.xempty + NB,
  s.dempty = s.dfull + NACC, s.stg_full = s.dempty + NACC;
  s.bis = s.stg_full + 1;
  s.xeo = s.bis + 1;
  s.stage_st = reinterpret_cast<uint32_t*>(s.xeo + 2);
  s.seg_info = s.stage_st + 2 * NSTAGE;
  s.dyn_total = s.seg_info + 2 * NACC;
  s.epi_word = s.dyn_total + 2;
  s.rel_cnt = s.epi_word + 1;
  s.stg_ph = s.rel_cnt + 1;
  s.q_rel = s.stg_ph + 1;
  return s;
}
constexpr int W_ROWS = 64;      // weight rows per MMA warp (of a stage's TILE)
constexpr float E2M1_MAX = 6.0f;  // the largest e2m1 magnitude: an NVFP4 block's scale is amax / E2M1_MAX
constexpr int EPI_WARPS = 4;      // the epilogue warps (one per 32 rows of a tile)
__device__ __forceinline__ uint8_t* b_codes(const Smem& sm, uint32_t slot) { return sm.xb + slot * B_SLOT_BYTES; }
__device__ __forceinline__ uint8_t* b_scales(const Smem& sm, uint32_t slot) {
  return sm.xb + slot * B_SLOT_BYTES + B_CHUNKS * CHUNK_B;
}
// a B chunk row's 16-byte unit u of token row t (swizzled: conflict-free ldmatrix of the 8 token rows)
__device__ __forceinline__ uint32_t b_unit(uint32_t l, uint32_t t, uint32_t u) {
  return l * CHUNK_B + t * 128 + ((u ^ t) << 4);
}

__device__ __forceinline__ uint32_t ld_nc_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// counters: wait until counter c reaches count * (epoch + 1)
__device__ __forceinline__ uint32_t dep_target(const MKArgs& a, uint16_t c, uint32_t epoch) {
  return __ldg(a.ctr_count + c) * (epoch + 1);
}
__device__ __forceinline__ void wait_dep(const MKArgs& a, uint16_t c, uint32_t epoch) {
  if (c == NO_CTR) return;
  wait_counter(a.counters + c, dep_target(a, c, epoch));
}

// ------------------------------------------------------------------------------------------------ producer
__device__ __forceinline__ void produce_gemv(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& cc, int lane) {
  const bool is_nvfp4 = d.kind == KIND_NVFP4;
  const uint32_t bytes = A_BYTES + (is_nvfp4 ? SFA_BYTES : 0);
  const void* tmap = reinterpret_cast<const void*>(d.tmap);
  const uint8_t* sf = reinterpret_cast<const uint8_t*>(d.sf);
  constexpr uint32_t SF_ATOM_K = 64;  // K values per NVFP4 scale atom
  const uint32_t kgroups = d.K / SF_ATOM_K;  // NVFP4 scale atoms per 128-row block of the weights
  auto load = [&](uint32_t st, uint32_t ue) {
    if (cc.n >= NSTAGE) mbar_wait(&sm.empty[cc.slot], cc.phase ^ 1);
    if (lane == 0) {
      const StageAt at = stage_at(d, st);
      uint64_t* bar = &sm.full[cc.slot];
      sm.stage_st[2 * cc.slot] = st, sm.stage_st[2 * cc.slot + 1] = ue;  // read by the MMA after the arrival
      mbar_expect_tx(bar, bytes);
      uint8_t* dst = sm.stages + cc.slot * STAGE_BYTES;
      const uint64_t pol = l2_evict_first();
      tma_load_2d_pol(dst, tmap, at.kc * STAGE_K, at.row0, bar, pol);
      if (is_nvfp4)
        bulk_g2s_pol(dst + A_BYTES, sf + ((size_t)(at.row0 / TILE) * kgroups + at.kc * 4) * SF_ATOM_BYTES, SFA_BYTES,
                     bar, pol);
    }
    __syncwarp();
    cc.next(NSTAGE);
  };
  const uint32_t rk = rot_k(in);  // the rotation: the last unit's k stages first; each slot's range end for the MMA
  for (uint32_t st = in.r1 - rk; st < in.r1; st++) load(st, in.r1);
  for (uint32_t st = in.r0; st < in.r1 - rk; st++) load(st, in.r1 - rk);
}

// ------------------------------------------------------------------------------------------------ MMA
// one stage: rows 64 w .. 64 w + 63 (4 m16 tiles) x 4 k-steps of 32 bytes, into acc (paired: the stage's half)
template <int KIND>
__device__ __forceinline__ void mma_stage(uint32_t sa, const uint8_t* sfs, uint32_t bco, uint32_t bsc, int w, int lane,
                                          float (&acc)[4][4]) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < 4; s++) {
    uint32_t b[2];
    {
      const uint32_t r = lane & 7, u = 2 * s + ((lane >> 3) & 1);
      ldsm_x2(b, bco + r * 128 + ((u ^ r) << 4));
    }
    uint32_t sfb = 0;
    if (KIND == KIND_NVFP4) {
      uint32_t v;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(bsc + g * 16 + s * 4));
      sfb = v;
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      uint32_t a[4];
      const int q = lane >> 3, r8 = lane & 7;
      const int row = w * W_ROWS + mt * 16 + r8 + 8 * (q & 1), u = 2 * s + (q >> 1);
      ldsm_x4(a, sa + row * STAGE_K + ((u ^ (row & 7)) << 4));
      if (KIND == KIND_NVFP4) {
        const int srow = w * W_ROWS + mt * 16 + g + 8 * (t & 1);
        const uint32_t sfa =
            *reinterpret_cast<const uint32_t*>(sfs + s * SF_ATOM_BYTES + (srow & 31) * 16 + ((srow >> 5) & 3) * 4);
        mma_nvf4(acc[mt], a, b, sfa, sfb);
      } else if (KIND == KIND_FP8) {
        mma_fp8(acc[mt], a, b);
      } else {
        mma_bf16(acc[mt], a, b);
      }
    }
  }
}

// a segment's sums into segment slot `slot`: row r's 8 token sums at [r][0..7] (paired: the up half's at [r][8..15])
__device__ __forceinline__ void dump_acc(float* slotp, int w, int lane, const float (&acc)[4][4], int col0) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; mt++) {
    const int r = w * W_ROWS + mt * 16 + g;
    *reinterpret_cast<float2*>(slotp + r * ACC_COLS + col0 + 2 * t) = make_float2(acc[mt][0], acc[mt][1]);
    *reinterpret_cast<float2*>(slotp + (r + 8) * ACC_COLS + col0 + 2 * t) = make_float2(acc[mt][2], acc[mt][3]);
  }
}

template <int KIND>
__device__ __forceinline__ void mma_gemv(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& cc, RingPos& seg,
                                         uint32_t bslot, uint32_t bph, int w, int lane, uint64_t* tr) {
  const ChunkRange cr = chunk_range(in, d);
  // each stage waits for its B chunk's group (a completed phase returns at once; slots in stage order), every group
  // by the instruction's end (each barrier's phase completes once per instruction: an empty group by an arrival)
  const uint32_t bco0 = smem_u32(b_codes(sm, bslot)), bsc0 = smem_u32(b_scales(sm, bslot));
  float acc[2][4][4];
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[h][i][0] = acc[h][i][1] = acc[h][i][2] = acc[h][i][3] = 0.f;
  // the first stage's B group before any ring slot: the slots an earlier attention item filled are complete once B
  // is (it is built after the dependency); waiting on the slot first, the MMA could pass a parity wait two phases
  // ahead of the producer and read stale slot words
  mbar_wait(&sm.xfull[bslot * BG + min(rot_group(in), (uint32_t)BG - 1u)], bph);
  // each stage's index and its range's end come from its ring slot's words, in the producer's order (the rotation's
  // [r1 - k, r1) then [r0, r1 - k)); c counts the stages
  for (uint32_t c = in.r0; c < in.r1; c++) {
    mbar_wait(&sm.full[cc.slot], cc.phase);
    const uint32_t st = *reinterpret_cast<volatile uint32_t*>(sm.stage_st + 2 * cc.slot);
    const uint32_t rend = *reinterpret_cast<volatile uint32_t*>(sm.stage_st + 2 * cc.slot + 1);
    const StageAt at = stage_at(d, st);
    const uint32_t l = chunk_slot(at.kc, cr);
    mbar_wait(&sm.xfull[bslot * BG + l / B_GCH], bph);
    if (tr && c == in.r0 && w == 0 && lane == 0) tr[TR_MMA0] = gtimer();
    const uint32_t sa = smem_u32(sm.stages + cc.slot * STAGE_BYTES);
    const uint8_t* sfs = sm.stages + cc.slot * STAGE_BYTES + A_BYTES;
    if (at.half)
      mma_stage<KIND>(sa, sfs, bco0 + l * CHUNK_B, bsc0 + l * SFB_CHUNK, w, lane, acc[1]);
    else
      mma_stage<KIND>(sa, sfs, bco0 + l * CHUNK_B, bsc0 + l * SFB_CHUNK, w, lane, acc[0]);
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.empty[cc.slot]);
    cc.next(NSTAGE);
    const bool ends_segment = st + 1 == rend || stage_at(d, st + 1).unit != at.unit;
    if (ends_segment) {
#pragma unroll 1
      for (int h = 0; h < (d.paired ? 2 : 1); h++) {  // paired: the gate half, then the up half
        mbar_wait(&sm.dempty[seg.slot], seg.phase ^ 1);
        float* sp = sm.accb + seg.slot * (TILE * ACC_COLS);
        dump_acc(sp, w, lane, h ? acc[1] : acc[0], 0);
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.dfull[seg.slot]);
        seg.next(NACC);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[h][i][0] = acc[h][i][1] = acc[h][i][2] = acc[h][i][3] = 0.f;
    }
  }
#pragma unroll
  for (int g = 0; g < BG; g++) mbar_wait(&sm.xfull[bslot * BG + g], bph);
  if (tr && w == 0 && lane == 0) tr[TR_MMA1] = gtimer();
}

// ------------------------------------------------------------------------------------------------ epilogue
__device__ __forceinline__ void atom_add_u64(unsigned long long* p, unsigned long long v, unsigned long long& r) {
  asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;" : "=l"(r) : "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void red_add_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// EPI_SILU: output row o of pair p (gate sums g, up sums u): act = bf16(silu(bf16(alpha g)) * bf16(alpha u)) in vLLM's
// NVFP4 quantization (the consumer's global scale; the 16 outputs of a block are 16 consecutive lanes), into the
// published operand (codes [n_out / 256][NT][128], scales after them [n_out / 256][NT][16])
__device__ __forceinline__ void silu_quant(const OpDesc& d, uint32_t o, int lane, const float (&g)[NT],
                                           const float (&u)[NT], uint32_t T) {
  const float gs = d.act_gs, alpha = d.alpha;
  uint8_t* act = reinterpret_cast<uint8_t*>(d.out);
  const uint32_t c = o / 256, b = (o % 256) / 2;
  uint8_t* codes = act + (size_t)c * CHUNK_B;  // token t's byte b at pub_byte(t, b)
  uint8_t* scales = act + (size_t)(d.n_out / 256) * CHUNK_B + (size_t)c * SFB_CHUNK + (o % 256) / 16;
  float x[NT], amax[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) {
    const float gg = bf16_round(alpha * g[t]), uu = bf16_round(alpha * u[t]);
    x[t] = bf16_round(silu_f32(gg) * uu);
    amax[t] = fabsf(x[t]);
  }
#pragma unroll
  for (int s = 1; s < 16; s <<= 1)
#pragma unroll
    for (int t = 0; t < NT; t++) amax[t] = fmaxf(amax[t], __shfl_xor_sync(0xffffffffu, amax[t], s));
  const float r6 = rcp_approx_ftz(E2M1_MAX), rgs = rcp_approx_ftz(gs);
#pragma unroll
  for (int t = 0; t < NT; t++) {
    const __nv_fp8_e4m3 sf(gs * (amax[t] * r6));
    const float sfq = float(sf);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rgs) : 0.f;
    const float xn = __shfl_xor_sync(0xffffffffu, x[t], 1);
    const uint32_t cb = f32x2_to_e2m1x2(x[t] * osc, xn * osc);
    if (t < (int)T) {
      if ((lane & 1) == 0) codes[pub_byte(t, b)] = (uint8_t)cb;
      if ((lane & 15) == 0) scales[t * 16] = *reinterpret_cast<const uint8_t*>(&sf);
    } else {
      if ((lane & 1) == 0) codes[pub_byte(t, b)] = 0;
      if ((lane & 15) == 0) scales[t * 16] = 0;
    }
  }
}

// the grouped conv's coefficients (OpDesc::coef): per token a row of 2 sides x 2 taps x 320 groups of 16 channels
constexpr uint32_t COEF_ROW = 1280;
constexpr uint32_t COEF_SIDE = 640;
constexpr uint32_t COEF_TAP = 320;
constexpr uint32_t CONV_GROUP_CH = 16;
// DF_COLFIN: a column's 8 values finished by the part that completes its warp's 32-column count, which reloads the
// totals until their count bits say every part is in (no fence: it would wait for in-flight TMA) and zeroes them.
// DF_CONV_FIN: x = conv side 1 of bf16(sums) + the residual (x_out), its squares into sq; else bf16(sums) into out
__device__ __forceinline__ void colfin_tile(const OpDesc& d, unsigned long long* acc, uint32_t tile, uint32_t idx,
                                            uint32_t rows_total, float (&v)[NT], unsigned long long (&sq)[NT], int lane,
                                            uint32_t epoch, uint32_t T) {
  const uint32_t npt = reinterpret_cast<const uint8_t*>(d.parts)[tile];
  const bool has_column = idx < d.n_out;
#pragma unroll
  for (int t = 0; t < NT; t++) sq[t] = 0ull;
  bool is_finisher = true;
  if (npt > 1) {
    __syncwarp();
    uint32_t* ctr = reinterpret_cast<uint32_t*>(d.colctr) + tile * EPI_WARPS + (threadIdx.x >> 5) - W_EPI0;
    uint32_t count_before = 0;
    if (lane == 0) count_before = atomicAdd(ctr, 1u);
    count_before = __shfl_sync(0xffffffffu, count_before, 0);
    is_finisher = count_before + 1 == npt * (epoch + 1);
    if (!is_finisher) return;
    if (has_column) {
#pragma unroll
      for (int t = 0; t < NT; t++) {
        if (t < (int)T) {
          unsigned long long* w = acc + (size_t)t * rows_total + idx;
          unsigned long long tot;
          do {
            tot = ld_relaxed_u64(w);
          } while ((uint32_t)(tot & ((1u << CNT_BITS) - 1)) != npt);
          __stcg(w, 0ull);
          v[t] = __ll2float_rn((long long)tot >> CNT_BITS) * CNT_INV;
        }
      }
    }
  }
  if (!has_column) return;
  const uint32_t no = d.n_out;
  if (!(d.dflags & DF_CONV_FIN)) {
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(d.out);
#pragma unroll
    for (int t = 0; t < NT; t++)
      if (t < (int)T) out[(size_t)t * no + idx] = __float2bfloat16(v[t]);
    return;
  }
  const __nv_bfloat16* bk = reinterpret_cast<const __nv_bfloat16*>(d.base) + 2 * K_NORM;  // side 1: [tap][5120]
  const float b0 = __bfloat162float(bk[idx]), b1 = __bfloat162float(bk[K_NORM + idx]);
  const __nv_bfloat16* cf = reinterpret_cast<const __nv_bfloat16*>(d.coef) + COEF_SIDE + idx / CONV_GROUP_CH;
  const void* ri = reinterpret_cast<const void*>(d.res_in);
  float* xo = reinterpret_cast<float*>(d.x_out);
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(d.out);
  const bool uses_fma = (d.dflags & DF_CONV_FMA) != 0u, rounds_res = (d.dflags & DF_RES_ROUND) != 0u;
  float prev_o = 0.f;
#pragma unroll
  for (int t = 0; t < NT; t++) {
    const float o = bf16_round(v[t]);
    if (out) out[(size_t)t * no + idx] = __float2bfloat16(o);
    const float c0 = __fadd_rn(b0, __bfloat162float(cf[(size_t)t * COEF_ROW])),
                c1 = __fadd_rn(b1, __bfloat162float(cf[(size_t)t * COEF_ROW + COEF_TAP]));
    const float tail = __fmul_rn(__fmul_rn(c1, prev_o), t >= 1 ? 1.0f : 0.0f);
    const float y = uses_fma ? __fmaf_rn(c0, o, tail) : __fadd_rn(__fmul_rn(c0, o), tail);
    prev_o = o;
    float r = 0.f;
    if (ri) r = d.res_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ri)[(size_t)t * no + idx])
                           : reinterpret_cast<const float*>(ri)[(size_t)t * no + idx];
    if (rounds_res) r = bf16_round(r);
    const float x = __fadd_rn(y, r);
    if (xo) xo[(size_t)t * no + idx] = x;
    sq[t] = t < (int)T ? (unsigned long long)__float2ll_rn(x * x * SSQ_SCALE) : 0ull;
  }
}

template <int V>
__device__ __forceinline__ void epilogue_gemv(const Instr& in, const OpDesc& d, const Smem& sm, RingPos& seg, int ept,
                                              uint32_t epoch, uint32_t T) {
  const int lane = ept & 31;
  const int row = ept;  // the output row of the tile this thread finishes
  const uint32_t rows_total = d.paired ? d.n_out : d.mtiles * TILE;  // an output vector's stride (per token)
  const uint32_t spu = d.paired ? 2 * d.spm : d.bw;  // stages per unit
  // the counted sums: one copy, every word back to zero by the part that completes it (the next op sharing the buffer
  // starts from zeros: its instructions start after this op's counter)
  unsigned long long* acc = reinterpret_cast<unsigned long long*>(d.acc);
  if (d.epi == EPI_RED && d.xr && in.idx == 0) {  // the next launch's sums of squares (instruction 0: mtiles x NT)
    unsigned long long* next_ssq =
        reinterpret_cast<unsigned long long*>(d.ssq) + (size_t)((epoch + 1) & 1) * d.mtiles * NT;
    for (uint32_t i = ept; i < d.mtiles * NT; i += NEPI) next_ssq[i] = 0ull;
  }
  // the instruction's segments, from the stage geometry; the rotation: the segment [r1 - k, r1) first, then those of
  // [r0, r1 - k); rend: the range's end, r1 for the first
  uint32_t st = rot_k(in) ? in.r1 - rot_k(in) : in.r0, rend = in.r1;
  for (;;) {
    if (st >= rend) {
      if (rend == in.r1 && rot_k(in))
        rend = in.r1 - rot_k(in), st = in.r0;
      else
        break;
    }
    const uint32_t unit = stage_at(d, st).unit, ue = min(rend, (unit + 1) * spu);
    float v[NT], u[NT];
    // EPI_RED with xr: the row's residuals loaded before the MMA's sums are waited for, under the segment's stream,
    // not one more round trip on the finisher's path
    float rp[NT];
    {
      const bool loads_res = d.epi == EPI_RED && d.xr && d.res_in != 0 && !(HAS_D<V> && (d.dflags & DF_COLFIN));
      const uint32_t idx = (unit % d.mtiles) * TILE + row;
#pragma unroll
      for (int t = 0; t < NT; t++)
        rp[t] = loads_res && t < (int)T && idx < d.n_out
                    ? (d.res_bf16 ? __bfloat162float(__ldcg(reinterpret_cast<const __nv_bfloat16*>(d.res_in) +
                                                            (size_t)t * d.n_out + idx))
                                  : __ldcg(reinterpret_cast<const float*>(d.res_in) + (size_t)t * d.n_out + idx))
                    : 0.f;
    }
    // EPI_SILU's prefix part of a split pair: the suffix part's flag waited and its sums loaded before the MMA's sums,
    // under the segment's stream (the suffix runs at the start of the next SM's range), not in the op's drain
    float sp_g[NT], sp_u[NT];
    const bool loads_suffix = d.epi == EPI_SILU && st == unit * spu && ue != (unit + 1) * spu;
    if (loads_suffix) {
      if (ept == 0) wait_counter(reinterpret_cast<uint32_t*>(d.sflag) + unit, epoch + 1);
      named_sync(BAR_EPI, NEPI);
      const float* spart = reinterpret_cast<const float*>(d.spart) + ((size_t)unit * TILE + row) * (2 * NT);
#pragma unroll
      for (int k = 0; k < NT; k++) sp_g[k] = __ldcg(spart + k), sp_u[k] = __ldcg(spart + NT + k);
    }
    {
      mbar_wait(&sm.dfull[seg.slot], seg.phase);
      const float* sp = sm.accb + seg.slot * (TILE * ACC_COLS) + row * ACC_COLS;
#pragma unroll
      for (int k = 0; k < NT; k += 4) {
        const float4 a = *reinterpret_cast<const float4*>(sp + k);
        v[k] = a.x, v[k + 1] = a.y, v[k + 2] = a.z, v[k + 3] = a.w;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.dempty[seg.slot]);
      seg.next(NACC);
    }
    const bool is_first_part = st == unit * spu, is_last_part = ue == (unit + 1) * spu;
    if (d.paired) {  // the up half, the slot's next use
      mbar_wait(&sm.dfull[seg.slot], seg.phase);
      const float* sp = sm.accb + seg.slot * (TILE * ACC_COLS) + row * ACC_COLS;
#pragma unroll
      for (int k = 0; k < NT; k += 4) {
        const float4 a = *reinterpret_cast<const float4*>(sp + k);
        u[k] = a.x, u[k + 1] = a.y, u[k + 2] = a.z, u[k + 3] = a.w;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.dempty[seg.slot]);
      seg.next(NACC);
    }
    if (d.epi == EPI_SILU) {
      float* spart =
          reinterpret_cast<float*>(d.spart) + ((size_t)unit * TILE + row) * (2 * NT);  // [pair][row][g 8 | u 8]
      uint32_t* flag = reinterpret_cast<uint32_t*>(d.sflag) + unit;
      if (!is_first_part) {  // the suffix part: its sums for the prefix part
#pragma unroll
        for (int k = 0; k < NT; k++) __stcg(spart + k, v[k]), __stcg(spart + NT + k, u[k]);
        named_sync(BAR_EPI, NEPI);
        if (ept == 0) red_release_add(flag, 1);
      } else {
        if (!is_last_part) {
#pragma unroll
          for (int k = 0; k < NT; k++) v[k] += sp_g[k], u[k] += sp_u[k];  // waited and loaded at the segment's start
        }
        silu_quant(d, unit * TILE + row, lane, v, u, T);
      }
    } else if (d.epi == EPI_LOGITS) {
      const uint32_t idx = (unit % d.mtiles) * TILE + row;
      // a tile in two parts: the later part (an instruction's first segment) leaves its sums in its instruction's slot
      // as tagged words f32 bits * 2^32 + (pool_len + launch parity), 8-byte single-copy atomic (no fence: one waits
      // for the SM's in-flight TMA); the earlier part reloads until the tags are this launch's
      if (!(is_first_part && is_last_part)) {
        unsigned long long* sl = reinterpret_cast<unsigned long long*>(d.spart) +
                                 ((size_t)(is_first_part ? in.idx + 1 : in.idx) * TILE + row) * NT;
        constexpr unsigned long long TAG_MASK = 0xffffffffull;  // a word's tag: its low 32 bits
        const unsigned long long tag = d.pool_len + (epoch & 1u);
        if (!is_first_part) {
#pragma unroll
          for (int t = 0; t < NT; t += 2)
            st_cg_v2_u64(sl + t, (unsigned long long)__float_as_uint(v[t]) << 32 | tag,
                         (unsigned long long)__float_as_uint(v[t + 1]) << 32 | tag);
        } else {
#pragma unroll
          for (int t = 0; t < NT; t += 2) {
            unsigned long long w0, w1;
            do {
              ld_relaxed_v2_u64(sl + t, w0, w1);
            } while ((w0 & TAG_MASK) != tag || (w1 & TAG_MASK) != tag);
            v[t] += __uint_as_float((uint32_t)(w0 >> 32)), v[t + 1] += __uint_as_float((uint32_t)(w1 >> 32));
          }
        }
      }
      if (is_first_part && idx < d.n_out) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(d.out);
#pragma unroll
        for (int t = 0; t < NT; t++)
          if (t < (int)T) out[(size_t)t * d.n_out + idx] = __float2bfloat16(d.alpha * v[t]);
      }
    } else if (HAS_D<V> && (d.dflags & DF_COLFIN)) {  // the draft's: a column's 8 values finished by one part
      const uint32_t tile = unit % d.mtiles, idx = tile * TILE + row;
      const uint32_t npt = reinterpret_cast<const uint8_t*>(d.parts)[tile];
      if (idx < d.n_out) {
        const float alpha = d.alpha;
#pragma unroll
        for (int t = 0; t < NT; t++) {
          const long long fx = __float2ll_rn(v[t] * alpha * CNT_SCALE);
          if (npt == 1) v[t] = __ll2float_rn(fx) * CNT_INV;
          else if (t < (int)T)
            red_add_u64(acc + (size_t)t * rows_total + idx, (unsigned long long)fx * (1ull << CNT_BITS) + 1ull);
        }
      }
      unsigned long long sq[NT];
      colfin_tile(d, acc, tile, idx, rows_total, v, sq, lane, epoch, T);
      if (d.xr) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) sq[t] += __shfl_xor_sync(0xffffffffu, sq[t], o);
        }
        if (lane == 0) {
          unsigned long long* sp =
              reinterpret_cast<unsigned long long*>(d.ssq) + ((size_t)(epoch & 1) * d.mtiles + tile) * NT;
#pragma unroll
          for (int t = 0; t < NT; t++)
            if (t < (int)T && sq[t]) atomicAdd(sp + t, sq[t]);
        }
      }
    } else {  // EPI_RED: counted fixed-point words; the part that completes a value stores its bf16
      const uint32_t tile = unit % d.mtiles, idx = tile * TILE + row;
      // xr: the finisher of a value also forms x = output + residual (fp32 x_out, bf16 r_out) and adds round(x^2 2^24)
      // into its token's sum of squares of the tile (the warp's sums, then one atomic per token: exact, in any order)
      unsigned long long sq[NT];
#pragma unroll
      for (int t = 0; t < NT; t++) sq[t] = 0ull;
      const uint32_t npt = reinterpret_cast<const uint8_t*>(d.parts)[tile];
      if (idx < d.n_out) {
        const bool is_solo = npt == 1;  // this segment is the whole value
        const float alpha = d.alpha;
        __nv_bfloat16* const out = reinterpret_cast<__nv_bfloat16*>(d.out);
        const uint32_t xr = d.xr, no = d.n_out;
        float* const xo = reinterpret_cast<float*>(d.x_out);
        __nv_bfloat16* const ro = reinterpret_cast<__nv_bfloat16*>(d.r_out);
        float res[NT];
        // the residuals loaded at the segment's start; a split value takes them after the atomics' results (issued
        // before them they delay the atomics' return); none: the fc's x = out
        auto load_res = [&]() {
#pragma unroll
          for (int t = 0; t < NT; t++) res[t] = rp[t];
          if (d.dflags & DF_RES_ROUND) {
#pragma unroll
            for (int t = 0; t < NT; t++) res[t] = bf16_round(res[t]);
          }
        };
        if (is_solo) load_res();
        auto finish = [&](int t, float val) {
          const __nv_bfloat16 o = __float2bfloat16(val);
          if (out) out[(size_t)t * no + idx] = o;  // xr: optional
          if (xr) {
            const float x = __bfloat162float(o) + res[t];
            if (xo) xo[(size_t)t * no + idx] = x;
            if (ro) ro[(size_t)t * no + idx] = __float2bfloat16(x);
            sq[t] = (unsigned long long)__float2ll_rn(x * x * SSQ_SCALE);
          }
        };
        if (is_solo) {
#pragma unroll
          for (int t = 0; t < NT; t++)
            if (t < (int)T) finish(t, __ll2float_rn(__float2ll_rn(v[t] * alpha * CNT_SCALE)) * CNT_INV);
        } else {
          unsigned long long tot[NT], inc[NT];
#pragma unroll
          for (int t = 0; t < NT; t++) {
            const long long fx = __float2ll_rn(v[t] * alpha * CNT_SCALE);
            inc[t] = (unsigned long long)fx * (1ull << CNT_BITS) + 1u;
            tot[t] = 0ull;
            if (t < (int)T) atom_add_u64(acc + (size_t)t * rows_total + idx, inc[t], tot[t]);
          }
          if (xr) {
            unsigned long long dep = 0ull;  // the loads after every atomic's result
#pragma unroll
            for (int t = 0; t < NT; t++) dep |= tot[t] & 0ull;
            if (dep == 0ull) load_res();
          }
#pragma unroll
          for (int t = 0; t < NT; t++) {
            tot[t] += inc[t];
            if (t < (int)T && (uint32_t)(tot[t] & ((1u << CNT_BITS) - 1)) == npt) {
              __stcg(acc + (size_t)t * rows_total + idx, 0ull);  // the value complete: its word back to zero
              finish(t, __ll2float_rn((long long)tot[t] >> CNT_BITS) * CNT_INV);
            }
          }
        }
      }
      if (d.xr) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) sq[t] += __shfl_xor_sync(0xffffffffu, sq[t], o);
        }
        if (lane == 0) {
          unsigned long long* sp =
              reinterpret_cast<unsigned long long*>(d.ssq) + ((size_t)(epoch & 1) * d.mtiles + tile) * NT;
#pragma unroll
          for (int t = 0; t < NT; t++)
            if (t < (int)T && sq[t]) atomicAdd(sp + t, sq[t]);
        }
      }
    }
    st = ue;
  }
}

// ------------------------------------------------------------------------------------------------ prologue
// rsqrtf(s / 5120.f + eps) bit for bit (div.rn's fast path as three FMAs, then rsqrt.approx.ftz)
__device__ __forceinline__ float rms_rsqrt5120(float s, float eps) {
  constexpr uint32_t RCP_N_BITS = 0x394ccccdu;  // 1 / 5120.f's bits
  constexpr float NEG_N = -5120.f;
  constexpr uint32_t F32_INF_BITS = 0x7f800000u;
  const float r1 = __uint_as_float(RCP_N_BITS);
  const float q0 = __fmaf_rn(r1, s, 0.f), e = __fmaf_rn(q0, NEG_N, s), q = __fmaf_rn(r1, e, q0);
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(q + eps));
  return s == __uint_as_float(F32_INF_BITS) ? 0.f : y;
}
__device__ __forceinline__ void bf16r2(float& a, float& b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  const uint32_t u = *reinterpret_cast<const uint32_t*>(&h);
  a = __uint_as_float(u << 16);
  b = __uint_as_float(u & 0xffff0000u);
}
constexpr int NG5 = K_NORM / 4 / NPRO;  // groups of 4 values per prologue thread in a 5120-value row
static_assert(NG5 * 4 * NPRO == (int)K_NORM, "the RMSNorm rows");

// a group of 4 values (vi .. vi + 3) of token row t into the B slot, if its chunk is the instruction's
template <int KIND>
__device__ __forceinline__ void enc_store4(uint32_t xbs, uint32_t sfs, const ChunkRange& cr, uint32_t vi, uint32_t t,
                                           float (&v)[4], float gs, float rgs, float inv_s) {
  const uint32_t cv = KIND == KIND_NVFP4 ? 256 : KIND == KIND_FP8 ? 128 : 64;
  const uint32_t kc = vi / cv, l = chunk_slot(kc, cr);
  if (KIND == KIND_NVFP4) {  // the 4 lanes of a 16-value block together (vLLM's quantization of bf16(v))
    bf16r2(v[0], v[1]);
    bf16r2(v[2], v[3]);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const __nv_fp8_e4m3 q(gs * (amax * rcp_approx_ftz(E2M1_MAX)));
    const float sfq = float(q);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rgs) : 0.f;
    const uint32_t c = f32x2_to_e2m1x2(v[0] * osc, v[1] * osc) | (f32x2_to_e2m1x2(v[2] * osc, v[3] * osc) << 8);
    if (l < cr.nck) {
      const uint32_t byte = (vi % 256) / 2;
      sts_u16(xbs + b_unit(l, t, byte >> 4) + (byte & 15), c);
      if ((vi & 15) == 0) sts_u8(sfs + l * SFB_CHUNK + t * 16 + (vi % 256) / 16, *reinterpret_cast<const uint8_t*>(&q));
    }
  } else if (KIND == KIND_FP8) {
    mul2(v[0], v[1], inv_s, inv_s), mul2(v[2], v[3], inv_s, inv_s);
    const uint32_t w = f32x2_to_e4m3x2(v[0], v[1]) | (f32x2_to_e4m3x2(v[2], v[3]) << 16);
    if (l < cr.nck) {
      const uint32_t byte = vi % 128;
      sts_u32(xbs + b_unit(l, t, byte >> 4) + (byte & 15), w);
    }
  } else {
    const __nv_bfloat162 h0 = __floats2bfloat162_rn(v[0], v[1]), h1 = __floats2bfloat162_rn(v[2], v[3]);
    if (l < cr.nck) {
      const uint32_t byte = (vi % 64) * 2;
      const uint32_t a = xbs + b_unit(l, t, byte >> 4) + (byte & 15);
      asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(a), "r"(*reinterpret_cast<const uint32_t*>(&h0)),
                   "r"(*reinterpret_cast<const uint32_t*>(&h1))
                   : "memory");
    }
  }
}

// PRO_NORM_IN / PRO_NORM_RESID (K = 5120): per token row, x = the producer's bf16 output (+ the residual), the row's
// RMSNorm (x * rs) * (1 + w), encoded into the instruction's chunks; NORM_RESID stores the instruction's slice of x
// (fp32 x_out, bf16 r_out)
template <int KIND, bool RESID>
__device__ __forceinline__ void pro_norm(const Instr& in, const OpDesc& d, const Smem& sm, int pt, uint32_t bslot,
                                         uint32_t T) {
  const ChunkRange cr = chunk_range(in, d);
  const uint32_t xbs = smem_u32(b_codes(sm, bslot)), sfs = smem_u32(b_scales(sm, bslot));
  const float gs = d.gs, eps = d.eps;
  const float inv_s = KIND == KIND_FP8 ? div_full(1.0f, gs) : 0.f;  // FP8: gs holds the input scale s_in
  const float rgs = KIND == KIND_NVFP4 ? rcp_approx_ftz(gs) : 0.f;
  const __nv_bfloat16* xin = reinterpret_cast<const __nv_bfloat16*>(d.in0);
  const uint32_t s0 = d.slice * in.idx, s1 = min(K_NORM, d.slice * (in.idx + 1));
  float w1[NG5][4];
#pragma unroll
  for (int i = 0; i < NG5; i++) {
    const uint2 wv = __ldg(reinterpret_cast<const uint2*>(d.nw) + pt + i * NPRO);
    w1[i][0] = bf16lo(wv.x) + 1.0f, w1[i][1] = bf16hi(wv.x) + 1.0f;
    w1[i][2] = bf16lo(wv.y) + 1.0f, w1[i][3] = bf16hi(wv.y) + 1.0f;
  }
#pragma unroll 1
  for (uint32_t t = 0; t < NT; t++) {
    float x[NG5][4];
    if (t < T) {
#pragma unroll
      for (int i = 0; i < NG5; i++) {
        const uint32_t q = pt + i * NPRO;
        const uint2 ov = __ldcg(reinterpret_cast<const uint2*>(xin + (size_t)t * K_NORM) + q);
        if (RESID && d.res_bf16) {
          const uint2 r = __ldcg(reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(d.res_in) +
                                                                (size_t)t * K_NORM) +
                                 q);
          x[i][0] = bf16lo_add(ov.x, bf16lo(r.x)), x[i][1] = bf16hi_add(ov.x, bf16hi(r.x));
          x[i][2] = bf16lo_add(ov.y, bf16lo(r.y)), x[i][3] = bf16hi_add(ov.y, bf16hi(r.y));
        } else if (RESID) {
          const float4 r = __ldcg(
              reinterpret_cast<const float4*>(reinterpret_cast<const float*>(d.res_in) + (size_t)t * K_NORM) +
              q);
          x[i][0] = bf16lo_add(ov.x, r.x), x[i][1] = bf16hi_add(ov.x, r.y);
          x[i][2] = bf16lo_add(ov.y, r.z), x[i][3] = bf16hi_add(ov.y, r.w);
        } else {
          x[i][0] = bf16lo(ov.x), x[i][1] = bf16hi(ov.x), x[i][2] = bf16lo(ov.y), x[i][3] = bf16hi(ov.y);
        }
        if (RESID && q * 4 >= s0 && q * 4 < s1) {
          if (d.x_out)
            reinterpret_cast<float4*>(reinterpret_cast<float*>(d.x_out) + (size_t)t * K_NORM)[q] =
                make_float4(x[i][0], x[i][1], x[i][2], x[i][3]);
          if (d.r_out) {
            const __nv_bfloat162 h0 = __floats2bfloat162_rn(x[i][0], x[i][1]),
                                 h1 = __floats2bfloat162_rn(x[i][2], x[i][3]);
            reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(d.r_out) + (size_t)t * K_NORM)[q] =
                make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < NG5; i++) x[i][0] = x[i][1] = x[i][2] = x[i][3] = 0.f;
    }
    float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      fma2(s4[0], s4[1], x[i][0], x[i][1], x[i][0], x[i][1]);
      fma2(s4[2], s4[3], x[i][2], x[i][3], x[i][2], x[i][3]);
    }
    float ss = (s4[0] + s4[1]) + (s4[2] + s4[3]);
    {
      float* ra = sm.red + (t & 1) * (NPRO / 32);
      const float wsum = warp_sum(ss);
      if ((pt & 31) == 0) ra[pt >> 5] = wsum;
      named_sync(BAR_PRO, NPRO);
      ss = ((((((0.f + ra[0]) + ra[1]) + ra[2]) + ra[3]) + ra[4]) + ra[5] + ra[6]) + ra[7];
    }
    const float rs = rms_rsqrt5120(ss, eps);
#pragma unroll
    for (int i = 0; i < NG5; i++) {
      float v[4];
#pragma unroll
      for (int k = 0; k < 4; k += 2) {
        v[k] = x[i][k], v[k + 1] = x[i][k + 1];
        mul2(v[k], v[k + 1], rs, rs);
        mul2(v[k], v[k + 1], w1[i][k], w1[i][k + 1]);
      }
      if (t >= T) v[0] = v[1] = v[2] = v[3] = 0.f;
      enc_store4<KIND>(xbs, sfs, cr, (pt + i * NPRO) * 4, t, v, gs, rgs, inv_s);
    }
  }
}

// PRO_NORM_X (K = 5120): rstd per token from the producer's published sums of squares (its M tiles' exact fixed-point
// sums, added in order), then the instruction's chunks of x (fp32, or bf16 x_bf16) normed (x * rs) * (1 + w) and
// encoded; every load of a batch in flight before its encode
template <int KIND>
__device__ __forceinline__ void pro_norm_x(const Instr& in, const OpDesc& d, const Smem& sm, int pt, uint32_t bslot,
                                           uint32_t T, uint32_t epoch) {
  const ChunkRange cr = chunk_range(in, d);
  const uint32_t xbs = smem_u32(b_codes(sm, bslot)), sfs = smem_u32(b_scales(sm, bslot));
  const float gs = d.gs, eps = d.eps;
  const float inv_s = KIND == KIND_FP8 ? div_full(1.0f, gs) : 0.f;
  const float rgs = KIND == KIND_NVFP4 ? rcp_approx_ftz(gs) : 0.f;
  float* rstd = sm.red;  // [NT]
  if (pt < NT) {
    const unsigned long long* sp =
        reinterpret_cast<const unsigned long long*>(d.ssq_in) + (size_t)(epoch & 1) * d.ssq_tiles * NT;
    unsigned long long tot = 0ull;
    for (uint32_t i = 0; i < d.ssq_tiles; i++) tot += __ldcg(sp + (size_t)i * NT + pt);
    rstd[pt] = rms_rsqrt5120(__ull2float_rn(tot) * SSQ_INV, eps);
  }
  named_sync(BAR_PRO, NPRO);
  constexpr uint32_t CV = KIND == KIND_NVFP4 ? 256 : KIND == KIND_FP8 ? 128 : 64;  // values per chunk row
  constexpr uint32_t QPR = CV / 4;  // groups of 4 per chunk row
  const uint32_t ng = cr.nck * NT * QPR;
  const bool is_x_bf16 = d.x_bf16;
  const uint8_t* xin = reinterpret_cast<const uint8_t*>(d.in0);
  const __nv_bfloat16* nw = reinterpret_cast<const __nv_bfloat16*>(d.nw);
  constexpr int BATCH = 8;
#pragma unroll 1
  for (uint32_t g0 = 0; g0 < ng; g0 += BATCH * NPRO) {
    float4 xv[BATCH];
    uint2 wv[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; b++) {  // group g = (l NT + t) QPR + q: a 16-value block = 4 consecutive lanes
      const uint32_t g = g0 + b * NPRO + pt;
      xv[b] = make_float4(0.f, 0.f, 0.f, 0.f), wv[b] = make_uint2(0, 0);
      if (g < ng) {
        const uint32_t q = g % QPR, lt = g / QPR, t = lt % NT, l = lt / NT;
        const uint32_t col = slot_chunk(l, cr) * CV + q * 4;
        if (t < T) {
          if (is_x_bf16) {
            const uint2 u = __ldcg(reinterpret_cast<const uint2*>(xin + ((size_t)t * K_NORM + col) * 2));
            xv[b] = make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
          } else {
            xv[b] = __ldcg(reinterpret_cast<const float4*>(xin + ((size_t)t * K_NORM + col) * 4));
          }
        }
        wv[b] = __ldg(reinterpret_cast<const uint2*>(nw + col));
      }
    }
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
      const uint32_t g = g0 + b * NPRO + pt;
      const uint32_t q = g % QPR, lt = g / QPR, t = lt % NT, l = lt / NT;
      const float rs = t < T ? rstd[t] : 0.f;
      float v[4] = {xv[b].x, xv[b].y, xv[b].z, xv[b].w};
      const float w1[4] = {bf16lo(wv[b].x) + 1.0f, bf16hi(wv[b].x) + 1.0f, bf16lo(wv[b].y) + 1.0f,
                           bf16hi(wv[b].y) + 1.0f};
#pragma unroll
      for (int k = 0; k < 4; k += 2) {
        mul2(v[k], v[k + 1], rs, rs);
        mul2(v[k], v[k + 1], w1[k], w1[k + 1]);
      }
      // g past ng: a lane of the last batch with nothing to store, still in the NVFP4 block's shuffles
      const uint32_t vi = (g < ng ? slot_chunk(l, cr) : 0) * CV + q * 4;
      enc_store4<KIND>(xbs, sfs, g < ng ? cr : ChunkRange{cr.bw, 0, cr.kc0}, vi, t, v, gs, rgs, inv_s);
    }
  }
}

// a group of 4 normed values (vi .. vi + 3) of token row t into a published operand (PRO_ACT's layout: codes
// [K chunks][NT][128], NVFP4 scales [K chunks][NT][16] after them, nch chunks); the 4 lanes of an NVFP4 block together
// (every lane of the warp in the call: st false stores nothing)
template <int KIND>
__device__ __forceinline__ void enc_pub4(uint8_t* pub, uint32_t nch, uint32_t vi, uint32_t t, float (&v)[4], float gs,
                                         float rgs, float inv_s, bool stores) {
  if (KIND == KIND_NVFP4) {  // vLLM's quantization of bf16(v), as enc_store4
    bf16r2(v[0], v[1]);
    bf16r2(v[2], v[3]);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const __nv_fp8_e4m3 q(gs * (amax * rcp_approx_ftz(E2M1_MAX)));
    const float sfq = float(q);
    const float osc = sfq != 0.f ? rcp_approx_ftz(sfq * rgs) : 0.f;
    const uint32_t c = f32x2_to_e2m1x2(v[0] * osc, v[1] * osc) | (f32x2_to_e2m1x2(v[2] * osc, v[3] * osc) << 8);
    if (stores) {
      const uint32_t kc = vi / 256;
      *reinterpret_cast<uint16_t*>(pub + (size_t)kc * CHUNK_B + pub_byte(t, (vi % 256) / 2)) = (uint16_t)c;
      if ((vi & 15) == 0)
        pub[(size_t)nch * CHUNK_B + (size_t)kc * SFB_CHUNK + t * 16 + (vi % 256) / 16] =
            *reinterpret_cast<const uint8_t*>(&q);
    }
  } else if (KIND == KIND_FP8) {
    mul2(v[0], v[1], inv_s, inv_s), mul2(v[2], v[3], inv_s, inv_s);
    const uint32_t w = f32x2_to_e4m3x2(v[0], v[1]) | (f32x2_to_e4m3x2(v[2], v[3]) << 16);
    if (stores) *reinterpret_cast<uint32_t*>(pub + (size_t)(vi / 128) * CHUNK_B + pub_byte(t, vi % 128)) = w;
  } else {
    const __nv_bfloat162 h0 = __floats2bfloat162_rn(v[0], v[1]), h1 = __floats2bfloat162_rn(v[2], v[3]);
    if (stores)
      *reinterpret_cast<uint2*>(pub + (size_t)(vi / 64) * CHUNK_B + pub_byte(t, (vi % 64) * 2)) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
  }
}

// PRO_NORM_Q: rstd per token from the producer's exact sums of squares, the instruction's slice of x's [NT][K / 16]
// NVFP4 blocks normed (x * rs) * (1 + w) (DF_NW_PLAIN: * w; DF_PLAIN: no norm; DF_CONV: the grouped conv, side 0),
// encoded into q_pub and counted (qctr), then the wait for every slice (one instruction of the op per SM)
constexpr uint32_t QP_PUBLISH = 1u, QP_WAIT = 2u;  // the slice published, every slice waited for (OP_QPUB: the first)
template <int KIND, int V>
__device__ __forceinline__ void pro_norm_q(const MKArgs& a, const Instr& in, const OpDesc& d, const Smem& sm, int pt,
                                           uint32_t T, uint32_t epoch, uint32_t i,
                                           uint32_t mode = QP_PUBLISH | QP_WAIT) {
  const float gs = d.gs, eps = d.eps;
  const float inv_s = KIND == KIND_FP8 ? div_full(1.0f, gs) : 0.f;
  const float rgs = KIND == KIND_NVFP4 ? rcp_approx_ftz(gs) : 0.f;
  const uint32_t df = d.dflags;
  const bool is_plain = HAS_D<V> && (df & DF_PLAIN) != 0u, is_conv = HAS_D<V> && (df & DF_CONV) != 0u;  // the draft's
  const float w_one = (df & DF_NW_PLAIN) ? 0.0f : 1.0f;
  const uint32_t K = d.K, BPT = K / 16;  // NVFP4 blocks per token row
  const uint32_t nblk = NT * BPT, blocks_per_inst = (nblk + d.n_inst - 1) / d.n_inst;
  const uint32_t b0 = min(nblk, in.idx * blocks_per_inst), b1 = min(nblk, b0 + blocks_per_inst);
  const uint32_t ng = (b1 - b0) * 4;  // groups of 4 values
  constexpr uint32_t CV = KIND == KIND_NVFP4 ? 256 : KIND == KIND_FP8 ? 128 : 64;
  const uint32_t nch = K / CV;
  uint8_t* pub = reinterpret_cast<uint8_t*>(d.q_pub);
  const uint8_t* xin = reinterpret_cast<const uint8_t*>(d.in0);
  const __nv_bfloat16* nw = reinterpret_cast<const __nv_bfloat16*>(d.nw);
  const bool is_x_bf16 = d.x_bf16;
  // x's 4 values (vi ..) of token t: fp32, or bf16 (x_bf16, DF_PLAIN: segment vi / 5120 of aux_all)
  auto load4 = [&](uint32_t t, uint32_t vi) -> float4 {
    if (is_plain) {
      const uint32_t seg = vi / K_NORM, k = vi - seg * K_NORM;
      const uint2 u = __ldcg(reinterpret_cast<const uint2*>(xin + (((size_t)seg * NT + t) * K_NORM + k) * 2));
      return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
    }
    if (is_x_bf16) {
      const uint2 u = __ldcg(reinterpret_cast<const uint2*>(xin + ((size_t)t * K_NORM + vi) * 2));
      return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
    }
    return __ldcg(reinterpret_cast<const float4*>(xin + ((size_t)t * K_NORM + vi) * 4));
  };
  if (mode & QP_PUBLISH) {
    // the first group's x and norm weight loads issued before the sums of squares' (their round trips overlap)
    float4 x_first = make_float4(0.f, 0.f, 0.f, 0.f);
    uint2 w_first = make_uint2(0u, 0u);
    {
      const uint32_t blk = b0 + (pt < (int)ng ? pt / 4 : 0), t = blk / BPT, vi = (blk % BPT) * 16 + (pt % 4) * 4;
      if (pt < (int)ng && t < T) {
        x_first = load4(t, vi);
        if (!is_plain) w_first = __ldg(reinterpret_cast<const uint2*>(nw + vi));
      }
    }
    // the sums of squares: thread (j = pt / 8, t = pt % 8) the tiles j, j + 32, ..; a warp's 4 j's by shuffles; the
    // 8 warps' sums in red (u64 [8][NT])
    unsigned long long* part = reinterpret_cast<unsigned long long*>(sm.red);
    if (!is_plain) {
      const uint32_t t = pt & 7, j = pt >> 3;
      const unsigned long long* sp =
          reinterpret_cast<const unsigned long long*>(d.ssq_in) + (size_t)(epoch & 1) * d.ssq_tiles * NT;
      unsigned long long s0 = 0ull;
      for (uint32_t k = j; k < d.ssq_tiles; k += NPRO / NT) s0 += __ldcg(sp + (size_t)k * NT + t);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
      part[(pt >> 5) * NT + t] = s0;  // the 4 lanes of a token hold the same sum: an unpredicated store
    }
    named_sync(BAR_PRO, NPRO);
    if (pt == 0 && a.trace) a.trace[(size_t)i * TR_N + TR_ELAST] = gtimer();
    auto rstd = [&](uint32_t t) -> float {
      unsigned long long tot = 0ull;
#pragma unroll
      for (int k = 0; k < NPRO / 32; k++) tot += part[k * NT + t];
      return rms_rsqrt5120(__ull2float_rn(tot) * SSQ_INV, eps);
    };
    for (uint32_t g0 = 0; g0 < ng; g0 += NPRO) {  // a uniform trip count: the NVFP4 block's shuffles
      const uint32_t g = g0 + pt;
      const bool in_range = g < ng;
      const uint32_t blk = b0 + (in_range ? g / 4 : 0), t = blk / BPT, vi = (blk % BPT) * 16 + (g % 4) * 4;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      if (in_range && t < T) {
        const float4 x = g0 == 0 ? x_first : load4(t, vi);
        v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
        if (!is_plain) {
          const uint2 w = g0 == 0 ? w_first : __ldg(reinterpret_cast<const uint2*>(nw + vi));
          const float w1[4] = {bf16lo(w.x) + w_one, bf16hi(w.x) + w_one, bf16lo(w.y) + w_one, bf16hi(w.y) + w_one};
          const float rs = rstd(t);
#pragma unroll
          for (int k = 0; k < 4; k += 2) {
            mul2(v[k], v[k + 1], rs, rs);
            mul2(v[k], v[k + 1], w1[k], w1[k + 1]);
          }
          if (is_conv) {  // y = bf16(fma(c0, n_t, c1 n_(t-1))), n = bf16(norm); none for t = 0
            float p[4] = {0.f, 0.f, 0.f, 0.f};
            if (t > 0) {
              const float4 xp = load4(t - 1, vi);
              const float rp = rstd(t - 1);
              p[0] = xp.x, p[1] = xp.y, p[2] = xp.z, p[3] = xp.w;
#pragma unroll
              for (int k = 0; k < 4; k += 2) {
                mul2(p[k], p[k + 1], rp, rp);
                mul2(p[k], p[k + 1], w1[k], w1[k + 1]);
              }
              bf16r2(p[0], p[1]), bf16r2(p[2], p[3]);
            }
            bf16r2(v[0], v[1]), bf16r2(v[2], v[3]);
            const __nv_bfloat16* cf = reinterpret_cast<const __nv_bfloat16*>(d.coef) + (size_t)t * COEF_ROW;
            const uint32_t grp = vi / CONV_GROUP_CH;
            const float d0 = __bfloat162float(cf[grp]), d1 = __bfloat162float(cf[COEF_TAP + grp]);
            const __nv_bfloat16* bk = reinterpret_cast<const __nv_bfloat16*>(d.base);  // side 0: [tap][5120]
            const uint2 c0u = __ldg(reinterpret_cast<const uint2*>(bk + vi));
            const uint2 c1u = __ldg(reinterpret_cast<const uint2*>(bk + K_NORM + vi));
            const float b0v[4] = {bf16lo(c0u.x), bf16hi(c0u.x), bf16lo(c0u.y), bf16hi(c0u.y)};
            const float b1v[4] = {bf16lo(c1u.x), bf16hi(c1u.x), bf16lo(c1u.y), bf16hi(c1u.y)};
#pragma unroll
            for (int k = 0; k < 4; k++)
              v[k] = bf16_round(__fmaf_rn(__fadd_rn(b0v[k], d0), v[k], __fmul_rn(__fadd_rn(b1v[k], d1), p[k])));
          }
        }
      }
      enc_pub4<KIND>(pub, nch, vi, t, v, gs, rgs, inv_s, in_range);
    }
    named_sync(BAR_PRO, NPRO);
    if (pt == 0) {
      if (a.trace) a.trace[(size_t)i * TR_N + TR_ELAST + 1] = gtimer();
      red_release_add(a.counters + d.qctr, 1);
      if (a.trace) a.trace[(size_t)i * TR_N + TR_Q] = gtimer();
    }
  }
  if ((mode & QP_WAIT) && pt == 0) {
    wait_counter(a.counters + d.qctr, dep_target(a, (uint16_t)d.qctr, epoch));
    if (a.trace) a.trace[(size_t)i * TR_N + TR_Q + 1] = gtimer();
  }
  named_sync(BAR_PRO, NPRO);
}

// PRO_ACT's B by bulk copies (one thread): the instruction's chunks of a published operand (codes [K chunks][NT][128]
// in the B slot's swizzle, NVFP4 scales [K chunks][NT][16] after them) into B slot bslot, completing its xfull phases
__device__ __forceinline__ void pro_act_bulk(const Instr& in, const OpDesc& d, const Smem& sm, uint32_t bslot,
                                             const uint8_t* src) {
  const ChunkRange cr = chunk_range(in, d);
  const uint32_t nchunks = d.kb / STAGE_K;
  const bool has_scales = d.kind == KIND_NVFP4;
  // the operand's generic stores by other SMs, seen through the counters, read by the async proxy; the slot's last
  // generic reads before its async writes
  asm volatile("fence.proxy.async.global;" ::: "memory");
  fence_async_smem();
  uint8_t* xb = b_codes(sm, bslot);
  uint8_t* sf = b_scales(sm, bslot);
  // the groups in the MMA's order, each on its barrier (an empty one: an arrival only); rotated: the group of the
  // first stage's chunk first
  const uint32_t gf = min(rot_group(in), (uint32_t)BG - 1u);
  for (uint32_t j = 0; j < (uint32_t)BG; j++) {
    const uint32_t g = j == 0 ? gf : (j - 1 < gf ? j - 1 : j);
    uint64_t* bar = &sm.xfull[bslot * BG + g];
    const uint32_t l0 = g * B_GCH, l1 = min(cr.nck, l0 + B_GCH);
    if (l0 >= l1) {
      mbar_arrive(bar);
      continue;
    }
    mbar_expect_tx(bar, (l1 - l0) * (CHUNK_B + (has_scales ? SFB_CHUNK : 0u)));
    for (uint32_t l = l0; l < l1; l++) {
      const uint32_t kc = slot_chunk(l, cr);
      bulk_g2s(xb + l * CHUNK_B, src + (size_t)kc * CHUNK_B, CHUNK_B, bar);
      if (has_scales)
        bulk_g2s(sf + l * SFB_CHUNK, src + (size_t)nchunks * CHUNK_B + (size_t)kc * SFB_CHUNK, SFB_CHUNK, bar);
    }
  }
}

// OP_EMBED, item t's part p of n (Instr r0, r1, aux): row t of the table (bf16 [V][K], in host memory through UVA:
// memory_plan.py) into r_out [NT][K] and its integer sum of squares. DF_DRAFT_TOK: the draft's query block, the verify
// step's last sampled token, then the mask token, whose row the device holds (res_in)
template <int V>
__device__ __forceinline__ void op_embed(const OpDesc& d, uint32_t t, uint32_t p, uint32_t n, int pt, uint32_t T,
                                         uint32_t epoch) {
  const uint32_t rv = d.K / 8;  // 16-byte vectors per row
  const uint32_t pv = rv / max(n, 1u), e0 = p * pv;  // part p of n: its vectors e0 .. e0 + pv
  unsigned long long* ssq = reinterpret_cast<unsigned long long*>(d.ssq);
  if (ssq && pt == 0 && p == 0) ssq[((epoch + 1) & 1) * NT + t] = 0ull;  // the next launch's
  const bool is_draft = HAS_D<V> && (d.dflags & DF_DRAFT_TOK) != 0u;
  const uint4* src = nullptr;
  if (t < T) {
    if (is_draft && t > 0) {
      src = reinterpret_cast<const uint4*>(d.res_in);
    } else {
      const int64_t tok = is_draft ? draft_state().bonus : (int64_t)rec_token(nullptr, t);
      src = reinterpret_cast<const uint4*>(d.in0) + tok * rv;
    }
  }
  constexpr int EB = 3;  // a part's 16-byte vectors per thread, all in flight
  uint4 v[EB];
#pragma unroll
  for (int k = 0; k < EB; k++) {
    const uint32_t e = k * NPRO + pt;
    v[k] = src && e < pv ? __ldg(src + e0 + e) : make_uint4(0, 0, 0, 0);
  }
  uint4* out = reinterpret_cast<uint4*>(d.r_out) + (size_t)t * rv + e0;
  unsigned long long sq = 0ull;
#pragma unroll
  for (int k = 0; k < EB; k++) {
    const uint32_t e = k * NPRO + pt;
    if (e < pv) out[e] = v[k];
    const uint32_t u[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const float a = bf16lo(u[q]), b = bf16hi(u[q]);
      sq += (unsigned long long)__float2ll_rn(a * a * SSQ_SCALE) +
            (unsigned long long)__float2ll_rn(b * b * SSQ_SCALE);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  if (ssq && (pt & 31) == 0 && t < T && sq) atomicAdd(ssq + (epoch & 1) * NT + t, sq);
}

}  // namespace mk
#include "draft.cuh"
namespace mk {

// ------------------------------------------------------------------------------------------------ spec sampler
// The spec sampler's phase ph (0 .. 2: A .. C; A0, the draft rows' stats, first with a0) of an item on the prologue
// warps, its scratch in the staging buffer (SsArgs, then SsSmem) so a GEMV's B and segment slots stay usable beside it;
// not inlined: its code outside mk_main's
constexpr uint32_t SS_PH_NONE = 3u;  // ss_phase's ph past C: phase A0 alone
constexpr uint32_t SS_ARGS_BYTES = (sizeof(SsArgs) + 15) / 16 * 16;
static_assert(SS_ARGS_BYTES + sizeof(SsSmem<NPRO>) <= STG_BYTES, "the spec sampler's scratch in the staging buffer");
__device__ __noinline__ void ss_phase(const SsDesc& sd, uint32_t T, uint8_t* base, int pt, uint32_t ph, uint32_t item,
                                      bool a0, uint64_t* tr) {
  SsArgs& sa = *reinterpret_cast<SsArgs*>(base);
  SsSmem<NPRO>& ssh = *reinterpret_cast<SsSmem<NPRO>*>(base + SS_ARGS_BYTES);
  if (pt == 0) sa = ss_args(sd, (int)T);
  named_sync(BAR_PRO, NPRO);
  const LpGrp<NPRO> gp{pt, BAR_PRO};
  if (a0) {
    ss_phase_a0(gp, ssh, sa, (int)item);
    named_sync(BAR_PRO, NPRO);
  }
  if (ph == 0) ss_phase_a(gp, ssh, sa, (int)item, false, tr);
  else if (ph == 1) ss_phase_b(gp, ssh, sa, (int)item, (int)sd.slices, false, tr);
  else if (ph == 2) ss_phase_c(gp, ssh, sa, (int)sd.slices, false);
  named_sync(BAR_PRO, NPRO);  // the group's shared memory free for the next phase
}

// ------------------------------------------------------------------------------------------------ kernel
// The launch's start, run by every warpgroup after its setmaxnreg: the barriers (thread 0), the record's header into
// rec_sh, the epoch
__device__ __forceinline__ void mk_start(const MKArgs& a, const Smem& sm, uint32_t* epoch_sh) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; i++) mbar_init(&sm.full[i], 1), mbar_init(&sm.empty[i], NMMA);
    for (int i = 0; i < NB * BG; i++) mbar_init(&sm.xfull[i], 1);
    for (int i = 0; i < NB; i++) mbar_init(&sm.xempty[i], NMMA);
    for (int i = 0; i < NACC; i++) mbar_init(&sm.dfull[i], NMMA), mbar_init(&sm.dempty[i], EPI_WARPS);
    mbar_init(sm.stg_full, 1);
    mbar_init(sm.bis, 1);
    mbar_init(&sm.xeo[0], NMMA), mbar_init(&sm.xeo[1], NMMA);  // reserved
    sm.dyn_total[0] = 0u, sm.dyn_total[1] = 0u, sm.rel_cnt[0] = 0u, sm.stg_ph[0] = 0u, sm.q_rel[0] = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    *epoch_sh = ld_relaxed(a.epoch);
  }
  for (int w = threadIdx.x; w < SR_BT; w += NTHREADS) rec_sh[w] = a.rec.w[w];
  __syncthreads();
}

// the launch's words, read by each role (volatile loads: not one value live across every role's code)
#define MK_WORDS                                                                                                       \
  const uint32_t epoch = *reinterpret_cast<volatile uint32_t*>(epoch_sh);                                              \
  const uint32_t T = (uint32_t)rec_i32(nullptr, SR_NTOK);                                                              \
  const uint32_t i0 = ld_nc_u32(a.sm_off + blockIdx.x), i1 = ld_nc_u32(a.sm_off + blockIdx.x + 1);                     \
  (void)epoch, (void)T
#define MK_SMEM                                                                                                        \
  extern __shared__ __align__(16) uint8_t smem_raw[];                                                                  \
  const Smem sm = smem_layout(smem_raw);                                                                               \
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;                                                          \
  (void)warp, (void)lane

// the roles, each in the code its warpgroup's setmaxnreg dominates alone (ptxas budgets a region by the setmaxnreg
// that reaches it)
template <int V>
__device__ __forceinline__ void role_producer(const MKArgs& a, uint32_t* epoch_sh) {
  MK_SMEM;
  MK_WORDS;
  RingPos cc;
  constexpr uint32_t NO_PREV_DEP = 0xffffffffu;  // no GEMV on this SM yet
  uint32_t rel_prev_dep = NO_PREV_DEP, n_gemv = 0;
  for (uint32_t i = i0; i < i1; i++) {
    const Instr in = a.instrs[i];
    if (HAS_T<V> && in.op == OP_ATTN) {
      const AttnDesc& g = *reinterpret_cast<const AttnDesc*>(&a.descs[in.desc]);
      produce_attn<NSTAGE>(in, g, sm.stages, STAGE_BYTES, sm.full, sm.empty, cc, a.counters,
                           dep_target(a, (uint16_t)g.prep_ctr, epoch), lane);
      continue;
    }
    if (HAS_D<V> && in.op == OP_D_ATTN) {
      const DraftDesc& g = *reinterpret_cast<const DraftDesc*>(&a.descs[in.desc]);
      produce_dattn<NSTAGE>(in, g, sm.stages, STAGE_BYTES, sm.full, sm.empty, cc, a.counters,
                            dep_target(a, (uint16_t)g.prep_ctr, epoch), dep_target(a, (uint16_t)g.ctx_ctr, epoch),
                            lane);
      continue;
    }
    if (in.op != OP_GEMV) continue;
    // an instruction on another counter than the SM's previous GEMV waits until this SM's epilogue has released the
    // previous one: the release's MEMBAR waits for the SM's in-flight copies, so the next op's loads go after it
    if (in.dep != rel_prev_dep && n_gemv > 0) {
      while (*reinterpret_cast<volatile uint32_t*>(sm.rel_cnt) < n_gemv) {
      }
    }
    rel_prev_dep = in.dep;
    n_gemv++;
    produce_gemv(in, a.descs[in.desc], sm, cc, lane);
  }
}
template <int V>
__device__ __forceinline__ void role_mma(const MKArgs& a, uint32_t* epoch_sh) {
  MK_SMEM;
  MK_WORDS;
  const int w = warp - W_MMA0;
  RingPos cc, seg, bs;
  for (uint32_t i = i0; i < i1; i++) {
    const Instr in = a.instrs[i];
    if (HAS_T<V> && in.op == OP_ATTN) {  // its stages are the prologue warps'
      cc.advance(attn_stages(in, *reinterpret_cast<const AttnDesc*>(&a.descs[in.desc])), NSTAGE);
      continue;
    }
    if (HAS_D<V> && in.op == OP_D_ATTN) {
      cc.advance(dq_stages(in, *reinterpret_cast<const DraftDesc*>(&a.descs[in.desc])), NSTAGE);
      continue;
    }
    if (in.op != OP_GEMV) continue;
    const OpDesc& d = a.descs[in.desc];
    uint64_t* tr = a.trace ? a.trace + (size_t)i * TR_N : nullptr;
    if (d.kind == KIND_NVFP4)
      mma_gemv<KIND_NVFP4>(in, d, sm, cc, seg, bs.slot, bs.phase, w, lane, tr);
    else if (d.kind == KIND_FP8)
      mma_gemv<KIND_FP8>(in, d, sm, cc, seg, bs.slot, bs.phase, w, lane, tr);
    else
      mma_gemv<KIND_BF16>(in, d, sm, cc, seg, bs.slot, bs.phase, w, lane, tr);
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.xempty[bs.slot]);
    bs.phase ^= 1u;  // NB 1: one phase per instruction
  }
}
template <int V>
__device__ __forceinline__ void role_epilogue(const MKArgs& a, uint32_t* epoch_sh) {
  MK_SMEM;
  MK_WORDS;
  const int ept = threadIdx.x - W_EPI0 * 32;
  RingPos seg;
  for (uint32_t i = i0; i < i1; i++) {
    const Instr in = a.instrs[i];
    if (in.op != OP_GEMV) continue;
    const OpDesc& d = a.descs[in.desc];
    // the residual's early loads: it is an earlier op's output, complete once this instruction's dependency is,
    // which the epilogue waits for too (the previous op may be the dependency itself, still writing it)
    if (d.epi == EPI_RED && d.xr && d.res_in != 0 && in.dep != NO_CTR && !(HAS_D<V> && (d.dflags & DF_COLFIN))) {
      if (ept == 0) wait_dep(a, in.dep, epoch);
      named_sync(BAR_EPI, NEPI);
    }
    epilogue_gemv<V>(in, d, sm, seg, ept, epoch, T);
    named_sync(BAR_EPI, NEPI);
    if (ept == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    if (ept == 0) *reinterpret_cast<volatile uint32_t*>(sm.rel_cnt) += 1;  // the release's MEMBAR is done
    if (ept == 0 && a.trace) a.trace[(size_t)i * TR_N + TR_EPI] = gtimer();
  }
}
template <int V>
__device__ __forceinline__ void role_prologue(const MKArgs& a, uint32_t* epoch_sh) {
  MK_SMEM;
  MK_WORDS;
  const int pt = threadIdx.x - W_PRO0 * 32;
  RingPos bs, cc;
  uint32_t xempty_phase = 0;  // the B slot's consumed barrier's parity: one phase per GEMV instruction
  // the fused kernel: the loop's end reloaded, not a register held across every op's code
  for (uint32_t i = i0; i < (HAS_T<V> && HAS_D<V> ? ld_nc_u32(a.sm_off + blockIdx.x + 1) : i1); i++) {
    const Instr in = a.instrs[i];
    const OpDesc& d = a.descs[in.desc];
    if (pt == 0 && a.trace) a.trace[(size_t)i * TR_N + TR_START] = gtimer();
    if (in.op == OP_GEMV) cc.advance(in.r1 - in.r0, NSTAGE);  // the MMA warps' stages
    if (in.op == OP_GEMV) {
      const uint32_t pro = d.pro;
      mbar_wait(&sm.xempty[bs.slot], xempty_phase ^ 1);  // the B slot free before the prologue writes it
      if (pt == 0) {
        wait_dep(a, in.dep, epoch);
        if (a.trace) a.trace[(size_t)i * TR_N + TR_DEP] = gtimer();
      }
      named_sync(BAR_PRO, NPRO);
      if (pro == PRO_ACT) {
        // the bulk copies below
      } else if (pro == PRO_NORM_Q) {
        const uint32_t qm = (in.aux & AUX_QPUB_DONE) ? QP_WAIT : QP_PUBLISH | QP_WAIT;
        if (d.kind == KIND_NVFP4) pro_norm_q<KIND_NVFP4, V>(a, in, d, sm, pt, T, epoch, i, qm);
        else if (d.kind == KIND_FP8) pro_norm_q<KIND_FP8, V>(a, in, d, sm, pt, T, epoch, i, qm);
        else pro_norm_q<KIND_BF16, V>(a, in, d, sm, pt, T, epoch, i, qm);
      } else if (pro == PRO_NORM_X) {
        if (d.kind == KIND_NVFP4) pro_norm_x<KIND_NVFP4>(in, d, sm, pt, bs.slot, T, epoch);
        else if (d.kind == KIND_FP8) pro_norm_x<KIND_FP8>(in, d, sm, pt, bs.slot, T, epoch);
        else pro_norm_x<KIND_BF16>(in, d, sm, pt, bs.slot, T, epoch);
      } else if (pro == PRO_NORM_IN) {
        if (d.kind == KIND_NVFP4) pro_norm<KIND_NVFP4, false>(in, d, sm, pt, bs.slot, T);
        else if (d.kind == KIND_FP8) pro_norm<KIND_FP8, false>(in, d, sm, pt, bs.slot, T);
        else pro_norm<KIND_BF16, false>(in, d, sm, pt, bs.slot, T);
      } else {
        if (d.kind == KIND_NVFP4) pro_norm<KIND_NVFP4, true>(in, d, sm, pt, bs.slot, T);
        else if (d.kind == KIND_FP8) pro_norm<KIND_FP8, true>(in, d, sm, pt, bs.slot, T);
        else pro_norm<KIND_BF16, true>(in, d, sm, pt, bs.slot, T);
      }
      named_sync(BAR_PRO, NPRO);
      if (pt == 0) {
        if (a.trace) a.trace[(size_t)i * TR_N + TR_B] = gtimer();
        if (pro == PRO_ACT || pro == PRO_NORM_Q)  // a published operand: its chunks by bulk copies
          pro_act_bulk(in, d, sm, bs.slot, reinterpret_cast<const uint8_t*>(pro == PRO_ACT ? d.in0 : d.q_pub));
        else
          for (int g = 0; g < BG; g++) mbar_arrive(&sm.xfull[bs.slot * BG + g]);
        if (pro == PRO_NORM_RESID && (d.x_out || d.r_out) && in.done != NO_CTR)
          red_release_add(a.counters + in.done, 1);
      }
      bs.phase ^= 1u;  // NB 1: the groups' phases, one per instruction
      xempty_phase ^= 1u;
    } else if (in.op == OP_EMBED) {
      if (pt == 0) wait_dep(a, in.dep, epoch);
      named_sync(BAR_PRO, NPRO);
      op_embed<V>(d, in.r0, in.r1, in.aux, pt, T, epoch);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_T<V> && in.op == OP_GDNT) {  // the Gated DeltaNet core of one item (gdn_t.cuh); in.idx: the item
      const GdnTDesc& g = *reinterpret_cast<const GdnTDesc*>(&d);
      const GtDep dep{in.dep != NO_CTR ? a.counters + in.dep : nullptr,
                      in.dep != NO_CTR ? dep_target(a, in.dep, epoch) : 0u,
                      in.pr0 != NO_CTR ? a.counters + in.pr0 : nullptr,
                      in.pr0 != NO_CTR ? dep_target(a, (uint16_t)in.pr0, epoch) : 0u, 0u, nullptr,
                      in.done != NO_CTR ? a.counters + in.done : nullptr,
                      in.done != NO_CTR ? dep_target(a, in.done, epoch) : 0u, nullptr};
      GtCarry cy;
      uint32_t stg_phase = *reinterpret_cast<volatile uint32_t*>(sm.stg_ph);
      gdn_t_core_any(in.idx, g, sm.stg, sm.stg_full, stg_phase, epoch, pt, BAR_PRO, BAR_GDNT, dep, cy);
      if (pt == 0) *reinterpret_cast<volatile uint32_t*>(sm.stg_ph) = stg_phase;  // read again after the sync below
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
      gdn_t_finish(in.idx, g, sm.stg, pt, BAR_PRO, dep, cy);
    } else if (HAS_T<V> && (in.op == OP_ATTN_PREP || in.op == OP_ATTN_COMBINE)) {
      if (pt == 0) wait_dep(a, in.dep, epoch);
      named_sync(BAR_PRO, NPRO);
      const AttnDesc& g = *reinterpret_cast<const AttnDesc*>(&d);
      if (in.op == OP_ATTN_PREP) attn_prep(in, g, pt, NPRO);
      else attn_combine(in, g, pt, sm.stg, BAR_PRO);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_T<V> && in.op == OP_ATTN) {  // its tiles: the producer waits for OP_ATTN_PREP where needed
      const AttnDesc& g = *reinterpret_cast<const AttnDesc*>(&d);
      if (pt == 0) wait_dep(a, in.dep, epoch);  // Q from OP_ATTN_PREP
      named_sync(BAR_PRO, NPRO);
      attn_item<NSTAGE>(in, g, sm.stages, STAGE_BYTES, sm.full, sm.empty, cc, sm.stg, pt, BAR_PRO);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_D<V> && in.op == OP_D_ATTN) {
      const DraftDesc& g = *reinterpret_cast<const DraftDesc*>(&d);
      if (pt == 0) wait_dep(a, in.dep, epoch);  // Q from OP_D_PREP; its tiles wait in the producer
      named_sync(BAR_PRO, NPRO);
      dattn_item<NSTAGE>(in, g, sm.stages, STAGE_BYTES, sm.full, sm.empty, cc, sm.stg, pt, BAR_PRO);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_D<V> && in.op >= OP_D_CTX && in.op <= OP_D_SELECT) {  // the draft's other ops (draft.cuh)
      const DraftDesc& g = *reinterpret_cast<const DraftDesc*>(&d);
      if (pt == 0) {
        wait_dep(a, in.dep, epoch);
        if (in.pr0 != NO_CTR) wait_dep(a, (uint16_t)in.pr0, epoch);
      }
      named_sync(BAR_PRO, NPRO);
      if (in.op == OP_D_CTX) d_ctx(in.r0, g, pt);
      else if (in.op == OP_D_PREP) d_prep(in.r0, g, pt);
      else if (in.op == OP_D_COMBINE) d_combine(in.r0, g, pt, sm.stg, BAR_PRO);
      else if (in.op == OP_D_TOPK)
        d_topk(in.r0, g, pt, sm.stg, BAR_PRO, a.trace ? a.trace + (size_t)i * TR_N : nullptr);
      else d_select(in.r0, g, pt, sm.stg, BAR_PRO, a.trace ? a.trace + (size_t)i * TR_N : nullptr);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_T<V> && in.op >= OP_SS_A && in.op <= OP_SS_C) {  // the spec sampler's phases (in.idx: the item)
      const SsDesc& sd = *reinterpret_cast<const SsDesc*>(&d);
      const bool ss_enabled = rec_i32(nullptr, SR_SS + SS_TOP_P) != 0;  // 0: vLLM samples the step's logits
      uint64_t* const ss_tr = a.trace ? a.trace + (size_t)i * TR_N : nullptr;
      // SS_AUX_A0: A0 alone (the draft rows' stats, no dependency), right after the LM head on each SM so it runs
      // while the LM head streams; SS_AUX_NO_A0: the phase A after such an instruction
      if (in.op == OP_SS_A && (in.aux & SS_AUX_A0)) {
        if (ss_enabled) ss_phase(sd, T, sm.stg, pt, SS_PH_NONE, in.idx, true, ss_tr);
      } else {
        if (pt == 0 && in.dep != NO_CTR) wait_dep(a, in.dep, epoch);
        named_sync(BAR_PRO, NPRO);
        if (ss_enabled)
          ss_phase(sd, T, sm.stg, pt, in.op - OP_SS_A, in.idx, in.op == OP_SS_A && !(in.aux & SS_AUX_NO_A0), ss_tr);
      }
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_T<V> && in.op == OP_PREV) {  // no dependency: the previous launch's ring slot, as it left it
      const PrevDesc& pd = *reinterpret_cast<const PrevDesc*>(&d);
      if (rec_i32(nullptr, SR_BK + BK_PREV)) prev_outputs(pd, pt, NPRO, BAR_PRO);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_T<V> && in.op == OP_BOOK) {  // the embedding's dependency: it reads the input ids
      const BookDesc& bd = *reinterpret_cast<const BookDesc*>(&d);
      if (rec_i32(nullptr, SR_BK + BK_ON)) {
        book_inputs(bd, T, pt, NPRO, BAR_PRO, in.aux ? in.aux : BOOK_IN | BOOK_BT);
        __threadfence();  // the inputs for the launch's readers on other SMs, the draft's after them
      }
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    } else if (HAS_D<V> && in.op == OP_QPUB) {  // the fc's slice, NVFP4; its count is the op's qctr
      if (pt == 0) wait_dep(a, in.dep, epoch);
      named_sync(BAR_PRO, NPRO);
      pro_norm_q<KIND_NVFP4, V>(a, in, d, sm, pt, T, epoch, i, QP_PUBLISH);
    } else if (in.op == OP_NOP) {
      if (pt == 0) wait_dep(a, in.dep, epoch);
      named_sync(BAR_PRO, NPRO);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);
    }
    if (in.op != OP_GEMV && pt == 0 && a.trace) a.trace[(size_t)i * TR_N + TR_EPI] = gtimer();
  }
}

// a traced launch's CTA entry (k 0) and exit (k 1), in the trace's row past the instructions' of the CTA's SM index
// (not inlined: its registers outside the roles' budgets)
constexpr int CTA_ENTRY = 0;  // cta_stamp's slots
constexpr int CTA_EXIT = 1;
__device__ __noinline__ void cta_stamp(uint64_t* trace, const uint32_t* sm_off, int k) {
  trace[(size_t)(ld_nc_u32(sm_off + gridDim.x) + blockIdx.x) * TR_N + k] = gtimer();
}
// V: 0 the verify step (the target's ops), 1 the draft's proposal (draft.cuh), 2 both in one launch (the draft's
// instructions after the target's on every SM, their dependencies the target's counters)
template <int V>
__global__ void __launch_bounds__(NTHREADS, 1) mk_main(const __grid_constant__ MKArgs a) {
  __shared__ uint32_t epoch_sh;
  const int wg = threadIdx.x >> 7, warp = threadIdx.x >> 5;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  // a traced launch: each CTA's entry and exit (not the draft kernel's: its registers)
  if (HAS_T<V> && a.trace && threadIdx.x == 0) cta_stamp(a.trace, a.sm_off, CTA_ENTRY);
  // per warpgroup: its setmaxnreg first (every warp of it; the decreases first), then the start and its roles
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_EPI) : "memory");
    mk_start(a, smem_layout(smem_raw), &epoch_sh);
    role_epilogue<V>(a, &epoch_sh);
  } else if (wg == 1) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_MMA) : "memory");
    mk_start(a, smem_layout(smem_raw), &epoch_sh);
    if (warp == W_PROD) role_producer<V>(a, &epoch_sh);
    else if (warp < W_MMA0 + NMMA) role_mma<V>(a, &epoch_sh);
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(REG_PRO) : "memory");
    mk_start(a, smem_layout(smem_raw), &epoch_sh);
    role_prologue<V>(a, &epoch_sh);
  }
  // the launch's end: the last CTA advances the epoch (every CTA read it at its start)
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t epoch = *reinterpret_cast<volatile uint32_t*>(&epoch_sh);
    __threadfence();
    const uint32_t n = atomicAdd(a.epoch + 1, 1u) + 1;
    if (n == gridDim.x * (epoch + 1)) {
      __threadfence();
      atomicAdd(a.epoch, 1u);
    }
    if (HAS_T<V> && a.trace) cta_stamp(a.trace, a.sm_off, CTA_EXIT);
  }
}

}  // namespace mk

