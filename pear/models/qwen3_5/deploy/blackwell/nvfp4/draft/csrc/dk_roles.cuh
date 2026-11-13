// The draft proposal's device code as the roles run it per instruction: the workers' ops (B operands, the attention,
// PREP, QUANT, SELECT), warpgroup 0's base taps, the epilogue, the producer / MMA (dk_gemv.cuh); the standalone kernel
// (dk_kernel.cu) and a host kernel (the verify megakernel, csrc/megakernel.cu) run the draft through the same roles.
#pragma once
#include "dk_operand.cuh"
#include "../../../common/csrc/dk_attention.cuh"
#include "dk_select.cuh"

namespace dk {

static_assert(ATT_SMEM <= BBUF_BYTES, "the attention's scratch in the B buffer");
static_assert(sizeof(GemvDesc) == sizeof(AttnDesc), "descriptors share a table");
static_assert(SEL_SS_OFF + sizeof(SelectSmem) <= BBUF_BYTES && SEL_SS_OFF % 128 == 0, "the selector's scratch in the B buffer");
static_assert(fp4_image_bytes(HID) <= SEL_SS_OFF, "the LM head's operand below the selector's dry run");
// the selector's lists, then its rows, from the ring's start (the B buffer follows the ring in every host)
static_assert(SEL_MAX_PARTS * NSTEP * TOPK * 8 <= NSTAGE * STAGE_STRIDE + SEL_SS_OFF &&
              SEL_ROWS * SEL_ROW * 2 + NSTEP * SEL_ROW <= NSTAGE * STAGE_STRIDE + SEL_SS_OFF, "the selector's lists, its rows");
static_assert(fp4_image_bytes(HID) <= BBUF_BYTES, "the LM head's NVFP4 B operand and its scales");
static_assert(NSTEP * TOPK * 8 * (1 + WG_WARPS) + NSTEP * WG_WARPS * 4 <= XCHG_FLOATS * 4,
              "the LM head's running lists in the exchange buffer");

// the role's timestamps of instruction ii (null when off or past MAX_INSTR)
// (null by a mask: no select against zero; a null tstamp gives an address the mask clears)
__device__ __forceinline__ uint64_t* stamps(const DkArgs& a, int role, uint32_t ii, uint32_t i0) {
  const uint64_t p = reinterpret_cast<uint64_t>(a.tstamp) +
                     sizeof(uint64_t) * (((size_t)blockIdx.x * NROLE + role) * (MAX_INSTR + 1) + (ii - i0)) * NSUB;
  const uint64_t stamp_mask =
      lt_mask64(0u, (uint32_t)(a.tstamp != nullptr)) & lt_mask64(ii - i0, (uint32_t)MAX_INSTR + 1);
  return reinterpret_cast<uint64_t*>(p & stamp_mask);
}

// the role's progress word: the instruction it is on (relative, + 1) << 16 | its stage / step within it
__device__ __forceinline__ void mark(const DkArgs& a, int role, uint32_t ii, uint32_t i0, uint32_t sub = 0) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(a.progress + blockIdx.x * NROLE + role),
               "r"(((ii - i0 + 1) << PROG_SHIFT) | (sub & PROG_SUB_MASK))
               : "memory");
  if (uint64_t* ts = stamps(a, role, ii, i0)) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    ts[0] = t;
  }
}

__device__ __forceinline__ Trace trace(const DkArgs& a, int role, uint32_t ii, uint32_t i0) {
  return Trace{a.progress + blockIdx.x * NROLE + role, ii - i0 + 1, stamps(a, role, ii, i0)};
}

__device__ __forceinline__ uint32_t ctr_target(const DkArgs& a, uint16_t ctr, uint32_t epoch) {
  return (epoch + 1) * a.ctr_count[ctr];
}

// A role's copy of its op's descriptor in shared memory (the descriptors are global memory: read again after the
// role's own stores, which the compiler cannot tell apart from them, each read could miss the small L1)
__device__ __forceinline__ const GemvDesc& local_desc(const GemvDesc& g, GemvDesc& copy, int t, int nthreads, int bar) {
  constexpr int WORDS = sizeof(GemvDesc) / 16;
  named_sync(bar, nthreads);  // (the previous op's readers are done)
  if (t < WORDS) reinterpret_cast<uint4*>(&copy)[t] = __ldg(reinterpret_cast<const uint4*>(&g) + t);
  named_sync(bar, nthreads);
  return copy;
}

// ------------------------------------------------------------------------------------------------ workers
// OP_PREP: CTA 0 zeroes the sums-of-squares buffers' next-launch halves; every CTA computes the query block's ids and
// the 1 / rms of its embedding rows (layer 0's input norm) into its worker constants.
__device__ void op_prep(const Instr& in, const DkArgs& a, WorkerConst& wc, float* red, int pt, uint32_t epoch) {
  if (in.dep != NO_CTR) {  // (in a host step: its sampler's counter, the step's tokens)
    if (pt == 0) wait_counter(a.counters + in.dep, ctr_target(a, in.dep, epoch));
    named_sync(BAR_WRK, NWT);
  }
  if (in.tile == 0)
    for (uint32_t i = pt; i < a.n_ssq * QB; i += NWT) a.ssq_zero[(i / QB) * 2 * QB + ((epoch + 1) & 1) * QB + i % QB] = 0;
  const StepState s = step_state(a);
  if (pt < QB) wc.ids[pt] = query_id(a, s, pt);
  named_sync(BAR_WRK, NWT);
  // one warp per token: the fp32 sum of the bf16 row's squares, 1 / rms as the compiled norm computes it
  const int w = pt >> 5, lane = pt & 31;
  if (w < QB) {
    const uint4* row = reinterpret_cast<const uint4*>(a.tab.embed + (size_t)wc.ids[w] * HID);
    float ss = 0.f;
    for (int i = lane; i < HID / 8; i += 32) {
      const uint4 u = __ldg(row + i);
      const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ss = fmaf(bf16lo(uu[j]), bf16lo(uu[j]), ss);
        ss = fmaf(bf16hi(uu[j]), bf16hi(uu[j]), ss);
      }
    }
    ss = warp_sum(ss);
    if (lane == 0) wc.rstd_emb[w] = rsqrt_tri(div_full(ss, (float)HID) + NORM_EPS);
  }
  named_sync(BAR_WRK, NWT);
}

// the B rows' 1 / rms for this op (from the producing ops' sums of squares, parity of this launch: fixed point)
__device__ __forceinline__ void op_rstd(const GemvDesc& d, WorkerConst& wc, int pt, uint32_t epoch) {
  if (pt >= NB) return;
  const bool is_qkv_ctx = d.bsrc == B_QKV && pt >= QB;
  const long long* ssq = reinterpret_cast<const long long*>(is_qkv_ctx ? d.b[BT_CTX_SSQ] : d.b[BT_SSQ]);
  if (ssq == nullptr || (pt >= QB && !is_qkv_ctx)) return;
  const float s = ssq_val(__ldcg(ssq + (epoch & 1) * QB + (pt % QB)));
  // vLLM's _C rms_norm (the committed tokens' hidden_norm) divides with div.rn, the compiled draft with div.full
  wc.rstd[pt] =
      is_qkv_ctx ? rsqrt_tri(__fdiv_rn(s, (float)HID) + NORM_EPS) : rsqrt_tri(div_full(s, (float)HID) + NORM_EPS);
}

// The B operand's chunks in slot order (slot i: the chunk of stage r0 + i), worker warp w row w of every chunk, two
// chunks per step (a half warp each); a chunk's slot barrier completes once every warp's row is in. Compact loops: a
// build's time follows its code size (instruction fetch).
__device__ __forceinline__ uint32_t chunk_kc(const Instr& in, uint32_t nch, uint32_t i) {
  return stage_at(in, in.r0 + min(i, nch - 1)).kc;
}
// chunk_kc for a loop over the slots: one division at construction, a rotation per slot
struct ChunkSeq {
  uint32_t kb0, rot, kbw, last_slot;
  __device__ __forceinline__ ChunkSeq(const Instr& in, uint32_t nch)
      : kb0(in.kb0), rot(in.r0 % in.kbw), kbw(in.kbw), last_slot(nch - 1) {}
  __device__ __forceinline__ uint32_t at(uint32_t i) const {
    const uint32_t j = rot + min(i, last_slot);
    return kb0 + (j >= kbw ? j - kbw : j);
  }
};
// the warp's rows of chunks i0, i0 + 1 (below nch) are in: each ready for the MMA once every warp's is (no fence: a
// release flag would wait for the loads in flight)
__device__ __forceinline__ void rows_done(const GemvDesc& d, const Smem& sm, uint32_t i0, uint32_t nch, int lane) {
  fence_async_smem();
  __syncwarp();
  const uint32_t i = i0 + (lane >> 4);
  if ((lane & (HALF - 1)) == 0 && i < nch) mbar_arrive(&sm.bbar[d.b_slot0 + i]);
}

// The first phase with the kernel projection's normed rows (F_XN: bf16(RMSNorm(x)) of every chunk, b[0], written by
// its first tile's SMs, copied by the conv phase once the projection completes): B_QKV's context rows only.
template <int SRC>
__device__ __forceinline__ void copy_prep(const Instr& in, const GemvDesc& d, const DkArgs& a, const Smem& sm,
                                          WorkerConst& wc, int pt, uint32_t epoch) {
  const int wid = pt >> 5, lane = pt & 31;
  const uint32_t h = lane >> 4, b = lane & (HALF - 1), nch = b_slots(in);
  const ChunkSeq cs(in, nch);
  if constexpr (SRC == B_QKV) {
    const uint32_t ctx_col = sm.tmem + T_CTX + (wid / WG_WARPS) * (CTX_STEPS * CTX_WORDS);  // (the warp's half)
    if (d.layer == 0) {  // computed (hidden_norm of the fc output, quantized) and cached in TMEM
      RowIn c = load_ctx(d, a, cs.at(h), wid, b);
      named_sync(BAR_WRK, NWT);
      op_rstd(d, wc, pt, epoch);
      named_sync(BAR_WRK, NWT);
      const Fp4Q f(d.gs);
      const bool has_ctx = (uint32_t)wid < a.step.n_ctx;
      const float r_ctx = wc.rstd[QB + wid];
#pragma unroll 1
      for (uint32_t i0 = 0; i0 < nch; i0 += CPS) {
        const uint32_t i = i0 + h;
        const RowIn cn = load_ctx(d, a, cs.at(i + CPS), wid, b);
        float v[BLK_K];
        ctx_values(c, has_ctx, r_ctx, v);
        uint32_t w3[3];
        ChunkOut::quant16(v, f, w3);
        tc_st4_u32(ctx_col + (i0 / CPS) * CTX_WORDS, w3[0], w3[1], w3[2], 0u);
        if (i < nch) ChunkOut(d, sm, i).store(QB + wid, b, w3);
        c = cn;
      }
    } else {  // copied from the TMEM cache
#pragma unroll 1
      for (uint32_t i0 = 0; i0 < nch; i0 += CPS) {
        const uint32_t i = i0 + h;
        uint32_t cq[CTX_WORDS];
        tc_ld4_u32(ctx_col + (i0 / CPS) * CTX_WORDS, cq);
        if (i < nch) ChunkOut(d, sm, i).store(QB + wid, b, {cq[0], cq[1], cq[2]});
      }
    }
  }
}

// the first phase of a two-phase B operand (instruction `in`, the next GEMV), after the previous op's B is built: its
// input's norm into the staging, B_QKV's context rows into the tiles (the input and its sums of squares are complete:
// the previous op waited for them; aux_seen: the aux counter this SM saw complete)
template <int SRC>
__device__ __forceinline__ void prep_operand(const Instr& in, const GemvDesc& d_smem, const DkArgs& a, const Smem& sm,
                                             WorkerConst& wc, int pt, uint32_t epoch, uint16_t& aux_seen) {
  const GemvDesc dr = d_smem;  // (registers: the shared copy's fields would be reloaded after every memory clobber)
  const GemvDesc& d = dr;
  const int wid = pt >> 5, lane = pt & 31;
  const uint32_t h = lane >> 4, b = lane & (HALF - 1), nch = b_slots(in);
  const ChunkSeq cs(in, nch);
  if (in.aux && in.aux != aux_seen) {  // (every worker's loads of its outputs after thread 0's acquire: the barrier)
    if (pt == 0) wait_counter(a.counters + in.aux - 1, ctr_target(a, (uint16_t)(in.aux - 1), epoch));
    named_sync(BAR_WRK, NWT);
    aux_seen = in.aux;
  }
  if (in.flags & F_XN) return copy_prep<SRC>(in, d, a, sm, wc, pt, epoch);
  // B_QKV's context rows: computed by layer 0 (and cached in TMEM, CTX_WORDS columns per step), copied by the others
  const bool is_ctx_new = SRC == B_QKV && d.layer == 0;
  RowIn q = load_prep(d, a, wc, cs.at(h), wid, b), c = q;
  if (is_ctx_new) c = load_ctx(d, a, cs.at(h), wid, b);
  named_sync(BAR_WRK, NWT);
  op_rstd(d, wc, pt, epoch);
  named_sync(BAR_WRK, NWT);
  const Fp4Q f(d.gs);
  const bool emb = d.xemb, has_ctx = (uint32_t)wid < a.step.n_ctx;
  const float r = row_rstd(emb, wc, wid), r_ctx = wc.rstd[QB + wid];
  const uint32_t ctx_col = sm.tmem + T_CTX + (wid / WG_WARPS) * (CTX_STEPS * CTX_WORDS);  // (this warp's half)
#pragma unroll 1
  for (uint32_t i0 = 0; i0 < nch; i0 += CPS) {
    const uint32_t i = i0 + h;
    const RowIn qn = load_prep(d, a, wc, cs.at(i + CPS), wid, b);
    RowIn cn = qn;
    if (is_ctx_new) cn = load_ctx(d, a, cs.at(i + CPS), wid, b);
    float v[BLK_K];
    norm16(q, emb, r, v);
    if (i < nch) {
      uint8_t* stg = sm.bbuf + d.stg_off + (i * QB + wid) * STG_ROW + 32 * b;
      sts128(stg, pack_bf16x8(v)), sts128(stg + 16, pack_bf16x8(v + 8));
    }
    if constexpr (SRC == B_QKV) {
      uint32_t cq[CTX_WORDS] = {0u, 0u, 0u, 0u};
      const uint32_t col = ctx_col + (i0 / CPS) * CTX_WORDS;
      if (is_ctx_new) {
        ctx_values(c, has_ctx, r_ctx, v);
        uint32_t w3[3];
        ChunkOut::quant16(v, f, w3);
        cq[0] = w3[0], cq[1] = w3[1], cq[2] = w3[2];
        tc_st4_u32(col, cq[0], cq[1], cq[2], 0u);
      } else {
        tc_ld4_u32(col, cq);
      }
      if (i < nch) ChunkOut(d, sm, i).store(QB + wid, b, {cq[0], cq[1], cq[2]});
    }
    q = qn;
    c = cn;
  }
}
// the second phase once the op's coefficients exist: the chunks' coefficients staged at once, then the conv over the
// staged rows, quantized into the tiles; the base taps from TMEM region conv_n % 2 (conv_n: the SM's two-phase GEMVs
// so far), handed back to warpgroup 0 once read
__device__ __forceinline__ void finish_operand(const Instr& in, const GemvDesc& d_smem, const Smem& sm, int pt, uint32_t conv_n,
                                               uint32_t epoch, const Trace& tr) {
  const GemvDesc dr = d_smem;  // (registers: the shared copy's fields would be reloaded after every memory clobber)
  const GemvDesc& d = dr;
  const int wid = pt >> 5, lane = pt & 31;
  const uint32_t h = lane >> 4, b = lane & (HALF - 1), nch = b_slots(in);
  const ChunkSeq cs(in, nch);
  const float* coef = reinterpret_cast<const float*>(d.b[BT_COEF]) + (epoch & 1) * COEF_HALF;  // (the launch's sums)
  uint8_t* coefs = sm.bbuf + ist_off(d);
  // the chunks' coefficients rounded to bf16 into the staging (16-byte piece q is half a group's record: a tap's 8
  // tokens, 32 bytes of sums; PT_PIECES pieces per thread, their loads in flight together, ahead of the rows' copies)
  constexpr uint32_t PIECES = CONV_REC / 16;
  constexpr int PT_PIECES = 2;  // a thread's pieces per round
  const uint32_t npc = nch * PIECES;
  float4 c[PT_PIECES][2];
  auto load = [&](uint32_t q0) {
#pragma unroll
    for (int u = 0; u < PT_PIECES; u++) {
      const uint32_t q = q0 + u * NWT, j = q % PIECES, g = cs.at(q / PIECES) * HALF + j / 2;
      const float4* src = reinterpret_cast<const float4*>(coef + coef_row(0, j & 1, g) * QB);
      if (q < npc) c[u][0] = __ldcg(src), c[u][1] = __ldcg(src + 1);
    }
  };
  auto store = [&](uint32_t q0) {
#pragma unroll
    for (int u = 0; u < PT_PIECES; u++) {
      const uint32_t q = q0 + u * NWT;
      if (q < npc)
        sts128(coefs + q * 16, make_uint4(pack_bf16x2(c[u][0].x, c[u][0].y), pack_bf16x2(c[u][0].z, c[u][0].w),
                                          pack_bf16x2(c[u][1].x, c[u][1].y), pack_bf16x2(c[u][1].z, c[u][1].w)));
    }
  };
  load(pt);
  if (in.flags & F_XN) {  // the projection's normed rows (its counter, the dependency, covers them): this warp's row
    const __nv_bfloat16* xn = reinterpret_cast<const __nv_bfloat16*>(d.b[BT_X]);
#pragma unroll 1
    for (uint32_t i = 0; i < nch; i++)
      cp_async16(sm.bbuf + d.stg_off + (i * QB + wid) * STG_ROW + 16 * lane,
                 xn + (size_t)wid * HID + cs.at(i) * CHUNK_K + 8 * lane);
    cp_async_commit();
  }
  store(pt);
  // (more than 16 chunks only)
  for (uint32_t q0 = pt + PT_PIECES * NWT; q0 < npc; q0 += PT_PIECES * NWT) load(q0), store(q0);
  const uint32_t r = conv_n & 1;
  if (pt == 0) mbar_wait(&sm.base_full[r], (conv_n >> 1) & 1);
  cp_async_wait<0>();
  named_sync(BAR_WRK, NWT);
  tc_fence_after();
  const uint32_t tbase = sm.tmem + T_BASE + r * BASE_REGION, last_step = (nch - 1) / CPS;  // (the last step's taps)
  const Fp4Q f(d.gs);
  ConvIn conv_in = load_conv(coefs + min(h, nch - 1) * CONV_REC, tbase, wid, b);
#pragma unroll 1
  for (uint32_t i0 = 0; i0 < nch; i0 += CPS) {
    const uint32_t i = i0 + h;
    const ConvIn next_conv_in = load_conv(coefs + min(i + CPS, nch - 1) * CONV_REC,
                                          tbase + min((i0 + CPS) / CPS, last_step) * BASE_WORDS, wid, b);
    if (i < nch) {
      float v[BLK_K];
      conv_values(conv_in, sm.bbuf + d.stg_off + i * (QB * STG_ROW), wid, b, v);
      ChunkOut(d, sm, i).put(wid, b, v, f);
    }
    rows_done(d, sm, i0, nch, lane);
    conv_in = next_conv_in;
  }
  tc_fence_before();
  __syncwarp();
  if (lane == 0) mbar_arrive(&sm.base_empty[r]);
}
// the workers' timestamps of a GEMV instruction (Trace::ts; 0: its start)
constexpr int TS_WRK_DEPS = 1;    // its dependencies met
constexpr int TS_WRK_BUILD = 2;   // its B operand built
constexpr int TS_WRK_COPIES = 3;  // a one-phase build: its first inputs' copies issued
constexpr int TS_WRK_HEAD = 4;    // its first chunks built
constexpr int TS_WRK_TAIL = 5;    // its last chunks built
constexpr int TS_WRK_PREP = 6;    // the next GEMV's first phase done

// a one-phase B operand: the inputs staged per batch of chunks (ist_n of them; the first batch's copies go out before
// the norms' scales), each chunk's row quantized from the staging
template <int SRC>
__device__ __forceinline__ void build_operand(const Instr& in, const GemvDesc& d_smem, const DkArgs& a, const Smem& sm,
                                              WorkerConst& wc, int pt, uint32_t epoch, const Trace& tr) {
  // the descriptor in registers (the fields used): through the shared copy, every field would be reloaded after each
  // copy's or store's memory clobber; the chunks by one rotation instead of a division each
  const GemvDesc dr = d_smem;
  const GemvDesc& d = dr;
  const ChunkSeq cs(in, min((uint32_t)in.kbw, in.r1 - in.r0));
  const int wid = pt >> 5, lane = pt & 31;
  const uint32_t h = lane >> 4, b = lane & (HALF - 1);
  const uint32_t nch = min((uint32_t)in.kbw, in.r1 - in.r0), nb = ist_n(d);
  uint8_t* stage = sm.bbuf + ist_off(d) + wid * in_row<SRC>();  // this warp's row of the batch's first chunk
  constexpr uint32_t CH = QB * in_row<SRC>();                    // a chunk's rows
#pragma unroll 1
  for (uint32_t i = 0; i < min(nb, nch); i++)
    stage_row<SRC>(d, a, wc, cs.at(i), wid, lane, stage + i * CH);
  cp_async_commit();
  op_rstd(d, wc, pt, epoch);
  named_sync(BAR_WRK, NWT);
  if (pt == 0) tr.stamp(TS_WRK_COPIES);
  const Fp4Q f(d.gs);
  const bool emb = d.xemb;
  const int tok = row_token<SRC>(a, wid);
  const float r = (SRC == B_NORM || SRC == B_FINAL) && tok >= 0 ? row_rstd(emb, wc, tok) : 0.f;
  __nv_bfloat16* xn_out =
      SRC == B_NORM && (in.flags & F_XN) ? reinterpret_cast<__nv_bfloat16*>(d.b[BT_XN_OUT]) : nullptr;
#pragma unroll 1
  for (uint32_t b0 = 0; b0 < nch; b0 += nb) {
    const uint32_t b1 = min(b0 + nb, nch);
    if (b0) {  // (the warp's own rows: it has read the previous batch)
#pragma unroll 1
      for (uint32_t i = b0; i < b1; i++)
        stage_row<SRC>(d, a, wc, cs.at(i), wid, lane, stage + (i - b0) * CH);
      cp_async_commit();
    }
    cp_async_wait<0>();
    __syncwarp();  // (a half warp reads what the whole warp copied)
#pragma unroll 1
    for (uint32_t i0 = b0; i0 < b1; i0 += CPS) {
      if (pt == 0 && (i0 == 0 || i0 + CPS >= nch)) tr.stamp(i0 == 0 ? TS_WRK_HEAD : TS_WRK_TAIL);
      const uint32_t i = i0 + h;
      if (i < b1) {
        float v[BLK_K];
        row_values<SRC>(a, staged_row<SRC>(stage + (i - b0) * CH, b, emb), wid, emb, r, v);
        ChunkOut(d, sm, i).put(wid, b, v, f);
        if (SRC == B_NORM && xn_out) {  // (the next two-phase GEMV's first phase copies these)
          uint4* o = reinterpret_cast<uint4*>(xn_out + (size_t)wid * HID + cs.at(i) * CHUNK_K + BLK_K * b);
          __stcg(o, pack_bf16x8(v)), __stcg(o + 1, pack_bf16x8(v + 8));
        }
      }
      rows_done(d, sm, i0, b1, lane);
    }
  }
  if (xn_out) {  // (the normed rows written: counted on the op's counter, which the conv phase copying them waits for)
    named_sync(BAR_WRK, NWT);
    if (pt == 0) red_release_add(a.counters + in.xn - 1, 1);
  }
}

// a GEMV instruction's B operand: the distinct K chunks of its stage range (the band's, or fewer), built once into
// the B buffer after the previous op's MMAs are done with it; B_FP4 the whole operand by all workers. seq: the
// instruction's index + 1 (published in Smem::wrk_gemv)
__device__ void worker_gemv(const Instr& in, const GemvDesc& d, const DkArgs& a, const Smem& sm, WorkerConst& wc, int pt,
                            uint32_t epoch, RingPos& cc, uint32_t seq, bool is_prepared, uint16_t aux_seen,
                            uint32_t& conv_n, const Trace& tr) {
  if (pt == 0) {
    if (in.dep != NO_CTR) wait_counter(a.counters + in.dep, ctr_target(a, in.dep, epoch));
    if (in.aux && in.aux != aux_seen) wait_counter(a.counters + in.aux - 1, ctr_target(a, (uint16_t)(in.aux - 1), epoch));
    tr.stamp(TS_WRK_DEPS);
    if (!is_prepared) wait_ring_drained(sm, cc);  // (prepared tiles keep clear of the previous op's)
    *sm.wrk_gemv = seq;
  }
  named_sync(BAR_WRK, NWT);
  if (is_prepared) {
    finish_operand(in, d, sm, pt, conv_n++, epoch, tr);
    if (pt == 0) tr.stamp(TS_WRK_BUILD);
    cc.advance(in.r1 - in.r0, NSTAGE);
    return;
  }
  switch (d.bsrc) {
    case B_AUX: build_operand<B_AUX>(in, d, a, sm, wc, pt, epoch, tr); break;
    case B_COPY: build_operand<B_COPY>(in, d, a, sm, wc, pt, epoch, tr); break;
    case B_NORM: build_operand<B_NORM>(in, d, a, sm, wc, pt, epoch, tr); break;
    case B_FINAL: build_operand<B_FINAL>(in, d, a, sm, wc, pt, epoch, tr); break;
    default: {  // B_FP4: OP_QUANT's image (its counter is the aux dependency) copied whole
      const uint8_t* img = reinterpret_cast<const uint8_t*>(d.e[ET_IMAGE]);
      for (uint32_t i = pt; i < fp4_image_bytes(d.K) / 16; i += NWT) cp_async16(sm.bbuf + i * 16, img + i * 16);
      cp_async_commit();
      cp_async_wait<0>();
      fence_async_smem();
      named_sync(BAR_WRK, NWT);
      if (pt < (int)b_slots(in)) mbar_arrive_n(&sm.bbar[pt], NWRK);  // (a slot's barrier counts every worker warp)
      break;
    }
  }
  if (pt == 0) tr.stamp(TS_WRK_BUILD);
  cc.advance(in.r1 - in.r0, NSTAGE);
}

// OP_QUANT: the SM's share [r0, r1) of the LM head's NVFP4 B operand's (row, block) pairs into its image (GemvDesc of
// the LM head's input: b[] as B_FP4's, e[1] the image), after the final layer's output
__device__ void op_quant(const Instr& in, const GemvDesc& d, const DkArgs& a, WorkerConst& wc, int pt, uint32_t epoch) {
  if (pt == 0 && in.dep != NO_CTR) wait_counter(a.counters + in.dep, ctr_target(a, in.dep, epoch));
  named_sync(BAR_WRK, NWT);
  op_rstd(d, wc, pt, epoch);
  named_sync(BAR_WRK, NWT);
  for (uint32_t i = in.r0 + pt; i < in.r1; i += NWT)
    quant_block(d, a, wc, i, reinterpret_cast<uint8_t*>(d.e[ET_IMAGE]));
}

// ------------------------------------------------------------------------------------------------ epilogue
// the epilogue's timestamps of a GEMV instruction (Trace::ts: at its last segment; 0: its start)
constexpr int TS_EPI_ACC = 1;      // the accumulator ready
constexpr int TS_EPI_PART = 2;     // the split-K part added (not the tile's finisher)
constexpr int TS_EPI_SUMS = 3;     // the tile's sums complete (its finisher), or qkv's part added
constexpr int TS_EPI_FINISH = 4;   // the finisher done
constexpr int TS_EPI_RELEASE = 5;  // the instruction's count released (EpiRole)
constexpr int TS_EPI_NSEG = 6;     // (a count: the instruction's segments)
template <int N>
__device__ void epilogue_gemv(const Instr& in, const GemvDesc& d_smem, const DkArgs& a, const Smem& sm, RingPos& seg, int ept,
                              uint32_t epoch, const Trace& tr) {
  const GemvDesc dr = d_smem;  // (registers: the shared copy's fields would be reloaded after every memory clobber)
  const GemvDesc& d = dr;
  const int lane = ept & 31;
  // E_LOGITS: the SM's running top-16 lists in the exchange buffer, then its candidate lists of the tile's warps
  uint64_t* run = reinterpret_cast<uint64_t*>(sm.xchg);
  uint64_t* cand = run + NSTEP * TOPK;
  uint32_t* ncand = reinterpret_cast<uint32_t*>(cand + NSTEP * WG_WARPS * TOPK);  // [NSTEP][WG_WARPS]
  if (d.epi == E_LOGITS) {
    for (int i = ept; i < NSTEP * TOPK; i += NEPI) run[i] = KEY_NONE;
    named_sync(BAR_EPI, NEPI);
  }
  if (d.epi == E_RESID) {  // the finisher's inputs are loaded ahead: the op's dependencies first
    if (ept == 0) {
      if (in.dep != NO_CTR) wait_counter(a.counters + in.dep, ctr_target(a, in.dep, epoch));
      if (in.aux) wait_counter(a.counters + in.aux - 1, ctr_target(a, (uint16_t)(in.aux - 1), epoch));
    }
    named_sync(BAR_EPI, NEPI);
  }
  const StageOrder ord(in);
  uint32_t nseg = 0;
  for (uint32_t k = 0; k < ord.len();) {
    const uint32_t n = ord.seg(k, in.kbw), mt = ord.at(k) / in.kbw;
    const bool is_last_seg = k + n == ord.len() && ept == 0;  // profiling: the last segment's points
    if (ept == 0) tr.at(k);
    const FinIn fin_in = load_fin(d, mt, ept, sm, a, epoch);
    const PartSlot ps = part_slot(d, in, mt);
    mbar_wait(&sm.dfull[seg.slot], seg.phase);
    if (is_last_seg) tr.stamp(TS_EPI_ACC);
    tc_fence_after();
    float v[N];
    load_acc<N>(sm, seg.slot, v);
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.dempty[seg.slot]);
    if (d.kind == KIND_NVFP4 && d.epi != E_LOGITS) {  // (the LM head's logits apply alpha to the tile's sums)
#pragma unroll
      for (int i = 0; i < N; i++) v[i] *= d.alpha;
    }
    // (no reduction round for qkv: the attention items sum its slots)
    const bool is_qkv_part = N == NB && d.epi == E_QKV;
    if constexpr (N == NB) {
      if (is_qkv_part) qkv_part(d, mt, ept, v, epoch, in.r0 <= mt * in.kbw, ps.slot);
    }
    const bool is_finisher = !is_qkv_part && reduce_part<N>(d, ps, mt, ept, v, sm);
    if (is_last_seg) tr.stamp(is_finisher || is_qkv_part ? TS_EPI_SUMS : TS_EPI_PART);
    if (is_finisher) {
      if constexpr (N == QB) {
        switch (d.epi) {
          case E_FC: finish_fc<N>(d, mt, ept, v, sm, epoch); break;
          case E_RESID: finish_resid(d, mt, ept, v, sm, fin_in, epoch); break;
          case E_SILU: finish_silu(d, mt, ept, v, sm); break;
          case E_LOGITS: finish_logits(d, mt, ept, v, sm, run, cand, ncand, a.tab.topk_chunk, a.tab.nstep); break;
          case E_COEF: finish_coef(d, mt, ept, v, epoch); break;
          default: finish_bf16<N>(d, mt, ept, v); break;
        }
      }
    }
    if (is_last_seg) tr.stamp(TS_EPI_FINISH);
    k += n;
    seg.next(N_ACC);
    nseg++;
  }
  if (ept == 0) tr.put(TS_EPI_NSEG, nseg);
  if (d.epi == E_LOGITS) {  // this SM's lists: part [instruction][NSTEP][TOPK]
    uint64_t* part = reinterpret_cast<uint64_t*>(d.e[ET_PARTS]) + (size_t)in.tile * NSTEP * TOPK;
    for (int i = ept; i < NSTEP * TOPK; i += NEPI) __stcg(part + i, run[i]);
  }
}

// The draft's own shared state (one per CTA). A host kernel declares one, binds the Smem view to it and to its ring
// (stages, full / empty), its accumulator hand-off (dfull / dempty), its producer-gate word, the draft's B area and
// TMEM, and has one thread init it before the roles start (the mbarrier inits fenced by the caller).
struct DraftShared {
  static constexpr int ERED_ROWS = 5;  // the epilogue's reduction rows: the warps' sums, then per-token results
  static constexpr int BASE_BARS = 4;  // base_full[N_BASE], then base_empty[N_BASE]
  __align__(16) float xchg[XCHG_FLOATS];       // the epilogue's exchange buffer
  __align__(16) float ered[ERED_ROWS * NB];    // the epilogue's reductions
  __align__(16) GemvDesc wdesc, edesc, pdesc;  // the workers' and the epilogue's GEMV, the prepared one
  StepSmem step;
  WorkerConst wc;
  AttnSmem att;
  uint64_t bbar[BBUF_CHUNKS];  // a B chunk built (every worker warp's rows)
  uint64_t base_bar[BASE_BARS];
  uint32_t eflag;
  volatile uint32_t prod_n, wrk_gemv;
  __device__ __forceinline__ void bind(Smem& sm) {
    sm.bbar = bbar;
    sm.base_full = base_bar;
    sm.base_empty = base_bar + N_BASE;
    sm.wrk_gemv = &wrk_gemv;
    sm.step = &step;
    sm.wc = &wc;
    sm.att = &att;
    sm.wdesc = &wdesc;
    sm.edesc = &edesc;
    sm.pdesc = &pdesc;
    sm.xchg = xchg;
    sm.ered = ered;
    sm.eflag = &eflag;
    sm.prod_n = &prod_n;
  }
  __device__ __forceinline__ void init(int t = 0, int nt = 1) {  // (thread t of nt, each one's inits fenced by it)
    for (int j = t; j < BBUF_CHUNKS; j += nt) mbar_init(&bbar[j], NWRK);
    for (int b = t; b < N_BASE; b += nt) {
      mbar_init(&base_bar[b], WG_WARPS);
      mbar_init(&base_bar[N_BASE + b], NWRK);
    }
    if (t == 0) wrk_gemv = 0, prod_n = 0;
  }
};

// ------------------------------------------------------------------------------------------------ roles
// One instruction of the SM's list per call (ii; i0 the list's first, i1 its end), by the standalone kernel's role
// loops or the verify megakernel's (the draft's op codes). A role's state across instructions is its struct's; the ring
// and accumulator positions (cc, seg) are the caller's, so a host kernel's roles continue theirs.

// warpgroup 0 (warp q: TMEM lane quadrant q): each two-phase GEMV's base conv taps (side 0) of this SM's chunks into
// TMEM region n % 2 (n: the SM's two-phase GEMVs so far), once the conv phase two such ops back has read it (read from
// global memory, every step of the conv phase would wait for an L2 round trip).
struct BaseRole {
  uint32_t n = 0;
  __device__ __forceinline__ void instr(const Instr& in, const DkArgs& a, const Smem& sm, int lane) {
    if (in.op != OP_GEMV || !is_two_phase(a.descs[in.desc].bsrc)) return;
    const uint32_t h = lane >> 4, b = lane & (HALF - 1);
    const uint32_t r = n & 1, nch = b_slots(in);
    if (n >= N_BASE) mbar_wait(&sm.base_empty[r], ((n >> 1) & 1) ^ 1);
    tc_fence_after();
    const __nv_bfloat16* base = reinterpret_cast<const __nv_bfloat16*>(a.descs[in.desc].b[BT_BASE]);
    const uint32_t col = sm.tmem + T_BASE + r * BASE_REGION;
#pragma unroll 1
    for (uint32_t i = 0; i < nch; i += CPS) store_base(base, chunk_kc(in, nch, i + h), b, col + (i / CPS) * BASE_WORDS);
    tc_wait_st();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.base_full[r]);
    n++;
  }
};

// warp W_PROD: each GEMV's weight tiles into the ring. The gate: a release waits for every TMA load in flight on the
// SM, so a GEMV's weights wait for this SM's release of its previous GEMV (its dependents cannot start before that
// release anyway).
struct ProdRole {
  uint32_t last_gemv = 0;  // 1 + this SM's previous GEMV instruction (relative to i0); 0: none yet
  __device__ __forceinline__ void instr(const Instr& in, uint32_t ii, uint32_t i0, const DkArgs& a, const Smem& sm,
                                        RingPos& cc, int lane) {
    if (lane == 0) mark(a, R_PROD, ii, i0);
    if (in.op != OP_GEMV) return;
    if (last_gemv && !(in.flags & F_NOGATE))
      while (*sm.epi_rel < last_gemv) __nanosleep(PACE_NS);
    last_gemv = ii - i0 + 1;
    produce_gemv(in, a.descs[in.desc], sm, cc, trace(a, R_PROD, ii, i0));
  }
};

// warp W_MMA: each GEMV's MMAs (bs: the B slots' phases)
struct MmaRole {
  BSlots bs;
  __device__ __forceinline__ void instr(const Instr& in, uint32_t ii, uint32_t i0, const DkArgs& a, const Smem& sm,
                                        RingPos& cc, RingPos& seg, int lane) {
    if (lane == 0) mark(a, R_MMA, ii, i0);
    if (in.op != OP_GEMV) return;
    const GemvDesc& d = a.descs[in.desc];
    if (d.kind == KIND_NVFP4) mma_gemv<KIND_NVFP4>(in, d, sm, cc, seg, bs, trace(a, R_MMA, ii, i0));
    else mma_gemv<KIND_BF16>(in, d, sm, cc, seg, bs, trace(a, R_MMA, ii, i0));
  }
};

// warpgroup 1: each GEMV's epilogue, then its release (which opens the producer's gate); begin: the step's tables,
// after counter dep in a host step (its sampler's: the step's tokens)
struct EpiRole {
  __device__ __forceinline__ void begin(const DkArgs& a, const Smem& sm, int ept, uint32_t epoch, uint16_t dep = NO_CTR) {
    if (dep != NO_CTR && ept == 0) wait_counter(a.counters + dep, ctr_target(a, dep, epoch));
    fill_step(a, *sm.step, ept);
  }
  __device__ __forceinline__ void instr(const Instr& in, uint32_t ii, uint32_t i0, const DkArgs& a, const Smem& sm,
                                        RingPos& seg, int ept, uint32_t epoch) {
    if (ept == 0) mark(a, R_EPI, ii, i0);
    if (in.op != OP_GEMV) return;
    const GemvDesc& d = local_desc(a.descs[in.desc], *sm.edesc, ept, NEPI, BAR_EPI);
    if (d.n == NB) epilogue_gemv<NB>(in, d, a, sm, seg, ept, epoch, trace(a, R_EPI, ii, i0));
    else epilogue_gemv<QB>(in, d, a, sm, seg, ept, epoch, trace(a, R_EPI, ii, i0));
    named_sync(BAR_EPI, NEPI);
    if (ept == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);  // (bar.sync before: all stores)
    if (ept == 0) {
      *sm.epi_rel = ii - i0 + 1;  // (opens the producer's gate)
      trace(a, R_EPI, ii, i0).stamp(TS_EPI_RELEASE);
    }
  }
};

// warpgroups 3-4: the B operands (and the next two-phase GEMV's first phase), the attention, PREP, QUANT, SELECT
struct WrkRole {
  bool is_prepared = false;  // the current GEMV's B operand has had its first phase during the previous op
  uint16_t aux_seen = 0;  // the last second dependency (counter + 1) a prepared operand saw complete
  uint32_t conv_n = 0;    // the two-phase GEMVs so far (their base taps' TMEM region: conv_n % 2)
  __device__ __forceinline__ void instr(const Instr& in, uint32_t ii, uint32_t i0, uint32_t i1, const DkArgs& a,
                                        const Smem& sm, RingPos& cc, int pt, uint32_t epoch) {
    WorkerConst& wc = *sm.wc;
    if (pt == 0) mark(a, R_WRK, ii, i0);
    if (in.op == OP_PREP) {
      op_prep(in, a, wc, sm.ered, pt, epoch);
    } else if (in.op == OP_GEMV) {
      worker_gemv(in, local_desc(a.descs[in.desc], *sm.wdesc, pt, NWT, BAR_WRK), a, sm, wc, pt, epoch, cc, ii + 1,
                  is_prepared, aux_seen, conv_n, trace(a, R_WRK, ii, i0));
      // the next GEMV's first phase right after this one, if its B operand is built in two
      is_prepared = false;
      const Instr nx = a.instrs[ii + 1 < i1 ? ii + 1 : ii];
      if (ii + 1 < i1 && nx.op == OP_GEMV && is_two_phase(a.descs[nx.desc].bsrc)) {
        const GemvDesc& nd = local_desc(a.descs[nx.desc], *sm.pdesc, pt, NWT, BAR_WRK);
        if (nd.bsrc == B_QKV) prep_operand<B_QKV>(nx, nd, a, sm, wc, pt, epoch, aux_seen);
        else prep_operand<B_NORM_CONV>(nx, nd, a, sm, wc, pt, epoch, aux_seen);
        is_prepared = true;
        if (pt == 0) trace(a, R_WRK, ii + 1, i0).stamp(TS_WRK_PREP);  // profiling: the next GEMV's first phase done
      }
    } else {
      if (in.op == OP_ATTN)
        attn_item(in, reinterpret_cast<const AttnDesc&>(a.descs[in.desc]), a, sm, cc, *sm.att, pt, epoch, trace(a, R_WRK, ii, i0));
      else if (in.op == OP_QUANT) op_quant(in, local_desc(a.descs[in.desc], *sm.wdesc, pt, NWT, BAR_WRK), a, wc, pt, epoch);
      else if (in.op == OP_SELECT) {  // (warmed by a dry run while the LM head streams: one call site)
        constexpr int SELECT_RUNS = 2;  // the dry run, then the real one
#pragma unroll 1
        for (int rep = 0; rep < SELECT_RUNS; rep++)
          select_op(in, reinterpret_cast<const SelectDesc&>(a.descs[in.desc]), a, sm.bbuf, sm.stages, pt, epoch,
                    trace(a, R_WRK, ii, i0), opaque_flag(rep == 0));
      }
      named_sync(BAR_WRK, NWT);
      if (pt == 0 && in.done != NO_CTR) red_release_add(a.counters + in.done, 1);  // (bar.sync before: all stores)
    }
  }
};

}  // namespace dk
