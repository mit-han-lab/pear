// The draft's tensor-core GEMV: M = 128 weight rows x N = 8 / 16 tokens per tile, the weights by TMA through a ring of
// 128-byte K chunks, the SM's K band of B built once per op by the worker warps (dk_operand.cuh), fp32 accumulators in
// TMEM, split-K parts reduced by fp32 atomics, the last part finishing the tile (dk_finish.cuh).
#pragma once
#include "dk_layout.cuh"

namespace dk {

struct Smem {
  uint8_t* stages;   // [NSTAGE][STAGE_STRIDE]: A (16 KB, SW128)
  uint8_t* bbuf;     // [BBUF_BYTES]: the B operand's chunk tiles of the SM's band (or the workers' scratch)
  uint64_t* bbar;    // [BBUF_CHUNKS]: a B chunk tile built (the workers' arrival; one phase per op that uses the slot)
  volatile uint32_t* wrk_gemv;  // the GEMV instruction (index + 1) the workers build B for
  uint64_t* full;    // [NSTAGE]: A landed (the producer's transaction bytes)
  uint64_t* empty;   // [NSTAGE]: the stage's MMAs done (tcgen05.commit)
  uint64_t* dfull;   // [N_ACC]: accumulator buffer complete
  uint64_t* dempty;  // [N_ACC]: accumulator buffer read (the epilogue's WG_WARPS warps)
  uint64_t* base_full;   // [N_BASE]: a TMEM base-tap region written (warpgroup 0's WG_WARPS warps)
  uint64_t* base_empty;  // [N_BASE]: a TMEM base-tap region read (the worker warps' conv phase)
  float* xchg;       // [XCHG_FLOATS]: epilogue exchange
  float* ered;       // [ERED_ROWS][NB]: epilogue reductions (the warps' sums, then per-token results)
  uint32_t* eflag;   // epilogue broadcast word
  volatile uint32_t* prod_n;  // ring stages the producer has issued: the roles that skip ops pace their barrier waits
  volatile uint32_t* epi_rel;  // 1 + the last instruction whose epilogue released its count (the producer gate)
  struct StepSmem* step;       // the step's tables (the epilogue's finishers)
  struct WorkerConst* wc;      // the workers' per-op constants
  struct AttnSmem* att;        // the attention item's state
  GemvDesc* wdesc;             // the role copies of the GEMV descriptors: the workers', the epilogue's, the prepared
  GemvDesc* edesc;             // op's (the next two-phase GEMV's first phase)
  GemvDesc* pdesc;
  uint32_t tmem;
};

constexpr uint32_t PROG_SHIFT = 16;          // a progress word: the instruction << PROG_SHIFT | its stage or step
constexpr uint32_t PROG_SUB_MASK = 0xffffu;  // a progress word's stage or step
// a role's progress word (hang diagnosis): its instruction << PROG_SHIFT | the stage or segment it waits on; and its
// timestamps within the instruction (profiling: null when off)
struct Trace {
  uint32_t* word;
  uint32_t instr;
  uint64_t* ts;  // [NSUB]: 0 the instruction's start, then points within it
  __device__ __forceinline__ void at(uint32_t sub) const {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(word), "r"((instr << PROG_SHIFT) | (sub & PROG_SUB_MASK))
                 : "memory");
  }
  __device__ __forceinline__ void stamp(int k) const {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    put(k, t);
  }
  __device__ __forceinline__ void put(int k, uint64_t v) const {
    if (ts != nullptr && k < NSUB) ts[k] = v;
  }
};

// ------------------------------------------------------------------------------------------------ producer
constexpr int TS_ISSUED = 1;  // the producer's and the MMA warp's timestamp: the instruction's stages all issued
// Warp W_PROD: the instruction's weight tiles into the ring, waiting only for free slots.
__device__ __forceinline__ void produce_gemv(const Instr& in, const GemvDesc& d, const Smem& sm, RingPos& cc, const Trace& tr) {
  const uint32_t bytes = A_BYTES;
  const void* tmap = reinterpret_cast<const void*>(d.tmap);
  const StageOrder ord(in);
  for (uint32_t k = 0; k < ord.len(); k++) {
    const auto [mt, kc] = stage_at(in, ord.at(k));
    if (elect_one()) tr.at(k);
    if (cc.n >= NSTAGE) mbar_wait(&sm.empty[cc.slot], cc.phase ^ 1);
    if (elect_one()) {
      uint64_t* bar = &sm.full[cc.slot];
      mbar_expect_tx(bar, bytes);
      uint8_t* dst = sm.stages + cc.slot * STAGE_STRIDE;
      const uint64_t pol = l2_evict_first();
      if (d.paired) {  // an M tile is 64 gate rows and the 64 up rows of the same outputs
        tma_load_2d_pol(dst, tmap, kc * CHUNK, mt * (TILE_M / 2), bar, pol);
        tma_load_2d_pol(dst + A_BYTES / 2, tmap, kc * CHUNK, d.rows / 2 + mt * (TILE_M / 2), bar, pol);
      } else {
        tma_load_2d_pol(dst, tmap, kc * CHUNK, mt * TILE_M, bar, pol);
      }
    }
    __syncwarp();
    cc.next(NSTAGE);
    *sm.prod_n = cc.n;
  }
  if (elect_one()) tr.stamp(TS_ISSUED);
}

// A role that skips other ops' stages can reach a slot's barrier several phases early, where a parity wait cannot
// tell phase k from k - 2: it first waits until the producer has issued ring position `ring_pos` (its own empty wait
// orders it).
__device__ __forceinline__ void pace(const Smem& sm, uint32_t ring_pos) {
  constexpr uint32_t ISSUE_POLL_NS = 32;  // the poll interval on the producer's issued count
  while (*sm.prod_n <= ring_pos) __nanosleep(ISSUE_POLL_NS);
}

// The B buffer free for another use: the MMA has committed the ring's last stage so far (position cc.n - 1; commits
// complete in order). Once the producer has issued it, that slot's previous use is consumed: the parity is unambiguous.
__device__ __forceinline__ void wait_ring_drained(const Smem& sm, const RingPos& cc) {
  if (cc.n == 0) return;
  const uint32_t last_pos = cc.n - 1;
  pace(sm, last_pos);
  mbar_wait(&sm.empty[last_pos % NSTAGE], (last_pos / NSTAGE) & 1);
}

// the phases of the B slots' barriers a role waits on: bit j flips with every op that builds slot j (its slots b_slot0
// .. + b_slots(in) - 1; B_IMAGE: slot b_slot0 alone, for the whole operand). Each use of a slot is built after the MMAs
// of its previous use: never two phases early.
struct BSlots {
  uint64_t bits = 0;
  __device__ __forceinline__ uint32_t phase_of(uint32_t j) const { return (uint32_t)(bits >> j) & 1u; }
  __device__ __forceinline__ void flip(uint32_t slot0, uint32_t n) {
    bits ^= (n >= 64 ? ~0ull : (1ull << n) - 1) << slot0;
  }
};
static_assert(BBUF_CHUNKS <= 64, "BSlots bits");
// the distinct K chunks of a GEMV instruction's range: its B slots
__device__ __forceinline__ uint32_t b_slots(const Instr& in) { return min((uint32_t)in.kbw, in.r1 - in.r0); }
// the B slot barriers an op's instruction uses: one per chunk, B_IMAGE's whole operand one
__device__ __forceinline__ uint32_t b_bars(const Instr& in, const GemvDesc& d) { return d.bsrc == B_IMAGE ? 1u : b_slots(in); }

// ------------------------------------------------------------------------------------------------ MMA issuer
// Warp W_MMA: per M-tile segment of the instruction, its stages' MMAs into one accumulator buffer (seg); each stage's
// commit frees its ring slot, the segment's commit hands the accumulator to the epilogue. bs: the B slots' phases.
__device__ __forceinline__ void mma_gemv(const Instr& in, const GemvDesc& d, const Smem& sm, RingPos& cc, RingPos& seg,
                                         BSlots& bs, const Trace& tr) {
  const uint32_t idesc = idesc_bf16(d.n);
  // the descriptors of ring slot 0 and of the first B tile (16-byte units: + a slot's / a tile's size): the range's
  // chunk tiles from d.b_off in slot order (B_IMAGE, the LM head: the whole operand, in K order, one barrier)
  const uint64_t a0 = sw128_desc(sm.stages);
  const uint64_t b0 = sw128_desc(sm.bbuf + d.b_off);
  constexpr uint32_t A_STEP = STAGE_STRIDE >> 4;
  const uint32_t b_step = (d.n * CHUNK) >> 4;
  const bool is_whole_operand = d.bsrc == B_IMAGE;
  const uint32_t jmask = is_whole_operand ? 0u : ~0u;  // (a stage's B barrier: slot j's, or B_IMAGE's one)
  const StageOrder ord(in);
  for (uint32_t k = 0; k < ord.len();) {
    const uint32_t st = ord.at(k), n = ord.seg(k, in.kbw);  // this segment's stages (one M tile)
    if (seg.n >= N_ACC) mbar_wait(&sm.dempty[seg.slot], seg.phase ^ 1);
    tc_fence_after();
    const uint32_t t_d = sm.tmem + seg.slot * T_ACC;
    uint32_t j = b_slot(in, st), kc = stage_at(in, st).kc;  // the next stage's B slot and K chunk (+ 1 per stage)
    for (uint32_t i = 0; i < n;) {
      // a warp-wide wait for the next stage, then one lane issues it and every following stage whose barriers have
      // completed, in one run (the catch-up once B is built: a warp-wide wait per stage is slower than the tensor pipe)
      if (elect_one()) tr.at(k + i);
      mbar_wait(&sm.full[cc.slot], cc.phase);
      mbar_wait(&sm.bbar[d.b_slot0 + (j & jmask)], bs.phase_of(d.b_slot0 + (j & jmask)));
      tc_fence_after();
      uint32_t run = 0;
      if (elect_one()) {
        RingPos r = cc;
        uint32_t jj = j, kk = kc;
        bool is_next_ready;
        do {
          if (run) tc_fence_after();
          const uint32_t pos = is_whole_operand ? kk : jj;  // the stage's B chunk position
          const uint64_t ad = a0 + r.slot * A_STEP, bd = b0 + pos * b_step;
#pragma unroll
          for (uint32_t q = 0; q < KSTEPS; q++) {
            const uint32_t accum = (i + run > 0 || q > 0) ? 1u : 0u;
            const uint64_t step = 2 * q;  // 32 bytes in the descriptor's 16-byte units
            tc_mma_bf16(t_d, ad + step, bd + step, idesc, accum);
          }
          tc_commit(&sm.empty[r.slot]);
          r.next(NSTAGE);
          jj = jj + 1 == in.kbw ? 0 : jj + 1;
          kk++;
          run++;
          is_next_ready = i + run < n &&
                          mbar_test(&sm.bbar[d.b_slot0 + (jj & jmask)], bs.phase_of(d.b_slot0 + (jj & jmask))) &&
                          mbar_test(&sm.full[r.slot], r.phase);
        } while (is_next_ready);
      }
      run = __shfl_sync(0xffffffffu, run, __ffs(__ballot_sync(0xffffffffu, run != 0)) - 1);  // (the issuing lane's)
      cc.advance(run, NSTAGE);
      j = (j + run) % in.kbw;
      kc += run;
      i += run;
    }
    if (elect_one()) tc_commit(&sm.dfull[seg.slot]);
    __syncwarp();
    seg.next(N_ACC);
    k += n;
  }
  bs.flip(d.b_slot0, b_bars(in, d));
  if (elect_one()) tr.stamp(TS_ISSUED);
}

// ------------------------------------------------------------------------------------------------ epilogue
// a tile's accumulator slots at most
template <int N>
__host__ __device__ constexpr uint32_t acc_slots() {
  constexpr int ACC_FLOATS = 64;  // fp32 slot values a finisher holds per row (schedule.py ACC_FLOATS)
  return ACC_FLOATS / N;
}
// a split-K part's place in its tile: the tile's parts, its accumulator slots in use, the part's slot
struct PartSlot {
  uint32_t parts, slots, slot;
};
// The part's place in tile mt (loaded ahead: before the segment's accumulator is ready); a band op's slot from the
// table after the tile counts, by the part's band and whether it starts mid-tile.
__device__ __forceinline__ PartSlot part_slot(const GemvDesc& d, const Instr& in, uint32_t mt) {
  constexpr uint32_t TAB_PARTS_MASK = 0xFFu;  // a tile_parts entry: the tile's split-K parts,
  constexpr uint32_t TAB_SLOTS_SHIFT = 8;     // then its accumulator slots in use
  constexpr uint32_t BAND_MASK = 0xFFu;       // a band op's Instr::slot: its band,
  constexpr uint32_t BAND_NB_SHIFT = 8;       // then the band count
  constexpr uint32_t BAND_NB_MASK = 0x7Fu;    // its bits (BAND_SLOT above them)
  const uint32_t tab = reinterpret_cast<const uint16_t*>(d.tile_parts)[mt];
  uint32_t slot = in.slot;
  if (slot & BAND_SLOT) {
    const uint32_t nb = (slot >> BAND_NB_SHIFT) & BAND_NB_MASK, b = slot & BAND_MASK,
                   j = max(in.r0, mt * in.kbw) > mt * in.kbw;
    slot = reinterpret_cast<const uint8_t*>(d.tile_parts)[2 * d.mtiles + (mt * nb + b) * 2 + j];
  }
  return {tab & TAB_PARTS_MASK, tab >> TAB_SLOTS_SHIFT, slot};
}
// The tile's split-K part (this thread's row, N columns) added into its accumulator slot (at most two parts a slot: 0 +
// a + b in either order); true on the part that completes the tile, its v then the slots summed in slot order (the same
// sums whichever part came last); the slots and the counter are left zeroed.
template <int N>
__device__ __forceinline__ bool reduce_part(const GemvDesc& d, const PartSlot& ps, uint32_t mt, int ept, float (&v)[N],
                                            const Smem& sm) {
  const uint32_t parts = ps.parts, slots = ps.slots;
  if (parts <= 1) return true;
  constexpr size_t SLOT = TILE_M * N;  // (floats)
  float* acc = reinterpret_cast<float*>(d.acc) + (size_t)mt * acc_slots<N>() * SLOT + (size_t)ept * N;
  float* slot_acc = acc + ps.slot * SLOT;
#pragma unroll
  for (int i = 0; i < N; i += 4) red_add_v4(slot_acc + i, v[i], v[i + 1], v[i + 2], v[i + 3]);
  named_sync(BAR_EPI, NEPI);
  if (ept == 0) {
    uint32_t* ctr = reinterpret_cast<uint32_t*>(d.tile_ctr) + mt;
    const bool is_last_part = atom_add_acq_rel(ctr, 1) == parts - 1;
    if (is_last_part) *ctr = 0;  // (no other part touches it again in this launch)
    *sm.eflag = is_last_part;
  }
  named_sync(BAR_EPI, NEPI);
  if (!*reinterpret_cast<volatile uint32_t*>(sm.eflag)) return false;
  // the slots summed in slot order, a group of GROUP_VALUES values per row in flight at once
  constexpr int GROUP_VALUES = 16;  // a row's values in flight at once (the registers' limit: no spills)
  constexpr uint32_t G = GROUP_VALUES / N;
  float4 t[N / 4];
#pragma unroll
  for (uint32_t k0 = 0; k0 < acc_slots<N>(); k0 += G) {
    if (k0 >= slots) break;
    float4 g[G][N / 4];
#pragma unroll
    for (uint32_t k = 0; k < G; k++)
#pragma unroll
      for (int i = 0; i < N / 4; i++)
        if (k0 + k < slots) g[k][i] = __ldcg(reinterpret_cast<const float4*>(acc + (k0 + k) * SLOT) + i);
#pragma unroll
    for (uint32_t k = 0; k < G; k++)
#pragma unroll
      for (int i = 0; i < N / 4; i++) {
        if (k0 + k == 0) t[i] = g[k][i];
        else if (k0 + k < slots) t[i].x += g[k][i].x, t[i].y += g[k][i].y, t[i].z += g[k][i].z, t[i].w += g[k][i].w;
      }
  }
#pragma unroll
  for (int i = 0; i < N / 4; i++)
    v[4 * i] = t[i].x, v[4 * i + 1] = t[i].y, v[4 * i + 2] = t[i].z, v[4 * i + 3] = t[i].w;
#pragma unroll
  for (uint32_t k = 0; k < acc_slots<N>(); k++)
#pragma unroll
    for (int i = 0; i < N / 4; i++)
      if (k < slots) __stcg(reinterpret_cast<float4*>(acc + k * SLOT) + i, make_float4(0.f, 0.f, 0.f, 0.f));
  return true;
}

// this thread's accumulator row of buffer `slot` (N columns)
template <int N>
__device__ __forceinline__ void load_acc(const Smem& sm, uint32_t slot, float (&v)[N]) {
  float h[8];
  tc_ld8(sm.tmem + slot * T_ACC, h);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = h[i];
  if (N > 8) {
    tc_ld8(sm.tmem + slot * T_ACC + 8, h);
#pragma unroll
    for (int i = 0; i < 8; i++) v[(8 + i) % N] = h[i];
  }
}

}  // namespace dk
