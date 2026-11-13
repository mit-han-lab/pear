// The decode step's instruction, its record (the device buffer the host fills before each launch: the input tokens,
// positions, paged KV cache blocks and Gated DeltaNet state slots), and OP_BOOK, which writes the V2 runner's inputs.
#pragma once
// (after the checkpoint format's csrc/common.cuh, which every includer includes first)

namespace mk {

struct alignas(16) Instr {  // 32 bytes (schedule.py INSTR)
  uint16_t op, desc;
  uint16_t dep, done;  // the counters this instruction waits on / counts (INSTR_NO_CTR: none)
  uint32_t r0, r1;     // GEMV: stage range [r0, r1) of the op's (M tile, 128-byte K chunk) space; other ops: their item
  uint32_t tile;       // this instruction's index among the op's n_inst
  uint32_t aux;        // per op: GEMV warm-up / park slots / pacing; attention: its zeroing counter; sampler: its ranks
  uint32_t pr0, pr1;   // GEMV with AUX_SPLIT: the stage range of the other CTA's instruction of the op (its CTA pair)
};
static_assert(sizeof(Instr) == 32, "Instr layout");
constexpr uint16_t INSTR_NO_CTR = 0xffff;  // Instr::dep / done: no counter

// the record's int32 words
enum StepRecWord : int {
  SR_FLAGS = 0,      // REC_FLAG_*: the input tokens at SR_TOK_PTR instead of SR_TOK, the position at SR_POS_PTR
  SR_TOK = 1,
  SR_TOK_PTR = 2,    // [2, 3] uint64
  SR_POS = 4,        // the new token's position (tokens before it)
  SR_MPOS = 5,       // [5, 6, 7] M-RoPE positions of the (t, h, w) sections
  SR_SEQ_LEN = 8,    // tokens attended to, the new one included
  SR_NMB = 9,        // attention block-table entries (manager blocks)
  SR_RNG = 10,       // [10, 11] uint64: the sampler's counter for this step
  SR_SEED = 12,      // [12, 13] uint64
  SR_NZERO = 14,     // newly allocated blocks to zero in the attention layers' caches (vLLM's KVBlockZeroer)
  SR_NTOK = 15,      // the step's token rows T (1..NT): tokens SR_TOK (T = 1) or SR_TOK_PTR[0..T), positions SR_POS + t
  SR_MAMBA = 16,     // [16 + 2g, 17 + 2g]: Gated DeltaNet group g's state slots: read (prior block), write (current)
  SR_ZERO_IDS = 24,  // [24, 32): their block ids
  SR_GDNT = 32,      // [32, 80): OP_GDNT's step words (gdn_t.cuh GdnStepWord: GS_BT + 2 GS_MAX_GROUPS)
  SR_POS_PTR = 80,   // [80, 81] uint64 (REC_FLAG_POS_ON_DEVICE): the step's first position, an int32 on the device:
                     // the V2 runner's num_computed_tokens[row] (asynchronous scheduling: SR_POS is an upper bound)
  SR_ATT_BT = 82,    // [82 + 2g, 83 + 2g] uint64: attention KV cache group g's block-table row on the device (int32
                     // kernel-block ids: the V2 runner's block_tables[g].gpu[row])
  SR_SS = 96,        // [96, 108): the spec sampler's step words (spec_sample.cuh SsStepWord)
  SR_BK = 108,       // [108, 112): the step's inputs written by the launch (BookWord)
  SR_BT = 112,       // the attention layers' block table (manager block ids, host-copied: the one-group synthetic step)
};
// SR_BK words: the request's row of V2's request states, whether OP_BOOK writes the step's inputs, what is left to
// apply of the prior launched step (BK_PREV), and the flag value OP_PREV stores to that step's host ring slot (BK_SEQ)
enum BookWord : int { BK_REQ = 0, BK_ON = 1, BK_PREV = 2, BK_SEQ = 3 };
// BK_PREV's fields: bit 0 PREV_OUT (its outputs into the host ring), bit 1 PREV_POST (V2's post_update), bits 2-3 its
// ring slot, bits 4-7 its rows T, bits 8-31 its request's row
constexpr uint32_t PREV_OUT = 1u, PREV_POST = 2u;
constexpr uint32_t PREV_SLOT_SHIFT = 2, PREV_SLOT_MASK = 3u, PREV_T_SHIFT = 4, PREV_T_MASK = 15u, PREV_REQ_SHIFT = 8;
constexpr int REC_FLAG_TOKEN_ON_DEVICE = 1, REC_FLAG_POS_ON_DEVICE = 2;  // SR_FLAGS bits

// The record's header (words < SR_BT) is a launch argument, copied by mk_main into each CTA's rec_sh before any read:
// no host-to-device copy (a copy-engine hop between the prior kernel and the launch) precedes the step. The words from
// SR_BT on (the block table) stay in the device record.
struct RecParam {
  int32_t w[SR_BT];
};
__shared__ __align__(16) int32_t rec_sh[SR_BT];

// a record word: the header's from rec_sh, the rest from the device record (rec_sh by a volatile load, read where used:
// a non-volatile one is hoisted out of the prologue warps' instruction loop into long-lived registers, and that or an
// asm load makes the 56-register roles spill)
__device__ __forceinline__ int32_t rec_i32(const int32_t* rec, int w) {
  if (w < SR_BT) return *reinterpret_cast<const volatile int32_t*>(rec_sh + w);
  int32_t v;
  asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(rec + w));
  return v;
}
// a uint64 word of the header (w even: 8-byte aligned in rec_sh)
__device__ __forceinline__ uint64_t rec_u64(int w) {
  return *reinterpret_cast<const volatile uint64_t*>(rec_sh + w);
}
// the step's first position: the device's (POS_ON_DEVICE) or the host's
__device__ __forceinline__ int32_t rec_pos(const int32_t* rec) {
  if (rec_i32(rec, SR_FLAGS) & REC_FLAG_POS_ON_DEVICE)
    return *reinterpret_cast<const volatile int32_t*>(rec_u64(SR_POS_PTR));
  return rec_i32(rec, SR_POS);
}
// attention KV cache group g's block-table row (kernel-block ids) on the device
__device__ __forceinline__ const int32_t* rec_att_bt(const int32_t* rec, int g) {
  return reinterpret_cast<const int32_t*>(rec_u64(SR_ATT_BT + 2 * g));
}
// the input token of row t: written by the host (one row), or a device tensor's (vLLM's input ids: async scheduling's
// fed-back tokens, which a teacher-forcing caller may have overwritten)
__device__ __forceinline__ int32_t rec_token(const int32_t* rec, uint32_t t) {
  if (rec_i32(rec, SR_FLAGS) & REC_FLAG_TOKEN_ON_DEVICE) {
    const int32_t* p = reinterpret_cast<const int32_t*>(rec_u64(SR_TOK_PTR));
    return *reinterpret_cast<const volatile int32_t*>(p + t);
  }
  return rec_i32(rec, SR_TOK);
}

// OP_BOOK's descriptor (an OpDesc's 256 bytes, schedule.py BOOK): the V2 runner's tensors the step's inputs come from
// and go to
struct BookDesc {
  uint64_t rec;
  uint64_t input_ids, positions, seq_lens, is_padding;  // the input buffers: int32, int64, int32, bool
  uint64_t last_sampled, draft_tokens, num_computed;    // request states: int64 [R], int64 [R][draft_stride], int32 [R]
  uint64_t bt_src, bt_dst, bt_stride, num_blocks;       // per KV cache group g: the tables' bases (uint64 [G]: the
                                                        // runner's, the inputs'), row strides (int64 [G]), blocks per
                                                        // request (int32 [G][num_blocks_stride])
  uint64_t sampled, num_sampled, num_rejected;          // the spec sampler's ring: int64 [RING][nt], int32 [RING]
  uint32_t draft_stride, groups, num_blocks_stride, nt;
  uint32_t pad[30];
};
static_assert(sizeof(BookDesc) == 256, "BookDesc layout");

// OP_BOOK (the prologue warps of SM 0, nth threads, first; T: the step's rows): what V2's prepare_inputs and
// gather_block_tables write for one request past its prompt (ids: its last sampled token, then its drafts; positions,
// seq_len, no padding, its block-table rows into row 0 of the inputs'); not inlined, so its code stays out of mk_main
__device__ __noinline__ void book_inputs(const BookDesc& d, uint32_t T, int pt, int nth, int bar) {
  const int32_t* rec = reinterpret_cast<const int32_t*>(d.rec);
  const int req = rec_i32(rec, SR_BK + BK_REQ);
  const uint32_t w = (uint32_t)rec_i32(rec, SR_BK + BK_PREV);
  // PREV_POST: the prior launched step's post_update, not applied yet: its last sampled token and computed tokens
  // (its rows less the rejected) from its ring slot, written into the states once every thread has read them (bar)
  const bool post_pending = (w & PREV_POST) != 0u;
  const int64_t r = (w >> PREV_SLOT_SHIFT) & PREV_SLOT_MASK, Tp = (w >> PREV_T_SHIFT) & PREV_T_MASK;
  // (every load first, in parallel: the slot's count and its rows' tokens beside the states)
  int32_t nc = reinterpret_cast<const int32_t*>(d.num_computed)[req];
  const int32_t ns = post_pending ? reinterpret_cast<const int32_t*>(d.num_sampled)[r] : 0;
  const int32_t rej = post_pending ? reinterpret_cast<const int32_t*>(d.num_rejected)[r] : 0;
  const int64_t ring_token = post_pending && pt < Tp ? reinterpret_cast<const int64_t*>(d.sampled)[r * d.nt + pt] : 0;
  const int64_t last_token = pt == 0 ? reinterpret_cast<const int64_t*>(d.last_sampled)[req] : 0;
  nc += post_pending ? (int32_t)Tp - rej : 0;
  const int last_row = ns > 0 ? ns - 1 : 0;  // (the thread holding the last sampled token)
  if (pt == last_row) reinterpret_cast<int32_t*>(d.input_ids)[0] = (int32_t)(ns > 0 ? ring_token : last_token);
  if (pt == 0) {
    reinterpret_cast<int32_t*>(d.seq_lens)[0] = nc + (int32_t)T;
  } else if (pt < (int)T) {
    const int64_t* drafts = reinterpret_cast<const int64_t*>(d.draft_tokens) + (int64_t)req * d.draft_stride;
    reinterpret_cast<int32_t*>(d.input_ids)[pt] = (int32_t)drafts[pt - 1];
  }
  if (pt < (int)T) {
    reinterpret_cast<int64_t*>(d.positions)[pt] = (int64_t)nc + pt;
    reinterpret_cast<bool*>(d.is_padding)[pt] = false;
  }
  const uint64_t* src = reinterpret_cast<const uint64_t*>(d.bt_src);
  const uint64_t* dst = reinterpret_cast<const uint64_t*>(d.bt_dst);
  const int64_t* stride = reinterpret_cast<const int64_t*>(d.bt_stride);
  const int32_t* nb = reinterpret_cast<const int32_t*>(d.num_blocks);
#pragma unroll 1
  for (uint32_t g = 0; g < d.groups; g++) {
    const int n = nb[(size_t)g * d.num_blocks_stride + req];
    const int32_t* row = reinterpret_cast<const int32_t*>(src[g]) + (int64_t)req * stride[g];
    int32_t* out = reinterpret_cast<int32_t*>(dst[g]);
#pragma unroll 1
    for (int j = pt; j < n; j += nth) out[j] = row[j];
  }
  if (post_pending) {  // (post_pending is the record's: every thread reaches the barrier)
    named_sync(bar, nth);
    if (pt == last_row && ns > 0) reinterpret_cast<int64_t*>(d.last_sampled)[req] = ring_token;
    if (pt == 0) reinterpret_cast<int32_t*>(d.num_computed)[req] = nc;
  }
}

}  // namespace mk
