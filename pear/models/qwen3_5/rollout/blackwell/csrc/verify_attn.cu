// Paged attention of a step's decode or verify rows on the tensor cores (tcgen05): T query tokens per request, causal
// within the block, over vLLM's bf16 KV cache; one CTA per SM takes an equal share of the step's pages and piece starts
// and a second kernel (programmatic dependent launch) merges the requests' pieces split over CTAs.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <mutex>
#include <unordered_map>

#include "sync.cuh"
#include "tcgen05.cuh"

namespace {

constexpr int HD = 256;                          // head dim
constexpr int PAGE = 64;                         // tokens per KV cache kernel block
constexpr int BOX_COLS = 64;                     // bf16 per TMA box row: the 128-byte swizzle span
constexpr uint32_t BOX_BYTES = PAGE * 128;       // a box: a page's 64 rows of 128 bytes
constexpr uint32_t STAGE_BYTES = 4 * BOX_BYTES;  // a page's K or V: 4 boxes of 64 dims
constexpr int STAGES = 6;                        // ring stages: 3 pages in flight
constexpr int MAX_ROWS = 48;                     // query rows per KV head (GQA group x T) at most
constexpr uint32_t Q_SLOT = 2 * HD + 16;         // a query row's smem slot, padded: conflict-free row reads
constexpr int MAX_REQUESTS = 256;
constexpr int MAX_SEGMENTS = 512;                // requests x KV heads
constexpr int NUM_WARPS = 6;                     // the producer, the MMA issuer, 4 epilogue warps
constexpr int THREADS = NUM_WARPS * 32;
constexpr int W_PRODUCER = 0, W_MMA = 1, W_EPILOGUE = 2;
constexpr int EPILOGUE_THREADS = 128;
constexpr int MERGE_THREADS = 256;               // the merge kernel's CTA: fits an SM beside an attention CTA
constexpr int MERGE_CTAS_PER_SM = 3;            // merge CTAs per SM: as many items in flight as the registers allow
constexpr int MERGE_ITEMS = 2;                   // 4-dim row items a merge thread loads at once
constexpr int MERGE_CHUNK = 4;                   // pieces an item loads at once
// a piece's start (its query rows into TMEM, the pipeline's restart) costs about a page per this many query tokens,
// plus one: so many units head every segment, so that the equal shares weigh the pieces they start
constexpr int PIECE_START_TOKENS_PER_PAGE = 2;
constexpr uint32_t TMEM_COLS = 512;
constexpr uint32_t T_O = 0, T_S = 256, T_Q = 384;  // TMEM columns: O [256], S double buffer [2][64], Q [128]
constexpr float RESCALE_LOG2 = 32.0f;  // O is rescaled once a row's max grows by more (base 2): P, l, O stay finite
constexpr double LOG2_E = 1.4426950408889634;  // the scores' scale into base 2
constexpr uint32_t V_ROW_CHUNKS = 32;            // 16-byte chunks of a V row in a stage: 4 boxes x 8
// kind::f16 instruction descriptors (bf16 x bf16 -> f32, M = 128): S = Q K^T (N 64, K-major keys), O += P V (N 128,
// V MN-major)
constexpr uint32_t ID_QK = (1u << 4) | (1u << 7) | (1u << 10) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
constexpr uint32_t ID_PV = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);

struct Params {
  const __nv_bfloat16* q;  // [tokens][query heads][HD]
  int64_t q_tok, q_head;   // element strides
  const __nv_bfloat16* key;  // the step's keys and values [tokens][KV heads][HD], written into the cache here
  const __nv_bfloat16* value;
  int64_t k_tok, k_head, v_tok, v_head;
  const int64_t* slot_mapping;  // [tokens]: the cache row of each (kernel block x PAGE + token), negative: none
  __nv_bfloat16* cache;         // [blocks][KV heads][PAGE][K | V]
  int64_t c_block, c_head, c_token;
  __nv_bfloat16* out;
  int64_t o_tok, o_head;
  const int* block_table;  // [requests][pages]: kernel block ids
  int64_t bt_stride;
  const int* seq_lens;     // [requests]: the attended length, the T query tokens last
  float* part_o;           // [slots][MAX_ROWS][HD]: split pieces' unnormalized O
  float* part_ml;          // [slots][MAX_ROWS][2]: their row max (base 2, scaled) and sum
  int n, T, group, kv_heads;
  int piece_units;  // the units heading every segment for the start of a piece (PIECE_START_TOKENS_PER_PAGE)
  float scale_log2;
};

struct alignas(1024) Smem {
  uint8_t ring[STAGES][STAGE_BYTES];
  uint8_t q[MAX_ROWS * Q_SLOT];
  int start[MAX_SEGMENTS + 1];  // the segments' first units (a segment: a request's KV head; its units: its pages)
  int lens[MAX_REQUESTS];
  int warp_sums[NUM_WARPS];
  uint64_t full[STAGES], empty[STAGES];
  uint64_t s_full[2], p_ready[2], pv_done[2];
  uint64_t q_full, q_empty, q_ready, o_done, o_free;
  uint32_t tmem_base;
};

struct MergeSmem {
  int start[MAX_SEGMENTS + 1];
  int lens[MAX_REQUESTS];
  int warp_sums[MERGE_THREADS / 32];
};

// ---- PTX helpers ----
__device__ __forceinline__ void named_sync(int id, int count) {
  asm volatile("bar.warp.sync -1;\n\tbar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float exp2_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}


// ---- the schedule: every role walks the same pieces (a segment's pages within the CTA's units) ----
struct Piece {
  int s, i, h, len;  // segment s: request i, KV head h, its attended length
  int pg0, pg1;      // its pages [pg0, pg1) in this CTA (none for an empty request)
  bool is_whole;     // the CTA holds every unit of the segment
};

struct PieceWalk {
  int s, u0, u1;

  __device__ PieceWalk(const Smem& sm, int S, int u0_, int u1_) : u0(u0_), u1(u1_) {
    int lo = 0, hi = S - 1;  // the last segment starting at or before u0
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (sm.start[mid] <= u0) lo = mid; else hi = mid - 1;
    }
    s = lo;
  }

  __device__ bool next(const Smem& sm, const Params& p, int S, Piece& pc) {
    for (;; ++s) {
      if (s >= S || sm.start[s] + p.piece_units >= u1) return false;
      const int first = sm.start[s] + p.piece_units;  // the segment's first page, after its start's units
      const int a = max(u0, first), b = min(u1, sm.start[s + 1]);
      if (a >= b) continue;  // the CTA holds none of the segment's pages
      pc.s = s;
      pc.i = s / p.kv_heads;
      pc.h = s % p.kv_heads;
      pc.len = sm.lens[pc.i];
      pc.pg0 = pc.len > 0 ? a - first : 0;
      pc.pg1 = pc.len > 0 ? b - first : 0;
      pc.is_whole = a == first && b == sm.start[s + 1];
      ++s;
      return true;
    }
  }
};

// the requests' lengths into smem and the segments' first units (every thread of the CTA)
template <int NT>
__device__ void schedule(const Params& p, int S, int* start, int* lens, int* warp_sums) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < p.n; i += NT) lens[i] = __ldg(p.seq_lens + i);
  __syncthreads();
  constexpr int PER = (MAX_SEGMENTS + NT) / NT;  // (MAX_SEGMENTS + 1 offsets)
  int units[PER];
  int total = 0;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int s = tid * PER + k;
    units[k] = s < S ? p.piece_units + max((lens[s / p.kv_heads] + PAGE - 1) / PAGE, 1) : 0;
    total += units[k];
  }
  int x = total;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  int run = x - total;
  for (int w = 0; w < warp; ++w) run += warp_sums[w];
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int s = tid * PER + k;
    if (s <= S) start[s] = run;
    run += units[k];
  }
  __syncthreads();
}

// the CTA holding unit u: the largest c with c U / G <= u
__device__ __forceinline__ int unit_owner(int64_t u, int64_t U, int G) { return (int)(((u + 1) * G - 1) / U); }

// ---- the producer warp: per piece its query rows (one bulk copy per row, the rows over the lanes; the CTA's first
// piece after its first page), per page its K and V stages (lane 0); before the TMA loads of a page holding the step's
// tokens, every lane writes their keys and values into its rows ----
__device__ void write_step_tokens(const Params& p, const Piece& pc, int pg, int lane) {
  for (int j = 0; j < p.T; ++j) {
    const int pos = pc.len - p.T + j;
    if (pos < 0 || pos / PAGE != pg) continue;
    const int64_t slot = __ldg(p.slot_mapping + pc.i * p.T + j);
    if (slot < 0) continue;
    const int64_t token = pc.i * p.T + j;
    const uint4* k = reinterpret_cast<const uint4*>(p.key + token * p.k_tok + pc.h * p.k_head);
    const uint4* v = reinterpret_cast<const uint4*>(p.value + token * p.v_tok + pc.h * p.v_head);
    uint4* row = reinterpret_cast<uint4*>(p.cache + (slot / PAGE) * p.c_block + pc.h * p.c_head + (slot % PAGE) * p.c_token);
    row[lane] = __ldg(k + lane);  // K: 32 lanes x 16 bytes
    row[HD / 8 + lane] = __ldg(v + lane);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");  // the rows, for the TMA loads
}

// a piece's R query rows into shared memory: lane 0 sets their bytes, every lane copies its rows
__device__ void load_query_rows(const Params& p, const Piece& pc, Smem& sm, int R, int lane) {
  if (lane == 0) mbar_expect_tx(&sm.q_full, (uint32_t)R * 2 * HD);
  __syncwarp();
  for (int r = lane; r < R; r += 32) {
    const int t = r / p.group, j = r % p.group;
    const __nv_bfloat16* src = p.q + (int64_t)(pc.i * p.T + t) * p.q_tok + (int64_t)(pc.h * p.group + j) * p.q_head;
    bulk_g2s(smem_u32(sm.q + r * Q_SLOT), src, 2 * HD, &sm.q_full);
  }
}

__device__ void produce(const CUtensorMap* map, const Params& p, Smem& sm, int S, int u0, int u1, int lane) {
  const uint64_t pol = l2_evict_first();
  const int R = p.group * p.T;
  PieceWalk walk(sm, S, u0, u1);
  Piece pc;
  uint32_t stage = 0, piece = 0;
  while (walk.next(sm, p, S, pc)) {
    if (pc.pg1 == pc.pg0) continue;
    const int* table = p.block_table + (int64_t)pc.i * p.bt_stride;
    int block_next = __ldg(table + pc.pg0);  // first: the first page's TMA loads wait on it
    PieceWalk peek = walk;
    Piece next_piece;
    if (lane == 0 && peek.next(sm, p, S, next_piece) && next_piece.pg1 > next_piece.pg0) {
      const int* next_block = p.block_table + (int64_t)next_piece.i * p.bt_stride + next_piece.pg0;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(next_block));  // the next piece's first block id into L2
    }
    // the CTA's first piece issues its query rows after its first page's TMA loads (the rows' copies would delay them),
    // a later piece before its pages (which wait on the ring)
    if (piece > 0) {
      mbar_wait(&sm.q_empty, (piece - 1) & 1);
      load_query_rows(p, pc, sm, R, lane);
    }
    const int first_new = max(pc.len - p.T, 0) / PAGE;  // the first page holding one of the step's tokens
    for (int pg = pc.pg0; pg < pc.pg1; ++pg) {
      const int block = block_next;
      if (pg + 1 < pc.pg1) block_next = __ldg(table + pg + 1);
      if (pg >= first_new) write_step_tokens(p, pc, pg, lane);
      __syncwarp();
      if (lane == 0) {
        for (int half = 0; half < 2; ++half, ++stage) {  // the page's K, then its V
          const uint32_t slot = stage % STAGES;
          if (stage >= STAGES) mbar_wait(&sm.empty[slot], ((stage / STAGES) & 1) ^ 1);
          mbar_expect_tx(&sm.full[slot], STAGE_BYTES);
#pragma unroll
          for (int b = 0; b < 4; ++b)
            tma_load_4d(smem_u32(sm.ring[slot] + b * BOX_BYTES), map, (half * 4 + b) * BOX_COLS, 0, pc.h, block,
                        &sm.full[slot], pol);
        }
      }
      if (piece == 0 && pg == pc.pg0) load_query_rows(p, pc, sm, R, lane);
    }
    ++piece;
  }
}

// ---- the MMA issuer: per page S_b = Q K^T, then (once the softmax has written P_b) O += P_b V ----
__device__ void mma(const Params& p, Smem& sm, int S, int u0, int u1, uint32_t tmem) {
  PieceWalk walk(sm, S, u0, u1);
  Piece pc;
  uint32_t page = 0, piece = 0;  // the CTA's pages and pieces so far
  while (walk.next(sm, p, S, pc)) {
    const uint32_t np = (uint32_t)(pc.pg1 - pc.pg0);
    if (np == 0) continue;
    mbar_wait(&sm.q_ready, piece & 1);
    tc_fence_after();
    for (uint32_t j = 0; j <= np; ++j) {
      if (j < np) {
        const uint32_t pj = page + j, b = pj & 1, st = 2 * pj, slot = st % STAGES;
        mbar_wait(&sm.full[slot], (st / STAGES) & 1);
        tc_fence_after();
        const uint32_t kbase = smem_u32(sm.ring[slot]);
#pragma unroll
        for (uint32_t kk = 0; kk < HD / 16; ++kk) {  // dims 16 kk..: box kk / 4, 32 bytes per k-step
          const uint64_t desc = sw128_desc(kbase + (kk >> 2) * BOX_BYTES + (kk & 3) * 32, 16);
          tc_mma_ts(tmem + T_S + 64 * b, tmem + T_Q + 8 * kk, desc, ID_QK, kk > 0 ? 1u : 0u);
        }
        tc_commit(&sm.s_full[b]);
        tc_commit(&sm.empty[slot]);
      }
      if (j >= 1) {
        const uint32_t pj = page + j - 1, b = pj & 1, st = 2 * pj + 1, slot = st % STAGES;
        if (j == 1 && piece > 0) mbar_wait(&sm.o_free, (piece - 1) & 1);
        mbar_wait(&sm.p_ready[b], (pj >> 1) & 1);
        mbar_wait(&sm.full[slot], (st / STAGES) & 1);
        tc_fence_after();
        const uint32_t vbase = smem_u32(sm.ring[slot]);
#pragma unroll
        for (uint32_t kg = 0; kg < PAGE / 16; ++kg) {  // keys 16 kg..: P columns 8 kg.., 2 swizzle atoms of V
          const uint32_t acc = (j > 1 || kg > 0) ? 1u : 0u;
          const uint32_t pa = tmem + T_S + 64 * b + 8 * kg;
          tc_mma_ts(tmem + T_O, pa, sw128_desc(vbase + kg * 2048, BOX_BYTES), ID_PV, acc);
          tc_mma_ts(tmem + T_O + 128, pa, sw128_desc(vbase + 2 * BOX_BYTES + kg * 2048, BOX_BYTES), ID_PV, acc);
          tc_mma_ts(tmem + T_O, pa + 32, sw128_desc(vbase + kg * 2048, BOX_BYTES), ID_PV, 1u);
          tc_mma_ts(tmem + T_O + 128, pa + 32, sw128_desc(vbase + 2 * BOX_BYTES + kg * 2048, BOX_BYTES), ID_PV, 1u);
        }
        tc_commit(&sm.empty[slot]);
        tc_commit(&sm.pv_done[b]);
      }
    }
    tc_commit(&sm.o_done);
    page += np;
    ++piece;
  }
}

// ---- the epilogue warps: Q into TMEM, the online softmax of every page (P over S in TMEM), O out ----
__device__ void epilogue(const Params& p, Smem& sm, int S, int u0, int u1, uint32_t tmem, int warp, int lane) {
  const int q = warp & 3;  // the warp's TMEM lane quadrant
  const int row = 32 * q + lane, R = p.group * p.T;
  const bool is_active = 32 * q < R, has_row = row < R;
  const int t = has_row ? row / p.group : 0, j = has_row ? row % p.group : 0;
  const uint32_t tq = tmem + ((uint32_t)(32 * q) << 16);
  const int ept = (int)threadIdx.x - W_EPILOGUE * 32;
  PieceWalk walk(sm, S, u0, u1);
  Piece pc;
  uint32_t page = 0, piece = 0;
  while (walk.next(sm, p, S, pc)) {
    const uint32_t np = (uint32_t)(pc.pg1 - pc.pg0);
    if (np == 0) {  // an empty request (a padded row): zeros
      for (int item = ept; item < R * (HD / 8); item += EPILOGUE_THREADS) {
        const int r = item / (HD / 8), d8 = item % (HD / 8);
        __nv_bfloat16* dst = p.out + (int64_t)(pc.i * p.T + r / p.group) * p.o_tok +
                             (int64_t)(pc.h * p.group + r % p.group) * p.o_head + 8 * d8;
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      }
      continue;
    }
    float m_run = -INFINITY, l_run = 0.f;
    if (is_active) {
      // the query rows into TMEM (Q columns: dims packed in pairs), rows past R zero
      mbar_wait(&sm.q_full, piece & 1);
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        uint32_t v[32];
        const uint4* src = reinterpret_cast<const uint4*>(sm.q + (has_row ? row : 0) * Q_SLOT + ch * 128);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint4 u = has_row ? src[k] : make_uint4(0, 0, 0, 0);
          v[4 * k] = u.x, v[4 * k + 1] = u.y, v[4 * k + 2] = u.z, v[4 * k + 3] = u.w;
        }
        tc_st_x32(tq + T_Q + 32 * ch, v);
      }
      tc_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(&sm.q_empty);
        mbar_arrive(&sm.q_ready);
      }
      const int lim = pc.len - p.T + t;  // the last key this row sees
#pragma unroll 1
      for (uint32_t jj = 0; jj < np; ++jj) {
        const uint32_t pj = page + jj, b = pj & 1;
        const int key0 = (pc.pg0 + (int)jj) * PAGE;
        mbar_wait(&sm.s_full[b], (pj >> 1) & 1);
        tc_fence_after();
        uint32_t sv[2][32];
        tc_ld_x32(tq + T_S + 64 * b, sv[0]);
        tc_ld_x32(tq + T_S + 64 * b + 32, sv[1]);
        tc_wait_ld();
        float s[64];
#pragma unroll
        for (int c = 0; c < 64; ++c) s[c] = __uint_as_float(sv[c >> 5][c & 31]);
        if (key0 + PAGE - 1 > lim) {
#pragma unroll
          for (int c = 0; c < 64; ++c)
            if (key0 + c > lim) s[c] = -INFINITY;
        }
        float mx4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
        for (int c = 0; c < 64; ++c) mx4[c & 3] = fmaxf(mx4[c & 3], s[c]);
        const float mx = has_row ? fmaxf(fmaxf(mx4[0], mx4[1]), fmaxf(mx4[2], mx4[3])) * p.scale_log2 : -INFINITY;
        // the row's max is kept until a key exceeds it by more than RESCALE_LOG2 (then O and l are rescaled); the first
        // finite max sets it (O's row holds no term yet)
        const float m_new = fmaxf(m_run, mx);
        const bool raises_max = m_new > m_run + RESCALE_LOG2;
        const bool rescales_o = raises_max && m_run != -INFINITY;
        const float alpha = rescales_o ? exp2_approx(m_run - m_new) : 1.f;
        if (__any_sync(0xffffffffu, rescales_o)) {
          mbar_wait(&sm.pv_done[(pj - 1) & 1], ((pj - 1) >> 1) & 1);  // (jj > 0: m_run is finite)
          tc_fence_after();
#pragma unroll 1
          for (int ch = 0; ch < HD / 32; ++ch) {
            uint32_t v[32];
            tc_ld_x32(tq + T_O + 32 * ch, v);
            tc_wait_ld();
#pragma unroll
            for (int k = 0; k < 32; ++k) v[k] = __float_as_uint(__uint_as_float(v[k]) * alpha);
            tc_st_x32(tq + T_O + 32 * ch, v);
          }
          tc_wait_st();
        }
        if (raises_max) {
          l_run *= alpha;
          m_run = m_new;
        }
        // P = 2^(s - m) as bf16 pairs over the S buffer's first 32 columns, its fp32 sum into l
        const float mz = m_run == -INFINITY ? INFINITY : m_run;
        // P (bf16) and P - bf16(P) (a second bf16 term: P to ~16 bits in O), in two halves of 32 keys: P's columns
        // 16 half.., the residual's 32 + 16 half.., over S already loaded
        float sum4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t pk[16], pl[16];
#pragma unroll
          for (int c2 = 0; c2 < 16; ++c2) {
            const int c = 32 * half + 2 * c2;
            const float p0 = has_row ? exp2_approx(__fmaf_rn(s[c], p.scale_log2, -mz)) : 0.f;
            const float p1 = has_row ? exp2_approx(__fmaf_rn(s[c + 1], p.scale_log2, -mz)) : 0.f;
            sum4[c2 & 3] += p0 + p1;
            const __nv_bfloat162 hi = __floats2bfloat162_rn(p0, p1);
            pk[c2] = *reinterpret_cast<const uint32_t*>(&hi);
            pl[c2] = pack_bf16x2(p0 - __low2float(hi), p1 - __high2float(hi));
          }
          tc_st_x16(tq + T_S + 64 * b + 16 * half, pk);
          tc_st_x16(tq + T_S + 64 * b + 32 + 16 * half, pl);
        }
        l_run += (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
        if (key0 + PAGE > pc.len && q == 0) {
          // the sequence's last page: V rows past its end zeroed (stale bytes could be NaN; P is 0 there)
          const uint32_t st = 2 * pj + 1, slot = st % STAGES;
          mbar_wait(&sm.full[slot], (st / STAGES) & 1);
          const int r0 = pc.len - key0;
          for (int idx = lane; idx < (PAGE - r0) * (int)V_ROW_CHUNKS; idx += 32) {
            const int r = r0 + idx / V_ROW_CHUNKS, c = idx % V_ROW_CHUNKS;
            *reinterpret_cast<uint4*>(sm.ring[slot] + (c >> 3) * BOX_BYTES + r * 128 + (c & 7) * 16) = make_uint4(0, 0, 0, 0);
          }
          fence_async_smem();
        }
        tc_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.p_ready[b]);
      }
      mbar_wait(&sm.o_done, piece & 1);
      tc_fence_after();
    }

    const int slot = (int)blockIdx.x + pc.s;
    if (is_active) {
      // O: normalized into the output (the whole segment), or this piece's fp32 partial
      const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
      __nv_bfloat16* dst = p.out + (int64_t)(pc.i * p.T + t) * p.o_tok + (int64_t)(pc.h * p.group + j) * p.o_head;
      float* part = p.part_o + ((int64_t)slot * MAX_ROWS + row) * HD;
#pragma unroll 1
      for (int ch = 0; ch < HD / 32; ++ch) {
        uint32_t v[32];
        tc_ld_x32(tq + T_O + 32 * ch, v);
        tc_wait_ld();
        if (!has_row) continue;
        if (pc.is_whole) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            uint4 w;
            w.x = pack_bf16x2(__uint_as_float(v[8 * k]) * inv, __uint_as_float(v[8 * k + 1]) * inv);
            w.y = pack_bf16x2(__uint_as_float(v[8 * k + 2]) * inv, __uint_as_float(v[8 * k + 3]) * inv);
            w.z = pack_bf16x2(__uint_as_float(v[8 * k + 4]) * inv, __uint_as_float(v[8 * k + 5]) * inv);
            w.w = pack_bf16x2(__uint_as_float(v[8 * k + 6]) * inv, __uint_as_float(v[8 * k + 7]) * inv);
            reinterpret_cast<uint4*>(dst + 32 * ch)[k] = w;
          }
        } else {
#pragma unroll
          for (int k = 0; k < 8; ++k)
            reinterpret_cast<uint4*>(part + 32 * ch)[k] = make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
        }
      }
      if (!pc.is_whole && has_row)
        *reinterpret_cast<float2*>(p.part_ml + ((int64_t)slot * MAX_ROWS + row) * 2) = make_float2(m_run, l_run);
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.o_free);
    }
    page += np;
    ++piece;
  }
}

__global__ void __launch_bounds__(THREADS, 1) verify_attn_kernel(const __grid_constant__ CUtensorMap kv_map, const Params p) {
  extern __shared__ uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int S = p.n * p.kv_heads;

  pdl_launch_next();  // the merge kernel may start as CTAs finish
  schedule<THREADS>(p, S, sm.start, sm.lens, sm.warp_sums);
  // the units over min(grid, units) CTAs: every working CTA holds at least one
  const int64_t U = sm.start[S];
  const int G = (int)min((int64_t)gridDim.x, U), c = blockIdx.x;
  if (c >= G) return;
  const int u0 = (int)((int64_t)c * U / G), u1 = (int)((int64_t)(c + 1) * U / G);

  const int active = (p.group * p.T + 31) / 32;  // epilogue warps holding query rows
  if (threadIdx.x == 0) {
    for (int k = 0; k < STAGES; ++k) {
      mbar_init(&sm.full[k], 1);
      mbar_init(&sm.empty[k], 1);
    }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&sm.s_full[b], 1);
      mbar_init(&sm.p_ready[b], active);
      mbar_init(&sm.pv_done[b], 1);
    }
    mbar_init(&sm.q_full, 1);
    mbar_init(&sm.q_empty, active);
    mbar_init(&sm.q_ready, active);
    mbar_init(&sm.o_done, 1);
    mbar_init(&sm.o_free, active);
    mbar_init_fence();
    tma_prefetch_map(&kv_map);
  }
  if (warp == W_MMA) tc_alloc(&sm.tmem_base, TMEM_COLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = sm.tmem_base;

  if (warp == W_PRODUCER) {
    produce(&kv_map, p, sm, S, u0, u1, lane);
  } else if (warp == W_MMA) {
    if (elect_one()) mma(p, sm, S, u0, u1, tmem);
  } else {
    epilogue(p, sm, S, u0, u1, tmem, warp, lane);
  }

  __syncwarp();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (warp == W_MMA) tc_dealloc(tmem, TMEM_COLS);
}

// ---- the merge: a thread per 4 dims of a query row of a segment split over CTAs (pieces in CTAs c_lo..c_hi, slots
// c + s), MERGE_ITEMS such items at a time; a chunk of MERGE_CHUNK pieces' (m, l) and O loaded at once, merged against
// the chunk's max ----
struct MergeItem {
  int s, r, dim, c_lo, k;  // segment s, query row r, its dims dim..dim + 3, the pieces of CTAs c_lo..c_lo + k - 1
};

__device__ __forceinline__ bool merge_item(const Params& p, const MergeSmem& sm, int64_t U, int G, int R, int item,
                                           MergeItem& it) {
  constexpr int CHUNKS = HD / 4;  // a row's 4-dim items
  it.s = item / (R * CHUNKS);
  it.r = (item / CHUNKS) % R;
  it.dim = 4 * (item % CHUNKS);
  it.c_lo = unit_owner(sm.start[it.s] + p.piece_units, U, G);
  it.k = unit_owner(sm.start[it.s + 1] - 1, U, G) - it.c_lo + 1;
  return it.k > 1;
}

__device__ __forceinline__ void merge_chunk(const Params& p, const MergeItem& it, int base, float2 (&ml)[MERGE_CHUNK],
                                            float4 (&o)[MERGE_CHUNK]) {
  const int64_t slot0 = it.c_lo + base + it.s;
#pragma unroll
  for (int u = 0; u < MERGE_CHUNK; ++u) {
    const bool has_piece = base + u < it.k;
    ml[u] = has_piece ? __ldcg(reinterpret_cast<const float2*>(p.part_ml + ((slot0 + u) * MAX_ROWS + it.r) * 2))
                      : make_float2(-INFINITY, 0.f);
    o[u] = has_piece ? __ldcg(reinterpret_cast<const float4*>(p.part_o + ((slot0 + u) * MAX_ROWS + it.r) * HD + it.dim))
                     : make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

__device__ __forceinline__ void merge_combine(const float2 (&ml)[MERGE_CHUNK], const float4 (&o)[MERGE_CHUNK], float& M,
                                              float& L, float4& acc) {
  float m_new = M;
#pragma unroll
  for (int u = 0; u < MERGE_CHUNK; ++u) m_new = fmaxf(m_new, ml[u].x);
  if (m_new == -INFINITY) return;  // (no key seen so far)
  const float a = exp2f(M - m_new);
  L *= a;
  acc.x *= a, acc.y *= a, acc.z *= a, acc.w *= a;
#pragma unroll
  for (int u = 0; u < MERGE_CHUNK; ++u) {
    const float w = ml[u].x == -INFINITY ? 0.f : exp2f(ml[u].x - m_new);
    L += w * ml[u].y;
    acc.x += w * o[u].x, acc.y += w * o[u].y, acc.z += w * o[u].z, acc.w += w * o[u].w;
  }
  M = m_new;
}

__device__ __forceinline__ void merge_store(const Params& p, const MergeItem& it, float L, const float4& acc) {
  const float inv = L > 0.f ? 1.f / L : 0.f;
  const int i = it.s / p.kv_heads, h = it.s % p.kv_heads, t = it.r / p.group, j = it.r % p.group;
  __nv_bfloat16* dst = p.out + (int64_t)(i * p.T + t) * p.o_tok + (int64_t)(h * p.group + j) * p.o_head + it.dim;
  *reinterpret_cast<uint2*>(dst) = make_uint2(pack_bf16x2(acc.x * inv, acc.y * inv), pack_bf16x2(acc.z * inv, acc.w * inv));
}

__global__ void __launch_bounds__(MERGE_THREADS) merge_kernel(const Params p, int grid_main) {
  __shared__ MergeSmem sm;
  const int S = p.n * p.kv_heads, R = p.group * p.T;
  schedule<MERGE_THREADS>(p, S, sm.start, sm.lens, sm.warp_sums);
  const int64_t U = sm.start[S];
  const int G = (int)min((int64_t)grid_main, U);
  pdl_wait();  // the attention kernel complete, its partials visible

  const int stride = gridDim.x * MERGE_THREADS, items = S * R * (HD / 4);
  for (int item = blockIdx.x * MERGE_THREADS + threadIdx.x; item < items; item += MERGE_ITEMS * stride) {
    MergeItem it[MERGE_ITEMS];
    bool is_split[MERGE_ITEMS];
    float M[MERGE_ITEMS], L[MERGE_ITEMS];
    float4 acc[MERGE_ITEMS];
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x) {
      is_split[x] = item + x * stride < items && merge_item(p, sm, U, G, R, item + x * stride, it[x]);
      M[x] = -INFINITY, L[x] = 0.f, acc[x] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // the items' first chunks loaded together, then any further chunks item by item
    float2 ml[MERGE_ITEMS][MERGE_CHUNK];
    float4 o[MERGE_ITEMS][MERGE_CHUNK];
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x)
      if (is_split[x]) merge_chunk(p, it[x], 0, ml[x], o[x]);
#pragma unroll
    for (int x = 0; x < MERGE_ITEMS; ++x) {
      if (!is_split[x]) continue;
      merge_combine(ml[x], o[x], M[x], L[x], acc[x]);
      for (int base = MERGE_CHUNK; base < it[x].k; base += MERGE_CHUNK) {
        merge_chunk(p, it[x], base, ml[x], o[x]);
        merge_combine(ml[x], o[x], M[x], L[x], acc[x]);
      }
      merge_store(p, it[x], L[x], acc[x]);
    }
  }
}

// ---- host ----
struct MapKey {
  uintptr_t ptr;
  int64_t blocks, heads, stride_block, stride_head, stride_token;
  bool operator==(const MapKey& o) const {
    return ptr == o.ptr && blocks == o.blocks && heads == o.heads && stride_block == o.stride_block &&
           stride_head == o.stride_head && stride_token == o.stride_token;
  }
};
struct MapKeyHash {
  size_t operator()(const MapKey& k) const {
    return std::hash<uintptr_t>()(k.ptr) ^ (std::hash<int64_t>()(k.blocks) * 31) ^ (std::hash<int64_t>()(k.stride_block) * 17);
  }
};

// 128-byte-swizzled boxes of 64 tokens x 64 dims over the layer's cache [blocks][KV heads][PAGE][K | V]
static CUtensorMap kv_tensor_map(const at::Tensor& kv) {
  static std::mutex mu;
  static std::unordered_map<MapKey, CUtensorMap, MapKeyHash> cache;
  const MapKey key{reinterpret_cast<uintptr_t>(kv.data_ptr()), kv.size(0), kv.size(1), kv.stride(0), kv.stride(1), kv.stride(2)};
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map;
  cuuint64_t dims[4] = {(cuuint64_t)(2 * HD), (cuuint64_t)PAGE, (cuuint64_t)kv.size(1), (cuuint64_t)kv.size(0)};
  cuuint64_t strides[3] = {(cuuint64_t)kv.stride(2) * 2, (cuuint64_t)kv.stride(1) * 2, (cuuint64_t)kv.stride(0) * 2};
  cuuint32_t box[4] = {BOX_COLS, PAGE, 1, 1};
  cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  const CUresult res = encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, kv.data_ptr(), dims, strides, box, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)res);
  cache.emplace(key, map);
  return map;
}

// q, out: [requests x T, query heads, HD] bf16; kv: [blocks, KV heads, PAGE, 2 HD] bf16 (K | V per token)
void verify_attention(const at::Tensor& q, const at::Tensor& key, const at::Tensor& value, const at::Tensor& slot_mapping,
                      at::Tensor& kv, const at::Tensor& block_table, const at::Tensor& seq_lens, at::Tensor& out,
                      at::Tensor& part_o, at::Tensor& part_ml, int64_t num_requests, int64_t query_len, double scale,
                      int64_t ctas) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == HD && q.stride(2) == 1, "q [tokens, heads, 256]");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "q rows 16-byte aligned (bulk copies)");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == HD && out.stride(2) == 1 && out.size(1) == q.size(1), "out [tokens, heads, 256]");
  TORCH_CHECK(kv.dim() == 4 && kv.size(2) == PAGE && kv.size(3) == 2 * HD && kv.stride(3) == 1, "kv [blocks, heads, 64, 512]");
  for (const at::Tensor* t : {&key, &value}) {
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 3 && t->size(1) == kv.size(1) && t->size(2) == HD &&
                    t->stride(2) == 1 && t->stride(0) % 8 == 0 && t->stride(1) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "key, value [tokens, KV heads, 256], rows 16-byte aligned");
    TORCH_CHECK(t->size(0) >= num_requests * query_len, "key, value rows");
  }
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong && slot_mapping.numel() >= num_requests * query_len, "slot_mapping");
  TORCH_CHECK(kv.stride(0) % 8 == 0 && kv.stride(1) % 8 == 0 && kv.stride(2) % 8 == 0, "kv rows 16-byte aligned");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1, "block_table int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens int32");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_ml.scalar_type() == at::kFloat);
  const int64_t kv_heads = kv.size(1), group = q.size(1) / kv_heads;
  TORCH_CHECK(q.size(1) == group * kv_heads && group * query_len <= MAX_ROWS, "GQA group x T <= 48");
  TORCH_CHECK(num_requests >= 1 && num_requests <= MAX_REQUESTS && num_requests * kv_heads <= MAX_SEGMENTS, "requests");
  TORCH_CHECK(q.size(0) >= num_requests * query_len && out.size(0) >= num_requests * query_len, "query rows");
  TORCH_CHECK(ctas >= 1 && part_o.numel() >= (ctas + num_requests * kv_heads) * MAX_ROWS * HD &&
                  part_ml.numel() >= (ctas + num_requests * kv_heads) * MAX_ROWS * 2,
              "workspace");
  const at::cuda::CUDAGuard guard(q.device());
  const CUtensorMap map = kv_tensor_map(kv);
  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.q_tok = q.stride(0);
  p.q_head = q.stride(1);
  p.key = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
  p.value = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
  p.k_tok = key.stride(0);
  p.k_head = key.stride(1);
  p.v_tok = value.stride(0);
  p.v_head = value.stride(1);
  p.slot_mapping = slot_mapping.data_ptr<int64_t>();
  p.cache = reinterpret_cast<__nv_bfloat16*>(kv.data_ptr());
  p.c_block = kv.stride(0);
  p.c_head = kv.stride(1);
  p.c_token = kv.stride(2);
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.o_tok = out.stride(0);
  p.o_head = out.stride(1);
  p.block_table = block_table.data_ptr<int>();
  p.bt_stride = block_table.stride(0);
  p.seq_lens = seq_lens.data_ptr<int>();
  p.part_o = part_o.data_ptr<float>();
  p.part_ml = part_ml.data_ptr<float>();
  p.n = (int)num_requests;
  p.T = (int)query_len;
  p.group = (int)group;
  p.kv_heads = (int)kv_heads;
  p.piece_units = 1 + (int)query_len / PIECE_START_TOKENS_PER_PAGE;
  p.scale_log2 = (float)(scale * LOG2_E);
  const size_t smem = sizeof(Smem) + 1024;
  static bool is_configured = false;
  if (!is_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(verify_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    is_configured = true;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  verify_attn_kernel<<<(int)ctas, THREADS, smem, stream>>>(map, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // the merge, launched as the attention kernel's programmatic dependent: its CTAs start as SMs free up
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3((unsigned)(MERGE_CTAS_PER_SM * ctas));
  config.blockDim = dim3(MERGE_THREADS);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, merge_kernel, p, (int)ctas));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def(
      "verify_attention(Tensor q, Tensor key, Tensor value, Tensor slot_mapping, Tensor(a!) kv, Tensor block_table, "
      "Tensor seq_lens, Tensor(b!) out, Tensor(c!) part_o, Tensor(d!) part_ml, int num_requests, int query_len, "
      "float scale, int ctas) -> ()");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("verify_attention", &verify_attention); }
