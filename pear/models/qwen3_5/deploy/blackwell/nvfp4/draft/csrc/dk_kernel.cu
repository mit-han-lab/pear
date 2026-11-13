// The draft-proposal megakernel: one persistent CTA per SM walks its instruction list (schedule.py); the warp roles
// (dk_layout.cuh) each replay the list, synchronized by mbarriers within the CTA and by counters across CTAs. Counter
// targets scale with the launch epoch, which the last step of every launch advances: nothing is reset between launches.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dk_roles.cuh"

namespace dk {

// ------------------------------------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(NTHREADS, 1) dk_main(const __grid_constant__ DkArgs a_param) {
  __shared__ DkArgs a_sh;  // the arguments in smem: referenced throughout by every role
  for (uint32_t i = threadIdx.x; i < sizeof(DkArgs) / 8; i += NTHREADS)
    reinterpret_cast<uint64_t*>(&a_sh)[i] = reinterpret_cast<const uint64_t*>(&a_param)[i];
  extern __shared__ __align__(SMEM_ALIGN) uint8_t smem_raw[];
  constexpr int ACC_BARS = 4;  // the accumulator buffers' dfull, then their dempty
  __shared__ uint64_t bars[2 * NSTAGE + ACC_BARS];  // the ring's full / empty, the accumulators' dfull / dempty
  __shared__ uint32_t tmem_slot, irange[2];
  __shared__ volatile uint32_t epi_rel;  // producer gate: 1 + the last instruction whose epilogue released its count
  __shared__ DraftShared dsh;
  __syncthreads();
  const DkArgs& a = a_sh;
  // aligned by pointer arithmetic (an integer round trip would hide the shared space: generic LD / ST everywhere)
  uint8_t* base = smem_raw + ((0u - smem_u32(smem_raw)) & (SMEM_ALIGN - 1));
  Smem sm;
  dsh.bind(sm);
  sm.stages = base;
  sm.bbuf = base + NSTAGE * STAGE_STRIDE;
  sm.epi_rel = &epi_rel;
  sm.full = bars;
  sm.empty = bars + NSTAGE;
  sm.dfull = bars + 2 * NSTAGE;
  sm.dempty = bars + 2 * NSTAGE + N_ACC;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // profiling: the CTA's entry, its roles joined and its exit (R_SF's last stamp slot)
  constexpr int TS_ENTRY = 1;   // the CTA's entry
  constexpr int TS_JOIN = 2;    // its roles joined
  constexpr int TS_EXIT = 3;    // its exit
  constexpr int TS_CLOCK0 = 4;  // (not a time: the SM clock at the entry, over the CTA)
  constexpr int TS_CLOCK1 = 5;  // (and at the exit)
  const Trace frame{a.progress + blockIdx.x * NROLE + R_SF, 0,
                    a.tstamp ? a.tstamp + (((size_t)blockIdx.x * NROLE + R_SF) * (MAX_INSTR + 1) + MAX_INSTR) * NSUB : nullptr};
  if (threadIdx.x == 0) frame.stamp(TS_ENTRY), frame.put(TS_CLOCK0, clock64());
  // the launch's epoch; CTA 0 advances it at the end once every CTA has read it
  const uint32_t epoch = ld_acquire(a.epoch);
  if (threadIdx.x == 0) {
    red_relaxed_add(a.epoch + 1, 1);
    dsh.init();
    for (int s = 0; s < NSTAGE; s++) {
      mbar_init(&sm.full[s], 1);
      mbar_init(&sm.empty[s], 1);
    }
    for (int b = 0; b < N_ACC; b++) {
      mbar_init(&sm.dfull[b], 1);
      mbar_init(&sm.dempty[b], WG_WARPS);
    }
    epi_rel = 0;
    irange[0] = a.sm_off[blockIdx.x];
    irange[1] = a.sm_off[blockIdx.x + 1];
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == W_MMA) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "n"(T_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  sm.tmem = tmem_slot;
  const uint32_t i0 = irange[0], i1 = irange[1];
  if (warp < W_EPI0) {  // warpgroup 0: the convs' base taps into TMEM
    set_max_regs<REG_IDLE>();
    BaseRole role;
#pragma unroll 1
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      role.instr(in, a, sm, lane);
    }
  } else if (warp >= W_PROD && warp < W_WRK0) {  // warpgroup 2: the TMA and MMA roles
    set_max_regs<REG_LO>();
    RingPos cc, seg;
    if (warp == W_PROD) {
      ProdRole role;
      for (uint32_t ii = i0; ii < i1; ii++) {
        const Instr in = a.instrs[ii];
        role.instr(in, ii, i0, a, sm, cc, lane);
      }
      if (lane == 0) mark(a, R_PROD, i1, i0);
    } else if (warp == W_MMA) {
      MmaRole role;
      for (uint32_t ii = i0; ii < i1; ii++) {
        const Instr in = a.instrs[ii];
        role.instr(in, ii, i0, a, sm, cc, seg, lane);
      }
      if (lane == 0) mark(a, R_MMA, i1, i0);
    }  // (warps 10, 11: idle)
  } else if (warp < W_PROD) {  // warpgroup 1: the epilogue
    set_max_regs<REG_EPI>();
    const int ept = threadIdx.x - W_EPI0 * 32;
    RingPos seg;
    EpiRole role;
    role.begin(a, sm, ept, epoch);
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      role.instr(in, ii, i0, a, sm, seg, ept, epoch);
    }
    if (ept == 0) mark(a, R_EPI, i1, i0);
  } else {  // warpgroups 3-4: the workers
    set_max_regs<REG_HI>();
    const int pt = threadIdx.x - W_WRK0 * 32;
    RingPos cc;
    WrkRole role;
    for (uint32_t ii = i0; ii < i1; ii++) {
      const Instr in = a.instrs[ii];
      role.instr(in, ii, i0, i1, a, sm, cc, pt, epoch);
    }
    if (pt == 0) mark(a, R_WRK, i1, i0);
  }
  tc_fence_before();
  __syncthreads();
  if (threadIdx.x == 0) frame.stamp(TS_JOIN);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (ctr_before(ld_relaxed(a.epoch + 1), (epoch + 1) * gridDim.x)) {
    }
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(a.epoch), "r"(epoch + 1) : "memory");
  }
  if (warp == W_MMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(sm.tmem), "n"(T_COLS));
  }
  if (threadIdx.x == 0) frame.stamp(TS_EXIT), frame.put(TS_CLOCK1, clock64());
}

}  // namespace dk

// ------------------------------------------------------------------------------------------------ host
namespace {

// the launch arguments from the schedule's tensors and the step's pointers (int64 addresses)
dk::DkArgs make_args(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters,
                     torch::Tensor ctr_count, torch::Tensor epoch, torch::Tensor ssq, torch::Tensor step, torch::Tensor tab,
                     torch::Tensor progress, torch::Tensor tstamp) {
  dk::DkArgs a{};
  a.instrs = reinterpret_cast<const dk::Instr*>(instrs.data_ptr());
  a.sm_off = reinterpret_cast<const uint32_t*>(sm_off.data_ptr());
  a.descs = reinterpret_cast<const dk::GemvDesc*>(descs.data_ptr());
  a.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  a.ctr_count = reinterpret_cast<const uint32_t*>(ctr_count.data_ptr());
  a.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
  a.ssq_zero = reinterpret_cast<long long*>(ssq.data_ptr());
  a.n_ssq = (uint32_t)(ssq.numel() / (2 * dk::QB));
  a.progress = reinterpret_cast<uint32_t*>(progress.data_ptr());
  a.tstamp = tstamp.numel() ? reinterpret_cast<uint64_t*>(tstamp.data_ptr()) : nullptr;
  TORCH_CHECK(step.device().is_cpu() && step.numel() * 8 == (int64_t)sizeof(dk::StepArgs), "StepArgs layout");
  TORCH_CHECK(tab.device().is_cpu() && tab.numel() * 8 == (int64_t)sizeof(dk::Tables), "Tables layout");
  memcpy(&a.step, step.data_ptr(), sizeof(dk::StepArgs));
  memcpy(&a.tab, tab.data_ptr(), sizeof(dk::Tables));
  return a;
}

}  // namespace

// One proposal: the schedule's launch over nsm CTAs.
void dk_run(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters, torch::Tensor ctr_count,
            torch::Tensor epoch, torch::Tensor ssq, torch::Tensor step, torch::Tensor tab, torch::Tensor progress, torch::Tensor tstamp, int64_t nsm) {
  const at::cuda::CUDAGuard guard(instrs.device());
  static bool is_smem_attr_set = false;
  if (!is_smem_attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(dk::dk_main, cudaFuncAttributeMaxDynamicSharedMemorySize, dk::SMEM_BYTES) == cudaSuccess,
                "shared memory");
    is_smem_attr_set = true;
  }
  const dk::DkArgs a = make_args(instrs, sm_off, descs, counters, ctr_count, epoch, ssq, step, tab, progress, tstamp);
  dk::dk_main<<<(int)nsm, dk::NTHREADS, dk::SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// TMA descriptor over a row-major byte matrix w[rows][row_bytes] for the MMA's A tiles: uint8 elements, box 128 bytes
// x box_rows rows, 128-byte swizzle; returned as a CPU uint8 tensor [128] to be copied into device memory.
torch::Tensor dk_make_tmap(torch::Tensor w, int64_t rows, int64_t row_bytes, int64_t box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
  if (!enc) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &q) ==
                        cudaSuccess &&
                    q == cudaDriverEntryPointSuccess,
                "cuTensorMapEncodeTiled unavailable");
  }
  constexpr int TMAP_BYTES = 128;  // an encoded CUtensorMap
  constexpr int BOX_BYTES = 128;   // a box row: the 128-byte swizzle's span
  constexpr int TMA_ALIGN = 16;    // the global address alignment TMA requires
  TORCH_CHECK(row_bytes % BOX_BYTES == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % TMA_ALIGN == 0,
              "shape/alignment");
  auto out = torch::zeros({TMAP_BYTES}, torch::kUInt8);
  CUtensorMap m;
  cuuint64_t gdim[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {BOX_BYTES, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, w.data_ptr(), gdim, gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  memcpy(out.data_ptr<uint8_t>(), &m, TMAP_BYTES);
  return out;
}

// the sizes of the packed host structs (schedule.py checks its layouts against them), the shared memory, the B buffer,
// the stage ring
std::vector<int64_t> dk_sizes() {
  return {(int64_t)sizeof(dk::Instr), (int64_t)sizeof(dk::GemvDesc), (int64_t)sizeof(dk::StepArgs), (int64_t)sizeof(dk::Tables),
          (int64_t)dk::SMEM_BYTES, (int64_t)dk::BBUF_BYTES, (int64_t)dk::NSTAGE * dk::STAGE_STRIDE, (int64_t)sizeof(dk::DkArgs)};
}

TORCH_LIBRARY(pear_deploy_blackwell_nvfp4_draft, m) {
  m.def("run(Tensor instrs, Tensor sm_off, Tensor descs, Tensor(a!) counters, Tensor ctr_count, Tensor(b!) epoch, "
        "Tensor(c!) ssq, Tensor step, Tensor tab, Tensor(d!) progress, Tensor(e!) tstamp, int nsm) -> ()");
  m.def("make_tmap(Tensor w, int rows, int row_bytes, int box_rows) -> Tensor");
  m.def("sizes() -> int[]");
}
TORCH_LIBRARY_IMPL(pear_deploy_blackwell_nvfp4_draft, CUDA, m) { m.impl("run", &dk_run); }
TORCH_LIBRARY_IMPL(pear_deploy_blackwell_nvfp4_draft, CompositeExplicitAutograd, m) {
  m.impl("make_tmap", &dk_make_tmap);
}
TORCH_LIBRARY_IMPL(pear_deploy_blackwell_nvfp4_draft, CatchAll, m) { m.impl("sizes", &dk_sizes); }
