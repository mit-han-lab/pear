// The decode megakernel on sm_120a: its torch ops (megakernel.cuh: the layouts, kernel.cuh: the device code).
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.cuh"

// ------------------------------------------------------------------------------------------------ host
namespace {

constexpr int TMAP_BYTES = 128;  // sizeof(CUtensorMap)
constexpr cuuint32_t BOX_BYTES = 128;  // a box row: the SWIZZLE_128B span
constexpr int64_t SF_ATOM_ROWS = 128;  // weight rows per NVFP4 scale atom
constexpr int64_t SF_ATOM_K = 64;  // K values per NVFP4 scale atom
constexpr int64_t NVFP4_BLOCK = 16;  // values per NVFP4 scale
constexpr uint32_t E2M1_X2_NIBBLES = 0xC8643210u;  // e2m1 magnitudes x 2 as nibbles {0, 1, 2, 3, 4, 6, 8, 12}
constexpr uint8_t E2M1_MAG_MASK = 7;  // an e2m1 code's magnitude bits
constexpr uint8_t E2M1_SIGN = 8;  // the low code's sign bit
constexpr uint8_t E2M1_SIGN_HI = 128;  // the high code's sign bit
constexpr int DEQUANT_THREADS = 256;
constexpr int64_t DEQUANT_MAX_BLOCKS = 170 * 16;  // 16 CTAs on each of the RTX 5090's 170 SMs
constexpr uint32_t CLOCK_POLLS_MASK = 255u;  // wait_flag reads the clock once per 256 polls

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "cuTensorMapEncodeTiled");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// a CUtensorMap (128 bytes) over a uint8 matrix [rows][row_bytes] at addr: box [box_rows][128 B], SWIZZLE_128B
torch::Tensor make_tmap(int64_t addr, int64_t rows, int64_t row_bytes, int64_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {BOX_BYTES, (cuuint32_t)box_rows}, es[2] = {1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, reinterpret_cast<void*>(addr), dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  auto out = torch::empty({TMAP_BYTES}, torch::dtype(torch::kUInt8));
  std::memcpy(out.data_ptr(), &m, TMAP_BYTES);
  return out;
}

// a CUtensorMap over a KV cache viewed as uint8 [rows][heads][row_bytes] (rows: tokens of every kernel block): box
// {128 B, 1 head, box_rows}, SWIZZLE_128B (attention.cuh)
torch::Tensor make_tmap3d(int64_t addr, int64_t rows, int64_t heads, int64_t row_bytes, int64_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[3] = {(cuuint64_t)row_bytes, (cuuint64_t)heads, (cuuint64_t)rows};
  cuuint64_t strides[2] = {(cuuint64_t)row_bytes, (cuuint64_t)(row_bytes * heads)};
  cuuint32_t box[3] = {BOX_BYTES, 1, (cuuint32_t)box_rows}, es[3] = {1, 1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, reinterpret_cast<void*>(addr), dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  auto out = torch::empty({TMAP_BYTES}, torch::dtype(torch::kUInt8));
  std::memcpy(out.data_ptr(), &m, TMAP_BYTES);
  return out;
}

template <int V>
void run_v(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters,
           torch::Tensor ctr_count, torch::Tensor epoch, int64_t nsm, torch::Tensor rec_host, int64_t trace) {
  const at::cuda::CUDAGuard guard(instrs.device());
  TORCH_CHECK(rec_host.device().is_cpu() && rec_host.dtype() == torch::kInt32 && rec_host.numel() >= mk::SR_BT,
              "the record's header: int32 [>= SR_BT] on the host");
  mk::MKArgs a;
  a.instrs = reinterpret_cast<const mk::Instr*>(instrs.data_ptr());
  a.sm_off = reinterpret_cast<const uint32_t*>(sm_off.data_ptr());
  a.descs = reinterpret_cast<const mk::OpDesc*>(descs.data_ptr());
  a.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  a.ctr_count = reinterpret_cast<const uint32_t*>(ctr_count.data_ptr());
  a.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
  a.trace = reinterpret_cast<uint64_t*>(trace);
  std::memcpy(a.rec.w, rec_host.data_ptr(), sizeof(a.rec.w));
  static bool is_smem_attr_set = false;
  const int smem = mk::SMEM_BYTES + mk::SMEM_ALIGN;
  if (!is_smem_attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(mk::mk_main<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
    is_smem_attr_set = true;
  }
  mk::mk_main<V><<<(unsigned)nsm, mk::NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// the verify step's kernel and the draft's (mk_main<V>)
void run(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters,
         torch::Tensor ctr_count, torch::Tensor epoch, int64_t nsm, torch::Tensor rec_host, int64_t trace) {
  run_v<0>(instrs, sm_off, descs, counters, ctr_count, epoch, nsm, rec_host, trace);
}
void run_draft(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters,
               torch::Tensor ctr_count, torch::Tensor epoch, int64_t nsm, torch::Tensor rec_host, int64_t trace) {
  run_v<1>(instrs, sm_off, descs, counters, ctr_count, epoch, nsm, rec_host, trace);
}
// the verify step with the draft's proposal in the same launch (mk_main<2>: the fused schedule, step.py / draft.py)
void run_fused(torch::Tensor instrs, torch::Tensor sm_off, torch::Tensor descs, torch::Tensor counters,
               torch::Tensor ctr_count, torch::Tensor epoch, int64_t nsm, torch::Tensor rec_host, int64_t trace) {
  run_v<2>(instrs, sm_off, descs, counters, ctr_count, epoch, nsm, rec_host, trace);
}

// The bf16 weights of an NVFP4 projection (codes [rows][K / 2], vLLM's swizzled e4m3 scale atoms, a power-of-two
// global scale gs) for vLLM's own draft paths: out[r][k] = bf16(e2m1(code) * (e4m3(scale) / gs)), these fp32
// operations in this order (the bf16 weights the draft's outputs are exact against)
__global__ void k_nvfp4_dequant(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ sf, float gs,
                                __nv_bfloat16* __restrict__ out, int64_t rows, int64_t K) {
  const int64_t half = K / 2, n = rows * half, D = K / SF_ATOM_K;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = i / half, k = 2 * (i - r * half), blk = k / NVFP4_BLOCK;
    const int64_t a = r / SF_ATOM_ROWS, rr = r % SF_ATOM_ROWS, b = rr / 32, c = rr % 32, d = blk / 4, e = blk % 4;
    __nv_fp8_e4m3 s8;
    s8.__x = sf[((((a * D + d) * 32 + c) * 4 + b) * 4) + e];
    const float s = float(s8) / gs;
    const uint8_t cb = codes[i];
    // the magnitudes from a table in a register (no local memory)
    const float m0 = 0.5f * ((E2M1_X2_NIBBLES >> (4 * (cb & E2M1_MAG_MASK))) & 15);
    const float m1 = 0.5f * ((E2M1_X2_NIBBLES >> (4 * ((cb >> 4) & E2M1_MAG_MASK))) & 15);
    const float v0 = (cb & E2M1_SIGN) ? -m0 : m0, v1 = (cb & E2M1_SIGN_HI) ? -m1 : m1;
    out[r * K + k] = __float2bfloat16_rn(v0 * s);
    out[r * K + k + 1] = __float2bfloat16_rn(v1 * s);
  }
}

torch::Tensor nvfp4_dequant(torch::Tensor codes, torch::Tensor scales, double gs) {
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.dim() == 2 && codes.is_contiguous(), "codes");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.is_contiguous(), "scales");
  const int64_t rows = codes.size(0), K = 2 * codes.size(1);
  TORCH_CHECK(rows % SF_ATOM_ROWS == 0 && K % SF_ATOM_K == 0 && scales.numel() == rows * K / NVFP4_BLOCK,
              "NVFP4 shapes");
  const at::cuda::CUDAGuard guard(codes.device());
  auto out = torch::empty({rows, K}, codes.options().dtype(torch::kBFloat16));
  const int64_t n = rows * K / 2;
  const int threads = DEQUANT_THREADS, blocks = (int)std::min<int64_t>((n + threads - 1) / threads, DEQUANT_MAX_BLOCKS);
  k_nvfp4_dequant<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), (float)gs,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// (host) Waits until flags[slot] (pinned host memory, a host ring slot's flag OP_PREV writes once the slot is written)
// equals seq, by acquire loads (the slot's data read after it is the device's); false after timeout_us
bool wait_flag(torch::Tensor flags, int64_t slot, int64_t seq, int64_t timeout_us) {
  TORCH_CHECK(flags.is_pinned() && flags.scalar_type() == torch::kInt32 && slot >= 0 && slot < flags.numel(), "flags");
  const int32_t* p = flags.data_ptr<int32_t>() + slot;
  const int32_t target = (int32_t)seq;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0;; i++) {
    if (__atomic_load_n(p, __ATOMIC_ACQUIRE) == target) return true;
    if ((i & CLOCK_POLLS_MASK) == CLOCK_POLLS_MASK) {
      if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(timeout_us)) return false;
      std::this_thread::yield();
    }
  }
}

std::vector<int64_t> sizes() {
  return {mk::SMEM_BYTES, (int64_t)sizeof(mk::Instr), (int64_t)sizeof(mk::OpDesc), mk::B_CHUNKS, mk::NSTAGE};
}

}  // namespace

TORCH_LIBRARY(pear_deploy_rtx_blackwell_nvfp4, m) {
  m.def("make_tmap", &make_tmap);
  m.def("make_tmap3d", &make_tmap3d);
  m.def("run", &run);
  m.def("run_draft", &run_draft);
  m.def("run_fused", &run_fused);
  m.def("wait_flag", &wait_flag);
  m.def("sizes", &sizes);
  // (a schema and a CUDA kernel, not a CompositeImplicitAutograd function: torch.compile's fake for it registers)
  m.def("nvfp4_dequant(Tensor codes, Tensor scales, float gs) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_deploy_rtx_blackwell_nvfp4, CUDA, m) {
  m.impl("nvfp4_dequant", &nvfp4_dequant);
}
