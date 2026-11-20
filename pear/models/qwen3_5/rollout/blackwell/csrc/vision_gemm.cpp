// The vision tower's fp32 GEMMs out = x @ w^T (+ bias, + tanh GELU) through cuBLASLt with the epilogue fused: on the
// SIMT cores, or emulated on the bf16 tensor cores (BF16x9: each fp32 operand as three bf16 parts, the nine products
// exact, summed in fp32), per shape with the heuristic's first candidate.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <map>
#include <mutex>
#include <optional>
#include <tuple>

namespace vision_gemm {

#define VG_CHECK(x)                                                                                    \
  do {                                                                                                 \
    cublasStatus_t s_ = (x);                                                                           \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "vision_mm: cuBLASLt error ", int(s_), " at line ", __LINE__); \
  } while (0)

constexpr int64_t kWorkspace = 32ll << 20;  // cuBLASLt's workspace per device, above what the tower's algorithms take

enum Epilogue : int64_t { kNone = 0, kBias = 1, kGeluBias = 2 };  // vision_mm's epilogue argument

struct Plan {
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulAlgo_t algo;
};

using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, bool>;  // device, M, N, K, epilogue, emulated

// descriptors and the chosen algorithm cached per shape, one persistent workspace per device: no allocation per call
// (CUDA-graph safe)
std::mutex g_mutex;
std::map<Key, Plan> g_plans;
std::map<int, cublasLtHandle_t> g_handles;
std::map<int, torch::Tensor> g_workspaces;
cublasLtEmulationDesc_t g_emulation = nullptr;

cublasLtHandle_t handle(int dev) {
  auto it = g_handles.find(dev);
  if (it != g_handles.end()) return it->second;
  cublasLtHandle_t h;
  VG_CHECK(cublasLtCreate(&h));
  g_handles[dev] = h;
  return h;
}

torch::Tensor& workspace(int dev, const torch::Tensor& like) {
  auto it = g_workspaces.find(dev);
  if (it == g_workspaces.end()) {
    it = g_workspaces.emplace(dev, torch::empty({kWorkspace}, like.options().dtype(torch::kUInt8))).first;
  }
  return it->second;
}

// BF16x9 on every call, not only where cuBLASLt's heuristics expect it to be faster; no inf/NaN scan and patch kernels
// per call: the tower's activations are finite (a non-finite input poisons its row through the next LayerNorm, with or
// without the patch, which only turns some NaN outputs back into the infinities an fp32 product would give)
cublasLtEmulationDesc_t emulation() {
  if (g_emulation == nullptr) {
    VG_CHECK(cublasLtEmulationDescCreate(&g_emulation));
    cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_EAGER;
    VG_CHECK(cublasLtEmulationDescSetAttribute(g_emulation, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy,
                                               sizeof(strategy)));
    int32_t special_values = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE;
    VG_CHECK(cublasLtEmulationDescSetAttribute(g_emulation, CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT,
                                               &special_values, sizeof(special_values)));
  }
  return g_emulation;
}

// column-major view: out^T [N, M] = (w [N, K] row-major = K x N col-major, transposed) * (x^T: K x M); the bias is
// per row of out^T
Plan& plan(int dev, int64_t M, int64_t N, int64_t K, int64_t epilogue, bool emulated) {
  Key key{dev, M, N, K, epilogue, emulated};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan p;
  VG_CHECK(cublasLtMatmulDescCreate(&p.desc, emulated ? CUBLAS_COMPUTE_32F_EMULATED_16BFX9 : CUBLAS_COMPUTE_32F,
                                    CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  if (emulated) {
    cublasLtEmulationDesc_t e = emulation();
    VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &e, sizeof(e)));
  }
  if (epilogue != kNone) {
    cublasLtEpilogue_t e = epilogue == kGeluBias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
    VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    cudaDataType_t bias_type = CUDA_R_32F;
    VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                            sizeof(bias_type)));
  }
  VG_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_32F, K, N, K));
  VG_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_32F, K, M, K));
  VG_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, N, M, N));
  cublasLtMatmulPreference_t pref;
  VG_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  int64_t ws = kWorkspace;
  VG_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t result;
  int found = 0;
  VG_CHECK(cublasLtMatmulAlgoGetHeuristic(handle(dev), p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, &result, &found));
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(found > 0, "vision_mm: no cuBLASLt algorithm for ", M, "x", N, "x", K);
  p.algo = result.algo;
  return g_plans.emplace(key, p).first->second;
}

}  // namespace vision_gemm

// x: [M, K] fp32 with contiguous rows, w: [N, K] fp32 contiguous, bias: [N] fp32 (epilogue 1: + bias, 2: tanh
// GELU(+ bias)). Returns [M, N] fp32.
torch::Tensor vision_mm(torch::Tensor x, torch::Tensor w, std::optional<torch::Tensor> bias, int64_t epilogue,
                        bool emulated) {
  using namespace vision_gemm;
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && x.is_contiguous() && w.is_contiguous());
  TORCH_CHECK(epilogue == kNone ||
              (bias.has_value() && bias->scalar_type() == at::kFloat && bias->numel() == w.size(0)));
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto out = torch::empty({M, N}, x.options());
  if (M == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int dev = x.get_device();
  std::lock_guard<std::mutex> lock(g_mutex);
  Plan& p = plan(dev, M, N, K, epilogue, emulated);
  if (epilogue != kNone) {
    const void* bias_ptr = bias->data_ptr();
    VG_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }
  const float alpha = 1.f, beta = 0.f;
  VG_CHECK(cublasLtMatmul(handle(dev), p.desc, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &beta, out.data_ptr(),
                          p.lc, out.data_ptr(), p.lc, &p.algo, workspace(dev, x).data_ptr(), kWorkspace,
                          at::cuda::getCurrentCUDAStream()));
  return out;
}

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("vision_mm(Tensor x, Tensor w, Tensor? bias, int epilogue, bool emulated) -> Tensor");
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("vision_mm", &vision_mm); }
