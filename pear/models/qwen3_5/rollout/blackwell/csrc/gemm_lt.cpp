// bf16 GEMMs out = x @ w^T through cuBLASLt, per shape with the heuristic candidate its table chose (candidate 0 if
// none): the target model's table (lt_mm, bf16 out) and the drafter's (dg_mm, bf16 or fp32 out), whose shapes the
// target's share at other batch sizes.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

#define LT_CHECK(x)                                                                        \
  do {                                                                                     \
    cublasStatus_t s_ = (x);                                                               \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt error ", int(s_), " at line ", __LINE__); \
  } while (0)

constexpr int64_t kWorkspace = 32ll << 20;  // cuBLASLt's workspace per device
constexpr int kCandidates = 16;  // heuristic candidates kept per shape

struct Plan {
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t la, lb, lc;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
};

using Shape = std::tuple<int64_t, int64_t, int64_t, bool>;  // M, N, K, fp32 out
using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, bool>;  // device, M, N, K, ldx, fp32 out

// descriptors and heuristic results cached per shape, one persistent workspace per device: no allocation per call
// (CUDA-graph safe)
std::mutex g_mutex;
std::map<Key, Plan> g_plans;
std::map<Shape, int> g_target_choice;  // shape -> candidate index, the target model's table
std::map<Shape, int> g_draft_choice;   // the drafter's table
std::map<int, cublasLtHandle_t> g_handles;
std::map<int, torch::Tensor> g_workspaces;

cublasLtHandle_t handle(int dev) {
  auto it = g_handles.find(dev);
  if (it != g_handles.end()) return it->second;
  cublasLtHandle_t h;
  LT_CHECK(cublasLtCreate(&h));
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

// column-major view: out^T [N, M] = (w [N, K] row-major = K x N col-major, transposed) * (x^T: K x M, ld = ldx)
Plan& plan(int dev, int64_t M, int64_t N, int64_t K, int64_t ldx, bool is_fp32) {
  Key key{dev, M, N, K, ldx, is_fp32};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, M, ldx));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, is_fp32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  int64_t ws = kWorkspace;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  p.algos.resize(kCandidates);
  int found = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle(dev), p.desc, p.la, p.lb, p.lc, p.lc, pref, kCandidates,
                                          p.algos.data(), &found));
  cublasLtMatmulPreferenceDestroy(pref);
  p.algos.resize(found);
  TORCH_CHECK(found > 0, "cuBLASLt: no algorithm for ", M, "x", N, "x", K);
  return g_plans.emplace(key, std::move(p)).first->second;
}

// x as cuBLASLt takes it: unit column stride, 16-byte aligned rows (a copy otherwise)
torch::Tensor operand(const torch::Tensor& x, const torch::Tensor& w) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "gemm_lt: x [M, K], w [N, K]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "gemm_lt: bf16");
  TORCH_CHECK(w.is_contiguous() && w.size(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) == 0,
              "gemm_lt: w row-major, aligned");
  if (x.stride(1) != 1 || x.stride(0) % 8 != 0 || x.stride(0) < x.size(1) ||
      (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) != 0)
    return x.contiguous();
  return x;
}

// x @ w^T on the candidate the table chose for its shape
torch::Tensor matmul(const std::map<Shape, int>& choices, const torch::Tensor& x_in, const torch::Tensor& w,
                     bool is_fp32) {
  torch::Tensor x = operand(x_in, w);
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto out = torch::empty({M, N}, x.options().dtype(is_fp32 ? torch::kFloat32 : torch::kBFloat16));
  if (M == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int dev = x.get_device();
  std::lock_guard<std::mutex> lock(g_mutex);
  Plan& p = plan(dev, M, N, K, x.stride(0), is_fp32);
  int idx = 0;
  auto c = choices.find({M, N, K, is_fp32});
  if (c != choices.end() && c->second < (int)p.algos.size()) idx = c->second;
  float alpha = 1.f, beta = 0.f;
  auto& ws = workspace(dev, x);
  LT_CHECK(cublasLtMatmul(handle(dev), p.desc, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &beta, out.data_ptr(),
                          p.lc, out.data_ptr(), p.lc, &p.algos[idx].algo, ws.data_ptr(), kWorkspace,
                          at::cuda::getCurrentCUDAStream()));
  return out;
}

torch::Tensor lt_mm(torch::Tensor x, torch::Tensor w) { return matmul(g_target_choice, x, w, false); }

void lt_set(int64_t M, int64_t N, int64_t K, int64_t idx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_target_choice[{M, N, K, false}] = (int)idx;
}

torch::Tensor dg_mm(torch::Tensor x, torch::Tensor w, bool is_fp32) { return matmul(g_draft_choice, x, w, is_fp32); }

void dg_set(int64_t M, int64_t N, int64_t K, bool is_fp32, int64_t idx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_draft_choice[{M, N, K, is_fp32}] = (int)idx;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("lt_mm(Tensor x, Tensor w) -> Tensor");
  m.def("lt_set(int M, int N, int K, int idx) -> ()", &lt_set);
  m.def("dg_mm(Tensor x, Tensor w, bool is_fp32) -> Tensor");
  m.def("dg_set(int M, int N, int K, bool is_fp32, int idx) -> ()", &dg_set);
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) {
  m.impl("lt_mm", &lt_mm);
  m.impl("dg_mm", &dg_mm);
}
