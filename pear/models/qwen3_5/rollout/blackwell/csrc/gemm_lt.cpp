// bf16 GEMMs out = x @ w^T through cuBLASLt, per shape with the heuristic candidate lt_set chose (candidate 0 if none).
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

using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t>;  // device, M, N, K, ldx

// descriptors and heuristic results cached per shape, one persistent workspace per device: no allocation per call
// (CUDA-graph safe)
std::mutex g_mutex;
std::map<Key, Plan> g_plans;
std::map<std::tuple<int64_t, int64_t, int64_t>, int> g_choice;  // (M, N, K) -> candidate index
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
Plan& plan(int dev, int64_t M, int64_t N, int64_t K, int64_t ldx) {
  Key key{dev, M, N, K, ldx};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, M, ldx));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N));
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
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "lt_mm: x [M, K], w [N, K]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "lt_mm: bf16");
  TORCH_CHECK(w.is_contiguous() && w.size(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) == 0,
              "lt_mm: w row-major, aligned");
  if (x.stride(1) != 1 || x.stride(0) % 8 != 0 || x.stride(0) < x.size(1) ||
      (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) != 0)
    return x.contiguous();
  return x;
}

cublasStatus_t run(const Plan& p, int idx, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                   int dev, cudaStream_t stream) {
  float alpha = 1.f, beta = 0.f;
  auto& ws = workspace(dev, x);
  return cublasLtMatmul(handle(dev), p.desc, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &beta, out.data_ptr(),
                        p.lc, out.data_ptr(), p.lc, &p.algos[idx].algo, ws.data_ptr(), kWorkspace, stream);
}

torch::Tensor lt_mm(torch::Tensor x_in, torch::Tensor w) {
  torch::Tensor x = operand(x_in, w);
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto out = torch::empty({M, N}, x.options().dtype(torch::kBFloat16));
  if (M == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int dev = x.get_device();
  std::lock_guard<std::mutex> lock(g_mutex);
  Plan& p = plan(dev, M, N, K, x.stride(0));
  int idx = 0;
  auto c = g_choice.find({M, N, K});
  if (c != g_choice.end() && c->second < (int)p.algos.size()) idx = c->second;
  LT_CHECK(run(p, idx, x, w, out, dev, at::cuda::getCurrentCUDAStream()));
  return out;
}

void lt_set(int64_t M, int64_t N, int64_t K, int64_t idx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_choice[{M, N, K}] = (int)idx;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pear_rollout_blackwell, m) {
  m.def("lt_mm(Tensor x, Tensor w) -> Tensor");
  m.def("lt_set(int M, int N, int K, int idx) -> ()", &lt_set);
}

TORCH_LIBRARY_IMPL(pear_rollout_blackwell, CUDA, m) { m.impl("lt_mm", &lt_mm); }
