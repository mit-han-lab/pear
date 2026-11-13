// Fused Gated DeltaNet decode step (one token per sequence) of Qwen3.5's linear-attention layers, per TP rank, in one
// kernel on the fused in_proj output [q k v z | b a]: the causal conv1d update, the gates, q/k L2 norm, the fp32
// delta-rule recurrence and the gated RMSNorm, with the rounding points of vLLM's sequence of kernels
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace gdn {

constexpr int DK = 128;  // key head dim
constexpr int DV = 128;  // value head dim
constexpr int NT = 128;  // threads per CTA: 4 warps x 32 rows
constexpr int MAX_TOKENS = 1024;
constexpr int MAX_KH = 16;

struct Params {
  const __nv_bfloat16* proj;  // [B, ...]: q[H*DK] k[H*DK] v[HV*DV] z[HV*DV] b[HV] a[HV]
  int64_t proj_stride;
  __nv_bfloat16* conv_state;  // [slots, 3, conv_dim] (row of 3 taps, dim contiguous)
  int64_t cs_slot_stride, cs_tok_stride;
  const __nv_bfloat16* conv_w;  // [conv_dim, 4]
  float* ssm;                   // [slots, HV, DV, DK]
  int64_t ss_slot_stride;
  const int32_t* state_idx;  // [B]
  const float* A_log;        // [HV]
  const void* dt_bias;       // [HV] (dt_bias_f32 ? fp32 : bf16)
  const void* norm_w;        // [DV] (norm_w_f32 ? fp32 : bf16)
  bool dt_bias_f32, norm_w_f32;
  __nv_bfloat16* out;  // [B, HV, DV]
  int64_t out_stride;
  int* counters;  // [MAX_TOKENS * MAX_KH], zero between launches
  int B, H, HV;
  float eps, scale;
};

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

// Triton's lowerings in vLLM's kernels (their PTX): tl.exp = ex2.approx of x * log2(e), tl.log = libdevice logf, fp32
// '/' and tl.sigmoid's = div.full, tl.sqrt = sqrt.approx.ftz, tl.rsqrt = rsqrt.approx.ftz
__device__ __forceinline__ float tl_exp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float sqrt_approx(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tl_sigmoid(float x) { return div_full(1.f, 1.f + tl_exp(0.f - x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// y = silu(sum_j bf16(s_j * w_j)) over the taps (s0, s1, s2, x), the sum in fp32 as vLLM's Triton kernel does
__device__ __forceinline__ float conv_tap(__nv_bfloat16 s0, __nv_bfloat16 s1, __nv_bfloat16 s2, __nv_bfloat16 xv,
                                          const __nv_bfloat16* w) {
  float acc = 0.f;
  acc += __bfloat162float(__hmul(s0, w[0]));
  acc += __bfloat162float(__hmul(s1, w[1]));
  acc += __bfloat162float(__hmul(s2, w[2]));
  acc += __bfloat162float(__hmul(xv, w[3]));
  return bf(div_full(acc, 1.f + tl_exp(0.f - acc)));
}

// one CTA per (value head, sequence), 4 warps: warp w holds state rows 32w..32w+31 in registers (lane l: columns
// 4l..4l+3), loaded with the gating inputs during the conv; the HVK value heads of a key head share its q/k taps: each
// reads the old ones, the last (threadfence, a workspace counter) writes the new ones after the output
template <int HVK>
__global__ void __launch_bounds__(NT, 3) gdn_decode_kernel(const Params p) {
  __shared__ float q_s[DK], k_s[DK], v_s[DV], o_s[DV];
  __shared__ float red[4];
  __shared__ int last_s;

  const int hv = blockIdx.x, b = blockIdx.y;
  const int kh = hv / HVK;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int slot = p.state_idx[b];
  __nv_bfloat16* outp = p.out + (int64_t)b * p.out_stride + hv * DV;
  if (slot <= 0) {  // padding (null block): zeros, states untouched
    outp[tid] = __float2bfloat16(0.f);
    return;
  }
  float* gstate = p.ssm + (int64_t)slot * p.ss_slot_stride + (int64_t)hv * DV * DK;
  // this warp's 32 state rows into registers first: their loads are in flight during the conv below
  float* h = gstate + (warp * 32) * DK + 4 * lane;
  float4 rows[32];
#pragma unroll
  for (int r = 0; r < 32; ++r) rows[r] = __ldcs(reinterpret_cast<const float4*>(h + r * DK));
  const __nv_bfloat16* x = p.proj + (int64_t)b * p.proj_stride;
  const int qk_dim = p.H * DK;
  // the gating inputs first (independent of the conv)
  const __nv_bfloat16 a_raw = x[2 * qk_dim + 2 * p.HV * DV + p.HV + hv];
  const __nv_bfloat16 b_raw = x[2 * qk_dim + 2 * p.HV * DV + hv];
  const float dtb = p.dt_bias_f32 ? reinterpret_cast<const float*>(p.dt_bias)[hv]
                                  : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p.dt_bias)[hv]);
  const float A_log_hv = p.A_log[hv];
  __nv_bfloat16* cs = p.conv_state + (int64_t)slot * p.cs_slot_stride;
  const int64_t t1 = p.cs_tok_stride, t2 = 2 * p.cs_tok_stride;

  // 1. conv: q and k channels of the key head (old taps only), v channels of this head (taps updated here)
  const int cq = kh * DK + tid, ck = qk_dim + kh * DK + tid, cv = 2 * qk_dim + hv * DV + tid;
  const __nv_bfloat16 xq = x[cq], xk = x[ck], xv = x[cv];
  const __nv_bfloat16 q0 = cs[cq], q1 = cs[t1 + cq], q2 = cs[t2 + cq];
  const __nv_bfloat16 k0 = cs[ck], k1 = cs[t1 + ck], k2 = cs[t2 + ck];
  const __nv_bfloat16 v0 = cs[cv], v1 = cs[t1 + cv], v2 = cs[t2 + cv];
  q_s[tid] = conv_tap(q0, q1, q2, xq, p.conv_w + (int64_t)cq * 4);
  k_s[tid] = conv_tap(k0, k1, k2, xk, p.conv_w + (int64_t)ck * 4);
  v_s[tid] = conv_tap(v0, v1, v2, xv, p.conv_w + (int64_t)cv * 4);
  cs[cv] = v1;
  cs[t1 + cv] = v2;
  cs[t2 + cv] = xv;
  __syncthreads();

  // 2. L2 norms of q (warp 0) and k (warp 1); gating
  if (warp < 2) {
    const float* src = warp ? k_s : q_s;
    float s = 0.f;
#pragma unroll
    for (int i = lane; i < DK; i += 32) s += src[i] * src[i];
    s = warp_sum(s);
    if (lane == 0) red[warp] = sqrt_approx(s + 1e-6f);
  }
  __syncthreads();
  const float qn = red[0], kn = red[1];
  float qv[4], kv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    qv[j] = div_full(q_s[4 * lane + j], qn) * p.scale;
    kv[j] = div_full(k_s[4 * lane + j], kn);
  }
  const float4 q4 = make_float4(qv[0], qv[1], qv[2], qv[3]), k4 = make_float4(kv[0], kv[1], kv[2], kv[3]);
  const float a_val = __bfloat162float(a_raw);
  const float b_val = __bfloat162float(b_raw);
  const float xx = a_val + dtb;
  const float sp = xx <= 20.f ? logf(1.f + tl_exp(xx)) : xx;
  const float decay = tl_exp((0.f - tl_exp(A_log_hv)) * sp);
  const float beta = bf(tl_sigmoid(b_val));
  __syncthreads();  // red[] is reused below

  // 3. recurrence: h *= exp(g); v -= h k; v *= beta; h += v k^T; o = h q (rows of this warp, in registers)
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    float4 hr = rows[r];
    hr.x *= decay;
    hr.y *= decay;
    hr.z *= decay;
    hr.w *= decay;
    const float hk = warp_sum(hr.x * k4.x + hr.y * k4.y + hr.z * k4.z + hr.w * k4.w);
    const float vv = (v_s[warp * 32 + r] - hk) * beta;
    hr.x += vv * k4.x;
    hr.y += vv * k4.y;
    hr.z += vv * k4.z;
    hr.w += vv * k4.w;
    const float o = warp_sum(hr.x * q4.x + hr.y * q4.y + hr.z * q4.z + hr.w * q4.w);
    __stcs(reinterpret_cast<float4*>(h + r * DK), hr);
    if (lane == 0) o_s[warp * 32 + r] = bf(o);
  }
  __syncthreads();

  // 4. gated RMSNorm: y = o * rsqrt(mean(o^2) + eps) * w * silu(z)
  {
    const float ov = o_s[tid];
    const float s = warp_sum(ov * ov);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    const float rstd = rsqrt_approx(div_full(red[0] + red[1] + red[2] + red[3], (float)DV) + p.eps);
    const float w = p.norm_w_f32 ? reinterpret_cast<const float*>(p.norm_w)[tid]
                                 : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p.norm_w)[tid]);
    const float z = __bfloat162float(x[2 * qk_dim + p.HV * DV + hv * DV + tid]);
    float y = ov * rstd * w;
    y *= z * tl_sigmoid(z);
    outp[tid] = __float2bfloat16(y);
  }
  // the key head's q/k taps: the last of its value heads (all have read the old taps by now) writes the new ones
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    int* cnt = p.counters + b * MAX_KH + kh;
    const int last = atomicAdd(cnt, 1) == HVK - 1;
    if (last) *cnt = 0;
    last_s = last;
  }
  __syncthreads();
  if (last_s) {
    __threadfence();
    cs[cq] = q1;
    cs[t1 + cq] = q2;
    cs[t2 + cq] = xq;
    cs[ck] = k1;
    cs[t1 + ck] = k2;
    cs[t2 + ck] = xk;
  }
}

void decode(const torch::Tensor& proj, torch::Tensor& conv_state, const torch::Tensor& conv_w, torch::Tensor& ssm,
            const torch::Tensor& state_idx, const torch::Tensor& A_log,
            const torch::Tensor& dt_bias, const torch::Tensor& norm_w, torch::Tensor& out, torch::Tensor& workspace,
            int64_t num_k_heads, double eps, double scale) {
  const int B = state_idx.size(0);
  const int HV = ssm.size(1), H = num_k_heads;
  TORCH_CHECK(ssm.size(2) == DV && ssm.size(3) == DK && ssm.stride(3) == 1 && ssm.stride(2) == DK &&
                  ssm.stride(1) == DV * DK && ssm.scalar_type() == at::kFloat,
              "ssm state layout");
  TORCH_CHECK(ssm.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(ssm.data_ptr()) % 16 == 0, "ssm alignment");
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == 3 && conv_state.stride(2) == 1 &&
                  conv_state.scalar_type() == at::kBFloat16,
              "conv state layout (slots, 3, dim)");
  TORCH_CHECK(conv_w.is_contiguous() && conv_w.size(1) == 4 && conv_w.scalar_type() == at::kBFloat16, "conv w");
  TORCH_CHECK(proj.stride(1) == 1 && proj.scalar_type() == at::kBFloat16 && proj.size(0) >= B, "proj");
  TORCH_CHECK(proj.size(1) >= 2 * H * DK + 2 * HV * DV + 2 * HV, "proj width");
  TORCH_CHECK(out.stride(-1) == 1 && out.size(0) >= B, "out");
  TORCH_CHECK(state_idx.scalar_type() == at::kInt && state_idx.stride(0) == 1, "state idx");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log fp32");
  TORCH_CHECK(HV % H == 0 && HV / H == 3 && H <= MAX_KH && B <= MAX_TOKENS, "heads / batch");
  TORCH_CHECK(conv_state.size(2) == 2 * H * DK + HV * DV, "conv dim");
  TORCH_CHECK(workspace.scalar_type() == at::kInt && workspace.numel() >= MAX_TOKENS * MAX_KH, "workspace");
  Params p;
  p.proj = reinterpret_cast<const __nv_bfloat16*>(proj.data_ptr());
  p.proj_stride = proj.stride(0);
  p.conv_state = reinterpret_cast<__nv_bfloat16*>(conv_state.data_ptr());
  p.cs_slot_stride = conv_state.stride(0);
  p.cs_tok_stride = conv_state.stride(1);
  p.conv_w = reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr());
  p.ssm = ssm.data_ptr<float>();
  p.ss_slot_stride = ssm.stride(0);
  p.state_idx = state_idx.data_ptr<int32_t>();
  p.A_log = A_log.data_ptr<float>();
  p.dt_bias = dt_bias.data_ptr();
  p.dt_bias_f32 = dt_bias.scalar_type() == at::kFloat;
  p.norm_w = norm_w.data_ptr();
  p.norm_w_f32 = norm_w.scalar_type() == at::kFloat;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_stride = out.stride(0);
  p.counters = workspace.data_ptr<int>();
  p.B = B;
  p.H = H;
  p.HV = HV;
  p.eps = (float)eps;
  p.scale = (float)scale;
  if (B == 0) return;
  gdn_decode_kernel<3><<<dim3(HV, B), NT, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn

TORCH_LIBRARY_FRAGMENT(pear_rollout_hopper, m) {
  m.def(
      "gdn_decode(Tensor proj, Tensor(a!) conv_state, Tensor conv_w, Tensor(b!) ssm, Tensor state_idx, Tensor A_log, "
      "Tensor dt_bias, Tensor norm_w, Tensor(c!) out, Tensor(d!) workspace, int num_k_heads, float eps, float scale) "
      "-> ()");
}
TORCH_LIBRARY_IMPL(pear_rollout_hopper, CUDA, m) { m.impl("gdn_decode", &gdn::decode); }
