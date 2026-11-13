import copy
import itertools

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from pear.models.qwen3_5.trainer.blackwell.gpu_backend import BlackwellTrainingGpuBackend
from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, raise_range_flag
from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels
from verl.utils.device import get_device_capability

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need a GPU")
# a GPU backend's own kernels are tested on the GPUs that select it
BLACKWELL = pytest.mark.skipif(
    not BlackwellTrainingGpuBackend.supports_compute_capability(get_device_capability()),
    reason="BlackwellTrainingGpuBackend's kernels",
)
HOPPER = pytest.mark.skipif(
    not HopperTrainingGpuBackend.supports_compute_capability(get_device_capability()),
    reason="HopperTrainingGpuBackend's kernels",
)

HIDDEN = 5120  # Qwen3.8-27B's hidden size (the norm kernels' block)
SMALL_HIDDEN = 512
LENGTHS = [700, 1500, 301]  # the sequences of a packed micro-batch
GDN_HEADS, GDN_VALUE_HEADS, GDN_DIM = 2, 4, 128  # the small config's Gated DeltaNet heads, Qwen3.5's head size
L2NORM_EPS = 1e-6  # FLA's q/k L2 norm eps, which HF's mixer uses
HEAD_IDS = 32  # score centering's head: the sampler's top 32 ids, the label among them
TEMPERATURE = 0.7
REPEATS = 5  # backward calls compared bit for bit
# a padded span's summary against the span's own, relative to its largest entry: the kernels' fp32 reduction order
# depends on the span length
SUMMARY_ROUNDING = 1e-3
SUMMARY_TOKENS = 8192  # a rank's boundary sequence in the summary test
# the summary test's log decay per token, in [-SUMMARY_DECAY, 0]: slow as the trainer's gates (faster decay leaves
# the transition below fp32's range over the rank's tokens)
SUMMARY_DECAY = 1e-3
SUMMARY_GAIN = 4  # the transition's error below FLA's at least by this factor
TF32_ROUNDING = 2**-10  # TF32's relative rounding error: 10 explicit mantissa bits, fp32 operands truncated on Hopper


@pytest.fixture(scope="module")
def gpu_backend():
    """The GPU backend Qwen3_5TrainingKernels.install_gpu_backend selects for the GPU, with its process-wide patches."""
    # the patches stay for the rest of the session: those of cuDNN's and FLA's delta rule modules and of transformers'
    # vision attention, which no later GPU file uses in this process, and Blackwell's cuBLAS variable, which the later
    # files' processes inherit: every check of an fp32 GEMM there has a tolerance both of cuBLAS's fp32 GEMMs meet
    gpu_backend = Qwen3_5TrainingKernels.select_gpu_backend(get_device_capability())
    gpu_backend.install()
    return gpu_backend


def rel_err(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def small_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=SMALL_HIDDEN,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        vocab_size=1000,
        layer_types=["linear_attention", "full_attention"],
    )


def hi_lo(x):
    """The bf16 activation autograd sees, registered with its fp16 twin and remainder, as the norms register them."""
    from pear.models.qwen3_5.trainer.operands import Fp16Operands

    x16 = x.half()
    handle = x.bfloat16().requires_grad_()
    Fp16Operands.register(handle, x16, (x - x16.float()).half())
    return handle


def test_fp16_operands_keep_the_bf16_weight_and_rerun_a_batch_out_of_range():
    from pear.models.qwen3_5.trainer.operands import Fp16Operands

    torch.manual_seed(0)
    w = torch.randn(128, 256, device="cuda")
    w = w.sign() * (w.abs() + 1e-3)  # above fp16's subnormal range once scaled
    assert torch.equal(Fp16Operands.weight16(w).float() / 256, w.bfloat16().float())
    x16 = torch.randn(64, 256, device="cuda").half()
    out = Fp16Operands.mm16(x16, Fp16Operands.weight16(w))
    expected = x16.double() @ w.bfloat16().double().t()
    assert out.dtype == torch.float32 and rel_err(out, expected) < 1e-6

    calls, reruns = [], []

    def run():
        calls.append(Fp16Operands.is_enabled)
        Fp16Operands.to16(torch.full((8,), 70000.0 if len(calls) == 1 else 1.0, device="cuda"))
        return len(calls)

    assert Fp16Operands.run_checked(run, lambda: reruns.append(True)) == 2
    assert calls == [True, False] and reruns == [True] and Fp16Operands.is_enabled


@triton.jit
def _row_range_kernel(X, FLAG, VALID, COLS: tl.constexpr, LIMIT: tl.constexpr):
    # the range check of a row's first VALID values
    cols = tl.arange(0, COLS)
    raise_range_flag(FLAG, tl.load(X + cols), cols < VALID, LIMIT)


@triton.jit
def _block_range_kernel(X, FLAG, VALID, ROWS: tl.constexpr, COLS: tl.constexpr, LIMIT: tl.constexpr):
    # the range check of a block's first VALID rows
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    x = tl.load(X + rows[:, None] * COLS + cols[None, :])
    raise_range_flag(FLAG, x, (rows < VALID)[:, None], LIMIT)


def test_the_range_flag_rises_for_a_value_past_fp16_s_range_under_the_mask():
    def is_raised(x, valid):
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        if x.dim() == 1:
            _row_range_kernel[(1,)](x, flag, valid, x.shape[0], FP16_LIMIT)
        else:
            _block_range_kernel[(1,)](x, flag, valid, x.shape[0], x.shape[1], FP16_LIMIT)
        return bool(flag.item())

    # a row of values and a block of rows, as the kernels check them
    for shape in ((8,), (4, 8)):
        x = torch.ones(shape, device="cuda")
        assert not is_raised(x, 3)
        x[-1] = -70000.0  # past the range, outside the mask
        assert not is_raised(x, 3)
        x[2] = -70000.0  # under the mask, by its magnitude
        assert is_raised(x, 3)


def test_fp32_output_linear_on_hi_lo_operands():
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, Fp32OutputLinear

    torch.manual_seed(1)
    x = torch.randn(2, 300, 512, device="cuda")
    w = (torch.randn(256, 512, device="cuda") * 512**-0.5).requires_grad_()
    handle = hi_lo(x)
    y = Fp32OutputLinear.apply(handle, w)
    Fp16Operands.release(handle)

    expected = x.double() @ w.detach().bfloat16().double().t()
    bf16_operands = x.bfloat16().double() @ w.detach().bfloat16().double().t()
    assert y.dtype == torch.float32
    assert rel_err(y, expected) < rel_err(bf16_operands, expected) / 100  # x to ~22 significant bits

    grad = torch.randn_like(y)
    y.backward(grad)
    g = grad.reshape(-1, 256).bfloat16()
    assert torch.equal(handle.grad.view(-1, 512), torch.mm(g, w.detach().bfloat16()))
    assert torch.equal(w.grad, torch.mm(g.t(), x.bfloat16().view(-1, 512)).float())


def test_fused_norms_match_the_hf_norm_on_the_fp32_stream():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    from pear.models.qwen3_5.trainer.rms_norm import AddRMSNorm, RMSNorm

    torch.manual_seed(2)
    norm = Qwen3_5RMSNorm(HIDDEN, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    weight = norm.weight.detach().bfloat16().requires_grad_()  # gathered in param_dtype
    hf_norm = copy.deepcopy(norm)
    with torch.no_grad():
        hf_norm.weight.copy_(weight)
    x = torch.randn(3, 257, HIDDEN, device="cuda", requires_grad=True)
    b = torch.randn(3, 257, HIDDEN, device="cuda").bfloat16().requires_grad_()

    y, y16, remainder = RMSNorm.apply(x, weight, norm.eps, torch.bfloat16, True, True)
    expected = hf_norm(x)
    assert y.dtype == torch.bfloat16 and rel_err(y, expected) < 4e-3
    assert rel_err(y16, expected) < 5e-4 and rel_err(y16.float() + remainder.float(), expected) < 1e-6
    grad = torch.randn_like(y)
    dx, dw = torch.autograd.grad(y, (x, weight), grad)
    dx_expected, dw_expected = torch.autograd.grad(expected, (x, hf_norm.weight), grad.float())
    assert dx.dtype == torch.float32 and rel_err(dx, dx_expected) < 1e-5 and rel_err(dw, dw_expected) < 1e-2

    x1, y, y16 = AddRMSNorm.apply(x, b, weight, norm.eps, torch.bfloat16, True)
    assert torch.equal(x1, x + b.float())
    expected = hf_norm(x + b.float())
    assert rel_err(y, expected) < 4e-3 and rel_err(y16, expected) < 5e-4
    dx1, dy = torch.randn_like(x1), torch.randn_like(y)
    dx, db = torch.autograd.grad((x1, y), (x, b), (dx1, dy))
    reference = x.detach().clone().requires_grad_()
    hf_x1 = reference + b.detach().float()
    (dx_expected,) = torch.autograd.grad((hf_x1, hf_norm(hf_x1)), reference, (dx1, dy.float()))
    assert rel_err(dx, dx_expected) < 1e-5 and db.dtype == torch.bfloat16 and rel_err(db, dx_expected) < 4e-3


def fp64_gated_delta_net(mixer, x):
    """The Gated DeltaNet's packed forward in fp64 from the mixer's parameters, its delta rule as the recurrence."""
    import torch.nn.functional as F

    T = x.shape[1]
    H, HV, K, V = mixer.num_k_heads, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim
    mixed_qkv = F.linear(x[0], mixer.in_proj_qkv.weight)
    z = F.linear(x[0], mixer.in_proj_z.weight).view(T, HV, V)
    beta = F.linear(x[0], mixer.in_proj_b.weight).sigmoid()
    g = -mixer.A_log.exp() * F.softplus(F.linear(x[0], mixer.in_proj_a.weight) + mixer.dt_bias)
    bounds = list(itertools.pairwise([0, *itertools.accumulate(LENGTHS)]))

    # the causal convolution and SiLU per sequence; q, k L2-normalized and repeated to the value heads
    width = mixer.conv1d.weight.shape[-1]
    conv_outputs = []
    for s, e in bounds:
        out = F.conv1d(mixed_qkv[s:e].t()[None], mixer.conv1d.weight, groups=mixed_qkv.shape[1], padding=width - 1)
        conv_outputs.append(out[0, :, : e - s].t())
    q, k, v = F.silu(torch.cat(conv_outputs)).split((H * K, H * K, HV * V), dim=-1)
    q, k = (t.view(T, H, K) / (t.view(T, H, K).pow(2).sum(-1, keepdim=True) + L2NORM_EPS).sqrt() for t in (q, k))
    q, k, v = q.repeat_interleave(HV // H, 1) * K**-0.5, k.repeat_interleave(HV // H, 1), v.view(T, HV, V)

    outputs = []
    for s, e in bounds:
        state = x.new_zeros(HV, K, V)
        for t in range(s, e):
            state = state * g[t].exp()[:, None, None]
            update = (v[t] - torch.einsum("hk,hkv->hv", k[t], state)) * beta[t][:, None]
            state = state + k[t][:, :, None] * update[:, None, :]
            outputs.append(torch.einsum("hk,hkv->hv", q[t], state))
    o = torch.stack(outputs)
    y = o * (o.pow(2).mean(-1, keepdim=True) + mixer.norm.eps).rsqrt() * mixer.norm.weight * F.silu(z)
    return F.linear(y.reshape(T, -1), mixer.out_proj.weight)[None]


def test_gated_delta_net_kernels_are_closer_to_fp64_than_the_hf_mixer(gpu_backend):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from pear.models.qwen3_5.trainer.gdn import GatedDeltaNet
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, Fp32OutputLinear
    from verl.models.transformers.qwen3_5 import qwen3_5_gated_delta_net_forward

    # the bf16 parameters FSDP gathers in param_dtype, for every mixer and the fp64 reference alike
    torch.manual_seed(3)
    hf_mixer = Qwen3_5GatedDeltaNet(small_text_config(), layer_idx=0).cuda()
    with torch.no_grad():
        hf_mixer.norm.weight.normal_(1, 0.1)
    hf_mixer = hf_mixer.bfloat16()
    kernels, reference = copy.deepcopy(hf_mixer), copy.deepcopy(hf_mixer).double()
    kernels.gpu_backend = gpu_backend
    for projection in (kernels.in_proj_qkv, kernels.in_proj_z, kernels.in_proj_a, kernels.in_proj_b, kernels.out_proj):
        Fp32OutputLinear.install(projection)
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], device="cuda", dtype=torch.long)
    x = torch.randn(1, sum(LENGTHS), SMALL_HIDDEN, device="cuda")
    projection = torch.randn(1, sum(LENGTHS), SMALL_HIDDEN, device="cuda")

    def run(module, forward, hidden):
        out = forward(module, hidden, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
        (out.double() * projection.double()).sum().backward()
        return out.detach(), module.in_proj_qkv.weight.grad, module.out_proj.weight.grad

    expected = run(reference, lambda module, hidden, **_: fp64_gated_delta_net(module, hidden), x.double())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = run(hf_mixer, qwen3_5_gated_delta_net_forward, x.bfloat16())
        handle = hi_lo(x)
        actual = run(kernels, GatedDeltaNet.forward, handle)
        Fp16Operands.release(handle)

    assert actual[0].dtype == torch.float32
    assert rel_err(actual[0], expected[0]) < rel_err(baseline[0], expected[0]) / 5
    for actual_grad, baseline_grad, expected_grad in zip(actual[1:], baseline[1:], expected[1:], strict=True):
        # the backward keeps bf16 operands on more precise saved activations: no further from fp64 than HF's
        assert rel_err(actual_grad, expected_grad) < rel_err(baseline_grad, expected_grad)


@HOPPER
def test_the_hopper_delta_rule_backward_reuses_the_recompute_s_tensors():
    from functools import partial
    from types import SimpleNamespace

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from torch.utils.checkpoint import checkpoint

    from pear.models.qwen3_5.trainer.hopper.delta_rule import HopperDeltaRule
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, RecomputeFlag

    torch.manual_seed(5)
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], device="cuda", dtype=torch.long)
    tokens = sum(LENGTHS)
    q, k = (F.normalize(torch.randn(1, tokens, GDN_HEADS, GDN_DIM, device="cuda"), dim=-1) for _ in range(2))
    v = torch.randn(1, tokens, GDN_VALUE_HEADS, GDN_DIM, device="cuda")
    g = -torch.rand(1, tokens, GDN_VALUE_HEADS, device="cuda")
    beta = torch.rand(1, tokens, GDN_VALUE_HEADS, device="cuda")
    do = torch.randn(1, tokens, GDN_VALUE_HEADS, GDN_DIM, device="cuda", dtype=torch.bfloat16)

    def leaves():
        return [t.bfloat16().requires_grad_() for t in (q, k, v)] + [t.clone().requires_grad_() for t in (g, beta)]

    expected_leaves = leaves()
    expected = chunk_gated_delta_rule(*expected_leaves, cu_seqlens=cu)[0]
    expected.backward(do)

    # as the mixer runs it in a checkpointed layer: the forward on fp16 twins, the recompute on the bf16 operands
    def delta_rule(q, k, v, g, beta):
        twins = (None,) * 3 if RecomputeFlag.in_recompute.get() else (q.half(), k.half(), v.half())
        return HopperDeltaRule.apply_packed(q, k, v, g, beta, cu, cu.cpu(), None, *twins, 0)[0]

    layer = SimpleNamespace(
        _gradient_checkpointing_func=partial(checkpoint, use_reentrant=False), gpu_backend=HopperTrainingGpuBackend
    )
    RecomputeFlag.install(SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))))
    actual_leaves = leaves()
    actual = layer._gradient_checkpointing_func(delta_rule, *actual_leaves)
    actual.backward(do)

    # the forward's fp16 output is closer to FLA's own fp32 forward than its bf16 one; the backward is FLA's on the
    # recompute's w, h and v_new (the checkpoint raises if what the two passes save differs in metadata)
    reference = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu)[0]
    assert rel_err(actual, reference) < rel_err(expected, reference)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert torch.equal(actual_leaf.grad, expected_leaf.grad)

    # a non-finite fp16 output raises the range flag
    Fp16Operands.reset()
    with torch.no_grad():
        twins = q.half(), k.half(), v.half()
        twins[2][0, 0, 0, 0] = float("inf")
        HopperDeltaRule.apply_packed(q.bfloat16(), k.bfloat16(), v.bfloat16(), g, beta, cu, cu.cpu(), None, *twins, 0)
    assert bool(Fp16Operands.range_flag(q.device).item())
    Fp16Operands.reset()


@HOPPER
def test_the_boundary_summary_s_transition_is_closer_to_fp64_than_fla_s():
    from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    from pear.models.qwen3_5.trainer.hopper.delta_rule import HopperDeltaRule, _summary_kernel

    # the fp16 twins' WY factors of one rank's boundary sequence, as the forward hands them to the pre-process
    torch.manual_seed(0)
    T, H, HV, K = SUMMARY_TOKENS, GDN_HEADS, GDN_VALUE_HEADS, GDN_DIM
    k = F.normalize(torch.randn(1, T, H, K, device="cuda"), dim=-1).half()
    v = torch.randn(1, T, HV, K, device="cuda").half()
    g = -torch.rand(1, T, HV, device="cuda") * SUMMARY_DECAY
    beta = torch.rand(1, T, HV, device="cuda")
    cu = torch.tensor([0, T], device="cuda", dtype=torch.long)
    chunk_indices = prepare_chunk_indices(cu, HopperDeltaRule.CHUNK)
    g = chunk_local_cumsum(
        g, chunk_size=HopperDeltaRule.CHUNK, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=chunk_indices
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=chunk_indices, chunk_size=HopperDeltaRule.CHUNK
    )

    # [h | M] by the chunk recurrence in fp64: h from a zero state on u, M from the identity on u = 0
    kk, ww, uu, gg = k[0].double().repeat_interleave(HV // H, 1), w[0].double(), u[0].double(), g[0].double()
    expected = torch.zeros(HV, K, 2 * K, dtype=torch.float64, device="cuda")
    expected[:, :, K:] = torch.eye(K, dtype=torch.float64, device="cuda")
    for s in range(0, T, HopperDeltaRule.CHUNK):
        e = min(s + HopperDeltaRule.CHUNK, T)
        rhs = torch.cat([uu[s:e], torch.zeros_like(uu[s:e])], -1) - torch.einsum("thk,hkc->thc", ww[s:e], expected)
        rhs = rhs * torch.exp2(gg[e - 1][None] - gg[s:e])[:, :, None]
        expected = expected * torch.exp2(gg[e - 1])[:, None, None] + torch.einsum("thk,thc->hkc", kk[s:e], rhs)

    ours = torch.zeros(HV, K, 2 * K, device="cuda")
    _summary_kernel[(2 * K // HopperDeltaRule.SUMMARY_COLUMNS, HV)](
        k,
        w,
        u,
        g,
        ours,
        cu,
        H=H,
        HV=HV,
        K=K,
        V=K,
        BT=HopperDeltaRule.CHUNK,
        BS=HopperDeltaRule.SUMMARY_COLUMNS,
        num_warps=HopperDeltaRule.SUMMARY_WARPS,
        num_stages=HopperDeltaRule.SUMMARY_STAGES,
    )
    fla = torch.zeros(HV, K, 2 * K, device="cuda")
    pre_process_fwd_kernel_merged[(2 * triton.cdiv(K, HopperDeltaRule.FLA_SUMMARY_BLOCK), HV)](
        k=k,
        v=u,
        w=w,
        g=g,
        gk=None,
        bg=None,
        u=u,
        hm=fla,
        cu_seqlens=cu,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=K,
        BT=HopperDeltaRule.CHUNK,
        BK1=K,
        BLOCK_SIZE=HopperDeltaRule.FLA_SUMMARY_BLOCK,
        MULTI_SEQS=False,
    )
    # the transition M from the recurrence, where FLA chains K x K products; the offset h at TF32's precision
    h, m = slice(0, K), slice(K, 2 * K)
    assert rel_err(ours[..., m], expected[..., m]) * SUMMARY_GAIN < rel_err(fla[..., m], expected[..., m])
    assert rel_err(ours[..., h], expected[..., h]) < TF32_ROUNDING


@BLACKWELL
def test_the_delta_rule_holds_no_memory_per_packing():
    from cudnn.linear_attention.ops import gdn

    from pear.models.qwen3_5.trainer.blackwell.delta_rule import BlackwellDeltaRule, CudnnGDNGraphs

    CudnnGDNGraphs.install()

    def step(lengths):
        cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.long)
        tokens = sum(lengths)
        q, k = (
            F.normalize(torch.randn(1, tokens, GDN_HEADS, GDN_DIM, device="cuda"), dim=-1).bfloat16().requires_grad_()
            for _ in range(2)
        )
        v = torch.randn(1, tokens, GDN_VALUE_HEADS, GDN_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(1, tokens, GDN_VALUE_HEADS, device="cuda")).requires_grad_()
        beta = torch.rand(1, tokens, GDN_VALUE_HEADS, device="cuda").requires_grad_()
        out, _ = BlackwellDeltaRule.apply_packed(q, k, v, g, beta, cu, cu.cpu(), None, None, None, None, 0)
        out.float().sum().backward()

    step(LENGTHS)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    # more packings than any graph cache keeps; cuDNN's own caches would hold a graph and its workspace for each
    for extra in range(1, 3 * CudnnGDNGraphs.GRAPHS_PER_CACHE + 1):
        step([length + extra for length in LENGTHS])
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= resident
    assert all(len(cache) <= CudnnGDNGraphs.GRAPHS_PER_CACHE for cache in (gdn.fprop_cache, gdn.bprop_cache))
    assert len(gdn.summary_cache) <= CudnnGDNGraphs.SUMMARY_GRAPHS


@BLACKWELL
def test_the_summaries_of_a_padded_span_match_its_own_to_rounding_and_reuse_the_bucket_graphs():
    from cudnn.linear_attention.ops import gdn

    from pear.models.qwen3_5.trainer.blackwell.delta_rule import BlackwellDeltaRule, CudnnGDNGraphs

    CudnnGDNGraphs.install()
    ops = torch.ops.cudnn
    tokens = BlackwellDeltaRule.SUMMARY_SPAN_BUCKET + sum(LENGTHS)
    q, k = (F.normalize(torch.randn(tokens, GDN_HEADS, GDN_DIM, device="cuda"), dim=-1) for _ in range(2))
    v, do = (torch.randn(tokens, GDN_VALUE_HEADS, GDN_DIM, device="cuda") for _ in range(2))
    g = -torch.rand(tokens, GDN_VALUE_HEADS, device="cuda")
    beta = torch.rand(tokens, GDN_VALUE_HEADS, device="cuda")
    # the forward's operands (fp16 twins) and the backward's (bf16), as BlackwellDeltaRule passes them
    operands = (k.half(), v.half(), g, beta, do.bfloat16(), q.bfloat16(), k.bfloat16(), g, beta)

    def summaries(rows, length):
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        forward = ops.gated_delta_net_summary(
            *rows[:4], cu, initial_state=None, output_transition=True, plan_name=None, **BlackwellDeltaRule.GDN_FLAGS
        )
        backward = ops.gated_delta_net_summary_bwd(
            *rows[4:],
            cu,
            GDN_DIM**-0.5,
            d_final_state=None,
            output_transition=True,
            plan_name=None,
            **BlackwellDeltaRule.GDN_FLAGS,
        )
        return (*forward, *backward)

    # a span inside the first bucket and one past it
    for s, e in ((LENGTHS[0], 2 * LENGTHS[0]), (3, 3 + BlackwellDeltaRule.SUMMARY_SPAN_BUCKET + LENGTHS[2])):
        padded_summaries = summaries(BlackwellDeltaRule.padded_span(operands, s, e), BlackwellDeltaRule.bucket(e - s))
        span_summaries = summaries([t[s:e] for t in operands], e - s)
        for a, b in zip(padded_summaries, span_summaries, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=SUMMARY_ROUNDING * b.abs().max().item())
    graphs = len(gdn.summary_cache)
    for length in LENGTHS:  # all in the first bucket, whose graphs exist
        summaries(BlackwellDeltaRule.padded_span(operands, 0, length), BlackwellDeltaRule.bucket(length))
    assert len(gdn.summary_cache) == graphs


def test_the_mlp_rounds_once_and_skips_the_down_projection_in_the_recompute():
    from pear.models.qwen3_5.trainer.mlp import Fp32SwiGLUMLP
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, RecomputeFlag

    torch.manual_seed(4)
    x = torch.randn(1, 3000, SMALL_HIDDEN, device="cuda")
    w_gate, w_up = (torch.randn(1024, SMALL_HIDDEN, device="cuda") * SMALL_HIDDEN**-0.5 for _ in range(2))
    w_down = torch.randn(SMALL_HIDDEN, 1024, device="cuda") * 1024**-0.5
    weights = [w.bfloat16().requires_grad_() for w in (w_gate, w_up, w_down)]
    expected = (lambda a, b: (a * torch.sigmoid(a) * b) @ w_down.bfloat16().double().t())(
        x.double() @ w_gate.bfloat16().double().t(), x.double() @ w_up.bfloat16().double().t()
    )
    xb = x.bfloat16()
    gate, up = xb @ weights[0].detach().t(), xb @ weights[1].detach().t()
    baseline = (torch.nn.functional.silu(gate) * up) @ weights[2].detach().t()  # Liger rounds each step to bf16

    handle = hi_lo(x)
    y = Fp32SwiGLUMLP.apply(handle, *weights)
    assert y.dtype == torch.float32 and rel_err(y, expected) < rel_err(baseline, expected) / 2
    grad = torch.randn_like(y)
    grads = torch.autograd.grad(y, (handle, *weights), grad)

    # the recompute regenerates the saved tensors (equal) and leaves the output uncomputed
    token = RecomputeFlag.in_recompute.set(True)
    try:
        recomputed = Fp32SwiGLUMLP.apply(handle, *weights)
        assert torch.equal(torch.autograd.grad(recomputed, (handle, *weights), grad)[1], grads[1])
    finally:
        RecomputeFlag.in_recompute.reset(token)
        Fp16Operands.release(handle)


def test_the_fused_head_matches_the_fp32_head_and_repeats_bit_for_bit():
    from pear.models.qwen3_5.trainer.head import Fp32HeadLogProbs, FusedFp32HeadLogProbs

    torch.manual_seed(5)
    rows, vocab = 2 * Fp32HeadLogProbs.CHUNK_ROWS + 37, 3000
    hidden = torch.randn(rows, SMALL_HIDDEN, device="cuda").bfloat16()
    weight = (torch.randn(vocab, SMALL_HIDDEN, device="cuda") * SMALL_HIDDEN**-0.5).bfloat16()
    head_ids = (hidden.float() @ weight.float().t()).topk(HEAD_IDS, -1).indices
    ids = torch.cat([head_ids[:, :1], head_ids], -1)  # the label among the head ids
    grads = torch.randn(rows, HEAD_IDS + 1, device="cuda"), torch.randn(rows, device="cuda")

    def run(function):
        leaves = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        log_probs, entropy = function.apply(*leaves, ids, TEMPERATURE)
        torch.autograd.backward((log_probs, entropy), grads)
        return log_probs.detach(), entropy.detach(), leaves[0].grad, leaves[1].grad

    expected = run(Fp32HeadLogProbs)
    actual = run(FusedFp32HeadLogProbs)
    assert torch.equal(actual[0], expected[0])  # verl's own log-normalizer and gather
    assert (actual[1] - expected[1]).abs().max() < 1e-5
    assert rel_err(actual[2], expected[2]) < 1e-2 and rel_err(actual[3], expected[3]) < 1e-2
    for _ in range(REPEATS):
        assert all(torch.equal(a, b) for a, b in zip(run(FusedFp32HeadLogProbs)[2:], actual[2:], strict=True))


@BLACKWELL
def test_the_vision_attention_is_more_precise_than_tf32():
    from pear.models.qwen3_5.trainer.blackwell.vision_attention import Fp32VisionAttention

    torch.manual_seed(6)
    S, H, D, images = 1024, 16, 72, 10  # 10 images of 1024 patches: two groups of the score budget
    q, k, v = (torch.randn(images * S, H, D, device="cuda", requires_grad=True) for _ in range(3))
    scale = D**-0.5
    out = Fp32VisionAttention.apply(q, k, v, S, scale)
    qq, kk, vv = (t.detach().double().view(images, S, H, D).transpose(1, 2).requires_grad_() for t in (q, k, v))
    expected = torch.softmax(qq @ kk.transpose(-1, -2) * scale, -1) @ vv
    assert rel_err(out, expected.transpose(1, 2).reshape(images * S, H, D)) < 1e-5
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (qq, kk, vv), grad.double().view(images, S, H, D).transpose(1, 2))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert rel_err(actual_grad, expected_grad.transpose(1, 2).reshape(images * S, H, D)) < 1e-4


@BLACKWELL
def test_blackwell_s_vision_attention_runs_the_family_s_forward_on_its_own_attention():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

    from pear.models.qwen3_5.trainer.blackwell.vision_attention import BlackwellVisionAttention
    from pear.models.qwen3_5.trainer.vision_attention import BatchedVisionAttention

    torch.manual_seed(7)
    hidden_size, heads = 64, 4
    sizes = [16, 16, 8, 16, 16, 4]  # patches per image: four runs of equal sizes
    attention = Qwen3_5VisionAttention(Qwen3_5VisionConfig(hidden_size=hidden_size, num_heads=heads)).cuda().eval()
    hidden = torch.randn(sum(sizes), hidden_size, device="cuda")
    cu_seqlens = torch.tensor([0, *itertools.accumulate(sizes)], dtype=torch.int32, device="cuda")
    angles = torch.randn(sum(sizes), hidden_size // heads // 2, device="cuda")
    position = (angles.cos().repeat(1, 2), angles.sin().repeat(1, 2))

    # the family's forward, SDPA on each run, in fp64
    expected = BatchedVisionAttention.forward(
        copy.deepcopy(attention).double(), hidden.double(), cu_seqlens, tuple(p.double() for p in position)
    )
    for _ in range(2):  # the second call takes the runs from their cache
        actual = BlackwellVisionAttention.attention_forward(attention, hidden, cu_seqlens, position_embeddings=position)
        assert actual.dtype == torch.float32 and rel_err(actual, expected) < TF32_ROUNDING


def test_the_vision_linears_are_more_precise_than_tf32():
    from pear.models.qwen3_5.trainer.vision_tower import Bf16x3Linear, Tf32Linear

    torch.manual_seed(6)
    x = torch.randn(4096, 1152, device="cuda", requires_grad=True)
    w = (torch.randn(4304, 1152, device="cuda") * 1152**-0.5).requires_grad_()
    bias = torch.randn(4304, device="cuda", requires_grad=True)
    expected = x.double() @ w.double().t() + bias.double()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        assert rel_err(Bf16x3Linear.apply(x, w, bias), expected) < 1e-5
        tf32_error = rel_err(Tf32Linear.apply(x, w, bias), expected)
        assert rel_err(Bf16x3Linear.apply(x, w, bias), expected) < tf32_error / 10
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    # the checkpoint compares the saved tensors of the forward (Bf16x3Linear) and of the recompute (Tf32Linear)
    saved = [Bf16x3Linear.apply(x, w, bias).grad_fn.saved_tensors, Tf32Linear.apply(x, w, bias).grad_fn.saved_tensors]
    assert [(t.shape, t.dtype) for t in saved[0]] == [(t.shape, t.dtype) for t in saved[1]]
