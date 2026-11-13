import copy
import itertools
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from verl.utils.device import get_device_capability
from verl.utils.net_utils import get_free_port

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Ulysses over two GPUs")

LOCALHOST = "127.0.0.1"
SP = 2
SMALL_HIDDEN = 512
LENGTHS = [900, 1700, 402]  # the second sequence crosses the ranks' boundary (3002 tokens, 1501 per rank)
KV_HEADS = (2, 1)  # the attention's KV heads: one per rank, and one shared by both ranks


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


def gathered(tensor):
    """The ranks' sequence slices [1, T / SP, ...] concatenated."""
    parts = [torch.empty_like(tensor) for _ in range(SP)]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=1)


def summed(tensor):
    tensor = tensor.clone()
    dist.all_reduce(tensor)
    return tensor


def gated_delta_net_checks(rank: int, gpu_backend) -> None:
    """The kernels' mixer over the ranks' slices (the delta rule's state chained across them, the halo exchanged)
    against the same mixer on the whole batch in one rank."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from pear.models.qwen3_5.trainer.gdn import GatedDeltaNet
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, Fp32OutputLinear
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group

    torch.manual_seed(0)
    mixer = Qwen3_5GatedDeltaNet(small_text_config(), layer_idx=0).cuda().bfloat16()
    mixer.gpu_backend = gpu_backend
    for projection in (mixer.in_proj_qkv, mixer.in_proj_z, mixer.in_proj_a, mixer.in_proj_b, mixer.out_proj):
        Fp32OutputLinear.install(projection)
    T = sum(LENGTHS)
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], device="cuda", dtype=torch.long)
    x = torch.randn(1, T, SMALL_HIDDEN, device="cuda")
    projection = torch.randn(1, T, SMALL_HIDDEN, device="cuda")
    local = slice(rank * T // SP, (rank + 1) * T // SP)

    def run(hidden, out_projection):
        mixer.zero_grad()
        handle = hi_lo(hidden)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = GatedDeltaNet.forward(mixer, handle, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
        (out * out_projection).sum().backward()
        Fp16Operands.release(handle)
        grads = [p.grad.clone() for p in (mixer.in_proj_qkv.weight, mixer.conv1d.weight, mixer.out_proj.weight)]
        return out.detach(), handle.grad, grads

    set_ulysses_sequence_parallel_group(None)
    whole, whole_grad, whole_weight_grads = run(x, projection)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    split, split_grad, split_weight_grads = run(x[:, local], projection[:, local])

    assert rel_err(gathered(split), whole) < 2e-3
    assert rel_err(gathered(split_grad), whole_grad) < 1e-2  # the halo's gradient returned to the previous rank
    for split_weight_grad, whole_weight_grad in zip(split_weight_grads, whole_weight_grads, strict=True):
        assert rel_err(summed(split_weight_grad), whole_weight_grad) < 1e-2


def fla_candidate_checks() -> None:
    """FLA's restricted autotuners compiled none of the candidates the Hopper GPU backend drops."""
    from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend

    compiled = [
        candidate
        for kernel in HopperTrainingGpuBackend.SPILLING_FLA_KERNELS
        for caches in kernel.fn.fn.device_caches.values()  # FLA's heuristics, its autotuner, the jit function
        for candidate in caches[0].values()
    ]
    assert compiled
    for candidate in compiled:
        assert candidate.metadata.num_warps >= HopperTrainingGpuBackend.FLA_MIN_WARPS, (
            candidate.name,
            candidate.metadata.num_warps,
        )


def attention_checks(rank: int, gpu_backend, kv_heads: int) -> None:
    """The kernels' full attention (fp16 operands in the forward, the q/k path in fp32; also in a checkpointed layer)
    and HF's attention under Ulysses against an fp64 reference on the whole batch; HF's attention with one all-to-all
    equals verl's."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from transformers.integrations import flash_attention

    from pear.models.qwen3_5.trainer.attention import FullAttention, UlyssesPackedAttention
    from pear.models.qwen3_5.trainer.operands import Fp16Operands, Fp32OutputLinear, RecomputeFlag
    from verl.models.transformers.monkey_patch import _ulysses_flash_attention_forward
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group

    torch.manual_seed(1)
    config = small_text_config()
    config._attn_implementation, config.num_key_value_heads = gpu_backend.attention_cls.implementation, kv_heads
    hf_attention = modeling.Qwen3_5Attention(config, layer_idx=1).cuda().bfloat16()
    with torch.no_grad():
        for norm in (hf_attention.q_norm, hf_attention.k_norm):
            norm.weight.normal_(0, 0.1)
    kernels = copy.deepcopy(hf_attention)
    kernels.gpu_backend = gpu_backend
    for projection in (kernels.q_proj, kernels.k_proj, kernels.v_proj, kernels.o_proj):
        Fp32OutputLinear.install(projection)
    for norm in (kernels.q_norm, kernels.k_norm):
        FullAttention.install_head_norm(norm)
    reference = copy.deepcopy(hf_attention).double()
    reference.config._attn_implementation = "eager"

    T = sum(LENGTHS)
    local = slice(rank * T // SP, (rank + 1) * T // SP)
    positions = torch.cat([torch.arange(n, device="cuda") for n in LENGTHS])[None]
    rotary = modeling.Qwen3_5TextRotaryEmbedding(config).cuda()
    x = torch.randn(1, T, SMALL_HIDDEN, device="cuda")
    cos, sin = rotary(x, positions[None].expand(3, -1, -1))  # fp32: they follow the fp32 residual stream
    projection = torch.randn(1, T, SMALL_HIDDEN, device="cuda")

    # fp64 reference: eager attention with the packed sequences' block-diagonal causal mask
    starts = torch.repeat_interleave(torch.tensor([0, *itertools.accumulate(LENGTHS)][:-1]), torch.tensor(LENGTHS))
    allowed = (starts[None, :] == starts[:, None]) & torch.ones(T, T, dtype=torch.bool).tril()
    mask = torch.zeros(T, T, dtype=torch.float64).masked_fill(~allowed, float("-inf")).cuda()[None, None]
    x64 = x.double().requires_grad_()
    expected, _ = reference(x64, (cos.double(), sin.double()), mask)
    (expected * projection.double()).sum().backward()

    def run(module, forward, hidden, embeddings):
        hidden = hidden[:, local].detach().requires_grad_()
        handle = hi_lo(hidden.detach()) if module is kernels else hidden.bfloat16()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = forward(module, handle, embeddings, None, position_ids=positions[:, local])
        (out.float() * projection[:, local]).sum().backward()
        if module is kernels:
            Fp16Operands.release(handle)
            return gathered(out.detach().float()), gathered(handle.grad.float())
        return gathered(out.detach().float()), gathered(hidden.grad)

    # the kernels' attention in a checkpointed layer: the first forward's saves are discarded, the recompute's used
    layer = SimpleNamespace(
        _gradient_checkpointing_func=partial(checkpoint, use_reentrant=False), gpu_backend=gpu_backend
    )
    RecomputeFlag.install(SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))))
    checkpointed = partial(layer._gradient_checkpointing_func, FullAttention.forward)

    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    local_embeddings = (cos[:, local], sin[:, local])
    flash_attention._flash_attention_forward = UlyssesPackedAttention.forward
    actual, actual_grad = run(kernels, FullAttention.forward, x, local_embeddings)
    recomputed, recomputed_grad = run(kernels, checkpointed, x, local_embeddings)
    bf16_embeddings = tuple(t.bfloat16() for t in local_embeddings)
    packed, packed_grad = run(hf_attention, modeling.Qwen3_5Attention.forward, x, bf16_embeddings)
    flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
    baseline, baseline_grad = run(hf_attention, modeling.Qwen3_5Attention.forward, x, bf16_embeddings)

    assert torch.equal(packed, baseline)  # the same kernel on the same heads
    assert rel_err(packed_grad, baseline_grad) < 1e-2
    assert rel_err(actual, expected) < rel_err(baseline, expected) / 2
    assert torch.equal(recomputed, actual)  # the checkpoint's first forward runs the same kernels
    # the backward keeps bf16 numerics: as close as HF's attention's, also from the recompute
    assert rel_err(actual_grad, x64.grad) < 2 * rel_err(baseline_grad, x64.grad)
    assert rel_err(recomputed_grad, x64.grad) < 2 * rel_err(baseline_grad, x64.grad)


def two_rank_checks(rank: int, port: int) -> None:
    from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://{LOCALHOST}:{port}", rank=rank, world_size=SP)
    gpu_backend = Qwen3_5TrainingKernels.select_gpu_backend(get_device_capability())
    gpu_backend.install()
    try:
        gated_delta_net_checks(rank, gpu_backend)
        if gpu_backend is HopperTrainingGpuBackend:
            fla_candidate_checks()
        for kv_heads in KV_HEADS:
            # the KV-head layouts the GPU backend's attention runs (HF's attention takes the others)
            if gpu_backend.attention_cls.eligible(gpu_backend.attention_cls.implementation, kv_heads, SP):
                attention_checks(rank, gpu_backend, kv_heads)
    finally:
        dist.destroy_process_group()


def test_the_mixers_under_ulysses_on_two_ranks():
    torch.multiprocessing.spawn(two_rank_checks, args=(get_free_port(LOCALHOST)[0],), nprocs=SP)
