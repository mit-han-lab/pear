import os
import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.sample import sampler as vllm_sampler
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker import mamba_utils

from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gdn import GDNCore
from pear.models.qwen3_5.rollout.blackwell.gemm import GemmTableLinearMethod
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.model import AllReduceBuffers, LogitsGatherBuffers
from pear.models.qwen3_5.rollout.blackwell.rope import Fp32Rope
from pear.models.qwen3_5.rollout.blackwell.sampler import BlackwellSampler
from pear.models.qwen3_5.rollout.mamba_copy import MambaCopy
from verl.utils.net_utils import get_free_port

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
H, HV, K, V = 8, 24, 128, 128  # Qwen3.8-27B's Gated DeltaNet heads on one of two tensor-parallel ranks
C = 2 * H * K + HV * V  # conv channels: q, k, v
SLOTS = 6  # state blocks of the pool; block 0 is the padding rows' slot
EPS = 1e-6  # the gated norm's epsilon
HIDDEN = 5120  # the model's hidden size, which the fused all-reduce is built for
VOCAB = 248320
MAX_TOKENS = 8192  # the replicas' max_num_batched_tokens
MAX_ROWS = 128  # their max_num_seqs
LOCALHOST = "127.0.0.1"
L2_EPS = 1e-6  # vLLM's q / k l2norm epsilon, under the square root
HEAD = 256  # the attention layers' head size
Q_HEADS, KV_HEADS = 12, 2  # their heads on one of two ranks
ROPE = {  # the model's rope_parameters
    "rope_type": "default",
    "rope_theta": 1e7,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}
ROPE_POSITIONS = 4096  # the test cache's positions (the model's: 262144)


class Buffer:
    """A vLLM CpuGpuBuffer of int64 values."""

    def __init__(self, values: list[int]):
        self.np = np.array(values, dtype=np.int64)
        self.gpu = torch.tensor(values, dtype=torch.int64, device=DEVICE)

    def copy_to_gpu(self, n: int) -> torch.Tensor:
        return self.gpu[:n]


@pytest.fixture
def patched(monkeypatch):
    # every test leaves vLLM's state copy as it found it
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", mamba_utils.do_mamba_copy_block)
    monkeypatch.setattr(mamba_utils, "collect_mamba_copy_meta", mamba_utils.collect_mamba_copy_meta)


def test_state_copies_match_torch(patched):
    MambaCopy.install()
    pools = [torch.randint(0, 255, (SLOTS, 4096 + 65536 + 48), dtype=torch.uint8, device=DEVICE) for _ in range(3)]
    expected = [pool.clone() for pool in pools]
    src, dst = 1, 4
    srcs, dsts, sizes = [], [], []
    for pool, want in zip(pools, expected, strict=True):
        for offset, size in ((0, 4096), (4096, 65536)):  # a conv state and a recurrent state of one layer
            srcs.append(pool[src, offset:].data_ptr())
            dsts.append(pool[dst, offset:].data_ptr())
            sizes.append(size)
            want[dst, offset : offset + size] = want[src, offset : offset + size]
    copy_bufs = types.SimpleNamespace(
        src_ptrs=Buffer(srcs), dst_ptrs=Buffer(dsts), sizes=Buffer(sizes), offset=len(sizes)
    )

    mamba_utils.do_mamba_copy_block(copy_bufs)

    torch.cuda.synchronize()
    assert all(torch.equal(pool, want) for pool, want in zip(pools, expected, strict=True))


def gdn_layer(generator: torch.Generator) -> types.SimpleNamespace:
    """A Gated DeltaNet layer's weights and state pools on one of two ranks."""
    shape = (SLOTS, C, 3) if is_conv_state_dim_first() else (SLOTS, 3, C)
    return types.SimpleNamespace(
        num_k_heads=2 * H,
        num_v_heads=2 * HV,
        head_k_dim=K,
        head_v_dim=V,
        tp_size=2,
        kv_cache=(
            torch.randn(shape, generator=generator, device=DEVICE).to(torch.bfloat16),
            torch.randn(SLOTS, HV, V, K, generator=generator, device=DEVICE) * 0.1,
        ),
        conv1d=types.SimpleNamespace(weight=(torch.randn(C, 4, generator=generator, device=DEVICE) * 0.5).bfloat16()),
        A_log=torch.randn(HV, generator=generator, device=DEVICE) * 0.5,
        dt_bias=torch.randn(HV, generator=generator, device=DEVICE).bfloat16(),
        norm=types.SimpleNamespace(weight=torch.randn(V, generator=generator, device=DEVICE).bfloat16(), eps=EPS),
    )


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)


def reference_decode(layer, qkvz: torch.Tensor, ba: torch.Tensor, slots: torch.Tensor):
    """One decode step in fp64: the causal conv and its new state, the gated delta rule, the gated RMSNorm."""
    conv_state, ssm = layer.kv_cache
    window = conv_state[slots].double()
    window = window.transpose(1, 2) if is_conv_state_dim_first() else window  # [rows, 3, C]
    x = qkvz[:, :C].double()
    w = layer.conv1d.weight.double()
    acc = (window * w[:, :3].T).sum(1) + x * w[:, 3]
    conv_out = acc * torch.sigmoid(acc)
    new_window = torch.cat([window[:, 1:], x[:, None]], 1)

    q = l2norm(conv_out[:, : H * K].view(-1, H, K)) * K**-0.5
    k = l2norm(conv_out[:, H * K : 2 * H * K].view(-1, H, K))
    q, k = q.repeat_interleave(HV // H, 1), k.repeat_interleave(HV // H, 1)
    v = conv_out[:, 2 * H * K :].view(-1, HV, V)
    g = -layer.A_log.double().exp() * F.softplus(ba[:, HV:].double() + layer.dt_bias.double())
    beta = torch.sigmoid(ba[:, :HV].double())
    state = ssm[slots].double() * g.exp()[..., None, None]
    u = (v - torch.einsum("rhvk,rhk->rhv", state, k)) * beta[..., None]
    state = state + u[..., None] * k[:, :, None, :]
    o = torch.einsum("rhvk,rhk->rhv", state, q)

    z = qkvz[:, C:].double().view(-1, HV, V)
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * layer.norm.weight.double() * F.silu(z)
    return y, new_window, state


def test_a_decode_step_matches_the_reference():
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    layer = gdn_layer(generator)
    rows = 4
    qkvz = torch.randn(rows + 1, C + HV * V, generator=generator, device=DEVICE).bfloat16()
    ba = torch.randn(rows + 1, 2 * HV, generator=generator, device=DEVICE).bfloat16()
    slots = torch.tensor([3, 1, 5, 2, 0], dtype=torch.int32, device=DEVICE)  # the last row pads (slot 0)
    md = types.SimpleNamespace(num_actual_tokens=rows + 1, non_spec_state_indices_tensor=slots)
    conv_state, ssm = (s.clone() for s in layer.kv_cache)
    want, want_window, want_state = reference_decode(layer, qkvz[:rows], ba[:rows], slots[:rows].long())
    out = torch.empty(rows + 1, HV, V, dtype=torch.bfloat16, device=DEVICE)

    GDNCore.decode_step(layer, qkvz, ba, out, md)

    torch.cuda.synchronize()
    torch.testing.assert_close(out[:rows].double(), want, rtol=2e-2, atol=2e-2)
    assert torch.equal(out[rows], torch.zeros_like(out[rows]))
    torch.testing.assert_close(layer.kv_cache[1][slots[:rows].long()].double(), want_state, rtol=1e-4, atol=1e-5)
    window = layer.kv_cache[0][slots[:rows].long()]
    window = window.transpose(1, 2) if is_conv_state_dim_first() else window
    assert torch.equal(window.double(), want_window)
    # the padding row's slot keeps its states
    assert torch.equal(layer.kv_cache[0][0], conv_state[0]) and torch.equal(layer.kv_cache[1][0], ssm[0])


def two_rank_checks(rank: int, port: int) -> None:
    """The fused all-reduce + fp32 residual + RMSNorm, the partial-embedding norm and the logits gather on one rank."""
    os.environ.update(MASTER_ADDR=LOCALHOST, MASTER_PORT=str(port))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=device)
    group = dist.group.WORLD
    BlackwellExtension.load()
    # held like the model holds them: the extension keeps only the buffers' addresses
    AllReduceBuffers.create(group, MAX_TOKENS, HIDDEN, device)
    LogitsGatherBuffers.create(group, MAX_ROWS, VOCAB, device)
    gather = LogitsGatherBuffers.instance
    generator = torch.Generator(device=device).manual_seed(1234)  # the same draws on both ranks
    w = (0.1 * torch.randn(HIDDEN, generator=generator, device=device)).bfloat16()

    # a sequence of token counts: rows a larger call wrote must not leak into the later ones
    for T in (848, 64, 1, 1000, 7, 64, MAX_TOKENS, 300, 64):
        residual = torch.randn(T, HIDDEN, generator=generator, device=device) * 3
        partials = [torch.randn(T, HIDDEN, generator=generator, device=device).bfloat16() for _ in range(2)]
        out, new_residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
            partials[rank].clone(), residual, w, EPS, False
        )
        out32, _ = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(partials[rank].clone(), residual, w, EPS, True)
        want = residual.double() + partials[0].double() + partials[1].double()
        want_out = want * torch.rsqrt(want.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.double())
        torch.testing.assert_close(new_residual.double(), want, rtol=0, atol=1e-4)
        torch.testing.assert_close(out.double(), want_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out32.double(), want_out, rtol=1e-5, atol=1e-5)
        peer = new_residual.clone()
        dist.broadcast(peer, 0)
        assert torch.equal(peer, new_residual), "the ranks' residuals differ"

    # the first layer: each token's embedding row is nonzero on one rank, so the residual is exactly the full row
    for T in (64, 848, 7):
        full = torch.randn(T, HIDDEN, generator=generator, device=device).bfloat16()
        owner = torch.randint(0, 2, (T, 1), generator=generator, device=device)
        out, residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(torch.where(owner == rank, full, 0), w, EPS)
        assert torch.equal(residual, full.float())
        want_out = full.double() * torch.rsqrt(full.double().pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.double())
        torch.testing.assert_close(out.double(), want_out, rtol=1e-2, atol=1e-2)

    # the logits: this rank's vocabulary shard as high and low rows, summed and gathered into [n, vocab]
    shard = slice(rank * VOCAB // 2, (rank + 1) * VOCAB // 2)
    for n in (65, 1, MAX_ROWS, 33):
        high = torch.randn(n, VOCAB, generator=generator, device=device)
        low = torch.randn(n, VOCAB, generator=generator, device=device) * 1e-3
        logits = torch.ops.pear_rollout_blackwell.lg_gather(
            torch.cat([high[:, shard], low[:, shard]]).contiguous(), n, gather.rank
        )
        torch.cuda.synchronize()
        assert torch.equal(logits, high + low)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the fused all-reduce exchanges between two GPUs")
def test_the_fused_all_reduce_and_the_logits_gather_on_two_ranks():
    torch.multiprocessing.spawn(two_rank_checks, args=(get_free_port(LOCALHOST)[0],), nprocs=2)


def test_the_linear_layers_gemms_match_an_fp32_product():
    BlackwellExtension.load()
    GemmTableLinearMethod.load_table()
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    method = GemmTableLinearMethod()
    # the MLP gate/up at 896 rows (a measured shape), a shape the table leaves to candidate 0, a row-strided input
    for M, N, K, pad in ((896, 17408, HIDDEN, 0), (7, 8192, HIDDEN, 0), (64, HIDDEN, 3072, 8)):
        x = torch.randn(M, K + pad, generator=generator, device=DEVICE).bfloat16()[:, :K]
        w = (torch.randn(N, K, generator=generator, device=DEVICE) * K**-0.5).bfloat16()

        out = method.apply(types.SimpleNamespace(weight=w), x)

        assert out.dtype == torch.bfloat16 and torch.equal(out, torch.ops.pear_rollout_blackwell.lt_mm(x, w))
        torch.testing.assert_close(out.float(), F.linear(x.float(), w.float()), rtol=1e-2, atol=1e-2)


@pytest.fixture
def sampler(monkeypatch):
    # every test leaves vLLM's sampler as it found it
    monkeypatch.setattr(vllm_sampler.Sampler, "forward", vllm_sampler.Sampler.forward)


def sampling_metadata(rows: int, temperature: float) -> SamplingMetadata:
    """Random rows with temperature only and processed top-K log-probs, as the replicas sample."""
    zeros = torch.zeros(rows, device=DEVICE)
    return SamplingMetadata(
        temperature=torch.full((rows,), temperature, device=DEVICE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=BlackwellSampler.MAX_LOGPROBS,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=zeros,
        presence_penalties=zeros,
        repetition_penalties=zeros + 1,
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_the_fused_sampler_matches_vllms(sampler):
    vllm_forward = vllm_sampler.Sampler.forward
    BlackwellExtension.load()
    BlackwellSampler.install()
    module = vllm_sampler.Sampler(logprobs_mode="processed_logprobs")
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    # a step's rows, one row at a low temperature, all rows at a high one, integer logits (ties: radix select)
    for rows, temperature, scale, integers in (
        (65, 1.0, 3.0, False),
        (1, 0.7, 2.0, False),
        (MAX_ROWS, 1.3, 8.0, False),
        (8, 1.0, 2.0, True),
    ):
        logits = torch.randn(rows, VOCAB, generator=generator, device=DEVICE) * scale
        logits = logits.round() if integers else logits
        md = sampling_metadata(rows, temperature)
        assert BlackwellSampler.eligible(module, logits, md, False, module.logprobs_mode)

        torch.cuda.manual_seed(rows)  # the same noise for both
        want = vllm_forward(module, logits.clone(), md)
        torch.cuda.manual_seed(rows)
        got = module(logits.clone(), md)

        assert torch.equal(got.sampled_token_ids, want.sampled_token_ids)
        assert torch.equal(got.logprobs_tensors.selected_token_ranks, want.logprobs_tensors.selected_token_ranks)
        torch.testing.assert_close(got.logprobs_tensors.logprobs, want.logprobs_tensors.logprobs, rtol=0, atol=1e-4)
        if not integers:  # tied log-probs may list their tokens in another order
            assert torch.equal(got.logprobs_tensors.logprob_token_ids, want.logprobs_tensors.logprob_token_ids)


@pytest.fixture
def engine_config():
    # vLLM's custom ops (the rotary embedding) read the current engine config as they are built
    with set_current_vllm_config(VllmConfig()):
        yield


def test_the_rotary_embedding_rotates_in_fp32(engine_config):
    rope = get_rope(HEAD, ROPE_POSITIONS, rope_parameters=ROPE, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.rotary_emb = rope.to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    tokens = 300
    positions = torch.randint(0, ROPE_POSITIONS, (3, tokens), generator=generator, device=DEVICE)  # T / H / W
    q = torch.randn(tokens, Q_HEADS * HEAD, generator=generator, device=DEVICE).bfloat16()
    k = torch.randn(tokens, KV_HEADS * HEAD, generator=generator, device=DEVICE).bfloat16()
    bf16_q, _ = rope.forward_native(positions, q.clone(), k.clone())

    Fp32Rope.install(layer)

    want_q, want_k = rope.forward_native(positions, q.float(), k.float())  # fp32 throughout
    assert not torch.equal(bf16_q, want_q.bfloat16())
    # vLLM's path with custom ops off: rotated in fp32, rounded once
    got_q, got_k = rope.forward_native(positions, q.clone(), k.clone())
    assert torch.equal(got_q, want_q.bfloat16()) and torch.equal(got_k, want_k.bfloat16())
    # the mRoPE kernel (custom ops on): within one bf16 step
    got_q, got_k = rope.forward_cuda(positions, q.clone(), k.clone())
    torch.testing.assert_close(got_q, want_q.bfloat16(), rtol=2**-7, atol=0)
    torch.testing.assert_close(got_k, want_k.bfloat16(), rtol=2**-7, atol=0)
    assert rope.cos_sin_cache.dtype == torch.float32
