import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.vision import VisionAllReduceBuffers, VisionGemmLinearMethod

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
HIDDEN = 1152  # the vision tower's hidden size, which its fused all-reduce is built for
EPS = 1e-6  # the tower's LayerNorm epsilon
# rows of x and (N, K) of the tower's linear layers on one of two ranks: qkv, the attention's projection, the MLP's two
# layers, the patch embedding
GEMM_ROWS = (1, 64, 1000)
GEMM_SHAPES = ((1728, 1152), (1152, 576), (2152, 1152), (1152, 2152), (1152, 1536))
# the emulated GEMM's error relative to sum |x w|: BF16x9's nine exact products summed in fp32; the SIMT SGEMM's
EMULATED_TOLERANCE = 1e-5
SIMT_TOLERANCE = 1e-5
# rows of the all-reduces: a frame's patches and more than one rotation of the receive buffers (two launches)
AR_ROWS = (1, 37, 1000, VisionAllReduceBuffers.ROWS + 100)
LAYERNORM_TOLERANCE = 1e-5  # the fused LayerNorm against fp64, relative to the output's scale


@pytest.mark.parametrize("rows", GEMM_ROWS)
@pytest.mark.parametrize("N, K", GEMM_SHAPES)
@pytest.mark.parametrize("is_emulated", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_the_vision_gemm_matches_fp64(rows, N, K, is_emulated, has_bias):
    BlackwellExtension.load()
    generator = torch.Generator(DEVICE).manual_seed(rows + N + K)
    x = torch.randn(rows, K, generator=generator, device=DEVICE)
    w = torch.randn(N, K, generator=generator, device=DEVICE) / K**0.5
    bias = torch.randn(N, generator=generator, device=DEVICE) if has_bias else None
    epilogue = VisionGemmLinearMethod.EPILOGUE_BIAS if has_bias else VisionGemmLinearMethod.EPILOGUE_NONE

    out = torch.ops.pear_rollout_blackwell.vision_mm(x, w, bias, epilogue, is_emulated)

    want = x.double() @ w.double().t() + (bias.double() if has_bias else 0)
    scale = x.double().abs() @ w.double().abs().t()
    tolerance = EMULATED_TOLERANCE if is_emulated else SIMT_TOLERANCE
    assert out.shape == (rows, N) and out.dtype == torch.float32
    assert ((out.double() - want).abs() <= tolerance * scale + 1e-6).all()


def all_reduce_checks(rank: int, rendezvous: str) -> None:
    """The vision tower's all-reduces on one of two ranks: fused with the residual add and the LayerNorm (the residual
    the fp32 sum of the residual and both partials, the norm within its error of fp64), and alone (the fp32 sum), at
    every row count, -0.0 among the partials."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, device_id=device)
    BlackwellExtension.load()
    VisionAllReduceBuffers.create(dist.group.WORLD, HIDDEN, device)
    generator = torch.Generator(device=device).manual_seed(5)  # the same draws on both ranks
    weight = 1 + 0.1 * torch.randn(HIDDEN, generator=generator, device=device)
    bias = 0.1 * torch.randn(HIDDEN, generator=generator, device=device)

    for T in AR_ROWS:
        residual = torch.randn(T, HIDDEN, generator=generator, device=device) * 3
        partials = [torch.randn(T, HIDDEN, generator=generator, device=device) for _ in range(2)]
        for partial in partials:
            partial[::7, ::5] = -0.0  # the receive buffers' "not arrived" mark, pushed as +0.0

        out, new_residual = torch.ops.pear_rollout_blackwell.vision_ar_add_layernorm(
            partials[rank].clone(), residual, weight, bias, EPS
        )
        reduced = torch.ops.pear_rollout_blackwell.vision_all_reduce(partials[rank].clone())

        want_residual = residual + (partials[0] + partials[1])
        assert torch.equal(new_residual, want_residual), (rank, T)
        assert torch.equal(reduced, partials[0] + partials[1]), (rank, T)
        want = F.layer_norm(want_residual.double(), (HIDDEN,), weight.double(), bias.double(), EPS)
        assert ((out.double() - want).abs() <= LAYERNORM_TOLERANCE * want.abs().amax()).all(), (rank, T)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="the vision all-reduces exchange between two GPUs")
def test_the_vision_all_reduces_on_two_ranks(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(all_reduce_checks, args=(str(tmp_path / "rendezvous"),), nprocs=2)
