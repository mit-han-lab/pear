import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.mlp import Qwen3_5RolloutMLP

CAPABILITY = current_platform.get_device_capability()
# the GPUs of the rollout kernels' backends
pytestmark = pytest.mark.skipif(
    not (
        BlackwellRolloutGpuBackend.supports_compute_capability(CAPABILITY)
        or HopperRolloutGpuBackend.supports_compute_capability(CAPABILITY)
    ),
    reason="the rollout kernels' GPUs",
)

DEVICE = "cuda"
# the MLP's intermediate channels on one rank of a GB300 replica (TP2) and of an H200 replica (TP4)
RANK_INTERMEDIATE = [pytest.param(8704, id="tp2"), pytest.param(4352, id="tp4")]
TOKENS = 300
FP32_SLACK = 2**-18  # how far fp32 arithmetic may land from fp64, relative to the magnitude of the result


@pytest.fixture
def replica_ops():
    # vLLM's custom ops as a replica's engine runs them: no torch.compile, every custom op on
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


def beyond_one_rounding(got: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Where got lies farther from the exact value than fp32 arithmetic rounded once to bf16 can."""
    _, exponent = torch.frexp(exact)
    half_step = torch.ldexp(torch.ones_like(exact), exponent - 9)  # |exact| in [2^(e-1), 2^e): bf16's step 2^(e-8)

    return (got.double() - exact).abs() > half_step + FP32_SLACK * exact.abs()


@pytest.mark.usefixtures("replica_ops")
@pytest.mark.parametrize("intermediate", RANK_INTERMEDIATE)
def test_silu_of_the_gate_times_up_is_rounded_once(intermediate):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    gate_up = torch.randn(TOKENS, 2 * intermediate, generator=generator, device=DEVICE).bfloat16()
    # the projections stand in: the MLP's own forward takes the gate / up rows and hands its product down
    mlp = Qwen3_5RolloutMLP.__new__(Qwen3_5RolloutMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = lambda x: (gate_up, None)
    mlp.down_proj = lambda product: (product, None)

    got = mlp(None)

    gate, up = gate_up.double().chunk(2, dim=-1)
    exact = torch.nn.functional.silu(gate) * up
    assert not beyond_one_rounding(got, exact).any()
    # the control: vLLM's kernel, silu(gate) rounded to bf16 before the product
    assert beyond_one_rounding(SiluAndMul()(gate_up), exact).any()
