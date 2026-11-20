import pytest
import torch
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
SEED = 0
GATE_COLUMNS = 3072  # an attention layer's output columns on one of two ranks: 12 heads of 256
# [T, D] steps and the gates' scale: a decode step to a prefill chunk, gates far into the sigmoid's tails
SHAPES = [(256, GATE_COLUMNS, 1.0), (64, GATE_COLUMNS, 4.0), (16, GATE_COLUMNS, 12.0), (1, GATE_COLUMNS, 30.0)]
SHAPES += [(256, GATE_COLUMNS, 100.0), (13, 512, 3.0)]
INPUTS = (1.0, -3.5, 1e-30, 6e4)  # x against every finite bf16 gate: unit, negative, subnormal-scale and large inputs
BF16_PATTERNS = 1 << 16
VECTOR = 8  # bf16 per 16-byte access: the kernel's rows are whole vectors


@pytest.fixture(scope="module", autouse=True)
def extension():
    BlackwellExtension.load()


def gated_rows(T: int, D: int, gate_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """An attention output [T, D] and its gate, a strided view of wider rows (the kernel reads rows of any stride)."""
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    x = (torch.randn(T, D, generator=generator, device=DEVICE) * 2).bfloat16()
    gate = (torch.randn(T, 2 * D, generator=generator, device=DEVICE) * gate_scale).bfloat16()[:, D:]

    return x, gate


def rounded_once(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """x * sigmoid(gate) as torch computes it in fp32, rounded to bf16 once."""
    return (x.float() * torch.sigmoid(gate.float())).bfloat16()


@pytest.mark.parametrize("T, D, gate_scale", SHAPES)
def test_the_output_gate_is_torch_s_fp32_gate_rounded_once(T, D, gate_scale):
    x, gate = gated_rows(T, D, gate_scale)

    out = torch.ops.pear_rollout_blackwell.output_gate(x, gate)

    assert torch.equal(out.view(torch.int16), rounded_once(x, gate).view(torch.int16))


def test_torch_s_bf16_gate_rounds_twice():
    # the control: torch's bf16 chain rounds sigmoid(gate) to bf16 before the product, which the kernel does not
    x, gate = gated_rows(*SHAPES[0])

    out = torch.ops.pear_rollout_blackwell.output_gate(x, gate)

    assert not torch.equal(out, x * torch.sigmoid(gate))


@pytest.mark.parametrize("x_value", INPUTS)
def test_every_finite_bf16_gate_is_torch_s_fp32_gate_rounded_once(x_value):
    gate = torch.arange(BF16_PATTERNS, dtype=torch.int32, device=DEVICE).to(torch.int16).view(torch.bfloat16)
    gate = gate[torch.isfinite(gate.float())]
    gate = gate[: gate.numel() // VECTOR * VECTOR].view(1, -1).contiguous()
    x = torch.full_like(gate, x_value)

    out = torch.ops.pear_rollout_blackwell.output_gate(x, gate)

    assert torch.equal(out.view(torch.int16), rounded_once(x, gate).view(torch.int16))
