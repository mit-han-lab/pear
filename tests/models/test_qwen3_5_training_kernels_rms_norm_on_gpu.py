import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def test_precise_rms_norm_matches_the_stock_forward_bitwise():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    from pear.models.qwen3_5.trainer.rms_norm import PreciseRMSNormFunction

    torch.manual_seed(0)
    norm = Qwen3_5RMSNorm(5120, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    x = torch.randn(64, 5120, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = norm(x)
    actual = PreciseRMSNormFunction.apply(x, norm.weight, norm.eps)
    assert torch.equal(actual, expected)

    grad = torch.randn_like(expected)
    dx_expected, dw_expected = torch.autograd.grad(expected, (x, norm.weight), grad)
    dx, dw = torch.autograd.grad(actual, (x, norm.weight), grad)
    # the backward is Liger's kernel fed the exact rstd: only bf16 rounding separates it from autograd's
    assert torch.allclose(dx.float(), dx_expected.float(), rtol=2e-2, atol=1e-2)
    assert torch.allclose(dw, dw_expected, rtol=2e-2, atol=1e-2)


def test_precise_rms_norm_of_an_fp32_stream_hands_the_autocast_dtype_on():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    from pear.models.qwen3_5.trainer.rms_norm import PreciseRMSNormFunction

    torch.manual_seed(0)
    norm = Qwen3_5RMSNorm(5120, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    weight = norm.weight.detach().bfloat16().requires_grad_()  # the weight FSDP gathers in param_dtype
    x = torch.randn(64, 5120, device="cuda", requires_grad=True)  # the fp32 residual stream
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = PreciseRMSNormFunction.apply(x, weight, norm.eps)
    # the stock norm's fp32 output, as autocast would cast it for the next linear layer
    stock = Qwen3_5RMSNorm(5120, eps=1e-6).cuda()
    with torch.no_grad():
        stock.weight.copy_(weight)
    expected = stock(x)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected.bfloat16())

    grad = torch.randn_like(actual)
    (dx,) = torch.autograd.grad(actual, x, grad)
    (dx_expected,) = torch.autograd.grad(expected, x, grad.float())
    assert dx.dtype == torch.float32
    assert torch.allclose(dx, dx_expected, rtol=1e-3, atol=1e-4)
