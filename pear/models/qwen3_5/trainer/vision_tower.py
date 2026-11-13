from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from pear.models.qwen3_5.trainer.ieee_fp32 import IEEEFp32


@triton.jit
def _split_bf16_kernel(X, HIGH, LOW, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    x = tl.load(X + offs, mask=m, other=0.0)
    high_part = x.to(tl.bfloat16)
    tl.store(HIGH + offs, high_part, mask=m)
    tl.store(LOW + offs, (x - high_part.to(tl.float32)).to(tl.bfloat16), mask=m)


class TowerLinear(torch.autograd.Function):
    """A linear layer of the vision tower by its backward, the stock one: fp32 GEMMs on the process's TF32 setting."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x2, weight = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])

        dx = torch.mm(dy2, weight).view(ctx.shape) if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy2.t(), x2) if ctx.needs_input_grad[1] else None
        db = dy2.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None

        return dx, dw, db


class Bf16x3Linear(TowerLinear):
    """y = x W^T + b for fp32 x, W split into bf16 high + low parts: x_high W_high + x_high W_low + x_low W_high."""

    SPLIT_ELEMENTS = 8192  # elements per program of the split kernel
    SPLIT_WARPS = 8  # its warps

    # the three products summed in fp32 on bf16 tensor cores
    @staticmethod
    def forward(ctx, x, weight, bias):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])

        x_high, x_low = Bf16x3Linear.split(x2)
        w_high, w_low = Bf16x3Linear.split(weight)

        if bias is not None:
            y = torch.addmm(bias, x_high, w_high.t(), out_dtype=torch.float32)
        else:
            y = torch.mm(x_high, w_high.t(), out_dtype=torch.float32)
        torch.addmm(y, x_high, w_low.t(), out_dtype=torch.float32, out=y)
        torch.addmm(y, x_low, w_high.t(), out_dtype=torch.float32, out=y)

        ctx.save_for_backward(x2, weight)
        ctx.has_bias, ctx.shape = bias is not None, shape

        return y.view(*shape[:-1], weight.shape[0])

    @staticmethod
    def split(x):
        """x (fp32) = high + low + O(2^-17 |x|) with high, low bf16."""
        x = x.contiguous()
        high_part = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        low_part = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)

        n = x.numel()
        _split_bf16_kernel[(triton.cdiv(n, Bf16x3Linear.SPLIT_ELEMENTS),)](
            x, high_part, low_part, n, Bf16x3Linear.SPLIT_ELEMENTS, num_warps=Bf16x3Linear.SPLIT_WARPS
        )

        return high_part, low_part


class Tf32Linear(TowerLinear):
    """F.linear on the process's fp32 setting (TF32) that saves what Bf16x3Linear saves."""

    # the recompute of a checkpointed block runs it where the forward ran Bf16x3Linear, and the checkpoint compares the
    # saved tensors' metadata between the two
    @staticmethod
    def forward(ctx, x, weight, bias):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = torch.nn.functional.linear(x2, weight, bias)

        ctx.save_for_backward(x2, weight)
        ctx.has_bias, ctx.shape = bias is not None, shape

        return y.view(*shape[:-1], weight.shape[0])


class VisionTower:
    """The fp32 vision tower's forward: its convolution in IEEE fp32 and its linear layers on bf16 splits."""

    # inside the tower's forward (not its checkpoints' recompute)
    in_forward: ContextVar = ContextVar("qwen3_5_training_kernels_vision_forward", default=False)

    @classmethod
    def install(cls, model) -> None:
        """Patch the tower's forward and its linear layers (its attention is the GPU backend's)."""
        # the forward's convolution in IEEE fp32 (cuDNN's TF32 off) and its linear layers on 3-term bf16 splits; the
        # backward and the recompute of the checkpointed blocks keep TF32; the module calls stay, for hooks
        visual = model.model.visual
        tower_forward = visual.forward

        def forward(*args, **kwargs):
            with IEEEFp32(torch.backends.cudnn):
                token = cls.in_forward.set(True)

                try:
                    return tower_forward(*args, **kwargs)
                finally:
                    cls.in_forward.reset(token)

        visual.forward = forward

        for module in visual.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = cls.linear_forward(module)

    @classmethod
    def linear_forward(cls, module):
        """The forward of one of the tower's linear layers."""

        def forward(x):
            linear = Bf16x3Linear if cls.in_forward.get() else Tf32Linear
            return linear.apply(x, module.weight, module.bias)

        return forward
