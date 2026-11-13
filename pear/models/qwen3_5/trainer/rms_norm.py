from types import MethodType

import torch
import triton
import triton.language as tl
from liger_kernel.ops.rms_norm import _str_to_casting_mode, rms_norm_backward
from liger_kernel.ops.utils import calculate_settings
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, Fp16Operands, raise_range_flag

BACKWARD_ROWS = 4  # rows per program of the backward (the weight gradient's partial sums)
NORM_WARPS = 8  # warps of the norm kernels


@triton.jit
def _square_kernel(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    tl.store(S + i, x * x, i < N)


@triton.jit
def _normalize_kernel(X, W, VAR, Y, RSTD, N: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D

    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    var = tl.load(VAR + row, i < N, other=0)

    rstd = tl.rsqrt(var + EPS)
    y = (x * rstd) * (1.0 + w)

    tl.store(Y + i, y, i < N)
    tl.store(RSTD + row, rstd, (i < N) & (col == 0))


class PreciseRMSNormFunction(torch.autograd.Function):
    """Qwen3.5's (1 + w) RMSNorm with the stock forward rounding and Liger's backward kernel."""

    NORM_ELEMENTS = 1024  # elements per program of the two element-wise kernels

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        shape = x.shape
        x = x.contiguous()

        # the stock forward squares in fp32 and reduces with torch's own tree; keeping that reduction (and no fp fusion)
        # keeps its rounding (this model amplifies rounding-level changes in its early layers)
        squares = torch.empty_like(x, dtype=torch.float32)
        _square_kernel[(triton.cdiv(x.numel(), PreciseRMSNormFunction.NORM_ELEMENTS),)](
            x, squares, x.numel(), PreciseRMSNormFunction.NORM_ELEMENTS, enable_fp_fusion=False
        )
        variance = squares.mean(-1, keepdim=True)

        # the output in the autocast dtype the next linear layers compute in: the norms of an fp32 residual stream would
        # otherwise hand them fp32, which autocast casts once for each layer that reads it
        is_autocast = torch.is_autocast_enabled("cuda")
        out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda") if is_autocast else x.dtype)
        rstd = torch.empty(x.numel() // shape[-1], device=x.device, dtype=torch.float32)
        _normalize_kernel[(triton.cdiv(x.numel(), PreciseRMSNormFunction.NORM_ELEMENTS),)](
            x,
            weight,
            variance,
            out,
            rstd,
            x.numel(),
            shape[-1],
            eps,
            PreciseRMSNormFunction.NORM_ELEMENTS,
            enable_fp_fusion=False,
        )

        ctx.save_for_backward(x.view(-1, shape[-1]), weight, rstd)
        ctx.settings = calculate_settings(shape[-1])

        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        block, warps = ctx.settings

        # Liger allocates dx like dy: an fp32 input keeps an fp32 gradient
        dx, dw = rms_norm_backward(
            dy.to(x.dtype).contiguous(), x, weight, rstd, 1.0, _str_to_casting_mode["gemma"], block, warps, False, None
        )
        return dx, dw, None


class PreciseRMSNorm:
    """The precise RMSNorm as the forward of a Qwen3.5 text model's norms."""

    @classmethod
    def install(cls, text_model) -> None:
        """Install the precise RMSNorm on the text model's layer norms and final norm (q/k norms stay stock)."""
        norms = [text_model.norm]

        for layer in text_model.layers:
            norms.extend((layer.input_layernorm, layer.post_attention_layernorm))

        for norm in norms:
            norm.forward = MethodType(cls.forward, norm)

    @staticmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PreciseRMSNormFunction.apply(x, self.weight, self.eps)


@triton.jit
def _rms_fwd_kernel(
    X,
    B,
    W,
    X1,
    Y,
    RSTD,
    Y16,
    REMAINDER,
    FLAG,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ADD: tl.constexpr,
    HAS_Y: tl.constexpr,
    HAS_Y16: tl.constexpr,
    HAS_REMAINDER: tl.constexpr,
    LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N

    x = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_ADD:
        x = x + tl.load(B + row * N + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(X1 + row * N + cols, x, mask=m)

    var = tl.sum(x * x, 0) / N
    rstd = tl.rsqrt(var + EPS)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    y = (x * rstd) * (1.0 + w)

    if HAS_Y:
        tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=m)
    if HAS_Y16:  # the fp16 twin, rounded once from the same fp32 value
        y16 = y.to(tl.float16)
        tl.store(Y16 + row * N + cols, y16, mask=m)
        if HAS_REMAINDER:  # the rounding's remainder, fp16 too
            tl.store(REMAINDER + row * N + cols, (y - y16.to(tl.float32)).to(tl.float16), mask=m)
        raise_range_flag(FLAG, y, m, LIMIT)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_bwd_kernel(
    DY,
    X,
    RSTD,
    W,
    DX_IN,
    DX,
    DB,
    DWP,
    M,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_DX_IN: tl.constexpr,
    HAS_DB: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    w1 = 1.0 + tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK,), dtype=tl.float32)

    for i in range(ROWS):
        row = pid.to(tl.int64) * ROWS + i
        if row < M:
            dy = tl.load(DY + row * N + cols, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
            rstd = tl.load(RSTD + row)
            x_hat = x * rstd
            wdy = dy * w1
            dx = rstd * (wdy - x_hat * (tl.sum(wdy * x_hat, 0) / N))
            if HAS_DX_IN:
                dx += tl.load(DX_IN + row * N + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(DX + row * N + cols, dx.to(DX.dtype.element_ty), mask=m)
            if HAS_DB:
                tl.store(DB + row * N + cols, dx.to(DB.dtype.element_ty), mask=m)
            dw += dy * x_hat

    tl.store(DWP + pid.to(tl.int64) * N + cols, dw, mask=m)


class RMSNorm(torch.autograd.Function):
    """Qwen3.5's y = x * rstd * (1 + w) on an fp32 x in one pass: (y in out_dtype, its fp16 twin, the remainder)."""

    # the stock norm's math (variance and normalization of the fp32 input in fp32, one rounding of the output) in one
    # pass instead of three; without autograd and with fp16 operands on, the fp16 output is the only one (the handle)
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype, wants_fp16=False, wants_remainder=False):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape

        y, y16, remainder = Fp16Operands.output_buffers(ctx, (M, N), out_dtype, x.device, wants_fp16, wants_remainder)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        _rms_fwd_kernel[(M,)](
            x2,
            x2,
            weight,
            x2,
            y if y is not None else y16,
            rstd,
            y16 if y16 is not None else rstd,
            remainder if remainder is not None else rstd,
            Fp16Operands.range_flag(x.device),
            N,
            eps,
            triton.next_power_of_2(N),
            False,
            y is not None,
            y16 is not None,
            remainder is not None,
            FP16_LIMIT,
            num_warps=NORM_WARPS,
        )

        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape

        outputs = Fp16Operands.twin_outputs(ctx, y, y16, remainder)
        return tuple(None if output is None else output.view(shape) for output in outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy, *_twins):
        x2, weight, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy = dy.reshape(M, N).contiguous()

        dx = torch.empty(M, N, device=x2.device, dtype=x2.dtype)
        programs = triton.cdiv(M, BACKWARD_ROWS)
        dw_parts = torch.empty(programs, N, device=x2.device, dtype=torch.float32)

        _rms_bwd_kernel[(programs,)](
            dy,
            x2,
            rstd,
            weight,
            dy,
            dx,
            dx,
            dw_parts,
            M,
            N,
            triton.next_power_of_2(N),
            BACKWARD_ROWS,
            False,
            False,
            num_warps=NORM_WARPS,
        )

        return dx.view(ctx.shape), dw_parts.sum(0).to(weight.dtype), None, None, None, None

    @staticmethod
    def autocast_dtype() -> torch.dtype:
        """The dtype the next linear layers compute in: the autocast dtype, fp32 outside autocast."""
        return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32


class AddRMSNorm(torch.autograd.Function):
    """x1 = x + b in fp32 and y = norm(x1) in one pass: (x1, y in out_dtype, y's fp16 twin)."""

    # without autograd and with fp16 operands on, y is the fp16 output itself; the backward folds the residual's
    # gradient into the norm's in one pass
    @staticmethod
    def forward(ctx, x, b, weight, eps, out_dtype, wants_fp16):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        b2 = b.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape

        x1 = torch.empty(M, N, device=x.device, dtype=torch.float32)
        y, y16, _ = Fp16Operands.output_buffers(ctx, (M, N), out_dtype, x.device, wants_fp16, False)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        _rms_fwd_kernel[(M,)](
            x2,
            b2,
            weight,
            x1,
            y if y is not None else y16,
            rstd,
            y16 if y16 is not None else rstd,
            rstd,
            Fp16Operands.range_flag(x.device),
            N,
            eps,
            triton.next_power_of_2(N),
            True,
            y is not None,
            y16 is not None,
            False,
            FP16_LIMIT,
            num_warps=NORM_WARPS,
        )

        ctx.save_for_backward(x1, weight, rstd)
        ctx.shape, ctx.b_dtype = shape, b.dtype

        y, y16, _ = Fp16Operands.twin_outputs(ctx, y, y16, None)
        return x1.view(shape), y.view(shape), None if y16 is None else y16.view(shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dx1, dy, _twin):
        x1, weight, rstd = ctx.saved_tensors
        M, N = x1.shape
        dy = dy.reshape(M, N).contiguous()

        dx = torch.empty(M, N, device=x1.device, dtype=torch.float32)
        db = torch.empty(M, N, device=x1.device, dtype=ctx.b_dtype)
        programs = triton.cdiv(M, BACKWARD_ROWS)
        dw_parts = torch.empty(programs, N, device=x1.device, dtype=torch.float32)

        _rms_bwd_kernel[(programs,)](
            dy,
            x1,
            rstd,
            weight,
            dx1.reshape(M, N).contiguous(),
            dx,
            db,
            dw_parts,
            M,
            N,
            triton.next_power_of_2(N),
            BACKWARD_ROWS,
            True,
            True,
            num_warps=NORM_WARPS,
        )

        return dx.view(ctx.shape), db.view(ctx.shape), dw_parts.sum(0).to(weight.dtype), None, None, None


class DecoderLayer:
    """Qwen3.5's decoder layer with its two RMSNorms fused on the fp32 residual stream, their fp16 twins registered."""

    @staticmethod
    def install(model, gpu_backend) -> None:
        """Patch the decoder layers' forward; each layer holds the GPU backend."""
        Qwen3_5DecoderLayer.forward = DecoderLayer.forward

        for layer in model.model.language_model.layers:
            layer.gpu_backend = gpu_backend

    @staticmethod
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        cu_seqlens_cpu = kwargs.pop("cu_seqlens_cpu", None)
        norm = self.input_layernorm
        out_dtype = RMSNorm.autocast_dtype()

        # the input norm feeds the projections of the attention or the Gated DeltaNet: the twin and its remainder
        # (the Gated DeltaNet's as the GPU backend says)
        wants_remainder = Fp16Operands.wants_remainder() and (
            self.layer_type != "linear_attention" or self.gpu_backend.gdn_wants_remainder
        )
        h, h16, remainder = RMSNorm.apply(
            hidden_states, norm.weight, norm.eps, out_dtype, Fp16Operands.is_enabled, wants_remainder
        )
        Fp16Operands.register(h, h if h.dtype == torch.float16 else h16, remainder)

        try:
            if self.layer_type == "linear_attention":
                out = self.linear_attn(
                    hidden_states=h,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
            else:
                out, _ = self.self_attn(
                    hidden_states=h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
        finally:
            Fp16Operands.release(h)
        del h, h16, remainder

        norm = self.post_attention_layernorm
        hidden_states, y, y16 = AddRMSNorm.apply(
            hidden_states, out.contiguous(), norm.weight, norm.eps, out_dtype, Fp16Operands.is_enabled
        )
        del out
        Fp16Operands.register(y, y16)

        try:
            out = self.mlp(y)
        finally:
            Fp16Operands.release(y)
        del y, y16

        return hidden_states + out
