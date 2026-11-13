import torch
import triton
import triton.language as tl

from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, Fp16Operands, RecomputeFlag, raise_range_flag


@triton.jit
def _swiglu_fwd_kernel(
    AB,
    H,
    ABQ,
    H16,
    FLAG,
    INTER,
    HAS_H: tl.constexpr,
    SAVE: tl.constexpr,
    HAS_H16: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < INTER

    a = tl.load(AB + row * 2 * INTER + cols, mask=m, other=0.0)
    b = tl.load(AB + row * 2 * INTER + INTER + cols, mask=m, other=0.0)
    h = a * tl.sigmoid(a) * b

    if HAS_H:
        tl.store(H + row * INTER + cols, h.to(H.dtype.element_ty), mask=m)
    if HAS_H16:  # the down projection's fp16 operand, rounded once from the same fp32 value
        tl.store(H16 + row * INTER + cols, h.to(tl.float16), mask=m)
        raise_range_flag(FLAG, h, m, LIMIT)
    if SAVE:
        tl.store(ABQ + row * 2 * INTER + cols, a.to(ABQ.dtype.element_ty), mask=m)
        tl.store(ABQ + row * 2 * INTER + INTER + cols, b.to(ABQ.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_bwd_kernel(DH, ABQ, DAB, INTER, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < INTER

    dh = tl.load(DH + row * INTER + cols, mask=m, other=0.0).to(tl.float32)
    a = tl.load(ABQ + row * 2 * INTER + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(ABQ + row * 2 * INTER + INTER + cols, mask=m, other=0.0).to(tl.float32)

    sig = tl.sigmoid(a)
    silu = a * sig
    db = dh * silu
    da = dh * (silu * (1.0 - sig) + sig) * b

    tl.store(DAB + row * 2 * INTER + cols, da.to(DAB.dtype.element_ty), mask=m)
    tl.store(DAB + row * 2 * INTER + INTER + cols, db.to(DAB.dtype.element_ty), mask=m)


class Fp32SwiGLUMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) with fp32 gate/up outputs, one rounding of the product and an fp32 output."""

    CHUNK_ROWS = 16384  # rows per gate/up GEMM: the chunk's fp32 [rows, 2 * intermediate] output stays ~2 GB
    SWIGLU_CHANNELS = 2048  # intermediate channels per program of the SwiGLU kernels
    SWIGLU_WARPS = 8  # warps of the SwiGLU kernels

    @staticmethod
    def install(model) -> None:
        """Every decoder layer's MLP, in place of Liger's SwiGLU (the projections' weights are read at call time)."""
        for layer in model.model.language_model.layers:
            layer.mlp.forward = Fp32SwiGLUMLP.mlp_forward(layer.mlp)

    @staticmethod
    def mlp_forward(mlp):
        """The forward of one decoder layer's MLP."""

        def forward(x):
            return Fp32SwiGLUMLP.apply(x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)

        return forward

    # the gate and up projections are one GEMM over row chunks whose fp32 output feeds silu(gate) * up in fp32, rounded
    # once for the down projection, whose fp32 output joins the residual stream; the forward's GEMMs take x's fp16 twin
    # and the product in fp16. The backward is Liger's: bf16 operands (gate/up stored as Liger stores them), fp32 sums.
    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down):
        shape = x.shape
        T = x.numel() // shape[-1]
        inter = w_gate.shape[0]
        needs_backward = any(ctx.needs_input_grad)
        in_recompute = RecomputeFlag.in_recompute.get()

        x16 = Fp16Operands.twin(x) if Fp16Operands.is_enabled else None
        if x16 is None and Fp16Operands.is_enabled and x.dtype == torch.float32:
            x16 = Fp16Operands.to16(x)
        w_cat = torch.cat((w_gate, w_up)).to(torch.bfloat16)
        if x16 is not None:
            x_op, w_op = x16.reshape(T, shape[-1]), Fp16Operands.weight16(w_cat)
        else:
            x_op, w_op = x.reshape(T, shape[-1]).to(torch.bfloat16), w_cat
        del w_cat

        # the bf16 product is saved for the backward or is the bf16 down projection's operand
        h = torch.empty(T, inter, device=x.device, dtype=torch.bfloat16) if needs_backward or x16 is None else None
        has_h16 = x16 is not None and not in_recompute
        h16 = torch.empty(T, inter, device=x.device, dtype=torch.float16) if has_h16 else None
        abq = torch.empty(T, 2 * inter, device=x.device, dtype=torch.bfloat16) if needs_backward else None
        placeholder = h if h is not None else h16

        for start in range(0, T, Fp32SwiGLUMLP.CHUNK_ROWS):
            rows = slice(start, start + Fp32SwiGLUMLP.CHUNK_ROWS)
            if x16 is not None:
                ab = Fp16Operands.mm16(x_op[rows], w_op)
            else:
                ab = torch.mm(x_op[rows], w_op.t(), out_dtype=torch.float32)
            _swiglu_fwd_kernel[(ab.shape[0], triton.cdiv(inter, Fp32SwiGLUMLP.SWIGLU_CHANNELS))](
                ab,
                h[rows] if h is not None else placeholder,
                abq[rows] if needs_backward else placeholder,
                h16[rows] if h16 is not None else placeholder,
                Fp16Operands.range_flag(x.device),
                inter,
                h is not None,
                needs_backward,
                h16 is not None,
                FP16_LIMIT,
                Fp32SwiGLUMLP.SWIGLU_CHANNELS,
                num_warps=Fp32SwiGLUMLP.SWIGLU_WARPS,
            )
            del ab
        del w_op

        # the checkpoint's recompute only regenerates the saved tensors and discards the output: no down projection
        if in_recompute:
            y = torch.empty(T, w_down.shape[0], device=x.device, dtype=torch.float32)
        elif h16 is not None:
            y = Fp16Operands.mm16(h16, Fp16Operands.weight16(w_down))
        else:
            y = torch.mm(h, w_down.to(torch.bfloat16).t(), out_dtype=torch.float32)
        del h16

        if needs_backward:
            ctx.save_for_backward(x.reshape(T, shape[-1]).to(torch.bfloat16), w_gate, w_up, w_down, abq, h)
        ctx.shape, ctx.x_dtype = shape, x.dtype

        return y.view(*shape[:-1], w_down.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x2, w_gate, w_up, w_down, abq, h = ctx.saved_tensors
        T, inter = h.shape

        dy = dy.reshape(T, -1).to(torch.bfloat16)
        dh = torch.mm(dy, w_down.to(torch.bfloat16))
        dw_down = torch.mm(dy.t(), h)
        del h

        dab = torch.empty(T, 2 * inter, device=dy.device, dtype=torch.bfloat16)
        _swiglu_bwd_kernel[(T, triton.cdiv(inter, Fp32SwiGLUMLP.SWIGLU_CHANNELS))](
            dh, abq, dab, inter, Fp32SwiGLUMLP.SWIGLU_CHANNELS, num_warps=Fp32SwiGLUMLP.SWIGLU_WARPS
        )
        del dh

        w_cat = torch.cat((w_gate, w_up)).to(torch.bfloat16)
        dx = torch.mm(dab, w_cat)
        dw_cat = torch.mm(dab.t(), x2)

        return (
            dx.view(ctx.shape).to(ctx.x_dtype),
            dw_cat[:inter].to(w_gate.dtype),
            dw_cat[inter:].to(w_up.dtype),
            dw_down.to(w_down.dtype),
        )
