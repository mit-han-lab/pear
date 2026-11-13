from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial

import torch
import triton
import triton.language as tl

from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)

FP16_LIMIT = 65000.0  # |x| above this would not survive the rounding to fp16 (largest finite value 65504)


@triton.jit
def raise_range_flag(FLAG, y, mask, LIMIT: tl.constexpr):
    # the device's flag raised where a value of y (one or two dimensions) under the mask is past LIMIT
    magnitudes = tl.where(mask, tl.abs(y), 0.0)
    if len(magnitudes.shape) == 2:
        magnitudes = tl.max(magnitudes, 1)

    out_of_range = tl.max(magnitudes, 0) > LIMIT
    tl.atomic_max(FLAG, out_of_range.to(tl.int32))


@triton.jit
def _to16_kernel(X, Y, FLAG, N, SCALE, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    v = tl.load(X + offs, mask=m, other=0.0).to(tl.float32) * SCALE
    tl.store(Y + offs, v.to(tl.float16), mask=m)

    raise_range_flag(FLAG, v, m, LIMIT)


class RecomputeFlag:
    """Whether the code runs in an activation checkpoint's recompute, which only regenerates the backward's inputs."""

    in_recompute: ContextVar = ContextVar("qwen3_5_training_kernels_recompute", default=False)
    # the checkpoint's first forward: its outputs count, while what it saves for the backward is discarded (only the
    # metadata is compared with the recompute's); inside an autograd Function's forward grad mode is off in both passes
    in_checkpointed_forward: ContextVar = ContextVar("qwen3_5_training_kernels_checkpointed_forward", default=False)

    @classmethod
    def install(cls, model) -> None:
        """Set the flags in every decoder layer's checkpoint, around the contexts it already has."""
        for layer in model.model.language_model.layers:
            checkpoint_fn = layer._gradient_checkpointing_func
            options = dict(checkpoint_fn.keywords)
            gpu_backend = layer.gpu_backend

            # verl's attention retention replays the first forward's attention outputs, which a bf16 recompute replaces
            if options.get("context_fn") is not None and not gpu_backend.recompute_is_fp16:
                raise ValueError(f"{gpu_backend.__name__} requires gradient_checkpointing_retain_attention=False")

            options["context_fn"] = partial(cls._contexts, options.get("context_fn"))
            layer._gradient_checkpointing_func = partial(checkpoint_fn.func, *checkpoint_fn.args, **options)

    @classmethod
    def saves_discarded(cls) -> bool:
        """Whether what an autograd Function saves now is discarded: in a checkpointed layer's first forward."""
        return cls.in_checkpointed_forward.get() and not cls.in_recompute.get()

    @staticmethod
    def placeholder(shape, dtype, device) -> torch.Tensor:
        """A tensor with the metadata and no memory, saved in place of one the checkpoint discards."""
        return torch.empty((), dtype=dtype, device=device).expand(shape)

    @classmethod
    def _contexts(cls, context_fn):
        """The checkpoint's (forward, recompute) contexts, each with its flag set."""
        forward_context, recompute_context = context_fn() if context_fn is not None else (nullcontext(), nullcontext())
        return cls._flagged(cls.in_checkpointed_forward, forward_context), cls._flagged(
            cls.in_recompute, recompute_context
        )

    @staticmethod
    @contextmanager
    def _flagged(flag, context):
        token = flag.set(True)

        try:
            with context:
                yield
        finally:
            flag.reset(token)


class Fp16Operands:
    """fp16 twins of the forward's GEMM operands, written next to the bf16 activations autograd sees."""

    WEIGHT_SCALE = 256.0  # 2^8: fp16 holds every bf16 weight times it exactly for magnitudes 2^-25 .. 255
    TO16_ELEMENTS = 4096  # elements per program of the conversion kernel
    TO16_WARPS = 8  # its warps

    # fp16 has 11 significant bits where bf16 has 8; every conversion raises a device flag past FP16_LIMIT, and
    # run_checked then reruns the batch on bf16 operands. Module calls take one tensor, so a twin travels in a registry
    # keyed by the bf16 tensor.
    is_enabled = True  # False while a batch reruns on bf16 operands
    range_flags: dict = {}  # device -> int32 flag
    twins: dict = {}  # id(x) -> (x, its fp16 twin, the twin's rounding remainder in fp16 or None)

    @classmethod
    def run_checked(cls, run, before_rerun):
        """run() on fp16 operands; if any operand left fp16's range, before_rerun() and run() again on bf16 operands."""
        cls.reset()
        out = run()

        if any(bool(flag.item()) for flag in cls.range_flags.values()):
            logger.warning("an fp16 operand left fp16's range; recomputing the batch on bf16 operands")
            before_rerun()
            cls.is_enabled = False

            try:
                out = run()
            finally:
                cls.is_enabled = True
                cls.reset()

        return out

    @classmethod
    def reset(cls) -> None:
        for flag in cls.range_flags.values():
            flag.zero_()

    @classmethod
    def range_flag(cls, device) -> torch.Tensor:
        """The device's range flag, set by any conversion whose value left fp16's range."""
        if device not in cls.range_flags:
            cls.range_flags[device] = torch.zeros(1, dtype=torch.int32, device=device)
        return cls.range_flags[device]

    @classmethod
    def wants_remainder(cls) -> bool:
        """Whether producers also write the remainder of the rounding (the first forward; the recompute does not)."""
        # the twin and its remainder: the projections add a second GEMM on fp16(x - fp16(x)), x to ~22 significant bits
        return cls.is_enabled and not RecomputeFlag.in_recompute.get()

    @staticmethod
    def output_buffers(ctx, shape, dtype, device, wants_fp16: bool, wants_remainder: bool):
        """The buffers a producer's kernel writes: (output in dtype, fp16 twin, remainder), None for one it skips."""
        has_output = any(ctx.needs_input_grad) or not wants_fp16
        output = torch.empty(shape, device=device, dtype=dtype) if has_output else None
        output16 = torch.empty(shape, device=device, dtype=torch.float16) if wants_fp16 else None
        remainder = torch.empty(shape, device=device, dtype=torch.float16) if wants_fp16 and wants_remainder else None
        return output, output16, remainder

    @staticmethod
    def twin_outputs(ctx, output, output16, remainder):
        """The buffers as a producer's forward returns them: (the one autograd sees, its fp16 twin, the remainder)."""
        if remainder is not None:
            ctx.mark_non_differentiable(remainder)
        if output is None:
            return output16, None, remainder
        if output16 is not None:
            ctx.mark_non_differentiable(output16)
        return output, output16, remainder

    @classmethod
    def register(cls, x, x16, remainder=None) -> None:
        """Attach a twin (and remainder) to x until release(x)."""
        if x16 is not None:
            cls.twins[id(x)] = (x, x16, remainder)

    @classmethod
    def twin(cls, x):
        """The fp16 twin of x, x itself when it is fp16 (a pass without autograd hands fp16 on), else None."""
        if x.dtype == torch.float16:
            return x

        entry = cls.entry(x)
        return None if entry is None else entry[1].view(x.shape)

    @classmethod
    def release(cls, x) -> None:
        cls.twins.pop(id(x), None)

    @classmethod
    def to16(cls, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """x * scale rounded once to fp16, with the range flag."""
        x = x.contiguous()
        y = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        n = x.numel()

        if n:
            _to16_kernel[(triton.cdiv(n, cls.TO16_ELEMENTS),)](
                x, y, cls.range_flag(x.device), n, scale, FP16_LIMIT, cls.TO16_ELEMENTS, num_warps=cls.TO16_WARPS
            )

        return y

    @classmethod
    def weight16(cls, w: torch.Tensor) -> torch.Tensor:
        """The weight's bf16 value times 2^8 in fp16 (exact); mm16 divides the scale out."""
        return cls.to16(w.to(torch.bfloat16), cls.WEIGHT_SCALE)

    @staticmethod
    def mm16(x16: torch.Tensor, w16: torch.Tensor, remainder: torch.Tensor = None) -> torch.Tensor:
        """x16 @ (w16 / 2^8)^T summed in fp32 (alpha divides the scale out), plus remainder @ (w16 / 2^8)^T."""
        out = torch.empty(x16.shape[0], w16.shape[0], device=x16.device, dtype=torch.float32)
        torch.addmm(out, x16, w16.t(), beta=0, alpha=1.0 / Fp16Operands.WEIGHT_SCALE, out_dtype=torch.float32, out=out)
        if remainder is not None:
            torch.addmm(
                out, remainder, w16.t(), beta=1, alpha=1.0 / Fp16Operands.WEIGHT_SCALE, out_dtype=torch.float32, out=out
            )
        return out

    @classmethod
    def entry(cls, x):
        """What is registered with x or with a tensor x is a view of: (the tensor, its twin, the remainder), or None."""
        entry = cls.twins.get(id(x))
        if entry is not None and entry[0] is x:
            return entry

        # a view of a registered tensor (same storage and size): the registry holds the registered tensors alive, so a
        # match cannot be a later tensor reusing their memory
        for entry in cls.twins.values():
            base = entry[0]
            if base.data_ptr() == x.data_ptr() and base.numel() == x.numel() and base.dtype == x.dtype:
                return entry

        return None


class Fp32OutputLinear(torch.autograd.Function):
    """x @ W^T summed in fp32 and left in fp32, on fp16 operands where x has a twin; the backward on bf16 operands."""

    @staticmethod
    def install(module) -> None:
        """Route the linear layer's calls through it (the module call stays, for hooks)."""

        def forward(x):
            return Fp32OutputLinear.apply(x, module.weight)

        module.forward = forward

    @staticmethod
    def forward(ctx, x, weight):
        with torch.autocast("cuda", enabled=False):
            w_bf16 = weight.to(torch.bfloat16)
            x16 = Fp16Operands.twin(x) if Fp16Operands.is_enabled else None
            if x16 is None and Fp16Operands.is_enabled and x.dtype == torch.float32:
                x16 = Fp16Operands.to16(x)

            if x16 is not None:
                entry = Fp16Operands.entry(x)
                remainder = None if entry is None or entry[2] is None else entry[2].view(x.shape)
                y = Fp16Operands.mm16(
                    x16.reshape(-1, x16.shape[-1]),
                    Fp16Operands.weight16(w_bf16),
                    remainder=None if remainder is None else remainder.reshape(-1, remainder.shape[-1]),
                )
            else:
                x_bf16 = x.to(torch.bfloat16)
                y = torch.mm(x_bf16.reshape(-1, x_bf16.shape[-1]), w_bf16.t(), out_dtype=torch.float32)

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x.to(torch.bfloat16), w_bf16)
            ctx.weight_dtype, ctx.x_dtype = weight.dtype, x.dtype

        return y.view(*x.shape[:-1], w_bf16.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x_bf16, w_bf16 = ctx.saved_tensors

        with torch.autocast("cuda", enabled=False):
            dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
            dx = torch.mm(dy2, w_bf16).view(x_bf16.shape).to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
            dw = None
            if ctx.needs_input_grad[1]:
                dw = torch.mm(dy2.t(), x_bf16.reshape(-1, x_bf16.shape[-1])).to(ctx.weight_dtype)

        return dx, dw
