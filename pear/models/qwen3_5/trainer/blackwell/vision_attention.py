import ctypes
import os

import torch

from pear.models.qwen3_5.trainer.ieee_fp32 import IEEEFp32
from pear.models.qwen3_5.trainer.vision_attention import BatchedVisionAttention


class Bf16x9Matmul(IEEEFp32):
    """fp32 matmuls without TF32 inside the block: cuBLAS's BF16x9 emulation, its special-value scan off."""

    CUBLAS = "libcublas.so.13"  # the cuBLAS torch loaded (CUDA 13)
    SPECIAL_VALUES_NONE = 0  # cuBLAS's emulation special-value support mask: none scanned for
    SPECIAL_VALUES_ALL = 0xFFFF  # the mask with every special value supported, cuBLAS's default

    # with CUBLAS_EMULATE_SINGLE_PRECISION=1 (set before cuBLAS creates its handles) an fp32 GEMM that does not
    # allow TF32 runs as BF16x9 on tensor cores; the inputs are finite here, so the inf/NaN scan-and-patch kernels
    # are skipped
    library: list = []  # the cuBLAS library with cublasSetEmulationSpecialValuesSupport, or None

    def __init__(self) -> None:
        super().__init__(torch.backends.cuda.matmul)

    def __enter__(self):
        super().__enter__()
        Bf16x9Matmul.special_values(Bf16x9Matmul.SPECIAL_VALUES_NONE)

    def __exit__(self, *exc):
        super().__exit__(*exc)
        Bf16x9Matmul.special_values(Bf16x9Matmul.SPECIAL_VALUES_ALL)

    @classmethod
    def special_values(cls, mask: int) -> None:
        """Set the emulation's special-value support on the current stream's cuBLAS handle."""
        if not cls.library:
            library = ctypes.CDLL(cls.CUBLAS, mode=os.RTLD_NOLOAD)
            cls.library.append(library if hasattr(library, "cublasSetEmulationSpecialValuesSupport") else None)

        if cls.library[0] is not None:
            handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
            cls.library[0].cublasSetEmulationSpecialValuesSupport(handle, ctypes.c_int(mask))


class Fp32VisionAttention(torch.autograd.Function):
    """Non-causal attention of B images of S patches on fp32 GEMMs: q, k, v [B*S, H, D] fp32 -> out [B*S, H, D]."""

    SCORE_BYTES = 512 << 20  # fp32 attention scores per group of images: 8 images of 1024 patches x 16 heads

    # the GEMMs run without TF32 (Bf16x9Matmul); the images go in groups whose [images, H, S, S] scores stay within
    # SCORE_BYTES
    @staticmethod
    def forward(ctx, q, k, v, S, scale):
        T, H, D = q.shape
        B = T // S
        images = max(1, Fp32VisionAttention.SCORE_BYTES // (H * S * S * 4))
        queries, keys, values = (t.reshape(B, S, H, D).transpose(1, 2) for t in (q, k, v))

        out = torch.empty(B, S, H, D, device=q.device, dtype=torch.float32)
        probs = []

        with Bf16x9Matmul():
            for b in range(0, B, images):
                i = slice(b, b + images)
                # the scale on q ([b, H, S, D]) rather than on the [b, H, S, S] scores: one pass less over them
                p = torch.softmax(torch.matmul(queries[i] * scale, keys[i].transpose(-1, -2)), -1)
                out[i] = torch.matmul(p, values[i]).transpose(1, 2)
                probs.append(p)

        ctx.save_for_backward(q, k, v, out, *probs)
        ctx.S, ctx.scale, ctx.images = S, scale, images

        return out.view(T, H, D)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q, k, v, out, *probs = ctx.saved_tensors
        S, scale, images = ctx.S, ctx.scale, ctx.images
        T, H, D = q.shape
        B = T // S

        queries, keys, values = (t.reshape(B, S, H, D).transpose(1, 2) for t in (q, k, v))
        outputs = out.transpose(1, 2)
        do = do.reshape(B, S, H, D).transpose(1, 2)

        dq = torch.empty(B, S, H, D, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv = torch.empty_like(dq)

        with Bf16x9Matmul():
            for n, b in enumerate(range(0, B, images)):
                i = slice(b, b + images)
                p = probs[n]
                dv[i] = torch.matmul(p.transpose(-1, -2), do[i]).transpose(1, 2)
                dp = torch.matmul(do[i], values[i].transpose(-1, -2))
                delta = (do[i] * outputs[i]).sum(-1, keepdim=True)
                ds = p.mul_(dp.sub_(delta))  # p is not needed afterwards
                del dp
                dq[i] = torch.matmul(ds, keys[i]).mul_(scale).transpose(1, 2)
                dk[i] = torch.matmul(ds.transpose(-1, -2), queries[i]).mul_(scale).transpose(1, 2)
                probs[n] = None

        return dq.view(T, H, D), dk.view(T, H, D), dv.view(T, H, D), None, None


class BlackwellVisionAttention(BatchedVisionAttention):
    """Blackwell's fp32 vision attention: Fp32VisionAttention on BF16x9 GEMMs, one call per run of equal-size images."""

    RUN_COPIES = 8  # micro-batches whose runs of equal-size images stay cached

    copies: list = []  # (cu_seqlens, its runs of equal-size images), the most recent micro-batches first

    @classmethod
    def install(cls, attention_cls: type) -> None:
        """Route the attention class through it on every implementation, on cuBLAS's BF16x9 emulation of fp32 GEMMs."""
        # read by cuBLAS when it creates its handles, at the first matmul after the model is built
        os.environ.setdefault("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
        attention_cls.forward = cls.attention_forward

    @staticmethod
    def attention_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        """The attention class's forward: the family's, on Fp32VisionAttention and the cached runs."""
        return BlackwellVisionAttention.forward(self, hidden_states, cu_seqlens, position_embeddings)

    @classmethod
    def image_runs(cls, cu_seqlens) -> list[tuple[int, int]]:
        """The runs of equal-size images, one device-to-host copy per micro-batch (every block gets the same tensor)."""
        for tensor, runs in cls.copies:
            if tensor is cu_seqlens:
                return runs

        runs = super().image_runs(cu_seqlens)
        cls.copies.insert(0, (cu_seqlens, runs))
        del cls.copies[cls.RUN_COPIES :]

        return runs

    @staticmethod
    def attend(attention, query, key, value, size: int, count: int):
        """Fp32VisionAttention over a run of images of size patches."""
        return Fp32VisionAttention.apply(query, key, value, size, attention.scaling)
