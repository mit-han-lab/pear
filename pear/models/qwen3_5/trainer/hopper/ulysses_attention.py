import torch
import torch.distributed as dist
import triton

from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, Fp16Operands, RecomputeFlag
from pear.models.qwen3_5.trainer.ulysses_attention import (
    PACK_TOKENS,
    PACK_WARPS,
    TrainingUlyssesAttention,
    pack_rope_kernel,
    unpack_rope_kernel,
)
from verl.utils.ulysses import all_to_all_tensor


class HopperUlyssesAttention(TrainingUlyssesAttention):
    """Ulysses causal attention on Hopper: FlashAttention-4's SM90 forward on fp16 operands, FlashAttention-3's bf16
    recompute and backward: (out in bf16 for autograd, out16 or None)."""

    implementation = "flash_attention_3"  # the attn_implementation whose layers it runs

    @classmethod
    def eligible(cls, implementation: str, kv_heads: int, sp: int) -> bool:
        """Whether it runs a layer: FlashAttention-3's configuration under Ulysses, the KV heads split or repeated."""
        return implementation == cls.implementation and sp > 1 and (kv_heads % sp == 0 or sp % kv_heads == 0)

    # one kernel rotates q and k, repeats each KV head for the ranks sharing it and rounds q, k, v once for one
    # all-to-all; the forward runs FA4's fp16 kernel (forward_is_fp16), the recompute FA3's bf16 forward, whose output
    # and LSE the backward reads; a forward whose saves are kept runs both
    @staticmethod
    def forward(ctx, q, k, v, cos, sin, cu_seqlens, max_length, scale, group):
        import flash_attn_interface as fa3

        forward_is_fp16 = Fp16Operands.is_enabled and not RecomputeFlag.in_recompute.get()
        sp = dist.get_world_size(group)
        _, s, hq, d = q.shape
        hk = k.shape[2]
        rot = cos.shape[-1]
        rep = max(sp // hk, 1)
        hq_r, hk_r = hq // sp, hk * rep // sp
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos2 = cos.reshape(-1, rot).float().contiguous()
        sin2 = sin.reshape(-1, rot).float().contiguous()

        out16 = None
        if forward_is_fp16:
            packed_qkv16 = HopperUlyssesAttention.pack(q, k, v, cos2, sin2, group, rep, torch.float16)
            out16 = HopperUlyssesAttention.flash_attention_4(
                *packed_qkv16.split((hq_r, hk_r, hk_r), dim=1), cu_seqlens, max_length, scale
            )[0]
            del packed_qkv16
            out16 = all_to_all_tensor(out16.unsqueeze(0), 1, 2, group)  # [1, T_local, hq, d]

        # the backward's inputs: FA3's bf16 forward, or placeholders with their metadata where they are discarded
        if forward_is_fp16 and (not any(ctx.needs_input_grad) or RecomputeFlag.saves_discarded()):
            total = s * sp
            metadata = (
                ((total, hq_r + 2 * hk_r, d), torch.bfloat16),
                ((total, hq_r, d), torch.bfloat16),
                ((hq_r, total), torch.float32),
            )
            packed_qkv, output_bf16, lse = (
                RecomputeFlag.placeholder(shape, dtype, q.device) for shape, dtype in metadata
            )
        else:
            packed_qkv = HopperUlyssesAttention.pack(q, k, v, cos2, sin2, group, rep, torch.bfloat16)
            output_bf16, lse = fa3._flash_attn_forward(
                *packed_qkv.split((hq_r, hk_r, hk_r), dim=1),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_length,
                max_seqlen_k=max_length,
                softmax_scale=scale,
                causal=True,
            )[:2]
        ctx.save_for_backward(packed_qkv, output_bf16, lse, cos2, sin2, cu_seqlens)
        ctx.dims = (hq, hk, rep, d, sp, max_length, scale, group, rot, q.dtype, k.dtype, v.dtype)

        if out16 is None:
            return all_to_all_tensor(output_bf16.unsqueeze(0), 1, 2, group), None

        ctx.mark_non_differentiable(out16)
        return out16.to(torch.bfloat16), out16

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout_local, _twin):
        import flash_attn_interface as fa3

        packed_qkv, output_bf16, lse, cos2, sin2, cu_seqlens = ctx.saved_tensors
        hq, hk, rep, d, sp, max_length, scale, group, rot, q_dtype, k_dtype, v_dtype = ctx.dims
        hq_r, hk_r = hq // sp, hk * rep // sp
        dout = all_to_all_tensor(dout_local.to(torch.bfloat16).contiguous(), 2, 1, group)[0]  # [T, hq_r, d]

        grads = torch.empty(packed_qkv.shape, device=packed_qkv.device, dtype=torch.bfloat16)
        dq, dk, dv = grads.split((hq_r, hk_r, hk_r), dim=1)
        query_bf16, key_bf16, value_bf16 = packed_qkv.split((hq_r, hk_r, hk_r), dim=1)
        fa3._flash_attn_backward(
            dout,
            query_bf16,
            key_bf16,
            value_bf16,
            output_bf16,
            lse,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            max_length,
            max_length,
            dq,
            dk,
            dv,
            scale,
            True,
        )
        del dout, query_bf16, key_bf16, value_bf16, dq, dk, dv

        grads = all_to_all_tensor(grads.unsqueeze(0), 1, 2, group)[0]  # [T_local, hq + 2 hk rep, d]
        t_local = grads.shape[0]

        dq = torch.empty(1, t_local, hq, d, device=grads.device, dtype=torch.float32)
        dk = torch.empty(1, t_local, hk, d, device=grads.device, dtype=torch.float32)
        dv = torch.empty(1, t_local, hk, d, device=grads.device, dtype=torch.float32)
        unpack_rope_kernel[(triton.cdiv(t_local, PACK_TOKENS), hq + 2 * hk)](
            grads, cos2, sin2, dq, dk, dv, t_local, hq, hk, sp, rep, d, rot, PACK_TOKENS, num_warps=PACK_WARPS
        )

        return dq.to(q_dtype), dk.to(k_dtype), dv.to(v_dtype), None, None, None, None, None, None

    @staticmethod
    def pack(q, k, v, cos2, sin2, group, rep, dtype):
        """This rank's heads of q, k (rotated) and v over the whole sequence, rounded once to dtype: [T, per, d]."""
        _, s, hq, d = q.shape
        hk = k.shape[2]

        out = torch.empty(s, hq + 2 * hk * rep, d, device=q.device, dtype=dtype)
        pack_rope_kernel[(triton.cdiv(s, PACK_TOKENS), hq + 2 * hk * rep)](
            q,
            k,
            v,
            cos2,
            sin2,
            out,
            Fp16Operands.range_flag(q.device),
            s,
            hq,
            hk,
            dist.get_world_size(group),
            rep,
            d,
            cos2.shape[-1],
            PACK_TOKENS,
            dtype == torch.float16,
            FP16_LIMIT,
            num_warps=PACK_WARPS,
        )

        return all_to_all_tensor(out.unsqueeze(0), 2, 1, group)[0]
