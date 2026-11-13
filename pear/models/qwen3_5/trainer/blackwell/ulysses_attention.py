import torch
import torch.distributed as dist
import triton

from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, Fp16Operands
from pear.models.qwen3_5.trainer.ulysses_attention import (
    PACK_TOKENS,
    PACK_WARPS,
    TrainingUlyssesAttention,
    pack_rope_kernel,
    unpack_rope_kernel,
)
from verl.utils.ulysses import all_to_all_tensor


class BlackwellUlyssesAttention(TrainingUlyssesAttention):
    """Ulysses causal attention with FlashAttention-4's forward on fp16 operands: (out in bf16 for autograd, out16)."""

    implementation = "flash_attention_4"  # the attn_implementation whose layers it runs

    @classmethod
    def eligible(cls, implementation: str, kv_heads: int, sp: int) -> bool:
        """Whether it runs a layer: FlashAttention-4 on fp16 operands under Ulysses, the KV heads split by the ranks."""
        return Fp16Operands.is_enabled and implementation == cls.implementation and sp > 1 and kv_heads % sp == 0

    # q, k [1, T_local, H, D] (fp32, before the rotary embedding), v and the rotary's cos, sin [1, T_local, R]: one
    # kernel rotates q and k, rounds q, k and v once to fp16 and packs them for one all-to-all
    @staticmethod
    def forward(ctx, q, k, v, cos, sin, cu_seqlens, max_length, scale, group):
        sp = dist.get_world_size(group)
        b, s, hq, d = q.shape
        hk = k.shape[2]
        rot = cos.shape[-1]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos2 = cos.reshape(-1, rot).float().contiguous()
        sin2 = sin.reshape(-1, rot).float().contiguous()

        packed_qkv16 = torch.empty(b, s, hq + 2 * hk, d, device=q.device, dtype=torch.float16)
        pack_rope_kernel[(triton.cdiv(s, PACK_TOKENS), hq + 2 * hk)](
            q,
            k,
            v,
            cos2,
            sin2,
            packed_qkv16,
            Fp16Operands.range_flag(q.device),
            s,
            hq,
            hk,
            sp,
            1,
            d,
            rot,
            PACK_TOKENS,
            True,
            FP16_LIMIT,
            num_warps=PACK_WARPS,
        )
        packed_qkv16 = all_to_all_tensor(packed_qkv16, 2, 1, group)  # [1, T, (hq + 2 hk) / sp, d]
        q16, k16, v16 = packed_qkv16[0].split((hq // sp, hk // sp, hk // sp), dim=1)

        out16, *_ = BlackwellUlyssesAttention.flash_attention_4(q16, k16, v16, cu_seqlens, max_length, scale)
        local_output16 = all_to_all_tensor(out16.unsqueeze(0), 1, 2, group)  # [1, T_local, hq, d]

        if not any(ctx.needs_input_grad):
            return local_output16, None

        ctx.save_for_backward(packed_qkv16, cu_seqlens, cos2, sin2)
        ctx.dims = (hq, hk, d, sp, max_length, scale, group, rot, q.dtype, k.dtype, v.dtype)
        ctx.mark_non_differentiable(local_output16)

        return local_output16.to(torch.bfloat16), local_output16

    # the backward keeps the stock numerics (FA4's fp16 backward drifts the q/v projections' gradients on long
    # sequences): bf16 operands and a bf16 forward's probabilities, recomputed here; one kernel unrotates and unpacks
    # the gradients
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout_local, _twin):
        import flash_attn.cute.interface as fa4

        packed_qkv16, cu_seqlens, cos2, sin2 = ctx.saved_tensors
        hq, hk, d, sp, max_length, scale, group, rot, q_dtype, k_dtype, v_dtype = ctx.dims
        dout = all_to_all_tensor(dout_local.contiguous(), 2, 1, group)[0]  # [T, hq / sp, d]

        query_bf16, key_bf16, value_bf16 = (
            t.to(torch.bfloat16) for t in packed_qkv16[0].split((hq // sp, hk // sp, hk // sp), dim=1)
        )
        output_bf16, lse, *_ = BlackwellUlyssesAttention.flash_attention_4(
            query_bf16, key_bf16, value_bf16, cu_seqlens, max_length, scale
        )

        dq, dk, dv = fa4._flash_attn_bwd(
            query_bf16,
            key_bf16,
            value_bf16,
            output_bf16,
            dout.to(torch.bfloat16),
            lse,
            scale,
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
        )
        del query_bf16, key_bf16, value_bf16, output_bf16, lse, dout

        grads = torch.cat((dq, dk, dv), dim=1).unsqueeze(0)  # [1, T, (hq + 2 hk) / sp, d]
        del dq, dk, dv
        grads = all_to_all_tensor(grads, 1, 2, group)  # [1, T_local, hq + 2 hk (rank-major chunks), d]

        b, t_local = grads.shape[:2]
        dq = torch.empty(b, t_local, hq, d, device=grads.device, dtype=torch.float32)
        dk = torch.empty(b, t_local, hk, d, device=grads.device, dtype=torch.float32)
        dv = torch.empty(b, t_local, hk, d, device=grads.device, dtype=torch.float32)
        unpack_rope_kernel[(triton.cdiv(t_local, PACK_TOKENS), hq + 2 * hk)](
            grads, cos2, sin2, dq, dk, dv, t_local, hq, hk, sp, 1, d, rot, PACK_TOKENS, num_warps=PACK_WARPS
        )

        return dq.to(q_dtype), dk.to(k_dtype), dv.to(v_dtype), None, None, None, None, None, None
