import torch
import torch.distributed as dist
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
import triton
import triton.language as tl
from transformers.integrations import flash_attention
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    prepare_fa_kwargs_from_position_ids,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from pear.models.qwen3_5.trainer.operands import FP16_LIMIT, Fp16Operands, Fp32OutputLinear, raise_range_flag
from pear.models.qwen3_5.trainer.rms_norm import RMSNorm
from verl.models.transformers.monkey_patch import _ulysses_flash_attention_forward, repeat_kv
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


@triton.jit
def _gated_out_kernel(
    A,
    G,
    OUT,
    O16,
    REMAINDER,
    FLAG,
    N,
    HAS_O: tl.constexpr,
    HAS_O16: tl.constexpr,
    HAS_REMAINDER: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    o = a * tl.sigmoid(g)

    if HAS_O:
        tl.store(OUT + offs, o.to(OUT.dtype.element_ty), mask=m)
    if HAS_O16:
        o16 = o.to(tl.float16)
        tl.store(O16 + offs, o16, mask=m)
        if HAS_REMAINDER:
            tl.store(REMAINDER + offs, (o - o16.to(tl.float32)).to(tl.float16), mask=m)
        raise_range_flag(FLAG, o, m, LIMIT)


@triton.jit
def _gated_out_bwd_kernel(A, G, DO, DA, DG, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    do = tl.load(DO + offs, mask=m, other=0.0).to(tl.float32)

    s = tl.sigmoid(g)
    tl.store(DA + offs, (do * s).to(DA.dtype.element_ty), mask=m)
    tl.store(DG + offs, (do * a * s * (1.0 - s)).to(DG.dtype.element_ty), mask=m)


class PackedBoundaries:
    """The packed sequence boundaries of a micro-batch from its position ids, once for every layer and recompute."""

    BOUNDARY_COPIES = 4  # micro-batches whose packed boundaries stay cached

    # every layer of a forward (and of its checkpointed recompute) receives the same position-ids tensor; transformers'
    # extraction from the gathered ids is a nonzero(), a host sync
    copies: list = []  # (local position ids, cu_seqlens, max_length), the most recent micro-batches first

    @classmethod
    def of(cls, position_ids, group):
        for tensor, cu_seqlens, max_length in cls.copies:
            if tensor is position_ids:
                return cu_seqlens, max_length

        rank_position_ids = [torch.empty_like(position_ids) for _ in range(dist.get_world_size(group))]
        dist.all_gather(rank_position_ids, position_ids, group=group)
        (cu_seqlens, _), (max_length, _) = prepare_fa_kwargs_from_position_ids(torch.cat(rank_position_ids, dim=-1))

        cls.copies.insert(0, (position_ids, cu_seqlens, max_length))
        del cls.copies[cls.BOUNDARY_COPIES :]

        return cu_seqlens, max_length


class UlyssesPackedAttention:
    """verl's Ulysses flash attention with q, k and v in one all-to-all and the boundaries once per micro-batch."""

    # each rank's head chunk keeps its q, k and v heads together, so they travel in one all-to-all (one in the backward
    # too); the attention call, its arguments and the output all-to-all are verl's
    @staticmethod
    def forward(
        query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=None, **kwargs
    ):
        sp = get_ulysses_sequence_parallel_world_size()
        if sp <= 1 or position_ids is None or kwargs.get("cu_seq_lens_q") is not None:
            return _ulysses_flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                *args,
                position_ids=position_ids,
                **kwargs,
            )

        repeats = max(sp // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        b, s, hq, d = query_states.shape
        hk = key_states.size(2)
        qkv = torch.cat(
            (
                query_states.reshape(b, s, sp, hq // sp, d),
                key_states.reshape(b, s, sp, hk // sp, d),
                value_states.reshape(b, s, sp, hk // sp, d),
            ),
            dim=3,
        ).reshape(b, s, -1, d)
        qkv = gather_seq_scatter_heads(qkv, seq_dim=1, head_dim=2)
        q, k, v = qkv.split((hq // sp, hk // sp, hk // sp), dim=2)

        cu_seqlens, max_length = PackedBoundaries.of(position_ids, get_ulysses_sequence_parallel_group())
        out = _flash_attention_forward(
            q,
            k,
            v,
            attention_mask,
            q.size(1),
            *args,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_length,
            max_length_k=max_length,
            **kwargs,
        )

        return gather_heads_scatter_seq(out, seq_dim=1, head_dim=2)


class GatedOutput(torch.autograd.Function):
    """attn_output * sigmoid(gate) in fp32, rounded once to bf16 and once to fp16: (o, o16, the fp16 remainder)."""

    GATE_ELEMENTS = 4096  # elements per program of the output gate's kernels

    # without autograd the fp16 output is the only one; a16: the attention's fp16 output when a is its bf16 handle;
    # recompute_is_fp16: the GPU backend's (TrainingGpuBackend), whether the recompute has it too
    @staticmethod
    def forward(ctx, a, g, a16, recompute_is_fp16):
        a_dtype, handle = a.dtype, a
        a = (a16 if a16 is not None else a).contiguous()
        g = g.contiguous()

        o, o16, remainder = Fp16Operands.output_buffers(
            ctx, a.shape, torch.bfloat16, a.device, Fp16Operands.is_enabled, Fp16Operands.wants_remainder()
        )

        n = a.numel()
        _gated_out_kernel[(triton.cdiv(n, GatedOutput.GATE_ELEMENTS),)](
            a,
            g,
            o if o is not None else o16,
            o16 if o16 is not None else a,
            remainder if remainder is not None else a,
            Fp16Operands.range_flag(a.device),
            n,
            o is not None,
            o16 is not None,
            remainder is not None,
            FP16_LIMIT,
            BLOCK=GatedOutput.GATE_ELEMENTS,
        )

        ctx.save_for_backward(a if recompute_is_fp16 else handle.contiguous(), g)
        ctx.a_dtype = a_dtype

        return Fp16Operands.twin_outputs(ctx, o, o16, remainder)

    @staticmethod
    def backward(ctx, do, *_twins):
        a, g = ctx.saved_tensors
        do = do.contiguous()

        da = torch.empty(a.shape, device=a.device, dtype=ctx.a_dtype)
        dg = torch.empty_like(g)

        n = a.numel()
        _gated_out_bwd_kernel[(triton.cdiv(n, GatedOutput.GATE_ELEMENTS),)](
            a, g, do, da, dg, n, BLOCK=GatedOutput.GATE_ELEMENTS
        )

        return da, dg, None, None


class FullAttention:
    """Qwen3.5's gated full attention with its query/key path in fp32 up to one rounding and its output gate in fp32."""

    @staticmethod
    def install(model, gpu_backend) -> None:
        """Patch the attention layers' forward, projections and q/k norms; each layer holds the GPU backend."""
        modeling.Qwen3_5Attention.forward = FullAttention.forward
        flash_attention._flash_attention_forward = UlyssesPackedAttention.forward

        for layer in model.model.language_model.layers:
            if layer.layer_type == "full_attention":
                attention = layer.self_attn
                attention.gpu_backend = gpu_backend

                # fp32 outputs: the q/k norms and the rotary embedding compute on them, v and the attention's output
                # are rounded once for the kernels; o_proj's output joins the fp32 residual stream
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
                    Fp32OutputLinear.install(projection)

                for norm in (attention.q_norm, attention.k_norm):
                    FullAttention.install_head_norm(norm)

    @staticmethod
    def install_head_norm(norm) -> None:
        """The q/k norm's forward as one kernel each way on its fp32 input, the output fp32 (the module call stays)."""

        def forward(x):
            shape = x.shape
            return RMSNorm.apply(x.reshape(-1, shape[-1]), norm.weight, norm.eps, torch.float32)[0].view(shape)

        norm.forward = forward

    @staticmethod
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        cos, sin = position_embeddings

        # the rotary embedding, the attention on fp16 operands and the Ulysses exchanges in one autograd node where the
        # GPU backend's attention applies; the stock attention on bf16 operands otherwise
        attention_cls = self.gpu_backend.attention_cls
        sp = get_ulysses_sequence_parallel_world_size()
        attn16, attn_weights = None, None
        if attention_cls.eligible(self.config._attn_implementation, key_states.shape[2], sp):
            group = get_ulysses_sequence_parallel_group()
            cu_seqlens, max_length = PackedBoundaries.of(kwargs["position_ids"], group)
            attn_output, attn16 = attention_cls.apply(
                query_states,
                key_states,
                value_states,
                cos,
                sin,
                cu_seqlens.to(torch.int32),
                int(max_length),
                self.scaling,
                group,
            )
            if attn16 is not None:
                attn16 = attn16.reshape(*input_shape, -1)
        else:
            query_states, key_states = modeling.apply_rotary_pos_emb(
                query_states.transpose(1, 2), key_states.transpose(1, 2), cos.float(), sin.float()
            )
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, modeling.eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states.to(torch.bfloat16),
                key_states.to(torch.bfloat16),
                value_states.transpose(1, 2).to(torch.bfloat16),
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        del query_states, key_states, value_states

        # the output gate's result with its fp16 twin and remainder for o_proj
        o, o16, remainder = GatedOutput.apply(
            attn_output.reshape(*input_shape, -1), gate, attn16, self.gpu_backend.recompute_is_fp16
        )
        del attn_output, attn16
        if o16 is None and o.dtype == torch.float16:
            o16 = o  # without autograd the fp16 output is the handle
        Fp16Operands.register(o, o16, remainder)

        try:
            return self.o_proj(o), attn_weights
        finally:
            Fp16Operands.release(o)
