import collections

import torch
import torch.distributed as dist

from pear.models.qwen3_5.trainer.ieee_fp32 import IEEEFp32


class BlackwellDeltaRule(torch.autograd.Function):
    """The gated delta rule over packed local tokens on cuDNN's FROST kernels, with Ulysses context parallelism."""

    PLAN = "gdn_frost"  # cuDNN's FROST engines for the forward and the backward (CuTe DSL, nvidia-cutlass-dsl >= 4.7)
    # span lengths the summaries run at: unlike the forward and backward plans they compile a kernel per length and a
    # boundary span is rarely a length seen before; zero operands, log decay and beta leave the state as it is (to the
    # kernels' fp32 rounding, whose order depends on length)
    SUMMARY_SPAN_BUCKET = 4096
    # the gated delta rule's options in every call: the gates and the normalization are computed before the kernels
    GDN_FLAGS = {
        "use_qk_l2norm_in_kernel": False,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "gate_domain": "log",
        "a_log": None,
        "dt_bias": None,
        "batch_invariant": False,
    }

    @staticmethod
    def apply_packed(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context, q16, k16, v16, layer_idx):
        """(o [1, T, HV, V], its fp16 twin or None) for q, k [1, T, H, K] (L2-normalized), v, g (log decay) and beta;
        every layer (layer_idx) runs the same kernels."""
        if q16 is not None:
            q16, k16, v16 = q16.contiguous(), k16.contiguous(), v16.contiguous()

        return BlackwellDeltaRule.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            cu_seqlens.to(device=q.device, dtype=torch.int32),
            cu_seqlens_cpu.tolist(),
            cp_context,
            q16,
            k16,
            v16,
        )

    # a sequence crossing a rank boundary needs the state the earlier ranks leave it, and in the backward the gradient
    # of the state it leaves the later ranks: as FLA does, each rank summarizes its boundary span (the final state H
    # and the transition M: X_final = X_init @ M + H; backward G and M^T), all-gathered and chained in fp32
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context, q16, k16, v16):
        ops = torch.ops.cudnn
        scale = k.shape[-1] ** -0.5
        # with fp16 twins of q, k, v the forward runs on them and returns (o, o16): the kernel's fp16 output and its
        # bf16 rounding for autograd; the backward runs on bf16 operands
        forward_is_fp16 = q16 is not None
        q3, k3, v3 = (q16[0], k16[0], v16[0]) if forward_is_fp16 else (q[0], k[0], v[0])
        g2, b2 = g[0], beta[0]
        HO, V, K = v3.shape[1], v3.shape[2], k3.shape[2]

        initial_state = None
        if cp_context is not None:
            group, rank = cp_context.group, dist.get_rank(cp_context.group)
            if not cp_context.is_last_rank:
                span_start, span_end = BlackwellDeltaRule.span(cu_seqlens_cpu, is_last=True)
                summary_state, transition = ops.gated_delta_net_summary(
                    *BlackwellDeltaRule.padded_span((k3, v3, g2, b2), span_start, span_end),
                    BlackwellDeltaRule.span_cu(cp_context, span_end - span_start, True, q.device),
                    initial_state=None,
                    output_transition=True,
                    plan_name=None,
                    **BlackwellDeltaRule.GDN_FLAGS,
                )
                summary_state, transition = summary_state[0].float(), transition[0].float()
            else:
                summary_state = torch.zeros(HO, V, K, device=q.device, dtype=torch.float32)
                transition = torch.zeros(HO, K, K, device=q.device, dtype=torch.float32)
            summary_states, transitions = BlackwellDeltaRule.gather_pairs(summary_state, transition, group)
            if not cp_context.is_first_rank:
                incoming_state = torch.zeros(HO, V, K, device=q.device, dtype=torch.float32)
                with IEEEFp32(torch.backends.cuda.matmul):
                    for j in range(rank - cp_context.pre_num_ranks, rank):
                        incoming_state = torch.baddbmm(summary_states[j], incoming_state, transitions[j])
                initial_state = torch.zeros(len(cu_seqlens_cpu) - 1, HO, V, K, device=q.device, dtype=torch.float32)
                initial_state[0] = incoming_state

        o, _, _ = ops.gated_delta_net_fwd(
            q3,
            k3,
            v3,
            g2,
            b2,
            cu_seqlens,
            float(scale),
            initial_state=initial_state,
            output_final_state=False,
            checkpoint_every_n_tokens=0,
            plan_name=BlackwellDeltaRule.PLAN,
            **BlackwellDeltaRule.GDN_FLAGS,
        )

        needs_backward = any(ctx.needs_input_grad)
        if needs_backward:
            ctx.save_for_backward(
                q[0], k[0], v[0], g2, b2, cu_seqlens, initial_state if initial_state is not None else cu_seqlens
            )
            ctx.has_initial_state = initial_state is not None
            ctx.cu_seqlens_cpu, ctx.scale, ctx.cp_context = cu_seqlens_cpu, scale, cp_context

        o = o.unsqueeze(0)
        if not (forward_is_fp16 and needs_backward):
            return o, None
        ctx.mark_non_differentiable(o)
        return o.to(torch.bfloat16), o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, _twin):
        ops = torch.ops.cudnn
        q3, k3, v3, g2, b2, cu_seqlens, initial_state = ctx.saved_tensors
        initial_state = initial_state if ctx.has_initial_state else None
        cu_seqlens_cpu, cp_context = ctx.cu_seqlens_cpu, ctx.cp_context
        do3 = do[0].contiguous()
        HO, V, K = v3.shape[1], v3.shape[2], k3.shape[2]

        d_final = None
        if cp_context is not None:
            group, rank = cp_context.group, dist.get_rank(cp_context.group)
            if not cp_context.is_first_rank:
                span_start, span_end = BlackwellDeltaRule.span(cu_seqlens_cpu, is_last=False)
                summary_gradient, transposed_transition = ops.gated_delta_net_summary_bwd(
                    *BlackwellDeltaRule.padded_span((do3, q3, k3, g2, b2), span_start, span_end),
                    BlackwellDeltaRule.span_cu(cp_context, span_end - span_start, False, do3.device),
                    float(ctx.scale),
                    d_final_state=None,
                    output_transition=True,
                    plan_name=None,
                    **BlackwellDeltaRule.GDN_FLAGS,
                )
                summary_gradient, transposed_transition = summary_gradient[0].float(), transposed_transition[0].float()
            else:
                summary_gradient = torch.zeros(HO, V, K, device=do3.device, dtype=torch.float32)
                transposed_transition = torch.zeros(HO, K, K, device=do3.device, dtype=torch.float32)
            summary_gradients, transposed_transitions = BlackwellDeltaRule.gather_pairs(
                summary_gradient, transposed_transition, group
            )
            if not cp_context.is_last_rank:
                outgoing_gradient = torch.zeros(HO, V, K, device=do3.device, dtype=torch.float32)
                with IEEEFp32(torch.backends.cuda.matmul):
                    for j in range(rank + cp_context.post_num_ranks, rank, -1):
                        outgoing_gradient = torch.baddbmm(
                            summary_gradients[j], outgoing_gradient, transposed_transitions[j]
                        )
                d_final = torch.zeros(len(cu_seqlens_cpu) - 1, HO, V, K, device=do3.device, dtype=torch.float32)
                d_final[-1] = outgoing_gradient

        dq, dk, dv, dg, dbeta, _, _, _ = ops.gated_delta_net_bwd(
            do3,
            q3,
            k3,
            v3,
            g2,
            b2,
            cu_seqlens,
            float(ctx.scale),
            initial_state=initial_state,
            d_final_state=d_final,
            state_checkpoints=None,
            checkpoint_every_n_tokens=0,
            plan_name=BlackwellDeltaRule.PLAN,
            **BlackwellDeltaRule.GDN_FLAGS,
        )

        grads = (dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0), dg.unsqueeze(0), dbeta.unsqueeze(0))
        return grads + (None,) * 6

    @staticmethod
    def span(cu_seqlens_cpu: list, is_last: bool) -> tuple[int, int]:
        """The local token span of the rank's first or last sequence."""
        i = len(cu_seqlens_cpu) - 2 if is_last else 0
        return cu_seqlens_cpu[i], cu_seqlens_cpu[i + 1]

    @staticmethod
    def padded_span(tensors, span_start: int, span_end: int) -> list[torch.Tensor]:
        """Each tensor's rows of the span, then zero rows up to the span's bucket length (state-preserving tokens)."""
        length = BlackwellDeltaRule.bucket(span_end - span_start)
        if length == span_end - span_start:
            return [t[span_start:span_end] for t in tensors]

        copies = [t.new_zeros(length, *t.shape[1:]) for t in tensors]

        for out, t in zip(copies, tensors, strict=True):
            out[: span_end - span_start] = t[span_start:span_end]

        return copies

    @staticmethod
    def span_cu(cp_context, length: int, is_last: bool, device) -> torch.Tensor:
        """The boundaries [0, bucket length] of the rank's first or last span, of that length, on the device."""
        # copied from pinned memory: torch.tensor(..., device=cuda) copies from pageable memory and synchronizes the
        # stream, which in every GDN layer and recompute drains the GPU's queue
        key = "_pear_span_last" if is_last else "_pear_span_first"

        # made once per micro-batch, on FLA's context
        if getattr(cp_context, key, None) is None:
            span = torch.tensor([0, BlackwellDeltaRule.bucket(length)], dtype=torch.int32, device="cpu").pin_memory()
            setattr(cp_context, key, span.to(device, non_blocking=True))

        return getattr(cp_context, key)

    @staticmethod
    def gather_pairs(a, b, group):
        """All-gather two fp32 tensors of one shape from every rank: [world, *shape] each."""
        payload = torch.stack((a, b)).contiguous()
        out = torch.empty(dist.get_world_size(group), *payload.shape, device=payload.device, dtype=payload.dtype)
        dist.all_gather_into_tensor(out, payload, group=group)
        return out[:, 0], out[:, 1]

    @staticmethod
    def bucket(length: int) -> int:
        """The summaries' span length for a span of `length` tokens: the next multiple of SUMMARY_SPAN_BUCKET."""
        return -(-length // BlackwellDeltaRule.SUMMARY_SPAN_BUCKET) * BlackwellDeltaRule.SUMMARY_SPAN_BUCKET


class CudnnGDNGraphs:
    """cuDNN's Gated DeltaNet ops with their graph caches bounded and a workspace per execution."""

    # graphs each of cuDNN's Gated DeltaNet forward and backward caches keeps: a micro-batch's layers share one shape,
    # so its forward, recompute and backward need one or two per cache; the rest keep recent shapes that recur
    GRAPHS_PER_CACHE = 8
    SUMMARY_GRAPHS = 64  # summary graphs kept, one per span bucket and direction

    # cuDNN frontend keeps one graph per problem shape (total tokens, sequences) without a bound, and each graph its own
    # grow-only workspace: every micro-batch of a new shape would keep another set of them for the rest of the process
    @classmethod
    def install(cls) -> None:
        """Register the ops as torch.ops.cudnn and bound their graphs in this process."""
        from cudnn.linear_attention.ops import common, gdn

        for module in (common, gdn):
            module.graph_workspace = cls.graph_workspace

        for name in ("fprop_cache", "bprop_cache"):
            setattr(gdn, name, LRUCache(cls.GRAPHS_PER_CACHE))

        gdn.summary_cache = LRUCache(cls.SUMMARY_GRAPHS)

    @staticmethod
    def graph_workspace(graph, device):
        """The graph's workspace for one execution, its plans built first; the caching allocator reuses the memory."""
        if not graph._is_built:
            if not graph._planning_done:
                graph.create_execution_plans()
            if graph.selected_engine is None:
                graph.build()
            else:
                graph.build_plans()

        return torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device=device)


class LRUCache(collections.OrderedDict):
    """A dict that keeps its `capacity` most recently used entries."""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)

        while len(self) > self.capacity:
            self.popitem(last=False)
