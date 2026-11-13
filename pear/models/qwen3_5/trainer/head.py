import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from pear.models.qwen3_5.trainer.operands import Fp16Operands
from pear.models.qwen3_5.trainer.rms_norm import RMSNorm


class Fp32HeadLogProbs(torch.autograd.Function):
    """The log-probs at given ids and the entropy of the lm_head's logits, in fp32, a chunk of rows at a time."""

    CHUNK_ROWS = 4096  # rows per chunk of fp32 logits: 4 GB at Qwen3.5's vocabulary, and the chunk's GEMMs stay large

    @staticmethod
    def forward(ctx, hidden_states, weight, ids, temperature):
        n = hidden_states.shape[0]
        log_probs = hidden_states.new_empty(ids.shape, dtype=torch.float32)
        log_normalizers = hidden_states.new_empty(n, dtype=torch.float32)
        entropy = hidden_states.new_empty(n, dtype=torch.float32)

        for rows in Fp32HeadLogProbs.chunks(n):
            logits = Fp32HeadLogProbs.logits(hidden_states[rows], weight, temperature)
            Fp32HeadLogProbs.write_log_probs(logits, rows, ids, log_normalizers, log_probs)

            # H = -sum_v p_v log p_v; lse - sum_v p_v z_v would cancel in fp32
            log_p = logits.sub_(log_normalizers[rows, None])
            entropy[rows] = -log_p.exp().mul_(log_p).sum(-1)

        ctx.save_for_backward(hidden_states, weight, ids, log_normalizers, entropy)
        ctx.temperature = temperature
        ctx.set_materialize_grads(False)  # an unused entropy gets no gradient pass

        return log_probs, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        hidden_states, weight, ids, log_normalizers, entropy = ctx.saved_tensors
        grad_hidden_states = torch.empty_like(hidden_states)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)

        for rows in Fp32HeadLogProbs.chunks(hidden_states.shape[0]):
            # dL/dz = g - p sum_j g_j for the log-probs at ids, and - p (log p + H) g_H for the entropy
            log_p = Fp32HeadLogProbs.logits(hidden_states[rows], weight, ctx.temperature).sub_(
                log_normalizers[rows, None]
            )
            probs = log_p.exp()
            grad_logits = probs * -grad_log_probs[rows].sum(-1, keepdim=True)
            if grad_entropy is not None:
                grad_logits -= log_p.add_(entropy[rows, None]).mul_(probs).mul_(grad_entropy[rows, None])

            # one scatter per column: a row's ids can coincide (a label among the head ids), and the atomic adds of
            # one call to the same element land in a varying order
            for column in range(ids.shape[1]):
                grad_logits.scatter_add_(1, ids[rows, column, None], grad_log_probs[rows, column, None])

            grad_logits = grad_logits.div_(ctx.temperature).to(hidden_states.dtype)
            Fp32HeadLogProbs.write_gradients(grad_logits, rows, hidden_states, weight, grad_hidden_states, grad_weight)

        return grad_hidden_states, grad_weight.to(weight.dtype), None, None

    @staticmethod
    def chunks(count: int):
        """The slices of count rows, a chunk of logits at a time."""
        for start in range(0, count, Fp32HeadLogProbs.CHUNK_ROWS):
            yield slice(start, start + Fp32HeadLogProbs.CHUNK_ROWS)

    @staticmethod
    def logits(hidden_states, weight, temperature):
        """One chunk's logits: the products of the bf16 values summed in fp32 and left in fp32."""
        return torch.mm(hidden_states, weight.t(), out_dtype=torch.float32).div_(temperature)

    @staticmethod
    def write_log_probs(logits, rows, ids, log_normalizers, log_probs) -> None:
        """A chunk's log-normalizers and its log-probs at the ids, from its logits."""
        log_normalizers[rows] = logits.logsumexp(-1)
        log_probs[rows] = logits.gather(1, ids[rows]) - log_normalizers[rows, None]

    @staticmethod
    def write_gradients(grad_logits, rows, hidden_states, weight, grad_hidden_states, grad_weight) -> None:
        """A chunk's hidden gradient and its part of the weight's, summed in fp32, from its logits' gradient."""
        grad_hidden_states[rows] = grad_logits @ weight
        torch.addmm(grad_weight, grad_logits.t(), hidden_states[rows], out_dtype=torch.float32, out=grad_weight)


class Fp32Head:
    """The lm_head called with ids: the fp32 log-probs at them and the entropy, within the module's own call."""

    # a call of the module lets FSDP gather its weight in param_dtype and reduce its gradient over the shard group; the
    # weight read outside the call is the sharded fp32 parameter, whose gradient would skip that reduction
    @staticmethod
    def install(head) -> None:
        """Let the lm_head take ids and the temperature for Fp32HeadLogProbs; without ids it is the linear layer."""
        linear_forward = head.forward

        def forward(hidden_states, ids=None, temperature=1.0):
            if ids is None:
                return linear_forward(hidden_states)
            return Fp32HeadLogProbs.apply(hidden_states.to(head.weight.dtype), head.weight, ids, temperature)

        head.forward = forward


@triton.jit
def _entropy_kernel(Z, LSE, ENT, V, BLOCK: tl.constexpr):
    # the entropy -sum_v p_v log p_v of one row given its log-normalizer, one pass over the fp32 logits; libdevice's exp
    # (Triton's fast exp biases the entropy and score centering's mass term with it)
    row = tl.program_id(0).to(tl.int64)
    zrow = Z + row * V
    lse = tl.load(LSE + row)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)

    for start in range(0, V, BLOCK):
        c = start + cols
        log_p = tl.load(zrow + c, mask=c < V, other=float("-inf")) - lse
        acc += tl.where(c < V, libdevice.exp(log_p) * log_p, 0.0)

    tl.store(ENT + row, -tl.sum(acc, 0))


@triton.jit
def _bwd_kernel(
    Z, LSE, GSUM, ENT, GENT, G, V, temperature, HAS_T: tl.constexpr, HAS_ENT: tl.constexpr, BLOCK: tl.constexpr
):
    # grad_z = (-p * sum_k g_k - [entropy] p (log p + H) g_H) / T, rounded to G's dtype (the gathered ids' part apart)
    row = tl.program_id(0).to(tl.int64)
    zrow = Z + row * V
    grow = G + row * V

    lse = tl.load(LSE + row)
    gs = tl.load(GSUM + row)
    if HAS_ENT:
        h = tl.load(ENT + row)
        gh = tl.load(GENT + row)
    cols = tl.arange(0, BLOCK)

    for start in range(0, V, BLOCK):
        c = start + cols
        z = tl.load(zrow + c, mask=c < V, other=0.0)
        if HAS_T:
            z = z / temperature
        log_p = z - lse
        p = libdevice.exp(log_p)
        g = p * -gs
        if HAS_ENT:
            g -= (log_p + h) * p * gh
        if HAS_T:
            g = g / temperature
        tl.store(grow + c, g.to(G.dtype.element_ty), mask=c < V)


@triton.jit
def _bwd_fixup_kernel(
    Z,
    LSE,
    GSUM,
    ENT,
    GENT,
    IDS,
    GLP,
    G,
    V,
    K1: tl.constexpr,
    temperature,
    HAS_T: tl.constexpr,
    HAS_ENT: tl.constexpr,
    KB: tl.constexpr,
):
    # at the gathered columns: the dense value again in fp32 plus every g_k of that id (the scatter), rounded once
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, KB)
    mk = k < K1
    ids = tl.load(IDS + row * K1 + k, mask=mk, other=-1)
    g = tl.load(GLP + row * K1 + k, mask=mk, other=0.0)

    # the sum of g over equal ids (a label among the head ids), in a fixed order
    equal_ids = (ids[:, None] == ids[None, :]) & mk[None, :]
    gtot = tl.sum(tl.where(equal_ids, g[None, :], 0.0), 1)

    lse = tl.load(LSE + row)
    gs = tl.load(GSUM + row)
    z = tl.load(Z + row * V + ids, mask=mk, other=0.0)
    if HAS_T:
        z = z / temperature
    log_p = z - lse
    p = libdevice.exp(log_p)
    val = p * -gs
    if HAS_ENT:
        h = tl.load(ENT + row)
        gh = tl.load(GENT + row)
        val -= (log_p + h) * p * gh
    val += gtot
    if HAS_T:
        val = val / temperature

    tl.store(G + row * V + ids, val.to(G.dtype.element_ty), mask=mk)


class FusedFp32HeadLogProbs(Fp32HeadLogProbs):
    """Fp32HeadLogProbs (log-probs at ids, entropy of the fp32 logits) with fused passes over each chunk."""

    ROW_COLUMNS = 2048  # vocabulary columns per iteration of the row kernels
    ROW_WARPS = 8  # warps of the entropy and gradient kernels, a row per program
    FIXUP_WARPS = 4  # warps of the fix-up kernel at the gathered columns

    # per chunk of rows the logits come from one GEMM left in fp32, as in the fp32 head; the log-normalizer and the
    # log-probs are the fp32 head's (bit-identical: score centering's mass term is ill-conditioned in them), the
    # entropy is one kernel, the backward one kernel plus a fix-up at the gathered columns
    @staticmethod
    def forward(ctx, hidden_states, weight, ids, temperature):
        n = hidden_states.shape[0]
        ids = ids.contiguous()
        log_probs = hidden_states.new_empty(ids.shape, dtype=torch.float32)
        lse = hidden_states.new_empty(n, dtype=torch.float32)
        entropy = hidden_states.new_empty(n, dtype=torch.float32)
        V = weight.shape[0]

        # fp16 operands: the final norm's twin of the hidden state and the weight's exact fp16 image; the backward
        # recomputes the logits from the same operands
        h16 = Fp16Operands.twin(hidden_states) if Fp16Operands.is_enabled else None
        if h16 is not None:
            FusedHead.release_final_outputs()
        w16 = Fp16Operands.weight16(weight) if h16 is not None else None

        for rows in FusedFp32HeadLogProbs.chunks(n):
            if h16 is not None:
                z = Fp16Operands.mm16(h16[rows], w16)
            else:
                z = torch.mm(hidden_states[rows], weight.t(), out_dtype=torch.float32)
            if float(temperature) != 1.0:
                z.div_(temperature)
            FusedFp32HeadLogProbs.write_log_probs(z, rows, ids, lse, log_probs)
            _entropy_kernel[(z.shape[0],)](
                z,
                lse[rows],
                entropy[rows],
                V,
                FusedFp32HeadLogProbs.ROW_COLUMNS,
                num_warps=FusedFp32HeadLogProbs.ROW_WARPS,
            )
            del z

        del w16
        ctx.save_for_backward(hidden_states, weight, ids, lse, entropy, h16 if h16 is not None else lse)
        ctx.has_fp16_twin, ctx.temperature = h16 is not None, temperature
        ctx.set_materialize_grads(False)  # an unused entropy gets no gradient pass

        return log_probs, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        hidden_states, weight, ids, lse, entropy, h16 = ctx.saved_tensors
        n, k1 = ids.shape
        w16 = Fp16Operands.weight16(weight) if ctx.has_fp16_twin else None
        V = weight.shape[0]
        has_t = float(ctx.temperature) != 1.0
        has_ent = grad_entropy is not None

        grad_hidden_states = torch.empty_like(hidden_states)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        grad_log_probs = grad_log_probs.float().contiguous()
        gsum = grad_log_probs.sum(-1)
        gent = grad_entropy.float().contiguous() if has_ent else gsum

        for rows in FusedFp32HeadLogProbs.chunks(n):
            if ctx.has_fp16_twin:
                z = Fp16Operands.mm16(h16[rows], w16)
            else:
                z = torch.mm(hidden_states[rows], weight.t(), out_dtype=torch.float32)
            r = z.shape[0]

            # the logits' gradient computed in fp32 and rounded once, as the fp32 head casts it
            g = torch.empty(r, V, device=z.device, dtype=hidden_states.dtype)
            args = (z, lse[rows], gsum[rows], entropy[rows], gent[rows])
            _bwd_kernel[(r,)](
                *args,
                g,
                V,
                float(ctx.temperature),
                has_t,
                has_ent,
                FusedFp32HeadLogProbs.ROW_COLUMNS,
                num_warps=FusedFp32HeadLogProbs.ROW_WARPS,
            )
            _bwd_fixup_kernel[(r,)](
                *args,
                ids[rows],
                grad_log_probs[rows],
                g,
                V,
                k1,
                float(ctx.temperature),
                has_t,
                has_ent,
                triton.next_power_of_2(max(k1, 2)),
                num_warps=FusedFp32HeadLogProbs.FIXUP_WARPS,
            )
            del z

            FusedFp32HeadLogProbs.write_gradients(g, rows, hidden_states, weight, grad_hidden_states, grad_weight)

        return grad_hidden_states, grad_weight.to(weight.dtype), None, None


class FusedHead:
    """The final norm writes its output's fp16 twin and the lm_head takes it through the fused fp32 head."""

    final_outputs: list = []  # the final norm's output registered with its twin, until the head takes it

    @classmethod
    def install(cls, model) -> None:
        norm = model.model.language_model.norm

        def final_norm(x):
            cls.release_final_outputs()
            y, y16, _ = RMSNorm.apply(x, norm.weight, norm.eps, RMSNorm.autocast_dtype(), Fp16Operands.is_enabled)
            if y16 is not None:
                Fp16Operands.register(y, y16)
                cls.final_outputs.append(y)
            return y

        norm.forward = final_norm

        head = model.lm_head
        fp32_head_forward = head.forward  # the family's Fp32Head: the linear layer without ids

        # the hidden state goes in as it comes: the fp32 head casts it to the weight's dtype, which would round the fp16
        # output of a pass without autograd to bf16
        def lm_head(hidden_states, ids=None, temperature=1.0):
            if ids is None:
                return fp32_head_forward(hidden_states)
            if hidden_states.dtype != torch.float16:
                hidden_states = hidden_states.to(head.weight.dtype)
            return FusedFp32HeadLogProbs.apply(hidden_states, head.weight, ids, temperature)

        head.forward = lm_head

    @classmethod
    def release_final_outputs(cls) -> None:
        while cls.final_outputs:
            Fp16Operands.release(cls.final_outputs.pop())
