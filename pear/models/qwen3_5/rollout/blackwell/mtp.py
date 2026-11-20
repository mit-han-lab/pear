import torch
import torch.distributed._symmetric_memory as symm_mem
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.models.interfaces import _require_is_multimodal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.triton_utils import tl, triton

from pear.models.qwen3_5.rollout.blackwell.gemm import DraftGemmTableLinearMethod, DraftPartialsLinearMethod, StreamGemm
from pear.models.qwen3_5.rollout.blackwell.model import TP_SIZE, BlackwellQwen3_5ForConditionalGeneration
from pear.models.qwen3_5.rollout.mtp import Qwen3_5RolloutMTP
from pear.models.qwen3_5.rollout.triton_launcher import TritonLauncher


@triton.jit(do_not_specialize=["start", "end", "hidden_stride"])
def _draft_fc_input_kernel(
    IDS,
    EMBEDS,
    TABLE,
    HIDDEN,
    hidden_stride,
    W_EMBED,
    W_HIDDEN,
    OUT,
    start,
    end,
    eps,
    H: tl.constexpr,
    HALF: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOKUP: tl.constexpr,
):
    # row t of the drafter's fc input on this rank: the (1 + w) RMSNorm of its partial embedding row (the token's row
    # in this rank's shard [start, end), else zeros, whose norm is zeros), then its half of the normed hidden state
    t = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    if LOOKUP:
        i = tl.load(IDS + t).to(tl.int64)
        in_shard = (i >= start) & (i < end)
        e = tl.load(TABLE + tl.where(in_shard, i - start, 0) * H + cols, mask=mask & in_shard, other=0.0)
    else:
        e = tl.load(EMBEDS + t * H + cols, mask=mask, other=0.0)
    e = e.to(tl.float32)
    w_e = tl.load(W_EMBED + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    e_normed = e * tl.rsqrt(tl.sum(e * e, axis=0) / H + eps) * w_e
    tl.store(OUT + t * (H + HALF) + cols, e_normed.to(OUT.dtype.element_ty), mask=mask)

    h = tl.load(HIDDEN + t * hidden_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w_h = tl.load(W_HIDDEN + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    h_normed = h * tl.rsqrt(tl.sum(h * h, axis=0) / H + eps) * w_h
    own = cols - RANK * HALF
    tl.store(OUT + t * (H + HALF) + H + own, h_normed.to(OUT.dtype.element_ty), mask=(own >= 0) & (own < HALF))


class DraftArgmaxBuffers:
    """This rank's exchange of the drafter's per-row best (value, token) and count of NaN and +inf logits with the
    peer: the slots the peer writes in symmetric memory, the local epochs, the rows' finished-chunk counts, the chunks'
    partials and the rows' counts over both shards."""

    MAX_CHUNKS = 32  # a row's chunks (kMaxChunks of csrc/draft_argmax.cu)
    instance: "DraftArgmaxBuffers | None" = None

    def __init__(self, group, max_rows: int, device: torch.device):
        # four words per row, (value, epoch, token) and (count, epoch) for each of the epoch's two parities, each word
        # tagged with its epoch: zero is no epoch's tag
        self.slots = symm_mem.empty(4 * max_rows, dtype=torch.int64, device=device)
        self.slots.zero_()
        self.epochs = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.done = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.part_val = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.float32, device=device)
        self.part_idx = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.int32, device=device)
        self.part_nonfinite = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.int32, device=device)
        # the rows' NaN and +inf logits over both shards, which every argmax adds to
        self.nonfinite = torch.zeros(max_rows, dtype=torch.int32, device=device)

        self.handle = symm_mem.rendezvous(self.slots, group.group_name)
        torch.ops.pear_rollout_blackwell.da_setup(
            self.slots,
            self.handle.buffer_ptrs[1 - self.handle.rank],
            self.epochs,
            self.done,
            self.part_val,
            self.part_idx,
            self.part_nonfinite,
            self.nonfinite,
        )

        torch.cuda.synchronize(device)
        torch.distributed.barrier(group)

    @classmethod
    def create(cls, group, max_rows: int, device: torch.device) -> None:
        """Make the process's buffers, once."""
        if cls.instance is None:
            cls.instance = cls(group, max_rows, device)


class DraftFcLinear(ReplicatedLinear):
    """The MTP predictor's fc [embedding | hidden] -> hidden as a row-parallel layer whose embedding block every rank
    holds whole: rank r's weight is [W_embedding | its half of W_hidden] and its input [norm(its partial embedding
    rows) | its half of norm(hidden)], so that the ranks' partial products sum to fc's output."""

    def __init__(self, hidden_size: int, rank: int, quant_config, prefix: str):
        super().__init__(
            hidden_size + hidden_size // TP_SIZE,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=False,
        )
        self.hidden_size, self.rank = hidden_size, rank

    def weight_loader(self, param, loaded_weight):
        # the checkpoint's [hidden, 2 hidden]: the embedding block, then this rank's columns of the hidden block
        half = self.hidden_size // TP_SIZE
        first = self.hidden_size + self.rank * half
        embedding_block, hidden_columns = loaded_weight[:, : self.hidden_size], loaded_weight[:, first : first + half]
        param.data.copy_(torch.cat([embedding_block, hidden_columns], dim=1))


class BlackwellQwen3_5MTP(Qwen3_5RolloutMTP):
    """The rollout's MTP drafter on the drafter's GEMM table and the target's fused kernels: its fc row-parallel on
    each rank's partial embedding rows (DraftFcLinear), its norms fused with the all-reduces on an fp32 residual
    (csrc/ar_norm.cu), its output gate (csrc/output_gate.cu), the MLP's SwiGLU and the row-parallel projections'
    split-K sums on the stream GEMM, and its greedy token one kernel after the lm_head's GEMM, the ranks' best swapped
    over NVLink (csrc/draft_argmax.cu)."""

    FC_INPUT_WARPS = 8  # warps per row of the fc-input kernel
    SWIGLU_MAX_ROWS = 128  # rows up to which the MLP's SwiGLU stream GEMM beats the GEMM table and the activation
    finishes_sample_rows = True  # forward takes sample_rows: the speculator's first pass finishes only those rows
    fc_input_kernel = TritonLauncher(_draft_fc_input_kernel)

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        group = get_tp_group()
        assert group.world_size == TP_SIZE

        # the head the kernels compute: fp32 logits (the target's head dtype, Qwen3_5MTPSpeculatorConfig) of bf16
        # weights, no soft cap or scale, every row of a shard a token of the vocabulary
        processor, shard_indices = self.logits_processor, self.lm_head.shard_indices
        assert vllm_config.model_config.dtype == torch.bfloat16 and processor.head_dtype == torch.float32
        assert processor.soft_cap is None and processor.scale == 1.0 and shard_indices.num_org_vocab_padding == 0
        assert processor.org_vocab_size - shard_indices.org_vocab_start_index >= self.lm_head.weight.shape[0]
        # the fc input's lookup: every row of an embedding shard a token of the vocabulary, no added vocabulary
        embedding_shard = self.model.embed_tokens.shard_indices
        assert embedding_shard.num_org_vocab_padding == 0
        assert embedding_shard.added_vocab_end_index == embedding_shard.added_vocab_start_index

        device = torch.device("cuda", torch.cuda.current_device())
        DraftArgmaxBuffers.create(group.device_group, vllm_config.scheduler_config.max_num_seqs, device)
        # the rows' counts of NaN and +inf logits, which the argmax adds to and the speculator reports per proposal
        self.nonfinite_counts = DraftArgmaxBuffers.instance.nonfinite

        # the fc rebuilt before its weights load
        fc = self.model.fc
        self.model.fc = DraftFcLinear(fc.output_size, group.rank_in_group, fc.quant_config, fc.prefix)

        # every exactly-unquantized linear layer on the drafter's GEMM table, before the weights load; the row-parallel
        # ones (fc, o_proj, down_proj) leave their sums to the next fused norm, which all-reduces them
        for module in self.modules():
            if isinstance(module, LinearBase) and type(module.quant_method) is UnquantizedLinearMethod:
                module.quant_method = DraftGemmTableLinearMethod()
        row_parallel = [self.model.fc]
        for layer in self.model.layers:
            assert not layer.self_attn.o_proj.reduce_results and not layer.mlp.down_proj.reduce_results
            row_parallel += [layer.self_attn.o_proj, layer.mlp.down_proj]
        for linear in row_parallel:
            if type(linear.quant_method) is DraftGemmTableLinearMethod:
                linear.quant_method = DraftPartialsLinearMethod()

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        # this rank's partial rows (the forward's fc sums the ranks'), the image rows on rank 0 and zeros on the other
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0 and get_tp_group().rank_in_group:
            multimodal_embeddings = BlackwellQwen3_5ForConditionalGeneration.zeros(multimodal_embeddings)
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.model.embed_tokens.partial_rows, is_multimodal=is_multimodal
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx=0,
        sample_rows=None,
    ):
        # inputs_embeds, when given, are this rank's partial rows (embed_input_ids); sample_rows, when given, the rows
        # the layer finishes past its attention (the others only write their KV) and the rows returned
        mtp = self.model
        layer = mtp.layers[spec_step_idx % mtp.num_mtp_layers]
        ar_add_rmsnorm = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm

        normed, residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(
            mtp.fc(self.fc_input(input_ids, inputs_embeds, hidden_states)),
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
        )
        attention_layer = layer.self_attn
        qkv, _ = attention_layer.qkv_proj(normed)
        q, k, v, gate = attention_layer._project_qkv_gate(qkv, positions)
        attention = torch.ops.pear_rollout_blackwell.output_gate(attention_layer.attn(q, k, v), gate)
        if sample_rows is not None:
            attention, residual = attention[sample_rows], residual[sample_rows]
        attention, _ = attention_layer.o_proj(attention)
        normed, residual = ar_add_rmsnorm(
            attention,
            residual,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
            False,
        )

        mlp = layer.mlp
        if StreamGemm.takes(normed, mlp.gate_up_proj.weight, self.SWIGLU_MAX_ROWS):
            down, _ = mlp.down_proj(torch.ops.pear_rollout_blackwell.sg_swiglu(normed, mlp.gate_up_proj.weight))
        else:
            down = mlp(normed)
        hidden_states, _ = ar_add_rmsnorm(down, residual, mtp.norm.weight, mtp.norm.variance_epsilon, False)

        return hidden_states

    def fc_input(self, input_ids, inputs_embeds, hidden_states):
        """[norm(partial embedding rows) | this rank's half of norm(hidden)] for the fc, one kernel."""
        mtp = self.model
        shard_indices = mtp.embed_tokens.shard_indices
        tokens, hidden_size = hidden_states.shape
        out = hidden_states.new_empty((tokens, hidden_size + hidden_size // TP_SIZE))
        if tokens:
            lookup = inputs_embeds is None
            self.fc_input_kernel(
                (lookup, hidden_size, mtp.fc.rank, input_ids.dtype if lookup else None, hidden_states.dtype),
                (tokens,),
                input_ids if lookup else hidden_states,
                hidden_states if lookup else inputs_embeds,
                mtp.embed_tokens.weight,
                hidden_states,
                hidden_states.stride(0),
                mtp.pre_fc_norm_embedding.weight,
                mtp.pre_fc_norm_hidden.weight,
                out,
                shard_indices.org_vocab_start_index,
                shard_indices.org_vocab_end_index,
                mtp.pre_fc_norm_embedding.variance_epsilon,
                num_warps=self.FC_INPUT_WARPS,
                H=hidden_size,
                HALF=hidden_size // TP_SIZE,
                RANK=mtp.fc.rank,
                BLOCK=triton.next_power_of_2(hidden_size),
                LOOKUP=lookup,
            )

        return out

    def get_top_tokens(self, hidden_states, spec_step_idx=0):
        # this rank's shard's fp32 logits on the drafter's GEMM table, whose argmax over the two ranks the kernel takes
        logits = torch.ops.pear_rollout_blackwell.dg_mm(hidden_states, self.lm_head.weight, True)
        return torch.ops.pear_rollout_blackwell.draft_argmax(logits, self.lm_head.shard_indices.org_vocab_start_index)
