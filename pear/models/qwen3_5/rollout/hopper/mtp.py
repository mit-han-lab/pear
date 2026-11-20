import json
from pathlib import Path

import torch
import torch.distributed._symmetric_memory as symm_mem
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.models.interfaces import _require_is_multimodal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.triton_utils import tl, triton

from pear.models.qwen3_5.rollout.hopper.all_reduce import AllReduceNorm
from pear.models.qwen3_5.rollout.mtp import Qwen3_5RolloutMTP


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
    PART: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOKUP: tl.constexpr,
):
    # row t of the drafter's fc input on this rank: the (1 + w) RMSNorm of its partial embedding row (the token's row of
    # this rank's shard [start, end), else zeros, whose norm is zeros), then this rank's part of the normed hidden state;
    # vLLM's fused_pre_fc_norms arithmetic
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
    e_normed = (e * tl.rsqrt(tl.sum(e * e, axis=0) / H + eps)) * w_e
    tl.store(OUT + t * (H + PART) + cols, e_normed.to(OUT.dtype.element_ty), mask=mask)

    h = tl.load(HIDDEN + t * hidden_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w_h = tl.load(W_HIDDEN + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    h_normed = (h * tl.rsqrt(tl.sum(h * h, axis=0) / H + eps)) * w_h
    own = cols - RANK * PART
    tl.store(OUT + t * (H + PART) + H + own, h_normed.to(OUT.dtype.element_ty), mask=(own >= 0) & (own < PART))


class DraftArgmaxBuffers:
    """The drafter's argmax exchange (csrc/draft_argmax.cu): the slots every rank writes its rows' best and counts of
    NaN and +inf logits into and a flag per rank, in symmetric memory, every rank's addresses of them, this rank's call
    count, its chunks' partials and the rows' counts over every shard."""

    WORLD = 4  # the ranks exchanging (kWorld of csrc/draft_argmax.cu)
    MAX_CHUNKS = 32  # a row's chunks (kMaxChunks)
    SLOT_WORDS = 2  # a row's slot: (value bits, token), then its count (kSlotWords)
    instance: "DraftArgmaxBuffers | None" = None

    def __init__(self, group, max_rows: int, device: torch.device):
        # the slots by the call's parity and the sending rank; the flags hold the last call each rank sent, so they start
        # at zero on every rank before any rank writes them (the barrier)
        self.slots = symm_mem.empty(2 * self.WORLD * max_rows * self.SLOT_WORDS, dtype=torch.int64, device=device)
        self.flags = symm_mem.empty(self.WORLD, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.calls = torch.zeros(1, dtype=torch.int32, device=device)
        self.part_val = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.float32, device=device)
        self.part_idx = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.int32, device=device)
        self.part_nonfinite = torch.empty(max_rows * self.MAX_CHUNKS, dtype=torch.int32, device=device)
        # the rows' NaN and +inf logits over every shard, which every argmax adds to
        self.nonfinite = torch.zeros(max_rows, dtype=torch.int32, device=device)

        slots_handle = symm_mem.rendezvous(self.slots, group.group_name)
        flags_handle = symm_mem.rendezvous(self.flags, group.group_name)
        self.handles = (slots_handle, flags_handle)  # held: the mappings live as long as their handles
        self.rank_slots = list(slots_handle.buffer_ptrs)
        self.rank_flags = list(flags_handle.buffer_ptrs)
        self.rank = slots_handle.rank

        torch.cuda.synchronize(device)
        torch.distributed.barrier(group)

    @classmethod
    def create(cls, group, max_rows: int, device: torch.device) -> None:
        """Make the process's buffers, once."""
        if cls.instance is None:
            cls.instance = cls(group, max_rows, device)

    def argmax(self, logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
        """The drafts (int64 [rows]) of this rank's fp32 shard logits [rows, shard] (tokens from vocab_start): the
        best over every rank's shard, the same on every rank; the rows' NaN and +inf logits added to nonfinite."""
        return torch.ops.pear_rollout_hopper.draft_argmax(
            logits,
            vocab_start,
            self.slots,
            self.flags,
            self.rank_slots,
            self.rank_flags,
            self.calls,
            self.part_val,
            self.part_idx,
            self.part_nonfinite,
            self.nonfinite,
            self.rank,
        )


class DraftGemmLinearMethod(UnquantizedLinearMethod):
    """The drafter's bf16 linear layers on its cuBLASLt GEMMs (csrc/draft_gemm.cpp): the table's heuristic candidate
    per shape, cuBLAS's own choice otherwise."""

    def apply(self, layer, x, bias=None):
        if bias is None:
            return torch.ops.pear_rollout_hopper.dg_mm(x, layer.weight, False)

        return super().apply(layer, x, bias)


class DraftFcLinear(ReplicatedLinear):
    """The MTP predictor's fc [embedding | hidden] -> hidden as a row-parallel layer whose embedding block every rank
    holds whole: rank r's weight is [W_embedding | rank r's part of W_hidden], its input [norm(its partial embedding
    rows) | its part of norm(hidden)], so that the ranks' partial products sum to fc's output (each embedding row is
    whole on one rank and zeros on the others)."""

    def __init__(self, hidden_size: int, rank: int, world_size: int, quant_config, prefix: str):
        self.hidden_size, self.rank, self.part = hidden_size, rank, hidden_size // world_size
        super().__init__(
            hidden_size + self.part,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=False,
        )

    def weight_loader(self, param, loaded_weight):
        # the checkpoint's [hidden, 2 hidden]: the embedding block, then this rank's columns of the hidden block
        first = self.hidden_size + self.rank * self.part
        embedding_block = loaded_weight[:, : self.hidden_size]
        hidden_columns = loaded_weight[:, first : first + self.part]
        param.data.copy_(torch.cat([embedding_block, hidden_columns], dim=1))


class HopperQwen3_5MTP(Qwen3_5RolloutMTP):
    """The rollout's MTP drafter on the Hopper replica's fused kernels: its fc row-parallel on each rank's partial
    embedding rows (DraftFcLinear, no embedding all-reduce and no gather), its three norms fused with the all-reduces
    (AllReduceNorm), and its greedy token two kernels after the lm_head's cuBLASLt GEMM (csrc/draft_gemm.cpp), the
    ranks' best exchanged over NVLink (DraftArgmaxBuffers)."""

    FC_INPUT_WARPS = 8  # warps per row of the fc-input kernel
    # (rows, N, K, fp32 out, candidate): the cuBLASLt heuristic candidate of the drafter's GEMM shapes where it beats
    # cuBLAS's own choice on H200 (one GPU, CUDA graphs, cold weights); other shapes take candidate 0
    GEMM_TABLE = Path(__file__).resolve().parent / "draft_gemm_table.json"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        group = get_tp_group()
        processor, shard_indices = self.logits_processor, self.lm_head.shard_indices

        # vLLM's argmax of the fp32 logits as they are: no soft cap, scale or masked padding in any shard
        if group.world_size != DraftArgmaxBuffers.WORLD:
            raise NotImplementedError(f"the drafter runs at TP {DraftArgmaxBuffers.WORLD}, not {group.world_size}")
        if processor.head_dtype != torch.float32 or processor.soft_cap is not None or processor.scale != 1.0:
            raise NotImplementedError("the drafter's argmax takes the plain fp32 logits of PEAR's lm_head")
        if shard_indices.num_org_vocab_padding or processor.org_vocab_size % group.world_size:
            raise NotImplementedError("the drafter's argmax and fc input take vocabulary shards without padding")

        device = torch.device("cuda", torch.cuda.current_device())
        DraftArgmaxBuffers.create(group.device_group, vllm_config.scheduler_config.max_num_seqs, device)
        # the rows' counts of NaN and +inf logits, which the argmax adds to and the speculator reports per proposal
        self.nonfinite_counts = DraftArgmaxBuffers.instance.nonfinite
        for rows, n, k, is_fp32, candidate in json.loads(self.GEMM_TABLE.read_text())["rows"]:
            torch.ops.pear_rollout_hopper.dg_set(rows, n, k, is_fp32, candidate)

        # the fc rebuilt and every unquantized linear layer on the drafter's GEMMs, before the weights load
        fc = self.model.fc
        self.model.fc = DraftFcLinear(fc.output_size, group.rank_in_group, group.world_size, fc.quant_config, fc.prefix)
        for module in self.model.modules():
            if isinstance(module, LinearBase) and type(module.quant_method) is UnquantizedLinearMethod:
                module.quant_method = DraftGemmLinearMethod()

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        # this rank's partial rows (the forward's fc sums the ranks'): text rows of its shard, image rows on rank 0
        # and zeros on the others
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0 and get_tp_group().rank_in_group:
            multimodal_embeddings = [torch.zeros_like(embedding) for embedding in multimodal_embeddings]
        inputs_embeds = self._embed_text_input_ids(input_ids, self.partial_rows, is_multimodal=is_multimodal)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0
    ):
        # inputs_embeds, when given, are this rank's partial rows (embed_input_ids). Every norm takes the preceding
        # row-parallel output's all-reduce; the first starts the fp32 residual stream from the fc's sum
        mtp = self.model
        layer = mtp.layers[spec_step_idx % mtp.num_mtp_layers]

        x = mtp.fc(self.fc_input(input_ids, inputs_embeds, hidden_states))
        x, residual = AllReduceNorm.apply(x, torch.zeros_like(x, dtype=torch.float32), layer.input_layernorm)
        x = layer.self_attn(positions=positions, hidden_states=x)
        x, residual = AllReduceNorm.apply(x, residual, layer.post_attention_layernorm)
        x = layer.mlp(x)
        x, _ = AllReduceNorm.apply(x, residual, mtp.norm)

        return x

    def fc_input(self, input_ids, inputs_embeds, hidden_states):
        """[norm(partial embedding rows) | this rank's part of norm(hidden)] for the fc, one kernel (the embedding rows
        looked up from this rank's shard unless given)."""
        mtp = self.model
        fc, shard_indices = mtp.fc, mtp.embed_tokens.shard_indices
        tokens, hidden_size = hidden_states.shape
        out = hidden_states.new_empty((tokens, hidden_size + fc.part))
        if tokens == 0:
            return out

        is_lookup = inputs_embeds is None
        _draft_fc_input_kernel[(tokens,)](
            input_ids,
            hidden_states if is_lookup else inputs_embeds,
            mtp.embed_tokens.weight,
            hidden_states,
            hidden_states.stride(0),
            mtp.pre_fc_norm_embedding.weight,
            mtp.pre_fc_norm_hidden.weight,
            out,
            shard_indices.org_vocab_start_index,
            shard_indices.org_vocab_end_index,
            mtp.pre_fc_norm_embedding.variance_epsilon,
            H=hidden_size,
            PART=fc.part,
            RANK=fc.rank,
            BLOCK=triton.next_power_of_2(hidden_size),
            LOOKUP=is_lookup,
            num_warps=self.FC_INPUT_WARPS,
        )

        return out

    def partial_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The embedding rows of the tokens in this rank's vocabulary shard, zeros for the others (vLLM's masked
        lookup without its all-reduce)."""
        embedding = self.model.embed_tokens
        shard_indices = embedding.shard_indices
        in_shard = (input_ids >= shard_indices.org_vocab_start_index) & (input_ids < shard_indices.org_vocab_end_index)
        rows = embedding.weight[torch.where(in_shard, input_ids - shard_indices.org_vocab_start_index, 0)]

        return rows.masked_fill_(~in_shard.unsqueeze(-1), 0)

    def get_top_tokens(self, hidden_states, spec_step_idx=0):
        # this rank's shard's fp32 logits as vLLM computes them (bf16 operands, fp32 accumulation and output) on the
        # table's cuBLASLt candidate, then the ranks' argmax
        logits = torch.ops.pear_rollout_hopper.dg_mm(hidden_states, self.lm_head.weight, True)

        return DraftArgmaxBuffers.instance.argmax(logits, self.lm_head.shard_indices.org_vocab_start_index)
