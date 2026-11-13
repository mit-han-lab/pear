import torch
import torch.distributed._symmetric_memory as symm_mem
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.triton_utils import tl, triton

from pear.models.qwen3_5.rollout.blackwell.gemm import GemmTableLinearMethod
from pear.models.qwen3_5.rollout.blackwell.rope import Fp32Rope
from pear.models.qwen3_5.rollout.blackwell.triton_launcher import TritonLauncher
from pear.models.qwen3_5.rollout.model import (
    Qwen3_5RolloutDecoderLayer,
    Qwen3_5RolloutForCausalLM,
    Qwen3_5RolloutForConditionalGeneration,
    Qwen3_5RolloutModel,
)

TP_SIZE = 2  # the fused all-reduce and the logits gather exchange with one peer


@triton.jit(do_not_specialize=["start", "end"])
def _shard_embed_kernel(IDS, W, OUT, start, end, H: tl.constexpr, BH: tl.constexpr):
    # this rank's vocabulary shard [start, end): the token's row, zeros for tokens of the other shard
    t = tl.program_id(0)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    i = tl.load(IDS + t).to(tl.int64)

    in_shard = (i >= start) & (i < end)
    row = tl.where(in_shard, i - start, 0)
    x = tl.load(W + row * H + cols, mask=in_shard & (cols < H), other=0.0)

    tl.store(OUT + t.to(tl.int64) * H + cols, x, mask=cols < H)


class AllReduceBuffers:
    """This rank's receive buffers of the fused all-reduce in symmetric memory, the peer's mapped, the local epochs."""

    CTAS_PER_SM = 2  # CTAs of the fused all-reduce per SM
    SENTINEL = -32768  # bf16 -0.0 as int16 (0x8000): the all-reduce's "not arrived" mark in its receive buffers
    instance: "AllReduceBuffers | None" = None

    def __init__(self, group, max_tokens: int, hidden_size: int, device: torch.device):
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count * self.CTAS_PER_SM

        # two rotations of [max_tokens, hidden_size]; the peer writes them, every element starts as "not arrived"
        self.receive_buffers = symm_mem.empty(2, max_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        self.receive_buffers.view(torch.int16).fill_(self.SENTINEL)
        self.counters = torch.zeros(self.ctas, dtype=torch.int32, device=device)

        self.handle = symm_mem.rendezvous(self.receive_buffers, group.group_name)
        peer = self.handle.buffer_ptrs[1 - self.handle.rank]
        torch.ops.pear_rollout_blackwell.ar_setup(self.receive_buffers, peer, self.counters, self.ctas)

        torch.cuda.synchronize(device)
        torch.distributed.barrier(group)

    @classmethod
    def create(cls, group, max_tokens: int, hidden_size: int, device: torch.device) -> None:
        """Make the process's buffers, once."""
        if cls.instance is None:
            cls.instance = cls(group, max_tokens, hidden_size, device)


class LogitsGatherBuffers:
    """This rank's gathered-logits buffer and its flags in symmetric memory (the peer writes both), the local epochs."""

    instance: "LogitsGatherBuffers | None" = None

    def __init__(self, group, max_rows: int, vocab_size: int, device: torch.device):
        self.rank = torch.distributed.get_rank(group)
        self.max_rows, self.vocab_size = max_rows, vocab_size
        ctas = torch.cuda.get_device_properties(device).multi_processor_count  # one co-resident CTA per SM

        self.buffer = symm_mem.empty(max_rows, vocab_size, dtype=torch.float32, device=device)
        self.flags = symm_mem.empty(ctas, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.epochs = torch.zeros(ctas, dtype=torch.int32, device=device)

        self.handles = (
            symm_mem.rendezvous(self.buffer, group.group_name),
            symm_mem.rendezvous(self.flags, group.group_name),
        )
        buffer_handle, flag_handle = self.handles
        torch.ops.pear_rollout_blackwell.lg_setup(
            self.buffer,
            buffer_handle.buffer_ptrs[1 - buffer_handle.rank],
            self.flags,
            flag_handle.buffer_ptrs[1 - flag_handle.rank],
            self.epochs,
            ctas,
        )

        torch.cuda.synchronize(device)
        torch.distributed.barrier(group)

    @classmethod
    def create(cls, group, max_rows: int, vocab_size: int, device: torch.device) -> None:
        """Make the process's buffers, once."""
        if cls.instance is None:
            cls.instance = cls(group, max_rows, vocab_size, device)


class BlackwellQwen3_5DecoderLayer(Qwen3_5RolloutDecoderLayer):
    """A dense decoder layer on the fp32 residual stream, whose row-parallel outputs the next fused norm reduces."""

    # every decoder layer's two norms become one op each (csrc/ar_norm.cu): the all-reduce of the row-parallel
    # projection's bf16 partial sums over the two ranks (one-shot NVLink push), the add into the fp32 residual and the
    # (1 + w) RMSNorm whose bf16 output feeds the next GEMMs; the row-parallel layers stop reducing their outputs
    reduce_results = False

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        # the first layer sums the ranks' partial embeddings into the residual
        eps = self.input_layernorm.variance_epsilon
        if residual is None:
            normed_states, residual = torch.ops.pear_rollout_blackwell.ar_rmsnorm(
                hidden_states, self.input_layernorm.weight, eps
            )
        else:
            normed_states, residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
                hidden_states, residual, self.input_layernorm.weight, eps, False
            )

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=normed_states)
        else:
            hidden_states = self.self_attn(hidden_states=normed_states, positions=positions)

        normed_states, residual = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            False,
        )

        return self.mlp(normed_states), residual


class UnreducedVocabParallelEmbedding(VocabParallelEmbedding):
    """The vocab-parallel lookup without its all-reduce: the first decoder layer's norm sums the ranks' partial rows."""

    EMBED_COLUMNS = 1024  # hidden columns per program of the shard lookup
    shard_embed = TritonLauncher(_shard_embed_kernel)

    def forward(self, input_):
        shard_indices = self.shard_indices
        weight = self.weight
        if (
            shard_indices.num_org_vocab_padding == 0
            and shard_indices.added_vocab_end_index == shard_indices.added_vocab_start_index
            and input_.dim() == 1
            and input_.is_contiguous()
            and weight.is_contiguous()
            and not torch.compiler.is_compiling()
        ):
            # one launch: the lookup masked to this rank's shard (vLLM: a compiled mask, the lookup, a fill)
            H = weight.shape[1]
            out = torch.empty(input_.shape[0], H, dtype=weight.dtype, device=weight.device)
            if input_.shape[0]:
                self.shard_embed(
                    (input_.dtype, weight.dtype, H),
                    (input_.shape[0], triton.cdiv(H, self.EMBED_COLUMNS)),
                    input_,
                    weight,
                    out,
                    shard_indices.org_vocab_start_index,
                    shard_indices.org_vocab_end_index,
                    H=H,
                    BH=self.EMBED_COLUMNS,
                )

            return out

        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            shard_indices.org_vocab_start_index,
            shard_indices.org_vocab_end_index,
            shard_indices.num_org_vocab_padding,
            shard_indices.added_vocab_start_index,
            shard_indices.added_vocab_end_index,
        )
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

        return output_parallel


class BlackwellQwen3_5Model(Qwen3_5RolloutModel):
    """The Qwen3.5 language model on an fp32 residual stream with fused all-reduces and RMSNorms, its rotary embedding
    and its final norm's output in fp32."""

    decoder_layer_cls = BlackwellQwen3_5DecoderLayer
    embedding_cls = UnreducedVocabParallelEmbedding

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        group = get_tp_group()
        # the layers' outputs leave through the fp32 residual: no pipeline stages, no auxiliary hidden states
        assert get_pp_group().world_size == 1 and not self.aux_hidden_state_layers and group.world_size == TP_SIZE

        device = torch.device("cuda", torch.cuda.current_device())
        LogitsGatherBuffers.create(
            group.device_group, vllm_config.scheduler_config.max_num_seqs, self.config.vocab_size, device
        )
        AllReduceBuffers.create(
            group.device_group, vllm_config.scheduler_config.max_num_batched_tokens, self.config.hidden_size, device
        )

        Fp32Rope.install(self)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)

        hidden_states, _ = torch.ops.pear_rollout_blackwell.ar_add_rmsnorm(
            hidden_states, residual, self.norm.weight, self.norm.variance_epsilon, True
        )

        return hidden_states


class HighLowLogitsProcessor(LogitsProcessor):
    """vLLM's logits with the fp32 hidden states into the bf16 lm_head as exact high + low parts, the shards gathered
    over NVLink."""

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        if not self.eligible(lm_head, hidden_states, embedding_bias):
            return super()._apply_head(lm_head, hidden_states, embedding_bias)

        high_low_logits, n = self.high_low_head(lm_head, hidden_states)

        return (high_low_logits[:n] + high_low_logits[n:]).reshape(*hidden_states.shape[:-1], -1)

    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        gather = LogitsGatherBuffers.instance
        if not (
            gather is not None
            and lm_head.tp_size == TP_SIZE
            and hidden_states.dim() == 2
            and hidden_states.shape[0] <= gather.max_rows
            and TP_SIZE * lm_head.weight.shape[0] == gather.vocab_size == self.org_vocab_size
            and self.eligible(lm_head, hidden_states, embedding_bias)
        ):
            return super()._get_logits(hidden_states, lm_head, embedding_bias)

        high_low_logits, n = self.high_low_head(lm_head, hidden_states)

        # high + low and the vocabulary gather in one NVLink kernel: [n, vocab], valid until the next step's logits
        return torch.ops.pear_rollout_blackwell.lg_gather(high_low_logits, n, gather.rank)

    @staticmethod
    def eligible(lm_head, hidden_states, embedding_bias) -> bool:
        """fp32 hidden states into an unquantized bf16 lm_head without bias."""
        return (
            hidden_states.dtype == torch.float32
            and lm_head.weight.dtype == torch.bfloat16
            and isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod)
            and embedding_bias is None
        )

    @staticmethod
    def high_low_head(lm_head, hidden_states) -> tuple[torch.Tensor, int]:
        """The head's fp32 product of [high rows; low rows] and the row count."""
        # one GEMM of twice the rows: decode-sized GEMMs are bound by the weight read
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        high_part = flat_states.to(torch.bfloat16)
        low_part = (flat_states - high_part.float()).to(torch.bfloat16)

        return (
            torch.mm(torch.cat([high_part, low_part]), lm_head.weight.t(), out_dtype=torch.float32),
            flat_states.shape[0],
        )


class BlackwellQwen3_5ForCausalLM(Qwen3_5RolloutForCausalLM):
    """The Qwen3.5 causal LM of a datacenter Blackwell replica: the fp32 residual model and the high + low logits."""

    model_cls = BlackwellQwen3_5Model
    logits_processor_cls = HighLowLogitsProcessor


class BlackwellQwen3_5ForConditionalGeneration(Qwen3_5RolloutForConditionalGeneration):
    """The Qwen3.5 vision-language model of a datacenter Blackwell replica, its linear layers on the GEMM table."""

    causal_lm_cls = BlackwellQwen3_5ForCausalLM
    linear_method_cls = GemmTableLinearMethod

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        # the image rows: rank 0 contributes them, the other rank zeros (its text rows are its partial lookup)
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0 and get_tp_group().rank_in_group:
            multimodal_embeddings = self.zeros(multimodal_embeddings)

        return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    @staticmethod
    def zeros(embeddings):
        """Zeros shaped like the (nested) multimodal embeddings."""
        if isinstance(embeddings, torch.Tensor):
            return torch.zeros_like(embeddings)

        return [BlackwellQwen3_5ForConditionalGeneration.zeros(e) for e in embeddings]
