import numpy as np
import torch
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides, is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)

from pear.models.qwen3_5.rollout.hopper.staging import StagedBuffer

HEAD_DIM = 256  # the attention layers' head dim (csrc/split_attn.cu HD)
QUERY_HEADS = 6  # query heads per KV head at TP4 (csrc/split_attn.cu HQ)
MAX_SEQS = 256  # sequences a step may have (csrc/split_attn.cu MAX_SEQS)
# the largest query a step is described with to FA3 at least: its metadata at max_seqlen_q == 1 has another layout
MIN_QUERY_LEN = 2


class AttentionStepBuffers(StagedBuffer):
    """A builder's fixed device buffers for the attention arguments of a step and their pinned staging copies."""

    # the decode-row count, FA3's query starts (+ total), and the query and KV lengths of the whole batch (FA3's
    # scheduler metadata) and of FA3's part (the decode rows zeroed): one upload, at fixed addresses for the graphs
    def __init__(self, max_seqs: int, device):
        sizes = [1, max_seqs + 1, max_seqs, max_seqs, max_seqs, max_seqs]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        super().__init__(int(self.offsets[-1]), device)
        self.num_decode_rows, self.cu, self.sq_all, self.sk_all, self.sq, self.sk = torch.split(
            self.device_buffer, sizes
        )

    def fill(self, num_decode_rows: int, starts, total: int, qlens, klens) -> None:
        n = len(qlens)
        stage = self.next_stage()
        o = self.offsets

        stage[0] = num_decode_rows
        stage[o[1] : o[1] + n] = starts
        stage[o[1] + n] = total
        stage[o[2] : o[2] + n] = qlens
        stage[o[3] : o[3] + n] = klens
        stage[o[4] : o[4] + n] = qlens
        stage[o[5] : o[5] + n] = klens
        stage[o[4] : o[4] + num_decode_rows] = 0
        stage[o[5] : o[5] + num_decode_rows] = 0

        self.upload(int(o[5]) + n)


class HopperFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """vLLM's FlashAttention metadata builder with the attention arguments of a FULL-graph step: its decode rows and
    FA3's part."""

    PAGE = 784  # the KV cache's block size in tokens (csrc/split_attn.cu PAGE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_buffers = None  # the attention arguments' device buffers, made at the first step that takes them

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        m = common_attn_metadata

        # the steps a FULL graph may replay, by vLLM's own test for its FA3 metadata
        if (
            common_prefix_len
            or fast_build
            or not self.use_full_cuda_graph
            or m.num_actual_tokens > self.max_cudagraph_size
            or not self.eligible(m)
        ):
            return super().build(common_prefix_len, m, fast_build=fast_build)

        self.aot_schedule = False  # vLLM's metadata without its scheduler metadata (FA3's batch is another one)

        try:
            md = super().build(common_prefix_len, m, fast_build=fast_build)
        finally:
            self.aot_schedule = True

        if md.use_cascade or md.mm_prefix_query_range_tensor is not None:
            return super().build(common_prefix_len, m, fast_build=fast_build)

        # a pure decode step (vLLM captures the mixed-step graphs at max_query_len = their token count)
        md._pear_attention_step = self.attention_step(m, md, is_pure_decode=m.max_query_len == 1)

        return md

    def eligible(self, m) -> bool:
        """A step the decode kernel computes exactly as FA3 would: plain causal attention over the paged cache."""
        return (
            self.aot_schedule
            and self.dcp_world_size == 1
            and self.headdim == HEAD_DIM
            and self.num_heads_kv == 1
            and self.num_heads_q == QUERY_HEADS
            and self.block_size == self.PAGE
            and self.kv_cache_dtype == torch.bfloat16
            and m.num_reqs <= min(self.vllm_config.scheduler_config.max_num_seqs, MAX_SEQS)
            and isinstance(m.causal, bool)
            and m.causal
            and m.mm_req_doc_ranges is None
            and self.rswa_window is None
            and self.aot_sliding_window in (None, (-1, -1))
        )

    def attention_step(self, m, md, is_pure_decode: bool) -> tuple:
        """The step's attention arguments: its decode rows (the leading one-token queries) and FA3's part on
        whole-batch metadata."""
        n = m.num_reqs
        qsl = m.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
        qlens = np.diff(qsl)
        klens = (m._seq_lens_cpu if m._seq_lens_cpu is not None else m.seq_lens.cpu())[:n].numpy().astype(np.int64)
        prefill_rows = np.flatnonzero(qlens != 1)
        num_decode_rows = int(prefill_rows[0]) if len(prefill_rows) else n

        buffers = self.step_buffers
        if buffers is None:
            max_seqs = min(self.vllm_config.scheduler_config.max_num_seqs, MAX_SEQS)
            buffers = self.step_buffers = AttentionStepBuffers(max_seqs, self.device)
        buffers.fill(num_decode_rows, qsl[:-1], int(qsl[-1]), qlens, klens)
        cu = buffers.cu[: n + 1]

        # FA3's scheduler metadata as for the whole batch on every SM, so its split-KV count per sequence (and the
        # prefill rows' summation order) is its own decision for the step; and never max_seqlen_q == 1, whose layout
        # differs: a mixed graph captured from single tokens would replay a real step's metadata in the wrong layout
        scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
            n,
            max(MIN_QUERY_LEN, md.max_query_len),
            md.max_seq_len,
            self.num_heads_q,
            self.num_heads_kv,
            self.headdim,
            self.headdim,
            self.kv_cache_dtype,
            buffers.sk_all[:n],
            cu,
            None,
            None,
            buffers.sq_all[:n],
            None,
            self.block_size,
            0,
            md.causal,
            -1,
            -1,
            False,
            md.max_num_splits,
            None,
            0,
        )
        md.scheduler_metadata = self._store_scheduler_metadata(scheduler_metadata)

        return buffers.num_decode_rows, cu, buffers.sq[:n], buffers.sk[:n], n, is_pure_decode


class HopperFlashAttentionImpl(FlashAttentionImpl):
    """vLLM's FlashAttention with the decode rows of a FULL-graph step on the split-KV decode kernel
    (csrc/split_attn.cu), the prefill rows of a mixed one on FlashAttention-3 beside it."""

    # FA3 runs decode rows (one query token each) as tiles that are mostly padding and rounds P to bf16; the kernel
    # has a row per query head and keeps more of P's bits
    MIXED_CTAS = 48  # the decode kernel's CTAs beside FlashAttention-3 in a mixed step (FA3 keeps the other SMs)
    DECODE_CTAS = 132  # the decode kernel's CTAs in a pure decode step, where it runs alone (H100's SMs)
    MAX_CTAS = 160  # the kernel's partial slots (csrc/split_attn.cu MAX_CTAS_ALLOWED)
    device_states: dict = {}  # device -> the kernel's partials, counters and the mixed steps' side stream

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None
    ):
        step = getattr(attn_metadata, "_pear_attention_step", None) if attn_metadata is not None else None
        if step is None or not self.eligible(output_scale, output_block_scale):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        self.attend(query, kv_cache, attn_metadata, output, step)

        return output

    def eligible(self, output_scale, output_block_scale) -> bool:
        """An attention layer the decode kernel computes (FA3, unquantized, no sinks, windows, softcap or ALiBi)."""
        return (
            output_scale is None
            and output_block_scale is None
            and self.vllm_flash_attn_version == 3
            and self.alibi_slopes is None
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap in (None, 0, 0.0)
            and self.dcp_world_size <= 1
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and self.attn_type == AttentionType.DECODER
        )

    def attend(self, query, kv_cache, md, output, step) -> None:
        """The decode kernel alone in a pure decode step; in a mixed one beside FA3 over the prefill rows."""
        num_decode_rows, cu, sq, sk, n, is_pure_decode = step
        num_tokens = md.num_actual_tokens
        kv = kv_cache.transpose(1, 2)  # [blocks, page, kv heads, K | V]
        q = query[:num_tokens].view(num_tokens, QUERY_HEADS, HEAD_DIM)
        out = output[:num_tokens].view(num_tokens, QUERY_HEADS, HEAD_DIM)
        state = self.device_state(query.device)

        if is_pure_decode:
            torch.ops.pear_rollout_hopper.split_decode_attention(
                q,
                kv,
                md.block_table,
                md.seq_lens,
                num_decode_rows,
                out,
                state["part_o"],
                state["part_ml"],
                state["counters"],
                self.scale,
                self.DECODE_CTAS,
                n,
            )
            return

        # the decode kernel (no allocations) on the side stream; FA3 allocates its softmax statistics, which in a graph
        # capture must come from the capturing stream
        key_cache, value_cache = kv.split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        main_stream = torch.cuda.current_stream()
        side_stream = state["stream"]
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            torch.ops.pear_rollout_hopper.split_decode_attention(
                q,
                kv,
                md.block_table,
                md.seq_lens,
                num_decode_rows,
                out,
                state["part_o"],
                state["part_ml"],
                state["counters"],
                self.scale,
                self.MIXED_CTAS,
                n,
            )

        torch.ops._vllm_fa3_C.fwd(
            query[:num_tokens],
            key_cache,
            value_cache,
            None,
            None,
            None,
            output[:num_tokens],
            cu,
            None,
            None,
            sq,
            sk,
            max(MIN_QUERY_LEN, md.max_query_len),
            md.max_seq_len,
            md.block_table,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.scale,
            True,
            -1,
            -1,
            0.0,
            True,
            md.scheduler_metadata,
            md.max_num_splits,
            None,
            self.MIXED_CTAS,
            None,
            1,
            0,
            None,
        )

        main_stream.wait_stream(side_stream)

    @classmethod
    def device_state(cls, device) -> dict:
        if device not in cls.device_states:
            cls.device_states[device] = {
                "part_o": torch.empty(
                    (cls.MAX_CTAS + MAX_SEQS) * QUERY_HEADS * HEAD_DIM, dtype=torch.float32, device=device
                ),
                "part_ml": torch.empty((cls.MAX_CTAS + MAX_SEQS) * QUERY_HEADS * 2, dtype=torch.float32, device=device),
                "counters": torch.zeros(MAX_SEQS, dtype=torch.int32, device=device),
                "stream": torch.cuda.Stream(device=device),
            }

        return cls.device_states[device]


class HopperFlashAttentionBackend(FlashAttentionBackend):
    """vLLM's FlashAttention backend with the decode kernel's builder and impl (registered in place of FLASH_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return HopperFlashAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return HopperFlashAttentionImpl
