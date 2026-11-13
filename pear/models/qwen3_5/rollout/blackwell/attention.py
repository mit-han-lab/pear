import torch
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends import flashinfer as vllm_flashinfer
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferImpl,
    FlashInferTrtllmAPIDecode,
    TRTLLMPrefill,
)

# the factor on the SM count FlashInfer's trtllm-gen decode launcher sizes its split-KV from (SMs / (batch x KV head
# groups) CTAs per sequence): a sequence over more CTAs, merged by the kernel's own multi-CTA reduction, so that a
# step waits less for its longest context
SPLIT_FACTOR = 8


class DecodeSplit:
    """FlashInfer's TRT-LLM decode attention split over SPLIT_FACTOR times more CTAs per sequence in vLLM's own
    FlashInfer path (BlackwellFlashInferImpl's other steps), through FlashInfer's SM count."""

    COUNTER_BYTES = 1 << 16  # the multi-CTA reduction's semaphore buffer (the kernel resets what it uses)
    counters: dict = {}  # device -> the semaphore buffer, kept instead of allocated per call

    @classmethod
    def install(cls) -> None:
        """Patch vLLM's call of FlashInfer's TRT-LLM decode."""
        # FlashInfer sizes the split from its own SM count, which neither it nor vLLM lets a caller set
        import flashinfer.decode as flashinfer_decode

        vllm_trtllm_batch_decode_with_kv_cache = vllm_flashinfer.trtllm_batch_decode_with_kv_cache
        flashinfer_get_device_sm_count = flashinfer_decode.get_device_sm_count

        def boosted_sm_count(device):
            return flashinfer_get_device_sm_count(device) * SPLIT_FACTOR

        def trtllm_batch_decode_with_kv_cache(*args, **kwargs):
            if kwargs.get("multi_ctas_kv_counter_buffer") is None:
                query = kwargs["query"] if "query" in kwargs else args[0]
                kwargs["multi_ctas_kv_counter_buffer"] = cls.counter(query.device)
            flashinfer_decode.get_device_sm_count = boosted_sm_count

            try:
                return vllm_trtllm_batch_decode_with_kv_cache(*args, **kwargs)
            finally:
                flashinfer_decode.get_device_sm_count = flashinfer_get_device_sm_count

        vllm_flashinfer.trtllm_batch_decode_with_kv_cache = trtllm_batch_decode_with_kv_cache

    @classmethod
    def counter(cls, device: torch.device) -> torch.Tensor:
        buffer = cls.counters.get(device)
        if buffer is None:
            buffer = cls.counters[device] = torch.zeros(cls.COUNTER_BYTES, dtype=torch.uint8, device=device)

        return buffer


class AttentionStep:
    """The attention arguments of one step, shared by its layers."""

    SORT_MIN_ROWS = 48  # decode rows from which the longest contexts launch first

    def __init__(self, md):
        self.num_decode_tokens = md.num_decode_tokens
        self.num_tokens = md.num_actual_tokens
        self.num_prefills = md.num_prefills
        workspace = vllm_flashinfer._get_trtllm_workspace_buffer()
        self.workspace = workspace
        self.workspace_bytes = workspace.numel() * workspace.element_size()

        if md.prefill is not None:
            prefill = md.prefill
            self.prefill_args = (
                prefill.block_tables,
                prefill.seq_lens,
                prefill.max_q_len,
                prefill.max_seq_len,
                prefill.cum_seq_lens_q,
                prefill.cum_seq_lens_kv,
            )

        self.order = None
        if md.decode is not None:
            decode = md.decode
            if self.num_decode_tokens >= self.SORT_MIN_ROWS:
                # the decode rows longest context first: the kernel splits every sequence into the same number of
                # CTAs, so the long contexts' CTAs set the tail
                self.order = torch.argsort(decode.seq_lens, descending=True)
                self.decode_args = (
                    decode.block_tables.index_select(0, self.order),
                    decode.seq_lens.index_select(0, self.order),
                    decode.max_seq_len,
                )
            else:
                self.decode_args = (decode.block_tables, decode.seq_lens, decode.max_seq_len)


class BlackwellFlashInferImpl(FlashInferImpl):
    """vLLM's FlashInfer attention with the TRT-LLM kernels launched directly: the cache views per layer, the step's
    arguments per step."""

    # vLLM's FlashInferImpl.forward spends host time per layer in a mixed step (permuting and splitting the KV cache
    # view, contiguity asserts, FlashInfer's wrappers with their checks and per-call allocations) for two kernel
    # launches
    module = None  # FlashInfer's TRT-LLM attention module
    sm_count = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.views = None  # (the KV cache's address, its key and value views for the attention kernels)
        self.update_views = None  # (the KV cache's address, its key and value views for the cache update)

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None
    ):
        if not self.eligible(attn_metadata, kv_cache, output_scale):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )

        return self.attend(query, kv_cache, attn_metadata, output)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # the layer's cache views, split once (two view ops per call otherwise)
        if self.kv_sharing_target_layer_name is not None or self.is_kvcache_nvfp4:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        views = self.update_views
        if views is None or views[0] != kv_cache.data_ptr():
            k, v = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            views = self.update_views = (kv_cache.data_ptr(), k, v)

        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, views[1], views[2], slot_mapping, self.cache_dtype, layer._k_scale, layer._v_scale
        )

    def eligible(self, md, kv_cache, output_scale) -> bool:
        """A bf16 KV cache without cascade, decode context parallelism, sinks, output quantization or sliding window."""
        if md is None or output_scale is not None or md.use_cascade:
            return False
        if kv_cache.dtype != torch.bfloat16 or self.dcp_world_size > 1 or self.sinks is not None:
            return False
        if self.bmm1_scale is None or self.o_sf_scale is not None or self.window_left != -1:
            return False

        if md.prefill is not None and not isinstance(md.prefill, TRTLLMPrefill):
            return False
        if md.decode is not None:
            decode = md.decode
            if not isinstance(decode, FlashInferTrtllmAPIDecode) or decode.kernel != FlashInferDecodeKernel.TRTLLM_GEN:
                return False
            if decode.q_len_per_req != 1 or decode.mask is not None or md.num_decode_tokens != md.num_decodes:
                return False

        return True

    def attend(self, query, kv_cache, md, output):
        """The prefill rows' context kernel, then the decode rows' kernel at the boosted split."""
        if BlackwellFlashInferImpl.module is None:
            from flashinfer.decode import get_trtllm_gen_fmha_module
            from flashinfer.utils import get_device_sm_count

            BlackwellFlashInferImpl.module = get_trtllm_gen_fmha_module()
            BlackwellFlashInferImpl.sm_count = get_device_sm_count(torch.device("cuda", torch.cuda.current_device()))
        module, sm_count = BlackwellFlashInferImpl.module, BlackwellFlashInferImpl.sm_count

        step = getattr(md, "_pear_attention_step", None)
        if step is None:
            step = md._pear_attention_step = AttentionStep(md)

        views = self.views
        if views is None or views[0] != kv_cache.data_ptr():
            permuted_cache = canonicalize_singleton_dim_strides(
                kv_cache.permute(*FlashInferBackend.get_kv_cache_stride_order())
            )
            k, v = permuted_cache.split(self.head_size, dim=-1)
            views = self.views = (kv_cache.data_ptr(), k, v)
        _, k_cache, v_cache = views

        num_decode_tokens, num_tokens = step.num_decode_tokens, step.num_tokens
        if not query.is_contiguous():
            query = query.contiguous()
        counter = DecodeSplit.counter(query.device)

        # the launchers' positional arguments as FlashInfer's own wrappers pass them for this layout (no sinks,
        # no sliding window, no output scales)
        if num_tokens > num_decode_tokens:
            tables, lens, max_q, max_kv, cum_q, cum_kv = step.prefill_args
            module.trtllm_paged_attention_context(
                output[num_decode_tokens:num_tokens],
                None,
                query[num_decode_tokens:num_tokens],
                k_cache,
                v_cache,
                step.workspace,
                counter,
                tables,
                lens,
                max_q,
                max_kv,
                self.bmm1_scale,
                self.bmm2_scale,
                -1.0,
                -1,
                0,
                step.num_prefills,
                -1,
                cum_q,
                cum_kv,
                sm_count,
                True,
                step.workspace_bytes,
                None,
                None,
                None,
                None,
                True,
                True,
                None,
                0,
                0,
            )

        if num_decode_tokens > 0:
            tables, lens, max_seq = step.decode_args
            if step.order is not None:
                decode_queries = query[:num_decode_tokens].index_select(0, step.order)
                decode_output = torch.empty_like(output[:num_decode_tokens])
            else:
                decode_queries, decode_output = query[:num_decode_tokens], output[:num_decode_tokens]
            module.trtllm_paged_attention_decode(
                decode_output,
                None,
                decode_queries,
                k_cache,
                v_cache,
                step.workspace,
                counter,
                tables,
                lens,
                1,
                max_seq,
                self.bmm1_scale,
                self.bmm2_scale,
                -1.0,
                -1,
                0,
                num_decode_tokens,
                -1,
                0,
                sm_count * SPLIT_FACTOR,
                True,
                step.workspace_bytes,
                None,
                None,
                None,
                None,
                None,
                True,
                None,
                0,
                0,
                False,
            )
            if step.order is not None:
                output[:num_decode_tokens].index_copy_(0, step.order, decode_output)

        return output


class BlackwellFlashInferBackend(FlashInferBackend):
    """vLLM's FlashInfer attention backend with BlackwellFlashInferImpl (registered in place of FLASHINFER)."""

    @staticmethod
    def get_impl_cls() -> type[FlashInferImpl]:
        return BlackwellFlashInferImpl
