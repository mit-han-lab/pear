import weakref

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

from pear.models.qwen3_5.rollout.blackwell.flashinfer_launchers import TrtllmContextArguments, TrtllmDecodeArguments

# the factor on the SM count FlashInfer's trtllm-gen decode launcher sizes its split-KV from (SMs / (batch x KV head
# groups) CTAs per sequence): a sequence over more CTAs, merged by the kernel's own multi-CTA reduction, so that a
# step waits less for its longest context
SPLIT_FACTOR = 8


class DecodeSplit:
    """FlashInfer's TRT-LLM decode attention split over SPLIT_FACTOR times more CTAs per sequence in vLLM's own
    FlashInfer path (BlackwellFlashInferImpl's other steps), through FlashInfer's SM count."""

    @staticmethod
    def install() -> None:
        """Patch vLLM's call of FlashInfer's TRT-LLM decode."""
        # FlashInfer sizes the split from its own SM count, which neither it nor vLLM lets a caller set; vLLM's
        # multi-CTA counter buffer, sized for its batch-token budget, covers the boosted count
        import flashinfer.decode as flashinfer_decode

        vllm_trtllm_batch_decode_with_kv_cache = vllm_flashinfer.trtllm_batch_decode_with_kv_cache
        flashinfer_get_device_sm_count = flashinfer_decode.get_device_sm_count

        def boosted_sm_count(device):
            return flashinfer_get_device_sm_count(device) * SPLIT_FACTOR

        def trtllm_batch_decode_with_kv_cache(*args, **kwargs):
            flashinfer_decode.get_device_sm_count = boosted_sm_count

            try:
                return vllm_trtllm_batch_decode_with_kv_cache(*args, **kwargs)
            finally:
                flashinfer_decode.get_device_sm_count = flashinfer_get_device_sm_count

        vllm_flashinfer.trtllm_batch_decode_with_kv_cache = trtllm_batch_decode_with_kv_cache


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


class KVCacheViews:
    """A layer's views of its KV cache, made at the first call with that cache and dropped when vLLM releases it."""

    def __init__(self):
        self.cache_ref = None  # a weak reference to the KV cache they view
        self.views = None

    def of(self, kv_cache, make):
        """kv_cache's views, make(kv_cache) at the first call with it."""
        if self.cache_ref is None or self.cache_ref() is not kv_cache:
            views = make(kv_cache)
            # the views hold the cache's memory, so they go with the cache. The reference is replaced first: dropping
            # the old views can free the old cache, whose reference would then drop the new views
            self.cache_ref = weakref.ref(kv_cache, self.release)
            self.views = views

        return self.views

    def release(self, cache_ref):
        """Drop the views of the KV cache vLLM released (the weak reference's callback)."""
        self.cache_ref = self.views = None


class BlackwellFlashInferImpl(FlashInferImpl):
    """vLLM's FlashInfer attention with the TRT-LLM kernels launched directly: the cache views per layer, the step's
    arguments per step."""

    # vLLM's FlashInferImpl.forward spends host time per layer in a mixed step (permuting and splitting the KV cache
    # view, contiguity asserts, FlashInfer's wrappers with their checks and per-call allocations) for two kernel
    # launches
    module = None  # FlashInfer's TRT-LLM attention module
    sm_count = None
    # the launchers' arguments this attention never varies, as FlashInfer's wrappers pass them for its layout: a bf16
    # cache in vLLM's page table, causal, without output scales, windows, sinks, block scales or log-sum-exp
    FIXED_ARGUMENTS = {
        "out_scale_factor": None,
        "o_sf_scale": -1.0,
        "o_sf_vec_size": -1,
        "o_sf_start_index": 0,
        "window_left": -1,
        "enable_pdl": True,
        "attention_sinks": None,
        "key_block_scales": None,
        "value_block_scales": None,
        "skip_softmax_threshold_scale_factor": None,
        "uses_shared_paged_kv_idx": True,
        "use_fp16_softmax": None,
        "lse": None,
        "lse_scale": 1.0,
        "lse_stride_tokens": 0,
        "lse_stride_heads": 0,
    }
    FIXED_CONTEXT_ARGUMENTS = {
        **FIXED_ARGUMENTS,
        "window_right": -1,
        "variable_window_token_starts": None,
        "variable_window_token_ends": None,
        "uses_spcompress": None,
        "is_causal": True,
    }
    # the decode rows: one query token each, dense attention
    FIXED_DECODE_ARGUMENTS = {
        **FIXED_ARGUMENTS,
        "max_q_len": 1,
        "sparse_mla_top_k": 0,
        "cum_seq_lens_q": None,
        "enable_block_sparse_attention": False,
        "sparse_mla_top_k_lens": None,
        "bf16q_fp8kv_transform_mode": 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.views = KVCacheViews()  # the KV cache's key and value views for the attention kernels
        self.update_views = KVCacheViews()  # its key and value views for the cache update

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None
    ):
        if not self.eligible(attn_metadata, kv_cache, output_scale):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )

        return self.attend(query, kv_cache, attn_metadata, output)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.kv_sharing_target_layer_name is not None or self.is_kvcache_nvfp4:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        # the layer's cache views, split once (two view ops per call otherwise)
        k, v = self.update_views.of(kv_cache, lambda cache: cache.transpose(1, 2).split(self.head_size, dim=-1))
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, k, v, slot_mapping, self.cache_dtype, layer._k_scale, layer._v_scale
        )

    def eligible(self, md, kv_cache, output_scale) -> bool:
        """A bf16 KV cache without decode context parallelism, sinks, output quantization or sliding window, its decode
        rows of one query token each."""
        if md is None or output_scale is not None:
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
            if decode.q_len_per_req != 1 or decode.q_cu_seq_lens is not None or decode.mask is not None:
                return False
            if md.num_decode_tokens != md.num_decodes:
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

        # the cache as the kernels read it (HND), the layout vLLM resolved for the KV cache group
        k_cache, v_cache = self.views.of(
            kv_cache,
            lambda cache: canonicalize_singleton_dim_strides(cache.permute(*md.kv_cache_layout.layer_view_order)).split(
                self.head_size, dim=-1
            ),
        )

        num_decode_tokens, num_tokens = step.num_decode_tokens, step.num_tokens
        if not query.is_contiguous():
            query = query.contiguous()
        # vLLM's multi-CTA semaphores, sized for its batch-token budget (each kernel resets the ones it uses)
        counter = self._multi_ctas_kv_counter_buffer

        if num_tokens > num_decode_tokens:
            tables, lens, max_q, max_kv, cum_q, cum_kv = step.prefill_args
            context_arguments = TrtllmContextArguments(
                **self.FIXED_CONTEXT_ARGUMENTS,
                out=output[num_decode_tokens:num_tokens],
                query=query[num_decode_tokens:num_tokens],
                key_cache=k_cache,
                value_cache=v_cache,
                workspace_buffer=step.workspace,
                multi_ctas_kv_counter_buffer=counter,
                block_tables=tables,
                seq_lens=lens,
                max_q_len=max_q,
                max_kv_len=max_kv,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                batch_size=step.num_prefills,
                cum_seq_lens_q=cum_q,
                cum_seq_lens_kv=cum_kv,
                sm_count=sm_count,
                workspace_size=step.workspace_bytes,
            )
            module.trtllm_paged_attention_context(*context_arguments)

        if num_decode_tokens > 0:
            tables, lens, max_seq = step.decode_args
            if step.order is not None:
                decode_queries = query[:num_decode_tokens].index_select(0, step.order)
                decode_output = torch.empty_like(output[:num_decode_tokens])
            else:
                decode_queries, decode_output = query[:num_decode_tokens], output[:num_decode_tokens]
            decode_arguments = TrtllmDecodeArguments(
                **self.FIXED_DECODE_ARGUMENTS,
                out=decode_output,
                query=decode_queries,
                key_cache=k_cache,
                value_cache=v_cache,
                workspace_buffer=step.workspace,
                multi_ctas_kv_counter_buffer=counter,
                block_tables=tables,
                seq_lens=lens,
                max_kv_len=max_seq,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                batch_size=num_decode_tokens,
                sm_count=sm_count * SPLIT_FACTOR,
                workspace_size=step.workspace_bytes,
            )
            module.trtllm_paged_attention_decode(*decode_arguments)
            if step.order is not None:
                output[:num_decode_tokens].index_copy_(0, step.order, decode_output)

        return output


class BlackwellFlashInferBackend(FlashInferBackend):
    """vLLM's FlashInfer attention backend with BlackwellFlashInferImpl (registered in place of FLASHINFER)."""

    @staticmethod
    def get_impl_cls() -> type[FlashInferImpl]:
        return BlackwellFlashInferImpl
