from typing import NamedTuple

import torch
from cuda.bindings import driver


class ChunkKernelKey(NamedTuple):
    """The key FlashInfer caches its compiled Blackwell chunked GDN kernel under: the parameters of
    flashinfer.gdn_kernels.blackwell.gdn_prefill._get_compiled_cache, in their order."""

    NO_TENSOR = "none"  # FlashInfer's dtype string of a tensor the call does not give

    kernel_variant: str  # the kernel class for the device's architecture
    target_key: tuple  # the device's compile target
    num_sm: int  # the SMs the persistent kernel's grid covers
    io_dtype_str: str  # of q, k, v and the output
    state_dtype_str: str  # of the recurrent states
    HQ: int  # q heads
    HV: int  # v heads
    is_GQA: bool  # whether there are no fewer q heads than v heads
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool  # whether the kernel writes states inside a sequence
    use_state_indices: bool  # whether the states are rows of a pool, picked by index
    cu_seqlens_dtype_str: str
    state_indices_dtype_str: str
    cu_checkpoints_dtype_str: str
    initial_state_inner_strides: tuple[int, ...] | None  # a pool's strides within a row
    output_state_inner_strides: tuple[int, ...] | None
    state_stride_divisibility: int  # the alignment of a pool's strides

    @classmethod
    def of_packed_call(cls, q, v, initial_state, cu_seqlens) -> "ChunkKernelKey":
        """The key of a call on packed initial and final states without checkpoints, from the helpers FlashInfer's
        entry point builds it with."""
        from flashinfer.gdn_kernels.blackwell import gdn_prefill

        target = gdn_prefill.gdn_device_target(q.device)
        heads, v_heads = q.size(1), v.size(1)

        return cls(
            kernel_variant=gdn_prefill._prefill_kernel_variant(target.major, target.minor),
            target_key=target.compile_key,
            num_sm=gdn_prefill.get_num_sm(q.device),
            io_dtype_str=str(q.dtype),
            state_dtype_str=str(initial_state.dtype),
            HQ=heads,
            HV=v_heads,
            is_GQA=heads >= v_heads,
            use_initial_state=True,
            store_final_state=True,
            enable_checkpoints=False,
            use_state_indices=False,
            cu_seqlens_dtype_str=str(cu_seqlens.dtype),
            state_indices_dtype_str=cls.NO_TENSOR,
            cu_checkpoints_dtype_str=cls.NO_TENSOR,
            initial_state_inner_strides=None,
            output_state_inner_strides=None,
            state_stride_divisibility=1,  # packed states: no pool whose strides to align
        )


class ChunkKernelArguments(NamedTuple):
    """The arguments of FlashInfer's compiled Blackwell chunked GDN kernel: the parameters of
    GatedDeltaNetChunkedKernel.__call__, which FlashInfer compiles, in their order."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor  # the decay, exponentiated
    beta: torch.Tensor
    o: torch.Tensor
    cu_seqlens: torch.Tensor
    s_in: torch.Tensor | None  # the initial states
    s_out: torch.Tensor | None  # the final states
    s_indices: torch.Tensor | None  # the states' rows in a pool
    s_checkpoints: torch.Tensor | None
    cu_checkpoints: torch.Tensor | None
    s_out_indices: torch.Tensor | None
    checkpoint_indices: torch.Tensor | None
    prefix_lens: torch.Tensor | None
    checkpoint_every_n_tokens: int
    scale: float
    tensormap_workspace: torch.Tensor
    stream: driver.CUstream


class FlashInferChunk:
    """FlashInfer's Blackwell chunked GDN kernel on packed states, launched through its compiled function."""

    # FlashInfer's entry points build the compile key, look it up and allocate a workspace at every call: host time in
    # every layer of a mixed step, while the GPU waits
    def __init__(self):
        self.compiled_kernel = None
        self.workspace = None

    def __call__(self, q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale) -> None:
        """One call of the chunked kernel over the prefill rows."""
        if self.compiled_kernel is None:
            # the first call goes through FlashInfer's entry point, which compiles
            from flashinfer.gdn_kernels.blackwell import gdn_prefill

            gdn_prefill.chunk_gated_delta_rule_sm100(q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale)

            key = ChunkKernelKey.of_packed_call(q, v, s_in, cu_seqlens)
            self.compiled_kernel = gdn_prefill._get_compiled_cache(*key)["compiled"]
            # the persistent kernel's TMA-descriptor workspace is sized per SM, whatever the batch (B); the kernel
            # rewrites it at each launch, all on the one stream the prefill rows run on
            workspace_size = gdn_prefill.GatedDeltaNetChunkedKernel.get_workspace_size(
                num_sm=key.num_sm, B=1, HQ=key.HQ, HV=key.HV, is_persistent=True
            )
            self.workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
            return

        arguments = ChunkKernelArguments(
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            o=out,
            cu_seqlens=cu_seqlens,
            s_in=s_in,
            s_out=s_out,
            s_indices=None,
            s_checkpoints=None,
            cu_checkpoints=None,
            s_out_indices=None,
            checkpoint_indices=None,
            prefix_lens=None,
            checkpoint_every_n_tokens=0,
            scale=scale,
            tensormap_workspace=self.workspace,
            stream=driver.CUstream(torch._C._cuda_getCurrentRawStream(q.device.index)),
        )
        self.compiled_kernel(*arguments)


class TrtllmContextArguments(NamedTuple):
    """The arguments of FlashInfer's TRT-LLM paged-attention context launcher (trtllm_paged_attention_context in
    csrc/trtllm_fmha_kernel_launcher.cu), in its order."""

    out: torch.Tensor
    out_scale_factor: torch.Tensor | None
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    workspace_buffer: torch.Tensor
    multi_ctas_kv_counter_buffer: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_q_len: int
    max_kv_len: int
    bmm1_scale: float
    bmm2_scale: float
    o_sf_scale: float
    o_sf_vec_size: int
    o_sf_start_index: int
    batch_size: int
    window_left: int
    window_right: int
    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor
    variable_window_token_starts: torch.Tensor | None
    variable_window_token_ends: torch.Tensor | None
    sm_count: int
    enable_pdl: bool
    workspace_size: int
    attention_sinks: torch.Tensor | None
    key_block_scales: torch.Tensor | None
    value_block_scales: torch.Tensor | None
    skip_softmax_threshold_scale_factor: float | None
    uses_shared_paged_kv_idx: bool | None
    use_fp16_softmax: bool | None
    uses_spcompress: bool | None
    is_causal: bool
    lse: torch.Tensor | None
    lse_scale: float
    lse_stride_tokens: int
    lse_stride_heads: int


class TrtllmDecodeArguments(NamedTuple):
    """The arguments of FlashInfer's TRT-LLM paged-attention decode launcher (trtllm_paged_attention_decode in
    csrc/trtllm_fmha_kernel_launcher.cu), in its order."""

    out: torch.Tensor
    out_scale_factor: torch.Tensor | None
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    workspace_buffer: torch.Tensor
    multi_ctas_kv_counter_buffer: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_q_len: int
    max_kv_len: int
    bmm1_scale: float
    bmm2_scale: float
    o_sf_scale: float
    o_sf_vec_size: int
    o_sf_start_index: int
    batch_size: int
    window_left: int
    sparse_mla_top_k: int
    sm_count: int
    enable_pdl: bool
    workspace_size: int
    attention_sinks: torch.Tensor | None
    cum_seq_lens_q: torch.Tensor | None
    key_block_scales: torch.Tensor | None
    value_block_scales: torch.Tensor | None
    skip_softmax_threshold_scale_factor: float | None
    uses_shared_paged_kv_idx: bool | None
    lse: torch.Tensor | None
    lse_scale: float
    lse_stride_tokens: int
    lse_stride_heads: int
    enable_block_sparse_attention: bool
    sparse_mla_top_k_lens: torch.Tensor | None
    bf16q_fp8kv_transform_mode: int
    use_fp16_softmax: bool | None
