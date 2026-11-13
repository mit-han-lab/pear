from __future__ import annotations

import copy
import functools
import math
from typing import TYPE_CHECKING, Any

import torch
from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import calc_rows_per_block, layer_norm_fwd_kernel
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

if TYPE_CHECKING:  # annotations only
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata


class DeployGDNAttention(QwenGatedDeltaNetAttention):
    """vLLM's Qwen Gated DeltaNet layer whose prefill core runs vLLM's kernels with less host work (registered out of
    tree in its place)."""

    # the prefill core runs outside the piecewise CUDA graphs, once per layer and prefill step: the lean path runs the
    # same kernels in the same order on the same inputs as vLLM's, to bitwise equal outputs and states
    is_warmed_up = False  # FlashInfer's state_indices variant compiled in this process
    WARM_TOKENS = 64  # tokens of the warm-up call that compiles FlashInfer's state_indices variant of its kernel
    # vLLM's layer_norm_fwd tile configuration: one warp per 256 values of a group, at most 8
    NORM_VALUES_PER_WARP = 256
    NORM_MAX_WARPS = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # the lean path runs FlashInfer's core: with another prefill backend every call is vLLM's
        self.is_lean = self.gdn_prefill_backend == "flashinfer"
        assert self.norm.group_size in (None, self.head_v_dim), "the gated norm per head"

    def _forward_core_fused_norm_packed(
        self, mixed_qkvz: torch.Tensor, ba: torch.Tensor, core_attn_out: torch.Tensor
    ) -> None:
        if not self.is_lean:
            return super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:  # vLLM's warm-up and profile runs
            super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)
            if not DeployGDNAttention.is_warmed_up:
                self.warm(mixed_qkvz)
                DeployGDNAttention.is_warmed_up = True
            return

        attn_metadata = attn_metadata[self.prefix]
        if (
            attn_metadata.num_prefills == 0
            or attn_metadata.num_decodes != 0
            or attn_metadata.spec_sequence_masks is not None
            or not attn_metadata.pear_has_state
        ):
            return super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)

        # a, b and the gate z as row-strided views of the projections (vLLM's path makes contiguous copies)
        num_tokens = attn_metadata.num_actual_tokens
        qkv_size, conv_weight, norm_weight = self.prefill_weights
        mixed_qkv = mixed_qkvz[:num_tokens, :qkv_size]
        z = mixed_qkvz[:num_tokens, qkv_size:]
        b, a = ba[:num_tokens].chunk(2, dim=-1)
        conv_state, ssm_state = self.kv_cache[0].transpose(-1, -2), self.kv_cache[1]

        conv_output = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            conv_weight,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=attn_metadata.has_initial_state,
            cache_indices=attn_metadata.non_spec_state_indices_tensor,
            query_start_loc=attn_metadata.non_spec_query_start_loc,
            metadata=attn_metadata,
        ).transpose(0, 1)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_output,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )

        # FlashInfer reads each sequence's state from, and writes it to, the layer's state pool through the prefill
        # state indices (vLLM: gather -> zero -> kernel -> scatter of packed copies), its output in place
        core_output = core_attn_out[:num_tokens]
        chunk_gated_delta_rule_sm100(
            q,
            k,
            v,
            torch.exp(g),
            beta,
            core_output,
            attn_metadata.pear_cu_seqlens,
            ssm_state,
            ssm_state,
            1.0 / math.sqrt(self.head_k_dim),
            state_indices=attn_metadata.pear_state_indices,
        )

        # the gated RMS norm per head, in place: vLLM's kernel with vLLM's tile config (rows per program from its
        # [tokens * heads, dim] view), here over [tokens, heads * dim] rows in groups of dim with the row-strided z
        output_rows = core_output.view(num_tokens, -1)
        num_heads, head_dim = core_output.shape[1], core_output.shape[2]
        rows_per_block = calc_rows_per_block(num_tokens * num_heads, core_output.device)
        rstd = torch.empty(num_heads * num_tokens, dtype=torch.float32, device=core_output.device)
        block_n = triton.next_power_of_2(head_dim)
        layer_norm_fwd_kernel[(triton.cdiv(num_tokens, rows_per_block), num_heads)](
            output_rows,
            output_rows,
            norm_weight,
            self.norm.bias,
            z,
            None,
            rstd,
            output_rows.stride(0),
            output_rows.stride(0),
            z.stride(0),
            num_tokens,
            head_dim,
            self.norm.eps,
            BLOCK_N=block_n,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_BIAS=self.norm.bias is not None,
            HAS_Z=True,
            NORM_BEFORE_GATE=self.norm.norm_before_gate,
            IS_RMS_NORM=True,
            num_warps=min(max(block_n // self.NORM_VALUES_PER_WARP, 1), self.NORM_MAX_WARPS),
            ACTIVATION=self.norm.activation,
        )

    def warm(self, activations: torch.Tensor) -> None:
        """Compile FlashInfer's state_indices variant of its kernel (a CuTe DSL JIT per process) in vLLM's warm-up."""
        q = torch.zeros(
            self.WARM_TOKENS, self.num_k_heads, self.head_k_dim, dtype=activations.dtype, device=activations.device
        )
        v = torch.zeros(
            self.WARM_TOKENS, self.num_v_heads, self.head_v_dim, dtype=activations.dtype, device=activations.device
        )
        g = torch.ones(self.WARM_TOKENS, self.num_v_heads, dtype=torch.float32, device=activations.device)

        # a pool of two rows of the layer's state dtype, the warm-up's sequence on row 1: vLLM's profile run has not
        # allocated the layer's state pool yet
        pool = torch.zeros(
            2,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            dtype=self.get_state_dtype()[1],
            device=activations.device,
        )
        cu_seqlens = torch.tensor([0, self.WARM_TOKENS], dtype=torch.int32, device=activations.device)

        chunk_gated_delta_rule_sm100(
            q,
            torch.zeros_like(q),
            v,
            g,
            g.clone(),
            torch.empty_like(v),
            cu_seqlens,
            pool,
            pool,
            1.0 / math.sqrt(self.head_k_dim),
            state_indices=torch.ones(1, dtype=torch.int32, device=activations.device),
        )

    # cached: the served weights are never updated in place
    @functools.cached_property
    def prefill_weights(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """The qkv channels, the conv weight [channels, width] and the norm weight repeated per head."""
        conv_weight = self.conv1d.weight
        qkv_size = self.key_dim * 2 + self.value_dim

        return (
            qkv_size,
            conv_weight.view(conv_weight.size(0), conv_weight.size(2)),
            self.norm.weight.repeat(self.num_v_heads),
        )


class DeployGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    """vLLM's GDN metadata builder that also marks a prefill-only step whose sequences all continue a cached state,
    with FlashInfer's int32 sequence starts and state indices."""

    # a prefill-only step's Mamba groups build the same metadata but for their state indices: within one prepare_attn
    # (install's wrap: () while it runs, else None) the first group's build is kept as (key, metadata), and the later
    # groups copy it with their own indices
    step_build: tuple | None = None

    @classmethod
    def install(cls) -> None:
        """Arm the shared build for the duration of each hybrid model state's prepare_attn."""
        vllm_prepare_attn = MambaHybridModelState.prepare_attn

        def armed_prepare_attn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cls.step_build = ()
            try:
                return vllm_prepare_attn(self, *args, **kwargs)
            finally:
                cls.step_build = None

        MambaHybridModelState.prepare_attn = armed_prepare_attn

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, *args: Any, **kwargs: Any
    ) -> GDNAttentionMetadata:
        # the step's shared inputs (one CommonAttentionMetadata per group, the same tensors in each) and the spec
        key = (
            id(common_attn_metadata.query_start_loc_cpu),
            id(common_attn_metadata.seq_lens),
            common_attn_metadata.num_reqs,
            common_attn_metadata.num_actual_tokens,
            self.kv_cache_spec,
        )
        step_build = DeployGDNMetadataBuilder.step_build
        if step_build and step_build[0] == key:
            return self.with_indices(step_build[1], common_attn_metadata)

        attn_metadata = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)

        # the lean path takes only steps whose sequences all continue a cached state: vLLM zeroes the initial state of a
        # sequence that has none, while FlashInfer would read its pool row as it is
        attn_metadata.pear_has_state = False
        is_prefill_only = (
            attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.spec_sequence_masks is None
        )
        if is_prefill_only:
            # the V2 runner's CPU sequence lengths (the device's would cost a sync): an upper bound that is exact in a
            # prefill-only step, since it exceeds the true lengths only past rejected draft tokens
            query_start_loc = common_attn_metadata.query_start_loc_cpu[: common_attn_metadata.num_reqs + 1]
            num_computed_tokens = common_attn_metadata.seq_lens_cpu_upper_bound[: common_attn_metadata.num_reqs] - (
                query_start_loc[1:] - query_start_loc[:-1]
            )
            attn_metadata.pear_has_state = bool((num_computed_tokens > 0).all())
        if attn_metadata.pear_has_state:  # FlashInfer's int32 inputs, once per step for all layers
            attn_metadata.pear_cu_seqlens = attn_metadata.prefill_query_start_loc.to(torch.int32)
            attn_metadata.pear_state_indices = attn_metadata.prefill_state_indices.to(torch.int32).contiguous()

        if step_build is not None and is_prefill_only:
            DeployGDNMetadataBuilder.step_build = (key, attn_metadata)

        return attn_metadata

    def with_indices(
        self, prefill_metadata: GDNAttentionMetadata, common_attn_metadata: CommonAttentionMetadata
    ) -> GDNAttentionMetadata:
        """A copy of prefill_metadata (this step's prefill-only build of another group) with this group's state indices,
        as vLLM's build derives them: the first column of the mamba block table (align mode: the request's last
        blocks)."""
        attn_metadata = copy.copy(prefill_metadata)
        block_table = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        attn_metadata.non_spec_state_indices_tensor = block_table[:, 0]
        attn_metadata.prefill_state_indices = attn_metadata.non_spec_state_indices_tensor
        if attn_metadata.pear_has_state:
            attn_metadata.pear_state_indices = attn_metadata.prefill_state_indices.to(torch.int32).contiguous()

        return attn_metadata


class DeployGDNBackend(GDNAttentionBackend):
    """vLLM's GDN attention backend with DeployGDNMetadataBuilder (registered in place of GDN_ATTN)."""

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return DeployGDNMetadataBuilder
