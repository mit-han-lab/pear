import numpy as np
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


class RolloutGDNAttention(QwenGatedDeltaNetAttention):
    """vLLM's Qwen Gated DeltaNet layer whose core a GPU backend computes from its packed projections."""

    def get_prefill_checkpoint_alignment(self) -> int | None:
        # vLLM's layer has FlashInfer's prefill write the states at a prefill's inner block boundaries and the scheduler
        # stop splitting prefills there; the backends' prefill kernels write a sequence's final state only, so the
        # scheduler splits every prefill at the block boundaries, whose states the prefix cache serves
        return None

    def computes_core(self, hidden_states: torch.Tensor) -> bool:
        """Whether the backend's core computes the layer: vLLM's fused decode kernel selected ("cuda"), the activations
        bf16, the gated norm's weight bf16 or fp32."""
        return (
            self.gdn_decode_kernel == "cuda"
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )

    def fused_core(self, mixed_qkvz, ba, core_attn_out):
        """vLLM's fused core op on the packed projections (it calls _forward_core_fused_norm_packed), then the output
        projection."""
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz, ba, core_attn_out, layer_name=_encode_layer_name(self.prefix)
        )
        output, _ = self.out_proj(core_attn_out.flatten(-2))

        return output


class GDNProjection:
    """One GEMM for a Gated DeltaNet layer's in_proj_qkvz and in_proj_ba, over one weight both are views of."""

    # in_proj_ba costs a launch per layer at any batch size. The views are made as the layer is built, so the loaders
    # write into the one tensor; every consumer takes the output's row strides
    @staticmethod
    def merge(layer) -> None:
        """Make the two projections' weights views of one [qkvz; ba] tensor, before the weights load."""
        qkvz, ba = layer.in_proj_qkvz, layer.in_proj_ba
        if (
            getattr(qkvz, "bias", None) is not None
            or getattr(ba, "bias", None) is not None
            or not isinstance(qkvz.quant_method, UnquantizedLinearMethod)
            or not isinstance(ba.quant_method, UnquantizedLinearMethod)
            or layer.disable_tp_for_ba_proj
            or layer.gqa_interleaved_layout
        ):
            return

        w1, w2 = qkvz.weight, ba.weight
        if w1.dtype != w2.dtype or w1.shape[1] != w2.shape[1] or not w1.is_contiguous() or not w2.is_contiguous():
            return

        n = w1.shape[0]
        qkvzba = torch.empty((n + w2.shape[0], w1.shape[1]), dtype=w1.dtype, device=w1.device)
        w1.data = qkvzba[:n]
        w2.data = qkvzba[n:]
        layer.register_buffer("qkvzba_weight", qkvzba, persistent=False)
        layer.qkvz_out = n

    @staticmethod
    def eligible(layer, hidden_states) -> bool:
        """A merged weight, and activations the layer's core computes."""
        return getattr(layer, "qkvzba_weight", None) is not None and layer.computes_core(hidden_states)

    @staticmethod
    def check(model) -> None:
        """Raise unless every merged layer's projections still view its merged weight (a loader replaced one)."""
        for module in model.modules():
            qkvzba = getattr(module, "qkvzba_weight", None)
            if qkvzba is None:
                continue
            qkvz, ba = module.in_proj_qkvz.weight, module.in_proj_ba.weight
            if qkvz.data_ptr() != qkvzba.data_ptr() or ba.data_ptr() != qkvzba[module.qkvz_out :].data_ptr():
                raise RuntimeError(f"{module.prefix}: in_proj_qkvz / in_proj_ba no longer view the merged weight")


class RolloutGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    """vLLM's GDN metadata builder whose other KV cache groups derive their metadata from the first group's, on the
    state slots vLLM's model state computes for every group at once."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # vLLM opts in its own class only. The GDN groups differ only in their state slots: the runner builds the first
        # group's metadata and derives the others' (update_block_table re-gathers a speculative step's slots too)
        self.supports_update_block_table = True
        # under mamba_cache_mode align, this group's state slots of the step, which vLLM's model state computes for
        # every group in one launch for the builders that declare them
        self.mamba_aligned_state_indices = None

    def state_slots(self, block_table: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """The requests' state slots in this group: under align mode the block of their last token (vLLM's precomputed
        indices), else their one block."""
        slots = self.mamba_aligned_state_indices if self.mamba_aligned_state_indices is not None else block_table

        return slots[:num_reqs, 0]

    @staticmethod
    def num_decodes(m) -> int:
        """The batch's leading decode rows as vLLM's GDN builder counts them: one token over a prior state; the rows
        from the first other one on are prefill rows."""
        num_reqs = m.num_reqs
        query_lens = np.diff(m.query_start_loc_cpu.numpy()[: num_reqs + 1])
        # a prompt's first chunk of one token has no prior state: vLLM hands a request a state slot as it was left
        is_first_chunk = (
            (query_lens > 0)
            & (m.seq_lens_cpu_upper_bound.numpy()[:num_reqs] <= query_lens)
            & m.is_prefilling.numpy()[:num_reqs]
        )
        is_prefill = (query_lens > 1) | is_first_chunk

        return int(np.argmax(is_prefill)) if is_prefill.any() else num_reqs
