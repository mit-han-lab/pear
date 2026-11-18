import numpy as np
import torch
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


class RolloutGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    """vLLM's GDN metadata builder whose other KV cache groups derive their metadata from the first group's, on the
    state slots vLLM's model state computes for every group at once."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # vLLM opts in its own class only. The GDN groups differ only in their state slots: the runner builds the first
        # group's metadata and derives the others' (update_block_table)
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
