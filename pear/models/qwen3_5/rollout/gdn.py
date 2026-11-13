import copy

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


class RolloutGDNAttention(QwenGatedDeltaNetAttention):
    """vLLM's Qwen Gated DeltaNet layer whose core a GPU backend computes from its packed projections."""

    def fused_core(self, mixed_qkvz, ba, core_attn_out):
        """vLLM's fused core op on the packed projections (it calls _forward_core_fused_norm_packed), then the output
        projection."""
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz, ba, core_attn_out, layer_name=_encode_layer_name(self.prefix)
        )
        output, _ = self.out_proj(core_attn_out.flatten(-2))

        return output


class RolloutGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    """vLLM's GDN metadata builder whose other KV cache groups derive their metadata from the first group's."""

    @staticmethod
    def group_common_metadata(common_attn_metadata, blk_table, slot_mapping):
        """A copy of the step's common metadata with a group's block table and slot mapping."""
        common_attn_metadata = copy.copy(common_attn_metadata)
        common_attn_metadata.block_table_tensor = blk_table
        common_attn_metadata.slot_mapping = slot_mapping

        return common_attn_metadata
