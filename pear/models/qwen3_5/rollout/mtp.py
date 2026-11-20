from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.models.qwen3_5 import Qwen3_5MTP

from pear.models.qwen3_5.rollout.attention import Qwen3_5RolloutAttention
from pear.models.qwen3_5.rollout.mlp import Qwen3_5RolloutMLP
from pear.models.qwen3_5.rollout.rope import Fp32Rope


class Qwen3_5RolloutMTP(Qwen3_5MTP):
    """vLLM's Qwen3.5 MTP drafter whose layers compute as the rollout model's: its attention and MLP, the rotary
    embedding in fp32."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # vLLM builds the MTP layers' attention and MLP from its own classes: they are rebased on the rollout's, as
        # vLLM's Transformers backend rebases its embeddings, the attention's output gate fused as its constructor sets
        for layer in self.model.layers:
            assert type(layer.self_attn) is Qwen3NextAttention and type(layer.mlp) is Qwen3NextMLP
            layer.self_attn.__class__ = Qwen3_5RolloutAttention
            layer.self_attn.use_fused_output_gate = True
            layer.mlp.__class__ = Qwen3_5RolloutMLP

        # the fp32 cos / sin cache the attention's q / k kernel rotates with
        Fp32Rope.install(self)
