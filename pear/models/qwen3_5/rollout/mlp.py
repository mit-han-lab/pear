from flashinfer.activation import silu_and_mul
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP


class Qwen3_5RolloutMLP(Qwen3NextMLP):
    """vLLM's dense MLP whose silu(gate) * up is computed in fp32 and rounded to bf16 once."""

    def forward(self, x):
        # FlashInfer's kernel: one store of the fp32 product (vLLM's own kernel rounds silu(gate) to bf16 first)
        gate_up, _ = self.gate_up_proj(x)
        output, _ = self.down_proj(silu_and_mul(gate_up))

        return output
