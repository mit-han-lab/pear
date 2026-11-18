from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig


class Qwen3_5MTPSpeculatorConfig(MTPSpeculatorConfig):
    """vLLM's MTP speculator config whose speculator drafts on the target's M-RoPE positions, each draft token the
    argmax over the ranks' vocabulary shards of logits in the target's lm_head dtype."""

    speculator_class = "pear.models.qwen3_5.rollout.speculator.Qwen3_5MTPSpeculator"
    # the weight sync refreshes no quantized copy of the target's lm_head (vLLM does in reset_lora_state alone)
    supported_options = MTPSpeculatorConfig.supported_options - {"draft_lm_head_quantization"}
    # the speculator's draft forward takes no prefill-context-parallel inputs
    supports_pcp = False

    @classmethod
    def adapt_draft_model_config(cls, spec: SpeculativeConfig) -> None:
        super().adapt_draft_model_config(spec)
        # the target's lm_head dtype (PEAR's hf_overrides), which vLLM hands no draft: dict hf_overrides are the
        # target's alone
        spec.draft_model_config.hf_config.head_dtype = getattr(spec.target_model_config.hf_config, "head_dtype", None)

    @classmethod
    def apply_options(cls, spec: SpeculativeConfig) -> None:
        super().apply_options(spec)
        # a greedy draft from each rank's argmax over its vocabulary shard: two values a row gathered, not the logits
        spec.use_local_argmax_reduction = True
