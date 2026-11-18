from pear.models.qwen3_5.rollout.blackwell.sampler import BlackwellSampler
from pear.models.qwen3_5.rollout.model_state import Qwen3_5RolloutModelState


class BlackwellRolloutModelState(Qwen3_5RolloutModelState):
    """The model state of a datacenter Blackwell replica: vLLM's hybrid model state with the fused sampler."""

    sampler_cls = BlackwellSampler
    # the first decoder layer sums the ranks' partial embedding rows, and a prompt embedding is a whole row on each
    # rank: vLLM refuses --enable-prompt-embeds for this model state
    supports_prompt_embeds = False
