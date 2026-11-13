from abc import ABC, abstractmethod

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample import sampler as vllm_sampler
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor


class RolloutSampler(ABC):
    """vLLM's sampler with a GPU backend's own kernels for random rows with processed top-K log-probs."""

    MAX_LOGPROBS = 32  # the largest top-K the kernels keep (kTopK and K of the two backends' csrc/sampler.cu)

    @classmethod
    def install(cls) -> None:
        """Patch vLLM's sampler."""
        vllm_forward = vllm_sampler.Sampler.forward

        def forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
            md = sampling_metadata
            mode = logprobs_mode_override or self.logprobs_mode
            if not cls.eligible(self, logits, md, predict_bonus_token, mode):
                return vllm_forward(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)

            return cls.draw(self.apply_logits_processors(logits, md, predict_bonus_token), md)

        vllm_sampler.Sampler.forward = forward

    @classmethod
    def eligible(cls, sampler, logits, md, predict_bonus_token, mode) -> bool:
        """Random rows with a temperature only, a top-K of processed log-probs, fp32 logits (vLLM's path otherwise)."""
        num_logprobs = md.max_num_logprobs

        return (
            mode == "processed_logprobs"
            and md.all_random
            and md.top_k is None
            and md.top_p is None
            and not md.generators
            and num_logprobs is not None
            and 1 <= num_logprobs <= cls.MAX_LOGPROBS
            and not md.logprob_token_ids
            and not sampler.use_fp64_gumbel
            and md.temperature is not None
            and logits.dtype == torch.float32
            and logits.dim() == 2
            and all(isinstance(p, MinPLogitsProcessor) and not p.min_p_count for p in md.logitsprocs.argmax_invariant)
        )

    @classmethod
    @abstractmethod
    def draw(cls, logits, md) -> SamplerOutput:
        """One token per row of the processed logits, with its top-K log-probs."""
