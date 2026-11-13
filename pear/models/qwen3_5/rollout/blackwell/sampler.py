import torch
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from pear.models.qwen3_5.rollout.sampler import RolloutSampler


class BlackwellSampler(RolloutSampler):
    """vLLM's sampling of random rows with processed top-K log-probs as three fused kernels (csrc/sampler.cu)."""

    # vLLM makes several passes over the fp32 logits [rows, vocab], each with one CTA per row; the kernels read the
    # logits twice and the noise once, over ROW_CTAS CTAs per row
    ROW_CTAS = 32  # CTAs per row

    @classmethod
    def eligible(cls, sampler, logits, md, predict_bonus_token, mode) -> bool:
        """The shared conditions, no greedy row or bonus token, fp32 temperatures and rows of whole float4."""
        return (
            super().eligible(sampler, logits, md, predict_bonus_token, mode)
            and not md.all_greedy
            and not predict_bonus_token
            and md.temperature.dtype == torch.float32
            and logits.shape[1] % 4 == 0  # rows read as float4
        )

    @classmethod
    def draw(cls, logits, md) -> SamplerOutput:
        logits = logits.contiguous()

        # the noise drawn as vLLM draws it, so the RNG stream advances identically; the race
        # argmax_j softmax(x)_j / q_j is taken as argmax_j x_j - log q_j
        q = torch.empty_like(logits)
        q.exponential_()
        sampled_ids, ids, logprobs, ranks = torch.ops.pear_rollout_blackwell.fused_sample(
            logits, q, md.temperature.contiguous(), md.max_num_logprobs, cls.ROW_CTAS
        )

        return SamplerOutput(
            sampled_token_ids=sampled_ids.unsqueeze(-1), logprobs_tensors=LogprobsTensors(ids, logprobs, ranks)
        )
