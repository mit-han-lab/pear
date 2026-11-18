import torch
from vllm.v1.outputs import LogprobsTensors

from pear.models.qwen3_5.rollout.sampler import RolloutSampler


class BlackwellSampler(RolloutSampler):
    """vLLM's sampling of the rows at a temperature alone, with processed top-K log-probs, as three fused kernels
    (csrc/sampler.cu)."""

    # vLLM makes several passes over the fp32 logits [rows, vocab], each with one CTA per row; the kernels read the
    # logits twice and the noise once, over ROW_CTAS CTAs per row
    ROW_CTAS = 32  # CTAs per row
    ROW_VECTOR = 4  # floats per load: the kernels read a row as float4

    @classmethod
    def supports(cls, logits: torch.Tensor) -> bool:
        return super().supports(logits) and logits.shape[1] % cls.ROW_VECTOR == 0

    def draw(self, logits, temperature, num_logprobs):
        logits = logits.contiguous()

        # the race argmax_j softmax(x)_j / q_j over exponential noise of torch's generator, taken as
        # argmax_j x_j - log q_j; every tensor-parallel rank draws the same noise (the same seed, the same draws)
        q = torch.empty_like(logits)
        q.exponential_()
        sampled, ids, logprobs, ranks = torch.ops.pear_rollout_blackwell.fused_sample(
            logits, q, temperature, num_logprobs, self.ROW_CTAS
        )

        # vLLM's V2 runner takes the sampled tokens as int64
        return sampled.long(), LogprobsTensors(ids, logprobs, ranks)
