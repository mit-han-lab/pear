import torch
from vllm.v1.outputs import LogprobsTensors

from pear.models.qwen3_5.rollout.sampler import RolloutRejectionSampler, RolloutSampler


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
        sampled, ids, logprobs, ranks, nonfinite = torch.ops.pear_rollout_blackwell.fused_sample(
            logits, q, temperature, num_logprobs, self.ROW_CTAS
        )

        # vLLM's V2 runner takes the sampled tokens as int64
        return sampled.long(), LogprobsTensors(ids, logprobs, ranks), nonfinite


class BlackwellRejectionSampler(RolloutRejectionSampler):
    """vLLM's rejection sampling of the verify steps the fused sampler serves, as three fused kernels
    (csrc/verify_sampler.cu): one read of every row, a CTA per verify row, the ranks a top list leaves open."""

    def verify(self, logits, input_batch, num_logprobs):
        sampling_states = self.sampler.sampling_states

        return torch.ops.pear_rollout_blackwell.verify_sample(
            logits.contiguous(),
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.seq_lens,
            self.sampler.req_states.prefill_len.gpu,
            sampling_states.temperature.gpu,
            sampling_states.seeds.gpu,
            self.rates,
            num_logprobs,
            self.num_speculative_steps + 1,
        )
