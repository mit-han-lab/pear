import torch
from vllm.v1.outputs import LogprobsTensors

from pear.models.qwen3_5.rollout.sampler import RolloutRejectionSampler, RolloutSampler


class HopperSampler(RolloutSampler):
    """vLLM's sampling of the rows at a temperature alone, with processed top-K log-probs, as four kernels
    (csrc/sampler.cu)."""

    # one pass over the logits gives log Z and the top-K, the draw is exact inverse-CDF sampling in two levels from one
    # uniform per row (Philox, counter = (call, row)), the rank counted only when the draw is outside the top-K; every
    # TP rank calls the sampler in lockstep, so the calls number the draws
    WORKSPACE_BYTES = 8 << 20  # the kernels' per-chunk and per-row records
    SEED_MASK = (1 << 63) - 1  # the Philox seed: torch's initial seed as a non-negative int64
    seed = None
    calls = 0
    workspaces: dict = {}  # device index -> the kernels' workspace

    def draw(self, logits, temperature, num_logprobs):
        # the kernels read the rows as float4 at 16-byte aligned addresses
        if logits.stride(1) != 1 or logits.stride(0) % 4 or logits.data_ptr() % 16:
            logits = logits.contiguous()

        return self.draw_rows(logits, temperature, num_logprobs)

    @classmethod
    def draw_rows(cls, logits, temperature, k: int) -> tuple[torch.Tensor, LogprobsTensors, torch.Tensor]:
        """Draw one token per row of fp32 logits [rows, vocab] at the rows' temperatures (int64), with its top-k
        log-probs and the row's count of NaN and +inf logits (int32)."""
        device, rows = logits.device, logits.shape[0]
        if cls.seed is None:
            cls.seed = torch.initial_seed() & cls.SEED_MASK
        if device.index not in cls.workspaces:
            cls.workspaces[device.index] = torch.empty(cls.WORKSPACE_BYTES, dtype=torch.uint8, device=device)

        sampled_ids = torch.empty(rows, dtype=torch.int64, device=device)
        ids = torch.empty(rows, k + 1, dtype=torch.int32, device=device)
        logprobs = torch.empty(rows, k + 1, dtype=torch.float32, device=device)
        ranks = torch.empty(rows, dtype=torch.int64, device=device)
        nonfinite = torch.empty(rows, dtype=torch.int32, device=device)

        cls.calls += 1
        torch.ops.pear_rollout_hopper.fused_sample(
            logits,
            temperature,
            cls.seed,
            cls.calls,
            k,
            cls.workspaces[device.index],
            sampled_ids,
            ids,
            logprobs,
            ranks,
            nonfinite,
        )

        return sampled_ids, LogprobsTensors(ids, logprobs, ranks), nonfinite


class HopperRejectionSampler(RolloutRejectionSampler):
    """vLLM's rejection sampling of the verify steps the fused sampler serves, as three fused kernels
    (csrc/verify_sampler.cu): one read of every row, a CTA per request, the ranks a top list leaves open."""

    def verify(self, logits, input_batch, num_logprobs):
        sampling_states = self.sampler.sampling_states

        return torch.ops.pear_rollout_hopper.verify_sample(
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
