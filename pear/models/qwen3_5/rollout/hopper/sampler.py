import torch
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from pear.models.qwen3_5.rollout.sampler import RolloutSampler


class HopperSampler(RolloutSampler):
    """vLLM's sampling of random rows with processed top-K log-probs as four kernels (csrc/sampler.cu)."""

    # one pass over the logits gives log Z and the top-K, the draw is exact inverse-CDF sampling in two levels from one
    # uniform per row (Philox, counter = (call, row)), the rank counted only when the draw is outside the top-K; every
    # TP rank calls the sampler in lockstep, so the calls number the draws
    WORKSPACE_BYTES = 8 << 20  # the kernels' per-chunk and per-row records
    SEED_MASK = (1 << 63) - 1  # the Philox seed: torch's initial seed as a non-negative int64
    seed = None
    calls = 0
    workspaces: dict = {}  # device index -> the kernels' workspace

    @classmethod
    def eligible(cls, sampler, logits, md, predict_bonus_token, mode) -> bool:
        """The shared conditions and a temperature for every row."""
        return (
            super().eligible(sampler, logits, md, predict_bonus_token, mode)
            and logits.shape[0] <= md.temperature.shape[0]
        )

    @classmethod
    def draw(cls, logits, md) -> SamplerOutput:
        if logits.stride(1) != 1 or logits.stride(0) % 4 or logits.data_ptr() % 16:
            logits = logits.contiguous()

        return cls.sample(logits, md.temperature, md.max_num_logprobs)

    @classmethod
    def sample(cls, logits, temperature, k: int) -> SamplerOutput:
        """Draw one token per row of fp32 logits [rows, vocab] at the rows' temperatures, with its top-k log-probs."""
        device, rows = logits.device, logits.shape[0]
        if cls.seed is None:
            cls.seed = torch.initial_seed() & cls.SEED_MASK
        if device.index not in cls.workspaces:
            cls.workspaces[device.index] = torch.empty(cls.WORKSPACE_BYTES, dtype=torch.uint8, device=device)

        sampled_ids = torch.empty(rows, dtype=torch.int64, device=device)
        ids = torch.empty(rows, k + 1, dtype=torch.int32, device=device)
        logprobs = torch.empty(rows, k + 1, dtype=torch.float32, device=device)
        ranks = torch.empty(rows, dtype=torch.int64, device=device)

        cls.calls += 1
        torch.ops.pear_rollout_hopper.fused_sample(
            logits,
            temperature[:rows].float(),
            cls.seed,
            cls.calls,
            k,
            cls.workspaces[device.index],
            sampled_ids,
            ids,
            logprobs,
            ranks,
        )

        return SamplerOutput(
            sampled_token_ids=sampled_ids.to(torch.int32).unsqueeze(-1),
            logprobs_tensors=LogprobsTensors(ids, logprobs, ranks),
        )
