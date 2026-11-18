from abc import ABC, abstractmethod

import numpy as np
import torch
from vllm.config import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from pear.models.qwen3_5.rollout.nonfinite_logits import NonfiniteLogits


class RolloutSampler(Sampler, ABC):
    """vLLM's V2 sampler with a GPU backend's own kernels for the batches whose requests sample at a temperature alone,
    with the top-K log-probs of the distribution sampled from."""

    LOGPROBS_MODE = "processed_logprobs"  # the kernels' log-probs: of the distribution at the temperature
    MAX_LOGPROBS = 32  # the largest top-K the kernels keep (kTopK and K of the two backends' csrc/sampler.cu)
    KERNEL = "the fused sampler"  # the kernels, as an error about their rows names them

    def __init__(self, sampler: Sampler):
        # vLLM's sampler as the runner built it: its requests' sampling state, its processors and options
        self.__dict__.update(sampler.__dict__)
        # per request slot: whether vLLM processes its logits by its temperature alone, the rows the kernels compute
        self.is_eligible = np.zeros_like(self.needs_logits_processing)

    @staticmethod
    def serves(sampler: Sampler) -> bool:
        """Whether the engine's sampling options leave the kernels rows to compute (vLLM's sampler takes the others)."""
        return (
            type(sampler) is Sampler  # not the watermarking sampler
            and sampler.logprobs_mode == RolloutSampler.LOGPROBS_MODE
            and not sampler.use_fp64_gumbel
            and not sampler.return_sampling_mask
            and sampler.trace_replay_state is None
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        super().add_request(req_idx, sampling_params)

        # vLLM's processors, the thinking budget and min-p / top-k / top-p leave the logits as they are
        self.is_eligible[req_idx] = (
            sampling_params.temperature > 0
            and not self.may_mask_logits[req_idx]
            and sampling_params.seed is None
            and not sampling_params.logprob_token_ids
        )

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        # the kernels draw one token per request from its one logits row (vLLM's sampler also takes several rows per
        # request)
        idx_mapping_np = input_batch.idx_mapping_np
        num_logprobs = self.kernel_num_logprobs(logits, idx_mapping_np)
        if num_logprobs is None or logits.shape[0] != len(idx_mapping_np):
            return super().__call__(logits, input_batch)

        num_nans = get_num_nans(logits) if self.compute_nans else None
        temperature = self.sampling_states.temperature.gpu[input_batch.idx_mapping]
        sampled, logprobs_tensors, nonfinite = self.draw(logits, temperature, num_logprobs)
        NonfiniteLogits.report(self.KERNEL, nonfinite, input_batch.req_ids)

        # a token per request, but for the chunked prefills not done with their prompt (vLLM's count)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def kernel_num_logprobs(self, logits: torch.Tensor, idx_mapping_np: np.ndarray) -> int | None:
        """The number of top log-probs the kernels compute for these requests, None when vLLM's samplers take them."""
        num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if self.is_eligible[idx_mapping_np].all() and 1 <= num_logprobs <= self.MAX_LOGPROBS and self.supports(logits):
            return num_logprobs

        return None

    @classmethod
    def supports(cls, logits: torch.Tensor) -> bool:
        """Whether the kernels take these logits: fp32 rows [rows, vocab]."""
        return logits.dtype == torch.float32 and logits.dim() == 2

    @abstractmethod
    def draw(
        self, logits: torch.Tensor, temperature: torch.Tensor, num_logprobs: int
    ) -> tuple[torch.Tensor, LogprobsTensors, torch.Tensor]:
        """One token per row of the logits at the row's temperature (int64), with its top num_logprobs log-probs and
        the row's count of NaN and +inf logits (int32)."""


class RolloutRejectionSampler(RejectionSampler, ABC):
    """vLLM's rejection sampler with a GPU backend's own kernels for the verify steps whose requests the rollout
    sampler serves: greedy drafts checked by vLLM's lossless rule (or its synthetic acceptance) at a temperature alone,
    with the processed top-K log-probs of every row."""

    METHODS = ("standard", "synthetic")  # the rejection rules the kernels apply
    KERNEL = "the verify sampler"  # the kernels, as an error about their rows names them

    def __init__(self, sampler: RolloutSampler, spec_config: SpeculativeConfig, device: torch.device):
        super().__init__(sampler, spec_config, device)
        # greedy drafts, every row verified: vLLM keeps block verification, FLy, draft distributions and adaptive
        # verification
        self.is_served = (
            spec_config.rejection_sample_method in self.METHODS
            and spec_config.draft_sample_method == "greedy"
            and not self.enable_adaptive_verification
        )
        # the synthetic acceptance's conditional rate per draft position, none for the lossless rule
        self.rates = self.synthetic_conditional_rates
        if self.rates is None:
            self.rates = torch.empty(0, dtype=torch.float32, device=device)

    def __call__(
        self, logits: torch.Tensor, input_batch: InputBatch, draft_logits: DraftLogitsCache | None = None
    ) -> SamplerOutput:
        num_logprobs = None
        if self.is_served and draft_logits is None:
            num_logprobs = self.sampler.kernel_num_logprobs(logits, input_batch.idx_mapping_np)
        if num_logprobs is None:
            return super().__call__(logits, input_batch, draft_logits)

        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None
        sampled, num_sampled, num_rejected, ids, logprobs, ranks, nonfinite = self.verify(
            logits, input_batch, num_logprobs
        )
        NonfiniteLogits.report(self.KERNEL, nonfinite, input_batch.req_ids, input_batch.cu_num_logits_np)

        # the rows of each request, as vLLM's output slices them (its first num_sampled rows are the kept tokens')
        is_expanded = logits.shape[0] != input_batch.num_reqs
        cu_num_logits = input_batch.cu_num_logits_np.tolist() if is_expanded else None

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=LogprobsTensors(ids, logprobs, ranks, cu_num_generated_tokens=cu_num_logits),
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    @abstractmethod
    def verify(
        self, logits: torch.Tensor, input_batch: InputBatch, num_logprobs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The verify step's sampled tokens [requests, drafts + 1] (int64), their counts and the rejected rows' counts
        per request (int32), and every row's log-prob ids, log-probs, kept token's rank and count of NaN and +inf
        logits (int32)."""
