from abc import ABC, abstractmethod

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


class RolloutSampler(Sampler, ABC):
    """vLLM's V2 sampler with a GPU backend's own kernels for the batches whose requests sample at a temperature alone,
    with the top-K log-probs of the distribution sampled from."""

    LOGPROBS_MODE = "processed_logprobs"  # the kernels' log-probs: of the distribution at the temperature
    MAX_LOGPROBS = 32  # the largest top-K the kernels keep (kTopK and K of the two backends' csrc/sampler.cu)

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
        idx_mapping_np = input_batch.idx_mapping_np
        num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if not (
            self.is_eligible[idx_mapping_np].all() and 1 <= num_logprobs <= self.MAX_LOGPROBS and self.supports(logits)
        ):
            return super().__call__(logits, input_batch)

        num_nans = get_num_nans(logits) if self.compute_nans else None
        temperature = self.sampling_states.temperature.gpu[input_batch.idx_mapping]
        sampled, logprobs_tensors = self.draw(logits, temperature, num_logprobs)

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

    @classmethod
    def supports(cls, logits: torch.Tensor) -> bool:
        """Whether the kernels take these logits: fp32 rows [rows, vocab]."""
        return logits.dtype == torch.float32 and logits.dim() == 2

    @abstractmethod
    def draw(
        self, logits: torch.Tensor, temperature: torch.Tensor, num_logprobs: int
    ) -> tuple[torch.Tensor, LogprobsTensors]:
        """One token per row of the logits at the row's temperature (int64), with its top num_logprobs log-probs."""
