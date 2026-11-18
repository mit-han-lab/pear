from typing import ClassVar

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from pear.models.qwen3_5.rollout.sampler import RolloutSampler


class Qwen3_5RolloutModelState(MambaHybridModelState):
    """vLLM's V2 model state of a hybrid model, with a GPU backend's sampler in place of vLLM's and, with a drafter,
    its rejection sampler over it."""

    sampler_cls: ClassVar[type[RolloutSampler]]  # the GPU backend's sampler
    # the GPU backend's rejection sampler: vLLM's unless the backend has kernels for verify steps
    rejection_sampler_cls: ClassVar[type[RejectionSampler]] = RejectionSampler

    def custom_sampler(self, sampler: Sampler) -> tuple[RolloutSampler, RejectionSampler | None] | None:
        # vLLM's samplers stay when the engine's sampling options leave the kernels no rows. A drafter's verify steps
        # go to the rejection sampler, which reads the GPU backend's sampler's request states and processing; the
        # runner hands the sampler the steps without drafts alone
        if not self.sampler_cls.serves(sampler):
            return None

        rollout_sampler = self.sampler_cls(sampler)
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None:
            return rollout_sampler, None

        return rollout_sampler, self.rejection_sampler_cls(rollout_sampler, speculative_config, self.device)
