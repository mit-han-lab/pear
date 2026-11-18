from typing import ClassVar

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from pear.models.qwen3_5.rollout.sampler import RolloutSampler


class Qwen3_5RolloutModelState(MambaHybridModelState):
    """vLLM's V2 model state of a hybrid model, with a GPU backend's sampler in place of vLLM's."""

    sampler_cls: ClassVar[type[RolloutSampler]]  # the GPU backend's sampler

    def custom_sampler(self, sampler: Sampler) -> tuple[RolloutSampler, None] | None:
        # vLLM's sampler stays when the engine's sampling options leave the kernels no rows; no drafter, no rejection
        # sampler (the GPU backends refuse a drafter)
        if not self.sampler_cls.serves(sampler):
            return None

        return self.sampler_cls(sampler), None
