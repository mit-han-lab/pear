from vllm.forward_context import get_forward_context

from pear.models.qwen3_5.rollout.hopper.all_reduce import DecodeAllReduce
from pear.models.qwen3_5.rollout.hopper.gdn import GDNProjection
from pear.models.qwen3_5.rollout.model import Qwen3_5RolloutForConditionalGeneration


class HopperQwen3_5ForConditionalGeneration(Qwen3_5RolloutForConditionalGeneration):
    """The Qwen3.5 vision-language model of a Hopper replica, its pure decode steps on the decode kernels."""

    def forward(self, *args, **kwargs):
        # a pure decode step: vLLM's uniform batch of one token per request
        descriptor = get_forward_context().batch_descriptor
        DecodeAllReduce.begin_step(
            bool(descriptor is not None and descriptor.uniform and descriptor.num_tokens == descriptor.num_reqs)
        )

        return super().forward(*args, **kwargs)

    def process_weights_after_loading(self) -> None:
        # every rank after every load (collectives): the merged GDN projections and the decode all-reduce's workspace
        GDNProjection.check(self)
        DecodeAllReduce.create_workspace(self)
