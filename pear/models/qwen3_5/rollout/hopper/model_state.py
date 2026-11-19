from vllm.v1.core.sched.output import NewRequestData

from pear.models.qwen3_5.rollout.hopper.sampler import HopperSampler
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.model_state import Qwen3_5RolloutModelState


class HopperRolloutModelState(Qwen3_5RolloutModelState):
    """The model state of a Hopper replica: vLLM's hybrid model state with the fused sampler and a new turn's M-RoPE
    positions from its first uncomputed token on."""

    sampler_cls = HopperSampler

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if not StepInputArrays.stage(self, req_index, new_req_data):
            super().add_request(req_index, new_req_data)
            return

        # vLLM's, without the positions of the whole prompt that StepInputArrays staged from the first uncomputed token
        rope_state, self.rope_state = self.rope_state, None
        try:
            super().add_request(req_index, new_req_data)
        finally:
            self.rope_state = rope_state
