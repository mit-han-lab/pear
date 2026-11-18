from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch

from pear.models.qwen3_5.rollout.hopper.all_reduce import AllReduceNorm
from pear.models.qwen3_5.rollout.hopper.sampler import HopperSampler
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.model_state import Qwen3_5RolloutModelState


class HopperRolloutModelState(Qwen3_5RolloutModelState):
    """The model state of a Hopper replica: vLLM's hybrid model state with the fused sampler, which tells the decoder
    layers' all-reduces whether a forward is a pure decode step, and a new turn's M-RoPE positions from its first
    uncomputed token on."""

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

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        # vLLM's V2 runner sets up every forward here, the captures' too. A FULL graph's capture promises its queries'
        # length (one token in a decode graph, its token count in a mixed one); a forward without a graph computes its
        # own batch (a graph replays what its capture recorded)
        if for_capture:
            is_pure_decode = input_batch.max_query_len == 1
        else:
            is_pure_decode = input_batch.decode_graph_eligible and input_batch.num_tokens == input_batch.num_reqs
        AllReduceNorm.begin_step(is_pure_decode)

        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
