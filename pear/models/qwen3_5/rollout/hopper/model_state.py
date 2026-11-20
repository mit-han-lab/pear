import numpy as np
from vllm.v1.core.sched.output import NewRequestData

from pear.models.qwen3_5.rollout.hopper.gdn import ReplayState
from pear.models.qwen3_5.rollout.hopper.sampler import HopperRejectionSampler, HopperSampler
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.model_state import Qwen3_5RolloutModelState


class HopperRolloutModelState(Qwen3_5RolloutModelState):
    """The model state of a Hopper replica: vLLM's hybrid model state with the fused samplers and a new turn's M-RoPE
    positions from its first uncomputed token on; with a drafter, the Gated DeltaNet layers' replaySSM bookkeeping
    (ReplayState) in place of vLLM's per-draft-token state copies."""

    sampler_cls = HopperSampler
    rejection_sampler_cls = HopperRejectionSampler

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.replay_state = None

    def initialize_kv_cache(self, kv_cache_config, block_tables) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        num_speculative_tokens = self.vllm_config.num_speculative_tokens
        if num_speculative_tokens == 0:
            return
        if not self._align_mode:
            raise NotImplementedError("the Gated DeltaNet verify steps under replaySSM need mamba_cache_mode align")

        # the GDN layers of each mamba group, in the order of vLLM's aligned indices (its sorted mamba group ids)
        forward_context = self.vllm_config.compilation_config.static_forward_context
        groups = [
            [forward_context[name] for name in kv_cache_config.kv_cache_groups[group_id].layer_names]
            for group_id in self._mamba_group_ids
        ]
        self.replay_state = ReplayState.instance = ReplayState(
            groups, self.max_num_reqs, 1 + num_speculative_tokens, self._mamba_ctx.block_size, self.device
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if not StepInputArrays.stage(self, req_index, new_req_data):
            super().add_request(req_index, new_req_data)
        else:
            # vLLM's, without the positions of the whole prompt that StepInputArrays staged from the first uncomputed
            # token
            rope_state, self.rope_state = self.rope_state, None
            try:
                super().add_request(req_index, new_req_data)
            finally:
                self.rope_state = rope_state

        # a new request (or one resumed from the prefix cache) has no verify block to replay
        if self.replay_state is not None:
            self.replay_state.replay[req_index] = 0

    def preprocess_state(self, input_batch, block_tables, kv_cache_config, num_computed_tokens) -> None:
        if self.replay_state is None:
            return super().preprocess_state(input_batch, block_tables, kv_cache_config, num_computed_tokens)
        if input_batch.num_reqs == 0:
            return

        self.replay_state.preprocess(self, input_batch, num_computed_tokens, self.verify_rows(input_batch))

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None) -> None:
        if self.replay_state is None:
            return super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        if idx_mapping.shape[0] == 0:
            return

        self.replay_state.postprocess(self, idx_mapping, num_sampled)

    def verify_rows(self, input_batch) -> np.ndarray | None:
        """The step's verify rows as vLLM's attention metadata counts its speculative rows (decoding, at most one token
        and the drafts), None when every row is one."""
        num_reqs = input_batch.num_reqs
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        if input_batch.prefill_runs_as_decode_np is not None:
            is_prefilling = is_prefilling & ~input_batch.prefill_runs_as_decode_np[:num_reqs]
        num_tokens = input_batch.num_scheduled_tokens[:num_reqs]
        is_verify = ~is_prefilling & (num_tokens > 0) & (num_tokens <= 1 + self.vllm_config.num_speculative_tokens)

        return None if is_verify.all() else is_verify
