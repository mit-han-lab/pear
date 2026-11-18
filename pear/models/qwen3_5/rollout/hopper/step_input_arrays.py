import numpy as np
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration


class StepInputArrays:
    """The host side of a step's inputs as array operations: a new turn's M-RoPE positions from its first uncomputed
    token on."""

    # the GPU idles until a new turn's step has its inputs, so their Python cost adds to the turn: vLLM computes a new
    # request's positions for its whole prompt
    MIN_PREFIX = 1024  # cached prompt tokens from which a new request's M-RoPE positions skip them

    @classmethod
    def stage(cls, model_state, req_index: int, request) -> bool:
        """Stage a new request's M-RoPE positions from its first uncomputed token on into the model state's, False where
        vLLM's apply."""
        # EVS reads a request's positions from its first token on (vLLM's MultiModalPruner)
        start = request.num_computed_tokens
        if not (
            start >= cls.MIN_PREFIX
            and request.prompt_token_ids is not None
            and request.prompt_embeds is None
            and model_state.mm_pruner is None
        ):
            return False
        tail = cls.tail_positions(model_state.model, request.prefill_token_ids, request.mm_features, start)
        if tail is None:
            return False

        positions, delta = tail
        rope_state = model_state.rope_state
        rope_state.prefill_delta.np[req_index] = delta
        for dim in range(rope_state.num_dims):
            row = rope_state.num_dims * req_index + dim
            rope_state.prefill_positions.stage_write(row, start, positions[dim, start:].numpy())

        return True

    @staticmethod
    def tail_positions(model, ids, features, start):
        """Qwen3-VL's M-RoPE positions of the tokens from start on (the [3, n] array unset before it) and the exact
        position delta; None where that layout does not apply (video, other models)."""
        model_cls = Qwen3VLForConditionalGeneration
        if not isinstance(model, model_cls) or type(model).get_mrope_input_positions is not (
            model_cls.get_mrope_input_positions
        ):
            return None
        if any(f.modality != "image" for f in features):
            return None

        config = model.config
        n = len(ids)
        out = np.empty((3, n), dtype=np.int64)
        st, last_max = 0, -1
        segments = model_cls._iter_mm_grid_hw(
            ids,
            features,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        )

        for offset, gh, gw, num in segments:
            if num == 0:
                continue
            if num != gh * gw:
                return None
            st_idx = last_max + 1
            text_len = offset - st
            a = max(st, start)
            if a < offset:  # text tokens [st, offset): st_idx + their index in the segment
                out[:, a:offset] = st_idx + np.arange(a - st, text_len)
            base = st_idx + text_len
            a = max(offset, start)
            if a < offset + num:  # image tokens: (t, h, w) grid positions after the text
                j = np.arange(a - offset, num)
                out[0, a : offset + num] = base
                out[1, a : offset + num] = base + j // gw
                out[2, a : offset + num] = base + j % gw
            last_max = base + max(gh, gw) - 1
            st = offset + num

        if st < n:
            st_idx = last_max + 1
            a = max(st, start)
            if a < n:
                out[:, a:] = st_idx + np.arange(a - st, n - st)
            last_max = st_idx + n - st - 1

        return torch.from_numpy(out), int(last_max + 1 - n)
