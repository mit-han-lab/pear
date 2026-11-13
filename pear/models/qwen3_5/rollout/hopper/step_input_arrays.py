import numpy as np
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.v1.worker import gpu_model_runner


class StepInputArrays:
    """The host side of a step's inputs as array operations: M-RoPE positions (a new turn's from its first uncomputed
    token on) and the input ids carried over from the previous step."""

    # the GPU idles until a new turn's step has its inputs, so their Python cost adds to the turn: vLLM loops over the
    # batch's requests and computes a new request's positions for its whole prompt
    MIN_PREFIX = 1024  # cached prompt tokens from which a new request's M-RoPE positions skip them

    @classmethod
    def install(cls) -> None:
        """Patch the model runner's host-side input preparation."""
        runner_cls = gpu_model_runner.GPUModelRunner
        vllm_init_mrope_positions = runner_cls._init_mrope_positions
        vllm_calc_mrope_positions = runner_cls._calc_mrope_positions
        vllm_prepare_input_ids = runner_cls._prepare_input_ids

        def init_mrope_positions(self, req_state):
            start = req_state.num_computed_tokens
            if start >= cls.MIN_PREFIX and req_state.prompt_token_ids is not None and req_state.prompt_embeds is None:
                tail = cls.tail_positions(self.get_model(), req_state.prompt_token_ids, req_state.mm_features, start)
                if tail is not None:
                    req_state.mrope_positions, req_state.mrope_position_delta = tail
                    req_state._pear_mrope_from = start
                    return

            vllm_init_mrope_positions(self, req_state)
            req_state._pear_mrope_from = 0

        def compute_all_positions(self, req_state):
            """A request's positions read before the first one computed lazily: compute them all."""
            if getattr(req_state, "_pear_mrope_from", 0):
                vllm_init_mrope_positions(self, req_state)
                req_state._pear_mrope_from = 0

        def calc_mrope_positions(self, scheduler_output):
            """vLLM's loop as array operations: a request's scheduled prompt tokens take their precomputed positions,
            its completion tokens the text positions from context_len + delta on (the three sections equal)."""
            req_ids = self.input_batch.req_ids
            n = len(req_ids)
            if n == 0 or self.input_batch.req_prompt_embeds:
                for req_id in req_ids:
                    compute_all_positions(self, self.requests[req_id])
                return vllm_calc_mrope_positions(self, scheduler_output)

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            requests = self.requests
            num_sched = np.fromiter((num_scheduled_tokens[r] for r in req_ids), dtype=np.int64, count=n)
            num_computed = self.input_batch.num_computed_tokens_cpu[:n].astype(np.int64)
            num_prompt = self.input_batch.num_prompt_tokens[:n].astype(np.int64)
            prompt_part = np.clip(num_prompt - num_computed, 0, num_sched)
            num_completion = num_sched - prompt_part
            starts = np.cumsum(num_sched) - num_sched
            out = self.mrope_positions.np
            cpu = self.mrope_positions.cpu

            for i in np.flatnonzero(prompt_part):
                req = requests[req_ids[i]]
                a, b = int(num_computed[i]), int(num_computed[i] + prompt_part[i])
                if a < getattr(req, "_pear_mrope_from", 0):
                    compute_all_positions(self, req)
                cpu[:, starts[i] : starts[i] + prompt_part[i]] = req.mrope_positions[:, a:b]

            rows = np.flatnonzero(num_completion)
            if len(rows):
                delta = np.fromiter(
                    (requests[req_ids[i]].mrope_position_delta for i in rows), dtype=np.int64, count=len(rows)
                )
                count = num_completion[rows]
                first_positions = delta + num_computed[rows] + prompt_part[rows]  # each row's first completion token
                dst0 = starts[rows] + prompt_part[rows]
                total = int(count.sum())
                rel = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
                dst = np.repeat(dst0, count) + rel
                out[:, dst] = (np.repeat(first_positions, count) + rel).astype(out.dtype)

        def prepare_input_ids(self, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens):
            """vLLM's asynchronous-scheduling path without speculative tokens as array operations: a request carried
            over from the previous step takes its last sampled token on the GPU."""
            if (
                self._pp_recv_work is not None
                or self.input_batch.prev_sampled_token_ids is None
                or scheduler_output.scheduled_spec_decode_tokens
                or self.enable_prompt_embeds
            ):
                return vllm_prepare_input_ids(
                    self, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens
                )

            prev_positions = self.prev_positions.np[:num_reqs]
            is_common = prev_positions >= 0
            prev_indices = prev_positions[is_common].astype(np.int64)
            flat_positions = np.asarray(cu_num_tokens[:num_reqs], dtype=np.int64)[is_common] - 1
            num_common_tokens = len(flat_positions)
            if num_common_tokens < total_num_scheduled_tokens:
                self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if num_common_tokens == 0:
                return

            prev_tokens = self.input_batch.prev_sampled_token_ids
            if (
                np.array_equal(prev_indices, flat_positions)
                and flat_positions[-1] == num_common_tokens - 1
                and flat_positions.max() == num_common_tokens - 1
            ):
                self.input_ids.gpu[:num_common_tokens].copy_(prev_tokens[:num_common_tokens, 0], non_blocking=True)
                return

            index = torch.from_numpy(flat_positions).pin_memory().to(self.device, non_blocking=True)
            src = torch.from_numpy(prev_indices).pin_memory().to(self.device, non_blocking=True)
            self.input_ids.gpu.scatter_(dim=0, index=index, src=prev_tokens[src, 0])

        runner_cls._init_mrope_positions = init_mrope_positions
        runner_cls._calc_mrope_positions = calc_mrope_positions
        runner_cls._prepare_input_ids = prepare_input_ids

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
