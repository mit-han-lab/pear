import numpy as np
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.worker import gpu_input_batch


class PromptTailCopy:
    """A new request's prompt copied into the runner's token buffer from its first uncomputed token on."""

    MIN_PREFIX = 4096  # cached prompt tokens from which a new request's copy skips them
    # vLLM's own logits processors, which do not read the prompt they are handed
    BUILTIN_PROCESSORS = (LogitBiasLogitsProcessor, MinPLogitsProcessor, MinTokensLogitsProcessor)

    # vLLM converts the whole prompt (a Python list) into the buffer as the request joins the batch, inside the step.
    # The runner reads the scheduled positions; the cached prefix is read only by penalties and prompt log-probs
    # (vLLM's copy then)
    @classmethod
    def install(cls, vllm_config) -> None:
        """Patch vLLM's input batch, unless the replica drafts tokens."""
        # the model runner builds the batch without a hook; drafters read the whole prompt
        if vllm_config.speculative_config is not None:
            return

        vllm_add_request = gpu_input_batch.InputBatch.add_request

        def add_request(self, request):
            if not cls.eligible(self, request):
                return vllm_add_request(self, request)

            tokens = request.prompt_token_ids
            num_computed = request.num_computed_tokens
            tail = np.zeros(len(tokens), dtype=self.token_ids_cpu.dtype)
            tail[num_computed:] = tokens[num_computed:]
            request.prompt_token_ids = tail

            try:
                return vllm_add_request(self, request)
            finally:
                request.prompt_token_ids = tokens

        gpu_input_batch.InputBatch.add_request = add_request

    @staticmethod
    def eligible(batch, request) -> bool:
        """A token prompt with a long cached prefix, and nothing in the batch that reads this request's prefix."""
        params = request.sampling_params
        return (
            params is not None
            and isinstance(request.prompt_token_ids, list)
            and request.prompt_embeds is None
            and request.prompt_is_token_ids is None
            and request.num_computed_tokens >= PromptTailCopy.MIN_PREFIX
            and params.prompt_logprobs is None
            and params.repetition_penalty == 1.0
            and params.frequency_penalty == 0.0
            and params.presence_penalty == 0.0
            and all(isinstance(p, PromptTailCopy.BUILTIN_PROCESSORS) for p in batch.logitsprocs.all)
        )
