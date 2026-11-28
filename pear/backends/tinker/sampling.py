from __future__ import annotations

import numpy as np
import tinker
from tinker.types.topk_logprobs import TopkLogprobs

# The service permits a larger head only when a generation requests at most this many token/head pairs.
MAX_HEAD_VALUES = 1000
UNRESTRICTED_HEAD_SIZE = 20


def head_request_limit(max_tokens, topk, allow_continuations=False):
    cap = max_tokens if topk <= UNRESTRICTED_HEAD_SIZE else MAX_HEAD_VALUES // topk
    if cap < 1:
        raise ValueError(f"topk cannot exceed {MAX_HEAD_VALUES}")
    if cap < max_tokens and allow_continuations is not True:
        raise ValueError(
            f"Tinker's current top-{topk} limit requires requests of at most {cap} tokens. "
            "Continuing them repeatedly bills the growing prefix and is unsuitable for full PEAR runs. "
            "Set allow_head_continuations=true only for bounded validation, request higher service limits, "
            "or choose topk=20 as an explicit change to the score-centering approximation."
        )
    return cap


async def sample_with_heads(client, prompt, params, topk, *, allow_continuations=False, usage=None):
    """Continue short API requests into one answer, retaining the sampled prefix and every probability."""
    cap = head_request_limit(params.max_tokens, topk, allow_continuations)

    async def sample(**kwargs):
        return await (usage.sample(client, **kwargs) if usage is not None else client.sample(**kwargs))

    if cap >= params.max_tokens:
        response = await sample(prompt=prompt, num_samples=1, sampling_params=params, topk_sample_logprobs=topk)
        return response, 1, prompt.length
    if params.stop and not all(isinstance(token, int) for token in params.stop):
        raise ValueError("continuing top-k sampling requires token ID stop sequences")

    parts, prefix, remaining = [], prompt, params.max_tokens
    prompt_tokens, cache_hits = 0, 0
    while remaining:
        requested = min(cap, remaining)
        seed = (params.seed + len(parts)) % (2**31) if params.seed is not None else None
        response = await sample(
            prompt=prefix,
            num_samples=1,
            sampling_params=params.model_copy(update={"max_tokens": requested, "seed": seed}),
            topk_sample_logprobs=topk,
        )
        sequence = response.sequences[0]
        if not 0 < len(sequence.tokens) <= requested:
            raise ValueError("sampler returned an empty or oversized continuation")
        if sequence.logprobs is None or len(sequence.logprobs) != len(sequence.tokens):
            raise ValueError("continuation is missing sampled log probabilities")
        heads = sequence.topk_logprobs_np
        if heads is None or heads.token_ids.shape != (len(sequence.tokens), topk):
            raise ValueError("continuation is missing sampler head probabilities")
        parts.append(sequence)
        prompt_tokens += prefix.length
        cache_hits += response.prompt_cache_hit_tokens
        remaining -= len(sequence.tokens)
        if sequence.stop_reason == "stop":
            break
        prefix = prefix.append(tinker.EncodedTextChunk(tokens=sequence.tokens))

    sequence = tinker.SampledSequence(
        stop_reason=parts[-1].stop_reason,
        # This answer combines multiple service sequences and has no single service provenance ID.
        sequence_id="",
        tokens_np=np.concatenate([np.asarray(part.tokens, dtype=np.int32) for part in parts]),
        logprobs_np=np.concatenate([np.asarray(part.logprobs, dtype=np.float32) for part in parts]),
        topk_logprobs_np=TopkLogprobs(
            token_ids=np.concatenate([part.topk_logprobs_np.token_ids for part in parts]),
            logprobs=np.concatenate([part.topk_logprobs_np.logprobs for part in parts]),
        ),
    )
    return tinker.SampleResponse(sequences=[sequence], prompt_cache_hit_tokens=cache_hits), len(parts), prompt_tokens
