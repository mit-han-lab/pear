from types import SimpleNamespace

import numpy as np
import pytest
import tinker
from tinker.types.topk_logprobs import TopkLogprobs

from pear.backends.tinker.sampling import sample_with_heads


class Continuations:
    def __init__(self, length=65, missing=None):
        self.length, self.missing, self.calls = length, missing, []

    async def sample(self, *, prompt, num_samples, sampling_params, topk_sample_logprobs):
        assert num_samples == 1
        k, cap = topk_sample_logprobs, sampling_params.max_tokens
        assert k <= 20 or cap * k <= 1000
        offset = prompt.length - 1
        prefix = [token for chunk in prompt.chunks for token in chunk.tokens]
        assert prefix == [999, *range(offset)]
        self.calls.append((cap, sampling_params.seed))
        end = min(self.length, offset + cap)
        tokens = np.arange(offset, end, dtype=np.int32)
        if self.missing == "tokens":
            tokens = tokens[:0]
        heads = TopkLogprobs(
            token_ids=np.tile(np.arange(k, dtype=np.int32), (len(tokens), 1)),
            logprobs=np.repeat(-tokens.astype(np.float32)[:, None], k, axis=1),
        )
        sequence = tinker.SampledSequence(
            sequence_id=str(offset),
            stop_reason="stop" if end == self.length else "length",
            tokens_np=tokens,
            logprobs_np=None if self.missing == "logprobs" else -tokens.astype(np.float32),
            topk_logprobs_np=None if self.missing == "heads" else heads,
        )
        return SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=offset)


@pytest.mark.asyncio
@pytest.mark.parametrize(("length", "stop"), [(65, "stop"), (100, "length")])
async def test_top32_continuations_preserve_every_token_and_probability(length, stop):
    client = Continuations(length)
    response, calls, prompt_tokens = await sample_with_heads(
        client,
        tinker.ModelInput.from_ints([999]),
        tinker.SamplingParams(max_tokens=70, seed=7, stop=[2]),
        32,
        allow_continuations=True,
    )
    [sequence] = response.sequences
    expected = list(range(min(length, 70)))
    assert sequence.tokens == expected and sequence.logprobs == [-token for token in expected]
    assert sequence.stop_reason == stop
    assert client.calls == [(31, 7), (31, 8), (8, 9)]
    assert calls == 3 and prompt_tokens == 1 + 32 + 63
    assert response.prompt_cache_hit_tokens == 31 + 62
    np.testing.assert_array_equal(sequence.topk_logprobs_np.logprobs[:, 0], -np.array(expected))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["tokens", "logprobs", "heads"])
async def test_incomplete_continuations_fail_without_training_on_missing_data(missing):
    with pytest.raises(ValueError):
        await sample_with_heads(
            Continuations(missing=missing),
            tinker.ModelInput.from_ints([999]),
            tinker.SamplingParams(max_tokens=70),
            32,
            allow_continuations=True,
        )


@pytest.mark.asyncio
async def test_smaller_heads_use_one_request():
    client = Continuations()
    _, calls, _ = await sample_with_heads(
        client, tinker.ModelInput.from_ints([999]), tinker.SamplingParams(max_tokens=256), 20
    )
    assert calls == 1 and client.calls == [(256, None)]


@pytest.mark.asyncio
async def test_expensive_continuations_require_explicit_opt_in():
    client = Continuations()
    with pytest.raises(ValueError, match="allow_head_continuations=true"):
        await sample_with_heads(
            client, tinker.ModelInput.from_ints([999]), tinker.SamplingParams(max_tokens=160000), 32
        )
    assert not client.calls
