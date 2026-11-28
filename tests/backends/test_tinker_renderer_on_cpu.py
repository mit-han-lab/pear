from __future__ import annotations

import io
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tinker
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import AutoImageProcessor, AutoTokenizer, PreTrainedTokenizerFast, Qwen2VLImageProcessor

from pear.backends.tinker.config import TinkerConfig
from pear.backends.tinker.rollout import Conversation
from pear.backends.tinker.runtime import _renderer, make_renderer


@pytest.fixture
def assets(monkeypatch):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]})
    # The actual processor and Cookbook renderer; only the asset download is substituted.
    processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216}
    )
    loads = [
        Mock(return_value=tokenizer),
        Mock(side_effect=lambda *a, **kw: processor.__class__(**processor.to_dict())),
    ]
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loads[0])
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", loads[1])
    _renderer.cache_clear()
    yield loads
    _renderer.cache_clear()


def test_production_renderer_preserves_camera_pixels_order_and_token_positions(assets):
    config = TinkerConfig.load("examples/tinker/libero.yaml", [])
    renderer = make_renderer(config)
    rng = np.random.default_rng(7)
    frames = [Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)) for size in [(64, 96), (320, 256)] * 2]

    def observation(images):
        return [{"type": "text", "text": "front and wrist cameras"}, *[{"type": "image", "image": im} for im in images]]

    actual = Conversation(renderer, "Observe the scene.", observation(frames[:2]))
    original = Conversation(renderer.renderer, "Observe the scene.", observation(frames[:2]))

    def check(left, right, expected_frames):
        assert left.length == right.length
        images = []
        for a, b in zip(left.chunks, right.chunks, strict=True):
            assert type(a) is type(b)
            assert a.length == b.length
            if isinstance(a, tinker.types.ImageChunk):
                assert a.format == "png"
                assert a.expected_tokens == b.expected_tokens
                frame = expected_frames[len(images)]
                np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(a.data))), np.asarray(frame))
                # These detailed frames expose the production JPEG bug that solid-color fixtures missed.
                assert not np.array_equal(np.asarray(Image.open(io.BytesIO(b.data))), np.asarray(frame))
                images.append(a)
            else:
                assert a.tokens == b.tokens
        assert len(images) == len(expected_frames)

    check(actual.prompt, original.prompt, frames[:2])
    for conversation in (actual, original):
        conversation.append_answer([1, 3, 0, 4, 2], "<think>move</think>")
    check(actual.next_prompt(observation(frames[2:])), original.next_prompt(observation(frames[2:])), frames)


def test_renderer_pins_both_assets_and_includes_revision_in_cache(assets):
    config = TinkerConfig.load("examples/tinker/libero.yaml", [])
    first = make_renderer(config)
    assert make_renderer(config) is first
    for loader in assets:
        loader.assert_called_once_with("Qwen/Qwen3.8-27B", revision=config.model_revision)
    changed = replace(config, model_revision="a" * 40)
    assert make_renderer(changed) is not first
    for loader in assets:
        loader.assert_called_with("Qwen/Qwen3.8-27B", revision=changed.model_revision)
    assert config.signature() != changed.signature()
    with pytest.raises(ValueError, match="set model_revision"):
        make_renderer(replace(config, model_revision=None))
    with pytest.raises(ValueError, match="commit hash"):
        replace(config, model_revision="main").validate()
