import types

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from pear.models.qwen3_5.rollout.hopper.model import HopperQwen3_5ForConditionalGeneration
from pear.models.qwen3_5.rollout.hopper.model_state import HopperRolloutModelState
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's M-RoPE positions on the GPU")

DEVICE = torch.device("cuda")
FIRST_TEXT = StepInputArrays.MIN_PREFIX + 200  # the turn's text before its first image
OUTPUT = 300  # a preempted turn's output tokens
MAX_LEN = 4096  # the replicas' max_model_len in the test
MAX_NUM_REQS = 4
TOKEN_BUDGET = 8192  # the replicas' max_num_batched_tokens
DECODE_TOKENS = 4  # positions after the prefill a step reads
VISION_START, VISION_END, IMAGE, VIDEO = 151652, 151653, 151655, 151656  # the model's vision token ids
MERGE = 2  # the vision tower's spatial merge
NUM_DIMS = 3  # Qwen3-VL's M-RoPE sections


def image_turn(first_text: int) -> tuple[types.SimpleNamespace, list[int], list]:
    """Qwen3-VL's vision config and a turn of text and three images, the first after `first_text` tokens, as token ids
    and vLLM's multimodal features."""
    config = types.SimpleNamespace(
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        vision_config=types.SimpleNamespace(spatial_merge_size=MERGE),
    )
    ids, features = [], []
    for text, (h, w) in ((first_text, (4, 6)), (17, (6, 4)), (5, (2, 2))):  # text, then an image of h x w patches
        ids += list(range(1, text + 1)) + [VISION_START]
        grid = types.SimpleNamespace(data=torch.tensor([1, MERGE * h, MERGE * w]))
        offset = types.SimpleNamespace(offset=len(ids))
        features.append(types.SimpleNamespace(modality="image", mm_position=offset, data={"image_grid_thw": grid}))
        ids += [IMAGE] * (h * w) + [VISION_END]
    ids += list(range(1, 41))  # the turn's text after its last image

    return config, ids, features


def new_turn(prefill: list[int], num_computed: int, mm_features: list, prompt_len: int) -> NewRequestData:
    """A request joining the batch with its first `num_computed` tokens cached, as vLLM's scheduler sends it."""
    return NewRequestData(
        req_id="turn",
        prompt_token_ids=prefill[:prompt_len],
        mm_features=mm_features,
        sampling_params=SamplingParams(temperature=1.0),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=num_computed,
        lora_request=None,
        prefill_token_ids=prefill,
    )


def hopper_model_state(model) -> HopperRolloutModelState:
    """The Hopper replicas' model state with vLLM's M-RoPE state on the GPU and what its add_request reads."""
    model_state = object.__new__(HopperRolloutModelState)
    model_state.__dict__.update(
        rope_state=RopeState(NUM_DIMS, MAX_NUM_REQS, TOKEN_BUDGET, MAX_LEN, DEVICE),
        model=model,
        mm_pruner=None,
        prompt_embeds_state=None,
        num_accepted_tokens_gpu=torch.ones(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE),
        _align_mode=False,
    )

    return model_state


def step_positions(rope_state: RopeState, req_index: int, prefill_len: int, start: int) -> torch.Tensor:
    """The M-RoPE positions vLLM's kernel gives a step of the request's tokens from start on, the prefill's rest and
    DECODE_TOKENS after it."""
    num_tokens = prefill_len - start + DECODE_TOKENS
    prefill_lens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    prefill_lens[req_index] = prefill_len
    num_computed = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    num_computed[req_index] = start

    rope_state.prepare_positions(
        torch.tensor([req_index], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, num_tokens], dtype=torch.int32, device=DEVICE),
        prefill_lens,
        num_computed,
    )

    return rope_state.get_positions(num_tokens).cpu()


class StagedTurns:
    """StepInputArrays.stage counted: the requests whose positions it staged."""

    def __init__(self, stage):
        self.stage = stage  # StepInputArrays' own
        self.count = 0

    def __call__(self, model_state, req_index: int, request) -> bool:
        staged = self.stage(model_state, req_index, request)
        self.count += staged

        return staged


@pytest.mark.parametrize("where", ["text", "image-start", "inside-image", "closing-text", "output"])
def test_a_new_turns_positions_in_every_step_equal_vllms(monkeypatch, where):
    config, ids, features = image_turn(FIRST_TEXT)
    image = features[1].mm_position.offset
    prefill = ids + list(range(1, OUTPUT + 1))  # a preempted turn sent again with its output tokens
    starts = {
        "text": StepInputArrays.MIN_PREFIX + 7,
        "image-start": image,
        "inside-image": image + 5,
        "closing-text": len(ids) - 3,
        "output": len(ids) + OUTPUT // 2,
    }
    request = new_turn(prefill, starts[where], features, prompt_len=len(ids))
    model = object.__new__(HopperQwen3_5ForConditionalGeneration)  # the Hopper replicas' model class, without weights
    object.__setattr__(model, "config", config)
    vllm, pear = hopper_model_state(model), hopper_model_state(model)
    staged_turns = StagedTurns(StepInputArrays.stage)
    monkeypatch.setattr(StepInputArrays, "stage", staged_turns)

    MambaHybridModelState.add_request(vllm, 1, request)
    pear.add_request(1, request)
    for model_state in (vllm, pear):
        model_state.rope_state.apply_staged_writes()

    start = request.num_computed_tokens
    assert staged_turns.count == 1 and pear.rope_state.prefill_delta.np[1] == vllm.rope_state.prefill_delta.np[1]
    assert torch.equal(
        step_positions(pear.rope_state, 1, len(prefill), start), step_positions(vllm.rope_state, 1, len(prefill), start)
    )
