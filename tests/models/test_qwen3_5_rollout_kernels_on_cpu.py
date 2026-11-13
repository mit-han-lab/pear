import pkgutil
import types

import flashinfer.decode as flashinfer_decode
import pytest
import tomllib
import torch
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models import ModelRegistry, qwen3_5
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends import flashinfer as vllm_flashinfer
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker import gpu_input_batch

from pear.models.kernels import RolloutKernels
from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.models.qwen3_5.rollout.blackwell.attention import SPLIT_FACTOR, AttentionStep, DecodeSplit
from pear.models.qwen3_5.rollout.blackwell.cudagraph import BlackwellMixedStepGraphs
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.cudagraph import DecodeGraphSizes
from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperMixedStepGraphs
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.prompt_tail_copy import PromptTailCopy
from pear.models.qwen3_5.rollout.rollout_kernels import Qwen3_5RolloutKernels
from pear.rollout.config import PearRolloutConfig
from repository import REPO

KERNELS = "pear.models.qwen3_5.rollout.rollout_kernels.Qwen3_5RolloutKernels"  # the recipes' rollout.kernels
PYPROJECT = REPO / "pyproject.toml"  # it names PEAR's vLLM plugin
ARCHITECTURE = VllmGpuBackend.ARCHITECTURE  # the replicas' model, which a GPU backend's model replaces
SORT_MIN_ROWS = AttentionStep.SORT_MIN_ROWS  # decode rows from which the longest contexts launch first
MIN_PREFIX = PromptTailCopy.MIN_PREFIX  # cached prompt tokens from which a new request's copy skips them
MAX_NUM_SEQS = 128  # the GB300 replicas' decode slots
TOKEN_BUDGET = 8192  # their max_num_batched_tokens
VOCAB = 248320
BLOCK = 16  # tokens per KV cache block
PROMPT = 2 * MIN_PREFIX  # a new turn's prompt
VISION_START, VISION_END, IMAGE, VIDEO = 151652, 151653, 151655, 151656  # the model's vision token ids
MERGE = 2  # the vision tower's spatial merge
# every GPU backend on a GPU it supports
GPU_BACKENDS = [
    pytest.param(DeviceCapability(10, 3), BlackwellRolloutGpuBackend, id="blackwell"),
    pytest.param(DeviceCapability(9, 0), HopperRolloutGpuBackend, id="hopper"),
]


def test_the_kernels_need_the_vllm_rollout_without_sleep():
    PearRolloutConfig(name="vllm", free_cache_engine=False, kernels=KERNELS)

    with pytest.raises(ValueError, match="rollout.name=vllm"):
        PearRolloutConfig(name="sglang", free_cache_engine=False, kernels=KERNELS)
    with pytest.raises(ValueError, match="free_cache_engine=False"):
        PearRolloutConfig(name="vllm", kernels=KERNELS)


@pytest.mark.parametrize(
    "capability, tp, architecture, gpu_backend",
    [
        (DeviceCapability(10, 0), 2, ARCHITECTURE, BlackwellRolloutGpuBackend),  # B200
        (DeviceCapability(10, 3), 2, ARCHITECTURE, BlackwellRolloutGpuBackend),  # GB300
        (DeviceCapability(9, 0), 4, ARCHITECTURE, HopperRolloutGpuBackend),  # H100
        (DeviceCapability(10, 3), 4, ARCHITECTURE, None),  # GB300 at another tensor parallelism
        (DeviceCapability(9, 0), 4, "Qwen3_5ForCausalLM", None),  # the text-only model
        (DeviceCapability(12, 0), 2, ARCHITECTURE, None),  # consumer Blackwell (no tcgen05)
        (DeviceCapability(8, 0), 4, ARCHITECTURE, None),  # Ampere
    ],
)
def test_the_replica_selects_the_gpu_backend_of_its_gpu_and_engine_config(
    monkeypatch, capability, tp, architecture, gpu_backend
):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architectures=[architecture]),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=tp),
    )

    if gpu_backend is None:
        with pytest.raises(ValueError, match="no GPU backend of rollout.kernels fits"):
            Qwen3_5RolloutKernels.select_gpu_backend(vllm_config)
    else:
        assert Qwen3_5RolloutKernels.select_gpu_backend(vllm_config) is gpu_backend
        assert Qwen3_5RolloutKernels.select_gpu_backend() is gpu_backend  # the server's choice, by the GPU alone


@pytest.mark.parametrize("capability, gpu_backend", GPU_BACKENDS)
def test_the_plugin_registers_the_model_where_the_server_switched_the_kernels_on(monkeypatch, capability, gpu_backend):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))  # the registry as it was, afterwards
    monkeypatch.setenv(RolloutKernels.ENV, "")

    Qwen3_5RolloutKernels.register()  # an worker process of a server without the kernels
    assert ModelRegistry._try_load_model_cls(ARCHITECTURE) is qwen3_5.Qwen3_5ForConditionalGeneration

    # the server's environment, inherited
    RolloutKernels.export(KERNELS).configure_server({}, MAX_NUM_SEQS, TOKEN_BUDGET)
    Qwen3_5RolloutKernels.register()
    assert ModelRegistry._try_load_model_cls(ARCHITECTURE) is gpu_backend.model_cls


def test_pyproject_s_vllm_plugin_is_the_family_s_registration():
    plugins = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]["vllm.general_plugins"]

    assert pkgutil.resolve_name(plugins["pear_qwen3_5_kernels"]) == Qwen3_5RolloutKernels.register


@pytest.mark.parametrize("capability, gpu_backend", GPU_BACKENDS)
def test_a_worker_on_vllms_own_model_fails_the_check(monkeypatch, capability, gpu_backend):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architectures=[ARCHITECTURE]),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=gpu_backend.tensor_parallel_size),
    )
    vllm_model = qwen3_5.Qwen3_5ForConditionalGeneration.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)
    gpu_backend_model = gpu_backend.model_cls.__new__(gpu_backend.model_cls)

    Qwen3_5RolloutKernels.check_model(gpu_backend_model, vllm_config)
    with pytest.raises(RuntimeError, match="vLLM plugin"):
        Qwen3_5RolloutKernels.check_model(vllm_model, vllm_config)


@pytest.fixture
def blackwell(monkeypatch):
    # vLLM's default decode sizes reach 1024 on datacenter Blackwell, 512 elsewhere
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda capability, device_id=0: capability == 100
    )


@pytest.fixture(params=[BlackwellMixedStepGraphs, HopperMixedStepGraphs], ids=["blackwell", "hopper"])
def graphs(request, monkeypatch):
    """A GPU backend's graph sizing on its platform, whose largest default graph vLLM picks."""
    on_blackwell = request.param is BlackwellMixedStepGraphs
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: on_blackwell and capability == 100,
    )
    return request.param


def engine_config(max_num_seqs: int, max_num_batched_tokens: int, mode: CUDAGraphMode, sizes=None):
    """The parts of a VllmConfig that its capture-size step reads."""
    # no sizes: the field left at its default, which vLLM infers (the field's type takes no explicit None)
    compilation_config = CompilationConfig(
        cudagraph_mode=mode, **({} if sizes is None else {"cudagraph_capture_sizes": sizes})
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=compilation_config,
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=2),
        num_speculative_tokens=0,
        performance_mode="throughput",
    )


def test_mixed_sizes_follow_the_decode_sizes():
    sizes = BlackwellMixedStepGraphs.sizes([1, 2, 4, 8, 16, 128, 256], MAX_NUM_SEQS, TOKEN_BUDGET)

    assert sizes[:7] == [1, 2, 4, 8, 16, 128, 160]
    assert [s for s in sizes if 256 < s <= 1024] == list(range(288, 1025, 32))
    assert [s for s in sizes if s > 1024] == list(range(1088, 2049, 64))


def test_mixed_sizes_stay_within_the_token_budget():
    assert BlackwellMixedStepGraphs.sizes([1, 8], MAX_NUM_SEQS, 600)[-1] == 576


# the replicas' decode slots and budget, smaller ones, a budget vLLM appends, a token budget below the decode sizes
@pytest.mark.parametrize(
    "max_num_seqs, max_num_batched_tokens", [(MAX_NUM_SEQS, TOKEN_BUDGET), (64, 8192), (600, 1000), (8, 12)]
)
def test_the_decode_sizes_are_vllms_default(graphs, max_num_seqs, max_num_batched_tokens):
    config = engine_config(max_num_seqs, max_num_batched_tokens, CUDAGraphMode.FULL_AND_PIECEWISE)

    VllmConfig._set_cudagraph_sizes(config)

    sizes = DecodeGraphSizes.default(max_num_seqs, max_num_batched_tokens, graphs.MAX_DECODE_TOKENS)
    assert sizes == config.compilation_config.cudagraph_capture_sizes


def test_the_engine_captures_mixed_steps(blackwell):
    compilation_config = {}
    BlackwellMixedStepGraphs.configure(compilation_config, MAX_NUM_SEQS, TOKEN_BUDGET)
    sizes = compilation_config["cudagraph_capture_sizes"]
    config = engine_config(MAX_NUM_SEQS, TOKEN_BUDGET, CUDAGraphMode.FULL_AND_PIECEWISE, sizes)

    VllmConfig._set_cudagraph_sizes(config)

    # vLLM's own decode sizes (up to twice max_num_seqs), then the mixed steps up to 2048 tokens
    captured = config.compilation_config.cudagraph_capture_sizes
    assert captured == sizes and captured[:5] == [1, 2, 4, 8, 16] and 256 in captured
    assert captured[-1] == config.compilation_config.max_cudagraph_capture_size == BlackwellMixedStepGraphs.MAX_TOKENS


def test_no_graphs_stay_no_graphs(blackwell):
    compilation_config = {}
    BlackwellMixedStepGraphs.configure(compilation_config, MAX_NUM_SEQS, TOKEN_BUDGET)
    config = engine_config(
        MAX_NUM_SEQS, TOKEN_BUDGET, CUDAGraphMode.NONE, compilation_config["cudagraph_capture_sizes"]
    )

    VllmConfig._set_cudagraph_sizes(config)

    assert not config.compilation_config.cudagraph_capture_sizes


def test_the_hopper_engine_captures_the_mixed_step_buckets():
    compilation_config = {}
    HopperMixedStepGraphs.configure(compilation_config, MAX_NUM_SEQS, TOKEN_BUDGET)
    sizes = compilation_config["cudagraph_capture_sizes"]
    config = engine_config(MAX_NUM_SEQS, TOKEN_BUDGET, CUDAGraphMode.FULL_AND_PIECEWISE, sizes)

    VllmConfig._set_cudagraph_sizes(config)

    # vLLM's own decode sizes, then the mixed-step buckets up to MAX_TOKENS: the largest graph the attention builder's
    # graph-size test compares against
    captured = config.compilation_config.cudagraph_capture_sizes
    decode_sizes = DecodeGraphSizes.default(MAX_NUM_SEQS, TOKEN_BUDGET, HopperMixedStepGraphs.MAX_DECODE_TOKENS)
    assert captured == sizes and set(captured) == {*decode_sizes, *HopperMixedStepGraphs.BUCKETS}
    assert captured[-1] == config.compilation_config.max_cudagraph_capture_size == HopperMixedStepGraphs.MAX_TOKENS


def mixed_step_dispatcher(sizes: list[int], max_num_seqs: int) -> CudagraphDispatcher:
    """vLLM's graph dispatcher of a FULL_AND_PIECEWISE engine with these capture sizes, its keys initialized."""
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=sizes,
        max_cudagraph_capture_size=max(sizes),
        splitting_ops=list(CompilationConfig._attention_ops),
    )
    config = types.SimpleNamespace(
        compilation_config=compilation_config,
        num_speculative_tokens=0,
        lora_config=None,
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_AND_PIECEWISE, 1)
    return dispatcher


def test_the_mixed_steps_replay_full_graphs_beside_the_decode_ones():
    max_num_seqs = 64
    sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 512, 1024, 2048]
    dispatcher = mixed_step_dispatcher(sizes, max_num_seqs)

    HopperMixedStepGraphs.add_mixed_keys(dispatcher)

    # a pure decode step on its decode graph, a mixed one on its FULL graph up to MAX_TOKENS, piecewise above
    decode_mode, decode = dispatcher.dispatch(num_tokens=40, uniform_decode=True)
    assert decode_mode == CUDAGraphMode.FULL and decode.uniform and decode.num_tokens == decode.num_reqs == 48
    for tokens, padded in ((20, 24), (40, 48), (1000, 1024)):
        mixed_mode, mixed = dispatcher.dispatch(num_tokens=tokens)
        assert mixed_mode == CUDAGraphMode.FULL and not mixed.uniform and mixed.num_tokens == padded
        assert mixed.num_reqs == min(padded, max_num_seqs)
    assert dispatcher.dispatch(num_tokens=1500)[0] == CUDAGraphMode.PIECEWISE
    assert {key.num_tokens for key in dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]} == {2048}


def test_the_decode_attention_splits_over_more_ctas(monkeypatch):
    calls = []

    def decode(query, **kwargs):
        calls.append((flashinfer_decode.get_device_sm_count(query.device), kwargs["multi_ctas_kv_counter_buffer"]))

    monkeypatch.setattr(vllm_flashinfer, "trtllm_batch_decode_with_kv_cache", decode)
    monkeypatch.setattr(flashinfer_decode, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(DecodeSplit, "counters", {})
    DecodeSplit.install()
    query = torch.zeros(4, 8)

    vllm_flashinfer.trtllm_batch_decode_with_kv_cache(query=query)
    vllm_flashinfer.trtllm_batch_decode_with_kv_cache(query=query)

    assert [sms for sms, _ in calls] == [148 * SPLIT_FACTOR] * 2
    assert calls[0][1] is calls[1][1]  # one semaphore buffer, kept
    assert flashinfer_decode.get_device_sm_count(query.device) == 148  # restored after each call


def test_the_longest_decode_contexts_launch_first(monkeypatch):
    monkeypatch.setattr(vllm_flashinfer, "_get_trtllm_workspace_buffer", lambda: torch.zeros(16, dtype=torch.uint8))
    lens = torch.randperm(SORT_MIN_ROWS) + 1
    decode = types.SimpleNamespace(block_tables=lens[:, None] * 10, seq_lens=lens, max_seq_len=SORT_MIN_ROWS)
    md = types.SimpleNamespace(
        num_decode_tokens=SORT_MIN_ROWS, num_actual_tokens=SORT_MIN_ROWS, num_prefills=0, prefill=None, decode=decode
    )

    step = AttentionStep(md)

    tables, seq_lens, _ = step.decode_args
    assert seq_lens.tolist() == sorted(lens.tolist(), reverse=True)
    assert torch.equal(tables[:, 0], seq_lens * 10)  # the rows move with their lengths
    md.num_decode_tokens = md.num_actual_tokens = SORT_MIN_ROWS - 1
    assert AttentionStep(md).order is None  # fewer rows keep their order


@pytest.fixture
def input_batch(monkeypatch):
    # every test leaves vLLM's input batch as it found it
    monkeypatch.setattr(gpu_input_batch.InputBatch, "add_request", gpu_input_batch.InputBatch.add_request)
    # the batch's host buffers are pinned where vLLM's platform allows it; a node without a driver cannot pin
    monkeypatch.setattr(gpu_input_batch, "PIN_MEMORY", False)


def new_turn(req_id: str, prompt: list[int], num_computed: int, params: SamplingParams):
    """A request joining the batch with its first `num_computed` prompt tokens cached."""
    return gpu_input_batch.CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=([],),
        num_computed_tokens=num_computed,
        output_token_ids=[],
    )


def test_a_new_turn_copies_its_uncomputed_tail(input_batch):
    PromptTailCopy.install(types.SimpleNamespace(speculative_config=None))
    batch = gpu_input_batch.InputBatch(
        max_num_reqs=2,
        max_model_len=PROMPT,
        max_num_batched_tokens=TOKEN_BUDGET,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        max_num_blocks_per_req=[PROMPT // BLOCK],
    )
    prompt = list(range(1, PROMPT + 1))
    num_computed = MIN_PREFIX + 100
    tail_request = new_turn("tail", prompt, num_computed, SamplingParams(temperature=1.0))
    penalty_request = new_turn("penalty", prompt, num_computed, SamplingParams(temperature=1.0, repetition_penalty=1.1))

    batch.add_request(tail_request)
    batch.add_request(penalty_request)

    assert tail_request.prompt_token_ids is prompt  # the request keeps its list
    rows = batch.token_ids_cpu[:, :PROMPT]
    assert not rows[0, :num_computed].any() and rows[0, num_computed:].tolist() == prompt[num_computed:]
    assert rows[1].tolist() == prompt  # penalties read the prompt: vLLM's copy
    assert batch.num_prompt_tokens[:2].tolist() == [PROMPT, PROMPT]


def test_a_new_turns_positions_from_its_first_uncomputed_token_equal_vllms():
    config = types.SimpleNamespace(
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        vision_config=types.SimpleNamespace(spatial_merge_size=MERGE),
    )
    model = object.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)  # the replicas' model class, without weights
    object.__setattr__(model, "config", config)
    ids, features = [], []
    for text, (h, w) in ((300, (4, 6)), (17, (6, 4)), (5, (2, 2))):  # text, then an image of h x w merged patches
        ids += list(range(1, text + 1)) + [VISION_START]
        grid = types.SimpleNamespace(data=torch.tensor([1, MERGE * h, MERGE * w]))
        offset = types.SimpleNamespace(offset=len(ids))
        features.append(types.SimpleNamespace(modality="image", mm_position=offset, data={"image_grid_thw": grid}))
        ids += [IMAGE] * (h * w) + [VISION_END]
    ids += list(range(1, 41))  # the turn's text after its last image
    want, want_delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(ids, features, config)

    # the first uncomputed token in the text, at an image's first token, inside an image, in the closing text
    image = features[1].mm_position.offset
    for start in (100, image, image + 5, len(ids) - 3):
        positions, delta = StepInputArrays.tail_positions(model, ids, features, start)

        assert torch.equal(positions[:, start:], want[:, start:]) and delta == want_delta
