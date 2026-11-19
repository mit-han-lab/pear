import inspect
import pathlib
import pkgutil
import subprocess
import sys
import types
import weakref

import flashinfer.decode as flashinfer_decode
import numpy as np
import pytest
import tomllib
import torch
import vllm
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.model_executor.layers import fused_qk_norm_rope
from vllm.model_executor.layers.mamba.abstract import get_mamba_prefill_checkpoint_alignment
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.models import ModelRegistry, qwen3_next
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.models import qwen3_5
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends import flashinfer as vllm_flashinfer
from vllm.v1.attention.backends.flashinfer import FlashInferImpl
from vllm.v1.attention.backends.gdn_attn import FLASHINFER_GDN_CHECKPOINT_ALIGNMENT
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_states import init_model_state, resolve_model_state_cls
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from pear.models.kernels import RolloutKernels
from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.models.qwen3_5.rollout import model as rollout_model
from pear.models.qwen3_5.rollout.attention import Qwen3_5RolloutAttention
from pear.models.qwen3_5.rollout.blackwell.attention import (
    SPLIT_FACTOR,
    AttentionStep,
    BlackwellFlashInferImpl,
    DecodeSplit,
)
from pear.models.qwen3_5.rollout.blackwell.cudagraph import BlackwellMixedStepGraphs
from pear.models.qwen3_5.rollout.blackwell.gdn import BlackwellGDNAttention, GDNCore
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.blackwell.model_state import BlackwellRolloutModelState
from pear.models.qwen3_5.rollout.blackwell.sampler import BlackwellSampler
from pear.models.qwen3_5.rollout.gdn import RolloutGDNMetadataBuilder
from pear.models.qwen3_5.rollout.hopper.attention import HEAD_DIM, QUERY_HEADS, AttentionStepBuffers
from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperCudaGraphManager, HopperMixedStepGraphs
from pear.models.qwen3_5.rollout.hopper.gdn import HopperGDNAttention
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.model import HopperQwen3_5ForConditionalGeneration
from pear.models.qwen3_5.rollout.hopper.model_state import HopperRolloutModelState
from pear.models.qwen3_5.rollout.hopper.sampler import HopperSampler
from pear.models.qwen3_5.rollout.hopper.staging import StagedBuffer
from pear.models.qwen3_5.rollout.hopper.step_input_arrays import StepInputArrays
from pear.models.qwen3_5.rollout.mlp import Qwen3_5RolloutMLP
from pear.models.qwen3_5.rollout.model import Qwen3_5RolloutDecoderLayer
from pear.models.qwen3_5.rollout.rollout_kernels import Qwen3_5RolloutKernels
from pear.models.qwen3_5.rollout.sampler import RolloutSampler
from pear.rollout.config import PearRolloutConfig
from repository import REPO

KERNELS = "pear.models.qwen3_5.rollout.rollout_kernels.Qwen3_5RolloutKernels"  # the recipes' rollout.kernels
PYPROJECT = REPO / "pyproject.toml"  # it names PEAR's vLLM plugin
ARCHITECTURE = VllmGpuBackend.ARCHITECTURE  # the replicas' model, which a GPU backend's model replaces
SORT_MIN_ROWS = AttentionStep.SORT_MIN_ROWS  # decode rows from which the longest contexts launch first
MAX_NUM_SEQS = 128  # the replicas' decode slots
TOKEN_BUDGET = 8192  # their max_num_batched_tokens
DECODE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]  # request counts of vLLM's decode graphs up to MAX_NUM_SEQS
# the token-strided capture sizes of 128 and of 64 decode slots: 1, 2, 4, every 8 below 256, then every 16, up to twice
# the slots (vLLM's VllmConfig._set_cudagraph_sizes)
STRIDED_SIZES = {128: [1, 2, 4, *range(8, 256, 8), 256], 64: [1, 2, 4, *range(8, 129, 8)]}
VOCAB = 248320
NO_LOGPROBS = -1  # vLLM's max_num_logprobs of a batch without log-probs
VISION_START, VISION_END, IMAGE, VIDEO = 151652, 151653, 151655, 151656  # the model's vision token ids
MERGE = 2  # the vision tower's spatial merge
NUM_DIMS = 3  # Qwen3-VL's M-RoPE sections
# every GPU backend on a GPU it supports, its model state and sampler
GPU_BACKENDS = [
    pytest.param(DeviceCapability(10, 3), BlackwellRolloutGpuBackend, id="blackwell"),
    pytest.param(DeviceCapability(9, 0), HopperRolloutGpuBackend, id="hopper"),
]
MODEL_STATES = [
    pytest.param(BlackwellRolloutGpuBackend, BlackwellRolloutModelState, BlackwellSampler, id="blackwell"),
    pytest.param(HopperRolloutGpuBackend, HopperRolloutModelState, HopperSampler, id="hopper"),
]
# the methods of vLLM's V2 sampler the samplers override: vLLM calls the others itself, on the batches the kernels
# leave to it (its __call__ samples them through its sample)
SAMPLER_OVERRIDES = {"__init__", "add_request", "__call__"}
# Qwen3.8-27B's Gated DeltaNet: its key and value heads (all ranks'), their head dim
GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD_DIM = 16, 48, 128
KV_HEADS = 1  # its attention's KV heads on one of an H100 replica's four ranks


def test_the_kernels_need_the_vllm_rollout_without_sleep():
    PearRolloutConfig(name="vllm", free_cache_engine=False, kernels=KERNELS)

    with pytest.raises(ValueError, match="rollout.name=vllm"):
        PearRolloutConfig(name="sglang", free_cache_engine=False, kernels=KERNELS)
    with pytest.raises(ValueError, match="free_cache_engine=False"):
        PearRolloutConfig(name="vllm", kernels=KERNELS)


def engine_config(tp: int, architecture: str = ARCHITECTURE) -> types.SimpleNamespace:
    """The parts of a replica's VllmConfig the GPU backends' selection reads, as vLLM resolves them for PEAR's runs."""
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(architectures=[architecture], dtype=torch.bfloat16),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=1, enable_batch_sharded_sampling=False
        ),
        # the checkpoint's mamba_ssm_dtype, which vLLM takes for the recurrent states
        cache_config=types.SimpleNamespace(mamba_cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        lora_config=None,
        speculative_config=None,
    )


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
    vllm_config = engine_config(tp, architecture)

    if gpu_backend is None:
        with pytest.raises(ValueError, match="no GPU backend of rollout.kernels fits"):
            Qwen3_5RolloutKernels.select_gpu_backend(vllm_config)
    else:
        assert Qwen3_5RolloutKernels.select_gpu_backend(vllm_config) is gpu_backend
        assert Qwen3_5RolloutKernels.select_gpu_backend() is gpu_backend  # the server's choice, by the GPU alone


@pytest.mark.parametrize("capability, gpu_backend", GPU_BACKENDS)
@pytest.mark.parametrize(
    "section, field, value, reason",
    [
        ("parallel_config", "pipeline_parallel_size", 2, "one pipeline stage"),
        ("parallel_config", "enable_batch_sharded_sampling", True, "batch-sharded sampling"),
        ("cache_config", "mamba_ssm_cache_dtype", "auto", "states of"),  # bf16 recurrent states
        (None, "lora_config", object(), "LoRA"),
        (None, "speculative_config", object(), "without a drafter"),
    ],
)
def test_the_gpu_backends_refuse_what_their_kernels_do_not_compute(
    monkeypatch, capability, gpu_backend, section, field, value, reason
):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    vllm_config = engine_config(gpu_backend.tensor_parallel_size)
    setattr(getattr(vllm_config, section) if section else vllm_config, field, value)

    with pytest.raises(ValueError, match=reason):
        Qwen3_5RolloutKernels.select_gpu_backend(vllm_config)


@pytest.mark.parametrize("capability, gpu_backend", GPU_BACKENDS)
def test_the_plugin_registers_the_model_where_the_server_switched_the_kernels_on(monkeypatch, capability, gpu_backend):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))  # the registry as it was, afterwards
    monkeypatch.setenv(RolloutKernels.ENV, "")

    Qwen3_5RolloutKernels.register()  # a worker process of a server without the kernels
    assert ModelRegistry._try_load_model_cls(ARCHITECTURE) is qwen3_5.Qwen3_5ForConditionalGeneration

    # the server's environment, inherited
    RolloutKernels.export(KERNELS).configure_server({}, MAX_NUM_SEQS, TOKEN_BUDGET)
    Qwen3_5RolloutKernels.register()
    assert ModelRegistry._try_load_model_cls(ARCHITECTURE) is gpu_backend.model_cls


def test_pyproject_s_vllm_plugin_is_the_family_s_registration():
    plugins = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]["vllm.general_plugins"]

    assert pkgutil.resolve_name(plugins["pear_qwen3_5_kernels"]) == Qwen3_5RolloutKernels.register


def test_the_plugin_s_module_imports_no_gpu_backend():
    # vLLM loads the plugin in every process of an installation, the model server's older vLLM too: the backends,
    # written against this vLLM, are imported only where the kernels are on
    backend_packages = ("pear.models.qwen3_5.rollout.blackwell", "pear.models.qwen3_5.rollout.hopper")
    code = (
        "import sys; import pear.models.qwen3_5.rollout.rollout_kernels; "
        f"assert not [m for m in sys.modules if m.startswith({backend_packages!r})]"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("capability, gpu_backend", GPU_BACKENDS)
def test_a_worker_on_vllms_own_model_fails_the_check(monkeypatch, capability, gpu_backend):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    vllm_config = engine_config(gpu_backend.tensor_parallel_size)
    vllm_model = qwen3_5.Qwen3_5ForConditionalGeneration.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)
    gpu_backend_model = gpu_backend.model_cls.__new__(gpu_backend.model_cls)

    Qwen3_5RolloutKernels.check_model(gpu_backend_model, vllm_config)
    with pytest.raises(RuntimeError, match="vLLM plugin"):
        Qwen3_5RolloutKernels.check_model(vllm_model, vllm_config)


@pytest.mark.parametrize("gpu_backend, model_state_cls, sampler_cls", MODEL_STATES)
def test_vllms_runner_steps_the_model_with_the_gpu_backend_s_model_state(gpu_backend, model_state_cls, sampler_cls):
    model = gpu_backend.model_cls.__new__(gpu_backend.model_cls)

    assert resolve_model_state_cls(None, model) is model_state_cls
    assert issubclass(model_state_cls, MambaHybridModelState) and model_state_cls.sampler_cls is sampler_cls


def test_vllm_refuses_prompt_embeddings_on_the_blackwell_replicas():
    # the first decoder layer sums the ranks' partial embedding rows; a prompt embedding is a whole row on each rank
    vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(enable_prompt_embeds=True))
    model = BlackwellRolloutGpuBackend.model_cls.__new__(BlackwellRolloutGpuBackend.model_cls)

    with pytest.raises(ValueError, match="enable-prompt-embeds"):
        init_model_state(vllm_config, model, None, torch.device("cpu"))


def vllm_sampler(**options) -> Sampler:
    """vLLM's V2 sampler with a replica's options, without its request states (the hooks read only these)."""
    sampler = Sampler.__new__(Sampler)
    sampler.__dict__.update(
        logprobs_mode="processed_logprobs",
        use_fp64_gumbel=False,
        return_sampling_mask=False,
        trace_replay_state=None,
        needs_logits_processing=np.zeros(MAX_NUM_SEQS, dtype=bool),
        may_mask_logits=np.zeros(MAX_NUM_SEQS, dtype=bool),
    )
    sampler.__dict__.update(options)

    return sampler


@pytest.mark.parametrize("gpu_backend, model_state_cls, sampler_cls", MODEL_STATES)
def test_the_model_state_samples_with_the_gpu_backend_s_sampler(gpu_backend, model_state_cls, sampler_cls):
    model_state = model_state_cls.__new__(model_state_cls)
    vllm_sampler_of_the_runner = vllm_sampler()

    sampler, rejection_sampler = model_state.custom_sampler(vllm_sampler_of_the_runner)

    assert type(sampler) is sampler_cls and rejection_sampler is None
    assert sampler.needs_logits_processing is vllm_sampler_of_the_runner.needs_logits_processing  # vLLM's state
    # vLLM's sampler stays for the engine options the kernels do not compute
    for options in ({"logprobs_mode": "raw_logprobs"}, {"use_fp64_gumbel": True}, {"return_sampling_mask": True}):
        assert model_state.custom_sampler(vllm_sampler(**options)) is None


@pytest.mark.parametrize(
    "params, may_mask_logits, is_eligible",
    [
        (SamplingParams(temperature=1.0, logprobs=1), False, True),  # a replica's request
        (SamplingParams(temperature=0.6, logprobs=1), False, True),  # another temperature
        (SamplingParams(temperature=0.0, logprobs=1), False, False),  # greedy
        (SamplingParams(temperature=1.0, seed=7), False, False),  # a seed of its own
        (SamplingParams(temperature=1.0, logprob_token_ids=[3]), False, False),  # log-probs of chosen tokens
        (SamplingParams(temperature=1.0, top_p=0.9), True, False),  # logits vLLM processes
    ],
)
def test_a_request_is_the_kernels_when_vllm_leaves_its_logits_to_the_temperature(
    monkeypatch, params, may_mask_logits, is_eligible
):
    # vLLM's bookkeeping, reduced to its verdict on the request's logits
    def vllm_add_request(self, req_idx, sampling_params):
        self.may_mask_logits[req_idx] = may_mask_logits

    monkeypatch.setattr(Sampler, "add_request", vllm_add_request)
    sampler = BlackwellSampler(vllm_sampler())

    sampler.add_request(3, params)

    assert sampler.is_eligible[3] == is_eligible


@pytest.mark.parametrize(
    "ineligible_row, num_logprobs, dtype, vocab",
    [
        (2, 1, torch.float32, VOCAB),  # a row vLLM processes
        (None, NO_LOGPROBS, torch.float32, VOCAB),  # no log-probs
        (None, BlackwellSampler.MAX_LOGPROBS + 1, torch.float32, VOCAB),  # more log-probs than the kernels keep
        (None, 1, torch.bfloat16, VOCAB),  # bf16 logits
        (None, 1, torch.float32, VOCAB + 1),  # rows the kernels cannot read as float4
    ],
)
def test_vllm_samples_the_batches_the_kernels_do_not_take(monkeypatch, ineligible_row, num_logprobs, dtype, vocab):
    monkeypatch.setattr(Sampler, "__call__", lambda self, logits, input_batch: "vllm")
    sampler = BlackwellSampler(vllm_sampler())
    sampler.is_eligible[:4] = True
    if ineligible_row is not None:
        sampler.is_eligible[ineligible_row] = False
    sampler.sampling_states = types.SimpleNamespace(max_num_logprobs=lambda idx_mapping_np: num_logprobs)
    input_batch = types.SimpleNamespace(idx_mapping_np=np.arange(4))

    assert sampler(torch.zeros(4, vocab, dtype=dtype), input_batch) == "vllm"


@pytest.mark.parametrize("sampler_cls", [RolloutSampler, BlackwellSampler, HopperSampler])
def test_the_samplers_override_no_other_method_of_vllms_sampler(sampler_cls):
    vllm_methods = {
        name
        for name, value in vars(Sampler).items()
        if callable(value) or isinstance(value, (classmethod, staticmethod, property))
    }

    overridden = (vllm_methods & set(vars(sampler_cls))) - SAMPLER_OVERRIDES
    assert not overridden, f"{sampler_cls.__name__} overrides vLLM's Sampler.{sorted(overridden)}"


def test_the_blackwell_mixed_steps_take_a_graph_every_32_tokens_then_every_64():
    assert BlackwellMixedStepGraphs.mixed_sizes(2048) == [*range(32, 1025, 32), *range(1088, 2049, 64)]
    assert BlackwellMixedStepGraphs.mixed_sizes(600) == list(range(32, 577, 32))


@pytest.mark.parametrize("max_num_seqs", sorted(STRIDED_SIZES))
@pytest.mark.parametrize("max_num_batched_tokens", [TOKEN_BUDGET, 600])
def test_the_blackwell_server_sets_the_strided_sizes_beside_the_mixed_steps_and_for_the_decode_steps(
    max_num_seqs, max_num_batched_tokens
):
    graphs = BlackwellMixedStepGraphs
    compilation_config = {}

    BlackwellRolloutGpuBackend.configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    # the mixed steps: the strided sizes, then every FINE_STEP tokens from the first multiple above max_num_seqs up to
    # COARSE_FROM, every COARSE_STEP above; the decode steps: the strided sizes of at most max_num_seqs requests
    limit = min(graphs.MAX_TOKENS, max_num_batched_tokens)
    first_mixed_size = (max_num_seqs // graphs.FINE_STEP + 1) * graphs.FINE_STEP
    mixed_sizes = [
        *range(first_mixed_size, min(limit, graphs.COARSE_FROM) + 1, graphs.FINE_STEP),
        *range(graphs.COARSE_FROM, limit + 1, graphs.COARSE_STEP),
    ]
    strided_sizes = STRIDED_SIZES[max_num_seqs]
    config = CompilationConfig(**compilation_config)  # as vLLM's engine args build it
    assert config.prefill_cudagraph_capture_token_sizes == sorted({*strided_sizes, *mixed_sizes})
    assert config.decode_cudagraph_capture_request_sizes == [size for size in strided_sizes if size <= max_num_seqs]


@pytest.mark.parametrize("max_num_seqs", sorted(STRIDED_SIZES))
@pytest.mark.parametrize("max_num_batched_tokens", [TOKEN_BUDGET, 600])
def test_the_hopper_server_sets_the_strided_sizes_beside_the_buckets(max_num_seqs, max_num_batched_tokens):
    compilation_config = {}

    HopperRolloutGpuBackend.configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    # every one a FULL graph (HopperCudaGraphManager); the decode graphs vLLM's, which it takes from the strided sizes
    limit = min(HopperMixedStepGraphs.MAX_TOKENS, max_num_batched_tokens)
    buckets = [size for size in HopperMixedStepGraphs.BUCKETS if size <= limit]
    config = CompilationConfig(**compilation_config)
    assert config.prefill_cudagraph_capture_token_sizes == sorted({*STRIDED_SIZES[max_num_seqs], *buckets})
    assert config.decode_cudagraph_capture_request_sizes is None


def attention_groups(cg_support: AttentionCGSupport) -> list[list[types.SimpleNamespace]]:
    """A KV cache group whose metadata builder takes CUDA graphs as cg_support."""
    builder = types.SimpleNamespace(get_cudagraph_support=lambda vllm_config, kv_cache_spec: cg_support)
    group = types.SimpleNamespace(
        layer_names=["layer"],
        kv_cache_spec=None,
        backend=types.SimpleNamespace(__name__="Backend"),
        get_metadata_builder=lambda ubatch_idx=0: builder,
    )
    return [[group]]


def graph_manager(
    manager_cls: type[ModelCudaGraphManager], gpu_backend, mode: CUDAGraphMode, decode_sizes: list[int] | None = None
) -> ModelCudaGraphManager:
    """A replica's graph manager with the descriptors vLLM's constructor builds for the sizes its GPU backend sets;
    decode_sizes stand for vLLM's own where the backend sets none."""
    compilation_config = {}
    gpu_backend.configure(compilation_config, MAX_NUM_SEQS, TOKEN_BUDGET)
    prefill_sizes = compilation_config["prefill_cudagraph_capture_token_sizes"]

    manager = manager_cls.__new__(manager_cls)
    manager.vllm_config = types.SimpleNamespace(uniform_decode_query_len=1, cache_config=CacheConfig())
    manager.compilation_config = types.SimpleNamespace(
        prefill_cudagraph_capture_token_sizes=prefill_sizes,
        decode_cudagraph_capture_request_sizes=compilation_config.get(
            "decode_cudagraph_capture_request_sizes", decode_sizes
        ),
        max_cudagraph_capture_size=max(prefill_sizes),
        max_decode_cudagraph_capture_tokens=None,
    )
    manager.cudagraph_mode = mode
    manager.max_num_reqs = MAX_NUM_SEQS
    manager.decode_request_sizes = None
    manager.decode_max_reqs = {1: MAX_NUM_SEQS}
    manager.decode_query_len = 1
    manager.varlen_decode = False
    manager.ubatch_runner = None
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map, manager._max_lora_case = manager._build_lora_dispatch_map()
    manager._candidates, manager._decode_descs, manager._capture_descs = {}, {}, {}
    manager._init_candidates()

    return manager


def test_the_hopper_mixed_steps_replay_full_graphs_beside_the_decode_ones():
    manager = graph_manager(
        HopperCudaGraphManager, HopperRolloutGpuBackend, CUDAGraphMode.FULL_AND_PIECEWISE, DECODE_SIZES
    )

    manager.make_mixed_graphs_full(attention_groups(AttentionCGSupport.ALWAYS))
    manager._graphs_captured = True

    # a pure decode step on its decode graph
    decode = manager.dispatch(40, 40, 1, 0)
    assert decode.cg_mode == CUDAGraphMode.FULL and decode.uniform_token_count == 1 and decode.num_reqs == 48
    # a mixed step on the FULL graph of the next capture size (a strided size, then a bucket), of as many requests as
    # tokens up to the engine's
    for num_reqs, num_tokens, padded in ((3, 20, 24), (40, 100, 104), (MAX_NUM_SEQS, 1000, 1008)):
        mixed = manager.dispatch(num_reqs, num_tokens, None, 0)
        assert mixed.cg_mode == CUDAGraphMode.FULL and mixed.uniform_token_count is None
        assert mixed.num_tokens == padded and mixed.num_reqs == min(padded, MAX_NUM_SEQS)
    # above the largest bucket, without a graph
    assert manager.dispatch(MAX_NUM_SEQS, 1500, None, 0).cg_mode == CUDAGraphMode.NONE
    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    mixed_sizes = sorted(d.num_tokens for d in full_descs if d.uniform_token_count is None)
    assert CUDAGraphMode.PIECEWISE not in manager._capture_descs
    assert mixed_sizes == sorted({*STRIDED_SIZES[MAX_NUM_SEQS], *HopperMixedStepGraphs.BUCKETS})


def test_the_blackwell_steps_pad_to_the_strided_sizes():
    manager = graph_manager(ModelCudaGraphManager, BlackwellRolloutGpuBackend, CUDAGraphMode.FULL_AND_PIECEWISE)
    manager._graphs_captured = True

    # a decode step on the FULL decode graph of the next strided size, a mixed step on the piecewise graph of the next
    # one
    decode = manager.dispatch(41, 41, 1, 0)
    assert decode.cg_mode == CUDAGraphMode.FULL and decode.uniform_token_count == 1 and decode.num_reqs == 48
    mixed = manager.dispatch(3, 20, None, 0)
    assert mixed.cg_mode == CUDAGraphMode.PIECEWISE and mixed.num_tokens == 24
    decode_sizes = sorted(d.num_reqs for d in manager._capture_descs[CUDAGraphMode.FULL])
    assert decode_sizes == [size for size in STRIDED_SIZES[MAX_NUM_SEQS] if size <= MAX_NUM_SEQS]


@pytest.mark.parametrize(
    "mode, decode_sizes, cg_support, error",
    [
        # every step on a token-keyed graph: a decode step would replay a mixed graph's stale layout
        (CUDAGraphMode.FULL, DECODE_SIZES, AttentionCGSupport.ALWAYS, "cudagraph_mode is FULL"),
        (CUDAGraphMode.PIECEWISE, DECODE_SIZES, AttentionCGSupport.ALWAYS, "cudagraph_mode is PIECEWISE"),
        # decode steps of more requests than the decode graphs take
        (CUDAGraphMode.FULL_AND_PIECEWISE, [1, 2, 4, 64], AttentionCGSupport.ALWAYS, "graphs of up to 64 requests"),
        # an attention backend whose metadata a FULL mixed graph would replay stale
        (CUDAGraphMode.FULL_AND_PIECEWISE, DECODE_SIZES, AttentionCGSupport.UNIFORM_BATCH, "UNIFORM_BATCH"),
    ],
)
def test_the_hopper_graphs_refuse_what_would_replay_a_stale_step(mode, decode_sizes, cg_support, error):
    manager = graph_manager(HopperCudaGraphManager, HopperRolloutGpuBackend, mode, decode_sizes)

    with pytest.raises(RuntimeError, match=error):
        manager.make_mixed_graphs_full(attention_groups(cg_support))


def test_the_hopper_graphs_ask_the_attention_backends_under_the_engine_s_config():
    # vLLM's memory profiling captures outside the engine's config, which FlashInfer's builder reads for its answer
    manager = graph_manager(
        HopperCudaGraphManager, HopperRolloutGpuBackend, CUDAGraphMode.FULL_AND_PIECEWISE, DECODE_SIZES
    )
    groups = attention_groups(AttentionCGSupport.UNIFORM_BATCH)
    builder = groups[0][0].get_metadata_builder()
    builder.get_cudagraph_support = lambda vllm_config, kv_cache_spec: (
        AttentionCGSupport.UNIFORM_BATCH if get_current_vllm_config() is manager.vllm_config else None
    )

    with pytest.raises(RuntimeError, match="UNIFORM_BATCH"):
        manager.make_mixed_graphs_full(groups)


def common_metadata(query_lens: list[int], seq_lens: list[int], is_prefilling: list[bool]) -> CommonAttentionMetadata:
    """A step's common attention metadata on the host, as vLLM's V2 runner gives it to a GDN group's builder."""
    qsl = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    num_reqs = len(query_lens)

    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq,
        num_reqs=num_reqs,
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.arange(1, num_reqs + 1, dtype=torch.int32)[:, None],
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64),
        is_prefilling=torch.tensor(is_prefilling),
        seq_lens_cpu_upper_bound=seq,
    )


@pytest.mark.parametrize(
    "query_lens, seq_lens, is_prefilling, num_decodes",
    [
        ([1, 1, 300], [50, 9, 300], [False, False, True], 2),  # decode rows, then a prompt's chunk
        ([1, 1, 1], [50, 1, 9], [False, True, False], 1),  # a one-token prompt: no state before it
        ([1, 1], [50, 700], [False, True], 2),  # a prompt's last token over its cached prefix
        ([1, 1, 1], [1, 1, 1], [False, False, False], 3),  # a decode graph's capture
        ([20, 1], [20, 5], [True, False], 0),  # a prompt first
        ([1, 1, 0], [9, 9, 0], [False, False, False], 3),  # a FULL graph's padding requests
    ],
)
def test_the_gdn_builders_count_the_decode_rows_as_vllms(query_lens, seq_lens, is_prefilling, num_decodes):
    m = common_metadata(query_lens, seq_lens, is_prefilling)

    # vLLM's GDN builder: a first chunk (no state before it) is a prefill row, whatever its length
    no_prior_state = (m.query_start_loc_cpu.diff() > 0) & (m.seq_lens_cpu_upper_bound <= m.query_start_loc_cpu.diff())
    vllm_split = split_decodes_and_prefills(
        m.replace(is_prefilling=no_prior_state & m.is_prefilling),
        decode_threshold=1,
        treat_short_extends_as_decodes=False,
    )
    assert RolloutGDNMetadataBuilder.num_decodes(m) == vllm_split[0] == num_decodes


def engine_of_layer(gdn_layer) -> types.SimpleNamespace:
    """The parts of a VllmConfig through which vLLM finds its Mamba layers: one GDN layer."""
    compilation_config = types.SimpleNamespace(
        static_forward_context={"language_model.model.layers.0.linear_attn": gdn_layer}
    )

    return types.SimpleNamespace(compilation_config=compilation_config)


@pytest.mark.parametrize("layer_cls", [BlackwellGDNAttention, HopperGDNAttention])
def test_the_gdn_layers_take_no_prefill_checkpoints(layer_cls):
    # vLLM's layer with FlashInfer's prefill in align mode: the scheduler stops splitting prefills at block boundaries
    vllm_layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    vllm_layer._checkpoint_alignment = FLASHINFER_GDN_CHECKPOINT_ALIGNMENT
    layer = layer_cls.__new__(layer_cls)
    layer._checkpoint_alignment = FLASHINFER_GDN_CHECKPOINT_ALIGNMENT

    # the backends' prefill kernels write a sequence's final state only: the scheduler splits every prefill there
    assert get_mamba_prefill_checkpoint_alignment(engine_of_layer(vllm_layer)) == FLASHINFER_GDN_CHECKPOINT_ALIGNMENT
    assert get_mamba_prefill_checkpoint_alignment(engine_of_layer(layer)) is None


@pytest.mark.parametrize("layer_cls", [BlackwellGDNAttention, HopperGDNAttention])
@pytest.mark.parametrize(
    "decode_kernel, activations_dtype, norm_dtype, computes",
    [
        ("cuda", torch.bfloat16, torch.float32, True),  # a replica's layer
        ("cuda", torch.bfloat16, torch.bfloat16, True),
        ("triton", torch.bfloat16, torch.float32, False),  # VLLM_GDN_DECODE_KERNEL=triton, or the CUDA kernel unbuilt
        ("cuda", torch.float16, torch.float32, False),
        ("cuda", torch.bfloat16, torch.float16, False),
    ],
)
def test_the_backends_compute_the_gdn_core_on_vllm_s_fused_decode_path_only(
    layer_cls, decode_kernel, activations_dtype, norm_dtype, computes
):
    layer = layer_cls.__new__(layer_cls)
    layer.gdn_decode_kernel = decode_kernel  # as vLLM's layer resolves it when it is built
    layer.norm = types.SimpleNamespace(weight=torch.zeros(1, dtype=norm_dtype))

    assert layer.computes_core(torch.zeros(1, dtype=activations_dtype)) is computes


def test_the_decoder_layer_builds_the_rollout_attention_and_mlp(monkeypatch):
    # their own constructors over vLLM's, reduced to what vLLM leaves (an unfused output gate) for want of an engine;
    # the layer's norms and M-RoPE factor are vLLM's, which need one too
    def unfused_attention(layer, *args, **kwargs):
        torch.nn.Module.__init__(layer)
        layer.use_fused_output_gate = False

    monkeypatch.setattr(Qwen3NextAttention, "__init__", unfused_attention)
    monkeypatch.setattr(Qwen2MoeMLP, "__init__", lambda mlp, **kwargs: torch.nn.Module.__init__(mlp))
    monkeypatch.setattr(rollout_model, "Qwen3_5RMSNorm", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(rollout_model, "mrope_positions_factor", lambda vllm_config: 1)
    text_config = types.SimpleNamespace(
        model_type=Qwen3_5RolloutDecoderLayer.TEXT_MODEL_TYPE,
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    model_config = types.SimpleNamespace(hf_text_config=text_config)
    vllm_config = types.SimpleNamespace(model_config=model_config, cache_config=None, quant_config=None)

    layer = Qwen3_5RolloutDecoderLayer(vllm_config, "full_attention", prefix="model.layers.3")

    # q, k and the gate from the layer's own fp32 kernel, its output gated by vLLM's fused kernel, its MLP's product
    # from FlashInfer's: each rounded to bf16 once
    assert type(layer.self_attn) is Qwen3_5RolloutAttention and layer.self_attn.use_fused_output_gate
    assert type(layer.mlp) is Qwen3_5RolloutMLP


def test_vllm_calls_its_fused_q_k_norm_rope_kernel_only_where_the_rollout_attention_overrides():
    # vLLM calls fused_qk_rmsnorm_rope_gate from one place, Qwen3NextAttention._project_qkv_gate, which the rollout
    # attention replaces: no step of a replica, decode, mixed or prefill, reaches the kernel
    call = f"{fused_qk_norm_rope.fused_qk_rmsnorm_rope_gate.__name__}("
    definition = pathlib.Path(fused_qk_norm_rope.__file__).resolve()
    callers = [
        path.resolve()
        for path in pathlib.Path(vllm.__file__).parent.rglob("*.py")
        if path.resolve() != definition and call in path.read_text(errors="replace")
    ]

    assert callers == [pathlib.Path(qwen3_next.__file__).resolve()]
    assert inspect.getsource(qwen3_next).count(call) == 1
    assert inspect.getsource(Qwen3NextAttention._project_qkv_gate).count(call) == 1
    assert Qwen3_5RolloutAttention._project_qkv_gate is not Qwen3NextAttention._project_qkv_gate


def test_the_decode_attention_splits_over_more_ctas(monkeypatch):
    sm_counts = []
    monkeypatch.setattr(
        vllm_flashinfer,
        "trtllm_batch_decode_with_kv_cache",
        lambda **kwargs: sm_counts.append(flashinfer_decode.get_device_sm_count(kwargs["query"].device)),
    )
    monkeypatch.setattr(flashinfer_decode, "get_device_sm_count", lambda device: 148)
    DecodeSplit.install()
    query = torch.zeros(4, 8)

    vllm_flashinfer.trtllm_batch_decode_with_kv_cache(query=query)

    assert sm_counts == [148 * SPLIT_FACTOR]
    assert flashinfer_decode.get_device_sm_count(query.device) == 148  # restored after the call


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


def test_the_blackwell_attention_keeps_no_kv_cache_vllm_released(monkeypatch):
    monkeypatch.setattr(FlashInferImpl, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(torch.ops._C_cache_ops, "reshape_and_cache_flash", lambda *args: None, raising=False)
    monkeypatch.setattr(vllm_flashinfer, "_get_trtllm_workspace_buffer", lambda: torch.zeros(16, dtype=torch.uint8))
    monkeypatch.setattr(BlackwellFlashInferImpl, "module", types.SimpleNamespace())
    impl = BlackwellFlashInferImpl()
    impl.__dict__.update(
        kv_sharing_target_layer_name=None,
        is_kvcache_nvfp4=False,
        head_size=HEAD_DIM,
        cache_dtype="auto",
        _multi_ctas_kv_counter_buffer=None,
    )
    # a KV cache of two blocks of 16 tokens, the key and value halves last, in memory whose release a weak reference
    # sees; a step of no tokens, which splits the views and launches nothing
    memory = np.zeros(2 * KV_HEADS * 16 * 2 * HEAD_DIM, dtype=np.int16)
    memory_ref = weakref.ref(memory)
    layer = types.SimpleNamespace(
        kv_cache=torch.from_numpy(memory).view(torch.bfloat16).view(2, KV_HEADS, 16, 2 * HEAD_DIM),
        _k_scale=None,
        _v_scale=None,
    )
    del memory
    md = types.SimpleNamespace(
        num_decode_tokens=0,
        num_actual_tokens=0,
        num_prefills=0,
        prefill=None,
        decode=None,
        kv_cache_layout=KVCacheLayout.LBHNC,
    )
    keys = torch.zeros(0, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    queries = torch.zeros(0, QUERY_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    impl.do_kv_cache_update(layer, keys, keys, layer.kv_cache, torch.zeros(0, dtype=torch.int64))
    impl.attend(queries, layer.kv_cache, md, torch.empty_like(queries))
    # vLLM's CUDA graph memory profiling releases its KV cache (clear_layer_kv_caches) before allocating the real one
    layer.kv_cache = torch.tensor([])

    assert memory_ref() is None


def test_the_blackwell_attention_takes_the_views_of_the_kv_cache_vllm_binds_next(monkeypatch):
    # the memory and the blocks of the key view each kernel received, the cache update's and the decode attention's
    received = []
    monkeypatch.setattr(FlashInferImpl, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda key, value, k, v, *args: received.append((k.untyped_storage().data_ptr(), k.shape[0])),
        raising=False,
    )
    monkeypatch.setattr(vllm_flashinfer, "_get_trtllm_workspace_buffer", lambda: torch.zeros(16, dtype=torch.uint8))
    decode_kernel = types.SimpleNamespace(
        trtllm_paged_attention_decode=lambda out, lse, query, k, v, *args: received.append(
            (k.untyped_storage().data_ptr(), k.shape[0])
        )
    )
    monkeypatch.setattr(BlackwellFlashInferImpl, "module", decode_kernel)
    monkeypatch.setattr(BlackwellFlashInferImpl, "sm_count", 1)
    impl = BlackwellFlashInferImpl()
    impl.__dict__.update(
        kv_sharing_target_layer_name=None,
        is_kvcache_nvfp4=False,
        head_size=HEAD_DIM,
        cache_dtype="auto",
        _multi_ctas_kv_counter_buffer=None,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )
    # a step of one decode row
    decode = types.SimpleNamespace(
        block_tables=torch.zeros(1, 1, dtype=torch.int32), seq_lens=torch.ones(1, dtype=torch.int32), max_seq_len=1
    )
    md = types.SimpleNamespace(
        num_decode_tokens=1,
        num_actual_tokens=1,
        num_prefills=0,
        prefill=None,
        decode=decode,
        kv_cache_layout=KVCacheLayout.LBHNC,
    )
    layer = types.SimpleNamespace(kv_cache=None, _k_scale=None, _v_scale=None)
    keys = torch.zeros(1, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    queries = torch.zeros(1, QUERY_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    # caches of two and then three blocks of 16 tokens, each no view of another: its views keep it alive, so the first
    # dies when its last view goes. vLLM releases a cache (clear_layer_kv_caches) before it binds the next
    for num_blocks in (2, 3):
        layer.kv_cache = torch.tensor([])
        layer.kv_cache = torch.zeros(num_blocks, KV_HEADS, 16, 2 * HEAD_DIM, dtype=torch.bfloat16)
        impl.do_kv_cache_update(layer, keys, keys, layer.kv_cache, torch.zeros(1, dtype=torch.int64))
        impl.attend(queries, layer.kv_cache, md, torch.empty_like(queries))

    assert received[2:] == [(layer.kv_cache.untyped_storage().data_ptr(), 3)] * 2


def test_the_blackwell_gdn_layer_keeps_no_kv_cache_vllm_released():
    layer = BlackwellGDNAttention.__new__(BlackwellGDNAttention)
    # what GDNCore.constants reads of a rank's layer: Qwen3.8-27B's sizes, weights it keeps by reference
    layer.__dict__.update(
        num_k_heads=GDN_K_HEADS,
        num_v_heads=GDN_V_HEADS,
        head_k_dim=GDN_HEAD_DIM,
        head_v_dim=GDN_HEAD_DIM,
        tp_size=1,
        conv1d=types.SimpleNamespace(weight=None),
        A_log=None,
        dt_bias=None,
        norm=types.SimpleNamespace(weight=None, eps=None),
    )
    # the conv and recurrent state pools of two blocks as vLLM binds them (bind_kv_cache), in memory whose release a
    # weak reference sees
    memory = np.zeros(2 * 2 * 8, dtype=np.float32)
    memory_ref = weakref.ref(memory)
    pages = torch.from_numpy(memory).view(2, 2 * 8)
    layer.kv_cache = (pages[:, :8].view(2, 2, 4), pages[:, 8:].view(2, 2, 4))
    del memory, pages

    GDNCore.constants(layer)
    # vLLM's CUDA graph memory profiling releases its KV cache (clear_layer_kv_caches) before allocating the real one
    layer.kv_cache = []

    assert memory_ref() is None


class HostStagedBuffer:
    """StagedBuffer's construction and upload on a machine without a GPU driver: pageable host buffers, plain copies,
    no upload events."""

    @staticmethod
    def init(buffer: StagedBuffer, size: int, device) -> None:
        buffer.device_buffer = torch.zeros(size, dtype=torch.int32, device=device)
        buffer.stages = [torch.zeros(size, dtype=torch.int32) for _ in range(StagedBuffer.BUFFERS)]
        buffer.upload_events = [None] * StagedBuffer.BUFFERS
        buffer.turn = 0

    @staticmethod
    def upload(buffer: StagedBuffer, size: int) -> None:
        buffer.device_buffer[:size].copy_(buffer.stages[buffer.turn][:size])
        buffer.turn ^= 1


@pytest.mark.parametrize(
    "num_decode_rows, qlens, klens",
    [
        (2, [1, 1, 30, 5], [40, 9, 30, 700]),  # decode rows, then a prompt's first chunk and one over a cached prefix
        (3, [1, 1, 1], [12, 3000, 7]),  # a pure decode step
    ],
)
def test_a_hopper_step_s_attention_arguments_land_in_their_sections(monkeypatch, num_decode_rows, qlens, klens):
    monkeypatch.setattr(StagedBuffer, "__init__", HostStagedBuffer.init)
    monkeypatch.setattr(StagedBuffer, "upload", HostStagedBuffer.upload)
    buffers = AttentionStepBuffers(MAX_NUM_SEQS, "cpu")
    n = len(qlens)
    starts = np.cumsum([0, *qlens[:-1]])

    buffers.fill(num_decode_rows, starts, sum(qlens), np.array(qlens), np.array(klens))

    assert buffers.num_decode_rows.tolist() == [num_decode_rows]
    assert buffers.cu[: n + 1].tolist() == [*starts.tolist(), sum(qlens)]
    assert buffers.sq_all[:n].tolist() == qlens and buffers.sk_all[:n].tolist() == klens  # the whole batch's
    # FA3's part: the decode rows zeroed
    decode_zeros = [0] * num_decode_rows
    assert buffers.sq[:n].tolist() == decode_zeros + qlens[num_decode_rows:]
    assert buffers.sk[:n].tolist() == decode_zeros + klens[num_decode_rows:]


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


def hopper_model(config: types.SimpleNamespace) -> HopperQwen3_5ForConditionalGeneration:
    """The Hopper replicas' model class with a vision config, without weights."""
    model = object.__new__(HopperQwen3_5ForConditionalGeneration)
    object.__setattr__(model, "config", config)

    return model


def test_a_new_turns_positions_from_its_first_uncomputed_token_equal_vllms():
    config, ids, features = image_turn(300)
    model = hopper_model(config)
    want, want_delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(ids, features, config)

    # the first uncomputed token in the text, at an image's first token, inside an image, in the closing text
    image = features[1].mm_position.offset
    for start in (100, image, image + 5, len(ids) - 3):
        positions, delta = StepInputArrays.tail_positions(model, ids, features, start)

        assert torch.equal(positions[:, start:], want[:, start:]) and delta == want_delta


class RecordedRopeState:
    """vLLM's RopeState of a replica's model state: the M-RoPE positions staged into it."""

    def __init__(self, max_num_reqs: int):
        self.num_dims = NUM_DIMS
        self.prefill_delta = types.SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32))
        self.prefill_positions = self
        self.staged = []  # (row, first column, positions) of each staged write
        self.whole_prompts = []  # the requests whose positions vLLM's init_prefill_positions computed

    def stage_write(self, row: int, start: int, positions: np.ndarray) -> None:
        self.staged.append((row, start, positions))

    def init_prefill_positions(self, req_idx, model, prefill_token_ids, mm_features) -> None:
        self.whole_prompts.append(req_idx)


def hopper_model_state(rope_state: RecordedRopeState, model) -> HopperRolloutModelState:
    """The Hopper replicas' model state with what its add_request reads, not built (no GPU)."""
    model_state = object.__new__(HopperRolloutModelState)
    model_state.__dict__.update(
        rope_state=rope_state,
        model=model,
        mm_pruner=None,
        prompt_embeds_state=None,
        num_accepted_tokens_gpu=torch.zeros(2, dtype=torch.int32),
        _align_mode=False,
    )

    return model_state


@pytest.mark.parametrize("num_computed, staged", [(StepInputArrays.MIN_PREFIX + 7, True), (100, False)])
def test_the_hopper_model_state_stages_a_new_turns_positions_from_its_first_uncomputed_token(num_computed, staged):
    config, ids, features = image_turn(StepInputArrays.MIN_PREFIX + 300)
    rope_state = RecordedRopeState(max_num_reqs=2)
    model_state = hopper_model_state(rope_state, hopper_model(config))
    request = NewRequestData(
        req_id="turn",
        prompt_token_ids=ids,
        mm_features=features,
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=num_computed,
        lora_request=None,
        prefill_token_ids=ids,
    )

    model_state.add_request(1, request)

    want, want_delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(ids, features, config)
    # vLLM's add_request ran (its acceptance count of the slot), with the model state's rope state
    assert model_state.rope_state is rope_state and model_state.num_accepted_tokens_gpu.tolist() == [0, 1]
    if staged:
        rows = [NUM_DIMS + dim for dim in range(NUM_DIMS)]  # the request's slot 1
        assert rope_state.whole_prompts == [] and rope_state.prefill_delta.np[1] == want_delta
        assert [(row, start) for row, start, _ in rope_state.staged] == [(row, num_computed) for row in rows]
        for dim, (_, _, positions) in enumerate(rope_state.staged):
            assert np.array_equal(positions, want[dim, num_computed:].numpy())
    else:
        assert rope_state.whole_prompts == [1] and rope_state.staged == []  # vLLM's positions of the whole prompt
