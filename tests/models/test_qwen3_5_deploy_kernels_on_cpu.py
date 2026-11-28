import os
import pkgutil
import sys
import types

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

from pear.models.kernels import DeployKernels
from pear.models.qwen3_5.deploy.blackwell.bf16.step import DecodeStep as Bf16DecodeStep
from pear.models.qwen3_5.deploy.blackwell.gpu_backend import (
    BlackwellBf16DeployGpuBackend,
    BlackwellDeployGpuBackend,
    BlackwellNvfp4Bf16KvDeployGpuBackend,
    BlackwellNvfp4Fp8KvDeployGpuBackend,
)
from pear.models.qwen3_5.deploy.blackwell.nvfp4.draft.extension import (
    BlackwellNvfp4Bf16KvDraftExtension,
    BlackwellNvfp4DraftExtension,
    BlackwellNvfp4Fp8KvDraftExtension,
)
from pear.models.qwen3_5.deploy.blackwell.nvfp4.extension import (
    BlackwellNvfp4Bf16KvExtension,
    BlackwellNvfp4Extension,
    BlackwellNvfp4Fp8KvExtension,
)
from pear.models.qwen3_5.deploy.blackwell.nvfp4.step import DecodeStep as Nvfp4DecodeStep
from pear.models.qwen3_5.deploy.deploy_kernels import Qwen3_5DeployKernels
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend, Nvfp4CheckpointMixin
from pear.models.qwen3_5.deploy.memory_plan import MemoryPlan
from pear.models.qwen3_5.deploy.rtx_blackwell.gpu_backend import (
    RtxBlackwellDeployGpuBackend,
    RtxBlackwellNvfp4DeployGpuBackend,
)
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.extension import RtxBlackwellNvfp4Extension
from repository import REPO

GB300 = DeviceCapability(10, 3)
RTX_5090 = DeviceCapability(12, 0)
RTX_5090_NAME = "NVIDIA GeForce RTX 5090"  # NVML's
RTX_PRO_6000_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"  # an SM120 GPU of more SMs and memory
NVFP4 = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}  # ModelOpt's mixed export's config.json
BF16 = {}  # an unquantized checkpoint: no quantization_config


class EngineConfig:
    """A stand-in for vLLM's engine config with what the GPU backends read: one session, a draft, NVFP4 or bf16."""

    @staticmethod
    def of(
        tp: int = 1,
        max_num_seqs: int = 1,
        is_v2_runner: bool = True,
        draft_method: str | None = DeployGpuBackend.DRAFT_METHOD,
        draft_tokens: int = BlackwellDeployGpuBackend.DRAFT_TOKENS,
        draft_sample_method: str = DeployGpuBackend.DRAFT_SAMPLE_METHOD,
        quantization: dict | None = None,
        checkpoint_dtype: str = "bfloat16",
        dtype: torch.dtype = torch.bfloat16,
        cache_dtype: str = "auto",
        calculates_kv_scales: bool = False,
    ) -> types.SimpleNamespace:
        hf_config = types.SimpleNamespace(
            quantization_config=NVFP4 if quantization is None else quantization,
            text_config=types.SimpleNamespace(dtype=checkpoint_dtype),
        )
        speculative_config = types.SimpleNamespace(
            method=draft_method, num_speculative_tokens=draft_tokens, draft_sample_method=draft_sample_method
        )

        return types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                architectures=[DeployGpuBackend.ARCHITECTURE], hf_config=hf_config, dtype=dtype
            ),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=tp),
            scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
            cache_config=types.SimpleNamespace(cache_dtype=cache_dtype, calculate_kv_scales=calculates_kv_scales),
            use_v2_model_runner=is_v2_runner,
            speculative_config=speculative_config if draft_method is not None else None,
        )


@pytest.mark.parametrize(
    "capability, overrides, gpu_backend",
    [
        (GB300, {}, BlackwellNvfp4Bf16KvDeployGpuBackend),  # "auto": the model's bf16
        (GB300, {"draft_method": None}, None),  # no draft: kernels=null serves without one
        (GB300, {"draft_method": "eagle3"}, None),
        (GB300, {"draft_tokens": 3}, None),
        (DeviceCapability(10, 0), {}, None),  # B200: the kernels are built for sm_103a
        (GB300, {"tp": 2}, None),
        (GB300, {"max_num_seqs": 2}, None),
        (GB300, {"is_v2_runner": False}, None),
        (GB300, {"draft_sample_method": "greedy"}, None),
        (GB300, {"quantization": {"quant_method": "fp8"}}, None),
        (GB300, {"cache_dtype": "bfloat16"}, BlackwellNvfp4Bf16KvDeployGpuBackend),
        (GB300, {"cache_dtype": "fp8"}, BlackwellNvfp4Fp8KvDeployGpuBackend),  # e4m3 on CUDA
        (GB300, {"cache_dtype": "fp8_e4m3"}, BlackwellNvfp4Fp8KvDeployGpuBackend),
        (GB300, {"cache_dtype": "fp8_e5m2"}, None),
        (GB300, {"cache_dtype": "fp8_e4m3", "calculates_kv_scales": True}, None),  # the attention reads it at scale 1
        # "auto": the checkpoint's fp8 (its decode runner reads it at scale 1 only)
        (GB300, {"quantization": {**NVFP4, "kv_cache_quant_algo": "FP8"}}, BlackwellNvfp4Fp8KvDeployGpuBackend),
        (GB300, {"quantization": BF16, "cache_dtype": "fp8_e4m3"}, None),
        (GB300, {"quantization": BF16}, BlackwellBf16DeployGpuBackend),
        (GB300, {"quantization": BF16, "checkpoint_dtype": "float16"}, None),
        (GB300, {"quantization": BF16, "dtype": torch.float32}, None),
    ],
)
def test_the_server_selects_the_gpu_backend_of_its_gpu_and_engine_config(
    monkeypatch, capability, overrides, gpu_backend
):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: capability)
    vllm_config = EngineConfig.of(**overrides)

    if gpu_backend is None:
        with pytest.raises(ValueError, match="no GPU backend"):
            Qwen3_5DeployKernels.select_gpu_backend(vllm_config)
    else:
        assert Qwen3_5DeployKernels.select_gpu_backend(vllm_config) is gpu_backend


@pytest.mark.parametrize(
    "gpu_name, overrides, gpu_backend",
    [
        (RTX_5090_NAME, {}, RtxBlackwellNvfp4DeployGpuBackend),
        (RTX_5090_NAME, {"cache_dtype": "fp8"}, RtxBlackwellNvfp4DeployGpuBackend),  # e4m3 on CUDA
        (RTX_5090_NAME, {"cache_dtype": "auto"}, None),  # the model's bf16: the attention reads e4m3
        (RTX_5090_NAME, {"cache_dtype": "fp8_e5m2"}, None),
        (RTX_5090_NAME, {"quantization": BF16}, None),
        (RTX_5090_NAME, {"draft_tokens": 3}, None),
        (RTX_5090_NAME, {"max_num_seqs": 2}, None),
        (RTX_PRO_6000_NAME, {}, None),  # the schedules are built for the 5090
    ],
)
def test_on_an_rtx_5090_the_server_selects_the_nvfp4_megakernel_on_an_fp8_kv_cache(
    monkeypatch, gpu_name, overrides, gpu_backend
):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: RTX_5090)
    monkeypatch.setattr(current_platform, "get_device_name", lambda device_id=0: gpu_name)
    vllm_config = EngineConfig.of(**{"cache_dtype": "fp8_e4m3", **overrides})

    if gpu_backend is None:
        with pytest.raises(ValueError, match="no GPU backend"):
            Qwen3_5DeployKernels.select_gpu_backend(vllm_config)
    else:
        assert Qwen3_5DeployKernels.select_gpu_backend(vllm_config) is gpu_backend


def test_the_server_switches_on_the_runner_the_kernels_hook(monkeypatch):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: GB300)
    monkeypatch.delenv(DeployGpuBackend.V2_MODEL_RUNNER_ENV, raising=False)

    Qwen3_5DeployKernels.configure_server({}, 1, 8192)

    assert os.environ[DeployGpuBackend.V2_MODEL_RUNNER_ENV] == "1"


def test_on_gb300_the_engine_compiles_its_graphs_anew(monkeypatch):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: GB300)
    monkeypatch.delenv(DeployGpuBackend.COMPILE_CACHE_ENV, raising=False)

    Qwen3_5DeployKernels.configure_server({}, 1, 8192)

    assert os.environ[DeployGpuBackend.COMPILE_CACHE_ENV] == "1"


def test_on_the_rtx_5090_the_engine_caches_its_graphs_by_the_megakernels_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: RTX_5090)
    monkeypatch.delenv(DeployGpuBackend.COMPILE_CACHE_ENV, raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

    Qwen3_5DeployKernels.configure_server({}, 1, 8192)

    cache_root = os.environ["VLLM_CACHE_ROOT"]
    assert os.path.dirname(cache_root) == str(tmp_path)
    assert os.path.basename(cache_root).startswith(RtxBlackwellNvfp4Extension.OP_NAMESPACE)
    assert DeployGpuBackend.COMPILE_CACHE_ENV not in os.environ


@pytest.mark.parametrize(
    "options, is_valid", [("", True), ("groups,embed,vision_host", True), ("groups,graphs", False)]
)
def test_the_memory_plan_takes_its_own_options_only(monkeypatch, options, is_valid):
    monkeypatch.setenv(MemoryPlan.ENV, options)

    if is_valid:
        assert MemoryPlan.options() <= set(MemoryPlan.OPTIONS)
    else:
        with pytest.raises(ValueError, match="graphs"):
            MemoryPlan.options()


def test_the_family_deploy_config_names_the_familys_classes():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "pear" / "configs"), version_base=None):
        config = compose(config_name="server", overrides=["model=checkpoint", "+qwen3_5=deploy"])

    assert pkgutil.resolve_name(config.kernels) is Qwen3_5DeployKernels
    assert issubclass(Qwen3_5DeployKernels, DeployKernels)
    assert config.engine.compilation_config.cudagraph_mode == "FULL_AND_PIECEWISE"
    assert config.env[DeployGpuBackend.V2_MODEL_RUNNER_ENV] == "1"


@pytest.mark.parametrize(
    "gpu_backend, megakernels",
    [
        (BlackwellNvfp4Bf16KvDeployGpuBackend, BlackwellDeployGpuBackend),
        (BlackwellNvfp4Fp8KvDeployGpuBackend, BlackwellDeployGpuBackend),
        (RtxBlackwellNvfp4DeployGpuBackend, RtxBlackwellDeployGpuBackend),
    ],
    ids=["gb300", "gb300_fp8_kv", "rtx5090"],
)
@pytest.mark.parametrize("linear_backend", ["auto", "cutlass"])
def test_an_nvfp4_backend_picks_its_prefill_gemms_unless_the_engine_names_them(
    monkeypatch, gpu_backend, megakernels, linear_backend
):
    # the GPU's install loads the megakernel and patches vLLM: not on a CPU
    monkeypatch.setattr(megakernels, "install", classmethod(lambda cls, vllm_config: None))
    monkeypatch.setattr(gpu_backend.extension, "load", classmethod(lambda cls: None))
    vllm_config = types.SimpleNamespace(kernel_config=types.SimpleNamespace(linear_backend=linear_backend))

    gpu_backend.install(vllm_config)

    expected = Nvfp4CheckpointMixin.LINEAR_BACKEND if linear_backend == "auto" else linear_backend
    assert vllm_config.kernel_config.linear_backend == expected


class ModelViewStandIn:
    """A stand-in for the decode step's view of the model with the caches the attention reads and the draft's layers."""

    LAYERS = 4  # decoder layers, the last one full attention

    @classmethod
    def of(
        cls,
        step_class: type,
        aux_layers: tuple,
        kv_dtype: torch.dtype = torch.bfloat16,
        block_tokens: int | None = None,
        kv_dtypes: tuple[torch.dtype, ...] = (),
    ) -> types.SimpleNamespace:
        """The view of one attention layer on a KV cache of kv_dtype in kernel blocks of block_tokens (a page's by
        default), and one more attention layer for each of kv_dtypes."""
        step_module = sys.modules[step_class.__module__]
        layers = [types.SimpleNamespace(core=None) for _ in range(cls.LAYERS - 1 - len(kv_dtypes))]
        for dtype in (kv_dtype, *kv_dtypes):
            attention = object.__new__(step_module.AttentionCore)
            attention.kv_cache = torch.empty(
                1,
                step_module.ATT_KV_HEADS,
                block_tokens or step_module.ATT_PAGE,
                step_module.ATT_ROW_BYTES // 2,
                dtype=dtype,
            )
            layers.append(types.SimpleNamespace(core=attention))

        return types.SimpleNamespace(layers=layers, aux_layers=aux_layers, block_size=step_module.ATT_PAGE)


@pytest.mark.parametrize("step_class", [Nvfp4DecodeStep, Bf16DecodeStep], ids=["nvfp4", "bf16"])
@pytest.mark.parametrize(
    "aux_layers, is_supported",
    [((1, 2, 4), True), ((), False), ((0, 2), False), ((2, ModelViewStandIn.LAYERS + 1), False)],
)
def test_the_decode_step_takes_a_draft_that_reads_its_decoder_layers(step_class, aux_layers, is_supported):
    view = ModelViewStandIn.of(step_class, aux_layers)

    assert (step_class.unsupported_reason(view) is None) == is_supported


@pytest.mark.parametrize(
    "kv_dtype, block_tokens, kv_dtypes, is_supported",
    [
        (torch.bfloat16, 64, (), True),
        (torch.bfloat16, 16, (), False),  # bf16: kernel blocks of a page
        (torch.float8_e4m3fn, 16, (), True),  # FP8: of a quarter, a half or a whole page
        (torch.float8_e4m3fn, 32, (), True),
        (torch.float8_e4m3fn, 64, (), True),
        (torch.float8_e4m3fn, 8, (), False),
        (torch.float8_e5m2, 16, (), False),
        (torch.float8_e4m3fn, 64, (torch.bfloat16,), False),  # one dtype for every attention layer
    ],
)
def test_the_nvfp4_decode_step_reads_a_bf16_or_an_fp8_kv_cache(kv_dtype, block_tokens, kv_dtypes, is_supported):
    view = ModelViewStandIn.of(Nvfp4DecodeStep, (1, 2, 4), kv_dtype, block_tokens, kv_dtypes)

    assert (Nvfp4DecodeStep.unsupported_reason(view) is None) == is_supported


@pytest.mark.parametrize(
    "gpu_backend, kv_dtype",
    [
        (BlackwellNvfp4Bf16KvDeployGpuBackend, torch.bfloat16),
        (BlackwellNvfp4Fp8KvDeployGpuBackend, torch.float8_e4m3fn),
    ],
)
def test_a_gb300_nvfp4_backend_launches_the_builds_of_its_kv_cache_dtype(gpu_backend, kv_dtype):
    assert gpu_backend.extension.KV_DTYPE == kv_dtype
    assert gpu_backend.draft_extension.KV_DTYPE == kv_dtype


@pytest.mark.parametrize(
    "builds",
    [
        (BlackwellNvfp4Bf16KvExtension, BlackwellNvfp4Fp8KvExtension),
        (BlackwellNvfp4Bf16KvDraftExtension, BlackwellNvfp4Fp8KvDraftExtension),
    ],
    ids=["megakernel", "draft"],
)
def test_the_gb300_nvfp4_kernels_have_a_build_per_kv_cache_dtype(builds):
    flags = [{flag for flag in build.CUDA_FLAGS if flag.startswith("-DMK_")} for build in builds]

    assert flags == [set(), {"-DMK_KV_FP8=1"}]
    assert len({build.OP_NAMESPACE for build in builds}) == len(builds)
    assert len({sys.modules[build.__module__] for build in builds}) == 1  # (CudaExtension: one csrc/)
    assert [build.KV_DTYPE for build in builds] == [torch.bfloat16, torch.float8_e4m3fn]


@pytest.mark.parametrize(
    "kernel, build",
    [
        (BlackwellNvfp4Extension, BlackwellNvfp4Bf16KvExtension),
        (BlackwellNvfp4Extension, BlackwellNvfp4Fp8KvExtension),
        (BlackwellNvfp4DraftExtension, BlackwellNvfp4Bf16KvDraftExtension),
        (BlackwellNvfp4DraftExtension, BlackwellNvfp4Fp8KvDraftExtension),
    ],
)
def test_the_package_calls_the_ops_of_the_build_its_process_loaded(monkeypatch, kernel, build):
    with pytest.raises(RuntimeError, match="not loaded"):
        kernel.ops()

    monkeypatch.setattr(build, "is_loaded", True)

    assert kernel.loaded() is build
    assert kernel.ops() is getattr(torch.ops, build.OP_NAMESPACE)


@pytest.mark.parametrize(
    "build, dtype, is_read",
    [
        (BlackwellNvfp4Bf16KvExtension, torch.bfloat16, True),
        (BlackwellNvfp4Bf16KvExtension, torch.uint8, False),
        (BlackwellNvfp4Fp8KvExtension, torch.uint8, True),  # vLLM allocates an FP8 cache as bytes
        (BlackwellNvfp4Fp8KvExtension, torch.float8_e4m3fn, True),
        (BlackwellNvfp4Fp8KvExtension, torch.bfloat16, False),
    ],
)
def test_a_build_of_the_gb300_nvfp4_megakernel_reads_the_kv_cache_of_its_dtype(build, dtype, is_read):
    kv_cache = torch.zeros(2, 4, 16, 512, dtype=dtype)
    attention_layer = types.SimpleNamespace(kv_cache=kv_cache)

    if is_read:
        viewed = build.kv_cache_of(attention_layer)
        assert viewed.dtype == build.KV_DTYPE
        assert viewed.data_ptr() == kv_cache.data_ptr() and viewed.shape == kv_cache.shape
    else:
        with pytest.raises(AssertionError, match="KV cache"):
            build.kv_cache_of(attention_layer)


@pytest.mark.parametrize("scales, is_served", [((1.0, 1.0, 1.0), True), ((1.0, 0.5, 1.0), False)])
def test_the_gb300_nvfp4_megakernel_reads_an_fp8_kv_cache_at_scale_one_only(scales, is_served):
    attention_layer = types.SimpleNamespace(**{f"_{part}_scale_float": scale for part, scale in zip("qkv", scales)})

    BlackwellNvfp4Bf16KvExtension.check_kv_scales(attention_layer)  # a bf16 cache has no scales
    if is_served:
        BlackwellNvfp4Fp8KvExtension.check_kv_scales(attention_layer)
    else:
        with pytest.raises(AssertionError, match="scale 1"):
            BlackwellNvfp4Fp8KvExtension.check_kv_scales(attention_layer)
