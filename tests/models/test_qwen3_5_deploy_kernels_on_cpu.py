import os
import pkgutil
import re
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
    BlackwellNvfp4DeployGpuBackend,
)
from pear.models.qwen3_5.deploy.blackwell.nvfp4.step import DecodeStep as Nvfp4DecodeStep
from pear.models.qwen3_5.deploy.deploy_kernels import Qwen3_5DeployKernels
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend, Nvfp4CheckpointMixin
from pear.models.qwen3_5.deploy.memory_plan import MemoryPlan
from pear.models.qwen3_5.deploy.rtx_blackwell.gpu_backend import (
    Rtx5090Nvfp4Fp8KvDeployGpuBackend,
    RtxBlackwellDeployGpuBackend,
    RtxBlackwellNvfp4DeployGpuBackend,
    RtxPro6000Nvfp4Bf16KvDeployGpuBackend,
    RtxPro6000Nvfp4Fp8KvDeployGpuBackend,
)
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtx5090.extension import Rtx5090Nvfp4Fp8KvExtension
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtxpro6000.extension import (
    RtxPro6000Nvfp4Bf16KvExtension,
    RtxPro6000Nvfp4Fp8KvExtension,
)
from pear.models.qwen3_5.deploy.rtx_blackwell.nvfp4.rtxpro6000.schedule import Schedule
from repository import REPO

GB300 = DeviceCapability(10, 3)
RTX_5090 = DeviceCapability(12, 0)
RTX_5090_NAME = "NVIDIA GeForce RTX 5090"  # NVML's
RTX_PRO_6000_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"  # an SM120 GPU of more SMs and memory
RTX_PRO_5000_NAME = "NVIDIA RTX PRO 5000 Blackwell"  # an SM120 GPU without schedules
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
        mamba_cache_dtype: str = "auto",
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
            cache_config=types.SimpleNamespace(
                cache_dtype=cache_dtype,
                mamba_ssm_cache_dtype=mamba_cache_dtype,
                calculate_kv_scales=calculates_kv_scales,
            ),
            use_v2_model_runner=is_v2_runner,
            speculative_config=speculative_config if draft_method is not None else None,
        )


@pytest.mark.parametrize(
    "capability, overrides, gpu_backend",
    [
        (GB300, {}, BlackwellNvfp4DeployGpuBackend),
        (GB300, {"draft_method": None}, None),  # no draft: kernels=null serves without one
        (GB300, {"draft_method": "eagle3"}, None),
        (GB300, {"draft_tokens": 3}, None),
        (DeviceCapability(10, 0), {}, None),  # B200: the kernels are built for sm_103a
        (GB300, {"tp": 2}, None),
        (GB300, {"max_num_seqs": 2}, None),
        (GB300, {"is_v2_runner": False}, None),
        (GB300, {"draft_sample_method": "greedy"}, None),
        (GB300, {"quantization": {"quant_method": "fp8"}}, None),
        (GB300, {"cache_dtype": "bfloat16"}, BlackwellNvfp4DeployGpuBackend),
        (GB300, {"cache_dtype": "fp8"}, None),  # the attention reads bf16 k and v
        (GB300, {"quantization": {**NVFP4, "kv_cache_quant_algo": "FP8"}}, None),  # "auto": the checkpoint's fp8
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
        (RTX_5090_NAME, {}, Rtx5090Nvfp4Fp8KvDeployGpuBackend),
        (RTX_5090_NAME, {"cache_dtype": "fp8"}, Rtx5090Nvfp4Fp8KvDeployGpuBackend),  # e4m3 on CUDA
        (RTX_5090_NAME, {"cache_dtype": "auto"}, None),  # the model's bf16: the attention reads e4m3
        (RTX_5090_NAME, {"cache_dtype": "fp8_e5m2"}, None),
        (RTX_5090_NAME, {"calculates_kv_scales": True}, None),  # the attention reads the cache at scale 1
        (RTX_5090_NAME, {"quantization": BF16}, None),
        (RTX_5090_NAME, {"draft_tokens": 3}, None),
        (RTX_5090_NAME, {"max_num_seqs": 2}, None),
        (RTX_PRO_5000_NAME, {}, None),  # no schedules for its SMs
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


@pytest.mark.parametrize(
    "overrides, gpu_backend",
    [
        ({}, RtxPro6000Nvfp4Bf16KvDeployGpuBackend),  # "auto": the model's bf16
        ({"cache_dtype": "bfloat16"}, RtxPro6000Nvfp4Bf16KvDeployGpuBackend),
        ({"cache_dtype": "fp8_e4m3"}, RtxPro6000Nvfp4Fp8KvDeployGpuBackend),
        ({"cache_dtype": "fp8"}, RtxPro6000Nvfp4Fp8KvDeployGpuBackend),  # e4m3 on CUDA
        ({"cache_dtype": "fp8_e5m2"}, None),
        ({"cache_dtype": "fp8_e4m3", "calculates_kv_scales": True}, None),  # the attention reads the cache at scale 1
        ({"mamba_cache_dtype": "bfloat16"}, None),  # the Gated DeltaNet reads fp32 states
        ({"quantization": BF16}, None),
        ({"draft_tokens": 3}, None),
    ],
)
def test_on_an_rtx_pro_6000_the_server_selects_the_megakernels_build_of_the_kv_cache_dtype(
    monkeypatch, overrides, gpu_backend
):
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: RTX_5090)
    monkeypatch.setattr(current_platform, "get_device_name", lambda device_id=0: RTX_PRO_6000_NAME)
    vllm_config = EngineConfig.of(**overrides)

    if gpu_backend is None:
        with pytest.raises(ValueError, match="no GPU backend"):
            Qwen3_5DeployKernels.select_gpu_backend(vllm_config)
    else:
        assert Qwen3_5DeployKernels.select_gpu_backend(vllm_config) is gpu_backend


@pytest.mark.parametrize(
    "gpu_backend, package",
    [
        (Rtx5090Nvfp4Fp8KvDeployGpuBackend, "rtx5090"),
        (RtxPro6000Nvfp4Bf16KvDeployGpuBackend, "rtxpro6000"),
        (RtxPro6000Nvfp4Fp8KvDeployGpuBackend, "rtxpro6000"),
    ],
)
def test_an_rtx_gpus_backend_launches_the_megakernel_of_its_package(gpu_backend, package):
    for launched in (gpu_backend.extension, gpu_backend.decode_launch, gpu_backend.draft_launch):
        assert launched.__module__.split(".")[-2] == package


@pytest.mark.parametrize("package", ["rtx5090", "rtxpro6000"])
def test_an_rtx_megakernels_sources_include_files_of_the_repository(package):
    csrc = REPO / "pear" / "models" / "qwen3_5" / "deploy" / "rtx_blackwell" / "nvfp4" / package / "csrc"
    common_csrc = csrc.parents[1] / "common" / "csrc"  # the files the GPUs' megakernels share

    for source in (*csrc.iterdir(), *common_csrc.iterdir()):
        for include in re.findall(r'#include "([^"]+)"', source.read_text()):
            assert (source.parent / include).is_file(), (source.name, include)


def test_each_build_of_the_rtx_megakernels_has_its_own_ops():
    builds = (Rtx5090Nvfp4Fp8KvExtension, RtxPro6000Nvfp4Bf16KvExtension, RtxPro6000Nvfp4Fp8KvExtension)

    assert len({build.OP_NAMESPACE for build in builds}) == len(builds)


def test_the_rtx_pro_6000s_megakernel_has_a_build_per_kv_cache_dtype():
    builds = (RtxPro6000Nvfp4Bf16KvExtension, RtxPro6000Nvfp4Fp8KvExtension)
    flags = [{flag for flag in build.CUDA_FLAGS if flag.startswith("-DMK_")} for build in builds]

    assert flags == [set(), {"-DMK_KV_FP8=1"}]
    assert [build.kv_element_bytes() for build in builds] == [2, 1]
    assert [build.QUERY_DTYPE for build in builds] == [torch.bfloat16, torch.float16]


@pytest.mark.parametrize(
    "build, dtype, is_read",
    [
        (RtxPro6000Nvfp4Bf16KvExtension, torch.bfloat16, True),
        (RtxPro6000Nvfp4Bf16KvExtension, torch.uint8, False),
        (RtxPro6000Nvfp4Fp8KvExtension, torch.uint8, True),  # vLLM allocates an FP8 cache as bytes
        (RtxPro6000Nvfp4Fp8KvExtension, torch.float8_e4m3fn, True),
        (RtxPro6000Nvfp4Fp8KvExtension, torch.bfloat16, False),
        (Rtx5090Nvfp4Fp8KvExtension, torch.uint8, True),
        (Rtx5090Nvfp4Fp8KvExtension, torch.float8_e4m3fn, True),
        (Rtx5090Nvfp4Fp8KvExtension, torch.bfloat16, False),
    ],
)
def test_a_build_of_an_rtx_megakernel_reads_the_kv_cache_of_its_dtype_as_bytes(build, dtype, is_read):
    blocks, heads, block_tokens, row_elems = 3, 4, 16, 512
    # vLLM's NHD cache: memory [blocks][tokens][heads], logical (blocks, heads, tokens, k | v)
    kv_cache = torch.zeros(blocks, block_tokens, heads, row_elems, dtype=dtype).permute(0, 2, 1, 3)

    if is_read:
        kv_bytes = build.kv_cache_bytes(kv_cache, heads, row_elems)
        assert kv_bytes.dtype == torch.uint8
        assert kv_bytes.numel() == blocks * block_tokens * heads * row_elems * build.kv_element_bytes()
    else:
        with pytest.raises(AssertionError, match="KV cache"):
            build.kv_cache_bytes(kv_cache, heads, row_elems)


def test_a_build_of_the_rtx_pro_6000s_megakernel_refuses_a_kv_cache_in_another_memory_order():
    kv_cache = torch.zeros(3, 4, 16, 512, dtype=torch.bfloat16)  # memory [blocks][heads][tokens]

    with pytest.raises(AssertionError):
        RtxPro6000Nvfp4Bf16KvExtension.kv_cache_bytes(kv_cache, 4, 512)


@pytest.mark.parametrize("build", [Rtx5090Nvfp4Fp8KvExtension, RtxPro6000Nvfp4Fp8KvExtension])
@pytest.mark.parametrize("scales, is_served", [((1.0, 1.0, 1.0), True), ((1.0, 0.5, 1.0), False)])
def test_an_rtx_megakernel_reads_an_fp8_kv_cache_at_scale_one_only(build, scales, is_served):
    attention_layer = types.SimpleNamespace(**{f"_{part}_scale_float": scale for part, scale in zip("qkv", scales)})

    RtxPro6000Nvfp4Bf16KvExtension.check_kv_scales(attention_layer)  # a bf16 cache has no scales
    if is_served:
        build.check_kv_scales(attention_layer)
    else:
        with pytest.raises(AssertionError, match="scale 1"):
            build.check_kv_scales(attention_layer)


@pytest.mark.parametrize("total, pair_stages, sms", [(5440, 32, 188), (640, 80, 188), (7, 2, 3)])
def test_a_paired_ops_split_puts_a_pair_in_at_most_two_parts(total, pair_stages, sms):
    counts = Schedule.paired_counts(total, pair_stages, sms)
    ends = [sum(counts[: i + 1]) for i in range(len(counts))]
    parts_of_pair: dict[int, int] = {}
    for count, end in zip(counts, ends):
        for pair in range((end - count) // pair_stages, -(-end // pair_stages)) if count else ():
            parts_of_pair[pair] = parts_of_pair.get(pair, 0) + 1

    assert len(counts) == sms and sum(counts) == total
    assert max(parts_of_pair.values()) <= 2


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
    monkeypatch.setattr(current_platform, "get_device_name", lambda device_id=0: RTX_5090_NAME)
    monkeypatch.delenv(DeployGpuBackend.COMPILE_CACHE_ENV, raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

    Qwen3_5DeployKernels.configure_server({}, 1, 8192)

    cache_root = os.environ["VLLM_CACHE_ROOT"]
    assert os.path.dirname(cache_root) == str(tmp_path)
    assert os.path.basename(cache_root).startswith(RtxBlackwellNvfp4DeployGpuBackend.COMPILE_CACHE_NAME)
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
        (BlackwellNvfp4DeployGpuBackend, BlackwellDeployGpuBackend),
        (Rtx5090Nvfp4Fp8KvDeployGpuBackend, RtxBlackwellDeployGpuBackend),
    ],
    ids=["gb300", "rtx5090"],
)
@pytest.mark.parametrize("linear_backend", ["auto", "cutlass"])
def test_an_nvfp4_backend_picks_its_prefill_gemms_unless_the_engine_names_them(
    monkeypatch, gpu_backend, megakernels, linear_backend
):
    # the GPU's install loads the megakernel and patches vLLM: not on a CPU
    monkeypatch.setattr(megakernels, "install", classmethod(lambda cls, vllm_config: None))
    vllm_config = types.SimpleNamespace(kernel_config=types.SimpleNamespace(linear_backend=linear_backend))

    gpu_backend.install(vllm_config)

    expected = Nvfp4CheckpointMixin.LINEAR_BACKEND if linear_backend == "auto" else linear_backend
    assert vllm_config.kernel_config.linear_backend == expected


class ModelViewStandIn:
    """A stand-in for the decode step's view of the model with the caches the attention reads and the draft's layers."""

    LAYERS = 4  # decoder layers, the last one full attention

    @classmethod
    def of(cls, step_class: type, aux_layers: tuple) -> types.SimpleNamespace:
        step_module = sys.modules[step_class.__module__]
        attention = object.__new__(step_module.AttentionCore)
        attention.kv_cache = torch.empty(
            1, step_module.ATT_KV_HEADS, step_module.ATT_PAGE, step_module.ATT_ROW_BYTES // 2
        )
        layers = [types.SimpleNamespace(core=None) for _ in range(cls.LAYERS - 1)]
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
