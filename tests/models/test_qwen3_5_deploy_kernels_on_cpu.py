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
    BlackwellNvfp4DeployGpuBackend,
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
            cache_config=types.SimpleNamespace(cache_dtype=cache_dtype),
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
        (BlackwellNvfp4DeployGpuBackend, BlackwellDeployGpuBackend),
        (RtxBlackwellNvfp4DeployGpuBackend, RtxBlackwellDeployGpuBackend),
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
