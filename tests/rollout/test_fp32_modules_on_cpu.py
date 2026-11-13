import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.model_loader import base_loader

from pear.rollout.vllm_patches import Fp32Modules
from pear.rollout.worker_extension import PearVLLMWorkerExtension

FP32_MODULES = ["visual", "A_log"]
LOADER = object()  # a weight's loader, which vLLM attaches to the parameter object (set_weight_attrs)
# values bf16 cannot represent, as an fp32 checkpoint and the trainer's weight sync hold them
FP32_VALUES = torch.tensor([0.1234567, 1.7654321, -0.3333333, 2.0000001])


class Rotary(torch.nn.Module):
    """A rotary embedding whose cache vLLM computes in fp32 and stores in the dtype the module is built in."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cos_sin_cache", self._compute_cos_sin_cache().to(torch.bfloat16), persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 7, dtype=torch.float32) / 3


class GatedDeltaNet(torch.nn.Module):
    """A Gated DeltaNet layer: its A_log built in fp32 by vLLM, its projection in the model dtype."""

    def __init__(self):
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.in_proj = torch.nn.Linear(4, 4).to(torch.bfloat16)


class Model(torch.nn.Module):
    """A served model in bf16 with a vision tower and a Gated DeltaNet layer, as vLLM's construction returns it."""

    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Sequential(torch.nn.Linear(4, 4), Rotary()).to(torch.bfloat16)
        self.gdn = GatedDeltaNet()
        for param in self.parameters():
            param.weight_loader = LOADER


@pytest.fixture
def construction(monkeypatch):
    # vLLM's construction builds the model above; every test leaves it and the environment as it found them
    monkeypatch.setattr(base_loader, "initialize_model", lambda **kwargs: Model())
    # set, then deleted: monkeypatch restores only what it changed, and deleting a variable that is not set changes
    # nothing, so the names a test exports would stay in this process for the tests after it
    monkeypatch.setenv(Fp32Modules.ENV, "")
    monkeypatch.delenv(Fp32Modules.ENV)


def test_the_worker_processes_get_the_server_names(construction):
    assert Fp32Modules.from_env() is None

    Fp32Modules.export(FP32_MODULES)

    assert Fp32Modules.from_env().names == FP32_MODULES


def test_no_names_exports_nothing(construction):
    Fp32Modules.export([])

    assert Fp32Modules.from_env() is None


def test_the_weights_load_their_fp32_values(construction):
    Fp32Modules(FP32_MODULES).install()
    model = base_loader.initialize_model(vllm_config=None, model_config=None, prefix="")
    # what vLLM's loaders and the weight sync do with each weight: param.data.copy_(the checkpoint's tensor)
    model.visual[0].weight.data.copy_(FP32_VALUES.expand(4, 4))
    model.gdn.A_log.data.copy_(FP32_VALUES)

    assert torch.equal(model.visual[0].weight, FP32_VALUES.expand(4, 4)) and torch.equal(model.gdn.A_log, FP32_VALUES)
    assert {p.dtype for p in model.visual.parameters()} == {torch.float32}
    assert model.gdn.in_proj.weight.dtype == torch.bfloat16
    # the same parameter objects, which carry their loaders, and the rotary cache recomputed in fp32
    assert all(param.weight_loader is LOADER for param in model.parameters())
    assert torch.equal(model.visual[1].cos_sin_cache, model.visual[1]._compute_cos_sin_cache())


def test_the_receiver_requires_every_fp32_weight_listed(construction):
    Fp32Modules(FP32_MODULES).install()
    extension = PearVLLMWorkerExtension.__new__(PearVLLMWorkerExtension)
    extension.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
        model=base_loader.initialize_model(vllm_config=None, model_config=None, prefix=""),
    )

    extension.check_fp32_modules(FP32_MODULES)
    # without A_log, the sender would round the weight vLLM keeps in fp32
    with pytest.raises(ValueError, match="gdn.A_log"):
        extension.check_fp32_modules(["visual"])


def test_the_receiver_requires_every_listed_weight_in_fp32(construction):
    # the cast at the model's construction did not run: the vision tower is in bf16, as vLLM built it
    extension = PearVLLMWorkerExtension.__new__(PearVLLMWorkerExtension)
    extension.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
        model=base_loader.initialize_model(vllm_config=None, model_config=None, prefix=""),
    )

    # the worker would round the fp32 weights the sender sends for the tower
    with pytest.raises(ValueError, match=r"visual\.0\.weight in torch\.bfloat16"):
        extension.check_fp32_modules(FP32_MODULES)


def test_the_test_of_the_names_needs_no_vllm():
    # the trainer's checkpoint engine imports the class for it
    code = "import sys; import pear.rollout.vllm_patches; assert 'vllm' not in sys.modules"

    subprocess.run([sys.executable, "-c", code], check=True)
