from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from pear.models.kernels import TrainingKernels
from pear.trainer.config import PearFSDPEngineConfig
from pear.trainer.engine import PearFSDPEngine
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead

# the fake below by class path, as fsdp_config names the kernels
KERNELS = f"{__name__}.FakeKernels"
EVENTS = []  # what the fake was asked to do, in order


class FakeKernels(TrainingKernels):
    """Training kernels that record what the engine asks of them."""

    @classmethod
    def check_config(cls, engine_config) -> None:
        EVENTS.append(("check_config",))

    @classmethod
    def install(cls, module, fp32_residual, use_fused_kernels, fused_kernels_backend) -> None:
        EVENTS.append(("install", fp32_residual, use_fused_kernels, fused_kernels_backend))

    @classmethod
    def check_model(cls, module, use_remove_padding, use_fused_kernels, fused_kernels_backend) -> None:
        EVENTS.append(("check_model", use_remove_padding, use_fused_kernels, fused_kernels_backend))

    @classmethod
    def install_gpu_backend(cls, module) -> None:
        EVENTS.append(("install_gpu_backend",))

    @classmethod
    def install_checkpointing(cls, module) -> None:
        EVENTS.append(("install_checkpointing",))

    @classmethod
    def run_checked(cls, run, before_rerun):
        EVENTS.append(("run_checked",))
        before_rerun()
        return run()


@pytest.fixture
def events():
    """The fake's record, empty at the start of a test."""
    EVENTS.clear()

    return EVENTS


@pytest.fixture
def nvidia_gpu(monkeypatch):
    """The engine registry's lookup as on an NVIDIA GPU machine."""
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "cuda")
    monkeypatch.setenv("VERL_ENGINE_VENDOR", "nvidia")


class EngineUnderTest:
    """A PearFSDPEngine without verl's constructor, which needs a process group."""

    @staticmethod
    def of(kernels=None, use_gpu_backend=False, **engine_config) -> PearFSDPEngine:
        engine = object.__new__(PearFSDPEngine)
        engine.kernels = kernels
        engine.use_gpu_backend = use_gpu_backend
        engine.support_fused_score_centering = kernels is not None
        engine.streamed_adamw = None
        engine.use_remove_padding = True
        engine.use_ulysses_sp = False
        engine.engine_config = SimpleNamespace(mixed_precision={"fp32_residual": True}, **engine_config)
        engine.model_config = SimpleNamespace(
            fused_kernel_options={"impl_backend": "torch"}, use_fused_kernels=True, enable_gradient_checkpointing=True
        )

        return engine


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_the_registry_builds_pear_s_engine_for_a_language_model_only(nvidia_gpu, strategy):
    assert EngineRegistry.get_engine_cls("language_model", strategy) is PearFSDPEngine
    assert not issubclass(EngineRegistry.get_engine_cls("value_model", strategy), PearFSDPEngine)


def test_a_config_with_kernels_refuses_to_build_another_engine(monkeypatch, nvidia_gpu, events):
    assert PearFSDPEngineConfig(strategy="fsdp2", kernels=KERNELS, use_gpu_backend=True).kernels == KERNELS
    # the driver builds the config too: it does not import the kernels to do so
    assert events == []

    # a vendor without an engine of its own: the registry falls back to the device's, which is verl's
    monkeypatch.setenv("VERL_ENGINE_VENDOR", "another")
    with pytest.raises(RuntimeError, match="FSDPEngineWithLMHead, not PearFSDPEngine"):
        PearFSDPEngineConfig(strategy="fsdp2", kernels=KERNELS)
    # without PEAR's keys the engine it builds is verl's in what it does: nothing to refuse
    PearFSDPEngineConfig(strategy="fsdp2")


def test_a_gpu_backend_and_an_fp32_residual_stream_need_kernels(nvidia_gpu):
    with pytest.raises(ValueError, match="actor.fsdp_config.use_gpu_backend requires actor.fsdp_config.kernels"):
        PearFSDPEngineConfig(strategy="fsdp2", use_gpu_backend=True)
    with pytest.raises(
        ValueError, match="actor.fsdp_config.mixed_precision.fp32_residual requires actor.fsdp_config.kernels"
    ):
        PearFSDPEngineConfig(strategy="fsdp2", mixed_precision={"fp32_residual": True})


def test_the_engine_installs_the_kernels_then_their_gpu_backend_on_the_module_verl_built(monkeypatch, events):
    monkeypatch.setattr(FSDPEngineWithLMHead, "_build_module", lambda self: events.append(("verl",)) or "module")

    assert EngineUnderTest.of(FakeKernels, use_gpu_backend=True)._build_module() == "module"
    assert events == [
        ("check_config",),
        ("verl",),
        ("install", True, True, "torch"),
        ("check_model", True, True, "torch"),
        ("install_gpu_backend",),
        ("install_checkpointing",),
    ]

    events.clear()
    assert EngineUnderTest.of(FakeKernels)._build_module() == "module"
    assert events == [("verl",), ("install", True, True, "torch")]

    events.clear()
    assert EngineUnderTest.of()._build_module() == "module"
    assert events == [("verl",)]


@pytest.mark.parametrize("forward_only", [False, True])
def test_a_batch_the_gpu_backend_gives_up_reruns_without_its_gradients(monkeypatch, events, forward_only):
    monkeypatch.setattr(
        FSDPEngineWithLMHead,
        "forward_backward_batch",
        lambda self, data, loss_function, forward_only: events.append(("verl", data, forward_only)) or "outputs",
    )
    monkeypatch.setattr(PearFSDPEngine, "optimizer_zero_grad", lambda self: events.append(("zero_grad",)))

    engine = EngineUnderTest.of(FakeKernels, use_gpu_backend=True)
    assert engine.forward_backward_batch("batch", None, forward_only) == "outputs"
    dropped = [] if forward_only else [("zero_grad",)]
    assert events == [("run_checked",), *dropped, ("verl", "batch", forward_only)]

    events.clear()
    assert EngineUnderTest.of(FakeKernels).forward_backward_batch("batch", None, forward_only) == "outputs"
    assert events == [("verl", "batch", forward_only)]


@pytest.mark.parametrize("ready", [False, True])
def test_the_streamed_step_takes_over_once_the_states_live_on_the_host(events, ready):
    engine = EngineUnderTest.of(optimizer_offload_step=False)
    engine.optimizer = SimpleNamespace(step=lambda: events.append(("step",)))
    engine.streamed_adamw = SimpleNamespace(is_ready=lambda: ready, step=lambda: events.append(("streamed step",)))

    engine._optimizer_update()

    # states not all on the host take verl's update
    assert events == [("streamed step",) if ready else ("step",)]


@pytest.mark.parametrize("has_kernels", [True, False])
def test_fused_score_centering_takes_its_term_from_the_kernels_forward_only(has_kernels):
    lengths = torch.tensor([3, 2])
    tokens, head_size = int(lengths.sum()), 4
    offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)])
    micro_batch = TensorDict(
        {"input_ids": torch.nested.nested_tensor_from_jagged(torch.zeros(tokens, dtype=torch.long), offsets=offsets)},
        batch_size=[],
    )
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=True,
        use_fused_kernels=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=True,
    )
    # a fused forward's output on the packed tokens: no logits, the log-probs at the rollout's head ids
    output = SimpleNamespace(
        log_probs=torch.zeros(1, tokens),
        entropy=torch.zeros(1, tokens),
        head_log_probs=torch.randn(1, tokens, head_size),
    )
    output_args = {"input_ids_rmpad_rolled": None, "temperature_rmpad": None, "pad_size": 0}
    engine = EngineUnderTest.of(FakeKernels if has_kernels else None)

    def centering(train_head_log_probs, data):
        return {"sc_correction": train_head_log_probs.sum(-1)}

    assert bool(engine.has_fused_score_centering(micro_batch)) is has_kernels
    if has_kernels:
        model_output = engine.prepare_model_outputs(output, output_args, micro_batch, centering)
        assert torch.equal(model_output["sc_correction"].values(), output.head_log_probs[0].sum(-1))
    else:
        # verl's engine in all it does: it has no fused forward that returns the head's log-probs
        with pytest.raises(NotImplementedError, match="score_centering=True is not supported with use_fused_kernels"):
            engine.prepare_model_outputs(output, output_args, micro_batch, centering)
