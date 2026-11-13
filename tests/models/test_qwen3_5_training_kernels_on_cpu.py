import io
import logging
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

PRECISE = {"fp32_residual": True, "fp32_modules": ["model.visual"], "fp32_modules_tf32": True}


@pytest.mark.parametrize(
    "strategy, mixed_precision",
    [
        ("fsdp", PRECISE),
        ("fsdp2", {**PRECISE, "fp32_residual": False}),
        ("fsdp2", {**PRECISE, "fp32_modules": []}),
        ("fsdp2", {**PRECISE, "fp32_modules_tf32": False}),
    ],
)
def test_qwen3_5_kernels_require_the_precise_fsdp2_actor(strategy, mixed_precision):
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    with pytest.raises(ValueError, match="the GPU backend of Qwen3_5TrainingKernels requires FSDP2"):
        Qwen3_5TrainingKernels.check_config(SimpleNamespace(strategy=strategy, mixed_precision=mixed_precision))
    Qwen3_5TrainingKernels.check_config(SimpleNamespace(strategy="fsdp2", mixed_precision=PRECISE))


def test_qwen3_5_kernels_take_the_dense_model_on_the_packed_fused_torch_forward():
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    dense = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    mixture_of_experts = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))

    Qwen3_5TrainingKernels.check_model(dense, True, True, "torch")
    with pytest.raises(NotImplementedError, match="dense Qwen3.5, not qwen3_5_moe"):
        Qwen3_5TrainingKernels.check_model(mixture_of_experts, True, True, "torch")
    for options in ((False, True, "torch"), (True, False, "torch"), (True, True, "triton"), (True, True, None)):
        with pytest.raises(ValueError, match="use_remove_padding and the torch backend"):
            Qwen3_5TrainingKernels.check_model(dense, *options)


def test_the_kernels_select_the_gpu_backend_of_the_gpu_s_compute_capability():
    from pear.models.qwen3_5.trainer.blackwell.gpu_backend import BlackwellTrainingGpuBackend
    from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    # B200, GB300 (SM100, SM103) and H100
    for capability, gpu_backend in (
        ((10, 0), BlackwellTrainingGpuBackend),
        ((10, 3), BlackwellTrainingGpuBackend),
        ((9, 0), HopperTrainingGpuBackend),
    ):
        assert Qwen3_5TrainingKernels.select_gpu_backend(capability) is gpu_backend
    # Ampere, consumer Blackwell (no tcgen05) and no GPU
    for capability in ((8, 0), (12, 0), (None, None)):
        with pytest.raises(NotImplementedError, match="supports compute capability"):
            Qwen3_5TrainingKernels.select_gpu_backend(capability)


def test_a_batch_rerun_on_bf16_operands_is_logged_whatever_the_root_logger_admits(monkeypatch):
    from pear.models.qwen3_5.trainer import operands

    # in a run the root logger's handlers are at ERROR (vLLM's server module, which every process imports): the
    # warning goes through the handler of its own module
    stream = io.StringIO()
    monkeypatch.setattr(operands.logger.handlers[0], "stream", stream)
    for handler in logging.getLogger().handlers:
        monkeypatch.setattr(handler, "level", logging.ERROR)
    monkeypatch.setattr(operands.Fp16Operands, "range_flags", {})
    runs = []

    def run():
        runs.append(operands.Fp16Operands.is_enabled)
        if len(runs) == 1:  # a conversion of the first run left fp16's range
            operands.Fp16Operands.range_flag("cpu").fill_(1)
        return len(runs)

    assert operands.Fp16Operands.run_checked(run, lambda: None) == 2
    assert runs == [True, False]
    assert "recomputing the batch on bf16 operands" in stream.getvalue()


def test_a_producer_s_buffers_follow_what_autograd_and_the_fp16_operands_ask_for():
    from pear.models.qwen3_5.trainer.operands import Fp16Operands

    class Context(SimpleNamespace):
        def mark_non_differentiable(self, *tensors):
            self.marked = tensors

    # (autograd needs the output, fp16 asked for, the remainder asked for): whether (output, twin, remainder) exist
    cases = {
        (True, True, True): (True, True, True),
        (True, True, False): (True, True, False),
        (False, True, True): (False, True, True),
        (True, False, True): (True, False, False),
        (False, False, False): (True, False, False),
    }
    for (needs_backward, wants_fp16, wants_remainder), written in cases.items():
        ctx = Context(needs_input_grad=(needs_backward, False), marked=())
        buffers = Fp16Operands.output_buffers(ctx, (2, 3), torch.bfloat16, "cpu", wants_fp16, wants_remainder)
        assert tuple(buffer is not None for buffer in buffers) == written
        dtypes = (torch.bfloat16, torch.float16, torch.float16)
        assert all(b.dtype == dtype and b.shape == (2, 3) for b, dtype in zip(buffers, dtypes) if b is not None)

        output, output16, remainder = buffers
        returned = Fp16Operands.twin_outputs(ctx, output, output16, remainder)
        if output is None:  # without autograd the fp16 output is the only one, in the first place
            assert returned[0] is output16 and returned[1] is None and returned[2] is remainder
        else:
            assert all(a is b for a, b in zip(returned, buffers))
            assert output16 is None or any(tensor is output16 for tensor in ctx.marked)


def test_the_recompute_flag_is_set_in_the_checkpoint_s_recompute_only():
    from pear.models.qwen3_5.trainer.blackwell.gpu_backend import BlackwellTrainingGpuBackend
    from pear.models.qwen3_5.trainer.operands import RecomputeFlag

    seen = []

    def layer_forward(x):
        seen.append((RecomputeFlag.in_recompute.get(), RecomputeFlag.saves_discarded()))
        return (x * 2).sin()

    # a decoder layer as transformers' gradient checkpointing leaves it, with verl's attention retention context
    layer = SimpleNamespace(
        _gradient_checkpointing_func=partial(
            checkpoint, use_reentrant=False, context_fn=lambda: (torch.enable_grad(), torch.enable_grad())
        ),
        gpu_backend=BlackwellTrainingGpuBackend,
    )
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
    RecomputeFlag.install(model)

    x = torch.randn(4, requires_grad=True)
    layer._gradient_checkpointing_func(layer_forward, x).sum().backward()
    # the first forward's saves are discarded, the recompute's are the backward's
    assert seen == [(False, True), (True, False)]
    assert not RecomputeFlag.in_recompute.get() and not RecomputeFlag.saves_discarded()


def test_hopper_s_recompute_flags_reject_attention_retention():
    from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
    from pear.models.qwen3_5.trainer.operands import RecomputeFlag

    layer = SimpleNamespace(
        _gradient_checkpointing_func=partial(
            checkpoint, use_reentrant=False, context_fn=lambda: (torch.enable_grad(), torch.enable_grad())
        ),
        gpu_backend=HopperTrainingGpuBackend,
    )
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
    with pytest.raises(
        ValueError, match="HopperTrainingGpuBackend requires gradient_checkpointing_retain_attention=False"
    ):
        RecomputeFlag.install(model)


def test_the_lru_cache_keeps_the_most_recently_used_entries():
    from pear.models.qwen3_5.trainer.blackwell.delta_rule import LRUCache

    cache = LRUCache(2)
    cache["a"], cache["b"] = 1, 2
    assert cache["a"] == 1  # read: now the most recent
    cache["c"] = 3
    assert list(cache) == ["a", "c"]


def test_a_summary_span_rounds_up_to_its_bucket_with_zero_rows():
    from pear.models.qwen3_5.trainer.blackwell.delta_rule import BlackwellDeltaRule

    bucket = BlackwellDeltaRule.SUMMARY_SPAN_BUCKET
    assert [BlackwellDeltaRule.bucket(n) for n in (1, bucket, bucket + 1)] == [bucket, bucket, 2 * bucket]
    x = torch.randn(bucket + 10, 3, 2)
    (padded,) = BlackwellDeltaRule.padded_span((x,), 4, 9)
    assert padded.shape == (bucket, 3, 2)
    assert torch.equal(padded[:5], x[4:9]) and not padded[5:].any()
    (whole,) = BlackwellDeltaRule.padded_span((x,), 10, 10 + bucket)
    assert whole.data_ptr() == x[10:].data_ptr()  # a span of a bucket's length is the rows themselves


def test_every_mlp_s_forward_takes_its_own_projections(monkeypatch):
    from pear.models.qwen3_5.trainer.mlp import Fp32SwiGLUMLP

    layers = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                gate_proj=SimpleNamespace(weight=f"gate {index}"),
                up_proj=SimpleNamespace(weight=f"up {index}"),
                down_proj=SimpleNamespace(weight=f"down {index}"),
            )
        )
        for index in range(2)
    ]
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    # the autograd function's call, replaced by what it is given
    monkeypatch.setattr(Fp32SwiGLUMLP, "apply", lambda *operands: operands)

    Fp32SwiGLUMLP.install(model)

    assert [layer.mlp.forward("x") for layer in layers] == [
        ("x", "gate 0", "up 0", "down 0"),
        ("x", "gate 1", "up 1", "down 1"),
    ]


def test_the_vision_tower_s_linear_layers_share_the_stock_backward():
    from pear.models.qwen3_5.trainer.vision_tower import Bf16x3Linear, Tf32Linear

    torch.manual_seed(0)
    leaves = [torch.randn(shape, requires_grad=True) for shape in ((3, 5, 7), (4, 7), (4,))]

    gradients = torch.autograd.grad(Tf32Linear.apply(*leaves).square().sum(), leaves)
    stock_gradients = torch.autograd.grad(torch.nn.functional.linear(*leaves).square().sum(), leaves)

    assert Bf16x3Linear.backward is Tf32Linear.backward
    for gradient, stock_gradient in zip(gradients, stock_gradients, strict=True):
        torch.testing.assert_close(gradient, stock_gradient)


def test_the_vision_tower_s_forward_runs_with_cudnn_s_tf32_off_and_restores_it():
    from pear.models.qwen3_5.trainer.vision_tower import VisionTower

    seen = []

    class Tower(torch.nn.Module):
        def forward(self, fails=False):
            seen.append((torch.backends.cudnn.allow_tf32, VisionTower.in_forward.get()))
            if fails:
                raise RuntimeError("the tower fails")

    visual = Tower()
    VisionTower.install(SimpleNamespace(model=SimpleNamespace(visual=visual)))
    allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True

    try:
        visual()
        with pytest.raises(RuntimeError, match="the tower fails"):
            visual(fails=True)

        assert seen == [(False, True), (False, True)]
        assert torch.backends.cudnn.allow_tf32 and not VisionTower.in_forward.get()
    finally:
        torch.backends.cudnn.allow_tf32 = allow_tf32
