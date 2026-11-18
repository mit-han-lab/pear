import itertools

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import get_device_capability

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Ulysses over two GPUs")

SP = 2
TEXT_HIDDEN = 512
VOCAB = 1000
LENGTHS = [900, 1700, 402]  # the second sequence crosses the ranks' boundary (3002 tokens, 1501 per rank)
M_ROPE_ROWS = 4  # Qwen3.5's position ids: the text row and the temporal, height and width rows
LOGIT_SCALE = 4.0  # the lm_head's weights times this: label log-probs that spread over several nats
# two layouts of the same sequences against each other, relative to their norm: the kernels' rounding stays far below,
# a mixer that reads across a sequence boundary or loses the state of the rank before it moves whole sequences
LAYOUT_TOLERANCE = 1e-2
GRAD_TOLERANCE = 2e-2  # the parameters' gradients: the backward's bf16 operands, the ranks' shares summed in bf16


def rel_err(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def gathered(tensor):
    """The ranks' sequence slices [1, T / SP, ...] concatenated."""
    parts = [torch.empty_like(tensor) for _ in range(SP)]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=1)


def tiny_model(gpu_backend):
    """A Qwen3.5 of a Gated DeltaNet layer and a full attention layer, built as the PEAR engine builds the actor: verl's
    patches and checkpointing, then the family's training kernels with the GPU's backend, the parameters in the dtypes
    FSDP2 gathers them in."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels
    from verl.models.transformers.attention_retention import retain_attention_in_checkpoints
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.fsdp_utils import _autocast_disabled

    config = Qwen3_5Config(
        text_config={
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "vocab_size": VOCAB,
            "layer_types": ["linear_attention", "full_attention"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": TEXT_HIDDEN,
        },
    )
    # the GPU backend's attention for the language model, SDPA for the fp32 vision tower (verl's FSDP2 wrap sets it)
    config._attn_implementation = {"text_config": gpu_backend.attention_cls.implementation, "vision_config": "sdpa"}
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config)

    # unit-gain weights: every mixer moves the residual stream
    text_model = model.model.language_model
    with torch.no_grad():
        text_model.embed_tokens.weight.normal_()
        for module in text_model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0, module.in_features**-0.5)
        model.lm_head.weight.normal_(0, LOGIT_SCALE * TEXT_HIDDEN**-0.5)

    apply_monkey_patch(
        model, ulysses_sp_size=SP, use_remove_padding=True, use_fused_kernels=True, fused_kernels_backend="torch"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if gpu_backend.recompute_is_fp16:  # the recipes retain the attention where the recompute runs the fp16 forward
        retain_attention_in_checkpoints(model)

    Qwen3_5TrainingKernels.install(model, fp32_residual=True, use_fused_kernels=True, fused_kernels_backend="torch")
    Qwen3_5TrainingKernels.check_model(model, True, True, "torch")
    Qwen3_5TrainingKernels.install_gpu_backend(model)
    Qwen3_5TrainingKernels.install_checkpointing(model)

    model.cuda()
    text_model.bfloat16()
    model.lm_head.bfloat16()
    model.model.visual.forward = _autocast_disabled(model.model.visual.forward)
    torch.backends.cuda.matmul.allow_tf32 = True  # mixed_precision.fp32_modules_tf32
    torch.backends.cudnn.allow_tf32 = True
    return model


def forward(model, input_ids, cu_seqlens, shift_labels, with_hidden_states):
    """The fused forward as verl's engine calls it on packed rows: (label log-probs, entropy, final hidden states)."""
    from pear.models.qwen3_5.trainer.operands import Fp16Operands

    positions = torch.cat([torch.arange(n, device="cuda") for n in cu_seqlens.diff().tolist()])
    Fp16Operands.reset()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=positions.expand(M_ROPE_ROWS, 1, -1).contiguous(),  # text only: every row the same
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens.cpu(),
            shift_labels=shift_labels,
            temperature=1.0,
            return_dict=True,
            output_hidden_states=with_hidden_states,
        )

    # every operand within fp16's range: the outputs are the fp16 forward's, which the engine would otherwise rerun
    assert not any(bool(flag.item()) for flag in Fp16Operands.range_flags.values())
    return output.log_probs, output.entropy, output.hidden_states[-1] if with_hidden_states else None


def layout_checks(rank: int, model) -> None:
    """The batch's sequences one at a time, packed, and packed under Ulysses: the same log-probs, entropies and final
    hidden states (the log-probs pass, without autograd), and with autograd the same parameter gradients."""
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group

    sequences = [torch.randint(VOCAB, (1, n), generator=torch.Generator().manual_seed(n)).cuda() for n in LENGTHS]
    input_ids = torch.cat(sequences, dim=1)
    labels = torch.cat([sequence.roll(-1, dims=1) for sequence in sequences], dim=1)  # each sequence's own next tokens
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], device="cuda")
    T = sum(LENGTHS)
    local = slice(rank * T // SP, (rank + 1) * T // SP)

    model.eval()
    with torch.no_grad():
        set_ulysses_sequence_parallel_group(None)
        one_at_a_time = [
            forward(model, sequence, cu.new_tensor([0, sequence.shape[1]]), sequence.roll(-1, dims=1), True)
            for sequence in sequences
        ]
        packed = forward(model, input_ids, cu, labels, True)
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        split = [gathered(output) for output in forward(model, input_ids, cu, labels[:, local], True)]

    expected = [torch.cat(outputs, dim=1) for outputs in zip(*one_at_a_time, strict=True)]
    for layout in (packed, split):
        for actual, reference in zip(layout, expected, strict=True):
            assert rel_err(actual, reference) < LAYOUT_TOLERANCE

    # the update: forward and backward through the checkpointed layers, a fixed weight per token
    text_model = model.model.language_model
    linear_layer, full_layer = text_model.layers
    parameters = (
        model.lm_head.weight,
        text_model.embed_tokens.weight,
        linear_layer.linear_attn.in_proj_qkv.weight,
        linear_layer.linear_attn.in_proj_a.weight,
        linear_layer.linear_attn.conv1d.weight,
        full_layer.self_attn.q_proj.weight,
        full_layer.mlp.down_proj.weight,
    )
    token_weights = torch.randn(1, T, generator=torch.Generator().manual_seed(T)).cuda()
    model.train()

    def update(shift_labels, weights):
        model.zero_grad(set_to_none=True)
        log_probs, _, _ = forward(model, input_ids, cu, shift_labels, False)
        (log_probs * weights).sum().backward()
        return log_probs.detach(), [parameter.grad.clone() for parameter in parameters]

    set_ulysses_sequence_parallel_group(None)
    packed_log_probs, packed_grads = update(labels, token_weights)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    split_log_probs, split_grads = update(labels[:, local], token_weights[:, local])

    assert rel_err(gathered(split_log_probs), packed_log_probs) < LAYOUT_TOLERANCE
    for split_grad, packed_grad in zip(split_grads, packed_grads, strict=True):
        dist.all_reduce(split_grad)  # each rank's backward holds its own tokens' share
        assert rel_err(split_grad, packed_grad) < GRAD_TOLERANCE


def two_rank_checks(rank: int, rendezvous: str) -> None:
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=SP)
    try:
        gpu_backend = Qwen3_5TrainingKernels.select_gpu_backend(get_device_capability())
        layout_checks(rank, tiny_model(gpu_backend))
    finally:
        dist.destroy_process_group()


def test_the_model_s_outputs_do_not_depend_on_the_batch_s_layout(tmp_path):
    # the ranks meet at a file of the test's own: a free port picked beforehand can be taken by another process
    # before rank 0 binds it
    torch.multiprocessing.spawn(two_rank_checks, args=(str(tmp_path / "rendezvous"),), nprocs=SP)
