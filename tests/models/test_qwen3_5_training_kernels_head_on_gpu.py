import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.modeling_outputs import BaseModelOutput

from pear.models.qwen3_5.trainer.forward_with_torch_backend import ForwardWithTorchBackend
from pear.models.qwen3_5.trainer.head import Fp32Head, Fp32HeadLogProbs
from verl.utils.fsdp_utils import MixedPrecisionPolicy, fully_shard
from verl.utils.net_utils import get_free_port

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fp32 head's GEMMs need a GPU")

ROWS = 2 * Fp32HeadLogProbs.CHUNK_ROWS + 37  # three chunks, the last one partial
LOCALHOST = "127.0.0.1"
HIDDEN, VOCAB, IDS = 64, 1000, 5
TEMPERATURE = 0.7
HEAD = 32  # score centering's head: the sampler's top 32 ids, the label among them
REPEATS = 20  # backward calls compared bit for bit
# fp32 sums of 64 exact bf16 products against fp64 (bf16 logits would be off by up to 0.03 at these magnitudes)
FORWARD_ATOL = 1e-4
# the logits' gradient feeds the bf16 GEMMs of the backward, so the gradients agree to bf16's precision
GRAD_RTOL = 1e-2


class Fp64Head:
    """The same log-probs and entropy in fp64 from the same bf16 values, through autograd."""

    def __init__(self, hidden, weight):
        self.hidden = hidden.detach().double().requires_grad_()
        self.weight = weight.detach().double().requires_grad_()

    def __call__(self, ids):
        log_softmax = (self.hidden @ self.weight.t() / TEMPERATURE).log_softmax(-1)
        return log_softmax.gather(1, ids), -(log_softmax.exp() * log_softmax).sum(-1)


@pytest.mark.parametrize("entropy_has_grad", [False, True])
def test_the_fp32_head_matches_fp64(entropy_has_grad):
    torch.manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN, device="cuda").bfloat16().requires_grad_()
    weight = torch.randn(VOCAB, HIDDEN, device="cuda").bfloat16().requires_grad_()
    ids = torch.randint(0, VOCAB, (ROWS, IDS), device="cuda")
    grad_log_probs, grad_entropy = torch.randn(ROWS, IDS, device="cuda"), torch.randn(ROWS, device="cuda")
    reference = Fp64Head(hidden, weight)

    log_probs, entropy = Fp32HeadLogProbs.apply(hidden, weight, ids, TEMPERATURE)
    expected_log_probs, expected_entropy = reference(ids)
    torch.testing.assert_close(log_probs.double(), expected_log_probs, rtol=0, atol=FORWARD_ATOL)
    torch.testing.assert_close(entropy.double(), expected_entropy, rtol=0, atol=FORWARD_ATOL)

    for outputs in ((log_probs, entropy), (expected_log_probs, expected_entropy)):
        loss = (outputs[0] * grad_log_probs).sum()
        if entropy_has_grad:
            loss = loss + (outputs[1] * grad_entropy).sum()
        loss.backward()
    for actual, expected in ((hidden.grad, reference.hidden.grad), (weight.grad, reference.weight.grad)):
        assert actual.dtype == torch.bfloat16
        assert ((actual.double() - expected).norm() / expected.norm()).item() < GRAD_RTOL


def test_the_fp32_head_backward_repeats_bit_for_bit_when_a_label_is_in_the_head():
    torch.manual_seed(2)
    # fp32 inputs keep the backward in fp32 end to end, so a changed order of two adds shows in the gradients
    hidden = torch.randn(ROWS, HIDDEN, device="cuda")
    weight = torch.randn(VOCAB, HIDDEN, device="cuda")
    # each row's top ids as its head and the top one as its label, as for a sampled confident token
    head_ids = (hidden @ weight.t()).topk(HEAD, -1).indices
    ids = torch.cat([head_ids[:, :1], head_ids], -1)
    grad_log_probs = torch.randn(ROWS, HEAD + 1, device="cuda")

    gradients = []
    for _ in range(REPEATS):
        leaves = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        log_probs, _ = Fp32HeadLogProbs.apply(*leaves, ids, TEMPERATURE)
        log_probs.backward(grad_log_probs)
        gradients.append([leaf.grad for leaf in leaves])

    for repeat in gradients[1:]:
        assert all(torch.equal(first, other) for first, other in zip(gradients[0], repeat, strict=True))


def test_the_qwen3_5_fused_forward_returns_the_labels_and_the_head_apart():
    torch.manual_seed(1)
    hidden = torch.randn(1, ROWS, HIDDEN, device="cuda").bfloat16()
    weight = torch.randn(VOCAB, HIDDEN, device="cuda").bfloat16()
    labels = torch.randint(0, VOCAB, (1, ROWS), device="cuda")
    head_ids = torch.randint(0, VOCAB, (1, ROWS, IDS), device="cuda")
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    head.weight.data.copy_(weight)
    Fp32Head.install(head)
    model = SimpleNamespace(model=lambda input_ids, **kwargs: BaseModelOutput(last_hidden_state=hidden), lm_head=head)

    output = ForwardWithTorchBackend.forward(model, input_ids=labels, shift_labels=labels, head_ids=head_ids)

    # the labels' log-probs are gathered over the SP group, which needs them contiguous
    assert output.log_probs.shape == labels.shape and output.log_probs.is_contiguous()
    expected, _ = Fp32HeadLogProbs.apply(hidden[0], weight, torch.cat([labels[0, :, None], head_ids[0]], -1), 1.0)
    torch.testing.assert_close(output.log_probs[0], expected[:, 0])
    torch.testing.assert_close(output.head_log_probs[0], expected[:, 1:])


def two_rank_head_gradient(rank: int, port: int) -> None:
    """One FSDP2 rank of an lm_head unit: the fp32 head reads the gathered bf16 weight and its gradient is reduced."""
    os.environ.update(MASTER_ADDR=LOCALHOST, MASTER_PORT=str(port))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=device)
    torch.manual_seed(0)  # the same fp32 weight on both ranks
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, device=device)
    weight = head.weight.detach().clone()
    Fp32Head.install(head)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(head, mesh=init_device_mesh("cuda", (2,)), mp_policy=policy)
    torch.manual_seed(1 + rank)  # different rows on each rank
    hidden = torch.randn(Fp32HeadLogProbs.CHUNK_ROWS, HIDDEN, device=device).bfloat16()
    ids = torch.randint(0, VOCAB, (Fp32HeadLogProbs.CHUNK_ROWS, IDS), device=device)

    log_probs, _ = head(hidden, ids=ids, temperature=TEMPERATURE)
    log_probs.sum().backward()

    # the gathered weight is the bf16 one, and the gradient is the ranks' mean (FSDP's reduce-scatter)
    expected_log_probs, _ = Fp32HeadLogProbs.apply(hidden, weight.bfloat16(), ids, TEMPERATURE)
    torch.testing.assert_close(log_probs, expected_log_probs)
    reference = Fp64Head(hidden, weight.bfloat16())
    reference(ids)[0].sum().backward()
    expected = reference.weight.grad.float()
    dist.all_reduce(expected)
    expected /= dist.get_world_size()
    actual = head.weight.grad.full_tensor().double()
    assert ((actual - expected.double()).norm() / expected.norm()).item() < GRAD_RTOL
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two ranks")
def test_the_fp32_head_of_an_fsdp_unit_reduces_its_gradient_on_two_ranks():
    torch.multiprocessing.spawn(two_rank_head_gradient, args=(get_free_port(LOCALHOST)[0],), nprocs=2)
