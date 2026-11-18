import math

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

from pear.rollout.top_p import ExactMaxTopP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the top-p kernels need a GPU")

DEVICE = torch.device("cuda")
VOCAB = 248320  # Qwen3.5's vocabulary
ROWS = 256  # a batch past vLLM's split pipeline (64 rows): the launch of its monolithic kernel
TOP_PS = (0.5, 0.9, 0.95, 0.99)
TOP_P = 0.95  # the rollouts' validation and evaluation top-p
SEED = 0
# vLLM's tolerance of its Triton top-p's kept counts against its sort reference: a few tokens, or a fraction of the
# largest count
MAX_COUNT_DIFFERENCE = 3
MAX_COUNT_FRACTION = 0.005
TAIL_MEAN, TAIL_STD = -1.4, 2.4  # a near-one-hot row's other logits
# the near-one-hot rows' top token: its fp64 probability from 1 - 4e-8 to 1 - 6e-9, its logit that far above the
# others' log-sum-exp (log of p / (1 - p))
PEAK_MARGINS = (17.0, 19.0)
FLOOR_LOGIT = -1e4  # every other token of a few-token row: probability 0 in fp32, yet finite as a model's logits are
FLOOR_ROW = (0.2, 0.4, 0.37, 0.03)  # that row's tokens' probabilities: top-p 0.95 keeps the first three


@pytest.fixture(scope="module", autouse=True)
def exact_max_top_p():
    ExactMaxTopP.install()


class TopPMasks:
    """The tokens of each row that vLLM's Triton top-p keeps (PEAR's exact-max copy for a batch without top-k), and
    those of its exact nucleus."""

    @staticmethod
    def kept(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """The tokens the Triton top-p keeps."""
        p = torch.full((logits.shape[0],), top_p, device=DEVICE)
        return apply_top_k_top_p_triton(logits.clone(), None, p) > -math.inf

    @staticmethod
    def nucleus(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Each row's exact nucleus in fp64: the fewest top tokens whose mass reaches top-p."""
        probabilities = torch.softmax(logits.double(), dim=1)
        sorted_probabilities, order = probabilities.sort(dim=1, descending=True)
        mass_above = sorted_probabilities.cumsum(dim=1) - sorted_probabilities
        return torch.zeros_like(logits, dtype=torch.bool).scatter_(1, order, mass_above < top_p)


@pytest.mark.parametrize("top_p", TOP_PS)
def test_the_exact_max_top_p_keeps_as_many_tokens_as_vllm_s_sort_reference_on_random_rows(top_p):
    # vLLM's own test's rows
    generator = torch.Generator(DEVICE).manual_seed(SEED)
    logits = torch.randn(ROWS, VOCAB, generator=generator, device=DEVICE)
    p = torch.full((ROWS,), top_p, device=DEVICE)

    counts = TopPMasks.kept(logits, top_p).sum(dim=1)

    # vLLM's criterion for its Triton top-p (vLLM 0.34's tests/v1/sample/test_topk_topp_sampler.py:196-205): the
    # kernel's pivot search stops after a fixed number of halvings, inside a flat row's boundary
    reference_counts = (apply_top_k_top_p_pytorch(logits.clone(), None, p) > -math.inf).sum(dim=1)
    difference = (counts - reference_counts).abs().max().item()
    assert difference <= MAX_COUNT_DIFFERENCE or difference < MAX_COUNT_FRACTION * reference_counts.max().item()


def test_a_near_one_hot_row_keeps_its_top_token_alone():
    # vLLM's kernel kept the whole vocabulary of such rows (an fp32 top probability above its search's sentinel 1.0)
    generator = torch.Generator(DEVICE).manual_seed(SEED)
    logits = torch.randn(ROWS, VOCAB, generator=generator, device=DEVICE) * TAIL_STD + TAIL_MEAN
    tail = logits.double().logsumexp(dim=1)
    margins = torch.linspace(*PEAK_MARGINS, ROWS, device=DEVICE, dtype=torch.float64)
    top = torch.randint(0, VOCAB, (ROWS,), generator=generator, device=DEVICE)
    logits[torch.arange(ROWS, device=DEVICE), top] = (tail + margins).float()

    kept = TopPMasks.kept(logits, TOP_P)

    assert torch.equal(kept, TopPMasks.nucleus(logits, TOP_P))
    assert torch.equal(kept.nonzero()[:, 1], top)


def test_a_floor_of_large_negative_logits_does_not_overflow_the_softmax():
    # vLLM's kernel shifted its softmax by the first block's mean + 10 std, which a floor of -1e4 logits pulls far
    # below the kept tokens: exp overflowed and the mask let through the whole vocabulary
    logits = torch.full((ROWS, VOCAB), FLOOR_LOGIT, device=DEVICE)
    # four tokens across the vocabulary's tiles, the last outside the nucleus
    tokens = torch.arange(len(FLOOR_ROW), device=DEVICE) * (VOCAB - 1) // (len(FLOOR_ROW) - 1)
    logits[:, tokens] = torch.tensor(FLOOR_ROW, device=DEVICE).log()

    kept = TopPMasks.kept(logits, TOP_P)

    assert torch.equal(kept, TopPMasks.nucleus(logits, TOP_P))
    assert torch.equal(kept.sum(dim=1), torch.full((ROWS,), len(FLOOR_ROW) - 1, device=DEVICE))
