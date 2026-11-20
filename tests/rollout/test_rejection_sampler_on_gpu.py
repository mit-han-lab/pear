import types

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import expand_idx_mapping
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

from pear.rollout.top_p import ExactMaxTopP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rejection sampler's kernels need a GPU")

# (target, draft, top-p): every verified row's distribution over the toy tokens and the drafts' distribution; the
# second target has a least token that top-p 0.95 cuts, which its draft proposes often
TOY_CASES = {
    "temperature_1": ((0.2, 0.4, 0.4), (0.6, 0.3, 0.1), 1.0),
    "top_p_0.95": ((0.2, 0.4, 0.37, 0.03), (0.5, 0.2, 0.1, 0.2), 0.95),
}
FALSE_ALARM = 1e-6  # the chance that an exact sampler fails a position's goodness-of-fit check
MIN_EXPECTED_COUNT = 5  # the least expected count of every kept token for a position's check


class ToyVerification:
    """Verify steps through vLLM's V2 rejection sampler over vLLM's V2 sampler, as the V2 runner builds them when the
    model state brings no sampler of its own: requests at temperature 1 whose every target row holds one toy
    distribution spread over the model's vocabulary, drafted from another, each moving on by the tokens it emitted."""

    DEVICE = torch.device("cuda")
    VOCAB = 248320  # Qwen3.8-27B's vocabulary: the toy tokens lie in different vocabulary blocks of the kernels
    REQUESTS = 256  # a verify step's requests, the replicas' max_num_seqs
    VERIFY_STEPS = 400_000  # per case, over all requests
    PROMPT_TOKENS = 16  # each request's prompt, prefilled before its first verify step
    TEMPERATURE = 1.0
    OUTSIDE_LOGIT = -1e4  # every other token's logit: probability 0 in fp32, yet finite as a model's logits are
    DRAW_SEED = 1234  # the drafts' generator, independent of the sampler's noise
    # the speculative config's fields the rejection sampler reads, at vLLM's defaults: standard rejection sampling, the
    # fly options read by its fly method alone
    SPEC_CONFIG = {
        "enable_adaptive_verification": False,
        "rejection_sample_method": "standard",
        "fly_entropy_threshold": 0.3,
        "fly_entropy_top_k": 3,
    }

    def __init__(self, target: tuple, draft: tuple, top_p: float, draft_tokens: int, is_probabilistic: bool) -> None:
        self.target = torch.tensor(target, dtype=torch.float64)
        self.draft = torch.tensor(draft, dtype=torch.float64)
        self.top_p = top_p
        self.draft_tokens = draft_tokens
        # the drafts' logits cached for the verification, as a probabilistic drafter leaves them; else greedy drafts
        self.is_probabilistic = is_probabilistic
        self.rounds = -(-self.VERIFY_STEPS // self.REQUESTS)
        self.step_positions = torch.arange(draft_tokens + 1, device=self.DEVICE)

        # the toy tokens from the vocabulary's first id to its last, and each id's toy token (the others' bin last)
        size = len(target)
        self.token_ids = torch.tensor(
            [index * (self.VOCAB - 1) // (size - 1) for index in range(size)], device=self.DEVICE
        )
        self.toy_index = torch.full((self.VOCAB,), size, device=self.DEVICE)
        self.toy_index[self.token_ids] = torch.arange(size, device=self.DEVICE)

    def run(self) -> torch.Tensor:
        """How often each toy token was emitted at each position of a verify step, the last column every other token:
        [draft tokens + 1, toy tokens + 1]."""
        rejection_sampler, slots = self.rejection_sampler()
        logits = self.target_logits()
        draft_logits = self.draft_logits() if self.is_probabilistic else None
        layout = self.batch_layout(slots)
        generator = torch.Generator(device=self.DEVICE).manual_seed(self.DRAW_SEED)
        starts = torch.full((self.REQUESTS,), self.PROMPT_TOKENS, dtype=torch.int64, device=self.DEVICE)
        counts = torch.zeros(self.draft_tokens + 1, len(self.target) + 1, dtype=torch.int64, device=self.DEVICE)

        for round_index in range(self.rounds):
            output = rejection_sampler(logits, self.verify_batch(layout, starts, generator, round_index), draft_logits)
            self.count(counts, output)
            # a request's next step starts after the tokens it emitted
            starts += output.num_sampled

        return counts.cpu()

    def nucleus(self) -> torch.Tensor:
        """The distribution every emitted token must follow: the target's most probable tokens while the mass ranked
        above a token is below top-p, renormalized."""
        order = torch.argsort(self.target, descending=True)
        mass_above = torch.cumsum(self.target[order], 0) - self.target[order]
        is_kept = torch.zeros_like(self.target, dtype=torch.bool)
        is_kept[order] = mass_above < self.top_p
        nucleus = torch.where(is_kept, self.target, 0.0)

        return nucleus / nucleus.sum()

    def rejection_sampler(self) -> tuple[RejectionSampler, np.ndarray]:
        """The requests added to vLLM's request state and sampler, the rejection sampler the V2 runner builds over the
        sampler, and the requests' slots."""
        max_tokens = self.rounds * (self.draft_tokens + 1)
        requests = RequestState(
            max_num_reqs=self.REQUESTS,
            max_model_len=self.PROMPT_TOKENS + max_tokens,
            max_num_batched_tokens=self.REQUESTS * (self.draft_tokens + 1),
            num_speculative_steps=self.draft_tokens,
            vocab_size=self.VOCAB,
            device=self.DEVICE,
        )
        # the runner's decode query length: a request's last token and its drafts
        sampler = Sampler(
            VllmConfig(), self.REQUESTS, self.VOCAB, self.DEVICE, requests, num_speculative_tokens=self.draft_tokens + 1
        )

        slots = []
        for index in range(self.REQUESTS):
            request_id = f"toy-{index}"
            requests.add_request(
                request_id, self.PROMPT_TOKENS, [0] * self.PROMPT_TOKENS, self.PROMPT_TOKENS, max_tokens
            )
            slots.append(requests.req_id_to_index[request_id])
            sampler.add_request(slots[-1], SamplingParams(temperature=self.TEMPERATURE, top_p=self.top_p, seed=index))
        requests.apply_staged_writes()
        sampler.apply_staged_writes()

        spec_config = types.SimpleNamespace(num_speculative_tokens=self.draft_tokens, **self.SPEC_CONFIG)
        return RejectionSampler(sampler, spec_config, self.DEVICE), np.array(slots)

    def target_logits(self) -> torch.Tensor:
        """Every verify row's logits: the target's log-probabilities at the toy tokens, OUTSIDE_LOGIT elsewhere."""
        row = torch.full((self.VOCAB,), self.OUTSIDE_LOGIT, device=self.DEVICE)
        row[self.token_ids] = self.target.log().float().to(self.DEVICE)

        return row.expand(self.REQUESTS * (self.draft_tokens + 1), self.VOCAB).contiguous()

    def draft_logits(self) -> DraftLogitsCache:
        """The drafts' logits as a probabilistic drafter caches them for the verification: the draft's
        log-probabilities at every request's every step."""
        cache = DraftLogitsCache(
            self.REQUESTS,
            self.draft_tokens,
            self.VOCAB,
            dtype=torch.float32,
            fill=self.OUTSIDE_LOGIT,
            device=self.DEVICE,
        )
        cache.values[:, :, self.token_ids] = self.draft.log().float().to(self.DEVICE)

        return cache

    def batch_layout(self, slots: np.ndarray) -> dict:
        """The input batch's fields every verify step shares: each request's rows, by the V2 runner's own expansion."""
        rows_per_request = self.draft_tokens + 1
        cu_num_logits_np = np.arange(self.REQUESTS + 1, dtype=np.int32) * rows_per_request
        cu_num_logits = torch.from_numpy(cu_num_logits_np).to(self.DEVICE)
        idx_mapping = torch.from_numpy(slots).to(self.DEVICE)
        expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
            idx_mapping, self.REQUESTS * rows_per_request, cu_num_logits, rows_per_request
        )

        return {
            "num_reqs": self.REQUESTS,
            "idx_mapping": idx_mapping,
            "idx_mapping_np": slots,
            "cu_num_logits": cu_num_logits,
            "cu_num_logits_np": cu_num_logits_np,
            "expanded_idx_mapping": expanded_idx_mapping,
            "expanded_local_pos": expanded_local_pos,
            "logits_indices": torch.arange(self.REQUESTS * rows_per_request, device=self.DEVICE),
        }

    def verify_batch(self, layout: dict, starts: torch.Tensor, generator: torch.Generator, round_index: int):
        """A verify step's input batch: each request's last token and its drafts, drawn from the draft distribution, at
        the positions from its start."""
        draws = torch.multinomial(
            self.draft.float().to(self.DEVICE), self.REQUESTS * self.draft_tokens, replacement=True, generator=generator
        )
        input_ids = torch.zeros(self.REQUESTS, self.draft_tokens + 1, dtype=torch.int32, device=self.DEVICE)
        input_ids[:, 1:] = self.token_ids[draws].view(self.REQUESTS, self.draft_tokens)
        # the runner's CPU copy of the lengths is an upper bound of them under asynchronous scheduling
        length_bound = self.PROMPT_TOKENS + (round_index + 1) * (self.draft_tokens + 1)

        return types.SimpleNamespace(
            **layout,
            input_ids=input_ids.flatten(),
            positions=(starts[:, None] + self.step_positions).flatten(),
            seq_lens=(starts + self.draft_tokens + 1).to(torch.int32),
            seq_lens_cpu_upper_bound=torch.full((self.REQUESTS,), length_bound, dtype=torch.int32),
        )

    def count(self, counts: torch.Tensor, output) -> None:
        """A step's emitted tokens added to the counts by position, a token out of the vocabulary as another token."""
        tokens = output.sampled_token_ids
        is_emitted = self.step_positions < output.num_sampled[:, None]
        is_in_vocabulary = (tokens >= 0) & (tokens < self.VOCAB)
        bins = torch.where(is_in_vocabulary, self.toy_index[tokens.clamp(0, self.VOCAB - 1)], len(self.target))

        for position in range(self.draft_tokens + 1):
            counts[position] += torch.bincount(bins[is_emitted[:, position], position], minlength=len(self.target) + 1)

    @staticmethod
    def p_value(counts: torch.Tensor, probabilities: torch.Tensor) -> float:
        """The chance that draws from the probabilities land at least as far from them as the counts, by Pearson's
        chi-squared test."""
        expected = probabilities * counts.sum()
        statistic = ((counts - expected) ** 2 / expected).sum()
        half_degrees = torch.tensor((len(counts) - 1) / 2, dtype=torch.float64)

        return torch.special.gammaincc(half_degrees, statistic / 2).item()


@pytest.fixture(scope="module", autouse=True)
def exact_max_top_p():
    # the rollout workers' top-p, which PearVLLMWorkerExtension installs over vLLM's
    ExactMaxTopP.install()


@pytest.mark.parametrize("draft_tokens", [3, 7], ids=["T4", "T8"])
@pytest.mark.parametrize("is_probabilistic", [False, True], ids=["greedy_draft", "probabilistic_draft"])
@pytest.mark.parametrize("case", TOY_CASES)
def test_every_verified_position_emits_the_target_distribution(case, is_probabilistic, draft_tokens):
    target, draft, top_p = TOY_CASES[case]
    verification = ToyVerification(target, draft, top_p, draft_tokens, is_probabilistic)

    counts = verification.run()

    nucleus = verification.nucleus()
    is_kept = nucleus > 0
    # the first draft position, which most verify steps reach, has the draws of a check
    assert counts[1].sum() * nucleus[is_kept].min() >= MIN_EXPECTED_COUNT
    for position, position_counts in enumerate(counts):
        # a token outside the nucleus, cut by top-p or any other of the vocabulary, is never emitted
        outside = position_counts[-1] + position_counts[:-1][~is_kept].sum()
        assert outside == 0, f"position {position}: {outside} tokens outside the target's nucleus"
        kept_counts = position_counts[:-1][is_kept].double()
        if kept_counts.sum() * nucleus[is_kept].min() < MIN_EXPECTED_COUNT:
            continue
        p_value = ToyVerification.p_value(kept_counts, nucleus[is_kept])
        assert p_value > FALSE_ALARM, (
            f"position {position}: counts {kept_counts.tolist()} against {nucleus[is_kept].tolist()}, p {p_value:.2e}"
        )
