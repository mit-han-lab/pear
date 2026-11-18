import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    gather_draft_sampled,
    unconditional_to_conditional_rates,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

from nonfinite_rows import NonfiniteRows
from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.sampler import RolloutRejectionSampler

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

VOCAB = 248320  # Qwen3.5's vocabulary: the verify rows' width
TOPK = 32  # the processed log-probs the rollout keeps
SEEDS = [11, 12, 13, 14, 15]  # the distribution tests' independent repetitions
MAX_REQUESTS = 128  # request-state slots, at least
BONUS_DRAFT = 12345  # a draft its row accepts, before a bonus row
DOMINANT_LOGIT = 60.0  # that draft's logit: its probability rounds to 1 in fp32
NONFINITE_REQUESTS = ["a-finite-request", "the-request"]  # the step's requests: random rows, then rows of a kind


class VerifyBatch:
    """A verify step's input batch as vLLM's V2 runner hands it to the rejection sampler: n requests of k + 1 logit
    rows (the last token's and the drafts'), random logits whose drafts the target likes to a random degree."""

    def __init__(self, n: int, k: int, generator: torch.Generator, vocab: int = VOCAB, temperature: float = 1.0):
        device = torch.device("cuda")
        rows = k + 1
        self.n, self.k, self.vocab = n, k, vocab
        self.num_logits = n * rows
        self.logits = torch.randn(self.num_logits, vocab, generator=generator, device=device) * 2
        # each row's draft (the next row's input) between unlikely and almost certain under the target
        drafts = torch.randint(0, vocab, (n, rows), generator=generator, device=device)
        boost = torch.rand(n, rows, generator=generator, device=device) * 14 + 6
        row_index = torch.arange(self.num_logits, device=device)
        self.logits[row_index[:-1], drafts.flatten()[1:]] += boost.flatten()[:-1]

        slots = max(MAX_REQUESTS, n)
        self.idx_mapping = torch.randperm(slots, generator=generator, device=device)[:n]
        self.expanded_idx_mapping = self.idx_mapping.repeat_interleave(rows)
        self.expanded_local_pos = torch.arange(rows, dtype=torch.int32, device=device).repeat(n)
        self.cu_num_logits = torch.arange(0, self.num_logits + 1, rows, dtype=torch.int32, device=device)
        self.cu_num_logits_np = self.cu_num_logits.cpu().numpy()
        starts = torch.randint(1000, 100000, (n,), generator=generator, device=device)
        self.positions = (starts.repeat_interleave(rows) + self.expanded_local_pos).long()
        self.logits_indices = torch.arange(self.num_logits, dtype=torch.int64, device=device)
        self.input_ids = drafts.flatten().int()
        self.seq_lens = (starts + rows).int()
        self.prefill_len = torch.zeros(slots, dtype=torch.int32, device=device)
        self.prefill_len[self.idx_mapping] = (starts - 10).int()
        self.temperature = torch.full((slots,), temperature, device=device)
        self.seeds = torch.randint(-(2**62), 2**62, (slots,), generator=generator, device=device)

    def kernels(self, rates: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """The fused kernels' sampled tokens, counts, log-prob ids, log-probs, ranks and rows' NaN and +inf counts."""
        rates = rates if rates is not None else torch.empty(0, device=self.logits.device)
        return torch.ops.pear_rollout_blackwell.verify_sample(
            self.logits,
            self.input_ids,
            self.positions,
            self.logits_indices,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.cu_num_logits,
            self.idx_mapping,
            self.seq_lens,
            self.prefill_len,
            self.temperature,
            self.seeds,
            rates,
            TOPK,
            self.k + 1,
        )

    def vllm(self, rates: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """vLLM's rejection sampler on the same batch: sampled tokens, counts, and the top-K head of every row."""
        draft_sampled, pos = gather_draft_sampled(
            self.input_ids,
            self.positions,
            self.logits_indices,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.prefill_len,
        )
        sampled, num_sampled = rejection_sample(
            self.logits,
            None,
            draft_sampled,
            self.cu_num_logits,
            pos,
            self.idx_mapping,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.seeds,
            self.k,
            rates,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled, self.seq_lens, self.cu_num_logits, self.idx_mapping, self.prefill_len
        )
        head = compute_topk_scores(self.logits, TOPK, torch.zeros_like(self.logits_indices))
        return sampled, num_sampled, num_rejected, head.logprob_token_ids, head.logprobs


class TestVerifySampler:
    """The fused verify sampler against vLLM's rejection sampler and fp64 references on identical inputs."""

    @classmethod
    def setup_class(cls):
        BlackwellExtension.load()

    @pytest.mark.parametrize("k", [3, 7])
    @pytest.mark.parametrize("is_synthetic", [True, False])
    def test_acceptance_matches_vllm(self, k, is_synthetic):
        # the same u = tl.rand(seed, pos) as vLLM's kernels: the same accepted drafts and counts, row by row
        generator = torch.Generator("cuda").manual_seed(1)
        batch = VerifyBatch(64, k, generator)
        rates = None
        if is_synthetic:
            rates = torch.tensor(unconditional_to_conditional_rates([0.64, 0.42, 0.29, 0.17, 0.11, 0.08, 0.05][:k]))
            rates = rates.float().cuda()
        sampled, num_sampled, num_rejected, _, _, _, nonfinite = batch.kernels(rates)
        ref_sampled, ref_num_sampled, ref_num_rejected, _, _ = batch.vllm(rates)
        torch.cuda.synchronize()

        assert (nonfinite == 0).all()
        assert torch.equal(num_sampled, ref_num_sampled.int())
        assert torch.equal(num_rejected, ref_num_rejected.int())
        for i, count in enumerate(num_sampled.tolist()):
            assert torch.equal(sampled[i, : count - 1], ref_sampled[i, : count - 1])
        accepted = (num_sampled - 1).float().mean().item()
        assert 0.2 < accepted < k - 0.2, f"degenerate acceptance {accepted}"

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_head_against_fp64(self, temperature):
        # every row's top-K processed log-probs, and the kept tokens' log-probs and ranks
        generator = torch.Generator("cuda").manual_seed(2)
        batch = VerifyBatch(32, 3, generator, temperature=temperature)
        sampled, num_sampled, _, ids, logprobs, ranks, _ = batch.kernels(None)
        torch.cuda.synchronize()

        scaled = batch.logits / temperature
        reference = torch.log_softmax(scaled.double(), dim=-1)
        top = reference.topk(TOPK, dim=-1)
        assert torch.equal(ids[:, 1:].long().sort(dim=-1).values, top.indices.sort(dim=-1).values)
        assert (logprobs[:, 1:].double() - top.values).abs().max().item() < 1e-5
        assert (logprobs[:, 1:].diff(dim=-1) <= 0).all()

        for i, count in enumerate(num_sampled.tolist()):
            for j in range(count):
                row, token = i * 4 + j, sampled[i, j].item()
                assert ids[row, 0].item() == token
                assert abs(logprobs[row, 0].item() - reference[row, token].item()) < 1e-5
                assert ranks[row].item() == (scaled[row] >= scaled[row, token]).sum().item()

    def test_head_matches_vllm(self):
        # the top-K head of vLLM's rejection sampler (compute_topk_scores) on the same logits
        generator = torch.Generator("cuda").manual_seed(3)
        batch = VerifyBatch(64, 3, generator)
        _, _, _, ids, logprobs, _, _ = batch.kernels(None)
        _, _, _, ref_ids, ref_logprobs = batch.vllm(None)
        torch.cuda.synchronize()

        assert torch.equal(ids[:, 1:].long(), ref_ids[:, 1:])
        assert (logprobs[:, 1:] - ref_logprobs[:, 1:]).abs().max().item() < 1e-5

    def test_prefill_and_placeholders_match_vllm(self):
        # a chunked prefill samples nothing; a request whose drafts are placeholders (positions inside its prefill)
        # rejects them and draws from the target's row
        generator = torch.Generator("cuda").manual_seed(4)
        batch = VerifyBatch(16, 3, generator)
        batch.seq_lens[:4] = batch.prefill_len[batch.idx_mapping[:4]] - 1
        batch.prefill_len[batch.idx_mapping[4:8]] = batch.positions[4 * 4 : 8 * 4 : 4].int() + 2
        sampled, num_sampled, num_rejected, _, _, _, _ = batch.kernels(None)
        ref_sampled, ref_num_sampled, ref_num_rejected, _, _ = batch.vllm(None)
        torch.cuda.synchronize()

        assert torch.equal(num_sampled, ref_num_sampled.int())
        assert torch.equal(num_rejected, ref_num_rejected.int())
        assert (num_sampled[:4] == 0).all()
        assert (num_sampled[4:8] <= 2).all()

    @pytest.mark.parametrize("kind", NonfiniteRows.KINDS)
    @pytest.mark.parametrize("position", ["verifying", "bonus"])
    def test_corrupted_rows_are_counted_and_the_other_rows_verify_as_vllm(self, kind, position):
        # two requests of two rows: the first's random, the second's last in the logits' allocation, both of the kind
        # (verifying its draft), or the bonus row of the kind behind a row that accepts its draft
        generator = torch.Generator("cuda").manual_seed(5)
        batch = VerifyBatch(2, 1, generator)
        row = NonfiniteRows.row(kind, VOCAB, generator)
        if position == "verifying":
            second_rows, corrupted_rows = [row, row], [2, 3]
        else:
            first_row = batch.logits[2].clone()
            first_row[BONUS_DRAFT] = DOMINANT_LOGIT
            batch.input_ids[3] = BONUS_DRAFT
            second_rows, corrupted_rows = [first_row, row], [3]
        batch.logits = torch.cat([batch.logits[:2], torch.stack(second_rows)])
        sampled, num_sampled, num_rejected, ids, logprobs, ranks, nonfinite = batch.kernels(None)
        ref_sampled, ref_num_sampled, ref_num_rejected, _, _ = batch.vllm(None)

        # every kept token inside the vocabulary (the next step's input), the corrupted rows' logits counted
        count = NonfiniteRows.count(kind, VOCAB)
        for request, kept in enumerate(num_sampled.tolist()):
            assert ((sampled[request, :kept] >= 0) & (sampled[request, :kept] < VOCAB)).all()
        assert nonfinite.tolist() == [count if r in corrupted_rows else 0 for r in range(batch.num_logits)]

        # the requests a step keeps: vLLM's acceptance, a drawn token of a finite logit, the kept tokens' rows' top-K
        # (vLLM's) and ranks (counted over the row's logits)
        requests = 1 if kind in NonfiniteRows.CORRUPTED_KINDS else batch.n
        assert torch.equal(num_sampled[:requests], ref_num_sampled[:requests].int())
        assert torch.equal(num_rejected[:requests], ref_num_rejected[:requests].int())
        for request in range(requests):
            kept = num_sampled[request].item()
            first, tokens = request * (batch.k + 1), sampled[request, :kept]
            rows = slice(first, first + kept)
            assert torch.equal(tokens[:-1], ref_sampled[request, : kept - 1])
            assert torch.isfinite(batch.logits[first + kept - 1, tokens[-1]])
            want = compute_topk_scores(batch.logits[rows], TOPK, tokens)
            assert torch.equal(ids[rows].long(), want.logprob_token_ids)
            torch.testing.assert_close(logprobs[rows], want.logprobs, rtol=0, atol=1e-4)
            for row, token in enumerate(tokens.tolist(), start=first):
                assert ranks[row].item() == (batch.logits[row] >= batch.logits[row, token]).sum().item()

        # the step's output raises on the corrupted rows as they reach the host, naming the step, kernel and rows
        want_rows = [(row, NONFINITE_REQUESTS[1], count) for row in corrupted_rows] if count else []
        NonfiniteRows.assert_step_output(
            RolloutRejectionSampler.KERNEL, nonfinite, NONFINITE_REQUESTS, want_rows, batch.cu_num_logits_np
        )


class TestVerifySamplingDistribution:
    """Chi-square tests of the emitted tokens against the lossless rule's target distributions: many seeds on fixed
    logits and greedy drafts, the acceptance high and low."""

    SUPPORT = 40  # tokens carrying most of a row's mass, spread over the vocabulary's chunks
    CALLS = 400  # kernel calls per test, each with fresh seeds
    REQUESTS = 256  # requests per call: one draw each
    P_MIN = 1e-3  # the smallest p-value a test passes with
    MIN_EXPECTED = 5.0  # the draws a chi-square bin expects at least

    @classmethod
    def setup_class(cls):
        BlackwellExtension.load()

    @classmethod
    def row(cls, generator: torch.Generator, peak: float) -> tuple[torch.Tensor, torch.Tensor]:
        """A row's logits (a support of SUPPORT tokens over a tail of the others) and its support's token ids."""
        device = torch.device("cuda")
        logits = torch.randn(VOCAB, generator=generator, device=device) - 9
        support = torch.randperm(VOCAB, generator=generator, device=device)[: cls.SUPPORT]
        logits[support] = torch.rand(cls.SUPPORT, generator=generator, device=device) * peak
        return logits, support

    @classmethod
    def chi_square(cls, tokens: torch.Tensor, logits: torch.Tensor, support: torch.Tensor, excluded: int = -1) -> float:
        """The p-value of the tokens as draws from softmax(logits) (without the excluded token): one bin per support
        token, one for the tail, the bins expecting fewer than MIN_EXPECTED draws pooled."""
        probs = torch.softmax(logits.double(), dim=-1)
        if excluded >= 0:
            probs[excluded] = 0
            probs /= probs.sum()
        bins = torch.full((VOCAB,), cls.SUPPORT, dtype=torch.long, device=logits.device)
        bins[support] = torch.arange(cls.SUPPORT, device=logits.device)
        expected = torch.zeros(cls.SUPPORT + 1, dtype=torch.float64, device=logits.device)
        expected.index_add_(0, bins, probs * tokens.numel())
        observed = torch.bincount(bins[tokens], minlength=cls.SUPPORT + 1).double()
        assert observed[expected == 0].sum() == 0, "draws of a token the distribution excludes"

        # the rare bins pooled (with the smallest frequent one while the pool expects too few)
        order = expected.argsort()
        rare = expected[order].cumsum(0) < cls.MIN_EXPECTED
        rare[: int(rare.sum().item()) + 1] = True
        pooled_expected = torch.cat([expected[order][~rare], expected[order][rare].sum().view(1)])
        pooled_observed = torch.cat([observed[order][~rare], observed[order][rare].sum().view(1)])
        statistic = ((pooled_observed - pooled_expected) ** 2 / pooled_expected).sum()
        df = torch.tensor(float(pooled_expected.numel() - 1), dtype=torch.float64, device=statistic.device)
        return torch.special.gammaincc(df / 2, statistic / 2).item()

    def draws(
        self, rows: list[torch.Tensor], drafts: list[int], rates: list[float] | None, seed: int
    ) -> tuple[torch.Tensor, ...]:
        """Every request's sampled tokens and counts over CALLS calls of REQUESTS requests on the same rows."""
        generator = torch.Generator("cuda").manual_seed(seed)
        k = len(drafts)
        batch = VerifyBatch(self.REQUESTS, k, generator)
        batch.logits = torch.stack(rows).repeat(self.REQUESTS, 1)
        batch.input_ids = torch.tensor([0, *drafts], dtype=torch.int32, device="cuda").repeat(self.REQUESTS)
        conditional = None
        if rates is not None:
            conditional = torch.tensor(unconditional_to_conditional_rates(rates)).float().cuda()
        sampled, counts = [], []
        for _ in range(self.CALLS):
            batch.seeds = torch.randint(-(2**62), 2**62, batch.seeds.shape, generator=generator, device="cuda")
            tokens, num_sampled, _, _, _, _, _ = batch.kernels(conditional)
            sampled.append(tokens)
            counts.append(num_sampled)
        return torch.cat(sampled), torch.cat(counts)

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("peak", [3.0, 12.0])
    def test_first_token_is_target(self, peak, seed):
        # lossless: the first emitted token follows the first row's distribution, whatever the draft; a draft the target
        # likes (high acceptance) and one it does not (low)
        generator = torch.Generator("cuda").manual_seed(seed)
        rows, supports = zip(*(self.row(generator, peak) for _ in range(2)))
        for draft in (supports[0][rows[0][supports[0]].argmax()].item(), supports[0][0].item() + 1):
            sampled, counts = self.draws(list(rows), [draft], None, seed)
            p = self.chi_square(sampled[:, 0], rows[0], supports[0])
            rate = (counts == 2).float().mean().item()
            print(f"seed {seed} peak {peak} draft {draft}: first token p = {p:.4f} (acceptance {rate:.4f})")
            assert p > self.P_MIN

    @pytest.mark.parametrize("seed", SEEDS)
    def test_second_token_and_bonus_are_target(self, seed):
        # after an accepted draft the next emitted token follows the next row; after all drafts, the bonus row
        generator = torch.Generator("cuda").manual_seed(seed)
        rows, supports = zip(*(self.row(generator, 12.0) for _ in range(3)))
        drafts = [supports[i][rows[i][supports[i]].argmax()].item() for i in range(2)]
        sampled, counts = self.draws(list(rows), drafts, None, seed)
        accepted_one, accepted_all = counts >= 2, counts == 3
        p_second = self.chi_square(sampled[accepted_one, 1], rows[1], supports[1])
        p_bonus = self.chi_square(sampled[accepted_all, 2], rows[2], supports[2])
        print(
            f"seed {seed}: second token p = {p_second:.4f} ({accepted_one.sum().item()} draws), "
            f"bonus p = {p_bonus:.4f} ({accepted_all.sum().item()} draws)"
        )
        assert p_second > self.P_MIN and p_bonus > self.P_MIN

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recovered_token_is_residual(self, seed):
        # synthetic acceptance: a rejected draft's recovered token follows the row without that draft
        generator = torch.Generator("cuda").manual_seed(seed)
        rows, supports = zip(*(self.row(generator, 12.0) for _ in range(2)))
        draft = supports[0][rows[0][supports[0]].argmax()].item()
        sampled, counts = self.draws(list(rows), [draft], [0.3], seed)
        rejected = counts == 1
        p = self.chi_square(sampled[rejected, 0], rows[0], supports[0], excluded=draft)
        rate = 1 - rejected.float().mean().item()
        print(f"seed {seed}: recovered token p = {p:.4f} ({rejected.sum().item()} draws), acceptance {rate:.4f}")
        assert p > self.P_MIN and abs(rate - 0.3) < 0.01
