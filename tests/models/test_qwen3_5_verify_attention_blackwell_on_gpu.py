import pytest
import torch
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.attention import VerifyAttention
from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = torch.device("cuda")
PAGE = VerifyAttention.PAGE
KV_HEADS = 2  # the model at TP2: 12 query heads, 2 KV heads of 256
QUERY_HEADS = 12
HEAD_SIZE = VerifyAttention.HEAD_SIZE
SCALE = HEAD_SIZE**-0.5
SEED = 0
REL_MEAN = 3e-3  # relative error per query row and head against fp64: the bf16 output and P roundings
REL_MAX = 5e-3
# short requests, page boundaries, requests split over CTAs
LENGTHS = [9, 64, 65, 127, 128, 129, 700, 3000, 448, 455, 10000, 40000, 100000]
PEAKED_QUERY_SCALE = 10.0  # queries that peak the scores: the online softmax's rescales


class VerifyAttentionCase:
    """Requests at the given attended lengths over a paged bf16 cache in random pages, the step's last T tokens' keys and
    values given beside the cache (their cache rows and the rows past each request's length poisoned: the kernel
    writes the first and must not read the second), queries at T per request."""

    def __init__(self, lengths: list[int], query_len: int, query_scale: float = 1.0):
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        self.lengths, self.query_len = lengths, query_len
        pages = [-(-length // PAGE) for length in lengths]
        order = torch.randperm(sum(pages), generator=torch.Generator().manual_seed(SEED)) + 1
        self.block_table = torch.zeros(len(lengths), max(pages), dtype=torch.int32)
        for i, count in enumerate(pages):
            self.block_table[i, :count] = order[sum(pages[:i]) : sum(pages[: i + 1])]
        self.block_table = self.block_table.to(DEVICE)
        self.seq_lens = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
        shape = (sum(pages) + 1, PAGE, KV_HEADS, HEAD_SIZE)
        self.keys = torch.randn(shape, generator=generator, device=DEVICE).bfloat16()
        self.values = torch.randn(shape, generator=generator, device=DEVICE).bfloat16()
        self.query = (
            torch.randn(len(lengths) * query_len, QUERY_HEADS, HEAD_SIZE, generator=generator, device=DEVICE)
            * query_scale
        ).bfloat16()

        # the cache as the model's attention reads it ([blocks, KV heads, PAGE, K | V])
        self.cache = torch.cat([self.keys, self.values], dim=-1).permute(0, 2, 1, 3).contiguous()
        for i, length in enumerate(lengths):
            if length % PAGE:
                self.cache[self.block_table[i, pages[i] - 1], :, length % PAGE :] = float("nan")
        positions = self.seq_lens.long()[:, None] - query_len + torch.arange(query_len, device=DEVICE)
        blocks = self.block_table.long().gather(1, positions // PAGE).flatten()
        offsets = (positions % PAGE).flatten()
        self.slot_mapping = blocks * PAGE + offsets
        self.key, self.value = self.keys[blocks, offsets], self.values[blocks, offsets]
        self.cache[blocks, :, offsets] = float("nan")

    def run(self) -> torch.Tensor:
        """PEAR's kernel over the case: the output, the step's rows written into the cache."""
        out = torch.empty_like(self.query)
        VerifyAttention.instance.run(
            self.query, self.key, self.value, self.slot_mapping, self.cache, self.block_table, self.seq_lens,
            len(self.lengths), self.query_len, SCALE, out,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out

    def reference(self) -> torch.Tensor:
        """fp64 attention [requests x T, query heads, head size]: query j of a request at position L - T + j sees the
        keys up to it."""
        outputs = []
        T, group = self.query_len, QUERY_HEADS // KV_HEADS
        for i, length in enumerate(self.lengths):
            pages = self.block_table[i, : -(-length // PAGE)].long()
            keys = self.keys[pages].reshape(-1, KV_HEADS, HEAD_SIZE)[:length].double()
            values = self.values[pages].reshape(-1, KV_HEADS, HEAD_SIZE)[:length].double()
            q = self.query[i * T : (i + 1) * T].double().view(T, KV_HEADS, group, HEAD_SIZE)
            positions = torch.arange(length, device=DEVICE)
            mask = positions[None, :] <= length - T + torch.arange(T, device=DEVICE)[:, None]
            scores = torch.einsum("tkgd,lkd->kgtl", q, keys) * SCALE
            scores = scores.masked_fill(~mask[None, None], float("-inf"))
            attention = torch.einsum("kgtl,lkd->tkgd", scores.softmax(-1), values)
            outputs.append(attention.reshape(T, QUERY_HEADS, HEAD_SIZE))
        return torch.cat(outputs)


@pytest.fixture(scope="module", autouse=True)
def workspace():
    BlackwellExtension.load()
    VerifyAttention.create(DEVICE)


@pytest.mark.parametrize("query_scale", [1.0, PEAKED_QUERY_SCALE], ids=["spread", "peaked"])
@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_the_verify_attention_is_within_a_bf16_rounding_of_fp64(query_len, query_scale):
    case = VerifyAttentionCase(LENGTHS, query_len, query_scale)

    out = case.run()

    reference = case.reference()
    error = (out.double() - reference).norm(dim=-1) / reference.norm(dim=-1)
    assert error.mean() < REL_MEAN and error.max() < REL_MAX, (error.mean(), error.max())


@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_the_verify_attention_writes_the_step_s_keys_and_values(query_len):
    case = VerifyAttentionCase(LENGTHS, query_len)

    case.run()

    blocks, offsets = case.slot_mapping // PAGE, case.slot_mapping % PAGE
    rows = case.cache[blocks, :, offsets]
    assert torch.equal(rows[..., :HEAD_SIZE], case.key) and torch.equal(rows[..., HEAD_SIZE:], case.value)


def test_a_cuda_graph_replays_the_verify_attention():
    case = VerifyAttentionCase(LENGTHS, 4)
    want = case.run()
    out = torch.empty_like(case.query)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        VerifyAttention.instance.run(
            case.query, case.key, case.value, case.slot_mapping, case.cache, case.block_table, case.seq_lens,
            len(case.lengths), case.query_len, SCALE, out,
        )  # fmt: skip
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out, want)
