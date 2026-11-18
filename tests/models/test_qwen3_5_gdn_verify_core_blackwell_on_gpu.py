import pytest
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.blackwell.extension import BlackwellExtension
from pear.models.qwen3_5.rollout.blackwell.gdn.replay import ReplayRecord, ReplayState
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not BlackwellRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="BlackwellRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
DIMS = GDNDims(H=8, HV=24, K=128, V=128)  # Qwen3.8-27B's Gated DeltaNet heads on one of two ranks
TAPS = ReplayRecord.TAPS  # the conv window before a token
MAX_TOKENS = 8  # a verify block's tokens at most: the records' size
REQUESTS = 6
STEPS = 24
MAX_COLUMNS = 64  # block-table columns per request
L2_EPS = 1e-6  # the q / k l2norm epsilon, under the square root
NORM_EPS = 1e-6  # the gated norm's epsilon
OUTPUT_TOLERANCE = 1e-2  # an output row's relative error against fp64: the bf16 rounding
STATE_TOLERANCE = 1e-4  # a state's largest error against fp64, relative to its largest entry
SEED = 0


class Reference:
    """The fp64 Gated DeltaNet of one request: its state and conv window followed token by token."""

    def __init__(self, weights: dict, state: torch.Tensor, window: torch.Tensor):
        self.weights = weights
        self.state = state.double().clone()
        self.window = window.double().clone()  # [TAPS, C]

    def block(self, qkvz: torch.Tensor, ba: torch.Tensor):
        """Every token's output [T, HV, V], the state after it [T, HV, V, K] and the window after it [T, TAPS, C]."""
        w, (H, HV, K, V), C = self.weights, DIMS, DIMS.C
        T = qkvz.shape[0]
        x = torch.cat([self.window, qkvz[:, :C].double()])
        conv = torch.stack([(x[j : j + TAPS + 1] * w["conv"].double().T).sum(0) for j in range(T)])
        conv = conv * torch.sigmoid(conv)
        q, k = conv[:, : H * K].view(T, H, K), conv[:, H * K : 2 * H * K].view(T, H, K)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + L2_EPS) * K**-0.5
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + L2_EPS)
        q, k = q.repeat_interleave(HV // H, 1), k.repeat_interleave(HV // H, 1)
        v = conv[:, 2 * H * K :].view(T, HV, V)
        b, a = ba.double().chunk(2, dim=-1)
        g = -w["A_log"].double().exp() * F.softplus(a + w["dt_bias"].double())
        beta = torch.sigmoid(b)

        state, outs, states, windows = self.state, [], [], []
        for j in range(T):
            state = state * g[j].exp()[:, None, None]
            u = (v[j] - torch.einsum("hvk,hk->hv", state, k[j])) * beta[j][:, None]
            state = state + u[:, :, None] * k[j][:, None, :]
            outs.append(torch.einsum("hvk,hk->hv", state, q[j]))
            states.append(state)
            windows.append(x[j + 1 : j + TAPS + 1])

        o = torch.stack(outs)
        z = qkvz[:, C:].double().view(T, HV, V)
        y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + NORM_EPS) * w["norm"].double() * F.silu(z)
        return y, torch.stack(states), torch.stack(windows)

    def commit(self, states: torch.Tensor, windows: torch.Tensor, count: int) -> None:
        """The state and window after the block's first count tokens."""
        self.state, self.window = states[count - 1], windows[count - 1]


class SharedMemoryPoison:
    """Every SM's shared memory filled with NaN bits, so that a kernel reading shared memory it never wrote fails."""

    NAN_BITS = 0xFFFFFFFF  # a NaN as fp32
    CTAS_PER_SM = 4  # the fill's CTAs per SM: every SM runs at least one
    SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

// bits in every word of the CTA's dynamic shared memory (inline PTX: the stores have no reader the compiler sees)
__global__ void poison_kernel(unsigned bits, int words) {
  extern __shared__ unsigned smem[];
  const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  for (int i = threadIdx.x; i < words; i += blockDim.x)
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(base + 4 * i), "r"(bits) : "memory");
}

void poison(int64_t bits, int64_t ctas_per_sm) {
  int device, sms, max_smem;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  C10_CUDA_CHECK(cudaFuncSetAttribute(poison_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
  poison_kernel<<<sms * ctas_per_sm, 1024, max_smem, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<unsigned>(bits), max_smem / 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

    def __init__(self):
        from torch.utils.cpp_extension import load_inline

        self.module = load_inline(
            name="shared_memory_poison",
            cpp_sources="void poison(int64_t bits, int64_t ctas_per_sm);",
            cuda_sources=self.SOURCE,
            functions=["poison"],
        )

    def fill(self) -> None:
        """NaN bits in every SM's shared memory, on the current stream."""
        self.module.poison(self.NAN_BITS, self.CTAS_PER_SM)


class ReplaySequence:
    """Verify steps of REQUESTS requests near block boundaries through the replaySSM kernels (rows and slots, the verify
    core, the flush and the bookkeeping after sampling), each request moving on by its committed tokens, every SM's
    shared memory full of NaN bits before the core and before the bookkeeping: every output against the fp64
    recurrence, the running state, every boundary block's state and window."""

    def __init__(self, query_len: int, block_size: int, is_conv_state_dim_first: bool, poison: SharedMemoryPoison):
        self.query_len, self.block_size, self.is_dim_first = query_len, block_size, is_conv_state_dim_first
        self.poison = poison
        self.generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        C, HV, K = DIMS.C, DIMS.HV, DIMS.K
        self.weights = {
            "conv": self.randn(C, TAPS + 1, scale=0.5, dtype=torch.bfloat16),
            "A_log": self.randn(HV, scale=0.5).add(-1.0),
            "dt_bias": self.randn(HV, scale=0.5, dtype=torch.bfloat16),
            "norm": (1 + self.randn(K, scale=0.1)).to(torch.bfloat16),
        }

        # the pools: slot 0 the null block, a block-table row of MAX_COLUMNS blocks per request
        num_slots = 1 + REQUESTS * MAX_COLUMNS
        taps = TAPS + query_len - 1
        self.state_pool = self.randn(num_slots, HV, K, K, scale=0.1)
        conv_shape = (num_slots, C, taps) if is_conv_state_dim_first else (num_slots, taps, C)
        self.conv_pool = self.randn(*conv_shape, dtype=torch.bfloat16)
        self.record = ReplayRecord.allocate(REQUESTS, DIMS, MAX_TOKENS, torch.device(DEVICE))
        scratch_bytes = torch.ops.pear_rollout_blackwell.gdn_verify_scratch_bytes(REQUESTS, DIMS.H, DIMS.HV)
        self.scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=DEVICE)
        self.tables = (1 + torch.arange(REQUESTS * MAX_COLUMNS, device=DEVICE, dtype=torch.int32)).view(
            REQUESTS, MAX_COLUMNS
        )

    def randn(self, *shape, scale: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (torch.randn(*shape, generator=self.generator, device=DEVICE) * scale).to(dtype)

    def window_of(self, slot: int) -> torch.Tensor:
        """A slot's conv window [TAPS, C]."""
        return self.conv_pool[slot, :, :TAPS].T if self.is_dim_first else self.conv_pool[slot, :TAPS]

    def run(self) -> dict:
        """The sequence's errors: outputs, running states, boundary states; the boundary blocks checked."""
        T, B, n = self.query_len, self.block_size, REQUESTS
        C, H, HV, K = DIMS.C, DIMS.H, DIMS.HV, DIMS.K
        w = self.weights
        ba_offset = ReplayRecord.ba_offset(DIMS, MAX_TOKENS)

        # requests at positions around block boundaries: their prefill left the state in the running block
        positions = [B * (2 + r % 3) - (r * 3) % (2 * T + 1) for r in range(n)]
        references = []
        for r in range(n):
            slot = 1 + r * MAX_COLUMNS + (positions[r] - 1) // B
            window = self.randn(TAPS, C, dtype=torch.bfloat16)
            if self.is_dim_first:
                self.conv_pool[slot, :, :TAPS] = window.T
            else:
                self.conv_pool[slot, :TAPS] = window
            references.append(Reference(w, self.state_pool[slot], window))

        # vLLM's align bookkeeping and replaySSM's tables
        state_columns = torch.tensor([(p - 1) // B for p in positions], dtype=torch.int32, device=DEVICE)
        source_columns, source_offsets = torch.zeros_like(state_columns), torch.zeros_like(state_columns)
        accepted = torch.ones_like(state_columns)
        replay, parity = torch.zeros_like(state_columns), torch.zeros_like(state_columns)
        rows = torch.zeros(n, ReplayState.ROW_INTS, dtype=torch.int32, device=DEVICE)
        slots = torch.zeros(1, n, ReplayState.SLOT_INTS, dtype=torch.int32, device=DEVICE)
        idx_mapping = torch.arange(n, dtype=torch.int64, device=DEVICE)
        query_start = torch.arange(0, n * T + 1, T, dtype=torch.int32, device=DEVICE)
        table_pointers = torch.tensor([self.tables.data_ptr()], dtype=torch.int64, device=DEVICE)
        layer_table = torch.ops.pear_rollout_blackwell.gdn_layer_table(
            [self.state_pool], [self.conv_pool], [self.record], [w["conv"]], [w["A_log"]], [w["dt_bias"]],
            [w["norm"]], [0], self.is_dim_first,
        )  # fmt: skip

        errors, state_error, boundary_error = [], 0.0, 0.0
        boundaries = {}  # slot -> (state, window) its boundary block must hold
        for step in range(STEPS):
            computed = torch.tensor(positions, dtype=torch.int32, device=DEVICE)
            torch.ops.pear_rollout_blackwell.gdn_replay_meta(
                idx_mapping, query_start, computed, None, state_columns, source_columns, source_offsets, accepted,
                table_pointers, self.tables.stride(0), n, n, B, rows, slots,
            )  # fmt: skip
            # the projections as the merged GEMM gives them: views of one [tokens, qkvz | ba] output
            projection = self.randn(n * T, C + HV * K + 2 * HV, dtype=torch.bfloat16)
            qkvz, ba = projection[:, : C + HV * K], projection[:, C + HV * K :]
            ba.mul_(2.0)
            out = torch.empty(n * T, HV, K, dtype=torch.bfloat16, device=DEVICE)
            self.poison.fill()
            torch.ops.pear_rollout_blackwell.gdn_verify(
                qkvz, ba, out, w["conv"], w["A_log"], w["dt_bias"], w["norm"], self.state_pool, self.conv_pool,
                self.record, rows, slots, replay, parity, self.scratch, 0, n, H, B, T, ba_offset, K**-0.5, NORM_EPS,
                False, self.is_dim_first,
            )  # fmt: skip

            # the committed tokens: every count 1..T, the whole block at a boundary's end now and then
            sampled = [1 + (step * 7 + r * 3) % T for r in range(n)]
            for r in range(n):
                if (positions[r] + T) % B == 0 and step % 2 == 0:
                    sampled[r] = T
            self.poison.fill()
            torch.ops.pear_rollout_blackwell.gdn_commit(
                layer_table, rows, slots, replay, parity, accepted, idx_mapping,
                torch.tensor(sampled, dtype=torch.int32, device=DEVICE), H, HV, B, T, ba_offset, K**-0.5,
            )  # fmt: skip
            assert not self.state_pool.isnan().any(), f"NaN in the state pool after step {step}'s commit"

            # the rows end with the step: a forward of one token over every row (a graph capture's) writes nothing
            caches = (self.state_pool.clone(), self.conv_pool.clone(), self.record.clone())
            idle_out = torch.full((1, HV, K), 7.0, dtype=torch.bfloat16, device=DEVICE)
            torch.ops.pear_rollout_blackwell.gdn_verify(
                qkvz[:1], ba[:1], idle_out, w["conv"], w["A_log"], w["dt_bias"], w["norm"], self.state_pool,
                self.conv_pool, self.record, rows, slots, replay, parity, self.scratch, 0, n, H, B, T, ba_offset,
                K**-0.5, NORM_EPS, False, self.is_dim_first,
            )  # fmt: skip
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip((self.state_pool, self.conv_pool, self.record), caches))
            assert bool((idle_out == 7.0).all())

            for r in range(n):
                y, states, windows = references[r].block(qkvz[r * T : (r + 1) * T], ba[r * T : (r + 1) * T])
                errors.append(((out[r * T : (r + 1) * T].double() - y).norm(dim=-1) / y.norm(dim=-1)).flatten())
                # the running slot holds the state at the start of this block (after a flush, at its end)
                destination = int(slots[0, r, 1])
                boundary = (positions[r] // B + 1) * B
                start = states[T - 1] if boundary == positions[r] + T and sampled[r] == T else references[r].state
                state_error = max(state_error, self.relative(self.state_pool[destination], start))
                if boundary <= positions[r] + sampled[r]:
                    # a committed boundary: its block holds the state and the window there
                    j = boundary - positions[r] - 1
                    boundaries[1 + r * MAX_COLUMNS + boundary // B - 1] = (states[j].clone(), windows[j].clone())
                references[r].commit(states, windows, sampled[r])
                positions[r] += sampled[r]
            for slot, (state, window) in boundaries.items():
                boundary_error = max(boundary_error, self.relative(self.state_pool[slot], state))
                assert torch.equal(self.window_of(slot).double(), window), f"boundary window of slot {slot}"

        return {"output": float(torch.cat(errors).max()), "state": state_error, "boundary": boundary_error}

    @staticmethod
    def relative(got: torch.Tensor, want: torch.Tensor) -> float:
        """The largest |error| relative to the largest |entry| of want."""
        return float((got.double() - want).abs().max() / want.abs().max())


@pytest.fixture(scope="module", autouse=True)
def extension():
    BlackwellExtension.load()


@pytest.fixture(scope="module")
def poison() -> SharedMemoryPoison:
    return SharedMemoryPoison()


@pytest.mark.parametrize("is_conv_state_dim_first", [False, True], ids=["taps_first", "channels_first"])
@pytest.mark.parametrize("block_size", [16, 832])
@pytest.mark.parametrize("query_len", [1, 2, 4, 8])
def test_the_verify_core_follows_fp64_through_replays_flushes_and_block_boundaries(
    query_len, block_size, is_conv_state_dim_first, poison
):
    errors = ReplaySequence(query_len, block_size, is_conv_state_dim_first, poison).run()

    assert errors["output"] < OUTPUT_TOLERANCE, errors
    assert errors["state"] < STATE_TOLERANCE and errors["boundary"] < STATE_TOLERANCE, errors
