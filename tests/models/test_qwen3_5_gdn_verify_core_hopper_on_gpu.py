import math

import pytest
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform

from pear.models.qwen3_5.rollout.gdn_dims import GDNDims
from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
from pear.models.qwen3_5.rollout.hopper.gdn import ReplayRecord, ReplayState
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend

# the GPU backend's own tests run on the GPUs that select it
pytestmark = pytest.mark.skipif(
    not HopperRolloutGpuBackend.supports_compute_capability(current_platform.get_device_capability()),
    reason="HopperRolloutGpuBackend's kernels",
)

DEVICE = "cuda"
DIMS = GDNDims(H=4, HV=12, K=128, V=128)  # Qwen3.8-27B's Gated DeltaNet heads on one of four ranks
TAPS = ReplayRecord.TAPS  # the conv window before a token
CONV_TAPS = TAPS + 1  # the causal conv's taps
REQUESTS = 8
MAX_COLUMNS = 64  # block-table columns per request
STEPS = {16: 24, 832: 32}  # verify steps of a sequence by its block size: blocks crossed at both
EPS = 1e-6  # the q / k l2norm's and the gated norm's epsilon
SOFTPLUS_THRESHOLD = 20.0  # softplus is the identity above it, as in vLLM's gating kernel
A_RANGE = (1.0, 16.0)  # FLA's initialization of the decay rates A
DT_RANGE = (1e-3, 1e-1)  # and of dt, whose softplus inverse is dt_bias
INITIAL_STATE_SCALE = 0.05
STATE_TOLERANCE = 1e-3  # a state's relative error against fp64: fp32 drift (at most 2.2e-4 measured)
OUTPUT_TOLERANCE = 2e-3  # the bf16 outputs' relative error against the bf16-rounded reference
SEED = 0
# every query length of the drafters' verify blocks (3 and 7 drafts) and block size (the tests' and the replicas'), with
# every commit count of a block (0: random ones)
CASES = [(T, block_size, commit) for T in (4, 8) for block_size in (16, 832) for commit in range(T + 1)]


class Reference:
    """The fp64 recurrence over each request's committed tokens, with the kernel's rounding points (bf16 tap products
    and conv outputs, bf16 betas, bf16 outputs before the gated norm)."""

    def __init__(self, sequence: "ReplaySequence", initial_states: torch.Tensor, initial_windows: torch.Tensor):
        self.sequence = sequence
        self.states = [state.clone() for state in initial_states]
        self.windows = [window.clone() for window in initial_windows]

    def conv(self, inputs: torch.Tensor) -> torch.Tensor:
        """The causal conv's bf16 outputs of inputs [taps + L, C] (bf16): bf16(silu(sum_j bf16(x_j w_j)))."""
        weight = self.sequence.conv_weight
        products = [
            (inputs[j : j + inputs.shape[0] - TAPS].float() * weight[:, j].float()).bfloat16().double()
            for j in range(CONV_TAPS)
        ]
        acc = ((products[0] + products[1]) + products[2]) + products[3]
        return (acc * torch.sigmoid(acc)).bfloat16().double()

    @staticmethod
    def tokens(state, q, k, v, g, beta) -> tuple[list, list]:
        """The gated delta rule over the rows of one value head: the states after each token and the raw outputs."""
        states, outs = [], []
        for t in range(k.shape[0]):
            state = state * torch.exp(g[t])
            u = (v[t] - state @ k[t]) * beta[t]
            state = state + torch.outer(u, k[t])
            states.append(state)
            outs.append(state @ q[t])
        return states, outs

    def block(self, qkvz: torch.Tensor, ba: torch.Tensor, committed: torch.Tensor, computed: torch.Tensor):
        """Every request's block outputs [n T, HV, V] and the boundaries its committed tokens cross (request, boundary
        position, state, window); the requests' states and windows move on by their committed tokens."""
        s, (H, HV, K, V), C = self.sequence, DIMS, DIMS.C
        T, B = s.query_len, s.block_size
        out = torch.empty(REQUESTS * T, HV, V, dtype=torch.float64)
        boundaries = []
        for r in range(REQUESTS):
            x = qkvz[r * T : (r + 1) * T]
            inputs = torch.cat([self.windows[r], x[:, :C]])
            y = self.conv(inputs)
            q, k = y[:, : H * K].view(T, H, K), y[:, H * K : 2 * H * K].view(T, H, K)
            v = y[:, 2 * H * K :].view(T, HV, V)
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + EPS) * K**-0.5
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + EPS)
            b, a = ba[r * T : (r + 1) * T, :HV].double(), ba[r * T : (r + 1) * T, HV:].double()
            g = -torch.exp(s.a_log.double()) * F.softplus(a + s.dt_bias.double(), threshold=SOFTPLUS_THRESHOLD)
            beta = torch.sigmoid(b).bfloat16().double()
            z = x[:, C:].double().view(T, HV, V)

            count = int(committed[r])
            state = self.states[r].clone()
            boundary = (int(computed[r]) // B + 1) * B  # the first boundary after the block's start
            boundary_token = boundary - int(computed[r]) - 1
            boundary_state = self.states[r].clone()
            for hv in range(HV):
                ih = hv // (HV // H)
                states, outs = self.tokens(self.states[r][hv], q[:, ih], k[:, ih], v[:, hv], g[:, hv], beta[:, hv])
                o = torch.stack(outs).bfloat16().double()
                rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS)
                out[r * T : (r + 1) * T, hv] = o * rstd * s.norm_weight.double() * (z[:, hv] * torch.sigmoid(z[:, hv]))
                state[hv] = states[count - 1]
                if boundary_token < count:
                    boundary_state[hv] = states[boundary_token]
            if boundary_token < count:
                window = inputs[boundary_token + 1 : boundary_token + 1 + TAPS]
                boundaries.append((r, boundary, boundary_state, window))
            self.states[r] = state
            self.windows[r] = inputs[count : count + TAPS]
        return out, boundaries

    @staticmethod
    def relative(got: torch.Tensor, want: torch.Tensor) -> float:
        """||got - want|| relative to ||want||."""
        return float((got - want).norm() / want.norm().clamp_min(1e-30))


class ReplaySequence:
    """Verify steps of a few requests under replaySSM (csrc/gdn_verify.cu through its ops, as the model state drives
    them: rows and slots, the verify core, the flush and the bookkeeping after sampling) with partial acceptance and
    blocks crossing mamba block boundaries, against the fp64 recurrence with the kernel's rounding points: every block's
    outputs, every crossed boundary's state and conv window, the flushed running states at the end. A control breaks
    what the check must catch: an accepted count one token too far, or no state saved at a crossed boundary."""

    def __init__(self, query_len: int, block_size: int, commit: int, control: str | None = None):
        self.query_len, self.block_size, self.commit, self.control = query_len, block_size, commit, control
        self.generator = torch.Generator().manual_seed(SEED)
        C, HV, V = DIMS.C, DIMS.HV, DIMS.V
        self.conv_weight = (torch.randn(C, CONV_TAPS, generator=self.generator) * 0.5).bfloat16()
        self.a_log = torch.log(torch.rand(HV, generator=self.generator) * (A_RANGE[1] - A_RANGE[0]) + A_RANGE[0])
        log_dt = math.log(DT_RANGE[0]) + torch.rand(HV, generator=self.generator) * math.log(DT_RANGE[1] / DT_RANGE[0])
        self.dt_bias = torch.log(torch.expm1(torch.exp(log_dt))).bfloat16()
        self.norm_weight = (torch.rand(V, generator=self.generator) + 0.5).bfloat16()
        # every request's block table: distinct slots, slot 0 the null block
        self.tables = (1 + torch.arange(REQUESTS * MAX_COLUMNS)).view(REQUESTS, MAX_COLUMNS).int()
        self.record = ReplayRecord.allocate(REQUESTS, DIMS, query_len, torch.device(DEVICE))

    def run(self) -> dict:
        """The sequence's worst errors: outputs, boundary states, flushed running states, boundary window mismatches."""
        ops = torch.ops.pear_rollout_hopper
        T, n, B = self.query_len, REQUESTS, self.block_size
        C, H, HV, K, V = DIMS.C, DIMS.H, DIMS.HV, DIMS.K, DIMS.V
        num_slots = 1 + n * MAX_COLUMNS
        ssm = torch.zeros(num_slots, HV, V, K, dtype=torch.float32, device=DEVICE)
        conv = torch.zeros(num_slots, TAPS + T - 1, C, dtype=torch.bfloat16, device=DEVICE)

        # the requests' starting positions straddle block boundaries, their states and windows in the running blocks
        g = self.generator
        positions = B * torch.randint(1, 3, (n,), generator=g) - torch.randint(1, min(4 * T, B), (n,), generator=g)
        initial_states = torch.randn(n, HV, V, K, generator=g, dtype=torch.float64) * INITIAL_STATE_SCALE
        initial_windows = torch.randn(n, TAPS, C, generator=g).bfloat16()
        columns = (positions - 1) // B
        for r in range(n):
            ssm[self.tables[r, columns[r]]] = initial_states[r].float().to(DEVICE)
            conv[self.tables[r, columns[r]], :TAPS] = initial_windows[r].to(DEVICE)

        # vLLM's align bookkeeping and replaySSM's tables
        replay = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        parity = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        state_columns = columns.int().to(DEVICE)
        source_columns = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
        source_offsets = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        accepted = torch.ones(n, dtype=torch.int32, device=DEVICE)
        tables = self.tables.to(DEVICE)
        table_pointers = torch.tensor([tables.data_ptr()], dtype=torch.int64, device=DEVICE)
        rows = torch.zeros(n, ReplayState.ROW_INTS, dtype=torch.int32, device=DEVICE)
        slots = torch.full((1, n, ReplayState.SLOT_INTS), -1, dtype=torch.int32, device=DEVICE)
        idx_mapping = torch.arange(n, dtype=torch.int64, device=DEVICE)
        query_start = torch.arange(0, n * T + 1, T, dtype=torch.int32, device=DEVICE)
        # the weights live as long as the layer table, which holds their addresses
        conv_weight, a_log, dt_bias, norm_weight = (
            w.to(DEVICE) for w in (self.conv_weight, self.a_log, self.dt_bias, self.norm_weight)
        )
        layer_table = ops.gdn_layer_table(
            [ssm], [conv], [self.record], [conv_weight], [a_log], [dt_bias], [norm_weight], [0], False
        )
        record_ba = ReplayRecord.ba_offset(DIMS, T)

        reference = Reference(self, initial_states, initial_windows)
        worst = {"output": 0.0, "boundary": 0.0, "state": 0.0, "window": 0}
        computed = positions.clone()
        for _ in range(STEPS[B]):
            qkvz = torch.randn(n * T, C + HV * V, generator=g).bfloat16()
            ba = torch.randn(n * T, 2 * HV, generator=g).bfloat16()
            committed = torch.randint(1, T + 1, (n,), generator=g)
            if self.commit:
                committed[:] = self.commit
            out = torch.empty(n * T, HV, V, dtype=torch.bfloat16, device=DEVICE)

            ops.gdn_replay_meta(
                idx_mapping, query_start, computed.int().to(DEVICE), None, replay, parity, state_columns,
                source_columns, source_offsets, accepted, table_pointers, tables.stride(0), n, n, B, rows, slots,
            )  # fmt: skip
            if self.control == "boundary":
                slots[:, :, 2] = torch.where(slots[:, :, 2] >= 0, 0, slots[:, :, 2])
            ops.gdn_verify(
                qkvz.to(DEVICE), ba.to(DEVICE), out, conv_weight, a_log, dt_bias, norm_weight, ssm, conv, self.record,
                rows, slots, 0, n, H, B, T, record_ba, K**-0.5, EPS, False, False,
            )  # fmt: skip
            sampled = committed + (1 if self.control == "commit" else 0)
            ops.gdn_commit(
                layer_table, rows, slots, replay, parity, accepted, idx_mapping, sampled.int().to(DEVICE), H, HV, B, T,
                record_ba, K**-0.5,
            )  # fmt: skip
            torch.cuda.synchronize()

            expected, boundaries = reference.block(qkvz, ba, committed, computed)
            worst["output"] = max(worst["output"], Reference.relative(out.double().cpu(), expected.bfloat16().double()))
            for r, boundary, state, window in boundaries:
                slot = self.tables[r, boundary // B - 1]
                worst["boundary"] = max(worst["boundary"], Reference.relative(ssm[slot].double().cpu(), state))
                worst["window"] = max(worst["window"], int((conv[slot, :TAPS].cpu() != window).sum()))
            computed += committed

        # the running state at the end: a request flushed at its block's end holds the state itself in its slot
        for r in range(n):
            if int(replay[r]) == 0:
                slot = int(slots[0, r, 1])
                worst["state"] = max(worst["state"], Reference.relative(ssm[slot].double().cpu(), reference.states[r]))
        return worst

    @staticmethod
    def passes(worst: dict) -> bool:
        """Whether every error of a sequence is within its tolerance."""
        return (
            worst["output"] < OUTPUT_TOLERANCE
            and worst["boundary"] < STATE_TOLERANCE
            and worst["state"] < STATE_TOLERANCE
            and worst["window"] == 0
        )


@pytest.fixture(scope="module", autouse=True)
def extension():
    HopperExtension.load()


@pytest.mark.parametrize("query_len, block_size, commit", CASES)
def test_the_verify_core_follows_fp64_through_replays_flushes_and_block_boundaries(query_len, block_size, commit):
    worst = ReplaySequence(query_len, block_size, commit).run()

    assert ReplaySequence.passes(worst), worst


@pytest.mark.parametrize("control", ["commit", "boundary"])
def test_the_check_catches_an_accepted_count_too_far_and_a_boundary_state_never_saved(control):
    worst = ReplaySequence(4, 16, 0, control).run()

    assert not ReplaySequence.passes(worst), worst
