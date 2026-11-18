import dataclasses
from typing import NamedTuple, Optional

import numpy as np
import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from pear.models.qwen3_5.rollout.blackwell.gdn.core import GDNCore
from pear.models.qwen3_5.rollout.gdn_dims import GDNDims
from pear.models.qwen3_5.rollout.staged_buffer import StagedBuffer


class ReplayRecord:
    """The layout of a Gated DeltaNet layer's replay records: per request two copies (a verify step reads one and writes
    the other) of its last verify block, the conv inputs from the window before it on, then its b / a rows."""

    TAPS = 3  # the conv window before the block (kTaps of blackwell/csrc/gdn_verify.cu)
    COPIES = 2
    ALIGNMENT = 256  # a copy's size is a multiple of it (bytes)

    @classmethod
    def ba_offset(cls, dims: GDNDims, max_tokens: int) -> int:
        """The b / a rows' offset in a copy (bytes): after the window's and the block's bf16 conv inputs."""
        return (cls.TAPS + max_tokens) * dims.C * torch.bfloat16.itemsize

    @classmethod
    def allocate(cls, num_requests: int, dims: GDNDims, max_tokens: int, device: torch.device) -> torch.Tensor:
        """Zeroed records of num_requests requests for blocks of up to max_tokens tokens."""
        copy_bytes = cls.ba_offset(dims, max_tokens) + max_tokens * 2 * dims.HV * torch.bfloat16.itemsize
        copy_bytes = triton.cdiv(copy_bytes, cls.ALIGNMENT) * cls.ALIGNMENT

        return torch.zeros(num_requests, cls.COPIES, copy_bytes, dtype=torch.uint8, device=device)


class ReplayStep(NamedTuple):
    """A step's replaySSM tables: its rows and their slots in every GDN KV cache group, its requests' replay counts and
    record copies."""

    rows: torch.Tensor  # int32 [rows, 4]: first token, verify tokens (0: not a verify row), request, tokens computed
    slots: torch.Tensor  # int32 [groups, rows, 4]: source, destination, the block of a boundary inside it (-1: none)
    replay: torch.Tensor  # int32 [requests]: the committed tokens of each request's last verify block
    parity: torch.Tensor  # int32 [requests]: the record copy that holds that block
    # uint8: each row's value-head items and key-head fragments, written by the core's prep kernel and read by its
    # stream kernel (the layers run in turn: one for all)
    scratch: torch.Tensor
    max_tokens: int  # tokens of a verify block at most: the records' size
    block_size: int  # the mamba block size (tokens): the prefix cache's state boundaries


class GDNVerifyCore:
    """A Gated DeltaNet layer's verify rows under replaySSM (blackwell/csrc/gdn_verify.cu): from each request's slot,
    the replay of its previous block's committed tokens, this block's outputs with the gated norm, the state written
    once."""

    @staticmethod
    def run(layer, mixed_qkvz, ba, core_attn_out, step: ReplayStep, num_rows: int) -> None:
        """The core of the step's first num_rows rows (a row of no verify tokens is skipped) in the layer's group."""
        constants = GDNCore.constants(layer)
        dims = constants.dims
        torch.ops.pear_rollout_blackwell.gdn_verify(
            mixed_qkvz,
            ba,
            core_attn_out,
            constants.conv_weight,
            constants.a_log,
            constants.dt_bias,
            constants.norm_weight,
            constants.state_pool,
            constants.conv_state,
            layer.replay_record,
            step.rows,
            step.slots,
            step.replay,
            step.parity,
            step.scratch,
            layer.replay_group,
            num_rows,
            dims.H,
            step.block_size,
            step.max_tokens,
            ReplayRecord.ba_offset(dims, step.max_tokens),
            constants.scale,
            constants.norm_eps,
            layer.norm.activation == "sigmoid",
            is_conv_state_dim_first(),
        )


class ReplayState:
    """The replaySSM bookkeeping of a speculative replica (one per process, made by its model state with the KV caches):
    each request's replay count and record copy, the step's rows and their slots in every GDN KV cache group and the
    layers' table of the flush after sampling."""

    # a step's row: first token, verify tokens (0: not a verify row), request, tokens computed before; a row's slots in a
    # group: source, destination, the block of a boundary inside its block (kRowInts, kSlotInts of
    # blackwell/csrc/gdn_verify.cu)
    ROW_INTS = 4
    SLOT_INTS = 4
    instance: Optional["ReplayState"] = None

    def __init__(self, groups: list[list], max_num_reqs: int, max_tokens: int, block_size: int, device: torch.device):
        layers = [layer for group_layers in groups for layer in group_layers]
        self.dims = GDNDims.of_layer(layers[0])
        self.max_tokens, self.block_size = max_tokens, block_size
        self.replay = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.parity = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.rows = torch.zeros(max_num_reqs, self.ROW_INTS, dtype=torch.int32, device=device)
        self.slots = torch.full((len(groups), max_num_reqs, self.SLOT_INTS), -1, dtype=torch.int32, device=device)
        dims = self.dims
        scratch_bytes = torch.ops.pear_rollout_blackwell.gdn_verify_scratch_bytes(max_num_reqs, dims.H, dims.HV)
        self.verify_scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        # a mixed step's verify rows, uploaded from pinned buffers
        self.verify_rows = StagedBuffer(max_num_reqs, device, torch.uint8)

        # each layer's group, the order of the slots' groups; the flush's per-layer descriptors
        group_ids = []
        for group_id, group_layers in enumerate(groups):
            for layer in group_layers:
                layer.replay_group = group_id
                group_ids.append(group_id)
        constants = [GDNCore.constants(layer) for layer in layers]
        self.scale = constants[0].scale
        self.layer_table = torch.ops.pear_rollout_blackwell.gdn_layer_table(
            [c.state_pool for c in constants],
            [c.conv_state for c in constants],
            [layer.replay_record for layer in layers],
            [c.conv_weight for c in constants],
            [c.a_log for c in constants],
            [c.dt_bias for c in constants],
            [c.norm_weight for c in constants],
            group_ids,
            is_conv_state_dim_first(),
        )

    def preprocess(self, model_state, input_batch, num_computed_tokens: torch.Tensor, is_verify) -> None:
        """The step's rows and slots before the forward (vLLM's align bookkeeping with them), and vLLM's pre-copy of the
        other rows' states when there are any."""
        ctx = model_state._mamba_ctx
        verify_rows = None
        if is_verify is not None:
            num_rows = is_verify.shape[0]
            self.verify_rows.next_stage()[:num_rows] = is_verify
            self.verify_rows.upload(num_rows)
            verify_rows = self.verify_rows.device_buffer[:num_rows]

        torch.ops.pear_rollout_blackwell.gdn_replay_meta(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            num_computed_tokens,
            verify_rows,
            model_state._mamba_state_idx_gpu,
            model_state._mamba_src_col_gpu,
            model_state._mamba_src_off_gpu,
            model_state.num_accepted_tokens_gpu,
            ctx.aligned_index_block_table_ptrs,
            ctx.block_table_stride_req,
            input_batch.num_reqs,
            input_batch.num_reqs_after_padding,
            self.block_size,
            self.rows,
            self.slots,
        )
        if is_verify is not None:
            ctx.run_fused_precopy(
                input_batch.num_reqs,
                model_state._mamba_state_idx_gpu,
                model_state._mamba_src_col_gpu,
                model_state._mamba_src_off_gpu,
                input_batch.idx_mapping,
            )

    def step(self) -> ReplayStep:
        """The tables of the step being run."""
        return ReplayStep(
            self.rows, self.slots, self.replay, self.parity, self.verify_scratch, self.max_tokens, self.block_size
        )

    def postprocess(self, model_state, idx_mapping: torch.Tensor, num_sampled: torch.Tensor) -> None:
        """After sampling: the boundary states the step's committed tokens reached, the requests' replay counts and
        record copies, vLLM's accepted counts."""
        torch.ops.pear_rollout_blackwell.gdn_commit(
            self.layer_table,
            self.rows,
            self.slots,
            self.replay,
            self.parity,
            model_state.num_accepted_tokens_gpu,
            idx_mapping,
            num_sampled.to(torch.int32),
            self.dims.H,
            self.dims.HV,
            self.block_size,
            self.max_tokens,
            ReplayRecord.ba_offset(self.dims, self.max_tokens),
            self.scale,
        )


@dataclasses.dataclass
class ReplayMetadata:
    """The GDN metadata of a speculative replica's step: its verify rows under replaySSM (rows of ReplayState's tables
    below num_rows; the core skips the others) and, in a mixed step, its prefill rows on GDNCore.mixed_step's
    metadata."""

    num_rows: int
    prefill: GDNAttentionMetadata | None = None
    prefill_rows: np.ndarray | None = None  # the step's prefill rows, ascending
    # their tokens: the step's last ones, or gathered when verify rows follow them
    prefill_tokens: slice | torch.Tensor | None = None
