import types

import numpy as np
import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata, GDNAttentionMetadataBuilder

from pear.models.qwen3_5.rollout.blackwell.gdn import CONV_TOKENS, BlackwellGDNMetadataBuilder, StagedBuffers
from pear.models.qwen3_5.rollout.hopper.gdn import HopperGDNMetadataBuilder

MAX_NUM_SEQS = 8  # of the engine config the builders read
MAX_NUM_BATCHED_TOKENS = 512
# a step of two decode rows, a prompt's chunk over its cached prefix, a one-token prompt and a new prompt
QUERY_LENS = [1, 1, 20, 1, 9]
SEQ_LENS = [30, 7, 120, 1, 9]
IS_PREFILLING = [False, False, True, True, True]
NUM_DECODES = 2  # the one-token prompt has no state before it: a prefill row


class HostStagedBuffers:
    """StagedBuffers' construction on a machine without a GPU driver: pageable host buffers, its device buffers on the
    host."""

    @staticmethod
    def init(buffers: StagedBuffers, device) -> None:
        slots, count = StagedBuffers.SLOTS, StagedBuffers.BUFFERS
        buffers.host_buffers = [torch.empty(slots, dtype=torch.int32) for _ in range(count)]
        buffers.device_buffers = [torch.empty(slots, dtype=torch.int32, device=device) for _ in range(count)]
        buffers.turn = 0


def vllm_builder(monkeypatch) -> None:
    """vLLM's GDN builder reduced to nothing the builders' paths read; the step's uploads on the host, a test's own."""
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", lambda self, *args: None)
    monkeypatch.setattr(StagedBuffers, "__init__", HostStagedBuffers.init)
    monkeypatch.setattr(StagedBuffers, "instance", None)


def engine_config() -> types.SimpleNamespace:
    """The parts of a VllmConfig the builders read as they are built."""
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)
    )


def common_metadata(block_table: torch.Tensor) -> CommonAttentionMetadata:
    """The step's common attention metadata on the host, as vLLM's V2 runner gives it to a GDN group's builder."""
    qsl = torch.tensor([0, *np.cumsum(QUERY_LENS)], dtype=torch.int32)
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32)

    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens,
        num_reqs=len(QUERY_LENS),
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(QUERY_LENS),
        max_seq_len=max(SEQ_LENS),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64),
        is_prefilling=torch.tensor(IS_PREFILLING),
        seq_lens_cpu_upper_bound=seq_lens,
    )


def block_table(first_slot: int) -> torch.Tensor:
    """A group's block table of the step: one state slot per request, from first_slot on."""
    return torch.arange(first_slot, first_slot + len(QUERY_LENS), dtype=torch.int32)[:, None]


@pytest.mark.parametrize("builder_cls", [BlackwellGDNMetadataBuilder, HopperGDNMetadataBuilder])
def test_the_runner_derives_the_other_gdn_groups_metadata_from_the_first_group_s(monkeypatch, builder_cls):
    vllm_builder(monkeypatch)

    builder = builder_cls(None, [], engine_config(), "cpu")

    # vLLM's builder opts in its own class only
    assert builder.supports_update_block_table


def test_a_blackwell_mixed_step_s_metadata_is_one_host_layout(monkeypatch):
    vllm_builder(monkeypatch)
    builder = BlackwellGDNMetadataBuilder(None, [], engine_config(), "cpu")

    md = builder.build(0, common_metadata(block_table(1)))

    D = NUM_DECODES
    chunks = [-(-n // CONV_TOKENS) for n in QUERY_LENS[D:]]
    assert (md.num_decodes, md.num_prefills, md.num_actual_tokens) == (D, len(QUERY_LENS) - D, sum(QUERY_LENS))
    assert md.has_initial_state.tolist() == [True, True, True, False, False]
    assert md.prefill_query_start_loc.tolist() == [0, 20, 21, 30]
    # a conv program per decode row, one per CONV_TOKENS tokens begun of a prefill row: its row, its chunk in the row
    assert md.batch_ptr.tolist() == list(range(D)) + [D + i for i, n in enumerate(chunks) for _ in range(n)]
    assert md.token_chunk_offset_ptr.tolist() == [0] * D + [c for n in chunks for c in range(n)]
    assert md.nums_dict[CONV_TOKENS]["tot"] == D + sum(chunks)
    assert md.non_spec_state_indices_tensor.tolist() == [1, 2, 3, 4, 5] and md._pear_layout.D == D


def test_another_blackwell_gdn_group_takes_the_mixed_step_with_its_own_state_slots(monkeypatch):
    vllm_builder(monkeypatch)
    first, other = (BlackwellGDNMetadataBuilder(None, [], engine_config(), "cpu") for _ in range(2))
    md = first.build(0, common_metadata(block_table(1)))
    slot_mapping = torch.zeros(sum(QUERY_LENS), dtype=torch.int64)

    group_md = other.update_block_table(md, block_table(11), slot_mapping)

    slots = list(range(11, 11 + len(QUERY_LENS)))
    assert group_md.non_spec_state_indices_tensor.tolist() == slots
    assert group_md.prefill_state_indices.tolist() == slots[NUM_DECODES:]
    assert group_md._pear_layout.idx.tolist() == slots and group_md._pear_layout.pf_idx.tolist() == slots[NUM_DECODES:]
    assert group_md.batch_ptr is md.batch_ptr  # the step's layout, shared
    assert md.non_spec_state_indices_tensor.tolist() == [1, 2, 3, 4, 5]  # the first group's, unchanged


def test_another_hopper_gdn_group_takes_the_mixed_step_with_its_own_state_slots(monkeypatch):
    vllm_builder(monkeypatch)
    first, other = (HopperGDNMetadataBuilder(None, [], engine_config(), "cpu") for _ in range(2))
    num_reqs = len(QUERY_LENS)
    layout = types.SimpleNamespace(
        N=num_reqs, P=num_reqs - NUM_DECODES, T=sum(QUERY_LENS), D=NUM_DECODES, seq_init=torch.ones(num_reqs)
    )
    md = first.mixed_step_metadata(block_table(1), sum(QUERY_LENS), layout)
    slot_mapping = torch.zeros(sum(QUERY_LENS), dtype=torch.int64)

    group_md = other.update_block_table(md, block_table(11), slot_mapping)

    # each group's slots at its builder's fixed address, which its FULL mixed-step graphs read
    assert group_md.non_spec_state_indices_tensor is other.mixed_step_slots and group_md._pear_layout is layout
    assert other.mixed_step_slots[:num_reqs].tolist() == list(range(11, 11 + num_reqs))
    assert md.non_spec_state_indices_tensor is first.mixed_step_slots
    assert first.mixed_step_slots[:num_reqs].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("builder_cls", [BlackwellGDNMetadataBuilder, HopperGDNMetadataBuilder])
def test_another_gdn_group_takes_a_decode_step_as_vllm_derives_it(monkeypatch, builder_cls):
    vllm_builder(monkeypatch)
    derived = []
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "update_block_table",
        lambda self, metadata, blk_table, slot_mapping: derived.append(metadata) or metadata,
    )
    builder = builder_cls(None, [], engine_config(), "cpu")
    md = GDNAttentionMetadata(
        num_prefills=0,
        num_decodes=4,
        num_prefill_tokens=0,
        num_decode_tokens=4,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
    )

    builder.update_block_table(md, block_table(11), torch.zeros(4, dtype=torch.int64))

    assert derived == [md]
