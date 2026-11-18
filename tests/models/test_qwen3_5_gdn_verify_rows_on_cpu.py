import dataclasses
import types

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

from host_staged_buffer import HostStagedBuffer
from pear.models.qwen3_5.rollout.blackwell.gdn.core import GDNCore
from pear.models.qwen3_5.rollout.blackwell.gdn.kernels import CONV_TOKENS
from pear.models.qwen3_5.rollout.blackwell.gdn.layer import BlackwellGDNAttention, BlackwellGDNMetadataBuilder
from pear.models.qwen3_5.rollout.blackwell.gdn.replay import GDNVerifyCore, ReplayMetadata, ReplayState
from pear.models.qwen3_5.rollout.gdn import RolloutGDNAttention
from pear.models.qwen3_5.rollout.hopper.gdn import HopperGDNAttention

MAX_NUM_SEQS = 8  # of the engine config the builder reads
MAX_NUM_BATCHED_TOKENS = 512
NUM_SPECULATIVE_TOKENS = 3  # the drafter's tokens a step
# a drafter's step: verify rows of three drafts, two drafts and none, then a new turn's chunk over its cached prefix
# and a new prompt
QUERY_LENS = [4, 3, 1, 20, 9]
SEQ_LENS = [30, 7, 50, 120, 9]
DRAFT_COUNTS = [3, 2, 0, -1, -1]  # vLLM's runner: a decode row's drafts, -1 for the other rows
NUM_VERIFIES = 3
OTHER_GROUP_OFFSET = 500  # another GDN group's slots: the first group's moved by this many


@pytest.fixture
def builder(monkeypatch) -> BlackwellGDNMetadataBuilder:
    # vLLM's GDN builder reduced to what decides the builder's path: an engine with a drafter; the uploads on the host
    def vllm_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.use_spec_decode = True
        self.num_spec = NUM_SPECULATIVE_TOKENS

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", vllm_init)
    HostStagedBuffer.install(monkeypatch)
    scheduler_config = types.SimpleNamespace(max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)

    return BlackwellGDNMetadataBuilder(None, [], types.SimpleNamespace(scheduler_config=scheduler_config), "cpu")


def common_metadata(query_lens: list[int], seq_lens: list[int], draft_counts: list[int]) -> CommonAttentionMetadata:
    """The step's common attention metadata on the host, as vLLM's V2 runner gives it to a GDN group's builder, its
    block table one state slot per request (replaySSM's)."""
    qsl = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    lens = torch.tensor(seq_lens, dtype=torch.int32)

    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=lens,
        num_reqs=len(query_lens),
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=10 * torch.arange(1, len(query_lens) + 1, dtype=torch.int32)[:, None],
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64),
        is_prefilling=torch.tensor([count < 0 for count in draft_counts]),
        seq_lens_cpu_upper_bound=lens,
    )


def build(builder, query_lens, seq_lens, draft_counts):
    """The builder's metadata of the step, given the runner's draft counts."""
    return builder.build(
        0,
        common_metadata(query_lens, seq_lens, draft_counts),
        num_accepted_tokens=torch.ones(len(query_lens), dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor(draft_counts, dtype=torch.int32),
    )


def test_a_drafter_s_mixed_step_runs_its_verify_rows_under_replay_and_its_prefill_rows_after_them(builder):
    md = build(builder, QUERY_LENS, SEQ_LENS, DRAFT_COUNTS)

    # the verify rows: the first rows of ReplayState's tables; the prefill rows' tokens from their first on
    S = NUM_VERIFIES
    assert isinstance(md, ReplayMetadata) and md.num_rows == S and md.prefill_rows.tolist() == [3, 4]
    assert md.prefill_tokens == slice(sum(QUERY_LENS[:S]), sum(QUERY_LENS))
    prefill = md.prefill
    chunks = [-(-n // CONV_TOKENS) for n in QUERY_LENS[S:]]
    assert (prefill.num_decodes, prefill.num_prefills) == (0, len(QUERY_LENS) - S)
    assert prefill.num_actual_tokens == prefill.num_prefill_tokens == sum(QUERY_LENS[S:])
    # the prefill rows' query starts rebased to their first token, their slots, their flags
    assert prefill.non_spec_query_start_loc.tolist() == prefill.prefill_query_start_loc.tolist() == [0, 20, 29]
    assert prefill.non_spec_state_indices_tensor.tolist() == [40, 50]
    assert prefill.has_initial_state.tolist() == [True, False]
    # a conv program per CONV_TOKENS tokens begun of a prefill row, by its row among the prefill rows and its chunk
    assert prefill.batch_ptr.tolist() == [i for i, n in enumerate(chunks) for _ in range(n)]
    assert prefill.token_chunk_offset_ptr.tolist() == [c for n in chunks for c in range(n)]
    assert prefill._pear_layout.D == 0


def test_another_gdn_group_takes_the_prefill_rows_with_its_own_state_slots(builder):
    # the runner builds the first group's metadata and derives every other group's (update_block_table): the verify
    # rows' slots of every group are ReplayState's, the prefill rows' that group's
    m = common_metadata(QUERY_LENS, SEQ_LENS, DRAFT_COUNTS)
    md = build(builder, QUERY_LENS, SEQ_LENS, DRAFT_COUNTS)
    group_table = m.block_table_tensor + OTHER_GROUP_OFFSET

    group_md = builder.update_block_table(md, group_table, m.slot_mapping)

    S = NUM_VERIFIES
    prefill, group_prefill = md.prefill, group_md.prefill
    assert (group_md.num_rows, group_md.prefill_tokens) == (md.num_rows, md.prefill_tokens)
    assert group_md.prefill_rows is md.prefill_rows
    assert group_prefill.non_spec_state_indices_tensor.tolist() == group_table[S:, 0].tolist()
    assert group_prefill._pear_layout.idx.tolist() == group_prefill._pear_layout.pf_idx.tolist() == [540, 550]
    # the first group's slots unchanged
    assert prefill.non_spec_state_indices_tensor.tolist() == prefill._pear_layout.idx.tolist() == [40, 50]


@pytest.mark.parametrize(
    "query_lens, seq_lens, draft_counts",
    [
        # a step of verify rows alone
        ([4, 3, 1], [30, 7, 50], [3, 2, 0]),
        # a FULL graph's verify rows, padded to the graph's requests: the padding rows hold no token, no drafts
        ([4, 4, 4, 0], [30, 7, 50, 0], [3, 3, 3, -1]),
    ],
)
def test_a_step_of_verify_rows_alone_runs_them_all_under_replay(builder, query_lens, seq_lens, draft_counts):
    m = common_metadata(query_lens, seq_lens, draft_counts)

    md = build(builder, query_lens, seq_lens, draft_counts)

    assert md == ReplayMetadata(num_rows=len(query_lens))
    # one metadata for every group (each layer reads its group's slots in ReplayState), and so a graph's capture
    assert builder.update_block_table(md, m.block_table_tensor + OTHER_GROUP_OFFSET, m.slot_mapping) is md
    assert builder.build_for_cudagraph_capture(m) == ReplayMetadata(num_rows=len(query_lens))


def test_prefill_rows_ahead_of_a_verify_row_run_on_their_gathered_tokens(builder):
    # vLLM's runner sorts the rows without drafts by (k + 1 tokens first, then length): a prompt chunk of k + 1 tokens
    # lands before a decode row without drafts
    query_lens, seq_lens, draft_counts = [4, 4, 1, 9], [30, 104, 50, 9], [3, -1, 0, -1]
    m = common_metadata(query_lens, seq_lens, draft_counts)

    md = build(builder, query_lens, seq_lens, draft_counts)
    group_md = builder.update_block_table(md, m.block_table_tensor + OTHER_GROUP_OFFSET, m.slot_mapping)

    # the verify core over every row (it skips the prefill rows), GDNCore over the prefill rows' tokens packed
    assert md.num_rows == len(query_lens) and md.prefill_rows.tolist() == [1, 3]
    assert md.prefill_tokens.tolist() == [4, 5, 6, 7, *range(9, 18)]
    prefill = md.prefill
    assert prefill.non_spec_query_start_loc.tolist() == prefill.prefill_query_start_loc.tolist() == [0, 4, 13]
    assert prefill.non_spec_state_indices_tensor.tolist() == [20, 40]
    assert prefill.has_initial_state.tolist() == [True, False]
    assert group_md.prefill.non_spec_state_indices_tensor.tolist() == group_md.prefill._pear_layout.pf_idx.tolist()
    assert group_md.prefill._pear_layout.idx.tolist() == [520, 540]


def test_the_replay_core_scatters_the_gathered_prefill_rows_outputs_back(builder, monkeypatch):
    query_lens, seq_lens, draft_counts = [4, 4, 1, 9], [30, 104, 50, 9], [3, -1, 0, -1]
    md = build(builder, query_lens, seq_lens, draft_counts)
    verified, prefilled = [], []
    monkeypatch.setattr(ReplayState, "instance", types.SimpleNamespace(step=lambda: None))
    monkeypatch.setattr(GDNVerifyCore, "run", lambda layer, qkvz, ba, out, step, num_rows: verified.append(num_rows))

    def mixed_step(layer, qkvz, ba, out, layout):
        # every gathered output row: its token's first projection column
        prefilled.append((qkvz, ba, layout))
        out.copy_(qkvz[:, :1, None].expand_as(out))

    monkeypatch.setattr(GDNCore, "mixed_step", mixed_step)
    num_tokens = sum(query_lens)
    qkvz = torch.arange(num_tokens, dtype=torch.float32)[:, None].expand(num_tokens, 3).contiguous()
    ba = qkvz[:, :2] + 100
    out = torch.full((num_tokens, 2, 1), -1.0)

    BlackwellGDNAttention.replay_core(BlackwellGDNAttention.__new__(BlackwellGDNAttention), qkvz, ba, out, md)

    tokens = md.prefill_tokens.tolist()
    ((got_qkvz, got_ba, layout),) = prefilled
    assert verified == [len(query_lens)] and layout is md.prefill._pear_layout
    assert got_qkvz[:, 0].tolist() == tokens and got_ba[:, 0].tolist() == [token + 100 for token in tokens]
    # the prefill rows' outputs written, the verify rows' left to the verify core
    assert out[:, 0, 0].tolist() == [token if token in tokens else -1.0 for token in range(num_tokens)]


def test_the_gpu_backends_gdn_layers_keep_one_state_slot_per_request(monkeypatch):
    # vLLM's spec of a drafter's GDN layer: a state slot per draft token besides the running one
    vllm_spec = MambaSpec(
        shapes=((3, 8), (2, 4, 4)), dtypes=(torch.bfloat16, torch.float32), block_size=-1, num_speculative_blocks=3
    )
    monkeypatch.setattr(QwenGatedDeltaNetAttention, "get_kv_cache_spec", lambda self, vllm_config: vllm_spec)

    # the shared layer keeps vLLM's spec; both GPU backends' layers replay a request's last verify block from one slot
    assert "get_kv_cache_spec" not in vars(RolloutGDNAttention)
    for layer_cls in (BlackwellGDNAttention, HopperGDNAttention):
        spec = layer_cls.__new__(layer_cls).get_kv_cache_spec(None)
        assert spec == dataclasses.replace(vllm_spec, num_speculative_blocks=0)
