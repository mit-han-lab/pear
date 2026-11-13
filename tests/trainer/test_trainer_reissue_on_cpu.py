# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The helpers are those of verl's tests/trainer/ppo/v1/test_reissue_inflight_on_cpu.py; the tests are of PearTrainer's
# resume of in-flight prompt groups and of its submissions.
import io
import logging
import uuid
from types import SimpleNamespace

import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

import pear.trainer.trainer as trainer_module
from pear.trainer.replay_buffer import PearReplayBuffer
from pear.trainer.trainer import PearTrainer, RestoredGroup
from verl.trainer.ppo.v1 import trainer_base
from verl.utils import tensordict_utils as tu

ROLLOUT_N = 4  # sessions per prompt group


def _uid() -> str:
    # uid must not contain "_" because trajectory keys are "{uid}_{session}_{index}".
    return uuid.uuid4().hex


class RecordingAgentLoopManager:
    """Records every batch handed to ``generate_sequences`` so re-issues can be asserted."""

    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


class ReplayBufferStub:
    """The parts of the async replay buffer the re-issue reads: the session cap and the staleness of a dispatch step."""

    is_stale = PearReplayBuffer.is_stale

    def __init__(self):
        self.max_inflight_sessions = None
        self.max_off_policy_threshold = None
        self.max_off_policy_strategy = "drop"


def _make_trainer_stub(trainer_mode: str = "separate_async", global_steps: int = 8):
    """Bind the real ``_reissue_inflight_prompts`` to a stub exposing only what it uses."""
    stub = type("Stub", (), {})()
    stub.trainer_mode = trainer_mode
    stub.global_steps = global_steps
    stub.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": ROLLOUT_N}}})
    stub.agent_loop_manager = RecordingAgentLoopManager()
    stub.replay_buffer = ReplayBufferStub()
    stub._reissue_queue = []
    stub.INFLIGHT_PROMPT_STATUSES = PearTrainer.INFLIGHT_PROMPT_STATUSES
    stub._reissue_inflight_prompts = PearTrainer._reissue_inflight_prompts.__get__(stub)
    stub._reissue_prompt_groups = PearTrainer._reissue_prompt_groups.__get__(stub)
    stub._reissue_groups = PearTrainer._reissue_groups.__get__(stub)
    return stub


def _submit_prompt(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Mirror async prompt submission: store the per-row prompt data as fields
    plus the status tag under the ``{uid}`` key (a single-row batch).

    ``global_steps`` is a scalar broadcast across the batch, which TransferQueue cannot store as a
    field; the trainer excludes it and re-derives it from the tag on restore.
    """
    batch = tu.get_tensordict(
        {
            "uid": [uid],
            "raw_prompt": [f"prompt-for-{uid}"],
            "index": torch.tensor([0]),
        }
    )
    tag = {"is_prompt": True, "status": status, "global_steps": global_steps}
    tq.kv_batch_put(keys=[uid], partition_id=partition_id, tags=[tag], fields=batch)


def _add_trajectory(partition_id: str, uid: str, session_id: int, global_steps: int) -> str:
    """Attach one finished trajectory to a prompt (composite key), as the rollout would."""
    key = f"{uid}_{session_id}_0"
    tq.kv_put(
        key=key,
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"status": "success", "seq_len": 3, "global_steps": global_steps},
    )
    return key


def _dispatched_session_ids_by_group(stub) -> dict[tuple[str, int], list[int]]:
    """(uid, the batch's global_steps) -> the session ids each dispatched row runs."""
    return {
        (uid, int(batch["global_steps"])): list(unfinished_session_ids)
        for batch in stub.agent_loop_manager.batches
        for uid, unfinished_session_ids in zip(batch["uid"], batch["__unfinished_session_ids__"], strict=True)
    }


def _prompt_status(partition_id: str, uid: str) -> str | None:
    tag = tq.kv_list(partition_id=partition_id).get(partition_id, {}).get(uid)
    return None if tag is None else tag.get("status")


def _clear_partition(partition_id: str) -> None:
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _submitting_trainer(max_inflight_sessions: int | None) -> PearTrainer:
    """PEAR's trainer with the parts a submission uses, without its constructor."""
    trainer = PearTrainer.__new__(PearTrainer)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 4
    trainer.agent_loop_manager = RecordingAgentLoopManager()
    trainer.replay_buffer = SimpleNamespace(max_inflight_sessions=max_inflight_sessions)
    return trainer


def test_submission_tags_the_groups_of_a_batch_without_extra_info_with_no_tool(monkeypatch):
    # a dataset whose rows carry no extra_info, as verl's own: its groups count under no tool's session cap
    trainer = _submitting_trainer(max_inflight_sessions=None)
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b"],
            "index": torch.tensor([0, 1]),
        }
    )
    tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
    puts = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))

    assert trainer._submit_batch_to_rollout(batch) == 2
    assert [tag["tool_selection"] for tag in puts[0]["tags"]] == [None, None]
    assert trainer.agent_loop_manager.batches == [batch]


def test_submission_tags_each_group_with_the_tools_its_row_selects(monkeypatch):
    # a session cap per tool counts a group under the tool_selection of its tag (None: a row that selects none)
    trainer = _submitting_trainer(max_inflight_sessions=16)
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b"],
            "index": torch.tensor([0, 1]),
            "extra_info": [{"index": 0, "tool_selection": ["sim"]}, {"index": 1}],
        }
    )
    tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
    puts = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))

    assert trainer._submit_batch_to_rollout(batch) == 2
    assert [tag["tool_selection"] for tag in puts[0]["tags"]] == [["sim"], None]


def test_submission_keeps_every_tag_verl_gives_a_prompt(monkeypatch):
    # verl's tags with one key more than it has today (PPOTrainer._prompt_tags): the tools are added to them
    trainer = _submitting_trainer(max_inflight_sessions=16)
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b"],
            "index": torch.tensor([0, 1]),
            "extra_info": [{"index": 0, "tool_selection": ["sim"]}, {"index": 1}],
        }
    )
    tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
    verl_tag = {"is_prompt": True, "status": "pending", "global_steps": trainer.global_steps, "priority": 1}
    monkeypatch.setattr(
        trainer_base.PPOTrainer, "_prompt_tags", lambda self, batch: [dict(verl_tag) for _ in range(len(batch))]
    )
    puts = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))

    assert trainer._submit_batch_to_rollout(batch) == 2
    assert puts[0]["tags"] == [{**verl_tag, "tool_selection": ["sim"]}, {**verl_tag, "tool_selection": None}]


def test_a_refill_without_restored_groups_takes_the_prompts_it_was_asked_for_and_submits_them():
    stub = type("Stub", (), {})()
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b", "prompt-c"],
            "index": torch.tensor([0, 1, 2]),
        }
    )
    requested = []
    submitted = []
    stub._next_train_batch = lambda num_prompts: requested.append(num_prompts) or batch
    stub._submit_batch_to_rollout = lambda value: submitted.append(value) or len(value)
    stub.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": ROLLOUT_N}}})
    stub._reissue_queue = []
    stub._add_prompts_to_generate = PearTrainer._add_prompts_to_generate.__get__(stub)

    assert stub._add_prompts_to_generate(3) == 3
    assert requested == [3]
    assert submitted == [batch]


def test_reissue_runs_only_the_sessions_inflight_groups_had_not_stored(tq_init, partition_id):
    """In-flight groups run the sessions they had not stored, under their dispatch step; the stored sessions and the
    finished groups stay untouched."""
    pending = _uid()
    running = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=2)
    _submit_prompt(partition_id, running, "running", global_steps=2)
    _submit_prompt(partition_id, finished, "finished", global_steps=2)
    stored = [_add_trajectory(partition_id, running, session_id=i, global_steps=2) for i in (0, 2)]
    finished_trajectory = _add_trajectory(partition_id, finished, session_id=0, global_steps=2)

    stub = _make_trainer_stub(global_steps=3)
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 2

        # pending stored nothing and starts over under the resumed step; running keeps its sessions and its step
        assert _dispatched_session_ids_by_group(stub) == {(pending, 3): [0, 1, 2, 3], (running, 2): [1, 3]}
        assert _prompt_status(partition_id, pending) == "pending"
        assert _prompt_status(partition_id, running) == "pending"
        assert _prompt_status(partition_id, finished) == "finished"

        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert all(key in remaining for key in stored) and finished_trajectory in remaining
        assert remaining[pending]["global_steps"] == 3 and remaining[running]["global_steps"] == 2
    finally:
        _clear_partition(partition_id)


def test_reissue_preserves_prompt_data_and_resets_global_steps(tq_init, partition_id):
    """A group that stored nothing keeps its prompt fields and starts over under the resumed dispatch step."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", global_steps=7)

    stub = _make_trainer_stub(global_steps=8)
    try:
        stub._reissue_inflight_prompts(partition_id)

        assert len(stub.agent_loop_manager.batches) == 1
        batch = stub.agent_loop_manager.batches[0]
        assert list(batch["uid"]) == [uid]
        assert list(batch["raw_prompt"]) == [f"prompt-for-{uid}"]
        assert int(batch["global_steps"]) == 8
        assert [list(session_ids) for session_ids in batch["__unfinished_session_ids__"]] == [[0, 1, 2, 3]]
        tag = tq.kv_list(partition_id=partition_id).get(partition_id, {})[uid]
        assert tag["global_steps"] == 8
    finally:
        _clear_partition(partition_id)


def test_reissue_combines_original_steps_under_resumed_step(tq_init, partition_id):
    """Groups that stored nothing all start over under the resumed step, in one batch, whatever their original step."""
    step2 = [_uid(), _uid()]
    step5 = [_uid()]
    for uid in step2:
        _submit_prompt(partition_id, uid, "pending", global_steps=2)
    for uid in step5:
        _submit_prompt(partition_id, uid, "running", global_steps=5)

    stub = _make_trainer_stub(global_steps=6)
    try:
        reissued = stub._reissue_inflight_prompts(partition_id)
        assert reissued == 3

        assert len(stub.agent_loop_manager.batches) == 1
        batch = stub.agent_loop_manager.batches[0]
        assert int(batch["global_steps"]) == 6
        assert set(batch["uid"]) == set(step2 + step5)
    finally:
        _clear_partition(partition_id)


def test_reissue_dispatches_one_batch_per_dispatch_step(tq_init, partition_id):
    """Groups that kept sessions go out in one batch per original step, each row with its own sessions to run."""
    step5 = [_uid(), _uid()]
    step6 = _uid()
    for uid in step5:
        _submit_prompt(partition_id, uid, "running", global_steps=5)
    _submit_prompt(partition_id, step6, "running", global_steps=6)
    for session_id in (0, 2):
        _add_trajectory(partition_id, step5[0], session_id=session_id, global_steps=5)
    _add_trajectory(partition_id, step5[1], session_id=1, global_steps=5)
    _add_trajectory(partition_id, step6, session_id=3, global_steps=6)

    stub = _make_trainer_stub(global_steps=7)
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 3

        assert sorted(int(batch["global_steps"]) for batch in stub.agent_loop_manager.batches) == [5, 6]
        assert _dispatched_session_ids_by_group(stub) == {
            (step5[0], 5): [1, 3],
            (step5[1], 5): [0, 2, 3],
            (step6, 6): [0, 1, 2],
        }
    finally:
        _clear_partition(partition_id)


def test_reissue_finishes_groups_that_stored_every_session(tq_init, partition_id):
    """A group whose sessions were all stored before its terminal status was written finishes without a dispatch."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", global_steps=4)
    for session_id in range(ROLLOUT_N):
        _add_trajectory(partition_id, uid, session_id=session_id, global_steps=4)

    stub = _make_trainer_stub(global_steps=5)
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 0
        assert stub.agent_loop_manager.batches == []

        tag = tq.kv_list(partition_id=partition_id).get(partition_id, {})[uid]
        assert tag["status"] == "finished" and tag["global_steps"] == 4 and tag["is_prompt"]
    finally:
        _clear_partition(partition_id)


def test_reissue_runs_again_a_session_whose_put_had_not_written_its_data(tq_init, partition_id):
    """A checkpoint taken during a put holds the put's key without its data and tag: that session runs again, so a
    group whose last session it was is not complete."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", global_steps=4)
    for session_id in range(ROLLOUT_N - 1):
        _add_trajectory(partition_id, uid, session_id=session_id, global_steps=4)
    # the first step of a put: the key registered, its fields and tag not written yet
    unwritten = f"{uid}_{ROLLOUT_N - 1}_0"
    tq.get_client().kv_retrieve_meta(keys=[unwritten], partition_id=partition_id, create=True)
    assert tq.kv_list(partition_id=partition_id)[partition_id][unwritten] == {}

    stub = _make_trainer_stub(global_steps=5)
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 1

        assert _dispatched_session_ids_by_group(stub) == {(uid, 4): [ROLLOUT_N - 1]}
        assert unwritten not in tq.kv_list(partition_id=partition_id).get(partition_id, {})
    finally:
        _clear_partition(partition_id)


def test_reissue_restarts_groups_too_stale_to_be_trained(tq_init, partition_id):
    """A group whose dispatch step is already too old to be trained drops what it stored and starts over."""
    stale = _uid()
    fresh = _uid()
    _submit_prompt(partition_id, stale, "running", global_steps=5)
    _submit_prompt(partition_id, fresh, "running", global_steps=6)
    stale_trajectory = _add_trajectory(partition_id, stale, session_id=0, global_steps=5)
    fresh_trajectory = _add_trajectory(partition_id, fresh, session_id=0, global_steps=6)

    stub = _make_trainer_stub(global_steps=7)
    stub.replay_buffer.max_off_policy_threshold = 2  # trained at step 7: dispatched at 6 is fine, at 5 is too old
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 2

        assert _dispatched_session_ids_by_group(stub) == {(stale, 7): [0, 1, 2, 3], (fresh, 6): [1, 2, 3]}
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert stale_trajectory not in remaining and fresh_trajectory in remaining
    finally:
        _clear_partition(partition_id)


def test_reissue_logs_what_it_resumed_whatever_the_root_logger_admits(tq_init, partition_id, monkeypatch):
    # in a run the root logger's handlers are at ERROR (vLLM's server module, which the runner imports): the trainer's
    # lines go through the handler of its own module
    stream = io.StringIO()
    monkeypatch.setattr(trainer_module.logger.handlers[0], "stream", stream)
    for handler in logging.getLogger().handlers:
        monkeypatch.setattr(handler, "level", logging.ERROR)
    _submit_prompt(partition_id, _uid(), "pending", global_steps=2)

    stub = _make_trainer_stub(global_steps=3)
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 1
        assert "Re-issued 1 in-flight prompts with 4 sessions to run" in stream.getvalue()
    finally:
        _clear_partition(partition_id)


def test_save_load_then_reissue_only_inflight(tq_init, partition_id, tmp_path):
    """End-to-end recovery: after a save/load round-trip, only in-flight prompts are re-issued."""
    pending = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=4)
    _submit_prompt(partition_id, finished, "finished", global_steps=4)
    _add_trajectory(partition_id, finished, session_id=0, global_steps=4)

    ckpt_dir = str(tmp_path / "transfer_queue")
    tq.save_checkpoint(ckpt_dir, metadata={"global_steps": 4})
    _clear_partition(partition_id)
    tq.load_checkpoint(ckpt_dir)

    stub = _make_trainer_stub(global_steps=5)
    try:
        reissued = stub._reissue_inflight_prompts(partition_id)
        assert reissued == 1
        submitted_uids = {uid for batch in stub.agent_loop_manager.batches for uid in batch["uid"]}
        assert submitted_uids == {pending}
        assert int(stub.agent_loop_manager.batches[0]["global_steps"]) == 5
    finally:
        _clear_partition(partition_id)


def test_session_cap_queues_inflight_groups_with_their_prompt_data(tq_init, partition_id):
    """Under a session cap the restored groups wait as "queued" (nothing generates) with their stored sessions, their
    dispatch step and their prompt fields."""
    running = _uid()
    finished = _uid()
    _submit_prompt(partition_id, running, "running", global_steps=7)
    _submit_prompt(partition_id, finished, "finished", global_steps=7)
    running_trajectory = _add_trajectory(partition_id, running, session_id=0, global_steps=7)

    stub = _make_trainer_stub(global_steps=8)
    stub.replay_buffer.max_inflight_sessions = 16
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 1
        assert stub.agent_loop_manager.batches == []
        assert stub._reissue_queue == [
            RestoredGroup(uid=running, dispatch_step=7, unfinished_session_ids=[1, 2, 3], tool_selection=None)
        ]
        assert _prompt_status(partition_id, running) == "queued"
        assert _prompt_status(partition_id, finished) == "finished"
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert running_trajectory in remaining and remaining[running]["global_steps"] == 7

        stub._reissue_groups(stub._reissue_queue, partition_id)

        batch = stub.agent_loop_manager.batches[0]
        assert list(batch["raw_prompt"]) == [f"prompt-for-{running}"] and int(batch["global_steps"]) == 7
        assert _dispatched_session_ids_by_group(stub) == {(running, 7): [1, 2, 3]}
        assert _prompt_status(partition_id, running) == "pending"
    finally:
        _clear_partition(partition_id)
