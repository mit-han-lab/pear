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

# The helpers are those of verl's tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py; the tests are of PearReplayBuffer's
# session caps and pass rates.
import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from pear.trainer.replay_buffer import PearReplayBuffer

# Small poll interval so the blocking consumer reacts to producer writes quickly.
POLL_INTERVAL = 0.05


def _replay_buffer(refill_fn, rollout_n: int, **arguments) -> PearReplayBuffer:
    """PEAR's replay buffer of an async trainer without session caps, with the arguments a test sets of it."""
    return PearReplayBuffer(
        **{
            "trainer_mode": "separate_async",
            "trainer_config": {},
            "max_off_policy_threshold": 8,
            "max_off_policy_strategy": "drop",
            "sampler_kwargs": {},
            "poll_interval": POLL_INTERVAL,
            "refill_fn": refill_fn,
            "rollout_n": rollout_n,
            "max_inflight_sessions": None,
            "max_inflight_sessions_by_tool": None,
            **arguments,
        }
    )


class FakeRefiller:
    """Produce fresh terminal prompts when the replay buffer requests replacements."""

    def __init__(self, partition_id: str, global_steps: int, sessions: int = 1, rewards: list[float] | None = None):
        self.partition_id = partition_id
        self.global_steps = global_steps
        self.sessions = sessions
        # When set, refilled groups carry these per-session rewards so they survive zero-variance filtering.
        self.rewards = rewards
        self.calls: list[int] = []
        self.produced_uids: list[str] = []

    def __call__(self, num_prompts: int) -> int:
        self.calls.append(num_prompts)
        specs = [
            PromptSpec(
                uid=_uid(),
                status="finished",
                sessions=self.sessions,
                global_steps=self.global_steps,
                rewards=self.rewards,
            )
            for _ in range(num_prompts)
        ]
        # Synchronous produce (already-terminal) so the sample loop sees them on its next poll.
        _produce(self.partition_id, specs).join_and_check()
        self.produced_uids.extend(spec.uid for spec in specs)
        return num_prompts


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


def _set_prompt_status(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Transition an existing prompt to a new status (e.g. running -> finished).

    Mirrors the rollout side flipping a GRPO group's status once it terminates.
    The prompt tag is updated in place; its trajectory values are untouched.
    """
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
    )


@dataclass
class PromptSpec:
    """A prompt group and the trajectories that precede its status update."""

    uid: str
    status: str
    sessions: int = 1
    global_steps: int = 0
    rewards: list[float] | None = None
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Write complete trajectory groups before publishing their prompt status."""

    def __init__(self, partition_id: str, specs: list[PromptSpec]):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.specs = specs
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for spec in self.specs:
                for session_id in range(spec.sessions):
                    key = _trajectory_key(spec.uid, session_id)
                    fields = {"input_ids": torch.tensor([1, 2, 3])}
                    tag = {"is_prompt": False, "seq_len": 3, "global_steps": spec.global_steps}
                    if spec.rewards is not None:
                        fields["extra_fields"] = {"reward_extra_info": {"acc": float(spec.rewards[session_id])}}
                    tq.kv_put(
                        key=key,
                        partition_id=self.partition_id,
                        fields=fields,
                        tag=tag,
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status, "global_steps": spec.global_steps},
                )
        except Exception as e:  # surfaced to the test via join_and_check()
            self.error = e

    def join_and_check(self, timeout: float = 10.0) -> None:
        self.join(timeout)
        assert not self.is_alive(), "RolloutProducer thread did not finish in time"
        if self.error is not None:
            raise self.error


class SampleConsumer(threading.Thread):
    """Runs the blocking ``ReplayBuffer.sample`` in a background thread so the test
    can assert that it stays blocked until the producer supplies enough data."""

    def __init__(self, rb: PearReplayBuffer, partition_id: str, batch_size: int):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.result: KVBatchMeta | None = None
        self.metrics: dict | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result, self.metrics = self.rb.sample(
                global_steps=0,
                partition_id=self.partition_id,
                batch_size=self.batch_size,
            )
        except Exception as e:
            self.error = e

    def result_or_raise(self, timeout: float = 10.0) -> KVBatchMeta:
        self.join(timeout)
        assert not self.is_alive(), "SampleConsumer thread did not finish in time"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _produce(partition_id: str, specs: list[PromptSpec]) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs)
    producer.start()
    return producer


def _clear_partition(partition_id: str) -> None:
    """Best-effort cleanup of every key written into a partition."""
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


def test_async_dapo_reports_the_pass_rates_of_the_groups_it_judged(tq_init, partition_id):
    solved = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    unsolved = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [solved, unsolved, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _replay_buffer(refiller, rollout_n=2, trainer_mode="colocate_async", filter_groups_metric="acc")
    try:
        _, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        # judged in this sample: the three groups (pass rates 1, 0, 0.5) and the two refills of the evicted ones
        # (0.5 each)
        assert refiller.calls == [2]
        assert metrics["validation/filter_groups/avg@n"] == pytest.approx(0.5)
        assert metrics["validation/filter_groups/passrate/zero"] == pytest.approx(1 / 5)
        assert metrics["validation/filter_groups/passrate/one"] == pytest.approx(1 / 5)
        assert rb._pass_rate_metrics(partition_id) == {}  # consumed: the next sample reports its own groups
    finally:
        _clear_partition(partition_id)


def test_the_pass_rates_are_also_reported_over_each_tool_s_groups(tq_init, partition_id):
    arm = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 0.0])
    arm_solved = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    table = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    plain = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 0.0])  # its row selects no tool
    _produce(partition_id, [arm, arm_solved, table, plain]).join_and_check()
    for spec, tool_selection in ((arm, ["arm"]), (arm_solved, ["arm"]), (table, ["table"])):
        tq.kv_put(key=spec.uid, partition_id=partition_id, tag={"tool_selection": tool_selection})
    rb = _replay_buffer(
        lambda num_prompts: num_prompts, rollout_n=2, trainer_mode="colocate_async", filter_groups_metric="acc"
    )
    try:
        rb._sync_metadata_from_transfer_queue()
        rb._dapo_filtered_keys(partition_id)
        metrics = rb._pass_rate_metrics(partition_id)

        prefix = rb._metrics_prefix(partition_id)
        assert metrics[f"{prefix}/filter_groups/avg@n"] == pytest.approx((0.5 + 1.0 + 0.0 + 0.5) / 4)
        assert metrics[f"{prefix}/filter_groups/avg@n/arm"] == pytest.approx(0.75)
        assert metrics[f"{prefix}/filter_groups/avg@n/table"] == pytest.approx(0.0)
        assert sorted(key for key in metrics if "/avg@n/" in key) == [
            f"{prefix}/filter_groups/avg@n/arm",
            f"{prefix}/filter_groups/avg@n/table",
        ]
    finally:
        _clear_partition(partition_id)


class SimulatedRollout(threading.Thread):
    """A rollout that starts every submitted group, stores one more session of each running group per tick and marks
    a group finished once all its sessions are stored; ``refill`` is the replay buffer's ``refill_fn``."""

    def __init__(self, partition_id: str, sessions: int, rewards_of, tick: float = 0.01, supply: int | None = None):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.sessions = sessions
        self.rewards_of = rewards_of  # group index -> the per-session DAPO metric values
        self.tick = tick
        self.supply = supply  # groups it can submit in all (a one-pass evaluation); None: as many as asked
        self.lock = threading.Lock()
        self.pending: list[str] = []
        self.stored: dict[str, int] = {}  # running uid -> sessions stored so far
        self.index: dict[str, int] = {}  # uid -> submission order
        self.refill_calls: list[int] = []
        self.max_inflight = 0  # sessions generating right after each refill, measured on the rollout's own state
        self.stopping = threading.Event()
        self.error: Exception | None = None

    def refill(self, num_prompts: int) -> int:
        with self.lock:
            if self.supply is not None:
                num_prompts = min(num_prompts, self.supply - len(self.index))
            for _ in range(num_prompts):
                uid = _uid()
                self.index[uid] = len(self.index)
                _set_prompt_status(self.partition_id, uid, "pending", global_steps=0)
                self.pending.append(uid)
            self.refill_calls.append(num_prompts)
            inflight = self.sessions * len(self.pending) + sum(self.sessions - n for n in self.stored.values())
            self.max_inflight = max(self.max_inflight, inflight)
        return num_prompts

    def run(self) -> None:
        try:
            while not self.stopping.is_set():
                with self.lock:
                    for uid in self.pending:
                        _set_prompt_status(self.partition_id, uid, "running", global_steps=0)
                        self.stored[uid] = 0
                    self.pending.clear()
                    for uid in list(self.stored):
                        session_id = self.stored[uid]
                        reward = self.rewards_of(self.index[uid])[session_id]
                        tq.kv_put(
                            key=_trajectory_key(uid, session_id),
                            partition_id=self.partition_id,
                            fields={
                                "input_ids": torch.tensor([1, 2, 3]),
                                "extra_fields": {"reward_extra_info": {"acc": reward}},
                            },
                            tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
                        )
                        self.stored[uid] += 1
                        if self.stored[uid] == self.sessions:
                            _set_prompt_status(self.partition_id, uid, "finished", global_steps=0)
                            del self.stored[uid]
                time.sleep(self.tick)
        except Exception as e:  # surfaced by stop()
            self.error = e

    def stop(self) -> None:
        self.stopping.set()
        self.join(timeout=10.0)
        if self.error is not None:
            raise self.error


def test_session_cap_fills_a_batch_larger_than_the_cap_by_refilling_as_groups_finish(tq_init, partition_id):
    # 12 sessions = 3 groups of 4 in flight, a batch of 10 groups: only refills as groups finish can fill it.
    rollout = SimulatedRollout(partition_id, sessions=4, rewards_of=lambda index: [0.0, 1.0, 0.0, 1.0])
    rb = _replay_buffer(
        rollout.refill, rollout_n=rollout.sessions, filter_groups_metric="acc", max_inflight_sessions=12
    )
    rollout.start()
    consumer = SampleConsumer(rb, partition_id, batch_size=10)
    try:
        consumer.start()
        batch = consumer.result_or_raise(timeout=60.0)

        assert len(_uids_of(batch.keys)) == 10 and len(batch.keys) == 40
        assert rollout.max_inflight <= 12
        assert len(rollout.refill_calls) > 1, "the rollout must be refilled while sampling, not fed once"
    finally:
        rollout.stop()
        _clear_partition(partition_id)


def test_session_cap_replaces_dapo_evicted_groups_within_the_cap(tq_init, partition_id):
    # every other group is all-correct: DAPO evicts it, and its room is refilled like any finished group's
    rollout = SimulatedRollout(
        partition_id, sessions=4, rewards_of=lambda index: [1.0] * 4 if index % 2 else [0.0, 1.0, 0.0, 1.0]
    )
    rb = _replay_buffer(rollout.refill, rollout_n=rollout.sessions, filter_groups_metric="acc", max_inflight_sessions=8)
    rollout.start()
    consumer = SampleConsumer(rb, partition_id, batch_size=5)
    try:
        consumer.start()
        batch = consumer.result_or_raise(timeout=60.0)

        assert all(rollout.index[uid] % 2 == 0 for uid in _uids_of(batch.keys)), (
            "only groups with signal are trained on"
        )
        assert len(_uids_of(batch.keys)) == 5
        assert rollout.max_inflight <= 8
        assert consumer.metrics[f"{rb._metrics_prefix(partition_id)}/filter_groups/evicted_samples"] >= 1
    finally:
        rollout.stop()
        _clear_partition(partition_id)


def test_session_cap_samples_the_last_groups_once_the_refill_runs_dry(tq_init, partition_id):
    # 5 groups in all, at most 3 in flight: once the last ones are out the refill submits nothing, and the loop must
    # still sample them instead of calling the refill again at once, forever
    rollout = SimulatedRollout(partition_id, sessions=4, rewards_of=lambda index: [0.0, 1.0, 0.0, 1.0], supply=5)
    rb = _replay_buffer(
        rollout.refill, rollout_n=rollout.sessions, filter_groups_metric="acc", max_inflight_sessions=12
    )
    rollout.start()
    consumer = SampleConsumer(rb, partition_id, batch_size=5)
    try:
        consumer.start()
        batch = consumer.result_or_raise(timeout=30.0)

        assert len(_uids_of(batch.keys)) == 5 and len(batch.keys) == 20
        assert rollout.max_inflight <= 12
    finally:
        rollout.stop()
        _clear_partition(partition_id)


def test_queued_groups_are_outside_the_inflight_sessions(tq_init, partition_id):
    # a restored group waiting for room under the cap generates nothing: only pending and running sessions count
    queued, pending, running = _uid(), _uid(), _uid()
    for uid, status in ((queued, "queued"), (pending, "pending"), (running, "running")):
        _set_prompt_status(partition_id, uid, status, global_steps=3)
    tq.kv_put(
        key=_trajectory_key(running),
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": 3},
    )
    rb = _replay_buffer(lambda num_prompts: num_prompts, rollout_n=4, max_inflight_sessions=12)
    try:
        rb._sync_metadata_from_transfer_queue()
        assert rb._inflight_sessions(partition_id) == 4 + 3
    finally:
        _clear_partition(partition_id)


def test_a_restored_group_counts_only_the_sessions_it_has_yet_to_run(tq_init, partition_id):
    # a group restored from a checkpoint is pending again with the sessions it stored before: only the others generate
    pending = _uid()
    _set_prompt_status(partition_id, pending, "pending", global_steps=3)
    for session_id in (0, 2):
        tq.kv_put(
            key=_trajectory_key(pending, session_id),
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "seq_len": 3, "global_steps": 3},
        )
    rb = _replay_buffer(lambda num_prompts: num_prompts, rollout_n=4, max_inflight_sessions=12)
    try:
        rb._sync_metadata_from_transfer_queue()
        assert rb._inflight_sessions(partition_id) == 2
    finally:
        _clear_partition(partition_id)


def _tool_capped_rb(refill_fn, max_inflight_sessions_by_tool: dict[str, int]) -> PearReplayBuffer:
    return _replay_buffer(
        refill_fn, rollout_n=4, max_inflight_sessions=16, max_inflight_sessions_by_tool=max_inflight_sessions_by_tool
    )


def _put_prompt_with_tools(partition_id: str, uid: str, status: str, tool_selection: list[str] | None) -> None:
    """A prompt as the trainer submits it: its tag names the tools its row selects."""
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": 3, "tool_selection": tool_selection},
    )


def test_the_inflight_sessions_of_a_tool_are_those_of_the_groups_whose_rows_select_it(tq_init, partition_id):
    sim, both, plain = _uid(), _uid(), _uid()
    for uid, tool_selection in ((sim, ["sim"]), (both, ["sim", "search"]), (plain, None)):
        _put_prompt_with_tools(partition_id, uid, "pending", tool_selection)
    # the rollout's status update writes no tool selection: the tag keeps the submitted one
    _set_prompt_status(partition_id, both, "running", global_steps=3)
    tq.kv_put(
        key=_trajectory_key(both),
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": 3},
    )
    rb = _tool_capped_rb(lambda num_prompts, **kwargs: num_prompts, {"sim": 8})
    try:
        rb._sync_metadata_from_transfer_queue()
        assert rb._inflight_sessions(partition_id, "sim") == 4 + 3
        assert rb._inflight_sessions(partition_id, "search") == 3
        assert rb._inflight_sessions(partition_id) == 4 + 3 + 4
    finally:
        _clear_partition(partition_id)


def test_the_refill_gets_the_room_of_each_capped_tool(tq_init, partition_id):
    # two running groups of 4 sessions under a cap of 16, one of them selecting the tool capped at 6: room for 2 groups,
    # 2 sessions of the tool
    _put_prompt_with_tools(partition_id, _uid(), "running", ["sim"])
    _put_prompt_with_tools(partition_id, _uid(), "running", None)
    refill_calls = []

    class Refilled(Exception):
        """Ends the wait at its first refill."""

    def refill(num_prompts, **kwargs):
        refill_calls.append((num_prompts, kwargs))
        raise Refilled

    rb = _tool_capped_rb(refill, {"sim": 6})
    try:
        with pytest.raises(Refilled):
            rb.wait_for_sampleable(global_steps=3, partition_id=partition_id, target_count=1)
        assert refill_calls == [(2, {"free_sessions_by_tool": {"sim": 2}})]
    finally:
        _clear_partition(partition_id)


def test_session_cap_by_tool_requires_the_session_cap():
    with pytest.raises(ValueError, match="max_inflight_sessions_by_tool caps tools within max_inflight_sessions"):
        _replay_buffer(lambda num_prompts, **kwargs: num_prompts, rollout_n=4, max_inflight_sessions_by_tool={"sim": 8})


def test_the_validation_partition_is_not_refilled_by_room():
    def refill(num_prompts, **kwargs):
        raise AssertionError("the session cap is the training partition's")

    rb = _tool_capped_rb(refill, {"sim": 8})

    assert not rb._refill_free_room("val")
