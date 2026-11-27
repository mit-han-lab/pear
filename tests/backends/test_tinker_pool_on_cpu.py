from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pear.agent_loop.tasks import EpisodeTask
from pear.backends.tinker.config import TinkerConfig
from pear.backends.tinker.pool import RolloutPool
from pear.backends.tinker.state import Group, RunState


class Rollout:
    def __init__(self, constant=False, fail=False):
        self.constant, self.fail = constant, fail
        self.active, self.maximum = Counter(), Counter()
        self.calls = []

    async def run(self, task, sampler, uid, member):
        self.calls.append((uid, member))
        self.active[task.env_class] += 1
        self.maximum[task.env_class] = max(self.maximum[task.env_class], self.active[task.env_class])
        try:
            await asyncio.sleep(0.001)
            if self.fail:
                raise RuntimeError("sampler failed")
            return SimpleNamespace(
                reward=0.0 if self.constant or uid == "g000000000000" else float(member), generated_tokens=1
            )
        finally:
            self.active[task.env_class] -= 1


def pool_for(tmp_path, *, state=None, rollout=None, batch_groups=2, limit=20):
    config = TinkerConfig(
        "test",
        "test",
        [],
        [],
        str(tmp_path),
        group_size=2,
        batch_groups=batch_groups,
        max_sessions=4,
        max_groups_per_step=limit,
    )
    tasks = [EpisodeTask("test.Env", {}, "test", 0)]
    pool = RolloutPool(config, tasks, rollout or Rollout(), object(), {"test.Env": 4}, state or RunState())
    return pool


@pytest.mark.asyncio
async def test_constant_groups_are_replaced_until_a_full_batch_exists(tmp_path):
    pool = pool_for(tmp_path)
    pool.start()
    try:
        batch = await asyncio.wait_for(pool.take_batch(), 2)
        assert len(batch) == 2 and all(group.uid != "g000000000000" for group in batch)
        assert pool.state.counters["filtered"] == 1
        assert pool.rollout.maximum["test.Env"] <= 4
    finally:
        await pool.close()
    assert pool.rollout.active["test.Env"] == 0


@pytest.mark.asyncio
async def test_resume_runs_only_missing_members(tmp_path):
    completed = SimpleNamespace(reward=0.0, generated_tokens=1)
    group = Group("restored", 0, 2, [completed, None])
    pool = pool_for(tmp_path, state=RunState(step=3, groups=[group]), batch_groups=1)
    pool.start()
    try:
        [batch] = await asyncio.wait_for(pool.take_batch(), 2)
        assert batch.uid == "restored" and batch.trajectories[0] is completed
        assert [(uid, member) for uid, member in pool.rollout.calls if uid == "restored"] == [("restored", 1)]
    finally:
        await pool.close()


def test_resume_restarts_a_stale_partial_group_without_mixing_old_members(tmp_path):
    partial = Group("partial", 0, 1, [SimpleNamespace(reward=0), None])
    ready = Group("ready", 0, 1, [SimpleNamespace(reward=0), SimpleNamespace(reward=1)])
    pool = pool_for(tmp_path, state=RunState(step=10, groups=[partial, ready]))
    pool._discard_stale()
    assert pool.state.groups == [partial]
    assert partial.trajectories == [None, None] and partial.dispatch_step == 10
    assert pool.state.counters["stale"] == 2


@pytest.mark.asyncio
async def test_producer_failure_reaches_training_and_closes_other_sessions(tmp_path):
    pool = pool_for(tmp_path, rollout=Rollout(fail=True))
    pool.start()
    try:
        with pytest.raises(RuntimeError, match="sampler failed"):
            await asyncio.wait_for(pool.take_batch(), 2)
    finally:
        await pool.close()
    assert pool.rollout.active["test.Env"] == 0


@pytest.mark.asyncio
async def test_an_explicit_refill_limit_reports_an_unusable_batch(tmp_path):
    pool = pool_for(tmp_path, rollout=Rollout(constant=True), limit=4)
    pool.start()
    try:
        with pytest.raises(RuntimeError, match="usable groups"):
            await asyncio.wait_for(pool.take_batch(), 2)
    finally:
        await pool.close()


def test_session_capacity_must_be_positive(tmp_path):
    pool = pool_for(tmp_path)
    with pytest.raises(ValueError, match="no session slots"):
        RolloutPool(pool.config, pool.tasks, pool.rollout, pool.sampler, {"test.Env": 0}, RunState())


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [1, 2])
async def test_eight_members_share_fewer_sessions(tmp_path, slots):
    pool = pool_for(tmp_path, batch_groups=1)
    pool.config.group_size, pool.config.max_sessions = 8, slots
    pool = RolloutPool(pool.config, pool.tasks, pool.rollout, pool.sampler, {"test.Env": slots}, RunState())
    pool.start()
    try:
        [group] = await asyncio.wait_for(pool.take_batch(), 2)
        assert group.complete and len(group.trajectories) == 8
        assert [member for uid, member in pool.rollout.calls if uid == group.uid] == list(range(8))
        assert pool.rollout.maximum["test.Env"] <= slots
        assert pool.state.counters["filtered"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_busy_families_do_not_drop_tasks_from_the_mixture(tmp_path):
    pool = pool_for(tmp_path, batch_groups=3)
    pool.config.group_size, pool.config.max_sessions, pool.config.shuffle = 8, 3, False
    tasks = [EpisodeTask(name, {}, name, i) for i, name in enumerate(["isaac", "isaac", "cpu"])]
    pool = RolloutPool(pool.config, tasks, pool.rollout, pool.sampler, {"isaac": 1, "cpu": 2}, RunState())
    pool.state.next_group = 1  # this fixture's group zero deliberately has constant rewards
    pool.start()
    try:
        groups = await asyncio.wait_for(pool.take_batch(), 2)
        assert sorted(group.task_index for group in groups) == [0, 1, 2]
        assert all(group.complete and len(group.trajectories) == 8 for group in groups)
        assert pool.rollout.maximum["isaac"] == 1 and pool.rollout.maximum["cpu"] <= 2
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [False, True])
async def test_zero_prefetch_waits_for_the_update_and_limits_restored_groups(tmp_path, resumed):
    original = pool_for(tmp_path, batch_groups=1)
    config = replace(original.config, group_size=8, max_sessions=8, prefetch_groups=0)
    state = RunState(next_group=1)
    if resumed:
        state.groups = [Group(name, 0, 0, [None] * 8) for name in ["restored_a", "restored_b"]]
    pool = RolloutPool(config, original.tasks, Rollout(), object(), {"test.Env": 8}, state)
    pool.start()
    try:
        [first] = await asyncio.wait_for(pool.take_batch(), 2)
        await asyncio.sleep(0.01)
        assert len(pool.rollout.calls) == 8
        assert all(uid == first.uid for uid, _ in pool.rollout.calls)
        state.step += 1
        pool.update_completed()
        [second] = await asyncio.wait_for(pool.take_batch(final=True), 2)
        pool.update_completed()
        await asyncio.sleep(0.01)
        assert len(pool.rollout.calls) == 16
        assert first.uid != second.uid and second.dispatch_step == 1
        if resumed:
            assert [first.uid, second.uid] == ["restored_a", "restored_b"]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_update_admits_more_work_while_prefetched_rollouts_are_running(tmp_path):
    started, release = asyncio.Queue(), asyncio.Event()

    async def run(task, sampler, uid, member):
        if uid != "g000000000001":
            await started.put((uid, member, sampler))
            await release.wait()
        return SimpleNamespace(reward=float(member), generated_tokens=1)

    original = pool_for(tmp_path, batch_groups=1)
    config = replace(original.config, prefetch_groups=1)
    pool = RolloutPool(
        config, original.tasks, SimpleNamespace(run=run), object(), {"test.Env": 4}, RunState(next_group=1)
    )
    pool.start()
    try:
        running = [await asyncio.wait_for(started.get(), 2) for _ in range(2)]
        assert {(uid, member) for uid, member, _ in running} == {("g000000000002", 0), ("g000000000002", 1)}
        [batch] = await asyncio.wait_for(pool.take_batch(), 2)
        assert batch.uid == "g000000000001" and pool.held_groups == 1

        pool.sampler = updated_sampler = object()
        pool.state.step += 1
        pool.update_completed()

        admitted = [await asyncio.wait_for(started.get(), 2) for _ in range(2)]
        assert {(uid, member) for uid, member, _ in admitted} == {("g000000000003", 0), ("g000000000003", 1)}
        assert all(sampler is updated_sampler for _, _, sampler in admitted)
        assert pool.state.groups[-1].dispatch_step == 1
        assert pool.state.counters["completed"] == 1 and len(pool.jobs) == 4
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_update", [False, True])
async def test_zero_prefetch_does_not_start_a_group_beyond_the_refill_limit(tmp_path, after_update):
    original = pool_for(tmp_path, batch_groups=1, limit=1)
    config = replace(original.config, group_size=8, max_sessions=8, prefetch_groups=0)
    pool = RolloutPool(
        config, original.tasks, Rollout(constant=not after_update), object(), {"test.Env": 8}, RunState(next_group=1)
    )
    pool.start()
    try:
        if after_update:
            await asyncio.wait_for(pool.take_batch(), 2)
            pool.rollout.constant = True
            pool.state.step += 1
            pool.update_completed()
            # A group can finish while the coordinator saves its checkpoint, before the next take_batch.
            await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="after 1 completed groups"):
            await asyncio.wait_for(pool.take_batch(), 2)
        assert len(pool.rollout.calls) == (16 if after_update else 8)
    finally:
        await pool.close()
