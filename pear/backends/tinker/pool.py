from __future__ import annotations

import asyncio
from collections import Counter

from .state import Group


class RolloutPool:
    """Keep bounded simulator sessions running while training consumes complete, nonconstant groups."""

    def __init__(self, config, tasks, rollout, sampler, caps, state) -> None:
        self.config, self.tasks, self.rollout, self.sampler = config, tasks, rollout, sampler
        self.caps, self.state = caps, state
        self.jobs = {}
        self.changed, self.wake = asyncio.Event(), asyncio.Event()
        self.task = None
        extra = config.prefetch_groups
        self.limit = config.batch_groups + (
            max(1, config.max_sessions // config.group_size) if extra is None else extra
        )
        self.held_groups, self.finished = 0, False
        self.batch_start = state.counters["completed"]
        for name in {task.env_class for task in tasks}:
            if caps[name] < 1:
                raise ValueError(f"{name} has no session slots")

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    def _discard_stale(self) -> None:
        active = {uid for uid, _ in self.jobs.values()}
        for group in list(self.state.groups):
            if not any(t is not None for t in group.trajectories):
                continue
            if group.uid not in active and self.state.step - group.dispatch_step > self.config.max_staleness:
                if group.complete:
                    self.state.groups.remove(group)
                else:
                    # A resumed partial group keeps its task but cannot mix fresh samples with stale members.
                    group.trajectories = [None] * self.config.group_size
                    group.dispatch_step = self.state.step
                self.state.counters["stale"] += 1

    def _fill(self) -> None:
        if self.finished:
            return
        if (
            self.config.prefetch_groups == 0
            and self.config.max_groups_per_step
            and self.state.counters["completed"] - self.batch_start >= self.config.max_groups_per_step
        ):
            return
        self._discard_stale()
        capacity = self.limit - self.held_groups
        reserved = set(self.jobs.values())
        by_uid = {group.uid: group for group in self.state.groups}
        usage = Counter(self.tasks[by_uid[uid].task_index].env_class for uid, _ in self.jobs.values())

        def schedule(group):
            task = self.tasks[group.task_index]
            if not any(t is not None for t in group.trajectories) and not any(uid == group.uid for uid, _ in reserved):
                group.dispatch_step = self.state.step
            for member, trajectory in enumerate(group.trajectories):
                if trajectory is not None or (group.uid, member) in reserved:
                    continue
                if len(self.jobs) >= self.config.max_sessions or usage[task.env_class] >= self.caps[task.env_class]:
                    return
                job = asyncio.create_task(self.rollout.run(task, self.sampler, group.uid, member))
                self.jobs[job] = group.uid, member
                reserved.add((group.uid, member))
                usage[task.env_class] += 1

        # A resumed checkpoint can contain more queued groups than the new prefetch limit. Keep their
        # tasks, but acquire sampling slots only for the admitted prefix.
        for group in self.state.groups[:capacity]:
            if not group.complete:
                schedule(group)
        # Queue tasks in dataset order even when a family is busy. Each group acquires individual slots;
        # statistical group size is independent of the number of simultaneously running simulator Apps.
        while len(self.jobs) < self.config.max_sessions and len(self.state.groups) < capacity:
            index = self.state.next_task(len(self.tasks), self.config.seed, self.config.shuffle)
            group = Group(f"g{self.state.next_group:012d}", index, self.state.step, [None] * self.config.group_size)
            self.state.next_group += 1
            self.state.groups.append(group)
            schedule(group)

    async def _run(self) -> None:
        try:
            while True:
                self.wake.clear()
                self._fill()
                waiter = asyncio.create_task(self.wake.wait())
                try:
                    done, _ = await asyncio.wait([*self.jobs, waiter], return_when=asyncio.FIRST_COMPLETED)
                finally:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                done.discard(waiter)
                by_uid = {group.uid: group for group in self.state.groups}
                for job in done:
                    uid, member = self.jobs.pop(job)
                    group = by_uid[uid]
                    group.trajectories[member] = job.result()
                    if group.complete:
                        self.state.counters["completed"] += 1
                        self.state.counters["episodes"] += len(group.trajectories)
                        self.state.counters["successes"] += sum(int(t.reward) for t in group.trajectories)
                        if len({t.reward for t in group.trajectories}) == 1:
                            self.state.groups.remove(group)
                            self.state.counters["filtered"] += 1
                if done:
                    self.changed.set()
        finally:
            for job in self.jobs:
                job.cancel()
            await asyncio.gather(*self.jobs, return_exceptions=True)
            self.jobs.clear()
            self.changed.set()

    async def take_batch(self, *, final=False):
        first_count = self.batch_start if self.config.prefetch_groups == 0 else self.state.counters["completed"]
        while True:
            if self.task.done():
                await self.task
                raise RuntimeError("rollout producer stopped")
            self._discard_stale()
            groups = [group for group in self.state.groups if group.complete]
            if len(groups) >= self.config.batch_groups:
                selected = groups[: self.config.batch_groups]
                self.state.groups = [group for group in self.state.groups if group not in selected]
                self.held_groups = len(selected) if self.config.prefetch_groups is not None else 0
                self.finished = final
                self.wake.set()
                return selected
            limit = self.config.max_groups_per_step
            if limit and self.state.counters["completed"] - first_count >= limit:
                raise RuntimeError(f"fewer than batch_groups usable groups after {limit} completed groups")
            self.changed.clear()
            waiter = asyncio.create_task(self.changed.wait())
            try:
                await asyncio.wait([waiter, self.task], return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    def update_completed(self):
        self.held_groups = 0
        self.batch_start = self.state.counters["completed"]
        self.wake.set()

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
