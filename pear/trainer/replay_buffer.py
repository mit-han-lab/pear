from collections import Counter, defaultdict

import numpy as np
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


class PearReplayBuffer(ReplayBufferAsync):
    """verl's asynchronous replay buffer under caps on the sessions in flight, overall and by tool, which also reports
    the pass rates of the groups it judges."""

    # the status of a restored prompt group that waits for room under the session cap, next to verl's own statuses
    QUEUED_STATUS = "queued"

    def __init__(
        self,
        *,
        rollout_n: int,
        max_inflight_sessions: int | None,
        max_inflight_sessions_by_tool: dict[str, int] | None,
        **replay_buffer_kwargs,
    ) -> None:
        # set before the parent's constructor, which validates them (_validate_mode_config)
        self.rollout_n = rollout_n
        self.max_inflight_sessions = max_inflight_sessions
        self.max_inflight_sessions_by_tool = dict(max_inflight_sessions_by_tool or {})
        super().__init__(**replay_buffer_kwargs)

        # partition_id => {prompt_key: the tools its row selects}, which the session cap per tool counts it under
        self.prompt_tool_selections: dict[str, dict[str, list[str] | None]] = defaultdict(dict)
        # pass rates of the groups judged since the last sample(), each with the tools its row selects
        self._group_pass_rates: dict[str, list[tuple[float, list[str] | None]]] = defaultdict(list)

    def _validate_mode_config(self) -> None:
        if self.max_inflight_sessions_by_tool and self.max_inflight_sessions is None:
            raise ValueError("max_inflight_sessions_by_tool caps tools within max_inflight_sessions, which it requires")

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        metrics.update(self._pass_rate_metrics(partition_id))

        return batch, metrics

    def _sync_metadata_from_transfer_queue(self) -> None:
        self.prompt_tool_selections.clear()
        super()._sync_metadata_from_transfer_queue()

    def _track_prompt(self, partition_id: str, key: str, tag: dict) -> None:
        self.prompt_tool_selections[partition_id][key] = tag.get("tool_selection")
        # not generating, so outside the in-flight count: the refill sends it under the cap
        if tag["status"] != self.QUEUED_STATUS:
            super()._track_prompt(partition_id, key, tag)

    def _on_group_judged(self, partition_id: str, uid: str, values: list[float]) -> None:
        tool_selection = self.prompt_tool_selections[partition_id].get(uid)
        self._group_pass_rates[partition_id].append((float(np.mean(values)), tool_selection))

    def _refill_evicted(self, evicted_uids: set[str]) -> None:
        # with a session cap, finished and evicted groups alike free room that _refill_free_room refills up to the cap
        if self.max_inflight_sessions is None:
            super()._refill_evicted(evicted_uids)

    def _refill_free_room(self, partition_id: str) -> bool:
        if self.max_inflight_sessions is None or partition_id == "val":
            return False

        free_groups = (self.max_inflight_sessions - self._inflight_sessions(partition_id)) // self.rollout_n
        refill_kwargs = {}
        if self.max_inflight_sessions_by_tool:  # each capped tool's room, which the refill fits its groups in
            refill_kwargs["free_sessions_by_tool"] = {
                tool: cap - self._inflight_sessions(partition_id, tool)
                for tool, cap in self.max_inflight_sessions_by_tool.items()
            }

        # a refill with no prompts left (a one-pass dataloader) submits nothing: sample and wait as uncapped
        return free_groups > 0 and self.refill_fn(free_groups, **refill_kwargs) > 0

    def _inflight_sessions(self, partition_id: str, tool: str | None = None) -> int:
        """Sessions still generating: those of a pending or running group without a stored trajectory (with ``tool``,
        of the groups whose rows select it)."""
        stored_session_n = Counter(
            uid for uid, _ in {tuple(key.split("_")[:2]) for key in self.partitions[partition_id]}
        )
        # a restored group is pending with the sessions it stored before the checkpoint
        inflight_uids = self.pending_keys[partition_id] | self.running_keys[partition_id]
        if tool is not None:
            tool_selections = self.prompt_tool_selections[partition_id]
            inflight_uids = {uid for uid in inflight_uids if tool in (tool_selections.get(uid) or ())}

        return sum(self.rollout_n - stored_session_n[uid] for uid in inflight_uids)

    def _pass_rate_metrics(self, partition_id: str) -> dict:
        """The DAPO metric's per-group mean over the groups judged since the last sample, for a 0/1 reward: the
        policy's pass rate on this step's prompts (avg@n), also over each tool's groups (avg@n/<tool>), and the share of
        groups it never or always solved (what the filter evicts)."""
        group_pass_rates = self._group_pass_rates.pop(partition_id, [])
        if not group_pass_rates:
            return {}

        rates = np.array([rate for rate, _ in group_pass_rates])
        prefix = self._metrics_prefix(partition_id)
        metrics = {
            f"{prefix}/filter_groups/avg@n": float(rates.mean()),
            f"{prefix}/filter_groups/passrate/zero": float((rates <= 0.0).mean()),
            f"{prefix}/filter_groups/passrate/one": float((rates >= 1.0).mean()),
        }

        for tool in sorted({tool for _, tool_selection in group_pass_rates for tool in tool_selection or ()}):
            tool_rates = [rate for rate, tool_selection in group_pass_rates if tool in (tool_selection or ())]
            metrics[f"{prefix}/filter_groups/avg@n/{tool}"] = float(np.mean(tool_rates))

        return metrics
