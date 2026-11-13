import pkgutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict

from pear.envs.tool import EnvironmentTool
from pear.trainer.replay_buffer import PearReplayBuffer
from pear.utils.logging_utils import get_logger
from verl.trainer.ppo.v1 import get_trainer_cls
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerManager

logger = get_logger(__name__, default_level="INFO")  # what a resume restored belongs to a run's log


@dataclass
class RestoredGroup:
    """A prompt group in flight at the checkpoint: its uid, its dispatch step and the sessions it had not finished."""

    uid: str
    dispatch_step: int
    unfinished_session_ids: list[int]
    tool_selection: list[str] | None  # the tools its row selects, under whose session caps it runs


class SessionCapMixin:
    """What a verl trainer does differently under PEAR's replay buffer: it builds the buffer and tags each prompt with
    the tools its row selects."""

    def _replay_buffer_class(self, sampler_cls: type) -> type[PearReplayBuffer]:
        # a class PearReplayBuffer does not extend is a custom sampler, which verl's trainer would build instead of
        # its own replay buffer and this one would silently ignore
        if not issubclass(PearReplayBuffer, sampler_cls):
            raise ValueError(
                "trainer.v1.sampler.custom_sampler is not supported: PEAR's trainers build PearReplayBuffer"
            )

        return PearReplayBuffer

    def _replay_buffer_kwargs(self, replay_buffer_kwargs: dict[str, Any]) -> dict[str, Any]:
        max_inflight_sessions = self.config.trainer.v1.sampler.max_inflight_sessions

        return dict(
            replay_buffer_kwargs,
            rollout_n=self.config.actor_rollout_ref.rollout.n,
            max_inflight_sessions=max_inflight_sessions,
            # within the run's cap, what each family's servers take at once
            max_inflight_sessions_by_tool=None if max_inflight_sessions is None else self._session_caps_by_tool(),
        )

    def _session_caps_by_tool(self) -> dict[str, int]:
        """The session cap of every tool of the run whose family sets one (BaseEnv.session_cap), by the tool's name:
        its env class."""
        tool_configs = EnvironmentTool.load_configs(self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
        session_caps = {
            config["env_class"]: pkgutil.resolve_name(config["env_class"]).session_cap(config)
            for config in tool_configs
        }
        session_caps = {tool: session_cap for tool, session_cap in session_caps.items() if session_cap is not None}
        logger.info("Session caps by tool: %s", session_caps)

        return session_caps

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        if self.replay_buffer.max_inflight_sessions is not None and "__rollout_n__" in batch:
            raise ValueError(
                "max_inflight_sessions counts every group as rollout.n sessions: no per-prompt __rollout_n__"
            )

        return super()._submit_batch_to_rollout(batch)

    def _prompt_tags(self, batch: TensorDict) -> list[dict]:
        tags = super()._prompt_tags(batch)
        # a session cap per tool counts the group under the tools its row selects
        for tag, tool_selection in zip(tags, self._tool_selections(batch), strict=True):
            tag["tool_selection"] = tool_selection

        return tags

    def _stamped_batch(self, rows: list[TensorDict]) -> TensorDict:
        """The rows as one batch that carries the trainer's step."""
        batch = rows[0] if len(rows) == 1 else tu.concat_tensordict(rows)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)

        return batch

    @staticmethod
    def _tool_selections(batch: TensorDict) -> list[list[str] | None]:
        """Each row's ``extra_info.tool_selection`` as a list (None without one): the tools whose session caps count
        its group."""
        extra_infos = tu.get(batch, "extra_info", default=[None] * len(batch))
        if not isinstance(extra_infos, list):  # one value stored for every row of the batch
            extra_infos = [extra_infos] * len(batch)
        tool_selections = [(extra_info or {}).get("tool_selection") for extra_info in extra_infos]

        return [None if tool_selection is None else list(tool_selection) for tool_selection in tool_selections]

    @staticmethod
    def _take_tool_room(
        free_sessions_by_tool: dict[str, int], tool_selection: list[str] | None, session_n: int
    ) -> bool:
        """Whether session_n sessions of a group whose row selects tool_selection fit the room left of every capped
        tool it selects, which they then take."""
        capped_tools = [tool for tool in tool_selection or () if tool in free_sessions_by_tool]
        if any(free_sessions_by_tool[tool] < session_n for tool in capped_tools):
            return False

        for tool in capped_tools:
            free_sessions_by_tool[tool] -= session_n

        return True


class PearTrainer(SessionCapMixin, PPOTrainerSeparateAsync):
    """verl's separate-async trainer under session caps: the replay buffer refills the rollout by the room the caps
    leave, and a resumed run finishes the prompt groups that were in flight at its checkpoint."""

    # verl's in-flight statuses and the one of a restored group still waiting for room under the session cap
    INFLIGHT_PROMPT_STATUSES = (*PPOTrainerSeparateAsync.INFLIGHT_PROMPT_STATUSES, PearReplayBuffer.QUEUED_STATUS)

    def __init__(self, config: DictConfig) -> None:
        # trainer.v1.trainer_mode names the verl trainer whose config a run reads: the one this trainer extends
        mode_trainer = get_trainer_cls(config.trainer.v1.trainer_mode)
        if not isinstance(self, mode_trainer):
            raise ValueError(
                f"trainer.v1.trainer_mode={config.trainer.v1.trainer_mode} selects {mode_trainer.__name__}, "
                f"which {type(self).__name__} does not extend"
            )

        super().__init__(config)
        # restored in-flight prompt groups the refill sends first, under the session cap
        self._reissue_queue: list[RestoredGroup] = []

    def _setup(self) -> None:
        # next to hybrid replicas the standalone ranks follow theirs, which the training worker group sets
        if self._enable_hybrid_replicas:
            super()._setup()
            return

        # without them the standalone servers start while the training engines initialize (on nodes of another role)
        with ThreadPoolExecutor(max_workers=1) as executor:
            self._standalone_server_manager_future = executor.submit(
                LLMServerManager.create, config=self._standalone_config(), start_rank=0
            )
            super()._setup()

    def _create_standalone_server_manager(self) -> LLMServerManager:
        if self._enable_hybrid_replicas:
            return super()._create_standalone_server_manager()

        return self._standalone_server_manager_future.result()

    def _reissue_prompt_groups(self, inflight_uids: list[str], items: dict[str, dict], partition_id: str) -> int:
        """Run the sessions that checkpointed in-flight prompt groups had not stored, under the session cap when one is
        set."""
        # a group keeps the sessions it stored and its dispatch step, so the staleness of the sessions still to run is
        # counted from the policy that generated the stored ones; a group already too stale to be trained starts over
        inflight_uid_set = set(inflight_uids)
        stored_keys_by_uid, unwritten_keys = defaultdict(list), []

        for key, tag in items.items():
            uid = key.split("_", 1)[0]
            if not tag.get("is_prompt", False) and uid in inflight_uid_set:
                # a put registers its keys, writes their fields, then sets their tags: a key without its tag was being
                # written when the checkpoint was taken and holds no data, so its session runs again
                if "status" in tag:
                    stored_keys_by_uid[uid].append(key)
                else:
                    unwritten_keys.append(key)
        if unwritten_keys:
            tq.kv_clear(keys=unwritten_keys, partition_id=partition_id)
            logger.info("Dropped %d trajectories the checkpoint caught before their data landed", len(unwritten_keys))

        stale_uids = {
            uid
            for uid in stored_keys_by_uid
            if self.replay_buffer.is_stale(self.global_steps, items[uid]["global_steps"])
        }
        stale_keys = [key for uid in stale_uids for key in stored_keys_by_uid[uid]]
        if stale_keys:
            tq.kv_clear(keys=stale_keys, partition_id=partition_id)

        rollout_n = self.config.actor_rollout_ref.rollout.n
        restored_groups, complete_uids = [], []
        for uid in inflight_uids:
            stored_session_ids = (
                set() if uid in stale_uids else {int(key.split("_")[1]) for key in stored_keys_by_uid[uid]}
            )
            unfinished_session_ids = [
                session_id for session_id in range(rollout_n) if session_id not in stored_session_ids
            ]
            if not unfinished_session_ids:
                complete_uids.append(uid)
            else:
                dispatch_step = items[uid]["global_steps"] if stored_session_ids else self.global_steps
                restored_groups.append(
                    RestoredGroup(
                        uid=uid,
                        dispatch_step=dispatch_step,
                        unfinished_session_ids=unfinished_session_ids,
                        tool_selection=items[uid].get("tool_selection"),
                    )
                )

        # every session was stored before the group's terminal status was written
        if complete_uids:
            tags = [
                {"is_prompt": True, "status": "finished", "global_steps": items[uid]["global_steps"]}
                for uid in complete_uids
            ]
            tq.kv_batch_put(keys=complete_uids, partition_id=partition_id, tags=tags)

        if self.replay_buffer.max_inflight_sessions is None:
            self._reissue_groups(restored_groups, partition_id)
        elif restored_groups:
            # sent at once, the restored groups' sessions would exceed the cap: the refill sends them first, as room
            # frees up
            tags = [
                {"is_prompt": True, "status": PearReplayBuffer.QUEUED_STATUS, "global_steps": group.dispatch_step}
                for group in restored_groups
            ]
            tq.kv_batch_put(keys=[group.uid for group in restored_groups], partition_id=partition_id, tags=tags)
            self._reissue_queue = restored_groups

        unfinished_session_n = sum(len(group.unfinished_session_ids) for group in restored_groups)
        kept_session_n = len(inflight_uids) * rollout_n - unfinished_session_n
        logger.info(
            "%s %d in-flight prompts with %d sessions to run; kept %d stored sessions, %d groups complete, "
            "%d stale groups restarted",
            "Re-issued" if self.replay_buffer.max_inflight_sessions is None else "Queued",
            len(restored_groups),
            unfinished_session_n,
            kept_session_n,
            len(complete_uids),
            len(stale_uids),
        )

        return len(restored_groups)

    def _add_batch_to_generate(self) -> None:
        # with a session cap the replay buffer keeps the rollout fed while it samples
        if self.replay_buffer.max_inflight_sessions is None:
            super()._add_batch_to_generate()

    def _add_prompts_to_generate(self, num_prompts: int, free_sessions_by_tool: dict[str, int] | None = None) -> int:
        """Fill the room of num_prompts groups' sessions, the queued restored groups' sessions before fresh prompts; a
        group whose row selects a tool without room for its sessions (free_sessions_by_tool, by tool) stays out."""
        rollout_n = self.config.actor_rollout_ref.rollout.n
        free_session_n = num_prompts * rollout_n
        free_sessions_by_tool = dict(free_sessions_by_tool or {})  # what is left of each capped tool's room

        # the restored groups in their order, up to the first that does not fit the room; one whose tool has no room for
        # it keeps its place in the queue
        reissue_groups, queued_groups = [], []
        for index, group in enumerate(self._reissue_queue):
            session_n = len(group.unfinished_session_ids)
            if session_n > free_session_n:
                queued_groups += self._reissue_queue[index:]
                break
            if self._take_tool_room(free_sessions_by_tool, group.tool_selection, session_n):
                reissue_groups.append(group)
                free_session_n -= session_n
            else:
                queued_groups.append(group)

        self._reissue_queue = queued_groups
        if reissue_groups:
            self._reissue_groups(reissue_groups)
        if free_session_n < rollout_n:
            return len(reissue_groups)

        if not free_sessions_by_tool:
            batch = self._next_train_batch(free_session_n // rollout_n)
            return len(reissue_groups) + self._submit_batch_to_rollout(batch)

        # fresh prompts one at a time (an async dataloader fetches single prompts): one whose tool has no room is
        # skipped, to come again in a later pass of the dataloader; at most num_prompts are skipped, the next refill
        # goes on
        rows, skipped_n = [], 0
        while free_session_n >= rollout_n and skipped_n < num_prompts:
            row = self._fetch_one_gen_batch()
            if self._take_tool_room(free_sessions_by_tool, self._tool_selections(row)[0], rollout_n):
                rows.append(row)
                free_session_n -= rollout_n
            else:
                skipped_n += 1

        if not rows:
            return len(reissue_groups)

        return len(reissue_groups) + self._submit_batch_to_rollout(self._stamped_batch(rows))

    def _reissue_groups(self, restored_groups: list[RestoredGroup], partition_id: str = "train") -> None:
        """Dispatch the sessions restored prompt groups have yet to run, one batch per dispatch step."""
        for dispatch_step in sorted({group.dispatch_step for group in restored_groups}):
            step_groups = [group for group in restored_groups if group.dispatch_step == dispatch_step]
            uids = [group.uid for group in step_groups]
            batch = tq.kv_batch_get(keys=uids, partition_id=partition_id)
            # the batch's step tags the trajectories it produces: the new sessions join the stored ones under it
            tu.assign_non_tensor_data(batch, "global_steps", dispatch_step)
            tu.assign_non_tensor_stack(
                batch, "__unfinished_session_ids__", [group.unfinished_session_ids for group in step_groups]
            )

            tags = [{"is_prompt": True, "status": "pending", "global_steps": dispatch_step} for _ in uids]
            tq.kv_batch_put(keys=uids, partition_id=partition_id, tags=tags)
            self.agent_loop_manager.generate_sequences(batch)
