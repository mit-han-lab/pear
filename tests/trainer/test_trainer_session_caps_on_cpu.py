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

# The tests of the trainer under session caps and of what PEAR's trainer adds to verl's: its replay buffer, its start
# of the standalone replicas (the setup helper after the one of verl's
# tests/trainer/ppo/v1/test_separate_async_step_switch_on_cpu.py).
import threading
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.trainer.ppo.v1.trainer_separate_async as upstream_module
from pear.trainer import trainer as trainer_module
from pear.trainer.replay_buffer import PearReplayBuffer
from pear.trainer.trainer import PearTrainer, RestoredGroup
from repository import REPO
from verl.trainer.ppo.v1 import trainer_base
from verl.utils import tensordict_utils as tu

ROBODOJO = "pear.envs.robodojo.env.RoboDojoEnv"  # the tools' names: their env classes
ROBOLAB = "pear.envs.robolab.env.RoboLabEnv"
# a run's tool config: LIBERO's servers set no session cap, RoboDojo's are 4 with 5 Apps each
TOOL_CONFIG = f"""
tools:
  - class_name: pear.envs.tool.EnvironmentTool
    config:
      env_class: pear.envs.libero.env.LiberoEnv
      server_n: 64
  - class_name: pear.envs.tool.EnvironmentTool
    config:
      env_class: {ROBODOJO}
      server_n: 4
      apps_per_gpu: 5
"""
SETUP_WAIT_S = 5  # how long the stub training setup waits for the standalone replicas to have started


@pytest.fixture
def restore(monkeypatch):
    """Runs _reissue_inflight_prompts on a restored partition; returns (prompts re-issued, keys cleared)."""

    def run(items: dict) -> tuple[int, set]:
        tq = MagicMock(kv_list=MagicMock(return_value={"train": items}))
        monkeypatch.setattr(trainer_base, "tq", tq)
        monkeypatch.setattr(trainer_module, "tq", tq)
        monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_data", MagicMock())
        monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_stack", MagicMock())
        reissued = PearTrainer._reissue_inflight_prompts(_trainer())
        return reissued, {key for call in tq.kv_clear.call_args_list for key in call.kwargs["keys"]}

    return run


def _trainer(max_inflight_sessions: int | None = None) -> SimpleNamespace:
    """An async trainer at step 7 with groups of 2 sessions and the parts the re-issue and the refill use."""
    trainer = SimpleNamespace(
        trainer_mode="separate_async",
        global_steps=7,
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}}),
        agent_loop_manager=MagicMock(),
        replay_buffer=SimpleNamespace(
            max_inflight_sessions=max_inflight_sessions, is_stale=lambda global_steps, prompt_global_steps: False
        ),
        _reissue_queue=[],
        INFLIGHT_PROMPT_STATUSES=PearTrainer.INFLIGHT_PROMPT_STATUSES,
        _tool_selections=PearTrainer._tool_selections,
        _take_tool_room=PearTrainer._take_tool_room,
    )
    trainer._reissue_prompt_groups = partial(PearTrainer._reissue_prompt_groups, trainer)
    trainer._stamped_batch = partial(PearTrainer._stamped_batch, trainer)
    trainer._reissue_groups = lambda groups, partition_id="train": PearTrainer._reissue_groups(
        trainer, groups, partition_id
    )
    return trainer


def test_consumed_rows_are_dropped_and_stored_sessions_kept(restore):
    items = {
        "fin": {"is_prompt": True, "status": "finished", "global_steps": 6},
        "fin_0_0": {"status": "success"},
        "run": {"is_prompt": True, "status": "running", "global_steps": 6},
        "run_0_0": {"status": "success"},
        "old_0_0": {},
        "pad_0_0": {"is_padding": True},  # the batch trained before the checkpoint, prompt keys gone
    }
    assert restore(items) == (1, {"old_0_0", "pad_0_0"})


def test_session_cap_queues_restored_groups_instead_of_sending_them(monkeypatch):
    tq = MagicMock(
        kv_list=MagicMock(
            return_value={
                "train": {
                    "pend": {"is_prompt": True, "status": "pending", "global_steps": 6},
                    "run": {"is_prompt": True, "status": "running", "global_steps": 5},
                    "run_0_0": {"status": "success"},
                    "que": {"is_prompt": True, "status": "queued", "global_steps": 4},  # still queued at the save
                    "fin": {"is_prompt": True, "status": "finished", "global_steps": 6},
                    "fin_0_0": {},
                }
            }
        )
    )
    monkeypatch.setattr(trainer_base, "tq", tq)
    monkeypatch.setattr(trainer_module, "tq", tq)
    trainer = _trainer(max_inflight_sessions=16)

    assert PearTrainer._reissue_inflight_prompts(trainer) == 3
    assert trainer.agent_loop_manager.generate_sequences.call_count == 0
    assert trainer._reissue_queue == [
        RestoredGroup(uid="pend", dispatch_step=7, unfinished_session_ids=[0, 1], tool_selection=None),
        RestoredGroup(uid="run", dispatch_step=5, unfinished_session_ids=[1], tool_selection=None),
        RestoredGroup(uid="que", dispatch_step=7, unfinished_session_ids=[0, 1], tool_selection=None),
    ]
    tags = tq.kv_batch_put.call_args.kwargs["tags"]
    assert [(tag["status"], tag["global_steps"]) for tag in tags] == [("queued", 7), ("queued", 5), ("queued", 7)]


def test_a_session_the_checkpoint_caught_mid_write_runs_again(monkeypatch):
    # the group's last session was being written at the save: its key holds no data and no tag, so the group is not
    # complete, the key goes and the session runs again
    tq = MagicMock(
        kv_list=MagicMock(
            return_value={
                "train": {
                    "run": {"is_prompt": True, "status": "running", "global_steps": 6},
                    "run_0_0": {"status": "success"},
                    "run_1_0": {},
                }
            }
        )
    )
    monkeypatch.setattr(trainer_base, "tq", tq)
    monkeypatch.setattr(trainer_module, "tq", tq)
    trainer = _trainer(max_inflight_sessions=16)

    assert PearTrainer._reissue_inflight_prompts(trainer) == 1
    assert trainer._reissue_queue == [
        RestoredGroup(uid="run", dispatch_step=6, unfinished_session_ids=[1], tool_selection=None)
    ]
    assert [call.kwargs["keys"] for call in tq.kv_clear.call_args_list] == [["run_1_0"]]
    assert [tag["status"] for tag in tq.kv_batch_put.call_args.kwargs["tags"]] == ["queued"]


def test_refill_sends_queued_sessions_before_fresh_prompts(monkeypatch):
    tq = MagicMock()
    monkeypatch.setattr(trainer_module, "tq", tq)
    monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_data", MagicMock())
    monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_stack", MagicMock())
    trainer = _trainer(max_inflight_sessions=16)
    trainer._reissue_queue = [
        RestoredGroup(uid="a", dispatch_step=7, unfinished_session_ids=[0, 1], tool_selection=None),
        RestoredGroup(uid="b", dispatch_step=7, unfinished_session_ids=[1], tool_selection=None),
        RestoredGroup(uid="c", dispatch_step=6, unfinished_session_ids=[0, 1], tool_selection=None),
    ]
    trainer._next_train_batch = MagicMock(return_value="fresh batch")
    trainer._submit_batch_to_rollout = MagicMock(side_effect=lambda batch: 2)

    # room for 2 groups = 4 sessions: a and b fit (3), c does not fit the one left
    assert PearTrainer._add_prompts_to_generate(trainer, 2) == 2
    assert tq.kv_batch_get.call_args.kwargs["keys"] == ["a", "b"]
    assert [tag["status"] for tag in tq.kv_batch_put.call_args.kwargs["tags"]] == ["pending", "pending"]
    assert [group.uid for group in trainer._reissue_queue] == ["c"] and trainer._next_train_batch.call_count == 0

    # room for 3 groups = 6 sessions: c takes 2, the 4 left take 2 fresh prompts
    assert PearTrainer._add_prompts_to_generate(trainer, 3) == 3
    assert tq.kv_batch_get.call_args.kwargs["keys"] == ["c"]
    trainer._next_train_batch.assert_called_once_with(2)
    assert trainer._reissue_queue == []
    assert trainer.agent_loop_manager.generate_sequences.call_count == 2


def _row(uid: str, tool_selection: list[str] | None) -> TensorDict:
    """One fetch of an async dataloader: a single prompt whose row selects tool_selection."""
    return tu.get_tensordict(
        {
            "uid": [uid],
            "raw_prompt": [f"prompt-{uid}"],
            "index": torch.tensor([0]),
            "extra_info": [{"index": 0, "tool_selection": tool_selection}],
        }
    )


def test_refill_skips_the_rows_of_a_tool_at_its_cap():
    # room for 3 groups of 2 sessions, 3 sessions of the capped tool: its second row does not fit and is skipped, the
    # rows of no capped tool fill the rest
    trainer = _trainer(max_inflight_sessions=16)
    rows = iter([_row("sim1", ["sim"]), _row("sim2", ["sim"]), _row("plain1", None), _row("other1", ["other"])])
    trainer._fetch_one_gen_batch = lambda: next(rows)
    submitted = []
    trainer._submit_batch_to_rollout = lambda batch: submitted.append(batch) or len(batch)

    assert PearTrainer._add_prompts_to_generate(trainer, 3, free_sessions_by_tool={"sim": 3}) == 3
    assert [list(batch["uid"]) for batch in submitted] == [["sim1", "plain1", "other1"]]
    assert int(submitted[0]["global_steps"]) == 7


def test_refill_skips_at_most_as_many_rows_as_groups_asked_for():
    # every row selects the tool, which has no room: the refill skips 2 rows for its 2 groups and sends none, the next
    # refill goes on
    trainer = _trainer(max_inflight_sessions=16)
    fetched_uids = []

    def fetch_one_gen_batch():
        fetched_uids.append(f"sim{len(fetched_uids)}")
        return _row(fetched_uids[-1], ["sim"])

    trainer._fetch_one_gen_batch = fetch_one_gen_batch
    trainer._submit_batch_to_rollout = MagicMock()

    assert PearTrainer._add_prompts_to_generate(trainer, 2, free_sessions_by_tool={"sim": 0}) == 0
    assert fetched_uids == ["sim0", "sim1"] and trainer._submit_batch_to_rollout.call_count == 0


def test_a_restored_group_whose_tool_has_no_room_keeps_its_place_in_the_queue(monkeypatch):
    tq = MagicMock()
    monkeypatch.setattr(trainer_module, "tq", tq)
    monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_data", MagicMock())
    monkeypatch.setattr(trainer_module.tu, "assign_non_tensor_stack", MagicMock())
    trainer = _trainer(max_inflight_sessions=16)
    trainer._reissue_queue = [
        RestoredGroup(uid="sim", dispatch_step=7, unfinished_session_ids=[0, 1], tool_selection=["sim"]),
        RestoredGroup(uid="plain", dispatch_step=7, unfinished_session_ids=[0, 1], tool_selection=None),
    ]
    trainer._fetch_one_gen_batch = MagicMock()

    # room for 1 group, none of the capped tool: the plain group goes, the capped one waits first in the queue
    assert PearTrainer._add_prompts_to_generate(trainer, 1, free_sessions_by_tool={"sim": 0}) == 1
    assert tq.kv_batch_get.call_args.kwargs["keys"] == ["plain"]
    assert [group.uid for group in trainer._reissue_queue] == ["sim"] and trainer._fetch_one_gen_batch.call_count == 0


def test_restored_groups_keep_the_tool_selection_of_their_tags(monkeypatch):
    tq = MagicMock(
        kv_list=MagicMock(
            return_value={
                "train": {
                    "sim": {"is_prompt": True, "status": "running", "global_steps": 6, "tool_selection": ["sim"]},
                    "plain": {"is_prompt": True, "status": "pending", "global_steps": 6},
                }
            }
        )
    )
    monkeypatch.setattr(trainer_base, "tq", tq)
    monkeypatch.setattr(trainer_module, "tq", tq)
    trainer = _trainer(max_inflight_sessions=16)

    assert PearTrainer._reissue_inflight_prompts(trainer) == 2
    restored = [(group.uid, group.tool_selection) for group in trainer._reissue_queue]
    assert restored == [("sim", ["sim"]), ("plain", None)]


@pytest.fixture
def tool_config_path(tmp_path) -> str:
    """The file of TOOL_CONFIG."""
    (tmp_path / "tool_config.yaml").write_text(TOOL_CONFIG)

    return str(tmp_path / "tool_config.yaml")


def _built_trainer(
    max_inflight_sessions: int | None = None,
    custom_sampler_path: str | None = None,
    tool_config_path: str | None = None,
) -> PearTrainer:
    """PEAR's trainer with the replay buffer of its config, without its constructor."""
    trainer = PearTrainer.__new__(PearTrainer)
    trainer.trainer_mode = "separate_async"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "actor_rollout_ref": {"rollout": {"n": 8, "multi_turn": {"tool_config_path": tool_config_path}}},
            "trainer": {
                "v1": {
                    "separate_async": {},
                    "sampler": {
                        "custom_sampler": {"path": custom_sampler_path, "name": "OtherSampler"},
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                        "max_inflight_sessions": max_inflight_sessions,
                    },
                }
            },
        }
    )
    trainer.replay_buffer = trainer._build_replay_buffer()
    return trainer


def test_the_trainer_builds_pears_replay_buffer_with_the_group_size_and_the_caps(tool_config_path, monkeypatch, caplog):
    # the trainer's logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(trainer_module.logger, "handlers", [caplog.handler])

    replay_buffer = _built_trainer(max_inflight_sessions=384, tool_config_path=tool_config_path).replay_buffer

    assert type(replay_buffer) is PearReplayBuffer
    assert (replay_buffer.rollout_n, replay_buffer.max_inflight_sessions) == (8, 384)
    # within the run's cap, what the servers of each tool's config take at once: RoboDojo's 4 x 5 Apps
    assert replay_buffer.max_inflight_sessions_by_tool == {ROBODOJO: 20}
    assert caplog.messages == [f"Session caps by tool: {{'{ROBODOJO}': 20}}"]


def test_a_run_without_a_session_cap_caps_no_tool_s_sessions(tool_config_path):
    replay_buffer = _built_trainer(tool_config_path=tool_config_path).replay_buffer

    assert (replay_buffer.max_inflight_sessions, replay_buffer.max_inflight_sessions_by_tool) == (None, {})


@pytest.mark.parametrize(
    ("tool_config", "variables", "session_caps"),
    [
        ("pear/envs/robodojo/configs/tool_config.yaml", {"ROBODOJO_SERVER_N": "4"}, {ROBODOJO: 4 * 8}),
        (
            "pear/envs/robodojo/configs/tool_config.yaml",
            {"ROBODOJO_SERVER_N": "4", "ROBODOJO_APPS_PER_GPU": "20"},
            {ROBODOJO: 4 * 20},
        ),
        ("pear/envs/robolab/configs/tool_config.yaml", {"ROBOLAB_SERVER_N": "4"}, {ROBOLAB: 4 * 10}),
        (
            "pear/envs/robolab/configs/tool_config.yaml",
            {"ROBOLAB_SERVER_N": "4", "ROBOLAB_APPS_PER_GPU": "20"},
            {ROBOLAB: 4 * 20},
        ),
        (
            "examples/models/qwen3_8_27b_pear/train/libero_robodojo_tool_config.yaml",
            {"ROBODOJO_SERVER_N": "4"},
            {ROBODOJO: 4 * 8},
        ),
        ("pear/envs/libero/configs/tool_config.yaml", {}, {}),
    ],
    ids=["robodojo", "robodojo-20-apps", "robolab", "robolab-20-apps", "libero-robodojo", "libero"],
)
def test_a_shipped_tool_config_caps_its_family_s_sessions_at_its_servers_apps(
    monkeypatch, tmp_path, tool_config, variables, session_caps
):
    # the variables a recipe gives its workers: the Apps of a server only when the recipe was given them
    for variable in ("ROBODOJO_APPS_PER_GPU", "ROBOLAB_APPS_PER_GPU"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in {"PEAR_LOG_DIR": str(tmp_path), **variables}.items():
        monkeypatch.setenv(variable, value)

    trainer = _built_trainer(max_inflight_sessions=384, tool_config_path=str(REPO / tool_config))

    assert trainer.replay_buffer.max_inflight_sessions_by_tool == session_caps


def test_a_custom_sampler_is_refused_instead_of_ignored(tmp_path):
    # verl loads the class the config names before the trainer is asked which one it builds
    sampler_path = tmp_path / "other_sampler.py"
    sampler_path.write_text("class OtherSampler:\n    pass\n")

    with pytest.raises(ValueError, match="custom_sampler is not supported"):
        _built_trainer(custom_sampler_path=str(sampler_path))


def test_the_replay_buffer_takes_the_arguments_verl_s_trainer_builds_its_own_with(tool_config_path):
    # an argument PEAR's trainer never names: it reaches the buffer because the arguments are verl's, extended
    trainer = _built_trainer(max_inflight_sessions=384, tool_config_path=tool_config_path)
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    assert trainer._build_replay_buffer().sync_refill_failed_groups is True


@pytest.mark.parametrize("max_inflight_sessions, batches_submitted", [(None, 1), (384, 0)])
def test_prepare_step_submits_a_batch_only_without_a_session_cap(
    monkeypatch, tool_config_path, max_inflight_sessions, batches_submitted
):
    trainer = _built_trainer(max_inflight_sessions, tool_config_path=tool_config_path)
    trainer.current_mode = upstream_module.HybridEngineMode.TRAINER
    submitted = []
    monkeypatch.setattr(upstream_module.PPOTrainer, "_add_batch_to_generate", lambda self: submitted.append(self))

    assert trainer.prepare_step() == {}
    assert len(submitted) == batches_submitted


@pytest.mark.parametrize(
    "max_inflight_sessions, outcome",
    [(None, nullcontext()), (384, pytest.raises(ValueError, match="__rollout_n__"))],
)
def test_only_the_session_cap_rejects_a_per_prompt_group_size(tool_config_path, max_inflight_sessions, outcome):
    trainer = _built_trainer(max_inflight_sessions, tool_config_path=tool_config_path)
    trainer.agent_loop_manager = MagicMock()
    trainer.global_steps = 1
    batch = tu.get_tensordict({"uid": np.array(["a", "b"], dtype=object), "__rollout_n__": np.array([4, 16])})

    with patch("verl.trainer.ppo.v1.trainer_base.tq"), outcome:
        trainer._submit_batch_to_rollout(batch)


def _setup_trainer(monkeypatch, hybrid_enabled: bool) -> tuple[PearTrainer, list]:
    """PEAR's trainer before its setup, with verl's training setup, its standalone servers and its checkpoint manager
    replaced: returns it with the (config, start rank) of every standalone start."""
    trainer = PearTrainer.__new__(PearTrainer)
    trainer._enable_hybrid_replicas = hybrid_enabled
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "gpu_memory_utilization": 0.45,
                    "name": "vllm",
                    "prometheus": {"enable": False},
                    "checkpoint_engine": {
                        "_target_": "verl.workers.config.CheckpointEngineConfig",
                        "backend": "nccl",
                    },
                }
            },
        }
    )
    created_configs = []

    def base_setup(self):
        self.actor_rollout_wg = object()
        self.llm_server_manager = SimpleNamespace(rollout_replicas=[0, 1] if hybrid_enabled else [])

    def create_standalone(*, config, start_rank):
        created_configs.append((config, start_rank))
        return SimpleNamespace(get_replicas=lambda: ["standalone"])

    monkeypatch.setattr(upstream_module.PPOTrainer, "_setup", base_setup)
    monkeypatch.setattr(upstream_module.LLMServerManager, "create", create_standalone)
    monkeypatch.setattr(upstream_module, "CheckpointEngineManager", lambda **kwargs: object())
    trainer.add_replicas_to_balancer = lambda: None
    return trainer, created_configs


def test_setup_starts_the_standalone_replicas_while_the_training_engines_initialize(monkeypatch):
    trainer, created_configs = _setup_trainer(monkeypatch, False)
    started = threading.Event()
    base_setup_saw_start = []

    def create_standalone(*, config, start_rank):
        started.set()
        created_configs.append((config, start_rank))
        return SimpleNamespace(get_replicas=lambda: ["standalone"])

    def base_setup(self):
        base_setup_saw_start.append(started.wait(timeout=SETUP_WAIT_S))
        self.actor_rollout_wg = object()
        self.llm_server_manager = SimpleNamespace(rollout_replicas=[])

    monkeypatch.setattr(upstream_module.LLMServerManager, "create", create_standalone)
    monkeypatch.setattr(upstream_module.PPOTrainer, "_setup", base_setup)

    trainer._setup()

    assert base_setup_saw_start == [True]
    assert [start_rank for _, start_rank in created_configs] == [0]
    assert trainer.standalone_server_manager.get_replicas() == ["standalone"]


def test_setup_next_to_hybrid_replicas_starts_the_standalone_ones_after_them_with_the_ranks_that_follow(monkeypatch):
    trainer, created_configs = _setup_trainer(monkeypatch, True)

    trainer._setup()

    assert [start_rank for _, start_rank in created_configs] == [2]  # the base setup's two hybrid replicas
