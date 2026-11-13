import json
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from pear.eval import embodied
from pear.eval.embodied import EmbodiedEvaluation, EmbodiedEvaluationTrainer
from pear.trainer.replay_buffer import PearReplayBuffer
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

STEP = 242
PROMPT = [7, 8]
TRAIN_PARTITION = "train"  # the transfer queue's partition an evaluation samples
POLL_INTERVAL_S = 0.05  # of a replay buffer in these tests: it sees what the queue holds at once


class RecordingTrainer(EmbodiedEvaluationTrainer):
    """The evaluation trainer over a few episode rows that select tool_selection, with the rollout replaced by a list of
    the submitted batches."""

    def __init__(self, rows: int, tool_selection: list[str] | None = None) -> None:
        self.global_steps = STEP
        self.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
        # rows as verl's collate_fn makes them: tensors, and numpy object arrays for the rest
        self.train_dataloader_it = iter(
            [
                {
                    "raw_prompt": np.array([[]], dtype=object),
                    "dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8),
                    "index": np.array([i]),
                    "extra_info": np.array([{"index": i, "tool_selection": tool_selection}], dtype=object),
                }
                for i in range(rows)
            ]
        )
        self.next_episode = None
        self.submitted = []

    def _submit_batch_to_rollout(self, batch) -> int:
        self.submitted.append(batch)
        return len(batch)


class SetupTrainer(EmbodiedEvaluationTrainer):
    """The evaluation trainer's setup over a training config, without a tokenizer or a dataloader."""

    def __init__(self, config) -> None:
        self.config = config

    def _init_tokenizer(self) -> None:
        return

    def _init_dataloader(self) -> None:
        return


class SpellingTrainer(EmbodiedEvaluationTrainer):
    """The evaluation trainer with a tokenizer that spells the token ids out."""

    def __init__(self) -> None:
        self.global_steps = STEP
        self.tokenizer = types.SimpleNamespace(
            pad_token_id=0,
            batch_decode=lambda rows, skip_special_tokens: [
                " ".join(str(t) for t in row if t) for row in rows.tolist()
            ],
        )


class EndedEpisodes:
    """Episodes that have ended in the transfer queue's train partition as the agent loop leaves them: a trajectory
    each, then its prompt's status."""

    def __init__(self, episode_n: int) -> None:
        self.records = {}  # by uid: the record an evaluation writes of the episode, as JSON gives it back

        for index in range(episode_n):
            uid = uuid.uuid4().hex
            response = list(range(1, 4 + index))
            rm_scores = torch.zeros(len(response))
            rm_scores[-1] = float(index)
            field = {
                "prompts": torch.tensor(PROMPT),
                "responses": torch.tensor(response),
                "rm_scores": rm_scores,
                "num_turns": 3 + index,
                "uid": uid,
                "data_source": f"suite_{index}",
                "extra_fields": {"episode": {"task": index}},
            }
            tq.kv_batch_put(
                keys=[f"{uid}_0_0"],
                partition_id=TRAIN_PARTITION,
                fields=list_of_dict_to_tensordict([field]),
                tags=[{"status": "success"}],
            )
            tq.kv_put(
                key=uid,
                partition_id=TRAIN_PARTITION,
                tag={"is_prompt": True, "status": "finished", "global_steps": STEP},
            )
            self.records[uid] = {
                "step": STEP,
                "uid": uid,
                "data_source": f"suite_{index}",
                "score": float(index),
                "num_turns": 3 + index,
                "extra_fields": {"episode": {"task": index}},
                "input": " ".join(str(token) for token in PROMPT),
                "output": " ".join(str(token) for token in response),
            }

    def clear(self, uids: set[str]) -> None:
        """Takes the episodes of these uids out of the queue: those an evaluation that ended early left there."""
        keys = [key for uid in uids for key in (uid, f"{uid}_0_0")]
        tq.kv_clear(keys=keys, partition_id=TRAIN_PARTITION)


class EvaluationEnded(Exception):
    """An evaluation's end before its last episode, as when its job is cancelled."""


class WatchedReplayBuffer(PearReplayBuffer):
    """PEAR's replay buffer over the transfer queue, which notes at every sample what the evaluation's two files hold
    by then, and can end the evaluation at a sample."""

    def __init__(self, output_dir: Path, end_at_sample: int | None = None) -> None:
        super().__init__(
            rollout_n=1,
            max_inflight_sessions=None,
            max_inflight_sessions_by_tool=None,
            trainer_mode="colocate_async",
            trainer_config={},
            max_off_policy_threshold=5,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=POLL_INTERVAL_S,
        )
        self.records_path = output_dir / f"eval_{STEP}.jsonl"
        self.summary_path = output_dir / f"eval_{STEP}_summary.json"
        self.end_at_sample = end_at_sample  # the sample (from 0) at which the evaluation ends; None: at none
        self.file_states = []  # at each sample: the records in the file, and whether the summary exists

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        if len(self.file_states) == self.end_at_sample:
            raise EvaluationEnded

        self.file_states.append((len(self.records_path.read_text().splitlines()), self.summary_path.exists()))

        return super().sample(global_steps, partition_id, batch_size)


class QueueEvaluation(SpellingTrainer):
    """The evaluation trainer's loop over the episodes of the transfer queue, through a replay buffer of the test."""

    def __init__(self, output_dir: Path, episode_n: int, replay_buffer: PearReplayBuffer) -> None:
        super().__init__()
        self.config = OmegaConf.create({"trainer": {"validation_data_dir": str(output_dir)}})
        self.train_dataset = range(episode_n)
        self.train_dataloader = []
        self.replay_buffer = replay_buffer


@pytest.mark.parametrize("response_lengths", [[3, 5], [4, 4]])
def test_records_read_the_trajectories_as_the_agent_loop_stores_them(tq_init, response_lengths):
    # one put per group of rollout.n = 1, as AgentLoopWorkerTQ: rows of different lengths come back jagged, equal ones
    # stacked, and the ints, strings and dicts as non-tensor columns
    partition_id, keys = f"test-{uuid.uuid4().hex}", []
    for i, length in enumerate(response_lengths):
        uid = uuid.uuid4().hex
        rm_scores = torch.zeros(length)
        rm_scores[-1] = float(i)
        field = {
            "prompts": torch.tensor(PROMPT),
            "responses": torch.arange(1, length + 1),
            "rm_scores": rm_scores,
            "num_turns": 3 + i,
            "uid": uid,
            "data_source": f"suite_{i}",
            "extra_fields": {"episode": {"task": i}},
        }
        keys.append(f"{uid}_0_0")
        tq.kv_batch_put(
            keys=keys[-1:],
            partition_id=partition_id,
            fields=list_of_dict_to_tensordict([field]),
            tags=[{"status": "success"}],
        )

    records = SpellingTrainer()._records(types.SimpleNamespace(keys=keys, partition_id=partition_id))

    assert [r["score"] for r in records] == [0.0, 1.0]
    assert [r["num_turns"] for r in records] == [3, 4]
    assert [r["data_source"] for r in records] == ["suite_0", "suite_1"]
    assert records[1]["extra_fields"] == {"episode": {"task": 1}}
    assert records[0]["input"] == "7 8"
    assert records[1]["output"] == " ".join(str(t) for t in range(1, response_lengths[1] + 1))
    tq.kv_clear(keys=keys, partition_id=partition_id)


def test_the_evaluation_launches_verl_with_its_own_runner(monkeypatch):
    launches = []
    monkeypatch.setattr(
        embodied, "run_ppo", lambda config, task_runner_class: launches.append((config, task_runner_class))
    )
    # what the entry point reads of a composed config before the launch
    config = OmegaConf.create({"run_sizes": {"sessions_per_replica": None}, "ray_kwargs": {}})

    EmbodiedEvaluation(config).run()

    assert launches == [(config, embodied.EmbodiedEvaluationRunner)]


def test_the_replicas_load_the_policy_from_the_model_path(monkeypatch):
    config = OmegaConf.create(
        {"trainer": {"eval_step": STEP}, "actor_rollout_ref": {"rollout": {"load_format": "dummy"}}}
    )
    created = []
    monkeypatch.setattr(embodied, "RewardLoopManager", lambda config, rm_resource_pool: None)
    monkeypatch.setattr(
        embodied, "LLMServerManager", types.SimpleNamespace(create=lambda config: created.append(config))
    )

    SetupTrainer(config)._setup()

    # verl's dummy weights wait for a trainer's sync that never comes: every turn ran to its token cap as noise
    assert created[0].actor_rollout_ref.rollout.load_format == "auto"
    assert config.actor_rollout_ref.rollout.load_format == "dummy"


def test_the_refill_submits_every_episode_once():
    trainer = RecordingTrainer(rows=3)

    # the dataloader is not restarted: the last row goes out alone, then the refill reports nothing submitted
    assert trainer._add_prompts_to_generate(2) == 2
    assert trainer._add_prompts_to_generate(2) == 1
    assert trainer._add_prompts_to_generate(2) == 0

    uids = [uid for batch in trainer.submitted for uid in list(batch["uid"])]
    assert len(uids) == 3 and len(set(uids)) == 3


def test_an_episode_whose_tool_has_no_room_waits_for_the_next_refill():
    # the replay buffer's per-tool room (RoboDojo's sessions within its Apps): training's refill skips such a row to a
    # later pass of the dataloader, which the evaluation's single pass never makes
    trainer = RecordingTrainer(rows=3, tool_selection=["robodojo"])

    assert trainer._add_prompts_to_generate(3, free_sessions_by_tool={"robodojo": 1}) == 1
    assert trainer._add_prompts_to_generate(3, free_sessions_by_tool={"robodojo": 0}) == 0
    assert trainer._add_prompts_to_generate(3, free_sessions_by_tool={"robodojo": 5}) == 2
    assert trainer._add_prompts_to_generate(3, free_sessions_by_tool={"robodojo": 5}) == 0

    assert [int(index) for batch in trainer.submitted for index in batch["index"]] == [0, 1, 2]


def test_the_records_hold_each_episode_and_the_summary_their_scores_by_data_source():
    records = EmbodiedEvaluationTrainer.assemble(
        STEP,
        ["a", "b", "c"],
        [1.0, 0.0, 1.0],
        [3, 4, 5],
        ["suite_a", "suite_a", "suite_b"],
        [{"env": 1}, {}, {}],
        ["in"] * 3,
        ["o1", "o2", "o3"],
    )

    assert records[1] == {
        "step": STEP,
        "uid": "b",
        "data_source": "suite_a",
        "score": 0.0,
        "num_turns": 4,
        "extra_fields": {},
        "input": "in",
        "output": "o2",
    }
    assert EmbodiedEvaluationTrainer.summarize(STEP, records) == {
        "step": STEP,
        "episodes": 3,
        "score": 2 / 3,
        "by_data_source": {"suite_a": {"episodes": 2, "score": 0.5}, "suite_b": {"episodes": 1, "score": 1.0}},
    }
    assert EmbodiedEvaluationTrainer.to_json(np.array([1, 2])) == [1, 2]


def test_an_episode_s_record_is_in_the_file_as_soon_as_it_is_sampled(tq_init, tmp_path):
    # where an earlier evaluation left its two files
    (tmp_path / f"eval_{STEP}.jsonl").write_text("an earlier evaluation's record\n")
    (tmp_path / f"eval_{STEP}_summary.json").write_text("{}")
    episodes = EndedEpisodes(3)
    replay_buffer = WatchedReplayBuffer(tmp_path)

    QueueEvaluation(tmp_path, episode_n=3, replay_buffer=replay_buffer).fit(agent_loop_manager=None)

    # at the first sample the earlier files are gone; then a record more at each sample, and no summary before the end
    assert replay_buffer.file_states == [(0, False), (1, False), (2, False)]
    records = [json.loads(line) for line in (tmp_path / f"eval_{STEP}.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert {record["uid"]: record for record in records} == episodes.records
    summary = json.loads((tmp_path / f"eval_{STEP}_summary.json").read_text())
    assert summary == EmbodiedEvaluationTrainer.summarize(STEP, list(episodes.records.values()))


def test_an_evaluation_that_ends_early_keeps_the_records_of_its_finished_episodes(tq_init, tmp_path):
    episodes = EndedEpisodes(3)
    replay_buffer = WatchedReplayBuffer(tmp_path, end_at_sample=2)

    with pytest.raises(EvaluationEnded):
        QueueEvaluation(tmp_path, episode_n=3, replay_buffer=replay_buffer).fit(agent_loop_manager=None)

    records = [json.loads(line) for line in (tmp_path / f"eval_{STEP}.jsonl").read_text().splitlines()]
    episodes.clear(set(episodes.records) - {record["uid"] for record in records})
    assert len(records) == 2
    assert all(record == episodes.records[record["uid"]] for record in records)
    # its absence marks the evaluation as unfinished
    assert not (tmp_path / f"eval_{STEP}_summary.json").exists()
