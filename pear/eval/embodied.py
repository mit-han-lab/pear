import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig

from pear.trainer.config import RunConfig
from pear.trainer.runner import PearTaskRunner
from pear.trainer.trainer import SessionCapMixin
from verl.experimental.reward_loop import RewardLoopManager
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager


class EmbodiedEvaluationTrainer(SessionCapMixin, PPOTrainer):
    """The training rollout on a policy's episodes, every finished group recorded instead of trained on."""

    LOAD_FORMAT = "auto"  # vLLM reads the policy's safetensors from the model path

    def _setup(self) -> None:
        # the separate-async trainer's standalone replicas and nothing of the training side: the replicas load the
        # policy from the model path, the replay buffer (SessionCapMixin's) caps and refills the sessions as in training
        self._init_tokenizer()
        self._init_dataloader()
        self.global_steps = int(self.config.trainer.eval_step)
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=None)

        config = deepcopy(self.config)
        # verl's default "dummy" leaves random weights for the trainer's first sync, and no trainer syncs here
        config.actor_rollout_ref.rollout.load_format = self.LOAD_FORMAT
        memory = config.actor_rollout_ref.rollout.get("standalone_gpu_memory_utilization")
        if memory is not None:
            config.actor_rollout_ref.rollout.gpu_memory_utilization = memory
        self.server_manager = LLMServerManager.create(config=config)

    def get_llm_client(self) -> FullyAsyncLLMServerClient:
        return self.server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def fit(self, agent_loop_manager: AgentLoopManagerTQ) -> None:
        self.agent_loop_manager = agent_loop_manager
        self.train_dataloader_it = iter(self.train_dataloader)
        self.next_episode = None  # the episode the refill submits next, held while its tool has no room

        output_dir = self.config.trainer.validation_data_dir
        records_path = os.path.join(output_dir, f"eval_{self.global_steps}.jsonl")
        summary_path = os.path.join(output_dir, f"eval_{self.global_steps}_summary.json")
        os.makedirs(output_dir, exist_ok=True)
        # the summary stands for an evaluation that recorded every episode: an earlier evaluation's is removed first
        if os.path.exists(summary_path):
            os.remove(summary_path)

        records = []
        with open(records_path, "w") as records_file:
            # a group at a time, while the cap keeps the replicas refilled: its records are in the file as soon as the
            # replay buffer hands it over, so an evaluation that is ended early keeps the episodes it finished
            for _ in range(len(self.train_dataset)):
                batch = self.replay_buffer.sample(global_steps=self.global_steps, partition_id="train", batch_size=1)[0]
                group_records = self._records(batch)
                records_file.writelines(json.dumps(record, default=self.to_json) + "\n" for record in group_records)
                records_file.flush()
                records.extend(group_records)
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        summary = self.summarize(self.global_steps, records)
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=1)
        print(json.dumps(summary))

    def on_step_end(self) -> None:
        return

    def on_sample_end(self) -> None:
        return

    def _add_prompts_to_generate(self, num_prompts: int, free_sessions_by_tool: dict[str, int] | None = None) -> int:
        # one pass over the episodes in order: once they run out the refill submits nothing and the replay buffer only
        # samples; an episode whose tool has no room for its sessions (free_sessions_by_tool, the buffer's caps by
        # tool) waits for the next refill, where training's refill skips it to a later pass of the dataloader
        rollout_n = self.config.actor_rollout_ref.rollout.n
        free_sessions_by_tool = dict(free_sessions_by_tool or {})  # what is left of each capped tool's room
        episodes = []

        while len(episodes) < num_prompts:
            if self.next_episode is None:
                try:
                    self.next_episode = self._fetch_one_gen_batch()
                except StopIteration:
                    break
            if not self._take_tool_room(free_sessions_by_tool, self._tool_selections(self.next_episode)[0], rollout_n):
                break
            episodes.append(self.next_episode)
            self.next_episode = None

        if not episodes:
            return 0

        return self._submit_batch_to_rollout(self._stamped_batch(episodes))

    def _next_dataloader_batch(self) -> dict:
        return next(self.train_dataloader_it)  # StopIteration at the end: no second pass, unlike training

    def _records(self, batch: tq.KVBatchMeta) -> list[dict[str, Any]]:
        texts = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
        )
        pad = self.tokenizer.pad_token_id
        inputs = self.tokenizer.batch_decode(self.dense(texts["prompts"], pad), skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(self.dense(texts["responses"], pad), skip_special_tokens=True)

        # non-tensor columns come back as a LinkedList, a NonTensorStack or a numpy array depending on the backend
        fields = ["uid", "rm_scores", "num_turns", "data_source", "extra_fields"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        return self.assemble(
            self.global_steps,
            list(data["uid"]),
            self.dense(data["rm_scores"], 0).sum(dim=1).tolist(),
            list(data["num_turns"]),
            list(data["data_source"]),
            list(data["extra_fields"]),
            inputs,
            outputs,
        )

    @staticmethod
    def dense(column: torch.Tensor, padding: int) -> torch.Tensor:
        """A tensor column of the transfer queue as one tensor: jagged when its rows differ in length, stacked when
        not."""
        return column.to_padded_tensor(padding) if column.is_nested else column

    @staticmethod
    def assemble(
        step: int,
        uids: list[str],
        scores: list[float],
        turns: list[int],
        sources: list[str],
        extras: list[dict[str, Any]],
        inputs: list[str],
        outputs: list[str],
    ) -> list[dict[str, Any]]:
        """One record per trajectory from the transfer queue's columns."""
        return [
            {
                "step": step,
                "uid": uid,
                "data_source": source,
                "score": score,
                "num_turns": int(turn),
                "extra_fields": extra,
                "input": text_in,
                "output": text_out,
            }
            for uid, score, turn, source, extra, text_in, text_out in zip(
                uids, scores, turns, sources, extras, inputs, outputs, strict=True
            )
        ]

    @staticmethod
    def summarize(step: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        """The mean score, overall and per data source."""
        scores = defaultdict(list)
        for record in records:
            scores[record["data_source"]].append(record["score"])

        return {
            "step": step,
            "episodes": len(records),
            "score": float(np.mean([r["score"] for r in records])),
            "by_data_source": {
                source: {"episodes": len(s), "score": float(np.mean(s))} for source, s in sorted(scores.items())
            },
        }

    @staticmethod
    def to_json(value: Any) -> Any:
        """JSON for what the transfer queue hands back: numpy and torch values."""
        return value.tolist() if hasattr(value, "tolist") else str(value)


@ray.remote
class EmbodiedEvaluationRunner(PearTaskRunner):
    """The runner of an evaluation: the evaluation trainer, and no distillation teacher."""

    def build_trainer(self, config: DictConfig) -> PPOTrainer:
        return EmbodiedEvaluationTrainer(config=config)

    def teacher_client(self, trainer: PPOTrainer) -> None:
        return None


class EmbodiedEvaluation:
    """An evaluation from its composed config: verl's launch with the evaluation's runner."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def run(self) -> None:
        RunConfig(self.config).prepare()

        # without the two checks of a training (pear/trainer/main.py): validate_config checks the actor's batch sizes
        # and losses, and an evaluation has no actor; auto_set_device changes trainer.device only on a machine whose
        # accelerator is not the configured one
        run_ppo(self.config, task_runner_class=EmbodiedEvaluationRunner)


@hydra.main(config_path="../configs", config_name="embodied", version_base=None)
def main(config: DictConfig) -> None:
    EmbodiedEvaluation(config).run()


if __name__ == "__main__":
    main()
