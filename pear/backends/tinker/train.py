from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

import tinker

from pear.eval.records import EvaluationRecords
from pear.utils.file_utils import append_line, write_text_atomically

from .loss import PearLoss, builtin_loss_config, training_batches
from .pool import RolloutPool
from .rollout import EpisodeRollout
from .state import Checkpoints, RunState
from .usage import Usage


async def update_policy(client, batches, config, normalizer, usage=None):
    """Bound outstanding native requests while preserving submission order and one optimizer step."""
    pending, metrics, optimizer = deque(), {"training_request_metrics": []}, None
    adam = tinker.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )

    async def collect(future):
        result = await asyncio.shield(future)
        metrics["training_request_metrics"].append(dict(result.metrics))
        for key, value in result.metrics.items():
            # Service identifiers and means are not additive. Keep their per-request values above.
            if (key.endswith(":sum") or key in {"loss", "score_correction"}) and type(value) in (int, float):
                metrics[key] = metrics.get(key, 0.0) + value

    try:
        for batch in batches:
            if len(pending) >= config.train_inflight_requests:
                await collect(pending.popleft())
            if config.objective == "pear":
                future = PearLoss(normalizer).forward_backward(client, batch)
                # Each candidate partition is processed by the SDK's forward-only phase and its
                # surrogate forward/backward phase. Top-32 therefore processes four contexts.
                passes = 2 * ((config.topk + 20) // 20)
            else:
                # SDK calls reserve ordered request IDs synchronously and return awaitable futures.
                future = client.forward_backward(
                    batch, config.objective, loss_fn_config=builtin_loss_config(config, normalizer)
                )
                passes = 1
            tokens = sum(d.model_input.length for d in batch) * passes
            future = usage.training(future, tokens) if usage else asyncio.ensure_future(future)
            pending.append(future)
            if config.objective == "pear":
                # A Python custom-loss callback must submit its backward pass before the optimizer is queued.
                await collect(pending.popleft())
        optimizer = asyncio.ensure_future(client.optim_step(adam))
        while pending:
            await collect(pending.popleft())
        update = await asyncio.shield(optimizer)
        metrics.update({f"optimizer/{key}": value for key, value in update.metrics.items()})
        if usage:
            usage.record("optimizer_completed", {"optimizer_steps": 1, "retained_training_tokens": normalizer})
        return metrics
    finally:
        # Retrieve already-submitted results on failures; never submit a replacement optimizer update.
        remaining = [*pending, *([optimizer] if optimizer is not None else [])]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


async def train(config, runtime, dataset, renderer, service, usage=None) -> dict:
    usage = usage or Usage(config.run_dir)
    checkpoints = Checkpoints(config)
    restored = checkpoints.load()
    state, model_path = restored if restored else (RunState(), config.init_checkpoint)
    if restored and state.step >= config.steps:
        return {"step": state.step, "model_path": model_path}
    record_stability = config.objective == "stable_reinforce"
    if restored and record_stability:
        # Stability records belong to the old ServiceClient session. Retain tasks and optimizer state,
        # but collect every member again so a resumed group never mixes valid and expired provenance.
        # Older checkpoints still contain trajectories; new native checkpoints keep only these counts.
        discarded = state.counters.pop("pending_episodes", 0) + sum(
            t is not None for group in state.groups for t in group.trajectories
        )
        discarded_tokens = state.counters.pop("pending_generated_tokens", 0) + sum(
            t.generated_tokens for group in state.groups for t in group.trajectories if t is not None
        )
        usage.record(
            "resume_discarded",
            {"resume_discarded_episodes": discarded, "resume_discarded_generated_tokens": discarded_tokens},
        )
        for group in state.groups:
            group.trajectories = [None] * config.group_size
            group.dispatch_step = state.step
        state.counters["session_discarded_episodes"] = state.counters.get("session_discarded_episodes", 0) + discarded
    if restored:
        client = await service.create_training_client_from_state_with_optimizer(
            model_path, base_model=config.base_model
        )
    elif model_path:
        client = await service.create_training_client_from_state(model_path, base_model=config.base_model)
    else:
        client = await service.create_lora_training_client(
            base_model=config.base_model,
            rank=config.rank,
            seed=config.seed,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
        )
    info = client.get_info()
    if not info.is_lora or info.lora_rank != config.rank:
        raise ValueError(f"the training checkpoint must use LoRA rank {config.rank}; received {info.lora_rank}")
    if not restored:
        model_path = await checkpoints.save(client, state)
    sampler = await client.save_weights_and_get_sampling_client(
        kind=config.sampler_kind, record_stability_info=record_stability
    )
    print(json.dumps({"event": "training_started", "step": state.step, "resumed": bool(restored)}), flush=True)
    rollout = EpisodeRollout(config, renderer, runtime.sessions, usage)
    pool = RolloutPool(config, dataset.tasks, rollout, sampler, runtime.caps, state)
    pool.start()
    try:
        while state.step < config.steps:
            started = time.monotonic()
            groups = await pool.take_batch(final=state.step + 1 == config.steps)
            checkpoints.save_pending(state, groups)
            normalizer, batches = training_batches(groups, config)
            print(json.dumps({"event": "batch_ready", "step": state.step, "generated_tokens": normalizer}), flush=True)
            loss_metrics = await update_policy(client, batches, config, normalizer, usage)
            # Publish before incrementing the dispatch version. Work started during publication retains the
            # earlier stamp, so its staleness is never understated.
            pool.sampler = await client.save_weights_and_get_sampling_client(
                kind=config.sampler_kind, record_stability_info=record_stability
            )
            state.step += 1
            pool.update_completed()
            if state.step % config.save_every == 0 or state.step == config.steps:
                model_path = await checkpoints.save(client, state)
            trajectories = [t for group in groups for t in group.trajectories]
            metrics = {
                "step": state.step,
                "seconds": time.monotonic() - started,
                "generated_tokens": normalizer,
                "sampling_requests": sum(turn.sampling_requests for t in trajectories for turn in t.turns),
                "prompt_tokens": sum(turn.prompt_tokens for t in trajectories for turn in t.turns),
                "cached_prompt_tokens": sum(turn.cached_prompt_tokens for t in trajectories for turn in t.turns),
                "train_batch_success": sum(t.reward for t in trajectories) / len(trajectories),
                "rollout_success": state.counters["successes"] / max(1, state.counters["episodes"]),
                "groups": len(groups),
                "trained_groups": [group.uid for group in groups],
                "max_sample_age": max(state.step - 1 - g.dispatch_step for g in groups),
                **state.counters,
                **loss_metrics,
                "usage": dict(usage.totals),
            }
            append_line(Path(config.run_dir) / "metrics.jsonl", json.dumps(metrics))
            print(json.dumps(metrics), flush=True)
    finally:
        await pool.close()
        await usage.close()
    return {"step": state.step, "model_path": model_path}


async def evaluate(
    config, runtime, dataset, renderer, service, model_path: str | None = None, step: int = 0, usage=None
):
    usage = usage or Usage(config.run_dir, "eval")
    if model_path and tinker.ParsedCheckpointTinkerPath.from_tinker_path(model_path).checkpoint_type == "training":
        # Persistent training checkpoints include optimizer state, but the sampler cannot load that format.
        client = await service.create_training_client_from_state(model_path, base_model=config.base_model)
        sampler = await client.save_weights_and_get_sampling_client(kind="snapshot")
    else:
        sampler = await service.create_sampling_client(
            model_path=model_path,
            base_model=None if model_path else config.base_model,
        )
    rollout = EpisodeRollout(config, renderer, runtime.sessions, usage)
    slots = asyncio.Semaphore(config.max_sessions)
    family_slots = {name: asyncio.Semaphore(cap) for name, cap in runtime.caps.items()}
    directory = Path(config.run_dir) / "eval"
    directory.mkdir(parents=True, exist_ok=True)
    records_path, summary_path = directory / f"eval_{step}.jsonl", directory / f"eval_{step}_summary.json"
    summary_path.unlink(missing_ok=True)
    records_path.write_text("")

    async def run(task, member):
        async with slots, family_slots[task.env_class]:
            uid = f"eval_{task.index}"
            trajectory = await rollout.run(task, sampler, uid, member, evaluation=True)
            [record] = EvaluationRecords.assemble(
                step,
                [uid],
                [trajectory.reward],
                [len(trajectory.turns)],
                [task.data_source],
                [
                    {
                        "episode": trajectory.episode.record(),
                        "end_reason": trajectory.end_reason.value,
                        "reward_extra_info": {"success": trajectory.reward, "done": float(trajectory.episode.done)},
                    }
                ],
                [trajectory.initial_text],
                ["\n".join(t.text for t in trajectory.turns)],
            )
            append_line(records_path, json.dumps(record))
            return record

    jobs = [
        asyncio.create_task(run(task, member)) for task in dataset.tasks for member in range(config.eval_group_size)
    ]
    try:
        records = await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await usage.close()
    summary = EvaluationRecords.summarize(step, records)
    write_text_atomically(summary_path, json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary
