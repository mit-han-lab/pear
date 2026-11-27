from __future__ import annotations

import asyncio
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tinker
import torch
from tinker_helpers import TestRenderer, TestSampler, TestService, TestTrainingClient

from pear.agent_loop.episode import EndReason
from pear.agent_loop.protocol import JsonBlockProtocol, NoToolCall
from pear.agent_loop.tasks import EpisodeTask, load_tasks
from pear.backends.tinker.config import TinkerConfig
from pear.backends.tinker.loss import (
    PearLoss,
    builtin_loss_config,
    group_advantages,
    sparse_targets,
    stable_reinforce_data,
    training_batches,
    trajectory_datum,
)
from pear.backends.tinker.preflight import preflight
from pear.backends.tinker.rollout import EpisodeRollout
from pear.backends.tinker.state import Checkpoints, Group, RunState
from pear.backends.tinker.train import evaluate, train
from pear.backends.tinker.usage import Usage
from pear.envs.session import EnvironmentSession


@pytest.fixture
def setup(tmp_path):
    config = TinkerConfig(
        base_model="test",
        renderer="test",
        episode_files=[],
        tool_configs=[],
        run_dir=str(tmp_path),
        max_prompt_tokens=4096,
        response_budget=1000,
        context_length=8192,
        train_request_tokens=8192,
        group_size=2,
        batch_groups=1,
        max_sessions=4,
        steps=1,
        topk=2,
        max_groups_per_step=20,
        learning_rate=0.01,
        objective="pear",
    )
    session = EnvironmentSession({"env_class": "tinker_helpers.TestEnvironment", "log_dir": str(tmp_path)})
    task = EpisodeTask("tinker_helpers.TestEnvironment", {"task": 0}, "test", 0)
    runtime = SimpleNamespace(sessions={task.env_class: session}, caps={task.env_class: 4})
    return config, session, task, runtime


def test_shared_task_expansion_preserves_order_and_reweighting(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(
        json.dumps(
            {
                "env_class": "test.Env",
                "data_source": "suite",
                "env": [
                    {"suite": ["a", "b"], "trial": "0-1,4"},
                    {"suite": ["c"], "trial": [7]},
                ],
            }
        )
    )
    tasks = load_tasks([path, path])
    assert [t.env_args for t in tasks[:7]] == [
        {"suite": suite, "trial": trial} for suite in ("a", "b") for trial in (0, 1, 4)
    ] + [{"suite": "c", "trial": 7}]
    assert [t.index for t in tasks] == list(range(14))
    assert [t.env_args for t in tasks[:7]] == [t.env_args for t in tasks[7:]]


def test_shared_parser_keeps_last_answer_block_after_thinking():
    assert JsonBlockProtocol.parse('```json\n{"x":0}\n```</think>```json\n{"x":1}\n```', "</think>") == '{"x":1}'
    with pytest.raises(NoToolCall, match="no ```json block"):
        JsonBlockProtocol.parse('```json\n{"x":0}\n```</think>answer', "</think>")
    with pytest.raises(NoToolCall, match="not valid JSON"):
        JsonBlockProtocol.parse('```json\n{"x":}\n```')


@pytest.mark.asyncio
async def test_cookbook_rollout_keeps_images_sampled_ids_and_masks(setup):
    config, session, task, _ = setup
    sampler = TestSampler()
    trajectory = await EpisodeRollout(config, TestRenderer(), {task.env_class: session}).run(task, sampler, "g0", 0)
    assert len(trajectory.turns) == 2
    assert trajectory.end_reason == EndReason.ENV
    assert session.client.closed == [trajectory.episode.uid]
    first, second = trajectory.turns
    prefix = second.prompt.chunks[: len(first.prompt.chunks)]
    assert prefix == first.prompt.chunks
    assert second.prompt.chunks[len(prefix)].tokens == first.tokens
    assert all(k == config.topk for _, _, k in sampler.calls)
    datum = trajectory_datum(trajectory, 1.0, config.topk, trajectory.generated_tokens)
    mask = datum.loss_fn_inputs["mask"].to_torch().bool()
    assert datum.loss_fn_inputs["target_tokens"].sparse_crow_indices is not None
    targets = datum.loss_fn_inputs["target_tokens"].to_torch(pad_value=-1)
    assert mask.sum() == trajectory.generated_tokens
    assert targets[mask, 0].tolist() == first.tokens + second.tokens
    assert (targets[~mask] == -1).all()
    assert sum(isinstance(c, tinker.types.ImageChunk) for c in datum.model_input.chunks) == 4
    assert sampler.calls[1][1].max_tokens == config.response_budget - (second.prompt.length - first.prompt.length)


@pytest.mark.asyncio
async def test_budget_exhaustion_precedes_execution_of_a_valid_action(setup):
    config, session, task, _ = setup
    text = '```json\n{"arm":{"x":1}}\n```'
    config.response_budget = len(text)
    trajectory = await EpisodeRollout(config, TestRenderer(), {task.env_class: session}).run(
        task, TestSampler(text=text), "g0", 0
    )
    assert trajectory.end_reason == EndReason.BUDGET
    assert trajectory.reward == 0
    assert [call for call, _ in session.client.calls] == ["open"]
    assert trajectory.generated_tokens == len(text)


@pytest.mark.asyncio
async def test_parse_failures_remain_negative_training_examples(setup):
    config, session, task, _ = setup
    trajectory = await EpisodeRollout(config, TestRenderer(), {task.env_class: session}).run(
        task, TestSampler(text="bad"), "g0", 0
    )
    assert trajectory.end_reason == EndReason.PARSE
    assert trajectory.generated_tokens == 3 and trajectory.reward == 0
    assert [call for call, _ in session.client.calls] == ["open"]


@pytest.mark.asyncio
async def test_lost_episodes_restart_and_discard_the_old_trajectory(setup):
    config, session, task, _ = setup
    session.client.lose_once = True
    usage = Usage(config.run_dir)
    trajectory = await EpisodeRollout(config, TestRenderer(), {task.env_class: session}, usage).run(
        task, TestSampler(), "g0", 0
    )
    assert len(session.client.closed) == 2
    assert session.client.closed[-1] == trajectory.episode.uid
    assert len(trajectory.turns) == 2
    await usage.close()
    assert usage.totals["episode_attempts_started"] == 2
    assert usage.totals["episode_attempts_restarted"] == usage.totals["episode_attempts_failed"] == 1
    assert usage.totals["generated_tokens"] > trajectory.generated_tokens


@pytest.mark.asyncio
async def test_preflight_records_real_environment_calls_and_camera_frames(setup):
    config, session, task, runtime = setup
    runtime.resource_plan = {"test": "fixture"}
    report = await preflight(config, runtime, SimpleNamespace(tasks=[task]))
    assert report["status"] == "passed"
    family = report["families"][task.env_class]
    assert family["native_steps"] == 1
    assert len(family["reset_cameras"]) == len(family["action_cameras"]) == 2
    assert all(frame["sha256"] for frame in family["action_cameras"])
    assert not session.client.steps


def test_group_advantages_use_sample_standard_deviation():
    expected = (torch.tensor([0.0, 1.0, 1.0]) - 2 / 3) / (torch.tensor([0.0, 1.0, 1.0]).std() + 1e-6)
    torch.testing.assert_close(torch.tensor(group_advantages([0, 1, 1])), expected)
    assert group_advantages([1, 1]) == [0.0, 0.0]


def test_score_centering_matches_the_patched_verl_reference():
    from collections.abc import Callable
    from pathlib import Path

    # Evaluate the actual function shipped in the patch, without importing the GPU training stack.
    patch = next((Path(__file__).resolve().parents[2] / "3rdparty/verl/patches").glob("0011-*.patch"))
    text = "\n".join(
        line[1:] for line in patch.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    source = text[text.index("def score_centering_correction(") : text.index("def score_centering_logits_processor(")]
    namespace = {"torch": torch, "Callable": Callable}
    exec(source, namespace)
    torch.manual_seed(2)
    logits = torch.randn(4, 7, requires_grad=True)
    old = torch.randn(4, 7).log_softmax(-1)
    heads = old.topk(3, dim=-1)
    ids = torch.cat([torch.tensor([[0], [1], [2], [3]]), heads.indices], dim=-1)
    logp = logits.log_softmax(-1).gather(1, ids)
    mask, advantages = torch.tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, -1.0, 0.5])
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1] * 4),
        loss_fn_inputs={
            "target_tokens": ids,
            "mask": mask,
            "head_logprobs": heads.values,
            "advantages": advantages,
        },
    )
    loss, _ = PearLoss(3)([datum], [logp])
    correction, _, _ = namespace["score_centering_correction"](logp[:, 1:], heads.values, torch.ones_like)
    reference = ((-advantages * logp[:, 0] + advantages * correction) * mask).sum() / 3
    torch.testing.assert_close(loss, reference)
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(gradient, expected)
    assert not gradient[1].any()


@pytest.mark.asyncio
async def test_split_training_heads_preserve_loss_and_gradients():
    torch.manual_seed(4)
    logits = torch.randn(64, 80, requires_grad=True)
    old = torch.randn(64, 80).log_softmax(-1).topk(32, dim=-1)
    ids = torch.cat([torch.randint(80, (64, 1)), old.indices], dim=-1)
    mask = torch.arange(64) % 4 != 0
    ids[~mask] = -1
    datum = tinker.Datum(
        tinker.ModelInput.from_ints([1] * 64),
        {
            **sparse_targets(ids.numpy()),
            "mask": mask.float(),
            "advantages": mask.float() * -0.7,
            "head_logprobs": old.values,
        },
    )
    objective = PearLoss(int(mask.sum()))
    reference, _ = objective([datum], [logits.log_softmax(-1).gather(1, ids.clamp_min(0))])
    expected = torch.autograd.grad(reference, logits)[0]

    class Client:
        async def forward_backward_custom(self, pieces, callback):
            assert [d.loss_fn_inputs["target_tokens"].shape[1] for d in pieces] == [20, 13]
            assert all(d.model_input == datum.model_input for d in pieces)
            targets = [d.loss_fn_inputs["target_tokens"].to_torch(pad_value=-1) for d in pieces]
            torch.testing.assert_close(torch.cat(targets, dim=-1), ids)
            logp = logits.log_softmax(-1)
            values = [logp.gather(1, target.clamp_min(0)) for target in targets]
            loss, _ = callback(pieces, values)
            loss.backward()
            return loss.detach()

    actual = await objective.forward_backward(Client(), [datum])
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(logits.grad, expected)
    assert not logits.grad[~mask].any()


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", ["pear", "stable_reinforce"])
async def test_train_publish_checkpoint_resume_and_evaluate(setup, objective):
    config, session, task, runtime = setup
    config = replace(config, objective=objective)
    dataset, renderer, service = SimpleNamespace(tasks=[task]), TestRenderer(), TestService()
    first = await train(config, runtime, dataset, renderer, service)
    assert first["step"] == 1
    client = service.clients[-1]
    assert client.updates == 1 and len(client.published) == 2
    assert client.stability_flags == [objective == "stable_reinforce"] * 2
    if objective == "stable_reinforce":
        assert len(client.builtin_calls) == 1
        data, loss_fn, loss_config = client.builtin_calls[0]
        assert loss_fn == objective and loss_config == {"kl_coeff": 0.0}
        assert len(data) == config.group_size
        assert all(d.loss_fn_input_spans for d in data)
    assert not torch.equal(client.published[0], client.published[1])
    metrics = json.loads((Checkpoints(config).root.parent / "metrics.jsonl").read_text().splitlines()[0])
    assert metrics["optimizer/grad_norm"] > 0
    assert len(metrics["trained_groups"]) == config.batch_groups
    state, saved = Checkpoints(config).load()
    assert state.step == 1 and saved == first["model_path"]
    second = await train(replace(config, steps=2), runtime, dataset, renderer, service)
    assert second["step"] == 2 and service.resumed_optimizer
    summary = await evaluate(config, runtime, dataset, renderer, service, second["model_path"], 2)
    assert summary["episodes"] == 1
    evaluation_client = service.clients[-1]
    assert evaluation_client.updates == 0 and evaluation_client.published_kinds == ["snapshot"]
    torch.testing.assert_close(evaluation_client.published[0], service.saved[second["model_path"]][0])
    assert not session.client.steps
    with pytest.raises(ValueError, match="recipe differs"):
        Checkpoints(replace(config, learning_rate=0.02)).load()
    # Evaluation reads only a model reference, even with a different recipe or no local rollout pickle.
    for path in Checkpoints(config).root.glob("*.pkl"):
        path.unlink()
    assert Checkpoints(replace(config, learning_rate=0.02)).reference()["model_path"] == second["model_path"]
    warm_config = replace(
        config, run_dir=str(Checkpoints(config).root.parent / "next_stage"), init_checkpoint=second["model_path"]
    )
    warm = await train(warm_config, runtime, dataset, renderer, service)
    assert warm["step"] == 1 and service.clients[-1].updates == 1


@pytest.mark.asyncio
async def test_failed_update_resumes_the_saved_batch_without_sampling_it_again(setup, monkeypatch):
    config, session, task, runtime = setup
    config = replace(config, max_sessions=config.group_size)
    service = TestService()
    original = PearLoss.forward_backward

    async def fail(*args):
        raise RuntimeError("service disconnected")

    monkeypatch.setattr(PearLoss, "forward_backward", fail)
    with pytest.raises(RuntimeError, match="service disconnected"):
        await train(config, runtime, SimpleNamespace(tasks=[task]), TestRenderer(), service)
    state, _ = Checkpoints(config).load()
    assert state.step == 0 and state.groups[0].complete
    monkeypatch.setattr(PearLoss, "forward_backward", original)

    # A complete retained batch must train even if no new rollout can finish.
    async def unavailable(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(EpisodeRollout, "run", unavailable)
    result = await asyncio.wait_for(train(config, runtime, SimpleNamespace(tasks=[task]), TestRenderer(), service), 3)
    assert result["step"] == 1 and service.resumed_optimizer
    assert not session.client.steps


@pytest.mark.asyncio
async def test_generated_heavy_targets_stay_sparse_and_microbatches_share_normalization(setup):
    config, session, task, _ = setup
    # A long answer has far more trained than observation positions. SDK sparse heuristics must not turn
    # its 33-candidate targets into a dense tensor, which the service rejects above 20 candidates.
    text = "thinking " * 1000 + '```json\n{"arm":{"x":1}}\n```'
    config = replace(config, topk=32, allow_head_continuations=True, response_budget=25000, context_length=30000)
    rollout = EpisodeRollout(config, TestRenderer(), {task.env_class: session})
    first = await rollout.run(task, TestSampler(text=text), "long", 0)
    second = await rollout.run(task, TestSampler(text=text.replace('"x":1', '"x":0')), "long", 1)
    group = Group("long", 0, 0, [first, second])
    normalizer, full = training_batches([group], replace(config, train_request_tokens=100000))
    [data] = list(full)
    _, split = training_batches([group], replace(config, train_request_tokens=data[0].model_input.length))
    microbatches = list(split)
    assert [len(batch) for batch in microbatches] == [1, 1]
    assert normalizer == first.generated_tokens + second.generated_tokens
    assert all(d.loss_fn_inputs["target_tokens"].sparse_crow_indices is not None for d in data)
    assert all(d.loss_fn_inputs["mask"].to_torch().mean() > 0.5 for d in data)
    assert all(t.head_ids.dtype == np.int32 for t in first.turns)
    assert all(t.head_logprobs.dtype == np.float32 for t in first.turns)
    logits = torch.linspace(-1, 1, 256, requires_grad=True)
    logp = logits.log_softmax(-1)
    values = [logp[d.loss_fn_inputs["target_tokens"].to_torch().clamp_min(0)] for d in data]
    loss, _ = PearLoss(normalizer)(data, values)
    split_loss = sum(PearLoss(normalizer)([d], [v])[0] for d, v in zip(data, values, strict=True))
    torch.testing.assert_close(loss, split_loss)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0], torch.autograd.grad(split_loss, logits)[0]
    )


@pytest.mark.asyncio
async def test_observation_tokens_exhaust_the_budget_without_another_sample(setup):
    config, session, task, _ = setup
    sampler = TestSampler()
    # Enough room for a complete answer, but not the following camera frames and observation text.
    config = replace(config, response_budget=100)
    trajectory = await EpisodeRollout(config, TestRenderer(), {task.env_class: session}).run(task, sampler, "g", 0)
    assert trajectory.end_reason == EndReason.BUDGET
    assert len(sampler.calls) == 1 and trajectory.episode.num_chunks == 1
    assert session.client.closed == [trajectory.episode.uid]


@pytest.mark.asyncio
async def test_native_loss_keeps_provenance_images_and_global_token_weighting(setup):
    config, session, task, _ = setup
    config = replace(config, objective="stable_reinforce", response_budget=160000, context_length=262144)
    rollout = EpisodeRollout(config, TestRenderer(), {task.env_class: session})
    sampler = TestSampler(text='```json\n{"arm":{"x":1}}\n```')
    positive = await rollout.run(task, sampler, "native", 0)
    negative = await rollout.run(task, TestSampler(text="bad"), "native", 1)
    assert positive.reward == 1 and negative.reward == 0
    assert len(sampler.calls) == len(positive.turns) == 2
    assert sampler.calls[0][1].max_tokens == 160000
    assert all(k == 0 for _, _, k in sampler.calls)
    assert all(turn.head_ids.shape[1] == 0 and turn.sequence_id for turn in positive.turns)
    normalizer, batches = training_batches([Group("native", 0, 0, [positive, negative])], config)
    [data] = list(batches)
    assert len(data) == 2
    for datum, trajectory, advantage in zip(data, [positive, negative], group_advantages([1, 0]), strict=True):
        assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
        mask = datum.loss_fn_inputs["advantages"].to_torch() != 0
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        assert targets.ndim == 1 and mask.sum() == trajectory.generated_tokens
        assert targets[mask].tolist() == [token for turn in trajectory.turns for token in turn.tokens]
        assert not advantages[~mask].any()
        torch.testing.assert_close(advantages[mask], torch.full_like(advantages[mask], advantage / normalizer))
        assert sum(isinstance(c, tinker.types.ImageChunk) for c in datum.model_input.chunks) == 2 * len(
            trajectory.turns
        )
        # Reconstruct every text target from its provenance, including masked observation text.
        turns = {turn.sequence_id: turn for turn in trajectory.turns}
        offset = 0
        sampled = []
        for span in datum.loss_fn_input_spans:
            turn = turns[span.sequence_id]
            if isinstance(span, tinker.SampledProvenanceSpan):
                assert span.offset == 0 and span.length == len(turn.tokens)
                expected = turn.tokens
                sampled.append(span.sequence_id)
                assert mask[offset : offset + span.length].all()
            else:
                prompt = [
                    token
                    for chunk in turn.prompt.chunks
                    for token in (chunk.tokens if isinstance(chunk, tinker.EncodedTextChunk) else [-1] * chunk.length)
                ]
                expected = prompt[span.offset : span.offset + span.length]
                assert not mask[offset : offset + span.length].any()
            actual = targets[offset : offset + span.length].tolist()
            assert all(a == e for a, e in zip(actual, expected, strict=True) if e != -1)
            offset += span.length
        assert sampled == [turn.sequence_id for turn in trajectory.turns]
        assert offset == datum.model_input.length == sum(s.length for s in datum.model_input_spans)
    _, split = training_batches(
        [Group("native", 0, 0, [positive, negative])], replace(config, train_request_tokens=data[0].model_input.length)
    )
    microbatches = list(split)
    assert [len(batch) for batch in microbatches] == [1, 1]
    for expected, [actual] in zip(data, microbatches, strict=True):
        torch.testing.assert_close(
            actual.loss_fn_inputs["advantages"].to_torch(), expected.loss_fn_inputs["advantages"].to_torch()
        )
    assert builtin_loss_config(config, normalizer) == {"kl_coeff": 0.0}
    assert builtin_loss_config(replace(config, loss_config={"kl_coeff": 0.02, "kl_estimator": "k3"}), normalizer) == {
        "kl_coeff": 0.02 / normalizer,
        "kl_estimator": "k3",
    }
    positive.turns[0].sequence_id = None
    with pytest.raises(ValueError, match="sequence_id"):
        stable_reinforce_data(positive, 1, normalizer)


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint_form", ["compact", "diagnostic", "legacy"])
async def test_native_resume_recollects_session_bound_rollouts(setup, monkeypatch, checkpoint_form):
    config, session, task, runtime = setup
    config = replace(
        config,
        objective="stable_reinforce",
        max_sessions=config.group_size,
        prefetch_groups=0,
        dump_training_batch=checkpoint_form != "compact",
    )
    service = TestService()
    original = TestTrainingClient.forward_backward

    async def fail(*args, **kwargs):
        raise RuntimeError("service disconnected")

    monkeypatch.setattr(TestTrainingClient, "forward_backward", fail)
    with pytest.raises(RuntimeError, match="service disconnected"):
        await train(config, runtime, SimpleNamespace(tasks=[task]), TestRenderer(), service)
    checkpoints = Checkpoints(config)
    state, _ = checkpoints.load()
    assert state.step == 0 and all(t is None for g in state.groups for t in g.trajectories)
    discarded = state.counters["pending_episodes"]
    discarded_tokens = state.counters["pending_generated_tokens"]
    assert discarded == config.group_size and discarded_tokens > 0
    dump = checkpoints.root / "batch.pkl"
    assert dump.exists() == (checkpoint_form != "compact")
    if dump.exists():
        with dump.open("rb") as file:
            groups = pickle.load(file)
        assert groups[0].complete
        assert sum(t.generated_tokens for g in groups for t in g.trajectories) == discarded_tokens
        if checkpoint_form == "legacy":
            state.groups = groups
            state.counters.pop("pending_episodes")
            state.counters.pop("pending_generated_tokens")
            with (checkpoints.root / "step_00000000.pkl").open("wb") as file:
                pickle.dump(state, file)
    old_episodes = set(session.client.closed)
    monkeypatch.setattr(TestTrainingClient, "forward_backward", original)
    result = await train(
        replace(config, dump_training_batch=False), runtime, SimpleNamespace(tasks=[task]), TestRenderer(), service
    )
    assert result["step"] == 1 and service.resumed_optimizer
    state, _ = Checkpoints(config).load()
    assert state.counters["session_discarded_episodes"] == discarded
    usage = Usage(config.run_dir)
    assert usage.totals["resume_discarded_episodes"] == discarded
    assert usage.totals["resume_discarded_generated_tokens"] == discarded_tokens
    assert len(set(session.client.closed) - old_episodes) >= config.group_size
    assert not session.client.steps


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", ["stable_reinforce", "pear", "importance_sampling", "ppo"])
async def test_checkpoints_keep_payloads_only_for_replayable_objectives(setup, objective):
    config, _, _, _ = setup
    native = objective == "stable_reinforce"
    checkpoints = Checkpoints(replace(config, objective=objective))
    # A native snapshot must not even attempt to pickle the trajectory payload.
    trajectory = SimpleNamespace(generated_tokens=7, payload=(lambda: None) if native else b"camera")
    pending = Group("pending", 0, 2, [trajectory, trajectory])
    queued = Group("queued", 1, 1, [trajectory, None])
    state = RunState(step=3, epoch=2, cursor=1, next_group=8, groups=[queued])
    client = await TestService().create_lora_training_client()
    path = await checkpoints.save(client, state)
    saved, model_path = checkpoints.load()
    assert model_path == path
    assert saved.counters.get("pending_episodes", 0) == int(native)
    checkpoints.save_pending(state, [pending])
    saved, model_path = checkpoints.load()
    assert model_path == path
    assert (saved.step, saved.epoch, saved.cursor, saved.next_group) == (3, 2, 1, 8)
    assert [(g.uid, g.task_index, g.dispatch_step) for g in saved.groups] == [("pending", 0, 2), ("queued", 1, 1)]
    if native:
        assert all(t is None for g in saved.groups for t in g.trajectories)
        assert saved.counters["pending_episodes"] == 3 and saved.counters["pending_generated_tokens"] == 21
    else:
        assert saved.groups == [pending, queued]
    assert state.groups == [queued] and queued.trajectories == [trajectory, None]
    assert pending.trajectories == [trajectory, trajectory] and "pending_episodes" not in state.counters
    assert not (checkpoints.root / "batch.pkl").exists()


def test_native_recipe_defaults_and_kl_validation():
    config = TinkerConfig.load("examples/tinker/stage1.yaml", [])
    assert config.objective == "stable_reinforce" and config.loss_config == {"kl_coeff": 0.0}
    for loss_config in ({"kl_coeff": -1}, {"kl_coeff": float("nan")}, {"kl_estimator": "other"}, {"topk": 32}):
        with pytest.raises(ValueError):
            replace(config, loss_config=loss_config).validate()
    for objective in ("pear", "importance_sampling", "ppo"):
        alternative = TinkerConfig.load("examples/tinker/stage1.yaml", [f"objective={objective}", "loss_config={}"])
        assert alternative.objective == objective and alternative.loss_config == {}
