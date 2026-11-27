import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import tinker
import torch

from pear.backends.tinker.config import TinkerConfig
from pear.backends.tinker.loss import sparse_targets
from pear.backends.tinker.runtime import resource_plan
from pear.backends.tinker.train import update_policy
from pear.backends.tinker.usage import Usage, reconcile


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "backward", "optimizer"])
async def test_ordered_bounded_submission_and_single_optimizer(tmp_path, failure):
    config = TinkerConfig("test", "test", [], [], str(tmp_path), train_inflight_requests=2)
    data = [
        tinker.Datum(
            tinker.ModelInput.from_ints([1, 2]),
            {"target_tokens": [2, 3], "logprobs": [-1.0, -1.0], "advantages": [0.1, 0.1]},
        )
    ]
    calls, waiting = [], []

    def result(index):
        return SimpleNamespace(metrics={"loss:sum": 1, "clock_cycle:unique": 100 + index, "accuracy:mean": index / 3})

    class Client:
        def forward_backward(self, batch, objective, loss_fn_config):
            assert batch == data and objective == "stable_reinforce" and loss_fn_config == {"kl_coeff": 0.0}
            future = asyncio.get_running_loop().create_future()
            calls.append(f"backward{len(waiting)}")
            waiting.append(future)
            if len(waiting) == 2:
                if failure == "backward":
                    waiting[0].set_exception(RuntimeError("backward failed"))
                    waiting[1].set_result(result(1))
                else:
                    waiting[0].set_result(result(0))
            return future

        def optim_step(self, adam):
            assert len(waiting) == 3 and sum(not future.done() for future in waiting) == 2
            calls.append("optimizer")
            for index, future in enumerate(waiting[1:], 1):
                future.set_result(result(index))
            future = asyncio.get_running_loop().create_future()
            if failure == "optimizer":
                future.set_exception(RuntimeError("optimizer failed"))
            else:
                future.set_result(SimpleNamespace(metrics={}))
            return future

    usage = Usage(tmp_path)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            await update_policy(Client(), [data, data, data], config, 6, usage)
    else:
        assert await update_policy(Client(), [data, data, data], config, 6, usage) == {
            "loss:sum": 3,
            "training_request_metrics": [result(i).metrics for i in range(3)],
        }
    await usage.close()
    assert calls == (
        ["backward0", "backward1"] if failure == "backward" else ["backward0", "backward1", "backward2", "optimizer"]
    )
    assert usage.totals["retained_training_tokens"] == (0 if failure else 6)
    assert usage.totals["optimizer_steps"] == (0 if failure else 1)
    if failure != "backward":
        assert usage.totals["training_tokens"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("topk", [19, 20, 32])
async def test_reference_usage_counts_both_real_sdk_phases(tmp_path, topk):
    config = TinkerConfig("test", "test", [], [], str(tmp_path), objective="pear", topk=topk)
    length = 3
    data = [
        tinker.Datum(
            tinker.ModelInput.from_ints([1] * length),
            {
                **sparse_targets(torch.arange(topk + 1).repeat(length, 1).numpy()),
                "mask": torch.ones(length),
                "advantages": torch.tensor([1.0, -1.0, 1.0]),
                "head_logprobs": torch.full((length, topk), -5.0),
            },
        )
    ]

    class Client(tinker.TrainingClient):
        # Keep the pinned SDK's actual custom-loss orchestration; replace only the service transport.
        def __init__(self):
            self.calls = []

        def _reserve_request_ids(self, count):
            return list(range(count))

        def _chunked_requests_generator(self, data):
            yield SimpleNamespace(data=data, estimated_bytes_count=0)

        def _run_fwd_bwd(self, requests, loss_fn, loss_config, *, forward_only):
            pieces = [d for request in requests for d in request.data]
            self.calls.append((forward_only, sum(d.model_input.length for d in pieces)))

            async def result():
                return tinker.ForwardBackwardOutput(
                    loss_fn_output_type="cross_entropy",
                    metrics={"loss:sum": 0.0},
                    loss_fn_outputs=[
                        {
                            "logprobs": tinker.TensorData.from_torch(
                                torch.full(d.loss_fn_inputs["target_tokens"].shape, -5.0)
                            )
                        }
                        for d in pieces
                    ],
                )

            return SimpleNamespace(result_async=result)

        async def optim_step(self, adam):
            return SimpleNamespace(metrics={})

    client, usage = Client(), Usage(tmp_path)
    await update_policy(client, [data], config, length, usage)
    await usage.close()
    assert [forward_only for forward_only, _ in client.calls] == [True, False]
    assert usage.totals["training_tokens"] == sum(tokens for _, tokens in client.calls)
    assert usage.totals["training_tokens"] == (2 if topk == 19 else 4) * length
    assert usage.totals["retained_training_tokens"] == length


@pytest.mark.asyncio
async def test_usage_keeps_a_sample_after_its_episode_is_cancelled(tmp_path):
    usage = Usage(tmp_path)
    ready = asyncio.Event()

    class Sampler:
        async def sample(self, **kwargs):
            await ready.wait()
            return SimpleNamespace(prompt_cache_hit_tokens=2, sequences=[SimpleNamespace(tokens=[4, 5, 6])])

    task = asyncio.create_task(usage.sample(Sampler(), prompt=tinker.ModelInput.from_ints([1, 2, 3, 4])))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    ready.set()
    await usage.close()
    restored = Usage(tmp_path)
    assert restored.totals == usage.totals
    assert restored.totals["sampling_waits_cancelled"] == 1
    assert restored.totals["sampling_requests_completed"] == 1
    assert restored.totals["generated_tokens"] == 3
    assert restored.totals["cached_prompt_tokens"] == restored.totals["uncached_prompt_tokens"] == 2
    assert json.loads((tmp_path / "usage_summary.json").read_text())["totals"] == dict(usage.totals)


def test_resources_require_available_capacity_and_a_node_that_can_place_each_server(monkeypatch):
    env = SimpleNamespace(
        server_n=lambda config, resources: config["server_n"],
        session_cap=lambda config: 2,
        SERVER_RESOURCES={"num_cpus": 0, "num_gpus": 1, "resources": {"isaac": 1}},
    )
    monkeypatch.setattr("pear.backends.tinker.runtime.pkgutil.resolve_name", lambda _: env)
    configs, capacity = {"test.Env": {"server_n": 2}}, {"GPU": 2, "isaac": 2}
    assert resource_plan(configs, capacity, capacity, [capacity])["families"]["test.Env"]["session_cap"] == 2
    with pytest.raises(ValueError, match="insufficient available"):
        resource_plan(configs, capacity, {"GPU": 1, "isaac": 2}, [capacity])
    with pytest.raises(ValueError, match="no node can place"):
        resource_plan(configs, capacity, capacity, [{"GPU": 2}, {"isaac": 2}])


def test_group_size_is_independent_of_global_session_count():
    config = TinkerConfig.load("examples/tinker/stage1.yaml", ["max_sessions=1"])
    assert config.group_size == 8
    assert replace(config, train_inflight_requests=1).signature() == config.signature()
    assert replace(config, prefetch_groups=0).signature() == config.signature()
    for value in (0, 2, None):
        replace(config, prefetch_groups=value).validate()
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="prefetch_groups"):
            replace(config, prefetch_groups=value).validate()


@pytest.mark.asyncio
@pytest.mark.parametrize("age_days", [0, 30])
async def test_billing_reconciliation_uses_hour_boundaries_and_only_this_runs_sessions(tmp_path, monkeypatch, age_days):
    usage = Usage(tmp_path)
    usage.record("sample_completed", {"generated_tokens": 7})
    hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    usage.identity["created_at"] = (hour - timedelta(days=age_days)).isoformat()
    (tmp_path / "usage_identity.json").write_text(json.dumps(usage.identity))
    response = tinker.types.BillingUsageResponse(
        data=[
            {
                "bucket_start": hour,
                "bucket_end": hour + timedelta(hours=1),
                "session_id": session,
                "user_name": "private account name",
                "estimated_cost_usd": tokens / 1000000,
                "event_info": {"type": "sampling_sample", "token_count": tokens},
            }
            for session, tokens in [("ours", 10), ("unrelated", 10000)]
        ],
        sessions={
            "ours": {"user_metadata": {"pear_run_id": usage.identity["run_id"]}},
            "unrelated": {"user_metadata": {"pear_run_id": "another run"}},
        },
    )

    windows = []

    class Admin:
        async def get_billing_usage(self, start, end):
            assert start.minute == end.minute == start.second == end.second == 0
            assert timedelta(0) < end - start <= timedelta(days=14)
            if windows:
                assert start == windows[-1][1]
            windows.append((start, end))
            return response if start <= hour < end else tinker.types.BillingUsageResponse(data=[], sessions={})

    monkeypatch.setattr(tinker, "AdminClient", Admin)
    report = await reconcile(tmp_path)
    assert len(windows) == (3 if age_days else 1)
    assert windows[0][0].date() == (hour - timedelta(days=age_days)).date()
    assert windows[-1][1] == hour + timedelta(hours=1)
    assert report["matched_sessions"] == 1
    assert report["service"]["generated_tokens"] == 10
    assert report["service_minus_local"]["generated_tokens"] == 3
    assert report["estimated_cost_usd"] == 10 / 1000000
    saved = (tmp_path / "billing_reconciliation.json").read_text()
    assert "unrelated" not in saved and "private account name" not in saved
