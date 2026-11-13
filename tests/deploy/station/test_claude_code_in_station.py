import asyncio
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from pear.deploy.station.agent_loop.clients.claude_code import ClaudeCodeClient
from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, SessionLost
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput, ReferenceImage
from pear.deploy.station.exceptions import OperatorPaused

TIMEOUT_S = 300.0  # how long the model may take to answer a turn: a model's default in a station's configuration


@pytest.mark.parametrize("mode", ["completed", "pause", "disconnect", "disconnect_usage", "identity", "timeout"])
def test_a_turn_ends_with_its_answer_with_a_pause_or_with_a_lost_session(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(ModelClient, "CLEANUP_TIMEOUT_S", 1.2)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8")
    observation = ModelInput(
        "actual feedback: previous action was not executed",
        tuple(CameraImage(role, frame) for role in ("head", "left_wrist", "right_wrist")),
    )
    events, creates, turns, deletes, interrupts = [], [], [], [], []
    cancel = threading.Event()

    async def run():
        gate = asyncio.Event()

        async def health(request):
            return web.json_response(
                {"service": "pear-station-claude-code", "protocol": 1, "model": "vllm", "status": "ok"}
            )

        async def create(request):
            value = await request.json()
            creates.append(value)
            return web.json_response({"session_id": value["session_id"], "pid": 123})

        async def turn(request):
            value = await request.json()
            turns.append(value)
            response = web.StreamResponse()
            await response.prepare(request)
            if mode == "disconnect":
                await response.write_eof()
                return response

            async def emit(kind, data):
                sid = request.match_info["sid"]
                event = {
                    "session_id": "wrong" if mode == "identity" else sid,
                    "turn_id": value["turn_id"],
                    "type": kind,
                    "data": data,
                }
                await response.write((json.dumps(event) + "\n").encode())

            if mode == "disconnect_usage":
                await emit("usage", {"input": 100, "output": 17, "final": False, "partial": True})
                await response.write_eof()
                return response
            await emit("phase", {"phase": "Thinking"})
            if mode in ("pause", "timeout"):
                await gate.wait()
            if mode == "identity":
                return response
            await emit("answer_delta", {"text": "stale answer" if cancel.is_set() else "hold"})
            await emit(
                "usage",
                {
                    "input": 29137,
                    "output": 2498,
                    "final": True,
                    "partial": cancel.is_set(),
                    "context": {"input": 3825, "context_limit": 260096},
                },
            )
            await emit(
                "result",
                {
                    "status": "interrupted" if cancel.is_set() else "completed",
                    "answer": "hold",
                    "reasoning": "",
                },
            )
            return response

        async def interrupt(request):
            interrupts.append(request.match_info["tid"])
            if mode == "timeout":
                await asyncio.sleep(0.5)
            gate.set()
            return web.json_response({"status": "stopped"})

        async def delete(request):
            deletes.append(request.match_info["sid"])
            return web.json_response({"closed": True})

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/sessions", create)
        app.router.add_post("/sessions/{sid}/turns", turn)
        app.router.add_post("/sessions/{sid}/turns/{tid}/interrupt", interrupt)
        app.router.add_delete("/sessions/{sid}", delete)
        async with TestServer(app) as server:
            client = ClaudeCodeClient(
                "system", timeout_s=3, base_url=str(server.make_url("")).rstrip("/"), model="vllm"
            )
            assert (await asyncio.to_thread(client.check))["status"] == "ok"
            assert not creates

            def emit(kind, value):
                events.append((kind, value))
                if kind == "phase" and mode in ("pause", "timeout"):
                    cancel.set()
                    client.cancel()

            async def infer():
                return await asyncio.to_thread(client.infer, observation, on_event=emit, cancellation=cancel)

            if mode in ("disconnect", "disconnect_usage", "identity", "timeout"):
                with pytest.raises(SessionLost):
                    await infer()
                with pytest.raises(SessionLost):
                    await infer()
            elif mode == "pause":
                with pytest.raises(OperatorPaused):
                    await infer()
                assert not any(kind == "answer_delta" for kind, _ in events)
                assert any(kind == "usage" and data["final"] for kind, data in events)
                final = [data for kind, data in events if kind == "performance" and data["final"]]
                assert len(final) == 1 and final[0]["output_tokens"] == 2498
                assert final[0]["partial"] and final[0]["status"] == "interrupted"
                assert len(interrupts) == 1
            else:
                assert (await infer()).answer == "hold"
                assert (await infer()).answer == "hold"
                assert len(creates) == 1
                assert turns[0]["turn_id"] != turns[1]["turn_id"]
                assert all(t["text"] == observation.text for t in turns)
            if mode == "disconnect_usage":
                final_usage = [data for kind, data in events if kind == "usage" and data["final"]]
                final_perf = [data for kind, data in events if kind == "performance" and data["final"]]
                assert len(final_usage) == 1 and final_usage[0]["output"] == 17
                assert final_usage[0]["partial"]
                assert len(final_perf) == 1 and final_perf[0]["output_tokens"] == 17
                assert final_perf[0]["partial"] and final_perf[0]["status"] == "failed"
            gate.set()
            await asyncio.to_thread(client.close)
            assert deletes == [client.session_id]

    asyncio.run(run())


def test_cancel_before_infer_does_not_create_a_session():
    client = ClaudeCodeClient("system", timeout_s=TIMEOUT_S, base_url="http://127.0.0.1:1", model="vllm")
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(OperatorPaused):
        client.infer(ModelInput("", ()), on_event=lambda *args: None, cancellation=cancel)

    client.close()


@pytest.mark.parametrize("interrupted", [False, True])
def test_live_metrics_replace_snapshots_across_requests(tmp_path, interrupted):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8")
    observation = ModelInput(
        "recorded only", tuple(CameraImage(r, frame) for r in ("head", "left_wrist", "right_wrist"))
    )
    events = []

    async def run():
        async def create(request):
            return web.json_response({"session_id": (await request.json())["session_id"]})

        async def turn(request):
            body = await request.json()
            response = web.StreamResponse()
            await response.prepare(request)

            async def emit(kind, data):
                await response.write(
                    (
                        json.dumps(
                            {
                                "session_id": request.match_info["sid"],
                                "turn_id": body["turn_id"],
                                "type": kind,
                                "data": data,
                            }
                        )
                        + "\n"
                    ).encode()
                )

            await emit("reasoning_delta", {"text": "Thinking"})
            for first, second in [(8, None), (17, None), (17, None), (20, 0), (20, 5), (20, 10)]:
                rows = [
                    {
                        "id": "main",
                        "output": first,
                        "complete": second is not None,
                        "generation_time_ms": 30,
                    }
                ]
                if second is not None:
                    rows.append(
                        {
                            "id": "continuation",
                            "output": second,
                            "complete": False,
                            "generation_time_ms": 10,
                        }
                    )
                await emit(
                    "usage",
                    {
                        "input": 200,
                        "output": first + (second or 0),
                        "partial": True,
                        "final": False,
                        "requests": rows,
                    },
                )
                await asyncio.sleep(0.03)
            rows[-1]["complete"] = not interrupted
            await emit(
                "usage",
                {
                    "input": 200,
                    "output": 30,
                    "partial": interrupted,
                    "final": True,
                    "requests": rows,
                },
            )
            await emit(
                "result",
                {"status": "interrupted" if interrupted else "completed", "answer": "done"},
            )
            return response

        async def delete(request):
            return web.json_response({"closed": True})

        app = web.Application()
        app.router.add_post("/sessions", create)
        app.router.add_post("/sessions/{sid}/turns", turn)
        app.router.add_delete("/sessions/{sid}", delete)
        async with TestServer(app) as server:
            client = ClaudeCodeClient(
                "system", timeout_s=TIMEOUT_S, base_url=str(server.make_url("")).rstrip("/"), model="vllm"
            )

            def infer():
                return client.infer(
                    observation,
                    on_event=lambda k, d: events.append((k, d)),
                    cancellation=threading.Event(),
                )

            if interrupted:
                with pytest.raises(OperatorPaused):
                    await asyncio.to_thread(infer)
            else:
                assert (await asyncio.to_thread(infer)).usage["output"] == 30
            await asyncio.to_thread(client.close)

    asyncio.run(run())

    live = [d for k, d in events if k == "performance" and not d["final"] and d["output_tokens"] is not None]
    assert [p["output_tokens"] for p in live[:6]] == [8, 17, 17, 20, 25, 30]
    assert live[1]["client_decode_throughput_tps"] > 0
    assert live[2]["client_decode_throughput_tps"] == live[1]["client_decode_throughput_tps"]
    assert all(p["output_throughput_tps"] > 0 for p in live)
    final = [d for k, d in events if k == "performance" and d["final"]]
    assert len(final) == 1 and final[0]["output_tokens"] == 30
    assert final[0]["partial"] is interrupted
    assert final[0]["client_decode_throughput_tps"] > 0
    assert final[0]["decode_throughput_tps"] == (None if interrupted else 700)


def test_episode_compactions_survive_turns_interruptions_and_duplicate_snapshots():
    async def run():
        calls = {}
        scenarios = [
            (
                [
                    (0, "none"),
                    (1, "completed"),
                    (1, "completed"),
                    (2, "completed"),
                    (2, "running"),
                    (2, "interrupted"),
                ],
                "interrupted",
            ),
            ([(0, "running"), (0, "failed")], "completed"),
            ([(0, "none"), (0, "none")], "completed"),
            ([(0, "running"), (1, "completed"), (1, "completed")], "completed"),
        ]

        async def create(request):
            return web.json_response({"session_id": (await request.json())["session_id"]})

        async def turn(request):
            sid = request.match_info["sid"]
            body = await request.json()
            index = calls.get(sid, 0)
            calls[sid] = index + 1
            snapshots, status = scenarios[index]
            response = web.StreamResponse()
            await response.prepare(request)

            async def emit(kind, data):
                await response.write(
                    (
                        json.dumps(
                            {
                                "session_id": sid,
                                "turn_id": body["turn_id"],
                                "type": kind,
                                "data": data,
                            }
                        )
                        + "\n"
                    ).encode()
                )

            for i, (count, phase) in enumerate(snapshots):
                await emit(
                    "usage",
                    {
                        "input": 100,
                        "output": 10,
                        "final": i == len(snapshots) - 1,
                        "compaction": {"status": phase, "count": count},
                    },
                )
            await emit("result", {"status": status, "answer": "done"})
            return response

        async def delete(request):
            return web.json_response({"closed": True})

        app = web.Application()
        app.router.add_post("/sessions", create)
        app.router.add_post("/sessions/{sid}/turns", turn)
        app.router.add_delete("/sessions/{sid}", delete)
        async with TestServer(app) as server:

            def new_client():
                return ClaudeCodeClient("system", timeout_s=TIMEOUT_S, base_url=str(server.make_url("")), model="vllm")

            client = new_client()
            observed = []

            def infer():
                return client.infer(
                    ModelInput("recorded observation", ()),
                    on_event=lambda k, d: observed.append(d["compaction"]) if k == "usage" else None,
                    cancellation=threading.Event(),
                )

            with pytest.raises(OperatorPaused):
                await asyncio.to_thread(infer)
            assert [d["episode_count"] for d in observed] == [0, 1, 1, 2, 2, 2]
            frozen = dict(observed[-1])
            for expected in (2, 2, 3):
                output = await asyncio.to_thread(infer)
                assert output.usage["compaction"]["episode_count"] == expected
            assert observed[5] == frozen  # Later turns do not mutate saved snapshots.
            await asyncio.to_thread(client.close)
            client = new_client()
            observed.clear()
            with pytest.raises(OperatorPaused):
                await asyncio.to_thread(infer)
            assert observed[0]["episode_count"] == 0  # New episode, not a global counter.
            await asyncio.to_thread(client.close)

    asyncio.run(run())


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "interrupted",
        "compacted",
        "old_service",
        "recovered",
        "unverified",
        "recovery_paused",
    ],
)
def test_reference_images_are_sent_once_and_block_the_task_when_a_compaction_loses_them(tmp_path, scenario):
    path = tmp_path / "goal.jpg"
    path.write_bytes(b"\xff\xd8goal")
    observation = ModelInput("camera observation", (), (ReferenceImage(path),))
    calls = []

    async def run():
        async def health(request):
            return web.json_response(
                {
                    "service": "pear-station-claude-code",
                    "protocol": 1,
                    "model": "vllm",
                    "status": "ok",
                    "capabilities": {
                        "reference_images": scenario != "old_service",
                        "reference_image_recovery": scenario in ("recovered", "unverified", "recovery_paused"),
                    },
                }
            )

        async def create(request):
            return web.json_response({"session_id": (await request.json())["session_id"]})

        async def turn(request):
            body = await request.json()
            calls.append(body)
            response = web.StreamResponse()
            await response.prepare(request)
            events = []
            if scenario in ("compacted", "recovered", "unverified", "recovery_paused"):
                events.append(("usage", {"compaction": {"count": 1, "status": "completed"}}))
            events.append(
                (
                    "result",
                    {
                        "status": "interrupted"
                        if scenario in ("interrupted", "recovery_paused") and len(calls) == 1
                        else "completed",
                        "answer": "hold",
                        "reference_context": {
                            "status": "recovery_required"
                            if scenario == "recovery_paused" and len(calls) == 1
                            else "ready",
                            "image_count": 1,
                            "verified_request_id": None if scenario == "unverified" else "verified-request",
                        },
                    },
                )
            )
            for kind, data in events:
                await response.write(
                    (
                        json.dumps(
                            {
                                "session_id": request.match_info["sid"],
                                "turn_id": body["turn_id"],
                                "type": kind,
                                "data": data,
                            }
                        )
                        + "\n"
                    ).encode()
                )
            return response

        async def delete(request):
            return web.json_response({"closed": True})

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/sessions", create)
        app.router.add_post("/sessions/{sid}/turns", turn)
        app.router.add_delete("/sessions/{sid}", delete)
        async with TestServer(app) as server:
            client = ClaudeCodeClient("system", timeout_s=TIMEOUT_S, base_url=str(server.make_url("")), model="vllm")

            async def infer():
                return await asyncio.to_thread(
                    client.infer, observation, on_event=lambda *a: None, cancellation=threading.Event()
                )

            try:
                if scenario == "old_service":
                    with pytest.raises(ValueError, match="does not support"):
                        await infer()
                    assert not calls
                elif scenario in ("compacted", "unverified"):
                    for _ in range(2):
                        with pytest.raises(ContextLimit):
                            await infer()
                    assert len(calls) == 1
                else:
                    if scenario in ("interrupted", "recovery_paused"):
                        with pytest.raises(OperatorPaused):
                            await infer()
                    else:
                        assert (await infer()).answer == "hold"
                    assert (await infer()).answer == "hold"
                    assert len(calls[0]["reference_images"]) == 1
                    assert "reference_images" not in calls[1]
                    assert all(c["text"] == observation.text for c in calls)
            finally:
                await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_a_failed_turn_whose_error_mentions_context_is_an_overflow(tmp_path):
    path = tmp_path / "goal.jpg"
    path.write_bytes(b"\xff\xd8goal")
    observation = ModelInput("camera observation", (), (ReferenceImage(path),))

    async def run(message):
        async def health(request):
            return web.json_response(
                {
                    "service": "pear-station-claude-code",
                    "protocol": 1,
                    "model": "vllm",
                    "status": "ok",
                    "capabilities": {"reference_images": True},
                }
            )

        async def create(request):
            return web.json_response({"session_id": (await request.json())["session_id"]})

        async def turn(request):
            body = await request.json()
            response = web.StreamResponse()
            await response.prepare(request)
            event = {
                "session_id": request.match_info["sid"],
                "turn_id": body["turn_id"],
                "type": "result",
                "data": {"status": "failed", "error": message},
            }
            await response.write((json.dumps(event) + "\n").encode())
            return response

        async def delete(request):
            return web.json_response({"closed": True})

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/sessions", create)
        app.router.add_post("/sessions/{sid}/turns", turn)
        app.router.add_delete("/sessions/{sid}", delete)
        async with TestServer(app) as server:
            client = ClaudeCodeClient("system", timeout_s=TIMEOUT_S, base_url=str(server.make_url("")), model="vllm")

            try:
                with pytest.raises(ValueError, match=message) as failure:
                    await asyncio.to_thread(
                        client.infer, observation, on_event=lambda *a: None, cancellation=threading.Event()
                    )
                return failure.value
            finally:
                await asyncio.to_thread(client.close)

    assert isinstance(asyncio.run(run("The Context window is full")), ContextLimit)
    assert not isinstance(asyncio.run(run("upstream unavailable")), ContextLimit)


@pytest.mark.parametrize(
    "argument,message",
    [
        ({"base_url": "server:8000"}, "base_url must be HTTP"),
        ({"base_url": "http://name:word@server"}, "without embedded credentials"),
        ({"model": ""}, "model must be the name"),
    ],
)
def test_the_client_refuses_a_service_url_or_a_model_it_cannot_use(argument, message):
    with pytest.raises(ValueError, match=message):
        ClaudeCodeClient("system", timeout_s=TIMEOUT_S, **{"base_url": "http://server", "model": "vllm", **argument})
