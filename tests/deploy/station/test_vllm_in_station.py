import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest

from pear.deploy.station.agent_loop.clients.client import ContextLimit
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput, ReferenceImage
from pear.deploy.station.agent_loop.clients.vllm import VLLMClient
from pear.deploy.station.exceptions import OperatorPaused

TIMEOUT_S = 300.0  # how long the model may take to answer a turn: a model's default in a station's configuration


class SentRequest:
    """A request the in-process server received."""

    @staticmethod
    def images(request):
        return [
            part
            for message in request["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]


@pytest.fixture
def service():
    """An in-process OpenAI-compatible server for the vLLM client: its URL, the requests it received and the modes
    queued for its next answers."""
    calls, modes = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "vllm", "object": "model", "created": 0, "owned_by": "test"}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            mode = modes.pop(0) if modes else "ok"
            if isinstance(mode, tuple) or mode in ("cache", "network", "context"):
                # a pair is a refusal of the test's own: its status and its message
                status, message = (
                    mode
                    if isinstance(mode, tuple)
                    else (
                        500 if mode == "network" else 400,
                        {
                            "cache": "Cache miss for image x and data is not provided",
                            "network": "unavailable",
                            "context": "maximum context length exceeded",
                        }[mode],
                    )
                )
                self.send_response(status)
                body = json.dumps({"error": {"message": message}}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            def chunk(delta, finish=None):
                return {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "vllm",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            try:
                if mode == "stall":
                    time.sleep(1)
                thinking = chunk({"reasoning": "visible thinking"})
                if mode == "live_usage":
                    thinking["usage"] = {
                        "prompt_tokens": 100,
                        "completion_tokens": 7,
                        "total_tokens": 107,
                    }
                send(thinking)
                if mode == "live_usage":
                    time.sleep(0.3)
                    update = chunk({"reasoning": " more thinking"})
                    update["usage"] = {
                        "prompt_tokens": 100,
                        "completion_tokens": 13,
                        "total_tokens": 113,
                    }
                    send(update)
                    time.sleep(0.1)
                if mode == "thinking_stall":
                    time.sleep(1)
                if mode == "late_cache":
                    send({"error": "Cache miss for image x and data is not provided"})
                    return
                if mode == "partial":
                    return
                send(chunk({"content": "complete answer"}, "length" if mode == "length" else "stop"))
                if mode == "no_usage":
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                send(
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "vllm",
                        "choices": [],
                        "metrics": {
                            "generation_time_ms": 190,
                            "tokens_per_second": 25,
                        }
                        if mode == "live_usage"
                        else None,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "prompt_tokens_details": {"cached_tokens": 80},
                        },
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, modes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.fixture
def inputs(tmp_path):
    def make(number):
        frames = []
        for role in ("head", "left_wrist", "right_wrist"):
            path = tmp_path / f"{number}-{role}.jpg"
            path.write_bytes(b"jpeg" + str(number).encode())
            frames.append(CameraImage(role, path))
        return ModelInput(f"observation {number}", tuple(frames))

    return make


def test_three_turns_keep_the_history_the_thinking_the_image_uuids_and_the_usage(service, inputs):
    url, calls, _ = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    assert client.check()["status"] == "ok"
    events = []

    for i in range(3):
        result = client.infer(inputs(i), on_event=lambda k, d: events.append((k, d)), cancellation=threading.Event())
        assert result.answer == "complete answer" and result.reasoning == "visible thinking"
        assert result.usage["cached_input"] == 80

    assert [len(SentRequest.images(c)) for c in calls] == [3, 6, 9]
    assert [sum(im["image_url"] is not None for im in SentRequest.images(c)) for c in calls] == [3, 3, 3]
    assert [im["uuid"] for im in SentRequest.images(calls[0])] == [
        im["uuid"] for im in SentRequest.images(calls[2])[:3]
    ]
    assert calls[2]["messages"][2] == {
        "role": "assistant",
        "reasoning": "visible thinking",
        "content": "complete answer",
    }
    assert calls[0]["chat_template_kwargs"]["preserve_thinking"] is True
    assert calls[0]["temperature"] == 1 and calls[0]["top_k"] == -1
    assert [d for k, d in events if k == "performance"][-1]["output_tokens"] == 20
    client.close()
    assert not client.history


def test_native_usage_updates_before_answer_and_final_totals_are_not_added(service, inputs):
    url, calls, modes = service
    modes.append("live_usage")
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    events = []

    try:
        result = client.infer(inputs(0), on_event=lambda k, d: events.append((k, d)), cancellation=threading.Event())

        assert calls[0]["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
        before_answer = events[: next(i for i, (k, _) in enumerate(events) if k == "answer_delta")]
        assert [d["output"] for k, d in before_answer if k == "usage"] == [7, 13]
        live = [d for k, d in before_answer if k == "performance"]
        assert live[-1]["output_tokens"] == 13
        assert live[-1]["client_decode_throughput_tps"] > 0
        assert live[0]["client_decode_throughput_tps"] is None
        assert all(not d["final"] and d["status"] == "streaming" for d in live)
        for value in live:
            assert value["output_throughput_tps"] == pytest.approx(value["output_tokens"] / value["elapsed_s"])
        assert result.usage["output"] == 20
        assert [d for k, d in events if k == "usage"][-1]["final"] is True
        assert events[-1][1]["final"] is True and events[-1][1]["output_tokens"] == 20
        assert events[-1][1]["decode_time_s"] == 0.19
        assert events[-1][1]["decode_throughput_tps"] == 100
    finally:
        client.close()


def test_missing_usage_does_not_infer_tokens_from_stream(service, inputs):
    url, _, modes = service
    modes.append("no_usage")
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    events = []

    try:
        result = client.infer(inputs(0), on_event=lambda k, d: events.append((k, d)), cancellation=threading.Event())

        assert result.answer and result.reasoning and result.usage == {}
        assert not [d for k, d in events if k == "usage"]
        for kind, value in events:
            if kind == "performance":
                assert value["output_tokens"] is None
                assert value["output_throughput_tps"] is None
                assert value["client_decode_throughput_tps"] is None
    finally:
        client.close()


@pytest.mark.parametrize(
    "modes,expected,success",
    [
        (["cache", "ok"], 2, True),
        (["cache", "cache"], 2, False),
        (["network"], 1, False),
        (["late_cache"], 1, False),
        (["partial"], 1, False),
        (["length"], 1, False),
        (["context"], 1, False),
    ],
)
def test_cache_recovery_happens_only_once_and_only_before_output(service, inputs, modes, expected, success):
    url, calls, pending = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())
    pending.extend(modes)
    before = len(client.history)

    if success:
        client.infer(inputs(1), on_event=lambda *a: None, cancellation=threading.Event())
    else:
        with pytest.raises(Exception):  # noqa: B017 - the failure modes raise SDK, HTTP and context errors alike
            client.infer(inputs(1), on_event=lambda *a: None, cancellation=threading.Event())
        assert len(client.history) == before

    assert len(calls) == expected + 1
    if expected == 2:
        assert [im["uuid"] for im in SentRequest.images(calls[1])] == [
            im["uuid"] for im in SentRequest.images(calls[2])
        ]
        assert all(im["image_url"] for im in SentRequest.images(calls[2]))
    client.close()


@pytest.mark.parametrize("max_output_tokens", [None, 4096])
def test_a_turn_has_no_token_limit_but_the_one_a_station_gives(service, inputs, max_output_tokens):
    url, calls, _ = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm", max_output_tokens=max_output_tokens)

    client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())

    assert calls[-1].get("max_tokens") == max_output_tokens
    client.close()


def test_an_answer_cut_where_the_context_ends_is_a_full_context(service, inputs):
    url, _, pending = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    pending.append("length")

    with pytest.raises(ContextLimit, match="context is full"):
        client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())

    assert len(client.history) == 1
    client.close()


def test_an_answer_cut_at_a_station_s_token_limit_is_an_unfinished_answer(service, inputs):
    url, _, pending = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm", max_output_tokens=4096)
    pending.append("length")

    with pytest.raises(ValueError, match="did not finish normally: length") as error:
        client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())

    assert type(error.value) is ValueError
    client.close()


def test_cancel_before_first_output_leaves_prior_history(service, inputs):
    url, calls, modes = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    modes.append("stall")
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()

    try:
        with pytest.raises(OperatorPaused):
            client.infer(inputs(0), on_event=lambda *a: None, cancellation=cancel)

        assert len(client.history) == 1 and len(calls) == 1
    finally:
        timer.join()
        client.close()


def test_image_limit_pauses_before_sending(service, inputs):
    url, calls, _ = service
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm", max_images=2)

    with pytest.raises(ContextLimit):
        client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())

    assert not calls
    client.close()


def test_cancellation_during_thinking_keeps_partial_output_out_of_history(service, inputs):
    url, calls, modes = service
    modes.append("thinking_stall")
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    cancel = threading.Event()
    events = []

    def emit(kind, data):
        events.append(kind)
        if kind == "reasoning_delta":
            cancel.set()

    try:
        with pytest.raises(OperatorPaused):
            client.infer(inputs(0), on_event=emit, cancellation=cancel)

        assert "reasoning_delta" in events and "answer_delta" not in events
        assert len(calls) == 1 and len(client.history) == 1
    finally:
        client.close()


def test_cancellation_after_cache_miss_prevents_reupload(service, inputs):
    url, calls, modes = service
    modes.append("cache")
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")
    cancel = threading.Event()

    def emit(kind, data):
        if kind == "notice":
            cancel.set()

    try:
        with pytest.raises(OperatorPaused):
            client.infer(inputs(0), on_event=emit, cancellation=cancel)

        assert len(calls) == 1 and len(client.history) == 1
    finally:
        client.close()


def test_reference_images_are_unlabelled_once_and_survive_cache_recovery(service, inputs, tmp_path):
    url, calls, modes = service
    ref = tmp_path / "goal.jpg"
    ref.write_bytes(b"goal jpeg")
    refs = (ReferenceImage(ref), ReferenceImage(ref))
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")

    try:
        for n in range(3):
            if n == 1:
                modes.extend(["cache", "ok"])
            client.infer(
                replace(inputs(n), reference_images=refs),
                on_event=lambda *a: None,
                cancellation=threading.Event(),
            )

        users = [m for m in calls[-1]["messages"] if m["role"] == "user"]
        assert [p["type"] for p in users[0]["content"][:3]] == ["image_url", "image_url", "text"]
        assert users[0]["content"][2]["text"] == "observation 0"
        for user in users[1:]:
            assert user["content"][0]["type"] == "text"
        ids = [p["uuid"] for u in users for p in u["content"] if p["type"] == "image_url"]
        assert all(ids.count(im.uuid) == 1 for im in refs)
        restored = calls[2]["messages"][1]["content"]
        assert restored[0]["image_url"]["url"].endswith("Z29hbCBqcGVn")
        assert users[0]["content"][0]["image_url"] is None
    finally:
        client.close()


def test_failed_first_request_retries_reference_images(service, inputs, tmp_path):
    url, calls, modes = service
    ref = tmp_path / "goal.jpg"
    ref.write_bytes(b"goal")
    item = replace(inputs(0), reference_images=(ReferenceImage(ref),))
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")

    try:
        modes.append("context")
        with pytest.raises(ContextLimit):
            client.infer(item, on_event=lambda *a: None, cancellation=threading.Event())
        client.infer(item, on_event=lambda *a: None, cancellation=threading.Event())

        assert all(c["messages"][-1]["content"][0]["uuid"] == item.reference_images[0].uuid for c in calls)
    finally:
        client.close()


@pytest.mark.parametrize("text", ["Maximum context length", "max_model_len", "context length exceeded"])
def test_a_vllm_bad_request_with_a_marker_is_an_overflow(service, inputs, text):
    url, _, modes = service
    modes.append((400, text))
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")

    try:
        with pytest.raises(ContextLimit):
            client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())
    finally:
        client.close()


def test_other_status_codes_are_not_an_overflow(service, inputs):
    url, _, modes = service
    modes.append((500, "maximum context length"))
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")

    try:
        with pytest.raises(openai.APIStatusError) as error:
            client.infer(inputs(0), on_event=lambda *a: None, cancellation=threading.Event())

        assert error.value.status_code == 500
    finally:
        client.close()


def test_the_data_url_of_a_sent_image_names_a_jpeg(service, inputs):
    url, calls, _ = service
    item = inputs(0)
    client = VLLMClient("system", timeout_s=TIMEOUT_S, base_url=url, model="vllm")

    try:
        client.infer(item, on_event=lambda *a: None, cancellation=threading.Event())

        sent = [im["image_url"]["url"] for im in SentRequest.images(calls[0])]
        assert sent == ["data:image/jpeg;base64," + im.base64() for im in item.images]
    finally:
        client.close()


@pytest.mark.parametrize(
    "argument,message",
    [
        ({"base_url": "ftp://server/v1"}, "base_url must be HTTP"),
        ({"base_url": "http://name:word@server/v1"}, "without embedded credentials"),
        ({"model": ""}, "model must be the name"),
        ({"max_output_tokens": 0}, "max_output_tokens must be a whole number above 0"),
        ({"context_limit": 1.5}, "context_limit must be a whole number above 0"),
        ({"max_images": True}, "max_images must be a whole number above 0"),
        ({"temperature": 2.5}, "temperature must be a number from 0 to 2"),
        ({"top_p": 0}, "top_p must be a number above 0 and at most 1"),
        ({"top_k": 1.5}, "top_k must be a whole number"),
        ({"repetition_penalty": float("inf")}, "repetition_penalty must be a finite number above 0"),
        ({"reasoning_effort": "low"}, "reasoning_effort must be xhigh"),
    ],
)
def test_the_client_refuses_an_argument_outside_what_it_takes(argument, message):
    with pytest.raises(ValueError, match=message):
        VLLMClient("system", timeout_s=TIMEOUT_S, **{"base_url": "http://server/v1", "model": "vllm", **argument})
