import asyncio
import base64
import threading
import time

import pytest

from pear.deploy.station.agent_loop.clients.client import ModelClient, SessionLost
from pear.deploy.station.agent_loop.clients.schemas import (
    CameraImage,
    HealthStatus,
    ModelInput,
    ModelOutput,
    ReferenceImage,
)
from pear.deploy.station.exceptions import OperatorPaused

SLACK_POLLS = 10  # a cancel must be noticed well within this many poll intervals


class HangingClient(ModelClient):
    """A client whose request never answers by itself."""

    LABEL = "Hanging"
    TIMEOUT_S = 30.0  # how long a request hangs when nothing stops it

    def __init__(self):
        super().__init__("system", timeout_s=self.TIMEOUT_S)
        self.started = threading.Event()

    def check(self):
        return {"status": HealthStatus.OK.value, "message": "Hanging client"}

    async def _infer_async(self, observation, on_event, cancel):
        self.started.set()
        return await self._race(asyncio.sleep(self.TIMEOUT_S), cancel, timeout_s=self.TIMEOUT_S)


class LoopClient(ModelClient):
    """A client that answers at once, remembers the event loop and the thread each request ran in, and fails its
    first ``failed_releases`` releases."""

    LABEL = "Loop"

    def __init__(self, failed_releases=0):
        super().__init__("system", timeout_s=HangingClient.TIMEOUT_S)
        self.failed_releases = failed_releases
        self.loops, self.thread_names, self.release_count = [], [], 0

    def check(self):
        return {"status": HealthStatus.OK.value, "message": "Loop client"}

    async def _infer_async(self, observation, on_event, cancel):
        self.loops.append(asyncio.get_running_loop())
        self.thread_names.append(threading.current_thread().name)
        return ModelOutput("answer")

    async def _release(self):
        self.release_count += 1
        if self.release_count <= self.failed_releases:
            raise RuntimeError("release failed")


class Request:
    """One infer call running in a thread, as the agent loop does."""

    def __init__(self, client, cancel=None):
        self.cancel = cancel or threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(client,), daemon=True)
        self.thread.start()
        assert client.started.wait(5)

    def _run(self, client):
        try:
            client.infer(ModelInput("observation", ()), on_event=lambda *args: None, cancellation=self.cancel)
        except BaseException as error:
            self.error = error


def test_a_second_infer_while_one_is_running_is_refused():
    client = HangingClient()
    request = Request(client)

    with pytest.raises(RuntimeError, match="already active"):
        client.infer(ModelInput("second", ()), on_event=lambda *args: None, cancellation=threading.Event())

    client.cancel()
    request.thread.join(5)
    assert isinstance(request.error, OperatorPaused)


def test_infer_after_close_is_refused():
    client = HangingClient()
    client.close()

    with pytest.raises(SessionLost, match="closed"):
        client.infer(ModelInput("observation", ()), on_event=lambda *args: None, cancellation=threading.Event())


def test_an_already_set_cancel_stops_before_any_request_starts():
    client = HangingClient()
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(OperatorPaused):
        client.infer(ModelInput("observation", ()), on_event=lambda *args: None, cancellation=cancel)

    assert not client.started.is_set()


@pytest.mark.parametrize("source", ["client", "event"])
def test_cancel_ends_an_in_flight_request_within_the_poll_interval(source):
    client = HangingClient()
    request = Request(client)
    begun = time.monotonic()

    if source == "client":
        client.cancel()
    else:
        request.cancel.set()
    request.thread.join(5)

    assert not request.thread.is_alive()
    assert time.monotonic() - begun < SLACK_POLLS * client.CANCEL_POLL_S
    assert isinstance(request.error, OperatorPaused)


def test_close_cancels_the_request_and_waits_for_it():
    client = HangingClient()
    request = Request(client)

    client.close()

    assert not request.thread.is_alive()
    assert isinstance(request.error, OperatorPaused)


def test_turns_asked_from_several_threads_run_in_the_client_s_one_event_loop():
    client = LoopClient()

    for _ in range(3):  # one after another, each from a thread of its own, as the agent loop asks
        thread = threading.Thread(
            target=client.infer,
            args=(ModelInput("observation", ()),),
            kwargs={"on_event": lambda *args: None, "cancellation": threading.Event()},
        )
        thread.start()
        thread.join(5)

    assert len(client.loops) == 3 and len({id(loop) for loop in client.loops}) == 1
    assert set(client.thread_names) == {ModelClient.LOOP_THREAD_NAME}
    client.close()
    assert not client._loop_thread.is_alive()


def test_a_failed_release_keeps_the_event_loop_for_a_later_close():
    client = LoopClient(failed_releases=1)

    with pytest.raises(RuntimeError, match="release failed"):
        client.close()

    assert client._loop_thread.is_alive()
    client.close()
    assert not client._loop_thread.is_alive() and client.release_count == 2


def test_the_file_bytes_are_ascii_base64(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"\xff\xd8\xff")

    assert CameraImage("head", path).base64() == base64.b64encode(b"\xff\xd8\xff").decode("ascii")
    assert ReferenceImage(path).base64() == base64.b64encode(b"\xff\xd8\xff").decode("ascii")
