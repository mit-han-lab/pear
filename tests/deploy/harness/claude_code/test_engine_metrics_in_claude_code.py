import asyncio

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer
from service_rigs import StandInUpstream

from pear.deploy.harness.claude_code.engine_metrics import EngineMetrics

INTERVAL_S = 0.02  # between two readings


class MetricsRun:
    """One request's metrics followed while a test changes the stand-in server's counters."""

    def __init__(self, upstream: StandInUpstream) -> None:
        self.upstream = upstream
        self.outputs: list[int] = []

    async def follow(self, change) -> float | None:
        """Follows the metrics while ``change`` runs on the server's counters, and returns the decode interval."""
        async with TestServer(self.upstream.app()) as server, ClientSession() as http_session:
            metrics = EngineMetrics(http_session, str(server.make_url("/metrics")), INTERVAL_S)
            baseline = await metrics.read()
            finished = asyncio.Event()
            follower = asyncio.create_task(metrics.follow(baseline, self.outputs.append, finished))
            await change()
            finished.set()

            return await follower


def test_the_counters_gains_while_a_request_runs_are_its_output_and_its_decode_interval():
    upstream = StandInUpstream()
    upstream.generation_tokens, upstream.finished_requests, upstream.decode_seconds = 500, 3, 9.0
    run = MetricsRun(upstream)

    async def stream():
        for _ in range(5):
            upstream.generation_tokens += 10
            await asyncio.sleep(INTERVAL_S * 2)
        upstream.finished_requests += 1
        upstream.decode_seconds += 0.25

    assert asyncio.run(run.follow(stream)) == 250
    assert run.outputs == sorted(run.outputs) and 0 < run.outputs[0] and run.outputs[-1] == 50


def test_a_request_that_shared_the_server_has_no_decode_interval_and_no_live_output():
    upstream = StandInUpstream()
    run = MetricsRun(upstream)

    async def shared():
        upstream.running_requests = 2
        upstream.generation_tokens += 40
        await asyncio.sleep(INTERVAL_S * 3)
        upstream.finished_requests += 2

    assert asyncio.run(run.follow(shared)) is None
    assert run.outputs == []


def test_a_request_whose_interval_never_reached_the_metrics_has_none():
    upstream = StandInUpstream()

    async def unfinished():
        upstream.generation_tokens += 40
        await asyncio.sleep(INTERVAL_S * 2)

    assert asyncio.run(MetricsRun(upstream).follow(unfinished)) is None
