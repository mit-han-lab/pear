import asyncio
from collections.abc import Callable

import aiohttp
from prometheus_client.parser import text_string_to_metric_families


class EngineMetrics:
    """The model server's own counts of the one request it runs, read from its Prometheus metrics: the server runs one
    sequence at a time, so what its counters gain while a request streams is that request's."""

    OUTPUT_TOKENS = "vllm:generation_tokens_total"  # the tokens the engine has generated
    DECODE_SECONDS = "vllm:request_decode_time_seconds_sum"  # the decode intervals of the requests that finished
    FINISHED_REQUESTS = "vllm:request_decode_time_seconds_count"  # the requests whose interval is in that sum
    RUNNING_REQUESTS = "vllm:num_requests_running"
    # readings, one interval apart, that wait for a finished request's decode interval to reach the metrics
    SETTLE_READINGS = 4
    MILLISECONDS_PER_SECOND = 1000

    def __init__(self, http_session: aiohttp.ClientSession, metrics_url: str, interval_s: float) -> None:
        self.http_session = http_session
        self.metrics_url = metrics_url
        self.interval_s = interval_s  # between two readings while a request streams

    async def read(self) -> dict[str, float]:
        """Returns the server's samples by name, each summed over its labels."""
        async with self.http_session.get(self.metrics_url) as response:
            response.raise_for_status()
            text = await response.text()

        samples: dict[str, float] = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                samples[sample.name] = samples.get(sample.name, 0.0) + sample.value

        return samples

    async def follow(
        self, baseline: dict[str, float], on_output: Callable[[int], None], finished: asyncio.Event
    ) -> float | None:
        """Reports the request's generated tokens to ``on_output`` at every reading until ``finished`` is set, from
        the ``baseline`` read before the request; returns its decode interval in milliseconds, None when another
        request shared the server or the server's metrics could not be read."""
        is_shared = False
        reading = baseline

        try:
            while not finished.is_set():
                try:
                    await asyncio.wait_for(finished.wait(), self.interval_s)
                except TimeoutError:
                    pass
                reading = await self.read()
                is_shared |= reading[self.RUNNING_REQUESTS] > 1
                if not is_shared:
                    on_output(int(reading[self.OUTPUT_TOKENS] - baseline[self.OUTPUT_TOKENS]))

            for _ in range(self.SETTLE_READINGS):
                if reading[self.FINISHED_REQUESTS] > baseline[self.FINISHED_REQUESTS]:
                    break
                await asyncio.sleep(self.interval_s)
                reading = await self.read()
        except (aiohttp.ClientError, KeyError, ValueError):
            return None

        if is_shared or reading[self.FINISHED_REQUESTS] - baseline[self.FINISHED_REQUESTS] != 1:
            return None

        return (reading[self.DECODE_SECONDS] - baseline[self.DECODE_SECONDS]) * self.MILLISECONDS_PER_SECOND
