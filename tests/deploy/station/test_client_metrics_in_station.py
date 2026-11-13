import json
import math
import time

import pytest

from pear.deploy.station.agent_loop.clients.metrics import Performance, TokenRate


def test_performance_uses_only_provided_usage():
    value = Performance.report("vllm", time.monotonic() - 2, 0.5, {})

    assert value["output_tokens"] is None and value["output_throughput_tps"] is None
    assert value["first_visible_output_s"] == 0.5
    assert "decode_tps" not in value and "ttft_s" not in value
    json.dumps(value, allow_nan=False)


def test_one_request_decodes_all_tokens_but_the_first():
    value = Performance.decode_rate([(20, 190)])

    assert value == {"decode_throughput_tps": 100, "decode_time_s": 0.19}


def test_several_requests_add_their_decoded_tokens_and_times():
    value = Performance.decode_rate([(11, 100), (21, 300)])

    assert value["decode_time_s"] == 0.4
    assert value["decode_throughput_tps"] == pytest.approx(30 / 0.4)


@pytest.mark.parametrize("samples", [[], [(20, None)], [(20, 190), (1, 190)], [(20, 190), (20, 0)]])
def test_an_unusable_sample_gives_no_decode_rate(samples):
    assert Performance.decode_rate(samples) == {"decode_throughput_tps": None, "decode_time_s": None}


@pytest.mark.parametrize("milliseconds", [True, "190", -1, math.nan, math.inf])
def test_server_timing_must_be_a_finite_positive_number(milliseconds):
    assert Performance.decode_rate([(20, milliseconds)])["decode_throughput_tps"] is None


@pytest.mark.parametrize("duration", [None, True, "190", 0, -1, float("nan"), float("inf")])
def test_decode_requires_valid_server_timing(duration):
    result = Performance.decode_rate([(20, duration)])

    assert result["decode_throughput_tps"] is None


@pytest.mark.parametrize("tokens", [None, True, "20", 0, 1])
def test_decode_requires_multiple_reported_tokens(tokens):
    assert Performance.decode_rate([(tokens, 190)])["decode_throughput_tps"] is None


def test_the_token_rate_spans_the_first_to_the_last_report_of_a_request():
    rate = TokenRate()
    rate.add("a", 10, 1.0)
    rate.add("a", 30, 3.0)

    assert rate.tokens_per_second() == 10


def test_requests_are_measured_separately_and_summed_into_one_token_rate():
    rate = TokenRate()
    for request, count, received in (("a", 10, 0.0), ("a", 20, 1.0), ("b", 5, 5.0), ("b", 35, 7.0)):
        rate.add(request, count, received)

    assert rate.tokens_per_second() == pytest.approx(40 / 3)


@pytest.mark.parametrize("count", [None, 0, True, "5"])
def test_a_count_that_is_not_a_positive_integer_is_ignored(count):
    rate = TokenRate()
    rate.add("a", count, 1.0)

    assert rate.tokens_per_second() is None


def test_one_report_has_no_token_rate():
    rate = TokenRate()
    rate.add("a", 10, 1.0)

    assert rate.tokens_per_second() is None
