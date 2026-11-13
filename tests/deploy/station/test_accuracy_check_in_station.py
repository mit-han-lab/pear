import numpy as np
import pytest

from pear.deploy.station.control.commands import AccuracyCheck


@pytest.mark.parametrize(
    "errors, passed",
    [
        ([1.0, 2.0, 3.0, 3.0, 3.0], True),  # the median exactly at its limit
        ([3.01, 3.01, 3.01], False),
        ([1.0, 1.0, 1.0, 6.0], True),  # the worst pose exactly at its limit
        ([1.0, 1.0, 1.0, 6.01], False),
    ],
)
def test_the_verdict_holds_the_median_and_the_worst_pose_to_their_limits(errors, passed):
    verdict = AccuracyCheck.verdict(errors)

    assert verdict["passed"] is passed
    assert verdict["median_mm"] == pytest.approx(float(np.median(errors)))
    assert verdict["max_mm"] == pytest.approx(max(errors))
