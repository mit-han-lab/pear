import json
import threading
from collections.abc import Iterator
from dataclasses import replace

import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.stability import StabilityError
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from stand_in_cameras import StandInCameras
from stand_in_robots import StandInYAMRobot
from waiting import Waiting

WAIT_SECONDS = 3  # for the stand-in controller to reach an expected state
POLL_SECONDS = 0.005  # between checks of that state
# what the tests of a fault in a hold change of the tests' station: a short settle window, a hold that may stay
# unstable only briefly, and sequences that may run late
MOVING_HOLD_STATION = (
    "control.settle_window_s=0.1",
    "control.settle_timeout_s=0.4",
    "control.deadline_tolerance_s=0.5",
)


@pytest.fixture
def moving_hold(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[StandInYAMRobot, threading.Event]]:
    """The open stand-in YAM robot, settled, whose left arm's J4 reports a velocity while the event it yields is set:
    the robot and the event; closed at the end."""
    config = ConfigFiles.load("yam", *MOVING_HOLD_STATION)

    with StandInYAMRobot(config, cameras=StandInCameras(YAMRobot.CAMERAS)) as station:
        station.wait_stationary()
        runtime = station.control_loop
        active = threading.Event()
        read = runtime.driver.read

        def feedback():
            states = read()
            if active.is_set():
                v = [0.0] * 7
                v[3] = 0.1
                states["left"] = replace(states["left"], velocity=tuple(v))
            return states

        monkeypatch.setattr(runtime.driver, "read", feedback)
        yield station, active


def test_latched_hold_fault_keeps_original_metrics_at_every_consumer(moving_hold):
    station, active = moving_hold
    runtime = station.control_loop
    guard = ActionGuard(runtime)
    targets = dict(runtime.targets)
    before = len(runtime.driver.sent_targets)

    active.set()
    Waiting.until(lambda: runtime.status == "FAULT", WAIT_SECONDS, POLL_SECONDS)

    detail = station.status(with_snapshot=True)["fault_details"]
    assert detail["code"] == "StabilityError"
    assert detail["metrics"]["samples"] >= 4
    assert any(
        v["side"] == "left" and v["joint"] == "J4" and v["metric"] == "velocity_mean"
        for v in detail["metrics"]["violations"]
    )
    assert runtime.fault_details == detail
    json.dumps(detail, allow_nan=False)

    active.clear()
    sampled_at = runtime.states["left"].sampled_monotonic_s
    Waiting.until(lambda: runtime.states["left"].sampled_monotonic_s != sampled_at, WAIT_SECONDS, POLL_SECONDS)
    wait_for_hold = lambda: runtime.wait_for_hold(runtime.HOLD_WAIT_S)  # noqa: E731  (the longest wait for a hold)
    for read in (station.hold_snapshot, station.observe, guard.check, wait_for_hold):
        with pytest.raises(StabilityError, match="left J4") as caught:
            read()
        assert caught.value.details == detail
        caught.value.metrics.clear()  # Caller mutation must not change the latched record.
    detail["metrics"].clear()
    assert runtime.snapshot()["fault_details"]["metrics"]["violations"]
    assert runtime.status == "FAULT" and runtime.operation is None
    assert runtime.driver.sent_targets[before:] == [targets]  # Only fault holding, no new action.
