import numpy as np
import pytest

from pear.deploy.station.control.joint.feedforward import FrictionFeedforward

CONTROL_PERIOD_S = 0.004  # of the SDK control thread
TORQUE = [1.0, 2.0, 0.0]  # per-joint friction torque, Nm


class SDKControlThread:
    """The updates the SDK's control thread makes of a friction feedforward, one each control period."""

    @staticmethod
    def run(
        friction: FrictionFeedforward, velocity: list[float], start: float, seconds: float
    ) -> tuple[float, np.ndarray | None]:
        """Updates ``friction`` with the planned ``velocity`` for ``seconds`` from the time ``start``, and returns the
        time then and the last feedforward."""
        t, torque = start, None
        for _ in range(round(seconds / CONTROL_PERIOD_S)):
            t += CONTROL_PERIOD_S
            torque = friction.update(velocity, t, None)

        return t, torque


def test_torque_follows_planned_direction_after_a_smooth_ramp():
    friction = FrictionFeedforward(TORQUE, [0.0] * len(TORQUE))
    friction.update([0, 0, 0], 0.0, None)

    t, torque = SDKControlThread.run(friction, [0.2, -0.2, 0.2], 0.0, FrictionFeedforward.ENGAGE_S / 2)
    assert torque == pytest.approx([0.5, -1.0, 0.0], abs=0.1)
    _, torque = SDKControlThread.run(friction, [0.2, -0.2, 0.2], t, FrictionFeedforward.ENGAGE_S)
    assert torque == pytest.approx([1.0, -2.0, 0.0])


def test_torque_holds_through_the_stop_then_fades_to_zero():
    friction = FrictionFeedforward(TORQUE, [0.0] * len(TORQUE))
    t, _ = SDKControlThread.run(friction, [0.3, -0.3, 0], 0.0, 0.5)

    # The plan decelerates and stops: the torque must not drop with the speed.
    t, torque = SDKControlThread.run(friction, [0.001, -0.001, 0], t, 0.05)
    assert torque == pytest.approx([1.0, -2.0, 0.0])
    t, torque = SDKControlThread.run(friction, [0, 0, 0], t, FrictionFeedforward.LATCH_S - 0.06)
    assert torque == pytest.approx([1.0, -2.0, 0.0])
    t, torque = SDKControlThread.run(friction, [0, 0, 0], t, 0.02 + FrictionFeedforward.FADE_S / 2)
    assert 0.3 < torque[0] < 0.7 and -1.4 < torque[1] < -0.6
    previous = torque
    t, torque = SDKControlThread.run(friction, [0, 0, 0], t, FrictionFeedforward.FADE_S)
    assert torque == pytest.approx([0, 0, 0])
    assert np.all(np.abs(torque) <= np.abs(previous))


def test_reversal_ramps_through_zero_without_a_step():
    friction = FrictionFeedforward([1.0], [0.0])
    t, _ = SDKControlThread.run(friction, [0.2], 0.0, 0.3)
    steps, last = [], 1.0

    for _ in range(100):
        t += CONTROL_PERIOD_S
        value = friction.update([-0.2], t, None)[0]
        steps.append(abs(value - last))
        last = value

    assert last == pytest.approx(-1.0)
    assert max(steps) <= 2 * CONTROL_PERIOD_S / FrictionFeedforward.ENGAGE_S + 1e-9


def test_stalled_caller_cannot_jump_the_ramp_and_bad_input_fades():
    friction = FrictionFeedforward([1.0], [0.0])
    friction.update([0.0], 0.0, None)
    assert friction.update([0.2], 5.0, None)[0] == pytest.approx(1.0)  # dt capped at 0.1 s.

    friction = FrictionFeedforward([1.0], [0.0])
    t, _ = SDKControlThread.run(friction, [0.2], 0.0, 0.3)
    _, torque = SDKControlThread.run(
        friction, [float("nan")], t, FrictionFeedforward.LATCH_S + FrictionFeedforward.FADE_S + 0.05
    )
    assert torque[0] == pytest.approx(0.0)


def test_negative_torque_is_rejected():
    with pytest.raises(ValueError):
        FrictionFeedforward([-0.1], [0.0])


def test_magnitude_grows_with_gravity_load():
    friction = FrictionFeedforward([0.5, 0.5], [0.2, 0.0])
    t = 0.0
    for _ in range(50):
        t += CONTROL_PERIOD_S
        torque = friction.update([0.2, -0.2], t, load=[-10.0, -10.0])
    assert torque == pytest.approx([2.5, -0.5])

    torque = friction.update([0.2, -0.2], t + CONTROL_PERIOD_S, load=[float("nan"), 0.0])
    assert torque == pytest.approx([0.5, -0.5])  # Malformed load counts as none.
    with pytest.raises(ValueError):
        FrictionFeedforward([0.5], [-0.1])
