import json
from types import SimpleNamespace

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import StandInClock
from pear.deploy.station.control.robots.yam.commands import gripper_calibration


class Slider:
    """The gripper's motor as a viscously damped slider between two hard stops, on the stand-in clock."""

    def __init__(self, clock: StandInClock, lower: float = -5.17, upper: float = 1.4) -> None:
        self.clock, self.lower, self.upper = clock, lower, upper
        self.position, self.last = 0.0, clock.monotonic()
        self.initial_feedback = SimpleNamespace(position=self.position)
        self.torques = []

    def command(self, torque: float) -> SimpleNamespace:
        """Moves the slider under ``torque`` for the time since the last command and returns the motor's feedback."""
        dt = self.clock.monotonic() - self.last
        self.last = self.clock.monotonic()
        old = self.position

        # Viscously damped slider with hard stops, independent of endpoint detection.
        self.position = np.clip(old + torque / 0.5 * dt, self.lower, self.upper)
        self.torques.append(torque)

        return SimpleNamespace(
            id=7,
            error_code="0x1",
            position=float(self.position),
            velocity=(self.position - old) / dt if dt else 0.0,
            torque=torque,
            temperature_mos=25,
            temperature_rotor=25,
        )


@pytest.mark.parametrize(
    "lower,upper,stroke_range",
    [
        (-5.17, 1.4, (6.07, 7.07)),
        (1.27, 6.47, (4.7, 5.7)),  # The working station's travel, not the force model's.
    ],
)
def test_full_stroke_measured_in_fixed_coordinates(stand_in_clock, lower, upper, stroke_range):
    motor = Slider(stand_in_clock, lower=lower, upper=upper)
    motor.position = motor.initial_feedback.position = (lower + upper) / 2
    samples = []

    limits = gripper_calibration.GripperMeasurement.measure(motor, samples, stroke_range=stroke_range)

    assert limits == pytest.approx([upper, lower], abs=0.005)
    assert max(map(abs, motor.torques)) <= 0.5
    assert {s["phase"] for s in samples} == {"close", "open"}


def test_obstructed_partial_stroke_is_not_a_calibration(stand_in_clock):
    motor = Slider(stand_in_clock, lower=-2.0)

    with pytest.raises(RuntimeError, match="stroke.*implausible"):
        gripper_calibration.GripperMeasurement.measure(motor, [], stroke_range=(4.7, 5.7))


def test_late_feedback_is_rejected(stand_in_clock):
    motor = Slider(stand_in_clock)
    command = motor.command

    def late(torque):
        stand_in_clock.sleep(0.2)
        return command(torque)

    motor.command = late
    with pytest.raises(RuntimeError, match="feedback exceeded"):
        gripper_calibration.GripperMeasurement.measure(motor, [], stroke_range=(4.7, 5.7))


@pytest.mark.parametrize("motor_id", [1, 2, 3, 4, 5, 6, 0x7FF])
def test_calibration_wire_guard_rejects_other_motors(motor_id):
    sent = []
    guard = gripper_calibration.GripperCalibrationMotor.send_guard(lambda *a, **kw: sent.append(a), [])
    frame = SimpleNamespace(arbitration_id=motor_id, is_extended_id=False, data=bytes(8))

    with pytest.raises(RuntimeError, match="only gripper"):
        guard(frame)
    assert not sent


@pytest.mark.parametrize("opcode", [0xFB, 0xFE])
def test_calibration_wire_guard_rejects_reset_and_zero(opcode):
    guard = gripper_calibration.GripperCalibrationMotor.send_guard(lambda *a, **kw: pytest.fail("sent"), [])
    frame = SimpleNamespace(arbitration_id=7, is_extended_id=False, data=b"\xff" * 7 + bytes([opcode]))

    with pytest.raises(RuntimeError, match="refuses"):
        guard(frame)


@pytest.mark.parametrize("failure", [RuntimeError("feedback failure"), KeyboardInterrupt()])
def test_failed_calibration_disables_and_preserves_failure_record(monkeypatch, tmp_path, failure):
    closed = []

    class Motor:
        timeout_register = 8000
        frames = []

        def __init__(self, config, side):
            pass

        def close(self):
            closed.append(True)

    def fail(*_, **kwargs):
        raise failure

    monkeypatch.setattr(gripper_calibration, "GripperCalibrationMotor", Motor)
    monkeypatch.setattr(gripper_calibration.GripperMeasurement, "measure", fail)
    with pytest.raises(type(failure)):
        gripper_calibration.GripperCalibration.run(ConfigFiles.load_configs("yam")[1], "left", tmp_path)

    assert closed == [True]
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["success"] is False
    assert "gripper_limits" not in record
