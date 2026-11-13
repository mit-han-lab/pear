import math
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.robots.yam.joint.grippers import YAMGripperOperation
from yam_rigs import GripperRig, GripperRigs, MotionHelpers

PERIOD_S = 0.02  # one control period
STROKE_LIMIT_S = 4.5  # the longest accepted full stroke
ENDPOINT_TOLERANCE = 0.003  # the opening error accepted at the end of a stroke
STATIC_FRICTION_BAND = 0.0015  # the opening error the modelled servo needs before it moves
INTEGRATION_STEP_S = 0.001  # the modelled plant's step, finer than the control period
SAMPLE_PHASE_TICKS = 750  # control periods a full stroke may take while the feedback's age varies
FOLLOW_TOLERANCE = 0.02  # how far a command may lead the measured opening
ENCODER_RESOLUTION = 0.0000732  # the opening's quantization step
SERVO_LAG_S = 0.025  # the time constant of the servo whose feedback's age varies
HISTORY = 21  # plant steps kept, enough to look one control period back
SERVO_LAG_TICKS = 2300  # control periods a full stroke may take against a lagging servo


class ContactCandidate:
    """How a test steps a closing gripper until the SDK's force limit holds it on its object."""

    CANDIDATE_STEPS = 400  # control periods a closing gripper may take to meet an object

    @classmethod
    def reach_candidate(cls, rig: GripperRig, op: YAMGripperOperation, side: str = "right") -> float:
        """Steps ``rig`` until the gripper of ``side`` has a contact to confirm, and returns the clock's reading
        then."""
        for _ in range(cls.CANDIDATE_STEPS):
            rig.tick()
            if side in op.contacts:
                assert side not in op.outcomes
                return rig.clock.now
            assert rig.control_loop.fault is None

        pytest.fail("No contact candidate")


class LaggingServo:
    """A full gripper stroke against a position servo modelled with static friction and a first-order lag."""

    @staticmethod
    def full_stroke(tau: float, opening: bool) -> float:
        """Returns the seconds a full stroke takes against a servo of the time constant ``tau``."""
        control_config, config = ConfigFiles.load_configs("yam")
        direction = 1 if opening else -1
        initial = actual = float(not opening)
        targets = {s: [0.0] * 6 + [initial] for s in ("left", "right")}

        def states() -> dict[str, SimpleNamespace]:
            return {
                s: SimpleNamespace(
                    position=[0.0] * 6 + [actual if s == "right" else initial],
                    effort=[0.0] * 7,
                    sdk_gripper_force_limited=False,
                )
                for s in targets
            }

        operation = YAMGripperOperation(control_config, config, targets, states(), {"right": opening})
        max_lead = 0
        for tick in range(SERVO_LAG_TICKS):
            sample = operation.advance(tick * PERIOD_S, states(), targets)
            q = sample.position
            max_lead = max(max_lead, abs(q[13] - actual))
            assert max_lead <= config.arms["right"].limits.tracking_error[6]
            targets["right"] = list(q[7:])
            # Integrated finer than the trajectory clock.
            for _ in range(round(PERIOD_S / INTEGRATION_STEP_S)):
                speed = (q[13] - actual - direction * STATIC_FRICTION_BAND) / tau
                actual = float(np.clip(actual + INTEGRATION_STEP_S * speed, 0, 1))
            if operation.is_complete:
                assert abs(actual - float(opening)) < ENDPOINT_TOLERANCE
                return tick * PERIOD_S

        raise AssertionError("Full stroke failed to finish")


@pytest.fixture
def rig(gripper_rigs: GripperRigs) -> GripperRig:
    """A gripper rig whose right gripper meets an object at the opening 0.8."""
    return gripper_rigs.make(contacts={"right": 0.8})


def test_contact_waits_for_a_full_loaded_stable_window(rig):
    op = rig.begin({"right": False})
    started = ContactCandidate.reach_candidate(rig, op)
    duration = rig.robot_config.gripper_contact_confirm_s
    while rig.clock.now + rig.control_loop.control_config.control_period_s - started < duration:
        rig.tick()
        assert not op.is_complete

    rig.finish()
    assert op.outcomes == {"right": "resistance_detected_not_grasp_success"}
    metrics = op.final_readings["right"]["contact_confirmation"]
    assert metrics["confirmed"] and metrics["window_seconds"] >= duration
    assert metrics["mean_effort_ratio_abs"] >= 0.8
    assert rig.control_loop.targets["right"][6] == pytest.approx(0.8)


def test_brief_contact_resumes_from_feedback_and_reaches_closed_endpoint(rig):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    for _ in range(3):
        rig.tick()
    assert not op.is_complete

    # A different measured opening makes reusing the pre-contact trajectory observable.
    actual = 0.77
    rig.driver.positions["right"] = (0.0,) * 6 + (actual,)
    rig.driver.contacts.clear()
    op.endpoint_times["right"] = rig.clock.now - rig.control_config.settle_timeout_s
    rig.tick()
    resumed = rig.control_loop.targets["right"][6]
    assert actual - 0.3 * rig.control_loop.control_config.control_period_s <= resumed < actual
    assert "right" not in op.contacts
    assert "right" not in op.endpoint_times
    assert rig.control_loop.fault is None

    rig.finish()
    assert op.outcomes == {"right": "closed_without_confirmed_object"}
    assert rig.control_loop.targets["right"][6] == 0.0
    assert all(q[s][:6] == (0.0,) * 6 for q in rig.driver.commands for s in q)


def test_resumed_trajectory_keeps_velocity_acceleration_and_jerk_limits(rig):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    rig.driver.contacts.clear()
    first = len(rig.samples)

    rig.finish()

    samples = np.asarray(rig.samples[first:])
    assert np.max(abs(samples[:, 0, 13])) <= 0.3 + 1e-9
    assert np.max(abs(samples[:, 1, 13])) <= 1.2 + 1e-9
    assert np.max(abs(np.diff(samples[:, 1, 13]) / rig.control_loop.control_config.control_period_s)) <= 6 + 1e-7


def test_resume_preserves_measured_velocity_instead_of_reusing_old_profile(rig):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    rig.driver.contacts.clear()
    read = rig.driver.read

    def feedback():
        states = read()
        states["right"] = replace(states["right"], position=(0.0,) * 6 + (0.77,), velocity=(0.0,) * 6 + (-0.1,))
        return states

    rig.driver.read = feedback
    rig.tick()
    assert 0.767 < rig.control_loop.targets["right"][6] < 0.769
    assert -0.102 < rig.control_loop.trajectory.input.current_velocity[13] < -0.1

    rig.driver.read = read
    rig.finish()
    assert op.outcomes == {"right": "closed_without_confirmed_object"}


def test_confirmation_uses_actual_force_buildup_not_only_sdk_flag(rig):
    rig.driver.contact_effort_nm = 0.55  # Triggers the SDK, but below 80% of its requested holding torque.
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    for _ in range(20):
        rig.tick()
    assert not op.is_complete
    assert op.final_readings["right"]["contact_confirmation"]["mean_effort_ratio_abs"] < 0.8

    rig.driver.contact_effort_nm = 1.03
    rig.finish()
    assert op.outcomes["right"] == "resistance_detected_not_grasp_success"


@pytest.mark.parametrize("contact", [0.8, 0.015])
def test_force_that_never_builds_times_out_and_cannot_restart(rig, contact):
    rig.driver.contacts["right"] = contact
    rig.driver.contact_effort_nm = 0.55
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    for _ in range(180):
        rig.tick()

    assert rig.control_loop.status == "FAULT"
    assert "contact did not stabilize" in rig.control_loop.fault
    details = rig.control_loop.fault_details
    assert details["code"] == "GripperContactTimeout"
    assert not details["metrics"]["confirmed"]
    assert details["metrics"]["mean_effort_ratio_abs"] < 0.8
    assert not op.is_complete

    targets = dict(rig.control_loop.targets)
    rig.driver.contacts.clear()
    for _ in range(50):
        rig.tick()
    assert rig.control_loop.targets == targets


def test_force_limit_inside_closed_tolerance_completes_as_closed_endpoint(rig):
    # Empty fingers meet just short of a deep closed bound while the profile still
    # moves; the SDK's clipped target cannot build the torque needed to confirm.
    rig.driver.contacts["right"] = 0.006
    rig.driver.contact_effort_nm = 0.55
    op = rig.begin({"right": False})

    rig.finish()

    assert op.outcomes == {"right": "closed_without_confirmed_object"}
    assert op.final_readings["right"]["contact_confirmation"] == {"closed_endpoint_under_force_limit": True}
    assert rig.control_loop.targets["right"][6] == 0.0
    assert rig.control_loop.fault is None


@pytest.mark.parametrize("unstable", ["velocity", "opening_span"])
def test_contact_requires_stable_opening_and_velocity(rig, unstable):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    read = rig.driver.read
    counter = 0

    def feedback():
        nonlocal counter
        counter += 1
        values = read()
        state = values["right"]
        if unstable == "velocity":
            state = replace(state, velocity=(0.0,) * 6 + (0.1,))
        else:
            state = replace(state, position=(0.0,) * 6 + (0.8 + (-1) ** counter * 0.015,))
        values["right"] = state
        return values

    rig.driver.read = feedback
    for _ in range(25):
        rig.tick()
    assert not op.is_complete

    rig.driver.read = read
    rig.finish()
    assert op.outcomes["right"] == "resistance_detected_not_grasp_success"


def test_brief_release_restarts_confirmation_window(rig):
    op = rig.begin({"right": False})
    first = ContactCandidate.reach_candidate(rig, op)
    for _ in range(6):
        rig.tick()
    rig.driver.contacts.clear()
    rig.tick()
    assert not op.contacts

    rig.driver.contacts["right"] = 0.8
    second = ContactCandidate.reach_candidate(rig, op)
    assert second > first
    for _ in range(5):
        rig.tick()
    assert rig.clock.now - first > 0.2 and rig.clock.now - second < 0.2
    assert not op.is_complete

    rig.finish()
    assert op.outcomes["right"] == "resistance_detected_not_grasp_success"


def test_left_confirmation_and_right_transient_are_independent(rig):
    rig.driver.contacts["left"] = 0.9
    op = rig.begin({"left": False, "right": False})
    ContactCandidate.reach_candidate(rig, op)
    assert op.outcomes["left"] == "resistance_detected_not_grasp_success"
    left_target = rig.control_loop.targets["left"]

    del rig.driver.contacts["right"]
    rig.finish()
    assert op.outcomes["right"] == "closed_without_confirmed_object"
    assert rig.control_loop.targets["left"] == left_target


def test_repeated_batch_cannot_confirm_and_does_not_hide_other_sides_fresh_feedback(rig):
    rig.driver.contacts = {"left": 1.0, "right": 1.0}
    op = rig.begin({"left": False, "right": False})
    read = rig.driver.read

    def feedback():
        values = read()
        values["right"] = replace(values["right"], sample_id=7)
        return values

    rig.driver.read = feedback
    for _ in range(30):
        rig.tick()

    assert op.outcomes == {"left": "resistance_detected_not_grasp_success"}
    assert op.contacts["right"].metrics["samples"] == 1


@pytest.mark.parametrize("kind", ["stop", "cancel", "deadline", "stale"])
def test_interrupted_contact_candidate_never_resumes(rig, kind):
    op = rig.begin({"right": False})
    ContactCandidate.reach_candidate(rig, op)
    if kind == "stop":
        rig.control_loop.stop()
    elif kind == "cancel":
        rig.cancellation.set()
    elif kind == "deadline":
        # the port keeps the operation's deadline private: no public way sets it
        rig.control_loop._operation_deadline = rig.clock.now
    else:
        rig.driver.is_stale = True
    rig.tick()

    targets = dict(rig.control_loop.targets)
    rig.driver.is_stale = False
    rig.driver.contacts.clear()
    for _ in range(100):
        rig.tick()
    assert rig.control_loop.targets == targets
    assert rig.control_loop.operation is None
    assert not op.is_complete


@pytest.mark.parametrize("tau,seconds", [(0, 4.5), (0.02, 4.5), (0.05, 4.5), (0.1, 4.5)])
@pytest.mark.parametrize("opening", [False, True])
def test_commands_to_a_lagging_gripper_stay_continuous_and_within_limits(tau, seconds, opening):
    control_config, c = ConfigFiles.load_configs("yam")
    initial = float(not opening)
    targets = {s: [0.0] * 6 + [initial] for s in ("left", "right")}
    actual = initial

    def states():
        return {
            s: SimpleNamespace(position=[0.0] * 6 + [actual], effort=[0.0] * 7, sdk_gripper_force_limited=False)
            for s in targets
        }

    operation = YAMGripperOperation(control_config, c, targets, states(), {"left": opening, "right": opening})
    previous_velocity = previous_acceleration = 0.0
    for i in range(int(seconds / 0.02)):
        sample = operation.advance((i + 1) * 0.02, states(), targets)
        q, v, a = sample.position, sample.velocity, sample.acceleration
        assert abs(q[6] - actual) <= c.arms["left"].limits.tracking_error[6]
        assert abs(v[6]) <= 0.3 + 1e-9 and abs(a[6]) <= c.arms["left"].limits.acceleration[6] + 1e-9
        assert abs(a[6] - previous_acceleration) <= c.arms["left"].limits.jerk[6] * 0.02 + 1e-7
        assert abs(v[6] - previous_velocity) <= c.arms["left"].limits.acceleration[6] * 0.02 + 1e-7
        previous_velocity, previous_acceleration = v[6], a[6]
        targets = {s: list(q[j * 7 : (j + 1) * 7]) for j, s in enumerate(targets)}
        actual += (q[6] - actual) * (1 if tau == 0 else 1 - math.exp(-0.02 / tau))
        if operation.is_complete:
            break

    assert operation.is_complete
    assert abs(actual - opening) < 0.003


@pytest.mark.parametrize("initial", [0.1, 0.4, 0.5, 0.7, 0.99])
@pytest.mark.parametrize("opening", [False, True])
def test_partial_strokes_never_issue_endpoint_roundoff_outside_bounds(initial, opening):
    control_config, c = ConfigFiles.load_configs("yam")
    actual = initial
    targets = {s: [0.0] * 6 + [actual] for s in ("left", "right")}

    def states():
        return {
            s: SimpleNamespace(position=[0.0] * 6 + [actual], effort=[0.0] * 7, sdk_gripper_force_limited=False)
            for s in targets
        }

    operation = YAMGripperOperation(control_config, c, targets, states(), {"left": opening, "right": opening})
    for i in range(800):
        # Scheduling jitter does not change the fixed trajectory limits.
        sample = operation.advance(100 + i * 0.02 + (i % 3) * 0.0002, states(), targets)
        q = sample.position
        assert 0 <= q[6] <= 1 and 0 <= q[13] <= 1
        actual = q[6]
        targets = {s: list(q[j * 7 : (j + 1) * 7]) for j, s in enumerate(targets)}
        if operation.is_complete:
            break

    assert operation.is_complete and targets["left"][6] == float(opening)


@pytest.mark.parametrize("opening", [False, True])
def test_arm_position_change_during_gripping_keeps_operation(gripper_rigs, opening):
    rig = gripper_rigs.make(opening=float(not opening))
    op = rig.begin({"right": opening})
    read = rig.driver.read

    def feedback():
        states = read()
        q = list(states["left"].position)
        q[0] += 0.02
        states["left"] = replace(states["left"], position=tuple(q))
        return states

    rig.driver.read = feedback
    rig.finish()

    assert rig.control_loop.fault is None
    assert rig.control_loop.snapshot()["motion"]["end_reason"] == "finished"
    assert op.outcomes["right"] == ("open" if opening else "closed_without_confirmed_object")
    assert all(command[s][:6] == (0.0,) * 6 for command in rig.driver.commands for s in command)


def test_opening_obstruction_is_a_fault_even_when_sdk_reports_force_limiting(gripper_rigs):
    rig = gripper_rigs.make(opening=0.3)
    op = rig.begin({"right": True})
    read = rig.driver.read
    rig.driver.read = lambda: {
        s: replace(
            v,
            position=(0.0,) * 6 + (0.3,),
            effort=(0.0,) * 6 + (0.4,),
            sdk_gripper_force_limited=True,
        )
        for s, v in read().items()
    }
    for _ in range(400):
        rig.tick()
        if rig.control_loop.status == "FAULT":
            break

    assert "opening endpoint was not reached" in rig.control_loop.fault
    assert not op.outcomes
    assert rig.control_loop.targets["right"][6] == pytest.approx(0.3)

    targets = dict(rig.control_loop.targets)
    rig.driver.read = read
    for _ in range(30):
        rig.tick()
    assert rig.control_loop.targets == targets  # no automatic restart after clearance


def test_confirmed_contact_discards_remainder_and_never_restarts_after_effort_disappears(gripper_rigs):
    rig = gripper_rigs.make(contacts={"right": 0.6})
    op = rig.begin({"right": False})
    rig.finish()
    targets = dict(rig.control_loop.targets)
    assert op.outcomes["right"] == "resistance_detected_not_grasp_success"

    rig.driver.contacts.clear()
    for _ in range(100):
        rig.tick()
    assert rig.control_loop.targets == targets


def test_sdk_contact_does_not_bypass_fresh_station_settling(gripper_rigs):
    rig = gripper_rigs.make(contacts={"right": 1.0})
    op = rig.begin({"right": False})
    read = rig.driver.read
    rig.driver.read = lambda: {s: replace(v, sample_id=7) for s, v in read().items()}

    for _ in range(30):
        rig.tick()

    assert "right" not in op.outcomes
    assert not rig.control_loop.snapshot()["arrival"]["settled"]


@pytest.mark.parametrize("opening", [False, True])
def test_effort_without_the_sdk_force_limit_neither_faults_nor_counts_as_contact(gripper_rigs, opening):
    rig = gripper_rigs.make(opening=float(not opening))
    op = rig.begin({"right": opening})
    read = rig.driver.read
    rig.driver.read = lambda: {
        s: replace(v, effort=(0.0,) * 6 + (1.03,), sdk_gripper_force_limited=False) for s, v in read().items()
    }

    rig.finish()

    assert rig.control_loop.fault is None
    assert op.outcomes["right"] == ("open" if opening else "closed_without_confirmed_object")


@pytest.mark.parametrize("opening", [False, True])
@pytest.mark.parametrize("error", [0.0033131793692275815, 0.0099])
def test_station_arrival_and_endpoint_completion_use_the_same_gripper_tolerance(opening, error):
    target = float(opening)
    actual = target + (-error if opening else error)
    targets = {s: [0.0] * 6 + [target] for s in ("left", "right")}
    states = {
        s: SimpleNamespace(
            position=[0.0] * 6 + [actual],
            effort=[0.0] * 6 + [0.33],
            sdk_gripper_force_limited=False,
        )
        for s in targets
    }
    control_config, config = ConfigFiles.load_configs("yam")
    operation = YAMGripperOperation(control_config, config, targets, states, {"left": opening, "right": opening})

    sample = operation.advance(100, states, targets)

    q = sample.position
    assert operation.is_complete
    assert operation.final_readings["right"]["final_opening"] == actual
    assert q[13] == target
    snapshot = {
        "targets": targets,
        "arms": {s: {"position": v.position} for s, v in states.items()},
    }
    assert MotionHelpers.tracking(config).arrival(*MotionHelpers.vectors(snapshot))["reached"]


def test_a_closed_gripper_resting_outside_the_tolerance_times_out(gripper_rigs):
    rig = gripper_rigs.make(opening=0.0)
    op = rig.begin({"right": False})
    read = rig.driver.read
    rig.driver.read = lambda: {
        s: replace(v, position=(0.0,) * 6 + (0.0101 if s == "right" else 0.0,)) for s, v in read().items()
    }

    for _ in range(400):
        rig.tick()
        if rig.control_loop.status == "FAULT":
            break

    assert not op.is_complete
    assert "closing endpoint was not reached" in rig.control_loop.fault


@pytest.mark.parametrize("opening", [False, True])
@pytest.mark.parametrize("sample_delay_ms", [4, 8])
def test_a_full_stroke_finishes_in_time_when_the_feedback_age_varies(opening, sample_delay_ms):
    initial = actual = float(not opening)
    direction = 1 if opening else -1
    targets = {s: [0.0] * 6 + [initial] for s in ("left", "right")}
    plant = deque([actual] * HISTORY, maxlen=HISTORY)

    def states(measured):
        return {
            s: SimpleNamespace(
                position=[0.0] * 6 + [measured if s == "right" else initial],
                effort=[0.0] * 7,
                sdk_gripper_force_limited=False,
            )
            for s in targets
        }

    operation = YAMGripperOperation(*ConfigFiles.load_configs("yam"), targets, states(actual), {"right": opening})
    for tick in range(SAMPLE_PHASE_TICKS):
        # Alternate the feedback phase, quantized as the encoder is.
        age = sample_delay_ms if tick % 3 == 0 else 0
        measured = round(plant[-1 - age] / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
        sample = operation.advance(tick * PERIOD_S, states(measured), targets)
        q = sample.position
        assert abs(q[13] - measured) < FOLLOW_TOLERANCE
        targets["right"] = list(q[7:])
        for _ in range(round(PERIOD_S / INTEGRATION_STEP_S)):
            speed = (q[13] - actual - direction * STATIC_FRICTION_BAND) / SERVO_LAG_S
            actual = float(np.clip(actual + INTEGRATION_STEP_S * speed, 0, 1))
            plant.append(actual)
        if operation.is_complete:
            break

    assert operation.is_complete
    assert abs(actual - float(opening)) < ENDPOINT_TOLERANCE
    assert tick * PERIOD_S < STROKE_LIMIT_S


@pytest.mark.parametrize("tau", [0.025, 0.05, 0.08])
@pytest.mark.parametrize("opening", [False, True])
def test_a_full_stroke_against_a_lagging_servo_with_static_friction_finishes_in_time(tau, opening):
    assert LaggingServo.full_stroke(tau, opening) < STROKE_LIMIT_S


@pytest.mark.parametrize("opening", [False, True])
def test_an_empty_full_stroke_finishes_in_time_within_the_gripper_limits(gripper_rigs, opening):
    rig = gripper_rigs.make(opening=float(not opening))
    start = rig.clock.now  # a rig's control loop settles when it is made: the stroke is timed from the rig's start
    op = rig.begin({"left": opening, "right": opening})

    rig.finish()

    assert rig.clock.now - start < 4.5
    assert op.outcomes == {s: "open" if opening else "closed_without_confirmed_object" for s in ("left", "right")}
    samples = np.asarray(rig.samples)
    for index in (6, 13):
        assert np.max(abs(samples[:, 0, index])) <= 0.3 + 1e-9
        assert np.max(abs(samples[:, 1, index])) <= rig.robot_config.arms["left"].limits.acceleration[6] + 1e-9
        assert np.max(abs(np.diff(samples[:, 1, index]) / 0.02)) <= rig.robot_config.arms["left"].limits.jerk[6] + 1e-7
    assert all(np.allclose(q[s][:6], 0) for q in rig.driver.commands for s in q)


def test_independent_sdk_contacts_hold_force_adjusted_targets_and_other_joints(gripper_rigs):
    rig = gripper_rigs.make(contacts={"left": 0.7, "right": 0.35})
    op = rig.begin({"left": False, "right": False})

    rig.finish()

    assert op.outcomes == {s: "resistance_detected_not_grasp_success" for s in rig.driver.contacts}
    control_period_s = rig.control_loop.control_config.control_period_s
    for side, position in rig.driver.contacts.items():
        # At most one trajectory tick precedes the stand-in's SDK contact feedback.
        assert min(q[side][6] for q in rig.driver.commands) >= position - 0.3 * control_period_s - 1e-9
        assert rig.control_loop.targets[side][6] == pytest.approx(position)
    assert all(np.allclose(q[s][:6], 0) for q in rig.driver.commands for s in q)


@pytest.mark.parametrize("failure", ["stale", "missing", "tracking"])
def test_fault_sends_only_unchanged_halt_target(gripper_rigs, failure):
    rig = gripper_rigs.make(contacts={"right": 1.0})
    rig.begin({"right": False})
    for _ in range(5):
        rig.tick()
    if failure == "stale":
        rig.driver.is_stale = True
    elif failure == "missing":
        rig.driver.lacks_effort = True
    else:
        rig.driver.positions["right"] = (0.2,) + (0.0,) * 5 + (1.0,)
    before = len(rig.driver.commands)

    rig.tick()

    assert rig.control_loop.status == "FAULT"
    assert len(rig.driver.commands) == before + 1
    assert rig.driver.commands[-1] == rig.driver.commands[before - 1]
    rig.tick()
    assert len(rig.driver.commands) == before + 1


@pytest.mark.parametrize("kind", ["stop", "cancel", "deadline"])
def test_interrupt_cannot_resume_old_gripper_trajectory(gripper_rigs, kind):
    rig = gripper_rigs.make()
    rig.begin({"right": False})
    for _ in range(30):
        rig.tick()
    if kind == "stop":
        rig.control_loop.stop()
    if kind == "cancel":
        rig.cancellation.set()
    if kind == "deadline":
        # the port keeps the operation's deadline private: no public way sets it
        rig.control_loop._operation_deadline = rig.clock.now

    for _ in range(80):
        rig.tick()

    assert rig.control_loop.status == "HOLD"
    stopped = rig.control_loop.targets["right"][6]
    for _ in range(30):
        rig.tick()
    assert rig.control_loop.targets["right"][6] == stopped
    assert rig.control_loop.targets["left"][6] == 1.0


def test_unconfirmed_obstruction_is_terminal_and_never_restarts_old_motion(gripper_rigs):
    rig = gripper_rigs.make(contacts={"right": 0.8}, contact_effort_nm=0.0)
    op = rig.begin({"right": False})
    for _ in range(400):
        rig.tick()

    held = rig.control_loop.targets["right"][6]
    assert held == 0.0
    assert "closing endpoint was not reached" in rig.control_loop.fault
    assert rig.control_loop.status == "FAULT" and not op.is_complete

    rig.driver.contacts.clear()
    for _ in range(40):
        rig.tick()
    assert rig.control_loop.targets["right"][6] == held


@pytest.mark.parametrize("gap", [0.084, 0.22])
def test_delayed_tick_keeps_gripper_trajectory_with_fresh_feedback(gripper_rigs, gap):
    rig = gripper_rigs.make()
    op = rig.begin({"right": False})
    rig.tick()
    sent = len(rig.driver.commands)

    rig.clock.now += gap - rig.control_loop.control_config.control_period_s
    rig.tick()

    assert rig.control_loop.status == "RUNNING", rig.control_loop.fault
    assert rig.control_loop.operation is op
    assert len(rig.driver.commands) == sent + 1
    rig.finish()
    assert op.outcomes == {"right": "closed_without_confirmed_object"}


@pytest.mark.parametrize("contact", [0.0, 0.6])
def test_one_contact_or_endpoint_cannot_restart_completed_other_side(gripper_rigs, contact):
    rig = gripper_rigs.make(contacts={"left": contact})
    op = rig.begin({"left": False, "right": False})
    rig.finish()
    before = dict(rig.control_loop.targets)

    for _ in range(50):
        rig.tick()

    assert rig.control_loop.targets == before
    assert op.outcomes["right"] == "closed_without_confirmed_object"
    assert op.outcomes["left"] in (
        "closed_without_confirmed_object",
        "resistance_detected_not_grasp_success",
    )


def test_brief_effort_noise_does_not_freeze_or_claim_contact(gripper_rigs):
    rig = gripper_rigs.make()
    op = rig.begin({"left": False, "right": False})
    read = rig.driver.read
    noise = {"value": 0.41}
    rig.driver.read = lambda: {
        s: replace(v, effort=(0.0,) * 6 + (noise["value"] if s == "left" else 0.0,)) for s, v in read().items()
    }

    rig.tick()
    assert "left" not in op.outcomes

    noise["value"] = 0
    rig.finish()
    assert op.outcomes["left"] == "closed_without_confirmed_object"
