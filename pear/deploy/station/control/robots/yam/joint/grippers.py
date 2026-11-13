import copy
from enum import StrEnum
from typing import Any, NamedTuple

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.operations import GripperEndpointTimeout, GripperOperation
from pear.deploy.station.control.joint.stability import SampleWindow
from pear.deploy.station.control.joint.trajectory import CommandSample, GripperProfile
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.layout import LAYOUT


class GripperOutcome(StrEnum):
    """How a gripper's operation ended, by the word results, reports and the model's feedback carry."""

    OPEN = "open"
    CLOSED_WITHOUT_CONFIRMED_OBJECT = "closed_without_confirmed_object"
    RESISTANCE_DETECTED = "resistance_detected_not_grasp_success"  # the SDK's force limit held: no grasp is confirmed


class GripperContactTimeout(TimeoutError):
    """Raised when the SDK's resistance did not become a steady, loaded contact in time."""

    def __init__(self, message: str, metrics: dict[str, Any]) -> None:
        super().__init__(message)
        self.metrics = copy.deepcopy(metrics)  # the contact's window when it timed out

    @property
    def details(self) -> dict[str, Any]:
        """What a fault records about this failure: which check of the contact failed."""
        return {"code": type(self).__name__, "message": str(self), "metrics": self.metrics}


class ContactSample(NamedTuple):
    """What a contact's window keeps of one reading of a gripper."""

    opening: float
    velocity: float
    effort_ratio: float  # the measured effort as a share of the SDK's holding torque


class ContactConfirmation:
    """Confirms a gripper's contact from new, unbroken readings, not from wall time: the gripper holds still, loaded."""

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig, start_time: float) -> None:
        self.control_config, self.robot_config = control_config, robot_config
        self.start_time = start_time  # when the SDK's force limit first held
        self.window = SampleWindow(control_config, robot_config.gripper_contact_confirm_s)
        self.metrics: dict[str, Any] = {"confirmed": False, "samples": 0, "window_seconds": 0.0}

    def add(self, state: YAMArmState) -> bool:
        """Adds a new reading of the gripper's arm and returns whether the contact is now confirmed."""
        sample = ContactSample(
            JointLayout.gripper_value(state.position),
            JointLayout.gripper_value(state.velocity),
            JointLayout.gripper_value(state.effort) / abs(state.sdk_gripper_target_effort_nm),
        )
        if not self.window.add(state.sampled_monotonic_s, state.sample_id, sample):
            return False

        samples = self.window.samples
        window_s = samples[-1].sampled_monotonic_s - samples[0].sampled_monotonic_s
        opening_span = float(np.ptp([window_sample.value.opening for window_sample in samples]))
        mean_speed = float(abs(np.mean([window_sample.value.velocity for window_sample in samples])))
        effort_ratio = float(abs(np.mean([window_sample.value.effort_ratio for window_sample in samples])))
        robot_config = self.robot_config
        is_confirmed = (
            SampleWindow.is_complete(samples, robot_config.gripper_contact_confirm_s, self.control_config.rate_hz)
            and opening_span <= robot_config.gripper_tolerance
            and mean_speed <= robot_config.velocity_tolerance
            and effort_ratio >= robot_config.gripper_contact_effort_ratio
        )

        self.metrics = {
            "confirmed": is_confirmed,
            "samples": len(samples),
            "window_seconds": window_s,
            "opening_span": opening_span,
            "mean_velocity_abs": mean_speed,
            "mean_effort_ratio_abs": effort_ratio,
        }

        return is_confirmed


class YAMGripperOperation(GripperOperation):
    """Moves each requested gripper to its endpoint while every arm target stays held; a closing gripper that the
    SDK's force limit stops ends once its contact is confirmed."""

    ENDPOINT_EPSILON = 1e-8  # how closely a command or a profile's speed must match its endpoint

    def __init__(
        self,
        control_config: ControlConfig,
        robot_config: YAMConfig,
        targets: dict[str, tuple[float, ...]],
        states: dict[str, YAMArmState],
        gripper_request: dict[str, bool],
    ) -> None:
        super().__init__(control_config, robot_config, dict(gripper_request))
        start_targets = np.array(LAYOUT.join(targets))
        # the arms' measured coordinates as the operation starts
        self.start_positions = np.array(LAYOUT.join({arm: state.position for arm, state in states.items()}))
        self.outcomes: dict[str, str] = {}  # each finished gripper's outcome, by arm
        self.final_readings: dict[str, dict[str, Any]] = {arm: {} for arm in self.opens_by_arm}
        self.sent_count = 0  # of the commands sent
        self.endpoint_times: dict[str, float] = {}  # when each gripper's command reached its endpoint
        self.contacts: dict[str, ContactConfirmation] = {}  # the contacts being confirmed, by arm
        self.profiles = {
            arm: self.new_profile(arm, start_targets[LAYOUT.gripper_indices[LAYOUT.arms.index(arm)]])
            for arm in self.opens_by_arm
        }

    @property
    def is_complete(self) -> bool:
        """Whether every requested gripper has its outcome."""
        return len(self.outcomes) == len(self.opens_by_arm)

    def advance(
        self, now: float, states: dict[str, YAMArmState], targets: dict[str, tuple[float, ...]]
    ) -> CommandSample:
        """Returns the command of every arm's held target with the next opening of each gripper still moving."""
        measured_positions = np.array(LAYOUT.join({arm: state.position for arm, state in states.items()}))
        commanded_positions = np.array(LAYOUT.join(targets))
        position = commanded_positions.copy()
        velocity, acceleration = np.zeros(LAYOUT.vector_length), np.zeros(LAYOUT.vector_length)

        for arm, opens in self.opens_by_arm.items():
            index = LAYOUT.gripper_indices[LAYOUT.arms.index(arm)]
            state = states[arm]
            measured_opening, commanded_opening = measured_positions[index], commanded_positions[index]
            effort = float(JointLayout.gripper_value(state.effort)) if state.effort else 0.0
            self.final_readings[arm].update(
                {
                    "final_opening": float(measured_opening),
                    "final_effort_nm": effort,
                    "initial_opening": float(self.start_positions[index]),
                }
            )
            if arm in self.outcomes:
                continue

            if arm in self.contacts and not state.sdk_gripper_force_limited:
                # only an unfinished contact resumes: from the measured opening and velocity, without its old
                # trajectory and endpoint timer; a stop, a pause or a fault ends the whole operation instead
                del self.contacts[arm]
                self.endpoint_times.pop(arm, None)
                start_opening = float(np.clip(measured_opening, 0, 1))
                self.profiles[arm] = self.new_profile(arm, start_opening, JointLayout.gripper_value(state.velocity))
                self.final_readings[arm]["contact_confirmation"] = {"released_before_confirmation": True}

            profile = self.profiles[arm]
            is_at_endpoint = (
                abs(measured_opening - profile.endpoint) < self.robot_config.gripper_tolerance
                and abs(commanded_opening - profile.endpoint) < self.ENDPOINT_EPSILON
                and abs(profile.input.current_velocity[0]) < self.ENDPOINT_EPSILON
            )
            if is_at_endpoint:
                outcome = GripperOutcome.OPEN if opens else GripperOutcome.CLOSED_WITHOUT_CONFIRMED_OBJECT
                self.outcomes[arm] = outcome.value
                continue

            if not opens and state.sdk_gripper_force_limited:
                position[index] = self._follow_force_limit(arm, now, state, measured_opening, commanded_opening)
                continue
            if arm in self.endpoint_times and now - self.endpoint_times[arm] > self.control_config.settle_timeout_s:
                direction = "opening" if opens else "closing"
                raise GripperEndpointTimeout(f"{arm}: {direction} endpoint was not reached")

            candidate, candidate_velocity, candidate_acceleration, is_finished = profile.advance()
            if is_finished and abs(candidate - profile.endpoint) < self.ENDPOINT_EPSILON:
                candidate = profile.endpoint
                self.endpoint_times.setdefault(arm, now)
            position[index] = candidate
            velocity[index], acceleration[index] = candidate_velocity, candidate_acceleration
            profile.accept(candidate)

        self.sent_count += 1

        return CommandSample(position, velocity, acceleration, self.is_complete, requested_target=position)

    def report(self) -> dict[str, Any]:
        """Returns what the robot's reports keep of the operation: each gripper's readings and outcome, and the
        commands sent."""
        return {"grippers": self.final_readings, "outcomes": dict(self.outcomes), "updates_sent": self.sent_count}

    def _follow_force_limit(
        self, arm: str, now: float, state: YAMArmState, measured_opening: float, commanded_opening: float
    ) -> float:
        """Returns the opening to command for a closing gripper whose force the SDK limits, and ends the gripper's
        operation once its contact is confirmed or it bottomed out."""
        if abs(measured_opening) < self.robot_config.gripper_tolerance:
            # bottomed out: the SDK clips its force-limited target at the closed bound, where the requested torque
            # cannot build, so the rule of the closed endpoint applies
            self.outcomes[arm] = GripperOutcome.CLOSED_WITHOUT_CONFIRMED_OBJECT.value
            self.final_readings[arm]["contact_confirmation"] = {"closed_endpoint_under_force_limit": True}
            return GripperProfile.CLOSED_ENDPOINT

        if arm not in self.contacts:
            self.contacts[arm] = ContactConfirmation(self.control_config, self.robot_config, now)
        contact = self.contacts[arm]
        # the trajectory pauses while the SDK regulates the force, and the closing intent stays: following the SDK's
        # target toward open could release its own force limiter
        opening = min(commanded_opening, state.sdk_gripper_target)
        is_confirmed = contact.add(state)
        self.final_readings[arm]["contact_confirmation"] = dict(contact.metrics)

        if is_confirmed:
            self.outcomes[arm] = GripperOutcome.RESISTANCE_DETECTED.value
            return state.sdk_gripper_target
        if now - contact.start_time > self.control_config.settle_timeout_s:
            raise GripperContactTimeout(f"{arm}: contact did not stabilize under SDK force limiting", contact.metrics)

        return opening
