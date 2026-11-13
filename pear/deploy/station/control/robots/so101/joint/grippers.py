from enum import StrEnum
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.operations import GripperEndpointTimeout, GripperOperation
from pear.deploy.station.control.joint.stability import SampleWindow
from pear.deploy.station.control.joint.trajectory import CommandSample
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.layout import ARM, GRIPPER_INDEX, VECTOR_LENGTH


class GripperOutcome(StrEnum):
    """How the jaw's operation ended, by the word its report and an action's result carry."""

    OPEN = "open"
    CLOSED = "closed"
    STOPPED_BEFORE_CLOSED = "stopped_before_closed"  # it came to rest on an object: blocked, never a confirmed grasp


class SO101GripperOperation(GripperOperation):
    """Moves the jaw to its open or its closed endpoint while the arm's targets stay held."""

    def __init__(
        self,
        control_config: ControlConfig,
        robot_config: SO101Config,
        targets: dict[str, tuple[float, ...]],
        states: dict[str, ArmState],
        opens_gripper: bool,
    ) -> None:
        super().__init__(control_config, robot_config, {ARM: opens_gripper})
        self.initial_opening = float(states[ARM].position[GRIPPER_INDEX])
        self.outcome: GripperOutcome | None = None
        self.opening_report: dict[str, float] = {}  # the opening at the start and the latest one
        self.sample_count = 0  # of the command samples sent
        self.finished_at: float | None = None  # when the jaw's trajectory reached its endpoint

        # the latest openings; a gap starts the window over, a reading stamped out of order does not
        self.rest_window = SampleWindow(control_config, control_config.settle_window_s, checks_stamp_order=False)
        self.profile = self.new_profile(ARM, float(np.clip(targets[ARM][GRIPPER_INDEX], 0, 1)))

    def advance(self, now: float, states: dict[str, ArmState], targets: dict[str, tuple[float, ...]]) -> CommandSample:
        """Returns the next command sample, with the jaw's next opening until it reached its endpoint or stopped on
        an object."""
        state = states[ARM]
        position = np.array(targets[ARM], dtype=float)
        velocity, acceleration = np.zeros(VECTOR_LENGTH), np.zeros(VECTOR_LENGTH)
        measured_opening = float(state.position[GRIPPER_INDEX])
        self.opening_report = {"initial_opening": self.initial_opening, "final_opening": measured_opening}

        if self.outcome is None:
            candidate_opening, velocity[GRIPPER_INDEX], acceleration[GRIPPER_INDEX], is_finished = (
                self.profile.advance()
            )
            if is_finished:
                candidate_opening = self.profile.endpoint
                if self.finished_at is None:
                    self.finished_at = now
            position[GRIPPER_INDEX] = candidate_opening
            self.profile.accept(candidate_opening)
            if self.finished_at is not None:
                self._judge_endpoint(now, state, measured_opening)

        self.sample_count += 1
        return CommandSample(position, velocity, acceleration, self.outcome is not None, requested_target=position)

    def report(self) -> dict[str, Any]:
        """Returns the jaw's openings, its outcome and the command samples sent, as Home's records keep them."""
        return {"gripper": self.opening_report, "outcome": self.outcome.value, "updates_sent": self.sample_count}

    def _judge_endpoint(self, now: float, state: ArmState, measured_opening: float) -> None:
        """Sets the outcome once the jaw is at its endpoint or, closing, at rest on an object, and raises once it is
        neither past the settle timeout."""
        is_opening = self.opens_by_arm[ARM]

        if abs(measured_opening - self.profile.endpoint) <= self.robot_config.gripper_tolerance:
            self.outcome = GripperOutcome.OPEN if is_opening else GripperOutcome.CLOSED
        elif not is_opening and self._is_at_rest(state):
            # without force sensing, a jaw closing on an object ends once still; its command stays closed, so the
            # servo keeps squeezing within its torque limit
            self.outcome = GripperOutcome.STOPPED_BEFORE_CLOSED
            self.blocked_gripper_openings = {ARM: measured_opening}
        elif now - self.finished_at > self.control_config.settle_timeout_s:
            direction = "opening" if is_opening else "closing"
            raise GripperEndpointTimeout(f"Gripper {direction} endpoint was not reached")

    def _is_at_rest(self, state: ArmState) -> bool:
        """Whether the jaw has stayed still for a whole settle window of fresh readings."""
        window, duration_s = self.rest_window, self.control_config.settle_window_s
        window.add(state.sampled_monotonic_s, state.sample_id, float(state.position[GRIPPER_INDEX]))

        samples = window.samples
        if not SampleWindow.is_complete(samples, duration_s, self.control_config.rate_hz):
            return False

        return float(np.ptp([sample.value for sample in samples])) <= self.robot_config.gripper_tolerance
