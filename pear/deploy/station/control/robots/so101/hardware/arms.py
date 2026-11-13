import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.hardware.arm_state import ServoFault, SO101ArmState
from pear.deploy.station.control.robots.so101.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX, JOINTS, MOTORS, VECTOR_LENGTH
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class SO101Hardware:
    """The SO-101 follower arm through lerobot's Feetech bus, read and commanded in rad and as the gripper's opening."""

    ENABLE_ATTEMPTS = 3  # how many times enabling the arm is tried
    BUS_RETRIES = 2  # per register access while enabling and checking
    RUN_RETRIES = 1  # per bus access while running

    def __init__(self, control_config: ControlConfig, robot_config: SO101Config) -> None:
        robot_config.require_commissioned()

        hardware_config = robot_config.hardware
        self.control_config = control_config
        self.joint_offsets = np.array(hardware_config.joint_offsets, dtype=float)
        self.status_poll_ticks = hardware_config.status_poll_ticks
        self.read_count = 0
        self.previous_read: tuple[float, np.ndarray] | None = None  # the time and the position of the last read
        self.servo_status: dict[str, Any] = {}  # the servos' torque and Status as last polled
        self.is_closed = False

        self.bus_lock = StartupChecks.lock_bus(hardware_config.port)
        try:
            self.follower = StartupChecks.make_follower(hardware_config)
            StartupChecks.connect_checked(self.follower)
            try:
                self._enable()
                self._check_servos()
            except BaseException:
                self.follower.bus.disconnect(disable_torque=True)
                raise
        except BaseException:
            self.bus_lock.close()
            raise

    def read(self) -> SO101ArmState:
        """Returns the arm's reading, polling the servos' status at a low rate."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")

        values = self.follower.bus.sync_read("Present_Position", num_retry=self.RUN_RETRIES)
        now = time.monotonic()
        position = self.bus_position(values, self.joint_offsets)

        velocity = np.zeros(VECTOR_LENGTH)
        if self.previous_read is not None:
            elapsed_s = now - self.previous_read[0]
            if 0 < elapsed_s <= self.control_config.state_timeout_s:
                velocity = (position - self.previous_read[1]) / elapsed_s
        self.previous_read = (now, position)

        self.read_count += 1
        self._poll()
        return SO101ArmState(tuple(position), tuple(velocity), now, self.read_count, dict(self.servo_status))

    def read_for_recovery(self) -> SO101ArmState:
        """Returns the arm's reading after checking that every servo is healthy and still holds torque."""
        self._check_servos()

        return self.read()

    def send(self, target: Sequence[float]) -> None:
        """Commands the joint vector ``target``."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")

        target_vector = np.asarray(target, dtype=float)
        if target_vector.shape != (VECTOR_LENGTH,) or not np.isfinite(target_vector).all():
            raise ValueError("Expected a finite 6D target")
        angles_deg = np.degrees(target_vector[:GRIPPER_INDEX] - self.joint_offsets)
        goal = {joint: float(angle_deg) for joint, angle_deg in zip(JOINTS, angles_deg)}
        goal["gripper"] = float(target_vector[GRIPPER_INDEX] * 100)  # the bus takes the gripper's opening in percent

        self.follower.bus.sync_write("Goal_Position", goal, num_retry=self.RUN_RETRIES)

    def close(self) -> None:
        """Disables torque and releases the bus, so that the arm goes limp."""
        if self.is_closed:
            return

        self.is_closed = True
        try:
            self.follower.disconnect()
        finally:
            self.bus_lock.close()

    @staticmethod
    def bus_position(values: Mapping[str, float], joint_offsets: np.ndarray) -> np.ndarray:
        """Returns the joint vector of the bus's position readings: the joints in rad with their offsets, then the
        gripper's opening."""
        joint_positions = np.radians([values[joint] for joint in JOINTS]) + joint_offsets

        return np.append(joint_positions, values["gripper"] / 100)  # the bus reports the gripper's opening in percent

    def _enable(self) -> None:
        """Holds the current pose, the goal verified before torque goes on."""
        # after power-up the goal registers read 0 and lerobot's connect() enables torque without setting them; a sync
        # write gets no reply, so the goal is read back: a servo that missed it would jump to 0. A retry may briefly
        # release torque, as configure() does
        bus = self.follower.bus

        for attempt in range(1, self.ENABLE_ATTEMPTS + 1):
            try:
                present_positions = bus.sync_read("Present_Position", normalize=False, num_retry=self.BUS_RETRIES)
                bus.sync_write("Goal_Position", present_positions, normalize=False, num_retry=self.BUS_RETRIES)
                goal_positions = bus.sync_read("Goal_Position", normalize=False, num_retry=self.BUS_RETRIES)
                if goal_positions != present_positions:
                    raise ConnectionError(f"Goal_Position read back {goal_positions}, wrote {present_positions}")
                self.follower.configure()  # the servos' settings; it ends by enabling torque
                return
            except ConnectionError as error:
                if attempt == self.ENABLE_ATTEMPTS:
                    raise
                logger.warning("Enable attempt %d failed, retrying: %s", attempt, error)

    def _check_servos(self) -> None:
        """Polls the servos' torque and Status, and raises ServoFault unless every servo holds torque and reports no
        error."""
        bus = self.follower.bus
        torque_enabled = bus.sync_read("Torque_Enable", normalize=False, num_retry=self.BUS_RETRIES)
        status = bus.sync_read("Status", normalize=False, num_retry=self.BUS_RETRIES)
        self.servo_status = {"torque": torque_enabled, "status": status, "polled_monotonic_s": time.monotonic()}

        self._check_torque(torque_enabled)
        self._check_status(status)

    def _poll(self) -> None:
        """Polls the servos' Status or their torque when it is their turn, interleaved with the position reads."""
        phase = self.read_count % self.status_poll_ticks
        bus = self.follower.bus

        if phase == 0:
            status = bus.sync_read("Status", normalize=False, num_retry=self.RUN_RETRIES)
            self.servo_status = {**self.servo_status, "status": status, "polled_monotonic_s": time.monotonic()}
            self._check_status(status)
        elif phase == self.status_poll_ticks // 2:  # half a poll cycle after the Status
            torque_enabled = bus.sync_read("Torque_Enable", normalize=False, num_retry=self.RUN_RETRIES)
            self.servo_status = {**self.servo_status, "torque": torque_enabled, "polled_monotonic_s": time.monotonic()}
            self._check_torque(torque_enabled)

    @staticmethod
    def _check_torque(torque_enabled: Mapping[str, int]) -> None:
        """Raises ServoFault unless every servo holds torque."""
        if motors_without_torque := [motor for motor in MOTORS if torque_enabled[motor] != 1]:
            raise ServoFault(f"Servo torque is off: {', '.join(motors_without_torque)}")

    @staticmethod
    def _check_status(status: Mapping[str, int]) -> None:
        """Raises ServoFault if a servo's Status reports an error that must stop the arm."""
        if faults := StartupChecks.servo_faults(status):
            raise ServoFault("Servo error: " + "; ".join(faults))
