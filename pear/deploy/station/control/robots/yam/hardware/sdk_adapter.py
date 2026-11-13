from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.feedforward import FrictionFeedforward
from pear.deploy.station.control.robots.yam.config import ArmConfig, YAMConfig
from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")

# the frames of the motors' MIT protocol: seven 0xFF bytes, then the byte that says what the frame does
MIT_FRAME_HEAD = b"\xff" * 7
MIT_MOTOR_ON = 0xFC
MIT_MOTOR_OFF = 0xFD


class SDKCompatibility:
    """The SDK adapted inside the worker: its classes take the methods of the adapters below, and a robot whose
    build fails powers off what it enabled."""

    def __init__(self, robot_config: YAMConfig) -> None:
        self.robot_config = robot_config
        self.arms_by_channel = {arm_config.channel: arm_config for arm_config in robot_config.arms.values()}
        # the SDK's chains and robots of the robot being built, until it is built
        self.constructing_chains: list[Any] = []
        self.constructing_robots: list[Any] = []

    def install(self) -> Callable[..., Any]:
        """Gives the SDK's classes the adapters' methods and replaces its startup check, and returns the factory of
        one robot."""
        # the SDK: imported where it is adapted, in the worker's process
        from i2rt.motor_drivers import dm_driver
        from i2rt.robots import get_robot, motor_chain_robot
        from i2rt.robots.utils import GripperForceLimiter

        dm_driver.run_startup_checks = self.skip_checks
        dm_driver.DMSingleMotorCanInterface = self.adapted(StationMotor, dm_driver.DMSingleMotorCanInterface)
        get_robot.DMChainCanInterface = self.adapted(StationChain, get_robot.DMChainCanInterface, compatibility=self)
        get_robot.MotorChainRobot = self.adapted(StationRobot, get_robot.MotorChainRobot, compatibility=self)
        motor_chain_robot.GripperForceLimiter = self.adapted(ReportingForceLimiter, GripperForceLimiter)

        return self.create_robot

    def create_robot(self, **robot_arguments: Any) -> Any:
        """Builds one SDK robot; a failed build disables the motors of every handle it built in part."""
        from i2rt.robots import get_robot

        try:
            return get_robot.get_yam_robot(**robot_arguments)
        except BaseException as error:
            # a failed factory returns no robot: the handles it built in part are kept until their motors are off
            errors = []
            for robot in reversed(self.constructing_robots):
                errors.extend(SDKShutdown.stop_robot_writer(robot))
            for chain in reversed(self.constructing_chains):
                errors.extend(SDKShutdown.close_chain(chain))
            if errors:
                error.add_note("Startup cleanup incomplete: " + "; ".join(errors))
            raise
        finally:
            self.constructing_robots.clear()
            self.constructing_chains.clear()

    @staticmethod
    def adapted(adapter: type, sdk_class: type, **attributes: Any) -> type:
        """Returns a class of ``sdk_class`` whose methods are first the adapter's, with ``attributes``."""
        return type(adapter.__name__, (adapter, sdk_class), attributes)

    @staticmethod
    def skip_checks(*args: Any, **kwargs: Any) -> None:
        """Does nothing in the place of the SDK's own startup check, whose repair is a flash write."""
        # StartupChecks audited both arms read-only before any chain was built; a chain's own call forces repair

    @staticmethod
    def select_joint_branch(
        raw_angle: float, direction: float, lower_limit: float, upper_limit: float, margin: float
    ) -> float:
        """Returns the one encoder branch of ``raw_angle`` within the limits, writing no motor zero."""
        if not np.isfinite([raw_angle, direction, lower_limit, upper_limit, margin]).all() or direction not in (-1, 1):
            raise ValueError("Invalid joint coordinate mapping")

        signed_angle = raw_angle * direction
        min_turns = math.ceil((lower_limit - margin - signed_angle) / (2 * math.pi))
        max_turns = math.floor((upper_limit + margin - signed_angle) / (2 * math.pi))
        if min_turns != max_turns:
            raise ValueError("Startup angle has no unique branch inside configured joint limits")

        return signed_angle + min_turns * 2 * math.pi


class StationMotor:
    """The adapter of the SDK's interface of one motor: a start may reset a motor's communication timeout once."""

    # what the SDK's interface has, which the adapter reads
    channel: str
    _drain_bus: Callable[[], None]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._startup_motor_id: int | None = None  # the motor being enabled
        self._startup_feedback: Any = None  # its feedback while it is enabled
        self._startup_clear_attempted = False  # whether its one reset was spent
        super().__init__(*args, **kwargs)

    def motor_on(self, motor_id: int, motor_type: str) -> Any:
        """Enables the motor ``motor_id``, which may clear its communication timeout once."""
        # the SDK's motor_on changes the root logger's level: the level before it is put back after it
        root_level = logging.getLogger().level
        self._startup_motor_id = motor_id
        self._startup_feedback = None
        self._startup_clear_attempted = False

        try:
            return super().motor_on(motor_id, motor_type)
        finally:
            self._startup_motor_id = None
            self._startup_feedback = None
            logging.getLogger().setLevel(root_level)
            if self._startup_clear_attempted:
                logger.warning("%s Motor %s: attempted one startup communication-timeout reset", self.channel, motor_id)

    def parse_recv_message(self, *args: Any, **kwargs: Any) -> Any:
        """Returns a motor's parsed reply, which is kept while that motor is being enabled."""
        feedback = super().parse_recv_message(*args, **kwargs)
        if self._startup_motor_id == feedback.id:
            self._startup_feedback = feedback

        return feedback

    def clean_error(self, motor_id: int) -> None:
        """Clears the communication timeout of the motor being enabled, once, and refuses every other error."""
        from i2rt.motor_drivers import dm_driver

        # a stopped service leaves the firmware's communication watchdog latched: only a start may clear it, once for
        # each motor
        feedback = self._startup_feedback
        if (
            self._startup_motor_id == motor_id
            and feedback is not None
            and int(feedback.error_code, 16) == dm_driver.MotorErrorCode.loss_communication
            and not self._startup_clear_attempted
        ):
            self._startup_clear_attempted = True
            super().clean_error(motor_id)
            # the SDK's clear sends three frames but motor_on consumes one reply: the rest is drained, so that a
            # buffered disabled status is not taken for the new enable reply
            self._drain_bus()
            return

        detail = (
            f"{feedback.error_code} ({feedback.error_message}); "
            f"MOS {feedback.temperature_mos} C, rotor {feedback.temperature_rotor} C"
            if feedback is not None
            else "no startup feedback"
        )
        raise RuntimeError(
            f"{self.channel} Motor {motor_id}: {detail}; startup refused error clearing "
            "(only one communication-timeout reset is allowed)"
        )


class FeedbackBatch(list):
    """The motors' feedback of one sample, tagged with its number and its time."""


class StationChain:
    """The adapter of the SDK's motor chain: its feedback samples are numbered, and each joint's encoder branch is
    chosen within the robot's limits."""

    compatibility: SDKCompatibility  # set on the class that install makes
    # what the SDK's motor chain has, which the adapter reads and writes
    channel: str
    motor_list: list[tuple[int, str]]
    state: list[Any]
    state_lock: threading.Lock
    absolute_positions: np.ndarray
    motor_direction: np.ndarray
    motor_offset: np.ndarray
    _joint_position_real_to_sim_idx: Callable[[float, int], float]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.compatibility.constructing_chains.append(self)
        self._station_sequence = 0  # the number of the latest complete feedback batch
        self._station_sampled_at = float("-inf")  # its local monotonic time
        super().__init__(*args, **kwargs)

    def start_thread(self) -> Any:
        """Maps each motor to its joint's branch within the robot's limits, then starts the chain's thread."""
        # replaces the factory's blanket +/-pi wrapping before the background control starts, from the unmodified
        # absolute encoder coordinates
        arm_config: ArmConfig = self.compatibility.arms_by_channel[self.channel]
        feedback_tolerance = self.compatibility.robot_config.feedback_position_tolerance

        for motor_index, (motor_id, _) in enumerate(self.motor_list):
            offset = 0.0
            if motor_id == StartupChecks.GRIPPER_MOTOR_ID:
                # a power cycle can return an equivalent angle on another turn: the calibrated branch is recovered
                # without changing the endpoints or writing a motor zero
                closed_position, open_position = arm_config.gripper_limits
                lower_limit, upper_limit = sorted((closed_position, open_position))
                margin = feedback_tolerance * abs(open_position - closed_position)
            else:
                # the limits are true angles: the branch is chosen in the SDK's coordinates
                offset = arm_config.joint_offsets[motor_index]
                lower_limit = arm_config.limits.lower_bounds[motor_index] - offset
                upper_limit = arm_config.limits.upper_bounds[motor_index] - offset
                margin = feedback_tolerance
            joint_angle = offset + SDKCompatibility.select_joint_branch(
                self.absolute_positions[motor_index],
                self.motor_direction[motor_index],
                lower_limit,
                upper_limit,
                margin,
            )
            # every read, command, clip and gravity term of the SDK uses this mapping
            self.motor_offset[motor_index] = (
                self.absolute_positions[motor_index] - joint_angle * self.motor_direction[motor_index]
            )

        return super().start_thread()

    def read_states(self, torques: Any = None) -> FeedbackBatch:
        """Returns the motors' feedback of one sample, tagged with the sample's number and local monotonic time."""
        from i2rt.motor_drivers.utils import MotorInfo

        batch = FeedbackBatch()
        with self.state_lock:
            for motor_index, motor_feedback in enumerate(self.state):
                batch.append(
                    MotorInfo(
                        id=motor_feedback.id,
                        error_code=motor_feedback.error_code,
                        target_torque=torques[motor_index] if torques is not None else 0.0,
                        vel=motor_feedback.velocity * self.motor_direction[motor_index],
                        eff=motor_feedback.torque * self.motor_direction[motor_index],
                        pos=self._joint_position_real_to_sim_idx(self.absolute_positions[motor_index], motor_index),
                        temp_rotor=motor_feedback.temperature_rotor,
                        temp_mos=motor_feedback.temperature_mos,
                        timestamp=time.time(),
                    )
                )
            batch.sample = (self._station_sequence, self._station_sampled_at)

        return batch

    def _update_absolute_positions(self, feedback: Any) -> None:
        """Takes a complete feedback batch, which gets the next number and the local monotonic time."""
        # the SDK calls this when it publishes a complete batch under state_lock, or once at the start, before
        # either of its writers runs
        super()._update_absolute_positions(feedback)
        self._station_sequence += 1
        self._station_sampled_at = time.monotonic()


class StationRobot:
    """The adapter of the SDK's robot: the robot's gains and gravity factors, and its friction feedforward in every
    command."""

    compatibility: SDKCompatibility  # set on the class that install makes
    # what the SDK's robot has, which the adapter reads and writes
    gravity_comp_factor: np.ndarray
    _command_lock: threading.Lock
    _commands: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.compatibility.constructing_robots.append(self)
        arm_config: ArmConfig = self.compatibility.arms_by_channel[kwargs["motor_chain"].channel]
        joint_count = LAYOUT.joint_count
        kwargs["limit_gripper_force"] = arm_config.gripper_force_n
        if arm_config.joint_position_gains is not None:
            # applied before the SDK robot starts its command writer; the gripper's Kp stays
            kwargs["kp"] = np.array(kwargs["kp"], dtype=float, copy=True)
            kwargs["kp"][:joint_count] = arm_config.joint_position_gains
        if arm_config.gravity_compensation_factors is not None:
            # the factory appends the gripper's factor, which stays
            kwargs["gravity_comp_factor"] = np.array(kwargs["gravity_comp_factor"], dtype=float, copy=True)
            kwargs["gravity_comp_factor"][:joint_count] = arm_config.gravity_compensation_factors

        # the SDK starts its control thread inside __init__, and the thread calls update()
        has_friction_feedforward = any(arm_config.friction_feedforward_nm) or any(arm_config.friction_load_ratio)
        self._station_friction = (
            FrictionFeedforward(arm_config.friction_feedforward_nm, arm_config.friction_load_ratio)
            if has_friction_feedforward
            else None
        )
        self._station_planned_velocity = np.zeros(joint_count)
        self._station_friction_torque = np.zeros(joint_count)
        self._station_gravity_torque: np.ndarray | None = None  # of the latest control tick
        super().__init__(*args, **kwargs)

    def update(self) -> None:
        """Runs one tick of the SDK's control thread with the friction feedforward in its command."""
        self._station_apply_friction()
        super().update()

    def _station_apply_friction(self) -> None:
        """Writes the friction feedforward into the torques of the live command."""
        # the SDK adds a command's torques to its gravity compensation and clips the sum; a command swapped in later
        # starts from the last torque, so that no tick drops it
        if self._station_friction is None:
            return

        with self._command_lock:
            planned_velocity = self._station_planned_velocity
        # the gravity torque is the previous control tick's
        friction_torque = self._station_friction.update(
            planned_velocity, time.monotonic(), self._station_gravity_torque
        )
        with self._command_lock:
            self._station_friction_torque = friction_torque
            torques = np.zeros(len(self._commands.torques))
            torques[: LAYOUT.joint_count] = friction_torque
            self._commands.torques = torques

    def _compute_gravity_compensation(self, joint_state: Any) -> Any:
        """Returns the SDK's gravity compensation, and keeps the gravity torque it applies to the joints."""
        # the friction feedforward grows with this load
        gravity = super()._compute_gravity_compensation(joint_state)
        factors = np.asarray(self.gravity_comp_factor, dtype=float)
        self._station_gravity_torque = (gravity * factors)[: LAYOUT.joint_count].copy()

        return gravity

    def _motor_state_to_joint_state(self, motor_state: Any) -> Any:
        """Returns the SDK's joint state of a feedback batch, tagged with the batch's number and time."""
        joint_state = super()._motor_state_to_joint_state(motor_state)
        joint_state.station_sample = motor_state.sample

        return joint_state


class ReportingForceLimiter:
    """The adapter of the SDK's gripper force limiter: it also keeps the torque it last requested."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # the SDK's holding torque, its friction compensation included; None until the first request
        self.station_target_effort_nm: float | None = None
        super().__init__(*args, **kwargs)

    def compute_target_gripper_torque(self, gripper_state: Any) -> Any:
        """Returns the SDK's torque request for the gripper, which is kept; reading telemetry never updates the
        limiter."""
        target_effort_nm = super().compute_target_gripper_torque(gripper_state)
        self.station_target_effort_nm = target_effort_nm

        return target_effort_nm


class OneAttemptInterface:
    """The adapter of the SDK's interface of one motor for the gripper's calibration: each request is sent once, and
    no error is ever cleared."""

    MAX_RETRY = 1  # the SDK's sends of one request

    def _send_message_get_response(self, id: int, motor_id: int, data: Any, **kwargs: Any) -> Any:
        """Sends one request to a motor once and returns its reply; ``id`` is the SDK's name of the frame's id."""
        kwargs["max_retry"] = self.MAX_RETRY
        return super()._send_message_get_response(id, motor_id, data, **kwargs)

    def clean_error(self, motor_id: int) -> None:
        """Refuses to clear a motor's error."""
        raise RuntimeError("Calibration refuses automatic motor recovery")


class SDKShutdown:
    """The power-off of the SDK's robots: their writers stopped, every motor sent the motor-off frame, and the CAN
    socket closed."""

    WRITER_JOIN_S = 0.5  # for an SDK robot's command writer to stop
    LOCK_TIMEOUT_S = 0.5  # for the lock of a chain's CAN writer
    DISABLE_REPEATS = 3  # sends of every motor's motor-off frame
    SEND_TIMEOUT_S = 0.02  # of one frame
    DISABLE_FRAME = [*MIT_FRAME_HEAD, MIT_MOTOR_OFF]  # the motor-off frame

    @classmethod
    def stop_robot_writer(cls, robot: Any) -> list[str]:
        """Stops the command writer of an SDK robot and returns what failed."""
        errors = []

        event, thread = getattr(robot, "_stop_event", None), getattr(robot, "_server_thread", None)
        if event is not None:
            event.set()
        if thread is not None:
            try:
                thread.join(timeout=cls.WRITER_JOIN_S)
                if thread.is_alive():
                    errors.append("SDK robot writer did not stop before close")
            except RuntimeError as error:
                logger.exception("An SDK robot's command writer could not be joined")
                errors.append(str(error))

        return errors

    @classmethod
    def close_chain(cls, chain: Any) -> list[str]:
        """Sends every motor of a chain its motor-off frame, closes the chain's CAN socket and returns what failed."""
        import can  # the CAN library: imported where a chain closes, in the worker's process

        chain.running = False
        driver = getattr(chain, "motor_interface", None)
        if driver is None:
            return []  # no transport was opened, so no motor could have been enabled

        # a chain whose _motor_on() failed has no command_lock yet
        lock = getattr(chain, "command_lock", None)
        lock_acquired = lock is None or lock.acquire(timeout=cls.LOCK_TIMEOUT_S)
        errors = []
        try:
            if not lock_acquired:
                errors.append("SDK CAN writer did not stop before close")
                return errors
            # motor_off() waits for replies and can stall during a shutdown: these are its frames, each sent within a
            # time limit, with no wait for a reply
            for _ in range(cls.DISABLE_REPEATS):
                for motor_id, _ in chain.motor_list:
                    try:
                        driver.bus.send(
                            can.Message(
                                arbitration_id=driver._get_frame_id(motor_id),
                                data=cls.DISABLE_FRAME,
                                is_extended_id=False,
                            ),
                            timeout=cls.SEND_TIMEOUT_S,
                        )
                    except Exception as error:
                        logger.exception("The motor-off frame of motor %s was not sent", motor_id)
                        errors.append(f"motor {motor_id}: {error}")
        finally:
            try:
                chain.close()
            except Exception as error:
                logger.exception("A chain's CAN socket did not close")
                errors.append(f"CAN close: {error}")
            finally:
                if lock_acquired and lock is not None:
                    lock.release()

        return errors
