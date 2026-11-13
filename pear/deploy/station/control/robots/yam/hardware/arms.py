from typing import Any

import numpy as np

from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.hardware.sdk_adapter import SDKCompatibility, SDKShutdown
from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class YAMHardware:
    """Both arms through the SDK, built once the read-only startup checks pass: their readings, their commands and
    their power-off."""

    def __init__(self, robot_config: YAMConfig) -> None:
        # the SDK: imported where the arms are built, in the worker's process; the robot's driver refused an
        # uncommissioned configuration before that process was started
        from i2rt.robots.utils import ArmType, GripperType

        self.robot_config = robot_config
        self.robots: dict[str, Any] = {}  # each arm's SDK robot, by the arm's name
        self.device_lock = StartupChecks.run(robot_config)

        try:
            create_robot = SDKCompatibility(robot_config).install()
            for arm in LAYOUT.arms:
                arm_config = robot_config.arms[arm]
                self.robots[arm] = create_robot(
                    channel=arm_config.channel,
                    arm_type=ArmType(arm_config.arm_type),
                    gripper_type=GripperType(arm_config.gripper_type),
                    zero_gravity_mode=False,
                    gripper_limits_override=np.array(arm_config.gripper_limits),
                    enable_auto_recovery=False,
                )
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, YAMArmState]:
        """Returns each arm's latest reading, with the SDK's view of its gripper."""
        states = {}

        for arm, robot in self.robots.items():
            # a stopped SDK thread freezes station_sample: the feedback's age then faults
            with robot._state_lock:
                joint_state = robot._joint_state
                sequence, sampled_monotonic_s = joint_state.station_sample
                closed_position, open_position = robot._gripper_limits
                states[arm] = YAMArmState(
                    position=tuple(joint_state.pos),
                    velocity=tuple(joint_state.vel),
                    sampled_monotonic_s=sampled_monotonic_s,
                    effort=tuple(joint_state.eff),
                    sample_id=sequence,
                    sdk_gripper_target=float(
                        (robot._last_gripper_command_qpos - closed_position) / (open_position - closed_position)
                    ),
                    sdk_gripper_force_limited=bool(robot._gripper_force_limiter._is_clogged),
                    sdk_gripper_force_limit_n=float(robot._limit_gripper_force),
                    sdk_gripper_target_effort_nm=getattr(
                        robot._gripper_force_limiter, "station_target_effort_nm", None
                    ),
                )

        return states

    def read_for_recovery(self) -> dict[str, YAMArmState]:
        """Returns each arm's reading once the SDK's loops are found running and every motor without an error; it
        never clears an error, restarts a loop or enables a motor."""
        from i2rt.motor_drivers.dm_driver import MotorErrorCode

        if set(self.robots) != set(LAYOUT.arms):
            raise RuntimeError("Recovery blocked: both SDK arms must be connected")
        for arm, robot in self.robots.items():
            if not robot.motor_chain.running:
                raise RuntimeError(f"Recovery blocked: {arm} motor communication loop stopped")
            if robot._stop_event.is_set() or not robot._server_thread.is_alive():
                raise RuntimeError(f"Recovery blocked: {arm} SDK control thread stopped")
            with robot.motor_chain.state_lock:
                motor_states = list(robot.motor_chain.state)
            if len(motor_states) != LAYOUT.coordinate_count:
                raise RuntimeError(f"Recovery blocked: {arm} motor feedback is incomplete")
            for motor_state in motor_states:
                error_code = motor_state.error_code
                error_code = int(error_code, 16) if isinstance(error_code, str) else int(error_code)
                if error_code != MotorErrorCode.normal:
                    raise RuntimeError(
                        f"Recovery blocked: {arm} motor {motor_state.id}: "
                        + MotorErrorCode.get_error_message(error_code)
                    )

        return self.read()

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Commands each arm's target with its planned velocity; an arm that fails leaves the others commanded."""
        errors = []

        for arm, robot in self.robots.items():
            try:
                velocity = None if velocities is None else velocities[arm]
                self.command_position_velocity(
                    robot, targets[arm], velocity, self.robot_config.arms[arm].limits.velocity
                )
            except Exception as error:
                logger.exception("The command of the %s arm failed", arm)
                errors.append(f"{arm}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        """Powers both arms off and releases their buses; raises what failed once both were tried."""
        errors = []

        try:
            for arm, robot in self.robots.items():
                try:
                    shutdown_errors = SDKShutdown.stop_robot_writer(robot) + SDKShutdown.close_chain(robot.motor_chain)
                    if shutdown_errors:
                        raise RuntimeError("Motor shutdown incomplete: " + "; ".join(shutdown_errors))
                except Exception as error:
                    logger.exception("SDK close failed for %s", arm)
                    errors.append(f"{arm}: {error}")
        finally:
            self.robots.clear()
            self.device_lock.close()

        if errors:
            raise RuntimeError("; ".join(errors))

    @staticmethod
    def command_position_velocity(
        robot: Any, position: tuple[float, ...], velocity: tuple[float, ...] | None, velocity_limit: list[float]
    ) -> None:
        """Writes one MIT-mode command of each joint of an arm: its position, its arm's velocity feedforward and the
        SDK's gains."""
        # the planned velocity makes Kd damp only the deviation from the plan, where command_joint_pos (v_des = 0)
        # brakes the arm into friction; the command is built whole and swapped in under the lock, because the SDK's
        # command_joint_state publishes an all-zero command before it takes one
        from i2rt.robots.motor_chain_robot import JointCommands

        coordinate_count, joint_count = LAYOUT.coordinate_count, LAYOUT.joint_count
        requested_positions = np.array(position, dtype=float)
        feedforward = np.zeros(coordinate_count) if velocity is None else np.array(velocity, dtype=float)
        if requested_positions.shape != (coordinate_count,) or feedforward.shape != (coordinate_count,):
            # a fault's text, which the model may read: its words stay as they are
            raise ValueError("Expected seven position and velocity coordinates")
        if not np.isfinite(requested_positions).all() or not np.isfinite(feedforward).all():
            raise ValueError("Command must be finite")

        # the gripper's velocity stays 0: the SDK's force limiter regulates its position alone
        joint_velocity_limit = np.asarray(velocity_limit, dtype=float)[:joint_count]
        feedforward[:joint_count] = np.clip(feedforward[:joint_count], -joint_velocity_limit, joint_velocity_limit)
        feedforward[joint_count] = 0.0
        # the SDK clips the arm's joints in place: the requested positions stay as they are, for the comparison
        clipped_positions = robot._clip_robot_joint_pos_command(requested_positions.copy())
        is_clipped = clipped_positions[:joint_count] != requested_positions[:joint_count]
        feedforward[:joint_count][is_clipped] = 0.0

        commands = JointCommands.init_all_zero(len(robot.motor_chain))
        commands.pos = robot.remapper.to_robot_joint_pos_space(clipped_positions)
        commands.vel = robot.remapper.to_robot_joint_vel_space(feedforward)
        commands.kp = robot._kp
        commands.kd = robot._kd
        planned_velocity = feedforward[:joint_count].copy()
        with robot._command_lock:
            # the friction feedforward follows the planned velocity: until the SDK's control thread next updates it,
            # the new command keeps the last torque
            commands.torques[:joint_count] = robot._station_friction_torque
            robot._station_planned_velocity = planned_velocity
            robot._commands = commands
