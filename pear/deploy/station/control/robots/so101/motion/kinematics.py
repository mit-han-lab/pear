import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.motion.kinematics import BaseKinematics, UnreachablePoseError
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX, JOINTS, LAYOUT
from pear.deploy.station.control.robots.so101.motion.actions import ModelState, SingleArmDelta

# lerobot (and placo under it) is imported where the kinematics load: it is heavy, and joint mode needs none of it


class SO101Kinematics(BaseKinematics):
    """Forward and inverse kinematics on the CPU, on the package's description of the arm with the station's joint
    limits; it never opens hardware."""

    ARM_QUANTITIES = ("x", "y", "z", "pitch", "roll")  # what the model reads and changes of the arm, in this order
    DECIMALS = {"x": 1, "y": 1, "z": 1, "pitch": 2, "roll": 2, "width": 1}  # of each quantity the model reads
    END_EFFECTOR_FRAME = "gripper_frame_link"  # the description's frame at the tip of the fixed finger
    SHOULDER_FRAME = "shoulder_link"  # the description's frame that shoulder_pan turns
    UP_AXIS = np.array([0.0, 0, 1])  # the base frame's vertical
    GIL_YIELD_S = 0.002  # the longest stretch of a solve before the other threads may take the GIL
    CONVERGENCE_RAD = 1e-7  # the largest joint change of a step of a solve that has converged
    IK_WEIGHT = 1.0  # the position's and the orientation's weight in a step of placo
    # one loaded model per process and settings: placo loads the description in one call that holds the GIL, which
    # must not repeat at every enable
    _instances: dict[tuple[Any, ...], "SO101Kinematics"] = {}  # the loaded kinematics by what they were loaded from
    _instances_lock = threading.Lock()  # held while one is looked up or loaded

    def __init__(self, robot_config: SO101Config) -> None:
        from lerobot.model.kinematics import RobotKinematics

        self.robot_config, self.geometry = robot_config, robot_config.geometry
        self.model = RobotKinematics(str(self.geometry.urdf), self.END_EFFECTOR_FRAME, list(JOINTS))

        placo_robot = self.model.robot
        limits = robot_config.limits
        for index, name in enumerate(JOINTS):
            placo_robot.set_joint_limits(name, limits.lower_bounds[index], limits.upper_bounds[index])
        self.model.solver.enable_joint_limits(True)

        self._lock = threading.Lock()  # placo's model holds one pose at a time: one caller uses it at a time

        # the arm plane's horizontal direction, fixed in the shoulder's frame: +x at zero pan
        self._frame(np.zeros(len(JOINTS)))
        shoulder_pose = placo_robot.get_T_world_frame(self.SHOULDER_FRAME)
        self._plane_axis = shoulder_pose[:3, :3].T @ np.array([1.0, 0, 0])

    @classmethod
    def shared_instance(cls, robot_config: SO101Config) -> "SO101Kinematics":
        """Returns the kinematics of ``robot_config``, loaded once per process; their own lock serializes the
        callers."""
        limits = robot_config.limits
        key = (
            repr(robot_config.geometry),
            tuple(limits.lower_bounds),
            tuple(limits.upper_bounds),
            robot_config.feedback_position_tolerance,
        )

        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(robot_config)
            return cls._instances[key]

    def pose(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Returns the model's quantities of the arm, unrounded (cm and rad), at the measured ``joint_positions``."""
        with self._lock:
            return self._pose_values(self._frame(joint_positions), self._plane_direction())

    def quantities(self, joint_positions: Sequence[float]) -> ModelState:
        """Returns what the model reads of the measured ``joint_positions``, rounded the way it is shown."""
        arm = {
            name: round(float(value), self.DECIMALS[name]) + 0.0
            for name, value in zip(self.ARM_QUANTITIES, self.pose(joint_positions))
        }
        closed_width, open_width = self.geometry.gripper_width_cm
        opening = float(np.clip(joint_positions[GRIPPER_INDEX], 0, 1))
        width = round(closed_width + opening * (open_width - closed_width), self.DECIMALS["width"]) + 0.0

        return ModelState.model_validate({"arm": arm, "gripper": {"width": width}})

    def joint_target(self, joint_positions: Sequence[float], action: SingleArmDelta) -> np.ndarray:
        """Returns the joint goal of the pose delta ``action`` from ``joint_positions``; the gripper's coordinate is left
        as it is."""
        delta = np.array([getattr(action.arm, name) for name in self.ARM_QUANTITIES], dtype=float)
        if np.max(abs(delta[3:])) > self.geometry.max_rotation_rad:
            raise ValueError("Rotation increment exceeds the planning bound")

        joint_positions = np.asarray(joint_positions, dtype=float)
        goal = joint_positions.copy()
        if not delta.any():
            return goal

        with self._lock:
            try:
                goal[:GRIPPER_INDEX] = self._solve(joint_positions, delta)
            except UnreachablePoseError:
                fraction = self.reachable_fraction(
                    lambda tried_fraction: self._solve(joint_positions, tried_fraction * delta)
                )
                raise UnreachablePoseError(self.unreachable_message(delta, fraction)) from None

        limits = self.robot_config.joint_limits()
        limits.check_positions(goal, LAYOUT, limits.feedback_tolerance)
        return goal

    @classmethod
    def unreachable_message(cls, requested_delta: np.ndarray, fraction: float) -> str:
        """Returns what the model is told of how much of the requested increment the arm can reach."""
        changes = ", ".join(
            f"{name} {cls.truncated_text(fraction * value, cls.DECIMALS[name])} of {value:+g}"
            for name, value in zip(cls.ARM_QUANTITIES, requested_delta)
            if value
        )

        return (
            "The target is out of reach. Along the requested change the arm can go "
            f"at most {math.floor(100 * fraction)}% ({changes})."
        )

    def _solve(self, joint_positions: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Returns the arm joints that reach ``delta`` from ``joint_positions``, iterating placo from the measured
        joints, and raises if the final pose misses."""
        geometry = self.geometry
        seed = np.asarray(joint_positions[: len(JOINTS)], dtype=float)
        frame = self._frame(seed)
        target_pose_at = self._target_pose_function(frame, delta)
        solution = seed.copy()
        yielded_at = time.monotonic()

        for _ in range(geometry.ik_iterations):
            goal = target_pose_at()
            next_solution = np.radians(
                self.model.inverse_kinematics(np.degrees(solution), goal, self.IK_WEIGHT, self.IK_WEIGHT)
            )
            has_converged = np.max(abs(next_solution - solution)) < self.CONVERGENCE_RAD
            solution = next_solution
            if has_converged:
                break
            # back-to-back placo calls starve the other threads of the GIL: a yield keeps the control loop at its rate
            # while a solve runs in the caller's thread
            if time.monotonic() - yielded_at > self.GIL_YIELD_S:
                time.sleep(0)
                yielded_at = time.monotonic()

        reached_frame = self._frame(solution)
        goal = target_pose_at()
        position_error = float(np.linalg.norm(reached_frame[:3, 3] - goal[:3, 3]))
        if position_error > geometry.position_residual_m:
            raise UnreachablePoseError(f"IK position error {1000 * position_error:.1f} mm")
        angle_error = self.rotation_distance(reached_frame[:3, :3], goal[:3, :3])
        if angle_error > geometry.rotation_residual_rad:
            raise UnreachablePoseError(f"IK orientation error {angle_error:.3f} rad")

        return solution

    def _target_pose_function(self, frame: np.ndarray, delta: np.ndarray) -> Callable[[], np.ndarray]:
        """Returns the function of placo's current iterate that returns the 4x4 target pose of ``delta`` from the
        pose ``frame``."""
        # the model's positions are cm, the description's and placo's m
        position = frame[:3, 3] + np.asarray(delta[:3]) / 100
        current_quantities = self._pose_values(frame, self._plane_direction())
        pitch, roll = current_quantities[3] + delta[3], current_quantities[4] + delta[4]

        def target_pose_at() -> np.ndarray:
            """Returns the target pose, its pointing direction in the arm plane of placo's current iterate."""
            pose = np.eye(4)
            pose[:3, :3] = self._pose_rotation(self._plane_direction(), pitch, roll)
            pose[:3, 3] = position
            return pose

        return target_pose_at

    def _frame(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Returns the pose of the fingertip's frame at the arm joints of ``joint_positions``, in rad."""
        return np.array(self.model.forward_kinematics(np.degrees(np.asarray(joint_positions[: len(JOINTS)]))))

    def _plane_direction(self) -> np.ndarray:
        """Returns the unit horizontal direction of the arm plane at the joints last set in placo."""
        shoulder_rotation = self.model.robot.get_T_world_frame(self.SHOULDER_FRAME)[:3, :3]
        plane_direction = shoulder_rotation @ self._plane_axis
        plane_direction[2] = 0.0

        return plane_direction / np.linalg.norm(plane_direction)

    def _pose_values(self, frame: np.ndarray, plane_direction: np.ndarray) -> np.ndarray:
        """Returns the model's quantities of the fingertip's pose ``frame``: the position in cm, the pitch and the
        roll."""
        # the pointing direction lies in the vertical plane that shoulder_pan sets: the pitch is its angle below
        # horizontal, and the roll follows wrist_roll
        pointing_axis, lateral_axis = frame[:3, 2], frame[:3, 1]
        pitch = math.atan2(-pointing_axis[2], float(pointing_axis @ plane_direction))
        horizontal_normal = np.cross(self.UP_AXIS, plane_direction)
        roll = math.atan2(
            float(np.cross(horizontal_normal, lateral_axis) @ -pointing_axis), float(horizontal_normal @ lateral_axis)
        )

        return np.r_[100 * frame[:3, 3], pitch, roll]  # the base frame's metres as the model's centimetres

    def _pose_rotation(self, plane_direction: np.ndarray, pitch: float, roll: float) -> np.ndarray:
        """Returns the fingertip's rotation that points ``pitch`` below horizontal in the arm plane, rolled by
        ``roll``."""
        pointing_axis = math.cos(pitch) * plane_direction - math.sin(pitch) * self.UP_AXIS
        horizontal_normal = np.cross(self.UP_AXIS, plane_direction)
        lateral_axis = horizontal_normal * math.cos(roll) + np.cross(-pointing_axis, horizontal_normal) * math.sin(roll)

        return np.column_stack([np.cross(lateral_axis, pointing_axis), lateral_axis, pointing_axis])
