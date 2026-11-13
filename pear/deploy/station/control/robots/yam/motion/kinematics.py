from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from pear.deploy.station.control.motion.kinematics import BaseKinematics, UnreachablePoseError
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.formats import ARM_QUANTITIES, DECIMALS
from pear.deploy.station.control.robots.yam.layout import ARM_TYPE, GRIPPER_TYPE, LAYOUT
from pear.deploy.station.control.robots.yam.motion.actions import CartesianDelta, ModelState


class ToolPose(NamedTuple):
    """A tool's pose in the robot's frame: centimeters and fixed-axis XYZ roll, pitch and yaw in radians."""

    position_cm: list[float]
    rpy_rad: list[float]


class YAMKinematics(BaseKinematics):
    """Offline forward and inverse kinematics of both arms on the SDK's model; it never opens hardware."""

    # the model's translations are meters, the actions' centimeters; a rotation increment adds to the fixed-axis XYZ
    # roll, pitch and yaw; the tool's frame: x along the fingers' closing (left), y down, z against the approach
    GRASP_TO_TOOL = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
    MODEL_DOF = 8  # the model's joints: an arm's six and the gripper's two fingers
    SITE = "grasp_site"  # the model's site the kinematics solve for
    SOLVE_THRESHOLD_FRACTION = 0.5  # of a residual the robot allows, the solver's own threshold
    GIMBAL_MARGIN = 1e-9  # the distance of sin(pitch) from +-1 under which roll is not defined
    UNREACHABLE_ERRORS = (ValueError,)  # a solve that fails in any way leaves the increment out of reach

    def __init__(self, robot_config: YAMConfig) -> None:
        # the model's libraries: imported where kinematics are made, in the robot's process and the planner's
        import mink
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        self.robot_config = robot_config
        self.models: dict[str, Any] = {}  # each arm's model, by the arm's name
        self.limits: dict[str, list[Any]] = {}  # the solver's limits of each arm's model

        for arm in LAYOUT.arms:
            arm_config = robot_config.arms[arm]
            if (arm_config.arm_type, arm_config.gripper_type) != (ARM_TYPE, GRIPPER_TYPE):
                # a planning error's text, which the model may read: its words stay as they are
                raise ValueError("These kinematics are verified only for Ultra v1 + linear_4310")
            model = Kinematics(combine_arm_and_gripper_xml(ArmType(ARM_TYPE), GripperType(GRIPPER_TYPE)), self.SITE)
            mujoco_model = model._configuration.model
            if mujoco_model.nq != self.MODEL_DOF or mujoco_model.nv != self.MODEL_DOF:
                raise ValueError("Expected six arm and two finger joints")

            # the robot's own joint bounds, which are stricter than the model's margins
            joint_count = LAYOUT.joint_count
            mujoco_model.jnt_range[:joint_count, 0] = arm_config.limits.lower_bounds[:joint_count]
            mujoco_model.jnt_range[:joint_count, 1] = arm_config.limits.upper_bounds[:joint_count]
            self.models[arm] = model
            self.limits[arm] = [mink.ConfigurationLimit(mujoco_model)]

    def quantities(self, joint_positions: Sequence[float]) -> ModelState:
        """Returns the four parts the model reads of the measured joints, rounded the way they are shown."""
        # a gripper's width comes from its measured opening and the model's finger travel: an estimate, not a
        # measured aperture
        joint_positions = np.asarray(joint_positions, dtype=float)
        tool_poses = self.tool_poses(joint_positions)
        coordinate_count = LAYOUT.coordinate_count
        values = {}

        for arm in LAYOUT.arms:
            tool_pose = tool_poses[arm]
            pose_values = [*tool_pose.position_cm, *tool_pose.rpy_rad]
            # + 0.0: a rounded -0.0 becomes 0.0, which the model reads without a sign
            values[arm + "_arm"] = {
                name: round(float(value), DECIMALS[name]) + 0.0
                for name, value in zip(ARM_QUANTITIES, pose_values, strict=True)
            }
        for arm_index, arm in enumerate(LAYOUT.arms):
            arm_positions = joint_positions[arm_index * coordinate_count : (arm_index + 1) * coordinate_count]
            finger_positions = self._model_positions(arm, arm_positions)[LAYOUT.joint_count : self.MODEL_DOF]
            values[arm + "_gripper"] = {"width": round(float(100 * np.sum(finger_positions)), DECIMALS["width"]) + 0.0}

        return ModelState.model_validate(values)

    def tool_poses(self, joint_positions: np.ndarray) -> dict[str, ToolPose]:
        """Returns each arm's tool pose in centimeters and fixed-axis XYZ roll, pitch and yaw radians."""
        joint_positions = np.asarray(joint_positions, dtype=float)
        coordinate_count = LAYOUT.coordinate_count
        tool_poses = {}

        for arm_index, arm in enumerate(LAYOUT.arms):
            pose = self.pose(arm, joint_positions[arm_index * coordinate_count : (arm_index + 1) * coordinate_count])
            tool_poses[arm] = ToolPose(
                position_cm=(100 * pose[:3, 3]).tolist(),
                rpy_rad=self.matrix_to_rpy(pose[:3, :3]).tolist(),
            )

        return tool_poses

    def pose(self, arm: str, arm_positions: np.ndarray) -> np.ndarray:
        """Returns the tool pose of ``arm`` at its coordinates ``arm_positions``, as a 4x4 matrix in meters."""
        pose = self.models[arm].fk(self._model_positions(arm, arm_positions)).copy()
        pose[:3, 3] += self._base(arm)
        pose[:3, :3] = pose[:3, :3] @ self.GRASP_TO_TOOL

        return pose

    def joint_target(self, joint_positions: np.ndarray, arm_deltas: dict[str, CartesianDelta]) -> np.ndarray:
        """Returns the joint vector that carries out each arm's validated increment from the measured joints."""
        joint_positions = np.asarray(joint_positions, dtype=float)
        goal = joint_positions.copy()
        gripper_indices = list(LAYOUT.gripper_indices)
        goal[gripper_indices] = np.clip(goal[gripper_indices], 0, 1)
        failures = []

        for arm, delta in arm_deltas.items():
            arm_start = LAYOUT.arms.index(arm) * LAYOUT.coordinate_count
            try:
                goal[arm_start : arm_start + LAYOUT.joint_count] = self._solve_arm(joint_positions, delta)
            except UnreachablePoseError:
                # every arm is tried, so that the report names all of them, not just the first
                failures.append((arm, delta, self._reachable_fraction_of(joint_positions, delta)))
        if failures:
            raise UnreachablePoseError(self.unreachable_message(failures))

        # the idle arm keeps its measured joints, which may rest just past a limit
        limits = self.robot_config.joint_limits()
        limits.check_positions(goal, LAYOUT, limits.feedback_tolerance)

        return goal

    def _solve_arm(self, joint_positions: np.ndarray, delta: CartesianDelta) -> np.ndarray:
        """Returns the joints of one arm that reach its increment from the unrounded measured pose, solved once."""
        from scipy.spatial.transform import Rotation  # imported where a pose is solved: the planner's process

        geometry, arm = self.robot_config.geometry, delta.side
        delta_cm, delta_rpy = np.array(delta.delta_position_cm), np.array(delta.delta_rpy_rad)
        if np.max(np.abs(delta_rpy)) > geometry.max_rotation_rad:
            raise ValueError(f"{arm}: RPY rotation exceeds offline planning bound")
        arm_start = LAYOUT.arms.index(arm) * LAYOUT.coordinate_count
        arm_positions = joint_positions[arm_start : arm_start + LAYOUT.coordinate_count]
        if not (np.any(delta_cm) or np.any(delta_rpy)):
            return arm_positions[: LAYOUT.joint_count].copy()

        target = self.pose(arm, arm_positions)
        target[:3, 3] += delta_cm / 100
        target[:3, :3] = Rotation.from_euler("xyz", self.matrix_to_rpy(target[:3, :3]) + delta_rpy).as_matrix()
        model_target = target.copy()
        model_target[:3, 3] -= self._base(arm)
        model_target[:3, :3] = target[:3, :3] @ self.GRASP_TO_TOOL.T
        is_solved, solved_positions = self.models[arm].ik(
            model_target,
            self.SITE,
            init_q=self._model_positions(arm, arm_positions),
            limits=self.limits[arm],
            pos_threshold=geometry.position_residual_m * self.SOLVE_THRESHOLD_FRACTION,
            ori_threshold=geometry.rotation_residual_rad * self.SOLVE_THRESHOLD_FRACTION,
        )
        if not is_solved:
            raise UnreachablePoseError(f"{arm}: IK did not converge at the final target")

        candidate = np.r_[solved_positions[: LAYOUT.joint_count], arm_positions[LAYOUT.joint_count]]
        reached_pose = self.pose(arm, candidate)
        position_error = float(np.linalg.norm(reached_pose[:3, 3] - target[:3, 3]))
        angle_error = self.rotation_distance(reached_pose[:3, :3], target[:3, :3])
        if position_error > geometry.position_residual_m or angle_error > geometry.rotation_residual_rad:
            raise ValueError(f"{arm}: final FK residual exceeds bound")

        return candidate[: LAYOUT.joint_count]

    def _reachable_fraction_of(self, joint_positions: np.ndarray, delta: CartesianDelta) -> float:
        """Returns the largest fraction of an arm's increment ``delta`` that solves from the measured joints."""

        def solve(fraction: float) -> None:
            """Solves ``fraction`` of the increment, and raises when it cannot."""
            scaled_action = delta.model_copy(
                update={
                    "delta_position_cm": [fraction * component for component in delta.delta_position_cm],
                    "delta_rpy_rad": [fraction * component for component in delta.delta_rpy_rad],
                }
            )
            self._solve_arm(joint_positions, scaled_action)

        return self.reachable_fraction(solve)

    def _base(self, arm: str) -> np.ndarray:
        """Returns where the base of ``arm`` is in the robot's frame, in meters: the left one on +y."""
        return np.array([0, self.robot_config.geometry.base_spacing_m / 2 * (1 if arm == "left" else -1), 0])

    def _model_positions(self, arm: str, arm_positions: np.ndarray) -> np.ndarray:
        """Returns the model's joints of ``arm`` at its coordinates: its six joints and its two fingers' positions."""
        joint_count = LAYOUT.joint_count
        finger_ranges = self.models[arm]._configuration.model.jnt_range[joint_count : self.MODEL_DOF]
        opening = np.clip(arm_positions[joint_count], 0, 1)
        finger_positions = finger_ranges[:, 0] + opening * (finger_ranges[:, 1] - finger_ranges[:, 0])

        return np.r_[arm_positions[:joint_count], finger_positions]

    @classmethod
    def unreachable_message(cls, failures: list[tuple[str, CartesianDelta, float]]) -> str:
        """Returns the report of each unreachable arm: the largest fraction of its own increment that it solves."""
        parts = []

        for arm, delta, fraction in failures:
            requested_changes = [*delta.delta_position_cm, *delta.delta_rpy_rad]
            changes = ", ".join(
                f"{name} {cls.truncated_text(fraction * value, DECIMALS[name])} of {value:+g}"
                for name, value in zip(ARM_QUANTITIES, requested_changes, strict=True)
                if value
            )
            reach = f"at most {math.floor(100 * fraction)}% ({changes})"
            if parts:
                parts.append(f"{arm}_arm: {reach}.")
            else:
                parts.append(f"{arm}_arm: the target is out of reach. Along the requested change it can go {reach}.")

        return " ".join(parts)

    @classmethod
    def matrix_to_rpy(cls, rotation: np.ndarray) -> np.ndarray:
        """Returns the fixed-axis XYZ roll, pitch and yaw of ``rotation``, the inverse of Rz(yaw) Ry(pitch) Rx(roll)."""
        sin_pitch = float(np.clip(-rotation[2, 0], -1, 1))
        if abs(sin_pitch) < 1 - cls.GIMBAL_MARGIN:
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            return np.array([roll, math.asin(sin_pitch), math.atan2(rotation[1, 0], rotation[0, 0])])

        return np.array([0.0, math.asin(sin_pitch), math.atan2(-rotation[0, 1], rotation[1, 1])])
