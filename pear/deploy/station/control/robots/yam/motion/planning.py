from __future__ import annotations

import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.trajectory import GripperProfile, JointMove
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.motion.planning import HomePlanning, PreparedAction
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.layout import BOTH_ARMS, LAYOUT
from pear.deploy.station.control.robots.yam.motion.actions import BimanualDelta, CartesianDelta, GripperAction
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.schemas import HomePlan, Observation


class YAMPlanning(HomePlanning):
    """The planning of both YAM arms: an action's arm move and gripper request, and the plan of Home."""

    @classmethod
    def checked_action(
        cls, robot_config: YAMConfig, kinematics: YAMKinematics | None, observation: Observation, action: StrictModel
    ) -> PreparedAction:
        """Returns the checked arm move and the gripper request of a Cartesian, bimanual or gripper action from the
        hold of ``observation``."""
        arm_deltas, gripper_opens = cls.action_parts(action)
        active_arms = [
            arm for arm, delta in arm_deltas.items() if any(delta.delta_position_cm) or any(delta.delta_rpy_rad)
        ]
        measured_positions = np.asarray(observation["joint_positions"], dtype=float)
        origin = SnapshotCheck.target_vector(observation["control_snapshot"])
        vector_shape = (LAYOUT.vector_length,)
        if (
            measured_positions.shape != vector_shape
            or origin.shape != vector_shape
            or not np.isfinite([measured_positions, origin]).all()
        ):
            raise ValueError(f"Expected finite {LAYOUT.vector_length}D measured and commanded positions")

        goal = kinematics.joint_target(measured_positions, arm_deltas) if arm_deltas else origin.copy()
        for arm_index, arm in enumerate(LAYOUT.arms):
            arm_start = arm_index * LAYOUT.coordinate_count
            gripper_index = arm_start + LAYOUT.joint_count
            if arm not in active_arms:
                goal[arm_start:gripper_index] = origin[arm_start:gripper_index]
            goal[gripper_index] = origin[gripper_index]
        move = JointMove.build(origin, goal, robot_config.joint_limits(), LAYOUT) if active_arms else None

        return PreparedAction(gripper_opens or None, move)

    @classmethod
    def checked_home(
        cls,
        control_config: ControlConfig,
        robot_config: YAMConfig,
        observation: Observation,
        side: str,
        opens_grippers: bool,
    ) -> tuple[HomePlan, JointMove | None]:
        """Returns the plan of the direct move of ``side`` Home and the move; without ``opens_grippers`` the
        grippers keep their held targets."""
        # the robot's home() checked the side and Home's pose
        selected_arms = list(LAYOUT.arms) if side == BOTH_ARMS else [side]
        initial_targets = SnapshotCheck.target_vector(observation["control_snapshot"])
        measured_positions = cls.measured_positions(observation, LAYOUT)
        start, goal = initial_targets.copy(), initial_targets.copy()
        offset = measured_positions - initial_targets
        limits = robot_config.joint_limits()
        limits.check_positions(initial_targets, LAYOUT, limits.feedback_tolerance)

        opens_by_arm = {}
        for arm in selected_arms:
            arm_start = LAYOUT.arms.index(arm) * LAYOUT.coordinate_count
            gripper_index = arm_start + LAYOUT.joint_count
            goal[arm_start:gripper_index] = robot_config.home.arm_pose(arm)[: LAYOUT.joint_count]
            if not opens_grippers:
                continue
            if cls.needs_opening(
                initial_targets[gripper_index], measured_positions[gripper_index], robot_config.gripper_tolerance
            ):
                opens_by_arm[arm] = True
            start[gripper_index] = goal[gripper_index] = GripperProfile.OPEN_ENDPOINT
            # the opening has its own check of the endpoint before this move is checked again
            offset[gripper_index] = 0.0

        move = cls.home_move(robot_config, start, goal, offset)
        arm_limits = robot_config.arm_limits()
        # the slowest of the grippers that open
        opening_s = max(
            (
                GripperProfile.duration_s(
                    initial_targets[LAYOUT.gripper_indices[LAYOUT.arms.index(arm)]], True, arm_limits[arm]
                )
                for arm in opens_by_arm
            ),
            default=0.0,
        )
        plan = {
            "side": side,
            "active_sides": selected_arms,
            "open_grippers": opens_grippers,
            "initial_targets": initial_targets.tolist(),
            "start_targets": start.tolist(),
            "goal_targets": goal.tolist(),
            "opening_commands": opens_by_arm,
            **cls.time_estimates(control_config, robot_config, move, opening_s, bool(opens_by_arm)),
        }

        return plan, move

    @staticmethod
    def action_parts(action: StrictModel) -> tuple[dict[str, CartesianDelta], dict[str, bool]]:
        """Returns each arm's increment and whether each gripper the action moves opens, without inventing moves
        that hold."""
        if isinstance(action, BimanualDelta):
            arm_deltas = {arm: getattr(action, arm + "_arm").to_cartesian_delta(arm) for arm in LAYOUT.arms}
            gripper_widths = {arm: getattr(action, arm + "_gripper").width for arm in LAYOUT.arms}
            return arm_deltas, {arm: width > 0 for arm, width in gripper_widths.items() if width != 0}
        if isinstance(action, CartesianDelta):
            return {action.side: action}, {}
        if isinstance(action, GripperAction):
            return {}, {action.side: action.command == "open"}

        raise ValueError("Expected a Cartesian, bimanual or gripper action")
