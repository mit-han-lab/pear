import numpy as np

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.limits import JointLimits
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.trajectory import GripperProfile, JointMove
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.motion.planning import HomePlanning, PreparedAction
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.layout import GRIPPER_INDEX, LAYOUT
from pear.deploy.station.control.robots.so101.motion.actions import GripperAction, SingleArmDelta
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.schemas import HomePlan, Observation


class SO101Planning(HomePlanning):
    """The SO-101's planning in the caller's thread: an action's arm move by inverse kinematics from its pose delta,
    and the direct return to Home."""

    TARGETS_CHANGED = "Home target changed; check and confirm again"  # the refusal of an approved move held no longer
    BOUND_TOLERANCE = JointLimits.EPSILON  # the slack on the joint bounds of a Home path

    @classmethod
    def checked_action(
        cls, robot_config: SO101Config, kinematics: SO101Kinematics, observation: Observation, action: StrictModel
    ) -> PreparedAction:
        """Returns the prepared ``action`` from ``observation``: the inverse kinematics start from the pose
        ``delta_base`` gives, the move from the held command."""
        arm_delta, opens_gripper = cls.action_parts(action)
        measured_positions = np.asarray(observation["joint_positions"], dtype=float)
        origin = SnapshotCheck.target_vector(observation["control_snapshot"])
        if measured_positions.shape != origin.shape or not np.isfinite([measured_positions, origin]).all():
            raise ValueError("Expected finite measured and commanded positions")

        move = None
        if arm_delta is not None:
            goal = kinematics.joint_target(cls.delta_base(robot_config, measured_positions, origin), arm_delta)
            goal[GRIPPER_INDEX] = origin[GRIPPER_INDEX]
            move = JointMove.build(origin, goal, robot_config.joint_limits(), LAYOUT)

        return PreparedAction(opens_gripper, move)

    @classmethod
    def checked_home(
        cls, control_config: ControlConfig, robot_config: SO101Config, observation: Observation
    ) -> tuple[HomePlan, JointMove | None]:
        """Returns the plan of the direct return to Home from ``observation`` and its move: the gripper opens first,
        then the arm returns."""
        initial_targets = SnapshotCheck.target_vector(observation["control_snapshot"])
        measured_positions = cls.measured_positions(observation, LAYOUT)
        limits = robot_config.joint_limits()
        limits.check_positions(initial_targets, LAYOUT, limits.feedback_tolerance)
        needs_opening = cls.needs_opening(
            initial_targets[GRIPPER_INDEX], measured_positions[GRIPPER_INDEX], robot_config.gripper_tolerance
        )

        start, goal = initial_targets.copy(), np.array(robot_config.home.position, dtype=float)
        # the gripper opens first and is verified on its own, then the arm returns
        start[GRIPPER_INDEX] = goal[GRIPPER_INDEX] = GripperProfile.OPEN_ENDPOINT
        offset = measured_positions - initial_targets
        offset[GRIPPER_INDEX] = 0.0

        move = cls.home_move(robot_config, start, goal, offset)
        opening_s = 0.0
        if needs_opening:
            opening_s = GripperProfile.duration_s(initial_targets[GRIPPER_INDEX], True, robot_config.limits)

        plan = {
            "initial_target": initial_targets.tolist(),
            "start_target": start.tolist(),
            "goal_target": goal.tolist(),
            "open_gripper": needs_opening,
            **cls.time_estimates(control_config, robot_config, move, opening_s, needs_opening),
        }
        return plan, move

    @classmethod
    def check_home_bounds(
        cls, robot_config: SO101Config, move: JointMove | None, start: np.ndarray, offset: np.ndarray
    ) -> None:
        """Raises unless the planned motion from the targets ``start`` stays within the joint bounds, also when the
        offset of the measured joints shifts it."""
        # the measured pose may rest past a limit within the feedback tolerance; the shifted motion may not go beyond
        # it, and the planned one only moves inward from a start held past a limit
        limits = robot_config.joint_limits()
        measured_positions = start + offset
        limits.check_positions(measured_positions, LAYOUT, limits.feedback_tolerance)
        lower_bounds, upper_bounds = limits.bounds(start)
        shifted_lower_bounds = np.minimum(lower_bounds, measured_positions)
        shifted_upper_bounds = np.maximum(upper_bounds, measured_positions)

        if move is None:
            points = [start]
        else:
            # the least and the greatest position of each coordinate along the move, as Ruckig names them
            extrema = move.trajectory.position_extrema
            points = [np.array([getattr(axis, bound) for axis in extrema]) for bound in ("min", "max")]
        for point in points:
            shifted_point = point + offset
            if (
                np.any(point < lower_bounds - cls.BOUND_TOLERANCE)
                or np.any(point > upper_bounds + cls.BOUND_TOLERANCE)
                or np.any(shifted_point < shifted_lower_bounds - cls.BOUND_TOLERANCE)
                or np.any(shifted_point > shifted_upper_bounds + cls.BOUND_TOLERANCE)
            ):
                raise ValueError("Home path leaves the joint bounds")

    @staticmethod
    def action_parts(action: StrictModel) -> tuple[SingleArmDelta | None, bool | None]:
        """Returns the action's arm delta (None when the arm holds) and what it asks of the gripper: True opens, False
        closes, None leaves it."""
        if isinstance(action, SingleArmDelta):
            width = action.gripper.width
            arm_delta = action if any(action.arm.model_dump().values()) else None
            return arm_delta, None if width == 0 else width > 0
        if isinstance(action, GripperAction):
            return None, action.command == "open"
        raise ValueError("Expected a pose delta or gripper action")

    @staticmethod
    def delta_base(
        robot_config: SO101Config, measured_positions: np.ndarray, commanded_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the pose a delta is added to: the held command while the arm tracks it."""
        # servos sit slightly short of their goals under gravity, so adding each delta to the measured pose would creep
        # the arm down; once any joint is off by more than the arrival tolerance the arm is blocked, and the measured
        # pose is used so that commands never wind up
        joint_error = np.abs(measured_positions - commanded_positions)[:GRIPPER_INDEX]

        return commanded_positions if np.all(joint_error <= robot_config.position_tolerance) else measured_positions
