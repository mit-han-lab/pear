from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import AccuracyCheck

if TYPE_CHECKING:  # annotations only: the station's packages are imported where the command runs
    import numpy as np

    from pear.deploy.station.control.joint.layout import JointLayout
    from pear.deploy.station.control.robots.so101.config import SO101Config
    from pear.deploy.station.control.robots.so101.robot import SO101Robot

# the station's packages and the robot's own modules, the commands' module among them, are imported inside the
# methods: the command line lists the commands without them, and the commands' module imports this one


class SO101AccuracyCheck(AccuracyCheck):
    """MOVES the arm: the fingertip's stop error at fixed poses from Home, without the feedforward and then with the
    station's."""

    # the arm joints in rad, wrist_roll at Home's; each moves shoulder_lift or elbow_flex at least 0.3 rad from Home
    # and keeps the fingertip at least 7 cm above the table. Most lower the arm, with gravity; "forward_high" and
    # "up" raise the forearm against it. "left" puts the fingertip at the base frame's +y
    POSES = {
        "forward_low": [0.0, -0.15, 0.33, 0.83, -1.613],
        "forward_high": [0.0, -0.35, -0.25, 1.0, -1.613],
        "up": [0.0, -1.0, -0.5, 1.3, -1.613],
        "near": [0.0, -1.25, 0.75, 1.3, -1.613],
        "left": [-0.6, -0.45, 0.5, 1.0, -1.613],
        "right": [0.6, -0.4, 0.4, 1.0, -1.613],
    }
    LOADED_JOINTS = (1, 2)  # shoulder_lift and elbow_flex
    MIN_TIP_HEIGHT_CM = 4.0  # the lowest the fingertip may stop at a pose
    LIMIT_MARGIN = 0.05  # the least distance of a pose's joint from its limit, rad
    MM_PER_CM = 10.0  # the kinematics' positions are in cm
    FEEDFORWARD_KEYS = ("gravity_compliance", "friction_offset_rad", "friction_load_ratio")
    FAILURE_HINT = (
        "Check that the friction-sweep feedforward sits under robot:, run friction-sweep again after any change of "
        "position_p_coefficient, and check that the gripper's load is that of the sweep."
    )

    def run(self) -> None:
        """Refuses an uncommissioned arm, as every command that moves it does first, then runs the check."""
        from pear.deploy.station.control.robot import ActionMode

        self.new_robot(ActionMode.JOINT).robot_config.require_commissioned()

        return super().run()

    def checked_arm(self, layout: JointLayout) -> str:
        """Returns the SO-101's one arm."""
        return layout.arms[0]

    def poses(self, robot_config: SO101Config) -> dict[str, list[float]]:
        """Returns the fixed poses by name: the five arm joints, rad."""
        return self.POSES

    def home_joints(self, robot_config: SO101Config, arm: str) -> list[float]:
        """Returns the five arm joints of the station's Home, rad."""
        return list(robot_config.home.position[: robot_config.LAYOUT.joint_count])

    def tool_point_mm(self, robot_config: SO101Config, arm: str, coordinates: Sequence[float]) -> np.ndarray:
        """Returns the fingertip's position at ``coordinates`` in the base frame, mm."""
        from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics

        return SO101Kinematics.shared_instance(robot_config).pose(coordinates)[:3] * self.MM_PER_CM

    def check_pose(self, robot_config: SO101Config, arm: str, coordinates: Sequence[float]) -> None:
        """Raises ValueError unless every arm joint of ``coordinates`` stays LIMIT_MARGIN inside its limits and the
        fingertip MIN_TIP_HEIGHT_CM above the table."""
        from pear.deploy.station.control.robots.so101.layout import JOINTS
        from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics

        limits = robot_config.limits
        for index, name in enumerate(JOINTS):
            lower, upper = (
                limits.lower_bounds[index] + self.LIMIT_MARGIN,
                limits.upper_bounds[index] - self.LIMIT_MARGIN,
            )
            if not lower <= coordinates[index] <= upper:
                raise ValueError(f"{name} at {coordinates[index]:.3f} rad is outside [{lower:.3f}, {upper:.3f}]")
        tip_height_cm = SO101Kinematics.shared_instance(robot_config).pose(coordinates)[2]
        if tip_height_cm < self.MIN_TIP_HEIGHT_CM:
            raise ValueError(
                f"the fingertip at {tip_height_cm:.1f} cm is below the table's clearance of {self.MIN_TIP_HEIGHT_CM} cm"
            )

    def compensation(self, robot_config: SO101Config, arm: str) -> dict[str, list[float]]:
        """Returns the feedforward of ``robot_config``, all zeros when it has none."""
        feedforward, joint_count = robot_config.feedforward, robot_config.LAYOUT.joint_count

        return {
            key: [0.0] * joint_count if feedforward is None else [float(value) for value in getattr(feedforward, key)]
            for key in self.FEEDFORWARD_KEYS
        }

    def uncompensated_overrides(self, arm: str) -> list[str]:
        """Returns the override that leaves the servos without goal offsets."""
        return ["robot.feedforward=null"]

    def park(self, robot: SO101Robot, start: dict[str, Any]) -> None:
        """Returns the arm to the targets it was enabled at, from Home or a fault, before its torque is released."""
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps

        MaintenanceSteps.park(robot, start[self.checked_arm(robot.robot_config.LAYOUT)])
