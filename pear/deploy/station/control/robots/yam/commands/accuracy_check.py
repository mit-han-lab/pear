from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import AccuracyCheck

if TYPE_CHECKING:  # annotations only: the station's packages are imported where a command runs
    from pear.deploy.station.control.joint.layout import JointLayout
    from pear.deploy.station.control.robot import BaseRobot
    from pear.deploy.station.control.robots.yam.config import YAMConfig
    from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics


class YAMAccuracyCheck(AccuracyCheck):
    """MOVES one arm of the YAM through its poses from Home, first with the package's gravity factors and no friction
    feedforward and then with the station's, while the other arm holds Home; without --yes it checks the poses and
    moves nothing."""

    # the robot's arms, written here, as the command line declares them without the station's packages
    ARM_CHOICES = ("left", "right")
    # each arm's poses, J1-J6 rad: far and near, high and low, out and in, one with the wrist turned; J2 travels
    # against gravity to near_high and near_mid and with it to the others, J3 lifts the forearm to each. The right
    # arm's are the left's mirrored (J1, J5 and J6 negated); J1 turns at most 0.2 rad toward the other arm. The
    # lowest tool point is far_low's, 14.8 cm above the table.
    POSES_BY_SIDE: dict[str, dict[str, list[float]]] = {
        "left": {
            "far_low": [0.0, 1.5, 0.9, 0.0, 0.0, 0.0],
            "far_out": [0.45, 1.9, 1.6, -0.3, 0.0, 0.0],
            "near_high": [0.0, 0.6, 1.25, 0.0, 0.0, 0.0],
            "near_mid": [0.25, 0.45, 1.0, -0.4, 0.0, 0.0],
            "wrist_turned": [0.5, 1.2, 1.1, -0.6, 0.6, 1.0],
            "inward": [-0.2, 1.2, 0.85, 0.3, -0.3, -0.5],
        },
        "right": {
            "far_low": [0.0, 1.5, 0.9, 0.0, 0.0, 0.0],
            "far_out": [-0.45, 1.9, 1.6, -0.3, 0.0, 0.0],
            "near_high": [0.0, 0.6, 1.25, 0.0, 0.0, 0.0],
            "near_mid": [-0.25, 0.45, 1.0, -0.4, 0.0, 0.0],
            "wrist_turned": [-0.5, 1.2, 1.1, -0.6, -0.6, -1.0],
            "inward": [0.2, 1.2, 0.85, 0.3, 0.3, 0.5],
        },
    }
    BOUND_MARGIN = 0.05  # kept between a pose and a joint bound, rad
    MIN_TOOL_HEIGHT_CM = 8.0  # a pose that brings the tool closer to the table is refused, as a sweep is
    LOADED_JOINTS = (1, 2)  # J2 and J3
    FAILURE_HINT = (  # formatted with the checked arm when the command is made
        "Check that the friction-sweep lines sit under robot.arms.{side}, run friction-sweep again, and check that "
        "the gripper and its load are those of the sweep."
    )

    def __init__(self, args: argparse.Namespace, robot_class_path: str) -> None:
        """Keeps the parsed command line, names its arm in the failure's hint and makes no kinematics yet."""
        super().__init__(args, robot_class_path)
        self.FAILURE_HINT = self.FAILURE_HINT.format(side=args.side)
        self.kinematics: dict[str, YAMKinematics] = {}  # by the text of the robot's settings they are made of

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the checked arm and the confirmation that it may move."""
        super().add_arguments(parser)
        parser.add_argument("--side", choices=cls.ARM_CHOICES, required=True)

    def checked_arm(self, layout: JointLayout) -> str:
        """Returns the arm the command line names."""
        return self.args.side

    def poses(self, robot_config: YAMConfig) -> dict[str, list[float]]:
        """Returns the poses of the arm the command line names."""
        return self.POSES_BY_SIDE[self.args.side]

    def home_joints(self, robot_config: YAMConfig, arm: str) -> list[float]:
        """Returns the joints J1-J6 of ``arm`` at its Home."""
        return list(robot_config.home.arm_pose(arm)[: robot_config.LAYOUT.joint_count])

    def tool_point_mm(self, robot_config: YAMConfig, arm: str, coordinates: Sequence[float]) -> Any:
        """Returns the tool's point of ``arm`` at ``coordinates`` (J1-J6 and the gripper), in the robot's frame, mm."""
        import numpy as np

        return self.arm_kinematics(robot_config).pose(arm, np.asarray(coordinates, dtype=float))[:3, 3] * 1000

    def check_pose(self, robot_config: YAMConfig, arm: str, coordinates: Sequence[float]) -> None:
        """Raises ValueError unless every joint of ``arm`` at ``coordinates`` keeps BOUND_MARGIN inside its bounds, the
        tool keeps MIN_TOOL_HEIGHT_CM above the table and stays on its own arm's side of the station."""
        joint_count = robot_config.LAYOUT.joint_count
        limits = robot_config.arms[arm].limits
        for joint, position in enumerate(coordinates[:joint_count]):
            lower_bound = limits.lower_bounds[joint] + self.BOUND_MARGIN
            upper_bound = limits.upper_bounds[joint] - self.BOUND_MARGIN
            if not lower_bound <= position <= upper_bound:
                raise ValueError(
                    f"{arm} J{joint + 1} at {position:.2f} is within {self.BOUND_MARGIN} rad of its bounds "
                    f"[{limits.lower_bounds[joint]:.2f}, {limits.upper_bounds[joint]:.2f}]"
                )

        tool_point_cm = self.tool_point_mm(robot_config, arm, coordinates) / 10
        if tool_point_cm[2] < self.MIN_TOOL_HEIGHT_CM:
            raise ValueError(f"{arm} at {list(coordinates)} puts the tool {tool_point_cm[2]:.1f} cm above the table")
        # the bases sit across the station (y) at +-half their spacing, the left one at +y: the tool is nearer its
        # own arm's base while it keeps that base's sign
        own_side_sign = 1 if arm == self.ARM_CHOICES[0] else -1
        if tool_point_cm[1] * own_side_sign <= 0:
            raise ValueError(f"{arm} at {list(coordinates)} puts the tool on the other arm's side of the station")

    def compensation(self, robot_config: YAMConfig, arm: str) -> dict[str, list[float]]:
        """Returns the gravity factors the SDK applies to ``arm`` and its friction feedforward."""
        from pear.deploy.station.control.robots.yam.commands.friction_sweep import FrictionSweep  # with the SDK

        arm_config = robot_config.arms[arm]

        return {
            "gravity_compensation_factors": [
                float(factor) for factor in FrictionSweep.applied_gravity_factors(arm_config)
            ],
            "friction_feedforward_nm": [float(value) for value in arm_config.friction_feedforward_nm],
            "friction_load_ratio": [float(value) for value in arm_config.friction_load_ratio],
        }

    def uncompensated_overrides(self, arm: str) -> list[str]:
        """Returns the overrides that give ``arm`` the gravity factors of the package's config.yaml and no friction
        feedforward."""
        from omegaconf import OmegaConf

        from pear.deploy.station.control.robots.yam.layout import LAYOUT
        from pear.deploy.station.control.robots.yam.robot import YAMRobot

        package_arm = OmegaConf.load(YAMRobot.CONFIG_PATH).robot.arms[arm]
        values = {
            "gravity_compensation_factors": OmegaConf.to_container(package_arm.gravity_compensation_factors)
            if package_arm.gravity_compensation_factors is not None
            else None,
            "friction_feedforward_nm": [0] * LAYOUT.joint_count,
            "friction_load_ratio": [0] * LAYOUT.joint_count,
        }

        return [f"robot.arms.{arm}.{key}={json.dumps(value)}" for key, value in values.items()]

    def park(self, robot: BaseRobot, start: dict[str, Any]) -> None:
        """Returns both arms Home, where the YAM rests, as a friction sweep ends."""
        robot.home()

    def arm_kinematics(self, robot_config: YAMConfig) -> YAMKinematics:
        """Returns the kinematics of ``robot_config``, made once for each of the robot's settings."""
        from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics  # with the model's

        key = repr(robot_config)
        if key not in self.kinematics:
            self.kinematics[key] = YAMKinematics(robot_config)

        return self.kinematics[key]
