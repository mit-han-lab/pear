from __future__ import annotations

import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only: the station's packages load where a test loads a configuration
    from omegaconf import DictConfig

    from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig


class ConfigFiles:
    """Where the station's tests find their configurations and their data, and how a test loads a robot's
    configuration from them."""

    DIRECTORY = Path(__file__).resolve().parent  # the directory of the station's tests
    CONFIGS = DIRECTORY / "configs"  # the tests' station of each robot: a station's own file in the robot's folder
    STATION_FILE_NAME = "config.yaml"  # the name of the station's own file in a robot's folder
    # every robot the tests run on: each robot with a station of the tests
    ROBOT_NAMES = tuple(sorted(path.parent.name for path in CONFIGS.glob(f"*/{STATION_FILE_NAME}")))
    STORED_TEXTS = DIRECTORY / "stored_texts"  # texts a test's output must equal byte for byte
    # an environment variable: set to anything, the tests write the stored texts again instead of comparing with them
    STORED_TEXTS_UPDATE_VARIABLE = "PEAR_UPDATE_STATION_STORED_TEXTS"
    YAM_RECORDED_CASES = DIRECTORY / "yam_recorded_cases"  # what YAM arms recorded, which tests replay
    # the overrides that give a robot's arms limits wide enough for a move to go anywhere a test of the control loop
    # sends it: YAM's commissioned bounds and caps are narrower; the SO-101's shipped limits serve as they are
    WIDE_LIMITS = {
        "yam": tuple(
            f"robot.arms.{arm}.limits.{name}={values}"
            for arm in ("left", "right")
            for name, values in (
                ("lower_bounds", [-3, -3, -3, -3, -3, -3, 0]),
                ("upper_bounds", [3, 3, 3, 3, 3, 3, 1]),
                ("velocity", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3]),
                ("acceleration", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                ("jerk", [2, 2, 2, 2, 2, 2, 2]),
                ("tracking_error", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            )
        ),
        "so101": (),
    }
    # what the tests of the SO-101's arm and of its commands change of its station of the tests: short settle
    # windows and fast caps
    SO101_QUICK_OVERRIDES = (
        "control.settle_window_s=0.1",
        "control.settle_timeout_s=1.0",
        "robot.home.settle_window_s=0.1",
        "robot.home.speed_scale=1.0",
        "robot.limits.velocity=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]",
        "robot.limits.acceleration=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0]",
        "robot.limits.jerk=[200.0, 200.0, 200.0, 200.0, 200.0, 200.0]",
        "robot.limits.stop_acceleration=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0]",
        "robot.limits.stop_jerk=[600.0, 600.0, 600.0, 600.0, 600.0, 600.0]",
    )

    @classmethod
    def station_file(cls, robot_name: str) -> Path:
        """Returns the own file of the tests' station of the robot ``robot_name``."""
        return cls.CONFIGS / robot_name / cls.STATION_FILE_NAME

    @classmethod
    def load(cls, robot_name: str, *overrides: str) -> DictConfig:
        """Returns the configuration of the tests' station of the robot ``robot_name`` with the ``key=value``
        ``overrides``, loaded as a station loads its own."""
        # the station's packages load where a test loads a configuration: the conftest imports this module wherever
        # pytest collects the station's tests
        from pear.deploy.station.control.robot import BaseRobot

        return BaseRobot.load_config(cls.station_file(robot_name), overrides)

    @classmethod
    def load_configs(cls, robot_name: str, *overrides: str) -> tuple[ControlConfig, RobotConfig]:
        """Returns the control loop's configuration and the robot's settings of the tests' station of the robot
        ``robot_name`` with the ``overrides``, as the robot makes them."""
        return cls.make_configs(cls.load(robot_name, *overrides))

    @staticmethod
    def make_configs(config: DictConfig) -> tuple[ControlConfig, RobotConfig]:
        """Returns the control loop's configuration and the robot's settings of the station's ``config``, as its
        robot makes them."""
        from pear.deploy.station.control.robot import ActionMode

        # made in joint mode, a robot opens nothing and loads no kinematics
        robot = pkgutil.resolve_name(config.robot_class)(config, action_mode=ActionMode.JOINT)

        return robot.control_config, robot.robot_config
