import dataclasses

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.trajectory import JointTrajectory
from pear.deploy.station.control.robots.so101.config import SO101Config, SO101Limits
from pear.deploy.station.control.robots.so101.layout import LAYOUT
from pear.deploy.station.control.robots.so101.robot import SO101Robot

# a station without the arm's bus and its camera: the SO-101's settings when no device is configured
NO_DEVICE_OVERRIDES = ("robot.hardware=null", "robot.camera=null")
ZERO_HOME_OVERRIDE = "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]"  # Home: every joint at zero, the gripper open
# what the tests of the arm change of the station of the tests besides its quick settings and its Home: the
# tolerances they were written for (Home's, an arrival's, the feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)


class BrakingTime:
    """How long a STOP of the arm takes."""

    @staticmethod
    def braking_seconds(control_config: ControlConfig, config: SO101Config, limits: SO101Limits) -> float:
        """Returns how long a STOP takes from a joint still speeding up, under ``limits``."""
        joint_limits = dataclasses.replace(config, limits=limits).joint_limits()
        trajectory = JointTrajectory(joint_limits, LAYOUT, control_config)
        trajectory.hold(np.zeros(LAYOUT.vector_length))
        trajectory.input.current_velocity = [1.0, 0, 0, 0, 0, 0]
        trajectory.input.current_acceleration = [5.0, 0, 0, 0, 0, 0]  # still speeding up

        trajectory.stop(from_standstill=False, braking_caps=joint_limits.braking_caps())

        return trajectory.duration_s


@pytest.fixture
def so101_config() -> DictConfig:
    """The SO-101's station of the tests with its quick settings, Home at zero, the tolerances of the tests of the arm
    and no feedforward; it takes the place of the conftest's in this file."""
    return ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES)


def test_a_station_file_resolves_its_paths_and_is_commissioned():
    _, config = ConfigFiles.load_configs("so101")

    assert config.geometry.urdf.is_file()
    assert (config.hardware.calibration_dir / "so101_follower.json").is_file()
    assert config.motion_blockers() == []
    assert config.camera.role == "wrist"


def test_a_configuration_without_hardware_is_blocked_from_real_motion():
    _, config = ConfigFiles.load_configs("so101", *NO_DEVICE_OVERRIDES)

    assert config.hardware is None and config.camera is None
    assert config.motion_blockers() == ["missing hardware section"]


def test_stop_limits_may_not_brake_more_gently_than_moves(so101_config):
    data = so101_config.robot
    data.limits.stop_jerk = [1.0] * 6

    with pytest.raises(ValueError, match="more gently"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SO101Config), data))


def test_feedforward_needs_the_urdf(so101_config):
    data = so101_config.robot
    data.geometry = None
    # a feedforward section with every value its default: the defaults are the package's config.yaml's
    data.feedforward = OmegaConf.load(SO101Robot.CONFIG_PATH).robot.feedforward

    with pytest.raises(ValueError, match="geometry"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SO101Config), data))


def test_stop_limits_shorten_braking(so101_config):
    control_config, config = ConfigFiles.make_configs(so101_config)
    gentle = dataclasses.replace(config.limits, stop_acceleration=None, stop_jerk=None)

    assert BrakingTime.braking_seconds(control_config, config, config.limits) < 0.6 * BrakingTime.braking_seconds(
        control_config, config, gentle
    )
