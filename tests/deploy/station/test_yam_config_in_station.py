import dataclasses

import pytest
from omegaconf import OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.control.robots.yam.config import ArmConfig, CameraConfig, CameraCrop
from pear.deploy.station.control.robots.yam.robot import YAMRobot


@pytest.mark.parametrize(
    "change",
    [
        {"x_offset": -1},
        {"y_offset": -1},
        {"width": 0},
        {"height": 0},
        {"x_offset": 161},
        {"y_offset": 121},
        {"width": 480.5},
        {"x_offset": True},
    ],
)
def test_camera_crop_rejects_invalid_pixel_bounds(change):
    with pytest.raises(ValueError):
        OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(CameraCrop), dict(x_offset=80, y_offset=60, width=480, height=360) | change
            )
        )


def test_camera_crop_is_optional_and_accepts_full_frame():
    assert CameraConfig(serial="camera", role="top").crop is None
    assert CameraCrop(x_offset=0, y_offset=0, width=640, height=480).width == 640


def test_config_requires_velocity_feedforward_for_friction():
    config = ConfigFiles.load_configs("yam")[1]
    data = dataclasses.asdict(config.arms["left"])
    data.update(friction_feedforward_nm=[0, 1, 0, 0, 0, 0], uses_velocity_feedforward=False)

    with pytest.raises(ValueError, match="velocity_feedforward"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(type(config.arms["left"])), data))


def test_config_requires_velocity_feedforward_for_load_friction():
    config = ConfigFiles.load_configs("yam")[1]
    data = dataclasses.asdict(config.arms["left"])
    data.update(
        friction_feedforward_nm=[0] * 6,
        friction_load_ratio=[0, 0.1, 0, 0, 0, 0],
        uses_velocity_feedforward=False,
    )

    with pytest.raises(ValueError, match="velocity_feedforward"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(type(config.arms["left"])), data))


@pytest.mark.parametrize("offsets", [[0.0] * 5, [0.0] * 7, [0.21] + [0.0] * 5, [float("nan")] * 6])
def test_joint_offsets_reject_invalid_values(offsets):
    data = dataclasses.asdict(ConfigFiles.load_configs("yam")[1].arms["left"])
    data.pop("joint_offsets")
    # an arm's defaults are the robot's config.yaml, which a station's own file is read over
    shipped_arm = OmegaConf.load(YAMRobot.CONFIG_PATH).robot.arms.left
    arm_config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ArmConfig), shipped_arm, data))
    assert arm_config.joint_offsets == [0.0] * 6

    data["joint_offsets"] = offsets
    with pytest.raises(ValueError, match="joint_offsets"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ArmConfig), data))


@pytest.mark.parametrize(
    "gains",
    [[80] * 5, [80] * 7, [-1] * 6, [501] * 6, [float("nan")] * 6, [float("inf")] * 6],
)
def test_arm_kp_rejects_invalid_motor_gains(gains):
    data = dataclasses.asdict(ConfigFiles.load_configs("yam")[1].arms["left"])
    data["joint_position_gains"] = gains

    with pytest.raises(ValueError, match="joint_position_gains"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ArmConfig), data))


@pytest.mark.parametrize("force", [0, -1, float("nan"), float("inf")])
def test_gripper_force_must_be_finite_and_positive(force):
    data = dataclasses.asdict(ConfigFiles.load_configs("yam")[1].arms["left"])
    data["gripper_force_n"] = force

    with pytest.raises(ValueError, match="gripper_force_n"):
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ArmConfig), data))
