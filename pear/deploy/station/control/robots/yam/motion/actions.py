from typing import Annotated, Literal

from pydantic import Field

from pear.deploy.station.control.motion.actions import GripperQuantity, JointPositions, StrictModel
from pear.deploy.station.control.robots.yam.layout import LAYOUT, ArmName

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]  # a vector of the robot's frame: x, y, z

JointPositions = JointPositions.for_layout(LAYOUT)  # a chunk of targets of both arms' joint vector


class ArmQuantities(StrictModel):
    """One arm's tool pose as the model reads and moves it: centimeters and fixed-axis XYZ roll, pitch and yaw in
    radians."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class ModelState(StrictModel):
    """Model-visible bimanual pose quantities: centimeters and fixed-axis XYZ RPY radians."""

    left_arm: ArmQuantities
    right_arm: ArmQuantities
    left_gripper: GripperQuantity
    right_gripper: GripperQuantity


class CartesianDelta(StrictModel):
    """Station translation and additive fixed-axis XYZ RPY increments, in cm/rad."""

    type: Literal["cartesian_delta"] = "cartesian_delta"
    side: ArmName
    frame: Literal["yam_station"] = "yam_station"
    delta_position_cm: Vector3 = Field(description="Station-frame translation [dx,dy,dz] in cm")
    delta_rpy_rad: Vector3


class ArmDelta(ArmQuantities):
    """One arm's change of its tool pose in a model's action."""

    def to_cartesian_delta(self, arm: str) -> CartesianDelta:
        """Returns this change as the Cartesian delta of ``arm``."""
        return CartesianDelta(
            side=arm,
            delta_position_cm=[self.x, self.y, self.z],
            delta_rpy_rad=[self.roll, self.pitch, self.yaw],
        )


class GripperDelta(GripperQuantity):
    """Width delta follows the robot convention: positive opens, negative closes, zero holds."""


class BimanualDelta(StrictModel):
    """One model turn with an explicit delta for every part; all-zero parts keep their targets."""

    type: Literal["bimanual_delta"] = "bimanual_delta"
    left_arm: ArmDelta
    right_arm: ArmDelta
    left_gripper: GripperDelta
    right_gripper: GripperDelta


class GripperAction(StrictModel):
    """Endpoint operation with stationary arms; close may stop on resistance."""

    type: Literal["gripper"] = "gripper"
    side: ArmName
    command: Literal["open", "close"]
