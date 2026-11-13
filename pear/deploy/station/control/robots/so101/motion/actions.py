from typing import Literal

from pear.deploy.station.control.motion.actions import GripperQuantity, JointPositions, StrictModel
from pear.deploy.station.control.robots.so101.layout import LAYOUT


class ArmQuantities(StrictModel):
    """The arm's pose as the model reads and changes it: the position in cm, the pitch below horizontal and the roll
    about the pointing axis in rad."""

    x: float
    y: float
    z: float
    pitch: float
    roll: float


class ModelState(StrictModel):
    """What the model reads of the measured joints: the arm's pose and the gripper's width."""

    arm: ArmQuantities
    gripper: GripperQuantity


class SingleArmDelta(StrictModel):
    """One model turn in the single-arm pose format: the arm's changes and what the gripper does."""

    type: Literal["single_arm_delta"] = "single_arm_delta"
    arm: ArmQuantities  # zero changes keep the arm held
    gripper: GripperQuantity  # the sign of its width alone: positive opens fully, negative closes, zero holds


class GripperAction(StrictModel):
    """An endpoint operation of the gripper while the arm holds; closing may stop on an object."""

    type: Literal["gripper"] = "gripper"
    command: Literal["open", "close"]


# the chunk of absolute targets of the SO-101's joint vector, settled once after its last row
JointPositions = JointPositions.for_layout(LAYOUT)
