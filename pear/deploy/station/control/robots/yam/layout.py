from typing import Literal, get_args

from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.texts import ControlTexts

ArmName = Literal["left", "right"]  # an arm's name, as the actions' models check it, in the order of the joint vector
OUTSIDE_BOUNDS_TEXT = "{arm}: joint or gripper target outside commissioned bounds: {details}"
BOUND_TEXT = "{axis}={value:.6f} {unit}, bounds=[{lower:.6f}, {upper:.6f}], feedback allowance={allowance:.6f}"
# each arm: the joints J1-J6 in rad, then the gripper's opening, 0 closed and 1 open
LAYOUT = JointLayout(
    arms=get_args(ArmName),
    coordinates=("J1", "J2", "J3", "J4", "J5", "J6", "gripper"),
    texts=ControlTexts(
        outside_bounds=OUTSIDE_BOUNDS_TEXT,
        outside_bounds_coordinate=BOUND_TEXT,
        command_outside_bounds=OUTSIDE_BOUNDS_TEXT,
        command_outside_bounds_coordinate=BOUND_TEXT,
        bound_axes=("joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "gripper"),
        stale_feedback="{arm}: stale/invalid feedback, age={age:.4f}s",
        tracking_error_text="Held pose has excessive tracking error: {details}",
        unsettled_start="Enable requires both arms to settle; startup timed out",
        failed_hold="Recovery blocked: the fault hold command failed; reinitialize hardware: ",
    ),
)
BOTH_ARMS = "both"  # the Home choice that returns every arm
# the SDK's arm model and gripper model that the kinematics, the gripper calibration and the friction sweep support
ARM_TYPE = "yam_ultra"
GRIPPER_TYPE = "linear_4310"
