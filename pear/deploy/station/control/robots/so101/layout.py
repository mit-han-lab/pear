from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.texts import ControlTexts

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")  # the arm's joints, base to wrist
MOTORS = (*JOINTS, "gripper")  # the servos on the bus, one for each coordinate
# the arm's joints in rad, then the gripper's opening, 0 closed and 1 open
LAYOUT = JointLayout(
    arms=("arm",),
    coordinates=MOTORS,
    texts=ControlTexts(
        outside_bounds="Joint or gripper target outside commissioned bounds: {details}",
        outside_bounds_coordinate=(
            "{axis}={value:.6f} {unit}, bounds=[{lower:.6f}, {upper:.6f}], allowance={allowance:.6f}"
        ),
        command_outside_bounds="Command leaves the position bounds: {details}",
        command_outside_bounds_coordinate="{axis}={value:.6f} {unit}",
        bound_axes=MOTORS,
        stale_feedback="Stale or invalid feedback, age={age:.4f}s",
        tracking_error_text="Excessive tracking error: {details}",
        unsettled_start="Enable requires the arm to settle; startup timed out",
        failed_hold="Recovery blocked: the fault hold command failed; disable and enable again: ",
    ),
)
ARM = LAYOUT.arms[0]  # the one arm, which names its targets and its readings
VECTOR_LENGTH = LAYOUT.vector_length  # of the joint vector: the five joints and the gripper
GRIPPER_INDEX = LAYOUT.gripper_indices[0]  # of the gripper's opening in the joint vector
