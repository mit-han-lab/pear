import json
from pathlib import Path

from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.motion.results import MotionStatus
from pear.deploy.station.control.robots.yam.joint.grippers import GripperOutcome
from pear.deploy.station.control.robots.yam.motion.actions import ArmQuantities, BimanualDelta, GripperQuantity
from pear.deploy.station.control.robots.yam.motion.results import YAMMotionResult
from pear.deploy.station.control.schemas import ActionFormat, Camera

LEFT_CAMERA = Camera("left", "left_wrist", "Left Wrist")
TOP_CAMERA = Camera("top", "head", "Overhead")
RIGHT_CAMERA = Camera("right", "right_wrist", "Right Wrist")
CAMERAS = (LEFT_CAMERA, TOP_CAMERA, RIGHT_CAMERA)  # in the order the page shows them
ARM_QUANTITIES = tuple(ArmQuantities.model_fields)  # one arm's pose as the model reads and moves it, in order
GRIPPER_QUANTITIES = tuple(GripperQuantity.model_fields)  # a gripper's, in order
DECIMALS = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}  # of each quantity the model reads


class BimanualPoseFormat:
    """What the model reads and is told about a bimanual pose action."""

    NEXT_OBSERVATION = " Use the new measured observation."  # after the result a model is told

    @classmethod
    def execution_feedback(cls, result: YAMMotionResult | None, *, is_dispatched: bool) -> str | None:
        """Returns what the model is told of an action's result: nothing for a plain success, else the result as
        JSON."""
        if result is None:
            return "The previous inference was interrupted; no action was executed. Observe again."

        if (
            is_dispatched
            and result.status == MotionStatus.COMPLETED
            and result.is_settled
            and result.target_reached
            and result.error is None
            and all(outcome == GripperOutcome.OPEN for outcome in result.gripper_outcomes.values())
        ):
            return None
        if result.status == MotionStatus.REJECTED and (result.error or {}).get("code") == UnreachablePoseError.__name__:
            # the offline inverse kinematics refuse the whole action before any arm or gripper command
            value = {
                "status": MotionStatus.REJECTED.value,
                "moved": False,
                "error": {
                    "code": "out_of_reach",
                    "message": "Nothing moved: the whole action was discarded (both arms and "
                    "grippers). " + result.error["message"],
                },
            }
            return json.dumps(value, allow_nan=False) + cls.NEXT_OBSERVATION

        value = {
            "status": result.status.value,
            "execution_requested": is_dispatched,
            "settled": result.is_settled,
            "reached": result.target_reached,
        }
        if result.error:
            value["error"] = result.error
        if result.gripper_outcomes:
            value["grippers"] = result.gripper_outcomes

        return json.dumps(value, allow_nan=False) + cls.NEXT_OBSERVATION


BIMANUAL_POSE = ActionFormat(
    name="bimanual_pose",
    schema={
        "left_arm": ARM_QUANTITIES,
        "right_arm": ARM_QUANTITIES,
        "left_gripper": GRIPPER_QUANTITIES,
        "right_gripper": GRIPPER_QUANTITIES,
    },
    cameras=(TOP_CAMERA, LEFT_CAMERA, RIGHT_CAMERA),
    environment_prompt=(Path(__file__).parent / "prompts" / "bimanual_pose.md").read_text().strip(),
    decimals=DECIMALS,
    action_model=BimanualDelta,
    result_feedback=BimanualPoseFormat.execution_feedback,
)
