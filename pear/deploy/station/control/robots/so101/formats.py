import json
from pathlib import Path

from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.motion.results import MotionResult, MotionStatus
from pear.deploy.station.control.robots.so101.hardware.cameras import WristCamera
from pear.deploy.station.control.robots.so101.motion.actions import SingleArmDelta
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.schemas import ActionFormat


class SingleArmPoseFormat:
    """What the model is told of an action's result in the SO-101's single-arm pose format."""

    @staticmethod
    def result_feedback(result: MotionResult | None, *, is_dispatched: bool) -> str:
        """Returns the sentence the model is given of the previous action's ``result``: its actual outcome."""
        if result is None:
            return "The previous inference was interrupted; no action was executed. Observe again."

        if result.status == MotionStatus.REJECTED and (result.error or {}).get("code") == UnreachablePoseError.__name__:
            # the offline inverse kinematics reject the whole action before any arm or gripper command
            value = {
                "status": "rejected",
                "moved": False,
                "error": {
                    "code": "out_of_reach",
                    "message": "Nothing moved: the whole action was discarded (arm and gripper). "
                    + result.error["message"],
                },
            }
            return json.dumps(value, allow_nan=False) + " Use the new measured observation."

        value = {
            "status": result.status.value,
            "execution_requested": is_dispatched,
            "settled": result.is_settled,
            "reached": result.target_reached,
        }
        if result.error:
            value["error"] = result.error
        if result.gripper_outcome:
            value["gripper"] = result.gripper_outcome

        return json.dumps(value, allow_nan=False) + " Use the new measured observation."


# the SO-101's one action format: the arm's pose and the gripper's width, as the model reads them and changes them
SINGLE_ARM_POSE_FORMAT = ActionFormat(
    name="single_arm_pose",
    schema={"arm": SO101Kinematics.ARM_QUANTITIES, "gripper": ("width",)},
    cameras=(WristCamera.CAMERA,),
    environment_prompt=(Path(__file__).with_name("prompts") / "single_arm_pose.md").read_text().strip(),
    decimals=SO101Kinematics.DECIMALS,
    action_model=SingleArmDelta,
    result_feedback=SingleArmPoseFormat.result_feedback,
)
