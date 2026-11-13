import json
import threading
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.motion.results import MotionStatus
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE, BimanualPoseFormat
from pear.deploy.station.control.robots.yam.motion.actions import (
    BimanualDelta,
    CartesianDelta,
    JointPositions,
    ModelState,
)
from pear.deploy.station.control.robots.yam.motion.results import YAMMotionResult
from stand_in_clients import ModelReply, StandInScriptedClient

ZERO = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)  # one arm's six quantities, all zero
MODEL_TIMEOUT_S = 60.0  # how long the model may take to answer a turn


class BimanualActions:
    """Bimanual deltas of the YAM arms and grippers."""

    @staticmethod
    def action(
        left: dict[str, float] | None = None,
        right: dict[str, float] | None = None,
        grips: tuple[float, float] = (0.0, 0.0),
    ) -> BimanualDelta:
        """Returns the delta that changes each arm's quantities of ``left`` and ``right`` and each gripper's width by
        ``grips``, every other part zero."""
        return BimanualDelta(
            left_arm={**ZERO, **(left or {})},
            right_arm={**ZERO, **(right or {})},
            left_gripper={"width": grips[0]},
            right_gripper={"width": grips[1]},
        )


@pytest.fixture
def completed_result() -> YAMMotionResult:
    """The result of an action that completed, settled and reached its target."""
    return YAMMotionResult(
        status=MotionStatus.COMPLETED,
        is_settled=True,
        target_reached=True,
        measured_position=[0.0] * 14,
        tracking_error=[0.0] * 14,
    )


def test_a_cartesian_delta_with_a_rotation_vector_field_is_refused():
    with pytest.raises(ValueError):
        CartesianDelta.model_validate(
            dict(side="right", delta_position_cm=[0, 0, 0], delta_rotation_vector_rad=[0.1, 0, 0])
        )


@pytest.mark.parametrize("robot_name", ["yam"])
def test_the_model_reads_four_parts_and_no_joint_vector(robot, tmp_path):
    requests = []

    def infer(request: Any, emit: Any, cancel: threading.Event) -> Any:
        requests.append(request)
        return ModelReply.output(BimanualActions.action())

    client = StandInScriptedClient(SystemPrompt.render("Hold", BIMANUAL_POSE), infer, timeout_s=MODEL_TIMEOUT_S)
    session = StationAgentLoop(
        robot,
        client,
        action_format=BIMANUAL_POSE,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )
    for _ in range(2):
        result = session.step(cancellation=threading.Event(), on_event=lambda kind, data: None, attempt_id=uuid4().hex)
        assert result.status == "completed", result
    session.close()

    for request in requests:
        state = json.loads(request.text.split("\n")[0])
        assert list(state) == ["left_arm", "right_arm", "left_gripper", "right_gripper"]
        assert list(state["left_arm"]) == ["x", "y", "z", "roll", "pitch", "yaw"]
        assert [im.camera_name for im in request.images] == ["head", "left_wrist", "right_wrist"]
        assert "joint_state" not in request.text
    assert "measured_position" not in requests[1].text
    assert "tracking_error" not in requests[1].text
    assert "Previous execution" not in requests[1].text


@pytest.mark.parametrize("bad", ["shape", "nan", "bool", "string", "gripper", "unknown"])
def test_malformed_joint_action_is_rejected(bad):
    payload = dict(positions=[[0.0] * 14], rate_hz=30.0)
    if bad == "shape":
        payload["positions"][0].pop()
    elif bad in ("nan", "bool", "string"):
        payload["positions"][0][0] = {"nan": float("nan"), "bool": True, "string": "0"}[bad]
    elif bad == "gripper":
        payload["positions"][0][13] = 1.1
    else:
        payload["velocity"] = [0.0] * 14

    with pytest.raises(ValidationError):
        JointPositions.model_validate(payload)


def test_fields_in_meters_are_refused():
    with pytest.raises(ValidationError):
        CartesianDelta(side="right", delta_position_m=[0, 0, 0.01], delta_rpy_rad=[0, 0, 0])
    with pytest.raises(ValidationError):
        ModelState(
            left_arm={**ZERO, "x_m": 0.0},
            right_arm=ZERO,
            left_gripper={"width": 0.0},
            right_gripper={"width": 0.0},
        )


def test_unreachable_rejection_feedback_says_nothing_moved():
    result = YAMMotionResult(
        status=MotionStatus.REJECTED,
        is_settled=True,
        target_reached=False,
        measured_position=None,
        tracking_error=None,
        error={"code": "UnreachablePoseError", "message": "left_arm: at most 18%."},
    )

    text = BimanualPoseFormat.execution_feedback(result, is_dispatched=True)

    assert json.loads(text.removesuffix(" Use the new measured observation.")) == {
        "status": "rejected",
        "moved": False,
        "error": {
            "code": "out_of_reach",
            "message": (
                "Nothing moved: the whole action was discarded (both arms and grippers). left_arm: at most 18%."
            ),
        },
    }


@pytest.mark.parametrize("grippers", [{}, {"left": "open"}, {"left": "open", "right": "open"}])
def test_normal_motion_and_gripper_opening_need_no_model_feedback(completed_result, grippers):
    result = replace(completed_result, gripper_outcomes=grippers)

    assert BimanualPoseFormat.execution_feedback(result, is_dispatched=True) is None


@pytest.mark.parametrize(
    "changes,is_dispatched",
    [
        ({"status": MotionStatus.REJECTED}, False),
        ({"status": MotionStatus.FAILED}, True),
        ({"is_settled": False}, True),
        ({"target_reached": False}, True),
        ({"error": {"code": "OperatorPaused", "message": "Interrupted during motion"}}, True),
        ({}, False),
        ({"gripper_outcomes": {"right": "closed_without_confirmed_object"}}, True),
        (
            {
                "target_reached": False,
                "gripper_outcomes": {"right": "resistance_detected_not_grasp_success"},
            },
            True,
        ),
        ({"gripper_outcomes": {"left": "open", "right": "unknown_outcome"}}, True),
    ],
)
def test_exceptional_feedback_is_retained(completed_result, changes, is_dispatched):
    result = replace(completed_result, **changes)

    text = BimanualPoseFormat.execution_feedback(result, is_dispatched=is_dispatched)

    value = json.loads(text.removesuffix(" Use the new measured observation."))
    assert value["status"] == result.status
    assert value["execution_requested"] == is_dispatched
    assert value["settled"] == result.is_settled
    assert value["reached"] == result.target_reached
    if result.error:
        assert value["error"] == result.error
    if result.gripper_outcomes:
        assert value["grippers"] == result.gripper_outcomes


def test_a_turn_without_a_result_reports_the_interruption():
    assert "no action was executed" in BimanualPoseFormat.execution_feedback(None, is_dispatched=False)
