import json
import threading
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from config_files import ConfigFiles
from pear.agent_loop.protocol import JsonBlockProtocol
from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop
from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.agent_loop.messages import ObservationMessage, SystemPrompt
from pear.deploy.station.control.motion.results import MotionResult
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.so101.formats import SINGLE_ARM_POSE_FORMAT, SingleArmPoseFormat
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.robots.so101.motion.results import SO101MotionResult
from pear.deploy.station.control.robots.so101.robot import SO101Robot
from stand_in_cameras import StandInCameras
from stand_in_clients import ModelReply, StandInScriptedClient
from stand_in_robots import StandInSO101Robot

# the tolerances the tests of the arm were written for, tighter than the station's (Home's, an arrival's, the
# feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)
SYSTEM_PROMPT = ""  # what a stand-in model is prompted with: it reads nothing of it
MODEL_TIMEOUT_S = 60.0  # how long the model may take to answer a turn of these tests


class ModelTurns:
    """The model's actions and the model sessions of the tests of the SO-101's pose format."""

    @staticmethod
    def action(width: float = 0.0, **arm: float) -> dict[str, dict[str, float]]:
        """Returns a model action with the given arm components and gripper width."""
        zero = {name: 0.0 for name in SO101Kinematics.ARM_QUANTITIES}

        return {"arm": zero | arm, "gripper": {"width": width}}

    @staticmethod
    def session(station: BaseRobot, client: ModelClient, directory: Path) -> StationAgentLoop:
        """Returns a model session on ``station`` in the SO-101's pose format, answering through ``client``, the
        turns' pictures saved in ``directory``."""
        return StationAgentLoop(
            station,
            client,
            action_format=SINGLE_ARM_POSE_FORMAT,
            timeout_s=MODEL_TIMEOUT_S,
            directory=directory,
            reference_images=(),
        )

    @staticmethod
    def step(policy: StationAgentLoop) -> MotionResult:
        """Runs one turn of ``policy`` that no operator pauses and nobody listens to, under a new attempt id."""
        return policy.step(cancellation=threading.Event(), on_event=lambda *_: None, attempt_id=uuid4().hex)


@pytest.fixture
def robot_at_home() -> Iterator[BaseRobot]:
    """The open SO-101 of the station of the tests, at the real arm's Home, away from the stretched-out singularity,
    with the tolerances of the tests of the arm and a drawn wrist camera; closed at the end."""
    config = ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, *ARM_TEST_OVERRIDES)

    with StandInSO101Robot(config, cameras=StandInCameras(SO101Robot.CAMERAS)) as instance:
        yield instance


@pytest.mark.parametrize(
    "bad",
    ["missing", "extra", "bool", "nan", "infinity", "shape", "partial", "reasoning", "duplicate", "extra_quantity"],
)
def test_invalid_final_action_never_parses(bad):
    data = ModelTurns.action()
    if bad == "missing":
        data.pop("gripper")
    elif bad == "extra":
        data["request_id"] = "old"
    elif bad in ("bool", "nan", "infinity", "shape"):
        data["arm"]["x"] = {"bool": True, "nan": float("nan"), "infinity": float("inf")}.get(bad, [0])
    elif bad == "extra_quantity":
        data["arm"]["yaw"] = 0.0

    answer = ModelReply.output(data).answer
    if bad == "partial":
        answer = answer[:-4]
    elif bad == "reasoning":
        answer = "<think>" + answer + "</think>"
    elif bad == "duplicate":
        answer = answer.replace('"x": 0.0', '"x": 1, "x": 0.0', 1)

    with pytest.raises(ValueError):
        ActionParser.parse(answer, SINGLE_ARM_POSE_FORMAT)


def test_an_observation_and_the_system_prompt_follow_the_format(robot_at_home, tmp_path):
    raw = robot_at_home.observe()
    raw["images"] = {"wrist": np.full((4, 8, 3), [240, 10, 20], dtype=np.uint8)}
    state = robot_at_home.model_state(raw["joint_positions"], SINGLE_ARM_POSE_FORMAT.name)

    message = ObservationMessage.build(raw, state, SINGLE_ARM_POSE_FORMAT, tmp_path, "turn", None, ())

    assert [im.camera_name for im in message.images] == ["wrist"]
    shown = json.loads(message.text.splitlines()[0])
    assert list(shown) == ["arm", "gripper"]
    assert list(shown["arm"]) == list(SO101Kinematics.ARM_QUANTITIES)
    assert shown["arm"]["x"] == round(state.arm.x, 1)
    assert shown["arm"]["roll"] == round(state.arm.roll, 2)
    with Image.open(message.images[0].path) as image:
        assert image.size == (8, 4) and image.mode == "RGB"
        assert image.getpixel((0, 0))[0] > 200

    prompt = SystemPrompt.render("Lift the cup", SINGLE_ARM_POSE_FORMAT)
    assert "Lift the cup" in prompt and "## ENVIRONMENT" in prompt
    assert "{environment}" not in prompt and "{instruction}" not in prompt
    example = JsonBlockProtocol.JSON_BLOCK_PATTERN.findall(prompt)[-1]  # the environment's answer example
    ActionParser.parse(f"```json\n{example}\n```", SINGLE_ARM_POSE_FORMAT)  # (raises unless it is a valid action)


def test_round_trip_moves_the_arm_and_feeds_back(robot_at_home, tmp_path):
    requests = []

    def infer(obs, emit, cancel):
        requests.append(obs)
        return ModelReply.output(ModelTurns.action(z=1.0))

    policy = ModelTurns.session(
        robot_at_home, StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S), tmp_path
    )
    before = robot_at_home.kinematics.pose(robot_at_home.observe()["joint_positions"])

    first, second = ModelTurns.step(policy), ModelTurns.step(policy)

    assert first.status == second.status == "completed"
    assert first.target_reached and first.is_settled
    after = robot_at_home.kinematics.pose(robot_at_home.observe()["joint_positions"])
    assert after[2] - before[2] == pytest.approx(2.0, abs=0.05)
    assert set(second.measured_state.arm.model_dump()) == set(SO101Kinematics.ARM_QUANTITIES)
    assert len(requests) == 2 and policy.turn_count == 2
    for request in requests:
        state = json.loads(request.text.split("\n")[0])
        assert list(state["arm"]) == list(SO101Kinematics.ARM_QUANTITIES)
        assert "joint" not in request.text and "measured_position" not in request.text
    assert "Previous execution" not in requests[0].text
    assert '"status": "completed"' in requests[1].text
    assert requests[0].images[0].uuid != requests[1].images[0].uuid

    policy.close()
    assert requests[0].images[0].path.exists()  # Closing does not delete task records.


def test_gripper_outcome_is_fed_back(robot_at_home, tmp_path):
    requests = []
    replies = iter([ModelTurns.action(width=-1.0), ModelTurns.action()])

    def infer(obs, emit, cancel):
        requests.append(obs)
        return ModelReply.output(next(replies))

    policy = ModelTurns.session(
        robot_at_home, StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S), tmp_path
    )

    assert ModelTurns.step(policy).gripper_outcome == "closed"
    ModelTurns.step(policy)
    assert '"gripper": "closed"' in requests[1].text

    policy.close()


def test_unreachable_action_is_rejected_and_fed_back(robot_at_home, monkeypatch, tmp_path):
    requests = []
    replies = iter([ModelTurns.action(x=60.0), ModelTurns.action()])

    def infer(obs, emit, cancel):
        requests.append(obs)
        return ModelReply.output(next(replies))

    policy = ModelTurns.session(
        robot_at_home, StandInScriptedClient(SYSTEM_PROMPT, infer, timeout_s=MODEL_TIMEOUT_S), tmp_path
    )
    # the first turn alone may not move the arm: the patch ends with it, and the device guard's stay
    with monkeypatch.context() as patch:
        patch.setattr(robot_at_home.control_loop, "dispatch_joint_move", lambda *a, **k: pytest.fail("Moved"))
        result = ModelTurns.step(policy)
        assert result.status == "rejected" and result.error["code"] == "UnreachablePoseError"

    ModelTurns.step(policy)
    assert '"code": "out_of_reach"' in requests[1].text and "at most" in requests[1].text

    policy.close()


def test_unreachable_rejection_feedback_says_nothing_moved():
    result = SO101MotionResult(
        status="rejected",
        is_settled=True,
        target_reached=False,
        measured_position=None,
        tracking_error=None,
        error={"code": "UnreachablePoseError", "message": "at most 18%."},
    )

    text = SingleArmPoseFormat.result_feedback(result, is_dispatched=True)

    assert json.loads(text.removesuffix(" Use the new measured observation.")) == {
        "status": "rejected",
        "moved": False,
        "error": {
            "code": "out_of_reach",
            "message": "Nothing moved: the whole action was discarded (arm and gripper). at most 18%.",
        },
    }
