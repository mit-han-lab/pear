import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PIL import Image

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.messages import ObservationMessage, SystemPrompt
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE
from stand_in_cameras import StandInCameras
from stand_in_clients import DemonstrationActions, ModelReply
from stand_in_robots import StandInRobots
from stored_text import StoredText

ZERO = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)  # one YAM arm's pose quantities, each zero
# a YAM action that moves nothing
ACTION = dict(left_arm=ZERO, right_arm=ZERO, left_gripper={"width": 0.0}, right_gripper={"width": 0.0})
INSTRUCTION = "Pick up the red cube."  # the task of the stored prompts
FEEDBACK = "an earlier execution report"  # what a stored observation tells of the previous action
FORMATS = (("yam", "bimanual_pose"), ("so101", "single_arm_pose"))  # each robot with its format of the stored texts
OUT_OF_REACH_CM = 200.0  # far beyond either arm's reach
TURNED_RAD = 0.5  # a wrist turn about the vertical axis, for formats with a yaw
RELAXED_TIMING_S = 0.5  # stale-feedback and deadline allowance for a loaded test host
# what the station of each robot that the stored observations and feedback were made on changes of its station of
# the tests: the SO-101's arm starts stretched out, every joint at zero, with that station's tolerances
STORED_TEXT_STATIONS = {
    "yam": (),
    "so101": (
        "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
        "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
        "robot.position_tolerance=0.035",
        "robot.feedback_position_tolerance=0.05",
    ),
}


@pytest.fixture(params=FORMATS, ids=lambda pair: f"{pair[0]}-{pair[1]}")
def robot_format(request: pytest.FixtureRequest) -> tuple[str, type[BaseRobot], Any]:
    """The name of a robot of the stored texts, its class on stand-in hardware and its action format."""
    robot_name, format_name = request.param
    robot = StandInRobots.robot_class(robot_name)

    return robot_name, robot, robot.ACTION_FORMATS[format_name]


@pytest.fixture
def station(robot_format: tuple[str, type[BaseRobot], Any]) -> Iterator[BaseRobot]:
    """The open robot of ``robot_format`` on the station its stored texts were made on, observing drawn cameras."""
    robot_name, robot, _ = robot_format
    config = ConfigFiles.load(
        robot_name,
        f"control.state_timeout_s={RELAXED_TIMING_S}",
        f"control.deadline_tolerance_s={RELAXED_TIMING_S}",
        *STORED_TEXT_STATIONS[robot_name],
    )

    with robot(config, cameras=StandInCameras(robot.CAMERAS)) as station:
        yield station


@pytest.mark.parametrize(
    "bad",
    ["missing", "extra", "bool", "nan", "infinity", "shape", "partial", "reasoning", "duplicate"],
)
def test_invalid_final_action_never_parses(bad):
    data = json.loads(json.dumps(ACTION))
    if bad == "missing":
        data.pop("right_gripper")
    elif bad == "extra":
        data["request_id"] = "old"
    elif bad in ("bool", "nan", "infinity", "shape"):
        data["left_arm"]["x"] = {
            "bool": True,
            "nan": float("nan"),
            "infinity": float("inf"),
            "shape": [0],
        }[bad]
    answer = ModelReply.output(data).answer
    if bad == "partial":
        answer = answer[:-4]
    elif bad == "reasoning":
        answer = "<think>" + answer + "</think>"
    elif bad == "duplicate":
        answer = answer.replace('"x": 0.0', '"x": 1, "x": 0.0', 1)

    with pytest.raises(ValueError):
        ActionParser.parse(answer, BIMANUAL_POSE)


def test_a_turn_shows_the_rounded_state_and_rgb_frames_in_order_and_the_prompt_holds_the_task(
    robot, action_format, tmp_path
):
    raw = robot.observe()
    raw["images"] = {camera.role: np.full((4, 8, 3), [240, 10, 20], dtype=np.uint8) for camera in action_format.cameras}
    state = robot.model_state(raw["joint_positions"], action_format.name)

    message = ObservationMessage.build(raw, state, action_format, tmp_path, "turn", None, ())

    assert [im.camera_name for im in message.images] == [camera.name for camera in action_format.cameras]
    shown = json.loads(message.text.splitlines()[0])
    for part, fields in state.model_dump().items():
        for name, value in fields.items():
            assert shown[part][name] == round(value, action_format.decimals[name])
    for frame in message.images:
        with Image.open(frame.path) as image:
            assert image.size == (8, 4) and image.mode == "RGB"
            assert image.getpixel((0, 0))[0] > 200
    prompt = SystemPrompt.render("Lift the cup", action_format)
    assert "Lift the cup" in prompt and "{environment}" not in prompt and "{instruction}" not in prompt


def test_the_system_prompt_is_its_stored_text(robot_format):
    robot_name, _, action_format = robot_format

    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"prompt-{robot_name}-{action_format.name}.txt",
        SystemPrompt.render(INSTRUCTION, action_format),
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )


def test_an_observation_s_text_is_its_stored_text(robot_format, station, tmp_path):
    robot_name, _, action_format = robot_format
    raw = station.observe()
    state = station.model_state(raw["joint_positions"], action_format.name)

    plain = ObservationMessage.build(raw, state, action_format, tmp_path, "turn-1", None, ())
    reported = ObservationMessage.build(raw, state, action_format, tmp_path, "turn-2", FEEDBACK, ())

    text = "\n".join(
        [
            plain.text,
            "--- images: " + ", ".join(f"{image.camera_name} {image.path.name}" for image in plain.images),
            "--- with feedback:",
            reported.text,
        ]
    )
    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"observation-{robot_name}-{action_format.name}.txt",
        text + "\n",
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )


def test_parsed_answers_and_parse_errors_are_their_stored_text(robot_format):
    robot_name, _, action_format = robot_format
    action = DemonstrationActions.action(action_format.name, 0, True)
    first_part = next(iter(action_format.schema))
    missing = {part: fields for part, fields in action.items() if part != first_part}
    text_value = {**action, first_part: {**action[first_part], "z": "up"}}
    answers = {
        "valid": f"Moving.\n```json\n{json.dumps(action)}\n```",
        "missing part": f"```json\n{json.dumps(missing)}\n```",
        "text after the block": f"```json\n{json.dumps(action)}\n```\nDone.",
        "non-number": f"```json\n{json.dumps(text_value)}\n```",
        "thinking block": f"<think>plan</think>\n```json\n{json.dumps(action)}\n```",
        "duplicate field": f'```json\n{{"{first_part}": {{"x": 0, "x": 1}}}}\n```',
        "no block": "I will move the arm.",
    }

    lines = []
    for label, answer in answers.items():
        try:
            parsed = ActionParser.parse(answer, action_format).model_dump(mode="json")
            lines.append(f"{label}: {json.dumps(parsed, sort_keys=True)}")
        except ValueError as error:
            lines.append(f"{label}: {type(error).__name__}: {error}")

    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"parse-{robot_name}-{action_format.name}.txt",
        "\n".join(lines) + "\n",
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )


def test_execution_feedback_is_its_stored_text(robot_format, station):
    robot_name, _, action_format = robot_format
    bob = DemonstrationActions.action(action_format.name, 0, True)
    arm_parts = [part for part, fields in action_format.schema.items() if "x" in fields]
    gripper_parts = [part for part in action_format.schema if part not in arm_parts]
    unreachable = {**bob, arm_parts[0]: {**bob[arm_parts[0]], "x": OUT_OF_REACH_CM}}
    closing = {
        **DemonstrationActions.action(action_format.name, 0, False),
        **{part: {"width": -1.0} for part in gripper_parts},
    }
    actions = {"completed move": bob, "out of reach": unreachable, "gripper closing": closing}
    if "yaw" in action_format.schema[arm_parts[0]]:
        actions["turned wrist"] = {**bob, arm_parts[0]: {**bob[arm_parts[0]], "yaw": TURNED_RAD}}

    lines = [f"interrupted: {action_format.result_feedback(None, is_dispatched=False)}"]
    for label, action in actions.items():
        parsed = action_format.action_model.model_validate(action)
        result = station.execute(parsed, reference_observation=station.observe())
        lines.append(f"{label}: {action_format.result_feedback(result, is_dispatched=True)}")

    StoredText.assert_matches(
        ConfigFiles.STORED_TEXTS / f"feedback-{robot_name}-{action_format.name}.txt",
        "\n".join(lines) + "\n",
        ConfigFiles.STORED_TEXTS_UPDATE_VARIABLE,
    )
