import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop
from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.motion.planning import HomePlanningError
from pear.deploy.station.control.motion.results import MotionResult
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.formats import SINGLE_ARM_POSE_FORMAT
from pear.deploy.station.control.robots.so101.layout import JOINTS, LAYOUT
from pear.deploy.station.control.robots.so101.motion.actions import SingleArmDelta
from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics
from pear.deploy.station.control.robots.so101.motion.planning import SO101Planning
from pear.deploy.station.control.robots.so101.robot import SO101Robot
from pear.deploy.station.control.schemas import HomePlan
from stand_in_cameras import StandInCameras
from stand_in_clients import StandInDemonstrationClient
from stand_in_robots import StandInSO101Robot
from waiting import Waiting

ZERO_HOME_OVERRIDE = "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]"  # Home: every joint at zero, the gripper open
# what the tests of the arm change of the station of the tests besides its quick settings and its Home: the
# tolerances they were written for (Home's, an arrival's, the feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)
TIMEOUT_S = 5.0  # for the stand-in arm to show an injected reading
LATE_TIMEOUT_S = 2.0  # Home's timeout when approved late: past planning and half the Home's estimate
AWAY_RAD = 0.3  # the base turn away from Home, so that Home has to move
POLL_S = 0.01  # between two looks at the stand-in arm's reading
ZERO = dict(x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0)  # a pose delta of the arm that changes nothing
# Elbow bent, wrist pitched down: a typical tabletop working pose.
WORK = np.array([0.2, -0.35, 0.55, 0.4, 0.3, 1.0])
SYSTEM_PROMPT = ""  # what a stand-in model is prompted with: it reads nothing of it
MODEL_TIMEOUT_S = 60.0  # how long the model may take to answer a turn of these tests


class MotionSteps:
    """The actions, the operator's answer to a Home plan, the Home settings and the model sessions of the tests of the
    arm's motion."""

    @staticmethod
    def approve(plan: HomePlan) -> bool:
        """Approves the Home plan, as the operator who reviewed it."""
        return True

    @staticmethod
    def with_home(config: DictConfig, **home: Any) -> DictConfig:
        """Returns ``config`` with these Home settings changed."""
        return OmegaConf.merge(config, {"robot": {"home": home}})

    @staticmethod
    def delta(width: float = 0.0, **arm: float) -> dict[str, Any]:
        """Returns a pose delta action with the given arm components and gripper width."""
        return {"type": "single_arm_delta", "arm": {**ZERO, **arm}, "gripper": {"width": width}}

    @staticmethod
    def delta_model(**arm: float) -> SingleArmDelta:
        """Returns a pose delta action's model with the given arm components."""
        return SingleArmDelta.model_validate({"arm": {**ZERO, **arm}, "gripper": {"width": 0.0}})

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
def so101_config() -> DictConfig:
    """The SO-101's station of the tests with its quick settings, Home at zero, the tolerances of the tests of the arm
    and no feedforward; it takes the place of the conftest's in this file."""
    return ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES)


@pytest.fixture
def arm_configs(so101_config: DictConfig) -> tuple[ControlConfig, SO101Config]:
    """The control loop's configuration and the SO-101's settings of ``so101_config``."""
    return ConfigFiles.make_configs(so101_config)


@pytest.fixture
def robot_at_home() -> Iterator[BaseRobot]:
    """The open SO-101 of the station of the tests, at the real arm's Home, away from the stretched-out singularity,
    with the tolerances of the tests of the arm and a drawn wrist camera; closed at the end."""
    config = ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, *ARM_TEST_OVERRIDES)

    with StandInSO101Robot(config, cameras=StandInCameras(SO101Robot.CAMERAS)) as instance:
        yield instance


@pytest.fixture(scope="module")
def kinematics() -> SO101Kinematics:
    """The SO-101's kinematics of the station of the tests with Home at zero and the tolerances of the tests of the
    arm, loaded once for the module."""
    return SO101Kinematics(ConfigFiles.load_configs("so101", ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES)[1])


def test_expired_deadline_sends_no_gripper_command(so101_robot, monkeypatch):
    runtime = so101_robot.control_loop
    so101_robot.wait_stationary()
    observation = so101_robot.observe()
    begin, begun = runtime.begin_grippers, []
    monkeypatch.setattr(runtime, "begin_grippers", lambda request: begun.append(request) or begin(request))
    sent = len(runtime.driver.hardware.sent_targets)

    with pytest.raises(TimeoutError, match="total timeout"):
        so101_robot.operate_grippers(
            False, observation=observation, guard=ActionGuard(runtime), deadline=time.monotonic() - 1
        )

    assert begun == []
    assert len(runtime.driver.hardware.sent_targets) == sent


def test_home_at_home_sends_nothing_and_verifies_the_pose(so101_robot):
    sent = len(so101_robot.control_loop.driver.hardware.sent_targets)

    report = so101_robot.home(approve_plan=MotionSteps.approve)

    assert len(so101_robot.control_loop.driver.hardware.sent_targets) == sent
    assert report["phases"] == []
    assert report["results"]["reached"] and report["results"]["gripper_open"]


def test_an_unknown_option_is_refused_before_the_arm_is_touched(so101_robot):
    observation = so101_robot.observe()

    with pytest.raises(TypeError, match="side"):
        so101_robot.home(side="left", approve_plan=MotionSteps.approve)

    assert so101_robot.hold_snapshot(observation)["status"] == "HOLD"


def test_path_beyond_a_limit_is_rejected_before_opening(so101_config):
    home = list(so101_config.robot.home.position)
    home[0] = so101_config.robot.limits.upper_bounds[0] - 0.01
    config = MotionSteps.with_home(so101_config, position=home)

    with StandInSO101Robot(config) as station:
        target = list(station.control_loop.targets["arm"])
        target[0], target[5] = 0.0, 0.5
        station.control_loop.move({"arm": target})
        station.control_loop.wait_for_hold(5)
        # The arm rests 0.1 rad past its command: Home would end that far past the limit.
        hardware = station.control_loop.driver.hardware
        hardware.position = (hardware.position[0] + 0.1, *hardware.position[1:])
        Waiting.until(lambda: station.observe()["joint_positions"][0] == hardware.position[0], TIMEOUT_S, POLL_S)
        sent = len(station.control_loop.driver.hardware.sent_targets)

        with pytest.raises(HomeFailed, match="Home path leaves the joint bounds") as failure:
            station.home(approve_plan=MotionSteps.approve)
        assert isinstance(failure.value.__cause__, HomePlanningError)

        assert len(station.control_loop.driver.hardware.sent_targets) == sent


@pytest.mark.parametrize("has_move", [False, True], ids=["at_home", "inward_move"])
def test_home_bounds_refuse_feedback_past_the_feedback_tolerance(arm_configs, has_move):
    control_config, config = arm_configs
    held = np.array(config.home.position, dtype=float)
    held[0] += AWAY_RAD
    robot = {"arms": {arm: {"position": q} for arm, q in LAYOUT.split(held).items()}, "targets": LAYOUT.split(held)}
    plan, move = SO101Planning.checked_home(
        control_config, config, {"joint_positions": held, "control_snapshot": robot}
    )
    start = np.array(plan["start_target"] if has_move else plan["goal_target"])
    # The base joint reads past its upper bound, on the side the move leaves.
    offset = np.zeros(len(start))
    offset[0] = config.limits.upper_bounds[0] + 2 * config.feedback_position_tolerance - start[0]

    with pytest.raises(ValueError, match="outside commissioned bounds"):
        SO101Planning.check_home_bounds(config, move if has_move else None, start, offset)


def test_home_opens_the_gripper_before_the_arm_returns(so101_robot):
    target = list(so101_robot.control_loop.targets["arm"])
    target[0], target[5] = 0.3, 0.2
    so101_robot.control_loop.move({"arm": target})
    so101_robot.control_loop.wait_for_hold(5)
    sent = len(so101_robot.control_loop.driver.hardware.sent_targets)

    report = so101_robot.home(approve_plan=MotionSteps.approve)

    commands = np.array(so101_robot.control_loop.driver.hardware.sent_targets[sent:])
    opened = int(np.argmax(commands[:, 5] >= 1.0))
    assert np.allclose(commands[: opened + 1, 0], 0.3)
    assert not np.allclose(commands[opened:, 0], 0.3)
    assert [phase["stage"] for phase in report["phases"]] == ["opening", "homing"]
    assert report["results"]["reached"]


def test_late_approval_sends_nothing(so101_robot):
    target = list(so101_robot.control_loop.targets["arm"])
    target[0], target[5] = 0.3, 0.2
    so101_robot.control_loop.move({"arm": target})
    so101_robot.control_loop.wait_for_hold(5)
    sent = len(so101_robot.control_loop.driver.hardware.sent_targets)
    started = time.monotonic()

    def slow_review(plan):
        # The operator answers once only half of the plan's estimated time is left.
        left = plan["operation_seconds_estimate"] / 2
        time.sleep(max(0.0, started + LATE_TIMEOUT_S - left - time.monotonic()))
        return True

    with pytest.raises(HomeFailed, match="confirm again"):
        so101_robot.home(approve_plan=slow_review, timeout_s=LATE_TIMEOUT_S)

    assert len(so101_robot.control_loop.driver.hardware.sent_targets) == sent
    assert so101_robot.control_loop.operation is None


def test_inaccurate_home_fails(so101_config):
    config = MotionSteps.with_home(so101_config, arrival_timeout_s=0.5)

    with StandInSO101Robot(config) as station:
        target = list(station.control_loop.targets["arm"])
        target[2] += 0.3
        station.control_loop.move({"arm": target})
        station.control_loop.wait_for_hold(5)
        sag = config.robot.home.joint_tolerance[2] + 0.01
        station.control_loop.driver.hardware.response = lambda goal, position: (*goal[:2], goal[2] + sag, *goal[3:])

        with pytest.raises(HomeFailed, match="Home feedback did not settle") as failure:
            station.home(approve_plan=MotionSteps.approve)

    results = failure.value.details["results"]
    assert not results["reached"]
    assert results["violations"][0]["joint"] == "elbow_flex"


def test_the_zero_pose_reads_its_known_position_pitch_and_roll(kinematics):
    q = np.zeros(6)

    x, y, z, pitch, roll = kinematics.pose(q)

    assert (round(x, 1), round(y, 1), round(z, 1)) == (39.1, 0.0, 22.6)
    assert abs(pitch) < 1e-4  # the URDF writes pi/2 as 1.5708
    # The URDF wrist frame is twisted 0.049 rad from the arm plane at wrist_roll 0.
    assert roll == pytest.approx(-0.049, abs=1e-3)


def test_pointing_direction_stays_in_the_arm_plane(kinematics):
    rng = np.random.default_rng(0)
    limits = kinematics.robot_config.limits
    lower, upper = np.array(limits.lower_bounds), np.array(limits.upper_bounds)
    # the fingertip's frame and the arm plane as the kinematics compute them, from their public model: the plane's
    # horizontal direction is fixed in the shoulder's frame, the base's +x at zero joints
    model = kinematics.model
    model.forward_kinematics(np.zeros(len(JOINTS)))
    plane_axis = model.robot.get_T_world_frame(SO101Kinematics.SHOULDER_FRAME)[:3, :3].T @ np.array([1.0, 0, 0])

    for _ in range(50):
        q = rng.uniform(lower, upper)
        frame = np.array(model.forward_kinematics(np.degrees(q[: len(JOINTS)])))
        u = model.robot.get_T_world_frame(SO101Kinematics.SHOULDER_FRAME)[:3, :3] @ plane_axis
        u[2] = 0.0
        u /= np.linalg.norm(u)
        lateral_plane_normal = np.cross([0, 0, 1], u)
        assert abs(frame[:3, 2] @ lateral_plane_normal) < 1e-4


def test_wrist_flex_down_reads_positive_pitch(kinematics):
    q = np.zeros(6)
    q[3] = math.pi / 2

    assert kinematics.pose(q)[3] == pytest.approx(math.pi / 2, abs=1e-3)


@pytest.mark.parametrize(
    "delta",
    [
        dict(z=-3.0),
        dict(x=2.0, y=-3.0),
        dict(pitch=0.3),
        dict(roll=-0.6),
        dict(x=-2.0, z=2.0, pitch=-0.2, roll=0.4),
    ],
)
def test_pose_deltas_are_reached_within_the_ik_residuals(kinematics, delta):
    before = kinematics.pose(WORK)

    goal = kinematics.joint_target(WORK, MotionSteps.delta_model(**delta))

    after = kinematics.pose(goal)
    expected = before + np.array([delta.get(n, 0.0) for n in ("x", "y", "z", "pitch", "roll")])
    assert np.allclose(after[:3], expected[:3], atol=0.1)  # cm
    assert np.allclose(after[3:], expected[3:], atol=0.01)  # rad
    assert goal[5] == WORK[5]


def test_zero_delta_keeps_joints(kinematics):
    assert np.array_equal(kinematics.joint_target(WORK, MotionSteps.delta_model()), WORK)


def test_an_unreachable_target_reports_how_far_the_arm_can_go(kinematics):
    with pytest.raises(UnreachablePoseError, match=r"at most \d+% \(x \+\d+\.\d of \+40\)"):
        kinematics.joint_target(WORK, MotionSteps.delta_model(x=40.0))


def test_rotation_increment_is_bounded(kinematics):
    with pytest.raises(ValueError, match="planning bound"):
        kinematics.joint_target(WORK, MotionSteps.delta_model(roll=4.0))


def test_quantities_are_rounded_and_typed(kinematics):
    state = kinematics.quantities(WORK)

    assert set(state.arm.model_dump()) == {"x", "y", "z", "pitch", "roll"}
    assert state.gripper.width == 6.0
    assert kinematics.quantities(np.r_[WORK[:5], 0.0]).gripper.width == 0.0


def test_an_interrupted_execution_is_not_reported_as_completed(robot_at_home, monkeypatch, tmp_path):
    original = robot_at_home.control_loop.dispatch_joint_move

    def interrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        robot_at_home.stop()
        return value

    monkeypatch.setattr(robot_at_home.control_loop, "dispatch_joint_move", interrupted)
    client = StandInDemonstrationClient(
        SYSTEM_PROMPT,
        action_format=SINGLE_ARM_POSE_FORMAT.name,
        delay_s=0,
        is_moving=True,
        timeout_s=MODEL_TIMEOUT_S,
    )
    policy = MotionSteps.session(robot_at_home, client, tmp_path)

    result = MotionSteps.step(policy)

    assert result.status == "failed" and not result.target_reached
    policy.close()


def test_gravity_sag_does_not_accumulate_over_steps(so101_robot):
    """Deltas start from the held command while the arm tracks it within tolerance."""
    so101_robot.execute(MotionSteps.delta(x=-8.0, z=-6.0, pitch=0.6))
    hardware = so101_robot.control_loop.driver.hardware
    send = hardware.send

    def sagging(target):  # the elbow settles 0.01 rad short, as under gravity
        send(target)
        hardware.position = tuple(q + 0.01 * (j == 2) for j, q in enumerate(hardware.position))

    hardware.send = sagging
    so101_robot.execute(MotionSteps.delta(x=0.5))
    z = so101_robot.kinematics.pose(so101_robot.observe()["joint_positions"])[2]

    for dx in (1.0, -1.0) * 3:
        assert so101_robot.execute(MotionSteps.delta(x=dx)).target_reached

    drift = so101_robot.kinematics.pose(so101_robot.observe()["joint_positions"])[2] - z
    assert abs(drift) < 0.05


def test_delta_base_falls_back_to_measured_when_blocked(arm_configs):
    _, config = arm_configs
    commanded = np.zeros(6)
    tracking = commanded + [0, 0, 0.02, 0, 0, 0.5]  # gripper differences are ignored
    blocked = commanded + [0, 0, 0.05, 0, 0, 0]

    assert SO101Planning.delta_base(config, tracking, commanded) is commanded
    assert SO101Planning.delta_base(config, blocked, commanded) is blocked
