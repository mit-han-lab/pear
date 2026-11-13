import copy
import json
import os
import signal
import time
from collections.abc import Iterator
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from config_files import ConfigFiles
from control_rig import StandInClock, Ticker
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.trajectory import JointMove
from pear.deploy.station.control.motion.planning import HomePlanningError
from pear.deploy.station.control.robot import ActionMode
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from pear.deploy.station.control.robots.yam.motion.actions import BimanualDelta, CartesianDelta, GripperAction
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.motion.planner import PlannerProcess, PlanningWorker
from pear.deploy.station.control.robots.yam.motion.planning import YAMPlanning
from pear.deploy.station.control.schemas import Observation
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import StandInYAMRobot
from yam_rigs import MotionHelpers

ZERO = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)  # one arm's six quantities, all zero
# a wide station: limits wide enough for every move of the tests, a velocity tolerance of 0.03 rad/s, Home at a third
# of the move caps with no caps of its own and 20 s for its arrival, and no cameras
WIDE_STATION = (
    *ConfigFiles.WIDE_LIMITS["yam"],
    "robot.velocity_tolerance=0.03",
    "robot.home.speed_scale=0.3333333333333333",
    "robot.home.limits=null",
    "robot.home.arrival_timeout_s=20",
    "robot.cameras=[]",
)
J1 = 0  # the index of the joint each arm turns away from Home
J4 = 3  # the index of the joint that rests outside its lower limit
GRIPPER = 6  # the index of an arm's gripper among its coordinates
OUTSIDE_RAD = 0.01  # how far past its limit a joint rests, within the feedback tolerance
OPENING_PAST = 0.005  # how far past fully open a gripper rests, within the feedback tolerance
AWAY_RAD = 0.3  # each arm's base turn away from Home, so that Home has to move
LIFT_CM = 1.0  # a Cartesian step
GRIPPER_TOLERANCE = 0.0625  # a configured gripper tolerance unlike the default, exact in binary
# where the left arm rests past a limit when it is enabled, for the tests of the actions that still work
OUTSIDE_CASES = ["joint 4 below its lower limit", "gripper past fully open"]


class PlannerWorkers:
    """Planner worker processes that hang or answer late, in place of the planner's own."""

    HANG_SECONDS = 60  # far longer than any planner timeout
    LATE_SECONDS = 0.2  # delay of the late answer

    @staticmethod
    def hanging(connection: Connection, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        """Starts as the planner does, then never answers its request and ignores SIGTERM."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        connection.send((True, os.getpid()))
        connection.recv()
        time.sleep(PlannerWorkers.HANG_SECONDS)

    @staticmethod
    def late(connection: Connection, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        """Starts as the planner does, then answers its request after LATE_SECONDS."""
        connection.send((True, os.getpid()))
        connection.recv()
        time.sleep(PlannerWorkers.LATE_SECONDS)
        connection.send((True, "late-result"))
        connection.recv()


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


class HomePlans:
    """The poses and the observations the tests of Home's plan start from, and the times a move is sampled at."""

    @staticmethod
    def distant_pose() -> np.ndarray:
        """Returns both arms' joints away from Home: the bases turned, the grippers partly open."""
        q = np.zeros(14)
        q[[0, 7]] = [0.6, -0.6]
        q[[6, 13]] = [0.4, 0.7]
        return q

    @staticmethod
    def observation(control_loop: YAMControlLoop) -> Observation:
        """Returns an observation of the control loop's current snapshot, as the planner receives it."""
        snapshot = control_loop.snapshot()
        return {"joint_positions": SnapshotCheck.state_vector(snapshot), "control_snapshot": snapshot}

    @staticmethod
    def sample_times(move: JointMove, rate_hz: float) -> np.ndarray:
        """Returns uniform times of ``move`` at ``rate_hz``, both ends included."""
        return np.linspace(0, move.duration_s, max(2, int(np.ceil(move.duration_s * rate_hz)) + 1))


@pytest.fixture
def outside_limit_rig(
    stand_in_clock: StandInClock,
) -> Iterator[tuple[YAMControlLoop, StandInYAMDriver, Ticker, tuple[float, ...]]]:
    """A YAM control loop on the wide station, stepped by hand, enabled with the left arm's J4 resting outside its
    lower limit: the loop, its driver, its ticker and the left arm's resting position; closed at the end."""
    control_config, config = ConfigFiles.load_configs("yam", *WIDE_STATION)
    hardware = StandInYAMDriver(control_config, config)
    resting = list(hardware.positions["left"])
    resting[J4] = config.arms["left"].limits.lower_bounds[J4] - OUTSIDE_RAD
    hardware.positions = {**hardware.positions, "left": tuple(resting)}
    runtime = YAMControlLoop(hardware, control_config, config)

    yield runtime, hardware, Ticker(runtime, stand_in_clock), tuple(resting)

    runtime.close()


@pytest.fixture
def past_limit_config() -> tuple[ControlConfig, YAMConfig]:
    """The configurations of the YAM station of the tests."""
    return ConfigFiles.load_configs("yam")


@pytest.fixture
def past_limit_rig(
    stand_in_clock: StandInClock, past_limit_config: tuple[ControlConfig, YAMConfig], outside: str
) -> Iterator[tuple[YAMControlLoop, StandInYAMDriver, Ticker]]:
    """A YAM control loop, stepped by hand, enabled with both bases turned away from Home and the left arm past the
    limit ``outside`` names: the loop, its driver and its ticker; closed at the end."""
    control_config, config = past_limit_config
    hardware = StandInYAMDriver(control_config, config)
    limits = config.arms["left"].limits
    resting = {arm: list(position) for arm, position in hardware.positions.items()}
    for position in resting.values():
        position[J1] = AWAY_RAD
    if outside.startswith("joint"):
        resting["left"][J4] = limits.lower_bounds[J4] - OUTSIDE_RAD
    else:
        resting["left"][GRIPPER] = limits.upper_bounds[GRIPPER] + OPENING_PAST
    hardware.positions = {arm: tuple(position) for arm, position in resting.items()}
    runtime = YAMControlLoop(hardware, control_config, config)

    yield runtime, hardware, Ticker(runtime, stand_in_clock)

    runtime.close()


def test_a_roll_increment_adds_to_the_fixed_axis_roll_of_each_arm(yam_kinematics):
    q = np.zeros(14)
    q[[6, 13]] = 1
    starts = {s: yam_kinematics.pose(s, q[i * 7 : (i + 1) * 7]) for i, s in enumerate(("left", "right"))}
    candidate = BimanualActions.action(left={"z": 1.0, "roll": 0.1}, right={"z": 1.5, "roll": 0.12})

    final = np.array(MotionHelpers.planned_target(yam_kinematics, q, candidate))

    for i, (side, height, roll) in enumerate((("left", 1.0, 0.1), ("right", 1.5, 0.12))):
        after = yam_kinematics.pose(side, final[i * 7 : (i + 1) * 7])
        before = starts[side]
        expected = Rotation.from_euler("xyz", YAMKinematics.matrix_to_rpy(before[:3, :3]) + [roll, 0, 0]).as_matrix()
        assert after[:3, 3] - before[:3, 3] == pytest.approx([0, 0, height / 100], abs=0.0005)
        assert after[:3, :3] == pytest.approx(expected, abs=0.005)
        wrong = Rotation.from_rotvec([roll, 0, 0]).as_matrix() @ before[:3, :3]
        assert np.max(abs(after[:3, :3] - wrong)) > 0.05
    assert final[[6, 13]] == pytest.approx([1, 1])


@pytest.mark.parametrize("moving", ["left", "right"])
def test_zero_part_keeps_command_target_despite_static_error(yam_kinematics, moving):
    q = np.zeros(14)
    q[[6, 13]] = 1
    state = MotionHelpers.snapshot(q)
    inactive = "left" if moving == "right" else "right"
    state["arms"][inactive]["position"][3] += 0.004
    measured = np.array([v for side in ("left", "right") for v in state["arms"][side]["position"]])
    station = SimpleNamespace(config=yam_kinematics.robot_config, status=lambda: copy.deepcopy(state))
    candidate = BimanualActions.action(**{moving: {"z": 1.0}})

    plan = YAMPlanning.checked_action(
        station.config,
        yam_kinematics,
        {"joint_positions": measured, "control_snapshot": state},
        candidate,
    )

    index = 0 if inactive == "left" else 7
    assert np.asarray(plan.move.goal)[index : index + 7] == pytest.approx(q[index : index + 7])


@pytest.mark.parametrize("arm", ["left", "right"], ids=["further_out", "out_from_inside"])
def test_cartesian_goal_past_the_limit_is_rejected(outside_limit_rig, arm):
    runtime, hardware, _, _ = outside_limit_rig
    snapshot = runtime.snapshot()
    held = SnapshotCheck.target_vector(snapshot)
    joint = LAYOUT.arms.index(arm) * LAYOUT.coordinate_count + J4
    goal = held.copy()
    goal[joint] = min(held[joint], runtime.limits.lower_bounds[joint]) - OUTSIDE_RAD
    kinematics = SimpleNamespace(joint_target=lambda q, arms: goal.copy())  # IK that solves to `goal`
    lift = CartesianDelta(side=arm, delta_position_cm=[0, 0, LIFT_CM], delta_rpy_rad=[0, 0, 0])
    sent = len(hardware.sent_targets)

    with pytest.raises(ValueError, match=f"{arm}: joint or gripper target outside commissioned bounds: joint 4="):
        YAMPlanning.checked_action(
            runtime.robot_config,
            kinematics,
            {"joint_positions": SnapshotCheck.state_vector(snapshot), "control_snapshot": snapshot},
            lift,
        )

    assert len(hardware.sent_targets) == sent


def test_home_path_shifted_past_the_measured_pose_is_rejected():
    control_config, config = ConfigFiles.load_configs("yam", *WIDE_STATION)
    lower = config.arms["left"].limits.lower_bounds[J4]
    config.home.left_pose[J4] = lower
    held = np.array(config.home.left_pose + config.home.right_pose, dtype=float)
    held[J4] = lower + OUTSIDE_RAD
    measured = held.copy()
    measured[J4] = lower - OUTSIDE_RAD
    # The move lowers J4 to the limit; shifted by the measured offset it ends past the measured pose.
    robot = {
        "arms": {arm: {"position": q} for arm, q in LAYOUT.split(measured).items()},
        "targets": LAYOUT.split(held),
    }

    with pytest.raises(HomePlanningError, match="joint 4="):
        YAMPlanning.checked_home(
            control_config, config, {"joint_positions": measured, "control_snapshot": robot}, "both", True
        )


@pytest.mark.parametrize("outside", OUTSIDE_CASES)
def test_home_plan_opening_the_grippers_is_accepted(past_limit_rig, past_limit_config):
    runtime, _, _ = past_limit_rig
    control_config, config = past_limit_config

    plan, move = YAMPlanning.checked_home(control_config, config, HomePlans.observation(runtime), "both", True)

    assert move is not None and plan["opening_commands"]


@pytest.mark.parametrize("outside", OUTSIDE_CASES)
def test_home_keeping_the_grippers_runs_to_a_hold(past_limit_rig, past_limit_config):
    runtime, _, ticker = past_limit_rig
    control_config, config = past_limit_config
    observation = HomePlans.observation(runtime)

    _, move = YAMPlanning.checked_home(control_config, config, observation, "both", False)
    YAMPlanning.recheck_home_move(config, observation, move)
    runtime.dispatch_joint_move(move)
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert np.array(LAYOUT.join(runtime.targets)) == pytest.approx(move.goal)


@pytest.mark.parametrize("outside", OUTSIDE_CASES)
def test_cartesian_action_of_the_other_arm_runs(past_limit_rig, past_limit_config):
    runtime, _, ticker = past_limit_rig
    _, config = past_limit_config
    held = runtime.targets["left"]
    lift = CartesianDelta(side="right", delta_position_cm=[0, 0, LIFT_CM], delta_rpy_rad=[0, 0, 0])

    prepared = YAMPlanning.checked_action(config, YAMKinematics(config), HomePlans.observation(runtime), lift)
    runtime.dispatch_joint_move(prepared.move)
    ticker.until_hold()

    assert runtime.status == "HOLD", runtime.fault
    assert runtime.targets["left"] == held


@pytest.mark.parametrize(
    "case",
    json.loads((ConfigFiles.YAM_RECORDED_CASES / "home_tracking_residuals.json").read_text())["cases"],
    ids=lambda case: case["source"],
)
def test_the_residuals_recorded_at_home_stay_within_the_bounds(case):
    goal = np.zeros(14)
    goal[[6, 13]] = 1

    YAMPlanning.check_home_bounds(ConfigFiles.load_configs("yam")[1], None, goal, np.asarray(case["residual"]))


@pytest.mark.parametrize("has_move", [False, True], ids=["at_home", "inward_move"])
def test_home_bounds_refuse_feedback_past_the_feedback_tolerance(has_move):
    control_config, config = ConfigFiles.load_configs("yam")
    q = HomePlans.distant_pose()
    plan, move = YAMPlanning.checked_home(
        control_config, config, {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}, "both", True
    )
    start = np.asarray(plan["start_targets"] if has_move else plan["goal_targets"])
    # The left base joint reads past its upper bound, on the side the move leaves.
    offset = np.zeros(14)
    offset[0] = config.arms["left"].limits.upper_bounds[0] + 2 * config.feedback_position_tolerance - start[0]

    with pytest.raises(ValueError, match="outside commissioned bounds"):
        YAMPlanning.check_home_bounds(config, move if has_move else None, start, offset)


@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_distant_home_opens_selected_grippers_and_retains_inactive_arm(yam_kinematics, side):
    control_config = ConfigFiles.load_configs("yam")[0]
    q = HomePlans.distant_pose()

    plan, move = YAMPlanning.checked_home(
        control_config,
        yam_kinematics.robot_config,
        {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)},
        side,
        True,
    )

    assert move is not None
    previous = q.copy()
    for i, name in enumerate(("left", "right")):
        if side in (name, "both"):
            previous[i * 7 + 6] = 1.0
    expected_grippers = previous[[6, 13]].copy()
    caps = np.array(
        yam_kinematics.robot_config.joint_limits().motion_caps(
            yam_kinematics.robot_config.home.speed_scale,
            caps=yam_kinematics.robot_config.home.limits,
        )
    )
    assert move.start == pytest.approx(previous)
    assert np.asarray(move.motion_caps) == pytest.approx(caps)
    assert move.at_time(0)[1] == pytest.approx(np.zeros(14))
    assert move.at_time(move.duration_s)[1] == pytest.approx(np.zeros(14))
    for t in HomePlans.sample_times(move, control_config.rate_hz):
        assert move.at_time(t)[0][[6, 13]] == pytest.approx(expected_grippers)
    previous = np.array(move.goal)
    for i, name in enumerate(("left", "right")):
        expected = np.zeros(6) if side in (name, "both") else q[i * 7 : i * 7 + 6]
        assert previous[i * 7 : i * 7 + 6] == pytest.approx(expected)


@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_keep_grippers_home_moves_only_arm_joints(side):
    control_config, config = ConfigFiles.load_configs("yam")
    q = HomePlans.distant_pose()

    plan, move = YAMPlanning.checked_home(
        control_config, config, {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}, side, False
    )

    assert plan["open_grippers"] is False
    assert plan["opening_commands"] == {} and plan["opening_seconds_estimate"] == 0
    assert move.start == pytest.approx(q)
    for t in HomePlans.sample_times(move, control_config.rate_hz):
        assert move.at_time(t)[0][[6, 13]] == pytest.approx(q[[6, 13]])
    goal = np.array(move.goal)
    for i, name in enumerate(("left", "right")):
        expected = config.home.arm_pose(name)[:6] if side in (name, "both") else q[i * 7 : i * 7 + 6]
        assert goal[i * 7 : i * 7 + 6] == pytest.approx(expected)


@pytest.mark.parametrize("overlap", ["self", "inter_arm"])
def test_home_accepts_overlapping_robot_models(overlap):
    control_config, config = ConfigFiles.load_configs("yam")
    q = HomePlans.distant_pose()
    if overlap == "self":
        q[10] = -1
    else:
        q[[0, 7]] = [-1.5, 1.5]
        q[[1, 3, 8, 10]] = 1

    plan, move = YAMPlanning.checked_home(
        control_config, config, {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}, "both", True
    )

    assert move is not None
    assert plan["opening_commands"] == {"left": True, "right": True}
    assert move.goal == pytest.approx(config.home.left_pose + config.home.right_pose)


def test_home_from_below_base_plane_is_allowed(yam_kinematics):
    control_config = ConfigFiles.load_configs("yam")[0]
    q = np.zeros(14)
    q[[6, 13]] = 1
    q[8], q[9] = 2.0, 0.5
    assert yam_kinematics.pose("right", q[7:])[2, 3] < 0

    plan, move = YAMPlanning.checked_home(
        control_config,
        yam_kinematics.robot_config,
        {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)},
        "both",
        True,
    )

    assert plan["opening_commands"] == {}
    assert move is not None
    assert move.goal[7:13] == pytest.approx(yam_kinematics.robot_config.home.right_pose[:6])


def test_initial_home_checks_residual_joint_bounds_before_opening():
    control_config, config = ConfigFiles.load_configs("yam")
    config.home.left_pose[2] = config.arms["left"].limits.lower_bounds[2]
    held = HomePlans.distant_pose()
    measured = held.copy()
    measured[2] = -0.036

    with pytest.raises(HomePlanningError, match="outside commissioned bounds"):
        YAMPlanning.checked_home(
            control_config,
            config,
            {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(held)},
            "both",
            True,
        )


@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_home_accepts_gripper_feedback_just_past_open_endpoint(side):
    # An opened gripper reading slightly past the open endpoint must not block a later Home.
    control_config, config = ConfigFiles.load_configs("yam")
    targets = HomePlans.distant_pose()
    targets[[6, 13]] = 1.0
    measured = targets.copy()
    measured[[6, 13]] = 1.0001204810738407
    observation = {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(targets)}

    plan, move = YAMPlanning.checked_home(control_config, config, observation, side, True)

    assert plan["opening_commands"] == {}
    YAMPlanning.recheck_home_move(config, observation, move)


@pytest.mark.parametrize("short", [0.5, 1.0, 2.0], ids=["within", "at", "beyond"])
def test_an_open_gripper_reopens_only_beyond_the_gripper_tolerance(short):
    control_config, config = ConfigFiles.load_configs("yam")
    config.gripper_tolerance = GRIPPER_TOLERANCE
    targets = HomePlans.distant_pose()
    targets[[6, 13]] = 1.0
    measured = targets.copy()
    measured[6] = 1.0 - short * GRIPPER_TOLERANCE

    plan, _ = YAMPlanning.checked_home(
        control_config,
        config,
        {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(targets)},
        "both",
        True,
    )

    assert plan["opening_commands"] == ({"left": True} if short > 1 else {})


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_home_estimate_uses_actual_gripper_caps_and_longest_selected_profile(yam_kinematics, side):
    from ruckig import InputParameter, Ruckig, Trajectory

    control_config = ConfigFiles.load_configs("yam")[0]
    config = copy.deepcopy(yam_kinematics.robot_config)
    config.arms["right"].limits.velocity[6] = 0.1
    config.home.settle_window_s = 0.2
    q = HomePlans.distant_pose()

    plan, move = YAMPlanning.checked_home(
        control_config, config, {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}, side, True
    )

    durations = []
    for name in plan["active_sides"]:
        i = 6 if name == "left" else 13
        limits = config.arms[name].limits
        inp, native = InputParameter(1), Trajectory(1)
        inp.current_position, inp.target_position = [q[i]], [1.0]
        inp.max_velocity = [limits.velocity[6]]
        inp.max_acceleration = [limits.acceleration[6]]
        inp.max_jerk = [limits.jerk[6]]
        assert Ruckig(1).calculate(inp, native) >= 0
        durations.append(native.duration)
    assert plan["opening_seconds_estimate"] == pytest.approx(max(durations))
    assert plan["operation_seconds_estimate"] == pytest.approx(
        max(durations) + move.duration_s + 2 * control_config.settle_window_s
    )


@pytest.mark.parametrize("change", ["none", "feedback", "targets"])
def test_home_recheck_validates_fresh_joint_bounds_and_approved_targets(change):
    control_config, config = ConfigFiles.load_configs("yam")
    config.home.left_pose[2] = config.arms["left"].limits.lower_bounds[2]
    q = HomePlans.distant_pose()
    plan, move = YAMPlanning.checked_home(
        control_config, config, {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}, "both", True
    )
    targets = np.asarray(plan["start_targets"])
    measured = targets.copy()
    if change == "feedback":
        measured[2] -= 0.02
    elif change == "targets":
        targets[0] += 0.01
    observation = {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(targets)}

    if change == "none":
        YAMPlanning.recheck_home_move(config, observation, move)
    else:
        message = "outside commissioned bounds" if change == "feedback" else "Home targets changed"
        with pytest.raises(ValueError, match=message):
            YAMPlanning.recheck_home_move(config, observation, move)


def test_cartesian_move_below_base_plane_is_allowed(yam_kinematics):
    q = np.zeros(14)
    q[[6, 13]] = 1
    q[8], q[9] = 2.0, 0.5
    assert yam_kinematics.pose("right", q[7:])[2, 3] < 0

    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config,
        yam_kinematics,
        {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)},
        MotionHelpers.action(side="right", xyz=(0, 0, 1)),
    )

    assert prepared.move is not None


@pytest.mark.parametrize("overlap", ["self", "inter_arm"])
def test_cartesian_planning_accepts_overlapping_robot_models(yam_kinematics, overlap):
    q = np.zeros(14)
    q[[6, 13]] = 1
    if overlap == "self":
        q[10] = -1
    else:
        q[[0, 7]] = [-1.5, 1.5]
        q[[1, 3, 8, 10]] = 1

    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config,
        yam_kinematics,
        {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)},
        MotionHelpers.action(side="right", xyz=(0, 0, 1)),
    )

    assert prepared.move is not None
    assert prepared.move.goal[:7] == pytest.approx(q[:7])


def test_a_checked_move_keeps_the_idle_arm_and_a_zero_action_plans_no_move(yam_kinematics):
    q = np.zeros(14)
    q[[6, 13]] = 1
    state = MotionHelpers.snapshot(q)
    state["arms"]["left"]["position"][3] = 0.003
    state["arms"]["right"]["position"][2] = 0.004
    observation = {"joint_positions": SnapshotCheck.state_vector(state), "control_snapshot": state}

    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config, yam_kinematics, observation, MotionHelpers.action(xyz=(0, 0, 1))
    )

    assert prepared.move.goal[:7] == pytest.approx(q[:7])
    assert prepared.move.goal[13] == 1
    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config, yam_kinematics, observation, MotionHelpers.action()
    )
    assert prepared.move is None and not prepared.gripper_request


def test_gripper_actions_plan_open_and_closed_endpoints(yam_kinematics):
    q = np.zeros(14)
    q[[6, 13]] = 0.5
    state = MotionHelpers.snapshot(q)
    station = SimpleNamespace(config=yam_kinematics.robot_config, status=lambda: state)

    for command in ("close", "open"):
        a = GripperAction(side="right", command=command)
        prepared = YAMPlanning.checked_action(
            station.config,
            yam_kinematics,
            {"joint_positions": q, "control_snapshot": state},
            a,
        )
        assert prepared.move is None
        assert prepared.gripper_request == {"right": command == "open"}


def test_spawned_planner_keeps_live_stand_in_controller_running_and_rejects_an_outdated_observation():
    config = ConfigFiles.load("yam")
    planner = PlannerProcess(*ConfigFiles.make_configs(config))
    assert planner.worker.process.pid != os.getpid()

    try:
        with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
            station.wait_stationary()
            observation = station.observe()
            a = MotionHelpers.action(xyz=(2, 9, 0), rotation=(0.2, 0, 0))
            move = planner.plan(station, observation, a).move
            assert np.max(move.motion_caps[0]) <= 1
            station.control_loop.dispatch_joint_move(move)
            after = station.control_loop.wait_for_hold(move.duration_s + 5)
            assert after["motion"]["end_reason"] == "finished"
            assert after["fault"] is None
            with pytest.raises(ValueError, match="Observation invalidated by another operation"):
                planner.plan(station, observation, a)
            assert station.control_loop.snapshot()["status"] == "HOLD"
            invalid = MotionHelpers.action(xyz=(100, 0, 0))
            with pytest.raises(ValueError, match="out of reach"):
                planner.plan(station, station.observe(), invalid)
            assert station.control_loop.snapshot()["status"] == "HOLD"
    finally:
        planner.close()


def test_idle_planner_survives_terminal_sigint_and_can_plan_again():
    control_config, config = ConfigFiles.load_configs("yam")
    planner = PlannerProcess(control_config, config)
    q = np.zeros(14)
    q[[6, 13]] = 1
    station = SimpleNamespace(control_period_s=control_config.control_period_s, verify_observation=lambda obs: None)
    observation = {"joint_positions": q, "control_snapshot": MotionHelpers.snapshot(q)}

    try:
        os.kill(planner.worker.process.pid, signal.SIGINT)
        prepared = planner.plan(station, observation, MotionHelpers.action())
        assert not prepared.gripper_request and prepared.move is None
        assert planner.worker.process.is_alive()
    finally:
        planner.close()


def test_recorded_gripper_opening_below_base_plane_is_allowed(yam_kinematics):
    case = json.loads((ConfigFiles.YAM_RECORDED_CASES / "gripper_below_base_plane.json").read_text())
    measured, commanded = np.array(case["measured"]), np.array(case["commanded"])

    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config,
        yam_kinematics,
        {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(commanded)},
        GripperAction(side="right", command="open"),
    )

    assert prepared.move is None and prepared.gripper_request == {"right": True}


@pytest.mark.parametrize("worker", [PlannerWorkers.hanging, PlannerWorkers.late])
def test_planner_timeout_has_bounded_cleanup_and_discards_late_results(monkeypatch, worker):
    # the planner process runs the stand-in worker in place of the planner's own
    monkeypatch.setattr(PlanningWorker, "main", worker)
    control_config, config = ConfigFiles.load_configs("yam")
    planner = PlannerProcess(control_config, config)
    station = SimpleNamespace(control_period_s=control_config.control_period_s, verify_observation=lambda obs: None)
    observation = {"joint_positions": np.zeros(14), "control_snapshot": {}}
    start = time.monotonic()

    with pytest.raises(TimeoutError):
        planner.plan(station, observation, MotionHelpers.action(), timeout_s=0.05)

    assert time.monotonic() - start < 3.5
    assert not planner.worker.process.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        planner.plan(station, observation, MotionHelpers.action())
    planner.close()


def test_planning_keeps_held_targets_without_building_unused_fk_reports(yam_kinematics, monkeypatch):
    q = np.zeros(14)
    q[[6, 13]] = [0.6, 0.7]
    measured = q.copy()
    measured[3] = 0.004

    def forbidden(*args: Any) -> None:
        raise AssertionError("Execution planning built an unused FK report")

    monkeypatch.setattr(yam_kinematics, "tool_poses", forbidden)

    prepared = YAMPlanning.checked_action(
        yam_kinematics.robot_config,
        yam_kinematics,
        {"joint_positions": measured, "control_snapshot": MotionHelpers.snapshot(q)},
        MotionHelpers.action(xyz=(0, 0, 1)),
    )

    assert prepared.move.goal[:7] == pytest.approx(q[:7])
