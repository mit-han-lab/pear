import copy
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import StandInClock
from pear.deploy.station.agent_loop.agent_loop import StationAgentLoop
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.robot import ActionMode
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.motion.actions import BimanualDelta, CartesianDelta, GripperAction
from stand_in_clients import ModelReply, StandInScriptedClient
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import StandInYAMRobot, SteppedStandInYAMRobot

ZERO = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)  # one arm's six quantities, all zero
MODEL_TIMEOUT_S = 60.0  # how long the model may take to answer a turn
# both arms' joints where their models overlap, the right gripper closed
OVERLAPPING_POSE = (-1.5, 1, 0, 1, 0, 0, 1, 1.5, 1, 0, 1, 0, 0, 0)


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


class StandInOverlappingYAMDriver(StandInYAMDriver):
    """YAM's stand-in joints that start where the arms' models overlap."""

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        super().__init__(control_config, robot_config)
        q = np.array(OVERLAPPING_POSE, dtype=float)
        self.positions = {side: tuple(q[i * 7 : (i + 1) * 7]) for i, side in enumerate(("left", "right"))}


class StandInOverlappingYAMRobot(StandInYAMRobot):
    """The stand-in YAM robot, its arms starting where their models overlap."""

    DRIVER_CLASS = StandInOverlappingYAMDriver


@pytest.fixture
def station(monkeypatch: pytest.MonkeyPatch) -> Iterator[StandInYAMRobot]:
    """The open stand-in YAM robot in joint mode, whose sequences may run late; closed at the end."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A test attempted to construct real hardware")

    monkeypatch.setattr(YAMDriver, "__init__", forbidden)
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        yield station


@pytest.fixture
def rig(
    stand_in_clock: StandInClock,
) -> Iterator[tuple[SteppedStandInYAMRobot, YAMControlLoop, StandInYAMDriver, StandInClock, Callable[[float], None]]]:
    """The open stand-in YAM robot in joint mode, its control loop stepped by hand on the stand-in clock and settled:
    the robot, its loop, its driver, the clock and what advances them; closed at the end."""
    clock = stand_in_clock
    config = ConfigFiles.load("yam")

    with SteppedStandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        runtime = station.control_loop
        hardware = runtime.driver

        def advance(seconds: float) -> None:
            """Steps the control loop for ``seconds`` of the clock, which must not fault."""
            for _ in range(round(seconds / runtime.control_config.control_period_s)):
                clock.now += runtime.control_config.control_period_s
                runtime.step()
                assert runtime.status != "FAULT", runtime.fault

        advance(0.6)
        yield station, runtime, hardware, clock, advance


@pytest.mark.parametrize("robot_name", ["yam"])
def test_combined_action_executes_both_arm_moves_and_grippers_in_one_result(robot, yam_kinematics, tmp_path):
    candidate = BimanualActions.action(left={"z": 1.0}, right={"z": 1.0}, grips=(1.0, 1.0))

    def infer(req: Any, emit: Any, cancel: threading.Event) -> Any:
        return ModelReply.output(candidate)

    # Already-open grippers keep this integration test fast; contact is tested independently.
    robot.control_loop.move({s: [0.0] * 6 + [1.0] for s in ("left", "right")})
    robot.control_loop.wait_for_hold(robot.control_loop.HOLD_WAIT_S)
    initial = robot.observe()["joint_positions"]
    client = StandInScriptedClient(SystemPrompt.render("Both arms up", BIMANUAL_POSE), infer, timeout_s=MODEL_TIMEOUT_S)
    session = StationAgentLoop(
        robot,
        client,
        action_format=BIMANUAL_POSE,
        timeout_s=MODEL_TIMEOUT_S,
        directory=tmp_path,
        reference_images=(),
    )

    result = session.step(cancellation=threading.Event(), on_event=lambda kind, data: None, attempt_id=uuid4().hex)

    assert result.status == "completed", result
    assert result.gripper_outcomes == {"left": "open", "right": "open"}
    for i, side in enumerate(("left", "right")):
        before = yam_kinematics.pose(side, initial[i * 7 : (i + 1) * 7])
        after = yam_kinematics.pose(side, result.measured_position[i * 7 : (i + 1) * 7])
        assert after[:3, 3] - before[:3, 3] == pytest.approx([0, 0, 0.01], abs=0.0005)


@pytest.mark.parametrize("robot_name", ["yam"])
def test_a_cartesian_delta_and_a_gripper_action_execute(robot):
    before = robot.kinematics.tool_poses(robot.observe()["joint_positions"])["right"]

    result = robot.execute(
        CartesianDelta(side="right", delta_position_cm=[0.0, 0.0, 1.0], delta_rpy_rad=[0.0, 0.0, 0.0])
    )

    assert result.status == "completed", result
    after = robot.kinematics.tool_poses(robot.observe()["joint_positions"])["right"]
    assert np.array(after.position_cm) - before.position_cm == pytest.approx([0, 0, 1], abs=0.05)
    result = robot.execute(GripperAction(side="right", command="open"))
    assert result.status == "completed" and result.measured_position[13] == pytest.approx(1.0, abs=0.01)


def test_paused_gripper_executor_cannot_resume_after_stop(rig, monkeypatch):
    station, runtime, hardware, clock, advance = rig
    starts = []
    begin = runtime.begin_grippers

    def record_begin(*args, **kwargs):
        starts.append(clock.now)
        return begin(*args, **kwargs)

    def pause(_):
        advance(0.06)
        station.stop()
        advance(1.4)
        assert runtime.status == "HOLD"

    monkeypatch.setattr(runtime, "begin_grippers", record_begin)
    # every module sleeps on the stand-in clock: the first wait for the grippers pauses instead
    monkeypatch.setattr(time, "sleep", pause)

    with pytest.raises(RuntimeError, match="cancelled"):
        station.operate_grippers(
            {"right": True}, observation=station.observe(), guard=ActionGuard(runtime), deadline=None
        )

    assert len(starts) == 1
    assert runtime.status == "HOLD" and runtime.operation is None


def test_stop_between_cartesian_phases_prevents_gripper_submission(monkeypatch):
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        station.control_loop.wait_for_hold(station.control_loop.HOLD_WAIT_S)
        observation = station.observe()
        guard = ActionGuard(station.control_loop)
        original = station.observe

        def stop_between_phases(*args, **kwargs):
            station.stop()
            return original(*args, **kwargs)

        monkeypatch.setattr(station, "observe", stop_between_phases)
        candidate = PreparedAction({"right": True}, None)
        sent = len(station.control_loop.driver.sent_targets)

        with pytest.raises(RuntimeError, match="cancelled"):
            station.run_prepared(observation, candidate, guard=guard, deadline=None)

        assert len(station.control_loop.driver.sent_targets) == sent


def test_expired_deadline_sends_no_gripper_command(station, monkeypatch):
    runtime = station.control_loop
    station.wait_stationary()
    observation = station.observe()
    begin, begun = runtime.begin_grippers, []
    monkeypatch.setattr(runtime, "begin_grippers", lambda request: begun.append(request) or begin(request))
    before = len(runtime.driver.sent_targets)

    with pytest.raises(TimeoutError, match="total timeout"):
        station.operate_grippers(
            {"right": False},
            observation=observation,
            guard=ActionGuard(runtime),
            deadline=time.monotonic() - 1,
        )

    assert begun == []
    assert len(runtime.driver.sent_targets) == before


def test_a_gripper_closing_on_resistance_ends_in_a_hold_at_the_contact(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real hardware may not open in tests")

    monkeypatch.setattr(YAMDriver, "__init__", forbidden)
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        device = station.control_loop.driver
        send, read = device.send, device.read

        def blocked_send(value, velocities=None):
            actual = {s: list(q) for s, q in value.items()}
            actual["right"][6] = max(0.45, actual["right"][6])
            send(actual, velocities)

        def feedback():
            states = read()
            q = states["right"]
            effort = 1.03 if q.position[6] <= 0.450001 else 0
            states["right"] = YAMArmState(
                q.position,
                q.velocity,
                q.sampled_monotonic_s,
                effort=tuple([0.0] * 6 + [effort]),
                sdk_gripper_force_limited=bool(effort),
                sdk_gripper_target=q.position[6],
                sdk_gripper_force_limit_n=50.0,
                sdk_gripper_target_effort_nm=50 * 0.096 / 6.57 + 0.3,
            )
            return states

        monkeypatch.setattr(device, "send", blocked_send)
        monkeypatch.setattr(device, "read", feedback)
        initial = copy.deepcopy(station.control_loop.snapshot()["targets"])
        station.wait_stationary()

        report = station.operate_grippers(
            {"right": False},
            observation=station.observe(),
            guard=ActionGuard(station.control_loop),
            deadline=None,
        )

        assert report["completed"]
        assert report["outcomes"]["right"] == "resistance_detected_not_grasp_success"
        state = station.control_loop.snapshot()
        assert state["status"] == "HOLD" and state["fault"] is None
        assert state["targets"]["left"] == initial["left"]
        assert state["targets"]["right"][:6] == initial["right"][:6]
        assert state["targets"]["right"][6] >= 0.444 - 1e-9


def test_gripper_execution_accepts_overlapping_robot_models():
    q = np.array(OVERLAPPING_POSE, dtype=float)
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInOverlappingYAMRobot(config) as station:
        rt = station.control_loop
        station.wait_stationary()
        count = len(rt.driver.sent_targets)

        result = station.execute(GripperAction(side="right", command="open"))

        assert result.status == "completed"
        assert result.gripper_outcomes == {"right": "open"}
        assert len(rt.driver.sent_targets) > count
        expected = q.copy()
        expected[13] = 1
        assert SnapshotCheck.target_vector(rt.snapshot()) == pytest.approx(expected)
        for command in rt.driver.sent_targets[count:]:
            assert command["left"] == pytest.approx(q[:7])
            assert command["right"][:6] == pytest.approx(q[7:13])
