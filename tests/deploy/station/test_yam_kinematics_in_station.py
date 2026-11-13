import copy
import math
import re
import socket

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.motion.kinematics import UnreachablePoseError
from pear.deploy.station.control.robots.yam.motion.actions import BimanualDelta
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from yam_rigs import MotionHelpers

# the measured joints the tests plan from: both arms at zero, the grippers partly open
HOME = np.array([0.0] * 6 + [0.8] + [0.0] * 6 + [0.7])
# what the report of an unreachable arm says it can do: "at most <percent>%"
REPORTED_REACH = re.compile(r"at most (\d+)%")


def test_model_loading_and_planning_never_open_hardware(monkeypatch):
    from i2rt.robots import get_robot

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline planning tried to open hardware")

    original_socket = socket.socket

    def guarded_socket(family=socket.AF_INET, *args, **kwargs):
        if family == getattr(socket, "AF_CAN", -1):
            forbidden()
        return original_socket(family, *args, **kwargs)

    monkeypatch.setattr(get_robot, "get_yam_robot", forbidden)
    monkeypatch.setattr(YAMRobot, "open", forbidden)
    monkeypatch.setattr(socket, "socket", guarded_socket)
    model = YAMKinematics(ConfigFiles.load_configs("yam")[1])

    assert len(MotionHelpers.planned_target(model, HOME, MotionHelpers.action(xyz=(0, 0, 1)))) == 14


def test_the_tool_pose_at_rest_is_reported_in_centimetres_in_the_tool_frame(yam_kinematics):
    assert np.linalg.det(YAMKinematics.GRASP_TO_TOOL) == pytest.approx(1)
    poses = yam_kinematics.tool_poses(HOME)
    for side in ("left", "right"):
        pose = poses[side]
        assert pose.rpy_rad == pytest.approx([-math.pi / 2, 0, math.pi / 2])
        assert pose.position_cm[0] == pytest.approx(25.79822, abs=0.001)
        assert pose.position_cm[2] == pytest.approx(17.850419, abs=0.001)
    assert poses["left"].position_cm[1] - poses["right"].position_cm[1] == pytest.approx(61)


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_cm_lift_is_one_hundredth_meter_and_other_arm_holds(yam_kinematics, side):
    offset = 0 if side == "left" else 7
    before = yam_kinematics.pose(side, HOME[offset : offset + 7])

    result = MotionHelpers.planned_target(yam_kinematics, HOME, MotionHelpers.action(side=side, xyz=(0, 0, 1)))

    final = np.array(result)
    after = yam_kinematics.pose(side, final[offset : offset + 7])
    assert after[:3, 3] - before[:3, 3] == pytest.approx([0, 0, 0.01], abs=0.0005)
    observed = yam_kinematics.tool_poses(final)[side]
    assert observed.position_cm == pytest.approx(100 * after[:3, 3])
    other = slice(7, 14) if side == "left" else slice(0, 7)
    assert final[other] == pytest.approx(HOME[other])
    assert final[[6, 13]] == pytest.approx(HOME[[6, 13]])


def test_zero_action_preserves_joints_and_grippers(yam_kinematics):
    assert MotionHelpers.planned_target(yam_kinematics, HOME, MotionHelpers.action()) == pytest.approx(HOME)


def test_rotation_and_translation_use_unrounded_observed_pose(yam_kinematics):
    from scipy.spatial.transform import Rotation

    state = HOME.copy()
    state[1:6] = [0.01, -0.02, 0.01, -0.01, 0.01]
    candidate = MotionHelpers.action(side="left", xyz=(0, 0, 1.234), rotation=(0.15, -0.03, 0))
    before = yam_kinematics.pose("left", state[:7])

    result = MotionHelpers.planned_target(yam_kinematics, state, candidate)

    pose = yam_kinematics.pose("left", np.array(result)[:7])
    assert pose[:3, 3] == pytest.approx(before[:3, 3] + [0, 0, 0.01234], abs=0.0005)
    expected = Rotation.from_euler(
        "xyz", Rotation.from_matrix(before[:3, :3]).as_euler("xyz") + candidate.delta_rpy_rad
    ).as_matrix()
    assert pose[:3, :3] == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("side", ["left", "right"])
def test_reachable_translation_over_25_cm_reaches_requested_pose(yam_kinematics, side):
    offset = 0 if side == "left" else 7
    before = yam_kinematics.pose(side, HOME[offset : offset + 7])
    goal = HOME[offset : offset + 7].copy()
    goal[0] = math.pi / 2
    target = yam_kinematics.pose(side, goal)
    delta_cm = 100 * (target[:3, 3] - before[:3, 3])
    assert np.linalg.norm(delta_cm) > 25
    candidate = MotionHelpers.action(
        side=side,
        xyz=delta_cm,
        rotation=YAMKinematics.matrix_to_rpy(target[:3, :3]) - YAMKinematics.matrix_to_rpy(before[:3, :3]),
    )

    result = np.asarray(MotionHelpers.planned_target(yam_kinematics, HOME, candidate))

    after = yam_kinematics.pose(side, result[offset : offset + 7])
    assert after[:3, 3] == pytest.approx(target[:3, 3], abs=0.0005)
    assert after[:3, :3] == pytest.approx(target[:3, :3], abs=0.005)
    other = slice(7, 14) if side == "left" else slice(0, 7)
    assert result[other] == pytest.approx(HOME[other])


def test_unreachable_large_translation_is_rejected_by_ik(yam_kinematics):
    with pytest.raises(UnreachablePoseError, match="right_arm: the target is out of reach"):
        MotionHelpers.planned_target(yam_kinematics, HOME, MotionHelpers.action(xyz=(100, 0, 0)))


def test_unreachable_report_names_every_arm_and_only_unreachable_ones(yam_kinematics):
    far = {s: MotionHelpers.action(side=s, xyz=(100, 0, 0)) for s in ("left", "right")}

    with pytest.raises(UnreachablePoseError) as error:
        yam_kinematics.joint_target(HOME, far)

    message = str(error.value)
    assert message.startswith("left_arm: the target is out of reach. Along the requested")
    assert " right_arm: at most " in message
    with pytest.raises(UnreachablePoseError) as error:
        yam_kinematics.joint_target(
            HOME, {"left": MotionHelpers.action(side="left", xyz=(0, 0, 1)), "right": far["right"]}
        )
    assert "left_arm" not in str(error.value)


def test_reported_reach_fraction_is_solvable(yam_kinematics):
    far = MotionHelpers.action(xyz=(100, 0, 5))
    # the fraction the public report gives: the reach and the solve of one arm are private to the kinematics
    with pytest.raises(UnreachablePoseError) as error:
        yam_kinematics.joint_target(HOME, {"right": far})
    fraction = int(REPORTED_REACH.search(str(error.value)).group(1)) / 100

    assert 0 < fraction < 1
    yam_kinematics.joint_target(HOME, {"right": MotionHelpers.action(xyz=(100 * fraction, 0, 5 * fraction))})


def test_unreachable_message_truncates_toward_zero_and_keeps_direction():
    left = MotionHelpers.action(side="left", xyz=(10, -2, 0), rotation=(0, 0.5, 0))
    right = MotionHelpers.action(xyz=(10, 0, 0))

    assert YAMKinematics.unreachable_message([("left", left, 0.0199), ("right", right, 0.307)]) == (
        "left_arm: the target is out of reach. Along the requested change it can go at most "
        "1% (x +0.1 of +10, y 0.0 of -2, pitch 0.00 of +0.5). "
        "right_arm: at most 30% (x +3.0 of +10)."
    )


def test_a_rotation_beyond_the_bound_is_rejected(yam_kinematics):
    with pytest.raises(ValueError, match="rotation"):
        MotionHelpers.planned_target(yam_kinematics, HOME, MotionHelpers.action(rotation=(math.pi + 0.1, 0, 0)))


def test_an_ik_failure_and_an_inaccurate_endpoint_are_rejected(yam_kinematics, monkeypatch):
    candidate = MotionHelpers.action(side="left", xyz=(0, 0, 1))
    monkeypatch.setattr(yam_kinematics.models["left"], "ik", lambda *a, **k: (False, k["init_q"]))

    with pytest.raises(UnreachablePoseError, match="out of reach"):
        MotionHelpers.planned_target(yam_kinematics, HOME, candidate)

    def flipped(*args, **kwargs):
        solved = copy.copy(kwargs["init_q"])
        solved[0] += 1
        return True, solved

    monkeypatch.setattr(yam_kinematics.models["left"], "ik", flipped)
    with pytest.raises(ValueError, match="FK residual"):
        MotionHelpers.planned_target(yam_kinematics, HOME, candidate)


@pytest.mark.parametrize("active", [("left",), ("right",), ("left", "right"), ()])
def test_ik_runs_once_per_moving_arm_and_never_for_a_zero_part(yam_kinematics, monkeypatch, active):
    calls = {side: 0 for side in ("left", "right")}
    seeds = {}
    model_qs = {}
    for side in calls:
        native = yam_kinematics.models[side].ik
        native_fk = yam_kinematics.models[side].fk

        def counted(*args, _side=side, _native=native, **kwargs):
            calls[_side] += 1
            seeds[_side] = kwargs["init_q"].copy()
            return _native(*args, **kwargs)

        # the model's joints of an arm's pose, which its forward kinematics receive: the model's joints of the
        # measured pose are private to the kinematics
        def recorded_fk(q, _side=side, _native_fk=native_fk):
            model_qs[_side] = np.array(q, dtype=float)
            return _native_fk(q)

        monkeypatch.setattr(yam_kinematics.models[side], "ik", counted)
        monkeypatch.setattr(yam_kinematics.models[side], "fk", recorded_fk)
    zero = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
    request = BimanualDelta(
        left_arm={**zero, "z": 4.0 if "left" in active else 0.0},
        right_arm={**zero, "z": 4.0 if "right" in active else 0.0},
        left_gripper={"width": 0.0},
        right_gripper={"width": 0.0},
    )

    result = MotionHelpers.planned_target(yam_kinematics, HOME, request)

    assert calls == {side: int(side in active) for side in calls}
    for side in active:
        offset = 0 if side == "left" else 7
        yam_kinematics.pose(side, HOME[offset : offset + 7])
        assert seeds[side] == pytest.approx(model_qs[side])
        final = yam_kinematics.pose(side, np.asarray(result)[offset : offset + 7])
        before = yam_kinematics.pose(side, HOME[offset : offset + 7])
        assert final[:3, 3] - before[:3, 3] == pytest.approx([0, 0, 0.04], abs=0.0005)
