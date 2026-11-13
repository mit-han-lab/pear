from __future__ import annotations

import math

import numpy as np
import pytest

from pear.envs.robodojo.controller import MotionController

# a left wrist link's pose, rotated 90 degrees about world z
WRIST = [-0.2995, -0.3523, 0.9215, 0.7070, 0.0, 0.0, 0.7072]
OPEN = {"left": [0.044, 0.044], "right": [0.01, 0.01]}
# both tool points at world (-0.3, -0.2, 0.92), the gripper pointing forward (rotated 90 degrees about world z)
START_QUAT = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
START = {arm: [-0.3, -0.2, 0.92, *START_QUAT] for arm in MotionController.ARMS}
ZERO = {
    part: {quantity: 0.0 for quantity in quantities} for part, quantities in MotionController.SCHEMA.items()
}  # an action that holds
LIFT = {**ZERO, "left_arm": {**ZERO["left_arm"], "z": 5.0}}  # the left tool point 5 cm up


def test_the_tool_point_lies_its_offset_along_the_wrist_s_x_axis_and_the_link_target_inverts_it():
    tool_pose = MotionController.tool_pose_of_link(WRIST)
    x_axis = MotionController.quat_to_matrix(WRIST[3:])[:, 0]

    np.testing.assert_allclose(
        np.subtract(tool_pose[:3], WRIST[:3]), MotionController.TOOL_OFFSET_M[0] * x_axis, atol=1e-12
    )
    np.testing.assert_allclose(MotionController.link_pose_of_tool(tool_pose)[:3], WRIST[:3], atol=1e-12)


def test_interpolation_ends_on_its_poses_along_the_shorter_arc():
    start = MotionController.tool_pose_of_link(WRIST)
    turn = np.array([[math.cos(1.0), -math.sin(1.0), 0.0], [math.sin(1.0), math.cos(1.0), 0.0], [0.0, 0.0, 1.0]])
    turned = MotionController.matrix_to_quat(turn @ MotionController.quat_to_matrix(start[3:]))  # 1 rad about z
    goal = start[:3] + MotionController.quat_normalize(turned).tolist()
    goal[2] += 0.1

    np.testing.assert_allclose(MotionController.interpolate(start, goal, 0.0), start, atol=1e-12)
    np.testing.assert_allclose(MotionController.interpolate(start, goal, 1.0), goal, atol=1e-12)
    middle = MotionController.interpolate(start, goal, 0.5)
    assert abs(middle[2] - (start[2] + 0.05)) < 5e-13
    assert abs(MotionController.quat_angle(start[3:], middle[3:]) - 0.5) < 5e-10
    # q and -q are one orientation
    assert abs(MotionController.quat_angle(start[3:], [-q for q in start[3:]])) < 5e-7


def test_the_quantities_shown_of_the_start_pose_and_of_a_gripper_that_points_down():
    exact = MotionController.quantities(START, OPEN)
    shown = MotionController.state_of_quantities(exact)

    assert shown["left_arm"] == {"x": 25.0, "y": 30.0, "z": 15.5, "roll": -1.57, "pitch": 0.0, "yaw": 1.57}
    assert shown["right_arm"]["y"] == 30.0  # (both arms at world x -0.3 here)
    assert (shown["left_gripper"], shown["right_gripper"]) == ({"width": 8.8}, {"width": 2.0})
    # pointing down with the fingers closing along the robot's y axis: roll 0, pitch 0, yaw 1.57
    down = MotionController.matrix_to_quat(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T).tolist()
    exact_down = MotionController.quantities({arm: [-0.3, -0.2, 0.92, *down] for arm in MotionController.ARMS}, OPEN)
    angles = [MotionController.state_of_quantities(exact_down)["left_arm"][q] for q in ("roll", "pitch", "yaw")]
    assert angles == [0.0, 0.0, 1.57]
    for values, quaternion in ((exact, START_QUAT), (exact_down, down)):
        pose = MotionController.tool_pose_of_quantities(values["left_arm"])
        np.testing.assert_allclose(pose[:3], [-0.3, -0.2, 0.92], atol=1e-9)
        alignment = abs(float(np.dot(pose[3:], MotionController.quat_normalize(quaternion))))
        assert abs(alignment - 1.0) < 5e-10


def test_a_change_moves_the_tool_point_along_the_robot_axes():
    left = MotionController.quantities(START, OPEN)["left_arm"]

    for axis, world in (("x", [0, 0.05, 0]), ("y", [-0.05, 0, 0]), ("z", [0, 0, 0.05])):
        moved = MotionController.tool_pose_of_quantities({**left, axis: left[axis] + 5})
        np.testing.assert_allclose(np.subtract(moved[:3], START["left"][:3]), world, atol=1e-9)


def test_quantities_and_tool_poses_invert_each_other():
    rng = np.random.default_rng(0)

    for _ in range(20):
        quaternion = MotionController.quat_normalize(rng.normal(size=4))
        position = rng.normal(size=3) * 0.3 + [0.0, -0.3, 1.0]
        poses = {"left": [*position, *quaternion], "right": [*(position + 0.1), *quaternion]}
        exact = MotionController.quantities(poses, {"left": [0.01, 0.01], "right": [0.0, 0.02]})
        back = MotionController.tool_pose_of_quantities(exact["left_arm"])
        np.testing.assert_allclose(back[:3], poses["left"][:3], atol=1e-9)
        assert abs(abs(float(np.dot(back[3:], quaternion))) - 1.0) < 5e-10
        assert abs(exact["right_gripper"]["width"] - 2.0) < 5e-8


class Robot:
    """An ARX X5 arm of RoboDojo's robot list, as the controller reads it."""

    type, gripper_scale = "target", [-0.01, 0.044]

    def __init__(self, arm: str) -> None:
        self.arm, self.arm_name, self.gripper_name = arm, f"{arm}_arm", f"{arm}_ee"


class ArmsScene:
    """The official env of a scene as the controller drives it: its arms reach every commanded joint target within a
    control step (their wrist links no lower than ``floor``), joints that stand for a wrist-link pose; its IK solves a pose
    into the joints that ``solutions`` gives for the pose's own, failing for the arms of ``failing``'s next entry (one per
    control step), and its episode ends at control step ``ending_step``. ``solved`` lists the arm of every IK solve."""

    def __init__(self) -> None:
        self.robot_list = [Robot(arm) for arm in MotionController.ARMS]
        self.robot_manager = self
        self.links = {arm: list(START[arm]) for arm in MotionController.ARMS}  # the wrist links' poses
        self.fingers = {arm: [0.044, 0.044] for arm in MotionController.ARMS}
        self.floor = -np.inf
        self.failing: list[set[str]] = []
        self.solutions = lambda joints: [joints]  # a pose's joints -> the joints its IK solve returns
        self.ending_step: int | None = None
        self.actions: list[dict] = []  # every official action taken
        self.solved: list[str] = []
        self.end_flag, self.unstable_envs = [False], set()

    def ik_solutions(self, link_pose, robot):
        self.solved.append(robot.arm)
        if self.failing and robot.arm in self.failing[0]:
            return []
        return self.solutions(self.joints_of(link_pose))

    def take_action(self, action):
        self.actions.append(action)
        if self.failing:
            self.failing.pop(0)
        for robot in self.robot_list:
            if self.process_name(robot.arm_name) in action:
                link = self.pose_of(action[self.process_name(robot.arm_name)])
                self.links[robot.arm] = [link[0], link[1], max(link[2], self.floor), *link[3:]]
            low, high = robot.gripper_scale
            opening = low + action[self.process_name(robot.gripper_name)][0] * (high - low)
            self.fingers[robot.arm] = [opening, opening]

    def is_episode_end(self):
        if len(self.actions) == self.ending_step:
            self.end_flag[0] = True
        return self.end_flag[0]

    def get_joint(self, robot, env_idx_list):
        return {0: self.joints_of(self.links[robot.arm])}

    def get_real_endpose(self, robot, env_idx_list):
        return {0: self.links[robot.arm]}

    def get_end_effector_real_val(self, robot, env_idx_list):
        return {0: self.fingers[robot.arm]}

    @staticmethod
    def process_name(name):
        return name if name.endswith("_state") else name + "_joint_state"

    @staticmethod
    def joints_of(link_pose):
        quaternion = MotionController.quat_normalize(link_pose[3:])  # (qw >= 0: its x, y, z name it)
        return np.asarray([*link_pose[:3], *quaternion[1:]], dtype=np.float32)  # in cuRobo's dtype

    @staticmethod
    def pose_of(joints):
        x, y, z = (float(v) for v in joints[3:])
        return [*(float(v) for v in joints[:3]), max(0.0, 1.0 - x * x - y * y - z * z) ** 0.5, x, y, z]


@pytest.fixture
def scene() -> ArmsScene:
    return ArmsScene()


@pytest.fixture
def controller(scene, monkeypatch) -> MotionController:
    # the official IK's preparation of a target is the App's (the GPU replays); here the scene's IK gives the solutions
    monkeypatch.setattr(
        MotionController, "_ik_solutions", lambda self, robot, link_pose: self.env.ik_solutions(link_pose, robot)
    )
    return MotionController(scene, {"state": {f"{arm}_ee_joint_state": [1.0] for arm in MotionController.ARMS}})


def test_an_arm_moves_along_waypoints_at_a_constant_speed(scene, controller):
    before = controller.state()

    assert controller.move(LIFT) == "reached"

    heights = [ArmsScene.pose_of(action["left_arm_joint_state"])[2] for action in scene.actions]
    np.testing.assert_allclose(np.diff([START["left"][2], *heights]), [0.05 / 3] * 3, atol=1e-6)  # 5 cm, 3 waypoints
    assert all(
        set(action) == {"left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state"}
        for action in scene.actions
    )
    assert all(action["left_arm_joint_state"].dtype == np.float32 for action in scene.actions)
    after = controller.state()
    assert after["left_arm"]["z"] == pytest.approx(before["left_arm"]["z"] + 5.0)
    assert after["right_arm"] == before["right_arm"]
    assert scene.solved == ["left"] * 3  # the arm the action does not move holds its joint targets, unsolved


def test_a_step_commands_at_most_the_joint_change_the_drives_follow(scene, controller, monkeypatch):
    monkeypatch.setattr(MotionController, "JOINT_STEP_RAD", 0.005)  # the fake's joints move 0.02 per 2 cm step

    assert controller.move(LIFT) == "reached"

    joints = [ArmsScene.joints_of(START["left"])] + [action["left_arm_joint_state"] for action in scene.actions]
    assert max(float(np.max(np.abs(b - a))) for a, b in zip(joints, joints[1:])) <= 0.005 + 1e-6
    assert len(scene.actions) >= 10  # 5 cm at 0.5 cm per control step at most


def test_a_step_passes_over_solutions_in_other_arm_configurations(scene, controller):
    # the scene's joint 2 stands for the elbow, 0.3 short of straight: the official IK's best solutions bend it the other
    # way (0.6 away) and turn joint 3 by 1.5; the arm's own comes last, and it is commanded as solved
    straight = MotionController.ELBOW_RANGE_RAD[1]
    scene.links["left"][2] = straight - 0.3
    controller.joint_targets["left"] = ArmsScene.joints_of(scene.links["left"])
    own = []

    def solutions(joints):
        bent_over = joints.copy()
        bent_over[2] = 2 * straight - joints[2]
        own.append(joints)
        return [bent_over, joints + np.float32([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]), joints]

    scene.solutions = solutions

    assert controller.move(LIFT) == "reached"

    assert [action["left_arm_joint_state"].tobytes() for action in scene.actions] == [
        joints.tobytes() for joints in own
    ]


def test_of_solutions_in_the_arm_configuration_the_official_best_is_taken(scene, controller):
    # two seeds' solutions of one configuration, the official IK's best 1 mm farther along the lift
    scene.solutions = lambda joints: [joints + np.float32([0.0, 0.0, 0.001, 0.0, 0.0, 0.0]), joints]

    assert controller.move(LIFT) == "reached"

    heights = [action["left_arm_joint_state"][2] for action in scene.actions]
    np.testing.assert_allclose(heights, START["left"][2] + np.arange(1, 4) * 0.05 / 3 + 0.001, atol=1e-6)


def test_a_solution_whole_turns_off_the_arm_joints_is_taken_back(scene, controller):
    # the official IK's seeds span placeholder joint limits: a revolute joint's solution may come whole turns off
    turns = np.float32([0.0, 0.0, 0.0, 2 * np.pi, -4 * np.pi, 0.0])
    scene.solutions = lambda joints: [joints + turns]
    start = ArmsScene.joints_of(START["left"])

    assert controller.move(LIFT) == "reached"

    commanded = np.array([action["left_arm_joint_state"] for action in scene.actions])
    np.testing.assert_allclose(commanded[:, 3:], np.tile(start[3:], (len(commanded), 1)), atol=1e-5)
    np.testing.assert_allclose(np.diff([START["left"][2], *commanded[:, 2]]), [0.05 / 3] * 3, atol=1e-6)


def test_an_arm_stops_where_its_elbow_is_straight(scene, controller):
    # the scene's joint 2 stands for the elbow: the arm starts 2 cm short of straight and the lift takes it 5 cm further,
    # where every solution has the elbow bent the other way
    straight = MotionController.ELBOW_RANGE_RAD[1]
    scene.links["left"][2] = straight - 0.02
    controller.joint_targets["left"] = ArmsScene.joints_of(scene.links["left"])

    assert controller.move(LIFT) == "stopped"

    elbows = [float(action["left_arm_joint_state"][MotionController.ELBOW_INDEX]) for action in scene.actions]
    assert max(elbows) <= straight and scene.links["left"][2] == pytest.approx(straight - 0.02 + 0.05 / 3)


def test_an_arm_that_is_blocked_stops_where_it_is(scene, controller):
    scene.floor = START["left"][2] - 0.02  # 2 cm below, the arm meets a surface

    assert controller.move({**ZERO, "left_arm": {**ZERO["left_arm"], "z": -5.0}}) == "stopped"

    assert scene.links["left"][2] == pytest.approx(scene.floor)
    assert (
        controller.joint_targets["left"].tobytes() == ArmsScene.joints_of(scene.links["left"]).tobytes()
    )  # held there


def test_an_arm_whose_ik_keeps_failing_stops_where_it_is(scene, controller):
    scene.failing = [{"left"}] * (MotionController.STALL_STEP_N + 1)

    assert controller.move(LIFT) == "stopped"

    assert len(scene.actions) == MotionController.STALL_STEP_N + 1  # its first step, then no progress
    start = ArmsScene.joints_of(START["left"]).tobytes()  # commanded to the joints the reset left: no target before
    assert all(action["left_arm_joint_state"].tobytes() == start for action in scene.actions)
    assert controller.joint_targets["left"].tobytes() == start


def test_every_step_commands_both_arms_whatever_their_solves(scene, controller):
    scene.failing = [
        set(MotionController.ARMS)
    ]  # a scene's first step, whose official command of a missing arm has no previous one

    controller.move(LIFT)

    for arm in MotionController.ARMS:
        assert scene.actions[0][f"{arm}_arm_joint_state"].tobytes() == ArmsScene.joints_of(START[arm]).tobytes()


def test_a_gripper_closes_all_the_way_on_a_negative_width_change(scene, controller):
    assert controller.move({**ZERO, "right_gripper": {"width": -1.0}}) == "reached"

    openings = [(action["left_ee_joint_state"], action["right_ee_joint_state"]) for action in scene.actions]
    assert openings == [([1.0], [0.0])]
    assert controller.state()["right_gripper"] == {"width": 0.0} and controller.state()["left_gripper"] == {
        "width": 8.8
    }


def test_an_episode_that_ends_during_an_action_ends_the_action(scene, controller):
    scene.ending_step = 1

    assert controller.move(LIFT) == "done"

    assert len(scene.actions) == 1
