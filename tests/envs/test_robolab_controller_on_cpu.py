from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pear.envs.robolab.controller import MotionController

DOWN = MotionController.matrix_to_quat(np.diag([1.0, -1.0, -1.0])).tolist()  # the gripper pointing down at yaw 0
# base_link's pose that puts the tool point 50 cm in front of the robot's base and 20 cm up, pointing down
START = MotionController.link_pose_of_tool([0.5, 0.0, 0.2, *DOWN])
OPEN_GAP_M = 0.085  # the fake gripper's finger pads' gap when open (closed: 0)
ZERO = {
    part: {quantity: 0.0 for quantity in quantities} for part, quantities in MotionController.SCHEMA.items()
}  # an action that holds
LIFT = {**ZERO, "arm": {**ZERO["arm"], "z": 5.0}}  # the tool point 5 cm up


class Env:
    """RoboLab's RobolabEnv as the controller drives it: base_link reaches every commanded target within a control step,
    its tool point no lower than ``floor``, pressed on which the gripper turns about the robot's y by ``press_tilt`` rad
    per metre of the target below it (or, ``wedged``, keeps its turn whatever the target), and turned ``sag`` rad off its
    target about the robot's y always (a load the arm holds); the gripper's closing
    (gripper_pos) moves GRIP_RATE per control step toward its binary command, carries the pads down by PAD_ADVANCE_M at
    most (pointing down) and stops at ``grip_stop`` (an object) or where the pads meet the floor; the episode ends at
    control step ``ending_step``."""

    GRIP_RATE = 0.4
    PRESS_TOL_M = 1e-6  # a target this little below the floor is on it (RoboLab's action is float32)
    device = "cpu"

    def __init__(self) -> None:
        self.link = list(START)
        self.grip = 0.0
        self.floor = -math.inf
        self.press_tilt = 0.0
        self.wedged = False
        self.sag = 0.0
        self.grip_stop: float | None = None
        self.ending_step: int | None = None
        self.diverging_step: int | None = None  # the control step whose readings come back non-finite
        self.ik_failing_step: int | None = None  # the control step whose IK solve fails (a state gone non-finite in it)
        self.all_terminated = False
        self.actions: list[list[float]] = []  # every native action taken
        self.body_names = ["base_link", *MotionController.PAD_BODIES]
        self.scene = {"robot": SimpleNamespace(data=self)}
        self.joint_names = [f"panda_joint{number}" for number in range(1, 8)]
        self.elbow = -2.0  # panda_joint4's position (bent)
        self.ik_params = {"lambda_val": 0.01}  # RoboLab's absolute-IK action term's, as Isaac Lab sets it
        self.ik_dampings: list[float] = []  # the IK's damping at every native action
        term = SimpleNamespace(cfg=SimpleNamespace(controller=SimpleNamespace(ik_params=self.ik_params)))
        self.unwrapped = SimpleNamespace(action_manager=SimpleNamespace(get_term={"arm_action": term}.__getitem__))

    @property
    def joint_pos(self) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 0.0, self.elbow, 0.0, 0.0, 0.0]], dtype=torch.float64)

    @property
    def body_pos_w(self) -> torch.Tensor:
        # the pads' bodies unturned, each placed so that its gripping face lies half the gap off the middle
        half_gap = (1.0 - self.grip) * OPEN_GAP_M / 2
        faces = [[0.0, half_gap, 0.0], [0.0, -half_gap, 0.0]]
        bodies = [np.subtract(face, offset).tolist() for face, offset in zip(faces, MotionController.PAD_FACES_M)]
        return torch.tensor([[[0.0, 0.0, 0.0], *bodies]], dtype=torch.float64)

    @property
    def body_quat_w(self) -> torch.Tensor:
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=torch.float64)

    def step(self, action: torch.Tensor):
        self.actions.append(action[0].tolist())
        self.ik_dampings.append(self.ik_params["lambda_val"])
        if len(self.actions) == self.ik_failing_step:  # as Isaac Lab's damped least squares raises on such a state
            raise torch.linalg.LinAlgError("linalg.inv: The diagonal element 1 is zero, the input matrix is singular.")
        tool = MotionController.tool_pose_of_link([float(v) for v in action[0, :7]])
        pressed_m = self.floor - tool[2]
        if pressed_m > self.PRESS_TOL_M:
            tool[2] = self.floor
            turn = MotionController.rpy_to_matrix([0.0, self.press_tilt * pressed_m, 0.0])
            tool[3:] = MotionController.matrix_to_quat(turn @ MotionController.quat_to_matrix(tool[3:])).tolist()
        else:
            tool[2] = max(tool[2], self.floor)
            if self.wedged:  # off the surface the gripper keeps the turn it has
                tool[3:] = MotionController.tool_pose_of_link(self.link)[3:]
        if self.sag:
            turn = MotionController.rpy_to_matrix([0.0, self.sag, 0.0])
            tool[3:] = MotionController.matrix_to_quat(turn @ MotionController.quat_to_matrix(tool[3:])).tolist()
        self.link = MotionController.link_pose_of_tool(tool)
        command = 1.0 if float(action[0, 7]) > 0.5 else 0.0  # RoboLab's binary gripper action
        self.grip += max(-self.GRIP_RATE, min(self.GRIP_RATE, command - self.grip))
        if self.grip_stop is not None and command == 1.0:
            self.grip = min(self.grip, self.grip_stop)
        self.grip = min(self.grip, (tool[2] - self.floor) / MotionController.PAD_ADVANCE_M)  # the pads on the floor
        if len(self.actions) == self.ending_step:
            self.all_terminated = True
        if len(self.actions) == self.diverging_step:
            self.link = [math.nan] * 7
        return self.observation(), None, None, None, None

    def observation(self) -> dict:
        return {
            "proprio_obs": {
                "ee_pos": torch.tensor([self.link[:3]], dtype=torch.float64),
                "ee_quat": torch.tensor([self.link[3:]], dtype=torch.float64),
                "gripper_pos": torch.tensor([[self.grip]], dtype=torch.float64),
            }
        }


@pytest.fixture
def env() -> Env:
    return Env()


@pytest.fixture
def controller(env) -> MotionController:
    return MotionController(env, env.observation(), 0.0)


def test_the_tool_point_and_base_link_poses_invert_each_other():
    rng = np.random.default_rng(0)
    for _ in range(20):
        link = [*(rng.normal(size=3) * 0.3), *MotionController.quat_normalize(rng.normal(size=4))]
        back = MotionController.link_pose_of_tool(MotionController.tool_pose_of_link(link))
        np.testing.assert_allclose(back[:3], link[:3], atol=1e-12)
        assert MotionController.quat_angle(back[3:], link[3:]) == pytest.approx(0.0, abs=1e-6)


def test_pointing_down_at_yaw_0_reads_roll_pi(controller):
    state = controller.state()

    assert state["arm"]["x"] == 50.0 and state["arm"]["y"] == 0.0 and state["arm"]["z"] == 20.0
    assert abs(state["arm"]["roll"]) == 3.14 and state["arm"]["pitch"] == 0.0 and state["arm"]["yaw"] == 0.0
    assert state["gripper"] == {"width": 8.5}


def test_the_height_reads_from_the_table_top(env):
    controller = MotionController(env, env.observation(), 0.047)  # a table top 4.7 cm above the robot's root

    assert controller.state()["arm"]["z"] == 15.3

    controller.move(LIFT)
    assert controller.state()["arm"]["z"] == 20.3  # a change moves the tool point by as much


def test_a_change_moves_the_tool_point_along_the_robot_axes(controller):
    arm = controller._measured_quantities()["arm"]
    for axis, offset in (("x", [0.05, 0, 0]), ("y", [0, 0.05, 0]), ("z", [0, 0, 0.05])):
        moved = MotionController.tool_pose_of_quantities({**arm, axis: arm[axis] + 5})
        np.testing.assert_allclose(np.subtract(moved[:3], [0.5, 0.0, 0.2]), offset, atol=1e-9)


def test_quantities_and_tool_poses_invert_each_other():
    rng = np.random.default_rng(1)
    for _ in range(20):
        tool = [*(rng.normal(size=3) * 0.3), *MotionController.quat_normalize(rng.normal(size=4))]
        arm = MotionController.quantities(MotionController.link_pose_of_tool(tool), 0.0)["arm"]
        back = MotionController.tool_pose_of_quantities(arm)
        np.testing.assert_allclose(back[:3], tool[:3], atol=1e-9)
        assert MotionController.quat_angle(back[3:], tool[3:]) == pytest.approx(0.0, abs=1e-6)


def test_an_arm_moves_along_waypoints_at_a_constant_speed(env, controller):
    assert controller.move(LIFT) == "reached"

    # 5 cm in 3 waypoints of 2 cm at most, commanded in float32 (RoboLab's action)
    heights = [MotionController.tool_pose_of_link(action[:7])[2] for action in env.actions]
    np.testing.assert_allclose(np.diff([0.2, *heights]), [0.05 / 3] * 3, atol=1e-6)
    assert all(action[7] == 0.0 for action in env.actions)  # the gripper stays open
    assert controller.state()["arm"]["z"] == 25.0


def test_an_arm_that_is_blocked_stops_where_it_is(env, controller):
    env.floor = 0.18  # 2 cm below, the tool point meets a surface

    assert controller.move({**ZERO, "arm": {**ZERO["arm"], "z": -5.0}}) == "stopped"

    assert controller.state()["arm"]["z"] == 18.0
    assert controller.link_target == pytest.approx(env.link)  # held where it is, not pressing on
    # the first waypoint (18.3 cm) above the surface, the second onto it, then no progress
    assert len(env.actions) == 2 + MotionController.STALL_STEP_N


def test_an_arm_that_pressing_on_a_surface_turned_turns_back_where_it_stopped(env, controller):
    env.floor, env.press_tilt = 0.18, 2.0  # a target 2 cm below the surface turns the gripper by 0.04 rad

    assert controller.move({**ZERO, "arm": {**ZERO["arm"], "z": -5.0}}) == "stopped"

    arm = controller.state()["arm"]
    assert arm["z"] == 18.0 and abs(arm["roll"]) == 3.14 and arm["pitch"] == 0.0  # pointing down again, on the surface
    assert controller.link_target == pytest.approx(env.link)  # held where it is, not pressing on


def test_an_arm_that_cannot_turn_back_stops_where_it_is(env, controller):
    env.floor, env.press_tilt = 0.18, 2.0
    env.wedged = True  # the turn the press gave it stays

    assert controller.move({**ZERO, "arm": {**ZERO["arm"], "z": -5.0}}) == "stopped"

    assert controller.link_target == pytest.approx(env.link)
    assert len(env.actions) < MotionController.MAX_ACTION_STEP_N  # the turn back stalls and ends the action


def test_a_load_s_turn_does_not_add_up_over_the_actions(env, controller):
    env.sag = 0.05  # a load the arm holds turns the gripper 0.05 rad off whatever it is asked
    move = {**ZERO, "arm": {**ZERO["arm"], "x": 2.0}}

    for _ in range(5):
        controller.move(move)

    tool = MotionController.tool_pose_of_link(env.link)
    assert MotionController.quat_angle(tool[3:], DOWN) == pytest.approx(0.05, abs=1e-3)  # one load's turn, not five


def test_a_turn_the_action_asks_for_is_kept_by_the_actions_after_it(env, controller):
    assert controller.move({**ZERO, "arm": {**ZERO["arm"], "yaw": 0.5}}) == "reached"
    env.sag = 0.05
    controller.move(LIFT)
    env.sag = 0.0

    assert controller.move(LIFT) == "reached"

    assert controller.state()["arm"]["yaw"] == 0.5 and controller.state()["arm"]["pitch"] == 0.0  # the turn asked for


def test_a_close_on_a_surface_draws_the_arm_back_for_the_pads(env, controller):
    env.floor = 0.18
    assert controller.move({**ZERO, "arm": {**ZERO["arm"], "z": -5.0}}) == "stopped"

    assert controller.move({**ZERO, "gripper": {"width": -1.0}}) == "reached"

    assert controller.state()["gripper"] == {"width": 0.0}  # closed all the way, not held open by the surface
    assert MotionController.tool_pose_of_link(env.link)[2] == pytest.approx(0.18 + MotionController.PAD_ADVANCE_M)


def test_an_action_that_changes_nothing_takes_one_control_step(env, controller):
    assert controller.move(ZERO) == "reached"

    assert len(env.actions) == 1 and env.actions[0][:7] == pytest.approx(START)


def test_a_gripper_closes_all_the_way_on_a_negative_width_change(env, controller):
    assert controller.move({**ZERO, "gripper": {"width": -1.0}}) == "reached"

    assert [action[7] for action in env.actions] == [1.0] * 3  # 0.4 per control step to 1
    # the arm draws back by the pads' advance: up, the gripper pointing down
    drawn_back = MotionController.link_pose_of_tool([0.5, 0.0, 0.2 + MotionController.PAD_ADVANCE_M, *DOWN])
    assert all(action[:7] == pytest.approx(drawn_back) for action in env.actions)
    assert controller.state()["gripper"] == {"width": 0.0}

    assert controller.move({**ZERO, "gripper": {"width": 0.5}}) == "reached"
    assert env.actions[-1][7] == 0.0 and controller.state()["gripper"] == {"width": 8.5}


def test_a_gripper_closing_on_an_object_stops_there(env, controller):
    env.grip_stop = 0.6

    assert controller.move({**ZERO, "gripper": {"width": -1.0}}) == "reached"

    assert controller.state()["gripper"]["width"] == pytest.approx(3.4)  # 8.5 cm x 0.4 still open
    assert len(env.actions) == 2 + MotionController.GRIP_STILL_STEP_N


def test_the_ik_is_damped_more_while_the_elbow_is_near_straight(env, controller):
    controller.move(LIFT)
    assert set(env.ik_dampings) == {MotionController.IK_DAMPING}

    env.elbow, env.ik_dampings = -0.5, []
    controller.move(LIFT)
    assert set(env.ik_dampings) == {MotionController.IK_DAMPING_STRAIGHT}


def test_an_episode_that_ends_during_an_action_ends_the_action(env, controller):
    env.ending_step = 1

    assert controller.move(LIFT) == "done"

    assert len(env.actions) == 1


def test_an_action_ends_when_the_ik_fails_on_a_diverged_state(env, controller):
    env.ik_failing_step = 2

    assert controller.move(LIFT) == "diverged"
    assert len(env.actions) == 2


def test_an_action_ends_once_the_readings_are_no_longer_finite(env, controller):
    env.diverging_step = 2

    assert controller.move(LIFT) == "diverged"
    assert len(env.actions) == 2
