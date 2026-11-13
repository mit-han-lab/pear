from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# numpy only: the verl side imports this module through env.py (its env class's SCHEMA); torch and RoboLab are imported
# inside the methods, in the App


class MotionController:
    """Moves RoboLab's DROID arm (a Franka with a Robotiq 2F-85) through one episode as the model's actions say, in
    RoboLab's control steps of its absolute-IK action: the tool point moves along waypoints toward its goal at a
    constant speed until it reaches it or stops short, the gripper opens or closes all the way."""

    # what an action sets: the tool point's pose in the base frame, z from the table, and the finger gap (cm, radians)
    SCHEMA: dict[str, tuple[str, ...]] = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
    # the decimals of each value the model reads
    DECIMAL_N_BY_QUANTITY = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}
    # the gripper frame's axes in base_link's (columns): z where the gripper points, y across the fingers
    GRIPPER_FRAME = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    # the tool point, the pads' centre with the gripper open, from base_link (the Robotiq 2F-85's flange)
    TOOL_OFFSET_M = (0.131, 0.0, 0.0)
    PAD_BODIES = ("left_inner_finger", "right_inner_finger")  # the finger pads' bodies
    # the centres of their gripping faces, in those bodies' frames
    PAD_FACES_M = ((0.131, 0.0417, 0.0), (0.1312, -0.0414, 0.0))
    ARM_ACTION = "arm_action"  # RoboLab's absolute-IK action term of the DROID arm
    ELBOW_JOINT = "panda_joint4"  # the Franka's elbow, straight at its upper limit
    STEP_M = 0.02  # the tool point's waypoints advance at most this far per control step (15 Hz)
    STEP_RAD = math.radians(10)  # the tool point's waypoints turn at most this much per control step
    # the next waypoint is taken once the tool point is within this of the current one: the lag it allows sets the
    # arm's speed and how far a blocked arm is pressed before it stops
    LEAD_M = 2 * STEP_M
    LEAD_RAD = STEP_RAD  # the next waypoint is taken once the tool point's turn is within this of the current one's
    # a goal's position is reached within this: the arm settles on its target exponentially, so a tighter one costs
    # whole steps every action (the time limit counts them)
    POS_TOL_M = 0.005
    ROT_TOL_RAD = 0.02  # a goal's orientation is reached within this
    PROGRESS_M = 0.001  # the least the arm's distance to its goal shrinks in a control step that counts as progress
    PROGRESS_RAD = 0.005  # the least the arm's angle to its goal shrinks in a control step that counts as progress
    # an arm without progress for this many control steps is blocked (a contact) or out of reach: it stops where it is
    # (each of these steps it presses on what blocks it, and spends the task's time)
    STALL_STEP_N = 4
    # the Robotiq's linkage carries its pads this far forward along the gripper's axis as they close from open (the
    # pads' bodies in the simulator); the simulated arm is too stiff to yield to a surface they meet, so a close draws
    # the arm back by it
    PAD_ADVANCE_M = 0.0136
    # the gripper is at its command (RoboLab's gripper_pos, finger_joint / (pi / 4): 0 open, 1 closed) within this
    GRIP_TOL = 0.02
    GRIP_STILL = 0.002  # a gripper_pos that changed less than this in a control step is still
    GRIP_STILL_STEP_N = 3  # a gripper still for this many control steps has stopped (on an object)
    IK_DAMPING = 0.01  # the damping of RoboLab's absolute-IK action's damped least squares, Isaac Lab's
    # the damping with the elbow near straight (the arm at the edge of its reach), where a step under IK_DAMPING swings
    # it to its limit
    IK_DAMPING_STRAIGHT = 0.05
    # panda_joint4 above this is near straight (its limit -0.07)
    ELBOW_STRAIGHT_RAD = -1.0
    MAX_ACTION_STEP_N = 100  # the control steps of one action at most
    STEP_ROUNDING = 1e-9  # a distance of a whole number of steps up to float rounding takes that many
    SLERP_LINEAR_DOT = 0.9995  # orientations this close (the cosine of half their angle) interpolate linearly
    GIMBAL_LOCK_TOL = 1e-9  # sin(pitch) this close to +-1: roll and yaw turn about one axis

    def __init__(self, env: Any, observation: dict[str, Any], table_height_m: float) -> None:
        self.env = env  # the episode's RobolabEnv, reset
        self.observation = observation  # its last observation
        self.table_height_m = table_height_m  # the table top above the robot's root: the readings' z counts from it
        robot = env.scene["robot"]
        self.robot = robot
        self.pad_indices = [list(robot.data.body_names).index(name) for name in self.PAD_BODIES]
        self.elbow_index = list(robot.data.joint_names).index(self.ELBOW_JOINT)
        # the IK controller holds its action term's controller config and reads the damping at every solve
        self.ik_params = env.unwrapped.action_manager.get_term(self.ARM_ACTION).cfg.controller.ik_params

        # the commands, at the start where the reset left the robot: the gripper's (RoboLab's DROID starts open), the
        # arm's (base_link's pose in the robot-root frame) and the orientation the actions ask of the tool point
        self.grip_command = 0.0
        self.link_target = self._measured_link_pose()
        self.turn_target = self._measured_tool_pose()[3:]

    def move(self, action: dict[str, dict[str, float]]) -> str:
        """Carries out the model's ``action`` from the robot as it is now and returns "reached", "stopped" (the arm or
        the gripper stopped short), "done" (the episode ended in it) or "diverged" (the readings stopped being
        finite in it)."""
        quantities = self._measured_quantities()
        is_moved = any(float(v) != 0.0 for v in action["arm"].values())
        is_turned = any(float(action["arm"][quantity]) != 0.0 for quantity in ("roll", "pitch", "yaw"))
        goal = (
            self.tool_pose_of_quantities(
                {
                    quantity: quantities["arm"][quantity] + float(action["arm"][quantity])
                    for quantity in self.SCHEMA["arm"]
                }
            )
            if is_moved
            else None
        )

        # a turn the action asks for counts from the orientation the model read; otherwise the tool point keeps the one
        # asked before: a press or a load turns the arm a little, and a goal from the orientation it reads would keep
        # that turn and add the next one
        if is_turned:
            self.turn_target = goal[3:]
        elif goal is not None:
            goal = [*goal[:3], *self.turn_target]

        width_change = float(action["gripper"]["width"])
        is_grip_set = width_change != 0.0
        is_closing = width_change < 0.0 and self.grip_command == 0.0  # an open gripper starts to close
        if is_grip_set:  # all the way the width's change points: open for a positive one, closed for a negative one
            self.grip_command = 0.0 if width_change > 0.0 else 1.0
        if is_closing:  # the arm draws back along the gripper's axis by the pads' advance, from its goal or where it is
            goal = goal if goal is not None else [*self._measured_tool_pose()[:3], *self.turn_target]
            axis = self.quat_to_matrix(goal[3:])[:, 2]  # where the gripper points
            goal = (np.asarray(goal[:3]) - self.PAD_ADVANCE_M * axis).tolist() + list(goal[3:])
        if goal is None and not is_grip_set:  # the arm holds where it is (the action still takes its control step)
            goal = [*self._measured_tool_pose()[:3], *self.turn_target]

        # the arm's waypoints from where it is (the first) to its goal: the next one each control step while the tool
        # point keeps within the lead (LEAD_M, LEAD_RAD) of the current one, so a blocked arm is never pressed far
        start = waypoint = self._measured_tool_pose()
        waypoint_number, waypoint_n = 0, 1
        if goal is not None:
            distance_m, angle_rad = self._distance(start, goal)
            waypoint_n = max(
                1, math.ceil(max(distance_m / self.STEP_M, angle_rad / self.STEP_RAD) - self.STEP_ROUNDING)
            )
        least_distance, stall_step_n = (math.inf, math.inf), 0
        is_arm_done, is_stopped, is_turning_back = goal is None, False, False
        grip, still_step_n = self._grip(), 0

        for _ in range(self.MAX_ACTION_STEP_N):  # (every action takes at least one control step)
            if not is_arm_done:
                distance_m, angle_rad = self._distance(self._measured_tool_pose(), waypoint)
                if waypoint_number < waypoint_n and distance_m <= self.LEAD_M and angle_rad <= self.LEAD_RAD:
                    waypoint_number += 1
                    waypoint = self.interpolate(start, goal, waypoint_number / waypoint_n)
                self.link_target = self.link_pose_of_tool(waypoint)

            if not self._step():
                return "diverged"
            if self.env.all_terminated:
                return "done"
            if not self._is_finite():
                return "diverged"

            if not is_arm_done:
                distance_m, angle_rad = self._distance(self._measured_tool_pose(), goal)
                if distance_m <= self.POS_TOL_M and angle_rad <= self.ROT_TOL_RAD:
                    is_arm_done = True
                elif (
                    distance_m < least_distance[0] - self.PROGRESS_M
                    or angle_rad < least_distance[1] - self.PROGRESS_RAD
                ):
                    stall_step_n = 0
                else:
                    stall_step_n += 1
                    if stall_step_n >= self.STALL_STEP_N:
                        is_stopped = True
                        tool_pose = self._measured_tool_pose()
                        turned_back_pose = [*tool_pose[:3], *goal[3:]]  # where it stopped, turned as the action asked
                        if is_turning_back or self._distance(tool_pose, turned_back_pose)[1] <= self.ROT_TOL_RAD:
                            is_arm_done = True
                            self.link_target = self._measured_link_pose()  # stop where it is, not pressing on
                        else:
                            # pressing on a surface turned the gripper (the IK gives up its turn for the position): it
                            # turns back where it stopped, or the next actions, which start from it, would keep the tilt
                            is_turning_back = True
                            start = waypoint = tool_pose
                            goal = turned_back_pose
                            distance_m, angle_rad = self._distance(start, goal)
                            waypoint_number, waypoint_n = (
                                0,
                                max(1, math.ceil(angle_rad / self.STEP_RAD - self.STEP_ROUNDING)),
                            )
                            least_distance, stall_step_n = (math.inf, math.inf), 0
                least_distance = (min(least_distance[0], distance_m), min(least_distance[1], angle_rad))

            # the fingers' stop is judged with the arm at rest: a surface the pads meet holds them until it draws back
            measured_grip = self._grip()
            still_step_n = still_step_n + 1 if is_arm_done and abs(measured_grip - grip) < self.GRIP_STILL else 0
            grip = measured_grip
            is_grip_done = (
                not is_grip_set
                or abs(grip - self.grip_command) <= self.GRIP_TOL
                or still_step_n >= self.GRIP_STILL_STEP_N
            )
            if is_arm_done and is_grip_done:
                break
        else:  # out of steps short of the goal: stop there, not pressing on
            if not is_arm_done:
                is_stopped = True
                self.link_target = self._measured_link_pose()

        return "reached" if not is_stopped and is_grip_done else "stopped"

    def state(self) -> dict[str, dict[str, float]]:
        """The quantities as the model reads them, by part: the height counted from the table top, every value rounded
        to DECIMAL_N_BY_QUANTITY, a value that rounds to zero as 0.0 (never -0.0)."""
        quantities = self._measured_quantities()
        quantities["arm"]["z"] -= 100.0 * self.table_height_m

        return {
            part: {quantity: round(v, self.DECIMAL_N_BY_QUANTITY[quantity]) + 0.0 for quantity, v in values.items()}
            for part, values in quantities.items()
        }

    def _step(self) -> bool:
        """One control step of RoboLab's DROID absolute-IK action (base_link's target pose and the gripper's command):
        False when the physics diverged within it."""
        import torch

        is_straight = float(self.robot.data.joint_pos[0, self.elbow_index]) > self.ELBOW_STRAIGHT_RAD
        self.ik_params["lambda_val"] = self.IK_DAMPING_STRAIGHT if is_straight else self.IK_DAMPING
        native_action = torch.tensor(
            [[*self.link_target, self.grip_command]], dtype=torch.float32, device=self.env.device
        )

        try:
            self.observation, _, _, _, _ = self.env.step(native_action)
        except torch.linalg.LinAlgError:
            # Isaac Lab's damped least squares inverts J J^T + lambda^2 I, invertible for every finite Jacobian: a
            # failed inverse is the articulation's state gone non-finite within the step's physics substeps, the
            # divergence _is_finite reads after a step
            return False

        return True

    def _is_finite(self) -> bool:
        """Whether every reading the controller and the model use is finite."""
        import torch

        # the Robotiq's passive finger joints can blow up in RoboLab's physics, after which no reading can be trusted
        proprio = self.observation["proprio_obs"]
        pads = self.robot.data.body_pos_w[0, self.pad_indices]

        return all(
            bool(torch.isfinite(value).all())
            for value in (proprio["ee_pos"], proprio["ee_quat"], proprio["gripper_pos"], pads)
        )

    def _measured_quantities(self) -> dict[str, dict[str, float]]:
        """The unrounded quantities of the robot as it is now."""
        return self.quantities(self._measured_link_pose(), self._pad_gap())

    def _measured_link_pose(self) -> list[float]:
        """base_link's measured pose in the robot-root frame [x, y, z, qw, qx, qy, qz] (RoboLab's ee_pos, ee_quat)."""
        proprio = self.observation["proprio_obs"]

        return [float(v) for v in proprio["ee_pos"][0]] + self.quat_normalize(proprio["ee_quat"][0].tolist()).tolist()

    def _measured_tool_pose(self) -> list[float]:
        """The measured tool-point pose."""
        return self.tool_pose_of_link(self._measured_link_pose())

    def _grip(self) -> float:
        """The gripper's measured closing (RoboLab's gripper_pos: 0 open, 1 closed)."""
        return float(self.observation["proprio_obs"]["gripper_pos"][0, 0])

    def _pad_gap(self) -> float:
        """The measured distance between the finger pads' gripping faces (m)."""
        data = self.robot.data
        positions = data.body_pos_w[0, self.pad_indices].double().cpu().numpy()
        quaternions = data.body_quat_w[0, self.pad_indices].double().cpu().numpy()
        faces = [
            position + self.quat_to_matrix(quaternion) @ np.asarray(face)
            for position, quaternion, face in zip(positions, quaternions, self.PAD_FACES_M)
        ]

        return float(np.linalg.norm(faces[0] - faces[1]))

    @classmethod
    def quantities(cls, link_pose: Sequence[float], pad_gap: float) -> dict[str, dict[str, float]]:
        """The unrounded quantities of base_link's pose (robot-root frame) and the finger pads' gap (m)."""
        tool_pose = cls.tool_pose_of_link(link_pose)
        frame = cls.quat_to_matrix(tool_pose[3:])
        arm = dict(zip(cls.SCHEMA["arm"], [100.0 * v for v in tool_pose[:3]] + cls.matrix_to_rpy(frame).tolist()))

        return {"arm": {quantity: float(v) for quantity, v in arm.items()}, "gripper": {"width": 100.0 * pad_gap}}

    @classmethod
    def tool_pose_of_quantities(cls, values: dict[str, float]) -> list[float]:
        """The tool-point pose [x, y, z, qw, qx, qy, qz] (robot-root frame, metres) of the arm's quantities."""
        frame = cls.rpy_to_matrix([values["roll"], values["pitch"], values["yaw"]])
        position = np.array([values["x"], values["y"], values["z"]]) / 100.0

        return position.tolist() + cls.quat_normalize(cls.matrix_to_quat(frame)).tolist()

    @classmethod
    def tool_pose_of_link(cls, link_pose: Sequence[float]) -> list[float]:
        """The tool-point pose (the gripper frame's orientation) of a base_link pose."""
        rotation = cls.quat_to_matrix(link_pose[3:])
        position = np.asarray(link_pose[:3], dtype=np.float64) + rotation @ np.asarray(cls.TOOL_OFFSET_M)

        return position.tolist() + cls.quat_normalize(cls.matrix_to_quat(rotation @ cls.GRIPPER_FRAME)).tolist()

    @classmethod
    def link_pose_of_tool(cls, tool_pose: Sequence[float]) -> list[float]:
        """The base_link pose that puts the tool point at ``tool_pose`` (the pose RoboLab's IK solves for)."""
        rotation = cls.quat_to_matrix(tool_pose[3:]) @ cls.GRIPPER_FRAME.T
        position = np.asarray(tool_pose[:3], dtype=np.float64) - rotation @ np.asarray(cls.TOOL_OFFSET_M)

        return position.tolist() + cls.quat_normalize(cls.matrix_to_quat(rotation)).tolist()

    @classmethod
    def quat_angle(cls, a: Sequence[float], b: Sequence[float]) -> float:
        """The rotation angle (radians) between two orientations."""
        dot = abs(float(np.dot(cls.quat_normalize(a), cls.quat_normalize(b))))

        return 2 * math.acos(min(1.0, dot))

    @classmethod
    def interpolate(cls, start: Sequence[float], goal: Sequence[float], t: float) -> list[float]:
        """The pose (position + orientation) a fraction ``t`` of the way from ``start`` to ``goal``."""
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)

        return (start[:3] + t * (goal[:3] - start[:3])).tolist() + cls.slerp(start[3:], goal[3:], t).tolist()

    @classmethod
    def slerp(cls, a: Sequence[float], b: Sequence[float], t: float) -> np.ndarray:
        """The orientation a fraction ``t`` of the way from ``a`` to ``b`` along the shorter arc."""
        a, b = cls.quat_normalize(a), cls.quat_normalize(b)
        dot = float(np.dot(a, b))
        if dot < 0:
            b, dot = -b, -dot

        if dot > cls.SLERP_LINEAR_DOT:
            return cls.quat_normalize(a + t * (b - a))

        theta = math.acos(dot)

        return cls.quat_normalize((math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta))

    @staticmethod
    def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
        """Fixed-axis roll, pitch, yaw about x, y, z (REP 103): Rz(yaw) Ry(pitch) Rx(roll)."""
        cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
        cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
        cy, sy = math.cos(rpy[2]), math.sin(rpy[2])

        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    @classmethod
    def matrix_to_rpy(cls, rot: np.ndarray) -> np.ndarray:
        """The inverse of rpy_to_matrix: at pitch +-90 degrees roll and yaw turn about one axis, and yaw takes it."""
        sp = min(1.0, max(-1.0, -float(rot[2, 0])))
        if abs(sp) < 1.0 - cls.GIMBAL_LOCK_TOL:
            return np.array([math.atan2(rot[2, 1], rot[2, 2]), math.asin(sp), math.atan2(rot[1, 0], rot[0, 0])])

        return np.array([0.0, math.asin(sp), math.atan2(-rot[0, 1], rot[1, 1])])

    @staticmethod
    def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
        """The w-x-y-z quaternion of a rotation matrix (Shepperd)."""
        trace = float(np.trace(rot))
        if trace > 0:
            s = 2.0 * math.sqrt(trace + 1.0)
            return np.array(
                [s / 4, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
            )

        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
        q = np.empty(4)
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (rot[j, i] + rot[i, j]) / s
        q[1 + k] = (rot[k, i] + rot[i, k]) / s

        return q

    @classmethod
    def quat_to_matrix(cls, q: Sequence[float]) -> np.ndarray:
        """The rotation matrix of a w-x-y-z quaternion."""
        w, x, y, z = cls.quat_normalize(q)

        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def quat_normalize(q: Sequence[float]) -> np.ndarray:
        """The unit quaternion of ``q`` with qw >= 0 (one of the two that name its orientation)."""
        q = np.asarray(q, dtype=np.float64)
        norm = np.linalg.norm(q)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("quaternion must be finite and nonzero")

        q = q / norm

        return -q if q[0] < 0 else q

    @classmethod
    def _distance(cls, a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
        """How far apart two poses are: their positions (metres) and orientations (radians)."""
        return float(np.linalg.norm(np.subtract(a[:3], b[:3]))), cls.quat_angle(a[3:], b[3:])
