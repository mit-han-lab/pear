from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# numpy only: the verl side imports this module through env.py (its env class's SCHEMA), the simulator side the
# whole controller


class MotionController:
    """Moves RoboDojo's two ARX X5 arms through one episode as the model's actions say, in official control steps: every
    arm the action moves along waypoints toward its goal at a constant speed until it reaches it or stops short, every
    gripper it sets opened or closed all the way."""

    # what an action sets: per arm the tool point's pose in the robot's frame, per gripper the finger gap (cm, radians)
    SCHEMA: dict[str, tuple[str, ...]] = {
        "left_arm": ("x", "y", "z", "roll", "pitch", "yaw"),
        "right_arm": ("x", "y", "z", "roll", "pitch", "yaw"),
        "left_gripper": ("width",),
        "right_gripper": ("width",),
    }
    ARMS = ("left", "right")  # RoboDojo's target robots, in the order of its robot list
    # the decimals of each value the model reads
    DECIMAL_N_BY_QUANTITY = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}
    # takes world vectors to the robot's frame: its x (forward) is world +y, its y (left) world -x, its z up
    WORLD_TO_ROBOT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ORIGIN_CM = np.array([-45.0, 0.0, 76.5])  # the robot frame's origin, midway between the arms' bases, in robot axes
    # the gripper frame's axes in the wrist link's: x the fingers' closing direction, z against where the gripper points
    GRIPPER_FRAME = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    # the tool point, the finger pads' centre, from the wrist link RoboDojo solves IK for
    TOOL_OFFSET_M = (0.155, 0.0, 0.0)
    STEP_M = 0.02  # a tool point's waypoints advance at most this far per control step (25 Hz)
    STEP_RAD = math.radians(10)  # a tool point's waypoints turn at most this much per control step
    # the largest joint change one control step commands: the drives follow it
    JOINT_STEP_RAD = 0.3
    POS_TOL_M = 0.003  # a goal's position is reached within this
    ROT_TOL_RAD = 0.02  # a goal's orientation is reached within this
    GRIP_TOL_M = 0.0025  # a gripper is at its opening within this much finger travel
    GRIP_STILL_M = 3e-4  # fingers that moved less than this in a control step are still
    GRIP_STILL_STEP_N = 3  # a gripper whose fingers are still for this many control steps has stopped (on an object)
    PROGRESS_M = 0.001  # the least an arm's distance to its goal shrinks in a control step that counts as progress
    PROGRESS_RAD = 0.005  # the least an arm's angle to its goal shrinks in a control step that counts as progress
    # an arm without progress for this many control steps is blocked (a contact), out of reach or unsolvable: it stops
    # where it is
    STALL_STEP_N = 6
    MAX_ACTION_STEP_N = 100  # the control steps of one action at most
    # a solution moving any joint further than this lies in another arm configuration (a flip of the wrist or shoulder,
    # past a singularity or the edge of its reach), which a step never swings the arm into: with no other it holds
    IK_JUMP_RAD = 1.0
    # the official IK's seeds (curobo_planner.py, _build_ik_cfg), every one of whose solutions a solve returns
    IK_SEED_N = 32
    ELBOW_INDEX = 2  # the X5's elbow, its third joint
    # the angle of joint3 to joint4 ((0.245, 0, -0.056) m in link3's frame, X5A.urdf) off link3's axis
    ELBOW_OFFSET_RAD = math.atan2(0.056, 0.245)
    # the elbow as the start bends it, from folded to straight, where joint3 to joint4 lines up with the upper arm:
    # bent the other way, its solutions need the wrist beyond where it stops (joint4 at about -2.2 rad) and the arm
    # moves against its waypoints
    ELBOW_RANGE_RAD = (-ELBOW_OFFSET_RAD, math.pi - ELBOW_OFFSET_RAD)
    STEP_ROUNDING = 1e-9  # a distance of a whole number of steps up to float rounding takes that many
    SLERP_LINEAR_DOT = 0.9995  # orientations this close (the cosine of half their angle) interpolate linearly
    GIMBAL_LOCK_TOL = 1e-9  # sin(pitch) this close to +-1: roll and yaw turn about one axis

    def __init__(self, env: Any, observation: dict[str, Any]) -> None:
        self.env = env  # the episode's official env (create_eval_env), reset
        self.robots = dict(zip(self.ARMS, [robot for robot in env.robot_manager.robot_list if robot.type == "target"]))

        # what every arm and gripper is commanded to, at the start where the reset left them: the gripper's opening
        # (1 open, 0 closed) and the arm's joint targets
        self.grip_commands = {arm: float(observation["state"][f"{arm}_ee_joint_state"][0]) for arm in self.ARMS}
        self.joint_targets = {arm: env.robot_manager.get_joint(robot, [0])[0] for arm, robot in self.robots.items()}

    def move(self, action: dict[str, dict[str, float]]) -> str:
        """Carries out the model's ``action`` from the robot as it is now and returns "reached", "stopped" (an arm or a
        gripper stopped short) or "done" (the episode ended in it)."""
        goals = self._goals(action)
        goal_poses = {arm: goal["pose"] for arm, goal in goals.items() if goal["pose"] is not None}
        gripper_arms = [arm for arm, goal in goals.items() if goal["opening"] is not None]

        for arm in gripper_arms:
            self.grip_commands[arm] = goals[arm]["opening"]

        # every moved arm's waypoints from where it is (the first), as many as the farthest arm needs (the arms move in
        # step): the next one each control step while the arm keeps within a step of the current one, so a blocked arm
        # is never pressed far
        tool_poses = {arm: self._measured_tool_pose(arm) for arm in goal_poses}
        start_poses, waypoints, waypoint_numbers = dict(tool_poses), dict(tool_poses), dict.fromkeys(goal_poses, 0)
        distances = [self._distance(tool_poses[arm], pose) for arm, pose in goal_poses.items()]
        waypoint_n = max(
            [1]
            + [
                math.ceil(max(distance_m / self.STEP_M, angle_rad / self.STEP_RAD) - self.STEP_ROUNDING)
                for distance_m, angle_rad in distances
            ]
        )

        # per moved arm its least distances to its goal and the control steps since they last shrank; per set gripper
        # its finger joints and the steps they have been still
        least_distances, stall_step_n_by_arm = (
            dict.fromkeys(goal_poses, (math.inf, math.inf)),
            dict.fromkeys(goal_poses, 0),
        )
        reached_arms, stopped_arms = set(), set()
        finger_joints, still_step_n_by_arm = (
            {arm: self._fingers(arm) for arm in gripper_arms},
            dict.fromkeys(gripper_arms, 0),
        )

        for _ in range(self.MAX_ACTION_STEP_N):  # (every action takes at least one control step)
            moving_arms = [arm for arm in goal_poses if arm not in reached_arms | stopped_arms]
            for arm in moving_arms:
                distance_m, angle_rad = self._distance(tool_poses[arm], waypoints[arm])
                if waypoint_numbers[arm] < waypoint_n and distance_m <= self.STEP_M and angle_rad <= self.STEP_RAD:
                    waypoint_numbers[arm] += 1
                    waypoints[arm] = self.interpolate(
                        start_poses[arm], goal_poses[arm], waypoint_numbers[arm] / waypoint_n
                    )

            self._step({arm: waypoints[arm] for arm in moving_arms})
            if self._ended():
                return "done"

            tool_poses = {arm: self._measured_tool_pose(arm) for arm in goal_poses}
            for arm in moving_arms:
                distance_m, angle_rad = self._distance(tool_poses[arm], goal_poses[arm])
                least_distance_m, least_angle_rad = least_distances[arm]
                if distance_m <= self.POS_TOL_M and angle_rad <= self.ROT_TOL_RAD:
                    reached_arms.add(arm)
                elif distance_m < least_distance_m - self.PROGRESS_M or angle_rad < least_angle_rad - self.PROGRESS_RAD:
                    stall_step_n_by_arm[arm] = 0
                else:
                    stall_step_n_by_arm[arm] += 1
                    if stall_step_n_by_arm[arm] >= self.STALL_STEP_N:
                        stopped_arms.add(arm)
                        self._hold(arm)
                least_distances[arm] = (min(least_distance_m, distance_m), min(least_angle_rad, angle_rad))

            for arm in gripper_arms:
                measured_fingers = self._fingers(arm)
                is_still = float(np.max(np.abs(measured_fingers - finger_joints[arm]))) < self.GRIP_STILL_M
                still_step_n_by_arm[arm] = still_step_n_by_arm[arm] + 1 if is_still else 0
                finger_joints[arm] = measured_fingers

            grippers_done = all(
                self._gripper_done(arm, finger_joints[arm], still_step_n_by_arm[arm]) for arm in gripper_arms
            )
            if len(reached_arms | stopped_arms) == len(goal_poses) and grippers_done:
                break
        else:  # out of steps short of the goals: stop there, not pressing on
            for arm in set(goal_poses) - reached_arms - stopped_arms:
                stopped_arms.add(arm)
                self._hold(arm)

        self.env.is_episode_end()  # the official deploy loop's check before its next decision
        if self._ended():
            return "done"

        return "reached" if not stopped_arms and grippers_done else "stopped"

    def state(self) -> dict[str, dict[str, float]]:
        """The quantities as the model reads them, by part."""
        return self.state_of_quantities(self._measured_quantities())

    def _goals(self, action: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Per arm the goal of the model's ``action`` from the robot's unrounded quantities as they are now: its tool
        point's world ``pose`` and its gripper's ``opening`` command, either None where the action changes nothing."""
        exact_quantities = self._measured_quantities()
        goals = {}
        for arm in self.ARMS:
            changes, quantities = action[f"{arm}_arm"], exact_quantities[f"{arm}_arm"]
            is_moved = any(float(v) != 0.0 for v in changes.values())
            pose = (
                self.tool_pose_of_quantities(
                    {quantity: quantities[quantity] + float(changes[quantity]) for quantity in quantities}
                )
                if is_moved
                else None
            )
            width_change = float(action[f"{arm}_gripper"]["width"])
            # all the way the width's change points: 1 (open) for a positive one, 0 (closed) for a negative one
            opening = None if width_change == 0.0 else 1.0 if width_change > 0.0 else 0.0
            if pose is not None or opening is not None:
                goals[arm] = {"pose": pose, "opening": opening}

        if not goals:  # both arms hold where they are (the action still takes its control step)
            goals = {
                arm: {"pose": self.tool_pose_of_quantities(exact_quantities[f"{arm}_arm"]), "opening": None}
                for arm in self.ARMS
            }

        return goals

    def _step(self, waypoints: dict[str, list[float]]) -> None:
        """One official control step with the official deploy's task checks: every arm of ``waypoints`` commanded to the
        official IK's best solution of its waypoint in the arm's configuration, every other arm held at its targets."""
        manager, native_action = self.env.robot_manager, {}

        # the official IK, solved here in the robot list's order, cuRobo's float32 joints passed on in their dtype (the
        # official step interpolates in it); no solution keeps the arm's joint targets. Every arm is named: the official
        # step keeps a missing arm's previous command, which a scene's first step has none of
        for arm, robot in self.robots.items():
            if arm in waypoints:
                # the official IK ranks its seeds by pose error alone, and its best may bend the elbow the other way or
                # lie whole turns of a joint off (its seeds span placeholder limits of +-10 rad): the first in its
                # ranking in the arm's configuration, its joints taken those turns back, continues the arm's path
                current_joints = manager.get_joint(robot, [0])[0]
                lower_limit, upper_limit = self.ELBOW_RANGE_RAD
                for joints in self._ik_solutions(robot, self.link_pose_of_tool(waypoints[arm])):
                    joints = joints - 2 * np.pi * np.round((joints - current_joints) / (2 * np.pi))
                    jump = float(np.max(np.abs(joints - current_joints)))
                    if lower_limit <= joints[self.ELBOW_INDEX] <= upper_limit and jump <= self.IK_JUMP_RAD:
                        if jump > self.JOINT_STEP_RAD:  # the share of the way the drives follow in one step
                            joints = current_joints + (joints - current_joints) * np.float32(self.JOINT_STEP_RAD / jump)
                        self.joint_targets[arm] = joints
                        break
            native_action[manager.process_name(robot.arm_name)] = self.joint_targets[arm]
            native_action[manager.process_name(robot.gripper_name)] = [self.grip_commands[arm]]

        self.env.take_action(native_action)
        self.env.is_episode_end()
        if self.env.end_flag[0]:  # the official deploy loop's own check, which ends it
            self.env.is_episode_end()

    def _ik_solutions(self, robot: Any, link_pose: list[float]) -> list[np.ndarray]:
        """Every solution the official IK finds among its seeds for ``robot``'s wrist link at the world pose
        ``link_pose``, best first, in the arm's joint order."""
        from utils.transformer import calculate_target_pose  # RoboDojo's, in the App

        # the target prepared as the official solve_ik_to_joint prepares it
        manager = self.env.robot_manager
        planner = manager.ik_solver[robot.robot_name]
        target = calculate_target_pose(list(robot.entity_origin_pose), planner.robot_origin_pose, link_pose)
        position, quaternion = planner._trans_from_world_to_base(
            np.array(planner.robot_origin_pose, dtype=np.float32), np.array(target, dtype=np.float32)
        )
        position = position + np.asarray(planner.frame_bias, dtype=np.float32)

        result = planner.ik_solver.solve_pose(
            goal_tool_poses=planner._build_goal_pose(position, quaternion),
            current_state=planner._build_joint_state(manager.get_joint(robot, [0])[0]),
            return_seeds=self.IK_SEED_N,
        )
        if result.success is None:
            return []

        names = list(result.js_solution.joint_names)
        positions = result.js_solution.position.detach().cpu().numpy().reshape(-1, len(names))
        joint_indices = [names.index(name) for name in planner.active_joints_name]

        return [
            positions[seed, joint_indices] for seed in np.flatnonzero(result.success.detach().cpu().numpy().reshape(-1))
        ]

    def _ended(self) -> bool:
        """Whether the episode ended or its task flagged its scene unstable (e.g. a missed support-arm placement)."""
        return bool(self.env.end_flag[0]) or 0 in self.env.unstable_envs

    def _hold(self, arm: str) -> None:
        """Stops ``arm`` where it is: its joint targets become the measured joints."""
        self.joint_targets[arm] = self.env.robot_manager.get_joint(self.robots[arm], [0])[0]

    def _gripper_done(self, arm: str, finger_joints: np.ndarray, still_step_n: int) -> bool:
        """Whether ``arm``'s gripper is at its commanded opening or its fingers stopped short of it (on an object)."""
        lower_limit, upper_limit = self.robots[arm].gripper_scale
        # the finger joint the command asks for
        opening = lower_limit + self.grip_commands[arm] * (upper_limit - lower_limit)

        return still_step_n >= self.GRIP_STILL_STEP_N or abs(float(finger_joints[0]) - opening) <= self.GRIP_TOL_M

    def _measured_quantities(self) -> dict[str, dict[str, float]]:
        """The unrounded quantities of the robot as it is now, read from the physics (no rendering)."""
        return self.quantities(
            {arm: self._measured_tool_pose(arm) for arm in self.ARMS},
            {arm: [float(x) for x in self._fingers(arm)] for arm in self.ARMS},
        )

    def _measured_tool_pose(self, arm: str) -> list[float]:
        """The measured tool-point pose of ``arm``."""
        return self.tool_pose_of_link(self.env.robot_manager.get_real_endpose(self.robots[arm], [0])[0])

    def _fingers(self, arm: str) -> np.ndarray:
        """The measured finger joints of ``arm``'s gripper."""
        return np.asarray(self.env.robot_manager.get_end_effector_real_val(self.robots[arm], [0])[0], dtype=np.float64)

    @classmethod
    def quantities(cls, poses: dict[str, list[float]], fingers: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        """The unrounded quantities of the arms' tool-point poses (world [x, y, z, qw, qx, qy, qz]) and fingers."""
        exact_quantities = {}
        for arm in cls.ARMS:
            position = cls.WORLD_TO_ROBOT @ np.asarray(poses[arm][:3], dtype=np.float64) * 100.0 - cls.ORIGIN_CM
            frame = cls.WORLD_TO_ROBOT @ cls.quat_to_matrix(poses[arm][3:]) @ cls.GRIPPER_FRAME
            exact_quantities[f"{arm}_arm"] = dict(zip(cls.SCHEMA[f"{arm}_arm"], [*position, *cls.matrix_to_rpy(frame)]))
            exact_quantities[f"{arm}_gripper"] = {"width": 100.0 * max(0.0, float(sum(fingers[arm][:2])))}

        return {part: {quantity: float(v) for quantity, v in exact_quantities[part].items()} for part in cls.SCHEMA}

    @classmethod
    def state_of_quantities(cls, exact_quantities: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """The quantities as the model reads them: rounded to DECIMAL_N_BY_QUANTITY, a value that rounds to zero as 0.0
        (never -0.0)."""
        return {
            part: {quantity: round(v, cls.DECIMAL_N_BY_QUANTITY[quantity]) + 0.0 for quantity, v in values.items()}
            for part, values in exact_quantities.items()
        }

    @classmethod
    def tool_pose_of_quantities(cls, values: dict[str, float]) -> list[float]:
        """The world tool-point pose [x, y, z, qw, qx, qy, qz] of an arm's quantities."""
        position = cls.WORLD_TO_ROBOT.T @ ((np.array([values["x"], values["y"], values["z"]]) + cls.ORIGIN_CM) / 100.0)
        link_rotation = (
            cls.WORLD_TO_ROBOT.T
            @ cls.rpy_to_matrix([values["roll"], values["pitch"], values["yaw"]])
            @ cls.GRIPPER_FRAME.T
        )

        return [float(v) for v in position] + [float(v) for v in cls.quat_normalize(cls.matrix_to_quat(link_rotation))]

    @classmethod
    def tool_pose_of_link(cls, link_pose: Sequence[float]) -> list[float]:
        """The tool-point pose of a wrist-link pose (the same orientation)."""
        link_pose = np.asarray(link_pose, dtype=np.float64)
        position = link_pose[:3] + cls.quat_to_matrix(link_pose[3:]) @ np.asarray(cls.TOOL_OFFSET_M)

        return position.tolist() + cls.quat_normalize(link_pose[3:]).tolist()

    @classmethod
    def link_pose_of_tool(cls, tool_pose: Sequence[float]) -> list[float]:
        """The wrist-link pose that puts the tool point at ``tool_pose`` (the pose RoboDojo's IK solves for)."""
        tool_pose = np.asarray(tool_pose, dtype=np.float64)
        position = tool_pose[:3] - cls.quat_to_matrix(tool_pose[3:]) @ np.asarray(cls.TOOL_OFFSET_M)

        return position.tolist() + cls.quat_normalize(tool_pose[3:]).tolist()

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
