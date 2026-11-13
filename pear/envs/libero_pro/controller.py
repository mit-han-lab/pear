from __future__ import annotations

import math
from typing import Any

import numpy as np

# robosuite is imported inside the methods: it exists only in the simulator's Python (the verl side imports this
# module through env.py, its env class's SCHEMA)


class MotionController:
    """Moves LIBERO's Panda as one action says, the arm's quantities changed by their amounts and the gripper opened or
    closed all the way: it steps the robosuite scene on native actions in closed loop until the targets are reached at
    rest or the robot stops short of them."""

    # what an action sets, per part: the tool point's pose in the robot's base frame and the fingers' gap (cm, radians)
    SCHEMA: dict[str, tuple[str, ...]] = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
    # the decimals of each value the model reads
    DECIMAL_N_BY_QUANTITY = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}
    POS_DELTA_M = 0.05  # the OSC's position change at offset 1.0 (robosuite's osc_pose output_max)
    ROT_DELTA_RAD = 0.5  # its orientation change at offset 1.0
    # the Panda gripper's native action only says which way its command goes (the fingers' opening, -1 closed .. 1
    # open); 0 holds the command
    GRIP_RATE = 0.25  # the command's change per control step: 0.01 per physics substep, 25 substeps
    GRIP_SPAN_CM = 8.0  # the fingers' gap from closed to open: each finger's target spans 0..4 cm
    # the motion law, one native action per control step: every axis's goal offset is a PD law on its remaining error,
    # (error - LEAD_S * velocity) / SPAN, clipped to [-1, 1]: full speed while far, braking in time to stop on the
    # target (the OSC caps the speed), then decoupled through the OSC's own model (_decouple)
    LEAD_S = 0.04  # the law's lead: the velocity counts against the error as this many seconds of motion
    SPAN_M = 0.015  # the position error at which an axis's offset saturates (at rest)
    SPAN_RAD = 0.15  # the orientation error at which an axis's offset saturates (at rest)
    DECOUPLE_DAMPING = 0.1  # of the coupling's inverse (normalized units)
    DECOUPLE_MAX = 0.5  # the decoupling's largest change of one offset
    # the model behind the decoupling is the free arm's: none near a joint's limit, and none far off the orientation
    # target (a turn underway, or a contact twisting the tool, where it would twist it further)
    DECOUPLE_JOINT_MARGIN_RAD = 0.2  # no decoupling with a joint this close to its limit
    DECOUPLE_ROT_ERR_RAD = 0.1  # no decoupling with the orientation this far off its target
    POS_TOL_M = 0.003  # a position target counts as reached within this
    ROT_TOL_RAD = 0.02  # an orientation target counts as reached within this
    REST_MPS = 0.03  # at rest, the tool point moves slower than this at the step's end
    REST_RADPS = 0.15  # at rest, the tool point turns slower than this at the step's end
    GRIP_STILL_M = 0.0005  # at rest, the fingers' gap changed less than this in the step (when the action sets it)
    GRIP_TOL_CM = 0.5  # the fingers also count as at rest this close to the width of the end they go to
    PROGRESS_M = 0.001  # progress: the position error fell this far below its least so far
    PROGRESS_RAD = 0.005  # progress: the orientation error fell this far below its least so far
    # without progress for this many steps at rest speed, something keeps the targets: the robot stops there
    BLOCKED_STEP_N = 2
    STALL_STEP_N = 6  # without progress for this many steps at any speed, the robot stops there too
    MAX_ACTION_STEP_N = 100  # the native actions one action may take
    GIMBAL_LOCK_TOL = 1e-9  # sin(pitch) this close to +-1: roll and yaw turn about one axis

    def __init__(self, scene: Any, observation: dict[str, np.ndarray], action: dict[str, Any]) -> None:
        import robosuite.utils.transform_utils as T

        self.scene = scene  # the episode's OffScreenRenderEnv
        self.gripper = scene.robots[0].gripper

        # the arm's targets: its current values plus the action's changes; a quantity changed by 0 keeps its value
        targets = self._measured_quantities(scene, observation)
        for quantity, change in action["arm"].items():
            targets[quantity] += float(change)

        # in the world frame, where the OSC works: the tool point's position and orientation (quaternion)
        base_pos, base_rot = self._base(scene)
        self.pos_target = base_pos + base_rot @ (np.array([targets["x"], targets["y"], targets["z"]]) / 100.0)
        self.quat_target = T.mat2quat(
            base_rot @ self._rpy_to_matrix([targets["roll"], targets["pitch"], targets["yaw"]])
        )

        # the gripper goes all the way the width's change points (LIBERO's own use of it): open for a positive change,
        # closed for a negative one, held where it is for 0 (None); a command stopped short grips with less force, or
        # misses a narrow object
        width_change = float(action["gripper"]["width"])
        self.opening_target = float(np.sign(width_change)) if width_change != 0.0 else None
        self.width_target = (
            None if self.opening_target is None else self.GRIP_SPAN_CM * (self.opening_target + 1.0) / 2.0
        )

        # the robot after the last step, and the progress toward the targets so far
        self.observation = observation
        self.pos, self.quat, self.gap = self._tool(observation)
        self.vel, self.ang_vel = self._velocity()
        self.least_pos_err = self.least_rot_err = math.inf
        self.stall_step_n = self.slow_stall_step_n = 0
        self.is_stopping = False

    def move(self, budget: int) -> tuple[dict[str, np.ndarray], list[list[float]], list[float], bool, str]:
        """Native actions toward the targets until they are reached at rest, the robot stops short of them, the task
        ends, or the episode's ``budget`` or the action's cap runs out: the last observation, the actions, their
        rewards, whether the episode ended and the outcome (reached | stopped | done | budget | timeout)."""
        actions: list[list[float]] = []
        rewards: list[float] = []

        for _ in range(min(self.MAX_ACTION_STEP_N, budget)):
            native_action = np.concatenate([self._arm_offsets(), [self._grip_direction()]])
            self.observation, reward, done, _ = self.scene.step(native_action)
            actions.append(native_action.tolist())
            rewards.append(float(reward))
            if done:
                return self.observation, actions, rewards, True, "done"
            outcome = self._step_outcome()
            if outcome is not None:
                return self.observation, actions, rewards, False, outcome

        done = len(actions) >= budget

        return self.observation, actions, rewards, done, "budget" if done else "timeout"

    @classmethod
    def state(cls, scene: Any, observation: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        """The quantities as the model reads them, by part, rounded to DECIMAL_N_BY_QUANTITY."""
        values = cls._measured_quantities(scene, observation)

        # + 0.0: a value that rounds to zero reads 0.0, never -0.0
        return {
            part: {
                quantity: round(float(values[quantity]), cls.DECIMAL_N_BY_QUANTITY[quantity]) + 0.0
                for quantity in quantities
            }
            for part, quantities in cls.SCHEMA.items()
        }

    def _arm_offsets(self) -> np.ndarray:
        """The six arm components of the next native action (the OSC's normalized goal offsets): the PD law per axis,
        then decoupled."""
        rot_err = self._rotation(self.quat_target, self.quat)
        twist = np.concatenate([self.vel, self.ang_vel])
        span = np.array([self.SPAN_M] * 3 + [self.SPAN_RAD] * 3)
        offsets = np.clip(
            (np.concatenate([self.pos_target - self.pos, rot_err]) - self.LEAD_S * twist) / span, -1.0, 1.0
        )

        if np.linalg.norm(rot_err) > self.DECOUPLE_ROT_ERR_RAD or self._joint_margin() < self.DECOUPLE_JOINT_MARGIN_RAD:
            return offsets

        return self._decouple(offsets, twist)

    def _joint_margin(self) -> float:
        """The arm's smallest distance of a joint from its limit (rad)."""
        robot = self.scene.robots[0]
        sim = self.scene.sim
        joint_positions = sim.data.qpos[robot._ref_joint_pos_indexes]
        lower_limits, upper_limits = sim.model.jnt_range[robot._ref_joint_indexes].T

        return float(np.min(np.minimum(joint_positions - lower_limits, upper_limits - joint_positions)))

    def _decouple(self, offsets: np.ndarray, twist: np.ndarray) -> np.ndarray:
        """The offsets that give the OSC a decoupled response, K^-1 offsets + (kd / kp) (I - K^-1) twist in normalized
        units, each changed by at most DECOUPLE_MAX."""
        # with uncoupling=True robosuite's OSC scales the position and the orientation part of its wrench by separate
        # task-space inertias: the tool accelerates by K (kp * offset - kd * twist), K = Lambda_full^-1
        # blockdiag(Lambda_pos, Lambda_ori), so a push along x also tilts the tool and a turn also moves it
        from robosuite.utils.control_utils import opspace_matrices

        ctrl = self.scene.robots[0].controller
        if not getattr(ctrl, "uncoupling", False):
            return offsets

        # the controller's matrices as of its last update (the last physics substep): no sim.forward() here, which would
        # change the solver's warm start and break the exact replay of the native actions
        lambda_full, lambda_pos, lambda_ori, _ = opspace_matrices(ctrl.mass_matrix, ctrl.J_full, ctrl.J_pos, ctrl.J_ori)
        block_inertia = np.zeros((6, 6))
        block_inertia[:3, :3], block_inertia[3:, 3:] = lambda_pos, lambda_ori
        scale = np.array([self.POS_DELTA_M] * 3 + [self.ROT_DELTA_RAD] * 3)
        coupling = np.linalg.solve(lambda_full, block_inertia) * (scale[None, :] / scale[:, None])

        # a damped inverse: K degenerates near the arm's singular poses
        coupling_inverse = np.linalg.solve(coupling.T @ coupling + self.DECOUPLE_DAMPING**2 * np.eye(6), coupling.T)
        decoupled_offsets = coupling_inverse @ offsets + float(ctrl.kd[0] / ctrl.kp[0]) * (
            np.eye(6) - coupling_inverse
        ) @ (twist / scale)
        correction = np.clip(decoupled_offsets - offsets, -self.DECOUPLE_MAX, self.DECOUPLE_MAX)

        return np.clip(offsets + correction, -1.0, 1.0)

    def _grip_direction(self) -> float:
        if self.opening_target is None:
            return 0.0  # hold the command where it is

        opening = self._opening()
        if opening > self.opening_target + self.GRIP_RATE / 2:
            return 1.0  # close
        if opening < self.opening_target - self.GRIP_RATE / 2:
            return -1.0  # open

        return 0.0

    def _step_outcome(self) -> str | None:
        """The action's end after a step: "reached" or "stopped" once the robot is at rest there, None while it goes
        on."""
        previous_gap = self.gap
        self.pos, self.quat, self.gap = self._tool(self.observation)
        self.vel, self.ang_vel = self._velocity()

        pos_err = float(np.linalg.norm(self.pos_target - self.pos))
        rot_err = float(np.linalg.norm(self._rotation(self.quat_target, self.quat)))
        is_slow = np.linalg.norm(self.vel) < self.REST_MPS and np.linalg.norm(self.ang_vel) < self.REST_RADPS

        # the fingers are done once their gap is at the action's width or stops changing (something between them);
        # they can close slowly on an object for many steps
        fingers_done = (
            self.width_target is None
            or abs(self.gap - previous_gap) < self.GRIP_STILL_M
            or abs(self.gap * 100.0 - self.width_target) < self.GRIP_TOL_CM
        )
        opening_reached = self.opening_target is None or abs(self._opening() - self.opening_target) < self.GRIP_RATE / 2
        is_at_rest = is_slow and fingers_done and opening_reached
        arm_reached = pos_err < self.POS_TOL_M and rot_err < self.ROT_TOL_RAD

        if is_at_rest and (arm_reached or self.is_stopping):
            return "stopped" if self.is_stopping else "reached"
        if self.is_stopping:
            return None

        # progress is the arm's: the gripper's command reaches its opening by construction
        if (
            arm_reached
            or pos_err < self.least_pos_err - self.PROGRESS_M
            or rot_err < self.least_rot_err - self.PROGRESS_RAD
        ):
            self.least_pos_err, self.least_rot_err = min(self.least_pos_err, pos_err), min(self.least_rot_err, rot_err)
            self.stall_step_n = self.slow_stall_step_n = 0
        else:
            self.stall_step_n += 1
            self.slow_stall_step_n = self.slow_stall_step_n + 1 if is_slow else 0

        if not arm_reached and (
            self.slow_stall_step_n >= self.BLOCKED_STEP_N or self.stall_step_n >= self.STALL_STEP_N
        ):
            self.is_stopping = True  # stop pushing: come to rest where it is
            self.pos_target, self.quat_target = self.pos.copy(), self.quat.copy()
            if is_at_rest:
                return "stopped"

        return None

    @classmethod
    def _measured_quantities(cls, scene: Any, observation: dict[str, np.ndarray]) -> dict[str, float]:
        """Every quantity of SCHEMA, unrounded: the tool point in the base frame (cm, rad) and the fingers' gap (cm)."""
        import robosuite.utils.transform_utils as T

        base_pos, base_rot = cls._base(scene)
        tool_pos, tool_quat, gap = cls._tool(observation)
        pos = base_rot.T @ (tool_pos - base_pos) * 100.0
        rpy = cls._matrix_to_rpy(base_rot.T @ T.quat2mat(tool_quat))
        values = dict(zip(cls.SCHEMA["arm"], [*pos, *rpy]))
        values["width"] = gap * 100.0

        return values

    @staticmethod
    def _base(scene: Any) -> tuple[np.ndarray, np.ndarray]:
        """The robot base's world position and rotation matrix."""
        sim = scene.sim
        body = sim.model.body_name2id("robot0_base")

        return sim.data.body_xpos[body].copy(), sim.data.body_xmat[body].reshape(3, 3).copy()

    def _velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """The tool point's world-frame linear (m/s) and angular (rad/s) velocity, as the OSC measures them."""
        sim = self.scene.sim
        site = self.scene.robots[0].controller.eef_name

        vel = np.array(sim.data.get_site_xvelp(site), dtype=np.float64)
        ang_vel = np.array(sim.data.get_site_xvelr(site), dtype=np.float64)

        return vel, ang_vel

    def _opening(self) -> float:
        return float(self.gripper.current_action[0])  # (opening, -opening): one value per finger actuator

    @staticmethod
    def _tool(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        """World-frame position and orientation (quaternion) of the tool point and the fingers' gap, in meters."""
        return (
            np.array(observation["robot0_eef_pos"], dtype=np.float64),
            np.array(observation["robot0_eef_quat"], dtype=np.float64),
            float(observation["robot0_gripper_qpos"][0] - observation["robot0_gripper_qpos"][1]),
        )

    @staticmethod
    def _rotation(quat_to: np.ndarray, quat_from: np.ndarray) -> np.ndarray:
        """The shortest world-frame rotation (axis-angle) from ``quat_from`` to ``quat_to``: the OSC's delta
        convention."""
        import robosuite.utils.transform_utils as T

        quat = T.quat_multiply(quat_to, T.quat_inverse(quat_from))
        if quat[3] < 0.0:  # w: robosuite's quaternions are (x, y, z, w)
            quat = -quat

        return T.quat2axisangle(quat)

    @staticmethod
    def _rpy_to_matrix(rpy) -> np.ndarray:
        """Fixed-axis roll, pitch, yaw about x, y, z (REP 103): Rz(yaw) Ry(pitch) Rx(roll)."""
        # robosuite's euler2mat is not the inverse of its mat2euler, so both directions are written out here
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
    def _matrix_to_rpy(cls, rot: np.ndarray) -> np.ndarray:
        """The inverse of _rpy_to_matrix; at pitch +-90 degrees roll and yaw turn about one axis, and yaw takes it
        all."""
        sp = min(1.0, max(-1.0, -float(rot[2, 0])))

        if abs(sp) < 1.0 - cls.GIMBAL_LOCK_TOL:
            return np.array([math.atan2(rot[2, 1], rot[2, 2]), math.asin(sp), math.atan2(rot[1, 0], rot[0, 0])])

        return np.array([0.0, math.asin(sp), math.atan2(-rot[0, 1], rot[1, 1])])
