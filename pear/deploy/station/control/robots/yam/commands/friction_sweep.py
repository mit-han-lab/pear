import json
import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from pear.deploy.station.control.commands import StationFileSnippet
from pear.deploy.station.control.joint.config import MotionLimits
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.trajectory import GripperProfile
from pear.deploy.station.control.robots.yam.commands.friction_fit import FrictionFit
from pear.deploy.station.control.robots.yam.config import ArmConfig
from pear.deploy.station.control.robots.yam.layout import ARM_TYPE, GRIPPER_TYPE, LAYOUT
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from pear.deploy.station.control.schemas import ControlState
from pear.utils.file_utils import write_text_atomically


class FrictionSweep:
    """The supervised friction identification of one arm's joints: slow constant-velocity sweeps, both ways, at poses
    of different gravity loads while the other arm holds, fitted into the lines of the station's file."""

    # the swept arm's starts, the tool clear of the table, rad: the arm lifted, and low, where J2 carries about twice
    # the load
    POSES = {"lifted": [0.0, 1.53, 1.50, -1.53, 0.0, 0.0], "low": [0.0, 2.25, 2.5, -1.3, 0.0, 0.0]}
    # the joints swept at each pose: low adds a load only to the joints that carry the arm, J2-J4
    SWEPT_JOINTS = {"lifted": [0, 1, 2, 3, 4, 5], "low": [1, 2, 3]}
    # each joint's sweep about its pose, rad; J4 starts near its lower bound, so it sweeps upward, clear of the stop
    # even with a zero 0.1 rad off
    SWEEP_RANGE = [(-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2), (0.0, 0.5), (-0.4, 0.4), (-0.6, 0.6)]
    SPEEDS = [0.05, 0.1, 0.2]  # of the sweeps, rad/s
    MAX_SPEED = 0.3  # rad/s
    ACCELERATION, JERK = 0.5, 5.0  # gentle ramps into and out of each constant-speed segment
    MIN_TOOL_HEIGHT_CM = 8.0  # a sweep that would bring the tool closer to the table is refused
    SDK_DAMPING_GAINS = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]  # the pinned SDK's Kd, J1-J6; the robot's settings set only Kp
    SDK_POSITION_GAINS = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]  # the pinned SDK's Kp, J1-J6
    # a sample is steady when its target speed differs from the sweep's by less than the rest of this share of it
    STEADY_SPEED_FRACTION = 0.95
    BIN_COUNT = 8  # of the position bins a sweep's samples are paired in
    BOUND_MARGIN = 0.05  # kept between a sweep and a joint limit, rad
    CHECK_POINT_COUNT = 9  # of a sweep's positions checked against the table's clearance
    SETTLE_OVERHEAD_S = 3  # allowance of each sweep for settling, in the estimate of the duration
    POLL_S = 0.02  # between two snapshots of a move
    BREAKAWAY_SPEED_FRACTION = 0.2  # of the sweep's speed, above which a joint counts as moving
    PERCENTILES = (5, 95)  # of the steady samples' positions, which the bins span
    # what a run that is not confirmed prints after its plan, moving nothing
    DRY_RUN_TEXT = "Dry plan only. Stop the GUI, clear the table, stay at the stop, then rerun with --yes."
    SNIPPET_INDENT = 6  # of an arm's keys in the station's file, under robot.arms.<side>

    def __init__(
        self,
        robot: YAMRobot,
        arm: str,
        output_dir: Path,
        joints: list[int],
        speeds: list[float],
        poses: list[str],
    ) -> None:
        self.robot = robot  # closed, in joint mode, the swept arm's friction feedforward off
        self.arm, self.output_dir = arm, output_dir
        self.joints = joints  # the swept joints' indices, each swept at the poses whose SWEPT_JOINTS hold it
        self.speeds = speeds
        self.poses = poses  # the names of the swept poses, of POSES

    @classmethod
    def checked_arguments(
        cls,
        joints_text: str | None,
        speeds_text: str | None,
        refuse: Callable[[str], NoReturn],
    ) -> tuple[list[int], list[float]]:
        """Returns the swept joints' indices and the speeds that the command line gives as comma separated text
        (None: the sweep's own); ``refuse`` refuses joints or speeds out of range."""
        joint_count = LAYOUT.joint_count
        if joints_text is None:
            joints = list(range(joint_count))
        else:
            joints = [int(joint_number) - 1 for joint_number in joints_text.split(",")]
        speeds = list(cls.SPEEDS) if speeds_text is None else [float(speed) for speed in speeds_text.split(",")]
        if not all(0 <= joint < joint_count for joint in joints) or not all(
            0 < speed <= cls.MAX_SPEED for speed in speeds
        ):
            refuse(f"Joints must be 1-{joint_count} and speeds within (0, {cls.MAX_SPEED}] rad/s")

        return joints, speeds

    def run(self, is_confirmed: bool) -> None:
        """Prints the plan and, once ``is_confirmed``, homes both arms, sweeps each joint at each pose and speed both
        ways, homes them again, fits the feedforward, saves the record and prints the results, then the lines of the
        station's file; a run that does not finish saves its record and suggests nothing."""
        robot, arm_config = self.robot, self.robot.robot_config.arms[self.arm]
        sweeps = self.plan()
        if arm_config.joint_position_gains is not None:
            position_gains = list(arm_config.joint_position_gains)
        else:
            position_gains = self.SDK_POSITION_GAINS
        model_gravity_nm = {pose: self.model_gravity(self.POSES[pose]).tolist() for pose in self.poses}
        self.print_plan(sweeps, position_gains, arm_config, model_gravity_nm)
        if not is_confirmed:
            print(self.DRY_RUN_TEXT)
            return

        signal.signal(signal.SIGINT, signal.default_int_handler)
        started = datetime.now()
        time_tag = started.strftime("%Y%m%dT%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        applied_factors = self.applied_gravity_factors(arm_config)
        results, raw_samples, fit = [], [], None

        robot.open()
        try:
            print("Homing both arms...")
            robot.home()
            control_loop = robot.control_loop
            for pose, joint, sweep_start, sweep_end, speed in sweeps:
                samples = self.sweep(control_loop, pose, joint, sweep_start, sweep_end, speed)
                result = self.analyse(
                    samples, joint, speed, position_gains, self.SDK_DAMPING_GAINS, arm_config.uses_velocity_feedforward
                )
                result = {"pose": pose, **result}
                results.append(result)
                raw_samples.append({"pose": pose, "joint": joint + 1, "speed": speed, "samples": samples})
                print(json.dumps(result))
            print("Returning home...")
            robot.home()
            fit = FrictionFit.fit(results, model_gravity_nm, applied_factors)
        except KeyboardInterrupt:
            print("Interrupted: stopping.", file=sys.stderr)
            robot.stop()
            raise
        finally:
            robot.close()
            path = self.output_dir / f"friction-{self.arm}-{time_tag}.json"
            record = {
                "side": self.arm,
                "kp": position_gains,
                "kd": self.SDK_DAMPING_GAINS,
                "velocity_feedforward": arm_config.uses_velocity_feedforward,
                "poses": {pose: self.POSES[pose] for pose in self.poses},
                "gravity_comp_factor": applied_factors,
                "model_gravity_nm": model_gravity_nm,
                "results": results,
                "raw": raw_samples,
                "fit": fit,  # None: the sweeps did not all finish
            }
            write_text_atomically(path, json.dumps(record))
            print(f"Saved {path}")

        for pose in self.poses:
            print(f"\n{pose} pose")
            self.print_table([result for result in results if result["pose"] == pose])
        self.print_fit(fit, applied_factors)
        for warning in fit["warnings"]:
            print(f"Warning: {warning}", file=sys.stderr)
        comments = [
            f"yam friction-sweep, {self.arm} arm, {started.isoformat(timespec='seconds')}; record: {path}",
            f"under robot.arms.{self.arm} in the station file, replacing these keys if they are there:",
        ]
        print()
        print(StationFileSnippet.text(comments, fit["suggested"], self.SNIPPET_INDENT), end="")

    def plan(self) -> list[tuple[str, int, float, float, float]]:
        """Returns every sweep as its pose, joint, start, end and speed, pose by pose, each checked against the
        joint's bounds and the table's clearance before anything moves."""
        arm_limits = self.robot.robot_config.arms[self.arm].limits
        kinematics = YAMKinematics(self.robot.robot_config)

        sweeps = []
        for pose in self.poses:
            sweep_pose = self.POSES[pose]
            for joint in range(LAYOUT.joint_count):
                lower_bound = arm_limits.lower_bounds[joint] + self.BOUND_MARGIN
                upper_bound = arm_limits.upper_bounds[joint] - self.BOUND_MARGIN
                if not lower_bound <= sweep_pose[joint] <= upper_bound:
                    raise ValueError(
                        f"The {pose} pose's J{joint + 1} {sweep_pose[joint]:.2f} is too close to its bounds"
                    )

            for joint in (joint for joint in self.joints if joint in self.SWEPT_JOINTS[pose]):
                sweep_start = sweep_pose[joint] + self.SWEEP_RANGE[joint][0]
                sweep_end = sweep_pose[joint] + self.SWEEP_RANGE[joint][1]
                lower_bound = arm_limits.lower_bounds[joint] + self.BOUND_MARGIN
                upper_bound = arm_limits.upper_bounds[joint] - self.BOUND_MARGIN
                if sweep_start < lower_bound or sweep_end > upper_bound:
                    raise ValueError(
                        f"{pose} J{joint + 1} sweep [{sweep_start:.2f}, {sweep_end:.2f}] is too close to its bounds"
                    )
                for angle in np.linspace(sweep_start, sweep_end, self.CHECK_POINT_COUNT):
                    arm_positions = [*sweep_pose, GripperProfile.OPEN_ENDPOINT]
                    arm_positions[joint] = angle
                    tool_height_cm = 100 * kinematics.pose(self.arm, np.array(arm_positions))[2, 3]
                    if tool_height_cm < self.MIN_TOOL_HEIGHT_CM:
                        raise ValueError(
                            f"{pose} J{joint + 1} at {angle:.2f} puts the tool {tool_height_cm:.1f} cm above the table"
                        )
                sweeps += [(pose, joint, sweep_start, sweep_end, speed) for speed in self.speeds]

        return sweeps

    def print_plan(
        self,
        sweeps: list[tuple[str, int, float, float, float]],
        position_gains: list[float],
        arm_config: ArmConfig,
        model_gravity_nm: dict[str, list[float]],
    ) -> None:
        """Prints the gains, the time all the sweeps take, and each pose with the model's gravity there and its
        sweeps."""
        # each sweep: its way there and back at its speed, a ramp at both ends of each way, and its settling
        duration_s = sum(
            2 * (sweep_end - sweep_start) / speed + 4 * speed / self.ACCELERATION + self.SETTLE_OVERHEAD_S
            for _, _, sweep_start, sweep_end, speed in sweeps
        )

        print(f"{self.arm} arm, Kp {position_gains}, velocity feedforward {arm_config.uses_velocity_feedforward}")
        print(f"{len(sweeps)} sweeps, about {duration_s / 60:.0f} min of motion. The other arm holds.")
        for pose in self.poses:
            print(f"{pose} pose {self.POSES[pose]}, model gravity J1-J6 {np.round(model_gravity_nm[pose], 2)} Nm")
            for sweep_pose, joint, sweep_start, sweep_end, speed in sweeps:
                if sweep_pose == pose:
                    print(f"  J{joint + 1}: {sweep_start:+.2f} <-> {sweep_end:+.2f} rad at {speed} rad/s")

    def sweep(
        self,
        control_loop: JointControlLoop,
        pose: str,
        joint: int,
        sweep_start: float,
        sweep_end: float,
        speed: float,
    ) -> list[tuple[Any, ...]]:
        """Moves the swept arm to the start of one sweep about ``pose`` at the robot's own caps, sweeps the joint
        there and back at ``speed``, and returns the samples logged on the way."""
        start_pose = list(self.POSES[pose])
        start_pose[joint] = sweep_start
        self.move(control_loop, start_pose, None)

        caps = self.sweep_caps(speed)
        end_pose = list(start_pose)
        end_pose[joint] = sweep_end
        samples = self.move(control_loop, end_pose, caps, is_recorded=True)
        samples += self.move(control_loop, start_pose, caps, is_recorded=True)

        return samples

    def move(
        self,
        control_loop: JointControlLoop,
        joint_pose: list[float],
        caps: MotionLimits | None,
        is_recorded: bool = False,
    ) -> list[tuple[Any, ...]]:
        """Moves the swept arm's joints to ``joint_pose`` within ``caps`` (None: the robot's own), waits for the
        hold, and returns the samples logged on the way if ``is_recorded``."""
        joint_count = LAYOUT.joint_count
        targets = {arm: tuple(arm_targets) for arm, arm_targets in control_loop.targets.items()}
        targets[self.arm] = tuple(joint_pose) + (LAYOUT.gripper_value(targets[self.arm]),)
        control_loop.move(targets, caps=caps)

        samples = []
        while True:
            snapshot = control_loop.snapshot()
            if is_recorded:
                arm_feedback = snapshot["arms"][self.arm]
                missing_effort = [float("nan")] * LAYOUT.coordinate_count  # of feedback without effort
                samples.append(
                    (
                        snapshot["captured_monotonic_s"],
                        list(snapshot["targets"][self.arm][:joint_count]),
                        list(arm_feedback["position"][:joint_count]),
                        list(arm_feedback["velocity"][:joint_count]),
                        list(arm_feedback.get("effort", missing_effort)[:joint_count]),
                    )
                )
            if snapshot["status"] == ControlState.FAULT:
                raise RuntimeError(f"Station fault: {snapshot['fault']}")
            if snapshot["status"] == ControlState.HOLD:
                break
            time.sleep(self.POLL_S)
        control_loop.wait_for_hold(JointControlLoop.HOLD_WAIT_S)

        return samples

    def sweep_caps(self, speed: float) -> MotionLimits:
        """Returns the caps of a sweep at ``speed``: the joints' gentle ramps, and the gripper's caps of the
        station."""
        arm_limits = self.robot.robot_config.arms[LAYOUT.arms[0]].limits  # the first arm's: both share the limits
        joint_count = LAYOUT.joint_count

        return MotionLimits(
            velocity=[speed] * joint_count + [LAYOUT.gripper_value(arm_limits.velocity)],
            acceleration=[self.ACCELERATION] * joint_count + [LAYOUT.gripper_value(arm_limits.acceleration)],
            jerk=[self.JERK] * joint_count + [LAYOUT.gripper_value(arm_limits.jerk)],
        )

    @classmethod
    def analyse(
        cls,
        samples: list[tuple[Any, ...]],
        joint: int,
        speed: float,
        position_gains: list[float],
        damping_gains: list[float],
        uses_velocity_feedforward: bool,
    ) -> dict[str, Any]:
        """Returns the friction, the gravity model's error and the breakaway torque of one joint from the samples of
        its sweep."""
        times, targets, positions, velocities, efforts = (np.array(column, dtype=float) for column in zip(*samples))
        joint_targets, joint_positions = targets[:, joint], positions[:, joint]
        joint_velocities, joint_efforts = velocities[:, joint], efforts[:, joint]
        planned_speed = np.gradient(joint_targets, times)  # the commanded speed, from the logged targets
        is_steady = np.abs(np.abs(planned_speed) - speed) < (1 - cls.STEADY_SPEED_FRACTION) * speed
        direction = np.sign(planned_speed)
        # in a constant-speed segment the plan's velocity is exactly +/- speed
        target_velocity = direction * speed if uses_velocity_feedforward else 0.0
        pd_torque = position_gains[joint] * (joint_targets - joint_positions) + damping_gains[joint] * (
            target_velocity - joint_velocities
        )

        lower_position, upper_position = (
            np.percentile(joint_positions[is_steady], cls.PERCENTILES) if is_steady.any() else (0, 0)
        )
        edges = np.linspace(lower_position, upper_position, cls.BIN_COUNT + 1)
        plus_torques, minus_torques, effort_gaps = [], [], []
        for bin_start, bin_end in zip(edges[:-1], edges[1:]):
            is_in_bin = is_steady & (joint_positions >= bin_start) & (joint_positions < bin_end)
            plus_bin_torques = pd_torque[is_in_bin & (direction > 0)]
            minus_bin_torques = pd_torque[is_in_bin & (direction < 0)]
            if len(plus_bin_torques) and len(minus_bin_torques):
                plus_torques.append(np.median(plus_bin_torques))
                minus_torques.append(np.median(minus_bin_torques))
                effort_gaps.append(
                    np.median(joint_efforts[is_in_bin & (direction > 0)])
                    - np.median(joint_efforts[is_in_bin & (direction < 0)])
                )
        if not plus_torques:
            return {"joint": joint + 1, "speed": speed, "error": "no paired constant-speed samples"}
        plus_torques, minus_torques = np.array(plus_torques), np.array(minus_torques)

        # the breakaway: the peak |PD torque| in each direction before the joint starts moving
        breakaway = {}
        for sign, name in ((1, "plus"), (-1, "minus")):
            is_moving = direction == sign
            if is_moving.any():
                start_index = np.argmax(is_moving)
                is_still = np.abs(joint_velocities[start_index:]) < cls.BREAKAWAY_SPEED_FRACTION * speed
                stop_index = start_index + (np.argmin(is_still) if not is_still.all() else len(is_still))
                breakaway[name] = float(np.max(np.abs(pd_torque[start_index : max(stop_index, start_index + 1)])))

        return {
            "joint": joint + 1,
            "speed": speed,
            "bins": len(plus_torques),
            "needed_torque_plus_nm": float(np.median(plus_torques)),
            "needed_torque_minus_nm": float(np.median(minus_torques)),
            "friction_nm": float(np.median(plus_torques - minus_torques) / 2),
            "gravity_model_error_nm": float(np.median(plus_torques + minus_torques) / 2),
            "friction_spread_nm": float(np.std((plus_torques - minus_torques) / 2)),
            "effort_friction_nm": float(np.median(effort_gaps) / 2) if effort_gaps else None,
            "breakaway_nm": breakaway,
        }

    @staticmethod
    def print_fit(fit: dict[str, Any], applied_factors: list[float]) -> None:
        """Prints each joint's gravity factor, applied and fitted, its Coulomb friction at each pose, and the
        constant and load parts of its friction, before the share of them that is fed forward."""
        print("\n joint  applied-factor  fitted-factor  constant  load-ratio  coulomb-by-pose")
        for name, joint_fit in fit["joints"].items():
            coulomb = ", ".join(f"{pose} {value:.2f}" for pose, value in joint_fit["coulomb_nm_by_pose"].items())
            print(
                f"  {name}   {applied_factors[int(name[1:]) - 1]:.3f}           {joint_fit['gravity_factor']:.3f}"
                f"          {joint_fit['friction_constant_nm']:6.2f}    {joint_fit['friction_load_ratio']:.3f}"
                f"       {coulomb}"
            )

    @staticmethod
    def print_table(results: list[dict[str, Any]]) -> None:
        """Prints each sweep's friction, the torques it needed both ways, the gravity model's error and the
        breakaway."""
        print(" joint  speed   friction  need(+)  need(-)  gravity-err  breakaway(+/-)")

        for result in results:
            if "error" in result:
                print(f"  J{result['joint']}   {result['speed']:.2f}   {result['error']}")
                continue
            breakaway = result["breakaway_nm"]
            print(
                f"  J{result['joint']}   {result['speed']:.2f}   {result['friction_nm']:6.2f}   "
                f"{result['needed_torque_plus_nm']:+6.2f}   {result['needed_torque_minus_nm']:+6.2f}   "
                f"{result['gravity_model_error_nm']:+6.2f}       "
                f"{breakaway.get('plus', float('nan')):.2f}/{breakaway.get('minus', float('nan')):.2f}"
            )

    @staticmethod
    def model_gravity(pose: list[float]) -> np.ndarray:
        """Returns the gravity torque of each joint that the SDK's model gives at ``pose``, before any factor, Nm."""
        # the model's libraries: imported where the sweep needs them
        import mujoco
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType(ARM_TYPE), GripperType(GRIPPER_TYPE)))
        data = mujoco.MjData(model)
        data.qpos[: len(pose)] = pose
        mujoco.mj_forward(model, data)

        return data.qfrc_bias[: len(pose)].copy()

    @staticmethod
    def applied_gravity_factors(arm_config: ArmConfig) -> list[float]:
        """Returns the factors J1-J6 of the gravity model that the SDK applies to the swept arm."""
        if arm_config.gravity_compensation_factors is not None:
            return list(arm_config.gravity_compensation_factors)

        from i2rt.robots.utils import ArmType, _load_arm_config

        return [float(factor) for factor in _load_arm_config(ArmType(arm_config.arm_type)).gravity_comp_factor]
