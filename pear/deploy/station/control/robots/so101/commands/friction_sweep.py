from __future__ import annotations

import argparse
import json
import pkgutil
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.deploy.station.control.commands import RobotCommand, StationFileSnippet
from pear.utils.file_utils import write_text_atomically

if TYPE_CHECKING:  # annotations only: the station's packages are imported where the command runs
    import numpy as np

    from pear.deploy.station.control.joint.control_loop import JointControlLoop
    from pear.deploy.station.control.robots.so101.config import SO101Config
    from pear.deploy.station.control.robots.so101.joint.feedforward import GravityModel
    from pear.deploy.station.control.robots.so101.motion.kinematics import SO101Kinematics

# the station's packages and the robot's own modules, the commands' module among them, are imported inside the
# methods: the command line lists the commands without them, and the commands' module imports this one


class FrictionSweep(RobotCommand):
    """MOVES the arm: identifies the gravity compliance and the friction of its joints by staircases both ways,
    suggests the feedforward and keeps a record in the run directory."""

    NAME = "friction-sweep"
    HELP = "MOVES the arm: identify gravity and friction (see its --help)"
    MOVES_ROBOT = True
    HAS_DRY_RUN = True  # without --yes it plans the sweeps, says what it would do and moves nothing
    # what its --help says of it
    DESCRIPTION = "Supervised identification of gravity compliance and friction: staircases both ways."
    # each joint in turn steps up through a range and back down, settling at every step with the feedforward off;
    # poses with different gravity loads separate a load-dependent friction term. The poses in rad and the gripper's
    # opening: "reach" holds the fingertip forward and low, where shoulder_lift carries several times its Home load
    POSES = {
        "home": None,  # the station's Home
        "reach": [0.03, -0.153, 0.329, 0.825, -1.613, 1.0],
    }
    SWEPT_JOINTS = {"home": [0, 1, 2, 3, 4], "reach": [1, 2, 3]}  # at each pose; the load-free ones only at Home
    # each joint's range about the pose, rad; wrist_flex starts near its upper limit at Home, so its range is mostly
    # downward
    SWEEP_RANGE = [(-0.25, 0.25), (-0.15, 0.15), (-0.15, 0.15), (-0.3, 0.2), (-0.4, 0.4)]
    INTERVALS_PER_DIRECTION = 4  # of a staircase
    SPEED_SCALE = 1 / 3  # of the moves between the steps
    DWELL_S = 0.3  # the extra rest after the control loop reports HOLD
    AVERAGING_WINDOW_S = 0.2  # over which the measured position is averaged
    MIN_TIP_HEIGHT_CM = 4.0  # the lowest the fingertip may go at a step
    LIMIT_MARGIN = 0.05  # the least distance of a range from a joint's limit, rad
    SNIPPET_INDENT = 2  # of the printed feedforward: under robot: in the station file
    OUTPUT_INDENT = 1  # of the record's JSON
    OUTPUT_FOLDER = "friction"  # of the run directory, where the records go

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the poses and whether the feedforward stays on, and keeps the parser for the refusal without
        --yes."""
        parser.description = cls.DESCRIPTION
        parser.set_defaults(command_parser=parser)
        parser.add_argument("--yes", action="store_true", help="required to move the real arm")
        parser.add_argument("--poses", nargs="+", choices=list(cls.POSES), default=list(cls.POSES))
        parser.add_argument(
            "--feedforward",
            action="store_true",
            help="keep the configured feedforward on; the fit is then a residual",
        )

    def run(self) -> None:
        """Runs the sweeps, keeps their record with the fitted report, and prints the suggested feedforward as the
        station file takes it, its warnings on stderr; without --yes, logs what it would do and refuses before the
        arm is enabled."""
        from pear.deploy.station.control.robot import ActionMode
        from pear.deploy.station.control.robots.so101.commands.commands import MaintenanceSteps
        from pear.deploy.station.control.robots.so101.commands.friction_fit import FrictionFit
        from pear.deploy.station.control.robots.so101.joint.feedforward import GravityModel
        from pear.deploy.station.control.robots.so101.layout import JOINTS

        config = self.load_config()
        output_dir = Path(config.run_dir) / self.OUTPUT_FOLDER
        if not self.args.feedforward:
            config.robot.feedforward = None  # the configured feedforward would hide what the sweep measures
        robot = pkgutil.resolve_name(config.robot_class)(config, action_mode=ActionMode.JOINT)
        robot_config = robot.robot_config

        sweeps = self.plan(robot_config, robot.KINEMATICS_CLASS.shared_instance(robot_config), self.args.poses)
        step_count = sum(2 * len(sweep["points"]) - 1 for sweep in sweeps)
        self.log(
            f"{len(sweeps)} staircases, {step_count} settled steps, feedforward "
            f"{'on' if self.args.feedforward else 'off'}; fingertip stays above {self.MIN_TIP_HEIGHT_CM} cm."
        )

        # the dry run ends here: the sweeps are planned and checked, nothing is enabled
        if not self.args.yes:
            self.args.command_parser.error("moving the real arm needs --yes")

        gravity_model = GravityModel(robot_config.geometry.urdf)
        results = []
        with robot:
            start = MaintenanceSteps.held_targets(robot)
            try:
                robot.home(report_progress=MaintenanceSteps.print_home_stage)
                for sweep in sweeps:
                    self.log(f"{sweep['pose']}: {JOINTS[sweep['joint']]}")
                    rows = self.run_sweep(robot.control_loop, gravity_model, sweep)
                    results.append({"pose": sweep["pose"], "joint": sweep["joint"], "rows": rows})
            finally:
                MaintenanceSteps.park(robot, start)

        report, fitted_feedforward = FrictionFit.fit(results)
        suggested_feedforward, warnings = FrictionFit.clamped_suggestion(fitted_feedforward)
        p_coefficient = robot_config.hardware.position_p_coefficient if robot_config.hardware else None
        now = datetime.now()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"sweep-{now.strftime('%Y%m%dT%H%M%S')}.json"
        record = {
            "config": str(self.args.config),
            "feedforward": self.args.feedforward,
            "p_coefficient": p_coefficient,
            "report": report,
            "suggested_feedforward": suggested_feedforward,
            "warnings": warnings,
            "sweeps": results,
        }
        write_text_atomically(path, json.dumps(record, indent=self.OUTPUT_INDENT))

        for warning in warnings:
            self.log(f"warning: {warning}")
        if self.args.feedforward:
            # residuals of the configured feedforward: pasted, they would replace it
            self.log(f"residual fit with the configured feedforward on, nothing to paste; record: {path}")
            return
        comments = [
            f"so101 friction-sweep, {now.isoformat(timespec='seconds')}, P {p_coefficient}; record: {path}",
            "under robot: in the station file, replacing feedforward if it is there:",
        ]
        snippet = StationFileSnippet.text(comments, {"feedforward": suggested_feedforward}, self.SNIPPET_INDENT)
        print(snippet, end="", flush=True)

    @classmethod
    def plan(cls, robot_config: SO101Config, kinematics: SO101Kinematics, poses: Sequence[str]) -> list[dict[str, Any]]:
        """Returns every sweep with its steps, checked against the limits and the table's clearance first."""
        import numpy as np

        from pear.deploy.station.control.robots.so101.layout import JOINTS

        limits = robot_config.limits
        lower_limits, upper_limits = np.array(limits.lower_bounds), np.array(limits.upper_bounds)

        sweeps = []
        for pose in poses:
            base = np.array(robot_config.home.position if cls.POSES[pose] is None else cls.POSES[pose], dtype=float)
            for joint_index in cls.SWEPT_JOINTS[pose]:
                sweep_start = base[joint_index] + cls.SWEEP_RANGE[joint_index][0]
                sweep_end = base[joint_index] + cls.SWEEP_RANGE[joint_index][1]
                if (
                    sweep_start < lower_limits[joint_index] + cls.LIMIT_MARGIN
                    or sweep_end > upper_limits[joint_index] - cls.LIMIT_MARGIN
                ):
                    raise ValueError(
                        f"{pose} {JOINTS[joint_index]} sweep [{sweep_start:.2f}, {sweep_end:.2f}] nears a limit"
                    )
                points = np.linspace(sweep_start, sweep_end, cls.INTERVALS_PER_DIRECTION + 1)
                for angle in points:
                    point_pose = base.copy()
                    point_pose[joint_index] = angle
                    tip_height_cm = kinematics.pose(point_pose)[2]  # z of the model's quantities
                    if tip_height_cm < cls.MIN_TIP_HEIGHT_CM:
                        raise ValueError(
                            f"{pose} {JOINTS[joint_index]} at {angle:.2f} puts the fingertip at {tip_height_cm:.1f} cm"
                        )
                sweeps.append({"pose": pose, "joint": joint_index, "base": base, "points": points})

        return sweeps

    def run_sweep(
        self, control_loop: JointControlLoop, gravity_model: GravityModel, sweep: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Steps through ``sweep`` up and back down and returns, for each step, its target, error and gravity
        torque."""
        from pear.deploy.station.control.robots.so101.layout import JOINTS

        joint_index, points, base = sweep["joint"], sweep["points"], sweep["base"]
        order = [(0, None)] + [(point_index, "up") for point_index in range(1, len(points))]
        order += [(point_index, "down") for point_index in range(len(points) - 2, -1, -1)]

        rows = []
        for point_index, direction in order:
            goal = base.copy()
            goal[joint_index] = points[point_index]
            measured_position = self.settle_and_measure(control_loop, goal)
            error = (measured_position - goal)[: len(JOINTS)]
            rows.append(
                {
                    "index": point_index,
                    "direction": direction,
                    "target": goal.tolist(),
                    "error": error.tolist(),
                    "gravity": gravity_model.torque(goal).tolist(),
                }
            )
            if direction:
                self.log(
                    f"  {sweep['pose']} {JOINTS[joint_index]} {direction:4s} q={points[point_index]:+.3f} "
                    f"error={error[joint_index]:+.4f} rad"
                )

        return rows

    def settle_and_measure(self, control_loop: JointControlLoop, goal: np.ndarray) -> np.ndarray:
        """Moves the arm to ``goal``, waits for it to settle and returns its measured position, averaged."""
        import numpy as np

        from pear.deploy.station.control.robots.so101.layout import ARM

        control_loop.move({ARM: tuple(goal)}, speed_scale=self.SPEED_SCALE)
        control_loop.wait_for_hold(control_loop.HOLD_WAIT_S)
        time.sleep(self.DWELL_S)

        samples, end_time = [], time.monotonic() + self.AVERAGING_WINDOW_S
        while time.monotonic() < end_time:
            samples.append(control_loop.snapshot()["arms"][ARM]["position"])
            time.sleep(control_loop.control_config.control_period_s)

        return np.mean(samples, axis=0)

    @staticmethod
    def log(message: str) -> None:
        """Prints one line of the sweep's progress."""
        print(message, file=sys.stderr, flush=True)
