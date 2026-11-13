from __future__ import annotations

import argparse
import importlib
import json
import math
import pkgutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only: the station's packages are imported where a command runs
    from omegaconf import DictConfig

    from pear.deploy.station.control.cameras import CameraSource
    from pear.deploy.station.control.joint.layout import JointLayout
    from pear.deploy.station.control.robot import ActionMode, BaseRobot
    from pear.deploy.station.control.schemas import ControlSnapshot, DetailedStatus


class StationFileSnippet:
    """Lines a command prints for the operator to put into the station's own file: values under the key the comments
    name, indented as they sit there."""

    @staticmethod
    def text(comments: Sequence[str], values: Mapping[str, Any], indent: int) -> str:
        """Returns ``comments`` as comment lines, then ``values`` as YAML, each list on one line, indented by
        ``indent`` spaces."""
        import yaml

        body = yaml.safe_dump(dict(values), default_flow_style=None, sort_keys=False)
        comment_lines = "".join(f"# {comment}\n" for comment in comments)

        return comment_lines + "".join(f"{' ' * indent}{line}\n" for line in body.splitlines())

    @staticmethod
    def clamped(values: Sequence[float], at_least: float, at_most: float) -> tuple[list[float], list[int]]:
        """Returns ``values`` clipped into [``at_least``, ``at_most``] and the indices of those that were clipped."""
        clipped = [min(max(float(value), at_least), at_most) for value in values]

        return clipped, [index for index, value in enumerate(values) if clipped[index] != value]


class RobotCommand(ABC):
    """The base of one maintenance command of a robot: its word on the command line, its own arguments and what it
    does."""

    NAME: str  # the word of the command line that chooses the command
    HELP: str  # the command's line in the robot's help, as the operator reads it
    MOVES_ROBOT = False  # whether the command moves the robot: it declares --yes, without which it is refused
    HAS_DRY_RUN = False  # whether a command that moves the robot shows, without --yes, what it would do
    READS_CONFIG = True  # whether the command reads the station's configuration: it then takes --config and overrides

    def __init__(self, args: argparse.Namespace, robot_class_path: str) -> None:
        self.args = args  # the parsed command line
        self.robot_class_path = robot_class_path  # the class of the robot whose command this is

    @classmethod  # noqa: B027  (a default, not abstract)
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the command's own arguments on its parser, --yes among them if it moves the robot (by default
        none)."""

    @abstractmethod
    def run(self) -> Any:
        """Carries the command out and returns its result, which is printed as JSON (None: nothing is printed)."""

    def load_config(self, extra_overrides: Sequence[str] = ()) -> DictConfig:
        """Returns the station's configuration that the command line names, with the command's own
        ``extra_overrides`` read last, and refuses another robot's."""
        from pear.deploy.station.control.robot import BaseRobot  # with the station's packages: where a command runs

        config = BaseRobot.load_config(self.args.config, [*self.args.overrides, *extra_overrides])
        robot_class = pkgutil.resolve_name(config.robot_class)
        own_robot_class = pkgutil.resolve_name(self.robot_class_path)
        if not issubclass(robot_class, own_robot_class):
            raise ValueError(f"{self.args.config} configures {robot_class.TITLE}, not {own_robot_class.TITLE}")

        return config

    def new_robot(
        self, action_mode: ActionMode, cameras: CameraSource | None = None, extra_overrides: Sequence[str] = ()
    ) -> BaseRobot:
        """Returns the robot of the station's configuration with the command's own ``extra_overrides``, closed, which
        takes the actions of ``action_mode`` and observes ``cameras``."""
        config = self.load_config(extra_overrides)

        return pkgutil.resolve_name(config.robot_class)(config, cameras=cameras, action_mode=action_mode)


class RobotCommands:
    """The base of a robot's maintenance commands, which the station's command line offers under the robot's name."""

    ROBOT_CLASS_PATH: str  # the robot's class by its path: its module is imported where a command needs the robot
    HELP: str  # the robot's line in the station's help, as the operator reads it
    # the refusal of a command that moves the robot when it runs without --yes, formatted with the command's word
    CONFIRMATION_REFUSAL: str
    COMMAND_CLASSES: tuple[type[RobotCommand], ...]  # the robot's commands, in the order the help lists them
    JSON_INDENT = 2  # of a printed result

    def __init__(self, command: RobotCommand) -> None:
        self.command = command

    @classmethod
    def in_module(cls, module_name: str) -> type[RobotCommands]:
        """Returns the one class built on this base that the module ``module_name`` defines: the commands of the
        robot whose package the module is in."""
        module = importlib.import_module(module_name)
        [commands_class] = [
            member
            for member in vars(module).values()
            if isinstance(member, type) and issubclass(member, cls) and member.__module__ == module_name
        ]

        return commands_class

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the robot's commands on ``parser``, the parser of the robot's name on the station's command
        line."""
        command_parsers = parser.add_subparsers(dest="command", required=True)

        for command_class in cls.COMMAND_CLASSES:
            command_parser = command_parsers.add_parser(command_class.NAME, help=command_class.HELP)
            command_parser.set_defaults(command_class=command_class)
            if command_class.READS_CONFIG:
                command_parser.add_argument(
                    "--config", type=Path, required=True, help="the station's own configuration file"
                )
            command_class.add_arguments(command_parser)
            # after the command's own arguments: every word they leave is an override
            if command_class.READS_CONFIG:
                command_parser.add_argument(
                    "overrides", nargs="*", metavar="key=value", help="a value read over the file's"
                )

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> RobotCommands:
        """Returns the robot's commands with the command that the parsed command line ``args`` chose; a command that
        moves the robot is refused without --yes, unless it has a dry run."""
        command_class = args.command_class
        if command_class.MOVES_ROBOT and not command_class.HAS_DRY_RUN and not args.yes:
            raise ValueError(cls.CONFIRMATION_REFUSAL.format(command=command_class.NAME))

        return cls(command_class(args, cls.ROBOT_CLASS_PATH))

    def run(self) -> None:
        """Runs the command and prints its result as JSON."""
        result = self.command.run()
        if result is not None:
            self.print_result(result)

    @classmethod
    def print_result(cls, result: Any) -> None:
        """Prints the result of a command as indented JSON."""
        print(json.dumps(result, indent=cls.JSON_INDENT))


class Smoke(RobotCommand):
    """The base of a robot's smoke command, which enables the robot, moves one joint a small distance and back, and
    releases it; a robot's command declares the joint it moves and --yes around the distance."""

    NAME = "smoke"
    MOVES_ROBOT = True
    MAX_DELTA_RAD: float  # the farthest a robot lets the joint move
    DEFAULT_DELTA_RAD: float  # how far the joint moves when the command line does not say
    # what the operator reads before the robot moves, formatted with the parsed command line: the distance and the
    # robot's own options
    NOTICE: str
    REPORTS_WHILE_ENABLED = False  # whether the report is printed before the robot is released
    HOLD_TIMEOUT_S = 10.0  # for each of the two moves to end in a settled hold

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the distance of the move."""
        parser.add_argument("--delta", type=float, default=cls.DEFAULT_DELTA_RAD)

    def run(self) -> dict[str, Any] | None:
        """Moves the joint by the distance and back, each time to a settled hold, and returns what it reports of the
        snapshots before the move, after it and after the return; None once it has printed them itself."""
        from pear.deploy.station.control.robot import ActionMode  # with the station's packages

        # the configuration is read and checked first: a wrong one is reported before a wrong distance
        robot = self.new_robot(ActionMode.JOINT)
        delta = self.args.delta
        if not math.isfinite(delta) or not 0 < abs(delta) <= self.MAX_DELTA_RAD:
            raise ValueError(f"Smoke-test delta must be nonzero and at most {self.MAX_DELTA_RAD} rad")
        arm, coordinate_index = self.moved_joint(robot.robot_config.LAYOUT)
        print(self.NOTICE.format(**vars(self.args)), file=sys.stderr, flush=True)

        with robot:
            control_loop = robot.control_loop
            snapshot_before = self.initial_snapshot(robot)
            targets = {arm_name: list(arm_targets) for arm_name, arm_targets in snapshot_before["targets"].items()}
            targets[arm][coordinate_index] += delta
            control_loop.move(targets)
            snapshot_after_move = control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)
            control_loop.move(snapshot_before["targets"])
            snapshot_after_return = control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)

            report = {
                "before": self.summary(snapshot_before),
                "moved": self.summary(snapshot_after_move),
                "returned": self.summary(snapshot_after_return),
            }
            if self.REPORTS_WHILE_ENABLED:
                RobotCommands.print_result(report)
                return None

        return report

    @abstractmethod
    def moved_joint(self, layout: JointLayout) -> tuple[str, int]:
        """Returns the arm and the index of the coordinate that the command moves, as its command line chose them."""

    def initial_snapshot(self, robot: BaseRobot) -> ControlSnapshot | DetailedStatus:
        """Returns the snapshot the move starts from (by default that of the settled hold the command waits for)."""
        return robot.control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)

    def summary(self, snapshot: ControlSnapshot | DetailedStatus) -> Any:
        """Returns what the command reports of ``snapshot`` (by default the whole of it)."""
        return snapshot


class AccuracyCheck(RobotCommand):
    """The base of a robot's accuracy check, which takes one arm through fixed poses from Home, first without its
    gravity and friction feedforward and then with the station's, and holds the tool's stop error with the
    feedforward to its limits; a robot's command declares the arm, the poses, its kinematics and its feedforward."""

    NAME = "accuracy-check"
    HELP = "MOVES the arm: stop accuracy without and with the feedforward"
    MOVES_ROBOT = True
    HAS_DRY_RUN = True  # without --yes it checks the poses, says what it would do and moves nothing
    MEDIAN_LIMIT_MM = 3.0  # of the tool's stop error over the poses, with the feedforward
    MAX_LIMIT_MM = 6.0  # of the tool's stop error at any pose, with the feedforward
    FADE_WAIT_S = 0.5  # after the settled hold: the friction feedforward latches 0.2 s, then fades over 0.3 s
    AVERAGING_WINDOW_S = 0.2  # over which the measured position is averaged
    HOLD_TIMEOUT_S = 30.0  # for each move to end in a settled hold
    MIN_LOADED_TRAVEL_RAD = 0.3  # the least travel from Home of a loaded joint at each pose: friction must show
    SECONDS_PER_POSE = 10  # in the estimate of the duration
    OUTPUT_FOLDER = "accuracy"  # of the run directory, where the records go
    PASSES = ("uncompensated", "compensated")  # the feedforward of each pass: the package's defaults, the station's
    DRY_RUN_TEXT = "Dry plan only; add --yes to move."
    REPORT_DECIMALS = 2  # of the errors in mm
    JOINT_DECIMALS = 4  # of the joint errors in rad
    FAILURE_HINT: str  # what the operator checks after a failure
    LOADED_JOINTS: tuple[int, ...]  # the arm joints that carry most of the gravity, by index

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the confirmation that the arm may move."""
        parser.add_argument("--yes", action="store_true", help="Confirm motion on the real arm")

    @classmethod
    def verdict(cls, errors_mm: Sequence[float]) -> dict[str, Any]:
        """Returns the median and the worst of ``errors_mm`` and whether both are within their limits."""
        import numpy as np

        median_mm, max_mm = float(np.median(errors_mm)), float(max(errors_mm))

        return {
            "median_mm": median_mm,
            "max_mm": max_mm,
            "passed": median_mm <= cls.MEDIAN_LIMIT_MM and max_mm <= cls.MAX_LIMIT_MM,
        }

    def run(self) -> None:
        """Checks the poses and, with --yes, measures the arm's stop error at each without the feedforward and then
        with it, keeps a record, prints both and passes or fails the arm; refuses an arm whose feedforward is still
        the package's."""
        from datetime import datetime

        from pear.deploy.station.control.robot import ActionMode  # with the station's packages

        robots = {
            "uncompensated": None,  # filled below, once the arm is known
            "compensated": self.new_robot(ActionMode.JOINT),
        }
        robot_config = robots["compensated"].robot_config
        arm = self.checked_arm(robot_config.LAYOUT)
        robots["uncompensated"] = self.new_robot(ActionMode.JOINT, extra_overrides=self.uncompensated_overrides(arm))
        compensation = {name: self.compensation(robot.robot_config, arm) for name, robot in robots.items()}
        if compensation["compensated"] == compensation["uncompensated"]:
            raise ValueError(
                f"{arm} has no feedforward of its own in {self.args.config}: run friction-sweep and paste the lines it "
                "prints first"
            )

        poses = self.poses(robot_config)
        for joints in poses.values():
            self.check_pose(robot_config, arm, [*joints, 1.0])
        self.log(
            f"{arm}: {len(poses)} poses ({', '.join(poses)}), each from Home, without the feedforward and then with "
            f"it; about {len(poses) * len(self.PASSES) * self.SECONDS_PER_POSE // 60 + 1} min."
        )
        if not self.args.yes:
            self.log(self.DRY_RUN_TEXT)
            return

        passes = {}
        for name in self.PASSES:
            self.log(f"{name} pass")
            rows = self.measure(robots[name], arm, poses)
            passes[name] = {
                "compensation": compensation[name],
                "poses": rows,
                **self.verdict([row["error_mm"] for row in rows.values()]),
            }
        verdict = self.verdict([row["error_mm"] for row in passes["compensated"]["poses"].values()])

        output_dir = Path(self.load_config().run_dir) / self.OUTPUT_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"accuracy-{arm}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        record = {
            "config": str(self.args.config),
            "checked_arm": arm,
            "limits": {"median_mm": self.MEDIAN_LIMIT_MM, "max_mm": self.MAX_LIMIT_MM},
            "passes": passes,
            "passed": verdict["passed"],
        }
        path.write_text(json.dumps(record, indent=1))

        self.print_report(passes, path)
        if not verdict["passed"]:
            worst_pose = max(passes["compensated"]["poses"].items(), key=lambda item: item[1]["error_mm"])[0]
            raise RuntimeError(
                f"accuracy check failed: median {verdict['median_mm']:.1f} mm (limit {self.MEDIAN_LIMIT_MM}), worst "
                f"{verdict['max_mm']:.1f} mm at {worst_pose} (limit {self.MAX_LIMIT_MM}). {self.FAILURE_HINT}"
            )
        print("PASS")

    def measure(self, robot: BaseRobot, arm: str, poses: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        """Enables ``robot``, goes Home, takes ``arm`` from Home to each of ``poses`` and back, and returns each pose's
        target, measured coordinates, tool error in mm and joint errors; parks the arm before releasing it, unless
        the operator interrupts."""
        import time

        import numpy as np

        rows = {}
        with robot:
            control_loop = robot.control_loop
            start = control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)["targets"]
            joint_count = robot.robot_config.LAYOUT.joint_count
            is_interrupted = False
            try:
                robot.home()
                home = {
                    name: tuple(targets)
                    for name, targets in control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)["targets"].items()
                }
                for pose_name, joints in poses.items():
                    targets = dict(home)
                    targets[arm] = (*joints, *home[arm][joint_count:])
                    control_loop.move(targets)
                    control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)
                    time.sleep(self.FADE_WAIT_S)
                    measured = self.averaged_position(control_loop, arm)
                    target = np.asarray(targets[arm], dtype=float)
                    error = self.tool_point_mm(robot.robot_config, arm, measured) - self.tool_point_mm(
                        robot.robot_config, arm, target
                    )
                    rows[pose_name] = {
                        "target": target.tolist(),
                        "measured": measured.tolist(),
                        "error_mm": round(float(np.linalg.norm(error)), self.REPORT_DECIMALS),
                        "joint_error": np.round(measured - target, self.JOINT_DECIMALS)[:joint_count].tolist(),
                    }
                    self.log(f"  {pose_name}: {rows[pose_name]['error_mm']} mm")
                    control_loop.move(home)
                    control_loop.wait_for_hold(self.HOLD_TIMEOUT_S)
            except KeyboardInterrupt:
                is_interrupted = True
                robot.stop()
                raise
            finally:
                if not is_interrupted:
                    self.park(robot, start)

        return rows

    def averaged_position(self, control_loop: Any, arm: str) -> Any:
        """Returns the coordinates of ``arm`` averaged over AVERAGING_WINDOW_S."""
        import time

        import numpy as np

        samples, end_time = [], time.monotonic() + self.AVERAGING_WINDOW_S
        while not samples or time.monotonic() < end_time:
            samples.append(control_loop.snapshot()["arms"][arm]["position"])
            time.sleep(control_loop.control_config.control_period_s)

        return np.mean(np.asarray(samples, dtype=float), axis=0)

    def print_report(self, passes: dict[str, dict[str, Any]], path: Path) -> None:
        """Prints each pose's tool error without and with the feedforward, then the medians and the worst."""
        before, after = passes["uncompensated"], passes["compensated"]
        width = max(len(name) for name in after["poses"])
        print(f"{'pose':<{width}}  without (mm)  with (mm)")
        for name, row in after["poses"].items():
            print(f"{name:<{width}}  {before['poses'][name]['error_mm']:12.2f}  {row['error_mm']:9.2f}")
        worst_pose = max(after["poses"].items(), key=lambda item: item[1]["error_mm"])[0]
        print(
            f"median {before['median_mm']:.2f} -> {after['median_mm']:.2f} mm (limit {self.MEDIAN_LIMIT_MM}), "
            f"worst {after['max_mm']:.2f} mm at {worst_pose} (limit {self.MAX_LIMIT_MM}); record: {path}"
        )

    @staticmethod
    def log(message: str) -> None:
        """Prints one line of the check's progress."""
        print(message, file=sys.stderr, flush=True)

    @abstractmethod
    def checked_arm(self, layout: JointLayout) -> str:
        """Returns the arm the command checks, as the command line chose it."""

    @abstractmethod
    def poses(self, robot_config: Any) -> dict[str, list[float]]:
        """Returns the checked arm's poses by name: its joints, rad, without the gripper."""

    @abstractmethod
    def home_joints(self, robot_config: Any, arm: str) -> list[float]:
        """Returns the joints of ``arm`` at Home, rad."""

    @abstractmethod
    def tool_point_mm(self, robot_config: Any, arm: str, coordinates: Sequence[float]) -> Any:
        """Returns the tool's point of ``arm`` at its ``coordinates`` (joints and gripper), mm."""

    @abstractmethod
    def check_pose(self, robot_config: Any, arm: str, coordinates: Sequence[float]) -> None:
        """Raises ValueError unless ``arm`` may go to ``coordinates``: inside its bounds and clear of the table."""

    @abstractmethod
    def compensation(self, robot_config: Any, arm: str) -> dict[str, list[float]]:
        """Returns the gravity and friction feedforward of ``arm`` in ``robot_config``."""

    @abstractmethod
    def uncompensated_overrides(self, arm: str) -> list[str]:
        """Returns the overrides that give ``arm`` the package's feedforward, which is none of its own."""

    @abstractmethod
    def park(self, robot: BaseRobot, start: dict[str, Any]) -> None:
        """Returns the robot to where it can be released, from Home or a fault; ``start`` holds the targets it was
        enabled at."""
