import argparse
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Self

from pear.deploy.station.control.commands import RobotCommands
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class GUICommand:
    """Starts the page's service of the station a configuration names, or reuses the one already running on its port:
    in this process until a stop signal, or in the background with --detach."""

    NAME = "gui"  # the word of the command line that chooses the command
    MODULE = "pear.deploy.station.cli"  # the module of the command line, which python -m runs
    CONFIG_OPTION = "--config"  # the option that names the station's own file

    def __init__(self, config_path: Path, overrides: list[str], is_detached: bool) -> None:
        self.config_path = config_path  # the station's own configuration file
        self.overrides = overrides  # key=value, each read over the file's
        self.is_detached = is_detached  # whether the service runs in the background, the command printing its URL

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Declares the station's file, --detach and the overrides."""
        parser.add_argument(cls.CONFIG_OPTION, type=Path, required=True, help="the station's own configuration file")
        parser.add_argument(
            "--detach", action="store_true", help="runs the service in the background and prints its login URL"
        )
        parser.add_argument("overrides", nargs="*", metavar="key=value", help="a value read over the file's")

    def run(self) -> None:
        """Prints the login URL of a matching service, which --detach starts in the background; without one, serves in
        this process."""
        from pear.deploy.station.gui.server.launcher import GUILauncher  # with the station's packages

        launcher = GUILauncher.from_station_config(self.config_path, self.overrides)
        if self.is_detached:
            print(launcher.start_or_reuse(self.foreground_command()))
            return

        login_url = launcher.reuse()
        if login_url is not None:
            print(login_url)
            return

        from pear.deploy.station.gui.server.server import GUIServer  # the console: only where the service runs

        GUIServer(launcher.config, lambda listening_url: print(listening_url, flush=True)).run()

    def foreground_command(self) -> list[str]:
        """Returns the command the service in the background runs: this one, in the foreground."""
        command_line = [self.NAME, self.CONFIG_OPTION, str(self.config_path), *self.overrides]

        return [sys.executable, "-u", "-m", self.MODULE, *command_line]


class StationCommand:
    """The `pear-station` command: the page's service of a station, and the maintenance commands of each robot."""

    ROBOTS_PACKAGE = "pear.deploy.station.control.robots"  # whose packages are the robots, each by its name
    COMMANDS_MODULE = "commands.commands"  # of a robot's package: its maintenance commands
    # a command's failures that the operator acts on: told in one line; any other is told with its traceback alone
    COMMAND_ERRORS = (ValueError, RuntimeError, OSError, ImportError)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args  # the parsed command line

    @classmethod
    def from_command_line(cls) -> Self:
        """Returns the command the command line gives."""
        parser = argparse.ArgumentParser(prog="pear-station", description=cls.__doc__)
        subcommands = parser.add_subparsers(dest="subcommand", required=True)
        gui_parser = subcommands.add_parser(GUICommand.NAME, help=GUICommand.__doc__)
        GUICommand.add_arguments(gui_parser)
        for robot_name, commands_class in cls.robot_commands():
            robot_parser = subcommands.add_parser(robot_name, help=commands_class.HELP)
            commands_class.add_arguments(robot_parser)
            robot_parser.set_defaults(commands_class=commands_class)

        args, unread_arguments = parser.parse_known_args()
        # argparse fills the overrides where it first meets a positional: a key=value after an option is left unread
        if hasattr(args, "overrides") and all("=" in word and not word.startswith("-") for word in unread_arguments):
            args.overrides += unread_arguments
        elif unread_arguments:
            parser.error(f"unrecognized arguments: {' '.join(unread_arguments)}")

        return cls(args)

    def run(self) -> None:
        """Runs the subcommand the command line chose; a failure the operator acts on ends it with status 1 and one
        line, after its traceback in the log."""
        try:
            self.subcommand().run()
        except KeyboardInterrupt:
            # an interrupted maintenance command ends with status 0 and no traceback; the page's service does not
            if self.args.subcommand == GUICommand.NAME:
                raise
        except self.COMMAND_ERRORS as error:
            self.fail(error)
        except Exception as error:
            # a file that is no YAML, or a configuration OmegaConf refuses: a command that read one loaded both
            import yaml
            from omegaconf.errors import OmegaConfBaseException

            if not isinstance(error, (yaml.YAMLError, OmegaConfBaseException)):
                raise
            self.fail(error)

    @classmethod
    def robot_commands(cls) -> list[tuple[str, type[RobotCommands]]]:
        """Returns the maintenance commands of each robot's package, by the robot's name, in the order of the names."""
        robots_package = importlib.import_module(cls.ROBOTS_PACKAGE)
        robot_names = sorted(module.name for module in pkgutil.iter_modules(robots_package.__path__) if module.ispkg)

        return [
            (robot_name, RobotCommands.in_module(f"{cls.ROBOTS_PACKAGE}.{robot_name}.{cls.COMMANDS_MODULE}"))
            for robot_name in robot_names
        ]

    def subcommand(self) -> GUICommand | RobotCommands:
        """Returns the subcommand the command line chose: the page's service, or one of a robot's commands, which
        refuses a command that moves the robot without --yes."""
        args = self.args
        if args.subcommand == GUICommand.NAME:
            return GUICommand(args.config, args.overrides, args.detach)

        return args.commands_class.from_arguments(args)

    @staticmethod
    def fail(error: Exception) -> None:
        """Ends the command with status 1, ``error`` told in one line after its traceback in the log."""
        logger.exception("pear-station failed")
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    StationCommand.from_command_line().run()


if __name__ == "__main__":
    main()
