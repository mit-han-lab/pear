import ast
import inspect
import pkgutil
import textwrap
from pathlib import Path

import pytest

from pear.deploy.station.cli import StationCommand
from pear.deploy.station.control.robot import BaseRobot
from repository import REPO, Repository

STATION = REPO / "pear" / "deploy" / "station"  # the station's package
STATION_MODULE = "pear.deploy.station"
ROBOTS = STATION / "control" / "robots"  # the robots' packages, each by its robot's name
ROBOTS_MODULE = f"{STATION_MODULE}.control.robots"
ROBOT_COMMANDS = dict(StationCommand.robot_commands())  # each robot package's maintenance commands, by its name
STATION_USERS = ("agent_loop", "gui")  # the packages that reach a robot only through the contract of its base class
ROBOT_NAME = "robot"  # what the station's users call the robot whose members they read
SERVER = STATION / "gui" / "server"  # the page's service: its launcher and its web server
SERVER_MODULE = f"{STATION_MODULE}.gui.server"
LAUNCHER_SOURCE = SERVER / "launcher.py"  # the launcher of `pear-station gui`, which the web server imports
# the launcher's own modules, beside it in the service's package; every other module there is the web server's
LAUNCHER_MODULES = ("launcher", "browser_token", "service_identity")


class Sources:
    """The Python files of each layer of the station, and what they contain."""

    @staticmethod
    def generic() -> list[str]:
        """The files of the layer that must not know any robot, relative to the package: every file outside a robot's
        package."""
        robot_packages = Repository.packages(ROBOTS)
        paths = [
            path for path in Sources.python(STATION) if not any(package in path.parents for package in robot_packages)
        ]

        return sorted(str(path.relative_to(STATION)) for path in paths)

    @staticmethod
    def station_users() -> list[str]:
        """The files of the packages that use a robot, relative to the package."""
        paths = [path for package in STATION_USERS for path in Sources.python(STATION / package)]

        return sorted(str(path.relative_to(STATION)) for path in paths)

    @staticmethod
    def python(directory: Path) -> list[Path]:
        """Every Python file under a directory; none when the directory does not exist."""
        return list(directory.rglob("*.py")) if directory.is_dir() else []

    @staticmethod
    def modules(tree: ast.AST) -> list[tuple[int, str]]:
        """The (line, module) of every import in a file, at any depth."""
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.append((node.lineno, node.module))
            elif isinstance(node, ast.Import):
                found.extend((node.lineno, alias.name) for alias in node.names)
        return found

    @staticmethod
    def robot_members(tree: ast.AST) -> list[tuple[int, str]]:
        """The (line, member) of every attribute read from something named ROBOT_NAME."""
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                owner = node.value
                name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
                if name == ROBOT_NAME:
                    found.append((node.lineno, node.attr))
        return found

    @staticmethod
    def strings(tree: ast.AST) -> list[tuple[int, str]]:
        """The (line, value) of every string constant in a file."""
        return [
            (node.lineno, node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]


class RobotWords:
    """The words only a robot package may use: robot, camera, arm and format names."""

    @staticmethod
    def all() -> set[str]:
        """Every robot-specific name of every installed robot."""
        words = set(ROBOT_COMMANDS)
        for robot in (pkgutil.resolve_name(commands.ROBOT_CLASS_PATH) for commands in ROBOT_COMMANDS.values()):
            words |= set(robot.ROBOT_CONFIG_CLASS.LAYOUT.arms)
            words |= {camera.role for camera in robot.CAMERAS} | {camera.name for camera in robot.CAMERAS}
            for action_format in robot.ACTION_FORMATS.values():
                words.add(action_format.name)
                words |= {camera.role for camera in action_format.cameras}
                words |= {camera.name for camera in action_format.cameras}
        return words


class RobotContract:
    """The public members of the robot's base class: what the station's users may read of a robot."""

    @staticmethod
    def members() -> set[str]:
        """The public members of BaseRobot: its methods and class attributes, and the attributes its constructor
        sets."""
        constructor = ast.parse(textwrap.dedent(inspect.getsource(BaseRobot.__init__)))
        instance_attributes = {
            node.attr
            for node in ast.walk(constructor)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
        }

        return {
            member
            for member in (*dir(BaseRobot), *BaseRobot.__annotations__, *instance_attributes)
            if not member.startswith("_")
        }


@pytest.mark.parametrize("name", Sources.generic())
def test_generic_layer_has_no_robot_literals(name):
    tree = ast.parse((STATION / name).read_text())
    robots = [f"{ROBOTS_MODULE}.{robot}" for robot in ROBOT_COMMANDS]
    words = RobotWords.all()

    literals = [(line, value) for line, value in Sources.strings(tree) if value in words]
    imports = [(line, module) for line, module in Sources.modules(tree) if module.startswith(tuple(robots))]
    references = [(line, value) for line, value in Sources.strings(tree) if value.startswith(tuple(robots))]

    assert not literals + imports + references, f"{name} names a robot: {literals + imports + references}"


@pytest.mark.parametrize("name", Sources.station_users())
def test_the_agent_loop_and_the_console_use_only_the_robot_s_contract(name):
    contract = RobotContract.members()

    members = Sources.robot_members(ast.parse((STATION / name).read_text()))

    outside = [(line, member) for line, member in members if member not in contract]
    assert not outside, f"{name} uses robot members outside the contract: {outside}"


def test_launcher_imports_nothing_of_the_web_server():
    tree = ast.parse(LAUNCHER_SOURCE.read_text())
    imported = [module for _, module in Sources.modules(tree)]
    own_modules = tuple(f"{SERVER_MODULE}.{module}" for module in LAUNCHER_MODULES)

    assert not [
        module for module in imported if module.startswith(f"{SERVER_MODULE}.") and not module.startswith(own_modules)
    ]
