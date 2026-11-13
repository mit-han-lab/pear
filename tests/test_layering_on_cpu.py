import ast
import re

import pytest

from repository import REPO, Repository

PEAR = REPO / "pear"
FAMILY_ROOTS = (PEAR / "envs", PEAR / "models")  # the packages of the env families and of the model families
SOURCE_SUFFIXES = (".py", ".yaml", ".md")
STATION = PEAR / "deploy" / "station"  # the robot station
STATION_MODULE = "pear.deploy.station"
# the station's parts from the lowest to the highest, a module in the longest that names it: it imports its own part
# and the parts before it
STATION_ORDER = (
    "exceptions",
    "control.schemas",
    "control.cameras",
    "control.joint",
    "control.motion",
    "control.robot",
    "control.commands",
    "control.robots",
    "agent_loop.clients",
    "agent_loop",
    "records",
    "records.recording",
    "records.video",
    "gui.console",
    "gui.server",
    "cli",
)
ROBOT_PART = "control.robots"  # the robots' packages, none of which imports another's
ROBOT_ROOT = STATION.joinpath(*ROBOT_PART.split("."))
CLIENT_PART = "agent_loop.clients"  # the model clients: text and images in, an answer out
ROBOT_SIDE = "control"  # what a client imports nothing of, though the order has it before the clients
# a word of a line: YAM, the YAM of YAMRobot, the so101 of so101_arm; yaml is one word, and no yam
WORD = re.compile(r"[A-Z0-9]+(?![a-z])|[A-Za-z][a-z0-9]*")


class StationImports:
    """What the robot station's modules import of the station itself, by the parts of STATION_ORDER."""

    @staticmethod
    def read() -> list[tuple[str, str, str]]:
        """Returns (file:line, importing module, imported name) for every import of the station's own in its files."""
        found = []
        for path in Repository.files(f"{STATION.relative_to(REPO)}/**/*.py"):
            file = path.relative_to(REPO)
            module = ".".join(file.with_suffix("").parts).removesuffix(".__init__")

            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level:
                    names = ["." * node.level + (node.module or "")]  # relative, as it is written: no part holds it
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue

                found += [
                    (f"{file}:{node.lineno}", module, name) for name in names if name.startswith((STATION_MODULE, "."))
                ]

        return found

    @classmethod
    def rank(cls, name: str) -> int | None:
        """Returns the place in STATION_ORDER of the part a module or an imported name is in, None when none holds
        it."""
        parts = [part for part in STATION_ORDER if cls.is_within(name, part)]

        return STATION_ORDER.index(max(parts, key=len)) if parts else None

    @staticmethod
    def is_within(name: str, part: str) -> bool:
        """Whether a module or an imported name is the station's ``part`` or below it."""
        return f"{name}.".startswith(f"{STATION_MODULE}.{part}.")


@pytest.fixture(scope="module")
def station_imports() -> list[tuple[str, str, str]]:
    return StationImports.read()


def test_the_generic_runtime_names_no_env_family_no_model_family_and_no_robot():
    families = [family for root in FAMILY_ROOTS for family in Repository.packages(root)]
    robots = Repository.packages(ROBOT_ROOT)
    # a family by the letters its package's name begins with: libero for libero_plus too, qwen for qwen3_5
    names = {re.match("[a-z]+", family.name).group() for family in families}
    # a robot by its package's name as a word
    robot_names = {robot.name for robot in robots}
    # everything of pear/ outside the families' and the robots' packages
    generic_files = [
        path
        for path in Repository.files("pear/**/*")
        if path.suffix in SOURCE_SUFFIXES and not any(parent in [*families, *robots] for parent in path.parents)
    ]

    naming_lines = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()}"
        for path in generic_files
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if any(name in line.lower() for name in names) or robot_names & {word.lower() for word in WORD.findall(line)}
    ]

    assert names and robot_names and generic_files
    assert not naming_lines


def test_a_module_of_the_station_imports_its_own_part_and_the_parts_before_it(station_imports):
    ranked_imports = [
        (where, StationImports.rank(module), StationImports.rank(name), name) for where, module, name in station_imports
    ]

    unranked_imports = [
        f"{where}: {name}" for where, own_rank, rank, name in ranked_imports if None in (own_rank, rank)
    ]
    upward_imports = [
        f"{where}: {name}"
        for where, own_rank, rank, name in ranked_imports
        if None not in (own_rank, rank) and rank > own_rank
    ]

    assert station_imports
    # the order holds every module of the station that imports and every name one imports of the station
    assert not unranked_imports
    assert not upward_imports


def test_a_model_client_imports_nothing_of_the_robot_side(station_imports):
    robot_side_imports = [
        f"{where}: {name}"
        for where, module, name in station_imports
        if StationImports.is_within(module, CLIENT_PART) and StationImports.is_within(name, ROBOT_SIDE)
    ]

    assert not robot_side_imports


def test_a_robot_s_package_imports_no_other_robot_s(station_imports):
    robots = [f"{ROBOT_PART}.{robot.name}" for robot in Repository.packages(ROBOT_ROOT)]

    foreign_imports = [
        f"{where}: {name}"
        for where, module, name in station_imports
        for robot in robots
        for other_robot in robots
        if robot != other_robot
        and StationImports.is_within(module, robot)
        and StationImports.is_within(name, other_robot)
    ]

    assert len(robots) > 1
    assert not foreign_imports
