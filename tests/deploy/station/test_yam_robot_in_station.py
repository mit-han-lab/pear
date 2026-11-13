from collections.abc import Iterator

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.robot import ActionMode
from pear.deploy.station.control.robots.yam import robot as yam_robot
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.motion.actions import JointPositions
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from stand_in_joint_driver import StandInYAMDriver
from stand_in_robots import StandInYAMRobot
from yam_rigs import MotionHelpers

# a wide station: limits wide enough for every move of the tests, a velocity tolerance of 0.03 rad/s, Home at a third
# of the move caps with no caps of its own and 20 s for its arrival, and no cameras
WIDE_STATION = (
    *ConfigFiles.WIDE_LIMITS["yam"],
    "robot.velocity_tolerance=0.03",
    "robot.home.speed_scale=0.3333333333333333",
    "robot.home.limits=null",
    "robot.home.arrival_timeout_s=20",
    "robot.cameras=[]",
)
# what the tests of Home through the robot change of the tests' station: sequences may run late, the arms settle in
# a short window, also at Home, and Home's last wait is short
DISTANT_STATION = (
    "control.deadline_tolerance_s=0.5",
    "control.settle_window_s=0.1",
    "robot.home.settle_window_s=0.1",
    "robot.home.arrival_timeout_s=1.5",
)


class HomePoses:
    """The pose away from Home the arms of the tests start at."""

    @staticmethod
    def distant_pose() -> np.ndarray:
        """Returns both arms' joints away from Home: the bases turned, the grippers partly open."""
        q = np.zeros(14)
        q[[0, 7]] = [0.6, -0.6]
        q[[6, 13]] = [0.4, 0.7]
        return q


class StandInDistantYAMDriver(StandInYAMDriver):
    """YAM's stand-in joints that start away from Home."""

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        super().__init__(control_config, robot_config)
        q = HomePoses.distant_pose()
        self.positions = {s: tuple(q[i * 7 : (i + 1) * 7]) for i, s in enumerate(("left", "right"))}


class StandInDistantYAMRobot(StandInYAMRobot):
    """The stand-in YAM robot, its arms starting away from Home."""

    DRIVER_CLASS = StandInDistantYAMDriver


@pytest.fixture
def distant_station() -> Iterator[StandInDistantYAMRobot]:
    """The open stand-in YAM robot, its arms away from Home, with the short settle windows of a whole Home's tests;
    settled at the start and closed at the end."""
    config = ConfigFiles.load("yam", *DISTANT_STATION)

    with StandInDistantYAMRobot(config) as station:
        station.wait_stationary()
        yield station


def test_entire_chunk_validated_before_first_action():
    with StandInYAMRobot(ConfigFiles.load("yam", *WIDE_STATION), action_mode=ActionMode.JOINT) as station:
        q = station.observe()["joint_positions"]
        count = len(station.control_loop.driver.sent_targets)
        bad = q.copy()
        bad[-1] = np.nan

        result = station.execute({"type": "joint_positions", "positions": [q.tolist(), bad.tolist()], "rate_hz": 30.0})

        assert result.status == "rejected"
        assert len(station.control_loop.driver.sent_targets) == count
        assert station.control_loop.status == "HOLD"


def test_cartesian_planner_failure_never_opens_motors(monkeypatch):
    config = ConfigFiles.load("yam")
    monkeypatch.setattr(StandInYAMRobot, "DRIVER_CLASS", lambda *args: pytest.fail("Opened motors"))
    monkeypatch.setattr(StandInYAMRobot.CONTROL_LOOP_CLASS, "open", lambda *args: pytest.fail("Opened motors"))

    def fail(*args, **kwargs):
        raise TimeoutError("stand-in planner startup failure")

    monkeypatch.setattr(yam_robot, "PlannerProcess", fail)

    with pytest.raises(TimeoutError, match="planner startup"):
        StandInYAMRobot(config).open()


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1"])
def test_a_malformed_action_fails_validation(bad):
    # Model instances can be mutated; Station.execute revalidates every action once,
    # so the planner and IK only ever receive validated actions.
    invalid = MotionHelpers.action(side="left").model_copy(update={"delta_position_cm": [bad, 0, 0]})
    with pytest.raises(ValueError):
        YAMRobot.ACTION_ADAPTER.validate(invalid)
    missing = MotionHelpers.action().model_dump()
    del missing["delta_position_cm"]
    with pytest.raises(ValueError):
        YAMRobot.ACTION_ADAPTER.validate(missing)


def test_home_rejects_non_boolean_gripper_choice(distant_station):
    with pytest.raises(ValueError, match="open_grippers"):
        distant_station.home(opens_grippers="no")


def test_an_unknown_option_is_refused_before_the_arms_are_touched(distant_station):
    observation = distant_station.observe()

    with pytest.raises(TypeError, match="speed"):
        distant_station.home("left", speed=0.5)

    assert distant_station.hold_snapshot(observation)["status"] == "HOLD"


def test_joint_mode_home_lazily_initializes_planner_and_preserves_joint_contract():
    config = ConfigFiles.load("yam", "control.deadline_tolerance_s=0.5")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        assert station.planner is None
        result = station.home("left")
        assert result["results"]["left"]["reached"] and station.planner is not None
        assert station.kinematics is None  # Home needs only joint trajectory planning.
        observation = station.observe()
        result = station.execute(
            JointPositions(positions=[observation["joint_positions"].tolist()], rate_hz=30.0),
            reference_observation=observation,
        )
        assert result.status == "completed"
        assert station.kinematics is None


def test_joint_home_does_not_require_cartesian_geometry():
    config = ConfigFiles.load("yam", *WIDE_STATION, "robot.geometry=null")

    with StandInYAMRobot(config, action_mode=ActionMode.JOINT) as station:
        result = station.home()
        assert all(value["reached"] for value in result["results"].values())
        assert station.kinematics is None
        assert station.status()["status"] == "READY"


def test_cartesian_mode_requires_the_station_geometry():
    config = ConfigFiles.load("yam", *WIDE_STATION, "robot.geometry=null")

    with pytest.raises(ValueError, match="Cartesian motion requires station geometry"):
        StandInYAMRobot(config, action_mode=ActionMode.CARTESIAN).open()
