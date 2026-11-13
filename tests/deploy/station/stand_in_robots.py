import pkgutil

from config_files import ConfigFiles
from control_rig import SteppedControlLoop
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.joint.control_loop import SO101ControlLoop
from pear.deploy.station.control.robots.so101.robot import SO101Robot
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.robot import YAMRobot
from stand_in_cameras import StandInCameras
from stand_in_joint_driver import StandInSO101Driver, StandInYAMDriver


class StandInYAMCameras(StandInCameras):
    """The YAM robot's cameras, drawn, made as the robot makes its camera source."""

    def __init__(self, robot_config: YAMConfig) -> None:
        super().__init__(YAMRobot.CAMERAS)


class StandInYAMRobot(YAMRobot):
    """The YAM robot on stand-in joints and drawn cameras, with its own control loop, kinematics and planner."""

    DRIVER_CLASS = StandInYAMDriver
    CAMERA_SOURCE_CLASS = StandInYAMCameras


class SteppedYAMControlLoop(SteppedControlLoop, YAMControlLoop):
    """The YAM robot's control loop without its thread."""


class SteppedStandInYAMRobot(StandInYAMRobot):
    """The stand-in YAM robot whose control loop the test steps."""

    CONTROL_LOOP_CLASS = SteppedYAMControlLoop


class StandInSO101Cameras(StandInCameras):
    """The SO-101's camera, drawn, made as the robot makes its camera source."""

    def __init__(self, robot_config: SO101Config) -> None:
        super().__init__(SO101Robot.CAMERAS)


class StandInSO101Robot(SO101Robot):
    """The SO-101 on its own driver over the stand-in arm and a drawn camera, with its own control loop and
    kinematics."""

    DRIVER_CLASS = StandInSO101Driver
    CAMERA_SOURCE_CLASS = StandInSO101Cameras


class SteppedSO101ControlLoop(SteppedControlLoop, SO101ControlLoop):
    """The SO-101's control loop without its thread."""


class SteppedStandInSO101Robot(StandInSO101Robot):
    """The stand-in SO-101 whose control loop the test steps."""

    CONTROL_LOOP_CLASS = SteppedSO101ControlLoop


class StandInRobots:
    """How a test makes a robot of either kind on stand-in hardware, observing drawn cameras, with the settle windows
    of the tests that act through a robot."""

    # what those tests change of each robot's station of the tests: YAM's arms settle in a short window, and its
    # sequences of targets may run late
    OVERRIDES = {"yam": ("control.settle_window_s=0.05", "control.deadline_tolerance_s=0.5"), "so101": ()}

    @staticmethod
    def robot_class(robot_name: str) -> type[BaseRobot]:
        """Returns the class of the robot ``robot_name`` that its station of the tests names."""
        return pkgutil.resolve_name(ConfigFiles.load(robot_name).robot_class)

    @classmethod
    def make(cls, robot_name: str) -> BaseRobot:
        """Returns the robot ``robot_name`` on stand-in hardware, not open yet, whose observations read drawn
        cameras."""
        config = ConfigFiles.load(robot_name, *cls.OVERRIDES[robot_name])
        robot_class = pkgutil.resolve_name(config.robot_class)

        return robot_class(config, cameras=StandInCameras(robot_class.CAMERAS))
