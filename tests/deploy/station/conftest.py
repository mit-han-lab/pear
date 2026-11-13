from __future__ import annotations

import importlib
import os
import pkgutil
import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from config_files import ConfigFiles
from first_line import FirstLine
from repository import REPO

# annotations only: pytest loads this file whenever it collects this directory, so the station's packages, and the
# helpers that import them, load where a fixture asks for them
if TYPE_CHECKING:
    from omegaconf import DictConfig
    from playwright.sync_api import Browser, BrowserContext, Page

    from control_rig import StandInClock, Ticker
    from pear.deploy.station.control.joint.config import ControlConfig
    from pear.deploy.station.control.joint.control_loop import JointControlLoop
    from pear.deploy.station.control.joint.driver import JointDriver
    from pear.deploy.station.control.robot import BaseRobot
    from pear.deploy.station.control.robots.yam.config import YAMConfig
    from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver
    from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
    from pear.deploy.station.control.schemas import ActionFormat
    from pear.deploy.station.gui.console.console import Console
    from pear.deploy.station.gui.server.launcher import GUILauncher
    from pear.deploy.station.gui.server.service_identity import ServiceIdentity
    from yam_rigs import GripperRigs

# the calls that open a device, by the module and the attribute a test would reach them through
DEVICE_OPENERS = (
    ("can", "Bus"),
    ("can.interface", "Bus"),
    ("pyrealsense2", "pipeline"),
    ("serial", "Serial"),  # the SO-101 Feetech bus
    ("cv2", "VideoCapture"),  # the SO-101 wrist camera
)
STAND_IN_MODULE_PREFIX = "stand_in_"  # of the modules of the tests' stand-ins: hardware, cameras, models, robots


@pytest.fixture(params=ConfigFiles.ROBOT_NAMES)
def robot_name(request: pytest.FixtureRequest) -> str:
    """The name of a robot the tests run on: a test that asks for it runs once for each robot, unless it is
    parametrized with the names it runs on."""
    return request.param


@pytest.fixture
def action_format(robot_name: str) -> ActionFormat:
    """The default action format of the robot ``robot_name``."""
    from stand_in_robots import StandInRobots

    return next(iter(StandInRobots.robot_class(robot_name).ACTION_FORMATS.values()))


@pytest.fixture
def idle_action(action_format: ActionFormat) -> dict[str, Any]:
    """An action in the robot's default format that holds still."""
    from stand_in_clients import DemonstrationActions

    return DemonstrationActions.action(action_format.name, call_count=0, is_moving=False)


@pytest.fixture
def robot(robot_name: str) -> Iterator[BaseRobot]:
    """The open robot ``robot_name`` on stand-in hardware, observing drawn cameras; closed at the end."""
    from stand_in_robots import StandInRobots

    with StandInRobots.make(robot_name) as robot:
        yield robot


@pytest.fixture
def so101_config() -> DictConfig:
    """The SO-101's station of the tests, with the short settle windows and the fast caps of the tests of its arm."""
    return ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES)


@pytest.fixture
def so101_robot(so101_config: DictConfig) -> Iterator[BaseRobot]:
    """The open SO-101 of ``so101_config`` on stand-in hardware, without cameras; closed at the end."""
    from stand_in_robots import StandInSO101Robot

    with StandInSO101Robot(so101_config) as robot:
        yield robot


@pytest.fixture(scope="module")
def yam_kinematics() -> YAMKinematics:
    """The YAM arms' kinematics of their station of the tests, loaded once for each test module."""
    from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics

    return YAMKinematics(ConfigFiles.load_configs("yam")[1])


@pytest.fixture
def stand_in_clock(monkeypatch: pytest.MonkeyPatch) -> StandInClock:
    """The clock every module reads and sleeps on during the test: it stands still until the test advances it."""
    from control_rig import StandInClock

    return StandInClock().install(monkeypatch)


@pytest.fixture
def yam_rig(stand_in_clock: StandInClock) -> Iterator[tuple[JointControlLoop, JointDriver, Ticker]]:
    """A YAM control loop with wide limits, on stand-in joints and the stand-in clock, stepped by hand: the loop, its
    driver and its ticker; closed at the end."""
    from control_rig import ControlRig, Robots
    from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
    from stand_in_joint_driver import StandInYAMDriver

    with ControlRig(YAMControlLoop, StandInYAMDriver, *Robots.load("yam"), stand_in_clock) as rig:
        yield rig


@pytest.fixture
def gripper_rigs(stand_in_clock: StandInClock) -> Iterator[GripperRigs]:
    """What makes the gripper rigs of a test, on its stand-in clock; every rig is closed at the end."""
    from yam_rigs import GripperRigs

    with GripperRigs(stand_in_clock) as rigs:
        yield rigs


@pytest.fixture
def console(robot_name: str, tmp_path: Path) -> Iterator[Console]:
    """A console of the robot ``robot_name`` on stand-in hardware, cameras and model, its control held by the tests'
    tab and its recording off; its motors disabled and the console closed at the end."""
    from console_rig import ConsoleRig

    with ConsoleRig(robot_name, tmp_path) as console:
        yield console


@pytest.fixture
def enabled_console(console: Console) -> Console:
    """``console`` with its robot's motors enabled, holding and settled."""
    from console_rig import ConsoleRig

    ConsoleRig.enable(console)

    return console


class PhysicalDevices:
    """What keeps a test from the station's devices: each call that would open one fails the test instead."""

    FAILURE = "Offline test attempted to open a physical device"  # why the test failed

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch  # undoes every replacement at the end of the test

    def block(self) -> None:
        """Puts a failure in the place of each device's opener, of the YAM driver and of the start of a page's service
        in the background."""
        self.block_openers()
        self.guard_yam_driver()
        self.guard_service_start()

    def block_openers(self) -> None:
        """Replaces the libraries' calls that open a device; a library that is not installed opens nothing."""
        for module_name, attribute in DEVICE_OPENERS:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            # also where a library's stand-in lacks the call, which then cannot be reached either
            self.monkeypatch.setattr(module, attribute, self.fail_on_device_open, raising=False)

    def guard_yam_driver(self) -> None:
        """Lets the YAM driver be made only from settings the product refuses before the SDK's process starts."""
        # the station's packages: this file loads in every environment that collects the directory, so they are
        # imported where a test runs
        from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver

        real_constructor = YAMDriver.__init__

        def guarded_constructor(driver: YAMDriver, control_config: ControlConfig, robot_config: YAMConfig) -> None:
            # settings the product refuses still reach its own check, which raises before any device
            if robot_config.motion_blockers():
                return real_constructor(driver, control_config, robot_config)
            self.fail_on_device_open()

        self.monkeypatch.setattr(YAMDriver, "__init__", guarded_constructor)

    def guard_service_start(self) -> None:
        """Lets a launcher start a service in the background only for a robot on stand-in hardware and cameras."""
        from pear.deploy.station.gui.server.launcher import GUILauncher

        real_start = GUILauncher.start

        def guarded_start(launcher: GUILauncher, command: list[str]) -> None:
            # a service in the background runs in its own process, out of reach of the guards above
            if not self.is_stand_in_robot(launcher.config.robot_class):
                self.fail_on_device_open()
            return real_start(launcher, command)

        self.monkeypatch.setattr(GUILauncher, "start", guarded_start)

    @staticmethod
    def is_stand_in_robot(robot_class_path: str) -> bool:
        """Whether the robot of ``robot_class_path`` drives stand-in joints and draws its cameras."""
        robot_class = pkgutil.resolve_name(robot_class_path)
        hardware_classes = (robot_class.DRIVER_CLASS, robot_class.CAMERA_SOURCE_CLASS)

        return all(hardware_class.__module__.startswith(STAND_IN_MODULE_PREFIX) for hardware_class in hardware_classes)

    @classmethod
    def fail_on_device_open(cls, *args: object, **kwargs: object) -> NoReturn:
        """Fails the test that called it."""
        pytest.fail(cls.FAILURE)


@pytest.fixture(autouse=True)
def block_physical_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary tests cannot open physical CAN, serial, RealSense, USB cameras or motor drivers."""
    PhysicalDevices(monkeypatch).block()


class BrowserStation:
    """The page's service a module of browser tests drives: the tests' station of one robot, with the module's models,
    started as `pear-station gui` starts it in the foreground, in a process of its own on a free port of this machine
    and a run directory of the module's, and stopped at the end."""

    # the station's robot, on stand-in hardware and cameras: the tests read the page of its three cameras
    ROBOT_NAME = "yam"
    START_TIMEOUT_S = 60.0  # for the service to print its login URL: it imports the station and opens the cameras
    STOP_TIMEOUT_S = 30.0  # for the service to end after its stop signal: it closes the console and the robot
    KILL_TIMEOUT_S = 10.0  # for the process group of a service that did not end to end once killed
    LOG_NAME = "service.log"  # in the run directory: what the service writes to stderr, its log records among it
    VIEWPORT = {"width": 1440, "height": 1100}  # of every test's browser window
    # the station's one model unless a module gives its own, and its default: the stand-in that narrates and answers
    # the demonstration action of an action format
    STAND_IN_MODEL = (
        "default_model=stand_in",
        "models.stand_in.label=Stand-in model",
        "models.stand_in.client_class=stand_in_clients.StandInDemonstrationClient",
    )

    def __init__(self, run_dir: Path, models: tuple[str, ...]) -> None:
        from pear.deploy.station.gui.server.launcher import ServiceConfig
        from pear.deploy.station.gui.server.service_identity import ServiceIdentity
        from pear.utils.net_utils import free_port

        self.run_dir = run_dir
        self.overrides = [f"port={free_port(0)}", f"run_dir={run_dir}", *models]  # read over the station's file
        config = ConfigFiles.load(self.ROBOT_NAME, *self.overrides)
        # a service in a process of its own is out of reach of the device guards: its robot must be a stand-in's
        if not PhysicalDevices.is_stand_in_robot(config.robot_class):
            PhysicalDevices.fail_on_device_open()
        self.identity: ServiceIdentity = ServiceIdentity.of_config(config, run_dir)  # what its /health answers
        self.base_url = ServiceConfig.from_station_config(config).base_url  # what the tests' paths are relative to
        self.token = ""  # the browser token a login gives, which the service writes as it starts
        self.process: subprocess.Popen[bytes] | None = None  # the service's process, from the start to the end

    def __enter__(self) -> BrowserStation:
        from pear.deploy.station.cli import GUICommand
        from pear.deploy.station.gui.server.browser_token import BrowserToken

        station_file = ConfigFiles.station_file(self.ROBOT_NAME)
        command = GUICommand(station_file, self.overrides, is_detached=False).foreground_command()
        # the service imports the stand-in robot and model that the station's file and the overrides name by class path
        python_path = (str(REPO), str(ConfigFiles.DIRECTORY), os.environ.get("PYTHONPATH"))
        log_path = self.run_dir / self.LOG_NAME

        with log_path.open("wb") as log_file:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, python_path))},
                start_new_session=True,
            )

        # the one line the service prints, its login URL, once it listens
        try:
            FirstLine.of(self.process, self.START_TIMEOUT_S, self.base_url)
        except pytest.fail.Exception:
            pytest.fail(f"the page's service did not start; its log:\n{log_path.read_text()}")
        self.token = BrowserToken(self.run_dir).read()

        return self

    def __exit__(self, *exception: object) -> None:
        # a stop signal ends the service once its motors are off; one whose motors a failed test left on keeps serving
        self.process.send_signal(signal.SIGTERM)

        try:
            self.process.wait(timeout=self.STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=self.KILL_TIMEOUT_S)
            raise AssertionError(
                f"the page's service did not end within {self.STOP_TIMEOUT_S} s of its stop signal (a test left its"
                f" motors on, or its exit hangs) and was killed; its log: {self.run_dir / self.LOG_NAME}"
            ) from None
        finally:
            self.process.stdout.close()


@pytest.fixture(scope="module")
def station_models() -> tuple[str, ...]:
    """The overrides that give the browser tests' station its models: the stand-in model alone, answering in the
    action format of the station's robot; a module of real models gives its own."""
    from stand_in_robots import StandInRobots

    action_format = next(iter(StandInRobots.robot_class(BrowserStation.ROBOT_NAME).ACTION_FORMATS))

    return (*BrowserStation.STAND_IN_MODEL, f"models.stand_in.client.action_format={action_format}")


@pytest.fixture(scope="module")
def browser_station(
    station_models: tuple[str, ...], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[BrowserStation]:
    """The page's service of a module of browser tests, one for each module, with ``station_models``; stopped at the
    end."""
    with BrowserStation(tmp_path_factory.mktemp("station"), station_models) as station:
        yield station


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """A module's headless Google Chrome, as the SOURCE's page tests ran (its codecs play the recordings' H.264, which
    Playwright's Chromium cannot); Playwright's event loop ends with the module, so a later test of the same process
    can run its own."""
    from playwright.sync_api import sync_playwright

    # a Chrome where Playwright does not look for one (installed without root), else the machine's own
    executable = os.environ.get("PEAR_STATION_CHROME")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            **({"executable_path": executable} if executable else {"channel": "chrome"})
        )
        yield browser
        browser.close()


@pytest.fixture
def context(browser: Browser, browser_station: BrowserStation) -> Iterator[BrowserContext]:
    """A test's own browser context on the module's service: its address, which every path of the tests is relative
    to, and the page's window."""
    context = browser.new_context(base_url=browser_station.base_url, viewport=BrowserStation.VIEWPORT)
    yield context
    context.close()


@pytest.fixture
def page(context: BrowserContext) -> Page:
    """The test's page in its browser context."""
    return context.new_page()


@pytest.fixture
def station_login(context: BrowserContext, browser_station: BrowserStation) -> None:
    """Logs the test's browser context in to the module's service, once its /health answer shows that it is that
    service: the tests' station, whose robot is on stand-in hardware."""
    from pear.deploy.station.gui.server.security import Security
    from pear.deploy.station.gui.server.service_identity import ServiceIdentity

    assert browser_station.identity.is_same(context.request.get(ServiceIdentity.HEALTH_PATH).json())
    assert context.request.post(Security.OPEN_PATH, data={"token": browser_station.token}).ok
