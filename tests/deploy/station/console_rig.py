import threading
from pathlib import Path
from typing import Any

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.schemas import Camera, CameraFrame, RobotStatus
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.phase import Phase
from stand_in_cameras import StandInCameras
from stand_in_clients import StandInDemonstrationClient
from stand_in_robots import StandInRobots
from waiting import Waiting


class ConsoleRig:
    """A console of a robot on stand-in hardware, cameras and model, its control held by the tests' tab and its
    recording off: as a context it gives the console, and at the end disables the motors and closes it."""

    CLIENT_ID = "test-client"  # the tests' browser tab, which holds control
    HEARTBEAT_S = 0.2  # between two heartbeats of the tests' tab
    JOIN_S = 2.0  # for the heartbeat thread to end
    WAIT_S = 15.0  # for the console to reach what a step of the rig waits for
    POLL_S = 0.01  # between two looks at it
    TASKS_FOLDER = "tasks"  # in the rig's directory, as in a service's run directory: the tasks' folders
    UPLOADS_FOLDER = "tasks-uploads"  # beside it: the draft images the browsers upload
    MODEL = "stand_in"  # the name of the stand-in model in the station's configuration
    MODEL_DELAY_S = 0.06  # how long the stand-in model thinks before each answer
    # the consoles of the tests: control timings relaxed for a loaded host and short settle windows; the stand-in model,
    # which answers the demonstration action that holds still, each after a short think
    OVERRIDES = (
        "control.state_timeout_s=0.5",
        "control.deadline_tolerance_s=0.5",
        "control.settle_window_s=0.05",
        "robot.home.settle_window_s=0.05",
        f"default_model={MODEL}",
        f"models.{MODEL}.label=Stand-in model",
        f"models.{MODEL}.client_class=stand_in_clients.StandInDemonstrationClient",
        f"models.{MODEL}.timeout_s=30",
        f"models.{MODEL}.client.delay_s={MODEL_DELAY_S}",
        f"models.{MODEL}.client.is_moving=false",
    )

    def __init__(self, robot_name: str, directory: Path, *overrides: str) -> None:
        self.robot_name = robot_name
        self.directory = directory  # the tasks' folders and the draft images go under it
        self.overrides = overrides  # the test's own changes to the configuration, read over the rig's
        self.heartbeat_exit = threading.Event()  # set to end the heartbeats
        self.heartbeat_thread = threading.Thread(target=self._beat, name="test-heartbeat", daemon=True)
        self.console: Console | None = None  # None until the context is entered

    def __enter__(self) -> Console:
        # the stand-in model answers in the robot's first format, which the model names as its own
        action_format = next(iter(StandInRobots.robot_class(self.robot_name).ACTION_FORMATS))
        config = ConfigFiles.load(
            self.robot_name,
            *self.OVERRIDES,
            f"models.{self.MODEL}.action_format={action_format}",
            f"models.{self.MODEL}.client.action_format={action_format}",
            *self.overrides,
        )
        console = self.console = Console(
            config, root=self.directory / self.TASKS_FOLDER, uploads_dir=self.directory / self.UPLOADS_FOLDER
        )

        try:
            console.commands.run("recording", {"enabled": False})
            console.lease.claim(self.CLIENT_ID)
            self.heartbeat_thread.start()
        except BaseException:
            console.close()
            raise

        return console

    def __exit__(self, *_: object) -> None:
        console = self.console

        try:
            self.disable(console)
            console.close()
        finally:
            self.heartbeat_exit.set()
            if self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(self.JOIN_S)

    @classmethod
    def enable(cls, console: Console) -> None:
        """Enables the motors of ``console`` and waits until the arms hold, settled."""
        console.commands.run("enable", None)

        Waiting.until(
            lambda: not console.is_busy and console.robot_status() == RobotStatus.READY, cls.WAIT_S, cls.POLL_S
        )

    @classmethod
    def disable(cls, console: Console) -> None:
        """Disables the motors of ``console``, if they are enabled, and waits until it is idle."""
        if console.robot_status() != RobotStatus.OFF:
            console.commands.run("disable", {"confirm": True})
            Waiting.until(lambda: not console.is_busy, cls.WAIT_S, cls.POLL_S)

    @classmethod
    def pause_after_a_step(cls, console: Console) -> None:
        """Runs a task until one step completes, then pauses it."""
        console.commands.run("start", {"instruction": "Lift"})
        Waiting.until(lambda: console.task_run.total_steps >= 1, cls.WAIT_S, cls.POLL_S)
        console.commands.run("pause", None)
        Waiting.until(lambda: not console.is_busy and console.phase == Phase.PAUSED, cls.WAIT_S, cls.POLL_S)

    def _beat(self) -> None:
        """Renews the tests' tab's control until the rig ends: the heartbeat thread's body."""
        while not self.heartbeat_exit.wait(self.HEARTBEAT_S):
            self.console.lease.heartbeat(self.CLIENT_ID)


class StandInModel:
    """The clients a test makes a console use in place of its configured model, by patching the console's
    new_client: each a stand-in demonstration client in the format of the model it is made for, or one of the test's
    class, with the test's arguments."""

    def __init__(
        self,
        console: Console,
        client_class: type[StandInDemonstrationClient] = StandInDemonstrationClient,
        **arguments: Any,
    ) -> None:
        self.console = console
        self.client_class = client_class
        self.arguments = arguments  # the client's arguments beside its prompt, its timeout and its format

    def __call__(self, model: str, instruction: str) -> ModelClient:
        """Returns a new client of ``model`` for the task ``instruction``, as Console.new_client does."""
        models_config = self.console.models_config
        action_format = models_config.action_format(model)

        return self.client_class(
            SystemPrompt.render(instruction, action_format),
            timeout_s=models_config.models[model].timeout_s,
            action_format=action_format.name,
            **self.arguments,
        )


class MarkedCameras(StandInCameras):
    """Stand-in cameras whose frames all carry one frame number, and which note the threads that read them."""

    FRAME_NUMBER = 999_999  # the frame numbers of stand-in cameras stay far below it during a test

    def __init__(self, cameras: tuple[Camera, ...]) -> None:
        super().__init__(cameras)
        self.preview_threads: set[str] = set()  # the names of the threads that read the cameras

    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns a fresh frame of every camera, each numbered FRAME_NUMBER, and notes the thread that reads them."""
        self.preview_threads.add(threading.current_thread().name)
        frames, errors = super().read_preview()

        return {role: {**frame, "frame_number": self.FRAME_NUMBER} for role, frame in frames.items()}, errors
