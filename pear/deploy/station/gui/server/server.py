import asyncio
import signal
from collections.abc import Callable

from aiohttp import web
from omegaconf import DictConfig

from pear.deploy.station.control.schemas import RobotStatus
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server.app import ServerApp
from pear.deploy.station.gui.server.browser_token import BrowserToken
from pear.deploy.station.gui.server.launcher import ServiceConfig
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from pear.utils.file_lock import FileLock
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class GUIServer:
    """The page's service in this process: the run directory's owner lock, the console, and the web server on this
    machine's loopback address until a stop signal finds the motors off."""

    SHUTDOWN_TIMEOUT_S = 2  # for the open connections to end once the service stops
    OWNER_LOCK_NAME = "owner.lock"  # in the run directory: held by the one service that runs in it
    TASKS_FOLDER = "tasks"  # in the run directory: the folders of the tasks
    UPLOADS_FOLDER = "tasks-uploads"  # in the run directory: the draft images the browsers upload

    def __init__(self, config: DictConfig, on_listening: Callable[[str], None]) -> None:
        self.config = config  # the station's configuration, which the console is made from
        self.service_config = ServiceConfig.from_station_config(config)
        self.on_listening = on_listening  # takes the login URL once the service accepts connections

    def run(self) -> None:
        """Serves until a stop signal arrives with the motors off."""
        run_dir = self.service_config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        owner_lock = self.take_owner_lock()
        token = BrowserToken(run_dir).create()
        console = Console(self.config, root=run_dir / self.TASKS_FOLDER, uploads_dir=run_dir / self.UPLOADS_FOLDER)
        identity = ServiceIdentity.of_config(self.config, run_dir)

        asyncio.run(self.serve(console, token, identity))
        owner_lock.close()

    def take_owner_lock(self) -> FileLock:
        """Returns the run directory's owner lock, and refuses to run beside the service that holds it."""
        try:
            return FileLock(self.service_config.run_dir / self.OWNER_LOCK_NAME)
        except BlockingIOError:
            raise SystemExit("A GUI owner is already running; reuse it.") from None

    async def serve(self, console: Console, token: str, identity: ServiceIdentity) -> None:
        """Serves the console on the port, as the service of ``identity``, until a stop signal finds the motors off,
        then closes it."""
        app = ServerApp(console, token, identity).build()
        runner = web.AppRunner(app, shutdown_timeout=self.SHUTDOWN_TIMEOUT_S)
        await runner.setup()
        await web.TCPSite(runner, ServiceConfig.LOOPBACK, self.service_config.port).start()
        exit_requests: asyncio.Queue[int] = asyncio.Queue()  # one for each stop signal

        loop = asyncio.get_running_loop()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signal_number, exit_requests.put_nowait, signal_number)
        # the log never holds the token: the login URL goes to on_listening alone
        listening_url = f"{self.service_config.base_url}/"
        logger.info("%s Control listening on %s; motors disabled", console.robot.TITLE, listening_url)
        self.on_listening(self.service_config.login_url(token))

        try:
            while True:
                await exit_requests.get()
                if await asyncio.to_thread(self.stop_for_exit, console):
                    break
        finally:
            await runner.cleanup()

    def stop_for_exit(self, console: Console) -> bool:
        """Stops the robot for an exit; returns whether its motors are off, else asks the operator to disable them."""
        console.stop(reason="Service exit requested")
        if console.robot_status() != RobotStatus.OFF:
            console.error = "Exit deferred. Disable Motors in the browser before shutdown."
            return False

        return True
