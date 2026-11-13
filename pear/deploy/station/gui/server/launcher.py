import json
import os
import pkgutil
import subprocess
import time
from dataclasses import dataclass, fields
from http.client import HTTPException
from pathlib import Path
from typing import Any, Self
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from omegaconf import MISSING, DictConfig, OmegaConf

from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.control.robot import ActionMode, BaseRobot
from pear.deploy.station.gui.server.browser_token import BrowserToken
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


@dataclass
class ServiceConfig:
    """Where the page's service of a station listens and keeps its files: its part of the station's configuration."""

    LOOPBACK = "127.0.0.1"  # the only address the service listens on

    port: int = MISSING  # of the service, on this machine's loopback address
    run_dir: Path = MISSING  # the service's log, lock and token, and the tasks' records

    @classmethod
    def from_station_config(cls, config: DictConfig) -> Self:
        """Returns the service's part of a station's configuration, as BaseRobot.load_config reads it."""
        node = OmegaConf.masked_copy(config, [config_field.name for config_field in fields(cls)])

        return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), node))

    @property
    def base_url(self) -> str:
        """Where the service answers."""
        return f"http://{self.LOOPBACK}:{self.port}"

    def login_url(self, token: str) -> str:
        """Returns the URL that logs a browser in to the service with ``token``."""
        return f"{self.base_url}/#token={token}"


class GUILauncher:
    """Starts the page's service of a station in the background, or finds the one already running on its port."""

    HEALTH_TIMEOUT_S = 2  # for the answer to one /health request
    STARTUP_POLLS = 60  # /health requests within which a started service must answer
    STARTUP_POLL_S = 0.25  # between two of them
    LOG_NAME = "service.log"  # in the run directory: what the service in the background writes to stderr
    LOG_MODE = 0o600  # owner-only, like the browser token beside it
    # the refusal of a service on the port that is not this station's
    FOREIGN_SERVICE_MESSAGE = (
        "Port contains a different or outdated service. Disable its motors and stop it explicitly before restarting."
    )

    def __init__(self, config: DictConfig) -> None:
        self.config = config  # the station's configuration, which the service is made from
        self.service_config = ServiceConfig.from_station_config(config)
        self.identity = ServiceIdentity.of_config(config, self.service_config.run_dir)
        self.process: subprocess.Popen[bytes] | None = None  # the service this launcher started; None before

    @classmethod
    def from_station_config(cls, config_path: Path, overrides: list[str]) -> Self:
        """Returns the launcher of the station whose own file is ``config_path``, read with the ``key=value``
        ``overrides``; what the service would refuse of it is refused here, before any service starts."""
        config = BaseRobot.load_config(config_path, overrides)
        # the models and the robot's settings, as the service reads them; in joint mode the robot loads no planning
        ModelsConfig.from_station_config(config)
        pkgutil.resolve_name(config.robot_class)(config, action_mode=ActionMode.JOINT)

        return cls(config)

    def start_or_reuse(self, command: list[str]) -> str:
        """Returns the login URL of a matching service on the port, else of the service ``command`` starts in the
        background."""
        login_url = self.reuse()
        if login_url is None:
            self.start(command)
            self.validate(self.wait_for_health())
            login_url = self.service_config.login_url(BrowserToken(self.service_config.run_dir).read())

        return login_url

    def reuse(self) -> str | None:
        """Returns the login URL of a matching service on the port, None when nothing listens there."""
        health = self.read_health()
        if health is None:
            return None

        self.validate(health)
        return self.service_config.login_url(BrowserToken(self.service_config.run_dir).read())

    def start(self, command: list[str]) -> None:
        """Launches ``command``, the service in the foreground, in a session of its own, its stderr the run directory's
        log."""
        run_dir = self.service_config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(run_dir / self.LOG_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND, self.LOG_MODE)

        with os.fdopen(descriptor, "ab") as log_file:
            # the log takes the service's log records; the login URL it prints is left out
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log_file, start_new_session=True
            )

    def wait_for_health(self) -> dict[str, Any]:
        """Polls until the started service answers, and refuses one that exits or takes too long."""
        log_path = self.service_config.run_dir / self.LOG_NAME

        for _ in range(self.STARTUP_POLLS):
            health = self.read_health()
            if health:
                return health
            if self.process.poll() is not None:
                raise RuntimeError(f"GUI startup failed; inspect {log_path}")
            time.sleep(self.STARTUP_POLL_S)

        raise RuntimeError(f"GUI startup timed out; inspect {log_path}")

    def read_health(self) -> dict[str, Any] | None:
        """Returns the /health answer of the service on the port, None when nothing listens there."""
        health_url = self.service_config.base_url + ServiceIdentity.HEALTH_PATH

        try:
            with urlopen(health_url, timeout=self.HEALTH_TIMEOUT_S) as response:
                health = json.load(response)
        except (HTTPError, HTTPException, ValueError):
            # an error page, another protocol or a body that is no JSON: some other program holds the port
            logger.warning("The port's /health answer is no station's", exc_info=True)
            raise RuntimeError(self.FOREIGN_SERVICE_MESSAGE) from None
        except (URLError, TimeoutError) as error:
            if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
                return None
            logger.warning("The port did not answer /health", exc_info=True)
            raise RuntimeError("Local port cannot be verified; no service was started") from error

        if not isinstance(health, dict):
            raise RuntimeError(self.FOREIGN_SERVICE_MESSAGE)
        return health

    def validate(self, health: dict[str, Any]) -> None:
        """Refuses a service that is outdated, drives another robot or runs another configuration."""
        if not self.identity.is_same(health):
            raise RuntimeError(self.FOREIGN_SERVICE_MESSAGE)
