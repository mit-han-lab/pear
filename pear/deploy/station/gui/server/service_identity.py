import hashlib
import json
import os
import pkgutil
from pathlib import Path
from typing import Any, Self

from omegaconf import DictConfig, OmegaConf

from pear.deploy.station.agent_loop.messages import SystemPrompt


class ServiceIdentity:
    """Who a page's service is, as its /health answer tells: the service, the version of the answer, the robot and the
    digest of the configuration; a running service of the same identity can be reused."""

    SERVICE_NAME = "pear-station-gui"  # the service's name in its /health answer, beside the robot it drives
    # of the /health answer, which the service writes and `pear-station gui` checks before reusing the service
    HEALTH_VERSION = 4
    HEALTH_PATH = "/health"  # where a service answers who it is

    def __init__(self, robot_name: str, configuration_id: str) -> None:
        self.robot_name = robot_name  # the name of the robot the service drives
        self.configuration_id = configuration_id  # the digest of what the service's behaviour depends on

    @classmethod
    def of_config(cls, config: DictConfig, run_dir: Path) -> Self:
        """Returns the identity of the service of a station's configuration that runs in ``run_dir``: its digest
        changes whenever a running service would behave differently from a new one."""
        robot_class = pkgutil.resolve_name(config.robot_class)
        identity_fields = {
            "config": OmegaConf.to_container(config, resolve=True),
            "prompts": [
                SystemPrompt.render("", action_format) for action_format in robot_class.ACTION_FORMATS.values()
            ],
            "run_dir": str(run_dir.resolve()),
        }

        return cls(robot_class.NAME, cls.json_digest(identity_fields))

    def health(self, instance_id: str) -> dict[str, Any]:
        """Returns the /health answer of the service of this identity in its run ``instance_id``."""
        return {
            "service": self.SERVICE_NAME,
            "robot": self.robot_name,
            "version": self.HEALTH_VERSION,
            "configuration_id": self.configuration_id,
            "pid": os.getpid(),
            "instance": instance_id,
        }

    def is_same(self, health: dict[str, Any]) -> bool:
        """Whether a running service's /health answer is the answer of a service of this identity."""
        return (
            health.get("service") == self.SERVICE_NAME
            and health.get("robot") == self.robot_name
            and health.get("version") == self.HEALTH_VERSION
            and bool(health.get("instance"))
            and health.get("configuration_id") == self.configuration_id
        )

    @staticmethod
    def json_digest(value: Any) -> str:
        """Returns the SHA-256 of a JSON value, the same for equal values whatever the order of their keys."""
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
