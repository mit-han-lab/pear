import pkgutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

from omegaconf import MISSING, DictConfig, OmegaConf

from pear.deploy.station.agent_loop.clients.client import ModelClient
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.schemas import ActionFormat


@dataclass
class OfferedModel:
    """One model an operator may choose: what the page calls it, its client, the action format it answers in and how
    long it may take to answer a turn."""

    MAX_TIMEOUT_S = 1800  # the longest a model may be given to answer a turn

    label: str = MISSING  # what the page calls the model
    client_class: str = MISSING  # the model's client: the class path of a ModelClient
    client: dict[str, Any] = MISSING  # the client's arguments beside the system prompt
    action_format: str | None = None  # the robot's action format the model answers in; None: the robot's first
    timeout_s: float = 300  # how long the model may take to answer a turn

    def check(self, name: str) -> None:
        """Raises unless the model, which the station calls ``name``, has a label, a timeout within bounds and a
        client's class."""
        if not self.label:
            raise ValueError(f"Model {name}: its label is empty")
        if not 0 < self.timeout_s <= self.MAX_TIMEOUT_S:
            raise ValueError(f"Model {name}: timeout_s must be above 0 and at most {self.MAX_TIMEOUT_S}")
        client_type = self.client_type
        if not (isinstance(client_type, type) and issubclass(client_type, ModelClient)):
            raise ValueError(f"Model {name}: client_class names {self.client_class}, which is no model client's class")

    def resolve_paths(self, config_dir: Path) -> None:
        """Makes each path among the client's arguments absolute, a relative one relative to ``config_dir``."""
        for argument in self.client_type.RELATIVE_PATH_ARGUMENTS:
            if self.client.get(argument):
                path = Path(self.client[argument]).expanduser()
                self.client[argument] = str(path if path.is_absolute() else (config_dir / path).resolve())

    def new_client(self, system_prompt: str) -> ModelClient:
        """Returns a new client of the model, prompted with ``system_prompt`` and given the model's timeout."""
        return self.client_type(system_prompt, timeout_s=self.timeout_s, **self.client)

    @property
    def client_type(self) -> type[ModelClient]:
        """The class of the model's client."""
        return pkgutil.resolve_name(self.client_class)


@dataclass
class ModelsConfig:
    """The models of a station's configuration: those an operator chooses from, in the action formats of its robot,
    and the one a task starts with."""

    robot_class: str = MISSING  # the station's robot, whose action formats the models answer in
    config_dir: str = MISSING  # the directory of the station's own file, which a client's relative path is relative to
    default_model: str = MISSING  # the model a task starts with, a key of models
    models: dict[str, OfferedModel] = MISSING  # the models an operator chooses from, under the station's own names

    def __post_init__(self) -> None:
        for name, offered_model in self.models.items():
            offered_model.check(name)
            offered_model.resolve_paths(Path(self.config_dir))
            try:
                # a client checks its own arguments as it is made; neither making nor closing it does any I/O
                offered_model.new_client("").close()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Model {name}: {error}") from error
        self.require(self.default_model)

        robot_type = self.robot_type
        for name, offered_model in self.models.items():
            if offered_model.action_format not in (None, *robot_type.ACTION_FORMATS):
                raise ValueError(
                    f"Model {name}: {robot_type.TITLE} has no format {offered_model.action_format!r}; "
                    f"use one of {', '.join(robot_type.ACTION_FORMATS)}"
                )

    @classmethod
    def from_station_config(cls, config: DictConfig) -> Self:
        """Returns the models of a station's configuration, as BaseRobot.load_config reads it."""
        node = OmegaConf.masked_copy(config, [config_field.name for config_field in fields(cls)])

        return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), node))

    def require(self, model: str) -> None:
        """Raises unless ``model`` is one of the models."""
        if model not in self.models:
            raise ValueError(f"Unknown or unavailable model {model!r}; choose one of {', '.join(self.models)}")

    def action_format(self, model: str) -> ActionFormat:
        """Returns the action format ``model`` answers in: its own, or the robot's first."""
        action_formats = self.robot_type.ACTION_FORMATS

        return action_formats[self.models[model].action_format or next(iter(action_formats))]

    @property
    def robot_type(self) -> type[BaseRobot]:
        """The class of the station's robot."""
        return pkgutil.resolve_name(self.robot_class)
