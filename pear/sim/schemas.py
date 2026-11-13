from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


@dataclass(frozen=True)
class ServerContext:
    """What an env class may use of the server it sets up."""

    config: dict[str, Any]  # the family's tool config, as the run loaded it
    log_dir: Path  # where the server and its simulator processes log
    gpu_ids: list[int]  # the GPUs Ray gave the server
    log: Callable[[str], None]  # appends a message to the server's log


class EnvResult(TypedDict):
    """What one call of an env returns: the protocol's keys, in this order, before the keys of the family's own."""

    images: list[Any] | None  # the cameras' frames in CAMERAS order; None once the episode ended
    state: dict[str, dict[str, float]] | None  # the parts' quantities as the model reads them; None: not readable
    actions: list[Any]  # the native actions the call carried out
    rewards: list[float]  # one per native action
    done: bool  # whether the episode ended
    executed: int  # how many native actions the call carried out
    outcome: str | None  # how the action ended, in the family's words; None without an action
    instruction: str  # the episode's task
