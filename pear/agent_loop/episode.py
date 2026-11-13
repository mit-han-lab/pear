from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class EndReason(str, Enum):
    """How an episode ends, each by the text its EPISODE_END record names it with."""

    ENV = "env"  # the environment finished it
    PARSE = "parse"  # an answer without a usable action
    BUDGET = "budget"  # its response tokens ran out


@dataclass(eq=False)  # each episode is a run of its own: two are never the same by their contents
class Episode:
    """One run of a trajectory, as the agent loop records it: its env args, the native actions executed and what the
    model's actions earned; the tool's client knows it by its uid."""

    env_args: dict[str, Any]  # the env class's arguments (the row's extra_info["env"], decoded)
    uid: str = field(default_factory=lambda: uuid4().hex)  # this trajectory's own id: the n of a group share their env
    actions: list[Any] = field(default_factory=list)  # every native action executed, in order
    num_chunks: int = 0  # the model's actions carried out
    chunk_rewards: list[float] = field(default_factory=list)  # the reward of each of the model's actions
    done: bool = False  # the simulator ended the episode
    success: bool = False  # it ended with the task done

    def update(self, result: dict[str, Any]) -> None:
        """Accounts for one of the model's actions by the simulator's ``result``, whose ``actions`` are the native
        actions taken."""
        self.actions.extend(result["actions"])
        self.chunk_rewards.append(float(sum(float(r) for r in result["rewards"])))

        self.num_chunks += 1
        self.done = bool(result["done"])
        self.success = self.is_success(result)

    def record(self) -> dict[str, Any]:
        """Returns the run as the val dumps and the evaluation's records keep it."""
        return {
            "env": self.env_args,
            "uid": self.uid,
            "actions": self.actions,
            "executed_steps": len(self.actions),
            "num_chunks": self.num_chunks,
            "chunk_rewards": self.chunk_rewards,
            "done": self.done,
            "success": self.success,
        }

    @staticmethod
    def is_success(result: dict[str, Any]) -> bool:
        """Returns whether ``result`` ends its episode with the task done: its action earned a reward."""
        return bool(result["done"]) and sum(float(reward) for reward in result["rewards"]) > 0.0
