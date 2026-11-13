from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError, RayActorError

from pear.sim.env import BaseEnv, EpisodeLost, StatefulEnv, StatelessEnv
from pear.sim.schemas import EnvResult
from pear.sim.server import ServerPool


class BaseClient(ABC):
    """The base of the agent-loop side of an env class: the environment tool runs each episode through it by the
    episode's uid, and it has the servers of the env class's pool carry the episode out."""

    def __init__(self, env_class: type[BaseEnv]) -> None:
        # the pool the driver started for env_class, found by its name (ValueError if it started none)
        self.server_pool = ray.get_actor(ServerPool.name_of(env_class), namespace=ServerPool.NAMESPACE)

    @abstractmethod
    async def open(self, uid: str, env_args: dict[str, Any]) -> EnvResult:
        """Opens the episode ``uid`` of ``env_args`` and returns its initial observation."""

    @abstractmethod
    async def step(self, uid: str, action: Any) -> EnvResult:
        """Carries out ``action`` in the episode ``uid`` and returns the result."""

    @abstractmethod
    async def close(self, uid: str) -> None:
        """Ends the episode ``uid``."""

    @staticmethod
    def _result(reply: bytes) -> EnvResult:
        result = pickle.loads(reply)  # as the process pickled it: the server passes it on unread
        if "error" in result:
            raise RuntimeError(result["error"])  # the env's code raised: its trajectory fails, its process goes on

        return result


class StatelessClient(BaseClient):
    """The agent-loop side of a stateless env: each call of an episode acquires a server from the env class's pool, has
    it replay the episode's native actions and carry out one action, and releases the server."""

    def __init__(self, env_class: type[StatelessEnv]) -> None:
        super().__init__(env_class)
        # what each open episode's calls replay: its env args and the native actions it executed so far
        self.replay_by_uid: dict[str, tuple[dict[str, Any], list[Any]]] = {}

    async def open(self, uid: str, env_args: dict[str, Any]) -> EnvResult:
        """Returns the initial observation of the episode ``uid`` of ``env_args``."""
        self.replay_by_uid[uid] = env_args, []

        return await self._rollout(env_args, [], None)

    async def step(self, uid: str, action: Any) -> EnvResult:
        """Carries out ``action`` in the episode ``uid`` after a replay of its native actions so far and returns the
        result."""
        env_args, actions = self.replay_by_uid[uid]
        result = await self._rollout(env_args, actions, action)
        actions.extend(result["actions"])

        return result

    async def close(self, uid: str) -> None:
        """Forgets the episode ``uid``; no server keeps anything of it."""
        self.replay_by_uid.pop(uid, None)  # (none when it never opened)

    async def _rollout(self, env_args: dict[str, Any], history: list[Any], action: Any) -> EnvResult:
        """Acquires a server (waiting for one), has it replay ``history`` in an episode of ``env_args`` and carry out
        ``action``, releases it and returns the result."""
        request = {"env_args": env_args, "history": history, "action": action}
        while True:
            server = await self.server_pool.acquire.remote()
            try:
                return self._result(await server.rollout.remote(request))
            except ActorDiedError:  # gone for good: it stays out of the pool, and the call goes to another server
                server = None
            except ActorUnavailableError:  # Ray is starting it again: the call goes to another server
                pass
            finally:
                if server is not None:  # free again, or once Ray started it again
                    await self.server_pool.release.remote(server)


class StatefulClient(BaseClient):
    """The agent-loop side of a stateful env: each episode acquires a server from the env class's pool at its open and
    releases it at its close, and the server keeps the episode in one of its processes in between."""

    def __init__(self, env_class: type[StatefulEnv]) -> None:
        super().__init__(env_class)
        self.server_by_uid: dict[str, Any] = {}  # the server each open episode acquired

    async def open(self, uid: str, env_args: dict[str, Any]) -> EnvResult:
        """Acquires a server for the episode ``uid`` of ``env_args`` (waiting for one) and returns the episode's initial
        observation."""
        while True:
            server = await self.server_pool.acquire.remote()
            self.server_by_uid[uid] = server  # the episode's close releases it, whatever the open's outcome
            try:
                return self._result(await server.open.remote(uid, env_args))
            except ActorDiedError:  # gone for good: it stays out of the pool, and the open goes to another server
                del self.server_by_uid[uid]
            except ActorUnavailableError:  # Ray is starting it again: back in the pool, the open goes to another one
                await self.server_pool.release.remote(self.server_by_uid.pop(uid))

    async def step(self, uid: str, action: Any) -> EnvResult:
        """Carries out ``action`` in the episode ``uid`` and returns the result."""
        try:
            reply = await self.server_by_uid[uid].step.remote(uid, action)
        except RayActorError as exc:  # its server went away, and the episode with it
            raise EpisodeLost(f"{type(exc).__name__}: {exc}") from exc

        return self._result(reply)

    async def close(self, uid: str) -> None:
        """Ends the episode ``uid`` on its server and releases the server."""
        server = self.server_by_uid.pop(uid, None)
        if server is None:  # (none when the open never got a server)
            return

        try:
            await server.close.remote(uid)
        except ActorDiedError:  # gone for good: it stays out of the pool
            return
        except ActorUnavailableError:  # Ray is starting it again: back in the pool all the same
            pass

        await self.server_pool.release.remote(server)
