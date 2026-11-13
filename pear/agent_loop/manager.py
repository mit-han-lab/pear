from __future__ import annotations

import pkgutil

import ray
from tensordict import TensorDict

from pear.envs.tool import EnvironmentTool
from pear.sim.server import ServerPool
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils import tensordict_utils as tu


class PearAgentLoopManager(AgentLoopManagerTQ):
    """verl's agent loop manager (actor_rollout_ref.rollout.agent.agent_loop_manager_class) that starts a server pool
    for the env class of each tool in its tool config, once per run in the driver, holds the pools for the run and deals
    the prompts to its workers round-robin."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the agent-loop workers, whose tools find these pools by name, start only after this constructor (create)
        self.server_pools = []  # held for the run: Ray ends an actor nobody holds a handle of
        for config in EnvironmentTool.load_configs(self.rollout_config.multi_turn.tool_config_path):
            self.server_pools.append(ServerPool.start(pkgutil.resolve_name(config["env_class"]), config))

        self._next_worker = 0  # round-robin cursor across dispatches (generate_sequences)

    def generate_sequences(self, prompts: TensorDict) -> None:
        # Round-robin across dispatches, not per batch: DAPO refills arrive as batches of one or two prompts, which
        # prompts.chunk(num_workers) would always hand to the first workers. Row i goes to worker (first_worker + i) %
        # worker_n; index_select_tensor_dict because a batch read back from TransferQueue holds nested tensors.
        worker_n = len(self.agent_loop_workers)
        first_worker, self._next_worker = self._next_worker, (self._next_worker + len(prompts)) % worker_n
        ray.get(
            [
                self.agent_loop_workers[(first_worker + offset) % worker_n].generate_sequences.remote(
                    tu.index_select_tensor_dict(prompts, list(range(offset, len(prompts), worker_n)))
                )
                for offset in range(min(worker_n, len(prompts)))
            ]
        )
