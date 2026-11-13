import argparse
import os

import ray

from pear.models.kernels import RolloutKernels
from pear.rollout.config import PearRolloutConfig
from pear.rollout.lmcache import LMCacheVLLMServer
from pear.rollout.vllm_patches import Fp32Modules
from verl.workers.rollout.replica import RolloutMode, RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


class PearVLLMServer(LMCacheVLLMServer):
    """The vLLM server of PEAR's replicas: weights kept in fp32, a model family's rollout kernels, a checked weight
    sync."""

    # by name: vLLM imports the class in each worker process, and importing its module here would be circular there
    WORKER_EXTENSION = "pear.rollout.worker_extension.PearVLLMWorkerExtension"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        # NCCL's implicit launch ordering (base.yaml sets it for the Ray workers) hangs the replica after a weight
        # sync: its two communicators (TP, weight sync) issue from different threads, so the enforced order differs
        # across ranks
        os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "0"
        super()._preprocess_engine_kwargs(engine_kwargs)
        # the worker processes started below inherit this environment: the weights they keep in fp32
        Fp32Modules.export(self.config.fp32_modules)

    def _postprocess_compilation_config(self, compilation_config: dict) -> None:
        if self.config.kernels:
            # the worker processes inherit the rollout kernels' name with this environment
            kernels = RolloutKernels.export(self.config.kernels)
            kernels.configure_server(compilation_config, self.config.max_num_seqs, self.config.max_num_batched_tokens)

    def _get_worker_extension_cls(self) -> str:
        return self.WORKER_EXTENSION

    async def run_server(self, args: argparse.Namespace) -> None:
        await super().run_server(args)

        if self.config.kernels:
            await self.engine.collective_rpc(method="check_kernels")

    async def release_kv_cache(self) -> None:
        if self.node_rank == 0 and self.rollout_mode != RolloutMode.COLOCATED:
            # the sender casts every weight to the replicas' dtype except the fp32 modules, which the workers verify
            await self.engine.collective_rpc("check_fp32_modules", args=(self.config.fp32_modules,))
            # with or without sleep mode: the receiver allocates next
            await self.engine.collective_rpc("release_cached_memory")

        await super().release_kv_cache()


class PearVLLMReplica(vLLMReplica):
    """vLLM's replica whose servers are PEAR's."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(PearVLLMServer)

    @classmethod
    def require_registered(cls, name: str) -> None:
        """Raise unless rollout.name builds this replica in this process."""
        replica_class = RolloutReplicaRegistry.get(name)
        if replica_class is not cls:
            raise RuntimeError(
                f"rollout.name={name} builds {replica_class.__name__}, not {cls.__name__}: the replicas would run "
                "without PEAR's server (LMCache, rollout.fp32_modules, rollout.kernels, the checks of a weight sync)"
            )


# verl plugin registration on import (VERL_USE_EXTERNAL_MODULES=pear.rollout.server, forwarded to the Ray actors
# through ray_kwargs.ray_init.runtime_env.env_vars): rollout.name=vllm builds PEAR's replicas
RolloutReplicaRegistry.register(PearRolloutConfig.VLLM_NAME, lambda: PearVLLMReplica)
