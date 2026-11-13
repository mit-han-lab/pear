from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from pear.trainer.engine import PearFSDPEngine
from verl.workers.config import FSDPEngineConfig


@dataclass
class PearFSDPEngineConfig(FSDPEngineConfig):
    """verl's FSDP engine config with the keys of PEAR's engine: a model family's training kernels and whether they
    install a GPU generation's backend."""

    # a model family's training kernels, by class path (pear/models/kernels.py); None: verl's engine as it is
    kernels: str | None = None
    # whether the kernels install the backend of the machine's GPU generation, beside what they install on any GPU
    use_gpu_backend: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.use_gpu_backend and not self.kernels:
            raise ValueError(
                "actor.fsdp_config.use_gpu_backend requires actor.fsdp_config.kernels, whose GPU backend it is."
            )
        # verl only keeps FSDP2 from casting the layers' inputs back: the fp32 stream itself is a model family's
        if (self.mixed_precision or {}).get("fp32_residual") and not self.kernels:
            raise ValueError(
                "actor.fsdp_config.mixed_precision.fp32_residual requires actor.fsdp_config.kernels, "
                "whose stream it is."
            )
        # importing this class registered PEAR's engine; another device or vendor key may still win the lookup
        if self.kernels:
            PearFSDPEngine.require_registered(self.strategy)


class RunConfig:
    """A run's composed config as an entry point hands it to verl's launch."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def prepare(self) -> None:
        """Sets what follows from the run's sizes and resolves the workers' environment."""
        self._set_sizes()

        # verl merges the workers' environment into its own outside this config, where a value that interpolates one of
        # the run's keys would stay its text
        OmegaConf.resolve(self.config.ray_kwargs)

    def _set_sizes(self) -> None:
        """Sets verl's keys that follow from run_sizes: a replica's decoding slots, the agent loop's processes and the
        sessions kept generating across all replicas."""
        sizes = self.config.run_sizes
        if sizes.sessions_per_replica is None:  # the run gave none: verl's keys stand as it composed them
            return

        rollout = self.config.actor_rollout_ref.rollout
        replica_n = rollout.nnodes * rollout.n_gpus_per_node // rollout.tensor_model_parallel_size

        rollout.max_num_seqs = sizes.seqs_per_session * sizes.sessions_per_replica
        rollout.agent.num_workers = replica_n * rollout.max_num_seqs // sizes.seqs_per_agent_worker
        self.config.trainer.v1.sampler.max_inflight_sessions = replica_n * sizes.sessions_per_replica
