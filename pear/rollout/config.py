from dataclasses import dataclass, field

from verl.workers.config import RolloutConfig


@dataclass
class PearRolloutConfig(RolloutConfig):
    """verl's rollout config with the keys of PEAR's vLLM server, its worker extension and its weight sync."""

    VLLM_NAME = "vllm"  # verl's rollout name of vLLM's replicas, under which PEAR's replica registers
    NCCL_BACKEND = "nccl"  # verl's name of the NCCL checkpoint engine, under which PEAR's engine registers
    # the module that registers PEAR's NCCL engine as the backend nccl (rollout.checkpoint_engine.custom_backend_module)
    NCCL_BACKEND_MODULE = "pear.rollout.checkpoint_engine"

    # weights the vLLM replicas keep in fp32 instead of dtype, by substrings of their names: built in fp32, then loaded
    # and synced in fp32. The sync casts every other weight to dtype, so a weight vLLM itself keeps in fp32 belongs
    # here too.
    fp32_modules: list[str] = field(default_factory=list)
    # a model family's vLLM classes and patches for its replicas, by class path (pear/models/kernels.py); None:
    # vLLM's own
    kernels: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.kernels and self.name != self.VLLM_NAME:
            raise ValueError("rollout.kernels extends vLLM's replicas: it requires rollout.name=vllm.")
        if self.kernels and self.free_cache_engine:
            raise ValueError(
                "rollout.kernels requires rollout.free_cache_engine=False: the rollout kernels' device state is built "
                "with the model, and a level-2 sleep discards that memory without a backup."
            )
        # without the module, backend=nccl is verl's own engine, which sends every weight in the trainer's dtype and
        # keeps the allocator's cache
        engine_config = self.checkpoint_engine
        if (
            engine_config.backend == self.NCCL_BACKEND
            and engine_config.custom_backend_module != self.NCCL_BACKEND_MODULE
        ):
            raise ValueError(
                "rollout.checkpoint_engine.backend=nccl requires rollout.checkpoint_engine.custom_backend_module="
                f"{self.NCCL_BACKEND_MODULE}: the module registers PEAR's NCCL engine under that name."
            )
