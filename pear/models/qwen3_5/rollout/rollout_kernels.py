from vllm.model_executor.models import ModelRegistry

from pear.models.kernels import RolloutKernels
from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend


class Qwen3_5RolloutKernels(RolloutKernels):
    """The vLLM classes and patches of the replicas of a Qwen3.5 (Gated DeltaNet) model, named by rollout.kernels: in
    each process the backend of its GPU."""

    # the GPU generations' backends, in priority order
    GPU_BACKENDS = (BlackwellRolloutGpuBackend, HopperRolloutGpuBackend)

    @classmethod
    def configure_server(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """The settings of the process that builds the engine config (the server), before it builds it."""
        cls.select_gpu_backend().configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def register(cls) -> None:
        """PEAR's vLLM general plugin (pyproject.toml): register the GPU backend's model in a process of an engine the
        server switched the kernels on for."""
        # vLLM loads general plugins first in each process; a worker resolves its model class and keeps it for the
        # process (init_worker's multimodal receiver cache) before it builds the worker, whose extension installs the
        # GPU backend: a registration there comes too late
        if cls.is_enabled():
            ModelRegistry.register_model(VllmGpuBackend.ARCHITECTURE, cls.select_gpu_backend().model_cls)

    @classmethod
    def install_worker(cls, vllm_config) -> None:
        """Register the GPU backend's classes and apply its patches in a worker process, before it builds its model."""
        # vLLM's compile cache does not key on these patches (it would load a graph compiled without them); verl's
        # runtime env disables that cache (VLLM_DISABLE_COMPILE_CACHE in verl/trainer/constants_ppo.py)
        cls.select_gpu_backend(vllm_config).install(vllm_config)

    @classmethod
    def check_model(cls, model, vllm_config) -> None:
        """Raise unless a worker built the GPU backend's model (PEAR installed without its vLLM plugin)."""
        gpu_backend = cls.select_gpu_backend(vllm_config)
        if type(model) is not gpu_backend.model_cls:
            raise RuntimeError(
                f"the worker built {type(model).__name__}, not {gpu_backend.model_cls.__name__}: PEAR's vLLM plugin "
                "(pyproject.toml, vllm.general_plugins) is missing from this environment; run scripts/install.sh verl"
            )

    @classmethod
    def select_gpu_backend(cls, vllm_config=None) -> type[VllmGpuBackend]:
        """The first GPU backend that supports this process's GPU and, given one, the engine config."""
        return VllmGpuBackend.select(
            cls.GPU_BACKENDS, vllm_config, "no GPU backend of rollout.kernels fits this replica"
        )
