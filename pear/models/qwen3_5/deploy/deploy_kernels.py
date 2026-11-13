from __future__ import annotations

from typing import TYPE_CHECKING

from pear.models.kernels import DeployKernels
from pear.models.qwen3_5.deploy.blackwell.gpu_backend import (
    BlackwellBf16DeployGpuBackend,
    BlackwellNvfp4DeployGpuBackend,
)
from pear.models.qwen3_5.deploy.gpu_backend import DeployGpuBackend
from pear.models.qwen3_5.deploy.memory_plan import MemoryPlan
from pear.models.qwen3_5.deploy.rtx_blackwell.gpu_backend import RtxBlackwellNvfp4DeployGpuBackend
from pear.models.qwen3_5.gpu_backend import VllmGpuBackend
from pear.utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotations only
    from vllm.config import VllmConfig

logger = get_logger(__name__, default_level="INFO")  # the backend a worker installs belongs to the server's log


class Qwen3_5DeployKernels(DeployKernels):
    """The vLLM classes and patches of a deploy of a Qwen3.5 (Gated DeltaNet) model, named by kernels: in each process
    the backend of its GPU and its checkpoint."""

    # the backends, in priority order
    GPU_BACKENDS = (BlackwellNvfp4DeployGpuBackend, BlackwellBf16DeployGpuBackend, RtxBlackwellNvfp4DeployGpuBackend)

    @classmethod
    def configure_server(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        cls.select_gpu_backend().configure(compilation_config, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def install_worker(cls, vllm_config: VllmConfig) -> None:
        gpu_backend = cls.select_gpu_backend(vllm_config)
        logger.info("deploy kernels: %s", gpu_backend.__name__)

        # where vLLM's tensors live, as the config's environment names it: before the worker loads the model
        MemoryPlan.install()
        gpu_backend.install(vllm_config)

    @classmethod
    def select_gpu_backend(cls, vllm_config: VllmConfig | None = None) -> type[DeployGpuBackend]:
        """The first GPU backend that supports this process's GPU and, given one, the engine config."""
        return VllmGpuBackend.select(
            cls.GPU_BACKENDS, vllm_config, "no GPU backend of the deploy kernels fits this server"
        )
