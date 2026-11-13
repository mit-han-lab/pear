from __future__ import annotations

from typing import Any

from pear.models.kernels import DeployKernels


class DeployWorkerExtension:
    """vLLM's worker extension of a deploy server: the model family's deploy kernels installed in each worker."""

    def __new__(cls, **kwargs: Any) -> DeployWorkerExtension:
        instance = super().__new__(cls)

        # before the worker builds its model (the config's kernels, which DeployServer names)
        kernels = DeployKernels.from_env()
        if kernels is not None:
            kernels.install_worker(kwargs["vllm_config"])

        return instance
