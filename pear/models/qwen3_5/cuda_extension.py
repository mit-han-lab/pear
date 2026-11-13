import inspect
import os
from pathlib import Path

import torch


class CudaExtension:
    """A GPU backend's CUDA sources (the csrc/ beside its module) as torch.ops.<OP_NAMESPACE>, built at first use under
    TORCH_EXTENSIONS_DIR."""

    # each subclass names its op namespace and its flags
    OP_NAMESPACE: str
    CUDA_FLAGS: tuple
    HOST_FLAGS = ("-O3",)  # of the host compiler, for the ops' wrappers
    LINK_FLAGS: tuple
    is_loaded = False

    @classmethod
    def load(cls) -> None:
        """Load the extension into this process; the first process on a machine builds it (torch locks the build)."""
        if cls.is_loaded:
            return

        from torch.utils.cpp_extension import load

        csrc = Path(inspect.getfile(cls)).resolve().parent / "csrc"

        # the build targets this GPU's architecture-specific variant (e.g. 10.3a on GB300); torch reads the target from
        # the environment, which is restored after the build
        major, minor = torch.cuda.get_device_capability()
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"

        try:
            load(
                name=cls.OP_NAMESPACE,
                sources=[str(path) for path in sorted(csrc.glob("*.cu"))],
                extra_cuda_cflags=list(cls.CUDA_FLAGS),
                extra_cflags=list(cls.HOST_FLAGS),
                extra_ldflags=list(cls.LINK_FLAGS),
                is_python_module=False,
            )
        finally:
            if arch_list is None:
                del os.environ["TORCH_CUDA_ARCH_LIST"]
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        cls.register_fakes()
        cls.is_loaded = True

    @staticmethod
    def register_fakes() -> None:
        """Register the fake implementations of the ops that compiled graphs trace (a backend without such ops has
        none)."""
