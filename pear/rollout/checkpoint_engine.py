from collections.abc import Generator, Iterable, Iterator

import torch

from pear.rollout.config import PearRolloutConfig
from pear.rollout.vllm_patches import Fp32Modules
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine, WorkerMetadata
from verl.utils.torch_dtypes import PrecisionType


# under verl's own backend name, in place of verl's class: rollout.checkpoint_engine.backend=nccl builds this engine
# in every process that imports this module, which verl does for rollout.checkpoint_engine.custom_backend_module
@CheckpointEngineRegistry.register(PearRolloutConfig.NCCL_BACKEND)
class PearNCCLCheckpointEngine(NCCLCheckpointEngine):
    """verl's NCCL checkpoint engine that sends the weights in the rollout's dtype and returns the unused cache
    first."""

    def __init__(self, bucket_size: int, rollout_dtype: str, fp32_modules: list[str], **kwargs) -> None:
        super().__init__(bucket_size, rollout_dtype=PrecisionType.to_dtype(rollout_dtype), **kwargs)
        self.fp32_modules = Fp32Modules(list(fp32_modules))

    def prepare(self) -> WorkerMetadata:
        # The cupy buffers of prepare and the NCCL communicator built in init_process_group are allocated outside
        # PyTorch's caching allocator; after an actor update its cache can hold the whole device, so return the unused
        # part first.
        torch.cuda.empty_cache()

        return super().prepare()

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int | None = None
    ) -> None:
        return await super().send_weights(self._cast_for_send(weights), global_steps=global_steps)

    def _cast_for_send(self, params: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """The weights in the rollout's dtype, except those whose name matches rollout.fp32_modules."""
        # half the bytes of the fp32 master weights, and the rollout's param.copy_() rounds the same way; the receiver
        # checks that the fp32 modules cover every weight it keeps wider
        dtype = self.rollout_dtype
        for name, tensor in params:
            is_cast = tensor.is_floating_point() and not self.fp32_modules.matches(name)
            yield name, tensor.to(dtype) if is_cast else tensor
