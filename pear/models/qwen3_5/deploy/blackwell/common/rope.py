from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.mm.rope import RopeState

if TYPE_CHECKING:  # annotations only
    import torch
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


class RopeStaging:
    """vLLM's M-RoPE prefill positions of a new request (RopeState) staged as int32 arrays and written by vLLM's own
    staged-write kernel with the same rows and contents."""

    WRITE_BLOCK = 1024  # vLLM's StagedWriteTensor.apply_write: _apply_write_kernel's BLOCK_SIZE

    @classmethod
    def install(cls) -> None:
        """Replace RopeState's prefill-position staging and its apply."""
        vllm_init = RopeState.__init__
        vllm_apply = RopeState.apply_staged_writes

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            vllm_init(self, *args, **kwargs)
            self.pear_staged_writes = []  # the staged (row, int32 positions) writes

        def init_prefill_positions(
            self, req_idx: int, model: torch.nn.Module, prefill_token_ids: list[int], mm_features: list
        ) -> None:
            if self.has_delta:
                positions, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
                self.prefill_delta.np[req_idx] = delta
            else:
                positions = model.get_xdrope_input_positions(prefill_token_ids, mm_features)

            # one int32 array per row (vLLM's path makes every position a Python int, num_dims per prompt token, and
            # converts them back into a tensor)
            rows = np.asarray(positions).astype(np.int32)  # [num_dims, length]
            for i in range(self.num_dims):
                self.pear_staged_writes.append((self.num_dims * req_idx + i, rows[i]))

        def apply_staged_writes(self) -> None:
            if self.pear_staged_writes:
                self.prefill_positions.apply_write()  # any list-staged writes first, in staging order
                cls.write(self.prefill_positions, self.pear_staged_writes)
                self.pear_staged_writes.clear()
            vllm_apply(self)

        RopeState.__init__ = __init__
        RopeState.init_prefill_positions = init_prefill_positions
        RopeState.apply_staged_writes = apply_staged_writes

    @staticmethod
    def write(staged_tensor: StagedWriteTensor, staged_writes: list[tuple[int, np.ndarray]]) -> None:
        """StagedWriteTensor.apply_write of the staged rows (each from column 0), their contents one array."""
        num_writes = len(staged_writes)
        indices = staged_tensor.write_indices.copy_to_uva(np.array([row for row, _ in staged_writes], dtype=np.int32))
        starts = staged_tensor.write_starts.copy_to_uva(np.zeros(num_writes, dtype=np.int32))
        cu_lens = staged_tensor.write_cu_lens.copy_to_uva(
            np.cumsum([len(positions) for _, positions in staged_writes]).astype(np.int32)
        )
        contents = async_tensor_h2d(
            np.concatenate([positions for _, positions in staged_writes]),
            device=staged_tensor.device,
            dtype=staged_tensor.dtype,
        )

        buffer_utils._apply_write_kernel[(num_writes,)](
            staged_tensor.gpu,
            staged_tensor.gpu.stride(0),
            indices,
            starts,
            contents,
            cu_lens,
            None,
            BLOCK_SIZE=RopeStaging.WRITE_BLOCK,
            MULTI_GROUP=False,
        )
