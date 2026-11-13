import numpy as np
import torch


class StagedBuffer:
    """One int32 device buffer at a fixed address, filled from two pinned host buffers in turn."""

    BUFFERS = 2  # pinned host buffers: one in flight, one being written

    def __init__(self, size: int, device):
        self.device_buffer = torch.zeros(size, dtype=torch.int32, device=device)

        # an upload does not wait for the previous step (a pageable copy would); a host buffer is rewritten once its
        # last upload is done
        self.stages = [torch.zeros(size, dtype=torch.int32).pin_memory() for _ in range(self.BUFFERS)]
        self.upload_events = [None] * self.BUFFERS
        self.turn = 0

    def next_stage(self) -> np.ndarray:
        """The turn's host buffer, once its last upload is done."""
        if self.upload_events[self.turn] is not None:
            self.upload_events[self.turn].synchronize()

        return self.stages[self.turn].numpy()

    def upload(self, size: int) -> None:
        """Copy the turn's first size slots to the device buffer and give the turn to the other host buffer."""
        self.device_buffer[:size].copy_(self.stages[self.turn][:size], non_blocking=True)
        self.upload_events[self.turn] = torch.cuda.Event()
        self.upload_events[self.turn].record()
        self.turn ^= 1
