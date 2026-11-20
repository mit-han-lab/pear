import torch

from pear.models.qwen3_5.rollout.staged_buffer import StagedBuffer


class HostStagedBuffer:
    """StagedBuffer's construction and upload on a machine without a GPU driver: pageable host buffers, plain copies,
    no upload events."""

    @staticmethod
    def init(buffer: StagedBuffer, size: int, device, dtype: torch.dtype = torch.int32) -> None:
        buffer.device_buffer = torch.zeros(size, dtype=dtype, device=device)
        buffer.stages = [torch.zeros(size, dtype=dtype) for _ in range(StagedBuffer.BUFFERS)]
        buffer.upload_events = [None] * StagedBuffer.BUFFERS
        buffer.turn = 0

    @staticmethod
    def upload(buffer: StagedBuffer, size: int) -> None:
        buffer.device_buffer[:size].copy_(buffer.stages[buffer.turn][:size])
        buffer.turn ^= 1

    @classmethod
    def install(cls, monkeypatch) -> None:
        """Every StagedBuffer of the test on the host, for the test's duration."""
        monkeypatch.setattr(StagedBuffer, "__init__", cls.init)
        monkeypatch.setattr(StagedBuffer, "upload", cls.upload)
