class IEEEFp32:
    """IEEE fp32 inside the block: the TF32 of cuBLAS's matmuls or of cuDNN's convolutions off."""

    # the engine leaves TF32 on for the fp32 vision tower; settings: torch.backends.cuda.matmul or torch.backends.cudnn
    def __init__(self, settings) -> None:
        self.settings = settings

    def __enter__(self):
        self.allow_tf32 = self.settings.allow_tf32
        self.settings.allow_tf32 = False

    def __exit__(self, *exc):
        self.settings.allow_tf32 = self.allow_tf32
