import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, FFmpeg


@dataclass(frozen=True)
class SourceClip:
    """The cameras' own videos of one recording segment, placed on the task's monotonic clock."""

    start_monotonic_s: float  # when the segment's first frame was captured
    duration_s: float
    files: dict[str, Path]  # camera role -> its video; a camera whose video failed has no entry

    @classmethod
    def of_segment(cls, directory: Path, manifest: dict[str, Any]) -> "SourceClip | None":
        """Returns the clip of a saved segment from its manifest, None when none of its videos is on disk."""
        names = {role: name for role, name in (manifest.get("sources") or {}).items() if name}
        files = {role: directory / name for role, name in names.items() if (directory / name).is_file()}

        return cls(manifest["started_monotonic_s"], manifest["frames"] / FRAME_RATE, files) if files else None


class SourceFrames:
    """One clip's camera videos decoded in step, each at the size it is shown: its pictures, frame by frame."""

    def __init__(self, clip: SourceClip, sizes: dict[str, tuple[int, int]], executable: str, log: IO[str]) -> None:
        self.frame_count = round(clip.duration_s * FRAME_RATE)
        self.sizes = sizes  # camera role -> (width, height) of its pictures
        self.decoders: dict[str, subprocess.Popen] = {}
        # a camera without a video has no picture; a video a few frames short keeps its last one
        self.pictures: dict[str, np.ndarray] = {}

        try:
            for role, path in clip.files.items():
                self.decoders[role] = FFmpeg.raw_decoder(executable, path, sizes[role], log)
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yields the pictures of each frame by camera role."""
        for _ in range(self.frame_count):
            for role, decoder in self.decoders.items():
                width, height = self.sizes[role]
                picture_bytes = decoder.stdout.read(width * height * 3)
                if len(picture_bytes) == width * height * 3:
                    self.pictures[role] = np.frombuffer(picture_bytes, np.uint8).reshape(height, width, 3)

            yield dict(self.pictures)

    def close(self) -> None:
        """Ends the decoders."""
        for decoder in self.decoders.values():
            FFmpeg.kill(decoder)
            decoder.stdout.close()
