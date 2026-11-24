from dataclasses import dataclass

import numpy as np

from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.sources import SourceClip


@dataclass(frozen=True)
class RenderChunk:
    """A piece of a styled video that one worker draws and encodes alone: the frames of one clip from ``first_frame``
    to ``end_frame`` that the video shows."""

    kind: str  # of the video: FULL_KIND or MOTION_KIND
    index: int  # its place among the video's chunks, which names its file
    clip_index: int  # of the clip whose frames it shows
    clip_start: int  # the clip's first frame among the frames of every clip back to back
    first_frame: int  # of the clip, where its decoding starts: the first frame it shows
    end_frame: int  # of the clip, after the last frame it shows
    first_output: int  # its first frame's place in the video
    frame_count: int  # the frames it gives the video


class RenderPlan:
    """The chunks of a styled render, video after video in the order the render publishes them, and the frames each
    video takes of each clip."""

    CHUNK_FRAMES = 10 * FRAME_RATE  # of a chunk at most: a worker's share of a video, and a short clip is one

    def __init__(
        self, kinds: tuple[str, ...], clips: list[SourceClip], motion_spans: list[tuple[float, float]]
    ) -> None:
        self.kinds = kinds
        self.clip_frame_counts = [round(clip.duration_s * FRAME_RATE) for clip in clips]
        self.cut = MotionCut(motion_spans, sum(self.clip_frame_counts))
        self.chunks: list[RenderChunk] = []
        # by video, the frames it takes of each clip
        self.clip_frames: dict[str, list[int]] = {kind: [] for kind in kinds}

        for kind in kinds:
            self._plan(kind)

    def frame_count(self, kind: str) -> int:
        """Returns the frames of the video of ``kind``."""
        return sum(self.clip_frames[kind])

    def _plan(self, kind: str) -> None:
        """Adds the chunks of the video of ``kind``: each clip's frames the video shows, CHUNK_FRAMES at a time."""
        index = first_output = clip_start = 0
        for clip_index, clip_frame_count in enumerate(self.clip_frame_counts):
            is_shown = (
                self.cut.is_kept[clip_start : clip_start + clip_frame_count]
                if kind == TaskFiles.MOTION_KIND
                else np.ones(clip_frame_count, bool)
            )
            shown_frames = np.flatnonzero(is_shown)
            for first in range(0, len(shown_frames), self.CHUNK_FRAMES):
                frames = shown_frames[first : first + self.CHUNK_FRAMES]
                chunk = RenderChunk(
                    kind, index, clip_index, clip_start, int(frames[0]), int(frames[-1]) + 1, first_output, len(frames)
                )
                self.chunks.append(chunk)
                index += 1
                first_output += len(frames)
            self.clip_frames[kind].append(len(shown_frames))
            clip_start += clip_frame_count
