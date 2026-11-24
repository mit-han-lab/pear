import multiprocessing.queues
import os
from enum import StrEnum
from pathlib import Path
from typing import IO

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.render_plan import RenderChunk
from pear.deploy.station.records.video.sources import SourceClip, SourceFrames
from pear.deploy.station.records.video.styled_frames import StyledFrames
from pear.deploy.station.records.video.task_timeline import TaskTimeline
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RenderStatus(StrEnum):
    """Where a render of styled videos stands, each by the word the page reads."""

    WAITING = "waiting"  # a segment it needs is not finished yet
    QUEUED = "queued"  # another render runs first
    RENDERING = "rendering"  # its workers draw and encode
    SAVED = "saved"  # its videos are in place
    FAILED = "failed"  # it ended without them


class RenderWorker:
    """A spawned process of one render: it draws and encodes the chunks of the styled videos it is given, one after
    another, each into a file of its own."""

    PROGRESS_FRAMES = FRAME_RATE  # frames rendered between two progress reports
    ENCODER_TIMEOUT_S = 120  # for the encoder to finish once its input is closed

    def __init__(
        self,
        task_directory: Path,
        clips: list[SourceClip],
        cameras: tuple[Camera, ...],
        folder: Path,
        chunks: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        self.files = TaskFiles(task_directory)
        self.clips = clips  # the segments' videos, in recording order
        self.cameras = cameras
        self.folder = folder  # where the chunks are written
        self.chunks = chunks  # from the render job: the next chunk, or None at the end
        self.messages = messages  # to the render job: the progress, each chunk done, or why one failed
        self.executable = executable
        self.styled_frames: dict[str, StyledFrames] = {}  # by video, made with its first chunk
        self.timeline: TaskTimeline | None = None
        self.cut: MotionCut | None = None

    @classmethod
    def main(
        cls,
        task_directory: str,
        clips: list[SourceClip],
        cameras: tuple[Camera, ...],
        folder: str,
        chunks: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        """Runs one worker of a render in the process that was spawned for it."""
        cls(Path(task_directory), clips, cameras, Path(folder), chunks, messages, executable).run()

    def run(self) -> None:
        """Renders the chunks it is given until the end, and reports each, or the first that failed."""
        os.setsid()
        os.nice(FFmpeg.RENDER_NICENESS)

        try:
            # every worker appends to the render's one log
            with (self.folder / TaskFiles.ENCODER_LOG_NAME).open("a") as log:
                self.timeline = TaskTimeline.load(self.files, self.clips)
                frame_count = sum(round(clip.duration_s * FRAME_RATE) for clip in self.clips)
                self.cut = MotionCut(self.timeline.motion_spans, frame_count)
                while (chunk := self.chunks.get()) is not None:
                    self.render(chunk, log)
                    self.messages.put({"done": chunk.index, "kind": chunk.kind})
        except BaseException as error:
            logger.exception("A render into %s failed", self.folder)
            self.messages.put({"error": f"{type(error).__name__}: {error}{self._log_tail()}"})

    def render(self, chunk: RenderChunk, log: IO[str]) -> None:
        """Writes the unpublished file of ``chunk``."""
        styled_frames = self._styled_frames(chunk.kind)
        # FFmpeg scales each camera's video to its tile, so the frames take the pictures as they are
        sizes = {role: (width, height) for role, (_, _, width, height) in styled_frames.tiles.items()}
        filter_script = None
        if chunk.kind == TaskFiles.MOTION_KIND:
            name = TaskFiles.CHUNK_NAME.format(kind=chunk.kind, index=chunk.index) + TaskFiles.CHUNK_FILTER_SUFFIX
            filter_script = self.folder / name
            filter_script.write_text(
                self.cut.select_filter(chunk.clip_start + chunk.first_frame, chunk.end_frame - chunk.first_frame)
            )
        path = self.folder / TaskFiles.chunk_video_name(chunk.kind, chunk.index)
        clip = self.clips[chunk.clip_index]
        pictures = SourceFrames(clip, sizes, self.executable, log, chunk.frame_count, chunk.first_frame, filter_script)
        encoder = FFmpeg.raw_encoder(self.executable, VIDEO_SIZE, path, log)

        try:
            reported_frames = 0  # frames of the chunk reported_frames as done
            for index, images in enumerate(pictures):
                time_s = (chunk.first_output + index) / FRAME_RATE
                encoder.stdin.write(styled_frames.frame(images, time_s).tobytes())
                if index + 1 - reported_frames == self.PROGRESS_FRAMES:
                    self.messages.put({"frames": self.PROGRESS_FRAMES, "kind": chunk.kind})
                    reported_frames = index + 1
            encoder.stdin.close()
            if encoder.wait(timeout=self.ENCODER_TIMEOUT_S) != 0:
                raise RuntimeError(f"The encoder of the {chunk.kind} video's chunk {chunk.index} failed")
            self.messages.put({"frames": chunk.frame_count - reported_frames, "kind": chunk.kind})
        finally:
            pictures.close()
            FFmpeg.kill(encoder)
            if filter_script:
                filter_script.unlink(missing_ok=True)

    def _styled_frames(self, kind: str) -> StyledFrames:
        """Returns the styled frames of the video of ``kind``, which are made once."""
        if kind not in self.styled_frames:
            # the turns and the progress of the motion-only video run on its own, shorter timeline
            timeline = self.timeline.cut_to_motion() if kind == TaskFiles.MOTION_KIND else self.timeline
            self.styled_frames[kind] = StyledFrames(
                self.cameras,
                timeline.instruction,
                timeline.model_label,
                timeline.badge,
                timeline.turns,
                timeline.duration_s,
            )

        return self.styled_frames[kind]

    def _log_tail(self) -> str:
        """Returns the last error lines of FFmpeg."""
        try:
            log_text = (self.folder / TaskFiles.ENCODER_LOG_NAME).read_text(errors="replace")
        except OSError:
            return ""

        detail = log_text.strip()[-FFmpeg.ERROR_TAIL_CHARS :]

        return f"; FFmpeg: {detail}" if detail else ""
