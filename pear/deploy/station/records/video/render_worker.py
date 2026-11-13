import multiprocessing.queues
import os
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.sources import SourceClip, SourceFrames
from pear.deploy.station.records.video.styled_frames import StyledFrames
from pear.deploy.station.records.video.task_timeline import TaskTimeline
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RenderStatus(StrEnum):
    """Where a render of styled videos stands, each by the word the page reads."""

    WAITING = "waiting"  # a segment it needs is not finished yet
    QUEUED = "queued"  # another render runs first
    RENDERING = "rendering"  # its worker draws and encodes
    RENDERED = "rendered"  # the worker's own word: its videos are written and not published yet
    SAVED = "saved"  # its videos are in place
    FAILED = "failed"  # it ended without them


class RenderWorker:
    """The spawned process of one render: it draws the styled full and motion-only videos of a task's segments from
    the cameras' own videos."""

    PROGRESS_FRAMES = FRAME_RATE  # frames rendered between two progress reports
    ENCODER_TIMEOUT_S = 120  # for an encoder to finish once its input is closed

    def __init__(
        self,
        task_directory: Path,
        clips: list[SourceClip],
        cameras: tuple[Camera, ...],
        folder: Path,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        self.files = TaskFiles(task_directory)
        self.clips = clips  # the segments' videos, in recording order
        self.cameras = cameras
        self.folder = folder  # where the videos are written
        self.messages = messages  # to the render job: the progress, then the result
        self.executable = executable

    @classmethod
    def main(
        cls,
        task_directory: str,
        clips: list[SourceClip],
        cameras: tuple[Camera, ...],
        folder: str,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        """Runs the worker of one render in the process that was spawned for it."""
        cls(Path(task_directory), clips, cameras, Path(folder), messages, executable).run()

    def run(self) -> None:
        """Renders the videos and reports the frames each received, or why it failed."""
        os.setsid()
        os.nice(FFmpeg.RENDER_NICENESS)

        try:
            with (self.folder / TaskFiles.ENCODER_LOG_NAME).open("w") as log:
                render_result = self.render(TaskTimeline.load(self.files, self.clips), log)
            self.messages.put({"status": RenderStatus.RENDERED.value, **render_result})
        except BaseException as error:
            logger.exception("The render into %s failed", self.folder)
            error_text = f"{type(error).__name__}: {error}{self._log_tail()}"
            self.messages.put({"status": RenderStatus.FAILED.value, "error": error_text})

    def render(self, timeline: TaskTimeline, log: IO[str]) -> dict[str, Any]:
        """Writes the unpublished full video, and the motion-only one when anything moved, and returns the frames
        each received."""
        frame_count = sum(round(clip.duration_s * FRAME_RATE) for clip in self.clips)
        cut = MotionCut(timeline.motion_spans, frame_count)

        shared_arguments = (self.cameras, timeline.instruction, timeline.model_label, timeline.badge)
        styled_frames_by_kind = {
            TaskFiles.FULL_KIND: StyledFrames(*shared_arguments, timeline.turns, timeline.duration_s)
        }
        if cut.kept_count:
            # the turns and the progress of the motion-only video run on its own, shorter timeline
            motion_timeline = timeline.cut_to_motion()
            styled_frames_by_kind[TaskFiles.MOTION_KIND] = StyledFrames(
                *shared_arguments, motion_timeline.turns, motion_timeline.duration_s
            )

        encoders: dict[str, subprocess.Popen] = {}
        try:
            for kind in styled_frames_by_kind:
                path = self.folder / TaskFiles.partial_name(TaskFiles.video_name(kind))
                encoders[kind] = FFmpeg.raw_encoder(self.executable, VIDEO_SIZE, path, log)
            clip_motion_frames = self._encode(styled_frames_by_kind, encoders, cut, frame_count, log)
            for kind, encoder in encoders.items():
                encoder.stdin.close()
                if encoder.wait(timeout=self.ENCODER_TIMEOUT_S) != 0:
                    raise RuntimeError(f"The {kind} video encoder failed")
        finally:
            for encoder in encoders.values():
                FFmpeg.kill(encoder)

        return {"frames": frame_count, "motion_frames": cut.kept_count, "clip_motion_frames": clip_motion_frames}

    def _encode(
        self,
        styled_frames_by_kind: dict[str, StyledFrames],
        encoders: dict[str, subprocess.Popen],
        cut: MotionCut,
        frame_count: int,
        log: IO[str],
    ) -> list[int]:
        """Gives every frame to the full video's encoder and the kept ones to the motion-only video's, and returns
        how many frames of each clip were kept."""
        # FFmpeg scales each camera's video to its tile, so the frames take the pictures as they are
        full_frames = styled_frames_by_kind[TaskFiles.FULL_KIND]
        sizes = {role: (width, height) for role, (_, _, width, height) in full_frames.tiles.items()}
        index = kept_count = 0
        clip_motion_frames = []

        for clip in self.clips:
            pictures = SourceFrames(clip, sizes, self.executable, log)
            kept_before_clip = kept_count
            try:
                for images in pictures:
                    encoders[TaskFiles.FULL_KIND].stdin.write(full_frames.frame(images, index / FRAME_RATE).tobytes())
                    if cut.is_kept[index]:
                        motion_frame = styled_frames_by_kind[TaskFiles.MOTION_KIND].frame(
                            images, kept_count / FRAME_RATE
                        )
                        encoders[TaskFiles.MOTION_KIND].stdin.write(motion_frame.tobytes())
                        kept_count += 1
                    index += 1
                    if index % self.PROGRESS_FRAMES == 0:
                        self.messages.put({"progress": index / frame_count})
            finally:
                pictures.close()
            clip_motion_frames.append(kept_count - kept_before_clip)

        return clip_motion_frames

    def _log_tail(self) -> str:
        """Returns the last error lines of FFmpeg."""
        try:
            log_text = (self.folder / TaskFiles.ENCODER_LOG_NAME).read_text(errors="replace")
        except OSError:
            return ""

        detail = log_text.strip()[-FFmpeg.ERROR_TAIL_CHARS :]

        return f"; FFmpeg: {detail}" if detail else ""
