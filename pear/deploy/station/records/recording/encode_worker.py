import json
import math
import multiprocessing.queues
import os
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from PIL import Image

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, SOURCE_VIDEO_SIZE, VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.frame_renderer import CaptionedFrames
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.task_files import TaskFiles
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SegmentStatus(StrEnum):
    """Where a recording segment stands, each by the word its manifest holds."""

    STARTING = "starting"  # its encoder starts
    RECORDING = "recording"  # the cameras are sampled
    PROCESSING = "processing"  # it was stopped, and its videos are finished
    SAVED = "saved"  # its videos are complete
    FAILED = "failed"  # it ended without them


FINISHED_STATUSES = (SegmentStatus.SAVED, SegmentStatus.FAILED)  # the statuses of a segment that changes no more


class EncodeWorker:
    """The spawned process of one recording segment: it draws the frames, runs the FFmpeg encoders and writes the
    segment's files."""

    NICENESS = 5  # so that encoding never competes with the control loop
    PACKET_TIMEOUT_S = 15  # without a packet for this long, the sampler is taken as gone
    ENCODER_TIMEOUT_S = 60  # for an encoder to exit once its input is closed
    MOTION_MIN_TIMEOUT_S = 120  # the shortest wait for the motion-only encoder
    MOTION_TIMEOUT_FACTOR = 3  # of the video's duration, for the motion-only encoder
    SOURCE_CRF = 17  # x264 quality of the cameras' own videos
    BLANK_FRAME = bytes(SOURCE_VIDEO_SIZE[0] * SOURCE_VIDEO_SIZE[1] * 3)  # one black camera picture
    # the commit of the code this process imported, "-dirty" marking local changes; the manifest's code_version
    CODE_VERSION_COMMAND = ("git", "describe", "--always", "--dirty", "--abbrev=12")
    CODE_VERSION_TIMEOUT_S = 2  # for git to name the commit

    def __init__(
        self,
        directory: Path,
        task: dict[str, Any],
        cameras: tuple[Camera, ...],
        packets: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        self.directory = directory  # the segment's folder
        self.task = task
        self.cameras = cameras
        self.packets = packets  # from the sampler: camera frames with the state to caption, then the end
        self.messages = messages  # to the segment: what its manifest takes over
        self.executable = executable

        self.log: IO[str] | None = None
        self.children: list[subprocess.Popen] = []  # every FFmpeg process this worker started
        self.full_video_path = directory / TaskFiles.partial_name(TaskFiles.video_name(TaskFiles.FULL_KIND))
        self.full_encoder: subprocess.Popen | None = None
        self.captioned_frames: CaptionedFrames | None = None
        self.source_file_names = TaskFiles.source_file_names(cameras)
        self.source_encoders: dict[str, subprocess.Popen] = {}  # camera role -> the encoder of its own video
        self.source_errors: list[str] = []
        self.previous_packet: dict[str, Any] | None = None  # whose pictures the frames up to the next packet show
        self.next_index = 0  # of the next frame to write
        self.motion_intervals: list[list[int]] = []  # [first frame, frame after the last] of each run in motion

    @classmethod
    def main(
        cls,
        directory: str,
        task: dict[str, Any],
        cameras: tuple[Camera, ...],
        packets: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        """Runs the worker of one segment in the process that was spawned for it."""
        cls(Path(directory), task, cameras, packets, messages, executable).run()

    def run(self) -> None:
        """Encodes the segment's packets up to its end, and reports every step to the segment."""
        os.setsid()
        os.nice(self.NICENESS)

        try:
            self._open()
            self._consume()
            self._finish_full()
            self._encode_motion()
            self.messages.put(
                {
                    "status": SegmentStatus.SAVED.value,
                    "motion": TaskFiles.video_name(TaskFiles.MOTION_KIND) if self.motion_intervals else None,
                    "motion_duration_s": sum(end - start for start, end in self.motion_intervals) / FRAME_RATE,
                    "motion_intervals": self.motion_intervals,
                    "no_motion": not self.motion_intervals,
                }
            )
        except BaseException as error:
            logger.exception("The recording in %s failed", self.directory)
            self.messages.put({"status": SegmentStatus.FAILED.value, "error": f"{type(error).__name__}: {error}"})
        finally:
            self._close()

    def _open(self) -> None:
        """Starts the encoders of the full video and of each camera's own, and reports that they are open."""
        self.log = (self.directory / TaskFiles.ENCODER_LOG_NAME).open("w")
        self.full_encoder = FFmpeg.raw_encoder(self.executable, VIDEO_SIZE, self.full_video_path, self.log)
        self.children.append(self.full_encoder)

        # the cameras' own videos are extras: one that fails is dropped without failing the full video
        for role, name in self.source_file_names.items():
            output_path = self.directory / TaskFiles.partial_name(name)
            self.source_encoders[role] = FFmpeg.raw_encoder(
                self.executable, SOURCE_VIDEO_SIZE, output_path, self.log, self.SOURCE_CRF
            )
            self.children.append(self.source_encoders[role])

        self.captioned_frames = CaptionedFrames(self.task, self.cameras)
        self.messages.put({"ready": True, "code_version": self._code_version()})

    def _consume(self) -> None:
        """Writes the frames of every packet until the end packet, and the timeline beside them."""
        with (self.directory / TaskFiles.TIMELINE_NAME).open("w") as timeline_file:
            while True:
                packet = self.packets.get(timeout=self.PACKET_TIMEOUT_S)

                # a change of state reaches the video also when its camera sample was dropped: the previous picture
                # stays, and its caption and its place in the motion change on time
                for update in packet.get("updates", []):
                    self._write_until(update["index"], timeline_file)
                    if self.previous_packet is not None:
                        self.previous_packet = {**self.previous_packet, "state": update["state"]}

                if "end" in packet:
                    self._write_until(max(1, math.ceil(packet["end"] * FRAME_RATE)), timeline_file)
                    return

                if self.previous_packet is None:
                    self.previous_packet = packet
                self._write_until(packet["index"], timeline_file)
                self.previous_packet = packet

    def _write_until(self, end_index: int, timeline_file: IO[str]) -> None:
        """Writes the frames before ``end_index`` that are not written yet, all with the previous packet's
        pictures."""
        start_index = self.next_index
        if self.previous_packet is None or end_index <= start_index:
            return

        frames = self.previous_packet["frames"]
        pixels = {role: self._camera_pixels(frames.get(role)) for role in self.source_encoders}

        for frame_index in range(start_index, end_index):
            frame_image = self.captioned_frames.render(self.previous_packet, frame_index)
            self.full_encoder.stdin.write(frame_image.tobytes())
            for role in list(self.source_encoders):
                try:
                    self.source_encoders[role].stdin.write(pixels[role])
                except OSError as error:
                    self._drop_source(role, error)

        self.next_index = end_index
        self._log_interval(start_index, end_index, timeline_file)

    def _log_interval(self, start_index: int, end_index: int, timeline_file: IO[str]) -> None:
        """Adds a run of frames to the motion intervals when the robot moved in it, and to the timeline."""
        if self.previous_packet["state"]["moving"]:
            if self.motion_intervals and self.motion_intervals[-1][1] == start_index:
                self.motion_intervals[-1][1] = end_index
            else:
                self.motion_intervals.append([start_index, end_index])

        timeline_row = {
            "start_frame": start_index,
            "end_frame": end_index,
            "time_s": start_index / FRAME_RATE,
            "captured_monotonic_s": self.previous_packet["captured"],
            "state": self.previous_packet["state"],
            # of each camera's frame, when it arrived and its number, under the keys the timeline file has
            "cameras": {
                role: {"received_monotonic": frame["received_monotonic_s"], "frame_number": frame["frame_number"]}
                for role, frame in self.previous_packet["frames"].items()
            },
            "camera_errors": self.previous_packet["errors"],
        }
        timeline_file.write(json.dumps(timeline_row, allow_nan=False) + "\n")

    def _finish_full(self) -> None:
        """Ends the encoders of the full video and of the cameras' own, puts their files in place and reports them."""
        # every input is closed first, so that the cameras' encoders flush beside the full video's
        self.full_encoder.stdin.close()
        for role in list(self.source_encoders):
            try:
                self.source_encoders[role].stdin.close()
            except OSError as error:
                self._drop_source(role, error)

        if self.full_encoder.wait(timeout=self.ENCODER_TIMEOUT_S) != 0:
            raise RuntimeError(f"Full video encoder failed; see {TaskFiles.ENCODER_LOG_NAME}")
        full_video_name = TaskFiles.video_name(TaskFiles.FULL_KIND)
        self.full_video_path.replace(self.directory / full_video_name)

        for role in list(self.source_encoders):
            try:
                exit_code = self.source_encoders[role].wait(timeout=self.ENCODER_TIMEOUT_S)
            except subprocess.TimeoutExpired as error:
                self._drop_source(role, error)
                continue
            if exit_code != 0:
                error = RuntimeError(f"encoder exited with {exit_code}; see {TaskFiles.ENCODER_LOG_NAME}")
                self._drop_source(role, error)
                continue
            name = self.source_file_names[role]
            (self.directory / TaskFiles.partial_name(name)).replace(self.directory / name)

        self.messages.put(
            {
                "full": full_video_name,
                "frames": self.next_index,
                "duration_s": self.next_index / FRAME_RATE,
                "sources": {
                    role: name if role in self.source_encoders else None
                    for role, name in self.source_file_names.items()
                },
                "source_error": "; ".join(self.source_errors) or None,
                "status": SegmentStatus.PROCESSING.value,
            }
        )

    def _encode_motion(self) -> None:
        """Cuts the full video to the frames in which the robot moved, when it moved at all."""
        if not self.motion_intervals:
            return

        motion_spans = [(start / FRAME_RATE, end / FRAME_RATE) for start, end in self.motion_intervals]
        filter_script = self.directory / TaskFiles.MOTION_FILTER_NAME
        filter_script.write_text(MotionCut(motion_spans, self.next_index).select_filter())

        motion_video_name = TaskFiles.video_name(TaskFiles.MOTION_KIND)
        output_path = self.directory / TaskFiles.partial_name(motion_video_name)
        full_video_path = self.directory / TaskFiles.video_name(TaskFiles.FULL_KIND)
        encoder = FFmpeg.motion_encoder(self.executable, full_video_path, filter_script, output_path, self.log)
        self.children.append(encoder)
        timeout_s = max(self.MOTION_MIN_TIMEOUT_S, self.next_index / FRAME_RATE * self.MOTION_TIMEOUT_FACTOR)
        if encoder.wait(timeout=timeout_s) != 0:
            raise RuntimeError("Motion video encoder failed; full video is available")

        output_path.replace(self.directory / motion_video_name)
        filter_script.unlink()

    def _drop_source(self, role: str, error: Exception) -> None:
        """Gives up one camera's own video: its encoder is ended and its unfinished file removed."""
        encoder = self.source_encoders.pop(role)
        name = self.source_file_names[role]
        self.source_errors.append(f"{name}: {type(error).__name__}: {error}")
        # collected at the worker's end, with the other FFmpeg processes
        if encoder.poll() is None:
            encoder.kill()
        (self.directory / TaskFiles.partial_name(name)).unlink(missing_ok=True)

    def _close(self) -> None:
        """Ends every FFmpeg process that still runs, and closes their log."""
        for child in self.children:
            FFmpeg.kill(child)
        if self.log is not None:
            self.log.close()

    @classmethod
    def _code_version(cls) -> str | None:
        """Returns the commit of the code this process imported, None outside a checkout of it."""
        try:
            result = subprocess.run(
                cls.CODE_VERSION_COMMAND,
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=cls.CODE_VERSION_TIMEOUT_S,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        return result.stdout.strip() or None

    @classmethod
    def _camera_pixels(cls, frame: dict[str, Any] | None) -> bytes:
        """Returns a camera's picture without a label, at the size of its own video; black for a camera whose
        frame is missing."""
        if frame is None:
            return cls.BLANK_FRAME

        picture = frame["rgb"]
        if picture.shape[:2] != SOURCE_VIDEO_SIZE[::-1]:
            picture = Image.fromarray(picture).resize(SOURCE_VIDEO_SIZE)

        return picture.tobytes()
