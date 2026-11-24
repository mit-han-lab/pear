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
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, SOURCE_VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.task_files import TaskFiles
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SegmentStatus(StrEnum):
    """Where a recording segment stands, each by the word its manifest holds."""

    STARTING = "starting"  # its encoders start
    RECORDING = "recording"  # the cameras are sampled
    PROCESSING = "processing"  # it was stopped, and its videos are finished
    SAVED = "saved"  # its videos are complete
    FAILED = "failed"  # it ended without them


FINISHED_STATUSES = (SegmentStatus.SAVED, SegmentStatus.FAILED)  # the statuses of a segment that changes no more


class EncodeWorker:
    """The spawned process of one recording segment: it runs the encoders of the cameras' own videos and writes the
    segment's timeline beside them."""

    NICENESS = 5  # so that encoding never competes with the control loop
    PACKET_TIMEOUT_S = 15  # without a packet for this long, the sampler is taken as gone
    ENCODER_TIMEOUT_S = 60  # for an encoder to exit once its input is closed
    SOURCE_CRF = 17  # x264 quality of the cameras' own videos
    BLANK_FRAME = bytes(SOURCE_VIDEO_SIZE[0] * SOURCE_VIDEO_SIZE[1] * 3)  # one black camera picture
    # the commit of the code this process imported, "-dirty" marking local changes; the manifest's code_version
    CODE_VERSION_COMMAND = ("git", "describe", "--always", "--dirty", "--abbrev=12")
    CODE_VERSION_TIMEOUT_S = 2  # for git to name the commit

    def __init__(
        self,
        directory: Path,
        cameras: tuple[Camera, ...],
        packets: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        self.directory = directory  # the segment's folder
        self.packets = packets  # from the sampler: camera frames with the robot's state, then the end
        self.messages = messages  # to the segment: what its manifest takes over
        self.executable = executable

        self.log: IO[str] | None = None
        self.children: list[subprocess.Popen] = []  # every FFmpeg process this worker started
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
        cameras: tuple[Camera, ...],
        packets: multiprocessing.queues.Queue,
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        """Runs the worker of one segment in the process that was spawned for it."""
        cls(Path(directory), cameras, packets, messages, executable).run()

    def run(self) -> None:
        """Encodes the segment's packets up to its end, and reports every step to the segment."""
        os.setsid()
        os.nice(self.NICENESS)

        try:
            self._open()
            self._consume()
            self._finish()
        except BaseException as error:
            logger.exception("The recording in %s failed", self.directory)
            self.messages.put({"status": SegmentStatus.FAILED.value, "error": f"{type(error).__name__}: {error}"})
        finally:
            self._close()

    def _open(self) -> None:
        """Starts the encoder of each camera's own video, and reports that they are open."""
        self.log = (self.directory / TaskFiles.ENCODER_LOG_NAME).open("w")
        for role, name in self.source_file_names.items():
            output_path = self.directory / TaskFiles.partial_name(name)
            self.source_encoders[role] = FFmpeg.raw_encoder(
                self.executable, SOURCE_VIDEO_SIZE, output_path, self.log, self.SOURCE_CRF
            )
            self.children.append(self.source_encoders[role])

        self.messages.put({"ready": True, "code_version": self._code_version()})

    def _consume(self) -> None:
        """Writes the frames of every packet until the end packet, and the timeline beside them."""
        with (self.directory / TaskFiles.TIMELINE_NAME).open("w") as timeline_file:
            while True:
                packet = self.packets.get(timeout=self.PACKET_TIMEOUT_S)

                # a change of state reaches the timeline also when its camera sample was dropped: the previous
                # picture stays, and its place in the motion changes on time
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

        for _ in range(start_index, end_index):
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

    def _finish(self) -> None:
        """Ends the encoders of the cameras' own videos, puts their files in place and reports the saved segment."""
        # every input is closed first, so that the encoders flush side by side
        for role in list(self.source_encoders):
            try:
                self.source_encoders[role].stdin.close()
            except OSError as error:
                self._drop_source(role, error)

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

        # the cameras' videos are what a segment keeps: one camera's may be lost, all of them may not
        if not self.source_encoders:
            raise RuntimeError("; ".join(self.source_errors))

        self.messages.put(
            {
                "status": SegmentStatus.SAVED.value,
                "frames": self.next_index,
                "duration_s": self.next_index / FRAME_RATE,
                "sources": {
                    role: name if role in self.source_encoders else None
                    for role, name in self.source_file_names.items()
                },
                "source_error": "; ".join(self.source_errors) or None,
                "motion_intervals": self.motion_intervals,
            }
        )

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
