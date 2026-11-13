import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from pear.deploy.station.agent_loop.schemas import TurnEvent, TurnPhase
from pear.deploy.station.records.recording.encode_worker import FINISHED_STATUSES, EncodeWorker, SegmentStatus
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, SOURCE_VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.segment import Segment
from pear.deploy.station.records.task_events import TaskEventKind, TaskEvents
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.render_job import RenderedRecordings
from pear.deploy.station.records.video.render_worker import RenderStatus
from pear.deploy.station.records.video.sources import SourceClip
from waiting import Waiting


class StandInSegment:
    """A recording segment that is still processing, as the recordings and their renders read one."""

    def __init__(self, task_id: str, segment_id: str) -> None:
        self.manifest = {"id": segment_id, "task_id": task_id, "status": SegmentStatus.PROCESSING.value}

    def snapshot(self) -> dict[str, Any]:
        """Returns the manifest as it stands."""
        return dict(self.manifest)

    def stop(self, reason: str, *, stopped_at: float | None) -> None:
        """Takes the stop of the segment, which goes on processing."""


class RecordedTask:
    """A task's folder as the station leaves it, written by a test: the task's record, its events and its recording
    segments, and what a segment wrote, read back."""

    # a task as the console starts it, with what its recording reads
    TASK = {
        "id": "a" * 32,
        "instruction": "Lift both arms, then hold.",
        "model": "vllm",
        "model_label": "Example Vision Model 27B (long caption label)",
    }
    # task folders the imported station wrote, one for each robot: a reader of today must still read them
    STORED_TASKS = Path(__file__).resolve().parent / "stored_tasks"
    STORED_ROW_ROBOT = "yam"  # the robot of the stored task whose cameras are StandInCameras.ROW
    # each camera's own video in a segment of StandInCameras.ROW, by role
    SOURCE_FILES = {"left": "left_wrist.mp4", "top": "head.mp4", "right": "right_wrist.mp4"}
    # a segment given no start begins at FIRST_START_S on the monotonic clock, and START_STEP_S later for each number
    FIRST_START_S = 1000.0
    START_STEP_S = 100.0
    # between the grey levels of the first frames of two cameras, so that each camera's video is its own; a level
    # rises by one at each frame, from 0 again past 255
    CAMERA_GREY_STEP = 60
    GREY_LEVELS = 256  # of a byte
    ENCODER_TIMEOUT_S = 60  # for FFmpeg to finish a camera's video
    EPISODE_ID = "episode-1"  # of the model session of the turns a test writes

    def __init__(self, directory: Path) -> None:
        self.files = TaskFiles(directory)
        self.turn_fields: dict[str, dict[str, Any]] = {}  # what each event of a turn carries, by attempt id

    @classmethod
    def create(cls, directory: Path, task: dict[str, Any]) -> "RecordedTask":
        """Returns a task's folder with the task's record and its first event."""
        recorded_task = cls(directory)
        directory.mkdir(parents=True, exist_ok=True)
        TaskFiles.write_json(recorded_task.files.task_path, task)
        recorded_task.append_event(TaskEventKind.TASK.value, task["created"], task)

        return recorded_task

    @classmethod
    def stored_robots(cls) -> list[str]:
        """Returns the robots whose stored task folders the tests read, sorted."""
        return sorted(path.name for path in cls.STORED_TASKS.iterdir() if path.is_dir())

    def copy_stored_task(self) -> float:
        """Copies in the records of the stored task of StandInCameras.ROW and returns its first event's time on the
        monotonic clock."""
        stored_task = next(path for path in (self.STORED_TASKS / self.STORED_ROW_ROBOT).iterdir() if path.is_dir())
        shutil.copytree(stored_task, self.files.directory, dirs_exist_ok=True)

        return next(TaskEvents(self.files.events_path).read())["monotonic_s"]

    def start_turn(self, attempt_id: str, number: int, monotonic_s: float) -> None:
        """Writes the start of a turn and what its model was shown, as the console journals them."""
        self.turn_fields[attempt_id] = {
            "attempt": attempt_id,
            "turn_number": number,
            "started_at": monotonic_s,
            "episode_id": self.EPISODE_ID,
            "phase": TurnPhase.OBSERVING.value,
        }
        self.append_event(TaskEventKind.TURN_STARTED.value, monotonic_s, attempt_id=attempt_id)
        self.append_event(TurnEvent.OBSERVATION.value, monotonic_s, {"turn_id": attempt_id}, attempt_id)

    def append_event(
        self, kind: str, monotonic_s: float, data: dict[str, Any] | None = None, attempt_id: str | None = None
    ) -> None:
        """Writes one event of the task, with the fields of the turn ``attempt_id`` when it is given."""
        event_data = {**self.turn_fields.get(attempt_id, {}), **(data or {})}
        TaskEvents(self.files.events_path).append({"type": kind, "data": event_data, "monotonic_s": monotonic_s})

    def add_segment(
        self,
        number: int,
        frame_count: int,
        motion_frame_count: int = 0,
        *,
        status: str = SegmentStatus.SAVED.value,
        started_monotonic_s: float | None = None,
        grey_levels: dict[str, list[int]] | None = None,
    ) -> Path:
        """Writes a finished segment and returns its folder: its manifest and, when it was saved, each camera's own
        video, of the flat grey frames of ``grey_levels`` by role or, without them, of every camera."""
        directory = self.files.segment_directory(f"{number:0{TaskFiles.SEGMENT_ID_DIGITS}d}")
        directory.mkdir(parents=True)
        is_saved = status == SegmentStatus.SAVED
        if started_monotonic_s is None:
            started_monotonic_s = self.FIRST_START_S + self.START_STEP_S * number
        if grey_levels is None:
            grey_levels = {
                role: [(position * self.CAMERA_GREY_STEP + index) % self.GREY_LEVELS for index in range(frame_count)]
                for position, role in enumerate(self.SOURCE_FILES)
            }

        for role, levels in grey_levels.items() if is_saved else ():
            self._write_video(directory / self.SOURCE_FILES[role], levels)

        manifest = {
            "id": directory.name,
            "status": status,
            "fps": FRAME_RATE,
            "frames": frame_count,
            "started_monotonic_s": started_monotonic_s,
            "full": TaskFiles.video_name(TaskFiles.FULL_KIND) if is_saved else None,
            "sources": {role: self.SOURCE_FILES[role] for role in grey_levels} if is_saved else None,
            "motion": TaskFiles.video_name(TaskFiles.MOTION_KIND) if motion_frame_count else None,
            "motion_intervals": [[0, motion_frame_count]] if motion_frame_count else [],
            "no_motion": is_saved and not motion_frame_count,
        }
        TaskFiles.write_json(directory / TaskFiles.MANIFEST_NAME, manifest)

        return directory

    def segment_clip(
        self, number: int, start_monotonic_s: float, duration_s: float, motion_intervals: list[list[int]]
    ) -> SourceClip:
        """Writes the manifest of a saved segment whose videos the test does not read, with ``motion_intervals`` ([first
        frame, frame after the last] of each run in motion), and returns the clip of one camera's video."""
        directory = self.files.segment_directory(f"{number:0{TaskFiles.SEGMENT_ID_DIGITS}d}")
        directory.mkdir(parents=True)
        manifest = {
            "id": directory.name,
            "status": SegmentStatus.SAVED.value,
            "fps": FRAME_RATE,
            "started_monotonic_s": start_monotonic_s,
            "duration_s": duration_s,
            "motion_intervals": motion_intervals,
        }
        TaskFiles.write_json(directory / TaskFiles.MANIFEST_NAME, manifest)

        return SourceClip(start_monotonic_s, duration_s, {"top": directory / self.SOURCE_FILES["top"]})

    @staticmethod
    def timeline_rows(segment_directory: Path) -> list[dict[str, Any]]:
        """Returns the rows of the timeline a recording segment wrote, in order."""
        timeline_text = (segment_directory / TaskFiles.TIMELINE_NAME).read_text()

        return [json.loads(line) for line in timeline_text.splitlines()]

    @classmethod
    def _write_video(cls, path: Path, grey_levels: list[int]) -> None:
        """Encodes a camera's own video as the recorder does, one flat grey frame for each of ``grey_levels``."""
        encoder = FFmpeg.raw_encoder(
            FFmpeg.executable(), SOURCE_VIDEO_SIZE, path, subprocess.DEVNULL, EncodeWorker.SOURCE_CRF
        )
        width, height = SOURCE_VIDEO_SIZE
        for grey_level in grey_levels:
            encoder.stdin.write(np.full((height, width, 3), grey_level, np.uint8).tobytes())

        encoder.stdin.close()
        assert encoder.wait(timeout=cls.ENCODER_TIMEOUT_S) == 0


class RecordingWaits:
    """A test's waits for what the recordings do in their own threads and workers."""

    SAVE_TIMEOUT_S = 30  # for a segment to leave the recording and processing statuses
    RENDER_TIMEOUT_S = 90  # for a render to finish its videos
    POLL_S = 0.01  # between two looks

    @classmethod
    def await_finished(cls, segment: Segment) -> None:
        """Returns once the segment is saved or failed."""
        Waiting.until(lambda: segment.snapshot()["status"] in FINISHED_STATUSES, cls.SAVE_TIMEOUT_S, cls.POLL_S)

    @classmethod
    def await_merged(cls, recordings: RenderedRecordings, task_id: str) -> dict[str, Any]:
        """Returns the status of a task's joined videos once no render of them waits or runs."""
        Waiting.until(
            lambda: (recordings.merged_status(task_id)["render"] or {}).get("status") in (None, RenderStatus.FAILED),
            cls.RENDER_TIMEOUT_S,
            cls.POLL_S,
        )

        return recordings.merged_status(task_id)

    @classmethod
    def await_styled(cls, recordings: RenderedRecordings, task_id: str, segment_id: str) -> dict[str, Any]:
        """Returns the row of a segment among its task's segments once no render of its videos waits or runs."""
        Waiting.until(
            lambda: (
                (cls._segment_row(recordings, task_id, segment_id)["render"] or {}).get("status")
                in (None, RenderStatus.FAILED)
            ),
            cls.RENDER_TIMEOUT_S,
            cls.POLL_S,
        )

        return cls._segment_row(recordings, task_id, segment_id)

    @staticmethod
    def _segment_row(recordings: RenderedRecordings, task_id: str, segment_id: str) -> dict[str, Any]:
        """Returns the row of the segment ``segment_id`` among its task's segments."""
        return next(row for row in recordings.segments(task_id) if row["id"] == segment_id)
