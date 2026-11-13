import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.context_usage import ContextUsage
from pear.deploy.station.records.recording.encode_worker import FINISHED_STATUSES, SegmentStatus
from pear.deploy.station.records.recording.segment import Segment
from pear.deploy.station.records.task_files import TaskFiles


class Recordings:
    """The recording segments of the running service: starts them, stops them and lists them."""

    MAX_PROCESSING_SEGMENTS = 2  # segments whose videos are still finished before another may start
    JOIN_S = 10  # for each segment at close, for its sampler thread to end
    KILL_JOIN_S = 3  # for the thread once its encoder was ended

    def __init__(self, root: Path, cameras: tuple[Camera, ...]) -> None:
        self.root = root  # the folder of the tasks' folders
        self.cameras = tuple(cameras)
        self.source_file_names = TaskFiles.source_file_names(self.cameras)  # by camera role, of a camera's own video
        self.is_enabled = True  # whether a task that starts or resumes is recorded
        self.active_segment: Segment | None = None  # the segment that is recording
        self.live_segments: list[Segment] = []  # the segments of this service that were not found finished yet
        self.usage_tracker = ContextUsage()

    def start(
        self, task: dict[str, Any], camera_feed: CameraSource, robot_status_reader: Callable[[], dict[str, Any]]
    ) -> None:
        """Starts the next segment of ``task``, unless recording is off."""
        latest_context = self.usage_tracker.context_for_task(task["id"])
        if not self.is_enabled:
            return

        self.live_segments = [
            segment for segment in self.live_segments if segment.snapshot()["status"] not in FINISHED_STATUSES
        ]
        if len(self.live_segments) >= self.MAX_PROCESSING_SEGMENTS:
            raise RuntimeError("Previous recordings are still processing; wait before resuming")

        files = TaskFiles(self.root / task["id"])
        numbers = [int(path.name) for path in files.recordings_directory.glob("*") if path.name.isdigit()]
        segment_id = f"{max(numbers, default=0) + 1:0{TaskFiles.SEGMENT_ID_DIGITS}d}"
        segment = Segment(
            files.segment_directory(segment_id),
            task,
            self.cameras,
            camera_feed,
            robot_status_reader,
            latest_context,
        )
        self.live_segments.append(segment)
        self.active_segment = segment

    def stop(self, reason: str, *, stopped_at: float | None = None) -> Segment | None:
        """Stops the segment that is recording, at ``stopped_at`` or now, and returns it; None when none is."""
        segment = self.active_segment
        if segment is not None:
            segment.stop(reason, stopped_at=stopped_at)
            self.active_segment = None

        return segment

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        """Follows one event of the task: its usage, and what the recording segment captions."""
        # the usage is followed while recording is off too, so that a segment started later begins with the latest
        # context and not without the model session's history
        latest_context, turn_usage = self.usage_tracker.on_event(kind, data)

        segment = self.active_segment
        if segment is not None:
            segment.on_event(kind, data, latest_context=latest_context, turn_usage=turn_usage)

    def snapshot(self) -> dict[str, Any]:
        """Returns what the page shows of the recording: whether it is on, the recording segment and the live ones."""
        segment = self.active_segment

        return {
            "enabled": self.is_enabled,
            "active": segment.snapshot() if segment else None,
            "jobs": [segment.snapshot() for segment in self.live_segments],
        }

    def manifests_of_task(self, task_id: str) -> list[dict[str, Any]]:
        """Returns the manifest of every segment of a task: the saved ones, and the live ones as they stand."""
        manifests = {manifest["id"]: manifest for manifest in self._saved_manifests(TaskFiles(self.root / task_id))}
        for segment in self.live_segments:
            if segment.manifest["task_id"] == task_id:
                manifests[segment.manifest["id"]] = segment.snapshot()

        return list(manifests.values())

    def close(self) -> None:
        """Stops the recording segment and waits for every live segment to finish its files."""
        self.stop("Service shutdown")

        for segment in self.live_segments:
            segment.thread.join(self.JOIN_S)
            if segment.thread.is_alive():
                segment.terminate()
                segment.thread.join(self.KILL_JOIN_S)

    @staticmethod
    def _saved_manifests(files: TaskFiles) -> list[dict[str, Any]]:
        """Returns the manifests on disk of a task's segments, one that was never finished as failed."""
        manifests = []
        for path in sorted(files.recordings_directory.glob(f"*/{TaskFiles.MANIFEST_NAME}")):
            try:
                manifest = json.loads(path.read_text())
                # a live segment's manifest comes from the segment: an unfinished one on disk is of a service that
                # ended, and records no more
                if manifest["status"] not in FINISHED_STATUSES:
                    manifest.update(
                        status=SegmentStatus.FAILED.value, error="Recording interrupted before finalization"
                    )
                manifests.append(manifest)
            except (OSError, ValueError, KeyError):
                continue

        return manifests
