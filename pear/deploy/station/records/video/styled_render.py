import json
import time
from pathlib import Path
from typing import Any

from pear.deploy.station.records.recording.encode_worker import FINISHED_STATUSES, SegmentStatus
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, FFmpeg
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.sources import SourceClip


class StyledRender:
    """The styled full and motion-only videos of one part of a task, rendered from the cameras' own videos of its
    finished segments: one segment's, or the task's segments joined."""

    def __init__(self, files: TaskFiles, manifests: list[dict[str, Any]], part: str) -> None:
        # ``part`` is a segment's id, whose videos are in that segment's folder, or MERGED_FOLDER for the task's
        # segments joined; ``manifests`` are the task's segments
        self.files = files
        if part == TaskFiles.MERGED_FOLDER:
            self.folder = files.merged_directory
        else:
            self.folder = files.styled_directory(part)
            manifests = [manifest for manifest in manifests if manifest.get("id") == part]

        self.manifests: list[dict[str, Any]] = []  # the finished segments that have videos, in the order of their ids
        self.clips: list[SourceClip] = []  # their videos
        self.skipped_segments: list[dict[str, str]] = []  # the finished segments left out, each with the reason
        self.pending_id: str | None = None  # the first segment that is not finished yet
        self._plan(manifests)

    def status(self) -> dict[str, Any]:
        """Returns the rendered videos, if there are any, and whether they still cover exactly the finished
        segments."""
        available_ids = [manifest["id"] for manifest in self.manifests]

        try:
            merged_manifest = json.loads((self.folder / TaskFiles.MERGED_MANIFEST_NAME).read_text())
            # joined videos whose manifest does not call them styled are of another kind: they count as outdated
            is_styled = merged_manifest.get("styled")
            joined_ids = (
                [entry["id"] for entry in merged_manifest["segments"][TaskFiles.FULL_KIND]] if is_styled else None
            )
            if not (self.folder / TaskFiles.video_name(TaskFiles.FULL_KIND)).is_file():
                raise ValueError("Merged full video is missing")
        except (OSError, ValueError, KeyError, TypeError):
            merged_manifest = joined_ids = None

        return {
            "segments": {
                TaskFiles.FULL_KIND: available_ids,
                TaskFiles.MOTION_KIND: [manifest["id"] for manifest in self.manifests if manifest.get("motion")],
            },
            "pending": self.pending_id,
            # a task that was resumed has one more segment, which an earlier render does not cover
            "current": joined_ids == available_ids,
            "merged": merged_manifest,
        }

    def check(self) -> None:
        """Refuses a render while a segment is not finished, and when no segment has videos."""
        if self.pending_id is not None:
            raise RuntimeError(
                f"Segment {int(self.pending_id)} is still recording or processing; merge after it finishes"
            )
        if not self.manifests:
            raise ValueError("No finished segment has camera sources to merge")

    def publish(self, render_result: dict[str, Any]) -> None:
        """Checks the frame counts of the rendered videos, then puts them in place and writes their manifest."""
        frame_counts = {
            TaskFiles.FULL_KIND: render_result["frames"],
            TaskFiles.MOTION_KIND: render_result["motion_frames"],
        }
        executable = FFmpeg.executable()
        for kind, frames in frame_counts.items():
            if frames and (found_frames := FFmpeg.packet_count(executable, self._partial_path(kind))) != frames:
                raise RuntimeError(f"Merged {kind} video has {found_frames} frames; {frames} were rendered")

        for kind, frames in frame_counts.items():
            if frames:
                self._partial_path(kind).replace(self.folder / TaskFiles.video_name(kind))
            else:
                (self.folder / TaskFiles.video_name(kind)).unlink(missing_ok=True)

        segments, skipped_segments = self._segments(render_result["clip_motion_frames"])
        full_frames, motion_frames = frame_counts[TaskFiles.FULL_KIND], frame_counts[TaskFiles.MOTION_KIND]
        merged_manifest = {
            "created": time.time(),
            "styled": True,
            "full": TaskFiles.video_name(TaskFiles.FULL_KIND),
            "frames": full_frames,
            "duration_s": full_frames / FRAME_RATE,
            "motion": TaskFiles.video_name(TaskFiles.MOTION_KIND) if motion_frames else None,
            "motion_frames": motion_frames,
            "motion_duration_s": motion_frames / FRAME_RATE,
            # start_frame says where each segment begins in the rendered video
            "segments": segments,
            "skipped": skipped_segments,
        }

        TaskFiles.write_json(self.folder / TaskFiles.MERGED_MANIFEST_NAME, merged_manifest)

    def discard(self) -> None:
        """Removes the videos that were rendered and not published; the published ones and every segment stay."""
        for kind in TaskFiles.VIDEO_KINDS:
            self._partial_path(kind).unlink(missing_ok=True)

    def _plan(self, manifests: list[dict[str, Any]]) -> None:
        """Sorts the segments into those to render, those left out and the first that is not finished."""
        # a manifest whose id is no segment's number is passed over
        numbered_manifests = (manifest for manifest in manifests if str(manifest.get("id")).isdigit())
        for manifest in sorted(numbered_manifests, key=lambda manifest: int(manifest["id"])):
            if manifest["status"] not in FINISHED_STATUSES:
                self.pending_id = self.pending_id or manifest["id"]
            elif manifest["status"] == SegmentStatus.FAILED:
                self.skipped_segments.append({"id": manifest["id"], "reason": "recording failed"})
            elif (clip := SourceClip.of_segment(self.files.segment_directory(manifest["id"]), manifest)) is None:
                self.skipped_segments.append({"id": manifest["id"], "reason": "camera sources missing"})
            else:
                self.manifests.append(manifest)
                self.clips.append(clip)

    def _segments(self, clip_motion_frames: list[int]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        """Returns where each segment begins in each rendered video, and the segments each video leaves out."""
        segments: dict[str, list[dict]] = {kind: [] for kind in TaskFiles.VIDEO_KINDS}
        skipped_segments = {
            TaskFiles.FULL_KIND: self.skipped_segments,
            TaskFiles.MOTION_KIND: list(self.skipped_segments),
        }
        start_frames = dict.fromkeys(TaskFiles.VIDEO_KINDS, 0)

        for manifest, motion_frames in zip(self.manifests, clip_motion_frames):
            kind_frames = ((TaskFiles.FULL_KIND, manifest["frames"]), (TaskFiles.MOTION_KIND, motion_frames))
            for kind, frames in kind_frames:
                if frames:
                    segments[kind].append({"id": manifest["id"], "start_frame": start_frames[kind], "frames": frames})
                    start_frames[kind] += frames
            if not motion_frames:
                skipped_segments[TaskFiles.MOTION_KIND].append({"id": manifest["id"], "reason": "no motion"})

        skipped_segments[TaskFiles.MOTION_KIND].sort(key=lambda entry: int(entry["id"]))

        return segments, skipped_segments

    def _partial_path(self, kind: str) -> Path:
        """Returns where the video of ``kind`` is while it is rendered and not published."""
        return self.folder / TaskFiles.partial_name(TaskFiles.video_name(kind))
