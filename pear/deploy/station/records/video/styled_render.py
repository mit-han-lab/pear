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
        """Returns the rendered videos, if there are any, whether they still cover exactly the finished segments,
        and whether the render that made them finished both."""
        available_ids = [manifest["id"] for manifest in self.manifests]

        try:
            merged_manifest = json.loads((self.folder / TaskFiles.MERGED_MANIFEST_NAME).read_text())
            # joined videos whose manifest does not call them styled are of another kind: they count as outdated
            is_styled = merged_manifest.get("styled")
            joined_ids = (
                [entry["id"] for entry in merged_manifest["segments"][TaskFiles.FULL_KIND]] if is_styled else None
            )
            for kind in TaskFiles.VIDEO_KINDS:
                if merged_manifest[kind] and not (self.folder / merged_manifest[kind]).is_file():
                    raise ValueError(f"Merged {kind} video is missing")
        except (OSError, ValueError, KeyError, TypeError):
            merged_manifest = joined_ids = None

        # a task that was resumed has one more segment, which an earlier render does not cover
        is_current = joined_ids == available_ids

        return {
            "segments": {
                TaskFiles.FULL_KIND: available_ids,
                TaskFiles.MOTION_KIND: [
                    manifest["id"] for manifest in self.manifests if manifest.get("motion_intervals")
                ],
            },
            "pending": self.pending_id,
            "current": is_current,
            # the motion-only video is published first: the full one ends the render
            "complete": is_current and bool(merged_manifest[TaskFiles.FULL_KIND]),
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

    def kinds(self) -> tuple[str, ...]:
        """Returns the videos a render makes, in the order it publishes them: the motion-only one first, when
        anything moved."""
        has_motion = any(manifest.get("motion_intervals") for manifest in self.manifests)

        return (TaskFiles.MOTION_KIND, TaskFiles.FULL_KIND) if has_motion else (TaskFiles.FULL_KIND,)

    def publish(self, render_results: dict[str, dict[str, Any]]) -> None:
        """Checks the frame count of the video rendered last and puts it in place, then writes the manifest of the
        videos the render has made so far; a video it has not made is removed."""
        executable = FFmpeg.executable()
        for kind, render_result in render_results.items():
            partial_path = self.partial_path(kind)
            if partial_path.is_file():
                if (found_frames := FFmpeg.packet_count(executable, partial_path)) != render_result["frames"]:
                    raise RuntimeError(
                        f"Merged {kind} video has {found_frames} frames; {render_result['frames']} were rendered"
                    )
                partial_path.replace(self.folder / TaskFiles.video_name(kind))
        for kind in TaskFiles.VIDEO_KINDS:
            if kind not in render_results:
                (self.folder / TaskFiles.video_name(kind)).unlink(missing_ok=True)

        full_frames = render_results.get(TaskFiles.FULL_KIND, {}).get("frames")
        motion_frames = render_results.get(TaskFiles.MOTION_KIND, {}).get("frames", 0)
        segments, skipped_segments = self._segments(render_results.get(TaskFiles.MOTION_KIND))
        merged_manifest = {
            "created": time.time(),
            "styled": True,
            "full": TaskFiles.video_name(TaskFiles.FULL_KIND) if full_frames else None,
            "frames": full_frames,
            "duration_s": full_frames / FRAME_RATE if full_frames else None,
            "motion": TaskFiles.video_name(TaskFiles.MOTION_KIND) if motion_frames else None,
            "motion_frames": motion_frames,
            "motion_duration_s": motion_frames / FRAME_RATE,
            # start_frame says where each segment begins in the rendered video
            "segments": segments,
            "skipped": skipped_segments,
        }

        TaskFiles.write_json(self.folder / TaskFiles.MERGED_MANIFEST_NAME, merged_manifest)

    def discard(self) -> None:
        """Removes what a render left unfinished, its chunks too; the published videos and every segment stay."""
        for path in self.folder.glob("*"):
            if any(marker in path.name for marker in TaskFiles.UNFINISHED_MARKERS):
                path.unlink(missing_ok=True)

    def partial_path(self, kind: str) -> Path:
        """Returns where the video of ``kind`` is while it is rendered and not published."""
        return self.folder / TaskFiles.partial_name(TaskFiles.video_name(kind))

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

    def _segments(self, motion_result: dict[str, Any] | None) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        """Returns where each segment begins in each video, and the segments each video leaves out; the frames of the
        full video come from the segments' manifests, those of the motion-only one from its render, if it ran."""
        segments: dict[str, list[dict]] = {kind: [] for kind in TaskFiles.VIDEO_KINDS}
        skipped_segments = {
            TaskFiles.FULL_KIND: self.skipped_segments,
            TaskFiles.MOTION_KIND: list(self.skipped_segments),
        }
        start_frames = dict.fromkeys(TaskFiles.VIDEO_KINDS, 0)
        clip_motion_frames = motion_result["clip_frames"] if motion_result else [0] * len(self.manifests)

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
