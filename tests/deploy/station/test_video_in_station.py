import json
import os
import queue
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from imageio_ffmpeg import count_frames_and_secs, read_frames

from console_rig import ConsoleRig, StandInModel
from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.recordings import Recordings
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video import render_worker as render_worker_module
from pear.deploy.station.records.video.action_code import ActionCode
from pear.deploy.station.records.video.ink import Palette
from pear.deploy.station.records.video.render_job import RenderedRecordings
from pear.deploy.station.records.video.render_worker import RenderWorker
from pear.deploy.station.records.video.sources import SourceClip
from pear.deploy.station.records.video.styled_frames import StyledFrames, StyledLayout
from pear.deploy.station.records.video.styled_render import StyledRender
from pear.deploy.station.records.video.task_timeline import TaskTimeline, TurnSpan
from recorded_task import RecordedTask, RecordingWaits, StandInSegment
from stand_in_cameras import StandInCameras
from waiting import Waiting

SOURCES = RecordedTask.SOURCE_FILES  # each camera's own video in a segment of StandInCameras.ROW, by role
MERGED = TaskFiles.MERGED_FOLDER  # the part of a task whose render joins its segments
POLL_S = 0.2  # a render waiting for a segment has looked at it again within three of these
INSTRUCTION = "Lift both arms."  # of the task whose videos the render worker draws
PROGRESS_FRAMES = 4  # frames the render worker draws between two progress reports
FIRST_LEAD_S = 0.03  # of the first clip before the first answer
FIRST_S = 0.3  # of the first clip
SECOND_LEAD_S = 0.1  # of the second clip before the second answer
SECOND_S = 1.0  # of the second clip
# reports of a service that counts the whole context as the input of each request
PLAIN_USAGE_FIRST = {"input": 1000, "output": 200, "context_limit": 260096, "final": True}
PLAIN_USAGE_SECOND = {"input": 1500, "output": 300, "context_limit": 260096, "final": True}
PLAIN_USAGE_LAST = {"input": 2500, "output": 100, "context_limit": 260096, "final": True}
# a report of a service that measures the context apart from the request's own tokens
REQUEST_CONTEXT_USAGE = {
    "input": 2100,
    "output": 90,
    "context": {"input": 2190, "context_limit": 220000, "status": "measured"},
    "compaction": {"status": "none", "count": 0, "episode_count": 0},
    "final": True,
}
# the actions the model answered, by the turn's attempt
ACTIONS = {
    "one": {"type": "pose_delta", "arm": {"z": 0.3}},
    "two": {"type": "pose_delta", "arm": {"x": -1.0}},
    "four": {"type": "pose_delta", "gripper": {"width": 2.0}},
}
IDLE_S = 0.3  # held still before and after the motion
MOVING_S = 0.5  # of the motion between the two holds
TURNS = 2  # the stand-in model reports a full context after this many turns, which pauses the task
TASK_TIMEOUT_S = 60  # for the recorded task's steps


class StandInStyledFrames:
    """A stand-in for the styled frames of a render, which keeps each call's time and the mean of each camera's
    picture."""

    FRAME_SIZE = (320, 180)  # width and height of the frames it returns
    TILE = (64, 48)  # every camera's tile, so the sources are decoded at this size
    GREY_TOLERANCE = 3  # lossy H.264 keeps a flat grey within this many levels
    instances: list["StandInStyledFrames"] = []  # every one made; each test starts a new list

    def __init__(
        self,
        cameras: tuple[Camera, ...],
        instruction: str,
        model_label: str,
        badge: str | None,
        turns: list[TurnSpan],
        duration_s: float,
    ) -> None:
        self.header = (instruction, model_label, badge)
        self.turns, self.duration_s = turns, duration_s
        self.tiles = {camera.role: (n * self.TILE[0], 0, *self.TILE) for n, camera in enumerate(cameras)}
        self.times: list[float] = []  # of each frame drawn
        self.pictures: list[dict[str, int]] = []  # of each frame drawn, the mean grey of each camera's picture
        self.instances.append(self)

    def frame(self, images: dict[str, np.ndarray], time_s: float) -> np.ndarray:
        """Returns a black frame, and keeps ``time_s`` and the mean of each camera's picture."""
        assert all(image.shape == (self.TILE[1], self.TILE[0], 3) for image in images.values())
        self.times.append(time_s)
        self.pictures.append({role: round(float(image.mean())) for role, image in images.items()})

        return np.zeros((self.FRAME_SIZE[1], self.FRAME_SIZE[0], 3), np.uint8)

    @classmethod
    def close_to(cls, pictures: list[dict[str, int]], expected: list[dict[str, int]]) -> bool:
        """Whether the mean greys of ``pictures`` are those of ``expected``, within GREY_TOLERANCE."""
        return len(pictures) == len(expected) and all(
            picture.keys() == want.keys()
            and all(abs(picture[role] - want[role]) <= cls.GREY_TOLERANCE for role in want)
            for picture, want in zip(pictures, expected)
        )


class StyledSupport:
    """Turns, frames and pixel readers shared by the styled frame tests."""

    LENGTHS_S = (2.0, 5.0, 1.0, 8.0, 4.0)  # of the turns, which fill the video back to back
    DURATION_S = sum(LENGTHS_S)
    CAMERAS = {"row": StandInCameras.ROW, "single": StandInCameras.SINGLE}  # the cameras a frame is drawn for
    # Camera tiles (x, y, width, height): a row of three cameras, and the only camera
    TILES = {
        "row": [(40, 180, 600, 450), (660, 180, 600, 450), (1280, 180, 600, 450)],
        "single": [(40, 180, 960, 720)],
    }
    COLOURS = ((200, 30, 30), (30, 200, 30), (30, 30, 200))  # of each camera's picture, in the cameras' order
    TOLERANCE = 6  # of a colour channel, for anti-aliased and blended pixels
    SOURCE = (480, 640)  # rows and columns of a recorded camera picture
    # an action in which every number is zero
    ACTION = {"type": "pose_delta", "arm": {"x": 0.0, "y": 0.0, "z": 0.0}, "gripper": {"width": 0.0}}

    @classmethod
    def turns(cls, action=ACTION):
        starts = np.cumsum((0.0, *cls.LENGTHS_S))
        return [
            TurnSpan(n + 1, float(starts[n]), float(starts[n + 1]), action, 1000 * (n + 1))
            for n in range(len(cls.LENGTHS_S))
        ]

    @classmethod
    def frames(cls, camera_set, turns, *, instruction="Pick up the cup.", model_label="Example model"):
        cameras = cls.CAMERAS[camera_set]
        return StyledFrames(cameras, instruction, model_label, "high", turns, cls.DURATION_S)

    @classmethod
    def images(cls, camera_set):
        cameras = cls.CAMERAS[camera_set]
        return {c.role: np.full((*cls.SOURCE, 3), colour, np.uint8) for c, colour in zip(cameras, cls.COLOURS)}

    @classmethod
    def is_colour(cls, pixels, colour):
        return (np.abs(pixels.astype(int) - colour) <= cls.TOLERANCE).all(axis=-1)

    @staticmethod
    def runs(mask):
        """The lengths of the runs of True in a row of pixels."""
        edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        return list(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))

    @classmethod
    def bar(cls, picture):
        """The filled and the empty runs along the middle of the progress bar."""
        row = picture[StyledLayout.BAR_TOP + StyledFrames.BAR_HEIGHT // 2]
        empty = np.add(Palette.NIGHT, Palette.LINE_OPACITY * (255 - np.array(Palette.NIGHT)))
        return cls.runs(cls.is_colour(row, Palette.AMBER)), cls.runs(cls.is_colour(row, empty))


class TurnChecks:
    """What every list of turn spans must be."""

    @staticmethod
    def assert_ordered(turns):
        assert all(turn.start_s < turn.end_s for turn in turns)
        assert all(turn.end_s <= following.start_s for turn, following in zip(turns, turns[1:]))


class FileBytes:
    """The bytes of files, to tell whether they changed."""

    @staticmethod
    def contents(*groups: Iterable[Path]) -> dict[Path, bytes]:
        """Returns the bytes of every file of ``groups``, by path."""
        return {path: path.read_bytes() for paths in groups for path in paths}


@pytest.fixture
def stand_in_frames(monkeypatch):
    """The stand-in styled frames the render worker draws with, in the order it makes them."""
    monkeypatch.setattr(StandInStyledFrames, "instances", [])
    monkeypatch.setattr(render_worker_module, "StyledFrames", StandInStyledFrames)
    # the encoders take frames of the stand-in's size
    monkeypatch.setattr(render_worker_module, "VIDEO_SIZE", StandInStyledFrames.FRAME_SIZE)
    monkeypatch.setattr(RenderWorker, "PROGRESS_FRAMES", PROGRESS_FRAMES)
    return StandInStyledFrames.instances


@pytest.fixture
def recorded(tmp_path):
    """A journal of six turns and two clips: 100-110 s with motion at 106-108 s, and 120-130 s with motion at
    123-124 s and 127-129 s."""
    task = {
        "id": "c" * 32,
        "instruction": "Stack the cubes.",
        "created": 99.0,
        "model": "example",
        "model_label": "Example Model",
        "images": [],
    }
    records = RecordedTask.create(tmp_path / task["id"], task)
    records.append_event("session", 99.5, {"episode_id": RecordedTask.EPISODE_ID, "message": "Model session started"})

    records.start_turn("one", 1, 101.0)
    records.append_event("usage", 104.0, PLAIN_USAGE_FIRST, "one")
    records.append_event("model_output", 105.0, {"answer": "..."}, "one")
    records.append_event("action", 105.0, ACTIONS["one"], "one")
    records.append_event("execution", 108.0, {"status": "completed"}, "one")
    # Answered while the recording was paused.
    records.start_turn("two", 2, 108.0)
    records.append_event("usage", 112.0, PLAIN_USAGE_SECOND, "two")
    records.append_event("model_output", 112.5, {"answer": "..."}, "two")
    records.append_event("action", 112.5, ACTIONS["two"], "two")
    records.append_event("execution", 121.0, {"status": "completed"}, "two")
    # An answer with no action in it, and no usage report.
    records.start_turn("three", 3, 121.0)
    records.append_event("model_output", 123.0, {"answer": "no json"}, "three")
    records.append_event("execution", 123.1, {"status": "rejected"}, "three")
    records.start_turn("four", 4, 123.1)
    records.append_event("usage", 124.5, REQUEST_CONTEXT_USAGE, "four")
    records.append_event("model_output", 125.0, {"answer": "..."}, "four")
    records.append_event("action", 125.0, ACTIONS["four"], "four")
    records.append_event("execution", 126.0, {"status": "completed"}, "four")
    # Answered after the last clip ended.
    records.start_turn("five", 5, 126.0)
    records.append_event("usage", 130.5, PLAIN_USAGE_LAST, "five")
    records.append_event("model_output", 131.0, {"answer": "..."}, "five")
    # Paused before it answered.
    records.start_turn("six", 6, 131.5)
    records.append_event("turn_end", 132.0, {"phase": "Interrupted"}, "six")

    clips = [
        records.segment_clip(1, 100.0, 10.0, [[180, 240]]),
        records.segment_clip(2, 120.0, 10.0, [[90, 120], [210, 270]]),
    ]
    return records.files, clips


def test_merge_renders_segments_with_sources_and_skips_the_rest(tmp_path):
    task_id = "d" * 32
    directory = tmp_path / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    # Back to back, the segments span the recorded turns.
    task.add_segment(1, 24, 8, started_monotonic_s=started)
    task.add_segment(2, 18, started_monotonic_s=started + 24 / 30)  # Paused before any motion.
    task.add_segment(3, 12, 5, started_monotonic_s=started + 42 / 30)
    task.add_segment(4, 0, status="failed")  # Interrupted by a service restart.
    task.add_segment(5, 6, grey_levels={})  # Every camera copy failed.
    # by segment, its frames inside the motion spans: the first 8 of the first segment, the first 5 of the third
    kept = {"001": 8, "002": 0, "003": 5}
    recordings = RenderedRecordings(tmp_path, StandInCameras.ROW)
    try:
        assert recordings.merged_status(task_id) == {
            "segments": {"full": ["001", "002", "003"], "motion": ["001", "003"]},
            "pending": None,
            "current": False,
            "merged": None,
            "render": None,
        }

        recordings.request_render(task_id, MERGED)
        status = RecordingWaits.await_merged(recordings, task_id)

        assert status["current"] is True and status["render"] is None
        merged = status["merged"]
        folder = directory / "recordings" / "merged"
        assert merged["styled"] is True
        assert merged["frames"] == 54 and merged["duration_s"] == 54 / 30
        assert merged["segments"]["full"] == [
            {"id": "001", "start_frame": 0, "frames": 24},
            {"id": "002", "start_frame": 24, "frames": 18},
            {"id": "003", "start_frame": 42, "frames": 12},
        ]
        # The motion-only video keeps the frames inside the task timeline's motion spans.
        assert merged["motion_frames"] == sum(kept.values()) > 0 and merged["motion"] == "motion.mp4"
        assert [part["id"] for part in merged["segments"]["motion"]] == [n for n in kept if kept[n]]
        failed = [{"id": "004", "reason": "recording failed"}, {"id": "005", "reason": "camera sources missing"}]
        assert merged["skipped"]["full"] == failed
        still = [{"id": n, "reason": "no motion"} for n in kept if not kept[n]]
        assert merged["skipped"]["motion"] == sorted(failed + still, key=lambda part: part["id"])
        assert json.loads((folder / "merged.json").read_text()) == merged
        for kind, frames in (("full", 54), ("motion", merged["motion_frames"])):
            assert count_frames_and_secs(str(folder / f"{kind}.mp4"))[0] == frames, kind
        assert sorted(p.name for p in folder.iterdir()) == ["encoder.log", "full.mp4", "merged.json", "motion.mp4"]
        # The segment list is unchanged by the merged folder.
        assert [row["id"] for row in recordings.manifests_of_task(task_id)] == ["001", "002", "003", "004", "005"]

        # Merging an unchanged task does no work.
        stamp = (folder / "full.mp4").stat().st_mtime_ns
        assert recordings.request_render(task_id, MERGED)["merged"]["created"] == merged["created"]
        assert (folder / "full.mp4").stat().st_mtime_ns == stamp

        # A later Resume adds a segment: the old merge is reported as outdated, then replaced.
        task.add_segment(6, 6, started_monotonic_s=started + 54 / 30)
        stale = recordings.merged_status(task_id)
        assert stale["current"] is False and stale["merged"]["frames"] == 54
        recordings.request_render(task_id, MERGED)
        status = RecordingWaits.await_merged(recordings, task_id)
        assert status["current"] is True and status["merged"]["frames"] == 60
    finally:
        recordings.close()


def test_merge_needs_finished_segments_with_sources_and_a_task_end_waits_for_them(tmp_path):
    task_id = "e" * 32
    directory = tmp_path / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    recordings = RenderedRecordings(tmp_path, StandInCameras.ROW)
    try:
        with pytest.raises(ValueError, match="No finished segment has camera sources to merge"):
            recordings.request_render(task_id, MERGED)
        task.add_segment(1, 4, grey_levels={})
        with pytest.raises(ValueError, match="No finished segment has camera sources to merge"):
            recordings.request_render(task_id, MERGED)
        task.add_segment(2, 5, started_monotonic_s=started)
        folder = directory / "recordings" / "merged"
        folder.mkdir()
        (folder / "full.mp4").write_bytes(b"left over from a merge before the styled videos")
        (folder / "merged.json").write_text(json.dumps({"segments": {"full": [{"id": "002"}]}}))
        assert recordings.merged_status(task_id)["current"] is False

        recordings.live_segments.append(StandInSegment(task_id, "003"))
        try:
            with pytest.raises(RuntimeError, match="Segment 3 is still recording or processing"):
                recordings.request_render(task_id, MERGED)
            assert recordings.merged_status(task_id)["pending"] == "003"
            # The task ends while its last segment is still processing: the merge waits for it.
            recordings.on_event("task", {"id": task_id})
            recordings.on_event("task_end", {"message": "Task cleared; history saved."})
            time.sleep(3 * POLL_S)
            assert recordings.merged_status(task_id)["render"]["status"] == "waiting"
        finally:
            recordings.live_segments.pop()

        status = RecordingWaits.await_merged(recordings, task_id)
        assert status["current"] is True and status["merged"]["frames"] == 5
        assert status["merged"]["skipped"]["full"] == [{"id": "001", "reason": "camera sources missing"}]
        assert not list(folder.glob("*.partial.*"))

        # A task without recordings starts no merge when it ends.
        recordings.on_event("task", {"id": "0" * 32})
        recordings.on_event("task_end", {"message": "Task cleared; history saved."})
        assert ("0" * 32, MERGED) not in recordings.render_jobs
    finally:
        recordings.close()


def test_a_paused_segment_renders_its_own_videos_once_saved(tmp_path):
    task_id = "c" * 32
    directory = tmp_path / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    task.add_segment(1, 24, 8, started_monotonic_s=started)
    recordings = RenderedRecordings(tmp_path, StandInCameras.ROW)
    try:
        recordings.active_segment = StandInSegment(task_id, "001")
        recordings.live_segments.append(recordings.active_segment)
        try:
            recordings.stop("Task paused")
            time.sleep(3 * POLL_S)
            [row] = recordings.segments(task_id)
            assert row["styled"] is None and row["render"]["status"] == "waiting"
        finally:
            recordings.live_segments.pop()  # The segment is saved.

        row = RecordingWaits.await_styled(recordings, task_id, "001")

        assert row["render"] is None and row["styled"]["frames"] == 24
        assert [part["id"] for part in row["styled"]["segments"]["full"]] == ["001"]
        folder = directory / "recordings" / "001" / "styled"
        assert count_frames_and_secs(str(folder / row["styled"]["full"]))[0] == 24
        assert [row["id"] for row in recordings.manifests_of_task(task_id)] == ["001"]
        # The segment's videos are the task's: a task with one segment merges nothing when it ends.
        recordings.on_event("task", {"id": task_id})
        recordings.on_event("task_end", {"message": "Task cleared; history saved."})
        assert (task_id, MERGED) not in recordings.render_jobs
    finally:
        recordings.close()


def test_no_render_starts_after_shutdown(tmp_path):
    task_id = "b" * 32
    recordings = RenderedRecordings(tmp_path, StandInCameras.ROW)
    recordings.active_segment = StandInSegment(task_id, "001")

    recordings.close()

    assert recordings.render_jobs == {}


def test_saved_segments_become_clips_with_the_sources_on_disk(tmp_path):
    directory = tmp_path / ("a" * 32)
    task = RecordedTask(directory)
    first = task.add_segment(1, 9, started_monotonic_s=500.25)
    second = task.add_segment(2, 6, started_monotonic_s=501.5)
    (second / SOURCES["top"]).unlink()  # This camera's copy failed.
    task.add_segment(3, 6, grey_levels={})
    task.add_segment(4, 0, status="failed")

    plan = StyledRender(
        TaskFiles(directory), Recordings(tmp_path, StandInCameras.ROW).manifests_of_task(directory.name), MERGED
    )

    assert [row["id"] for row in plan.manifests] == ["001", "002"]
    assert plan.clips == [
        SourceClip(500.25, 9 / 30, {role: first / name for role, name in SOURCES.items()}),
        SourceClip(501.5, 6 / 30, {role: second / name for role, name in SOURCES.items() if role != "top"}),
    ]
    assert plan.skipped_segments == [
        {"id": "003", "reason": "camera sources missing"},
        {"id": "004", "reason": "recording failed"},
    ]


def test_every_camera_is_decoded_in_step_and_the_motion_video_gets_only_its_frames(tmp_path, stand_in_frames):
    directory = tmp_path / ("b" * 32)
    task = RecordedTask(directory)
    # Flat pictures whose brightness names the clip, the camera and the frame.
    task.add_segment(
        1,
        6,
        started_monotonic_s=10.0,
        grey_levels={
            "left": [20 + 20 * k for k in range(6)],
            "top": [200 - 20 * k for k in range(6)],
            "right": [100 + 10 * k for k in range(6)],
        },
    )
    # The second clip has no overhead source, and its right one is two frames short: its last picture stays.
    task.add_segment(
        2, 4, started_monotonic_s=20.0, grey_levels={"left": [30 + 25 * k for k in range(4)], "right": [60, 90]}
    )
    turns = [
        TurnSpan(1, 0.0, 4 / 30, {"z": 1.0}, 10),
        TurnSpan(2, 4 / 30, 7 / 30, None, 20),
        TurnSpan(3, 7 / 30, 10 / 30, {"z": -1.0}, 30),
    ]
    timeline = TaskTimeline(
        instruction=INSTRUCTION,
        model_label="Model",
        badge="xhigh",
        duration_s=10 / 30,
        turns=turns,
        motion_spans=[(1 / 30, 3 / 30), (8 / 30, 10 / 30)],
    )
    manifests = Recordings(tmp_path, StandInCameras.ROW).manifests_of_task(directory.name)
    merge, messages = StyledRender(TaskFiles(directory), manifests, MERGED), queue.Queue()
    merge.folder.mkdir()
    worker = RenderWorker(directory, merge.clips, StandInCameras.ROW, merge.folder, messages, FFmpeg.executable())

    with (merge.folder / "encoder.log").open("w") as log:
        rendered = worker.render(timeline, log)

    assert rendered == {"frames": 10, "motion_frames": 4, "clip_motion_frames": [2, 2]}
    full, motion = stand_in_frames
    assert full.header == motion.header == (INSTRUCTION, "Model", "xhigh")
    assert full.turns == turns and full.duration_s == 10 / 30
    assert full.times == [n / 30 for n in range(10)]
    expected = [{"left": 20 + 20 * k, "top": 200 - 20 * k, "right": 100 + 10 * k} for k in range(6)]
    # The second clip has no overhead source: the frames show that camera as missing.
    expected += [{"left": 30 + 25 * k, "right": 60 if k == 0 else 90} for k in range(4)]
    assert StandInStyledFrames.close_to(full.pictures, expected)
    # The motion-only video runs on the timeline's own motion-only timeline.
    motion_timeline = timeline.cut_to_motion()
    assert (motion.turns, motion.duration_s) == (motion_timeline.turns, motion_timeline.duration_s)
    assert motion.times == [n / 30 for n in range(4)]
    assert StandInStyledFrames.close_to(motion.pictures, [expected[n] for n in (1, 2, 8, 9)])
    assert [messages.get_nowait()["progress"] for _ in range(messages.qsize())] == [0.4, 0.8]
    for kind, frames in (("full", 10), ("motion", 4)):
        assert FFmpeg.packet_count(FFmpeg.executable(), merge.folder / f"{kind}.partial.mp4") == frames

    merge.publish(rendered)

    manifest = merge.status()["merged"]
    assert manifest["segments"] == {
        "full": [{"id": "001", "start_frame": 0, "frames": 6}, {"id": "002", "start_frame": 6, "frames": 4}],
        "motion": [{"id": "001", "start_frame": 0, "frames": 2}, {"id": "002", "start_frame": 2, "frames": 2}],
    }
    assert manifest["motion_duration_s"] == 4 / 30
    assert sorted(p.name for p in merge.folder.iterdir()) == [
        "encoder.log",
        "full.mp4",
        "merged.json",
        "motion.mp4",
    ]


def test_a_task_without_motion_publishes_no_motion_video(tmp_path, stand_in_frames):
    directory = tmp_path / ("c" * 32)
    RecordedTask(directory).add_segment(1, 5, started_monotonic_s=10.0)
    manifests = Recordings(tmp_path, StandInCameras.ROW).manifests_of_task(directory.name)
    merge, messages = StyledRender(TaskFiles(directory), manifests, MERGED), queue.Queue()
    merge.folder.mkdir()
    (merge.folder / "motion.mp4").write_bytes(b"from an earlier merge")
    timeline = TaskTimeline(
        instruction=INSTRUCTION,
        model_label="Model",
        badge=None,
        duration_s=5 / 30,
        turns=[TurnSpan(1, 0.0, 5 / 30, None, None)],
        motion_spans=[],
    )
    worker = RenderWorker(directory, merge.clips, StandInCameras.ROW, merge.folder, messages, FFmpeg.executable())

    with (merge.folder / "encoder.log").open("w") as log:
        rendered = worker.render(timeline, log)
    merge.publish(rendered)

    assert len(stand_in_frames) == 1 and rendered["clip_motion_frames"] == [0]
    manifest = merge.status()["merged"]
    assert manifest["motion"] is None and manifest["motion_frames"] == 0
    assert manifest["skipped"]["motion"] == [{"id": "001", "reason": "no motion"}]
    assert not (merge.folder / "motion.mp4").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into read-only folders")
def test_failed_render_keeps_the_sources_and_the_previous_merge(tmp_path):
    task_id = "9" * 32
    directory = tmp_path / task_id
    task = RecordedTask(directory)
    started = task.copy_stored_task()
    task.add_segment(1, 30, 4, started_monotonic_s=started)
    recordings = RenderedRecordings(tmp_path, StandInCameras.ROW)
    try:
        recordings.request_render(task_id, MERGED)
        first = RecordingWaits.await_merged(recordings, task_id)["merged"]
        folder = directory / "recordings" / "merged"
        published = FileBytes.contents(folder.glob("*.mp4"), [folder / "merged.json"])
        task.add_segment(2, 24, started_monotonic_s=started + 1)
        sources = FileBytes.contents((directory / "recordings").glob("0*/*.mp4"))
        assert len(sources) == 2 * len(SOURCES)

        folder.chmod(0o555)  # The disk refuses the new videos.
        try:
            recordings.request_render(task_id, MERGED)
            status = RecordingWaits.await_merged(recordings, task_id)
        finally:
            folder.chmod(0o755)

        assert status["render"]["status"] == "failed" and "Permission denied" in status["render"]["error"]
        assert status["current"] is False and status["merged"] == first
        assert FileBytes.contents(folder.glob("*.mp4"), [folder / "merged.json"]) == published
        assert FileBytes.contents((directory / "recordings").glob("0*/*.mp4")) == sources

        # Merging again once the disk accepts the videos replaces the outdated merge.
        recordings.request_render(task_id, MERGED)
        status = RecordingWaits.await_merged(recordings, task_id)
        assert status["current"] is True and status["merged"]["frames"] == 54
        assert not list(folder.glob("*.partial.*"))
    finally:
        recordings.close()


@pytest.mark.parametrize("camera_set", sorted(StyledSupport.TILES))
def test_frame_has_the_video_size_and_each_camera_in_its_tile(camera_set):
    styled = StyledSupport.frames(camera_set, StyledSupport.turns())

    picture = styled.frame(StyledSupport.images(camera_set), 3.0)

    assert picture.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3) and picture.dtype == np.uint8
    assert list(styled.tiles.values()) == StyledSupport.TILES[camera_set]
    inset = StyledFrames.CAMERA_RADIUS
    for (x, y, width, height), colour in zip(styled.tiles.values(), StyledSupport.COLOURS):
        assert (picture[y + inset : y + height - inset, x + inset : x + width - inset] == colour).all()
        # Rounded corners, and nothing of the camera outside its tile.
        assert not StyledSupport.is_colour(picture[y, x], colour)
        assert not StyledSupport.is_colour(picture[y + height // 2, x - 1], colour)
        assert not StyledSupport.is_colour(picture[y + height, x + width // 2], colour)
    # Pictures already at their tile's size are pasted as they are.
    tile_images = {
        role: np.full((height, width, 3), colour, np.uint8)
        for (role, (_, _, width, height)), colour in zip(styled.tiles.items(), StyledSupport.COLOURS)
    }
    assert np.array_equal(styled.frame(tile_images, 3.0), picture)


def test_missing_camera_leaves_its_tile_empty_and_labelled():
    styled = StyledSupport.frames("row", StyledSupport.turns())
    images = StyledSupport.images("row")
    role = next(iter(styled.tiles))
    del images[role]
    x, y, width, height = styled.tiles[role]
    inset = StyledFrames.CAMERA_RADIUS

    tile = styled.frame(images, 3.0)[y + inset : y + height - inset, x + inset : x + width - inset]

    assert StyledSupport.is_colour(tile, Palette.WELL).mean() > 0.95
    assert tile.max() > max(Palette.WELL) + 100  # the "Camera unavailable" label


@pytest.mark.parametrize("camera_set", sorted(StyledSupport.TILES))
def test_progress_has_one_segment_per_turn(camera_set):
    styled = StyledSupport.frames(camera_set, StyledSupport.turns())
    count = len(StyledSupport.LENGTHS_S)

    filled, empty = StyledSupport.bar(styled.frame({}, 0.0))
    assert not filled and len(empty) == count  # the playhead covers the first turn's start
    filled, empty = StyledSupport.bar(styled.frame({}, StyledSupport.DURATION_S - StyledFrames.FRAME_S))
    assert len(filled) == count and not empty
    span = VIDEO_SIZE[0] - 2 * StyledLayout.MARGIN
    expected = [length / StyledSupport.DURATION_S * span for length in StyledSupport.LENGTHS_S]
    # The gaps between turns and the playhead at the end take a few pixels.
    tolerance = 2 * StyledFrames.PLAYHEAD_CLEARANCE[0] + StyledFrames.BAR_GAP
    assert all(abs(run - length) <= tolerance for run, length in zip(filled, expected))
    filled, empty = StyledSupport.bar(styled.frame({}, sum(StyledSupport.LENGTHS_S[:2]) + 0.5))
    assert len(filled) == 3 and len(empty) == 3  # in the third turn, the rest of which is empty


@pytest.mark.parametrize("camera_set", sorted(StyledSupport.TILES))
def test_non_zero_values_are_highlighted(camera_set):
    action = {**StyledSupport.ACTION, "arm": {**StyledSupport.ACTION["arm"], "z": -3.0}}

    zeros = StyledSupport.frames(camera_set, StyledSupport.turns()).frame({}, 1.0)
    moving = StyledSupport.frames(camera_set, StyledSupport.turns(action)).frame({}, 1.0)

    x0, y0, x1, y1 = StyledLayout.for_cameras(len(StyledSupport.CAMERAS[camera_set])).code_box
    pixels = (slice(y0, y1), slice(x0, x1))
    is_amber = StyledSupport.is_colour(moving[pixels], Palette.LIGHT_AMBER)
    assert is_amber.sum() > 0 and not StyledSupport.is_colour(zeros[pixels], Palette.LIGHT_AMBER).any()
    rows = np.flatnonzero(is_amber.any(axis=1))
    assert rows[-1] - rows[0] < StyledFrames.CODE_PITCH  # one value on one line
    outside = np.ones(zeros.shape[:2], bool)
    outside[pixels] = False
    assert np.array_equal(zeros[outside], moving[outside])


def test_action_code_marks_numbers_and_wraps_wide_objects():
    action = {"type": "pose_delta", "arm": {"x": 0.0, "y": 0.0, "z": -3.0, "pitch": 0.2}, "gripper": {"width": 6}}

    code = ActionCode(action)

    assert code.line_count == 5  # braces alone, one top-level key per line
    assert [text for _, _, text, role in code.spans if role == "lit"] == ["-3.0", "0.2", "6"]
    assert [text for _, _, text, role in code.spans if role == "num"] == ["0.0", "0.0"]
    limit = 30
    wrapped = ActionCode(action, limit)
    assert wrapped.line_count > code.line_count and wrapped.column_count <= limit
    assert [span[2:] for span in wrapped.spans if span[3] == "lit"] == [
        ("-3.0", "lit"),
        ("0.2", "lit"),
        ("6", "lit"),
    ]
    assert ActionCode(None).spans == [(0, 0, ActionCode.NO_ACTION, "note")]


def test_turn_change_animates_then_settles():
    styled = StyledSupport.frames("row", StyledSupport.turns())
    start = StyledSupport.LENGTHS_S[0]
    top, bottom = styled.number_rows
    rows = slice(top, bottom)

    beginning, settled, later = (styled.frame({}, start + offset)[rows] for offset in (0.05, 1.0, 2.0))

    assert not np.array_equal(beginning, settled)
    assert np.array_equal(settled, later)


def test_long_instruction_and_model_are_cut_to_fit():
    styled = StyledSupport.frames(
        "row", StyledSupport.turns(), instruction="Pick up the cup " * 30, model_label="Model " * 40
    )

    header = styled.frame({}, 1.0)[: StyledLayout.CAMERA_LABEL_BASELINE - StyledFrames.LABEL_SIZE]

    dot = np.flatnonzero(StyledSupport.is_colour(header, Palette.AMBER).any(axis=0))[0]
    assert dot > VIDEO_SIZE[0] // 2  # the model name was cut short
    instruction = np.flatnonzero((header[:, :dot] == Palette.WHITE).all(axis=-1).any(axis=0))
    assert instruction.size and instruction[-1] < dot - StyledFrames.TITLE_GAP


@pytest.mark.parametrize("robot", RecordedTask.stored_robots())
def test_turns_follow_the_answers_with_their_actions(tmp_path, robot):
    golden = next((RecordedTask.STORED_TASKS / robot).glob("*/task.json")).parent
    directory = shutil.copytree(golden, tmp_path / golden.name)
    label = json.loads((directory / "task.json").read_text())["model_label"]  # the stored task's model, as recorded
    journal = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    answers = [event for event in journal if event["type"] == "model_output"]
    first, second = (answer["monotonic_s"] for answer in answers)
    records = RecordedTask(directory)
    clips = [
        records.segment_clip(1, first - FIRST_LEAD_S, FIRST_S, []),
        records.segment_clip(2, second - SECOND_LEAD_S, SECOND_S, []),
    ]
    assert clips[0].start_monotonic_s + FIRST_S < clips[1].start_monotonic_s

    timeline = TaskTimeline.load(TaskFiles(directory), clips)

    # The stored task's model has no reasoning effort to show.
    assert (timeline.instruction, timeline.model_label, timeline.badge) == ("Pick up the red cube.", label, None)
    assert timeline.duration_s == pytest.approx(FIRST_S + SECOND_S)
    # The third turn reached the context limit before it answered.
    assert [turn.number for turn in timeline.turns] == [1, 2]
    assert [turn.start_s for turn in timeline.turns] == pytest.approx([FIRST_LEAD_S, FIRST_S + SECOND_LEAD_S])
    assert [turn.end_s for turn in timeline.turns] == pytest.approx([FIRST_S + SECOND_LEAD_S, FIRST_S + SECOND_S])
    TurnChecks.assert_ordered(timeline.turns)
    for turn, answer in zip(timeline.turns, answers):
        output = json.loads((directory / f"{answer['data']['attempt']}-output.json").read_text())
        parts = json.loads(output["answer"].removeprefix("```json\n").removesuffix("\n```"))
        assert {key: value for key, value in turn.action.items() if key != "type"} == parts
        assert turn.tokens is None  # the stored task's model reported no usage
    assert timeline.motion_spans == []


def test_turns_map_across_clips_with_actions_and_context_tokens(recorded):
    timeline = TaskTimeline.load(*recorded)

    assert (timeline.instruction, timeline.model_label, timeline.badge) == (
        "Stack the cubes.",
        "Example Model",
        None,
    )
    assert timeline.duration_s == 20.0
    assert [(turn.number, turn.start_s, turn.end_s) for turn in timeline.turns] == [
        (1, 5.0, 10.0),
        (2, 10.0, 13.0),  # an answer between clips shows from the next clip's start
        (3, 13.0, 15.0),
        (4, 15.0, 20.0),
    ]
    assert [turn.action for turn in timeline.turns] == [
        ACTIONS["one"],
        ACTIONS["two"],
        None,
        ACTIONS["four"],
    ]
    # The context fill: a turn without a report keeps the last one; a request context counts itself.
    assert [turn.tokens for turn in timeline.turns] == [1200, 1800, 1800, 2190]
    TurnChecks.assert_ordered(timeline.turns)
    assert all(type(turn) is TurnSpan for turn in timeline.turns)


def test_the_cut_to_motion_keeps_the_turns_that_moved(recorded):
    timeline = TaskTimeline.load(*recorded)

    assert timeline.motion_spans == [(6.0, 8.0), (13.0, 14.0), (17.0, 19.0)]
    motion = timeline.cut_to_motion()
    assert motion.duration_s == 5.0
    assert motion.motion_spans == [(0.0, 5.0)]
    # The second turn's motion fell between the clips.
    assert [(turn.number, turn.start_s, turn.end_s) for turn in motion.turns] == [
        (1, 0.0, 2.0),
        (3, 2.0, 3.0),
        (4, 3.0, 5.0),
    ]
    assert [turn.tokens for turn in motion.turns] == [1200, 1800, 2190]
    TurnChecks.assert_ordered(motion.turns)


@pytest.mark.parametrize(
    ("task", "label", "badge"),
    [
        ({"model_label": "Example 27B BF16 · xhigh", "reasoning_effort": "xhigh"}, "Example 27B BF16", "xhigh"),
        (
            {"model_label": "Example 27B · xhigh · prompt B", "reasoning_effort": "xhigh"},
            "Example 27B · prompt B",
            "xhigh",
        ),
        (
            {"model_label": "Example 27B · harness · pose", "action_format": "pose"},
            "Example 27B · harness · pose",
            None,
        ),
        ({"model_label": None}, "example", None),
    ],
)
def test_badge_is_the_reasoning_effort_shown_once(tmp_path, task, label, badge):
    record = {"id": "e" * 32, "instruction": "Hold.", "created": 1.0, "model": "example", **task}
    files = RecordedTask.create(tmp_path / record["id"], record).files

    timeline = TaskTimeline.load(files, [])

    assert (timeline.model_label, timeline.badge) == (label, badge)
    assert (timeline.duration_s, timeline.turns, timeline.motion_spans) == (0.0, [], [])


def test_motion_spans_match_the_recorder_and_a_task_without_turns_has_none(tmp_path):
    task = RecordedTask.TASK
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    robot = {"control_state": "HOLD"}
    try:
        recordings.start(task, source, lambda: dict(robot))
        job = recordings.active_segment
        time.sleep(IDLE_S)
        robot["control_state"] = "RUNNING"
        time.sleep(MOVING_S)
        robot["control_state"] = "HOLD"
        time.sleep(IDLE_S)
        recordings.stop("Paused")
        RecordingWaits.await_finished(job)
    finally:
        recordings.close()
        source.close()
    info = job.snapshot()
    assert info["status"] == "saved", info
    files = RecordedTask.create(tmp_path / task["id"], {**task, "created": job.started_at}).files
    clip = SourceClip.of_segment(job.directory, info)

    timeline = TaskTimeline.load(files, [clip])

    moving = []
    for row in RecordedTask.timeline_rows(job.directory):
        if not row["state"]["moving"]:
            continue
        if moving and moving[-1][1] == row["start_frame"]:
            moving[-1][1] = row["end_frame"]
        else:
            moving.append([row["start_frame"], row["end_frame"]])
    assert moving and timeline.motion_spans == [(start / FRAME_RATE, end / FRAME_RATE) for start, end in moving]
    assert timeline.duration_s == info["duration_s"]
    assert timeline.turns == []


def test_stopped_segment_is_rendered_into_styled_full_and_motion_videos(enabled_console, monkeypatch):
    controller = enabled_console
    controller.commands.run("recording", {"enabled": True})
    monkeypatch.setattr(
        controller, "new_client", StandInModel(controller, delay_s=0, is_moving=True, context_limit=TURNS)
    )
    controller.commands.run("start", {"instruction": "Lift the arm, then hold."})
    Waiting.until(lambda: not controller.is_busy, TASK_TIMEOUT_S, ConsoleRig.POLL_S)
    assert controller.task_run.total_steps == TURNS
    task_id = controller.task_run.task["id"]
    directory = controller.root / task_id
    recordings = controller.recordings

    segment = RecordingWaits.await_styled(recordings, task_id, "001")
    controller.commands.run("clear", None)  # New task: a task with one segment merges nothing.
    Waiting.until(lambda: not controller.is_busy, TASK_TIMEOUT_S, ConsoleRig.POLL_S)

    assert segment["render"] is None and segment["styled"] is not None, segment
    assert recordings.merged_status(task_id)["render"] is None
    clips = StyledRender(TaskFiles(directory), recordings.manifests_of_task(task_id), MERGED).clips
    names = {camera.role: f"{camera.name}.mp4" for camera in controller.robot.CAMERAS}
    # the clip's motion spans are a field the port's clip added: the clip is compared on the SOURCE's fields
    assert [(clip.start_monotonic_s, clip.duration_s, clip.files) for clip in clips] == [
        (
            segment["started_monotonic_s"],
            segment["frames"] / 30,
            {role: directory / "recordings" / "001" / name for role, name in names.items()},
        )
    ]
    # The records and the clip share one clock: every action happened inside the recorded clip.
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    actions = [event["monotonic_s"] for event in events if event["type"] == "action"]
    assert len(actions) == TURNS
    assert all(clips[0].start_monotonic_s <= t < clips[0].start_monotonic_s + clips[0].duration_s for t in actions)
    styled, folder = segment["styled"], directory / "recordings" / "001" / "styled"
    assert styled["frames"] == segment["frames"]
    assert styled["motion_frames"] == sum(end - start for start, end in segment["motion_intervals"]) > 0
    for kind, frames in (("full", styled["frames"]), ("motion", styled["motion_frames"])):
        assert next(read_frames(str(folder / styled[kind])))["size"] == VIDEO_SIZE
        assert count_frames_and_secs(str(folder / styled[kind]))[0] == frames, kind
