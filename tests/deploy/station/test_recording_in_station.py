import json
import os
import queue
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from imageio_ffmpeg import count_frames_and_secs, read_frames
from PIL import ImageChops

from console_rig import ConsoleRig, MarkedCameras
from pear.deploy.station.records.recording.captions import CaptionPhase, Captions
from pear.deploy.station.records.recording.ffmpeg import FFmpeg
from pear.deploy.station.records.recording.frame_renderer import CaptionedFrames, CaptionedLayout
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.recording.recordings import Recordings
from pear.deploy.station.records.task_files import TaskFiles
from recorded_task import RecordedTask, RecordingWaits
from stand_in_cameras import StandInCameras
from waiting import Waiting

SOURCES = RecordedTask.SOURCE_FILES  # each camera's own video in a segment of StandInCameras.ROW, by role
# where full.mp4 shows each camera of StandInCameras.ROW: [x, y, width, height] by role
TILES = {role: [n * 640, 152, 640, 480] for n, role in enumerate(("left", "top", "right"))}
ENCODER_SHOWS_S = 15  # for the encoder of a camera's own video to show among the processes


class DecodedVideo:
    """The pictures of a video file, decoded."""

    @staticmethod
    def frames(path: Path) -> Iterator[np.ndarray]:
        """Yields the RGB picture of each frame of the video at ``path``."""
        frame_reader = read_frames(str(path))
        metadata = next(frame_reader)
        width, height = metadata["size"]
        for frame_bytes in frame_reader:
            yield np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)


def test_real_encoding_keeps_time_trims_waits_and_clears_previous_output(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    assert recordings.is_enabled is True
    source = StandInCameras(StandInCameras.ROW)
    robot = {"control_state": "HOLD"}
    try:
        recordings.start(RecordedTask.TASK, source, lambda: dict(robot))
        job = recordings.active_segment
        recordings.on_event("turn_started", {"turn_number": 1, "attempt": "one"})
        recordings.on_event("observation", {"attempt": "one", "request_id": "request-one"})
        recordings.on_event("capacity", {"attempt": "one", "input": 207539, "context_limit": 260096})
        recordings.on_event("phase", {"phase": "Thinking"})
        time.sleep(0.5)
        recordings.on_event("usage", {"attempt": "one", "input": 207539, "output": 181})
        action = {"left_arm": {"z": 0.3}, "left_gripper": {"width": 1}}
        recordings.on_event("action", action)
        recordings.on_event("phase", {"phase": "Executing"})
        robot["control_state"] = "RUNNING"
        time.sleep(0.6)
        robot["control_state"] = "HOLD"
        recordings.on_event("turn_started", {"turn_number": 2, "attempt": "two"})
        recordings.on_event("phase", {"phase": "Thinking"})
        time.sleep(0.5)
        requested = time.monotonic()
        recordings.stop("Paused", stopped_at=requested)
        recordings.stop("Repeated stop")
        assert time.monotonic() - requested < 0.1

        RecordingWaits.await_finished(job)
        info = job.snapshot()
        assert info["status"] == "saved", info
        assert info["model"] == RecordedTask.TASK["model"]
        assert info["model_label"] == RecordedTask.TASK["model_label"]
        full_frames, _ = count_frames_and_secs(str(job.directory / "full.mp4"))
        motion_frames, _ = count_frames_and_secs(str(job.directory / "motion.mp4"))
        assert abs(full_frames / 30 - (requested - job.started_at)) < 1 / 30 + 0.01
        assert 0.4 <= motion_frames / 30 <= 0.8

        rows = RecordedTask.timeline_rows(job.directory)
        thinking = [row for row in rows if row["state"].get("turn_number") == 2]
        assert thinking and all(row["state"]["action"] is None for row in thinking)
        assert all(row["state"]["context_usage"]["turn_number"] == 1 for row in thinking)
        assert all(row["state"]["inference_usage"] is None for row in thinking)
        assert all(not row["state"]["moving"] for row in thinking)
        executing = [row for row in rows if row["state"]["moving"]]
        assert executing and all(row["state"]["action"] == action for row in executing)
        assert all(row["state"]["context_usage"]["tokens"] == 207720 for row in executing)
        assert all(row["state"]["inference_usage"]["tokens"] == 207720 for row in executing)
        assert all(row["state"]["context_usage"]["context_limit"] == 260096 for row in executing)
        assert all(row["state"]["context_usage"]["request_id"] == "request-one" for row in executing)
        assert sum(row["end_frame"] - row["start_frame"] for row in executing) == motion_frames
        assert set(rows[0]["cameras"]) == {"left", "top", "right"}

        # Clean camera copies share full.mp4's frame numbers, so a re-edit can use
        # timeline rows directly. Each matches the panel burned into the full video.
        assert info["sources"] == SOURCES and info["source_error"] is None
        assert info["camera_tiles"] == TILES
        assert type(info["started_wall_s"]) is float
        assert abs(time.time() - info["started_wall_s"] - (time.monotonic() - job.started_at)) < 1
        saved = json.loads((job.directory / "recording.json").read_text())
        assert saved["sources"] == SOURCES and saved["started_wall_s"] == info["started_wall_s"]
        for role, name in SOURCES.items():
            frames, _ = count_frames_and_secs(str(job.directory / name))
            assert frames == full_frames, role
            x, y, width, height = TILES[role]
            pairs = zip(DecodedVideo.frames(job.directory / "full.mp4"), DecodedVideo.frames(job.directory / name))
            for n, (full, camera) in enumerate(pairs):
                if n % 7 == 0:
                    panel = full[y : y + height, x : x + width].astype(int)
                    assert np.abs(panel - camera).mean() < 4, (role, n)
        assert not list(job.directory.glob("*.partial.mp4"))
    finally:
        recordings.close()
        source.close()


def test_recording_disabled_does_not_start_encoder(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Disabled recording looked for an encoder")

    monkeypatch.setattr(FFmpeg, "executable", forbidden)
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    recordings.is_enabled = False

    recordings.start(RecordedTask.TASK, None, None)

    assert not recordings.live_segments and recordings.active_segment is None


def test_interrupted_manifest_is_not_reported_as_still_recording(tmp_path):
    task_id = RecordedTask.TASK["id"]
    directory = tmp_path / task_id / "recordings" / "001"
    directory.mkdir(parents=True)
    (directory / "recording.json").write_text('{"id":"001","status":"recording"}')

    assert Recordings(tmp_path, StandInCameras.ROW).manifests_of_task(task_id)[0]["status"] == "failed"


def test_immediate_pause_produces_a_decodable_full_video(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        recordings.stop("Immediate Pause")

        RecordingWaits.await_finished(job)

        assert job.snapshot()["status"] == "saved", job.snapshot()
        frames, _ = count_frames_and_secs(str(job.directory / "full.mp4"))
        assert frames >= 1 and job.snapshot()["no_motion"]
    finally:
        recordings.close()
        source.close()


def test_dropped_pictures_keep_action_timing_and_captions(tmp_path, monkeypatch):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    robot = {"control_state": "HOLD"}
    try:
        recordings.start(RecordedTask.TASK, source, lambda: dict(robot))
        job = recordings.active_segment

        def skip_picture(packet):
            raise queue.Full

        monkeypatch.setattr(job.packets, "put_nowait", skip_picture)
        recordings.on_event("turn_started", {"turn_number": 1, "attempt": "one"})
        recordings.on_event("phase", {"phase": "Thinking"})
        time.sleep(0.15)
        action = {"left_gripper": {"width": -1}}
        recordings.on_event("usage", {"attempt": "one", "input": 51696, "output": 187})
        recordings.on_event("action", {**action, "attempt": "one"})
        recordings.on_event("phase", {"phase": "Executing"})
        robot["control_state"] = "RUNNING"
        time.sleep(0.3)
        robot["control_state"] = "HOLD"
        recordings.on_event("turn_started", {"turn_number": 2, "attempt": "two"})
        time.sleep(0.15)
        recordings.stop("Paused")

        RecordingWaits.await_finished(job)

        assert job.snapshot()["status"] == "saved", job.snapshot()
        assert job.snapshot()["dropped_samples"] > 5
        assert 0.2 <= job.snapshot()["motion_duration_s"] <= 0.4
        rows = RecordedTask.timeline_rows(job.directory)
        assert all(row["state"]["action"] == action for row in rows if row["state"]["moving"])
        assert all(row["state"]["context_usage"]["tokens"] == 51883 for row in rows if row["state"]["moving"])
        assert any(row["state"].get("turn_number") == 2 for row in rows)
    finally:
        recordings.close()
        source.close()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to find the encoder")
def test_failed_camera_source_keeps_full_video_and_other_cameras(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        target = str(job.directory / "head.partial.mp4")

        def encoder():
            for entry in Path("/proc").iterdir():
                try:
                    if target in (entry / "cmdline").read_bytes().decode(errors="replace"):
                        return int(entry.name)
                except (OSError, ValueError):
                    continue
            return None

        Waiting.until(lambda: encoder() is not None, ENCODER_SHOWS_S, RecordingWaits.POLL_S)
        os.kill(encoder(), signal.SIGKILL)
        time.sleep(0.3)
        recordings.stop("Paused")

        RecordingWaits.await_finished(job)

        info = job.snapshot()
        assert info["status"] == "saved", info
        assert info["sources"] == {**SOURCES, "top": None}
        assert "head.mp4" in info["source_error"]
        assert (job.directory / "full.mp4").is_file()
        assert (job.directory / "left_wrist.mp4").is_file()
        assert not (job.directory / "head.mp4").exists()
        assert not list(job.directory.glob("*.partial.mp4"))
    finally:
        recordings.close()
        source.close()


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"tokens": 51883, "output": 187, "turn_number": 28},
            "Context Usage: 51,883 tokens | Last: turn 28",
        ),
        (
            {"tokens": 207539, "context_limit": 260096, "turn_number": 71},
            "Context Usage: 207,539 / 260,096 tokens (79.8%) | Last: turn 71",
        ),
        (
            {"tokens": 207720, "output": 181, "context_limit": 260096, "turn_number": 71},
            "Context Usage: 207,720 / 260,096 tokens (79.9%) | Last: turn 71",
        ),
        (
            {"tokens": 0, "output": 0, "context_limit": 100, "turn_number": 1},
            "Context Usage: 0 / 100 tokens (0.0%) | Last: turn 1",
        ),
        (
            {"tokens": 51696, "context_limit": 0, "turn_number": 28},
            "Context Usage: 51,696 tokens | Last: turn 28",
        ),
    ],
)
def test_the_context_caption_shows_the_latest_total_and_the_turn_it_is_from(usage, expected):
    assert Captions.context({"context_usage": usage}) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"phase": "Thinking"}, "Current Turn Usage: Pending"),
        ({"phase": "Executing"}, "Current Turn Usage: Not provided"),
        (
            {
                "phase": "Thinking",
                "turn_number": 29,
                "inference_usage": {
                    "input": 51696,
                    "output": 187,
                    "tokens": 51883,
                    "turn_number": 28,
                },
            },
            "Current Turn Usage: Pending",
        ),
        (
            {
                "phase": "Thinking",
                "turn_number": 28,
                "inference_usage": {
                    "input": 51696,
                    "output": None,
                    "tokens": 51696,
                    "turn_number": 28,
                },
            },
            "Current Turn Usage: 51,696 + Pending",
        ),
        (
            {
                "phase": "Executing",
                "turn_number": 28,
                "inference_usage": {
                    "input": 51696,
                    "output": 187,
                    "tokens": 51883,
                    "turn_number": 28,
                },
            },
            "Current Turn Usage: 51,696 + 187 = 51,883 tokens",
        ),
    ],
)
def test_inference_caption_is_current_turn_only_and_preserves_missing_output(state, expected):
    assert Captions.inference(state) == expected


def test_a_partial_report_reads_in_progress_until_it_is_final():
    state = {
        "turn_number": 1,
        "phase": "Thinking",
        "inference_usage": {
            "turn_number": 1,
            "input": 100,
            "output": 20,
            "partial": True,
            "final": False,
        },
    }

    caption = Captions.inference(state)

    assert "in progress" in caption and "incomplete" not in caption and "at least" not in caption
    state["inference_usage"]["final"] = True
    caption = Captions.inference(state)
    assert "incomplete" in caption and "at least 20" in caption


def test_resume_seeds_latest_context_even_if_previous_turn_was_not_recorded(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    recordings.is_enabled = False
    # the console sends the task's event before it starts the recording
    recordings.on_event("task", RecordedTask.TASK)
    recordings.on_event("turn_started", {"turn_number": 1, "attempt": "one", "episode_id": "e"})
    recordings.on_event("usage", {"attempt": "one", "input": 51696, "output": 187})
    recordings.on_event("resume_context", {"history_retained_by_model": True, "previous_turn_committed": True})
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.is_enabled = True
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        assert job.caption_state["context_usage"]["tokens"] == 51883
        recordings.stop("Immediate Pause")

        RecordingWaits.await_finished(job)

        assert job.snapshot()["status"] == "saved", job.snapshot()
        assert RecordedTask.timeline_rows(job.directory)[0]["state"]["context_usage"]["tokens"] == 51883
        # a segment that was stopped takes no more usage
        job.on_event(
            "usage",
            {"attempt": "one", "input": 60000, "output": 200},
            latest_context={"tokens": 60200},
            turn_usage={"tokens": 60200},
        )
        assert job.caption_state["context_usage"]["tokens"] == 51883
    finally:
        recordings.close()
        source.close()


def test_episode_compaction_caption_persists_across_turns_and_resume(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    recordings.is_enabled = False
    # the console sends the task's event before it starts the recording
    recordings.on_event("task", RecordedTask.TASK)
    recordings.on_event("turn_started", {"attempt": "one", "turn_number": 1, "episode_id": "e"})
    usage = {
        "attempt": "one",
        "input": 100,
        "output": 20,
        "context": {"input": 80, "context_limit": 260096},
        "compaction": {"status": "completed", "count": 2, "episode_count": 2},
    }
    recordings.on_event("usage", usage)
    earlier = recordings.usage_tracker.latest_context
    recordings.on_event("turn_started", {"attempt": "two", "turn_number": 2, "episode_id": "e"})
    assert "Compactions: 2 total" in Captions.context({"context_usage": recordings.usage_tracker.latest_context})
    for status in ("none", "running", "interrupted", "failed"):
        recordings.on_event(
            "usage",
            {
                **usage,
                "attempt": "two",
                "compaction": {"status": status, "count": 0, "episode_count": 2},
            },
        )
        caption = Captions.context({"context_usage": recordings.usage_tracker.latest_context})
        assert "Compactions: 2 total" in caption
        if status != "none":
            assert f"({status})" in caption
    assert earlier["compaction"] == {"status": "completed", "count": 2, "episode_count": 2}

    recordings.on_event(
        "usage",
        {
            **usage,
            "attempt": "two",
            "compaction": {"status": "none", "count": 0, "episode_count": 2},
        },
    )
    recordings.is_enabled = True
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        recordings.on_event("turn_started", {"attempt": "three", "turn_number": 3, "episode_id": "e"})
        recordings.on_event(
            "usage",
            {
                **usage,
                "attempt": "three",
                "compaction": {"status": "none", "count": 0, "episode_count": 2},
            },
        )
        time.sleep(0.15)
        recordings.stop("Paused")
        RecordingWaits.await_finished(job)
        assert job.snapshot()["status"] == "saved", job.snapshot()
        assert all(
            "Compactions: 2 total" in Captions.context(row["state"])
            for row in RecordedTask.timeline_rows(job.directory)
        )
    finally:
        recordings.close()
        source.close()

    recordings.on_event("session", {"episode_id": "new"})
    assert recordings.usage_tracker.latest_context is None


def test_the_captioned_frames_refresh_the_context_without_changing_the_action_or_the_turn():
    renderer = CaptionedFrames(RecordedTask.TASK, StandInCameras.ROW)
    state = {"phase": "Thinking", "turn_number": 1}
    before = renderer.render({"state": state, "frames": {}}, 0)
    state = {
        **state,
        "context_usage": {
            "tokens": 207720,
            "output": 181,
            "context_limit": 260096,
            "turn_number": 71,
        },
    }

    after = renderer.render({"state": state, "frames": {}}, 0)

    box = ImageChops.difference(before, after).getbbox()
    assert box is not None
    assert 600 < box[0] < box[2] <= 1888
    assert 710 <= box[1] < box[3] < 744


def test_the_captioned_frames_place_the_cameras_at_their_tiles():
    colours = {"left": (200, 0, 0), "top": (0, 200, 0), "right": (0, 0, 200)}
    frames = {role: {"rgb": np.full((480, 640, 3), colour, dtype=np.uint8)} for role, colour in colours.items()}

    picture = np.asarray(
        CaptionedFrames(RecordedTask.TASK, StandInCameras.ROW).render(
            {"state": {"phase": CaptionPhase.STARTING}, "frames": frames}, 0
        )
    )

    for role, (x, y, width, height) in TILES.items():
        panel = picture[y : y + height, x : x + width]
        assert (panel == colours[role]).all(), role
        assert (picture[y - 1, x : x + width] != colours[role]).any(), role
        assert (picture[y + height, x : x + width] != colours[role]).any(), role


def test_one_camera_fills_the_left_beside_a_text_column():
    lay = CaptionedLayout.for_cameras(StandInCameras.SINGLE)
    assert lay.tiles == {"wrist": [0, 152, 960, 720]}
    assert TaskFiles.source_file_names(StandInCameras.SINGLE) == {"wrist": "wrist.mp4"}

    task = {"instruction": "Pick", "model_label": "vLLM", "action_format": "single_arm_pose"}
    renderer = CaptionedFrames(task, StandInCameras.SINGLE)
    assert renderer.model_caption.endswith("|  single_arm_pose")
    frames = {"wrist": {"rgb": np.full((480, 640, 3), 200, dtype=np.uint8)}}
    picture = np.asarray(renderer.render({"state": {"phase": CaptionPhase.STARTING}, "frames": frames}, 0))
    assert (picture[152:872, 0:960] == 200).all()
    assert not (picture[152:872, 992:1888] == 200).all()


def test_a_frame_is_kept_when_its_centre_lies_in_a_motion_span():
    # Span edges a hair off the frame boundaries keep or drop whole frames by their centre.
    cut = MotionCut([(1 / 30 - 1e-9, 3 / 30 + 1e-9), (8 / 30, 10 / 30)], 10)

    assert np.flatnonzero(cut.is_kept).tolist() == [1, 2, 8, 9] and cut.kept_count == 4
    assert MotionCut([], 10).kept_count == 0


def test_the_motion_filter_selects_each_run_of_kept_frames():
    cut = MotionCut([(1 / 30 - 1e-9, 3 / 30 + 1e-9), (8 / 30, 10 / 30)], 10)

    assert cut.select_filter() == "select='between(n,1,2)+between(n,8,9)',setpts=N/(30*TB)"


@pytest.mark.parametrize("robot_name", ["yam"])
def test_recording_after_reconnect_reads_the_new_cameras(enabled_console, monkeypatch):
    c = enabled_console
    ConsoleRig.pause_after_a_step(c)
    reader = MarkedCameras(c.robot.CAMERAS)
    monkeypatch.setattr(c.preview, "open_reader", lambda: reader)
    c.commands.run("reconnect_cameras", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)

    c.commands.run("recording", {"enabled": True})
    c.commands.run("resume", None)
    Waiting.until(lambda: c.recordings.active_segment is not None, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    Waiting.until(lambda: "gui-video-sampler" in reader.preview_threads, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
    c.commands.run("pause", None)
    Waiting.until(lambda: not c.is_busy, ConsoleRig.WAIT_S, ConsoleRig.POLL_S)
