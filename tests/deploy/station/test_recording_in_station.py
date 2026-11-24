import json
import os
import queue
import signal
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from imageio_ffmpeg import count_frames_and_secs

from console_rig import ConsoleRig, MarkedCameras
from pear.deploy.station.records.recording.ffmpeg import FFmpeg
from pear.deploy.station.records.recording.motion_cut import MotionCut
from pear.deploy.station.records.recording.recordings import Recordings
from pear.deploy.station.records.task_files import TaskFiles
from recorded_task import RecordedTask, RecordingWaits
from stand_in_cameras import StandInCameras
from waiting import Waiting

SOURCES = RecordedTask.SOURCE_FILES  # each camera's own video in a segment of StandInCameras.ROW, by role
ENCODER_SHOWS_S = 15  # for the encoder of a camera's own video to show among the processes


def test_real_encoding_keeps_time_and_marks_the_motion(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    assert recordings.is_enabled is True
    source = StandInCameras(StandInCameras.ROW)
    robot = {"control_state": "HOLD"}
    try:
        recordings.start(RecordedTask.TASK, source, lambda: dict(robot))
        job = recordings.active_segment
        assert recordings.is_recording()
        time.sleep(0.5)
        robot["control_state"] = "RUNNING"
        time.sleep(0.6)
        robot["control_state"] = "HOLD"
        time.sleep(0.5)
        requested = time.monotonic()
        recordings.stop("Paused", stopped_at=requested)
        recordings.stop("Repeated stop")
        assert time.monotonic() - requested < 0.1
        assert not recordings.is_recording()

        RecordingWaits.await_finished(job)
        info = job.snapshot()
        assert info["status"] == "saved", info
        assert abs(info["frames"] / 30 - (requested - job.started_at)) < 1 / 30 + 0.01
        motion_frames = sum(end - start for start, end in info["motion_intervals"])
        assert 0.4 <= motion_frames / 30 <= 0.8

        rows = RecordedTask.timeline_rows(job.directory)
        assert all(set(row["state"]) == {"moving", "control_state"} for row in rows)
        moving_rows = [row for row in rows if row["state"]["moving"]]
        assert moving_rows and all(row["state"]["control_state"] == "RUNNING" for row in moving_rows)
        assert sum(row["end_frame"] - row["start_frame"] for row in moving_rows) == motion_frames
        assert rows[-1]["end_frame"] == info["frames"]
        assert set(rows[0]["cameras"]) == {"left", "top", "right"}

        # each camera's own video has a frame for every frame of the timeline, so a render reads them in step
        assert info["sources"] == SOURCES and info["source_error"] is None
        assert type(info["started_wall_s"]) is float
        assert abs(time.time() - info["started_wall_s"] - (time.monotonic() - job.started_at)) < 1
        saved = json.loads((job.directory / "recording.json").read_text())
        assert saved["sources"] == SOURCES and saved["started_wall_s"] == info["started_wall_s"]
        for role, name in SOURCES.items():
            frames, _ = count_frames_and_secs(str(job.directory / name))
            assert frames == info["frames"], role
        # the cameras' videos are the only videos a segment records
        assert sorted(path.name for path in job.directory.glob("*.mp4")) == sorted(SOURCES.values())
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


def test_immediate_pause_produces_decodable_camera_videos(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        recordings.stop("Immediate Pause")

        RecordingWaits.await_finished(job)

        assert job.snapshot()["status"] == "saved", job.snapshot()
        for name in SOURCES.values():
            frames, _ = count_frames_and_secs(str(job.directory / name))
            assert frames >= 1, name
        assert job.snapshot()["motion_intervals"] == []
    finally:
        recordings.close()
        source.close()


def test_dropped_pictures_keep_the_motion_on_time(tmp_path, monkeypatch):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    robot = {"control_state": "HOLD"}
    try:
        recordings.start(RecordedTask.TASK, source, lambda: dict(robot))
        job = recordings.active_segment

        def skip_picture(packet):
            raise queue.Full

        monkeypatch.setattr(job.packets, "put_nowait", skip_picture)
        time.sleep(0.15)
        robot["control_state"] = "RUNNING"
        time.sleep(0.3)
        robot["control_state"] = "HOLD"
        time.sleep(0.15)
        recordings.stop("Paused")

        RecordingWaits.await_finished(job)

        info = job.snapshot()
        assert info["status"] == "saved", info
        assert info["dropped_samples"] > 5
        assert 0.2 <= sum(end - start for start, end in info["motion_intervals"]) / 30 <= 0.4
        rows = RecordedTask.timeline_rows(job.directory)
        assert any(row["state"]["moving"] for row in rows) and not rows[-1]["state"]["moving"]
    finally:
        recordings.close()
        source.close()


class CameraEncoders:
    """The FFmpeg processes that encode a recording segment's camera videos, found among the host's processes."""

    @staticmethod
    def kill(segment_directory: Path, names: list[str]) -> None:
        """Kills the encoder of each camera video in ``names`` once it shows."""
        for name in names:
            target = str(segment_directory / TaskFiles.partial_name(name))
            Waiting.until(partial(CameraEncoders._is_shown, target), ENCODER_SHOWS_S, RecordingWaits.POLL_S)
            os.kill(CameraEncoders._find(target), signal.SIGKILL)

    @classmethod
    def _is_shown(cls, target: str) -> bool:
        """Whether a process writes ``target``."""
        return cls._find(target) is not None

    @staticmethod
    def _find(target: str) -> int | None:
        """Returns the process whose command line writes ``target``, None while there is none."""
        for entry in Path("/proc").iterdir():
            try:
                if target in (entry / "cmdline").read_bytes().decode(errors="replace"):
                    return int(entry.name)
            except (OSError, ValueError):
                continue
        return None


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to find the encoder")
def test_failed_camera_source_keeps_the_other_cameras(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        CameraEncoders.kill(job.directory, ["head.mp4"])
        time.sleep(0.3)
        recordings.stop("Paused")

        RecordingWaits.await_finished(job)

        info = job.snapshot()
        assert info["status"] == "saved", info
        assert info["sources"] == {**SOURCES, "top": None}
        assert "head.mp4" in info["source_error"]
        assert (job.directory / "left_wrist.mp4").is_file()
        assert not (job.directory / "head.mp4").exists()
        assert not list(job.directory.glob("*.partial.mp4"))
    finally:
        recordings.close()
        source.close()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to find the encoder")
def test_a_segment_whose_every_camera_video_failed_fails(tmp_path):
    recordings = Recordings(tmp_path, StandInCameras.ROW)
    source = StandInCameras(StandInCameras.ROW)
    try:
        recordings.start(RecordedTask.TASK, source, lambda: {"control_state": "HOLD"})
        job = recordings.active_segment
        CameraEncoders.kill(job.directory, list(SOURCES.values()))
        time.sleep(0.3)
        recordings.stop("Paused")

        RecordingWaits.await_finished(job)

        info = job.snapshot()
        assert info["status"] == "failed", info
        assert all(name in info["error"] for name in SOURCES.values())
        assert not list(job.directory.glob("*.mp4"))
    finally:
        recordings.close()
        source.close()


def test_a_frame_is_kept_when_its_centre_lies_in_a_motion_span():
    # Span edges a hair off the frame boundaries keep or drop whole frames by their centre.
    cut = MotionCut([(1 / 30 - 1e-9, 3 / 30 + 1e-9), (8 / 30, 10 / 30)], 10)

    assert np.flatnonzero(cut.is_kept).tolist() == [1, 2, 8, 9] and cut.kept_count == 4
    assert MotionCut([], 10).kept_count == 0


def test_the_motion_filter_selects_each_run_of_kept_frames_of_a_clip():
    cut = MotionCut([(1 / 30 - 1e-9, 3 / 30 + 1e-9), (8 / 30, 10 / 30)], 10)

    assert cut.select_filter(0, 10) == "select='between(n,1,2)+between(n,8,9)',setpts=N/(30*TB)"
    # a clip from frame 5 numbers its frames from 0
    assert cut.select_filter(5, 5) == "select='between(n,3,4)',setpts=N/(30*TB)"


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
