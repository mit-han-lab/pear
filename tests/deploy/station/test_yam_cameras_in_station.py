import ctypes
import dataclasses
import io
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from config_files import ConfigFiles
from pear.deploy.station.agent_loop.messages import ObservationMessage
from pear.deploy.station.control.cameras import CameraFeed
from pear.deploy.station.control.robots.yam.config import CameraConfig, CameraCrop, YAMConfig
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE
from pear.deploy.station.control.robots.yam.hardware import cameras
from pear.deploy.station.control.robots.yam.hardware.cameras import CaptureWorker, LiveCameras
from pear.deploy.station.gui.console.preview import Preview
from waiting import Waiting

WAIT_S = 12  # the longest a camera's recovery may take
POLL_S = 0.02  # between two looks at the cameras


class StandInCapture:
    """Capture processes that fail, hang or finish slowly on demand, each run as a capture process of the cameras."""

    @staticmethod
    def capture(serial, pixels, stamp, lock, messages, stop, reset):
        """Fails, hangs or streams, by the marker file the serial names."""
        if "persistent" in serial and not reset:
            Path(serial).write_text("soft retry attempted")
            messages.send("Frame didn't arrive within 1000")
            return
        if serial != "healthy" and not Path(serial).exists():
            Path(serial).write_text("failed")
            if "hung" in serial:
                lock.acquire()
                # Deliberately hold both the Python GIL and the shared frame lock.
                ctypes.PyDLL(None).sleep(60)
            else:
                messages.send("Frame didn't arrive within 3000")
            return
        if serial != "healthy":
            Path(serial).write_text("reset" if reset else "restarted")

        data = np.frombuffer(pixels, dtype=np.uint8)
        number = 0
        while not stop.wait(0.025):
            number += 1
            with lock:
                data[:] = number % 255
                stamp[0], stamp[1] = time.monotonic(), number

    @staticmethod
    def slow_cleanup(serial, pixels, stamp, lock, messages, stop, reset):
        """Streams nothing and finishes its cleanup slowly once told to stop."""
        Path(serial).write_text("opened")
        stop.wait(10)
        # A real read may still be in flight when close requests shutdown.
        time.sleep(0.6)
        Path(serial).write_text("closed gracefully")


class YAMSettings:
    """The YAM settings of the tests' station with the cameras a test gives."""

    @staticmethod
    def with_cameras(camera_configs: list[CameraConfig]) -> YAMConfig:
        """Returns the tests' YAM settings whose cameras are ``camera_configs``."""
        return dataclasses.replace(ConfigFiles.load_configs("yam")[1], cameras=camera_configs)


@pytest.fixture
def source():
    """Live cameras with no thread and no device, three fresh frames in their cache."""
    # No SDK or devices: exercise the cache/observation boundary directly.
    c = LiveCameras.__new__(LiveCameras)
    c._lock = threading.Lock()
    c._exit_event = threading.Event()
    c._roles = ("left", "top", "right")
    c._crops = {}
    c._threads = []
    c._errors = {}
    c._frames = {
        role: {
            "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "received_monotonic_s": time.monotonic(),
            "frame_number": 1,
        }
        for role in c._roles
    }

    return c


def test_partial_preview_does_not_weaken_model_observations(source):
    source._errors["top"] = "Device disconnected"

    frames, errors = source.read_preview()

    assert set(frames) == {"left", "right"}
    assert errors == {"top": "Device disconnected"}
    frames["left"]["rgb"][:] = 255
    assert not source._frames["left"]["rgb"].any()
    with pytest.raises(RuntimeError, match="Device disconnected"):
        source.read()


@pytest.mark.parametrize("failure", ["stale", "missing", "closed"])
def test_stale_missing_and_closed_frames_never_reach_policy(source, failure):
    if failure == "stale":
        source._frames["top"]["received_monotonic_s"] -= 1
    elif failure == "missing":
        del source._frames["top"]
    else:
        source.close()

    frames, errors = source.read_preview()

    assert "top" in errors and "top" not in frames
    with pytest.raises(RuntimeError):
        source.read()


def test_complete_observation_stays_strict_and_copied(source):
    frames = source.read()

    assert set(frames) == set(source._roles)
    frames["top"]["rgb"][:] = 255
    assert not source._frames["top"]["rgb"].any()


def test_top_crop_is_identical_for_preview_and_model_input(source, tmp_path):
    source._crops = {"top": CameraCrop(x_offset=80, y_offset=60, width=480, height=360)}
    y, x = np.indices((480, 640))
    original = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
    now = time.monotonic()
    for frame in source._frames.values():
        frame.update(rgb=original.copy(), received_monotonic_s=now)

    preview_frames, errors = source.read_preview()
    model_frames = source.read()
    assert not errors
    for role in source._roles:
        expected = original[60:420, 80:560] if role == "top" else original
        for frames in (preview_frames, model_frames):
            np.testing.assert_array_equal(frames[role]["rgb"], expected)
            assert frames[role]["received_monotonic_s"] == now
            assert frames[role]["frame_number"] == 1
            assert not np.shares_memory(frames[role]["rgb"], source._frames[role]["rgb"])
        np.testing.assert_array_equal(source._frames[role]["rgb"], original)

    # Exercise both downstream JPEG encoders without opening hardware or a policy; the preview owns its reader.
    reader = SimpleNamespace(read_preview=lambda: (preview_frames, {}), close=lambda: None)
    preview = Preview(CameraFeed(None), tuple(preview_frames), lambda: reader)
    try:
        deadline = time.monotonic() + 2
        while not all(preview.get(role) for role in source._roles):
            assert time.monotonic() < deadline, preview.error_text
            time.sleep(0.01)
        raw = {
            "images": {role: frame["rgb"] for role, frame in model_frames.items()},
            "camera_timestamps": model_frames,
            "captured_monotonic_s": time.monotonic(),
        }
        ObservationMessage.build(raw, SimpleNamespace(model_dump=lambda: {}), BIMANUAL_POSE, tmp_path, "crop", None, ())
        for role, name in (("top", "head"), ("left", "left_wrist"), ("right", "right_wrist")):
            expected_size = (480, 360) if role == "top" else (640, 480)
            with Image.open(io.BytesIO(preview.get(role).jpeg)) as shown:
                assert shown.size == expected_size
            with Image.open(tmp_path / f"crop-{name}.jpg") as sent:
                assert sent.size == expected_size
    finally:
        preview.close()

    model_frames["top"]["rgb"][:] = 0
    np.testing.assert_array_equal(preview_frames["top"]["rgb"], original[60:420, 80:560])


def test_capture_cleanup_failure_preserves_original_error(source, monkeypatch, caplog):
    import pyrealsense2 as rs

    waits = []

    class Frame:
        def get_data(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def get_frame_number(self):
            return 16

    class Pipeline:
        def start(self, config):
            pass

        def wait_for_frames(self, timeout):
            waits.append(timeout)
            if len(waits) > 16:
                raise RuntimeError("Device disconnected")
            return SimpleNamespace(get_color_frame=lambda: Frame())

        def stop(self):
            raise RuntimeError("stop() cannot be called before start()")

    monkeypatch.setattr(rs, "pipeline", Pipeline)
    monkeypatch.setattr(
        rs,
        "context",
        lambda: SimpleNamespace(query_devices=lambda: [SimpleNamespace(get_info=lambda key: "stand-in-camera")]),
    )
    # the capture's logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(logging.getLogger(cameras.__name__), "handlers", [caplog.handler])
    errors = []
    messages = SimpleNamespace(send=errors.append, close=lambda: None)

    CaptureWorker.main(
        "stand-in-camera",
        bytearray(480 * 640 * 3),
        [0.0, 0.0],
        threading.Lock(),
        messages,
        threading.Event(),
        False,
    )

    assert waits[:15] == [3000] * 15 and waits[15:] == [1000, 1000]
    assert errors == ["Device disconnected"]
    assert "cleanup failed" in caplog.text


def test_recovery_reset_targets_only_the_configured_serial(monkeypatch):
    import pyrealsense2 as rs

    resets, errors = [], []
    devices = [
        SimpleNamespace(get_info=lambda key, s=s: s, hardware_reset=lambda s=s: resets.append(s))
        for s in ("left", "top", "right")
    ]
    monkeypatch.setattr(rs, "context", lambda: SimpleNamespace(query_devices=lambda: devices))

    CaptureWorker.main(
        "top",
        None,
        None,
        None,
        SimpleNamespace(send=errors.append, close=lambda: None),
        SimpleNamespace(wait=lambda delay: True),
        True,
    )

    assert resets == ["top"] and not errors


def test_missing_usb_device_is_reported_without_opening_or_resetting(monkeypatch):
    import pyrealsense2 as rs

    monkeypatch.setattr(rs, "context", lambda: SimpleNamespace(query_devices=lambda: []))
    errors = []

    CaptureWorker.main(
        "top",
        None,
        None,
        None,
        SimpleNamespace(send=errors.append, close=lambda: None),
        None,
        True,
    )

    assert errors == ["Camera not detected over USB; check its USB connection"]


def test_failed_camera_recovers_and_other_camera_stays_live(monkeypatch, tmp_path):
    monkeypatch.setattr(cameras.CaptureWorker, "main", StandInCapture.capture)
    marker = tmp_path / "failed"
    source = cameras.LiveCameras(
        YAMSettings.with_cameras(
            [
                CameraConfig(role="left", serial="healthy"),
                CameraConfig(
                    role="top",
                    serial=str(marker),
                    crop=CameraCrop(x_offset=80, y_offset=60, width=480, height=360),
                ),
            ]
        )
    )

    try:
        Waiting.until(lambda: "left" in source.read_preview()[0] and "top" in source.read_preview()[1], WAIT_S, POLL_S)
        first = source.read_preview()[0]["left"]["frame_number"]
        Waiting.until(lambda: not source.read_preview()[1], WAIT_S, POLL_S)
        assert source.read()["left"]["frame_number"] > first
        assert source.read()["left"]["rgb"].shape == (480, 640, 3)
        assert source.read()["top"]["rgb"].shape == (360, 480, 3)
        assert marker.read_text() == "restarted"
    finally:
        source.close()

    assert all(not t.is_alive() for t in source._threads)


def test_hung_sdk_and_abandoned_lock_cannot_stall_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(cameras.CaptureWorker, "main", StandInCapture.capture)
    monkeypatch.setattr(cameras.LiveCameras, "START_TIMEOUT_S", 1.5)
    marker = tmp_path / "hung"
    source = cameras.LiveCameras(YAMSettings.with_cameras([CameraConfig(role="top", serial=str(marker))]))

    try:
        Waiting.until(marker.exists, WAIT_S, POLL_S)
        start = time.monotonic()
        for _ in range(20):
            source.read_preview()
            threading.Event().wait(0.01)
        assert time.monotonic() - start < 0.6
        Waiting.until(lambda: not source.read_preview()[1], WAIT_S, POLL_S)
        assert marker.read_text() == "restarted"
    finally:
        source.close()


def test_reset_is_limited_to_camera_stream_failures():
    assert cameras.LiveCameras.needs_reset("Frame didn't arrive within 3000")
    assert cameras.LiveCameras.needs_reset("Camera worker timed out")
    assert not cameras.LiveCameras.needs_reset("Device or resource busy")
    assert not cameras.LiveCameras.needs_reset("No device connected")
    assert not cameras.LiveCameras.needs_reset("Permission denied")
    assert cameras.LiveCameras.needs_reset("xioctl(VIDIOC_S_FMT) failed, errno=5 Last Error: Input/output error")


def test_repeated_stream_failure_escalates_after_soft_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(cameras.CaptureWorker, "main", StandInCapture.capture)
    marker = tmp_path / "persistent"
    source = cameras.LiveCameras(YAMSettings.with_cameras([CameraConfig(role="top", serial=str(marker))]))

    try:
        Waiting.until(marker.exists, WAIT_S, POLL_S)
        assert marker.read_text() == "soft retry attempted"
        Waiting.until(lambda: not source.read_preview()[1], WAIT_S, POLL_S)
        assert marker.read_text() == "reset"
    finally:
        source.close()


def test_close_allows_inflight_read_and_cleanup_to_finish(monkeypatch, tmp_path):
    monkeypatch.setattr(cameras.CaptureWorker, "main", StandInCapture.slow_cleanup)
    marker = tmp_path / "graceful"
    source = cameras.LiveCameras(YAMSettings.with_cameras([CameraConfig(role="top", serial=str(marker))]))
    Waiting.until(marker.exists, WAIT_S, POLL_S)

    source.close()

    assert marker.read_text() == "closed gracefully"
    assert all(not t.is_alive() for t in source._threads)
