import time
from pathlib import Path

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.robots.so101.config import WRIST_ROLE, SO101Config
from pear.deploy.station.control.schemas import Camera, CameraFrame

# lerobot is imported where the camera opens: it is heavy, and only a station with its camera needs it


class WristCamera(CameraSource):
    """The wrist camera through lerobot's OpenCVCamera, read as frames with the times they arrived."""

    CAMERA = Camera(WRIST_ROLE, "wrist", "Wrist Camera")  # the SO-101's one camera, whose role keys its frames
    START_TIMEOUT_MS = 2000  # the longest wait for the first frame

    def __init__(self, robot_config: SO101Config) -> None:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        camera_config = robot_config.camera
        self.role = camera_config.role  # what its frames are keyed by
        self.camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=Path(camera_config.path),
                fps=camera_config.fps,
                width=camera_config.width,
                height=camera_config.height,
                fourcc=camera_config.fourcc,
            )
        )

        self.camera.connect(warmup=True)
        self._last_frame_stamp: float | None = None  # of the last frame counted
        self._frame_count = 0  # of the different frames seen

        try:
            self.camera.async_read(timeout_ms=self.START_TIMEOUT_MS)  # starts lerobot's background reader
        except BaseException:
            self.camera.disconnect()
            raise

    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns the latest fresh frame and, when there is none, why."""
        camera, role = self.camera, self.role
        thread = camera.thread
        if thread is None or not thread.is_alive():
            return {}, {role: "Camera reader stopped; reconnect the camera"}

        with camera.frame_lock:
            frame, stamp = camera.latest_frame, camera.latest_timestamp
        if frame is None or stamp is None:
            return {}, {role: "Waiting for camera frames"}
        # lerobot stamps a frame with time.perf_counter(), the same clock as time.monotonic() on Linux
        if time.monotonic() - stamp > self.MAX_AGE_S:
            return {}, {role: "Camera frame is stale"}

        if stamp != self._last_frame_stamp:
            self._last_frame_stamp, self._frame_count = stamp, self._frame_count + 1
        # an RGB image of height x width x 3 bytes, with its stamp and its number among the frames seen
        return {role: {"rgb": frame.copy(), "received_monotonic_s": stamp, "frame_number": self._frame_count}}, {}

    def close(self) -> None:
        """Disconnects the camera."""
        if self.camera.is_connected:
            self.camera.disconnect()
