import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from pear.deploy.station.control.cameras import CameraFeed, CameraSource
from pear.deploy.station.exceptions import OperatorPaused
from pear.deploy.station.gui.console.periodic_thread import PeriodicThread
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreviewFrame:
    """One camera's latest preview: its JPEG, when its frame arrived and the frame's number."""

    jpeg: bytes
    received_monotonic_s: float
    frame_number: int


class Preview:
    """The cameras as the page shows them: the latest frame of each camera as JPEG and the cameras' errors, and the
    opening and reconnection of the cameras."""

    FRAME_RATE = 15  # previews encoded per second
    JPEG_QUALITY = 80  # of a preview
    THREAD_NAME = "gui-preview"  # of the thread that encodes the previews
    RECONNECTING_ERROR = "Reconnecting cameras"  # what the cameras show while their reader is replaced

    def __init__(
        self, camera_feed: CameraFeed, roles: tuple[str, ...], open_reader: Callable[[], CameraSource]
    ) -> None:
        self.camera_feed = camera_feed  # what the robot, the recordings and the previews read the cameras through
        self.roles = roles  # the robot's camera roles, in the page's order
        self.open_reader = open_reader  # opens a new reader of the robot's cameras
        self.lock = threading.Lock()  # held by every change of the previews and the errors
        self.frames: dict[str, PreviewFrame] = {}  # the latest preview of each role
        self.error_by_role: dict[str, str] = {}  # each camera's error
        self.error_text: str | None = None  # the cameras' errors in one sentence; None without one
        self.camera_error: str | None = None  # why no previews are encoded; None while they are
        self.encoding_thread: PeriodicThread | None = None  # None while no previews are encoded

        try:
            camera_feed.set_reader(open_reader())
        except Exception as exc:
            logger.exception("The cameras did not open")
            self.camera_error = str(exc)
        else:
            self._start_encoding()

    @property
    def is_encoding(self) -> bool:
        """Whether previews are encoded: the cameras' reader is open."""
        return self.encoding_thread is not None

    def get(self, role: str) -> PreviewFrame | None:
        """Returns the latest preview of ``role``; None while that camera has an error or no previews are encoded."""
        with self.lock:
            if self.encoding_thread is None or role in self.error_by_role:
                return None
            return self.frames.get(role)

    def status(self) -> dict[str, Any]:
        """Returns what the page shows of the cameras: while previews are encoded, their errors and each role's
        frame age and number; else why none are."""
        with self.lock:
            if self.encoding_thread is None:
                return {"error": self.camera_error, "roles": {}}

            return {
                "error": self.error_text,
                "errors": dict(self.error_by_role),
                "roles": {
                    role: {"age": time.monotonic() - frame.received_monotonic_s, "frame": frame.frame_number}
                    for role, frame in self.frames.items()
                },
            }

    def reconnect(self, cancellation: threading.Event) -> None:
        """Replaces the cameras' reader with a new one and encodes previews of it; it never moves the arms."""
        with self.lock:
            self.camera_error = self.RECONNECTING_ERROR
        reader = None

        try:
            self._stop_encoding()
            self.camera_feed.close()
            OperatorPaused.check_cancelled(cancellation)
            reader = self.open_reader()
            # a reader may still be connecting its cameras: its progress shows at once, while enabling the motors
            # and observing still require a complete set of fresh frames
            OperatorPaused.check_cancelled(cancellation)
            self.camera_feed.set_reader(reader)
            self._start_encoding()
        except BaseException as exc:
            with self.lock:
                self.camera_error = str(exc)
            if reader is not None and self.camera_feed.reader is not reader:
                reader.close()
            raise

    def close(self) -> None:
        """Stops encoding previews and closes the cameras' reader."""
        self._stop_encoding()
        self.camera_feed.close()

    def _start_encoding(self) -> None:
        """Starts encoding previews of the feed's reader, from no frame and no error."""
        with self.lock:
            self.frames, self.error_by_role, self.error_text, self.camera_error = {}, {}, None, None
            self.encoding_thread = PeriodicThread(
                self._encode_frames, 1 / self.FRAME_RATE, self.THREAD_NAME, is_paced_from_start=True
            )

    def _encode_frames(self) -> None:
        """Encodes the latest frame of each camera as JPEG, and records each camera's error in place of its frame."""
        try:
            frames, error_by_role = self.camera_feed.read_preview()
            jpeg_frames = {}
            for role, frame in frames.items():
                buffer = io.BytesIO()
                Image.fromarray(frame["rgb"]).save(buffer, format="JPEG", quality=self.JPEG_QUALITY)
                jpeg_frames[role] = PreviewFrame(
                    buffer.getvalue(), frame["received_monotonic_s"], frame["frame_number"]
                )

            with self.lock:
                self.frames.update(jpeg_frames)
                self.error_by_role = error_by_role
                self.error_text = "; ".join(f"{role}: {text}" for role, text in error_by_role.items()) or None
        except Exception as exc:
            if str(exc) != self.error_text:  # a failure that repeats at every frame is logged once
                logger.exception("The camera previews failed")
            with self.lock:
                self.error_text = str(exc)
                self.error_by_role = {role: str(exc) for role in self.roles}

    def _stop_encoding(self) -> None:
        """Stops encoding previews; raises when the encoding thread does not end."""
        encoding_thread = self.encoding_thread
        if encoding_thread is None:
            return

        encoding_thread.close()
        if encoding_thread.thread.is_alive():
            raise RuntimeError("Camera preview did not stop")
        with self.lock:
            self.encoding_thread = None
