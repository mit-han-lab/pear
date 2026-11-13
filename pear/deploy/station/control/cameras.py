import threading
from abc import ABC, abstractmethod

from pear.deploy.station.control.schemas import CameraFrame


class CameraSource(ABC):
    """A robot's cameras, as the robot observes them and the page previews and records them."""

    MAX_AGE_S = 0.2  # the age past which a frame is stale

    def read(self) -> dict[str, CameraFrame]:
        """Returns one fresh frame of each camera role, and raises when a camera has none."""
        frames, errors = self.read_preview()
        if errors:
            raise RuntimeError(f"Camera unavailable: {errors}")

        return frames

    @abstractmethod
    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns the latest fresh frames and, for each role without one, why."""

    @abstractmethod
    def close(self) -> None:
        """Stops the cameras."""


class CameraFeed(CameraSource):
    """The one camera source that the robot, the preview and the recordings read while its reader is replaced."""

    def __init__(self, reader: CameraSource | None) -> None:
        self.reader = reader  # the camera reader every read goes to; None while closed
        self.lock = threading.Lock()  # held by each read, so a reader closes only between reads

    def read(self) -> dict[str, CameraFrame]:
        """Returns one fresh frame of each camera role, all from the current reader."""
        with self.lock:
            return self._require_reader().read()

    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns the current reader's latest frames and, for each role without one, why."""
        with self.lock:
            return self._require_reader().read_preview()

    def set_reader(self, reader: CameraSource) -> None:
        """Reads from ``reader`` from now on."""
        with self.lock:
            self.reader = reader

    def close(self) -> None:
        """Closes the current reader after the read in progress; one that fails to close is closed again later."""
        with self.lock:
            if self.reader is not None:
                self.reader.close()
                self.reader = None

    def _require_reader(self) -> CameraSource:
        """Returns the current reader, and raises while there is none."""
        if self.reader is None:
            raise RuntimeError("Camera unavailable: no camera reader is open")

        return self.reader
