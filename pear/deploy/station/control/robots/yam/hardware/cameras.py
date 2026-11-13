import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.robots.yam.config import CameraConfig, YAMConfig
from pear.deploy.station.control.schemas import CameraFrame
from pear.utils.file_utils import write_text_atomically
from pear.utils.logging_utils import get_logger
from pear.utils.worker_process import WorkerProcess

FRAME_SHAPE = (CameraConfig.FRAME_HEIGHT, CameraConfig.FRAME_WIDTH, 3)  # of an RGB frame: rows, columns, colors

logger = get_logger(__name__, default_level="INFO")


class CaptureWorker:
    """The capture process of one RealSense camera, which copies each frame into memory its supervisor shares."""

    # the camera's SDK may hold the GIL or hang, so it never runs in the process that supervises the cameras
    FRAME_TIMEOUT_MS = 1000  # for each frame once the stream is warm
    RESET_WAIT_S = 3  # after a hardware reset, before the stream starts
    MESSAGE_CHARS = 1000  # of an error message sent to the supervisor

    def __init__(
        self,
        serial: str,
        pixels: Any,
        frame_stamp: Any,
        lock: Any,
        error_sender: Connection,
        stop_event: Any,
        resets_camera: bool,
    ) -> None:
        self.serial = serial
        self.pixels = pixels  # the shared memory of the latest frame
        self.frame_stamp = frame_stamp  # the shared monotonic time and number of the latest frame
        self.lock = lock  # held while the frame and its stamp change
        self.error_sender = error_sender  # where the process's error goes to its supervisor
        self.stop_event = stop_event  # set by the supervisor to end the process
        self.resets_camera = resets_camera  # whether the camera is reset in hardware before its stream starts

    @staticmethod
    def main(
        serial: str,
        pixels: Any,
        frame_stamp: Any,
        lock: Any,
        error_sender: Connection,
        stop_event: Any,
        resets_camera: bool,
    ) -> None:
        """The target of the capture process."""
        CaptureWorker(serial, pixels, frame_stamp, lock, error_sender, stop_event, resets_camera).run()

    def run(self) -> None:
        """Streams the camera's frames into the shared memory until told to stop; an error goes to the supervisor."""
        import pyrealsense2 as rs  # the camera's SDK: imported in the capture process alone

        pipeline = None
        try:
            context = rs.context()
            device = next(
                (
                    device
                    for device in context.query_devices()
                    if device.get_info(rs.camera_info.serial_number) == self.serial
                ),
                None,
            )
            if device is None:
                raise RuntimeError("Camera not detected over USB; check its USB connection")
            if self.resets_camera:
                # this camera alone is reset, never its hub or the CAN adapters
                logger.warning("Camera %s: requesting hardware reset", self.serial)
                device.hardware_reset()
                if self.stop_event.wait(self.RESET_WAIT_S):
                    return

            pipeline, _ = RealSenseCameras.start_color_stream(self.serial)
            target = np.frombuffer(self.pixels, dtype=np.uint8).reshape(FRAME_SHAPE)
            warmup_frames = RealSenseCameras.WARMUP_FRAMES
            while not self.stop_event.is_set():
                timeout_ms = RealSenseCameras.WARMUP_TIMEOUT_MS if warmup_frames else self.FRAME_TIMEOUT_MS
                frame = pipeline.wait_for_frames(timeout_ms).get_color_frame()
                if not frame:
                    raise RuntimeError("Missing RGB frame")
                if warmup_frames:
                    warmup_frames -= 1
                    continue
                rgb = np.asarray(frame.get_data())
                with self.lock:
                    np.copyto(target, rgb)
                    self.frame_stamp[0] = time.monotonic()
                    self.frame_stamp[1] = frame.get_frame_number()
        except Exception as error:
            logger.exception("Camera %s: capture failed", self.serial)
            self.error_sender.send(str(error)[: self.MESSAGE_CHARS])
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    logger.exception("Camera %s cleanup failed", self.serial)
            self.error_sender.close()


class LiveCameras(CameraSource):
    """The robot's RealSense cameras, each read by a capture process that a thread of its own supervises and starts
    again when it fails."""

    # a camera that fails leaves the other cameras' previews as they are
    START_TIMEOUT_S = 15.0  # for a capture process's first frame
    FRAME_TIMEOUT_S = 6.0  # between two frames
    RESET_INTERVAL_S = 60.0  # between two hardware resets of a camera
    STOP_GRACE_S = 3.5  # lets an in-flight warmup wait and the pipeline's stop finish before the process is ended
    POLL_S = 0.02  # between two looks at a capture process, and the longest wait for its lock
    HEALTHY_STREAM_S = 10  # a camera that streamed this long has its failures counted from zero again
    MAX_RETRY_DELAY_S = 30  # between two capture processes of a camera
    MAX_BACKOFF_STEPS = 5  # the delay doubles with each failure up to 2 ** this many seconds
    RESETTING_FAILURES = 2  # failures in a row before a hardware reset is considered
    TERMINATE_JOIN_S = 1  # after SIGTERM, and again after SIGKILL, for a capture process to end
    CLOSE_MARGIN_S = 2.5  # past the longest end of a capture process, for the supervising threads to end
    # the errors after which a camera is reset in hardware: a busy, a refused or a missing camera may be another
    # program's and is never reset
    RESET_ERRORS = (
        "Frame didn't arrive",
        "Missing RGB",
        "worker timed out",
        "xioctl(VIDIOC_S_FMT) failed, errno=5",
    )

    def __init__(self, robot_config: YAMConfig) -> None:
        cameras = robot_config.cameras
        if not cameras:
            raise ValueError("Live capture requires configured cameras")

        self._exit_event = threading.Event()  # set to end every supervising thread
        self._lock = threading.Lock()  # guards the frames and the errors below
        self._frames: dict[str, CameraFrame] = {}  # each camera's latest frame, by its role
        self._errors = {camera.role: "Connecting camera" for camera in cameras}  # why a camera has no frame
        self._roles = tuple(camera.role for camera in cameras)
        self._crops = {camera.role: camera.crop for camera in cameras}
        self._threads = [
            threading.Thread(target=self._supervise, args=(camera,), daemon=True, name=f"rgb-{camera.role}")
            for camera in cameras
        ]
        for thread in self._threads:
            thread.start()

    def read_preview(self) -> tuple[dict[str, CameraFrame], dict[str, str]]:
        """Returns the fresh frames, cropped, and why each other camera has none."""
        with self._lock:
            frames, errors = {}, dict(self._errors)
            now = time.monotonic()

            for role in self._roles:
                frame = self._frames.get(role)
                if self._exit_event.is_set():
                    errors[role] = "Camera reader closed"
                elif role in errors:
                    continue
                elif frame is None:
                    errors[role] = "Waiting for camera frames"
                elif now - frame["received_monotonic_s"] > self.MAX_AGE_S:
                    errors[role] = "Camera frame is stale"
                else:
                    rgb, crop = frame["rgb"], self._crops.get(role)
                    if crop is not None:
                        # one crop for the preview and the observations; the raw frame and its stamps are kept,
                        # without resampling
                        rgb = rgb[
                            crop.y_offset : crop.y_offset + crop.height, crop.x_offset : crop.x_offset + crop.width
                        ]
                    frames[role] = frame | {"rgb": rgb.copy()}

            return frames, errors

    def close(self) -> None:
        """Ends every capture process and its supervising thread, and raises if a thread did not end in time."""
        self._exit_event.set()
        deadline = time.monotonic() + self.STOP_GRACE_S + self.CLOSE_MARGIN_S

        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Camera reader did not stop")

    def _supervise(self, camera: CameraConfig) -> None:
        """Runs one capture process of ``camera`` after another until the cameras close, waiting longer after each
        failure and resetting a camera that keeps failing."""
        failures, last_reset_time, resets_camera = 0, float("-inf"), False

        while not self._exit_event.is_set():
            error, start_time, last_frame_time = self._run_capture(camera, resets_camera)
            if resets_camera:
                last_reset_time = start_time
            if self._exit_event.is_set():
                break

            if last_frame_time - start_time > self.HEALTHY_STREAM_S:
                failures = 0  # the camera streamed long enough before this failure
            failures += 1
            delay_s = min(self.MAX_RETRY_DELAY_S, 2 ** min(failures - 1, self.MAX_BACKOFF_STEPS))
            # first the stream is opened again plainly: one passing timeout must not cut the camera off with a reset
            resets_camera = (
                failures >= self.RESETTING_FAILURES
                and self.needs_reset(error)
                and time.monotonic() - last_reset_time >= self.RESET_INTERVAL_S
            )
            with self._lock:
                self._frames.pop(camera.role, None)
                self._errors[camera.role] = f"{error}; retrying in {delay_s}s"
            logger.warning(
                "Camera %s (%s): %s; retry in %ss%s",
                camera.role,
                camera.serial,
                error,
                delay_s,
                " with camera reset" if resets_camera else "",
            )
            self._exit_event.wait(delay_s)

    def _run_capture(self, camera: CameraConfig, resets_camera: bool) -> tuple[str, float, float]:
        """Runs one capture process of ``camera`` until it fails, stalls or the cameras close, and returns why it
        ended, when it started and when its last frame arrived."""
        context = WorkerProcess.CONTEXT
        # a killed process can leave its lock held: each process gets new memory and a new lock, which the supervisor
        # only ever takes with a timeout
        pixels = context.RawArray("B", int(np.prod(FRAME_SHAPE)))
        frame_stamp = context.RawArray("d", 2)  # the monotonic time and the number of the latest frame
        lock, stop_event = context.Lock(), context.Event()
        receiver, error_sender = context.Pipe(duplex=False)
        worker = WorkerProcess(
            CaptureWorker.main,
            (camera.serial, pixels, frame_stamp, lock, error_sender, stop_event, resets_camera),
            f"camera-{camera.role}",
            self.TERMINATE_JOIN_S,
            is_group_leader=False,
        )
        error = "Camera worker exited"
        start_time, last_frame_time = time.monotonic(), 0.0

        try:
            worker.start()
            error_sender.close()
            while not self._exit_event.wait(self.POLL_S):
                if receiver.poll():
                    try:
                        error = receiver.recv()
                    except EOFError:
                        pass
                    break

                frame = None
                if lock.acquire(timeout=self.POLL_S):
                    try:
                        if frame_stamp[0] > last_frame_time:
                            last_frame_time = frame_stamp[0]
                            frame = {
                                "rgb": np.frombuffer(pixels, dtype=np.uint8).reshape(FRAME_SHAPE).copy(),
                                "received_monotonic_s": last_frame_time,
                                "frame_number": int(frame_stamp[1]),
                            }
                    finally:
                        lock.release()
                if frame is not None:
                    with self._lock:
                        self._frames[camera.role] = frame
                        self._errors.pop(camera.role, None)

                frame_timeout_s = self.FRAME_TIMEOUT_S if last_frame_time else self.START_TIMEOUT_S
                if time.monotonic() - (last_frame_time or start_time) > frame_timeout_s:
                    error = "Camera worker timed out"
                    break
                if not worker.is_alive():
                    break
        except Exception as exception:
            logger.exception("Camera %s (%s): its capture process failed", camera.role, camera.serial)
            error = str(exception)
        finally:
            stop_event.set()
            worker.stop(self.STOP_GRACE_S)
            receiver.close()
            error_sender.close()

        return error, start_time, last_frame_time

    @classmethod
    def needs_reset(cls, error: str) -> bool:
        """Whether a camera that failed with ``error`` may be reset in hardware."""
        return any(text in error for text in cls.RESET_ERRORS)


class RealSenseCameras:
    """The RealSense cameras on the USB: their identities, a camera's RGB stream, and snapshots of several at once."""

    WARMUP_FRAMES = 15  # the frames a started stream drops before its frames are read
    WARMUP_TIMEOUT_MS = 3000  # for each of those frames
    SNAPSHOT_FRAME_TIMEOUT_MS = 3000  # for each frame of a snapshot
    JPEG_QUALITY = 95  # of a snapshot's image
    JSON_INDENT = 2  # of a snapshot's record

    @staticmethod
    def discover() -> list[dict[str, Any]]:
        """Returns the serial, the port, the model and the firmware of every connected RealSense camera."""
        import pyrealsense2 as rs  # the camera's SDK: imported where a camera is used

        context = rs.context()

        return [
            {
                "serial": device.get_info(rs.camera_info.serial_number),
                "usb_serial": device.get_info(rs.camera_info.asic_serial_number),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "model": device.get_info(rs.camera_info.name),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
            for device in context.query_devices()
        ]

    @classmethod
    def snapshot(
        cls, serials: list[str], output_dir: Path, frame_count: int, roles: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Captures every camera of ``serials`` at once, writes an index page of their images under ``output_dir``
        and returns each camera's record."""
        if not serials or len(set(serials)) != len(serials):
            raise ValueError("Provide distinct camera serials")
        available_serials = {device["serial"] for device in cls.discover()}
        if missing_serials := set(serials) - available_serials:
            raise ValueError(f"RealSense serials not found: {sorted(missing_serials)}; use SDK, not USB serials")

        # every camera streams at once: the capture exercises the USB's real bandwidth
        with ThreadPoolExecutor(max_workers=len(serials)) as pool:
            futures = [pool.submit(cls.capture, serial, output_dir, frame_count) for serial in serials]
            results = [future.result() for future in futures]

        sections = [
            f"<h2>{roles[result['serial']]} — {result['serial']}</h2>"
            f'<img src="{result["serial"]}.jpg" width="{CameraConfig.FRAME_WIDTH}">'
            for result in results
        ]
        (output_dir / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>YAM camera identity</title>'
            '<body style="background:#18202b;color:white;font-family:sans-serif">'
            "<h1>YAM station camera snapshots</h1>" + "".join(sections)
        )

        return results

    @classmethod
    def capture(cls, serial: str, output_dir: Path, frame_count: int) -> dict[str, Any]:
        """Saves the last of ``frame_count`` frames of a camera with the stream's statistics and intrinsics, and
        returns that record."""
        import pyrealsense2 as rs
        from PIL import Image

        pipeline, profile = cls.start_color_stream(serial)
        try:
            for _ in range(cls.WARMUP_FRAMES):
                pipeline.wait_for_frames(cls.WARMUP_TIMEOUT_MS)
            times, numbers = [], []
            for _ in range(frame_count):
                frame = pipeline.wait_for_frames(cls.SNAPSHOT_FRAME_TIMEOUT_MS).get_color_frame()
                if not frame:
                    raise RuntimeError(f"{serial}: missing color frame")
                times.append(time.monotonic())
                numbers.append(frame.get_frame_number())

            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{serial}.jpg"
            Image.fromarray(np.asarray(frame.get_data()).copy()).save(path, quality=cls.JPEG_QUALITY)
            intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
            result = {
                "serial": serial,
                "model": profile.get_device().get_info(rs.camera_info.name),
                "image": str(path),
                "width": intrinsics.width,
                "height": intrinsics.height,
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "ppx": intrinsics.ppx,
                    "ppy": intrinsics.ppy,
                    "model": str(intrinsics.model),
                    "coeffs": list(intrinsics.coeffs),
                },
                "frames": frame_count,
                "distinct_frame_numbers": len(set(numbers)),
                "dropped_frames": sum(
                    max(next_number - previous_number - 1, 0)
                    for previous_number, next_number in zip(numbers, numbers[1:])
                ),
                "observed_fps": (frame_count - 1) / (times[-1] - times[0]) if frame_count > 1 else None,
                "received_monotonic": times[-1],
                "device_timestamp_ms": frame.get_timestamp(),
                "timestamp_domain": str(frame.get_frame_timestamp_domain()),
            }
            write_text_atomically(output_dir / f"{serial}.json", json.dumps(result, indent=cls.JSON_INDENT))

            return result
        finally:
            pipeline.stop()

    @staticmethod
    def start_color_stream(serial: str) -> tuple[Any, Any]:
        """Starts the RGB stream of the camera ``serial`` and returns its pipeline and the pipeline's profile."""
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(serial)
        stream_config.enable_stream(
            rs.stream.color,
            CameraConfig.FRAME_WIDTH,
            CameraConfig.FRAME_HEIGHT,
            rs.format.rgb8,
            CameraConfig.FRAME_RATE,
        )

        return pipeline, pipeline.start(stream_config)
