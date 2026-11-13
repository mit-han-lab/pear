import copy
import math
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pear.deploy.station.agent_loop.clients.schemas import ModelEvent
from pear.deploy.station.agent_loop.schemas import TurnEvent, TurnPhase
from pear.deploy.station.control.cameras import CameraSource
from pear.deploy.station.control.schemas import MOVING_CONTROL_STATES, ActionPhase, Camera, ControlState
from pear.deploy.station.records.recording.captions import CaptionPhase
from pear.deploy.station.records.recording.encode_worker import FINISHED_STATUSES, EncodeWorker, SegmentStatus
from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE, VIDEO_SIZE, FFmpeg
from pear.deploy.station.records.recording.frame_renderer import UNKNOWN_MODEL_LABEL, CaptionedLayout
from pear.deploy.station.records.task_events import TaskEventKind
from pear.deploy.station.records.task_files import TaskFiles
from pear.utils.logging_utils import get_logger
from pear.utils.worker_process import WorkerProcess

logger = get_logger(__name__)


class Segment:
    """One recording segment of a task: it samples the cameras and hands the frames to its encode worker."""

    MAX_QUEUED_PACKETS = 3  # samples waiting for the worker; one more is dropped
    START_TIMEOUT_S = 10  # for the worker to report that its encoders are open
    MAX_PENDING_UPDATES = 300  # changes of state the worker has not been given yet
    SETTLING_MOTION_S = 0.2  # of settling that still counts as motion
    END_TIMEOUT_S = 10  # for the end packet to enter the queue
    FINALIZE_MIN_S = 120  # the shortest wait for the worker to finish the files
    FINALIZE_FACTOR = 5  # of the recorded duration, for the worker to finish the files
    POLL_S = 0.1  # between two looks at the finishing worker

    def __init__(
        self,
        directory: Path,
        task: dict[str, Any],
        cameras: tuple[Camera, ...],
        camera_feed: CameraSource,
        robot_status_reader: Callable[[], dict[str, Any]],
        latest_context: dict[str, Any] | None,
    ) -> None:
        self.directory = directory  # the segment's folder, named by its id
        self.camera_feed = camera_feed
        self.robot_status_reader = robot_status_reader  # returns the robot's status with its control snapshot
        self.lock = threading.Lock()  # guards the caption state, the manifest and the times
        # what the frames sampled from now on are captioned with
        self.caption_state: dict[str, Any] = {
            "phase": CaptionPhase.STARTING.value,
            "turn_number": None,
            "action": None,
            "context_usage": latest_context,
            "inference_usage": None,
        }
        self.ended_at: float | None = None  # when the segment was stopped, on the monotonic clock
        self.stop_event = threading.Event()
        self.started_at: float | None = None  # when its first frame was captured, on the monotonic clock
        self.settle_started_at: float | None = None  # since when the arms have been settling

        self.manifest_path = directory / TaskFiles.MANIFEST_NAME
        self.manifest: dict[str, Any] = {
            "id": directory.name,
            "task_id": task["id"],
            "model": task.get("model"),
            "model_label": task.get("model_label") or task.get("model") or UNKNOWN_MODEL_LABEL,
            "status": SegmentStatus.STARTING.value,
            "created": time.time(),
            "fps": FRAME_RATE,
            "width": VIDEO_SIZE[0],
            "height": VIDEO_SIZE[1],
            "full": None,
            "motion": None,
            "sources": None,
            "source_error": None,
            "camera_tiles": CaptionedLayout.for_cameras(cameras).tiles,
            "dropped_samples": 0,
            "error": None,
        }

        executable = FFmpeg.executable()
        directory.mkdir(parents=True)
        TaskFiles.write_json(self.manifest_path, self.manifest)

        self.packets = WorkerProcess.CONTEXT.Queue(maxsize=self.MAX_QUEUED_PACKETS)
        self.messages = WorkerProcess.CONTEXT.Queue()
        self.encoder = WorkerProcess(
            EncodeWorker.main,
            (str(directory), task, cameras, self.packets, self.messages, executable),
            "gui-video-encoder",
            FFmpeg.WORKER_EXIT_TIMEOUT_S,
            is_group_leader=True,
        )
        self.encoder.start()

        self._wait_for_worker()
        self._send_first_frame()

        self.thread = threading.Thread(target=self.run, name="gui-video-sampler", daemon=True)
        self.thread.start()

    def on_event(
        self,
        kind: str,
        data: dict[str, Any],
        *,
        latest_context: dict[str, Any] | None,
        turn_usage: dict[str, Any] | None,
    ) -> None:
        """Takes one event of the task into what the frames are captioned with."""
        with self.lock:
            if self.ended_at is not None:
                return

            if kind == TaskEventKind.TURN_STARTED:
                self.caption_state = {
                    "phase": TurnPhase.OBSERVING.value,
                    "turn_number": data["turn_number"],
                    "attempt": data["attempt"],
                    "action": None,
                }
            elif kind == ModelEvent.PHASE:
                self.caption_state = {**self.caption_state, "phase": data["phase"]}
            elif kind == TurnEvent.ACTION:
                action = {key: value for key, value in data.items() if key != "attempt"}
                self.caption_state = {**self.caption_state, "action": copy.deepcopy(action)}
            elif kind in (TaskEventKind.EXECUTION, TaskEventKind.TURN_END):
                phase = data.get("phase", data.get("status", CaptionPhase.COMPLETED.value))
                self.caption_state = {**self.caption_state, "phase": phase}

            self.caption_state = {
                **self.caption_state,
                "context_usage": latest_context,
                "inference_usage": turn_usage,
            }

    def snapshot(self) -> dict[str, Any]:
        """Returns the manifest as it stands, with the duration so far while the segment is not finished."""
        with self.lock:
            manifest = dict(self.manifest)
            if self.started_at is not None and manifest["status"] in (
                SegmentStatus.RECORDING,
                SegmentStatus.PROCESSING,
            ):
                manifest.setdefault("duration_s", (self.ended_at or time.monotonic()) - self.started_at)

            return manifest

    def stop(self, reason: str, *, stopped_at: float | None) -> None:
        """Ends the sampling at ``stopped_at``, or now; the worker finishes the files afterwards."""
        # nothing here waits for a process, writes to disk or puts into a queue: a stop is on the caller's way
        with self.lock:
            if self.ended_at is None:
                self.ended_at = max(self.started_at, stopped_at or time.monotonic())
                if self.manifest["status"] not in FINISHED_STATUSES:
                    self.manifest.update(
                        status=SegmentStatus.PROCESSING.value, reason=reason, ended_monotonic_s=self.ended_at
                    )
                self.stop_event.set()

    def run(self) -> None:
        """Samples until the segment is stopped, then waits for the worker to finish the files: the sampler
        thread's body."""
        try:
            updates = self._sample()
            if updates is None:
                return

            self._finalize(updates)
        except BaseException as error:
            logger.exception("The recording segment in %s failed", self.directory)
            with self.lock:
                self.manifest.update(status=SegmentStatus.FAILED.value, error=f"{type(error).__name__}: {error}")
        finally:
            self.terminate()
            try:
                TaskFiles.write_json(self.manifest_path, self.snapshot())
            finally:
                self._dispose()

    def terminate(self) -> None:
        """Ends the worker and the FFmpeg processes it started."""
        self.encoder.stop(0)

    def _wait_for_worker(self) -> None:
        """Waits for the worker's report that its encoders are open."""
        try:
            ready_message = self.messages.get(timeout=self.START_TIMEOUT_S)
            if not ready_message.get("ready"):
                raise RuntimeError(ready_message.get("error", "Video encoder did not start"))
            self.manifest["code_version"] = ready_message.get("code_version")
        except BaseException:
            self._abort("Video encoder could not start")
            raise

    def _send_first_frame(self) -> None:
        """Starts the segment's clock and gives the worker a first frame, so that a segment stopped at once still
        has a full video of one frame."""
        # one instant on both clocks: the task's events and the segment's timeline can be aligned
        self.started_at, started_wall_s = time.monotonic(), time.time()
        self.manifest.update(
            status=SegmentStatus.RECORDING.value, started_monotonic_s=self.started_at, started_wall_s=started_wall_s
        )

        try:
            frames, errors = self.camera_feed.read_preview()
            state = {**self.caption_state, "moving": False, "control_state": ControlState.HOLD.value}
            self.packets.put_nowait(
                {"index": 0, "captured": self.started_at, "frames": frames, "errors": errors, "state": state}
            )
        except BaseException:
            self._abort("Could not capture the first recording frame")
            raise

    def _sample(self) -> list[dict[str, Any]] | None:
        """Feeds camera packets to the worker until the segment is stopped, and returns the changes of state it
        was not given yet, None when the recording failed."""
        index = 1
        updates: list[dict[str, Any]] = []
        last_state = None

        while not self.stop_event.is_set():
            self._drain_messages()
            if self.snapshot()["status"] == SegmentStatus.FAILED:
                return None
            if not self.encoder.is_alive():
                raise RuntimeError("Video encoder exited unexpectedly")

            frames, errors = self.camera_feed.read_preview()
            with self.lock:
                if self.ended_at is not None:
                    break
                state = dict(self.caption_state)
                robot_status = self.robot_status_reader()
                now = time.monotonic()

            index = max(index, math.floor((now - self.started_at) * FRAME_RATE))
            self._describe_motion(state, robot_status, now)
            if state != last_state:
                updates.append({"index": index, "state": state})
                last_state = state
            if len(updates) > self.MAX_PENDING_UPDATES:
                raise RuntimeError("Video encoder cannot keep up with task events")

            packet = {
                "index": index,
                "captured": now,
                "frames": frames,
                "errors": errors,
                "state": state,
                "updates": updates,
            }
            try:
                self.packets.put_nowait(packet)
                updates = []
            except queue.Full:
                with self.lock:
                    self.manifest["dropped_samples"] += 1

            index = max(index + 1, math.floor((time.monotonic() - self.started_at) * FRAME_RATE) + 1)
            self.stop_event.wait(max(0, self.started_at + index / FRAME_RATE - time.monotonic()))

        return updates

    def _describe_motion(self, state: dict[str, Any], robot_status: dict[str, Any], now: float) -> None:
        """Adds to a sampled state whether the robot moves, its control state and the phase of a settling arm."""
        is_settling = robot_status.get("motion", {}).get("settling", False)
        if not is_settling:
            self.settle_started_at = None
        elif self.settle_started_at is None:
            self.settle_started_at = now

        # the commands, the grippers' too, and at most SETTLING_MOTION_S of settling count as motion
        state["moving"] = robot_status.get("control_state") in MOVING_CONTROL_STATES and (
            not is_settling or now - self.settle_started_at <= self.SETTLING_MOTION_S
        )
        state["control_state"] = robot_status.get("control_state", ControlState.CLOSED.value)  # a closed robot has none
        if is_settling and state["phase"] == ActionPhase.EXECUTING:
            state["phase"] = ActionPhase.SETTLING.value

    def _finalize(self, updates: list[dict[str, Any]]) -> None:
        """Gives the worker the end of the recording and waits until it has finished the files."""
        self.packets.put({"end": self.ended_at - self.started_at, "updates": updates}, timeout=self.END_TIMEOUT_S)

        deadline = time.monotonic() + max(self.FINALIZE_MIN_S, (self.ended_at - self.started_at) * self.FINALIZE_FACTOR)
        while self.encoder.is_alive():
            self._drain_messages()
            if time.monotonic() > deadline:
                raise TimeoutError("Video finalization timed out")
            self.encoder.process.join(self.POLL_S)

        self._drain_messages()
        if self.snapshot()["status"] not in FINISHED_STATUSES:
            raise RuntimeError("Video encoder exited without a completed recording")

    def _drain_messages(self) -> None:
        """Takes every message of the worker into the manifest, and writes the manifest after each."""
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break

            with self.lock:
                self.manifest.update(message)
            TaskFiles.write_json(self.manifest_path, self.snapshot())

    def _abort(self, error: str) -> None:
        """Gives up a segment that could not start: the worker is ended and the manifest says why."""
        self.terminate()
        self.manifest.update(status=SegmentStatus.FAILED.value, error=error)
        TaskFiles.write_json(self.manifest_path, self.manifest)
        self._dispose()

    def _dispose(self) -> None:
        """Closes the two queues."""
        # a worker that crashed must not leave the feeder thread of the packets waiting for good
        self.packets.cancel_join_thread()
        self.packets.close()
        self.messages.close()
