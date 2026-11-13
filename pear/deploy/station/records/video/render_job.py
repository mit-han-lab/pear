import multiprocessing.queues
import queue
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from pear.deploy.station.control.schemas import Camera
from pear.deploy.station.records.recording.ffmpeg import FFmpeg
from pear.deploy.station.records.recording.recordings import Recordings
from pear.deploy.station.records.recording.segment import Segment
from pear.deploy.station.records.task_events import TaskEventKind
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.render_worker import RenderStatus, RenderWorker
from pear.deploy.station.records.video.styled_render import StyledRender
from pear.utils.logging_utils import get_logger
from pear.utils.worker_process import WorkerProcess

logger = get_logger(__name__)


class RenderJob:
    """One render of styled videos: it waits for the segments it needs, then renders them in a spawned worker."""

    POLL_S = 0.5  # between two looks at the unfinished segments, and at the worker's reports
    PROGRESS_TIMEOUT_S = 300  # a worker that reports no progress for this long is taken as hung
    JOIN_S = 10  # at close, for the thread to end once its worker is stopped

    def __init__(
        self,
        files: TaskFiles,
        part: str,
        cameras: tuple[Camera, ...],
        manifests_reader: Callable[[], list[dict[str, Any]]],
        single_render_lock: threading.Lock,
    ) -> None:
        self.files = files
        self.part = part  # a segment's id, or MERGED_FOLDER for the task's segments joined
        self.cameras = cameras
        self.manifests_reader = manifests_reader  # returns the manifests of the task's segments as they stand
        self.single_render_lock = single_render_lock  # held while rendering, so that one render runs at a time
        self.lock = threading.Lock()  # guards the state and the worker
        self.state: dict[str, Any] = {"status": RenderStatus.WAITING.value, "progress": None, "error": None}
        self.stop_event = threading.Event()
        self.renderer: WorkerProcess | None = None  # the render's worker, once the rendering has begun
        self.thread = threading.Thread(target=self.run, name="gui-video-merge", daemon=True)
        self.thread.start()

    def snapshot(self) -> dict[str, Any]:
        """Returns the render's status, its progress and its error."""
        with self.lock:
            return dict(self.state)

    def close(self) -> None:
        """Stops the waiting or the rendering; the published videos and the segments stay as they are."""
        self.stop_event.set()
        with self.lock:
            self._terminate()
        self.thread.join(self.JOIN_S)

    def run(self) -> None:
        """Waits, renders and publishes: the render thread's body."""
        styled_render = None

        try:
            styled_render = self._wait_for_segments()
            if not styled_render.status()["current"]:
                self._update(status=RenderStatus.QUEUED.value)
                with self.single_render_lock:
                    render_result = self._render(styled_render)
                styled_render.publish(render_result)
            self._update(status=RenderStatus.SAVED.value, progress=1.0)
        except BaseException as error:
            logger.exception("The render of %s of the task in %s failed", self.part, self.files.directory)
            self._update(status=RenderStatus.FAILED.value, error=str(error) or type(error).__name__)
        finally:
            if styled_render is not None:
                styled_render.discard()

    def _wait_for_segments(self) -> StyledRender:
        """Returns the styled render of the part once none of its segments is still recording or processing."""
        while True:
            styled_render = StyledRender(self.files, self.manifests_reader(), self.part)
            if styled_render.pending_id is None:
                styled_render.check()
                return styled_render
            if self.stop_event.wait(self.POLL_S):
                raise RuntimeError("Merge stopped by service shutdown")

    def _render(self, styled_render: StyledRender) -> dict[str, Any]:
        """Runs the worker, follows its progress and returns what it rendered."""
        styled_render.folder.mkdir(exist_ok=True)
        messages = WorkerProcess.CONTEXT.Queue()

        with self.lock:
            if self.stop_event.is_set():
                raise RuntimeError("Merge stopped by service shutdown")
            self.state.update(status=RenderStatus.RENDERING.value, progress=0.0)
            self.renderer = WorkerProcess(
                RenderWorker.main,
                (
                    str(self.files.directory),
                    styled_render.clips,
                    self.cameras,
                    str(styled_render.folder),
                    messages,
                    FFmpeg.executable(),
                ),
                "gui-video-merge",
                FFmpeg.WORKER_EXIT_TIMEOUT_S,
                is_group_leader=True,
            )
            self.renderer.start()

        try:
            return self._follow(messages)
        finally:
            with self.lock:
                self._terminate()
            messages.close()

    def _follow(self, messages: multiprocessing.queues.Queue) -> dict[str, Any]:
        """Passes the worker's progress on until the worker reports its result, which is returned."""
        deadline = time.monotonic() + self.PROGRESS_TIMEOUT_S
        while True:
            # looked at before the read: whatever a worker that ended has sent is in the queue by then
            is_running = self.renderer.is_alive()
            try:
                message = messages.get(timeout=self.POLL_S)
            except queue.Empty:
                if not is_running:
                    raise RuntimeError("Merge worker exited without a result") from None
                if time.monotonic() > deadline:
                    raise TimeoutError("Merge worker stopped reporting progress") from None
                continue

            if "progress" in message:
                self._update(progress=message["progress"])
                deadline = time.monotonic() + self.PROGRESS_TIMEOUT_S
            elif message["status"] == RenderStatus.FAILED:
                raise RuntimeError(message["error"])
            else:
                return message

    def _update(self, **fields: Any) -> None:
        """Changes fields of the state."""
        with self.lock:
            self.state.update(fields)

    def _terminate(self) -> None:
        """Ends the worker and the FFmpeg processes it started; the caller holds the lock."""
        if self.renderer is not None:
            self.renderer.stop(0)


class RenderedRecordings(Recordings):
    """The recording segments of the running service together with the renders of their styled videos: a segment's
    as soon as it is saved, and the segments of a task joined once the task ends."""

    def __init__(self, root: Path, cameras: tuple[Camera, ...]) -> None:
        super().__init__(root, cameras)
        self.task_id: str | None = None  # of the task whose events arrive
        # (task id, segment id or MERGED_FOLDER) -> the latest render of those videos
        self.render_jobs: dict[tuple[str, str], RenderJob] = {}
        self.render_lock = threading.Lock()  # guards the render jobs and whether the service is closed
        self.single_render_lock = threading.Lock()  # held by the one render that runs at a time
        self.is_closed = False  # no render starts once the service shuts down

    def stop(self, reason: str, *, stopped_at: float | None = None) -> Segment | None:
        """Stops the segment that is recording and starts the render of its styled videos, which waits until the
        segment is saved."""
        segment = super().stop(reason, stopped_at=stopped_at)
        if segment is not None:
            with self.render_lock:
                self._start_render(segment.manifest["task_id"], segment.manifest["id"])

        return segment

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        """Follows one event of the task; at the end of a task with several segments their joined render starts."""
        if kind == TaskEventKind.TASK:
            self.task_id = data["id"]
        elif kind == TaskEventKind.TASK_END and self.task_id is not None:
            with self.render_lock:
                has_several_segments = len(self.manifests_of_task(self.task_id)) > 1
                if has_several_segments and not self._is_rendering(self.task_id, TaskFiles.MERGED_FOLDER):
                    self._start_render(self.task_id, TaskFiles.MERGED_FOLDER)
            self.task_id = None

        super().on_event(kind, data)

    def segments(self, task_id: str) -> list[dict[str, Any]]:
        """Returns the manifests of a task's segments, each with its styled videos once they are rendered and with
        a render of them that is under way."""
        manifests = self.manifests_of_task(task_id)
        statuses = [self._status(task_id, manifest["id"], manifests) for manifest in manifests]
        for manifest, status in zip(manifests, statuses):
            manifest.update(styled=status["merged"] if status["current"] else None, render=status["render"])

        return manifests

    def merged_status(self, task_id: str) -> dict[str, Any]:
        """Returns the joined videos of a task, if there are any, whether they still cover its finished segments,
        and a render of them that is under way."""
        return self._status(task_id, TaskFiles.MERGED_FOLDER, self.manifests_of_task(task_id))

    def request_render(self, task_id: str, part: str) -> dict[str, Any]:
        """Starts the render of the styled videos of one finished segment, or of every segment when ``part`` is
        MERGED_FOLDER, unless they are up to date or already rendering, and returns their status."""
        with self.render_lock:
            if not self._is_rendering(task_id, part):
                styled_render = self._styled_render(task_id, part, self.manifests_of_task(task_id))
                styled_render.check()
                if not styled_render.status()["current"]:
                    self._start_render(task_id, part)

        return self._status(task_id, part, self.manifests_of_task(task_id))

    def close(self) -> None:
        """Stops the recording, the renders and then the segments that still finish their files."""
        with self.render_lock:
            self.is_closed = True
        self.stop("Service shutdown")

        # the renders stop first: one that waits for a segment must not begin to render during the shutdown
        for render_job in self.render_jobs.values():
            render_job.close()
        super().close()

    def _status(self, task_id: str, part: str, manifests: list[dict[str, Any]]) -> dict[str, Any]:
        """Returns the rendered videos of ``part``, if there are any, whether they cover its finished segments, and
        a render of them that is under way."""
        render_job = self.render_jobs.get((task_id, part))
        render_state = render_job.snapshot() if render_job is not None else None
        is_saved = render_state is not None and render_state["status"] == RenderStatus.SAVED

        return {
            **self._styled_render(task_id, part, manifests).status(),
            "render": None if is_saved else render_state,
        }

    def _styled_render(self, task_id: str, part: str, manifests: list[dict[str, Any]]) -> StyledRender:
        """Returns the styled render of ``part`` of a task whose segments have ``manifests``."""
        return StyledRender(TaskFiles(self.root / task_id), manifests, part)

    def _is_rendering(self, task_id: str, part: str) -> bool:
        """Whether a render of ``part`` waits or runs."""
        render_job = self.render_jobs.get((task_id, part))

        return render_job is not None and render_job.thread.is_alive()

    def _start_render(self, task_id: str, part: str) -> None:
        """Starts the render of ``part``, which begins once its segments are saved; the caller holds the render
        lock."""
        if self.is_closed:
            return

        manifests_reader = partial(self.manifests_of_task, task_id)
        self.render_jobs[task_id, part] = RenderJob(
            TaskFiles(self.root / task_id), part, self.cameras, manifests_reader, self.single_render_lock
        )
