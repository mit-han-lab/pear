import multiprocessing.queues
import os
import queue
import threading
import time
from collections import Counter
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
from pear.deploy.station.records.video.render_plan import RenderChunk, RenderPlan
from pear.deploy.station.records.video.render_worker import RenderStatus, RenderWorker
from pear.deploy.station.records.video.styled_render import StyledRender
from pear.deploy.station.records.video.task_timeline import TaskTimeline
from pear.utils.logging_utils import get_logger
from pear.utils.worker_process import WorkerProcess

logger = get_logger(__name__)


class RenderJob:
    """One render of styled videos: it waits for the segments it needs, then has spawned workers draw the videos in
    chunks side by side, as many as the host's cores keep busy, and publishes each video once its chunks are joined,
    the motion-only one first."""

    POLL_S = 0.5  # between two looks at the unfinished segments, and at the workers' reports
    PROGRESS_TIMEOUT_S = 300  # workers that report no progress for this long are taken as hung
    JOIN_S = 10  # at close, for the thread to end once its workers are stopped
    # cores one chunk keeps busy: its drawing, the encoder's threads and the decoder. A render runs as many chunks at
    # once as the host's cores give room for, and one while a task records
    CORES_PER_CHUNK = 3

    def __init__(
        self,
        files: TaskFiles,
        part: str,
        cameras: tuple[Camera, ...],
        manifests_reader: Callable[[], list[dict[str, Any]]],
        single_render_lock: threading.Lock,
        recording_reader: Callable[[], bool],
    ) -> None:
        self.files = files
        self.part = part  # a segment's id, or MERGED_FOLDER for the task's segments joined
        self.cameras = cameras
        self.manifests_reader = manifests_reader  # returns the manifests of the task's segments as they stand
        self.single_render_lock = single_render_lock  # held while rendering, so that one render runs at a time
        self.recording_reader = recording_reader  # returns whether a segment records
        self.lock = threading.Lock()  # guards the state and the workers
        # the video whose chunks come next is the state's kind, and its progress the state's
        self.state: dict[str, Any] = {
            "status": RenderStatus.WAITING.value,
            "kind": None,
            "progress": None,
            "error": None,
        }
        self.stop_event = threading.Event()
        self.workers: list[tuple[WorkerProcess, multiprocessing.queues.Queue]] = []  # each with the queue of its chunks
        self.thread = threading.Thread(target=self.run, name="gui-video-merge", daemon=True)
        self.thread.start()

    def snapshot(self) -> dict[str, Any]:
        """Returns the render's status, the video it renders, its progress and its error."""
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
            if not styled_render.status()["complete"]:
                self._update(status=RenderStatus.QUEUED.value)
                with self.single_render_lock:
                    self._render(styled_render)
            self._update(status=RenderStatus.SAVED.value, kind=None, progress=1.0)
        except BaseException as error:
            logger.exception("The render of %s of the task in %s failed", self.part, self.files.directory)
            self._update(status=RenderStatus.FAILED.value, error=str(error) or type(error).__name__)
        finally:
            if styled_render is not None:
                styled_render.discard()

    @classmethod
    def worker_count(cls) -> int:
        """Returns how many chunks a render draws at once while no task records: as many as the cores this process
        may run on give room for."""
        return max(1, len(os.sched_getaffinity(0)) // cls.CORES_PER_CHUNK)

    def _wait_for_segments(self) -> StyledRender:
        """Returns the styled render of the part once none of its segments is still recording or processing."""
        while True:
            styled_render = StyledRender(self.files, self.manifests_reader(), self.part)
            if styled_render.pending_id is None:
                styled_render.check()
                return styled_render
            if self.stop_event.wait(self.POLL_S):
                raise RuntimeError("Merge stopped by service shutdown")

    def _render(self, styled_render: StyledRender) -> None:
        """Draws the part's videos in chunks, and publishes each video once its chunks are joined."""
        styled_render.folder.mkdir(exist_ok=True)
        (styled_render.folder / TaskFiles.ENCODER_LOG_NAME).write_text("")
        motion_spans = TaskTimeline.load(self.files, styled_render.clips).motion_spans
        plan = RenderPlan(styled_render.kinds(), styled_render.clips, motion_spans)
        messages = WorkerProcess.CONTEXT.Queue()

        try:
            self._follow(styled_render, plan, messages)
        finally:
            with self.lock:
                self._terminate()
            messages.close()

    def _follow(self, styled_render: StyledRender, plan: RenderPlan, messages: multiprocessing.queues.Queue) -> None:
        """Hands the chunks out in order to as many workers as may run, follows the workers' reports, and joins and
        publishes each video once all its chunks are done."""
        executable = FFmpeg.executable()
        chunk_queue = list(plan.chunks)  # the chunks not handed out yet, in order
        chunk_by_worker: dict[int, RenderChunk] = {}  # worker -> the chunk it renders
        chunks_to_finish = Counter(chunk.kind for chunk in plan.chunks)  # by video, its chunks not done yet
        done_frame_counts = dict.fromkeys(plan.kinds, 0)
        videos_to_publish = list(plan.kinds)
        render_results: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + self.PROGRESS_TIMEOUT_S
        self._update(status=RenderStatus.RENDERING.value, kind=videos_to_publish[0], progress=0.0)

        while videos_to_publish:
            if self.stop_event.is_set():
                raise RuntimeError("Merge stopped by service shutdown")
            self._hand_out(styled_render, chunk_queue, chunk_by_worker, messages, executable)
            # looked at before the read: whatever a worker that ended has sent is in the queue by then
            has_lost_worker = any(not self.workers[worker][0].is_alive() for worker in chunk_by_worker)
            try:
                message = messages.get(timeout=self.POLL_S)
            except queue.Empty:
                if has_lost_worker:
                    raise RuntimeError("A render worker exited without its chunk") from None
                if time.monotonic() > deadline:
                    raise TimeoutError("Render workers stopped reporting progress") from None
                continue

            deadline = time.monotonic() + self.PROGRESS_TIMEOUT_S
            if "error" in message:
                raise RuntimeError(message["error"])
            if "frames" in message:
                done_frame_counts[message["kind"]] += message["frames"]
            if "done" in message:
                worker = next(
                    n
                    for n, chunk in chunk_by_worker.items()
                    if (chunk.kind, chunk.index) == (message["kind"], message["done"])
                )
                del chunk_by_worker[worker]
                chunks_to_finish[message["kind"]] -= 1

            # the videos are published in the plan's order, each once all its chunks are done
            while videos_to_publish and not chunks_to_finish[videos_to_publish[0]]:
                kind = videos_to_publish.pop(0)
                self._join(styled_render, plan, kind, executable)
                render_results[kind] = {"frames": plan.frame_count(kind), "clip_frames": plan.clip_frames[kind]}
                styled_render.publish(render_results)
            if videos_to_publish:
                kind = videos_to_publish[0]
                self._update(kind=kind, progress=done_frame_counts[kind] / plan.frame_count(kind))

    def _hand_out(
        self,
        styled_render: StyledRender,
        chunk_queue: list[RenderChunk],
        chunk_by_worker: dict[int, RenderChunk],
        messages: multiprocessing.queues.Queue,
        executable: str,
    ) -> None:
        """Gives the next chunks to idle workers while fewer chunks render than may, and starts a worker when none is
        idle."""
        chunk_limit = 1 if self.recording_reader() else self.worker_count()
        while chunk_queue and len(chunk_by_worker) < chunk_limit:
            worker = next((n for n in range(len(self.workers)) if n not in chunk_by_worker), None)
            if worker is None:
                worker = self._start_worker(styled_render, messages, executable)
            chunk = chunk_queue.pop(0)
            self.workers[worker][1].put(chunk)
            chunk_by_worker[worker] = chunk

    def _start_worker(
        self, styled_render: StyledRender, messages: multiprocessing.queues.Queue, executable: str
    ) -> int:
        """Starts one more worker and returns its number."""
        chunks = WorkerProcess.CONTEXT.Queue()
        with self.lock:
            if self.stop_event.is_set():
                raise RuntimeError("Merge stopped by service shutdown")
            worker = WorkerProcess(
                RenderWorker.main,
                (
                    str(self.files.directory),
                    styled_render.clips,
                    self.cameras,
                    str(styled_render.folder),
                    chunks,
                    messages,
                    executable,
                ),
                "gui-video-render",
                FFmpeg.WORKER_EXIT_TIMEOUT_S,
                is_group_leader=True,
            )
            worker.start()
            self.workers.append((worker, chunks))

            return len(self.workers) - 1

    def _join(self, styled_render: StyledRender, plan: RenderPlan, kind: str, executable: str) -> None:
        """Joins the chunks of the video of ``kind`` into its unpublished file, and removes them."""
        folder = styled_render.folder
        paths = [folder / TaskFiles.chunk_video_name(kind, chunk.index) for chunk in plan.chunks if chunk.kind == kind]
        with (folder / TaskFiles.ENCODER_LOG_NAME).open("a") as log:
            FFmpeg.join(executable, paths, folder / TaskFiles.CHUNK_LIST_NAME, styled_render.partial_path(kind), log)
        for path in paths:
            path.unlink()

    def _update(self, **fields: Any) -> None:
        """Changes fields of the state."""
        with self.lock:
            self.state.update(fields)

    def _terminate(self) -> None:
        """Ends the workers and the FFmpeg processes they started; the caller holds the lock."""
        for worker, chunks in self.workers:
            worker.stop(0)
            # a chunk put for a worker that ended must not keep its queue's feeder waiting
            chunks.cancel_join_thread()
            chunks.close()
        self.workers = []


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
                if not styled_render.status()["complete"]:
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
            TaskFiles(self.root / task_id),
            part,
            self.cameras,
            manifests_reader,
            self.single_render_lock,
            self.is_recording,
        )
