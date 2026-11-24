import asyncio
import json
import os
import tempfile
import time
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

from aiohttp import web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.task_run import RECORD_URL
from pear.deploy.station.gui.server.history_routes import TaskRoutes
from pear.deploy.station.gui.server.trace_page import TracePage
from pear.deploy.station.records.task_files import TaskFiles


class Archive:
    """A zip file of a task's folder and its trace page, all in one folder."""

    # already-compressed media is stored: deflating it only costs CPU on the robot host
    STORED_SUFFIXES = (".mp4", ".jpg", ".jpeg", ".png", ".webp")
    TRACE_NAME = "trace.html"  # the trace page's name in the archive

    @classmethod
    def pack(cls, directory: Path, folder_name: str, trace_page: TracePage, archive_path: Path) -> None:
        """Writes the files of the task's folder ``directory`` and its trace page into a zip at ``archive_path``, all
        in the archive's folder ``folder_name``."""
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in sorted(directory.rglob("*")):
                if cls.includes(path):
                    is_stored = path.suffix.lower() in cls.STORED_SUFFIXES
                    name = f"{folder_name}/{path.relative_to(directory)}"
                    archive.write(path, name, compress_type=zipfile.ZIP_STORED if is_stored else zipfile.ZIP_DEFLATED)
            archive.writestr(f"{folder_name}/{cls.TRACE_NAME}", trace_page.html(), compress_type=zipfile.ZIP_DEFLATED)

    @staticmethod
    def includes(path: Path) -> bool:
        """Whether an archive takes the file: a regular file with content, no link and not being written."""
        is_unfinished = any(marker in path.name for marker in TaskFiles.UNFINISHED_MARKERS)
        return path.is_file() and not path.is_symlink() and not is_unfinished and path.stat().st_size > 0


class MediaRoutes(TaskRoutes):
    """The styled videos, and the downloads of a task: its archive and its trace page."""

    # the videos of a folder of joined or styled videos: full.mp4, motion.mp4
    MERGED_VIDEO_NAMES = tuple(TaskFiles.video_name(kind) for kind in TaskFiles.VIDEO_KINDS)
    CHUNK_BYTES = 1 << 20  # of an archive, read and sent at a time
    ARCHIVE_PREFIX = ".download-"  # of an archive under construction, a private temporary file in the tasks' folder
    ARCHIVE_SUFFIX = ".zip.tmp"  # of an archive under construction
    CONTENT_DISPOSITION = 'attachment; filename="{filename}"'  # of a response the browser saves as a file
    # of a task's downloads, its archive, the archive's folder and its trace page
    DOWNLOAD_NAME = "{robot}-{start_time}-{short_id}"
    START_TIME_FORMAT = "%Y-%m-%d-%H%M"  # of the task's start in that name, on the station's clock
    SHORT_ID_CHARS = 8  # of the task's id in that name

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        # a crash during a download leaves at most these
        for stale_archive in console.root.glob(f"{self.ARCHIVE_PREFIX}*{self.ARCHIVE_SUFFIX}"):
            stale_archive.unlink(missing_ok=True)

    def add_routes(self, router: web.UrlDispatcher) -> None:
        """Adds the routes to ``router``."""
        router.add_get("/api/history/{task_id}/export", self.export)
        router.add_get("/api/history/{task_id}/trace", self.trace)
        router.add_get("/api/recordings/{task_id}/" + TaskFiles.MERGED_FOLDER + "/{name}", self.merged_video)
        router.add_get("/api/recordings/{task_id}/{segment}/" + TaskFiles.STYLED_FOLDER + "/{name}", self.styled_video)

    async def export(self, request: web.Request) -> web.StreamResponse:
        """Answers an archive of a task's whole folder with its trace page."""
        files = self.task_files(request)
        folder_name = self.download_name(files)
        pack = partial(Archive.pack, files.directory, folder_name, self.trace_page(files))

        return await self.send_archive(request, pack, f"{folder_name}.zip")

    async def trace(self, request: web.Request) -> web.Response:
        """Answers a task's trace page, which the browser saves as a file."""
        files = self.task_files(request)
        trace_html = await asyncio.to_thread(self.trace_page(files).html)
        filename = f"{self.download_name(files)}-{Archive.TRACE_NAME}"

        return web.Response(
            text=trace_html,
            content_type="text/html",
            headers={"Content-Disposition": self.CONTENT_DISPOSITION.format(filename=filename)},
        )

    async def merged_video(self, request: web.Request) -> web.FileResponse:
        """Answers one of the task's styled videos of all its segments joined."""
        files = self.task_files(request)
        name = request.match_info["name"]
        if name not in self.MERGED_VIDEO_NAMES:
            raise web.HTTPNotFound()

        path = files.merged_directory / name
        return self.send_video(request, path, f"{files.directory.name}-{TaskFiles.MERGED_FOLDER}-{name}")

    async def styled_video(self, request: web.Request) -> web.FileResponse:
        """Answers one segment's styled full or motion-only video."""
        files = self.task_files(request)
        segment_id, name = request.match_info["segment"], request.match_info["name"]
        if not TaskFiles.SEGMENT_ID_PATTERN.fullmatch(segment_id) or name not in self.MERGED_VIDEO_NAMES:
            raise web.HTTPNotFound()

        path = files.styled_directory(segment_id) / name
        return self.send_video(request, path, f"{files.directory.name}-{segment_id}-{TaskFiles.STYLED_FOLDER}-{name}")

    def download_name(self, files: TaskFiles) -> str:
        """Returns the name of a task's downloads: the robot's, the task's start and the start of its id."""
        start_time = time.localtime(json.loads(files.task_path.read_text())["created"])
        return self.DOWNLOAD_NAME.format(
            robot=self.console.robot.NAME,
            start_time=time.strftime(self.START_TIME_FORMAT, start_time),
            short_id=files.directory.name[: self.SHORT_ID_CHARS],
        )

    def trace_page(self, files: TaskFiles) -> TracePage:
        """Returns the trace page of the task in ``files``."""
        record_prefix = RECORD_URL.format(task_id=files.directory.name, name="")
        return TracePage(files, self.console.robot.TITLE, record_prefix)

    def send_video(self, request: web.Request, path: Path, filename: str) -> web.FileResponse:
        """Returns the video at ``path``, which the browser saves as the robot's ``filename`` when the request asks."""
        if not path.is_file() or path.is_symlink():
            raise web.HTTPNotFound()

        headers = {}
        if request.query.get("download") == "1":
            headers["Content-Disposition"] = self.CONTENT_DISPOSITION.format(
                filename=f"{self.console.robot.NAME}-{filename}"
            )
        return web.FileResponse(path, headers=headers)

    async def send_archive(
        self, request: web.Request, pack: Callable[[Path], None], filename: str
    ) -> web.StreamResponse:
        """Packs an archive with ``pack`` into a private temporary file, then sends it with its length known."""
        # no command lock is held: a command never waits for this download's lock
        descriptor, archive_name = tempfile.mkstemp(
            prefix=self.ARCHIVE_PREFIX, suffix=self.ARCHIVE_SUFFIX, dir=self.console.root
        )
        os.close(descriptor)
        archive_path = Path(archive_name)

        try:
            await asyncio.to_thread(pack, archive_path)
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/zip",
                    "Content-Disposition": self.CONTENT_DISPOSITION.format(filename=filename),
                }
            )
            response.content_length = archive_path.stat().st_size
            await response.prepare(request)
            with archive_path.open("rb") as archive_file:
                while chunk := await asyncio.to_thread(archive_file.read, self.CHUNK_BYTES):
                    await response.write(chunk)
            await response.write_eof()

            return response
        finally:
            archive_path.unlink(missing_ok=True)
