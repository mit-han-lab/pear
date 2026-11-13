import asyncio
import os
import tempfile
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

from aiohttp import web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server.history_routes import TaskRoutes
from pear.deploy.station.records.recording.encode_worker import FINISHED_STATUSES
from pear.deploy.station.records.task_files import TaskFiles


class Archive:
    """A zip file of task records."""

    # already-compressed media is stored: deflating it only costs CPU on the robot host
    STORED_SUFFIXES = (".mp4", ".jpg", ".jpeg", ".png", ".webp")

    @staticmethod
    def includes(path: Path) -> bool:
        """Whether an archive takes the file: a regular file that is no link and is not being written."""
        is_unfinished = any(marker in path.name for marker in TaskFiles.UNFINISHED_MARKERS)
        return path.is_file() and not path.is_symlink() and not is_unfinished

    @classmethod
    def pack(cls, archive_path: Path, collect: Callable[[], list[tuple[Path, Path]]]) -> None:
        """Writes the files ``collect`` returns, each with its name in the archive, into a zip at ``archive_path``."""
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path, name in collect():
                is_stored = path.suffix.lower() in cls.STORED_SUFFIXES
                archive.write(path, name, compress_type=zipfile.ZIP_STORED if is_stored else zipfile.ZIP_DEFLATED)


class MediaRoutes(TaskRoutes):
    """The recorded videos, and the archives of a task and of a segment's camera sources."""

    # the videos of a folder of joined or styled videos: full.mp4, motion.mp4
    MERGED_VIDEO_NAMES = tuple(TaskFiles.video_name(kind) for kind in TaskFiles.VIDEO_KINDS)
    CHUNK_BYTES = 1 << 20  # of an archive, read and sent at a time
    ARCHIVE_PREFIX = ".download-"  # of an archive under construction, a private temporary file in the tasks' folder
    ARCHIVE_SUFFIX = ".zip.tmp"  # of an archive under construction
    CONTENT_DISPOSITION = 'attachment; filename="{filename}"'  # of a response the browser saves as a file

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        # a crash during a download leaves at most these
        for stale_archive in console.root.glob(f"{self.ARCHIVE_PREFIX}*{self.ARCHIVE_SUFFIX}"):
            stale_archive.unlink(missing_ok=True)

    def add_routes(self, router: web.UrlDispatcher) -> None:
        """Adds the routes to ``router``."""
        router.add_get("/api/history/{task_id}/export", self.export)
        # added before the route of a segment's file, which would match it too
        router.add_get("/api/recordings/{task_id}/{segment}/source.zip", self.source)
        router.add_get("/api/recordings/{task_id}/{segment}/{name}", self.video)
        router.add_get("/api/recordings/{task_id}/{segment}/" + TaskFiles.STYLED_FOLDER + "/{name}", self.styled_video)

    async def export(self, request: web.Request) -> web.StreamResponse:
        """Answers an archive of a task's whole folder."""
        files = self.task_files(request)
        filename = f"{self.console.robot.NAME}-task-{files.directory.name}.zip"

        return await self.send_archive(request, partial(self.task_entries, files.directory), filename)

    async def source(self, request: web.Request) -> web.StreamResponse:
        """Answers an archive of one finished segment's camera sources and videos with the task's own records."""
        files = self.task_files(request)
        segment_id = request.match_info["segment"]
        segment_directory = files.segment_directory(segment_id)
        if not TaskFiles.SEGMENT_ID_PATTERN.fullmatch(segment_id):
            raise web.HTTPNotFound()
        if not segment_directory.is_dir() or segment_directory.is_symlink():
            raise web.HTTPNotFound()

        segments = await asyncio.to_thread(self.console.recordings.manifests_of_task, files.directory.name)
        segment_record = next((segment for segment in segments if segment.get("id") == segment_id), None)
        if segment_record is None:
            raise web.HTTPNotFound()
        if segment_record["status"] not in FINISHED_STATUSES:
            raise web.HTTPConflict(text="Recording is still being processed")

        filename = f"{self.console.robot.NAME}-{files.directory.name}-{segment_id}-source.zip"
        return await self.send_archive(
            request, partial(self.segment_entries, files.directory, segment_directory), filename
        )

    async def video(self, request: web.Request) -> web.FileResponse:
        """Answers one video of a segment, or of the task's joined videos."""
        files = self.task_files(request)
        segment_id, name = request.match_info["segment"], request.match_info["name"]
        if segment_id == TaskFiles.MERGED_FOLDER:
            allowed_names = self.MERGED_VIDEO_NAMES
        elif TaskFiles.SEGMENT_ID_PATTERN.fullmatch(segment_id):
            allowed_names = (*self.MERGED_VIDEO_NAMES, *self.console.recordings.source_file_names.values())
        else:
            allowed_names = ()
        if name not in allowed_names:
            raise web.HTTPNotFound()

        path = files.recordings_directory / segment_id / name
        return self.send_video(request, path, f"{files.directory.name}-{segment_id}-{name}")

    async def styled_video(self, request: web.Request) -> web.FileResponse:
        """Answers one segment's styled full or motion-only video."""
        files = self.task_files(request)
        segment_id, name = request.match_info["segment"], request.match_info["name"]
        if not TaskFiles.SEGMENT_ID_PATTERN.fullmatch(segment_id) or name not in self.MERGED_VIDEO_NAMES:
            raise web.HTTPNotFound()

        path = files.styled_directory(segment_id) / name
        return self.send_video(request, path, f"{files.directory.name}-{segment_id}-{TaskFiles.STYLED_FOLDER}-{name}")

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
        self, request: web.Request, collect: Callable[[], list[tuple[Path, Path]]], filename: str
    ) -> web.StreamResponse:
        """Packs the files ``collect`` returns into a private temporary file, then sends it with its length known."""
        # no command lock is held: a command never waits for this download's lock
        descriptor, archive_name = tempfile.mkstemp(
            prefix=self.ARCHIVE_PREFIX, suffix=self.ARCHIVE_SUFFIX, dir=self.console.root
        )
        os.close(descriptor)
        archive_path = Path(archive_name)

        try:
            await asyncio.to_thread(Archive.pack, archive_path, collect)
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

    @staticmethod
    def task_entries(directory: Path) -> list[tuple[Path, Path]]:
        """Returns every file an archive takes of a task's folder ``directory``, each with its name in the archive."""
        return [(path, path.relative_to(directory)) for path in directory.rglob("*") if Archive.includes(path)]

    @staticmethod
    def segment_entries(directory: Path, segment_directory: Path) -> list[tuple[Path, Path]]:
        """Returns the files an archive takes of a task's folder ``directory`` and of one of its segments' folders,
        each with its name in the archive."""
        paths = [path for path in directory.iterdir() if Archive.includes(path)]
        paths += [path for path in segment_directory.iterdir() if Archive.includes(path)]

        return [(path, path.relative_to(directory)) for path in sorted(paths)]
