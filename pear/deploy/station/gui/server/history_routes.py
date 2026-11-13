import asyncio
import json
import re
from typing import Any

from aiohttp import web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.console.task_run import RECORD_URL
from pear.deploy.station.records.annotations import Annotations
from pear.deploy.station.records.task_events import TaskEvents
from pear.deploy.station.records.task_files import TaskFiles


class TaskRoutes:
    """The base of the routes about past tasks: the console, and the folder of the task a route names."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def task_files(self, request: web.Request) -> TaskFiles:
        """Returns the folder of the task the route names; an id of no task is not found."""
        files = TaskFiles.find(self.console.root, request.match_info["task_id"])
        if files is None:
            raise web.HTTPNotFound()

        return files


class HistoryRoutes(TaskRoutes):
    """The list of the tasks, their tags, and the events, recordings and records of a past task."""

    RECORD_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_.-]+")  # the name of a file of a task's folder that may be served
    EVENT_PAGE_SIZE = 100  # events one request reads, unless it asks for all

    def add_routes(self, router: web.UrlDispatcher) -> None:
        """Adds the routes to ``router``."""
        router.add_get("/api/history", self.history)
        router.add_get("/api/tags", self.tags)
        router.add_get("/api/history/{task_id}/annotations", self.task_annotations)
        router.add_post("/api/history/{task_id}/annotations", self.update_annotations)
        router.add_get("/api/history/{task_id}/events", self.events)
        router.add_get("/api/history/{task_id}/recordings", self.recordings)
        router.add_get("/api/history/{task_id}/recordings/merged", self.merged_recordings)
        router.add_post("/api/history/{task_id}/recordings/merged", self.render_merged_recordings)
        router.add_post("/api/history/{task_id}/recordings/{segment}/styled", self.styled_recording)
        router.add_get(RECORD_URL, self.record)

    async def history(self, request: web.Request) -> web.Response:
        """Answers every task's record with its tags, the latest first."""
        return web.json_response(await asyncio.to_thread(self.all_tasks))

    async def tags(self, request: web.Request) -> web.Response:
        """Answers how many tasks carry each tag."""
        tasks = await asyncio.to_thread(self.all_tasks)
        return web.json_response(Annotations.tag_counts(tasks))

    async def task_annotations(self, request: web.Request) -> web.Response:
        """Answers a task's tags."""
        files = self.task_files(request)
        return web.json_response(await asyncio.to_thread(Annotations.read, files))

    async def update_annotations(self, request: web.Request) -> web.Response:
        """Replaces a task's tags, and answers them as they are kept."""
        files = self.task_files(request)
        body = await request.json()
        if not isinstance(body, dict) or body.keys() != {"tags"}:
            raise ValueError("Send tags")

        # tags are records, not robot commands: any authenticated tab may edit them
        annotations = await asyncio.to_thread(self.console.annotations.update, files, tags=body["tags"])
        return web.json_response(annotations)

    async def events(self, request: web.Request) -> web.Response:
        """Answers a page of a task's events after the event the request names, or all of them."""
        files = self.task_files(request)
        after_id = int(request.query.get("after", "0"))
        limit = None if request.query.get("all") == "1" else self.EVENT_PAGE_SIZE

        return web.json_response(await asyncio.to_thread(self.read_events, files, after_id, limit))

    async def recordings(self, request: web.Request) -> web.Response:
        """Answers the manifests of a task's recording segments."""
        files = self.task_files(request)
        segments = await asyncio.to_thread(self.console.recordings.segments, files.directory.name)

        return web.json_response(segments)

    async def merged_recordings(self, request: web.Request) -> web.Response:
        """Answers the state of a task's joined videos."""
        files = self.task_files(request)
        return web.json_response(await asyncio.to_thread(self.console.recordings.merged_status, files.directory.name))

    async def render_merged_recordings(self, request: web.Request) -> web.Response:
        """Starts the render of a task's joined videos, and answers their state."""
        # joining the files is a records operation, not a robot command: any authenticated tab
        files = self.task_files(request)
        merged_status = await asyncio.to_thread(
            self.console.recordings.request_render, files.directory.name, TaskFiles.MERGED_FOLDER
        )

        return web.json_response(merged_status)

    async def styled_recording(self, request: web.Request) -> web.Response:
        """Starts the render of one segment's styled videos again, and answers their state."""
        files = self.task_files(request)
        segment_id = request.match_info["segment"]
        if not segment_id.isdigit():
            raise web.HTTPNotFound()

        styled_status = await asyncio.to_thread(
            self.console.recordings.request_render, files.directory.name, segment_id
        )
        return web.json_response(styled_status)

    async def record(self, request: web.Request) -> web.FileResponse:
        """Answers one file of a task's folder."""
        files = self.task_files(request)
        name = request.match_info["name"]
        if not self.RECORD_NAME_PATTERN.fullmatch(name) or name.startswith("."):
            raise web.HTTPNotFound()
        path = files.directory / name
        if not path.is_file():
            raise web.HTTPNotFound()

        return web.FileResponse(path)

    def all_tasks(self) -> list[dict[str, Any]]:
        """Returns every task's record with its tags, the latest first."""
        # every task is listed: tag filters must reach the oldest evaluations too
        paths = sorted(
            self.console.root.glob(f"*/{TaskFiles.TASK_NAME}"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        return [{**json.loads(path.read_text()), **Annotations.read(TaskFiles(path.parent))} for path in paths]

    @staticmethod
    def read_events(files: TaskFiles, after_id: int, limit: int | None) -> list[dict[str, Any]]:
        """Returns the task's events after the event ``after_id``, at most ``limit`` of them (None: all), each event of
        a turn with the turn's number."""
        events = []
        if files.events_path.exists():
            for event in TaskEvents(files.events_path).read_numbered():
                if event["id"] > after_id:
                    events.append(event)
                    if len(events) == limit:
                        break

        return events
