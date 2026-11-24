import base64
import json
from pathlib import Path

from pear.deploy.station.agent_loop.schemas import TurnEvent
from pear.deploy.station.records.task_events import TaskEvents
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.task_timeline import TaskTimeline


class TracePage:
    """A task's trace page: the page's trace template with the task's records and every picture they show inlined,
    one HTML file that opens anywhere."""

    TEMPLATE_PATH = Path(__file__).parents[1] / "static" / "trace.html"  # built with the page
    DATA_MARK = "__PEAR_TRACE_DATA__"  # the template's text that the records replace
    IMAGE_URL_PREFIX = "data:image/jpeg;base64,"  # of a picture inlined: every recorded picture is a JPEG file

    def __init__(self, files: TaskFiles, robot_title: str, record_prefix: str) -> None:
        self.files = files
        self.robot_title = robot_title  # the page's name of the robot
        self.record_prefix = record_prefix  # of the URL at which the page's service serves a file of the task's folder

    def html(self) -> str:
        """Returns the page."""
        files = self.files
        task = json.loads(files.task_path.read_text())
        session = json.loads(files.session_path.read_text()) if files.session_path.exists() else {}
        events = TaskEvents(files.events_path).read_shown() if files.events_path.exists() else []
        for event in events:
            if event["type"] == TurnEvent.OBSERVATION:
                images = event["data"].get("images", {})
                event["data"]["images"] = {
                    camera: inlined_url for camera, url in images.items() if (inlined_url := self._inlined(url))
                }

        answers = TaskTimeline.answers(files) if files.events_path.exists() else []
        trace = {
            "robot": self.robot_title,
            "task": {
                **task,
                "images": [
                    {**image, "url": inlined_url}
                    for image in task.get("images", [])
                    if (inlined_url := self._inlined(image["url"]))
                ],
            },
            "system_prompt": session.get("system_prompt"),
            "events": events,
            # the conversation's tokens after its last answered turn, as the styled videos show them
            "token_usage": answers[-1][3] if answers else None,
        }

        # no "<" in the data can end its script element
        data = json.dumps(trace, ensure_ascii=False).replace("<", "\\u003c")

        return self.TEMPLATE_PATH.read_text().replace(self.DATA_MARK, data)

    def _inlined(self, url: str) -> str | None:
        """Returns the picture the page's service serves at ``url`` as a data URL; None when its file is gone."""
        path = self.files.directory / url.removeprefix(self.record_prefix)
        if not path.is_file():
            return None

        return self.IMAGE_URL_PREFIX + base64.b64encode(path.read_bytes()).decode()
