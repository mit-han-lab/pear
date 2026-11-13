import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pear.utils.file_utils import append_line


class TaskEventKind(StrEnum):
    """The events of a task besides the agent loop's and the model clients', each by the name it has in the task's
    event file."""

    TASK = "task"  # a task started: its record
    TASK_END = "task_end"  # the task ended
    SESSION = "session"  # a model session of the task started
    TURN_STARTED = "turn_started"  # a turn started: its attempt id, number and episode
    # the tokens of a request's context before its answer: no client reports it, a stored task's events may hold it
    CAPACITY = "capacity"
    EXECUTION = "execution"  # what the action did: its result
    TURN_END = "turn_end"  # the turn ended without a result: the phase it ended in


class TaskEvents:
    """A task's event file: one event on each line, in the order the events happened."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: dict[str, Any]) -> None:
        """Writes ``event`` as the file's next line."""
        append_line(self.path, json.dumps(event, allow_nan=False))

    def read(self) -> Iterator[dict[str, Any]]:
        """Yields the events in the file's order; a line that is no JSON raises where it is read."""
        with self.path.open() as events_file:
            for line in events_file:
                yield json.loads(line)

    def read_numbered(self) -> Iterator[dict[str, Any]]:
        """Yields the events in the file's order, each event of a turn with the turn's number and start time: where
        the file gives none, the turns count in the order they first appear."""
        turn_fields: dict[str, dict[str, Any]] = {}  # by attempt id
        last_number = 0

        for event in self.read():
            data = event.get("data", {})
            attempt_id = data.get("attempt")
            if attempt_id:
                if attempt_id not in turn_fields:
                    last_number = max(last_number + 1, data.get("turn_number", 0))
                    turn_fields[attempt_id] = {
                        "turn_number": last_number,
                        "started_at": data.get("started_at", event.get("time")),
                    }
                event = {**event, "data": {**turn_fields[attempt_id], **data}}

            yield event
