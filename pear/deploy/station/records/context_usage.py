import threading
from typing import Any

from pear.deploy.station.agent_loop.clients.schemas import ModelEvent, Usage
from pear.deploy.station.agent_loop.schemas import TurnEvent
from pear.deploy.station.records.task_events import TaskEventKind


class ContextUsage:
    """The tokens a model's context held at its latest report and the tokens of the current turn, followed over the
    events of one task."""

    METRIC_KEYS = ("input", "output", "context_limit")  # the counts read from a capacity or usage event

    def __init__(self) -> None:
        self.lock = threading.Lock()  # held while one event is followed
        self.task_id: str | None = None  # of the task the reports belong to
        self.episode_id: str | None = None  # of the model session the reports belong to
        self._reset()

    def on_event(self, kind: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Follows one event of the task and returns the latest context and the current turn's usage, each None
        while there is none."""
        with self.lock:
            if kind in (TaskEventKind.TASK, TaskEventKind.TASK_END):
                self._reset()
                self.task_id, self.episode_id = data.get("id"), None
            elif kind == TaskEventKind.SESSION:
                self._reset()
                self.episode_id = data.get("episode_id")
            elif kind == TaskEventKind.TURN_STARTED:
                self._start_turn(data)
            elif self.attempt_id and data.get("attempt") == self.attempt_id:
                self._record(kind, data)

            # the two are replaced, never changed, so a frame that was given one keeps its values
            return self.latest_context, self.turn_usage

    def context_for_task(self, task_id: str) -> dict[str, Any] | None:
        """Returns the latest context of the task ``task_id``; the reports of another task are forgotten first."""
        with self.lock:
            if task_id != self.task_id:
                self._reset()
                self.task_id, self.episode_id = task_id, None

            return self.latest_context

    def _reset(self) -> None:
        """Forgets every report."""
        self.attempt_id: str | None = None  # of the current turn
        self.request_id: str | None = None  # of the current turn's request
        self.turn_number: int | None = None
        self.latest_context: dict[str, Any] | None = None  # the context's tokens at the latest report
        self.turn_usage: dict[str, Any] | None = None  # the tokens of the current turn's request
        # the counts of the current turn's capacity and usage events
        self.capacity: dict[str, int | None] = {}
        self.usage: dict[str, int | None] = {}

    def _start_turn(self, data: dict[str, Any]) -> None:
        """Begins a turn, which keeps the latest context of its own model session only."""
        if data.get("episode_id") != self.episode_id:
            self._reset()

        self.episode_id = data.get("episode_id")
        self.attempt_id, self.turn_number = data["attempt"], data["turn_number"]
        self.request_id = None
        self.turn_usage = None
        self.capacity, self.usage = {}, {}

    def _record(self, kind: str, data: dict[str, Any]) -> None:
        """Takes an event of the current turn: the id of its request, or a report of that request's tokens."""
        if kind == TurnEvent.OBSERVATION:
            self.request_id = data.get("turn_id", data.get("request_id"))
        elif kind in (TaskEventKind.CAPACITY, ModelEvent.USAGE) and data.get("request_id") in (None, self.request_id):
            if kind == ModelEvent.USAGE and isinstance(data.get("context"), dict):
                self._record_request_context(data)
            else:
                self._record_metrics(kind, data)

    def _record_request_context(self, usage: Usage) -> None:
        """Takes a usage report of a service that measures the context apart from the request's own tokens."""
        context = usage["context"]
        metadata = {"turn_number": self.turn_number, "episode_id": self.episode_id, "request_id": self.request_id}
        input_tokens, output_tokens = usage.get("input"), usage.get("output")

        self.turn_usage = {
            **metadata,
            "input": input_tokens,
            "output": output_tokens,
            "tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
            "partial": bool(usage.get("partial")),
            "final": usage.get("final", True),
            "compaction": dict(usage.get("compaction", {})),
        }

        self.latest_context = {
            **metadata,
            **context,
            "tokens": context.get("input"),
            "request_context": True,
            "compaction": dict(usage.get("compaction", {})),
        }

    def _record_metrics(self, kind: str, data: dict[str, Any]) -> None:
        """Takes the counts of a capacity or usage event, whose input is the whole context."""
        metrics = {
            key: value if type(value) is int and value >= 0 else None
            for key in self.METRIC_KEYS
            if (value := data.get(key)) is not None
        }
        if kind == TaskEventKind.CAPACITY:
            self.capacity = metrics
        else:
            self.usage = metrics

        used_tokens = self.usage.get("input")
        if used_tokens is None:
            used_tokens = self.capacity.get("input")
        if used_tokens is None:
            return

        output_tokens = self.usage.get("output")
        # the input holds the history already: the inputs of earlier turns are not added to it
        self.latest_context = {
            "input": used_tokens,
            "output": output_tokens,
            "tokens": used_tokens + (output_tokens or 0),
            "context_limit": self.capacity.get("context_limit") or self.usage.get("context_limit"),
            "turn_number": self.turn_number,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
        }
        self.turn_usage = self.latest_context
