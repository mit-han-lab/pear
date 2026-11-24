from typing import Any

from pear.deploy.station.agent_loop.clients.schemas import ModelEvent, Usage
from pear.deploy.station.agent_loop.schemas import TurnEvent
from pear.deploy.station.records.task_events import TaskEventKind


class ContextUsage:
    """The tokens a model's context held at its latest report, followed over the events of one task."""

    METRIC_KEYS = ("input", "output", "context_limit")  # the counts read from a capacity or usage event

    def __init__(self) -> None:
        self.episode_id: str | None = None  # of the model session the reports belong to
        self._reset()

    def on_event(self, kind: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Follows one event of the task and returns the latest context, None while there is none."""
        if kind in (TaskEventKind.TASK, TaskEventKind.TASK_END):
            self._reset()
            self.episode_id = None
        elif kind == TaskEventKind.SESSION:
            self._reset()
            self.episode_id = data.get("episode_id")
        elif kind == TaskEventKind.TURN_STARTED:
            self._start_turn(data)
        elif self.attempt_id and data.get("attempt") == self.attempt_id:
            self._record(kind, data)

        return self.latest_context

    def _reset(self) -> None:
        """Forgets every report."""
        self.attempt_id: str | None = None  # of the current turn
        self.request_id: str | None = None  # of the current turn's request
        self.turn_number: int | None = None
        self.latest_context: dict[str, Any] | None = None  # the context's tokens at the latest report
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
