import json
from dataclasses import dataclass, replace
from typing import Any

from pear.deploy.station.agent_loop.schemas import TurnEvent
from pear.deploy.station.records.context_usage import ContextUsage
from pear.deploy.station.records.task_events import TaskEventKind
from pear.deploy.station.records.task_files import TaskFiles
from pear.deploy.station.records.video.sources import SourceClip


@dataclass(frozen=True)
class TurnSpan:
    """One turn of a styled video: what it shows and when."""

    number: int  # the turn's number from 1, shown as "Turn N"
    start_s: float  # from the video's start, where the turn's action begins to show
    end_s: float  # where it stops showing: the next turn's start, or the video's end
    action: dict[str, Any] | None  # the parsed action of the model; None when the turn has none
    tokens: int | None  # the tokens of the context, shown for the turn


@dataclass(frozen=True)
class TaskTimeline:
    """What a task's styled video shows over the cameras' own videos of its segments, played back to back."""

    LABEL_SEPARATOR = " · "  # between the name and the tags of a model's label, one of which may be the badge

    instruction: str
    model_label: str
    badge: str | None  # the chip of the reasoning effort; None for no chip
    duration_s: float  # of the clips back to back
    turns: list[TurnSpan]  # in order, none overlapping another; a turn the model never answered has none
    motion_spans: list[tuple[float, float]]  # (start_s, end_s) of what the motion-only video keeps

    @classmethod
    def load(cls, files: TaskFiles, clips: list[SourceClip]) -> "TaskTimeline":
        """Returns the timeline of a task's records over its recorded clips, which are given in recording order."""
        task = json.loads(files.task_path.read_text())
        # only a model with a reasoning effort records one
        badge = task.get("reasoning_effort")
        name, *tags = (task.get("model_label") or task["model"]).split(cls.LABEL_SEPARATOR)

        clip_spans = [(clip.start_monotonic_s, clip.start_monotonic_s + clip.duration_s) for clip in clips]
        duration_s = sum((clip.duration_s for clip in clips), 0.0)
        answers = [
            (number, cls._elapsed_seconds(clip_spans, answered_at), action, tokens)
            for number, answered_at, action, tokens in cls._answers(files)
        ]

        return cls(
            instruction=task["instruction"],
            # the badge shows once, as the chip
            model_label=cls.LABEL_SEPARATOR.join([name, *(tag for tag in tags if tag != badge)]),
            badge=badge,
            duration_s=duration_s,
            turns=cls._spans(answers, duration_s),
            motion_spans=cls._motion_spans(clips),
        )

    def cut_to_motion(self) -> "TaskTimeline":
        """Returns this timeline cut to its motion spans, played back to back, as the motion-only video shows it."""
        duration_s = self._elapsed_seconds(self.motion_spans, self.duration_s)
        answers = [
            (turn.number, self._elapsed_seconds(self.motion_spans, turn.start_s), turn.action, turn.tokens)
            for turn in self.turns
        ]

        return replace(
            self, duration_s=duration_s, turns=self._spans(answers, duration_s), motion_spans=[(0.0, duration_s)]
        )

    @staticmethod
    def _answers(files: TaskFiles) -> list[tuple[int, float, dict[str, Any] | None, int | None]]:
        """Returns, for each turn the model answered: its number, the monotonic time of the answer, the action or
        None, and the context's tokens."""
        usage = ContextUsage()
        turns: dict[str, dict[str, Any]] = {}  # by attempt id, in the order the turns started

        for line in files.events_path.read_text().splitlines():
            event = json.loads(line)
            kind, data = event["type"], event["data"]
            latest_context, _ = usage.on_event(kind, data)

            if kind == TaskEventKind.TURN_STARTED:
                turns[data["attempt"]] = {
                    "number": data["turn_number"],
                    "fields": set(data),
                    "answered_at": None,
                    "action": None,
                }
            turn = turns.get(data.get("attempt"))
            if turn is None:
                continue

            if kind == TurnEvent.MODEL_OUTPUT:
                turn["answered_at"] = event["monotonic_s"]
            elif kind == TurnEvent.ACTION:
                # the event file holds the turn's own fields in each event of the turn
                turn["action"] = {key: value for key, value in data.items() if key not in turn["fields"]}

            # the context after the turn's last event: the snapshot the recorder captions too
            turn["tokens"] = latest_context["tokens"] if latest_context else None

        return [
            (turn["number"], turn["answered_at"], turn["action"], turn["tokens"])
            for turn in turns.values()
            if turn["answered_at"] is not None
        ]

    @staticmethod
    def _spans(
        answers: list[tuple[int, float, dict[str, Any] | None, int | None]], duration_s: float
    ) -> list[TurnSpan]:
        """Returns each turn from its answer to the next answer or ``duration_s``, without the turns that never
        show."""
        end_times = [start_s for _, start_s, _, _ in answers[1:]] + [duration_s]

        return [
            TurnSpan(number=number, start_s=start_s, end_s=end_s, action=action, tokens=tokens)
            for (number, start_s, action, tokens), end_s in zip(answers, end_times)
            if end_s > start_s
        ]

    @staticmethod
    def _motion_spans(clips: list[SourceClip]) -> list[tuple[float, float]]:
        """Returns the motion spans the recorder kept of each segment, moved to where the segment's clip plays."""
        motion_spans, offset_s = [], 0.0
        for clip in clips:
            # a clip's videos are in its segment's folder, beside the segment's manifest
            manifest_path = next(iter(clip.files.values())).parent / TaskFiles.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            # a segment that failed after its videos were saved has no motion intervals
            for start, end in manifest.get("motion_intervals", []):
                motion_spans.append((offset_s + start / manifest["fps"], offset_s + end / manifest["fps"]))
            offset_s += clip.duration_s

        return motion_spans

    @staticmethod
    def _elapsed_seconds(spans: list[tuple[float, float]], moment: float) -> float:
        """Returns the seconds of ``spans``, played back to back, before ``moment``; a moment between two spans
        counts as the start of the next."""
        return sum((min(max(moment - start, 0.0), end - start) for start, end in spans), 0.0)
