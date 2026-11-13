import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from uuid import uuid4


class ModelEvent(StrEnum):
    """What a client reports of a running request, each by the name its event has on the page and in a task's
    events."""

    PHASE = "phase"  # what a turn does now: the model's ModelPhase, or a phase the agent loop names itself
    REASONING_DELTA = "reasoning_delta"  # the next text of the reasoning, with its ReasoningKind
    ANSWER_DELTA = "answer_delta"  # the next text of the answer
    NOTICE = "notice"  # a message for the operator
    USAGE = "usage"  # the token counts so far: a Usage
    PERFORMANCE = "performance"  # the timings so far
    REQUEST_METRICS = "request_metrics"  # the size of the conversation as it was sent


class ModelPhase(StrEnum):
    """What a model does while its request runs, each by the word the page shows."""

    THINKING = "Thinking"  # it works on its answer
    COMPACTING = "Compacting"  # its service shortens the conversation to fit the context


class ReasoningKind(StrEnum):
    """How much of a model's reasoning a client passes on."""

    FULL = "full"  # all of its text
    SUMMARY = "summary"  # a summary its service writes
    NONE = "none"  # nothing


class HealthStatus(StrEnum):
    """What a check of a model's connection found."""

    OK = "ok"  # the model answers
    OFFLINE = "offline"  # the check failed
    UNCHECKED = "unchecked"  # no check has reached the model yet


class ModelHealth(TypedDict):
    """What a client's check reports, as the page reads it."""

    status: str  # a word of HealthStatus
    message: str  # a sentence for the operator
    reference_images: NotRequired[bool]  # whether the service takes a task's reference images, where a check tells


class ContextTokens(TypedDict, total=False):
    """The tokens of a conversation's context, where a service measures them apart from a request's input."""

    input: int | None  # the tokens the context holds; None while the measurement is pending
    context_limit: int  # the tokens the context can hold
    status: str  # pending or measured


class Compaction(TypedDict, total=False):
    """The compactions of a conversation's context that a service reports."""

    status: str  # the service's word for the turn's compaction; none without one
    count: int  # the compactions of the turn
    episode_count: int  # the compactions of the episode, the turn's among them


class RequestUsage(TypedDict, total=False):
    """One of the requests a service makes for a turn."""

    id: str
    output: int | None  # the tokens it has generated
    generation_time_ms: float  # how long it decoded, on the server's clock
    complete: bool  # whether it has ended


class Usage(TypedDict, total=False):
    """The token counts of a turn's request, as its usage events report them while it runs and at its end."""

    input: int  # the tokens of the request's prompt
    output: int | None  # the tokens generated
    cached_input: int  # the prompt's tokens the server took from its cache
    context_limit: int  # the tokens the model's context can hold
    final: bool  # whether no later report of the request follows
    partial: bool  # whether the counts are lower bounds: the request ended before a service had counted all of it
    context: ContextTokens
    compaction: Compaction
    requests: list[RequestUsage]


# takes one event of a running request: its kind and its data
EventHandler = Callable[[ModelEvent, dict[str, Any]], None]


class ImageFile:
    """A saved image of a turn, as a request carries its file."""

    path: Path  # the image's JPEG file

    def base64(self) -> str:
        """Returns the file's bytes as ASCII base64."""
        return base64.b64encode(self.path.read_bytes()).decode("ascii")


@dataclass(frozen=True)
class CameraImage(ImageFile):
    """One camera frame of a turn, under the camera's name as the model reads it."""

    camera_name: str
    path: Path
    uuid: str = field(default_factory=lambda: str(uuid4()))  # what a server's image cache knows the frame by


@dataclass(frozen=True)
class ReferenceImage(ImageFile):
    """A task's reference image, which a model is sent once, at the start of its episode."""

    path: Path
    uuid: str = field(default_factory=lambda: str(uuid4()))  # what a server's image cache knows the image by


@dataclass(frozen=True)
class ModelInput:
    """One turn's text and images."""

    text: str
    images: tuple[CameraImage, ...]
    reference_images: tuple[ReferenceImage, ...] = ()


@dataclass(frozen=True)
class ModelOutput:
    """One turn's answer with its reasoning and token usage."""

    answer: str
    reasoning: str = ""
    reasoning_kind: str = ReasoningKind.NONE.value  # a word of ReasoningKind, or the one a service gave
    usage: Usage = field(default_factory=dict)
