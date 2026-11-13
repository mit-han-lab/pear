from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import NotRequired, TypedDict

import numpy as np


class RobotStatus(StrEnum):
    """What a robot reports of itself: closed, a settled hold ready for an action, busy, or a latched fault."""

    OFF = "OFF"
    READY = "READY"
    BUSY = "BUSY"
    FAULT = "FAULT"


class ControlState(StrEnum):
    """The state of the control loop under a robot."""

    HOLD = "HOLD"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FAULT = "FAULT"
    CLOSED = "CLOSED"


MOVING_CONTROL_STATES = (
    ControlState.RUNNING,
    ControlState.STOPPING,
)  # the states in which the arms follow an operation


class MotionEndReason(StrEnum):
    """How the last operation of a control loop ended, each by the word a snapshot's motion names it with."""

    INITIAL_HOLD = "initial_hold"  # no operation ran yet: the hold that enabling took
    FINISHED = "finished"
    STOPPED = "stopped"
    CANCELLED = "cancelled"  # the operator's pause
    ACTION_TIMEOUT = "action_timeout"
    SEQUENCE_DEADLINE = "sequence_deadline"  # a sequence that fell behind its own rate
    FAULT = "fault"
    RECOVERED = "recovered"


class RecoveryPhase(StrEnum):
    """What a fault recovery is doing, in the words the page shows."""

    CHECKING_STABILITY = "Checking stability"
    RELEASING_TARGET = "Releasing old target"
    VERIFYING_HOLD = "Verifying new hold"


class ActionPhase(StrEnum):
    """What a robot is doing with an action, in the words its progress reports."""

    PLANNING = "Planning"
    EXECUTING = "Executing"
    SETTLING = "Settling"  # the action's motion is over and the arms come to rest


class CameraFrame(TypedDict):
    """One camera image and when it arrived."""

    rgb: np.ndarray
    received_monotonic_s: float
    frame_number: int


class ArmFeedback(TypedDict):
    """One arm's latest measurement, as a snapshot carries it."""

    position: list[float]
    velocity: list[float]
    sampled_monotonic_s: float
    sample_id: object  # identifies the raw feedback batch, so a repeated batch adds nothing
    age_seconds: float


class Arrival(TypedDict):
    """Whether the arms reached what the operation requested and have settled there."""

    settled: bool
    reached: bool
    signed_error: list[float]
    tolerance: list[float]


class Motion(TypedDict):
    """How the last operation ended and whether it is settling; a robot's driver adds its telemetry."""

    end_reason: str | None  # the word of a MotionEndReason or a robot's own; None while an operation runs
    settling: bool


class RecoveryProgress(TypedDict):
    """A fault recovery in progress."""

    phase: str  # the word of a RecoveryPhase
    metrics: dict


class SnapshotFields(TypedDict):
    """The control state of a robot's arms at one instant."""

    captured_monotonic_s: float
    fault: str | None
    fault_details: dict | None
    generation: int  # advances with every operation, stop and fault
    targets: dict[str, list[float]]  # per arm
    arms: dict[str, ArmFeedback]
    arrival: Arrival
    settling: dict  # the stability window's metrics
    recovery: RecoveryProgress | None
    motion: Motion


class ControlSnapshot(SnapshotFields):
    """A control loop's snapshot, as observations and results carry it."""

    status: str  # the word of a ControlState


class DetailedStatus(SnapshotFields):
    """A robot's status with its control snapshot."""

    status: str  # the word of a RobotStatus
    control_state: str  # the word of a ControlState


class BriefStatus(TypedDict):
    """A robot's status; only ``status`` while it is closed."""

    status: str  # the word of a RobotStatus
    settled: NotRequired[bool]
    reached: NotRequired[bool]
    fault: NotRequired[str | None]


class Observation(TypedDict):
    """One sample of a robot's session: measured joints, camera frames and the control snapshot."""

    observation_id: str
    session_id: str
    captured_monotonic_s: float
    joint_positions: np.ndarray  # the measured joint vector
    images: dict[str, np.ndarray]  # per camera role
    camera_timestamps: dict[str, dict]  # per camera role, the frame's fields but its image
    control_snapshot: ControlSnapshot


class HomePlan(TypedDict):
    """A checked Home plan that an operator reviews before anything moves; a robot adds its own fields."""

    operation_seconds_estimate: float  # how long Home takes once approved


class HomeReport(TypedDict):
    """What a completed Home did and how close each joint ended to its Home target."""

    plan: HomePlan
    before: ControlSnapshot
    phases: list[dict]  # the completed stages in order, each with its stage name and snapshot after it
    after: ControlSnapshot
    results: dict  # per-joint accuracy over the final stability window


@dataclass(frozen=True)
class Camera:
    """One camera of a robot."""

    role: str  # key in observations, previews and recordings
    name: str  # what the model and saved files call it
    label: str  # shown in the browser


@dataclass(frozen=True)
class ActionFormat:
    """One format of the state a model reads and the action it writes."""

    name: str
    schema: dict[str, tuple[str, ...]]  # the parts and the quantities an action sets, in the order the model reads
    cameras: tuple[Camera, ...]  # the cameras whose frames the model sees, in order
    environment_prompt: str  # the ENVIRONMENT section of the system prompt; empty: no section
    decimals: dict[str, int]  # rounding of each quantity in observations
    action_model: type  # the model class of one parsed action
    result_feedback: Callable[..., str | None]  # (result, *, is_dispatched) -> text for the next turn, or None
