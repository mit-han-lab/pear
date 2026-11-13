from enum import StrEnum

from pear.deploy.station.agent_loop.clients.schemas import ModelPhase
from pear.deploy.station.agent_loop.schemas import TurnPhase
from pear.deploy.station.control.schemas import ActionPhase, RobotStatus
from pear.deploy.station.records.recording.captions import CaptionPhase

STOPPED_STATUSES = (RobotStatus.READY, RobotStatus.OFF)  # the robot's statuses in which a stop has ended


class Phase(StrEnum):
    """What the console or one of its task's turns is doing, in the words the page shows."""

    DISABLED = "Disabled"
    ENABLING = "Enabling"
    HOLDING = "Holding"
    PAUSED = "Paused"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    FAULT = "Fault"
    RECOVERING_CONTROL = "Recovering control"
    RECONNECTING_CAMERAS = "Reconnecting cameras"
    CHECKING_HOME = "Checking home"
    CONFIRM_HOME = "Confirm home"
    OPENING_GRIPPERS = "Opening grippers"
    CHECKING_DIRECT_HOME = "Checking direct Home"
    HOMING = "Homing"
    HOME_COMPLETE = "Home complete"
    HOME_FAILED = "Home failed"
    # the phases of a turn, in the words the agent loop, the model's client, the robot and the recording use for them
    OBSERVING = TurnPhase.OBSERVING
    THINKING = ModelPhase.THINKING
    COMPACTING = ModelPhase.COMPACTING
    PLANNING = ActionPhase.PLANNING
    EXECUTING = ActionPhase.EXECUTING
    SETTLING = ActionPhase.SETTLING
    COMPLETED = CaptionPhase.COMPLETED
    REJECTED = "Rejected"
    FAILED = "Failed"
    INTERRUPTED = "Interrupted"
    CONTEXT_LIMIT = "Context limit"

    @classmethod
    def idle_for(cls, robot_status: str, *, is_latched: bool, has_task: bool) -> "Phase":
        """Returns the phase of a console with no running operation: a STOP latch first, then a paused task, then
        the motors, which ``robot_status``, the word of a RobotStatus, tells."""
        if is_latched:
            return cls.STOPPED
        if has_task:
            return cls.PAUSED

        return cls.HOLDING if robot_status == RobotStatus.READY else cls.DISABLED
