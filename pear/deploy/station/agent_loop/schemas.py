from enum import StrEnum


class TurnPhase(StrEnum):
    """The phase of a turn the agent loop names itself, by the word the page shows (the model's phases are ModelPhase,
    the robot's ActionPhase)."""

    OBSERVING = "Observing"  # the arms come to rest and are observed


class TurnEvent(StrEnum):
    """What the agent loop reports of a turn besides its phases, each by the name its event has on the page and in a
    task's events."""

    OBSERVATION = "observation"  # what the model is shown
    MODEL_OUTPUT = "model_output"  # the model's answer
    ACTION = "action"  # the action read from the answer
    DIAGNOSTIC = "diagnostic"  # what the robot recorded of the error that ended an action
