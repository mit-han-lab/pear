from enum import StrEnum
from typing import Any

from pear.deploy.station.agent_loop.clients.schemas import ModelPhase
from pear.deploy.station.agent_loop.schemas import TurnPhase


class CaptionPhase(StrEnum):
    """The phases a recording segment names by itself; every other phase it shows is the word of an event."""

    STARTING = "Starting"  # no turn has started yet
    COMPLETED = "Completed"  # an action's result or a turn's end that names neither a phase nor a status


class Captions:
    """The two usage lines a recorded frame carries: the size of the context, and the tokens of the current turn."""

    # the phases in which a missing count is still to come
    WAITING_PHASES = (CaptionPhase.STARTING, TurnPhase.OBSERVING, ModelPhase.THINKING, ModelPhase.COMPACTING)
    SHOWN_COMPACTION_STATUSES = ("running", "interrupted", "failed")  # compaction statuses written after the count

    @classmethod
    def context(cls, state: dict[str, Any]) -> str:
        """Returns the line of the latest context size, with the turn that reported it."""
        usage = state.get("context_usage")
        if usage is None:
            return "Context Usage: " + cls._missing_text(state)
        if usage.get("request_context"):
            return cls._request_context(usage)

        used_tokens, limit = usage["tokens"], usage.get("context_limit")
        if limit:
            caption = f"Context Usage: {used_tokens:,} / {limit:,} tokens ({100 * used_tokens / limit:.1f}%)"
        else:
            caption = f"Context Usage: {used_tokens:,} tokens"

        return f"{caption} | Last: turn {usage['turn_number']}"

    @classmethod
    def inference(cls, state: dict[str, Any]) -> str:
        """Returns the line of the input and output tokens of the current turn."""
        usage = state.get("inference_usage")
        missing_text = cls._missing_text(state)
        if usage is None or usage["turn_number"] != state.get("turn_number"):
            return "Current Turn Usage: " + missing_text
        if "partial" in usage:
            return cls._partial_inference(usage)

        input_tokens, output_tokens = usage["input"], usage["output"]
        if output_tokens is None:
            return f"Current Turn Usage: {input_tokens:,} + {missing_text}"

        return f"Current Turn Usage: {input_tokens:,} + {output_tokens:,} = {usage['tokens']:,} tokens"

    @classmethod
    def _request_context(cls, usage: dict[str, Any]) -> str:
        """Returns the context line of a service that measures the context apart, with its compactions."""
        compaction = usage.get("compaction", {})
        status = compaction.get("status", "none")
        episode_count = compaction.get("episode_count")

        if type(episode_count) is int:
            suffix = f" | Compactions: {episode_count} total"
            if status in cls.SHOWN_COMPACTION_STATUSES:
                suffix += f" ({status})"
        else:
            # compaction data without the episode's count holds the turn's count only
            suffix = "" if status == "none" else f" | Compact: {status} ({compaction.get('count', 0)})"

        used_tokens, limit = usage["tokens"], usage.get("context_limit")
        if used_tokens is None:
            return "Request Context: Pending" + suffix

        capacity_text = f" / {limit:,}" if limit else ""

        return f"Request Context: {used_tokens:,}{capacity_text} tokens" + suffix

    @classmethod
    def _partial_inference(cls, usage: dict[str, Any]) -> str:
        """Returns the turn's line of a service whose counts may be lower bounds or still running."""
        is_pending = usage.get("final") is False
        is_partial = usage["partial"] and not is_pending
        suffix = " (in progress)" if is_pending else " (incomplete)" if is_partial else ""
        counts_text = (
            f"{cls._count_text(usage['input'], is_partial)} in + {cls._count_text(usage['output'], is_partial)} out"
        )

        return f"Current Turn Usage: {counts_text}{suffix}"

    @staticmethod
    def _count_text(value: int | None, is_partial: bool) -> str:
        """Returns a token count as text, marked as a lower bound when it is one."""
        if value is None:
            return "Not provided"

        return ("at least " if is_partial else "") + f"{value:,}"

    @classmethod
    def _missing_text(cls, state: dict[str, Any]) -> str:
        """Returns what stands for a count that is not there: still to come, or never given."""
        return "Pending" if state.get("phase", CaptionPhase.STARTING.value) in cls.WAITING_PHASES else "Not provided"
