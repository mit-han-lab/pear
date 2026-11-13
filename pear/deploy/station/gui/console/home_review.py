import threading
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pear.deploy.station.control.schemas import HomePlan
from pear.deploy.station.exceptions import OperatorPaused


class HomeStatus(StrEnum):
    """How a Home stands, as the page receives it beside the Home's stage."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HomeReview:
    """A Home plan waiting for the operator's confirmation, and the operator's answer to it."""

    def __init__(self, lock: threading.RLock) -> None:
        self.lock = lock  # the console's: its snapshot reads the plan under it, as one moment with the rest
        self.plan: dict[str, Any] | None = None  # the plan under review, with its plan_id; None while none waits
        self.decision: bool | None = None  # the operator's answer to the plan; None until it arrives
        self.answered = threading.Event()  # set by the operator's answer, or by a STOP

    def prepare(self, plan: HomePlan) -> dict[str, Any]:
        """Returns ``plan`` under a new plan_id, with no answer and no wake left from before; the plan is published
        only after this, so that a fast answer is never erased."""
        review_plan = {**plan, "plan_id": uuid4().hex}
        self.decision = None
        self.answered.clear()

        return review_plan

    def wait_for_decision(self, review_plan: dict[str, Any], cancellation: threading.Event) -> bool:
        """Publishes ``review_plan`` and returns the operator's answer to it; a STOP ends the wait, and the end of
        the console's operation withdraws the plan."""
        with self.lock:
            self.plan = review_plan

        # a wake without an answer is a STOP, seen as the cancellation, or one left from an earlier STOP
        while self.decision is None:
            OperatorPaused.check_cancelled(cancellation)
            self.answered.wait()
            self.answered.clear()
        OperatorPaused.check_cancelled(cancellation)

        # the answer is taken as the plan is withdrawn: a later answer finds no plan
        with self.lock:
            decision, self.plan = self.decision, None

        return decision

    def confirm(self, plan_id: object, is_confirmed: bool) -> None:
        """Answers the plan under review, and refuses an answer to a plan that is no longer current."""
        # under the lock the plan stays current until the answer is recorded
        with self.lock:
            if not self.plan or plan_id != self.plan["plan_id"]:
                raise ValueError("This home plan is no longer current")
            self.decision = is_confirmed
        self.answered.set()
