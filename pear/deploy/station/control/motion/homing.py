from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.trajectory import GripperProfile, JointMove
from pear.deploy.station.control.motion.execution import Execution
from pear.deploy.station.control.schemas import ControlSnapshot, HomePlan, HomeReport, MotionEndReason, Observation
from pear.deploy.station.exceptions import OperatorPaused


class HomeFailed(RuntimeError):
    """Raised when Home stopped before it finished; the error that stopped it is its cause."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details  # the stage, the error, the completed stages and each joint's accuracy at the failure


class HomeStage(StrEnum):
    """The stages of a return to Home, by the words its progress reports and its records carry."""

    CHECKING = "checking"
    OPENING = "opening"
    HOMING = "homing"
    COMPLETED = "completed"


class Homing(ABC):
    """One return to Home of a robot under the action's guard: the check, the plan and its approval, the grippers'
    opening, the recheck, the move and its accuracy; a robot gives its own plan, recheck and accuracy."""

    WAIT_MARGIN_S = 5.0  # past the move and Home's arrival timeout before the last wait gives up

    # where a robot's plan holds what the flow reads of it
    START_TARGETS_KEY: str  # the joint vector the arms hold once the grippers have opened
    GOAL_TARGETS_KEY: str  # the joint vector at Home
    OPENING_KEY: str  # what the grippers that open first are asked for; false: none opens

    def __init__(
        self, robot: Execution, guard: ActionGuard, report_progress: Callable[[dict[str, Any]], None] | None
    ) -> None:
        self.robot, self.guard, self.report_progress = robot, guard, report_progress
        self.control_config = guard.control_loop.control_config
        self.robot_config = guard.control_loop.robot_config
        self.stage = HomeStage.CHECKING
        self.completed_stages: list[dict[str, Any]] = []  # each with its stage and what it left
        # the arms Home returns; a robot's plan may narrow them
        self.arms: Sequence[str] = self.robot_config.LAYOUT.arms
        # the joint vector the arms are judged against; None until a plan is approved
        self.targets: list[float] | None = None

    def run(self, *, approve_plan: Callable[[HomePlan], bool] | None, deadline: float) -> HomeReport:
        """Plans Home, has ``approve_plan`` approve the plan, opens the grippers, returns the arms and verifies them,
        all by ``deadline``, and returns the report; a failure raises HomeFailed, an operator's pause as it is."""
        robot, guard = self.robot, self.guard

        try:
            self.report_stage(HomeStage.CHECKING)
            robot.wait_for_settled_hold(guard=guard, deadline=deadline)
            observation = robot.observe()
            snapshot_before = observation["control_snapshot"]
            plan, move = self.plan(observation, deadline)

            guard.check()
            if approve_plan is not None and approve_plan(copy.deepcopy(plan)) is not True:
                raise ValueError("Home cancelled before execution")
            guard.check()
            self.check_time(plan, deadline)

            self.targets = plan[self.START_TARGETS_KEY]
            self._open_grippers(plan, observation, deadline)
            if move is not None:
                self.report_stage(HomeStage.CHECKING)
                observation = robot.observe()
                self.recheck(observation, move, deadline)

            self.targets = plan[self.GOAL_TARGETS_KEY]
            snapshot_after = self._return_arms(move, observation, deadline)

            guard.check()
            results = self.accuracy(snapshot_after)
            self.report_stage(HomeStage.COMPLETED, results=results)

            return {
                "plan": plan,
                "before": snapshot_before,
                "phases": self.completed_stages,
                "after": snapshot_after,
                "results": results,
            }
        except OperatorPaused:
            raise
        except Exception as error:
            details = {
                "stage": self.stage.value,
                "error": str(error),
                "completed_stages": self.completed_stages,
                "results": {} if self.targets is None else self.accuracy(guard.control_loop.snapshot()),
            }
            raise HomeFailed(str(error), details) from error

    @abstractmethod
    def plan(self, observation: Observation, deadline: float) -> tuple[HomePlan, JointMove | None]:
        """Returns the robot's checked plan of Home from ``observation``, which the operator reviews, and its arm
        move, None when the arms are at Home already."""

    @abstractmethod
    def is_opened(self, gripper_report: dict[str, Any]) -> bool:
        """Whether the report of the grippers' operation says that every gripper it moved is open."""

    @abstractmethod
    def recheck(self, observation: Observation, move: JointMove, deadline: float) -> None:
        """Raises unless the approved ``move`` is still valid with the feedback of ``observation``, taken once the
        grippers have opened."""

    @abstractmethod
    def tolerance(self) -> np.ndarray:
        """Returns Home's tolerance of each coordinate of an arm."""

    @abstractmethod
    def accuracy(self, snapshot: ControlSnapshot) -> dict[str, Any]:
        """Returns how close the arms of ``snapshot`` are to the targets they are judged against, as the robot's
        reports keep it."""

    def report_stage(self, stage: HomeStage, **data: Any) -> None:
        """Enters ``stage`` and tells the caller's progress of it with ``data``, checking the guard around the
        report."""
        self.stage = stage

        self.guard.check()
        if self.report_progress is not None:
            self.report_progress({"stage": stage.value, **data})
        self.guard.check()

    def wait(self, timeout_s: float) -> ControlSnapshot:
        """Waits until the control loop's own history shows the arms settled at their targets within Home's
        tolerance, and returns the snapshot then."""
        control_loop, home = self.guard.control_loop, self.robot_config.home
        targets = self.robot_config.LAYOUT.split(self.targets)

        def is_home_window_stable(snapshot: ControlSnapshot) -> bool:
            """Whether the loop's history shows the arms stable at their targets; ``snapshot`` receives the window's
            measurements."""
            metrics = control_loop.stability(
                arms=self.arms,
                duration_s=home.settle_window_s,
                targets=targets,
                position_tolerance=self.tolerance(),
            )
            snapshot["home_accuracy_window"] = metrics

            return metrics["stable"]

        return control_loop.wait_for_hold(
            timeout_s, accepts=is_home_window_stable, guard=self.guard, label="Home feedback"
        )

    def arm_accuracy(self, snapshot: ControlSnapshot, arm: str) -> dict[str, Any]:
        """Returns the error of ``arm`` in ``snapshot`` against its targets, judged by Home's own tolerance."""
        layout, tolerance = self.robot_config.LAYOUT, self.tolerance()
        position = np.asarray(snapshot["arms"][arm]["position"])
        error = position - np.asarray(layout.split(self.targets)[arm])
        open_error = abs(layout.gripper_value(position) - GripperProfile.OPEN_ENDPOINT)

        return {
            "reached": bool(np.all(abs(error) <= tolerance)),
            "gripper_open": bool(open_error <= self.robot_config.gripper_tolerance),
            "error": error.tolist(),
            # a JSON record refuses infinity: a gripper that is not judged has no tolerance
            "tolerance": [float(bound) if np.isfinite(bound) else None for bound in tolerance],
            "violations": [
                {
                    layout.COORDINATE_LABEL: layout.coordinates[coordinate_index],
                    "error": float(error[coordinate_index]),
                    "limit": float(tolerance[coordinate_index]),
                }
                for coordinate_index in np.flatnonzero(abs(error) > tolerance)
            ],
        }

    def _open_grippers(self, plan: HomePlan, observation: Observation, deadline: float) -> None:
        """Opens the grippers that the plan asks to open first, and raises unless every one of them opened."""
        gripper_request = plan[self.OPENING_KEY]
        if not gripper_request:
            return

        self.report_stage(HomeStage.OPENING)
        opening_report = self.robot.operate_grippers(
            gripper_request, observation=observation, guard=self.guard, deadline=deadline
        )
        if not self.is_opened(opening_report):
            raise RuntimeError("Home opening did not complete; no arm move dispatched")
        self.completed_stages.append({"stage": HomeStage.OPENING.value, **opening_report})

    def _return_arms(self, move: JointMove | None, observation: Observation, deadline: float) -> ControlSnapshot:
        """Starts ``move``, if there is one, from the hold of ``observation``, waits until the arms have arrived at
        Home and returns the snapshot then."""
        robot, guard = self.robot, self.guard
        move_s = 0.0
        if move is not None:
            move_s = move.duration_s
            self.report_stage(HomeStage.HOMING, duration_seconds=move_s)
            robot.submit_when_settled(
                guard, guard.control_loop.dispatch_joint_move, move, observation, deadline=deadline
            )

        wait_s = move_s + self.robot_config.home.arrival_timeout_s + self.WAIT_MARGIN_S
        snapshot_after = self.wait(robot.remaining_s(deadline, wait_s))
        if move is not None:
            if snapshot_after["motion"]["end_reason"] != MotionEndReason.FINISHED:
                raise RuntimeError("Home move was interrupted")
            self.completed_stages.append({"stage": HomeStage.HOMING.value, "after": snapshot_after})

        return snapshot_after

    @staticmethod
    def check_time(plan: HomePlan, deadline: float) -> None:
        """Raises unless the approved ``plan`` still has the time it estimates before ``deadline``."""
        if time.monotonic() + plan["operation_seconds_estimate"] >= deadline:
            raise TimeoutError("Home approved too late to finish before its timeout; nothing sent, confirm again")
