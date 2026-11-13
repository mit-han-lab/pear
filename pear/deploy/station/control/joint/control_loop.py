from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig, MotionLimits, RobotConfig
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.joint.operations import (
    GripperOperation,
    JointMoveOperation,
    Operation,
    OperationInterrupted,
    SequenceOperation,
    TargetOperation,
)
from pear.deploy.station.control.joint.recovery import RecoveryPolicy, RecoveryRequest
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.stability import StabilityWindow
from pear.deploy.station.control.joint.tracking import Tracking
from pear.deploy.station.control.joint.trajectory import CommandSample, JointMove, JointTrajectory
from pear.deploy.station.control.schemas import (
    MOVING_CONTROL_STATES,
    ControlSnapshot,
    ControlState,
    MotionEndReason,
    Observation,
)
from pear.deploy.station.exceptions import OperatorPaused
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class Braking(Enum):
    """What a controlled STOP brakes with."""

    MOVE_CAPS = "move_caps"  # the caps of the motion it stops
    STATION_LIMITS = "station_limits"  # the station's STOP acceleration and jerk


class JointControlLoop:
    """One owner and one bounded trajectory for every arm of a joint-streaming robot, stepped at its control rate."""

    THREAD_NAME = "station-control"  # of the control thread
    RECOVERY_WINDOW_FLOOR_S = 0.5  # a recovery's stability window is never shorter
    START_EPSILON = 1e-9  # how closely a move must start at the held command
    STATIONARY_EPSILON = 1e-8  # planned speed below which the commanded state counts as at rest
    MOVE_MARGIN_S = 5.0  # past a move's duration and settle timeout before the move times out
    SEQUENCE_MARGIN_S = 30.0  # past a sequence's duration before it times out
    GRIPPER_TIMEOUT_S = 45.0  # for a gripper operation
    HOLD_WAIT_S = 30.0  # for a HOLD, where a maintenance command waits for one
    SHUTDOWN_WAIT_S = 10.0  # for the stop that precedes a shutdown
    THREAD_JOIN_S = 2.0  # for the control thread to end

    # what a robot's control loop names: its gripper operation (made with the two configurations, the held targets,
    # the arms' states and the gripper request), how it recovers from a fault and what its STOP brakes with
    GRIPPER_OPERATION_CLASS: type[GripperOperation]
    RECOVERY_POLICY_CLASS: type[RecoveryPolicy]
    BRAKING: Braking
    CHECKS_VELOCITY_RMS: bool  # whether a recovery's stability window also bounds the velocity RMS

    def __init__(self, driver: JointDriver, control_config: ControlConfig, robot_config: RobotConfig) -> None:
        self.driver = driver
        # the loop's own copies, which it and its parts run with: a caller's later change to its objects is not seen
        self._control_config, self._robot_config = copy.deepcopy(control_config), copy.deepcopy(robot_config)
        self.layout = self._robot_config.LAYOUT
        self.limits = self._robot_config.joint_limits()
        self.tracking = Tracking(self._robot_config, self.limits, self.layout)
        self.recovery_policy = self.RECOVERY_POLICY_CLASS()

        self.lock = threading.RLock()  # guards everything below; an ActionGuard and the robot take it too
        self.generation = 0  # advances with every operation, stop and fault
        self.trajectory = self.new_trajectory()
        self._stop_operation = TargetOperation(self.trajectory)
        self.operation: Operation | None = None
        # set by the operator to stop the running operation: its ActionGuard attaches it
        self.operation_cancellation: threading.Event | None = None
        self._operation_deadline: float | None = None
        self.status, self.fault, self.fault_details = ControlState.HOLD, None, None
        # how the last operation ended: a MotionEndReason, or the word a robot stopped with; None while one runs
        self._motion_end_reason: str | None = MotionEndReason.INITIAL_HOLD
        self._exit_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._settle_since: float | None = None
        self._settling_window = self._new_window(self._control_config.settle_window_s, is_strict=False)
        self._recovery_request: RecoveryRequest | None = None
        self._fault_hold_error: str | None = None
        self._fault_read_error: str | None = None  # of the last failed read in FAULT, which is logged once
        # per arm, the opening a gripper stopped at on an object
        self._blocked_gripper_openings: dict[str, float] | None = None

        self.states = self._await_settled_feedback()
        self._is_settled = bool(self._settling_window.metrics.get("stable", False))
        # the hold is exactly where the arms are: feedback may rest just past a limit against a mechanical stop,
        # and clipping it into the bounds would jump the arm at enable
        self.targets = {
            arm: tuple(float(position) for position in self.states[arm].position) for arm in self.layout.arms
        }
        self.trajectory.hold(self.layout.join(self.targets))
        self.driver.send(self.targets)

    @classmethod
    def open(cls, driver: JointDriver, control_config: ControlConfig, robot_config: RobotConfig) -> JointControlLoop:
        """Returns a started control loop on ``driver``, which is closed if the start fails."""
        try:
            control_loop = cls(driver, control_config, robot_config)
            control_loop.start()

            return control_loop
        except BaseException:
            driver.close()
            raise

    @property
    def control_config(self) -> ControlConfig:
        """A copy of the configuration the control loop runs with."""
        return copy.deepcopy(self._control_config)

    @property
    def robot_config(self) -> RobotConfig:
        """A copy of the robot's settings the control loop runs with."""
        return copy.deepcopy(self._robot_config)

    def start(self) -> None:
        """Starts the control thread."""
        if self._thread is not None:
            raise RuntimeError("Runtime already started")

        self._thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
        self._thread.start()

    def move(
        self, targets: dict[str, tuple[float, ...]], *, speed_scale: float = 1.0, caps: MotionLimits | None = None
    ) -> None:
        """Starts a maintenance move from the held targets to the absolute joint ``targets``, one for each arm,
        within ``caps``."""
        with self.lock:
            if set(targets) != set(self.layout.arms):
                raise ValueError(f"An atomic station target must include {' and '.join(self.layout.arms)}")

            move = JointMove.build(
                self.trajectory.input.current_position,
                self.layout.join(targets),
                self.limits,
                self.layout,
                speed_scale=speed_scale,
                caps=caps,
            )
            self.dispatch_joint_move(move)
            logger.info("move requested: %s", targets)

    def dispatch_joint_move(self, move: JointMove, observation: Observation | None = None) -> bool:
        """Checks and starts a checked joint move under one lock, and returns whether it started: not while the HOLD
        of ``observation`` is unsettled."""
        with self.lock:
            self.verify_reference(observation)
            if observation is not None and (not self._is_settled or self.status != ControlState.HOLD):
                return False
            if self.status != ControlState.HOLD:
                raise RuntimeError(f"Cannot start a move in {self.status}")
            if not isinstance(move, JointMove):
                raise TypeError("Expected a checked JointMove")
            trajectory_input = self.trajectory.input
            if not np.allclose(move.start, trajectory_input.current_position, atol=self.START_EPSILON, rtol=0):
                raise ValueError("Trajectory start no longer matches held commands")
            if max(abs(velocity) for velocity in trajectory_input.current_velocity) > self.STATIONARY_EPSILON:
                raise RuntimeError("Joint move requires stationary planned state")

            self.trajectory.head_for(move.goal, move.motion_caps)
            self._begin(
                JointMoveOperation(self.trajectory, move),
                move.duration_s + self._control_config.settle_timeout_s + self.MOVE_MARGIN_S,
            )

            return True

    def start_sequence(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
        *,
        rate_hz: float,
        observation: Observation | None = None,
    ) -> bool:
        """Starts streaming the absolute joint vectors ``positions`` at ``rate_hz``, and returns whether it started:
        not while the HOLD of ``observation`` is unsettled."""
        operation = SequenceOperation(self.trajectory, positions, rate_hz)

        with self.lock:
            self.verify_reference(observation)
            if self.status != ControlState.HOLD:
                raise RuntimeError("A finite sequence requires HOLD")
            if observation is not None and not self._is_settled:
                return False

            self._begin(operation, len(operation.positions) / operation.rate_hz + self.SEQUENCE_MARGIN_S)

            return True

    def begin_grippers(self, gripper_request: Any) -> GripperOperation:
        """Starts the robot's gripper operation for ``gripper_request`` and returns it; the held arm targets stay
        unchanged."""
        with self.lock:
            if self.status != ControlState.HOLD:
                raise RuntimeError("Another operation owns the targets")
            self.check_fresh(self.states)

            operation = self.GRIPPER_OPERATION_CLASS(
                self._control_config, self._robot_config, self.targets, self.states, gripper_request
            )
            # grippers advance their own profiles: only the shared command state and the caps a stop uses are
            # synchronized, rather than planning a move that changes nothing
            self.trajectory.head_for(self.layout.join(self.targets))
            self._blocked_gripper_openings = None
            self._begin(operation, self.GRIPPER_TIMEOUT_S)

            return operation

    def verify_reference(self, observation: Observation | None) -> None:
        """Raises unless the feedback is fresh and ``observation`` belongs to the current operation; the caller holds
        the lock."""
        self.check_fresh(self.states)
        SnapshotCheck.validate_reference(self._feedback_snapshot(), observation)

    def stop(self, *, reason: str = MotionEndReason.STOPPED) -> None:
        """Brakes the current operation into a HOLD, with ``reason`` as how the motion ended; in a HOLD, invalidates
        pending observations."""
        with self.lock:
            self._finish_recovery(OperatorPaused("Control recovery interrupted by the operator"))
            if self.status != ControlState.RUNNING:
                if self.status == ControlState.HOLD:
                    self.generation += 1
                return

            self._motion_end_reason = reason
            stops_from_standstill = self.operation.stops_from_standstill
            self.operation = None
            self.operation_cancellation = None
            self.generation += 1
            braking_caps = self.limits.braking_caps() if self.BRAKING is Braking.STATION_LIMITS else None
            try:
                self.trajectory.stop(from_standstill=stops_from_standstill, braking_caps=braking_caps)
            except Exception as error:
                self._fail(error)
                return

            margin_s = self._control_config.settle_timeout_s + self.MOVE_MARGIN_S
            self._operation_deadline = time.monotonic() + self.trajectory.duration_s + margin_s
            self._settle_since = None
            self._settling_window.reset()
            self._is_settled = False
            self.status = ControlState.STOPPING
            logger.info("controlled stop requested")

    def recover(
        self, *, cancellation: threading.Event | None = None, timeout_s: float | None = None
    ) -> ControlSnapshot:
        """Reacquires a stable hold after a fault, without enabling motors or replaying an action, and returns the
        snapshot then."""
        has_automatic_timeout = timeout_s is None
        if timeout_s is None:
            timeout_s = self.recovery_policy.default_timeout_s(self._control_config)
        SnapshotCheck.check_timeout(timeout_s, "Recovery timeout")
        OperatorPaused.check_cancelled(cancellation)

        with self.lock:
            if self.status != ControlState.FAULT:
                raise RuntimeError("Control recovery requires FAULT")
            if self._recovery_request is not None:
                raise RuntimeError("Control recovery is already running")
            if self._fault_hold_error is not None:
                raise RuntimeError(self.layout.texts.failed_hold + self._fault_hold_error)
            if self._exit_event.is_set() or (self._thread is not None and not self._thread.is_alive()):
                raise RuntimeError("Recovery blocked: control thread is not running")

            # a recovery always needs a new window, even if the fault's telemetry looks stable
            duration_s = max(self.RECOVERY_WINDOW_FLOOR_S, self._control_config.settle_window_s)
            request = RecoveryRequest(
                self._new_window(duration_s, is_strict=True),
                time.monotonic() + timeout_s,
                cancellation,
                has_automatic_timeout=has_automatic_timeout,
            )
            self._recovery_request = request

        try:
            # the request's own deadline, which a release moves, ends the wait
            return SnapshotCheck.poll(lambda: self._recovery_outcome(request), self._control_config.control_period_s)
        finally:
            with self.lock:
                if self._recovery_request is request:
                    self._recovery_request = None

    def step(self) -> None:
        """Runs one control period: reads, checks, and advances the operation or the recovery."""
        with self.lock:
            if self.status == ControlState.CLOSED:
                return
            if self.status == ControlState.FAULT:
                self._step_fault()
                return

            try:
                now = time.monotonic()
                states = self._read()
                # only a checked reading is published: a malformed one must not break status or STOP after a fault
                self.states = states
                feedback = self._feedback_snapshot()
                self.check_tracking(feedback)
                if self.status == ControlState.HOLD:
                    self._is_settled = self._settling_window.check_hold(feedback)
                    self.driver.refresh_hold(self.targets)
                if self.operation_cancellation is not None and self.operation_cancellation.is_set():
                    self.stop(reason=MotionEndReason.CANCELLED)
                if self.status in MOVING_CONTROL_STATES:
                    self._advance(now, feedback)
            except Exception as error:
                self._fail(error)

    def snapshot(self) -> ControlSnapshot:
        """Returns the control loop's state, feedback, arrival, settling, recovery and motion, measured now."""
        with self.lock:
            snapshot = self._feedback_snapshot()
            snapshot["fault_details"] = copy.deepcopy(snapshot["fault_details"])
            capture_time = snapshot["captured_monotonic_s"]
            state_timeout_s = self._control_config.state_timeout_s
            is_fresh = all(
                ArmState.is_fresh(capture_time - state.sampled_monotonic_s, state_timeout_s)
                for state in self.states.values()
            )
            recovery_request, end_reason = self._recovery_request, self._motion_end_reason

            snapshot["arrival"] = {
                **self.tracking.arrival(SnapshotCheck.state_vector(snapshot), self.requested_position()),
                "settled": self.status == ControlState.HOLD and self._is_settled and is_fresh,
            }
            snapshot["settling"] = copy.deepcopy(self._settling_window.metrics)
            snapshot["recovery"] = (
                {"phase": recovery_request.phase.value, "metrics": copy.deepcopy(recovery_request.window.metrics)}
                if recovery_request is not None
                else None
            )
            snapshot["motion"] = {
                **self.driver.telemetry(),
                # the plain word, also of a stop that a robot gave a word of its own
                "end_reason": None if end_reason is None else str(end_reason),
                "settling": self.status in MOVING_CONTROL_STATES and self._settle_since is not None,
            }

            return snapshot

    def requested_position(self) -> list[float]:
        """Returns the joint vector the current operation heads for, a gripper stopped on an object where it
        stopped; the caller holds the lock."""
        requested_position = list(self.trajectory.input.target_position)
        for arm, opening in (self._blocked_gripper_openings or {}).items():
            requested_position[self.layout.gripper_indices[self.layout.arms.index(arm)]] = opening

        return requested_position

    def stability(self, **criteria: Any) -> dict[str, Any]:
        """Returns the control loop's stability history evaluated with ``criteria``, without sampling again."""
        with self.lock:
            return self._settling_window.evaluate(**criteria)

    def wait_for_hold(
        self,
        timeout_s: float,
        *,
        accepts: Callable[[ControlSnapshot], bool] | None = None,
        guard: ActionGuard | None = None,
        label: str = "Motion",
    ) -> ControlSnapshot:
        """Waits for a settled HOLD that ``accepts`` accepts and returns its snapshot; any failure stops the arms."""
        # accepts may annotate the snapshot but never commands; even an unguarded caller is bound to the operation
        # seen on entry
        guard = ActionGuard(self) if guard is None else guard

        def read_snapshot() -> ControlSnapshot:
            with self.lock:
                guard.check()
                return SnapshotCheck.validate_snapshot(self.snapshot(), self._control_config)

        try:
            return SnapshotCheck.wait_ready(
                read_snapshot,
                timeout_s=timeout_s,
                control_period_s=self._control_config.control_period_s,
                guard=guard,
                make_timeout_error=lambda snapshot: TimeoutError(f"{label} did not settle before timeout: {snapshot}"),
                accepts=accepts,
            )
        except BaseException:
            self.stop()
            raise

    def shutdown(self) -> None:
        """Stops, waits for the HOLD if possible, and always releases the hardware."""
        try:
            self.stop()
            if self.status not in (ControlState.FAULT, ControlState.CLOSED):
                self.wait_for_hold(self.SHUTDOWN_WAIT_S, label="Stop")
        finally:
            self.close()

    def close(self) -> None:
        """Releases the hardware without waiting for motion; a live owner calls ``shutdown``."""
        self._exit_event.set()
        if self._thread:
            self._thread.join(timeout=self.THREAD_JOIN_S)
            if self._thread.is_alive():
                raise RuntimeError("Control thread did not stop; hardware not closed concurrently")

        with self.lock:
            self._finish_recovery(RuntimeError("Station closed during recovery"))
            if self.status != ControlState.CLOSED:
                try:
                    self.driver.close()
                finally:
                    self.status = ControlState.CLOSED
                    self.generation += 1

    def attach_cancellation(self, cancellation: threading.Event | None) -> None:
        """Has the operation that runs, if one does, stop once ``cancellation`` is set; the caller holds the lock."""
        if self.status == ControlState.RUNNING:
            self.operation_cancellation = cancellation

    def new_trajectory(self) -> JointTrajectory:
        """Returns a trajectory at this control loop's limits, layout and rate."""
        return JointTrajectory(self.limits, self.layout, self._control_config)

    def check_fresh(self, states: dict[str, ArmState]) -> None:
        """Raises unless every arm's reading is recent and finite."""
        now = time.monotonic()

        for arm in self.layout.arms:
            age_s = now - states[arm].sampled_monotonic_s
            if not ArmState.is_fresh(age_s, self._control_config.state_timeout_s, ArmState.CLOCK_SLACK_S):
                raise RuntimeError(self.layout.texts.stale_feedback.format(arm=arm, age=age_s))

    def check_states(self, states: dict[str, ArmState]) -> None:
        """Raises unless ``states`` is a complete, finite and fresh reading within the feedback bounds."""
        if set(states) != set(self.layout.arms):
            raise RuntimeError("Missing arm feedback")
        for state in states.values():
            state.check(self.layout)
        self.check_fresh(states)

        # encoder quantization and resting against a mechanical stop can put measured feedback slightly beyond a
        # bound; commands still use the exact bounds
        positions = self.layout.join({arm: state.position for arm, state in states.items()})
        self.limits.check_positions(positions, self.layout, self.limits.feedback_tolerance)

    def check_tracking(self, feedback: dict[str, Any]) -> None:
        """Raises a TrackingError if a measured joint of ``feedback`` is too far from its held target; the caller
        holds the lock."""
        self.tracking.check(SnapshotCheck.state_vector(feedback), SnapshotCheck.target_vector(feedback))

    def planned_velocities(self, sample: CommandSample) -> dict[str, tuple[float, ...]] | None:
        """Returns the planned velocity of ``sample`` for each arm that is fed it, zero for the rest, None if none
        is."""
        fed_arms = [arm for arm in self.layout.arms if self._robot_config.feeds_velocity(arm)]
        if not fed_arms:
            return None

        velocities = self.layout.split(sample.velocity)
        zeros = (0.0,) * self.layout.coordinate_count

        return {arm: tuple(velocities[arm]) if arm in fed_arms else zeros for arm in self.layout.arms}

    def transfer_hold(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Commands ``targets`` as the new hold during a recovery; a failed write blocks later ones; the caller holds
        the lock."""
        try:
            self.driver.send(targets, velocities)
        except Exception as error:
            self._fault_hold_error = str(error)
            raise RuntimeError(f"Recovery hold write failed: {error}") from error

        self.targets = targets

    def complete_recovery(self, request: RecoveryRequest) -> None:
        """Ends ``request`` in a HOLD at the current targets, with a new operation generation; the caller holds the
        lock."""
        OperatorPaused.check_cancelled(request.cancellation)

        # later actions start at the acknowledged hold, with zero derivatives
        self.trajectory = self.new_trajectory()
        self.trajectory.hold(self.layout.join(self.targets))
        self._stop_operation = TargetOperation(self.trajectory)
        self.operation = self.operation_cancellation = self._operation_deadline = None
        self._settle_since = None
        request.window.relax(self._control_config.settle_window_s)
        self._settling_window, self._is_settled = request.window, True
        self.generation += 1
        self.status, self.fault, self.fault_details = ControlState.HOLD, None, None
        self._motion_end_reason = MotionEndReason.RECOVERED
        self._finish_recovery()

    def _await_settled_feedback(self) -> dict[str, ArmState]:
        """Returns the arms' reading once the transient of enabling has settled; the hardware may engage its own hold
        first."""

        def read_settling() -> tuple[bool, dict[str, ArmState]]:
            states = self._read()
            return self._settling_window.add({"arms": self._arm_feedback(states, time.monotonic())}), states

        return SnapshotCheck.poll(
            read_settling,
            self._control_config.control_period_s,
            self._control_config.settle_timeout_s,
            lambda _: RuntimeError(self.layout.texts.unsettled_start),
        )

    def _run(self) -> None:
        """Steps the control loop once every control period until it is closed or a step fails."""
        control_period_s = self._control_config.control_period_s

        try:
            deadline = time.monotonic()
            while not self._exit_event.is_set():
                self.step()

                deadline += control_period_s
                now = time.monotonic()
                if deadline < now:
                    # never several control periods in a burst after a stall
                    deadline = now + control_period_s
                self._exit_event.wait(max(0, deadline - now))
        except BaseException:  # the thread ends with the state as it was; the failure is logged before it propagates
            logger.exception("The control thread failed")
            raise

    def _begin(self, operation: Operation, timeout_s: float) -> None:
        """Makes ``operation`` the running one, which times out after ``timeout_s``."""
        self.operation = operation
        self.operation_cancellation = None
        self._operation_deadline = time.monotonic() + timeout_s
        self._settle_since = None
        self._settling_window.reset()
        self._is_settled = False
        self.generation += 1
        self.status = ControlState.RUNNING
        self._motion_end_reason = None

    def _recovery_outcome(self, request: RecoveryRequest) -> tuple[bool, ControlSnapshot | None]:
        """Returns whether ``request`` is done, with the snapshot then; raises its error, the operator's pause or its
        timeout."""
        with self.lock:
            if request.is_done:
                if request.error is not None:
                    raise request.error
                return True, self.snapshot()

            OperatorPaused.check_cancelled(request.cancellation)
            request.check_deadline()

            return False, None

    def _step_fault(self) -> None:
        """Reads the arms in FAULT and, if a recovery was requested, advances it."""
        # only an explicit recovery request may change the hold target in FAULT; a failed read keeps the last
        # reading, whose age then shows it is stale
        request = self._recovery_request
        try:
            self.states = self._read(is_for_recovery=request is not None)
            if request is not None:
                self._check_recovery(request)
            self._fault_read_error = None
        except Exception as error:
            if request is None and str(error) != self._fault_read_error:
                logger.warning("feedback read failed in FAULT: %s", error, exc_info=error)
                self._fault_read_error = str(error)
            self._finish_recovery(error)

    def _check_recovery(self, request: RecoveryRequest) -> None:
        """Advances ``request`` by one control period with the recovery policy."""
        OperatorPaused.check_cancelled(request.cancellation)
        request.check_deadline()

        feedback = self._feedback_snapshot()
        self.recovery_policy.step(self, request, feedback, request.window.add(feedback))

    def _advance(self, now: float, feedback: dict[str, Any]) -> None:
        """Sends the next sample of the running or stopping operation, and enters HOLD once it has finished and
        settled."""
        if now > self._operation_deadline:
            if self.status == ControlState.STOPPING:
                raise TimeoutError("Controlled stop exceeded its deadline")
            self.stop(reason=MotionEndReason.ACTION_TIMEOUT)
            if self.status == ControlState.FAULT:
                return

        operation = self.operation or self._stop_operation
        try:
            sample = operation.advance(now, self.states, self.targets)
        except OperationInterrupted as error:
            self.stop(reason=error.reason)
            if self.status == ControlState.FAULT:
                return
            sample = self._stop_operation.advance(now, self.states, self.targets)

        self.limits.check_command(sample.position, self.layout.join(self.targets), self.layout)
        targets = {
            arm: tuple(float(position) for position in positions)
            for arm, positions in self.layout.split(sample.position).items()
        }
        self.driver.send(targets, self.planned_velocities(sample))
        self.targets = targets
        self.trajectory.accept(sample)
        if not sample.is_finished:
            return

        if self._settle_since is None:
            self._settle_since = now
            self._settling_window.reset()
        if self._settling_window.add(feedback):
            if isinstance(self.operation, GripperOperation):
                self._blocked_gripper_openings = self.operation.blocked_gripper_openings
            self.operation = None
            self.operation_cancellation = self._operation_deadline = None
            self._is_settled = True
            if self.status == ControlState.RUNNING:
                self._motion_end_reason = MotionEndReason.FINISHED
            self.status = ControlState.HOLD
            logger.info("motion finished at %s", self.targets)
        elif now - self._settle_since > self._control_config.settle_timeout_s:
            raise self._settling_window.stability_error("Motion remained unstable before timeout")

    def _fail(self, error: Exception) -> None:
        """Latches ``error`` as the fault and commands the targets the operation and the recovery policy hold then."""
        self._finish_recovery(error)
        if self.operation is not None:
            self.targets = self.operation.fault_targets(error, time.monotonic(), self.states, self.targets)
        self.targets = self.recovery_policy.fault_targets(self)

        self.operation = None
        self.generation += 1
        self.status, self.fault = ControlState.FAULT, str(error)
        self.fault_details = copy.deepcopy(getattr(error, "details", None))
        self._is_settled = False
        self._settling_window.reset()
        self._motion_end_reason = MotionEndReason.FAULT
        self._fault_read_error = None
        logger.error("station fault: %s", error, exc_info=error)

        try:
            self.driver.send(self.targets)
            self._fault_hold_error = None
        except Exception as hold_error:
            self._fault_hold_error = str(hold_error)
            logger.exception("Could not command a hold; a physical stop may be required")

    def _finish_recovery(self, error: Exception | None = None) -> None:
        """Ends the recovery request, if there is one, with ``error`` or without one."""
        if self._recovery_request is not None:
            self._recovery_request.error = error
            self._recovery_request.is_done = True
            self._recovery_request = None

    def _read(self, *, is_for_recovery: bool = False) -> dict[str, ArmState]:
        """Returns a checked reading of every arm; for a recovery, only after the hardware was found healthy."""
        states = self.driver.read_for_recovery() if is_for_recovery else self.driver.read()
        self.check_states(states)

        return states

    def _new_window(self, duration_s: float, *, is_strict: bool) -> StabilityWindow:
        """Returns a new stability window of ``duration_s`` over this control loop's arms."""
        return StabilityWindow(
            self._control_config,
            self._robot_config,
            duration_s=duration_s,
            is_strict=is_strict,
            checks_velocity_rms=self.CHECKS_VELOCITY_RMS,
        )

    def _feedback_snapshot(self) -> dict[str, Any]:
        """Returns the current control data, fresh on every call; the caller holds the lock."""
        capture_time = time.monotonic()

        return {
            "captured_monotonic_s": capture_time,
            "status": self.status.value,
            "fault": self.fault,
            "fault_details": self.fault_details,
            "generation": self.generation,
            "targets": {arm: self.targets[arm] for arm in self.layout.arms},
            "arms": self._arm_feedback(self.states, capture_time),
        }

    def _arm_feedback(self, states: dict[str, ArmState], now: float) -> dict[str, Any]:
        """Returns each arm's reading as a snapshot's feedback at the monotonic time ``now``."""
        return {arm: states[arm].feedback(now) for arm in self.layout.arms}
