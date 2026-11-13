import copy
import math
import time
from collections import deque
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.layout import JointLayout


class StabilityError(RuntimeError):
    """Raised when measured settling failed; it carries the stability window's metrics."""

    def __init__(self, reason: str, metrics: dict[str, Any]) -> None:
        self.reason = reason
        self.metrics = copy.deepcopy(metrics)

        violation_text = (
            "; ".join(
                f"{JointLayout.axis_name(violation)} {violation['metric']}={violation['value']:.6f} >"
                f" {violation['limit']:.6f} {violation['units']}"
                for violation in metrics.get("violations", [])
            )
            or "insufficient fresh samples for a complete stability window"
        )
        super().__init__(
            f"{reason}: {violation_text} (window={metrics.get('window_seconds', 0):.3f}s, "
            f"samples={metrics.get('samples', 0)})"
        )

    @property
    def details(self) -> dict[str, Any]:
        """What a fault records about this failure."""
        return {
            "code": type(self).__name__,
            "reason": self.reason,
            "message": str(self),
            "metrics": copy.deepcopy(self.metrics),
        }


class WindowSample(NamedTuple):
    """One reading of a sample window."""

    sampled_monotonic_s: float  # the local monotonic time of the reading
    value: Any  # what the window's user keeps of the reading


class ArmReading(NamedTuple):
    """What a stability window keeps of one reading of an arm."""

    position: np.ndarray
    velocity: np.ndarray


class SampleWindow:
    """The recent readings of one arm that are new and unbroken: a repeated reading adds nothing, and a gap between
    two starts the window over."""

    MIN_SAMPLES = 3  # the fewest samples a complete window holds
    FRESH_FRACTION = 0.8  # of a window's control periods that must bring a new sample

    def __init__(self, control_config: ControlConfig, history_s: float, *, checks_stamp_order: bool = True) -> None:
        self.control_config = control_config
        self.history_s = history_s  # how far back of the latest sample the window keeps samples
        # whether a reading stamped no later than the last one also starts the window over, as a gap does
        self.checks_stamp_order = checks_stamp_order
        self.samples: deque[WindowSample] = deque()  # the oldest first
        self.last_identity: object = None  # of the reading added last

    def add(self, sampled_monotonic_s: float, sample_id: object, value: Any) -> bool:
        """Adds a reading unless it repeats the one added last, and returns whether it was new."""
        identity = sampled_monotonic_s if sample_id is None else sample_id
        if identity == self.last_identity:
            return False

        samples = self.samples
        if samples and (
            (self.checks_stamp_order and sampled_monotonic_s <= samples[-1].sampled_monotonic_s)
            or sampled_monotonic_s - samples[-1].sampled_monotonic_s > self.control_config.state_timeout_s
        ):
            samples.clear()
        self.last_identity = identity
        samples.append(WindowSample(sampled_monotonic_s, value))
        while len(samples) > 1 and samples[1].sampled_monotonic_s <= sampled_monotonic_s - self.history_s:
            samples.popleft()

        return True

    def clear(self) -> None:
        """Forgets every reading."""
        self.samples.clear()
        self.last_identity = None

    def recent(self, duration_s: float) -> list[WindowSample]:
        """Returns the samples of the last ``duration_s``, with the one before them that starts that span."""
        samples = list(self.samples)
        start_index = 0
        while (
            start_index + 1 < len(samples)
            and samples[start_index + 1].sampled_monotonic_s <= samples[-1].sampled_monotonic_s - duration_s
        ):
            start_index += 1

        return samples[start_index:]

    @classmethod
    def is_complete(cls, samples: Sequence[WindowSample], duration_s: float, rate_hz: float) -> bool:
        """Whether ``samples`` span ``duration_s`` with enough new samples for a control loop of ``rate_hz``."""
        elapsed_s = samples[-1].sampled_monotonic_s - samples[0].sampled_monotonic_s
        min_sample_n = max(cls.MIN_SAMPLES, math.ceil(duration_s * rate_hz * cls.FRESH_FRACTION))

        return elapsed_s >= duration_s and len(samples) >= min_sample_n


class StabilityWindow:
    """Measured settling: the mean signed velocity of each arm over its fresh feedback."""

    # the metrics, each by its key in the window's metrics and by the metric a violation names
    VELOCITY_MEAN = "velocity_mean"
    VELOCITY_RMS = "velocity_rms"
    POSITION_SPAN = "position_span"
    TARGET_ERROR_MAX = "target_error_max"
    RATE_SUFFIX = "/s"  # after the unit of a coordinate, the unit of its velocity

    def __init__(
        self,
        control_config: ControlConfig,
        robot_config: RobotConfig,
        *,
        duration_s: float,
        is_strict: bool,
        checks_velocity_rms: bool,
    ) -> None:
        self.control_config, self.robot_config = control_config, robot_config
        self.layout = robot_config.LAYOUT
        self.duration_s, self.is_strict = duration_s, is_strict
        self.checks_velocity_rms = checks_velocity_rms  # whether a strict window also bounds the velocity RMS
        home = robot_config.home
        # the history kept: enough for every window read from it, Home's own among them
        self._history_s = max(duration_s, control_config.settle_window_s, home.settle_window_s if home else 0)
        # arms report apart, so each keeps its own samples
        self._sample_windows = {arm: SampleWindow(control_config, self._history_s) for arm in self.layout.arms}
        self.metrics: dict[str, Any] = {}
        self._hold_unstable_since: float | None = None

    def check_hold(self, snapshot: dict[str, Any]) -> bool:
        """Adds ``snapshot`` during a HOLD and returns whether the window is stable; raises once the arms stay
        unstable past the settle timeout."""
        is_stable = self.add(snapshot)
        now = time.monotonic()

        if is_stable:
            self._hold_unstable_since = None
        elif self._hold_unstable_since is None:
            self._hold_unstable_since = now
        elif now - self._hold_unstable_since > self.control_config.settle_timeout_s:
            raise self.stability_error("Persistent motion during HOLD")

        return is_stable

    def add(self, snapshot: dict[str, Any]) -> bool:
        """Adds each arm's new sample of ``snapshot`` and returns whether the window is now stable."""
        now = time.monotonic()
        arm_feedback = snapshot["arms"]
        sample_times = {arm: arm_feedback[arm]["sampled_monotonic_s"] for arm in self.layout.arms}
        state_timeout_s = self.control_config.state_timeout_s
        if any(
            not math.isfinite(sample_time)
            or sample_time > now + ArmState.CLOCK_SLACK_S
            or now - sample_time > state_timeout_s
            for sample_time in sample_times.values()
        ):
            self.reset()
            raise RuntimeError("Stale/invalid stability feedback")

        # arms report independently: a repeated sample of one must neither count again nor hide another's new one
        coordinate_shape = (self.layout.coordinate_count,)
        for arm, sample_time in sample_times.items():
            feedback = arm_feedback[arm]
            position, velocity = np.array(feedback["position"]), np.array(feedback["velocity"])
            if (
                position.shape != coordinate_shape
                or velocity.shape != coordinate_shape
                or not np.isfinite([position, velocity]).all()
            ):
                raise ValueError("Invalid settling feedback")
            self._sample_windows[arm].add(sample_time, feedback.get("sample_id"), ArmReading(position, velocity))

        self.metrics = self.evaluate()

        return self.metrics["stable"]

    def evaluate(
        self,
        *,
        arms: Sequence[str] | None = None,
        duration_s: float | None = None,
        targets: dict[str, Sequence[float]] | None = None,
        position_tolerance: float | Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Returns the metrics of a window of the kept history, without sampling again: of ``arms`` over
        ``duration_s`` (by default every arm over the window's own), against ``targets`` too when Home gives them."""
        arms = self.layout.arms if arms is None else tuple(arms)
        duration_s = self.duration_s if duration_s is None else duration_s
        if not arms or set(arms) - set(self.layout.arms) or not 0 < duration_s <= self._history_s:
            raise ValueError("Invalid stability window selection")

        windows = [self._sample_windows[arm].recent(duration_s) for arm in arms]
        if not all(windows):
            return {"window_seconds": 0.0, "samples": 0, "stable": False}

        elapsed_s = min(window[-1].sampled_monotonic_s - window[0].sampled_monotonic_s for window in windows)
        sample_count = min(map(len, windows))
        metrics = {"window_seconds": elapsed_s, "samples": sample_count, "stable": False}
        control_config = self.control_config
        if not all(SampleWindow.is_complete(window, duration_s, control_config.rate_hz) for window in windows) or any(
            not ArmState.is_fresh(
                time.monotonic() - window[-1].sampled_monotonic_s,
                control_config.state_timeout_s,
                ArmState.CLOCK_SLACK_S,
            )
            for window in windows
        ):
            return metrics

        position_span = np.concatenate(
            [np.ptp([sample.value.position for sample in window], axis=0) for window in windows]
        )
        velocity_mean = np.concatenate(
            [abs(np.mean([sample.value.velocity for sample in window], axis=0)) for window in windows]
        )
        velocity_rms = np.concatenate(
            [
                np.sqrt(np.mean(np.array([sample.value.velocity for sample in window]) ** 2, axis=0))
                for window in windows
            ]
        )
        velocity_tolerance = self.robot_config.velocity_tolerance
        violations = self._violations(arms, self.VELOCITY_MEAN, velocity_mean, velocity_tolerance, self.RATE_SUFFIX)
        if self.is_strict:
            # position span and velocity RMS count only here: a recovery must not mistake alternating motion for a
            # stationary arm
            if self.checks_velocity_rms:
                violations += self._violations(
                    arms, self.VELOCITY_RMS, velocity_rms, velocity_tolerance, self.RATE_SUFFIX
                )
            violations += self._violations(
                arms, self.POSITION_SPAN, position_span, self.robot_config.position_tolerance, ""
            )

        target_error_maxima = {}
        if targets is not None:
            tolerance = np.broadcast_to(position_tolerance, (self.layout.coordinate_count,))
            for arm, window in zip(arms, windows):
                positions = np.array([sample.value.position for sample in window])
                target_errors = np.max(abs(positions - targets[arm]), axis=0)
                target_error_maxima[arm] = target_errors.tolist()
                violations += self._violations((arm,), self.TARGET_ERROR_MAX, target_errors, tolerance, "")

        metrics.update(
            {
                self.POSITION_SPAN: position_span.tolist(),
                self.VELOCITY_MEAN: velocity_mean.tolist(),
                self.VELOCITY_RMS: velocity_rms.tolist(),
                "violations": violations,
            }
        )
        if targets is not None:
            metrics[self.TARGET_ERROR_MAX] = target_error_maxima
        metrics["stable"] = not violations

        return metrics

    def reset(self) -> None:
        """Requires a new complete window, after the caller's readiness condition failed."""
        for sample_window in self._sample_windows.values():
            sample_window.clear()
        self.metrics = {}
        self._hold_unstable_since = None

    def relax(self, duration_s: float) -> None:
        """Makes a strict window a plain one of ``duration_s``, with the history it has."""
        self.duration_s, self.is_strict = duration_s, False

    def stability_error(self, reason: str) -> StabilityError:
        """Returns a StabilityError of ``reason`` that carries this window's metrics."""
        return StabilityError(reason, dict(self.metrics))

    def _violations(
        self, arms: Sequence[str], metric: str, values: np.ndarray, limit: float | np.ndarray, unit_suffix: str
    ) -> list[dict[str, Any]]:
        """Returns one violation for each value of the selected ``arms`` beyond its limit; ``unit_suffix`` follows the
        coordinate's unit."""
        coordinate_count = self.layout.coordinate_count
        limits = np.broadcast_to(limit, values.shape)

        return [
            {
                **self.layout.arm_axis_labels(arms[index // coordinate_count], index % coordinate_count),
                "metric": metric,
                "value": float(values[index]),
                "limit": float(limits[index]) if np.ndim(limit) else limit,
                "units": self.layout.axis_unit(index) + unit_suffix,
            }
            for index in np.flatnonzero(values > limits)
        ]
