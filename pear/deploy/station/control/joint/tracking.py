from collections.abc import Sequence
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.config import RobotConfig
from pear.deploy.station.control.joint.layout import JointLayout
from pear.deploy.station.control.joint.limits import JointLimits


class TrackingError(RuntimeError):
    """Raised for measured joints too far from their held targets: a hard fault from the actual feedback."""

    def __init__(self, metrics: dict[str, Any], layout: JointLayout) -> None:
        details = "; ".join(
            f"{layout.axis_name(violation)} error={violation['error']:.6f} > {violation['limit']:.6f}"
            f" {violation['units']}"
            for violation in metrics["violations"]
        )
        super().__init__(layout.texts.tracking_error_text.format(details=details))


class Tracking:
    """Measured joints against their targets: hard tracking faults and arrival within tolerance."""

    def __init__(self, robot_config: RobotConfig, limits: JointLimits, layout: JointLayout) -> None:
        self.layout = layout
        arm_tolerance = [robot_config.position_tolerance] * layout.joint_count + [robot_config.gripper_tolerance]
        self.arrival_tolerance = np.array(arm_tolerance * len(layout.arms))
        self.fault_bounds = np.array(limits.tracking_error, dtype=float)
        # a gripper stopped on an object legitimately stays short of its target
        self.fault_bounds[list(layout.gripper_indices)] = np.inf

    def check(self, measured_positions: Sequence[float], targets: Sequence[float]) -> None:
        """Raises a TrackingError if a joint is further from its target than its tracking bound."""
        if np.any(abs(np.asarray(measured_positions) - np.asarray(targets)) > self.fault_bounds):
            raise TrackingError(self.metrics(measured_positions, targets, self.fault_bounds), self.layout)

    def arrival(self, measured_positions: Sequence[float], requested_positions: Sequence[float]) -> dict[str, Any]:
        """Returns whether the measured joints reached the requested ones, independently of settling and faults."""
        tracking_metrics = self.metrics(measured_positions, requested_positions, self.arrival_tolerance)

        return {
            "reached": not tracking_metrics["violations"],
            "signed_error": tracking_metrics["signed_error"],
            "tolerance": self.arrival_tolerance.tolist(),
        }

    def metrics(
        self, measured_positions: Sequence[float], reference: Sequence[float], tolerance: np.ndarray
    ) -> dict[str, Any]:
        """Returns the signed errors of the measured joints against ``reference`` and the joints beyond
        ``tolerance``."""
        errors = np.asarray(measured_positions, dtype=float) - np.asarray(reference, dtype=float)
        coordinate_count = self.layout.coordinate_count

        violations = [
            {
                **self.layout.arm_axis_labels(self.layout.arms[index // coordinate_count], index % coordinate_count),
                "error": float(abs(errors[index])),
                "limit": float(tolerance[index]),
                "units": self.layout.axis_unit(index),
            }
            for index in np.flatnonzero(~(abs(errors) <= tolerance))
        ]

        return {"signed_error": errors.tolist(), "violations": violations}
