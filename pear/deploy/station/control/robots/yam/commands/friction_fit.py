from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class FrictionFit:
    """The feedforward of one YAM arm, fitted from its friction sweeps at several poses and speeds: the gravity model's
    factor of each joint, its Coulomb friction and the part of the friction that grows with the gravity load."""

    JOINT_COUNT = 6
    FRICTION_FRACTION = 0.85  # of the measured friction that the suggestion feeds forward; gravity goes in full
    GRAVITY_MIN_NM = 1.0  # a pose whose model gravity on a joint is smaller tells nothing of that joint's factor
    MIN_LOAD_SPREAD_NM = 2.0  # the least spread of the gravity load across poses that a load term is fitted over
    # what the arm's settings accept (yam/config.py): a suggestion outside is clipped, with a warning
    FACTOR_RANGE = (0.5, 1.5)
    FRICTION_RANGE_NM = (0.0, 3.0)
    LOAD_RATIO_RANGE = (0.0, 0.5)
    DECIMALS = 3  # of the suggested values

    @staticmethod
    def coulomb_friction(speeds: Sequence[float], frictions_nm: Sequence[float]) -> float:
        """Returns the friction at zero speed: the intercept of a line through the frictions measured at ``speeds``,
        kept between zero and the reading at the slowest speed."""
        speeds, frictions = np.asarray(speeds, dtype=float), np.asarray(frictions_nm, dtype=float)
        slowest_reading = max(0.0, float(frictions[np.argmin(speeds)]))
        if len(np.unique(speeds)) < 2:
            return slowest_reading

        intercept = float(np.polyfit(speeds, frictions, 1)[1])
        return min(max(intercept, 0.0), slowest_reading)

    @classmethod
    def gravity_factor(
        cls, raw_gravity_nm: Sequence[float], residual_nm: Sequence[float], applied_factor: float
    ) -> float:
        """Returns the factor of the raw model gravity that the measured gravity torques fit best, by least squares
        over the poses whose model gravity is at least GRAVITY_MIN_NM; ``applied_factor`` without such a pose."""
        raw_gravity, residual = np.asarray(raw_gravity_nm, dtype=float), np.asarray(residual_nm, dtype=float)
        is_loaded = np.abs(raw_gravity) >= cls.GRAVITY_MIN_NM
        if not is_loaded.any():
            return float(applied_factor)

        gravity = raw_gravity[is_loaded]
        measured_gravity = applied_factor * gravity + residual[is_loaded]  # what the joint needed against gravity
        return float(gravity @ measured_gravity / (gravity @ gravity))

    @classmethod
    def load_friction(cls, loads_nm: Sequence[float], frictions_nm: Sequence[float]) -> tuple[float, float]:
        """Returns the constant part of the friction and its growth per N*m of gravity load, fitted through the
        poses' ``loads_nm`` and Coulomb ``frictions_nm``; a constant alone when the loads spread too little."""
        loads, frictions = np.asarray(loads_nm, dtype=float), np.asarray(frictions_nm, dtype=float)
        if len(loads) < 2 or np.ptp(loads) < cls.MIN_LOAD_SPREAD_NM:
            return float(frictions.mean()), 0.0

        ratio = max(0.0, float(np.polyfit(loads, frictions, 1)[0]))
        return max(0.0, float(np.mean(frictions - ratio * loads))), ratio

    @classmethod
    def fit(
        cls,
        results: Sequence[Mapping[str, Any]],
        model_gravity_nm: Mapping[str, Sequence[float]],
        applied_factors: Sequence[float],
    ) -> dict[str, Any]:
        """Returns, for each joint, what its sweeps' ``results`` (each with its pose; a result with an error is left
        out) give at the poses' raw ``model_gravity_nm`` and the ``applied_factors``, the feedforward they suggest
        and the warnings of joints left out or values clipped."""
        joints, warnings = {}, []
        suggested: dict[str, list[float]] = {
            "gravity_compensation_factors": [],
            "friction_feedforward_nm": [],
            "friction_load_ratio": [],
        }

        for joint in range(cls.JOINT_COUNT):
            name = f"J{joint + 1}"
            usable = [result for result in results if result["joint"] == joint + 1 and "error" not in result]
            if not usable:
                warnings.append(f"{name}: no usable sweep; its friction is not fed forward")
                suggested["gravity_compensation_factors"].append(round(float(applied_factors[joint]), cls.DECIMALS))
                suggested["friction_feedforward_nm"].append(0.0)
                suggested["friction_load_ratio"].append(0.0)
                continue

            poses = list(dict.fromkeys(result["pose"] for result in usable))
            by_pose = {pose: [result for result in usable if result["pose"] == pose] for pose in poses}
            raw_gravity = [float(model_gravity_nm[pose][joint]) for pose in poses]
            residuals = [float(np.median([r["gravity_model_error_nm"] for r in by_pose[pose]])) for pose in poses]
            factor = cls.gravity_factor(raw_gravity, residuals, float(applied_factors[joint]))
            coulomb = {
                pose: cls.coulomb_friction(
                    [r["speed"] for r in by_pose[pose]], [r["friction_nm"] for r in by_pose[pose]]
                )
                for pose in poses
            }
            # the load the station feeds forward against: the new factor times the raw model's gravity
            constant, ratio = cls.load_friction([abs(factor * g) for g in raw_gravity], list(coulomb.values()))
            joints[name] = {
                "coulomb_nm_by_pose": coulomb,
                "gravity_factor": factor,
                "friction_constant_nm": constant,
                "friction_load_ratio": ratio,
            }

            for key, value, value_range in (
                ("gravity_compensation_factors", factor, cls.FACTOR_RANGE),
                ("friction_feedforward_nm", cls.FRICTION_FRACTION * constant, cls.FRICTION_RANGE_NM),
                ("friction_load_ratio", cls.FRICTION_FRACTION * ratio, cls.LOAD_RATIO_RANGE),
            ):
                clipped = min(max(value, value_range[0]), value_range[1])
                if clipped != value:
                    warnings.append(
                        f"{name}: {key} {value:.3g} is outside [{value_range[0]:g}, {value_range[1]:g}]; "
                        f"clipped to {clipped:g}"
                    )
                suggested[key].append(round(float(clipped), cls.DECIMALS))

        return {"joints": joints, "suggested": suggested, "warnings": warnings}
