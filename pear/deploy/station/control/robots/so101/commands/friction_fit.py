from typing import Any

import numpy as np

from pear.deploy.station.control.commands import StationFileSnippet
from pear.deploy.station.control.robots.so101.layout import JOINTS


class FrictionFit:
    """The SO-101's gravity compliance and friction goal offsets, fitted from its settled staircases both ways
    at poses with different gravity loads."""

    FRICTION_FRACTION = 0.85  # of the measured friction that the suggestion feeds forward
    # what the feedforward's settings accept (so101/config.py): a suggestion outside is clipped, with a warning
    SUGGESTED_RANGES = {
        "gravity_compliance": (0.0, 0.3),
        "friction_offset_rad": (0.0, 0.05),
        "friction_load_ratio": (0.0, 0.1),
    }
    MIN_LOAD_NM = 0.02  # the gravity torque under which a joint counts as load-free
    MIN_GRAVITY_SPREAD = 0.05  # the N*m between poses under which no load-dependent friction is fitted
    REPORT_DECIMALS = 4  # of the fitted values
    COMPLIANCE_DECIMALS = 3  # of the suggested gravity compliance

    @classmethod
    def fit(cls, results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[float]]]:
        """Returns, for each joint, the compliance fitted through the origin and the friction fitted against the
        gravity load by pose, and the feedforward they suggest."""
        report = {}
        for joint_index, joint_name in enumerate(JOINTS):
            pairs = cls._pairs(results, joint_index)
            if not pairs:
                continue

            # with g the gravity torque, c the compliance and f the friction band: e_up = -c*g - f, e_down = -c*g + f
            gravity = np.array([pair[1] for pair in pairs])
            mean_error = np.array([pair[2] for pair in pairs])
            friction = np.array([pair[3] for pair in pairs])
            is_loaded = np.abs(gravity).max() > cls.MIN_LOAD_NM
            compliance = float(-(mean_error @ gravity) / (gravity @ gravity)) if is_loaded else 0.0
            residual = mean_error + compliance * gravity

            pose_summaries = {}
            for pose in dict.fromkeys(pair[0] for pair in pairs):
                is_this_pose = np.array([pair[0] == pose for pair in pairs])
                pose_summaries[pose] = {
                    "abs_gravity": float(np.abs(gravity[is_this_pose]).mean()),
                    "friction": float(friction[is_this_pose].mean()),
                }

            friction_offset, ratio = float(friction.mean()), 0.0
            if len(pose_summaries) > 1:
                pose_gravity = np.array([summary["abs_gravity"] for summary in pose_summaries.values()])
                pose_friction = np.array([summary["friction"] for summary in pose_summaries.values()])
                if np.ptp(pose_gravity) > cls.MIN_GRAVITY_SPREAD:
                    ratio = max(0.0, float(np.polyfit(pose_gravity, pose_friction, 1)[0]))
                    friction_offset = max(0.0, float(np.mean(pose_friction - ratio * pose_gravity)))

            report[joint_name] = {
                "pairs": len(pairs),
                "compliance_rad_per_nm": round(compliance, cls.REPORT_DECIMALS),
                "gravity_residual_rad": {
                    "mean": round(float(residual.mean()), cls.REPORT_DECIMALS),
                    "max_abs": round(float(np.abs(residual).max()), cls.REPORT_DECIMALS),
                },
                "friction_rad": round(friction_offset, cls.REPORT_DECIMALS),
                "friction_load_ratio": round(ratio, cls.REPORT_DECIMALS),
                "friction_spread_rad": round(float(friction.std()), cls.REPORT_DECIMALS),
                "by_pose": {
                    pose: {key: round(number, cls.REPORT_DECIMALS) for key, number in summary.items()}
                    for pose, summary in pose_summaries.items()
                },
            }

        # under the keys of the feedforward section of the robot's settings
        suggested_feedforward = {
            "gravity_compliance": [
                round(max(0.0, report.get(joint_name, {}).get("compliance_rad_per_nm", 0.0)), cls.COMPLIANCE_DECIMALS)
                for joint_name in JOINTS
            ],
            "friction_offset_rad": [
                round(
                    cls.FRICTION_FRACTION * max(0.0, report.get(joint_name, {}).get("friction_rad", 0.0)),
                    cls.REPORT_DECIMALS,
                )
                for joint_name in JOINTS
            ],
            "friction_load_ratio": [
                round(
                    cls.FRICTION_FRACTION * report.get(joint_name, {}).get("friction_load_ratio", 0.0),
                    cls.REPORT_DECIMALS,
                )
                for joint_name in JOINTS
            ],
        }
        return report, suggested_feedforward

    @staticmethod
    def _pairs(results: list[dict[str, Any]], joint_index: int) -> list[tuple[str, float, float, float]]:
        """Returns, for each step of a joint's sweeps measured up and down, its pose, its gravity torque, the mean of
        its two errors and their half difference."""
        pairs = []
        for sweep in results:
            if sweep["joint"] != joint_index:
                continue
            rows_up = {row["index"]: row for row in sweep["rows"] if row["direction"] == "up"}
            rows_down = {row["index"]: row for row in sweep["rows"] if row["direction"] == "down"}
            for point_index in sorted(set(rows_up) & set(rows_down)):
                error_up, error_down = (
                    rows_up[point_index]["error"][joint_index],
                    rows_down[point_index]["error"][joint_index],
                )
                gravity = rows_up[point_index]["gravity"][joint_index]
                pairs.append((sweep["pose"], gravity, (error_up + error_down) / 2, (error_down - error_up) / 2))

        return pairs

    @classmethod
    def clamped_suggestion(cls, suggested: dict[str, list[float]]) -> tuple[dict[str, list[float]], list[str]]:
        """Returns the feedforward ``suggested`` clipped into what its settings accept, and a warning for each
        value clipped that names its joint and the value."""
        clamped, warnings = {}, []
        for key, (at_least, at_most) in cls.SUGGESTED_RANGES.items():
            clamped[key], clipped_indices = StationFileSnippet.clamped(suggested[key], at_least, at_most)
            warnings += [
                f"{JOINTS[index]}: {key} {suggested[key][index]:.4g} is outside [{at_least:g}, {at_most:g}]; "
                f"clipped to {clamped[key][index]:g}"
                for index in clipped_indices
            ]

        return clamped, warnings
