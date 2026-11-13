import pytest

from pear.deploy.station.control.robots.so101.commands.friction_fit import FrictionFit


def test_fit_recovers_compliance_and_feeds_085_of_both_friction_terms_forward():
    c, f0, ratio = 0.08, 0.01, 0.02  # shoulder_lift
    results = []
    for pose, gravity in (("home", -0.1), ("reach", -0.42)):
        rows = []
        for i in range(1, 4):
            g = gravity * (1 + 0.05 * i)
            f = f0 + ratio * abs(g)
            for direction, sign in (("up", -1), ("down", 1)):
                error = [0.0, -c * g + sign * f, 0.0, 0.0, 0.0]
                rows.append(
                    {
                        "index": i,
                        "direction": direction,
                        "error": error,
                        "gravity": [0.0, g, 0.0, 0.0, 0.0],
                    }
                )
        results.append({"pose": pose, "joint": 1, "rows": rows})

    report, suggested = FrictionFit.fit(results)

    lift = report["shoulder_lift"]
    assert lift["compliance_rad_per_nm"] == pytest.approx(c, abs=1e-4)
    assert lift["friction_load_ratio"] == pytest.approx(ratio, abs=1e-4)
    assert lift["friction_rad"] == pytest.approx(f0, abs=1e-4)
    assert lift["gravity_residual_rad"]["max_abs"] == pytest.approx(0.0, abs=1e-4)
    assert suggested["gravity_compliance"] == pytest.approx([0.0, c, 0.0, 0.0, 0.0], abs=1e-3)
    assert suggested["friction_offset_rad"] == pytest.approx([0.0, 0.85 * f0, 0.0, 0.0, 0.0], abs=1e-4)
    assert suggested["friction_load_ratio"] == pytest.approx([0.0, 0.85 * ratio, 0.0, 0.0, 0.0], abs=1e-4)


def test_suggestion_clamps_outside_values_and_warns_only_for_affected_joints():
    suggested, warnings = FrictionFit.clamped_suggestion(
        {
            "gravity_compliance": [-0.01, 0.4, 0.3, 0.08, 0.0],
            "friction_offset_rad": [0.0, 0.06, 0.05, 0.01, 0.0],
            "friction_load_ratio": [0.0, 0.12, 0.1, 0.02, 0.0],
        }
    )

    assert suggested == {
        "gravity_compliance": [0.0, 0.3, 0.3, 0.08, 0.0],
        "friction_offset_rad": [0.0, 0.05, 0.05, 0.01, 0.0],
        "friction_load_ratio": [0.0, 0.1, 0.1, 0.02, 0.0],
    }
    assert len(warnings) == 4
    assert {warning.split(":", 1)[0] for warning in warnings} == {"shoulder_pan", "shoulder_lift"}
    for key in suggested:
        assert any(f"shoulder_lift: {key}" in warning and "clipped to" in warning for warning in warnings)
