import pytest

from pear.deploy.station.control.commands import StationFileSnippet
from pear.deploy.station.control.robots.yam.commands.friction_fit import FrictionFit

SPEEDS = [0.05, 0.1, 0.2]  # of the sweeps, rad/s
POSE_GRAVITY = {"lifted": -5.0, "low": -11.0}  # J2's raw model gravity at each pose, N*m


def synthetic_results(
    c0: float = 0.4, ratio: float = 0.2, k: float = 1.25, applied: float = 1.1, viscous: float = 2.0
) -> tuple[list[dict], dict[str, list[float]], list[float]]:
    """Returns sweeps of J2 at both poses whose friction is c0 + ratio * |k * g| + viscous * speed and whose gravity
    model is off by (k - applied) * g, with the model's gravity of each pose and the factors applied in the sweeps."""
    results = [
        {
            "pose": pose,
            "joint": 2,
            "speed": speed,
            "friction_nm": c0 + ratio * abs(k * g) + viscous * speed,
            "gravity_model_error_nm": (k - applied) * g,
        }
        for pose, g in POSE_GRAVITY.items()
        for speed in SPEEDS
    ]
    gravity = {pose: [0.0, g, 0.0, 0.0, 0.0, 0.0] for pose, g in POSE_GRAVITY.items()}

    return results, gravity, [1.0, applied, 1.1, 1.2, 1.0, 1.0]


def test_coulomb_friction_is_the_zero_speed_intercept_within_the_slowest_reading():
    assert FrictionFit.coulomb_friction(SPEEDS, [1.96, 2.33, 3.21]) == pytest.approx(1.52, abs=0.02)
    assert FrictionFit.coulomb_friction(SPEEDS, [2.04, 1.97, 2.01]) == pytest.approx(2.02, abs=0.01)
    assert FrictionFit.coulomb_friction(SPEEDS, [1.0, 0.5, 0.2]) == 1.0  # above the slowest reading: clipped
    assert FrictionFit.coulomb_friction(SPEEDS, [0.1, 0.5, 1.3]) == 0.0  # a negative intercept: clipped
    assert FrictionFit.coulomb_friction([0.05], [0.7]) == 0.7


def test_gravity_factor_combines_the_poses_by_least_squares():
    gravity = [-5.0, -11.0]

    assert FrictionFit.gravity_factor(gravity, [(1.25 - 1.1) * g for g in gravity], 1.1) == pytest.approx(1.25)
    assert FrictionFit.gravity_factor([0.5], [0.3], 1.1) == 1.1  # under 1 N*m: the applied factor


def test_recalibrating_from_other_applied_factors_gives_the_same_factor():
    gravity = [-5.0, -11.0]

    assert FrictionFit.gravity_factor(gravity, [(1.25 - 1.2) * g for g in gravity], 1.2) == pytest.approx(1.25)


def test_load_friction_fits_a_line_across_poses_with_enough_spread():
    assert FrictionFit.load_friction([6.1, 13.3], [1.56, 3.02]) == pytest.approx((0.323, 0.2028), abs=1e-3)
    assert FrictionFit.load_friction([6.0, 7.0], [1.0, 1.2]) == pytest.approx((1.1, 0.0))


def test_fit_suggests_085_of_the_friction_and_the_full_factor():
    fit = FrictionFit.fit(*synthetic_results())

    suggested = fit["suggested"]
    assert suggested["gravity_compensation_factors"][1] == pytest.approx(1.25, abs=1e-3)
    assert suggested["friction_feedforward_nm"][1] == pytest.approx(0.85 * 0.4, abs=2e-3)
    assert suggested["friction_load_ratio"][1] == pytest.approx(0.85 * 0.2, abs=2e-3)
    assert not any(warning.startswith("J2") for warning in fit["warnings"])


def test_fit_skips_sweeps_without_paired_samples():
    results, gravity, applied = synthetic_results()
    missing = "no paired constant-speed samples"
    broken = [
        *results,
        {"pose": "low", "joint": 2, "speed": 0.2, "error": missing},
        {"pose": "lifted", "joint": 5, "speed": 0.05, "error": missing},
    ]

    fit = FrictionFit.fit(broken, gravity, applied)

    assert fit["suggested"] == FrictionFit.fit(results, gravity, applied)["suggested"]
    assert fit["suggested"]["friction_feedforward_nm"][4] == 0.0
    assert fit["suggested"]["gravity_compensation_factors"][4] == 1.0  # the applied factor stays
    assert any(warning.startswith("J5: no usable sweep") for warning in fit["warnings"])


def test_fit_clamps_what_the_config_refuses_and_warns():
    fit = FrictionFit.fit(*synthetic_results(c0=4.0, ratio=0.0))

    assert fit["suggested"]["friction_feedforward_nm"][1] == 3.0
    assert any("J2" in warning and "3.4" in warning for warning in fit["warnings"])


def test_station_file_snippet_indents_values_under_their_parent():
    text = StationFileSnippet.text(["under robot.arms.left"], {"x": [1.0, 2.5]}, 6)

    assert text == "# under robot.arms.left\n      x: [1.0, 2.5]\n"
    assert StationFileSnippet.clamped([-0.1, 0.2, 4.0], 0.0, 3.0) == ([0.0, 0.2, 3.0], [0, 2])
