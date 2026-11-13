import json
import time
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig

from config_files import ConfigFiles
from pear.deploy.station.control.motion.homing import HomeFailed
from pear.deploy.station.control.robot import ActionMode
from stand_in_robots import StandInSO101Robot

ZERO_HOME_OVERRIDE = "robot.home.position=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]"  # Home: every joint at zero, the gripper open
# what the tests of the arm change of the station of the tests besides its quick settings and its Home: the
# tolerances they were written for (Home's, an arrival's, the feedback's past a limit), and no feedforward
ARM_TEST_OVERRIDES = (
    "robot.home.joint_tolerance=[0.035, 0.035, 0.035, 0.035, 0.035]",
    "robot.position_tolerance=0.035",
    "robot.feedback_position_tolerance=0.05",
    "robot.feedforward=null",
)
ZERO = dict(x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0)  # a pose delta of the arm that changes nothing


class RobotActions:
    """The actions of the tests of the SO-101 as one robot."""

    @staticmethod
    def delta(width: float = 0.0, **arm: float) -> dict[str, Any]:
        """Returns a pose delta action with the given arm components and gripper width."""
        return {"type": "single_arm_delta", "arm": {**ZERO, **arm}, "gripper": {"width": width}}


@pytest.fixture
def so101_config() -> DictConfig:
    """The SO-101's station of the tests with its quick settings, Home at zero, the tolerances of the tests of the arm
    and no feedforward; it takes the place of the conftest's in this file."""
    return ConfigFiles.load("so101", *ConfigFiles.SO101_QUICK_OVERRIDES, ZERO_HOME_OVERRIDE, *ARM_TEST_OVERRIDES)


def test_pose_action_completes_and_reports_the_model_state(so101_robot):
    before = so101_robot.kinematics.pose(so101_robot.observe()["joint_positions"])

    result = so101_robot.execute(RobotActions.delta(x=-8.0, z=-6.0, pitch=0.6))

    assert (result.status, result.is_settled, result.target_reached) == ("completed", True, True)
    assert result.measured_state.arm.pitch == pytest.approx(before[3] + 0.6, abs=0.02)
    assert result.gripper_outcome is None


def test_the_gripper_closes_opens_and_stops_on_an_object(so101_robot):
    assert so101_robot.execute(RobotActions.delta(width=-1.0)).gripper_outcome == "closed"
    assert so101_robot.execute(RobotActions.delta(width=1.0)).gripper_outcome == "open"

    so101_robot.control_loop.driver.hardware.blocked_gripper_opening = 0.4
    blocked = so101_robot.execute(RobotActions.delta(width=-1.0))
    assert blocked.status == "completed"
    assert blocked.gripper_outcome == "stopped_before_closed"
    assert not blocked.target_reached
    # The jaw keeps squeezing toward closed; later moves are judged where it stopped.
    assert so101_robot.control_loop.targets["arm"][5] == 0.0

    lifted = so101_robot.execute(RobotActions.delta(z=2.0))
    assert lifted.target_reached and lifted.measured_position[5] == pytest.approx(0.4)


def test_unreachable_action_is_rejected_without_motion(so101_robot):
    sent = len(so101_robot.control_loop.driver.hardware.sent_targets)

    result = so101_robot.execute(RobotActions.delta(x=40.0))

    assert result.status == "rejected"
    assert "at most" in result.error["message"]
    assert len(so101_robot.control_loop.driver.hardware.sent_targets) == sent


def test_planning_past_the_deadline_is_refused(so101_robot):
    action = so101_robot.ACTION_ADAPTER.validate(RobotActions.delta(z=1.0))

    with pytest.raises(TimeoutError, match="total timeout"):
        so101_robot.plan(so101_robot.observe(), action, deadline=time.monotonic() - 1)


def test_home_requires_approval_and_opens_the_gripper_first(so101_robot):
    so101_robot.execute(RobotActions.delta(width=-1.0, x=-5.0))

    with pytest.raises(HomeFailed, match="cancelled"):
        so101_robot.home(approve_plan=lambda plan: False)

    stages, plans = [], []
    report = so101_robot.home(
        approve_plan=lambda plan: plans.append(plan) or True,
        report_progress=lambda p: stages.append(p["stage"]),
    )
    assert report["results"]["reached"] and report["results"]["gripper_open"]
    assert json.loads(json.dumps(plans[0]))["open_gripper"] is True  # the GUI journals it
    assert stages.index("opening") < stages.index("homing")
    assert np.allclose(so101_robot.observe()["joint_positions"], so101_robot.robot_config.home.position, atol=1e-6)


def test_joint_mode_runs_sequences_only(so101_config):
    with StandInSO101Robot(so101_config, action_mode=ActionMode.JOINT) as station:
        start = list(station.observe()["joint_positions"])
        rows = [start[:1] + [q] + start[2:] for q in np.linspace(0, 0.2, 5)]

        result = station.execute({"type": "joint_positions", "positions": rows, "rate_hz": 25})

        assert result.status == "completed"
        assert station.execute(RobotActions.delta(z=1.0)).status == "rejected"


def test_robots_share_one_loaded_kinematics_model(so101_config):
    with StandInSO101Robot(so101_config) as first, StandInSO101Robot(so101_config) as second:
        assert first.kinematics is second.kinematics
