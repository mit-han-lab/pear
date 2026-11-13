from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from config_files import ConfigFiles
from control_rig import StandInClock
from pear.deploy.station.control.joint.action_guard import ActionGuard
from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.control_loop import JointControlLoop
from pear.deploy.station.control.joint.operations import GripperOperation
from pear.deploy.station.control.joint.snapshot_checks import SnapshotCheck
from pear.deploy.station.control.joint.stability import StabilityWindow
from pear.deploy.station.control.joint.tracking import Tracking
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.joint.control_loop import YAMControlLoop
from pear.deploy.station.control.robots.yam.layout import LAYOUT
from pear.deploy.station.control.robots.yam.motion.actions import CartesianDelta
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.motion.planning import YAMPlanning
from pear.deploy.station.control.schemas import ControlState
from stand_in_joint_driver import StandInYAMDriver


class ControlHelpers:
    """Moves of the YAM arms' held targets as a maintenance command makes them, and the check of a reading that a YAM
    control loop makes."""

    @staticmethod
    def jog(control_loop: JointControlLoop, arm: str, joint_number: int, delta_rad: float) -> None:
        """Moves joint ``joint_number``, counted from 1, of ``arm`` by ``delta_rad`` from the held targets."""
        with control_loop.lock:
            targets = {arm_name: list(target) for arm_name, target in control_loop.targets.items()}
            targets[arm][joint_number - 1] += delta_rad
            control_loop.move(targets)

    @staticmethod
    def grip(control_loop: JointControlLoop, arm: str, opening: float) -> None:
        """Moves the gripper of ``arm`` to ``opening`` from the held targets."""
        with control_loop.lock:
            targets = dict(control_loop.targets)
            targets[arm] = LAYOUT.with_gripper(targets[arm], opening)
            control_loop.move(targets)

    @staticmethod
    def check_states(
        states: dict[str, YAMArmState], control_config: ControlConfig, robot_config: YAMConfig, now: float
    ) -> None:
        """Checks a reading of both arms against the bounds and the freshness that a YAM control loop with these
        configurations requires at the monotonic time ``now``."""
        with pytest.MonkeyPatch.context() as monkeypatch:
            clock = StandInClock().install(monkeypatch)
            driver = StandInYAMDriver(control_config, robot_config)
            control_loop = YAMControlLoop(driver, control_config, robot_config)
            clock.now = now

            try:
                control_loop.check_states(states)
            finally:
                control_loop.close()


class StandInGripperContactDriver(StandInYAMDriver):
    """YAM's stand-in joints whose grippers stop on objects, with the effort and the force limiting the SDK reports of
    a gripper on its object."""

    OBJECT_EFFORT_NM = 1.03  # the effort a gripper reports while its object holds it at the contact opening
    FORCE_LIMITED_EFFORT_NM = 0.3  # the effort from which a reading reports the gripper force-limited
    FORCE_LIMIT_N = 50.0  # the SDK's gripper force limit a reading reports
    # the SDK's holding torque a reading reports: what its force limiter of the linear gripper asks for at
    # FORCE_LIMIT_N, its friction compensation of 0.3 N*m included
    TARGET_EFFORT_NM = FORCE_LIMIT_N * 0.096 / 6.57 + 0.3
    CONTACT_SLACK = 1e-9  # how far past its contact opening a gripper still rests on its object

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        super().__init__(control_config, robot_config)
        self.contacts: dict[str, float] = {}  # each gripper's contact opening, by arm: its object stops it there
        self.contact_effort_nm = self.OBJECT_EFFORT_NM  # the effort a gripper on its object reports
        self.lacks_effort = False  # whether the readings carry no effort at all
        # each command of the control loop, before an object stopped a gripper; the latest last
        self.commands: list[dict[str, tuple[float, ...]]] = []

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Keeps the command and sends it with each gripper on an object held at its contact opening."""
        self.commands.append({arm: tuple(target) for arm, target in targets.items()})

        contact_targets = dict(targets)
        for arm, contact_opening in self.contacts.items():
            target = contact_targets[arm]
            contact_targets[arm] = LAYOUT.with_gripper(target, max(LAYOUT.gripper_value(target), contact_opening))

        super().send(contact_targets, velocities)

    def read(self) -> dict[str, YAMArmState]:
        """Returns the readings with each gripper's effort, the SDK's force limiting and its holding torque."""
        states = super().read()

        for arm, state in states.items():
            gripper_opening = LAYOUT.gripper_value(state.position)
            is_on_object = arm in self.contacts and gripper_opening <= self.contacts[arm] + self.CONTACT_SLACK
            effort_nm = self.contact_effort_nm if is_on_object else 0.0
            states[arm] = dataclasses.replace(
                state,
                effort=() if self.lacks_effort else LAYOUT.with_gripper((0.0,) * LAYOUT.coordinate_count, effort_nm),
                sdk_gripper_force_limited=effort_nm >= self.FORCE_LIMITED_EFFORT_NM,
                sdk_gripper_force_limit_n=self.FORCE_LIMIT_N,
                sdk_gripper_target_effort_nm=self.TARGET_EFFORT_NM,
                sdk_gripper_target=gripper_opening,
            )

        return states


class GripperRig:
    """A YAM control loop, on the stand-in clock and stand-in joints whose grippers stop on objects, that a test steps
    by hand through a gripper operation."""

    START_OPENING = 1.0  # each gripper's opening at the start: open
    MAX_STEPS = 600  # the most control periods a gripper operation may take

    def __init__(
        self,
        clock: StandInClock,
        *,
        opening: float = START_OPENING,
        contacts: dict[str, float] | None = None,
        contact_effort_nm: float = StandInGripperContactDriver.OBJECT_EFFORT_NM,
    ) -> None:
        self.clock = clock
        self.control_config, self.robot_config = ConfigFiles.load_configs("yam")
        self.driver = StandInGripperContactDriver(self.control_config, self.robot_config)
        self.driver.positions = dict.fromkeys(LAYOUT.arms, (0.0,) * LAYOUT.joint_count + (opening,))
        self.driver.contacts = contacts or {}
        self.driver.contact_effort_nm = contact_effort_nm

        self.control_loop = YAMControlLoop(self.driver, self.control_config, self.robot_config)
        self.cancellation = threading.Event()  # the operator's pause of the operations the rig begins
        self.samples: list[list[list[float]]] = []  # the planned velocity and acceleration after each step

    def begin(self, openings: dict[str, bool]) -> GripperOperation:
        """Starts the gripper operation that opens or closes each gripper of ``openings``, as an action starts it."""
        guard = ActionGuard(self.control_loop, self.cancellation)

        return guard.submit(self.control_loop.begin_grippers, openings)

    def tick(self) -> None:
        """Steps the control loop one control period later and keeps its planned velocity and acceleration."""
        self.clock.now += self.control_loop.control_config.control_period_s
        self.control_loop.step()

        trajectory_input = self.control_loop.trajectory.input
        self.samples.append([list(trajectory_input.current_velocity), list(trajectory_input.current_acceleration)])

    def finish(self) -> None:
        """Steps until the operation ends, which must end in a hold."""
        for _ in range(self.MAX_STEPS):
            self.tick()
            if self.control_loop.status in (ControlState.HOLD, ControlState.FAULT):
                break

        assert self.control_loop.status == ControlState.HOLD, self.control_loop.fault

    def close(self) -> None:
        """Closes the control loop and its stand-in joints."""
        self.control_loop.close()


class GripperRigs:
    """The gripper rigs of a test, on its stand-in clock; as a context, what makes them, every rig closed at the end."""

    def __init__(self, clock: StandInClock) -> None:
        self.clock = clock
        self.rigs: list[GripperRig] = []

    def __enter__(self) -> GripperRigs:
        return self

    def make(self, **settings: Any) -> GripperRig:
        """Returns a new gripper rig with the keyword ``settings`` of a GripperRig."""
        rig = GripperRig(self.clock, **settings)
        self.rigs.append(rig)

        return rig

    def __exit__(self, *_: Any) -> None:
        for rig in self.rigs:
            rig.close()


class MotionHelpers:
    """Cartesian actions of the YAM arms, control snapshots written by hand, and the checks the motion tests share."""

    ARM = "right"  # the arm an action moves unless a test names another

    @staticmethod
    def action(
        *, side: str = ARM, xyz: Sequence[float] = (0.0, 0.0, 0.0), rotation: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> CartesianDelta:
        """Returns the Cartesian delta of the arm ``side``: ``xyz`` in cm and the roll, pitch and yaw ``rotation`` in
        rad."""
        return CartesianDelta(side=side, delta_position_cm=list(xyz), delta_rpy_rad=list(rotation))

    @staticmethod
    def snapshot(joint_positions: Sequence[float]) -> dict[str, Any]:
        """Returns a control snapshot, written by hand, of both arms holding at ``joint_positions``, at rest and
        fresh."""
        positions = np.asarray(joint_positions)
        width = LAYOUT.coordinate_count
        arm_slices = [(arm, slice(index * width, (index + 1) * width)) for index, arm in enumerate(LAYOUT.arms)]

        return {
            "status": ControlState.HOLD.value,
            "fault": None,
            "generation": 0,
            "targets": {arm: positions[arm_slice].tolist() for arm, arm_slice in arm_slices},
            "arms": {
                arm: {
                    "position": positions[arm_slice].tolist(),
                    "velocity": [0.0] * width,
                    "age_seconds": 0.0,
                    "effort": [0.0] * width,
                }
                for arm, arm_slice in arm_slices
            },
        }

    @staticmethod
    def observation(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Returns an observation, of no session, that refers to ``snapshot``."""
        return {
            "session_id": None,
            "joint_positions": SnapshotCheck.state_vector(snapshot),
            "control_snapshot": snapshot,
        }

    @staticmethod
    def stamp(snapshot: dict[str, Any], now: float) -> dict[str, Any]:
        """Returns ``snapshot`` with each arm's sampling time: ``now`` less the arm's age."""
        for arm_feedback in snapshot["arms"].values():
            arm_feedback["sampled_monotonic_s"] = now - arm_feedback.get("age_seconds", 0)

        return snapshot

    @staticmethod
    def planned_target(kinematics: YAMKinematics, joint_positions: np.ndarray, action: StrictModel) -> np.ndarray:
        """Returns the joint vector that ``action`` plans from a hold at ``joint_positions``, checked as an action's
        planning checks it, with the offline kinematics."""
        observation = {"joint_positions": joint_positions, "control_snapshot": MotionHelpers.snapshot(joint_positions)}
        prepared_action = YAMPlanning.checked_action(kinematics.robot_config, kinematics, observation, action)

        return joint_positions.copy() if prepared_action.move is None else prepared_action.move.goal

    @staticmethod
    def tracking(robot_config: YAMConfig) -> Tracking:
        """Returns the tracking and the arrival checks of a YAM control loop with ``robot_config``."""
        return Tracking(robot_config, robot_config.joint_limits(), LAYOUT)

    @staticmethod
    def vectors(snapshot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Returns the measured positions and the held targets of a control snapshot."""
        return SnapshotCheck.state_vector(snapshot), SnapshotCheck.target_vector(snapshot)


class ReplayControlLoop:
    """The control loop of a replay: it holds, and each snapshot is the next a test's function gives, stamped now,
    with the arrival and the stability that a YAM control loop judges of it."""

    def __init__(
        self, control_config: ControlConfig, robot_config: YAMConfig, read_snapshot: Callable[[], dict[str, Any]]
    ) -> None:
        self.control_config, self.robot_config = control_config, robot_config
        self.read_snapshot = read_snapshot  # gives the next snapshot to replay
        self.lock = threading.RLock()  # what a guard of a wait takes
        # the replay holds and starts no operation: a replayed fault is found where a wait reads the snapshot
        self.status, self.fault, self.generation = ControlState.HOLD, None, 0
        self.window = StabilityWindow(
            control_config,
            robot_config,
            duration_s=control_config.settle_window_s,
            is_strict=False,
            checks_velocity_rms=True,
        )

    def snapshot(self) -> dict[str, Any]:
        """Returns the next snapshot, stamped now, with its arrival and the window's settling."""
        snapshot = MotionHelpers.stamp(self.read_snapshot(), time.monotonic())
        is_stable = (
            snapshot["status"] == ControlState.HOLD and not snapshot["fault"] and self.window.check_hold(snapshot)
        )

        arrival = MotionHelpers.tracking(self.robot_config).arrival(*MotionHelpers.vectors(snapshot))
        snapshot["arrival"] = {**arrival, "settled": is_stable}
        snapshot["settling"] = dict(self.window.metrics)

        return snapshot


class ReplayRobot:
    """A YAM robot's waits and checks of a hold, the robot's own code, over a replay of the snapshots a test's function
    gives."""

    session_id = None  # the replay's session: none, as an observation written by hand has none
    FOREIGN_OBSERVATION = BaseRobot.FOREIGN_OBSERVATION
    is_holding = BaseRobot.is_holding
    hold_snapshot = BaseRobot.hold_snapshot
    remaining_s = staticmethod(BaseRobot.remaining_s)
    wait_for_settled_hold = BaseRobot.wait_for_settled_hold
    wait_stationary = BaseRobot.wait_stationary
    control_period_s = BaseRobot.control_period_s  # the SOURCE's replay station had its interval

    def __init__(
        self,
        control_config: ControlConfig,
        robot_config: YAMConfig,
        read_snapshot: Callable[[], dict[str, Any]],
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self.control_config, self.robot_config = control_config, robot_config
        # what the robot's borrowed waits and checks read: the same objects, as a test edits them
        self._control_config, self._robot_config = control_config, robot_config
        self.control_loop = ReplayControlLoop(control_config, robot_config, read_snapshot)
        self.on_stop = on_stop  # called at every stop of the robot; None: nothing is

    def stop(self) -> None:
        """Calls ``on_stop``: the replay holds already."""
        if self.on_stop is not None:
            self.on_stop()
