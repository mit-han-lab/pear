import logging
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.joint.feedforward import FrictionFeedforward
from pear.deploy.station.control.robots.yam.hardware import sdk_adapter
from pear.deploy.station.control.robots.yam.hardware.arms import YAMHardware
from yam_rigs import ControlHelpers

CONTROL_PERIOD_S = 0.004  # of the SDK control thread
DISABLE_TIMEOUT_S = 0.02  # send timeout of the motor disable frame


class SDKRobotRig:
    """Makes the SDK robot of an adapted rig ready to take position and velocity commands."""

    LIMIT = [1, 1, 1, 1, 1, 1, 0.3]  # per-joint velocity cap of the command path

    @staticmethod
    def robot_for(adapter: SimpleNamespace, side: str) -> tuple[object, list[float]]:
        """Returns the SDK robot of the arm ``side`` with a command path, and its gripper's limits."""
        from i2rt.robots.utils import JointMapper

        robot = adapter.create(side)
        robot.motor_chain = [None] * 7
        limits = adapter.config.arms[side].gripper_limits
        robot.remapper = JointMapper({6: limits}, 7)
        robot._clip_robot_joint_pos_command = lambda q: q
        robot._command_lock = threading.Lock()
        robot._kp, robot._kd = np.ones(7) * 20, np.ones(7) * 0.5

        return robot, limits


class StartupMotors:
    """The SDK's interface of one motor, adapted, whose requests a test answers."""

    @staticmethod
    def startup_motor(sdk: tuple, statuses: list[int]) -> tuple[object, list[int]]:
        """Returns a motor interface whose replies carry ``statuses`` in turn, and the commands it sends."""
        import can

        dm_driver, _ = sdk
        sdk_adapter.SDKCompatibility(ConfigFiles.load_configs("yam")[1]).install()
        motor = dm_driver.DMSingleMotorCanInterface.__new__(dm_driver.DMSingleMotorCanInterface)
        sent = []
        replies = iter(statuses)

        def request(_id, motor_id, data, **kwargs):
            sent.append(data[-1])
            return can.Message(
                arbitration_id=motor_id + 16,
                data=[next(replies) << 4 | motor_id, 0x80, 0, 0x80, 0, 0, 25, 23],
                is_extended_id=False,
            )

        motor.channel = "can_follower_r"
        motor.name = "startup-test"
        motor.receive_mode = dm_driver.ReceiveMode.p16
        motor.bus = SimpleNamespace(send=lambda message: sent.append(message.data[-1]), channel_info=motor.channel)
        motor._send_message_get_response = request
        motor.try_receive_message = lambda *args, **kwargs: None

        return motor, sent


@pytest.fixture
def adapter(monkeypatch):
    """The SDK adapted as the worker adapts it, over a stand-in motor chain and robot, and what makes one arm's robot;
    the SDK's classes and the adapter's clock are restored at the end."""
    from i2rt.motor_drivers import dm_driver
    from i2rt.robots import get_robot, motor_chain_robot

    monkeypatch.setattr(motor_chain_robot, "GripperForceLimiter", motor_chain_robot.GripperForceLimiter)

    control_config, config = ConfigFiles.load_configs("yam")
    clock = SimpleNamespace(now=100.0)
    raw = np.array([0, 3.3, 0, 0, 0, 0, config.arms["left"].gripper_limits[0]])
    events = []

    class Chain:
        def __init__(self, motors, offsets, directions, channel, **kwargs):
            assert kwargs["start_thread"] is False
            self.channel = channel
            self.motor_list = motors
            self.motor_offset = np.array(offsets, dtype=float)
            self.motor_direction = np.array(directions)
            self.state_lock = threading.Lock()
            self.running = True
            self.publish()

        def publish(self):
            with self.state_lock:
                self.state = [
                    SimpleNamespace(
                        id=i + 1,
                        position=float(q),
                        velocity=0.0,
                        torque=float(i),
                        error_code="0x1",
                        temperature_rotor=25,
                        temperature_mos=26,
                    )
                    for i, q in enumerate(raw)
                ]
                self._update_absolute_positions(self.state)

        def _update_absolute_positions(self, feedback):
            self.absolute_positions = np.array([s.position for s in feedback])

        def _joint_position_real_to_sim_idx(self, value, i):
            return (value - self.motor_offset[i]) * self.motor_direction[i]

        def start_thread(self):
            events.append(("start", np.array([v.pos for v in self.read_states()])))

    original_robot = get_robot.MotorChainRobot

    class Robot:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        # Test the actual pinned SDK conversion, with only its remapper replaced.
        _motor_state_to_joint_state = original_robot._motor_state_to_joint_state
        command_joint_pos = original_robot.command_joint_pos

    monkeypatch.setattr(get_robot, "DMChainCanInterface", Chain)
    monkeypatch.setattr(get_robot, "MotorChainRobot", Robot)

    def sdk_checks(**kwargs):
        raise AssertionError("The chain repeated the read-only startup audit with repair")

    monkeypatch.setattr(dm_driver, "run_startup_checks", sdk_checks)
    monkeypatch.setattr(sdk_adapter, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: 1.0))
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", dm_driver.DMSingleMotorCanInterface)
    factory = sdk_adapter.SDKCompatibility(config).install()
    from i2rt.robots.utils import ArmType, GripperType

    def create(side="left"):
        robot = factory(
            channel=config.arms[side].channel,
            arm_type=ArmType.YAM_ULTRA,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
            gripper_limits_override=np.array(config.arms[side].gripper_limits),
        )
        robot._gripper_index = None
        robot.remapper = SimpleNamespace(to_command_joint_pos_space=lambda q: q, to_command_joint_vel_space=lambda v: v)
        robot._state_lock = threading.Lock()
        robot._gripper_limits = robot.gripper_limits
        robot._last_gripper_command_qpos = config.arms[side].gripper_limits[1]
        robot._gripper_force_limiter = SimpleNamespace(_is_clogged=False)
        robot._limit_gripper_force = robot.limit_gripper_force
        return robot

    return SimpleNamespace(
        control_config=control_config,
        config=config,
        clock=clock,
        raw=raw,
        events=events,
        create=create,
        checks=dm_driver.run_startup_checks,
    )


@pytest.fixture
def sdk(monkeypatch):
    """The SDK's motor driver and robot factory modules, whose classes the adapter's install replaces; restored at the
    end."""
    from i2rt.motor_drivers import dm_driver
    from i2rt.robots import get_robot, motor_chain_robot

    # The production worker is disposable. Restore its module adaptations in tests.
    for module, name in (
        (dm_driver, "DMSingleMotorCanInterface"),
        (dm_driver, "run_startup_checks"),
        (get_robot, "DMChainCanInterface"),
        (get_robot, "MotorChainRobot"),
        (motor_chain_robot, "GripperForceLimiter"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))

    return dm_driver, get_robot


@pytest.mark.parametrize("j2", [np.pi - 0.01, np.pi + 0.01, 3.3, -0.1, 3.66519])
@pytest.mark.parametrize("gripper", [1.214282081185262, 3.5, 6.340623149389457])
def test_reopen_selects_legal_j2_and_keeps_fixed_gripper_origin(adapter, j2, gripper):
    adapter.raw[1], adapter.raw[6] = j2, gripper

    for _ in range(2):
        robot = adapter.create()
        started = adapter.events[-1][1]
        assert started[1] == pytest.approx(j2)
        assert started[6] == pytest.approx(gripper)
        assert robot.motor_chain.motor_offset[6] == 0
    assert adapter.checks(repair=True) is None  # Never the SDK's Flash-writing check.


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("opening", [0.0, 0.03, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("turns", [-2, -1, 0, 1, 2])
def test_gripper_reopen_recovers_calibrated_branch_and_holds_same_encoder_pose(adapter, side, opening, turns):
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    arm = adapter.config.arms[side]
    closed, opened = arm.gripper_limits
    calibrated = closed + opening * (opened - closed)
    raw = calibrated + turns * 2 * np.pi
    adapter.raw[6] = raw

    for _ in range(2):
        robot = adapter.create(side)
        chain = robot.motor_chain
        feedback = adapter.events[-1][1][6]
        assert feedback == pytest.approx(calibrated)
        assert (feedback - closed) / (opened - closed) == pytest.approx(opening)
        # Use the installed SDK inverse transform: holding must command the
        # same raw position, not move the mechanism by one or more turns.
        command = DMChainCanInterface._joint_position_sim_to_real_idx(chain, feedback, 6)
        assert command == pytest.approx(raw)
        assert arm.gripper_limits == [closed, opened]


@pytest.mark.parametrize("side,offset", [("left", -0.104), ("right", -0.101)])
@pytest.mark.parametrize("raw", [-1.59, -0.5, 0.0, 1.6])
def test_joint_offset_shifts_feedback_and_commands_by_the_same_amount(adapter, side, offset, raw):
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    assert adapter.config.arms[side].joint_offsets == [0, 0, 0, offset, 0, 0]
    adapter.raw[3] = raw

    robot = adapter.create(side)

    feedback = adapter.events[-1][1]
    assert feedback[3] == pytest.approx(raw + offset)
    assert feedback[[0, 2, 4, 5]] == pytest.approx(adapter.raw[[0, 2, 4, 5]])
    # Holding the corrected feedback must command the unchanged encoder position.
    command = DMChainCanInterface._joint_position_sim_to_real_idx(robot.motor_chain, feedback[3], 3)
    assert command == pytest.approx(raw)
    assert DMChainCanInterface._joint_position_sim_to_real_idx(robot.motor_chain, 0.0, 3) == pytest.approx(-offset)


def test_joint_offset_branch_check_uses_corrected_limits(adapter, monkeypatch):
    lower = adapter.config.arms["left"].limits.lower_bounds[3]
    margin = adapter.config.feedback_position_tolerance
    monkeypatch.setattr(sdk_adapter.SDKShutdown, "close_chain", lambda chain: [])

    adapter.raw[3] = lower - margin + 0.104 + 0.001
    adapter.create()
    assert adapter.events[-1][1][3] == pytest.approx(lower - margin + 0.001)
    adapter.raw[3] = lower - margin + 0.104 - 0.001  # Legal raw, illegal true angle.
    with pytest.raises(ValueError, match="no unique branch"):
        adapter.create()


@pytest.mark.parametrize("side,raw", [("left", 0.03528141975402832), ("right", -4.805961608886719)])
def test_power_cycle_angles_resolve_inside_gripper_limits(adapter, side, raw):
    adapter.raw[6] = raw

    robot = adapter.create(side)

    closed, opened = adapter.config.arms[side].gripper_limits
    q = robot.motor_chain.read_states()[6].pos
    assert q == pytest.approx(raw + 2 * np.pi)
    assert 0 <= (q - closed) / (opened - closed) <= 1


@pytest.mark.parametrize("opening,accepted", [(-0.019, True), (1.019, True), (-0.021, False), (1.021, False)])
def test_gripper_branch_keeps_original_feedback_margin(adapter, opening, accepted, monkeypatch):
    closed, opened = adapter.config.arms["left"].gripper_limits
    adapter.raw[6] = closed + opening * (opened - closed) - 2 * np.pi
    # Stub cleanup as well: this fixture never constructs a transport.
    monkeypatch.setattr(sdk_adapter.SDKShutdown, "close_chain", lambda chain: [])

    if accepted:
        robot = adapter.create()
        q = robot.motor_chain.read_states()[6].pos
        assert (q - closed) / (opened - closed) == pytest.approx(opening)
    else:
        with pytest.raises(ValueError, match="no unique branch"):
            adapter.create()
        assert not adapter.events  # Reject before the SDK background writer starts.


def test_ambiguous_gripper_branch_is_rejected_before_writer_starts(adapter, monkeypatch):
    adapter.config.arms["left"].gripper_limits = (8.0, 0.0)
    adapter.raw[6] = 0.5  # Both 0.5 and 0.5 + 2π lie in this invalid calibration.
    monkeypatch.setattr(sdk_adapter.SDKShutdown, "close_chain", lambda chain: [])

    with pytest.raises(ValueError, match="no unique branch"):
        adapter.create()
    assert not adapter.events


@pytest.mark.parametrize("direction", [-1, 1])
def test_every_joint_boundary_selects_one_branch(direction):
    config = ConfigFiles.load_configs("yam")[1]

    for arm in config.arms.values():
        for lo, hi in zip(arm.limits.lower_bounds[:6], arm.limits.upper_bounds[:6]):
            for q in (lo, (lo + hi) / 2, hi):
                for turns in (-2, -1, 0, 1, 2):
                    assert sdk_adapter.SDKCompatibility.select_joint_branch(
                        direction * q + turns * 2 * np.pi,
                        direction,
                        lo,
                        hi,
                        config.feedback_position_tolerance,
                    ) == pytest.approx(q)
    for lo, hi in ((-7, 7), (0, 0.1)):
        with pytest.raises(ValueError, match="unique"):
            # the margin the source's signature defaulted to: the port takes it without a default
            sdk_adapter.SDKCompatibility.select_joint_branch(3.3, direction, lo, hi, 0.0)


@pytest.mark.parametrize("freeze", ["raw", "converted"])
def test_raw_and_converted_freezes_cannot_refresh_feedback(adapter, freeze):
    robot = adapter.create()
    robot._joint_state = robot._motor_state_to_joint_state(robot.motor_chain.read_states())
    devices = YAMHardware.__new__(YAMHardware)
    devices.robots = {s: robot for s in ("left", "right")}
    initial = devices.read()["left"]

    for _ in range(8):
        adapter.clock.now += 0.02
        if freeze == "converted":
            adapter.raw[1] += 0.001
            robot.motor_chain.publish()
        else:
            robot._joint_state = robot._motor_state_to_joint_state(robot.motor_chain.read_states())
        state = devices.read()["left"]
        assert state.sampled_monotonic_s == initial.sampled_monotonic_s
        assert state.sample_id == initial.sample_id
        assert state.position == initial.position
    with pytest.raises(RuntimeError, match="stale"):
        ControlHelpers.check_states(devices.read(), adapter.control_config, adapter.config, adapter.clock.now)

    robot.motor_chain.publish()
    robot._joint_state = robot._motor_state_to_joint_state(robot.motor_chain.read_states())
    fresh = devices.read()["left"]
    assert fresh.sample_id > initial.sample_id and fresh.sampled_monotonic_s > initial.sampled_monotonic_s
    assert fresh.effort == tuple(float(i) for i in range(7))
    # The SDK's raw-radian gripper command crosses the adapter as a normalized opening.
    assert fresh.sdk_gripper_target == pytest.approx(1.0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_adapter_applies_arm_kp_and_keeps_official_damping_and_gripper_gains(adapter, side):
    from i2rt.robots.utils import ArmType, GripperType

    adapter.config.arms[side].joint_position_gains = [80, 400, 300, 40, 10, 20]

    robot = adapter.create(side)

    assert robot.kp[:6] == pytest.approx([80, 400, 300, 40, 10, 20])
    assert robot.kd[:6] == pytest.approx([5, 5, 5, 1.5, 1.5, 1.5])
    assert robot.limit_gripper_force == 50.0
    assert tuple((robot.kp[6], robot.kd[6])) == GripperType.LINEAR_4310.get_motor_kp_kd(ArmType.YAM_ULTRA)


def test_adapter_keeps_sdk_arm_kp_when_unconfigured(adapter):
    adapter.config.arms["left"].joint_position_gains = None
    adapter.config.arms["right"].joint_position_gains = [80, 400, 400, 40, 10, 20]

    robot = adapter.create()

    assert robot.kp[:6] == pytest.approx([80, 80, 80, 40, 10, 10])
    assert adapter.create("right").kp[:6] == pytest.approx([80, 400, 400, 40, 10, 20])


@pytest.mark.parametrize("side", ["left", "right"])
def test_adapter_applies_arm_gravity_factor_and_keeps_gripper_factor(adapter, side):
    adapter.config.arms[side].gravity_compensation_factors = [1, 1.25, 1.2, 1.2, 1, 1]

    robot = adapter.create(side)

    assert robot.gravity_comp_factor == pytest.approx([1, 1.25, 1.2, 1.2, 1, 1, 1])


def test_adapter_keeps_sdk_gravity_factor_when_unconfigured(adapter):
    adapter.config.arms["left"].gravity_compensation_factors = None

    robot = adapter.create()

    assert robot.gravity_comp_factor == pytest.approx([1, 1.1, 1.1, 1.2, 1, 1, 1])


def test_gripper_force_is_configured_per_side_before_sdk_initialization(adapter):
    adapter.config.arms["left"].gripper_force_n = 60.0

    assert adapter.create("left").limit_gripper_force == 60.0
    assert adapter.create("right").limit_gripper_force == 50.0


def test_reporting_limiter_preserves_sdk_behavior_and_reports_actual_requested_torque(adapter):
    from i2rt.robots import motor_chain_robot
    from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType

    args = (60.0, GripperType.LINEAR_4310, ArmType.YAM_ULTRA, 20)
    reporting, official = motor_chain_robot.GripperForceLimiter(*args), GripperForceLimiter(*args)
    state = dict(
        current_qvel=0.0,
        current_eff=0.55,
        current_qpos=4.0,
        target_qpos=6.0,
        last_command_qpos=4.0,
        current_normalized_qpos=0.5,
        target_normalized_qpos=0.0,
    )

    assert reporting.update(state) == pytest.approx(official.update(state))
    assert reporting.station_target_effort_nm == pytest.approx(60 * 0.096 / 6.57 + 0.3)
    state.update(target_normalized_qpos=1.0, target_qpos=1.0)
    assert reporting.update(state) == official.update(state) == 1.0
    assert reporting.station_target_effort_nm is None


def test_pinned_official_force_limiter_detects_closing_contact_and_releases_for_opening():
    from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType

    limiter = GripperForceLimiter(50.0, GripperType.LINEAR_4310, ArmType.YAM_ULTRA, kp=20)
    state = dict(
        current_qvel=0.0,
        current_eff=0.55,
        current_qpos=4.0,
        target_qpos=6.0,
        last_command_qpos=4.0,
        current_normalized_qpos=0.5,
        target_normalized_qpos=0.0,
    )

    adjusted = limiter.update(state)

    assert limiter._is_clogged
    assert adjusted != state["target_qpos"]
    assert limiter.compute_target_gripper_torque(state) == pytest.approx(50 * 0.096 / 6.57 + 0.3)
    state.update(target_normalized_qpos=1.0, target_qpos=1.0)
    assert limiter.update(state) == 1.0
    assert not limiter._is_clogged


@pytest.mark.parametrize("side", ["left", "right"])
def test_commands_feed_arm_velocity_forward_and_keep_gripper_at_zero(adapter, side):
    robot, limits = SDKRobotRig.robot_for(adapter, side)
    velocity = [0.4, -0.7, 0.1, 0.0, 0.2, -0.3, 0.25]

    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], velocity, SDKRobotRig.LIMIT)

    assert robot._commands.vel[:6] == pytest.approx(velocity[:6])
    assert robot._commands.vel[6] == 0.0
    assert robot._commands.pos[6] == pytest.approx(sum(limits) / 2)
    assert robot._commands.kp == pytest.approx(robot._kp)
    assert robot._commands.kd == pytest.approx(robot._kd)


def test_omitted_velocity_holds_with_zero_feedforward(adapter):
    robot, _ = SDKRobotRig.robot_for(adapter, "left")

    YAMHardware.command_position_velocity(robot, [0.1] * 6 + [0.5], None, SDKRobotRig.LIMIT)

    assert robot._commands.vel == pytest.approx(np.zeros(7))
    assert robot._commands.pos[:6] == pytest.approx([0.1] * 6)


def test_velocity_is_capped_and_zeroed_where_the_sdk_clips_position(adapter):
    robot, _ = SDKRobotRig.robot_for(adapter, "left")

    def clip_j2(q):  # Mimic the SDK's in-place arm clip.
        q[1] = min(q[1], 1.0)
        return q

    robot._clip_robot_joint_pos_command = clip_j2
    YAMHardware.command_position_velocity(robot, [0, 1.5, 0, 0, 0, 0, 0.5], [3, 0.8, -3, 0, 0, 0, 0], SDKRobotRig.LIMIT)

    assert robot._commands.vel[:3] == pytest.approx([1.0, 0.0, -1.0])
    assert robot._commands.pos[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "position,velocity",
    [([0.0] * 7, [float("nan")] * 7), ([0.0] * 6, [0.0] * 6), ([float("inf")] * 7, None)],
)
def test_malformed_commands_are_rejected_without_writing(adapter, position, velocity):
    robot, _ = SDKRobotRig.robot_for(adapter, "left")
    robot._commands = previous = object()

    with pytest.raises(ValueError):
        YAMHardware.command_position_velocity(robot, position, velocity, SDKRobotRig.LIMIT)
    assert robot._commands is previous


def test_control_thread_never_observes_a_zero_gain_command(adapter):
    """The SDK publishes kp = kd = 0 outside its lock."""
    robot, _ = SDKRobotRig.robot_for(adapter, "left")
    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], None, SDKRobotRig.LIMIT)
    stop, seen = threading.Event(), []

    def control_thread():
        while not stop.is_set():
            with robot._command_lock:
                commands = robot._commands
            seen.append(float(np.min(commands.kp)))

    reader = threading.Thread(target=control_thread)
    reader.start()
    try:
        for i in range(3000):
            YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], [0.01 * (i % 50)] * 7, SDKRobotRig.LIMIT)
    finally:
        stop.set()
        reader.join()

    assert seen and min(seen) > 0


def test_adapter_writes_friction_into_command_torques(adapter):
    adapter.config.arms["left"].friction_feedforward_nm = [0, 1.5, 0, 0, 0, 0]
    robot, _ = SDKRobotRig.robot_for(adapter, "left")
    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], [0, 0.3, 0, 0, 0, 0, 0], SDKRobotRig.LIMIT)
    assert robot._station_planned_velocity == pytest.approx([0, 0.3, 0, 0, 0, 0])

    for _ in range(30):
        adapter.clock.now += CONTROL_PERIOD_S
        robot._station_apply_friction()
    assert robot._commands.torques == pytest.approx([0, 1.5, 0, 0, 0, 0, 0])

    # A new command carries the latest torque until the control thread updates it.
    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], [0, 0.0, 0, 0, 0, 0, 0], SDKRobotRig.LIMIT)
    assert robot._commands.torques == pytest.approx([0, 1.5, 0, 0, 0, 0, 0])
    for _ in range(round((FrictionFeedforward.LATCH_S + FrictionFeedforward.FADE_S) / CONTROL_PERIOD_S) + 10):
        adapter.clock.now += CONTROL_PERIOD_S
        robot._station_apply_friction()
    assert robot._commands.torques == pytest.approx(np.zeros(7))


def test_adapter_leaves_torques_alone_without_friction(adapter):
    adapter.config.arms["left"].friction_feedforward_nm = [0] * 6
    robot, _ = SDKRobotRig.robot_for(adapter, "left")
    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], [0, 0.3, 0, 0, 0, 0, 0], SDKRobotRig.LIMIT)

    robot._station_apply_friction()

    assert robot._commands.torques == pytest.approx(np.zeros(7))


def test_adapter_scales_friction_with_the_applied_gravity_torque(adapter):
    adapter.config.arms["left"].friction_feedforward_nm = [0, 0.5, 0, 0, 0, 0]
    adapter.config.arms["left"].friction_load_ratio = [0, 0.15, 0, 0, 0, 0]
    robot, _ = SDKRobotRig.robot_for(adapter, "left")
    robot._station_gravity_torque = np.array([0, -10.0, 4.0, 0, 0, 0])
    YAMHardware.command_position_velocity(robot, [0.0] * 6 + [0.5], [0, 0.3, 0, 0, 0, 0, 0], SDKRobotRig.LIMIT)

    for _ in range(30):
        adapter.clock.now += CONTROL_PERIOD_S
        robot._station_apply_friction()

    assert robot._commands.torques == pytest.approx([0, 2.0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("status", [0x0, 0x8, 0x9, 0xA, 0xB, 0xC, 0xE, 0xF])
def test_startup_hardware_error_never_clears_or_reenables_motor(sdk, status):
    motor, sent = StartupMotors.startup_motor(sdk, [status])
    level = logging.getLogger().level

    with pytest.raises(RuntimeError, match="startup refused error clearing") as captured:
        motor.motor_on(1, "DM4310")

    assert f"can_follower_r Motor 1: {hex(status)}" in str(captured.value)
    assert "MOS 25.0 C, rotor 23.0 C" in str(captured.value)
    assert sent == [0xFC]
    assert logging.getLogger().level == level


def test_normal_startup_never_clears(sdk):
    motor, sent = StartupMotors.startup_motor(sdk, [0x1])

    assert motor.motor_on(1, "DM4310").error_code == "0x1"
    assert sent == [0xFC]


def test_startup_communication_timeout_clears_once_using_sdk(sdk):
    motor, sent = StartupMotors.startup_motor(sdk, [0xD, 0x1, 0xD, 0x1])
    level = logging.getLogger().level

    for motor_id in (1, 2):
        assert motor.motor_on(motor_id, "DM4310").error_code == "0x1"

    assert sent == [0xFC, 0xFB, 0xFB, 0xFB, 0xFC] * 2
    assert logging.getLogger().level == level
    with pytest.raises(RuntimeError, match="no startup feedback"):
        motor.clean_error(2)
    assert len(sent) == 10  # Runtime recovery remains disabled, even after a successful reset.


@pytest.mark.parametrize("status", [0xD, 0x8, 0xA, 0x0])
def test_startup_reset_does_not_loop_or_clear_a_following_fault(sdk, status):
    motor, sent = StartupMotors.startup_motor(sdk, [0xD, status])
    level = logging.getLogger().level

    with pytest.raises(RuntimeError, match="startup refused error clearing") as captured:
        motor.motor_on(1, "DM4310")

    assert f"Motor 1: {hex(status)}" in str(captured.value)
    assert sent == [0xFC, 0xFB, 0xFB, 0xFB, 0xFC]
    assert logging.getLogger().level == level


@pytest.mark.parametrize("clear_replies", [1, 3])
@pytest.mark.parametrize("reply_delay", [0.0, 0.003])
@pytest.mark.parametrize("enabled_status", [0x1, 0x0, 0x8, 0xD])
def test_startup_uses_enable_reply_after_draining_clear_replies(
    sdk, monkeypatch, clear_replies, reply_delay, enabled_status
):
    import can

    dm_driver, _ = sdk
    pending, sent = deque(), []
    statuses = dict.fromkeys(range(1, 8), 0xD)
    cleared = dict.fromkeys(statuses, 0)

    class Bus:
        state = "active"
        channel_info = "startup-test"

        def send(self, message):
            motor_id, command = message.arbitration_id, message.data[-1]
            sent.append((motor_id, command))
            if command == 0xFB:
                statuses[motor_id] = 0x0
                cleared[motor_id] += 1
                if cleared[motor_id] > clear_replies:
                    return
            else:
                assert command == 0xFC
                if cleared[motor_id]:
                    statuses[motor_id] = enabled_status
            response = can.Message(
                arbitration_id=motor_id + 16,
                data=[statuses[motor_id] << 4 | motor_id, 0x80, 0, 0x80, 0, 0, 25, 23],
                is_extended_id=False,
            )
            pending.append((time.monotonic() + reply_delay, response))

        def recv(self, timeout):
            if pending:
                time.sleep(min(timeout, max(0, pending[0][0] - time.monotonic())))
                if pending[0][0] <= time.monotonic():
                    return pending.popleft()[1]
            else:
                time.sleep(timeout)
            return None

    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: Bus())
    sdk_adapter.SDKCompatibility(ConfigFiles.load_configs("yam")[1]).install()
    motor = dm_driver.DMSingleMotorCanInterface(channel="startup-test")
    # Exercise the real SDK send/receive/parser/clear/enable methods, with only CAN replaced.
    motor_ids = range(1, 8) if enabled_status == 0x1 else [1]
    for motor_id in motor_ids:
        if enabled_status == 0x1:
            assert motor.motor_on(motor_id, "DM4310").error_code == "0x1"
        else:
            with pytest.raises(RuntimeError, match=f"{hex(enabled_status)} .*startup refused"):
                motor.motor_on(motor_id, "DM4310")
        assert not pending  # A clear reply must never masquerade as the enable reply.
    assert sent == [(motor_id, command) for motor_id in motor_ids for command in [0xFC, 0xFB, 0xFB, 0xFB, 0xFC]]


@pytest.mark.parametrize("failure", ["motor_on", "branch", "robot"])
@pytest.mark.parametrize("disable_error", [False, True])
def test_partial_construction_disables_all_motors_and_closes_socket(sdk, monkeypatch, failure, disable_error):
    dm_driver, get_robot = sdk
    config = ConfigFiles.load_configs("yam")[1]
    events = []

    class Transport:
        def __init__(self, **kwargs):
            self.bus = SimpleNamespace(send=self.send)

        def _drain_bus(self, **kwargs):
            return 0

        def _get_frame_id(self, motor_id):
            return motor_id

        def motor_on(self, motor_id, motor_type):
            events.append(("enable", motor_id))
            if failure == "motor_on" and motor_id == 4:
                raise RuntimeError("enable reply lost")
            raw = [0, -1.0 if failure == "branch" else 0, 0, 0, 0, 0, 3.5]
            return SimpleNamespace(
                id=motor_id,
                position=raw[motor_id - 1],
                velocity=0.0,
                torque=0.0,
                error_code="0x1",
                temperature_rotor=25,
                temperature_mos=25,
            )

        def send(self, frame, timeout):
            assert timeout == DISABLE_TIMEOUT_S
            assert bytes(frame.data) == b"\xff" * 7 + b"\xfd"
            events.append(("disable", frame.arbitration_id))
            if disable_error and frame.arbitration_id == 2:
                raise OSError("injected disable send failure")

        def close(self):
            events.append("socket_close")

    class BrokenRobot:
        def __init__(self, **kwargs):
            self.motor_chain = kwargs["motor_chain"]
            self._stop_event = SimpleNamespace(set=lambda: events.append("stop_writer"))
            self._server_thread = SimpleNamespace(
                join=lambda timeout: events.append("join_writer"), is_alive=lambda: False
            )
            raise RuntimeError("robot initialization failed")

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", Transport)
    monkeypatch.setattr(dm_driver, "run_startup_checks", lambda *a, **kw: None)
    monkeypatch.setattr(get_robot.DMChainCanInterface, "start_thread", lambda self: events.append("start_writer"))
    monkeypatch.setattr(get_robot, "MotorChainRobot", BrokenRobot)
    factory = sdk_adapter.SDKCompatibility(config).install()
    from i2rt.robots.utils import ArmType, GripperType

    with pytest.raises((ValueError, RuntimeError), match="enable reply|unique branch|initialization") as captured:
        factory(
            channel=config.arms["left"].channel,
            arm_type=ArmType.YAM_ULTRA,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
            gripper_limits_override=np.array(config.arms["left"].gripper_limits),
            enable_auto_recovery=False,
        )

    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == "disable"] == list(range(1, 8)) * 3
    assert events[-1] == "socket_close"
    assert ("start_writer" in events) is (failure == "robot")
    if failure == "robot":
        assert events.index("stop_writer") < events.index(("disable", 1))
    if disable_error:
        assert any("cleanup incomplete" in note for note in captured.value.__notes__)
