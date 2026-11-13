import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware import arms, sdk_worker
from pear.deploy.station.control.robots.yam.hardware.arms import YAMHardware
from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver
from pear.utils.worker_process import WorkerProcess

DISABLE_TIMEOUT_S = 0.02  # send timeout of the motor disable frame


class StandInHardware:
    """Both arms of the SDK's worker, which answer reads and close without touching hardware."""

    def __init__(self, config):
        pass

    def read(self):
        """Returns a reading."""
        return "feedback"

    def close(self):
        """Closes nothing."""


class StandInBrokenHardware:
    """Both arms of the SDK's worker, whose construction starts a thread and then fails."""

    def __init__(self, config):
        threading.Thread(target=time.sleep, args=(60,), daemon=False).start()
        raise RuntimeError("construction failed after starting an SDK thread")


class StandInWorker:
    """The SDK's worker on stand-in arms: the target of a new interpreter, which puts the stand-in in the place of the
    worker's arms before it runs the worker."""

    @staticmethod
    def main(connection: Connection, robot_config: YAMConfig, devices: type) -> None:
        """Runs the worker whose arms are ``devices``."""
        sdk_worker.YAMHardware = devices
        sdk_worker.SDKWorker.main(connection, robot_config)


class SDKWorkers:
    """How a test runs the SDK's worker process on stand-in arms, and ends it."""

    @staticmethod
    def start_worker(devices: type) -> tuple[Connection, multiprocessing.Process]:
        """Starts the worker whose arms are ``devices`` and returns the test's end of its pipe and the process."""
        # spawned, as the robot's driver starts it: a forked child would inherit pytest's faulthandler watchdog
        # without its thread, and the worker's own watchdog would wait for that thread forever
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=StandInWorker.main, args=(child, ConfigFiles.load_configs("yam")[1], devices))
        process.start()
        child.close()

        return parent, process

    @staticmethod
    def cleanup(connection: Connection, process: multiprocessing.Process) -> None:
        """Ends the process if it still runs and closes the test's end of the pipe."""
        if process.is_alive():
            process.terminate()
        process.join(2)
        connection.close()


@pytest.fixture
def devices():
    """Both arms through the SDK, built without the SDK: two healthy stand-in robots and a reading that is a mark."""
    devices = YAMHardware.__new__(YAMHardware)
    devices.robots = {
        side: SimpleNamespace(
            motor_chain=SimpleNamespace(
                running=True,
                state_lock=threading.Lock(),
                state=[SimpleNamespace(id=i, error_code="0x1") for i in range(1, 8)],
            ),
            _stop_event=threading.Event(),
            _server_thread=SimpleNamespace(is_alive=lambda: True),
        )
        for side in ("left", "right")
    }
    devices.read = Mock(return_value={"healthy": True})

    return devices


def test_sdk_worker_serves_reads_and_closes():
    connection, process = SDKWorkers.start_worker(StandInHardware)

    try:
        assert connection.poll(10)
        assert connection.recv() == (True, None)
        connection.send(("read", None))
        assert connection.poll(2)
        assert connection.recv() == (True, "feedback")
        connection.send(("close", None))
        assert connection.poll(2)
        assert connection.recv() == (True, None)
        process.join(2)
        assert not process.is_alive()
    finally:
        SDKWorkers.cleanup(connection, process)


def test_sdk_worker_exits_when_caller_stops_heartbeats():
    connection, process = SDKWorkers.start_worker(StandInHardware)

    try:
        assert connection.poll(10)
        assert connection.recv() == (True, None)
        process.join(2)
        assert not process.is_alive()
    finally:
        SDKWorkers.cleanup(connection, process)


def test_failed_sdk_construction_cannot_leave_background_threads():
    connection, process = SDKWorkers.start_worker(StandInBrokenHardware)

    try:
        assert connection.poll(10)
        success, error = connection.recv()
        assert not success and "construction failed" in error
        process.join(2)
        assert not process.is_alive()
    finally:
        SDKWorkers.cleanup(connection, process)


def test_unresponsive_sdk_process_is_terminated():
    parent, peer = WorkerProcess.CONTEXT.Pipe()
    worker = WorkerProcess(time.sleep, (60,), "yam-sdk", YAMDriver.EXIT_TIMEOUT_S, is_group_leader=False)
    worker.start()
    process = worker.process
    client = YAMDriver.__new__(YAMDriver)
    client.connection, client.worker, client.is_closed = parent, worker, False

    # Keep the peer endpoint open in this process: exercise timeout, not EOF.
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            client._receive(0.05)
        assert not process.is_alive()
        assert client.is_closed
    finally:
        peer.close()
        SDKWorkers.cleanup(client.connection, process)


@pytest.mark.parametrize("send_failure", [False, True])
def test_shutdown_disables_all_motors_before_closing_socket(send_failure):
    events, frames = [], []

    def send(frame, timeout):
        assert events == ["stop", "join"]
        assert not chain.running
        assert chain.command_lock.locked()
        assert timeout == DISABLE_TIMEOUT_S
        frames.append(frame)
        if send_failure and frame.arbitration_id == 2:
            raise OSError("send failed")

    chain = SimpleNamespace(
        running=True,
        command_lock=threading.Lock(),
        motor_list=[(n, "unused") for n in range(1, 8)],
        motor_interface=SimpleNamespace(bus=SimpleNamespace(send=send), _get_frame_id=lambda n: n),
        close=lambda: events.append("close"),
    )
    robot = SimpleNamespace(
        motor_chain=chain,
        _stop_event=SimpleNamespace(set=lambda: events.append("stop")),
        _server_thread=SimpleNamespace(join=lambda timeout: events.append("join"), is_alive=lambda: False),
    )
    # the port powers an arm off inside YAMHardware.close: an arm's hardware of this one robot, its lock a stand-in
    hardware = YAMHardware.__new__(YAMHardware)
    hardware.robots, hardware.device_lock = {"left": robot}, SimpleNamespace(close=lambda: None)

    if send_failure:
        with pytest.raises(RuntimeError, match="shutdown incomplete"):
            hardware.close()
    else:
        hardware.close()

    assert events == ["stop", "join", "close"]
    assert [f.arbitration_id for f in frames] == list(range(1, 8)) * 3
    assert all(bytes(f.data) == b"\xff" * 7 + b"\xfd" for f in frames)
    assert not chain.command_lock.locked()


def test_a_healthy_sdk_passes_the_check_and_returns_its_reading(devices):
    assert devices.read_for_recovery() == {"healthy": True}
    devices.read.assert_called_once_with()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("code", ["0x0", "0x8", "0x9", "0xa", "0xb", "0xc", "0xd", "0xe"])
def test_disabled_and_faulted_motors_block_recovery(devices, side, code):
    devices.robots[side].motor_chain.state[3].error_code = code

    with pytest.raises(RuntimeError, match=f"{side} motor 4"):
        devices.read_for_recovery()
    devices.read.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_arm", "missing_motor", "chain", "writer", "stop"])
def test_stopped_or_incomplete_sdk_blocks_recovery(devices, failure):
    robot = devices.robots["right"]
    if failure == "missing_arm":
        del devices.robots["right"]
    elif failure == "missing_motor":
        robot.motor_chain.state.pop()
    elif failure == "chain":
        robot.motor_chain.running = False
    elif failure == "writer":
        robot._server_thread.is_alive = lambda: False
    else:
        robot._stop_event.set()

    with pytest.raises(RuntimeError, match="Recovery blocked"):
        devices.read_for_recovery()
    devices.read.assert_not_called()


def test_the_client_forwards_the_recovery_check_to_its_worker():
    hardware = YAMDriver.__new__(YAMDriver)
    hardware._call = Mock(return_value={"healthy": True})

    assert hardware.read_for_recovery() == {"healthy": True}
    hardware._call.assert_called_once_with("read_for_recovery")


def test_a_reading_of_the_sdk_cache_does_not_change_with_the_cache():
    # No SDK or hardware constructors, sockets, or command methods exist here.
    robot = SimpleNamespace(
        _state_lock=threading.Lock(),
        _joint_state=SimpleNamespace(
            station_sample=(1, 100.0),
            pos=np.zeros(7),
            vel=np.zeros(7),
            eff=np.arange(7, dtype=float),
        ),
        _gripper_limits=(0.0, 2.0),
        _last_gripper_command_qpos=1.0,
        _gripper_force_limiter=SimpleNamespace(_is_clogged=True, station_target_effort_nm=1.03),
        _limit_gripper_force=50.0,
    )
    devices = YAMHardware.__new__(YAMHardware)
    devices.robots = {"left": robot}

    state = devices.read()["left"]

    assert state.effort[2] == 2.0
    assert state.sdk_gripper_target_effort_nm == 1.03
    assert state.sdk_gripper_force_limited
    robot._joint_state.eff[:] = 99
    assert state.effort[2] == 2.0


def test_feedback_ages_are_not_refreshed_by_repeated_sdk_cache_reads():
    robot = SimpleNamespace(
        _state_lock=threading.Lock(),
        _joint_state=SimpleNamespace(
            station_sample=(1, time.monotonic()),
            pos=(0.0,) * 7,
            vel=(0.0,) * 7,
            eff=np.zeros(7),
        ),
        _gripper_limits=(0.0, 2.0),
        _last_gripper_command_qpos=1.0,
        _gripper_force_limiter=SimpleNamespace(_is_clogged=False),
        _limit_gripper_force=50.0,
    )
    devices = YAMHardware.__new__(YAMHardware)
    devices.robots = {"left": robot}

    first = devices.read()["left"].sampled_monotonic_s
    time.sleep(0.005)
    assert devices.read()["left"].sampled_monotonic_s == first
    robot._joint_state.station_sample = (2, time.monotonic())
    assert devices.read()["left"].sampled_monotonic_s > first


def test_second_arm_failure_closes_already_constructed_first_arm(monkeypatch):
    config = ConfigFiles.load_configs("yam")[1]
    closed = []
    monkeypatch.setattr(arms.StartupChecks, "run", lambda _: SimpleNamespace(close=lambda: closed.append("lease")))
    # the port powers an arm off inside YAMHardware.close, through these two steps of SDKShutdown: the first
    # stands for the arm's shutdown, as SdkShutdown.close_robot did
    monkeypatch.setattr(arms.SDKShutdown, "stop_robot_writer", lambda robot: closed.append(robot) or [])
    monkeypatch.setattr(arms.SDKShutdown, "close_chain", lambda chain: [])
    first = SimpleNamespace(motor_chain=None)

    def create(**kwargs):
        if kwargs["channel"] == config.arms["left"].channel:
            return first
        raise RuntimeError("right construction failed")

    monkeypatch.setattr(arms.SDKCompatibility, "install", lambda _: create)
    with pytest.raises(RuntimeError, match="right construction"):
        arms.YAMHardware(config)

    assert closed == [first, "lease"]
