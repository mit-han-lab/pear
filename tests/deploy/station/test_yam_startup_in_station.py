import builtins

import pytest

from config_files import ConfigFiles
from pear.deploy.station.control.joint.device_lock import DeviceLock
from pear.deploy.station.control.robots.yam.hardware.driver import YAMDriver
from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks


def test_channel_lease_prevents_second_owner():
    lease = DeviceLock(StartupChecks.LOCK_PROGRAM, ["unit_test_channel"])
    try:
        with pytest.raises(BlockingIOError):
            DeviceLock(StartupChecks.LOCK_PROGRAM, ["unit_test_channel"])
    finally:
        lease.close()

    DeviceLock(StartupChecks.LOCK_PROGRAM, ["unit_test_channel"]).close()


@pytest.mark.parametrize("timeout_disabled", [False, True])
def test_preflight_is_readonly_and_releases_buses_on_failure(monkeypatch, timeout_disabled):
    from i2rt.motor_config_tool import dm_motor_registers, utils
    from i2rt.motor_drivers import motor_check
    from i2rt.robots import get_robot

    config = ConfigFiles.load_configs("yam")[1]
    for side, arm in config.arms.items():
        arm.channel = f"unit_test_{side}"
    checked, opened, closed, reads = [], [], [], []

    class Bus:
        def __init__(self, channel, bustype):
            self.channel = channel
            opened.append(channel)

        def close(self):
            closed.append(self.channel)

    def check(channel, motors, **kwargs):
        assert kwargs["repair"] is False
        assert len(motors) == 7
        checked.append(channel)

    def read(bus, motor, register):
        assert register == "TIMEOUT"
        reads.append((bus.channel, motor))
        return 0 if timeout_disabled and bus.channel == "unit_test_right" else 8000

    monkeypatch.setattr(StartupChecks, "verify_sdk_revision", lambda: None)
    monkeypatch.setattr(StartupChecks, "verify_adapters", lambda _: None)
    monkeypatch.setattr(StartupChecks, "ensure_idle", lambda _: None)
    monkeypatch.setattr(utils, "RawCanInterface", Bus)
    monkeypatch.setattr(dm_motor_registers, "read_register", read)
    monkeypatch.setattr(motor_check, "run_startup_checks", check)
    monkeypatch.setattr(get_robot, "get_yam_robot", lambda **_: pytest.fail("motor enable"))
    if timeout_disabled:
        with pytest.raises(RuntimeError, match="timeout is disabled"):
            StartupChecks.preflight(config)
    else:
        result = StartupChecks.preflight(config)
        assert all(row["checks"] == "passed" for row in result.values())
        assert len(reads) == 14

    assert checked == opened == closed == ["unit_test_left", "unit_test_right"]
    DeviceLock(StartupChecks.LOCK_PROGRAM, checked).close()  # No retained ownership after success or failure.


def test_incomplete_real_config_refuses_before_sdk_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("i2rt"), "SDK was imported before commissioning gate"
        return original(name, *args, **kwargs)

    control_config, config = ConfigFiles.load_configs("yam")
    config.arms["left"].gripper_limits = None
    monkeypatch.setattr(builtins, "__import__", guarded)

    with pytest.raises(RuntimeError, match="not commissioned"):
        YAMDriver(control_config, config)
