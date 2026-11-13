import json
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np

from pear.deploy.station.control.joint.device_lock import DeviceLock
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.layout import LAYOUT


class StartupChecks:
    """The checks that run before any motor is enabled: the SDK's revision, the ownership and the identity of the CAN
    buses, and the motors' registers, all read-only; and the devices the operating system lists."""

    SDK_COMMIT = "120c3c81400171174604e503943f8d1ebc891058"  # the audited Git revision of the SDK
    SDK_JOINT_MARGIN = 0.15  # by how much the SDK's factory widens the joint bounds of its XML model, rad
    CAN_LINK_TYPE = "280"  # ARPHRD_CAN, the sysfs type of a CAN network interface
    IDLE_PROBE_S = 0.1  # between two reads of a bus's transmit counter
    USB_KEYS = ("serial", "product", "idVendor", "idProduct")  # what a USB device's identity is read from
    CAN_STATES = ("up", "unknown")  # the operational states of a CAN interface that is up
    CHANNEL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_-"  # the characters of a CAN interface's name
    LOCK_PROGRAM = "yam-station"  # the program the buses' lock files are named after, every YAM program's
    MOTOR_IDS = range(1, LAYOUT.coordinate_count + 1)  # an arm's motors: its six joints, then its gripper
    GRIPPER_MOTOR_ID = LAYOUT.coordinate_count  # the gripper's motor, the last of an arm

    @classmethod
    def run(cls, robot_config: YAMConfig) -> DeviceLock:
        """Runs every read-only check before the motors are enabled, and returns the lock of the buses, which the
        caller owns."""
        cls.verify_sdk_revision()
        device_lock = cls.claim_buses(robot_config)

        try:
            cls.check_motor_configuration(robot_config)
        except BaseException:
            device_lock.close()
            raise

        return device_lock

    @classmethod
    def preflight(cls, robot_config: YAMConfig) -> dict[str, dict[str, str]]:
        """Runs the checks of enabling the arms on their own, powering no motor, and returns what was checked of each
        arm."""
        robot_config.require_commissioned()
        cls.run(robot_config).close()

        return {
            arm: {"channel": arm_config.channel, "adapter_serial": arm_config.adapter_serial, "checks": "passed"}
            for arm, arm_config in robot_config.arms.items()
        }

    @classmethod
    def inventory(cls) -> dict[str, Any]:
        """Returns the CAN interfaces and the USB cameras the operating system lists; opens no device."""
        cameras = {}

        for path in sorted(Path("/sys/class/video4linux").glob("*")):
            identity = cls.usb_identity(path / "device")
            serial = identity.get("serial", path.name)
            cameras.setdefault(serial, identity | {"nodes": []})["nodes"].append(f"/dev/{path.name}")

        return {"can": cls.can_interfaces(), "usb_cameras": cameras}

    @classmethod
    def verify_sdk_revision(cls) -> None:
        """Raises unless the installed SDK is the audited Git revision."""
        direct_url = json.loads(distribution("i2rt").read_text("direct_url.json") or "{}")
        if direct_url.get("vcs_info", {}).get("commit_id") != cls.SDK_COMMIT:
            raise RuntimeError("Installed i2rt differs from the audited Git revision")

    @classmethod
    def claim_buses(cls, robot_config: YAMConfig) -> DeviceLock:
        """Returns the lock of both arms' buses once their adapters are found as configured and the buses idle."""
        channels = [arm_config.channel for arm_config in robot_config.arms.values()]
        # the buses are locked in sorted order, each name checked before its bus: the first invalid name is refused
        # once the buses before it are locked, and they are released with it
        sorted_channels = sorted(channels)
        valid_count = next(
            (
                index
                for index, channel in enumerate(sorted_channels)
                if not channel or any(char not in cls.CHANNEL_CHARS for char in channel)
            ),
            len(sorted_channels),
        )
        device_lock = DeviceLock(cls.LOCK_PROGRAM, sorted_channels[:valid_count])
        if valid_count < len(sorted_channels):
            device_lock.close()
            raise ValueError("Invalid CAN interface name")

        try:
            cls.verify_adapters(robot_config)
            cls.ensure_idle(channels)
        except BaseException:
            device_lock.close()
            raise

        return device_lock

    @classmethod
    def verify_adapters(cls, robot_config: YAMConfig) -> None:
        """Raises unless each arm's interface is up and its adapter has the configured serial."""
        detected_interfaces = cls.can_interfaces()

        for arm, arm_config in robot_config.arms.items():
            interface = detected_interfaces.get(arm_config.channel, {})
            if interface.get("serial") != arm_config.adapter_serial:
                raise RuntimeError(f"{arm}: CAN adapter identity mismatch")
            if interface.get("state") not in cls.CAN_STATES:
                raise RuntimeError(f"{arm}: CAN interface is not up")

    @classmethod
    def ensure_idle(cls, channels: list[str]) -> None:
        """Raises if another program sends on one of the buses."""
        counter_paths = [Path("/sys/class/net") / channel / "statistics/tx_packets" for channel in channels]
        counts_before = [counter_path.read_text() for counter_path in counter_paths]
        time.sleep(cls.IDLE_PROBE_S)

        if any(
            counter_path.read_text() != count_before for counter_path, count_before in zip(counter_paths, counts_before)
        ):
            raise RuntimeError("CAN traffic already present; stop the existing controller first")

    @classmethod
    def check_motor_configuration(cls, robot_config: YAMConfig) -> None:
        """Raises unless both arms' limits lie within the SDK's model and every motor's type, configuration and
        communication timeout are as audited, from register reads alone; the caller owns the idle buses."""
        # the SDK: imported where it is used, by these checks alone of the robot's process
        from i2rt.motor_config_tool.dm_motor_registers import read_register
        from i2rt.motor_config_tool.utils import RawCanInterface
        from i2rt.motor_drivers.motor_check import run_startup_checks
        from i2rt.robots.get_robot import _load_joint_limits_from_xml
        from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

        joint_count = LAYOUT.joint_count
        # both arms are audited before either object that enables motors is built
        for arm in LAYOUT.arms:
            arm_config = robot_config.arms[arm]
            arm_type, gripper_type = ArmType(arm_config.arm_type), GripperType(arm_config.gripper_type)
            bounds = _load_joint_limits_from_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())[:joint_count]
            bounds = bounds + np.array([-cls.SDK_JOINT_MARGIN, cls.SDK_JOINT_MARGIN])
            if np.any(np.array(arm_config.limits.lower_bounds[:joint_count]) < bounds[:, 0]) or np.any(
                np.array(arm_config.limits.upper_bounds[:joint_count]) > bounds[:, 1]
            ):
                raise RuntimeError(f"{arm}: joint limits exceed the SDK model bounds")

            motors = list(_load_arm_config(arm_type).motor_list)
            motors.append((cls.GRIPPER_MOTOR_ID, gripper_type.get_motor_type(arm_type)))
            run_startup_checks(
                arm_config.channel,
                motors,
                check_motor_types=True,
                check_motor_config=True,
                control_mode="MIT",
                repair=False,
            )

            bus = RawCanInterface(channel=arm_config.channel, bustype="socketcan")
            try:
                if any(read_register(bus, motor_id, "TIMEOUT") == 0 for motor_id in cls.MOTOR_IDS):
                    raise RuntimeError(f"{arm}: firmware communication timeout is disabled")
            finally:
                bus.close()

    @classmethod
    def can_interfaces(cls) -> dict[str, dict[str, Any]]:
        """Returns each CAN interface of the system with its adapter's USB identity, its state and its transmit
        counter."""
        interfaces = {}

        for interface_path in sorted(Path("/sys/class/net").iterdir()):
            if (interface_path / "type").read_text().strip() != cls.CAN_LINK_TYPE:
                continue
            interfaces[interface_path.name] = cls.usb_identity(interface_path / "device") | {
                "state": (interface_path / "operstate").read_text().strip(),
                "tx_packets": int((interface_path / "statistics/tx_packets").read_text()),
            }

        return interfaces

    @classmethod
    def usb_identity(cls, device: Path) -> dict[str, str]:
        """Returns the USB identity of the device at ``device`` or of the nearest USB device above it; empty when
        none."""
        for node in (device.resolve(), *device.resolve().parents):
            if (node / "idVendor").exists():
                return {key: (node / key).read_text().strip() for key in cls.USB_KEYS if (node / key).exists()} | {
                    "usb_path": node.name
                }

        return {}
