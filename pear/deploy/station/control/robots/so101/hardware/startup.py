from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pear.deploy.station.control.joint.device_lock import DeviceLock
from pear.deploy.station.control.robots.so101.config import HardwareConfig, SO101Config
from pear.deploy.station.control.robots.so101.layout import MOTORS

# lerobot is imported inside the methods: it is heavy, and only a command that opens the arm needs it


class StartupChecks:
    """The checks of the arm's settings, bus and servos that precede any motion, with the steps on the bus they
    share."""

    # the bits of the Feetech STS servos' Status register
    STATUS_BITS = {0x01: "voltage", 0x02: "encoder", 0x04: "temperature", 0x08: "current", 0x20: "overload"}
    GRIPPER_LOAD_BITS = 0x08 | 0x20  # a jaw squeezing an object at its torque limit may legitimately report these
    READ_RETRIES = 2  # per register read of a preflight
    POSITION_DECIMALS = 2  # of a servo's position in a preflight's report
    PREFLIGHT_REGISTERS = (  # each with whether the bus normalizes its value
        ("Present_Position", True),
        ("Torque_Enable", False),
        ("Status", False),
        ("Present_Temperature", False),
        ("Present_Voltage", False),
    )
    INVENTORY_ROOTS = {"serial": "/dev/serial/by-id", "video": "/dev/v4l/by-id"}  # stable device paths
    LOCK_PROGRAM = "so101-station"  # the program the bus's lock file is named after, every SO-101 program's

    @classmethod
    def preflight(cls, robot_config: SO101Config) -> dict[str, Any]:
        """Returns the servos' registers, their faults and whether the checks passed, read with the calibration
        checked and torque left exactly as found."""
        robot_config.require_commissioned()

        with cls.connected_bus(robot_config.hardware) as bus:
            registers = {
                name: bus.sync_read(name, normalize=is_normalized, num_retry=cls.READ_RETRIES)
                for name, is_normalized in cls.PREFLIGHT_REGISTERS
            }

        faults = cls.servo_faults(registers["Status"])
        return {
            "port": robot_config.hardware.port,
            "motors": list(MOTORS),
            "position_deg_or_percent": {
                motor: round(value, cls.POSITION_DECIMALS) for motor, value in registers["Present_Position"].items()
            },
            "torque_enabled": registers["Torque_Enable"],
            "temperature_c": registers["Present_Temperature"],
            # the servos report their voltage in tenths of a volt
            "voltage_v": {motor: value / 10 for motor, value in registers["Present_Voltage"].items()},
            "status": registers["Status"],
            "faults": faults,
            "checks": "failed" if faults else "passed",
        }

    @classmethod
    def inventory(cls) -> dict[str, dict[str, str]]:
        """Returns the serial adapters and the USB cameras by their stable paths; opens no device."""
        devices = {}
        for kind, root in cls.INVENTORY_ROOTS.items():
            directory = Path(root)
            devices[kind] = (
                {device_path.name: str(device_path.resolve()) for device_path in sorted(directory.iterdir())}
                if directory.exists()
                else {}
            )

        return devices

    @classmethod
    @contextmanager
    def connected_bus(cls, hardware_config: HardwareConfig) -> Iterator[Any]:
        """Holds the arm's bus, connected with its calibration checked, for as long as the context lasts; torque is
        left as found."""
        bus_lock = cls.lock_bus(hardware_config.port)

        try:
            follower = cls.make_follower(hardware_config)
            cls.connect_checked(follower)
            try:
                yield follower.bus
            finally:
                follower.bus.disconnect(disable_torque=False)
        finally:
            bus_lock.close()

    @classmethod
    def lock_bus(cls, port: str) -> DeviceLock:
        """Returns this process's hold on the arm's serial device, and raises when its name is unexpected or another
        process holds it."""
        device_name = Path(port).resolve().name
        if not device_name or not all(
            character.isascii() and (character.isalnum() or character in "_-") for character in device_name
        ):
            raise ValueError(f"Unexpected serial device name: {device_name!r}")

        try:
            return DeviceLock(cls.LOCK_PROGRAM, [device_name])
        except BlockingIOError as error:
            raise RuntimeError(f"{port} is owned by another {cls.LOCK_PROGRAM} process") from error

    @staticmethod
    def make_follower(hardware_config: HardwareConfig) -> Any:
        """Returns lerobot's SO-101 follower without cameras, which opens nothing yet."""
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        return SO101Follower(
            SO101FollowerConfig(
                port=hardware_config.port,
                id=hardware_config.calibration_id,
                calibration_dir=hardware_config.calibration_dir,
                use_degrees=True,
                position_p_coefficient=hardware_config.position_p_coefficient,
            )
        )

    @staticmethod
    def connect_checked(follower: Any) -> None:
        """Opens the bus, pings every servo and requires the calibration file to match the servos' registers."""
        if not follower.calibration:
            raise RuntimeError(f"No calibration file at {follower.calibration_fpath}; run pear-station so101 calibrate")

        follower.bus.connect(handshake=True)
        try:
            if not follower.bus.is_calibrated:
                raise RuntimeError("Servo calibration registers differ from the calibration file")
        except BaseException:
            follower.bus.disconnect(disable_torque=False)
            raise

    @classmethod
    def servo_faults(cls, status: dict[str, int]) -> list[str]:
        """Returns a description of each servo whose Status bits must stop the arm."""
        faults = []
        for motor, value in status.items():
            mask = value & ~cls.GRIPPER_LOAD_BITS if motor == "gripper" else value
            names = [name for bit, name in cls.STATUS_BITS.items() if mask & bit]
            if mask and not names:
                names = [f"0x{mask:02x}"]
            if names:
                faults.append(f"{motor}: {', '.join(names)}")

        return faults
