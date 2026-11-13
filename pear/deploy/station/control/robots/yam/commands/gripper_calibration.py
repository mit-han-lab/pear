import json
import logging
import math
import signal
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.sdk_adapter import (
    MIT_FRAME_HEAD,
    MIT_MOTOR_OFF,
    MIT_MOTOR_ON,
    OneAttemptInterface,
    SDKCompatibility,
)
from pear.deploy.station.control.robots.yam.hardware.startup import StartupChecks
from pear.deploy.station.control.robots.yam.layout import ARM_TYPE, GRIPPER_TYPE
from pear.utils.file_utils import write_text_atomically

CALIBRATION_TORQUE = 0.5  # the calibration's torque limit, Nm
CALIBRATION_DAMPING = 0.5  # the damping of the calibration's commands


class GripperCalibrationMotor:
    """Synchronous I/O with the gripper's motor alone, through the SDK's interface of one motor, with no SDK robot
    and no thread."""

    MOTOR_TYPE = "DM4310"  # the gripper motor's model
    SEND_TIMEOUT_S = 0.05  # of one frame
    FRAME_LENGTH = 8  # bytes of a frame of the MIT protocol

    def __init__(self, robot_config: YAMConfig, arm: str) -> None:
        arm_config = robot_config.arms[arm]
        if (arm_config.arm_type, arm_config.gripper_type) != (ARM_TYPE, GRIPPER_TYPE):
            raise ValueError("This commissioning routine supports original Ultra + linear_4310")
        StartupChecks.verify_sdk_revision()
        # the SDK: imported once its revision is verified
        from i2rt.motor_config_tool.dm_motor_registers import read_register
        from i2rt.motor_config_tool.utils import RawCanInterface
        from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
        from i2rt.motor_drivers.motor_check import run_startup_checks

        self.driver: Any = None  # the SDK's interface of the gripper's motor, once it is made
        self.frames: list[list[int]] = []  # every frame the calibration sent
        self.device_lock = StartupChecks.claim_buses(robot_config)

        try:
            run_startup_checks(
                arm_config.channel,
                [(StartupChecks.GRIPPER_MOTOR_ID, self.MOTOR_TYPE)],
                check_motor_types=True,
                check_motor_config=True,
                control_mode="MIT",
                repair=False,
            )
            bus = RawCanInterface(channel=arm_config.channel, bustype="socketcan")
            try:
                self.timeout_register = read_register(bus, StartupChecks.GRIPPER_MOTOR_ID, "TIMEOUT")
                if not self.timeout_register:
                    raise RuntimeError("Gripper firmware communication timeout is disabled")
            finally:
                bus.close()

            interface_class = SDKCompatibility.adapted(OneAttemptInterface, DMSingleMotorCanInterface)
            self.driver = interface_class(channel=arm_config.channel)
            self.driver.bus.send = self.send_guard(self.driver.bus.send, self.frames)
            # the SDK's public method can enter its error recovery, which the interface refuses; no chain is built, so
            # the arm's other motors stay untouched; it changes the root logger's level, which is put back after it
            root_level = logging.getLogger().level
            try:
                self.initial_feedback = self.driver.motor_on(StartupChecks.GRIPPER_MOTOR_ID, self.MOTOR_TYPE)
            finally:
                logging.getLogger().setLevel(root_level)
        except BaseException:
            self.close()
            raise

    def command(self, torque: float) -> Any:
        """Commands the gripper's motor with ``torque`` alone, damped, and returns its feedback."""
        if not math.isfinite(torque) or abs(torque) > CALIBRATION_TORQUE:
            raise ValueError(f"Calibration torque must be within +/-{CALIBRATION_TORQUE} Nm")

        return self.driver.set_control(
            StartupChecks.GRIPPER_MOTOR_ID, self.MOTOR_TYPE, 0.0, 0.0, 0.0, CALIBRATION_DAMPING, torque
        )

    def close(self) -> None:
        """Powers the gripper's motor off, closes its interface and releases the bus."""
        try:
            if self.driver is not None:
                try:
                    # explicit, unlike a chain's close, which closes its socket alone
                    self.driver.motor_off(StartupChecks.GRIPPER_MOTOR_ID)
                finally:
                    self.driver.close()
                    self.driver = None
        finally:
            self.device_lock.close()

    @classmethod
    def send_guard(cls, send: Callable[..., Any], frames: list[list[int]]) -> Callable[..., Any]:
        """Returns ``send`` restricted to the gripper's frames, each sent within a time limit and kept in ``frames``."""

        def guarded_send(message: Any, *args: Any, **kwargs: Any) -> Any:
            """Sends ``message`` if it is a frame of the gripper's motor that clears no error and writes no encoder."""
            data = bytes(message.data)
            if (
                message.arbitration_id != StartupChecks.GRIPPER_MOTOR_ID
                or message.is_extended_id
                or len(data) != cls.FRAME_LENGTH
            ):
                raise RuntimeError(f"Calibration may command only gripper motor {StartupChecks.GRIPPER_MOTOR_ID}")
            if data[:-1] == MIT_FRAME_HEAD and data[-1] not in (MIT_MOTOR_ON, MIT_MOTOR_OFF):
                raise RuntimeError("Calibration refuses error clearing and encoder writes")

            kwargs["timeout"] = cls.SEND_TIMEOUT_S
            result = send(message, *args, **kwargs)
            frames.append(list(data))

            return result

        return guarded_send


class GripperMeasurement:
    """The closed and the open endpoint of a gripper, found with a low torque and damping."""

    OK_ERROR_CODE = "0x1"  # the error code of a motor without an error, as the SDK's feedback spells it
    SEEK_DEADLINE_S = 20.0  # for one endpoint
    RAMP_S = 0.3  # the torque ramps up over this time
    MAX_INTERVAL_S = 0.1  # the longest gap between two control steps, and the longest wait for a reply
    MAX_VELOCITY = 1.8  # rad/s
    MAX_TRAVEL = 7.2  # rad from the initial position
    MAX_TEMPERATURE_C = 65  # of the motor's driver and of its rotor
    STABLE_WINDOW_S = 0.4  # how long a position must hold still to be an endpoint
    MIN_SEEK_S = 0.8  # the shortest seek of an endpoint
    STABLE_SPREAD = 0.01  # rad over the window
    STABLE_VELOCITY = 0.05  # rad/s
    STEP_S = 0.02  # of one control step

    @classmethod
    def measure(
        cls, motor: GripperCalibrationMotor, samples: list[dict[str, Any]], *, stroke_range: list[float]
    ) -> list[float]:
        """Returns the gripper's closed and open endpoint in motor radians; ``samples`` receives every step."""
        closed_position = cls.seek_endpoint(motor, 1, samples)
        open_position = cls.seek_endpoint(motor, -1, samples)

        # the SDK's motor_stroke is a constant of its force conversion, no bound of acceptance: the stroke must fall in
        # the range measured on the real hardware
        if not stroke_range[0] <= closed_position - open_position <= stroke_range[1]:
            raise RuntimeError(f"Measured gripper stroke {closed_position - open_position:.3f} rad is implausible")

        return [closed_position, open_position]

    @classmethod
    def seek_endpoint(cls, motor: GripperCalibrationMotor, sign: int, samples: list[dict[str, Any]]) -> float:
        """Pushes the gripper toward one end with a torque that ramps up and returns the position it rests at; a seek
        that times out yields no endpoint."""
        start_time = time.monotonic()
        previous_step_time = start_time
        initial_position = motor.initial_feedback.position
        recent_samples = deque()  # the (time, position) of the latest window's replies

        while time.monotonic() - start_time < cls.SEEK_DEADLINE_S:
            now = time.monotonic()
            if now - previous_step_time > cls.MAX_INTERVAL_S:
                raise RuntimeError(f"Calibration control interval exceeded {cls.MAX_INTERVAL_S * 1000:.0f} ms")
            torque = sign * CALIBRATION_TORQUE * min((now - start_time) / cls.RAMP_S, 1.0)
            state = motor.command(torque)
            reply_time = time.monotonic()
            if reply_time - now > cls.MAX_INTERVAL_S:
                raise RuntimeError(f"Calibration feedback exceeded {cls.MAX_INTERVAL_S * 1000:.0f} ms")
            values = (state.position, state.velocity, state.torque)
            if (
                state.id != StartupChecks.GRIPPER_MOTOR_ID
                or state.error_code != cls.OK_ERROR_CODE
                or not all(map(math.isfinite, values))
            ):
                raise RuntimeError("Invalid gripper feedback")
            if abs(state.velocity) > cls.MAX_VELOCITY or abs(state.position - initial_position) > cls.MAX_TRAVEL:
                raise RuntimeError("Gripper exceeded commissioning speed or travel")
            if max(state.temperature_mos, state.temperature_rotor) > cls.MAX_TEMPERATURE_C:
                raise RuntimeError("Gripper temperature exceeds commissioning limit")

            samples.append(
                {
                    "phase": "close" if sign > 0 else "open",
                    "elapsed": reply_time - start_time,
                    "position": state.position,
                    "velocity": state.velocity,
                    "torque": state.torque,
                }
            )
            recent_samples.append((reply_time, state.position))
            while len(recent_samples) > 1 and reply_time - recent_samples[1][0] >= cls.STABLE_WINDOW_S:
                recent_samples.popleft()
            positions = [position for _, position in recent_samples]
            if (
                reply_time - start_time >= cls.MIN_SEEK_S
                and reply_time - recent_samples[0][0] >= cls.STABLE_WINDOW_S
                and max(positions) - min(positions) < cls.STABLE_SPREAD
                and abs(state.velocity) < cls.STABLE_VELOCITY
            ):
                return sum(positions) / len(positions)

            previous_step_time = now
            time.sleep(max(0.0, cls.STEP_S - (time.monotonic() - now)))

        raise RuntimeError(f"Gripper did not reach a stable endpoint within {cls.SEEK_DEADLINE_S:g} s")


class GripperCalibration:
    """The supervised calibration of one gripper: its endpoints measured, and a record of the run kept whatever its
    outcome."""

    RUN_MARGIN_S = 4.0  # of the whole run, past the seeks of its two endpoints
    RUN_DEADLINE_S = 2 * GripperMeasurement.SEEK_DEADLINE_S + RUN_MARGIN_S  # of the whole run, by the wall clock
    CLEANUP_DEADLINE_S = 1.0  # of the motor's power-off once the run ended
    JSON_INDENT = 2  # of the record

    @classmethod
    def run(cls, robot_config: YAMConfig, arm: str, output_dir: Path) -> dict[str, Any]:
        """Measures the endpoints of ``arm``'s gripper, powering that gripper alone, keeps a record under
        ``output_dir`` and returns the endpoints with the record's path."""
        arm_config = robot_config.arms[arm]
        stroke_range = arm_config.gripper_stroke_range
        if stroke_range is None:
            raise ValueError("Set gripper_stroke_range from verified hardware before calibration")

        output_dir.mkdir(parents=True, exist_ok=True)
        recorded_at = datetime.now(UTC)
        path = output_dir / f"gripper-{arm}-{recorded_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        record = {
            "side": arm,
            "channel": arm_config.channel,
            "adapter_serial": arm_config.adapter_serial,
            "recorded_at": recorded_at.isoformat(),
            "sdk_commit": StartupChecks.SDK_COMMIT,
            "success": False,
            "samples": [],
            "control_frames": [],
            "coordinate": "raw motor radians; fixed zero software offset",
            "torque_limit_nm": CALIBRATION_TORQUE,
            "damping": CALIBRATION_DAMPING,
            "stroke_range": stroke_range,
        }
        motor = None

        def raise_deadline(*_: Any) -> None:
            """Raises the timeout of the run: the alarm's handler."""
            raise TimeoutError("Gripper calibration exceeded its wall-clock deadline")

        previous_handler = signal.signal(signal.SIGALRM, raise_deadline)
        signal.setitimer(signal.ITIMER_REAL, cls.RUN_DEADLINE_S)
        try:
            try:
                motor = GripperCalibrationMotor(robot_config, arm)
                record["firmware_timeout_register"] = motor.timeout_register
                record["control_frames"] = motor.frames
                record["gripper_limits"] = GripperMeasurement.measure(
                    motor, record["samples"], stroke_range=stroke_range
                )
            finally:
                # the cleanup keeps a time limit, after a timeout or a Ctrl+C too
                signal.setitimer(signal.ITIMER_REAL, cls.CLEANUP_DEADLINE_S)
                if motor is not None:
                    motor.close()
            record["success"] = True
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            write_text_atomically(path, json.dumps(record, indent=cls.JSON_INDENT, allow_nan=False) + "\n")

        return {"side": arm, "gripper_limits": record["gripper_limits"], "record": str(path)}
