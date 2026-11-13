import dataclasses
import time
from collections.abc import Callable, Sequence

from pear.deploy.station.control.joint.arm_state import ArmState
from pear.deploy.station.control.joint.config import ControlConfig, RobotConfig
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.robots.so101.config import SO101Config
from pear.deploy.station.control.robots.so101.hardware.arm_state import SO101ArmState
from pear.deploy.station.control.robots.so101.hardware.driver import SO101Driver
from pear.deploy.station.control.robots.so101.layout import VECTOR_LENGTH
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState

STALE_S = 1.0  # the age of a stale reading of a stand-in's arm
SENT_KEPT = 5000  # the most recent targets a stand-in keeps for a test to inspect


class StandInJointDriver(JointDriver):
    """A robot's joints without a device: every arm reaches each command at once, and a test makes the readings
    stale, makes an arm's bus fail and reads back what was sent."""

    START_GRIPPER_OPENING = 0.5  # each gripper's opening before the first command, with every joint at zero
    ARM_STATE_CLASS: type[ArmState] = ArmState  # one arm's reading: the robot's own class, with its telemetry

    def __init__(self, control_config: ControlConfig, robot_config: RobotConfig) -> None:
        layout = robot_config.LAYOUT
        start_position = (0.0,) * layout.joint_count + (self.START_GRIPPER_OPENING,)
        self.positions = dict.fromkeys(layout.arms, start_position)  # each arm's reading, by the arm's name
        self.is_closed = False
        self.is_stale = False  # whether every reading is STALE_S old
        self.failed_arm: str | None = None  # the arm whose bus fails every read; None: no bus fails
        self.sent_targets: list[dict[str, tuple[float, ...]]] = []  # the targets commanded, the latest last
        # the planned velocities of the last command, by arm; None: it held its targets
        self.velocities: dict[str, tuple[float, ...]] | None = None

    def read(self) -> dict[str, ArmState]:
        """Returns every arm where its last command put it, at rest."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")
        if self.failed_arm:
            raise RuntimeError(f"{self.failed_arm}: stand-in bus failure")

        sampled_monotonic_s = time.monotonic() - (STALE_S if self.is_stale else 0)

        return {
            arm: self.ARM_STATE_CLASS(position, (0.0,) * len(position), sampled_monotonic_s)
            for arm, position in self.positions.items()
        }

    def read_for_recovery(self) -> dict[str, ArmState]:
        """Returns every arm's reading: the joints are always healthy enough to hold again."""
        return self.read()

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Puts every arm at its target at once and keeps the targets and the planned velocities."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")

        self.velocities = None if velocities is None else {arm: tuple(value) for arm, value in velocities.items()}
        self.positions = {arm: tuple(target) for arm, target in targets.items()}
        self.sent_targets.append(self.positions.copy())
        if len(self.sent_targets) > SENT_KEPT:
            self.sent_targets.pop(0)

    def close(self) -> None:
        """Ends the readings and the commands."""
        self.is_closed = True


class StandInYAMDriver(StandInJointDriver):
    """Both YAM arms without their SDK: readings carry the SDK's fields of the force-limited gripper, unset."""

    ARM_STATE_CLASS = YAMArmState


class StandInSO101Arm:
    """The SO-101's arm without its bus: it reaches every command at once, and a test makes its readings stale, its
    reads fail, its jaw stop on an object or the arm answer a command its own way."""

    def __init__(self, control_config: ControlConfig, robot_config: SO101Config) -> None:
        home = robot_config.home
        start_position = home.position if home is not None else (0.0,) * VECTOR_LENGTH
        self.position = tuple(float(value) for value in start_position)  # the arm's reading: Home, until a command
        self.is_closed = False
        self.is_stale = False  # whether every reading is STALE_S old
        self.failure: BaseException | None = None  # what every read raises while it is set
        self.blocked_gripper_opening: float | None = None  # where the jaw stops while it closes; None: nowhere
        # what the arm reaches for a commanded target from its position, a sag model for example; None: the target
        self.response: Callable[[tuple[float, ...], tuple[float, ...]], Sequence[float]] | None = None
        self.sent_targets: list[tuple[float, ...]] = []  # the targets commanded, the latest last
        self.read_count = 0  # the reads so far, which number the readings

    def read(self) -> SO101ArmState:
        """Returns the arm where its last command put it, at rest."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")
        if self.failure is not None:
            raise self.failure

        self.read_count += 1

        return SO101ArmState(
            self.position,
            (0.0,) * VECTOR_LENGTH,
            time.monotonic() - (STALE_S if self.is_stale else 0),
            self.read_count,
        )

    def read_for_recovery(self) -> SO101ArmState:
        """Returns the arm's reading: its servos are always healthy and hold their torque."""
        return self.read()

    def send(self, target: Sequence[float]) -> None:
        """Puts the arm where ``target`` takes it, at once, and keeps the target."""
        if self.is_closed:
            raise RuntimeError("Hardware is closed")

        position = [float(value) for value in target]
        if self.response is not None:
            position = [float(value) for value in self.response(tuple(position), self.position)]
        if self.blocked_gripper_opening is not None:
            position[-1] = max(position[-1], self.blocked_gripper_opening)
        self.position = tuple(position)

        self.sent_targets.append(tuple(float(value) for value in target))
        if len(self.sent_targets) > SENT_KEPT:
            self.sent_targets.pop(0)

    def close(self) -> None:
        """Ends the readings and the commands."""
        self.is_closed = True


class StandInSO101Driver(SO101Driver):
    """The SO-101's own driver over the stand-in arm, by default without the feedforward."""

    HARDWARE_CLASS = StandInSO101Arm
    # whether the configured feedforward drives the arm: an offset tuned for the real servos pushes an arm that
    # reaches every command exactly off its target
    USES_FEEDFORWARD = False

    def __init__(self, control_config: ControlConfig, robot_config: SO101Config) -> None:
        if not self.USES_FEEDFORWARD:
            robot_config = dataclasses.replace(robot_config, feedforward=None)

        super().__init__(control_config, robot_config)
