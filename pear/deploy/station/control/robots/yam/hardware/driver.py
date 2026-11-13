from typing import Any

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.driver import JointDriver
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.arm_state import YAMArmState
from pear.deploy.station.control.robots.yam.hardware.sdk_worker import SDKRequest, SDKWorker
from pear.utils.worker_process import WorkerProcess


class YAMDriver(JointDriver):
    """Both YAM arms as the control loop reads and commands them, through the process that holds the SDK: a call
    that is not answered in time ends that process, so that no SDK target keeps running."""

    # the SDK keeps holding the last command by itself: a HOLD renews nothing, and a snapshot gets no telemetry
    START_TIMEOUT_S = SDKWorker.STARTUP_DEADLINE_S + 1  # for the worker's first message: its own limit and a margin
    CALL_TIMEOUT_S = 0.2  # for the worker's answer to a read, a recovery's read or a send
    CLOSE_TIMEOUT_S = SDKWorker.CLOSE_DEADLINE_S + 0.5  # for its answer to a close: its own limit and a margin
    CLOSE_GRACE_S = 0.5  # after its answer to a close, for the worker to end by itself
    EXIT_TIMEOUT_S = 1  # after SIGTERM, and again after SIGKILL, for the worker to end

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        robot_config.require_commissioned()  # before the worker is started or the SDK imported
        self.connection, child_connection = WorkerProcess.CONTEXT.Pipe()
        self.worker = WorkerProcess(
            SDKWorker.main, (child_connection, robot_config), "yam-sdk", self.EXIT_TIMEOUT_S, is_group_leader=False
        )
        self.is_closed = False  # once the worker was ended

        self.worker.start()
        child_connection.close()
        try:
            self._receive(self.START_TIMEOUT_S)
        except BaseException:
            self._terminate(0)
            raise

    def read(self) -> dict[str, YAMArmState]:
        """Returns both arms' latest readings."""
        return self._call(SDKRequest.READ)

    def read_for_recovery(self) -> dict[str, YAMArmState]:
        """Returns both arms' readings once the worker found the SDK's loops running and no motor in error."""
        return self._call(SDKRequest.READ_FOR_RECOVERY)

    def send(
        self, targets: dict[str, tuple[float, ...]], velocities: dict[str, tuple[float, ...]] | None = None
    ) -> None:
        """Commands both arms' targets with their planned velocities."""
        self._call(SDKRequest.SEND, (targets, velocities))

    def close(self) -> None:
        """Has the worker power the motors off, then ends it within the time limits."""
        if not self.is_closed:
            try:
                self._call(SDKRequest.CLOSE)
            finally:
                self._terminate(self.CLOSE_GRACE_S)

    def _call(self, request: SDKRequest, value: Any = None) -> Any:
        """Sends one request to the worker and returns its answer; a broken pipe ends the worker."""
        if self.is_closed:
            raise RuntimeError("SDK worker is closed")

        try:
            self.connection.send((request, value))
            return self._receive(self.CLOSE_TIMEOUT_S if request == SDKRequest.CLOSE else self.CALL_TIMEOUT_S)
        except (EOFError, BrokenPipeError, OSError) as error:
            self._terminate(0)
            raise RuntimeError("SDK worker exited; firmware timeout must stop actuation") from error

    def _receive(self, timeout_s: float) -> Any:
        """Returns the worker's next answer; no answer within ``timeout_s`` ends the worker."""
        if not self.connection.poll(timeout_s):
            self._terminate(0)
            raise RuntimeError("SDK worker timed out and was terminated")

        is_success, value = self.connection.recv()
        if not is_success:
            raise RuntimeError(value)

        return value

    def _terminate(self, grace_s: float) -> None:
        """Ends the worker, which may first end by itself for ``grace_s``, and closes the pipe."""
        self.worker.stop(grace_s)
        self.connection.close()
        self.is_closed = True
