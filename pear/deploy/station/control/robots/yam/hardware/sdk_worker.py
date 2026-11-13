import faulthandler
import os
import signal
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.hardware.arms import YAMHardware
from pear.utils.logging_utils import get_logger

logger = get_logger(__name__, default_level="INFO")


class SDKRequest(StrEnum):
    """What the robot's process asks of the SDK's worker, by the word a request carries."""

    READ = "read"
    READ_FOR_RECOVERY = "read_for_recovery"
    SEND = "send"
    CLOSE = "close"


class SDKWorker:
    """The process that holds the SDK and both arms, so that the SDK's threads, a failed build and a stalled robot's
    process can never leave the motors running."""

    STARTUP_DEADLINE_S = 30  # for building both arms, after which the process exits
    REQUEST_DEADLINE_S = 1.0  # for one turn of the loop, waiting for a request and answering it included
    HEARTBEAT_S = 0.5  # the longest the robot's process may stay silent before the motors are powered off
    CLOSE_DEADLINE_S = 4.0  # for powering the motors off

    def __init__(self, connection: Connection, robot_config: YAMConfig) -> None:
        self.connection = connection
        self.robot_config = robot_config
        self.hardware: YAMHardware | None = None  # both arms, once they are built
        self.last_error_response: str | None = None  # of the request that failed last; None once one succeeds

    @staticmethod
    def main(connection: Connection, robot_config: YAMConfig) -> None:
        """The target of the worker's process."""
        SDKWorker(connection, robot_config).run()

    def run(self) -> None:
        """Builds both arms, answers requests until the robot's process stops asking, then powers the motors off and
        exits."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            faulthandler.dump_traceback_later(self.STARTUP_DEADLINE_S, exit=True)
            self.hardware = YAMHardware(self.robot_config)
            self.connection.send((True, None))
            self.serve()
        except BaseException as error:
            logger.exception("The SDK's worker failed")
            try:
                self.connection.send((False, f"{type(error).__name__}: {error}"))
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            faulthandler.dump_traceback_later(self.CLOSE_DEADLINE_S, exit=True)
            try:
                if self.hardware is not None:
                    self.hardware.close()
            finally:
                self.connection.close()
                # the SDK may leave threads that are no daemons after a partial build: exiting here closes every CAN
                # socket and releases the bus locks
                os._exit(0)

    def serve(self) -> None:
        """Answers one request after another; a silent robot's process or its close ends the loop."""
        while True:
            faulthandler.dump_traceback_later(self.REQUEST_DEADLINE_S, exit=True)
            if not self.connection.poll(self.HEARTBEAT_S):
                break

            request, value = self.connection.recv()
            if request == SDKRequest.CLOSE:
                faulthandler.dump_traceback_later(self.CLOSE_DEADLINE_S, exit=True)
                self.hardware.close()
                self.hardware = None
                self.connection.send((True, None))
                break
            try:
                self.connection.send((True, self.dispatch(request, value)))
                self.last_error_response = None
            except Exception as error:
                error_response = f"{type(error).__name__}: {error}"
                if error_response != self.last_error_response:  # a failure repeated every control period, once
                    logger.exception("SDK request %s failed", request)
                self.last_error_response = error_response
                self.connection.send((False, error_response))

    def dispatch(self, request: SDKRequest, value: Any) -> Any:
        """Returns the answer to one request of the arms."""
        if request == SDKRequest.READ:
            return self.hardware.read()
        if request == SDKRequest.READ_FOR_RECOVERY:
            return self.hardware.read_for_recovery()
        if request == SDKRequest.SEND:
            return self.hardware.send(*value)

        raise ValueError(f"Unknown SDK operation: {request}")
