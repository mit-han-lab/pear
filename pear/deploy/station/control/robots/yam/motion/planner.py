from __future__ import annotations

import signal
import threading
import time
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

from pear.deploy.station.control.joint.config import ControlConfig
from pear.deploy.station.control.joint.trajectory import JointMove
from pear.deploy.station.control.motion.actions import StrictModel
from pear.deploy.station.control.motion.planning import PreparedAction
from pear.deploy.station.control.robot import BaseRobot
from pear.deploy.station.control.robots.yam.config import YAMConfig
from pear.deploy.station.control.robots.yam.motion.actions import GripperAction
from pear.deploy.station.control.robots.yam.motion.kinematics import YAMKinematics
from pear.deploy.station.control.robots.yam.motion.planning import YAMPlanning
from pear.deploy.station.control.schemas import HomePlan, Observation
from pear.utils.logging_utils import get_logger
from pear.utils.worker_process import WorkerProcess

logger = get_logger(__name__, default_level="INFO")


class PlanningRequest(StrEnum):
    """What the robot's process asks of the planner, by the word a request carries."""

    PLAN = "plan"
    HOME = "home"
    HOME_RECHECK = "home_recheck"


class PlanningWorker:
    """The planner process: it plans the requests it receives, and has no hardware."""

    def __init__(self, connection: Connection, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        self.connection = connection
        self.control_config, self.robot_config = control_config, robot_config
        self.kinematics: YAMKinematics | None = None  # made at the first request that needs them

    @classmethod
    def main(cls, connection: Connection, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        """The target of the planner's process."""
        cls(connection, control_config, robot_config).run()

    def run(self) -> None:
        """Answers one request after another until the robot's process sends None or closes the pipe."""
        # the robot's process owns cancellation and closes pending requests itself
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            self.connection.send((True, None))
            while True:
                request = self.connection.recv()
                if request is None:
                    break
                kind, observation, payload = request
                try:
                    self.connection.send((True, self.handle(kind, observation, payload)))
                except Exception as error:
                    logger.exception("Planning request %s failed", kind)
                    self.connection.send((False, error))
        except (EOFError, BrokenPipeError):
            pass
        except Exception as error:
            logger.exception("The planner failed")
            try:
                self.connection.send((False, error))
            except (EOFError, BrokenPipeError):
                pass
        finally:
            self.connection.close()

    def handle(self, kind: PlanningRequest, observation: Observation, payload: Any) -> Any:
        """Returns what the request ``kind`` asks for of ``observation``."""
        if kind == PlanningRequest.HOME:
            return YAMPlanning.checked_home(self.control_config, self.robot_config, observation, *payload)
        if kind == PlanningRequest.HOME_RECHECK:
            return YAMPlanning.recheck_home_move(self.robot_config, observation, payload)
        if kind == PlanningRequest.PLAN:
            if self.kinematics is None and not isinstance(payload, GripperAction):
                self.kinematics = YAMKinematics(self.robot_config)
            return YAMPlanning.checked_action(self.robot_config, self.kinematics, observation, payload)

        raise ValueError(f"Unknown planning request: {kind}")


class PlannerProcess:
    """The planner process as the robot's process uses it: one request at a time, bounded in time, bound to its
    observation, and never retried."""

    STARTUP_TIMEOUT_S = 30  # for the planner's first message
    REQUEST_TIMEOUT_S = 30  # the longest a request is waited for, whatever time its caller gives
    STOP_JOIN_S = 1  # for the planner to end: once it is told to, after SIGTERM and after SIGKILL

    def __init__(self, control_config: ControlConfig, robot_config: YAMConfig) -> None:
        self.connection, child_connection = WorkerProcess.CONTEXT.Pipe()
        self.worker = WorkerProcess(
            PlanningWorker.main,
            (child_connection, control_config, robot_config),
            "yam-planner",
            self.STOP_JOIN_S,
            is_group_leader=False,
        )
        self._lock = threading.Lock()  # held by the one request in flight
        self.is_closed = False  # after a failure, or once the robot closed it

        try:
            self.worker.start()
            child_connection.close()
            if not self.connection.poll(self.STARTUP_TIMEOUT_S):
                raise TimeoutError("Planner startup timed out")
            is_success, value = self.connection.recv()
            if not is_success:
                raise value
        except BaseException:
            child_connection.close()
            self.close()
            raise

    def plan(
        self, robot: BaseRobot, observation: Observation, action: StrictModel, *, timeout_s: float = REQUEST_TIMEOUT_S
    ) -> PreparedAction:
        """Returns the checked plan of ``action`` from the hold of ``observation``, within ``timeout_s``."""
        return self._request(robot, observation, action, timeout_s, PlanningRequest.PLAN)

    def plan_home(
        self,
        robot: BaseRobot,
        observation: Observation,
        side: str,
        *,
        opens_grippers: bool,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> tuple[HomePlan, JointMove | None]:
        """Returns the plan of ``side``'s direct move Home from the hold of ``observation`` and the move, within
        ``timeout_s``."""
        return self._request(robot, observation, (side, opens_grippers), timeout_s, PlanningRequest.HOME)

    def recheck_home(
        self, robot: BaseRobot, observation: Observation, move: JointMove, *, timeout_s: float = REQUEST_TIMEOUT_S
    ) -> None:
        """Raises unless the approved Home ``move`` is still valid with the feedback of ``observation``."""
        self._request(robot, observation, move, timeout_s, PlanningRequest.HOME_RECHECK)

    def close(self) -> None:
        """Tells the planner to end and ends it, within the time limits; a planner that survives SIGKILL raises."""
        if self.is_closed:
            return
        self.is_closed = True

        try:
            if self.worker.process.pid is not None:  # a planner whose start failed is sent nothing
                try:
                    self.connection.send(None)
                except (OSError, EOFError):
                    pass
                self.worker.stop(self.STOP_JOIN_S)
                if self.worker.is_alive():
                    raise RuntimeError("Planner could not be killed")
        finally:
            self.connection.close()

    def _request(
        self, robot: BaseRobot, observation: Observation, payload: Any, timeout_s: float, kind: PlanningRequest
    ) -> Any:
        """Sends one request and returns its answer while ``observation`` stays the hold of ``robot``; a request
        abandoned on the way closes the planner."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Planner already has a request")

        has_pending_request = False
        try:
            if self.is_closed:
                raise RuntimeError("Planner is closed; reopen the station")
            robot.verify_observation(observation)
            observation_data = {key: observation[key] for key in ("joint_positions", "control_snapshot")}
            self.connection.send((kind, observation_data, payload))
            has_pending_request = True

            deadline = time.monotonic() + min(timeout_s, self.REQUEST_TIMEOUT_S)
            while True:
                robot.verify_observation(observation)
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError("Offline planning timed out")
                if self.connection.poll(min(robot.control_period_s, remaining_s)):
                    is_success, value = self.connection.recv()
                    has_pending_request = False
                    if not is_success:
                        raise value
                    robot.verify_observation(observation)
                    return value
        except BaseException:
            # an abandoned request's late answer must never reach a later request
            if has_pending_request:
                self.close()
            raise
        finally:
            self._lock.release()
