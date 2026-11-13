from typing import TYPE_CHECKING

from pear.deploy.station.control.schemas import BriefStatus, RobotStatus
from pear.deploy.station.gui.console.periodic_thread import PeriodicThread
from pear.deploy.station.gui.console.phase import STOPPED_STATUSES, Phase
from pear.deploy.station.records.recording.encode_worker import SegmentStatus

if TYPE_CHECKING:  # an annotation only: its module imports this one
    from pear.deploy.station.gui.console.console import Console


class Watchdog:
    """A background check that pauses the robot when the browser that holds control, the recording, the cameras or
    the control fail."""

    PERIOD_S = 0.1  # between two checks
    THREAD_NAME = "gui-watchdog"  # of the thread that checks
    PHASES_WITHOUT_CAMERAS = (Phase.RECONNECTING_CAMERAS, Phase.RECOVERING_CONTROL)  # need no camera feeds

    def __init__(self, console: "Console") -> None:
        self.console = console
        self.periodic_thread = PeriodicThread(
            self._check, self.PERIOD_S, self.THREAD_NAME, is_waiting_first=True, is_ended_by_failure=True
        )

    def close(self) -> None:
        """Stops checking."""
        self.periodic_thread.close()

    def _check(self) -> None:
        """Checks the browser, the recording, the cameras and the robot once."""
        console = self.console
        lost_owner = console.lease.lost_owner()
        if lost_owner is not None:
            self._release_lost_browser(lost_owner)

        robot_state = console.robot.status()
        segment = console.recordings.active_segment
        if segment and segment.snapshot()["status"] == SegmentStatus.FAILED:
            console.error = "Recording failed: " + str(segment.snapshot()["error"])
            console.stop(reason=console.error)

        if robot_state["status"] == RobotStatus.FAULT:
            console.save_control_diagnostic()

        if (
            console.is_busy
            and not console.cancellation.is_set()
            and console.phase not in self.PHASES_WITHOUT_CAMERAS
            and console.preview.is_encoding
            and console.preview.error_text
        ):
            console.stop(reason="Camera unavailable; restore the feed before resuming.")

        if robot_state["status"] == RobotStatus.FAULT:
            robot_state = self._stop_for_fault()

        # under the lock a phase a command shows meanwhile is not replaced by the idle one
        with console.lock:
            if console.phase == Phase.STOPPING and not console.is_busy and robot_state["status"] in STOPPED_STATUSES:
                idle_phase = Phase.idle_for(
                    robot_state["status"],
                    is_latched=console.is_stop_latched,
                    has_task=console.task_run.task is not None,
                )
                console.set_phase(idle_phase)

    def _release_lost_browser(self, owner: str) -> None:
        """Pauses the operation of a browser that went silent, then frees control for the next one."""
        console = self.console
        # camera maintenance has no motor or task side effects: a lost browser must not cancel it and leave every
        # camera reader closed
        if console.phase != Phase.RECONNECTING_CAMERAS:
            console.stop(reason="Control connection lost; resume explicitly after reconnecting.")
        console.lease.release(owner)

    def _stop_for_fault(self) -> BriefStatus:
        """Pauses for a robot fault and shows it, unless recovery is under way; returns the robot's status as read
        under the lock."""
        console = self.console

        with console.lock:
            # recovery may have completed after the earlier status read
            robot_state = console.robot.status()
            if robot_state["status"] == RobotStatus.FAULT and console.phase not in (
                Phase.FAULT,
                Phase.RECOVERING_CONTROL,
            ):
                console.stop(reason=robot_state.get("fault") or "Robot fault")
                console.error = str(robot_state.get("fault") or "Robot fault")
                console.set_phase(Phase.FAULT)

        return robot_state
