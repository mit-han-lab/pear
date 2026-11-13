import os
from collections.abc import Iterable
from pathlib import Path

from pear.utils.file_lock import FileLock


class DeviceLock:
    """One user's hold on a robot's devices among the processes of a machine, each device by a lock file named after
    the program, ``program_name`` (each robot names its own), and the device, from the lock's making until it is
    closed."""

    # where the lock files are: one fixed place for every process of the machine, which no variable of a process's
    # environment moves
    DIRECTORY = Path("/tmp")
    FILE_NAME = "{program_name}-{uid}-{device_name}.lock"  # a device's lock file

    def __init__(self, program_name: str, device_names: Iterable[str]) -> None:
        self.file_locks: list[FileLock] = []

        try:
            for device_name in sorted(device_names):
                file_name = self.FILE_NAME.format(program_name=program_name, uid=os.getuid(), device_name=device_name)
                self.file_locks.append(FileLock(self.DIRECTORY / file_name))
        except BaseException:  # another process holds one of the devices: none of them is held
            self.close()
            raise

    def close(self) -> None:
        """Gives every device up."""
        for file_lock in self.file_locks:
            file_lock.close()
        self.file_locks.clear()
