import fcntl
from pathlib import Path


class FileLock:
    """An advisory lock on a file, which one holder has at a time from its making until it is closed."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("a+")

        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # another holder has it
            self.file.close()
            raise

    def close(self) -> None:
        """Gives the lock up."""
        self.file.close()
