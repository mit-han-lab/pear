import contextlib
import os
import secrets
import tempfile
from pathlib import Path


class BrowserToken:
    """The secret in a run directory that logs a browser in to its service, readable by its owner only."""

    FILE_NAME = "browser-token"  # in the run directory
    RANDOM_BYTE_COUNT = 32  # of randomness in a new token
    SHARED_BITS = 0o077  # group and other permissions, which the token file must not have

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / self.FILE_NAME

    def create(self) -> str:
        """Returns the directory's token, written first when the directory has none."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, draft = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.FILE_NAME}-")  # owner-only

        try:
            with os.fdopen(descriptor, "w") as token_file:
                token_file.write(secrets.token_urlsafe(self.RANDOM_BYTE_COUNT))
            # the link shows the token whole, and never replaces one that another service put there first
            with contextlib.suppress(FileExistsError):
                os.link(draft, self.path)
        finally:
            os.unlink(draft)

        if self.path.stat().st_mode & self.SHARED_BITS:
            raise RuntimeError(f"{self.path} is open to other users; remove it and start the GUI service again")
        return self.read()

    def read(self) -> str:
        """Returns the token a running service was started with."""
        token = self.path.read_text().strip()
        if not token:
            raise RuntimeError(f"{self.path} is empty; remove it and start the GUI service again")

        return token
