import os
import subprocess
import time
from pathlib import Path

from pear.utils.system_utils import die_with_parent


class MPSDaemon:
    """A CUDA MPS control daemon, whose server runs the CUDA work of its Apps on one GPU side by side; it dies with the
    thread that started it, as a server's Apps do."""

    START_TIMEOUT_S = 60  # an MPS control daemon creates its socket within this
    START_POLL_S = 1  # how often its start looks for the socket

    def __init__(self, pipe_dir: Path, log_path: Path, gpu_uuid: str) -> None:
        self.pipe_dir, self.log_path, self.gpu_uuid = pipe_dir, log_path, gpu_uuid

    def start(self) -> None:
        """Starts the daemon and waits for its socket."""
        self.pipe_dir.mkdir(parents=True)

        # in the foreground the daemon and its server log here; the server opens a context on every GPU it sees, so it
        # sees only its own
        with self.log_path.open("ab") as log_file:
            popen = subprocess.Popen(
                ["nvidia-cuda-mps-control", "-f"],
                env={**self.variables(), "CUDA_VISIBLE_DEVICES": self.gpu_uuid},
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=die_with_parent,
            )

        deadline = time.monotonic() + self.START_TIMEOUT_S
        while not (self.pipe_dir / "control").exists():
            if popen.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"the MPS control daemon of {self.pipe_dir} did not start; see {self.log_path}")
            time.sleep(self.START_POLL_S)

    def variables(self) -> dict[str, str]:
        """The variables of a process whose CUDA work the daemon's server runs."""
        return {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir)}
