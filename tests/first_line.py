import os
import select
import subprocess
import time

import pytest


class FirstLine:
    """The output of a test's process up to its first line that holds a text, read within a time limit."""

    PIPE_READ_BYTES = 65536  # the most one read of a process's pipe takes

    @classmethod
    def of(cls, process: subprocess.Popen, limit_s: float, text: str = "") -> str:
        deadline = time.monotonic() + limit_s
        output = b""

        # from the pipe itself, what it holds once it holds something: a read of a line waits for the line's end
        # without a limit, and a line the stream has buffered does not show in the wait for the next one
        while not any(text.encode() in line for line in output.split(b"\n")[:-1]):
            is_readable = select.select([process.stdout], [], [], max(deadline - time.monotonic(), 0.0))[0]
            chunk = os.read(process.stdout.fileno(), cls.PIPE_READ_BYTES) if is_readable else b""
            if not chunk:  # the time limit, or the end of the process's output
                process.kill()
                pytest.fail(f"no line with {text!r} of {process.args} within {limit_s} s: {output.decode()}")
            output += chunk

        return output.decode()
