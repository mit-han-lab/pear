import os
import subprocess
from pathlib import Path
from typing import IO

FRAME_RATE = 30  # frames per second of every recorded and rendered video
VIDEO_SIZE = (1920, 1080)  # width and height of a full video and of a styled one, in pixels
# width and height of one camera's own video, in pixels: a picture of another size is resized to it
SOURCE_VIDEO_SIZE = (640, 480)


class FFmpeg:
    """FFmpeg's command lines and the processes that run them."""

    QUIET_ARGUMENTS = ("-hide_banner", "-loglevel", "error", "-nostdin")  # print errors only, never read the terminal
    DEFAULT_CRF = 21  # x264 quality of the full and styled videos; lower is better
    PRESET = "veryfast"  # x264's trade of encoding time against file size
    THREADS = 2  # threads of one FFmpeg process
    # niceness of a render and of the count of its frames: below a recording's encoder, so both yield to a running
    # task
    RENDER_NICENESS = 10
    PROBE_TIMEOUT_S = 300  # for counting the packets of one video
    ERROR_TAIL_CHARS = 400  # of FFmpeg's stderr, kept in the error message
    WORKER_EXIT_TIMEOUT_S = 2  # after SIGTERM, and again after SIGKILL, for a worker that runs FFmpeg to end

    @staticmethod
    def executable() -> str:
        """Returns the FFmpeg binary that the imageio-ffmpeg wheel brings: no system package and no PATH entry."""
        # imported here, where a recording or a render starts: the workers are given the binary's path
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()

    @classmethod
    def raw_encoder(
        cls, executable: str, size: tuple[int, int], output_path: Path, log: IO[str], crf: int = DEFAULT_CRF
    ) -> subprocess.Popen:
        """Starts an encoder of the RGB frames of ``size`` written to its stdin."""
        return subprocess.Popen(
            cls._base_arguments(executable)
            + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FRAME_RATE)]
            + ["-i", "pipe:0"]
            + cls._video_arguments(crf)
            + [str(output_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )

    @classmethod
    def raw_decoder(cls, executable: str, source_path: Path, size: tuple[int, int], log: IO[str]) -> subprocess.Popen:
        """Starts a decoder that writes every coded frame of a video to its stdout as an RGB picture of ``size``."""
        return subprocess.Popen(
            [executable, *cls.QUIET_ARGUMENTS, "-threads", str(cls.THREADS), "-i", str(source_path)]
            + ["-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}"]
            + ["pipe:1"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
        )

    @classmethod
    def motion_encoder(
        cls, executable: str, source_path: Path, filter_script: Path, output_path: Path, log: IO[str]
    ) -> subprocess.Popen:
        """Starts an encoder of the frames of a video that a filter script keeps."""
        return subprocess.Popen(
            cls._base_arguments(executable)
            + ["-threads", str(cls.THREADS), "-i", str(source_path), "-filter_script:v", str(filter_script)]
            + ["-r", str(FRAME_RATE)]
            + cls._video_arguments(cls.DEFAULT_CRF)
            + [str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=log,
        )

    @classmethod
    def packet_count(cls, executable: str, path: Path) -> int:
        """Returns the frames of a video, counted from its packets without decoding them, at low priority."""
        process = subprocess.Popen(
            [executable, *cls.QUIET_ARGUMENTS, "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            os.setpriority(os.PRIO_PROCESS, process.pid, cls.RENDER_NICENESS)
        except OSError:
            pass  # it has ended already, or the priority may not be set: the priority is a courtesy only

        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=cls.PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("Video merge timed out") from None

        if process.returncode != 0:
            detail = stderr_bytes.decode(errors="replace").strip()[-cls.ERROR_TAIL_CHARS :]
            raise RuntimeError(f"FFmpeg exited with {process.returncode}: {detail}")

        return sum(1 for line in stdout_bytes.splitlines() if line and not line.startswith(b"#"))

    @staticmethod
    def kill(process: subprocess.Popen) -> None:
        """Kills an FFmpeg process that still runs, and collects it."""
        if process.poll() is None:
            process.kill()
            process.wait()

    @classmethod
    def _base_arguments(cls, executable: str) -> list[str]:
        """Returns the start of a command line that writes a file and may replace it."""
        return [executable, *cls.QUIET_ARGUMENTS, "-y"]

    @classmethod
    def _video_arguments(cls, crf: int) -> list[str]:
        """Returns the arguments of an H.264 video without sound that a browser plays, at the quality ``crf``."""
        return [
            *("-an", "-c:v", "libx264", "-preset", cls.PRESET, "-crf", str(crf)),
            *("-threads", str(cls.THREADS), "-pix_fmt", "yuv420p", "-movflags", "+faststart"),
        ]
