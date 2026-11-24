import os
import subprocess
from pathlib import Path
from typing import IO

FRAME_RATE = 30  # frames per second of every recorded and rendered video
VIDEO_SIZE = (1920, 1080)  # width and height of a styled video, in pixels
# width and height of one camera's own video, in pixels: a picture of another size is resized to it
SOURCE_VIDEO_SIZE = (640, 480)


class FFmpeg:
    """FFmpeg's command lines and the processes that run them."""

    QUIET_ARGUMENTS = ("-hide_banner", "-loglevel", "error", "-nostdin")  # print errors only, never read the terminal
    DEFAULT_CRF = 21  # x264 quality of the styled videos; lower is better
    PRESET = "veryfast"  # x264's trade of encoding time against file size
    THREADS = 2  # threads of one FFmpeg process
    # a seek lands this share of a frame before the frame it is for: a decoder that seeks to a time keeps the frames
    # from it on, and the float time of the frame itself could round past the frame
    SEEK_LEAD_FRAMES = 0.5
    # niceness of a render and of the count of its frames: below a recording's encoder, so both yield to a running
    # task
    RENDER_NICENESS = 10
    PROBE_TIMEOUT_S = 300  # for counting the packets of one video, and for joining videos
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
        cls,
        executable: str,
        size: tuple[int, int],
        output_path: Path,
        log: IO[str],
        crf: int = DEFAULT_CRF,
    ) -> subprocess.Popen:
        """Starts an encoder of the RGB frames of ``size`` written to its stdin."""
        return subprocess.Popen(
            [executable, *cls.QUIET_ARGUMENTS, "-y"]
            + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FRAME_RATE)]
            + ["-i", "pipe:0"]
            + ["-an", "-c:v", "libx264", "-preset", cls.PRESET, "-crf", str(crf), "-threads", str(cls.THREADS)]
            + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )

    @classmethod
    def raw_decoder(
        cls,
        executable: str,
        source_path: Path,
        size: tuple[int, int],
        log: IO[str],
        first_frame: int,
        filter_script: Path | None,
    ) -> subprocess.Popen:
        """Starts a decoder that writes the coded frames of a video from ``first_frame`` on to its stdout as RGB
        pictures of ``size``: every frame, or those a filter script keeps, which counts them from ``first_frame``."""
        seek_arguments = ["-ss", f"{(first_frame - cls.SEEK_LEAD_FRAMES) / FRAME_RATE:.6f}"] if first_frame else []
        filter_arguments = ["-filter_script:v", str(filter_script)] if filter_script else []
        return subprocess.Popen(
            [executable, *cls.QUIET_ARGUMENTS, "-threads", str(cls.THREADS), *seek_arguments, "-i", str(source_path)]
            + filter_arguments
            + ["-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}"]
            + ["pipe:1"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
        )

    @classmethod
    def join(cls, executable: str, paths: list[Path], list_path: Path, output_path: Path, log: IO[str]) -> None:
        """Joins videos of the same encoding, one after another, into ``output_path`` without encoding them again;
        ``list_path`` holds their list meanwhile."""
        # the list's lines quote each path: a quote in a path closes, escapes and reopens its quoting
        list_path.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in paths))
        try:
            subprocess.run(
                [executable, *cls.QUIET_ARGUMENTS, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
                + ["-c", "copy", "-movflags", "+faststart", str(output_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                timeout=cls.PROBE_TIMEOUT_S,
                check=True,
            )
        finally:
            list_path.unlink(missing_ok=True)

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
