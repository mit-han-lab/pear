import numpy as np

from pear.deploy.station.records.recording.ffmpeg import FRAME_RATE


class MotionCut:
    """The frames of a full video that its motion-only video keeps."""

    def __init__(self, motion_spans: list[tuple[float, float]], frame_count: int) -> None:
        # a frame is kept when its centre lies in a motion span, so the float edges of a span cannot split a frame
        frame_centres = (np.arange(frame_count) + 0.5) / FRAME_RATE
        self.is_kept = np.zeros(frame_count, bool)  # of each frame of the full video
        for start_s, end_s in motion_spans:
            self.is_kept |= (frame_centres >= start_s) & (frame_centres < end_s)
        self.kept_count = int(self.is_kept.sum())

    def select_filter(self, first_frame: int, frame_count: int) -> str:
        """Returns the FFmpeg filter that keeps this cut's frames among the ``frame_count`` frames from ``first_frame``
        on, counted from 0 at ``first_frame``, and plays them back to back."""
        # where a run of kept frames begins, and where it ends
        is_kept = self.is_kept[first_frame : first_frame + frame_count]
        run_edges = np.flatnonzero(np.diff(is_kept, prepend=False, append=False))
        runs = "+".join(f"between(n,{start},{end - 1})" for start, end in zip(run_edges[::2], run_edges[1::2]))

        return f"select='{runs}',setpts=N/({FRAME_RATE}*TB)"
