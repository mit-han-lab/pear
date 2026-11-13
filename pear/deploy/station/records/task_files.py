import json
import re
from pathlib import Path

from pear.deploy.station.control.schemas import Camera
from pear.utils.file_utils import write_text_atomically


class TaskFiles:
    """One task's folder: the form of its ids, and the name of every file and subfolder the task leaves in it."""

    HEX_ID_PATTERN = re.compile(r"[a-f0-9]{32}")  # an id that is the hex of a uuid4: a task's, an uploaded image's
    SEGMENT_ID_DIGITS = 3  # a recording segment's id is its number from 1, padded with zeros to this many digits
    SEGMENT_ID_PATTERN = re.compile(rf"[0-9]{{{SEGMENT_ID_DIGITS},}}")  # a recording segment's id
    JSON_INDENT = 2  # spaces of one level in a JSON file of the records

    TASK_NAME = "task.json"  # the task as it started: id, instruction, model, images
    EVENTS_NAME = "events.jsonl"  # the task's events, one on each line
    # the operator's tags, apart from the task's record: an edit of them leaves the time the history is sorted by
    ANNOTATIONS_NAME = "annotations.json"
    SESSION_NAME = "session.json"  # the task's model session: episode, model, action format, system prompt
    REFERENCE_IMAGE_NAME = "reference-{upload_id}.jpg"  # a reference image the operator gave with the task
    # what a turn leaves, each as <attempt id>-<word>.json: what the model was shown, what it answered and what
    # its action did
    OBSERVATION_RECORD = "observation"
    OUTPUT_RECORD = "output"
    RESULT_RECORD = "result"

    RECORDINGS_FOLDER = "recordings"  # the recording segments, each in a folder of its id
    MERGED_FOLDER = "merged"  # beside the segments' folders, the styled videos of all segments joined
    STYLED_FOLDER = "styled"  # in a segment's folder, the styled videos of that segment
    MANIFEST_NAME = "recording.json"  # a segment's manifest
    MERGED_MANIFEST_NAME = "merged.json"  # the manifest of a folder of styled videos
    ENCODER_LOG_NAME = "encoder.log"  # what the FFmpeg processes of a segment, or of a render, wrote to stderr
    TIMELINE_NAME = "timeline.jsonl"  # what each run of frames of a segment's full video shows, one run on each line
    MOTION_FILTER_NAME = "motion-filter.txt"  # the FFmpeg filter that cuts a segment's motion-only video
    FULL_KIND = "full"  # the video of everything that was recorded
    MOTION_KIND = "motion"  # the same video cut to the frames in which the robot moved
    VIDEO_KINDS = (FULL_KIND, MOTION_KIND)  # the videos of a segment and of a folder of styled videos
    VIDEO_SUFFIX = ".mp4"  # FFmpeg chooses the container by it, so a video that is still written ends with it too
    PARTIAL_MARKER = ".partial"  # before the suffix of a video that is still written
    # in the name of a file that is still written: a video, and a file on its way through a temporary one
    UNFINISHED_MARKERS = (PARTIAL_MARKER + ".", ".tmp")

    def __init__(self, directory: Path) -> None:
        self.directory = directory  # the task's folder, named by the task's id

    @classmethod
    def find(cls, root: Path, task_id: str) -> "TaskFiles | None":
        """Returns the folder of the task ``task_id`` under ``root``, None when the id has not the form of one or
        no such folder exists."""
        if not cls.HEX_ID_PATTERN.fullmatch(task_id) or not (root / task_id).is_dir():
            return None

        return cls(root / task_id)

    @property
    def task_path(self) -> Path:
        """The task's record."""
        return self.directory / self.TASK_NAME

    @property
    def events_path(self) -> Path:
        """The task's event file."""
        return self.directory / self.EVENTS_NAME

    @property
    def annotations_path(self) -> Path:
        """The file of the operator's tags."""
        return self.directory / self.ANNOTATIONS_NAME

    @property
    def session_path(self) -> Path:
        """The record of the task's model session."""
        return self.directory / self.SESSION_NAME

    @property
    def recordings_directory(self) -> Path:
        """The folder of the task's recording segments."""
        return self.directory / self.RECORDINGS_FOLDER

    @property
    def merged_directory(self) -> Path:
        """The folder of the styled videos of the task's segments joined."""
        return self.recordings_directory / self.MERGED_FOLDER

    def reference_image_path(self, upload_id: str) -> Path:
        """Returns the task's copy of the uploaded image ``upload_id``."""
        return self.directory / self.REFERENCE_IMAGE_NAME.format(upload_id=upload_id)

    def turn_record_path(self, attempt_id: str, record: str) -> Path:
        """Returns one record of the turn ``attempt_id``: its OBSERVATION_RECORD, OUTPUT_RECORD or RESULT_RECORD."""
        return self.directory / f"{attempt_id}-{record}.json"

    def segment_directory(self, segment_id: str) -> Path:
        """Returns the folder of the recording segment ``segment_id``."""
        return self.recordings_directory / segment_id

    def styled_directory(self, segment_id: str) -> Path:
        """Returns the folder of the styled videos of the segment ``segment_id``."""
        return self.segment_directory(segment_id) / self.STYLED_FOLDER

    @classmethod
    def video_name(cls, kind: str) -> str:
        """Returns the file name of the video of ``kind``, one of VIDEO_KINDS."""
        return kind + cls.VIDEO_SUFFIX

    @classmethod
    def partial_name(cls, video_name: str) -> str:
        """Returns the name a video has while it is written."""
        return video_name.removesuffix(cls.VIDEO_SUFFIX) + cls.PARTIAL_MARKER + cls.VIDEO_SUFFIX

    @classmethod
    def source_file_names(cls, cameras: tuple[Camera, ...]) -> dict[str, str]:
        """Returns, by camera role, the file name of each camera's own video in a segment's folder."""
        return {camera.role: camera.name + cls.VIDEO_SUFFIX for camera in cameras}

    @classmethod
    def write_json(cls, path: Path, value: object, *, escapes_non_ascii: bool = True) -> None:
        """Writes ``value`` to ``path`` as indented JSON without NaN: a reader sees the old file or the new one."""
        text = json.dumps(value, indent=cls.JSON_INDENT, allow_nan=False, ensure_ascii=escapes_non_ascii)
        write_text_atomically(path, text + "\n")
