import json
import re
import threading
import time
from typing import Any

from pear.deploy.station.records.task_files import TaskFiles


class Annotations:
    """The operator's tags of each task, in a file of their own beside the task's unchanged record."""

    MAX_TAGS = 20  # tags one task may have
    MAX_TAG_LENGTH = 40  # characters of one tag
    MAX_TAG_INPUTS = 100  # tags one request may send, before duplicates and blanks are dropped
    # whitespace is collapsed first, so any control character that is left is refused
    CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
    write_lock = threading.Lock()  # one write of a tags file at a time, whoever writes it

    @staticmethod
    def read(files: TaskFiles) -> dict[str, list[str]]:
        """Returns the tags of a task, none when it has no tags file or the file is damaged."""
        try:
            annotations = json.loads(files.annotations_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # a missing file holds no tags, and a damaged one must not hide the task: the next update replaces it
            return {"tags": []}
        if not isinstance(annotations, dict):
            return {"tags": []}

        tags = annotations.get("tags")

        return {"tags": [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []}

    @classmethod
    def update(cls, files: TaskFiles, *, tags: object) -> dict[str, list[str]]:
        """Replaces a task's tags, checked before anything is written, and returns the tags as they are kept."""
        annotations = {"tags": cls.clean_tags(tags)}

        with cls.write_lock:
            # the tags stay readable text: they are not escaped
            TaskFiles.write_json(
                files.annotations_path, {**annotations, "updated": time.time()}, escapes_non_ascii=False
            )

        return annotations

    @staticmethod
    def tag_counts(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Returns how many tasks carry each tag; a case variant counts toward the oldest task's spelling."""
        counts: dict[str, int] = {}
        spellings: dict[str, str] = {}
        for task in sorted(tasks, key=lambda task: task.get("created") or 0):
            for tag in task.get("tags", []):
                key = tag.casefold()
                spellings.setdefault(key, tag)
                counts[key] = counts.get(key, 0) + 1

        return sorted(
            ({"tag": spellings[key], "count": count} for key, count in counts.items()),
            key=lambda tag_count: (-tag_count["count"], tag_count["tag"].casefold()),
        )

    @classmethod
    def clean_tags(cls, raw_tags: object) -> list[str]:
        """Returns the tags trimmed, with single spaces and without case-insensitive duplicates, in their first
        spelling."""
        if not isinstance(raw_tags, list) or len(raw_tags) > cls.MAX_TAG_INPUTS:
            raise ValueError("Tags must be a list of text")

        tags: list[str] = []
        seen_keys: set[str] = set()
        for raw_tag in raw_tags:
            if not isinstance(raw_tag, str):
                raise ValueError("Tags must be text")
            tag = " ".join(raw_tag.split())
            if not tag:
                continue
            if len(tag) > cls.MAX_TAG_LENGTH:
                raise ValueError(f"Tags are limited to {cls.MAX_TAG_LENGTH} characters")
            if cls.CONTROL_CHARACTER_PATTERN.search(tag):
                raise ValueError("Tags cannot contain control characters")
            if tag.casefold() not in seen_keys:
                seen_keys.add(tag.casefold())
                tags.append(tag)

        if len(tags) > cls.MAX_TAGS:
            raise ValueError(f"A task can have at most {cls.MAX_TAGS} tags")

        return tags
