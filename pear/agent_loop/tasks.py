from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeTask:
    env_class: str
    env_args: dict[str, Any]
    data_source: str
    index: int


def parse_range(ranges: str) -> list[int]:
    """Comma-separated numbers and inclusive ranges, in their original order."""
    numbers = []
    for part in ranges.split(","):
        if "-" in part:
            first, last = part.split("-")
            numbers.extend(range(int(first), int(last) + 1))
        elif part.strip():
            numbers.append(int(part))
    return numbers


def load_tasks(paths: list[str | Path]) -> list[EpisodeTask]:
    """Expand episode files in order; repeated files retain their curriculum weight."""
    tasks = []
    for path in paths:
        with open(path) as file:
            spec = json.load(file)
        for product in spec["env"] if isinstance(spec["env"], list) else [spec["env"]]:
            keys = list(product)
            values = [parse_range(v) if isinstance(v, str) else list(v) for v in product.values()]
            for combination in itertools.product(*values):
                env_args = dict(zip(keys, combination))
                tasks.append(EpisodeTask(spec["env_class"], env_args, str(env_args[spec["data_source"]]), len(tasks)))
    return tasks
