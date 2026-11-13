import os
from pathlib import Path


def write_text_atomically(path: str | Path, text: str) -> None:
    """Writes ``text`` to ``path`` through a temporary file beside it: a reader sees the old file or the new one."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as temporary_file:
        temporary_file.write(text)

    os.replace(temporary_path, path)


def append_line(path: str | Path, line: str) -> None:
    """Appends ``line`` and a newline to ``path``."""
    with open(path, "a", encoding="utf-8") as file:
        file.write(line + "\n")
