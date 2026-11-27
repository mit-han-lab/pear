from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pear.utils.file_utils import write_text_atomically


def run_metadata(config) -> dict:
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
        return result.stdout if result.returncode == 0 else b""

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        "config": asdict(config),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tinker", "tinker-cookbook", "tml-renderers", "transformers", "ray", "torch")
        },
    }


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(path, json.dumps(report, indent=2, allow_nan=False) + "\n")
