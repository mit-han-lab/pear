import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

from repository import REPO

# the list of the repository's files is git's: a tree without its metadata has none
pytestmark = pytest.mark.skipif(not (REPO / ".git").exists(), reason="needs the repository's git metadata")
# other projects' text, which PEAR does not write: the code inside the patches, a robot's description with its licence
FOREIGN_PREFIXES = (
    "3rdparty/verl/patches/",
    "3rdparty/LMCache/patches/",
    "pear/deploy/station/control/robots/so101/description/",
)
# the pins and the locks: their four-part version numbers read as addresses
PIN_FILE_NAMES = ("constraints.in", "constraints.txt", "overrides.txt", "uv.lock", "package-lock.json", "versions.txt")
# this file names what it looks for
THIS_FILE = "tests/test_repository_hygiene_on_cpu.py"
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".mp4", ".pdf", ".gz")
# the files whose comments are read: Python by its tokens, with its docstrings, the others by their # and // lines
COMMENTED_SUFFIXES = (
    *(".py", ".yaml", ".yml", ".toml", ".sh", ".env", ".cfg", ".ts", ".tsx", ".js", ".mjs", ".css"),
    *(".c", ".cu", ".cuh"),
)


class RepositoryText:
    """The text files of the repository that PEAR writes, and the comments of its code and configuration files."""

    ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    # the addresses of no machine: this machine's own, and the ranges reserved for documentation
    NO_MACHINE = re.compile(r"127\.0\.0\.1|0\.0\.0\.0|192\.0\.2\.\d+|198\.51\.100\.\d+|203\.0\.113\.\d+")
    # a home directory or pytest's temporary root, which name the user
    HOME_PATH = re.compile(r"/(?:home|Users)/[\w.-]+|pytest-of-[\w.-]+")
    DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    NOTE = re.compile(r"\b(?:TODO|FIXME)\b")
    LINE_COMMENT = re.compile(r"(?:^|\s)(?:#|//)\s?(.*)$")

    @classmethod
    def by_name(cls) -> dict[str, str]:
        """The text of every file git tracks or would track, without other projects' text and the binary files."""
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        texts = {}
        for name in listing.splitlines():
            path = REPO / name
            if name.startswith(FOREIGN_PREFIXES) or name == THIS_FILE or name.endswith(BINARY_SUFFIXES):
                continue
            # a tracked file the working tree no longer has is a deletion that is not committed yet
            if not path.is_file():
                continue

            try:
                texts[name] = path.read_text()
            except UnicodeDecodeError:
                continue

        return texts

    @classmethod
    def comments(cls, name: str, text: str) -> list[tuple[int, str]]:
        """The (line, text) of every comment of a code or configuration file, and of every docstring of a Python file."""
        if not name.endswith(COMMENTED_SUFFIXES):
            return []
        if not name.endswith(".py"):
            lines = enumerate(text.splitlines(), start=1)
            return [(number, match.group(1)) for number, line in lines if (match := cls.LINE_COMMENT.search(line))]

        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        comments = [(token.start[0], token.string) for token in tokens if token.type == tokenize.COMMENT]
        documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        docstrings = [
            (getattr(node, "lineno", 1), docstring)
            for node in ast.walk(ast.parse(text))
            if isinstance(node, documented) and (docstring := ast.get_docstring(node))
        ]
        return comments + docstrings


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    return RepositoryText.by_name()


def test_no_file_names_a_machine_by_its_address(texts):
    addresses = [
        f"{name}: {address}"
        for name, text in texts.items()
        if Path(name).name not in PIN_FILE_NAMES
        for address in sorted(set(RepositoryText.ADDRESS.findall(text)))
        if not RepositoryText.NO_MACHINE.fullmatch(address)
    ]

    assert texts
    assert addresses == []


def test_no_file_names_a_user_s_home_directory(texts):
    home_paths = [
        f"{name}: {home_path}"
        for name, text in texts.items()
        for home_path in sorted(set(RepositoryText.HOME_PATH.findall(text)))
    ]

    assert home_paths == []


def test_no_comment_and_no_docstring_carries_a_date(texts):
    dated_lines = [
        f"{name}:{number}"
        for name, text in texts.items()
        for number, comment in RepositoryText.comments(name, text)
        if RepositoryText.DATE.search(comment)
    ]

    assert dated_lines == []


def test_no_file_carries_a_note_of_unfinished_work(texts):
    noted_files = [name for name, text in texts.items() if RepositoryText.NOTE.search(text)]

    assert noted_files == []
