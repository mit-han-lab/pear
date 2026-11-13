import os
import subprocess
import sys

from pear.utils.file_utils import append_line, write_text_atomically
from repository import REPO

# a locale whose encoding is ASCII, with Python's UTF-8 mode and its locale coercion off: what open() would write
# in without an encoding of its own
ASCII_LOCALE = {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
NON_ASCII_TEXT = "Grüße"
# writes the text with both helpers into the two files it is given; the text as ASCII escapes, since under that locale
# Python reads its command line as ASCII
WRITES = (
    "import sys; from pear.utils.file_utils import append_line, write_text_atomically; "
    f"write_text_atomically(sys.argv[1], {ascii(NON_ASCII_TEXT)}); append_line(sys.argv[2], {ascii(NON_ASCII_TEXT)})"
)


def test_an_atomic_write_replaces_the_file_and_leaves_no_temporary_one(tmp_path):
    path = tmp_path / "numbers.json"
    path.write_text("old")

    write_text_atomically(path, "new")

    assert path.read_text() == "new"
    assert sorted(file.name for file in tmp_path.iterdir()) == ["numbers.json"]


def test_an_appended_line_follows_the_lines_before_it(tmp_path):
    path = tmp_path / "actions.jsonl"

    append_line(path, "first")
    append_line(str(path), "second")

    assert path.read_text() == "first\nsecond\n"


def test_both_helpers_write_utf_8_whatever_the_locale(tmp_path):
    written, appended = tmp_path / "written.txt", tmp_path / "appended.txt"

    process = subprocess.run(
        [sys.executable, "-c", WRITES, str(written), str(appended)],
        env={**os.environ, **ASCII_LOCALE},
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert written.read_bytes() == NON_ASCII_TEXT.encode()
    assert appended.read_bytes() == (NON_ASCII_TEXT + "\n").encode()
