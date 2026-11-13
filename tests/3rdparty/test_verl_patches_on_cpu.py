import os
import re
import shutil
import subprocess

from repository import REPO

VERL = REPO / "3rdparty" / "verl"
PATCHES = VERL / "patches"
# a python that prints the test files it is given instead of running them
RECORDING_PYTHON = """#!/bin/sh
for argument in "$@"; do
  case "$argument" in tests/*) echo "$argument" ;; esac
done
"""
# a python that prints, instead of running the tests, what it is told of Ray's usage report and of its metrics thread
RAY_SETTINGS_PYTHON = """#!/bin/sh
case "$1" in -m) echo "usage report ${RAY_USAGE_STATS_ENABLED:-unset}, metrics thread ${RAY_enable_open_telemetry:-unset}" ;; esac
"""
# a line a patch adds that imports pear, at any indentation
PEAR_IMPORT = re.compile(r"^\+\s*(import|from)\s+pear\b")
# the author checkout_verl.sh gives the checkout's commits and the date of every patch, as the export writes them
NEUTRAL_HEADERS = ["From: pear <pear@localhost>", "Date: Mon, 17 Sep 2001 00:00:00 +0000"]


def test_no_patch_makes_verl_import_pear():
    patches = sorted(PATCHES.glob("*.patch"))
    pear_imports = [
        f"{patch.name}: {line}"
        for patch in patches
        for line in patch.read_text().splitlines()
        if PEAR_IMPORT.match(line)
    ]

    assert patches
    assert not pear_imports


def test_every_patch_carries_the_neutral_author_and_date():
    # a commit made in the checkout would export a person's name and address and the time it was made at
    headers = {}
    for patch in sorted(PATCHES.glob("*.patch")):
        mail_headers = patch.read_text().split("\n\n", 1)[0].splitlines()
        headers[patch.name] = [header for header in mail_headers if header.startswith(("From: ", "Date: "))]
    other_headers = {name: found for name, found in headers.items() if found != NEUTRAL_HEADERS}

    assert headers
    assert not other_headers


def test_verl_s_tests_of_the_patched_files_are_the_same_from_any_directory(tmp_path):
    # inside a repository git names only the patched files below the working directory: none, from 3rdparty/verl
    python = tmp_path / "python"
    python.write_text(RECORDING_PYTHON)
    python.chmod(0o755)

    selections = [
        subprocess.run(
            ["bash", str(VERL / "scripts" / "run_patched_verl_tests.sh"), "cpu"],
            env={"PATH": os.environ["PATH"], "PYTHON": str(python)},
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for directory in (REPO, VERL)
    ]

    assert selections[0].startswith("tests/")
    assert selections[0] == selections[1]


def test_verl_s_tests_run_without_ray_s_usage_report_and_without_its_metrics_thread(tmp_path):
    # ray.shutdown leaves the OpenTelemetry metrics thread running, and it can crash the test process at its exit
    python = tmp_path / "python"
    python.write_text(RAY_SETTINGS_PYTHON)
    python.chmod(0o755)

    process = subprocess.run(
        ["bash", str(VERL / "scripts" / "run_patched_verl_tests.sh"), "cpu"],
        env={"PATH": os.environ["PATH"], "PYTHON": str(python)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert process.stdout.splitlines()[-1] == "usage report 0, metrics thread 0"


def test_a_patch_git_cannot_read_ends_the_selection_of_verl_s_tests(tmp_path):
    # without git's status its failure would read as "the patches touch no test": a green run of nothing
    verl = tmp_path / "verl"
    for name in ("scripts", "src", "patches"):
        (verl / name).mkdir(parents=True)
    script = shutil.copy(VERL / "scripts" / "run_patched_verl_tests.sh", verl / "scripts")
    (verl / "patches" / "0001-not-a-patch.patch").write_text("not a patch\n")

    process = subprocess.run(["bash", script, "cpu"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)

    assert process.returncode != 0
    assert "the patches touch no" not in process.stdout
