import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from repository import REPO

# the program the installer places LIBERO-PRO's perturbed tasks with
PROGRAM = REPO / "scripts/install/sim/libero_pro/place_tasks.py"
# a download of the dataset: two directories of a perturbed task set, and a set no checksum names
DOWNLOAD_FILES = {
    "bddl_files/libero_goal_lan/open_the_drawer.bddl": "(define (problem open))",
    "init_files/libero_goal_lan/open_the_drawer.pruned_init": "states",
    "bddl_files/libero_goal_environment/open_the_drawer.bddl": "(define (problem other))",
}
TASK_SET_FILES = tuple(name for name in DOWNLOAD_FILES if "libero_goal_lan" in name)


class Dataset:
    """A download of LIBERO-PRO's dataset, the checksums of its perturbed tasks and a checkout to place them in."""

    def __init__(self, directory: Path) -> None:
        self.download = directory / "download"
        self.libero = directory / "libero"
        self.checksums = directory / "tasks.sha256"

        for name, text in DOWNLOAD_FILES.items():
            path = self.download / name
            path.parent.mkdir(parents=True)
            path.write_text(text)
        self.checksums.write_text(
            "".join(f"{hashlib.sha256(DOWNLOAD_FILES[name].encode()).hexdigest()}  {name}\n" for name in TASK_SET_FILES)
        )

    def place(self) -> subprocess.CompletedProcess:
        command = [sys.executable, PROGRAM, "--download", self.download, "--destination", self.libero]
        return subprocess.run([*command, "--checksums", self.checksums], capture_output=True, text=True)


@pytest.fixture
def dataset(tmp_path):
    return Dataset(tmp_path)


def test_the_task_sets_the_checksums_name_are_placed_and_no_other(dataset):
    # what an earlier revision left in a task set's directory goes with it
    stale_file = dataset.libero / "bddl_files/libero_goal_lan/stale.bddl"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale")

    process = dataset.place()

    placed_files = sorted(str(path.relative_to(dataset.libero)) for path in dataset.libero.rglob("*") if path.is_file())
    assert (process.returncode, placed_files) == (0, sorted(TASK_SET_FILES))


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_a_file_that_is_not_the_pinned_one_ends_it_before_anything_is_placed(dataset, damage):
    damaged_file = dataset.download / TASK_SET_FILES[0]
    if damage == "changed":
        damaged_file.write_text("another text")
    else:
        damaged_file.unlink()

    process = dataset.place()

    assert process.returncode != 0
    assert TASK_SET_FILES[0] in process.stderr
    assert not dataset.libero.exists()
