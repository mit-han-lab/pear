import subprocess
import sys

import pytest

from repository import REPO

RULE = REPO / ".github" / "scripts" / "pr_title.py"


@pytest.fixture
def rule():
    # as the workflow runs it: the title in the variable, the verdict in the exit status
    def run(title: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(RULE)], env={"PR_TITLE": title}, capture_output=True, text=True)

    return run


@pytest.mark.parametrize(
    "title",
    [
        "[envs] fix: the reset of a stale episode",
        "[examples, scripts, docs] fix: the Hub's libraries are offline on a GPU machine",
        "[eval,install] refactor: a benchmark runs in a process of its own",
        "[tests] test: every verl patch carries the neutral author",
        "[3rdparty] chore: the pinned commit moves",
    ],
)
def test_a_title_of_modules_a_type_and_a_description_passes(rule, title):
    process = rule(title)

    assert (process.returncode, process.stdout) == (0, "")


@pytest.mark.parametrize(
    "title",
    [
        "fix the reset of a stale episode",
        "[envs] the reset of a stale episode",
        "[kernels] fix: a module the list does not hold",
        "[envs, ] fix: an empty module",
        "[envs] feature: a type the list does not hold",
        "[envs] fix:",
        "[envs]fix: no space after the modules",
        "Draft: [envs] fix: a word before the modules",
    ],
)
def test_another_title_fails_and_is_told_the_modules_and_the_types(rule, title):
    process = rule(title)

    assert process.returncode == 1
    assert "Modules: agent_loop, envs, " in process.stdout
    assert "Types: feat, fix, " in process.stdout
