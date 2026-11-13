import os
import shutil
import subprocess
from pathlib import Path

import pytest

from repository import REPO

THIRD_PARTY = REPO / "3rdparty"
# the two patched projects: directory, checkout script, export script, prefix of the names in UPSTREAM
PROJECTS = {
    "verl": ("checkout_verl.sh", "export_verl_patches.sh", "VERL"),
    "LMCache": ("checkout_lmcache.sh", "export_lmcache_patches.sh", "LMCACHE"),
}
PATCH_AUTHOR = "pear <pear@localhost>"
PATCH_DATE = "Mon, 17 Sep 2001 00:00:00 +0000"  # the date the export gives every patch
SHORT_COMMIT_LENGTH = 8  # the characters of the upstream commit that the scripts' messages show
# the recipes that launch on a Ray cluster, and the line by which each verifies verl's checkout before its launch
RAY_RECIPES = sorted(
    {*REPO.glob("examples/models/*/train/run_*.sh"), *REPO.glob("examples/eval/eval_*.sh")}
    - {REPO / "examples/eval/eval_general.sh"}
)
CHECK_LINE = 'bash "$ROOT/3rdparty/verl/scripts/checkout_verl.sh" --check | tee -a "$RUN/train.log"\n'
LAUNCH_MODULE = "pear.eval.embodied"  # what the LIBERO evaluation recipe launches
# a Python that records the module it is told to run instead of running it: a recipe's launch then costs nothing;
# a program it is given with -c, as the recipe's environment gives one, gets no answer
RECORDING_PYTHON = '#!/bin/sh\nif [ "$1" = -m ]; then echo "$@" >> "$COMMAND_RECORD"; fi\n'


class StandInProject:
    """A copy of one project's scripts under a directory of the test, with a small repository as its upstream and one
    patch to it."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.checkout_script, self.export_script, prefix = PROJECTS[name]
        self.root = root
        self.project_dir = root / "3rdparty" / name
        self.patches_dir = self.project_dir / "patches"
        # verl's checkout is in the repository, LMCache's where LMCACHE_SRC says
        self.checkout_dir = self.project_dir / "src" if name == "verl" else root / "lmcache_src"
        self.upstream_dir = root / "upstream"

        shutil.copytree(THIRD_PARTY / name / "scripts", self.project_dir / "scripts")
        # the functions both projects' scripts source, beside the projects as in the repository
        shutil.copy(THIRD_PARTY / "patched_checkout.sh", root / "3rdparty")
        self.upstream_commit = self._create_upstream()
        upstream = f"{prefix}_URL=file://{self.upstream_dir}\n{prefix}_COMMIT={self.upstream_commit}\n"
        (self.project_dir / "UPSTREAM").write_text(upstream)
        self.patches_dir.mkdir()
        self.add_patch("value = 2\n", "[module] feat: the value is 2")

    def run(self, script: str, *arguments: str, **variables: str) -> subprocess.CompletedProcess:
        """Runs one of the project's scripts, in an environment of the test alone."""
        environment = {"PATH": os.environ["PATH"], "HOME": str(self.root), "LMCACHE_SRC": str(self.checkout_dir)}

        return subprocess.run(
            ["bash", str(self.project_dir / "scripts" / script), *arguments],
            env={**environment, **variables},
            capture_output=True,
            text=True,
        )

    def checkout(self, *arguments: str, **variables: str) -> subprocess.CompletedProcess:
        return self.run(self.checkout_script, *arguments, **variables)

    def export(self) -> subprocess.CompletedProcess:
        return self.run(self.export_script)

    def git(self, directory: Path, *arguments: str) -> str:
        """The output of a git command in a directory, as the patches' author."""
        identity = ["-c", "user.name=pear", "-c", "user.email=pear@localhost"]
        process = subprocess.run(
            ["git", "-C", str(directory), *identity, *arguments], capture_output=True, text=True, check=True
        )

        return process.stdout.strip()

    def add_patch(self, module_text: str, subject: str) -> Path:
        """One more patch file: the module's text changed on top of the patches there are, in a clone of its own."""
        clone_dir = self.root / f"clone_{len(list(self.patches_dir.iterdir()))}"
        self.git(self.root, "clone", "-q", str(self.upstream_dir), str(clone_dir))
        patches = sorted(self.patches_dir.glob("*.patch"))
        if patches:
            self.git(clone_dir, "am", "-q", "--keep-non-patch", *map(str, patches))
        (clone_dir / "module.py").write_text(module_text)
        self.git(clone_dir, "commit", "-q", "-a", "-m", subject, f"--date={PATCH_DATE}")
        options = ["-q", "-N", "--no-signature", "--zero-commit", "--full-index", f"--start-number={len(patches) + 1}"]
        self.git(clone_dir, "format-patch", *options, "-o", str(self.patches_dir), "HEAD~1..HEAD")

        return sorted(self.patches_dir.glob("*.patch"))[-1]

    def commit_in_checkout(self, module_text: str, subject: str, *options: str) -> None:
        (self.checkout_dir / "module.py").write_text(module_text)
        self.git(self.checkout_dir, "commit", "-q", "-a", "-m", subject, *options)

    def launch_recipe(self, **variables: str) -> tuple[int, str]:
        """Runs a copy of the LIBERO evaluation recipe next to this project, with a Python that records the modules it
        is told to run: the recipe's status and those command lines."""
        python = self.root / "work" / "envs" / "verl" / "bin" / "python"
        command_record = self.root / "commands"
        if not python.exists():
            shutil.copytree(REPO / "examples", self.root / "examples")
            # the environment the recipe sources
            (self.root / "scripts").mkdir()
            shutil.copy(REPO / "scripts" / "gpu_process_variables.sh", self.root / "scripts")
            python.parent.mkdir(parents=True)
            python.write_text(RECORDING_PYTHON)
            python.chmod(0o755)
        command_record.write_text("")
        settings = {"WORK_DIR": str(self.root / "work"), "GPU": "gb300", "ROLLOUT_NNODES": "1", "MODEL": str(self.root)}
        run = {"EVAL_STEP": "1", "RUN": str(self.root / "run"), "COMMAND_RECORD": str(command_record)}
        environment = {"PATH": os.environ["PATH"], "HOME": str(self.root)}

        process = subprocess.run(
            ["bash", str(self.root / "examples" / "eval" / "eval_libero.sh")],
            env={**environment, **settings, **run, **variables},
            capture_output=True,
            text=True,
        )

        return process.returncode, command_record.read_text()

    def leftovers(self) -> list[str]:
        """The work directories the scripts make next to the checkout, which they are to remove."""
        return [path.name for path in self.checkout_dir.parent.glob(f"{self.checkout_dir.name}.*")]

    def _create_upstream(self) -> str:
        self.upstream_dir.mkdir()
        self.git(self.upstream_dir, "init", "-q", "-b", "main")
        (self.upstream_dir / "module.py").write_text("value = 1\n")
        self.git(self.upstream_dir, "add", "module.py")
        self.git(self.upstream_dir, "commit", "-q", "-m", "the upstream commit")

        return self.git(self.upstream_dir, "rev-parse", "HEAD")


@pytest.fixture(params=sorted(PROJECTS))
def project(request, tmp_path) -> StandInProject:
    return StandInProject(request.param, tmp_path)


def test_a_missing_checkout_is_built_as_the_upstream_commit_with_the_patches(project):
    process = project.checkout()

    short_commit = project.upstream_commit[:SHORT_COMMIT_LENGTH]
    state = f"{project.checkout_dir} = {short_commit} + 1 patches (3rdparty/{project.name}/patches)"
    assert (process.returncode, process.stdout) == (0, f"{project.name}: {state}\n")
    assert (project.checkout_dir / "module.py").read_text() == "value = 2\n"
    assert project.git(project.checkout_dir, "log", "-1", "--format=%an <%ae>|%s") == (
        f"{PATCH_AUTHOR}|[module] feat: the value is 2"
    )
    assert project.leftovers() == []
    assert project.checkout("--check").returncode == 0


def test_a_check_of_a_missing_checkout_fails_and_builds_nothing(project):
    # a launch only verifies: the installer builds
    process = project.checkout("--check")

    assert process.returncode == 1
    assert "is missing" in process.stdout
    assert not project.checkout_dir.exists()


def test_a_checkout_built_from_other_patches_fails_the_check(project):
    project.checkout()
    project.add_patch("value = 3\n", "[module] feat: the value is 3")

    process = project.checkout("--check")

    assert process.returncode == 1
    assert "has other commits than the patches" in process.stdout


@pytest.mark.parametrize("file_name", ["module.py", "new_module.py"])
def test_a_changed_file_and_a_file_git_does_not_track_both_fail_the_check(project, file_name):
    # verl imports a module that was never added as it imports a committed one
    project.checkout()
    (project.checkout_dir / file_name).write_text("value = 5\n")

    process = project.checkout("--check")

    assert process.returncode == 1
    assert "has uncommitted or untracked files" in process.stdout


def test_a_checkout_of_another_history_fails_the_check_and_stays(project):
    # a directory the caller names is never removed, whatever it holds
    project.checkout_dir.mkdir(parents=True)
    project.git(project.checkout_dir, "init", "-q")
    (project.checkout_dir / "other.py").write_text("")
    project.git(project.checkout_dir, "add", "other.py")
    project.git(project.checkout_dir, "commit", "-q", "-m", "another history")

    process = project.checkout()

    assert process.returncode == 1
    assert f"is not based on {project.upstream_commit[:SHORT_COMMIT_LENGTH]}" in process.stdout
    assert (project.checkout_dir / "other.py").exists()


def test_a_patch_that_does_not_apply_leaves_no_directory_behind(project):
    [patch] = project.patches_dir.glob("*.patch")
    patch.write_text(patch.read_text().replace("-value = 1", "-value = 0"))

    process = project.checkout()

    assert process.returncode != 0
    assert not project.checkout_dir.exists()
    assert project.leftovers() == []


def test_an_argument_other_than_check_is_refused(project):
    process = project.checkout("--verify")

    assert (process.returncode, process.stdout) == (1, "unknown argument --verify (--check)\n")


def test_a_verl_change_in_progress_passes_the_check_when_its_variable_says_so(tmp_path):
    verl = StandInProject("verl", tmp_path)
    verl.checkout()
    verl.commit_in_checkout("value = 3\n", "[module] feat: a change in progress")

    refused_check = verl.checkout("--check")
    accepted_check = verl.checkout("--check", VERL_CHANGE_IN_PROGRESS="1")

    assert refused_check.returncode == 1
    assert accepted_check.returncode == 0
    assert "has other commits than the patches: accepted as a change in progress" in accepted_check.stdout


@pytest.mark.parametrize("recipe", RAY_RECIPES, ids=lambda recipe: recipe.name)
def test_every_recipe_verifies_verl_s_checkout_before_its_launch(recipe):
    # by its text: a recipe builds no checkout, and a checkout that fails the check ends it under pipefail
    text = recipe.read_text()

    assert text.count(CHECK_LINE) == 1
    assert text.index(CHECK_LINE) < text.index("python -m pear.")
    assert "set -euo pipefail\n" in text


def test_a_recipe_launches_with_the_patches_checkout_or_an_accepted_change_alone(tmp_path):
    verl = StandInProject("verl", tmp_path)
    verl.checkout()

    patches_status, patches_commands = verl.launch_recipe()
    verl.commit_in_checkout("value = 3\n", "[module] feat: a change in progress")
    refused_status, refused_commands = verl.launch_recipe()
    accepted_status, accepted_commands = verl.launch_recipe(VERL_CHANGE_IN_PROGRESS="1")

    assert (patches_status, LAUNCH_MODULE in patches_commands) == (0, True)
    assert (refused_status, refused_commands) == (1, "")
    assert (accepted_status, LAUNCH_MODULE in accepted_commands) == (0, True)
    assert "has other commits than the patches: accepted" in (tmp_path / "run" / "train.log").read_text()


def test_lmcache_s_wheel_is_never_built_from_a_change_in_progress(tmp_path):
    lmcache = StandInProject("LMCache", tmp_path)
    lmcache.checkout()
    lmcache.commit_in_checkout("value = 3\n", "[module] feat: a change in progress")

    process = lmcache.checkout("--check", VERL_CHANGE_IN_PROGRESS="1", CHANGE_IN_PROGRESS="1")

    assert process.returncode == 1


def test_an_export_adds_a_commit_s_patch_and_leaves_the_other_files_as_they_are(project):
    project.checkout()
    [first_patch] = project.patches_dir.glob("*.patch")
    first_patch_text = first_patch.read_text()
    project.commit_in_checkout("value = 3\n", "[module] feat: the value is 3")

    process = project.export()

    patches = sorted(project.patches_dir.glob("*.patch"))
    assert process.returncode == 0, process.stdout + process.stderr
    assert [patch.name for patch in patches] == [first_patch.name, "0002-module-feat-the-value-is-3.patch"]
    assert patches[0].read_text() == first_patch_text
    assert f"From: {PATCH_AUTHOR}\n" in patches[1].read_text()
    assert f"Date: {PATCH_DATE}\n" in patches[1].read_text()
    assert "Subject: [PATCH] [module] feat: the value is 3\n" in patches[1].read_text()
    assert project.leftovers() == []
    assert project.checkout("--check").returncode == 0


def test_an_export_refuses_a_commit_of_another_author(project):
    # a patch would carry the name and the address
    project.checkout()
    project.commit_in_checkout("value = 3\n", "[module] feat: the value is 3", "--author=Someone <someone@example.com>")

    process = project.export()

    assert process.returncode == 1
    assert "is by Someone <someone@example.com>" in process.stdout
    assert len(list(project.patches_dir.glob("*.patch"))) == 1


def test_an_export_refuses_a_file_git_does_not_track(project):
    project.checkout()
    (project.checkout_dir / "new_module.py").write_text("")

    process = project.export()

    assert process.returncode == 1
    assert "has uncommitted or untracked files" in process.stdout


def test_patches_that_do_not_rebuild_the_checkout_are_not_written(project):
    # a merge whose resolution no patch carries: the two sides' patches do not apply one after the other
    project.checkout()
    [patch] = project.patches_dir.glob("*.patch")
    patch_text = patch.read_text()
    project.git(project.checkout_dir, "checkout", "-q", "-b", "side")
    project.commit_in_checkout("value = 3\n", "[module] feat: the value is 3")
    project.git(project.checkout_dir, "checkout", "-q", "pear")
    project.commit_in_checkout("value = 4\n", "[module] feat: the value is 4")
    subprocess.run(["git", "-C", str(project.checkout_dir), "merge", "-q", "side"], capture_output=True)
    project.commit_in_checkout("value = 5\n", "the merge, resolved")

    process = project.export()

    assert process.returncode != 0
    assert [path.read_text() for path in project.patches_dir.glob("*.patch")] == [patch_text]
    assert project.leftovers() == []
