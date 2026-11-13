import os
import subprocess
from pathlib import Path

import pytest

from repository import REPO

SCRIPT = REPO / "scripts/export_checkpoint.sh"
LAUNCHES = ("20260101T000000Z", "20260102T000000Z")  # of the run, by their start times
# the training environment's Python as the script calls it: it records where it would import from and its arguments
PYTHON = """#!/bin/bash
echo "PYTHONPATH=$PYTHONPATH $*" >> "$COMMANDS_PATH"
"""
# PEAR's model merger on the tree's pear and verl: the run's config, then verl's merge of the checkpoint's actor
MERGER_COMMAND = (
    f"PYTHONPATH={REPO}:{REPO}/3rdparty/verl/src -m pear.utils.model_merger --config {{config_path}} merge "
    "--backend fsdp --trust-remote-code --local_dir {checkpoint_dir}/actor --target_dir {model_dir}"
)


@pytest.fixture
def run_dir(tmp_path):
    # a run directory as the recipes lay it out: a checkpoint, and the config Hydra recorded at each launch
    (tmp_path / "run/ckpt/global_step_3").mkdir(parents=True)
    for launch in LAUNCHES:
        (tmp_path / f"run/hydra/{launch}/.hydra").mkdir(parents=True)
        (tmp_path / f"run/hydra/{launch}/.hydra/config.yaml").touch()

    return tmp_path / "run"


@pytest.fixture
def export(tmp_path):
    (tmp_path / "envs/verl/bin").mkdir(parents=True)
    (tmp_path / "envs/verl/bin/python").write_text(PYTHON)
    (tmp_path / "envs/verl/bin/python").chmod(0o755)

    def run(checkpoint_dir: Path, **variables: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        environment = {
            "PATH": os.environ["PATH"],
            "WORK_DIR": str(tmp_path),
            "COMMANDS_PATH": str(tmp_path / "commands"),
            **variables,
        }
        process = subprocess.run(
            ["bash", str(SCRIPT), str(checkpoint_dir), str(tmp_path / "model")],
            env=environment,
            capture_output=True,
            text=True,
        )
        commands_path = tmp_path / "commands"

        return process, commands_path.read_text().splitlines() if commands_path.exists() else []

    return run


def test_a_checkpoint_is_merged_with_the_config_of_its_run_s_latest_launch(export, run_dir, tmp_path):
    checkpoint_dir = run_dir / "ckpt/global_step_3"

    process, commands = export(checkpoint_dir)

    assert process.returncode == 0, process.stdout + process.stderr
    # as the script finds it, from the checkpoint
    config_path = f"{checkpoint_dir}/../../hydra/{LAUNCHES[-1]}/.hydra/config.yaml"
    assert commands == [
        MERGER_COMMAND.format(config_path=config_path, checkpoint_dir=checkpoint_dir, model_dir=tmp_path / "model")
    ]


def test_a_checkpoint_outside_its_run_directory_is_merged_with_the_config_named(export, run_dir, tmp_path):
    checkpoint_dir = tmp_path / "moved/global_step_3"
    checkpoint_dir.mkdir(parents=True)
    config_path = run_dir / f"hydra/{LAUNCHES[0]}/.hydra/config.yaml"

    process, commands = export(checkpoint_dir, RUN_CONFIG=str(config_path))

    assert process.returncode == 0, process.stdout + process.stderr
    assert commands == [
        MERGER_COMMAND.format(config_path=config_path, checkpoint_dir=checkpoint_dir, model_dir=tmp_path / "model")
    ]


@pytest.mark.parametrize("named_config", [None, "nowhere/config.yaml"])
def test_a_checkpoint_without_its_run_s_config_is_not_merged(named_config, export, tmp_path):
    checkpoint_dir = tmp_path / "moved/global_step_3"
    checkpoint_dir.mkdir(parents=True)

    process, commands = export(checkpoint_dir, **({"RUN_CONFIG": named_config} if named_config else {}))

    assert process.returncode == 1
    sought_path = named_config or f"{checkpoint_dir}/../../hydra/*/.hydra/config.yaml"
    assert process.stdout == (
        f"no config of the checkpoint's run at {sought_path}: name the run's config as RUN_CONFIG\n"
    )
    assert commands == []
