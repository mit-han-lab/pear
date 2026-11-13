import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pear.utils.model_merger import PearModelMerger
from repository import REPO
from verl.model_merger.base_model_merger import generate_config_from_args, parse_args


@pytest.fixture
def run_config(tmp_path):
    # the config Hydra records at a run's launch, with the run's rollout
    def make(rollout: dict) -> Path:
        config_path = tmp_path / "config.yaml"
        OmegaConf.save({"actor_rollout_ref": {"rollout": rollout}}, config_path)

        return config_path

    return make


@pytest.fixture
def executed_commands(monkeypatch):
    # the commands the merger would turn its process into
    commands = []
    monkeypatch.setattr(os, "execv", lambda path, command: commands.append((path, command)))

    return commands


def test_verl_s_merger_runs_with_the_fp32_modules_of_the_run_s_config(run_config, executed_commands):
    config_path = run_config({"name": "vllm", "fp32_modules": ["visual", "A_log"]})

    PearModelMerger(config_path, ["merge", "--backend", "fsdp"]).run()

    verl_command = ["-m", "verl.model_merger", "merge", "--backend", "fsdp", "--fp32_modules", "visual", "A_log"]
    assert executed_commands == [(sys.executable, [sys.executable, *verl_command])]


@pytest.mark.parametrize("fp32_modules", [["visual", "A_log"], []])
def test_verl_reads_the_run_s_fp32_modules_from_the_command(
    fp32_modules, run_config, executed_commands, monkeypatch, tmp_path
):
    actor_dir, model_dir = str(tmp_path / "actor"), str(tmp_path / "model")
    verl_arguments = ["merge", "--backend", "fsdp", "--local_dir", actor_dir, "--target_dir", model_dir]
    PearModelMerger(run_config({"fp32_modules": fp32_modules}), verl_arguments).run()

    # verl's own reading of what follows `python -m` in the command, as its __main__ does it
    monkeypatch.setattr(sys, "argv", executed_commands[0][1][2:])
    merger_config = generate_config_from_args(parse_args())

    assert merger_config.fp32_modules == fp32_modules
    assert (merger_config.operation, merger_config.backend) == ("merge", "fsdp")
    assert (merger_config.local_dir, merger_config.target_dir) == (actor_dir, model_dir)


def test_a_file_that_is_not_a_run_s_config_is_not_merged(run_config, executed_commands):
    config_path = run_config({"name": "vllm"})

    with pytest.raises(ValueError, match="holds no actor_rollout_ref.rollout.fp32_modules"):
        PearModelMerger(config_path, ["merge", "--backend", "fsdp"]).run()

    assert executed_commands == []


def test_every_run_s_config_names_its_fp32_modules():
    # the trainer's config, which every recipe composes: a model family's config replaces the list
    trainer_config = OmegaConf.load(REPO / "pear/configs/base.yaml")

    assert OmegaConf.select(trainer_config, PearModelMerger.FP32_MODULES_KEY) is not None


def test_the_command_line_s_config_is_pear_s_and_the_rest_verl_s(monkeypatch):
    verl_arguments = ["merge", "--backend", "fsdp", "--local_dir", "actor", "--target_dir", "model"]
    monkeypatch.setattr(sys, "argv", ["pear.utils.model_merger", "--config", "run/config.yaml", *verl_arguments])

    merger = PearModelMerger.from_command_line()

    assert merger.config_path == Path("run/config.yaml")
    assert merger.verl_arguments == verl_arguments


def test_the_merger_runs_as_the_module_the_export_script_names():
    process = subprocess.run(
        [sys.executable, "-m", "pear.utils.model_merger", "--help"], capture_output=True, text=True
    )

    assert process.returncode == 0
    assert "--config" in process.stdout
