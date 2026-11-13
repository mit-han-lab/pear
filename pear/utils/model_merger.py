from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


class PearModelMerger:
    """verl's model merger with the fp32 modules of a PEAR run: the weights the run's replicas load in fp32
    (rollout.fp32_modules in its config) are written in fp32, the others in bf16."""

    FP32_MODULES_KEY = "actor_rollout_ref.rollout.fp32_modules"  # in a run's config

    def __init__(self, config_path: Path, verl_arguments: list[str]) -> None:
        self.config_path = config_path  # what Hydra recorded at the run's launch
        self.verl_arguments = verl_arguments  # verl's command line, which names no fp32 module

    @classmethod
    def from_command_line(cls) -> PearModelMerger:
        """The merger of the command line: --config, then verl's own arguments (scripts/export_checkpoint.sh)."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--config", type=Path, required=True, help="the config Hydra recorded at the run's launch")

        args, verl_arguments = parser.parse_known_args()

        return cls(args.config, verl_arguments)

    def run(self) -> None:
        fp32_modules = OmegaConf.select(OmegaConf.load(self.config_path), self.FP32_MODULES_KEY)
        # PEAR's trainer config always holds the key, [] for none: a file without it is not a run's config
        if fp32_modules is None:
            raise ValueError(f"{self.config_path} holds no {self.FP32_MODULES_KEY}: it is not the config of a PEAR run")

        command = [sys.executable, "-m", "verl.model_merger", *self.verl_arguments, "--fp32_modules", *fp32_modules]
        os.execv(sys.executable, command)


if __name__ == "__main__":
    PearModelMerger.from_command_line().run()
