from __future__ import annotations

import json
import os
from typing import Any

import hydra
import uvloop
from omegaconf import DictConfig, OmegaConf
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

from pear.models.kernels import DeployKernels


class DeployServer:
    """vLLM's OpenAI-compatible server of one checkpoint for one session at a time, from its composed config."""

    # the class vLLM extends every worker with, by its qualified name
    WORKER_EXTENSION = "pear.deploy.server.worker_extension.DeployWorkerExtension"

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def run(self) -> None:
        engine_args = self.engine_args()
        self.prepare(engine_args)

        options = ["--model", self.config.model, "--worker-extension-cls", self.WORKER_EXTENSION]
        options += self.command_line({**engine_args, **self.config.server})
        args = make_arg_parser(FlexibleArgumentParser()).parse_args(options)
        validate_parsed_serve_args(args)
        uvloop.run(run_server(args))

    def engine_args(self) -> dict[str, Any]:
        """The engine's arguments: the config's, with speculative decoding when it names a draft."""
        engine_args = OmegaConf.to_container(self.config.engine, resolve=True)
        if self.config.draft:
            speculative_config = OmegaConf.to_container(self.config.speculative, resolve=True)
            engine_args["speculative_config"] = {**speculative_config, "model": self.config.draft}

        return engine_args

    def prepare(self, engine_args: dict[str, Any]) -> None:
        """Set the environment the engine's processes inherit: the config's variables and the deploy kernels."""
        os.environ.update(self.config.env)
        if self.config.kernels:
            DeployKernels.export(self.config.kernels).configure_server(
                engine_args["compilation_config"], engine_args["max_num_seqs"], engine_args["max_num_batched_tokens"]
            )
        else:
            # a class path inherited from the shell would install kernels in the workers
            os.environ.pop(DeployKernels.ENV, None)

    @staticmethod
    def command_line(options: dict[str, Any]) -> list[str]:
        """Options as vLLM's command line: --key value, --key or --no-key for a boolean, a list's items, a dict's JSON."""
        args = []
        for key, value in options.items():
            name = key.replace("_", "-")
            if isinstance(value, bool):
                args.append(f"--{name}" if value else f"--no-{name}")
            elif isinstance(value, list):
                args += [f"--{name}", *map(str, value)]
            elif value is not None:
                args += [f"--{name}", json.dumps(value) if isinstance(value, dict) else str(value)]

        return args


@hydra.main(config_path="../../configs", config_name="server", version_base=None)
def main(config: DictConfig) -> None:
    DeployServer(config).run()


if __name__ == "__main__":
    main()
