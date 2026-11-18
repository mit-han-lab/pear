import json
import subprocess
import sys


class ServeCommandLine:
    """vLLM's serve command line, read as `vllm serve` reads it and as verl reads a replica's, in a process of its own:
    vLLM's command-line modules set up the logging of the process that imports them."""

    MODEL = "/model"  # the model a command line serves: a test reads the arguments after it
    DEVICE_TYPE = "cuda"  # the servers' device, which vLLM's parser asks the platform for and a CPU machine lacks
    # the parse: the arguments, the names to read, the model and the device type in as JSON, the values out as JSON
    PROGRAM = """
import json
import sys
from unittest import mock

from vllm.entrypoints.cli import serve
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

arguments, names, model, device_type = json.loads(sys.argv[1])
parser = FlexibleArgumentParser()
subparsers = parser.add_subparsers(dest="subparser")
with mock.patch.object(current_platform, "device_type", device_type):
    for command in serve.cmd_init():
        command.subparser_init(subparsers)
args = parser.parse_args(["serve", model, *arguments])
print(json.dumps({name: getattr(args, name) for name in names}))
"""

    @classmethod
    def values(cls, arguments: list[str], names: tuple[str, ...]) -> dict:
        """The values `vllm serve` parses for these engine or server arguments, by name, from a command line."""
        payload = json.dumps([arguments, list(names), cls.MODEL, cls.DEVICE_TYPE])
        process = subprocess.run(
            [sys.executable, "-c", cls.PROGRAM, payload], capture_output=True, text=True, check=True
        )

        return json.loads(process.stdout.splitlines()[-1])
