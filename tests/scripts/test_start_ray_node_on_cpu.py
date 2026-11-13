import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from repository import REPO

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the node start reads the port range of /proc")
CALLER_UMASK = 0o022  # a caller whose new directories would be open to every user
GPU_N = 2  # the GPUs the stand-in nvidia-smi lists
# The programs the node start calls: the machine's nvidia-smi, the training environment's python, which names the
# directory of its packages, and its ray, which records its arguments instead of starting a node.
STAND_INS = {
    "bin/nvidia-smi": f'#!/bin/sh\nfor gpu in $(seq {GPU_N}); do echo "GPU $gpu: a stand-in"; done\n',
    "env/bin/python": "#!/bin/sh\necho /packages\n",
    "env/bin/ray": '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$RAY_RECORD"\n',
}


class NodeStart:
    """scripts/start_ray_node.sh on stand-in programs, the machine's own directory under the test's."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.local_dir = directory / "local" / "pear"
        self.ray_record = directory / "ray_arguments"

        for name, text in STAND_INS.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            path.chmod(0o755)

    def run(self, role: str, **variables: str) -> list[str]:
        """Starts the head of a cluster as a machine of ``role`` and returns the arguments ray got."""
        environment = {
            "PATH": f"{self.directory / 'bin'}:{os.environ['PATH']}",
            "WORK_DIR": str(self.directory / "work"),
            "VERL_ENV": str(self.directory / "env"),
            "LOCAL_DIR": str(self.local_dir),
            "RAY_RECORD": str(self.ray_record),
            **variables,
        }

        subprocess.run(
            ["bash", str(REPO / "scripts/start_ray_node.sh"), role],
            env=environment,
            umask=CALLER_UMASK,
            check=True,
            capture_output=True,
            text=True,
        )

        return self.ray_record.read_text().splitlines()


@pytest.fixture
def node_start(tmp_path) -> NodeStart:
    return NodeStart(tmp_path)


def test_the_machine_s_own_directory_the_node_start_makes_is_owner_only(node_start):
    node_start.run("trainer")

    assert stat.S_IMODE(node_start.local_dir.stat().st_mode) == 0o700
    assert (node_start.local_dir / "ray").is_dir()  # Ray's directory, inside it


def test_a_machine_s_own_directory_that_exists_keeps_its_mode(node_start):
    node_start.local_dir.mkdir(parents=True)
    node_start.local_dir.chmod(0o755)

    node_start.run("trainer")

    assert stat.S_IMODE(node_start.local_dir.stat().st_mode) == 0o755


def test_a_rollout_machine_registers_its_libero_slots_next_to_its_role(node_start):
    arguments = node_start.run("rollout", LIBERO_SERVERS_PER_NODE="12")
    [resources] = [argument for argument in arguments if argument.startswith("--resources=")]

    # what a LIBERO family counts its servers from (server_n of its env class)
    assert json.loads(resources.removeprefix("--resources=")) == {"rollout": GPU_N, "libero": 12}
