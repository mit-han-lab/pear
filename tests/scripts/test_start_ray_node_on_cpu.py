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
RAY_STAND_IN = '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$RAY_RECORD"\nprintf \'%s\\n\' "$0" > "$RAY_RECORD.executable"\n'
# The programs the node start calls: the machine's nvidia-smi, the training environment's python, which names the
# directory of its packages, and its ray, which records its arguments instead of starting a node.
STAND_INS = {
    "bin/nvidia-smi": f'#!/bin/sh\nfor gpu in $(seq {GPU_N}); do echo "GPU $gpu: a stand-in"; done\n',
    "env/bin/python": "#!/bin/sh\necho /packages\n",
    "env/bin/ray": RAY_STAND_IN,
    "tinker/bin/ray": RAY_STAND_IN,
    "work/envs/robodojo/bin/ray": RAY_STAND_IN,
    "work/envs/robolab/bin/python": "#!/bin/sh\nexit 0\n",
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

    def run(self, role: str, address: str | None = None, **variables: str) -> list[str]:
        """Starts the head of a cluster as a machine of ``role`` and returns the arguments ray got."""
        environment = {
            "PATH": f"{self.directory / 'bin'}:{os.environ['PATH']}",
            "WORK_DIR": str(self.directory / "work"),
            "VERL_ENV": str(self.directory / "env"),
            "TINKER_ENV": str(self.directory / "tinker"),
            "LOCAL_DIR": str(self.local_dir),
            "RAY_RECORD": str(self.ray_record),
            **variables,
        }

        subprocess.run(
            ["bash", str(REPO / "scripts/start_ray_node.sh"), role, *([address] if address else [])],
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


@pytest.fixture
def isaac_runtime(node_start):
    runtime = node_start.directory / "isaac"
    for name in ("kit_exts", "RoboDojo", "omni_config", "host/lib"):
        (runtime / name).mkdir(parents=True)
    (runtime / "omni_config/omniverse.toml").touch()
    return str(runtime)


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


@pytest.mark.parametrize("role", ["cpu", "libero"])
@pytest.mark.parametrize("address", [None, "ray-head:6379"])
def test_cpu_nodes_need_no_nvidia_tools(node_start, role, address):
    (node_start.directory / "bin/nvidia-smi").write_text("#!/bin/sh\nexit 99\n")
    arguments = node_start.run(role, address, NUM_CPUS="4")
    assert "--num-gpus=0" in arguments and "--num-cpus=4" in arguments
    assert ("--head" if address is None else f"--address={address}") in arguments
    resources = json.loads(next(a.removeprefix("--resources=") for a in arguments if a.startswith("--resources=")))
    assert resources == {"libero": 4 if role == "libero" else 0}


@pytest.mark.parametrize("cpu_slots", [0, 2])
def test_isaac_nodes_offer_only_explicit_spare_cpu_capacity(node_start, isaac_runtime, cpu_slots):
    arguments = node_start.run(
        "robodojo", "ray-head:6379", SIM_RUNTIME_DIR=isaac_runtime, SIM_LIBERO_CPUS=str(cpu_slots)
    )
    assert f"--num-cpus={cpu_slots}" in arguments and f"--num-gpus={GPU_N}" in arguments
    resources = json.loads(next(a.removeprefix("--resources=") for a in arguments if a.startswith("--resources=")))
    assert resources == {"robodojo": GPU_N, **({"libero": cpu_slots} if cpu_slots else {})}


def test_cpu_worker_rejects_oversubscribed_libero_capacity(node_start):
    with pytest.raises(subprocess.CalledProcessError) as error:
        node_start.run("libero", NUM_CPUS="2", LIBERO_SERVERS_PER_NODE="3")
    assert "must fit the offered CPU slots" in error.value.stderr


@pytest.mark.parametrize("role", ["cpu", "libero"])
def test_cpu_nodes_require_an_explicit_allocation(node_start, role):
    (node_start.directory / "bin/getconf").write_text("#!/bin/sh\necho 256\n")
    (node_start.directory / "bin/getconf").chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError) as error:
        node_start.run(role)
    assert "set NUM_CPUS to the CPU slots allocated" in error.value.stderr
    assert not node_start.ray_record.exists()


@pytest.mark.parametrize("role", ["robodojo", "robolab"])
@pytest.mark.parametrize("override", [False, True])
def test_isaac_ray_changes_only_with_an_explicit_override(node_start, isaac_runtime, role, override):
    selected = node_start.directory / ("work/envs/robodojo" if role == "robodojo" else "env")
    variables = {"SIM_RUNTIME_DIR": isaac_runtime}
    if override:
        selected = node_start.directory / "tinker"
        variables["RAY_ENV"] = str(selected)
    node_start.run(role, "ray-head:6379", **variables)
    assert Path(f"{node_start.ray_record}.executable").read_text().strip() == str(selected / "bin/ray")
