import pkgutil

import pytest

from pear.sim.schemas import ServerContext
from repository import REPO

# the three LIBERO families, each with its own env class
ENV_CLASS_PATHS = (
    "pear.envs.libero.env.LiberoEnv",
    "pear.envs.libero_plus.env.LiberoPlusEnv",
    "pear.envs.libero_pro.env.LiberoProEnv",
)
ROLLOUT_MACHINE_N = 3  # the rollout machines of the tests' cluster
SLOTS_PER_MACHINE = 12  # the libero slots each of them registered


@pytest.fixture(params=ENV_CLASS_PATHS, ids=lambda path: path.rsplit(".", 1)[1])
def env_class(request) -> type:
    return pkgutil.resolve_name(request.param)


@pytest.fixture
def process(env_class, tmp_path, monkeypatch):
    """The simulator process a family sets up for a server on a node with a LIBERO run's variables."""
    monkeypatch.setenv("LIBERO_PYTHON", "/envs/libero/bin/python")
    monkeypatch.setenv("LIBERO_MESA_DIR", "/envs/mesa")
    monkeypatch.setenv("PYTHONPATH", "/node/a:/node/b")
    monkeypatch.delenv("PEAR_ROOT", raising=False)

    [process], max_concurrent_start_n = env_class.setup_env(ServerContext({}, tmp_path, [], print))

    assert max_concurrent_start_n == 1
    return process


def test_a_libero_process_runs_in_the_family_s_python_out_of_this_repository(process, tmp_path):
    assert process.command[:4] == ["/envs/libero/bin/python", "-u", "-m", "pear.sim.process"]
    assert (process.cwd, process.log_dir) == (REPO, tmp_path)
    # this repository, then the node's own path
    assert process.variables["PYTHONPATH"] == f"{REPO}:/node/a:/node/b"


def test_a_libero_process_renders_with_mesa_s_software_device_and_no_gpu(process):
    assert process.variables["LD_LIBRARY_PATH"].split(":")[0] == "/envs/mesa/lib"
    assert process.variables["LIBGL_DRIVERS_PATH"] == "/envs/mesa/lib/dri"
    assert process.variables["GALLIUM_DRIVER"] == "llvmpipe"
    assert process.variables["CUDA_VISIBLE_DEVICES"] == ""


def test_a_libero_family_starts_a_server_on_every_slot_its_cluster_offers(env_class, monkeypatch):
    # what the environment of a launch or of a machine says of the servers counts for nothing
    monkeypatch.setenv("LIBERO_SERVERS_PER_NODE", "16")
    monkeypatch.setenv("LIBERO_SERVER_N", "64")
    slot_n = ROLLOUT_MACHINE_N * SLOTS_PER_MACHINE
    cluster_resources = {"CPU": 360.0, "GPU": 12.0, "rollout": 12.0, "libero": float(slot_n)}

    assert env_class.server_n({}, cluster_resources) == slot_n
    # a slot each: Ray places as many servers on a machine as it registered slots
    assert env_class.SERVER_RESOURCES["resources"] == {"libero": 1}


def test_a_libero_family_refuses_a_cluster_that_offers_no_slot(env_class):
    with pytest.raises(RuntimeError, match=r"scripts/start_ray_node\.sh .*LIBERO_SERVERS_PER_NODE"):
        env_class.server_n({}, {"CPU": 360.0, "GPU": 12.0, "rollout": 12.0})
