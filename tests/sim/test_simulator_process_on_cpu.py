import importlib
import pickle
import sys
from pathlib import Path

import pytest

from pear.sim.process import SimulatorProcess
from pear.sim.schemas import ServerContext
from repository import REPO

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="a process dies with its parent through prctl (Linux)")
START_TIMEOUT_S = ROLLOUT_TIMEOUT_S = 60.0  # a child imports Python and pear in seconds
# The process's env class: a family whose own entries of the path name another pear ahead of its module's directory, and
# whose processes work in a directory without one; a call returns the file its process imported pear's contract from.
ENV = r"""
import os, sys
from pathlib import Path

import pear.sim.env
from pear.sim.env import StatelessEnv
from pear.sim.process import SimulatorProcess

MODULE_DIR = Path(__file__).parent


class ImportedTreeEnv(StatelessEnv):
    @classmethod
    def setup_env(cls, context):
        environment = {**os.environ, "PYTHONPATH": f"{MODULE_DIR / 'decoy'}{os.pathsep}{MODULE_DIR}"}
        return [SimulatorProcess("sim", cls, sys.executable, [], environment, MODULE_DIR, context.log_dir)], 1

    @classmethod
    def task_name(cls, env_args):
        return "imported tree"

    def rollout(self, history, action):
        return {"contract_file": pear.sim.env.__file__}
"""


@pytest.fixture
def env_class(tmp_path, monkeypatch) -> type:
    """ImportedTreeEnv, in a module of the test's directory."""
    (tmp_path / "imported_tree_env.py").write_text(ENV)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "imported_tree_env", raising=False)

    return importlib.import_module("imported_tree_env").ImportedTreeEnv


@pytest.mark.parametrize(
    ("family_path", "process_path"),
    [("/family/a:/family/b", f"{REPO}:/family/a:/family/b"), ("", str(REPO)), (None, str(REPO))],
    ids=["entries", "empty", "unset"],
)
def test_a_process_has_this_repository_ahead_of_its_family_s_entries_of_the_path(
    env_class, tmp_path, family_path, process_path
):
    environment = {"HOME": "/home"} if family_path is None else {"HOME": "/home", "PYTHONPATH": family_path}

    process = SimulatorProcess("sim", env_class, sys.executable, [], environment, tmp_path, tmp_path)

    assert process.variables == {"HOME": "/home", "PYTHONPATH": process_path}


def test_a_process_imports_pear_from_the_tree_of_its_starter(env_class, tmp_path):
    (tmp_path / "decoy" / "pear").mkdir(parents=True)
    (tmp_path / "decoy" / "pear" / "__init__.py").write_text("raise ImportError('the decoy tree')")
    [process], _ = env_class.setup_env(ServerContext({}, tmp_path / "logs", [], print))
    process.log_dir.mkdir()

    process.start()
    assert process.wait_ready(START_TIMEOUT_S), f"the process ended in its start; see {process.log_path}"
    reply = pickle.loads(process.call({"env_args": {}, "history": [], "action": None}, ROLLOUT_TIMEOUT_S))
    process.close()
    process.wait_or_kill(None)

    assert Path(reply["contract_file"]) == REPO / "pear" / "sim" / "env.py"
