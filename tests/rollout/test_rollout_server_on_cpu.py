import os
import subprocess
import sys

from vllm.utils.import_utils import resolve_obj_by_qualname

from pear.rollout.server import PearVLLMReplica, PearVLLMServer
from pear.rollout.worker_extension import PearVLLMWorkerExtension
from verl.workers.rollout.replica import RolloutReplicaRegistry


def test_the_rollout_name_vllm_builds_pears_replica():
    # the import above registered it, as verl's import of its external modules does in every process
    assert RolloutReplicaRegistry.get("vllm") is PearVLLMReplica


def test_the_server_names_pears_worker_extension():
    # vLLM resolves the name in every worker process, as here
    assert resolve_obj_by_qualname(PearVLLMServer.WORKER_EXTENSION) is PearVLLMWorkerExtension


def test_a_worker_process_imports_the_extension_before_verl():
    # vLLM imports the extension's module first in each worker process, which then imports verl with the modules a
    # recipe names (VERL_USE_EXTERNAL_MODULES), the server's among them: the server's module must find its own imports
    module = PearVLLMServer.WORKER_EXTENSION.rpartition(".")[0]
    environment = {**os.environ, "VERL_USE_EXTERNAL_MODULES": PearVLLMServer.__module__}

    subprocess.run([sys.executable, "-c", f"import {module}"], env=environment, check=True)
