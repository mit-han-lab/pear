import importlib.util
import os
import stat
import subprocess
import sys

import pytest

from repository import REPO

# scripts/gpu_process_variables.sh as scripts/start_ray_node.sh sources it for an environment, then a command
GPU_ENVIRONMENT_COMMAND = 'WORK_DIR=$1 LOCAL_DIR=$1; source "$3/scripts/gpu_process_variables.sh" "$2"; "${@:4}"'
# vLLM's own answer
USAGE_REPORT_CODE = "from vllm.usage.usage_lib import is_usage_stats_enabled; print(is_usage_stats_enabled())"
# LMCache's own answer
LMCACHE_USAGE_REPORT_CODE = (
    "from lmcache.usage_telemetry.identity import is_usage_tracking_enabled; print(is_usage_tracking_enabled())"
)
# the Hub library's own answers: whether it is offline, its directory, the login it would send
HUB_CODE = "from huggingface_hub import constants as c; print(c.HF_HUB_OFFLINE, c.HF_HOME, c.HF_TOKEN_PATH)"
# FlashInfer asked for a kernel file that none of its directories holds
MISSING_KERNEL_CODE = "from flashinfer.jit.cubin_loader import get_artifact; get_artifact('missing.cubin', '0' * 64)"


@pytest.fixture
def gpu_environment(tmp_path):
    # by default the environment the tests run in
    def run(*command: str, python_env: str = sys.prefix, **variables: str) -> subprocess.CompletedProcess:
        # a home of its own: vLLM also looks for ~/.config/vllm/do_not_track
        environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), **variables}

        return subprocess.run(
            ["bash", "-c", GPU_ENVIRONMENT_COMMAND, "bash", str(tmp_path), python_env, str(REPO), *command],
            env=environment,
            capture_output=True,
            text=True,
        )

    return run


def test_cuda_home_is_the_toolkit_among_the_packages_of_the_environment_s_python(gpu_environment, tmp_path):
    # an environment of another Python version than today's, whose Python names the directory of its packages
    site_packages = tmp_path / "env/lib/python3.99/site-packages"
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!/bin/sh\necho {site_packages}\n")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)

    process = gpu_environment("printenv", "CUDA_HOME", python_env=str(tmp_path / "env"))

    assert (process.returncode, process.stdout.strip()) == (0, f"{site_packages}/nvidia/cu13")


def test_a_gpu_machine_s_environment_turns_vllm_s_usage_report_off(gpu_environment):
    process = gpu_environment(sys.executable, "-c", USAGE_REPORT_CODE)

    assert (process.returncode, process.stdout.strip()) == (0, "False")


def test_the_usage_report_is_back_on_for_who_sets_the_switch_to_0(gpu_environment):
    process = gpu_environment(sys.executable, "-c", USAGE_REPORT_CODE, VLLM_NO_USAGE_STATS="0")

    assert (process.returncode, process.stdout.strip()) == (0, "True")


def test_a_gpu_machine_s_environment_names_lmcache_s_switch_for_no_usage_report(gpu_environment):
    process = gpu_environment("printenv", "LMCACHE_TRACK_USAGE")

    assert (process.returncode, process.stdout.strip()) == (0, "false")


@pytest.mark.parametrize(("variables", "is_reporting"), [({}, "False"), ({"LMCACHE_TRACK_USAGE": "true"}, "True")])
def test_lmcache_reports_its_usage_only_for_who_sets_the_switch_to_true(gpu_environment, variables, is_reporting):
    if importlib.util.find_spec("lmcache") is None:
        pytest.skip("LMCache is built into a GPU machine's environment only")

    process = gpu_environment(sys.executable, "-c", LMCACHE_USAGE_REPORT_CODE, **variables)

    assert (process.returncode, process.stdout.strip()) == (0, is_reporting), process.stderr


def test_a_gpu_machine_s_environment_keeps_the_hub_s_libraries_offline_and_in_the_work_directory(
    gpu_environment, tmp_path
):
    # the home directory of the test holds no login either: the token's path alone tells where one would be read
    process = gpu_environment(sys.executable, "-c", HUB_CODE)

    hub_dir = f"{tmp_path}/cache/hf"
    assert (process.returncode, process.stdout.split()) == (0, ["True", hub_dir, f"{hub_dir}/token"])


def test_the_hub_is_reachable_again_for_who_sets_the_switch_to_0(gpu_environment):
    process = gpu_environment(sys.executable, "-c", HUB_CODE, HF_HUB_OFFLINE="0")

    assert (process.returncode, process.stdout.split()[0]) == (0, "False")


def test_a_gpu_machine_s_environment_turns_the_tokenizers_thread_pool_off(gpu_environment):
    process = gpu_environment("printenv", "TOKENIZERS_PARALLELISM")

    assert (process.returncode, process.stdout.strip()) == (0, "false")


def test_a_gpu_machine_s_environment_has_flashinfer_refuse_to_download_a_kernel(gpu_environment, tmp_path):
    # the kernels' source is a directory of the test: nothing is asked of FlashInfer's artifact server either way
    process = gpu_environment(sys.executable, "-c", MISSING_KERNEL_CODE, FLASHINFER_CUBINS_REPOSITORY=str(tmp_path))

    assert process.returncode != 0
    assert "refusing to download" in process.stderr


def test_the_environment_s_programs_lead_the_path_and_its_name_keys_flashinfer_s_workspace(gpu_environment, tmp_path):
    # the server's environment, as its recipe names it, with a Python that names the directory of its packages
    python_env = tmp_path / "envs/server"
    python = python_env / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!/bin/sh\necho {python_env}/site-packages\n")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)

    process = gpu_environment(
        "sh", "-c", 'printf "%s\\n" "$PATH" "$FLASHINFER_WORKSPACE_BASE"', python_env=str(python_env)
    )

    path, workspace = process.stdout.splitlines()
    assert process.returncode == 0, process.stderr
    assert path.startswith(f"{python_env}/bin:")
    assert workspace == f"{tmp_path}/flashinfer_server"


def test_without_an_environment_the_variables_end_the_caller(gpu_environment):
    process = gpu_environment("printenv", "CUDA_HOME", python_env="")

    assert process.returncode != 0
    assert "the Python environment of the processes" in process.stderr
