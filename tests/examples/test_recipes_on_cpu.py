import os
import signal
import subprocess
import time

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import MissingConfigException
from omegaconf import DictConfig

from repository import REPO

# the general evaluation serves and scores with other programs: it is not one of the family recipes
EVALUATION_RECIPES = sorted(set((REPO / "examples/eval").glob("eval_*.sh")) - {REPO / "examples/eval/eval_general.sh"})
SERVER_RECIPE_DIR = REPO / "examples/models/qwen3_8_27b_pear/deploy"  # the server's recipe and its gpu/ files
SERVER_GPUS = sorted(path.stem for path in (SERVER_RECIPE_DIR / "gpu").glob("*.yaml"))  # GPUs with engine arguments
SERVER_KERNELS = "pear.models.qwen3_5.deploy.deploy_kernels.Qwen3_5DeployKernels"  # the family's deploy kernels
# stand-ins on a machine without GPUs. For the general evaluation's serve: four GPUs, a training environment's Python
# that names its packages' directory, and a vLLM server that records its process and stays. For its run, under client/:
# replicas that answer and a wait that ends at once, and an evaluation environment's Python that writes its arguments
# and what it is told of the Hub. For the station's recipe: a station environment's Python that writes its arguments
STAND_INS = {
    "bin/nvidia-smi": "#!/bin/sh\nprintf 'GPU 0\\nGPU 1\\nGPU 2\\nGPU 3\\n'\n",
    "env/bin/python": "#!/bin/sh\necho /nowhere\n",
    "bin/vllm": '#!/bin/sh\necho $$ >> "$REPLICA_PIDS_FILE"\nexec sleep 600\n',
    "client/bin/curl": "#!/bin/sh\n",
    "client/bin/sleep": "#!/bin/sh\n",
    "client/env/bin/python": '#!/bin/sh\necho "$* HF_HUB_OFFLINE=$HF_HUB_OFFLINE'
    ' HF_HUB_DISABLE_TELEMETRY=$HF_HUB_DISABLE_TELEMETRY" >> "$CLIENT_LOG"\n',
    "envs/station/bin/python": '#!/bin/sh\necho "$*" >> "$STATION_LOG"\n',
    # For the server's recipe, under server/, its work directory: the GPU's PCI bus, the CPUs next to it and a taskset,
    # which write what they are asked, and a server environment's Python that names its packages' directory and writes
    # its arguments and the switch of vLLM's usage report
    "server/bin/nvidia-smi": "#!/bin/sh\necho 00000000:1B:00.0\n",
    "server/bin/cat": '#!/bin/sh\necho "cat $*" >> "$SERVER_LOG"\necho 0-7\n',
    "server/bin/taskset": '#!/bin/sh\necho "taskset $1 $2" >> "$SERVER_LOG"\nshift 2\nexec "$@"\n',
    "server/envs/server/bin/python": '#!/bin/sh\n[ "$1" != -c ] || exec echo /nowhere\n'
    'printf "%s\\n" "$@" "VLLM_NO_USAGE_STATS=$VLLM_NO_USAGE_STATS" >> "$SERVER_LOG"\n',
}
WAIT_S = 30  # for a replica to start or to end


class ServerRecipeConfig:
    """The server's config as run_server.sh composes it: the recipe's directory searched for gpu/<GPU>.yaml, which
    overrides the family's deploy config."""

    @staticmethod
    def of(gpu: str) -> DictConfig:
        overrides = [
            f"hydra.searchpath=[pkg://pear.models,file://{SERVER_RECIPE_DIR}]",
            "+qwen3_5=deploy",
            f"+gpu={gpu}",
        ]
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "pear" / "configs"), version_base=None):
            return compose(config_name="server", overrides=[*overrides, "model=/model"])


@pytest.fixture
def stand_ins(tmp_path):
    """The stand-ins, written into the test's directory."""
    for name, text in STAND_INS.items():
        stand_in = tmp_path / name
        stand_in.parent.mkdir(parents=True, exist_ok=True)
        stand_in.write_text(text)
        stand_in.chmod(0o755)


def test_the_evaluation_recipes_differ_in_their_settings_and_their_family_alone():
    # below the user-adjustable settings and outside the family's own section, the recipes are one text
    shared_texts = set()
    for recipe in EVALUATION_RECIPES:
        settings, _, body = recipe.read_text().partition("# ---- end user-adjustable ----\n")
        gpu_settings, _, family_and_rest = body.partition("# ---- environment family ----\n")
        family, _, launch = family_and_rest.partition("# ---- derived ----\n")
        assert settings and gpu_settings and family and launch, recipe.name
        shared_texts.add(gpu_settings + launch)

    assert len(EVALUATION_RECIPES) > 1
    assert len(shared_texts) == 1


@pytest.mark.usefixtures("stand_ins")
def test_a_signal_to_the_general_evaluation_s_serve_ends_its_replicas(tmp_path):
    # left running, the replicas would hold the machine's GPUs after their script is gone
    pids_file = tmp_path / "replica_pids"
    variables = {"WORK_DIR": str(tmp_path), "VERL_ENV": str(tmp_path / "env"), "LOCAL_DIR": str(tmp_path / "local")}
    run = {"GPU": "h100", "MODEL": str(tmp_path), "NAME": "model", "RUN": str(tmp_path / "run")}
    path = f"{tmp_path}/bin:{os.environ['PATH']}"

    serve = subprocess.Popen(
        ["bash", "examples/eval/eval_general.sh", "serve"],
        env={"PATH": path, "REPLICA_PIDS_FILE": str(pids_file), **variables, **run},
        cwd=REPO,
    )
    deadline = time.monotonic() + WAIT_S
    while not pids_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    replica_pids = [int(pid) for pid in pids_file.read_text().split()]
    serve.send_signal(signal.SIGTERM)
    serve.wait(WAIT_S)

    assert replica_pids
    for pid in replica_pids:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.usefixtures("stand_ins")
def test_the_general_evaluation_s_client_reaches_the_hub_with_its_telemetry_off(tmp_path):
    client_log = tmp_path / "client_log"
    variables = {"WORK_DIR": str(tmp_path), "LOCAL_DIR": str(tmp_path / "local"), "CLIENT_LOG": str(client_log)}
    run = {"GPU": "h100", "MODEL": str(tmp_path), "NAME": "model", "RUN": str(tmp_path / "run")}
    programs = {
        "PATH": f"{tmp_path}/client/bin:{os.environ['PATH']}",
        "EVAL_PYTHON": f"{tmp_path}/client/env/bin/python",
    }

    process = subprocess.run(
        ["bash", "examples/eval/eval_general.sh", "run", "localhost:8000"],
        env={**programs, **variables, **run},
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    [client] = [call for call in client_log.read_text().splitlines() if call.startswith("-m pear.eval.general run ")]
    assert client.endswith(" HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1")


@pytest.mark.usefixtures("stand_ins")
def test_the_station_s_recipe_gives_the_station_its_file_its_run_and_the_served_model(tmp_path):
    station_log = tmp_path / "station_log"
    variables = {"WORK_DIR": str(tmp_path), "STATION_CONFIG": "/station.yaml", "STATION_LOG": str(station_log)}
    run = {"SERVER": "192.0.2.1:8000", "NAME": "model", "RUN": "/run"}

    process = subprocess.run(
        ["bash", "examples/models/qwen3_8_27b_pear/deploy/run_station.sh"],
        env={"PATH": os.environ["PATH"], **variables, **run},
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert station_log.read_text().split() == [
        *("-m", "pear.deploy.station.cli", "gui", "--config", "/station.yaml"),
        "run_dir=/run",
        "default_model=served",
        "models.served.label=Qwen3.8-27B",
        "models.served.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient",
        "models.served.client.base_url=http://192.0.2.1:8000/v1",
        "models.served.client.model=model",
    ]


@pytest.mark.usefixtures("stand_ins")
def test_the_server_s_recipe_serves_from_the_server_s_environment_on_its_gpu_s_cpus_with_vllm_s_usage_report_off(
    tmp_path,
):
    server_log = tmp_path / "server_log"
    variables = {
        "WORK_DIR": str(tmp_path / "server"),
        "LOCAL_DIR": str(tmp_path / "local"),
        "SERVER_LOG": str(server_log),
    }
    run = {"GPU": "gb300", "MODEL": "/model", "DRAFT": "/draft", "RUN": "/run"}

    process = subprocess.run(
        ["bash", "examples/models/qwen3_8_27b_pear/deploy/run_server.sh", "kernels=null"],
        env={"PATH": f"{tmp_path}/server/bin:{os.environ['PATH']}", **variables, **run},
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    calls = server_log.read_text().splitlines()
    # Hydra's folder of the run, named by the time the server started
    [run_dir] = [call for call in calls if call.startswith("hydra.run.dir=")]
    assert run_dir.startswith("hydra.run.dir=/run/hydra/")
    assert [call for call in calls if call != run_dir] == [
        "cat /sys/bus/pci/devices/0000:1b:00.0/local_cpulist",
        "taskset -c 0-7",
        *("-m", "pear.deploy.server.main", "--config-dir", str(SERVER_RECIPE_DIR)),
        *("+qwen3_5=deploy", "+gpu=gb300", "model=/model", "draft=/draft"),
        "+server={host: 0.0.0.0, port: 8000, served_model_name: qwen3_8_27b}",
        "kernels=null",
        "VLLM_NO_USAGE_STATS=1",
    ]


@pytest.mark.parametrize("gpu", SERVER_GPUS)
def test_a_gpu_s_engine_arguments_compose_with_the_family_s_deploy_config(gpu):
    config = ServerRecipeConfig.of(gpu)

    assert config.model == "/model"
    assert config.kernels == SERVER_KERNELS
    assert config.engine.max_num_seqs == 1


def test_on_gb300_the_engine_runs_trt_llm_attention_and_leaves_the_megakernels_their_memory():
    engine = ServerRecipeConfig.of("gb300").engine

    assert engine.attention_config.use_trtllm_attention
    assert engine.gpu_memory_utilization < 1


def test_on_the_rtx_5090_the_window_fits_an_fp8_kv_cache_beside_weights_in_host_memory():
    # the family's package, which imports vLLM: the other recipes' tests do not need it
    from pear.models.qwen3_5.deploy.memory_plan import MemoryPlan

    config = ServerRecipeConfig.of("rtx5090")

    assert config.engine.kv_cache_dtype == "fp8_e4m3"
    assert config.engine.max_model_len < ServerRecipeConfig.of("gb300").engine.max_model_len
    assert config.engine.compilation_config.cudagraph_capture_sizes == [1, 4, 8]
    assert set(config.env.PEAR_DEPLOY_MEMORY_PLAN.split(",")) == set(MemoryPlan.OPTIONS)


def test_a_gpu_without_engine_arguments_stops_the_server():
    with pytest.raises(MissingConfigException, match="gpu/a100"):
        ServerRecipeConfig.of("a100")
