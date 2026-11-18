import subprocess

import pytest
from omegaconf import OmegaConf

from repository import REPO
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config
from vllm_serve import ServeCommandLine

MODEL_FILE = REPO / "examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"
# what a recipe sets before it sources the file
SETTINGS = "MODEL=/model RUN=/run MAX_MODEL_LEN=260096"
# the configs whose engine arguments every replica of a training takes
REPLICA_CONFIGS = (REPO / "pear/configs/base.yaml", REPO / "pear/models/qwen3_5/config.yaml")
# the engine settings these configs pin where vLLM's default differs
PINNED_FIELDS = (
    "prefix_cache_retention_interval",
    "enable_extensible_kv_cache",
    "mm_device_do_normalize",
    "numa_bind",
)


def sourced(gpu: str, command: str, settings: str = "") -> subprocess.CompletedProcess:
    script = f"set -euo pipefail; GPU={gpu} {SETTINGS} {settings}; source {MODEL_FILE}; {command}"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize(("size", "tokens"), [("512 512", "256"), ("640 480", "300"), ("640 360", "220")])
def test_an_image_is_its_sides_in_whole_tokens(size, tokens):
    assert sourced("gb300", f"model_image_tokens {size}").stdout.strip() == tokens


@pytest.mark.parametrize(("gpu", "sizes"), [("gb300", "4 2 32"), ("h100", "8 4 32")])
def test_a_gpu_gives_the_model_its_sizes(gpu, sizes):
    assert sourced(gpu, 'echo "$GPUS_PER_NODE $TP $SESSIONS_PER_REPLICA"').stdout.strip() == sizes


def test_a_size_the_recipe_set_wins():
    process = sourced("gb300", 'echo "$TP $SESSIONS_PER_REPLICA"', "TP=4 SESSIONS_PER_REPLICA=64")

    assert process.stdout.strip() == "4 64"


@pytest.mark.parametrize("gpu", ["gb300", "h100"])
def test_every_argument_array_is_filled_and_no_key_comes_twice(gpu):
    command = 'printf "%s\\n" "${MODEL_ARGS[@]}" "${MODEL_ROLLOUT_ARGS[@]}" "${MODEL_TRAINING_ARGS[@]}"'
    arguments = sourced(gpu, command).stdout.splitlines()
    keys = [argument.split("=", 1)[0].lstrip("+") for argument in arguments]
    arrays = ("MODEL_ARGS", "MODEL_ROLLOUT_ARGS", "MODEL_TRAINING_ARGS", "MODEL_SERVER_ARGS")
    counts = sourced(gpu, "echo " + " ".join(f"${{#{array}[@]}}" for array in arrays)).stdout.split()

    assert "actor_rollout_ref.model.path=/model" in arguments
    assert len(keys) == len(set(keys))
    assert len(counts) == len(arrays) and "0" not in counts


def test_an_unknown_gpu_ends_the_recipe():
    process = sourced("a100", "echo not reached")

    assert process.returncode == 1
    assert process.stdout == "unknown GPU a100 (gb300 | h100)\n"


@pytest.mark.parametrize("gpu", ["gb300", "h100"])
def test_the_general_evaluation_s_servers_take_the_training_replicas_engine_settings(gpu):
    # the replicas' engine arguments of base.yaml and the family's config, as verl writes them on their command line
    engine_kwargs = {}
    for config in REPLICA_CONFIGS:
        rollout = OmegaConf.to_container(OmegaConf.load(config))["actor_rollout_ref"]["rollout"]
        engine_kwargs.update(rollout["engine_kwargs"]["vllm"])
    server_args = sourced(gpu, 'printf "%s\\n" "${MODEL_SERVER_ARGS[@]}"').stdout.splitlines()

    replica = ServeCommandLine.values(build_cli_args_from_config(engine_kwargs), PINNED_FIELDS)
    server = ServeCommandLine.values(server_args, PINNED_FIELDS)

    # every boundary's state kept, the KV cache from vLLM's estimate, the pixels normalized on the CPU, no NUMA binding
    assert server == replica == dict(zip(PINNED_FIELDS, (None, False, False, False), strict=True))
