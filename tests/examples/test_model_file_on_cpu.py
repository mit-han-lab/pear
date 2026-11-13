import subprocess

import pytest

from repository import REPO

MODEL_FILE = REPO / "examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"
# what a recipe sets before it sources the file
SETTINGS = "MODEL=/model RUN=/run MAX_MODEL_LEN=260096"


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
