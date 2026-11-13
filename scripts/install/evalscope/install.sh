#!/usr/bin/env bash
# What the general-capability evaluation needs under $WORK_DIR: its environment, $WORK_DIR/envs/evalscope, with
# evalscope at its pinned version, and the datasets of its benchmarks (pear/configs/general.yaml) with the NLTK data
# their scorers read, with which pear/eval/general.py runs them against an OpenAI endpoint. On a CPU machine.
#   WORK_DIR=<work dir> bash scripts/install.sh evalscope
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

EVALSCOPE_ENV=$ENVS/evalscope
PYTHON=$EVALSCOPE_ENV/bin/python
# the benchmarks' datasets and what the libraries that prepare them keep, where examples/eval/eval_general.sh reads
# them
DATASET_DIR=$WORK_DIR/cache/evalscope

require_tools curl getent tar sha256sum
install_uv
create_environment "$EVALSCOPE_ENV" "$EVALSCOPE_PYTHON_VERSION"
uv pip install --python "$PYTHON" --constraint "$INSTALL_DIR/evalscope/constraints.txt" \
  -r "$ROOT/pyproject.toml" --extra evalscope
# the datasets, one after the other, then the scorers' NLTK data: an evaluation then downloads neither while its
# replicas wait
PYTHONPATH=$ROOT HF_HUB_DISABLE_TELEMETRY=1 "$PYTHON" -m pear.eval.general download --dataset-dir "$DATASET_DIR"
