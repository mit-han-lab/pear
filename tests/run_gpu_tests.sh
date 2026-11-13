#!/usr/bin/env bash
# The tests that need a GPU machine, which CI has none of: PEAR's *_on_gpu.py, then verl's own GPU tests and distributed
# scripts in the files the patches touch (3rdparty/verl/scripts/run_patched_verl_tests.sh).
#   export WORK_DIR=<the directory scripts/install.sh built>
#   bash tests/run_gpu_tests.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
# JIT caches of this run alone, on the machine's own disk: the tests compile what the checkout holds
LOCAL_DIR=$(mktemp -d "${TMPDIR:-/var/tmp}/pear_gpu_tests.XXXXXX")
trap 'rm -rf "$LOCAL_DIR"' EXIT
# shellcheck source=../scripts/gpu_process_variables.sh
source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"

cd "$ROOT"
# without the robot station's directory, whose conftest imports the station's packages
python -m pytest tests --ignore=tests/deploy/station --ignore-glob='*_on_cpu.py' --ignore-glob='*_in_sim.py'
bash 3rdparty/verl/scripts/run_patched_verl_tests.sh gpu
bash 3rdparty/verl/scripts/run_patched_verl_tests.sh distributed
