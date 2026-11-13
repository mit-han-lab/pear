#!/usr/bin/env bash
# Everything a run on LIBERO needs under $WORK_DIR, from nothing: the model (scripts/download_model.sh) and the
# targets libero, numactl and verl, then a check of what they made, on a GPU machine (aarch64 with GB300s or x86_64
# with H100s, a driver for CUDA 13). After the model's download and the targets libero and verl on CPU machines it
# finds them made and only finishes and checks.
#   WORK_DIR=<work dir> bash scripts/install.sh libero_run > install.log 2>&1
#   the variables of what it runs reach them: MODEL_FILE (the model's download), CUDA_ARCH and MAX_JOBS (verl)
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

# the simulator check's episode and the action its process carries out
CHECK_ENV_ARGS='{"suite": "libero_spatial", "task_id": 0, "trial_id": 0}'
CHECK_ACTION='{"arm": {"x": 2.0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": 0}}'

# The training environment's tests on the machine's GPU, under the variables its processes run with, then the
# simulator as a LIBERO server runs it: a process of the env class's setup_env carries out an action (its log:
# $WORK_DIR/runs/install/libero_run).
check_environments() {
  (
    export VERL_ENV=$ENVS/verl LOCAL_DIR=$WORK_DIR/cache/environment_test
    # shellcheck source=../../../gpu_process_variables.sh
    source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
    "$VERL_ENV/bin/python" -m pytest "$ROOT/tests/test_training_environment_on_gpu.py"
  )
  LIBERO_PYTHON=$ENVS/libero/bin/python \
    LIBERO_CONFIG_PATH=$ENVS/libero/etc/libero \
    LIBERO_MESA_DIR=$ENVS/mesa \
    "$ENVS/verl/bin/python" -m pear.sim.checker \
    --tool-config "$ROOT/pear/envs/libero/configs/tool_config.yaml" \
    --env-args "$CHECK_ENV_ARGS" \
    --action "$CHECK_ACTION" \
    --log-dir "$WORK_DIR/runs/install/libero_run"
}

require_tools curl getent tar sha256sum git patch perl gcc g++ nvidia-smi
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "no GPU on $(hostname): run it on a GPU machine" >&2
  exit 1
fi

# for the targets below: an environment's files are copies of those in uv's cache, not links to them
export UV_LINK_MODE=copy
echo "== 1. uv ($WORK_DIR) on $(hostname), $(gcc --version | head -n 1)"
install_uv
echo "== 2. model"
bash "$ROOT/scripts/download_model.sh"
echo "== 3. libero"
bash "$INSTALL_DIR/sim/libero/install.sh"
echo "== 4. numactl"
bash "$INSTALL_DIR/numactl/install.sh"
echo "== 5. verl"
bash "$INSTALL_DIR/verl/install.sh"
echo "== 6. checks"
check_environments
